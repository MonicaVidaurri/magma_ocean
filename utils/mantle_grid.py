import numpy as np


class MantleGrid:
    """
    Pre-computed static mantle grid arrays that remain constant during integration.

    All arrays depend only on planet geometry and thermodynamic parameters, not on mantle temperature, so they are
    built once before the ODE run and reused at every function evaluation. The grid spans the mantle from the surface
    (depth 0) to the core-mantle boundary (depth ``radius_planet - radius_core``) with uniform spacing.
    """
    __slots__ = ('depths', 'pressures_gpa', 'solidus', 'liquidus', 'radii', 'r2', 'adiabat_factor', 'adiabat_slope',
                 'liq_offset', 'density', 'gravity', 'thermo', 'radius_planet', 'radius_core', 'mass_mantle',
                 'mass_adiabat', 'cum_r2', 'latent_integrand', 'cum_latent', 'max_monotonic_temperature')

    def __init__(self, **arrays):
        for name, value in arrays.items():
            setattr(self, name, value)


def cumulative_integral(depths, integrand):
    """Cumulative trapezoid integral of node values, starting at 0 at the first node."""
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(depths)
    return np.concatenate(([0.0], np.cumsum(increments)))


def solidus_temperature(depth, gravity, density, thermo):
    """
    Mantle solidus and its depth derivative.

    The solidus is the minimum of a low- and a high-pressure linear branch in pressure, blended with a hyperbolic
    tangent of half-width ``solidus_crossover_width_km`` across the branch crossover so that the depth at which an
    adiabat meets it moves smoothly with temperature.

    Parameters
    ----------
    depth : float or ndarray
        Depth below the surface (m).
    gravity : float
        Gravitational acceleration, assumed constant with depth (m/s^2).
    density : float
        Mantle density, assumed uniform (kg/m^3).
    thermo : dict
        ``params['planet']['thermodynamics']``.

    Returns
    -------
    temperature : float or ndarray
        Solidus temperature (K).
    dtemperature_ddepth : float or ndarray
        Derivative of the solidus with depth (K/m).
    """
    slope_low  = thermo['solidus_slope_low_p']
    int_low    = thermo['solidus_intercept_low_p']
    slope_high = thermo['solidus_slope_high_p']
    int_high   = thermo['solidus_intercept_high_p']

    dpressure_ddepth = gravity * density / 1e9
    pressure_gpa     = dpressure_ddepth * depth
    solidus_low_p    = slope_low * pressure_gpa + int_low
    solidus_high_p   = slope_high * pressure_gpa + int_high

    crossover_depth = (int_high - int_low) / (slope_low - slope_high) / dpressure_ddepth
    crossover_width = thermo['solidus_crossover_width_km'] * 1e3
    tanh_arg = np.tanh((depth - crossover_depth) / crossover_width)
    blend    = 0.5 * (1.0 + tanh_arg)
    dblend   = 0.5 * (1.0 - tanh_arg**2) / crossover_width

    temperature = (1.0 - blend) * solidus_low_p + blend * solidus_high_p
    dtemperature_ddepth = (((1.0 - blend) * slope_low + blend * slope_high) * dpressure_ddepth
                           + dblend * (solidus_high_p - solidus_low_p))
    return temperature, dtemperature_ddepth


def build_mantle_grid(radius_planet, radius_core, gravity, mass_mantle, params):
    """
    Build a MantleGrid from planet parameters. Call once during model setup.

    Parameters
    ----------
    radius_planet : float
        Total radius of the planet (m).
    radius_core : float
        Core radius (m).
    gravity : float
        Surface gravitational acceleration (m/s^2). Assumed constant with depth.
    mass_mantle : float
        Total mass of the mantle (kg). Density is assumed uniform.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    MantleGrid
    """
    thermo = params['planet']['thermodynamics']
    num    = params['numerical']

    thermal_expansion = thermo['thermal_expansion']
    heat_capacity     = thermo['specific_heat_mantle']
    step_size         = num['melt_grid_step']
    liq_offset        = thermo['liquidus_offset']

    mantle_thickness = radius_planet - radius_core
    volume_mantle    = (4.0 / 3.0) * np.pi * (radius_planet**3 - radius_core**3)
    density_mantle   = mass_mantle / volume_mantle

    # Uniform nodes that land exactly on the core-mantle boundary.
    num_nodes     = int(np.ceil(mantle_thickness / step_size)) + 1
    depths        = np.linspace(0.0, mantle_thickness, num_nodes)
    pressures_gpa = gravity * density_mantle * depths / 1e9

    solidus, _ = solidus_temperature(depths, gravity, density_mantle, thermo)
    liquidus   = solidus + liq_offset

    radii = radius_planet - depths
    r2    = radii**2

    # Adiabat factor: temp_adiabat = temp_potential * adiabat_factor, with adiabat_factor = 1 + adiabat_slope * depth.
    adiabat_slope  = thermal_expansion * gravity / heat_capacity
    adiabat_factor = 1.0 + adiabat_slope * depths

    # Sensible heat of an adiabatic mantle is Cp * T_pot * integral(rho * f dV), so this mass sets its heat capacity.
    mass_adiabat = 4.0 * np.pi * density_mantle * cumulative_integral(depths, adiabat_factor * r2)[-1]

    # The melt-fraction field is linear in T_pot between the solidus and liquidus, so d(melt mass)/dT_pot integrates
    # this temperature-independent quantity over the partially molten depth range.
    cum_r2           = cumulative_integral(depths, r2)
    latent_integrand = adiabat_factor * r2 / liq_offset
    cum_latent       = cumulative_integral(depths, latent_integrand)

    # The melt fraction decreases monotonically with depth only while the adiabat is shallower than the solidus.
    # Crossing searches assume this; find the potential temperature at which it first fails.
    delta_factor  = np.diff(adiabat_factor)
    delta_solidus = np.diff(solidus)
    max_monotonic_temperature = float(np.min(delta_solidus / delta_factor))

    return MantleGrid(
        depths=depths, pressures_gpa=pressures_gpa, solidus=solidus, liquidus=liquidus, radii=radii, r2=r2,
        adiabat_factor=adiabat_factor, adiabat_slope=adiabat_slope, liq_offset=liq_offset, density=density_mantle,
        gravity=gravity, thermo=thermo, radius_planet=radius_planet, radius_core=radius_core, mass_mantle=mass_mantle,
        mass_adiabat=mass_adiabat, cum_r2=cum_r2, latent_integrand=latent_integrand, cum_latent=cum_latent,
        max_monotonic_temperature=max_monotonic_temperature)
