import numpy as np


class MantleGrid:
    """
    Pre-computed static mantle grid arrays that remain constant during integration.

    All arrays depend only on planet geometry and thermodynamic parameters —
    not on mantle temperature — so they can be built once before the ODE run
    and reused at every function evaluation.
    """
    __slots__ = ('depths', 'pressures_gpa', 'solidus', 'liquidus', 'radii', 'r2',
                 'adiabat_factor', 'vol_proxy_full', 'liq_offset')

    def __init__(self, depths, pressures_gpa, solidus, liquidus, radii, r2,
                 adiabat_factor, vol_proxy_full, liq_offset):
        self.depths        = depths         # Depth grid from surface to CMB (m)
        self.pressures_gpa = pressures_gpa  # Hydrostatic pressure (GPa)
        self.solidus       = solidus        # Solidus temperature profile (K)
        self.liquidus      = liquidus       # Liquidus temperature profile (K)
        self.radii         = radii          # Radius at each depth node (m)
        self.r2            = r2             # radii**2 for volume integration (m^2)
        self.adiabat_factor = adiabat_factor  # 1 + (alpha*g/Cp)*z; multiply by T_pot
        self.vol_proxy_full = vol_proxy_full  # (Rp^3 - Rc^3) / 3 (m^3)
        self.liq_offset     = liq_offset      # Liquidus - solidus offset (K), scalar


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
        Surface gravitational acceleration (m/s^2).
    mass_mantle : float
        Total mass of the mantle (kg).
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

    slope_low  = thermo['solidus_slope_low_p']
    int_low    = thermo['solidus_intercept_low_p']
    slope_high = thermo['solidus_slope_high_p']
    int_high   = thermo['solidus_intercept_high_p']
    liq_offset = thermo['liquidus_offset']

    volume_mantle  = (4.0 / 3.0) * np.pi * (radius_planet**3 - radius_core**3)
    density_mantle = mass_mantle / volume_mantle

    depths        = np.arange(0, radius_planet - radius_core + step_size, step_size)
    pressures_gpa = gravity * density_mantle * depths / 1e9

    solidus  = np.minimum(slope_low  * pressures_gpa + int_low,
                          slope_high * pressures_gpa + int_high)
    liquidus = solidus + liq_offset

    radii = radius_planet - depths
    r2    = radii**2

    # Adiabat factor: temp_adiabat = temp_potential * adiabat_factor
    adiabat_factor = 1.0 + (thermal_expansion * gravity / heat_capacity) * depths

    vol_proxy_full = (radius_planet**3 - radius_core**3) / 3.0

    return MantleGrid(depths, pressures_gpa, solidus, liquidus, radii, r2,
                      adiabat_factor, vol_proxy_full, liq_offset)
