import numpy as np
from scipy.integrate import trapezoid

def get_meltfracb(
        gravity,
        temp_potential,
        radius_planet,
        radius_core,
        mass_mantle,
        params):
    """
    Calculates the bulk volume-averaged melt fraction of the ENTIRE mantle 
    and the pressure at the base of the deepest melt layer.

    Parameters
    ----------
    gravity : float
        Surface gravitational acceleration (m/s^2).
    temp_potential : float
        Mantle potential temperature (K).
    radius_planet : float
        Radius of the planet (m).
    radius_core : float
        Radius of the planetary core (m).
    mass_mantle : float
        Total mass of the mantle (kg).
    params : dict
        Master configuration dictionary containing [planet.thermodynamics] 
        and [numerical].

    Returns
    -------
    exchange_pressure_gpa : float
        Pressure at the deepest point of melting (GPa).
    bulk_melt_fraction : float
        Volume-averaged melt fraction relative to the entire mantle (0.0 to 1.0).
    """

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']
    num    = params['numerical']

    alpha = thermo['thermal_expansion']
    cp    = thermo['heat_capacity']
    step  = num['melt_grid_step']

    # --- Planet Geometry & Grid ---
    volume_mantle  = (4.0 / 3.0) * np.pi * (radius_planet**3 - radius_core**3)
    density_mantle = mass_mantle / volume_mantle

    # Depth grid from surface down to the Core-Mantle Boundary (CMB)
    depths = np.arange(0, radius_planet - radius_core + step, step)

    # Hydrostatic pressure in Pascals and GPa
    pressures_pa  = gravity * density_mantle * depths
    pressures_gpa = pressures_pa / 1e9

    # --- Thermodynamics (Solidus & Liquidus) ---
    temp_solidus = np.minimum(
        thermo['solidus_slope_low_p'] * pressures_gpa + thermo['solidus_intercept_low_p'], 
        thermo['solidus_slope_high_p'] * pressures_gpa + thermo['solidus_intercept_high_p']
    )
    temp_liquidus = temp_solidus + thermo['liquidus_offset']

    # Mantle adiabatic temperature profile
    temp_adiabat = temp_potential + temp_potential * (alpha * gravity * depths / cp)

    # --- Melt Fraction Calculation ---
    melt_fraction = np.zeros_like(depths)
    
    # Fully molten regions
    melt_fraction[temp_adiabat >= temp_liquidus] = 1.0
    
    # Partially molten regions (linear interpolation)
    partial_mask = (temp_adiabat > temp_solidus) & (temp_adiabat < temp_liquidus)
    melt_fraction[partial_mask] = (temp_adiabat[partial_mask] - temp_solidus[partial_mask]) / \
                                  (temp_liquidus[partial_mask] - temp_solidus[partial_mask])

    # --- Base Pressure Analysis ---
    idx_melt = np.where(melt_fraction > 0.0)[0]

    if len(idx_melt) < 2:
        return 0.0, 0.0

    idx_base = idx_melt[-1]

    # Exchange pressure at the deepest point of melting
    if idx_base < len(depths) - 1:
        exchange_pressure_gpa = pressures_gpa[idx_base + 1]
    else:
        exchange_pressure_gpa = pressures_gpa[idx_base]

    # --- Bulk Volume-Averaged Melt Fraction ---
    radii = radius_planet - depths
    
    # Integrate f(r) * r^2 dr over the ENTIRE depth of the mantle
    # Note: trapezoid handles the mapping of radii over the depth grid
    integral_f_vol = trapezoid(melt_fraction * radii**2, depths)
    
    # Exact geometric volume proxy for the entire mantle (V / 4pi)
    volume_mantle_proxy = (radius_planet**3 - radius_core**3) / 3.0

    bulk_melt_fraction = integral_f_vol / volume_mantle_proxy
    
    return exchange_pressure_gpa, min(bulk_melt_fraction, 1.0)
