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

    Unlike `get_meltfrac` (which averages only over the active magma ocean), 
    this function calculates the melt fraction relative to the total mantle volume. 
    This is required for bulk properties like whole-mantle degassing or bulk 
    rheology in tectonic phases.

    Parameters
    ----------
    gravity : float
        Surface gravitational acceleration in m/s^2.
    temp_potential : float
        Mantle potential temperature in Kelvin.
    radius_planet : float
        Radius of the planet in meters.
    radius_core : float
        Radius of the planetary core in meters.
    mass_mantle : float
        Total mass of the mantle in kg.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    exchange_pressure_gpa : float
        Pressure at the deepest point of melting in GPa.
    bulk_melt_fraction : float
        Volume-averaged melt fraction relative to the entire mantle.
    """

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']
    num    = params['numerical']

    # --- Thermodynamic Constants & Grid ---
    thermal_expansion = thermo['thermal_expansion']    # alpha (1/K)
    heat_capacity     = thermo['specific_heat_mantle'] # cp (J/kg/K)
    step_size         = num['melt_grid_step']          # (m)
    
    slope_low_p       = thermo['solidus_slope_low_p']
    int_low_p         = thermo['solidus_intercept_low_p']
    slope_high_p      = thermo['solidus_slope_high_p']
    int_high_p        = thermo['solidus_intercept_high_p']
    liq_offset        = thermo['liquidus_offset']

    # --- Planet Geometry & Grid ---
    volume_mantle  = (4.0 / 3.0) * np.pi * (radius_planet**3 - radius_core**3)
    density_mantle = mass_mantle / volume_mantle

    # Depth grid from surface down to the Core-Mantle Boundary (CMB)
    depths = np.arange(0, radius_planet - radius_core + step_size, step_size)

    # Hydrostatic pressure in Pascals and GPa
    pressures_pa  = gravity * density_mantle * depths
    pressures_gpa = pressures_pa / 1e9

    # --- Thermodynamics (Solidus & Liquidus) ---
    temp_solidus  = np.minimum(slope_low_p * pressures_gpa + int_low_p, 
                               slope_high_p * pressures_gpa + int_high_p)
    temp_liquidus = temp_solidus + liq_offset

    # Mantle adiabatic temperature profile
    temp_adiabat  = temp_potential + temp_potential * (thermal_expansion * gravity * depths / heat_capacity)

    # --- Melt Fraction Calculation ---
    melt_fraction = np.zeros_like(depths)
    
    # Fully molten regions
    melt_fraction[temp_adiabat >= temp_liquidus] = 1.0
    
    # Partially molten regions
    partial_mask = (temp_adiabat > temp_solidus) & (temp_adiabat < temp_liquidus)
    melt_fraction[partial_mask] = (temp_adiabat[partial_mask] - temp_solidus[partial_mask]) / \
                                  (temp_liquidus[partial_mask] - temp_solidus[partial_mask])

    # --- Base Pressure Analysis ---
    idx_melt = np.where(melt_fraction > 0.0)[0]

    if len(idx_melt) < 2:
        return 0.0, 0.0

    idx_base = idx_melt[-1]

    # Exchange pressure at the first purely solid grid point below the melt
    if idx_base < len(depths) - 1:
        exchange_pressure_gpa = pressures_gpa[idx_base + 1]
    else:
        exchange_pressure_gpa = pressures_gpa[idx_base]

    # --- Bulk Volume-Averaged Melt Fraction ---
    radii = radius_planet - depths
    
    # Integrate f(z) * r^2 dz over the ENTIRE depth grid
    integral_f_vol = trapezoid(melt_fraction * radii**2, depths)
    
    # Exact geometric volume proxy for the entire mantle (divided by 4/3 pi)
    volume_mantle_proxy = (radius_planet**3 - radius_core**3) / 3.0

    bulk_melt_fraction  = integral_f_vol / volume_mantle_proxy
    
    # Final numerical safety clamp
    bulk_melt_fraction  = min(bulk_melt_fraction, 1.0)

    return exchange_pressure_gpa, bulk_melt_fraction
