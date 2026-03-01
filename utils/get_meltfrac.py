import numpy as np
from scipy.integrate import trapezoid

def get_meltfrac(
        gravity,
        temp_potential,
        radius_planet,
        radius_core,
        mass_mantle,
        params):
    """
    Calculates the volume-averaged melt fraction of the magma ocean 
    using parameters from a master TOML dictionary.

    Parameters
    ----------
    gravity : float
        Surface gravity (m/s^2).
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
        Pressure at the base of the magma ocean (GPa).
    avg_melt_fraction : float
        Volume-averaged melt fraction (0.0 to 1.0).
    """

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']
    num    = params['numerical']

    alpha = thermo['thermal_expansion']
    cp    = thermo['heat_capacity']
    
    # --- Planet Geometry & Grid ---
    volume_mantle  = (4.0 / 3.0) * np.pi * (radius_planet**3 - radius_core**3)
    density_mantle = mass_mantle / volume_mantle

    # Depth grid from surface down to the CMB
    step = num['melt_grid_step']
    depths = np.arange(0, radius_planet - radius_core + step, step)

    # Hydrostatic pressure (Pa and GPa)
    pressures_pa  = gravity * density_mantle * depths
    pressures_gpa = pressures_pa / 1e9

    # --- Thermodynamics (Solidus & Liquidus) ---
    # Apply piecewise linear solidus from TOML
    temp_solidus = np.minimum(
        thermo['solidus_slope_low_p'] * pressures_gpa + thermo['solidus_intercept_low_p'], 
        thermo['solidus_slope_high_p'] * pressures_gpa + thermo['solidus_intercept_high_p']
    )
    temp_liquidus = temp_solidus + thermo['liquidus_offset']

    # Mantle adiabatic temperature profile
    temp_adiabat = temp_potential + temp_potential * (alpha * gravity * depths / cp)

    # --- Melt Fraction Calculation ---
    melt_fraction = np.zeros_like(depths)
    
    # Fully molten
    melt_fraction[temp_adiabat >= temp_liquidus] = 1.0
    
    # Partially molten (linear interpolation)
    partial_mask = (temp_adiabat > temp_solidus) & (temp_adiabat < temp_liquidus)
    melt_fraction[partial_mask] = (temp_adiabat[partial_mask] - temp_solidus[partial_mask]) / \
                                  (temp_liquidus[partial_mask] - temp_solidus[partial_mask])

    # --- Magma Ocean Boundary Analysis ---
    idx_melt = np.where(melt_fraction > 0.0)[0]

    # Solid planet check
    if len(idx_melt) < 2:
        return 0.0, 0.0

    idx_base = idx_melt[-1]

    # Determine exchange pressure at the base of the melting zone
    if idx_base < len(depths) - 1:
        exchange_pressure_gpa = pressures_gpa[idx_base + 1]
    else:
        exchange_pressure_gpa = pressures_gpa[idx_base]

    # --- Volume-Averaged Melt Fraction ---
    z_mo = depths[:idx_base + 1]
    r_mo = radius_planet - z_mo
    f_mo = melt_fraction[:idx_base + 1]

    # Integrate f(r) over the spherical shell volume
    integral_f_vol = trapezoid(f_mo * r_mo**2, z_mo)
    
    # Geometric volume proxy (V / 4pi)
    volume_mo_proxy = (r_mo[0]**3 - r_mo[-1]**3) / 3.0

    avg_melt_fraction = integral_f_vol / volume_mo_proxy
    
    return exchange_pressure_gpa, min(avg_melt_fraction, 1.0)
