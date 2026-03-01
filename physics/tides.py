import numpy as np
from math import isclose
from TidalPy.toolbox import quick_dual_body_tidal_dissipation

def calculate_tidal_dissipation(
        eccentricity, 
        orbital_freq, 
        spin_freq_planet, 
        spin_freq_host,
        radius_planet, 
        radius_host, 
        mass_planet, 
        mass_host,
        temp_mantle, 
        radius_core, 
        kin_visc_solid,
        rho_mantle, 
        melt_fraction,
        params,
        tides_on_flag
    ):
    """
    Calculates tidal dissipation rates, orbital evolution, and spin changes 
    for both the host star and the planet using nested TOML parameters.
    """

    if not tides_on_flag:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # --- Unpack Parameters ---
    univ_G = params['constants']['G']
    s_tide = params['star']['tides']
    p_tide = params['planet']['tides']

    # --- Derived Physical Properties ---
    g_p = univ_G * mass_planet / (radius_planet**2)
    g_h = univ_G * mass_host / (radius_host**2)
    
    rho_bulk_p = mass_planet / ((4.0 / 3.0) * np.pi * radius_planet**3)
    rho_bulk_h = mass_host / ((4.0 / 3.0) * np.pi * radius_host**3)

    # Moments of Inertia (MOI)
    moi_p = (2.0 / 5.0) * mass_planet * radius_planet**2
    moi_h = (2.0 / 5.0) * mass_host * radius_host**2

    # --- Planet Rheology & Shear Melting Model ---
    # Participation scale: Only the mantle participates in planet tides
    tidal_scale_p = (radius_planet - radius_core) / radius_planet
    
    # Dynamic Viscosity (Pa-s)
    visc_p = kin_visc_solid * rho_mantle
    
    # Calculate effective Shear Modulus (accounting for melt)
    shear_solid  = p_tide['shear_modulus_solid']
    shear_liquid = p_tide['shear_modulus_liquid_limit']
    phi_crit     = p_tide['crit_melt_frac']
    
    if melt_fraction <= 0.0:
        shear_p = shear_solid
    elif melt_fraction < phi_crit:
        # Partial melt softening
        softening = np.exp((p_tide['shear_melt_param_1'] / temp_mantle) - p_tide['shear_melt_param_2'])
        shear_p = shear_solid * softening
    elif melt_fraction <= (phi_crit + p_tide['melt_transition_width']):
        # Transition/Breakdown region
        temp_breakdown = p_tide['temp_solidus_ref'] + phi_crit * (p_tide['temp_liquidus_ref'] - p_tide['temp_solidus_ref'])
        softening = np.exp((p_tide['shear_melt_param_1'] / temp_breakdown) - p_tide['shear_melt_param_2'])
        falloff = np.exp(-p_tide['shear_melt_falloff_slope'] * (melt_fraction - phi_crit))
        shear_p = shear_solid * softening * falloff
    else:
        # Fully liquid/rheological breakdown
        shear_p = shear_liquid

    # Safety clamp
    shear_p = max(shear_p, shear_liquid)

    # --- Run Tidal Solver ---
    dissipation = quick_dual_body_tidal_dissipation(
        radii=(radius_host, radius_planet),
        masses=(mass_host, mass_planet),
        gravities=(g_h, g_p),
        densities=(rho_bulk_h, rho_bulk_p),
        mois=(moi_h, moi_p),
        viscosities=(s_tide['viscosity'], visc_p),
        shear_moduli=(s_tide['shear_modulus'], shear_p),
        rheologies=(s_tide['rheology_model'], p_tide['rheology_model']),
        complex_compliance_inputs=None,
        obliquities=(s_tide['obliquity'], p_tide['obliquity']),
        spin_frequencies=(spin_freq_host, spin_freq_planet), 
        tidal_scales=(s_tide['tidal_scale'], tidal_scale_p),
        fixed_k2s=(s_tide['fixed_k2'], p_tide['fixed_k2']),
        fixed_qs=(s_tide['fixed_Q'], p_tide['fixed_Q']),
        eccentricity=eccentricity,
        orbital_frequency=orbital_freq,
        max_tidal_order_l=2,
        eccentricity_truncation_lvl=10,
        use_obliquity=(not isclose(p_tide['obliquity'], 0.0)),
        da_dt_scale=1.0,
        de_dt_scale=1.0,
        dspin_dt_scale=1.0
    )
    
    # --- Unpack and Return ---
    return (
        dissipation['semi_major_axis_derivative'],
        dissipation['eccentricity_derivative'],
        dissipation['host']['spin_rate_derivative'],
        dissipation['secondary']['spin_rate_derivative'],
        dissipation['host']['tidal_heating'],
        dissipation['secondary']['tidal_heating']
    )
