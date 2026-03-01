def get_heat_cap3(
        temp_mantle,
        mass_mantle,
        gravity,
        radius_planet,
        radius_core,
        mass_magma_ocean,
        dmass_solid_dt,
        func_get_meltfrac,
        params):
    """
    Calculates the effective heat capacity (thermal inertia) of the mantle, 
    incorporating both sensible heat and the latent heat of fusion.

    Parameters
    ----------
    temp_mantle : float
        Mantle potential temperature (K).
    mass_mantle : float
        Total mass of the mantle (kg).
    gravity : float
        Surface gravity (m/s^2).
    radius_planet : float
        Radius of the planet (m).
    radius_core : float
        Radius of the core (m).
    mass_magma_ocean : float
        Mass of the active magma ocean (kg).
    dmass_solid_dt : float
        Derivative of solid mass with respect to temperature (kg/K).
    func_get_meltfrac : callable
        Function to compute the melt fraction.
    params : dict
        Master configuration dictionary containing [planet.material].

    Returns
    -------
    effective_heat_capacity : float
        Total effective heat capacity (thermal inertia) in J/K.
    """

    # --- Unpack Parameters ---
    material = params['planet']['material']
    cp_mantle = material['specific_heat_mantle']
    dh_fusion = material['latent_heat_fusion']

    # --- Melt Fraction Calculation ---
    # The get_meltfrac function returns (exchange_pressure, volume_melt_fraction)
    _, volume_melt_fraction = func_get_meltfrac(
        gravity, temp_mantle, radius_planet, radius_core, mass_mantle
    )

    # --- Effective Heat Capacity Calculation ---
    # Sensible Heat Capacity (J/K)
    sensible_heat_capacity = cp_mantle * mass_mantle

    # Latent Heat Capacity (J/K)
    # dmass_solid_dt is kg/K, latent_heat_fusion is J/kg
    latent_heat_capacity = dh_fusion * dmass_solid_dt

    # Total Effective Thermal Inertia
    effective_heat_capacity = sensible_heat_capacity + latent_heat_capacity

    return effective_heat_capacity
