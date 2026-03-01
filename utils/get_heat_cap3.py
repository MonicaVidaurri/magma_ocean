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
    incorporating both sensible heat and the latent heat of fusion released/absorbed 
    during phase changes.

    Parameters
    ----------
    temp_mantle : float
        Mantle potential temperature in Kelvin.
    mass_mantle : float
        Total mass of the mantle in kg.
    gravity : float
        Surface gravitational acceleration in m/s^2.
    radius_planet : float
        Radius of the planet in meters.
    radius_core : float
        Radius of the planetary core in meters.
    mass_magma_ocean : float
        Mass of the active magma ocean (liquid + suspended crystals) in kg.
    dmass_solid_dt : float
        The derivative of solid mass with respect to temperature (dMs/dT). 
        Must have units of kg/K.
    func_get_meltfrac : callable
        Function to compute the melt fraction given (gravity, temp, Rp, Rc, Mass, params).
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    effective_heat_capacity : float
        The total effective heat capacity of the mantle in J/K.
    """

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']

    # --- Thermodynamic Constants ---
    specific_heat_mantle = thermo['specific_heat_mantle']  # cp (J/kg/K)
    latent_heat_fusion   = thermo['latent_heat_fusion']    # deltaH (J/kg)

    # --- Melt Fraction Calculation ---
    # The get_meltfrac function returns (exchange_pressure, volume_melt_fraction)
    # We pass the master 'params' dict to the callable so it can access its own constants.
    _, volume_melt_fraction = func_get_meltfrac(gravity, temp_mantle, radius_planet, radius_core, mass_mantle, params)

    # --- Effective Heat Capacity Calculation ---
    # Sensible Heat Capacity (J/K):
    # The entire mantle (both solid and liquid) must change temperature, 
    # so we use the total mass of the mantle, not just the liquid fraction.
    sensible_heat_capacity = specific_heat_mantle * mass_mantle

    # Latent Heat Capacity (J/K):
    # Energy absorbed/released as the mass changes phase per degree Kelvin.
    # Note: dmass_solid_dt (dMs/dT) is typically negative as the planet cools.
    latent_heat_capacity = latent_heat_fusion * dmass_solid_dt

    # Total Effective Thermal Inertia
    effective_heat_capacity = sensible_heat_capacity + latent_heat_capacity

    return effective_heat_capacity
