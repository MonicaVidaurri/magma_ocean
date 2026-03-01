import numpy as np

def viscosity(
        temp_mantle,
        temp_surface,
        density_mantle,
        mass_frac_water,
        melt_fraction,
        params):
    """
    Calculates the kinematic viscosity of the mantle across solid and liquid 
    regimes using the Lebrun et al. (2013) parameterization.

    Parameters
    ----------
    temp_mantle : float
        Potential temperature of the convective mantle in Kelvin.
    temp_surface : float
        Surface temperature of the planet in Kelvin.
    density_mantle : float
        Bulk density of the mantle in kg/m^3.
    mass_frac_water : float
        Mass fraction of water in the convecting region.
    melt_fraction : float
        Volume-averaged melt fraction of the convecting region (0.0 to 1.0).
    params : dict
        Master configuration dictionary parsed from the TOML file.

    Returns
    -------
    kinematic_viscosity : float
        The kinematic viscosity of the mantle or magma ocean in m^2/s.
    """

    # --- Unpack Parameters ---
    rheo = params['planet']['rheology']
    univ = params['constants']
    
    gas_constant = univ['gas_constant']
    phi_crit     = rheo['critical_crystal_fraction']
    
    # --- Calculate Base Viscosities ---
    
    # A. Liquid Magma Viscosity (Dynamic, Pa·s)
    # Empirical: A * exp(B / (T - T_ref))
    dynamic_visc_liquid = rheo['liquid_visc_pre_exponential'] * np.exp(
        rheo['liquid_visc_activation_temp'] / (temp_mantle - rheo['liquid_visc_ref_temp'])
    )

    # B. Solid Mantle Viscosity (Dynamic, Pa·s)
    # Arrhenius: A * exp(E / (R * T))
    dynamic_visc_solid = rheo['solid_visc_pre_exponential'] * np.exp(
        rheo['solid_visc_activation_energy'] / (gas_constant * temp_mantle)
    )

    # --- Determine Rheological Regime ---
    
    # Ensure melt fraction is within physical bounds [0, 1]
    phi_melt = np.clip(melt_fraction, 0.0, 1.0)
    phi_crystal = 1.0 - phi_melt

    if phi_crystal < phi_crit:
        # REGIME 1: Liquid-supported (Magma Ocean)
        # Viscosity of a crystal suspension (Roscoe-style)
        relative_solid_effect = phi_crystal / phi_crit
        dynamic_viscosity = dynamic_visc_liquid / (1.0 - relative_solid_effect)**2.5
        
    else:
        # REGIME 2: Matrix-supported (Solid Mantle)
        # Melt weakens the solid matrix exponentially
        dynamic_viscosity = dynamic_visc_solid * np.exp(-rheo['melt_weakening_factor'] * phi_melt)

    # --- Final Conversion ---
    # Convert Dynamic (Pa·s) to Kinematic (m^2/s)
    kinematic_viscosity = dynamic_viscosity / density_mantle

    return kinematic_viscosity
