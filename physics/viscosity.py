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

    This model dynamically switches between a liquid-supported magma ocean regime 
    (modeled as a crystal suspension) and a solid-supported mantle regime 
    (modeled as solid-state creep weakened by melt pockets) based on a critical 
    crystal fraction threshold.

    Parameters
    ----------
    temp_mantle : float
        Potential temperature of the convective mantle in Kelvin.
    temp_surface : float
        Surface temperature of the planet in Kelvin. (Currently unused in the core math).
    density_mantle : float
        Bulk density of the mantle in kg/m^3.
    mass_frac_water : float
        Mass fraction of water in the convecting region. (Currently unused in the core math).
    melt_fraction : float
        Volume-averaged melt fraction of the convecting region (0.0 to 1.0).
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    kinematic_viscosity : float
        The kinematic viscosity of the mantle or magma ocean in m^2/s.
    """

    # --- Unpack Parameters ---
    univ = params['constants']
    rheo = params['planet']['rheology']

    # --- Constants ---
    gas_constant = univ['gas_constant']  # J/(mol K)

    # --- Liquid Magma Viscosity (Dynamic, Pa·s) ---
    # Empirical parameters for silicate melt viscosity
    A_coeff_liquid        = rheo['liquid_visc_pre_exponential'] # Pa s
    B_coeff_liquid        = rheo['liquid_visc_activation_temp'] # K
    temp_reference_liquid = rheo['liquid_visc_ref_temp']        # K
    # Arrhenius parameters for solid-state creep
    pre_exponential_solid = rheo['solid_visc_pre_exponential'] # Pa s
    activation_energy     = rheo['solid_visc_activation_energy'] # J/mol

    # --- Solid Mantle Viscosity (Dynamic, Pa·s) ---
    dynamic_visc_liquid = A_coeff_liquid * np.exp(B_coeff_liquid / (temp_mantle - temp_reference_liquid))
    dynamic_visc_solid = pre_exponential_solid * np.exp(activation_energy / (gas_constant * temp_mantle))

    crystal_fraction = 1.0 - melt_fraction
    critical_crystal_fraction = rheo['critical_crystal_fraction']
    
    # Solid Regime Viscosity
    melt_weakening_factor = rheo['melt_weakening_factor']
    visc_solid_regime = dynamic_visc_solid * np.exp(-melt_weakening_factor * melt_fraction)

    # Liquid Regime Viscosity (with a safety cap to prevent division by zero)
    relative_solid_effect = crystal_fraction / critical_crystal_fraction
    safe_relative_effect = min(relative_solid_effect, 0.9999) # Prevents the infinity crash
    visc_liquid_regime = dynamic_visc_liquid / (1.0 - safe_relative_effect)**2.5

    # Smooth Logarithmic Blending
    # Centers the transition at the critical crystal fraction.
    # Width is read from config (melt_transition_width); wider = smoother viscosity cliff.
    transition_width = rheo['melt_transition_width']
    blend = 0.5 * (1.0 + np.tanh((melt_fraction - (1.0 - critical_crystal_fraction)) / transition_width))
    
    # Smoothly interpolate across the orders of magnitude
    log_visc = (1.0 - blend) * np.log(visc_solid_regime) + blend * np.log(visc_liquid_regime)
    dynamic_viscosity = np.exp(log_visc)

    # --- Convert to Kinematic Viscosity ---
    kinematic_viscosity = dynamic_viscosity / density_mantle

    return kinematic_viscosity
