import numpy as np

def viscosity(
        temp_mantle,
        temp_surface,
        density_mantle,
        mass_frac_water,
        melt_fraction):
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

    Returns
    -------
    kinematic_viscosity : float
        The kinematic viscosity of the mantle or magma ocean in m^2/s.
    """

    # --- Constants ---
    gas_constant = 8.31447  # J/(mol K)

    # --- Liquid Magma Viscosity (Dynamic, Pa·s) ---
    # Empirical parameters for silicate melt viscosity
    A_coeff_liquid        = 0.00024  # Pa s
    B_coeff_liquid        = 4600.0   # K
    temp_reference_liquid = 1000.0   # K

    dynamic_visc_liquid = A_coeff_liquid * np.exp(B_coeff_liquid / (temp_mantle - temp_reference_liquid))

    # --- Solid Mantle Viscosity (Dynamic, Pa·s) ---
    # Arrhenius parameters for solid-state creep
    pre_exponential_solid = 3.7489e9  # Pa s
    activation_energy     = 350e3     # J/mol

    dynamic_visc_solid = pre_exponential_solid * np.exp(activation_energy / (gas_constant * temp_mantle))

    # --- Rheological Regime Switch ---
    # The rheology transitions at a critical crystal fraction (phi_c ~ 0.6).
    # Crystal fraction is (1.0 - melt_fraction).
    crystal_fraction          = 1.0 - melt_fraction
    critical_crystal_fraction = 0.6

    # Constrain melt fraction to physical bounds
    melt_fraction = np.clip(melt_fraction, 0.0, 1.0)

    if crystal_fraction < critical_crystal_fraction:
        # Liquid-supported regime (Magma Ocean)
        # Uses a Roscoe-style formulation for the viscosity of a crystal suspension
        relative_solid_effect = crystal_fraction / critical_crystal_fraction
        dynamic_viscosity     = dynamic_visc_liquid / (1.0 - relative_solid_effect)**2.5
        
    else:
        # Matrix-supported regime (Solid Mantle)
        # Melt weakens the solid matrix exponentially
        melt_weakening_factor = 26.0
        dynamic_viscosity     = dynamic_visc_solid * np.exp(-melt_weakening_factor * melt_fraction)

    # --- Convert to Kinematic Viscosity ---
    kinematic_viscosity = dynamic_viscosity / density_mantle

    return kinematic_viscosity
