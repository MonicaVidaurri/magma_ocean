import numpy as np

def viscosity2(
        temp_mantle,
        mass_frac_water,
        gravity,
        density_mantle,
        pressure_pa,
        params):
    """
    Calculates the kinematic viscosity of the mantle incorporating water-weakening 
    effects, following the parameterization of Sandu et al. (2011) and Li et al. (2008).

    This model determines the concentration of hydroxyl (OH) defects in an olivine 
    matrix, calculates the corresponding water fugacity, and applies it to an 
    Arrhenius creep law.

    Parameters
    ----------
    temp_mantle : float
        The potential temperature of the mantle in Kelvin.
    mass_frac_water : float
        The bulk mass fraction of water in the mantle (dimensionless).
    gravity : float
        Surface gravitational acceleration in m/s^2. (Currently unused).
    density_mantle : float
        Bulk density of the mantle in kg/m^3.
    pressure_pa : float
        Local mantle pressure in Pascals.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    kinematic_viscosity : float
        The kinematic viscosity of the mantle in m^2/s.
    """

    # --- Unpack Parameters ---
    univ  = params['constants']
    mat   = params['planet']['material']
    sandu = params['planet']['rheology']['sandu_2011']
    
    # --- Constants & Molar Masses ---
    gas_constant   = univ['gas_constant']    # J/(mol K)
    molar_mass_H2O = univ['molar_mass_H2O']  # kg/mol

    # Olivine solid solution properties: (Mg,Fe)2SiO4
    fraction_forsterite   = mat['fraction_forsterite']
    fraction_fayalite     = mat['fraction_fayalite']
    molar_mass_forsterite = mat['molar_mass_forsterite']  # kg/mol
    molar_mass_fayalite   = mat['molar_mass_fayalite']    # kg/mol

    # Bulk molar mass of the idealized olivine mantle
    molar_mass_olivine = (fraction_forsterite * molar_mass_forsterite + 
                          fraction_fayalite * molar_mass_fayalite)

    # --- Water Concentration Conversion ---
    # Numerical safety: Prevent log(0) crash if the mantle completely desiccates
    mass_frac_water = max(mass_frac_water, 1e-12)

    # Convert bulk water mass fraction to ppm H/Si (H atoms per 10^6 Si atoms)
    # Assumes 1 mole of olivine contains exactly 1 mole of Si.
    # Each H2O molecule contributes 2 H atoms.
    concentration_OH = (mass_frac_water * molar_mass_olivine * 1e6 * 2.0) / molar_mass_H2O

    # --- Water Fugacity (Li et al., 2008) ---
    # Empirical polynomial fit linking OH concentration to water fugacity
    c0 = sandu['fugacity_poly_c0']
    c1 = sandu['fugacity_poly_c1']
    c2 = sandu['fugacity_poly_c2']
    c3 = sandu['fugacity_poly_c3']

    ln_C_OH = np.log(concentration_OH)
    ln_fugacity_H2O = c0 + (c1 * ln_C_OH) + (c2 * ln_C_OH**2) + (c3 * ln_C_OH**3)
    fugacity_H2O = np.exp(ln_fugacity_H2O)

    # --- Dynamic Viscosity Calculation ---
    # Rheological parameters for wet olivine dislocation creep (Sandu et al., 2011)
    visc_reference    = sandu['visc_reference']    # eta_0 (Pa s), calibration constant
    fugacity_exponent = sandu['fugacity_exponent'] # r
    activation_energy = sandu['activation_energy'] # Qa (J/mol)
    
    # Activation volume is set to 0.0, rendering pressure effects negligible in this regime
    activation_volume = sandu['activation_volume'] # V (m^3/mol) 

    # Arrhenius exponent
    arrhenius_term = np.exp((activation_energy + pressure_pa * activation_volume) / (gas_constant * temp_mantle))

    # Dynamic viscosity (Pa s) incorporating the water fugacity weakening term
    dynamic_viscosity = visc_reference * (fugacity_H2O**-fugacity_exponent) * arrhenius_term

    # --- Kinematic Viscosity ---
    kinematic_viscosity = dynamic_viscosity / density_mantle

    return kinematic_viscosity
