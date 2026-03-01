import numpy as np

def get_fO2_borisov(
        temp_kelvin,
        pressure_gpa,
        fe3_fet_ratio,
        params):
    """
    Calculates log10 oxygen fugacity using an alternative empirical 
    parameterization (e.g., Borisov et al.).
    
    Parameters
    ----------
    temp_kelvin : float
        Temperature in Kelvin.
    pressure_gpa : float
        Pressure in GPa.
    fe3_fet_ratio : float or ndarray
        The ratio of Ferric iron to Total iron (Fe3+/FeT).
    params : dict
        The master configuration dictionary.
    """
    
    # --- Extract Composition and Weights ---
    comp = params['planet']['oxide_composition']
    
    # Calculate moles (using weights from TOML)
    n_sio2 = comp['mass_frac_SiO2'] / comp['molar_mass_SiO2']
    n_tio2 = comp['mass_frac_TiO2'] / comp['molar_mass_TiO2']
    n_alo1_5 = 2.0 * (comp['mass_frac_Al2O3'] / comp['molar_mass_Al2O3'])
    n_feot = comp['mass_frac_FeOt'] / comp['molar_mass_FeO']
    n_mgo = comp['mass_frac_MgO'] / comp['molar_mass_MgO']
    n_cao = comp['mass_frac_CaO'] / comp['molar_mass_CaO']
    n_nao1_5 = 2.0 * (comp['mass_frac_Na2O'] / comp['molar_mass_Na2O'])
    n_ko1_5 = 2.0 * (comp['mass_frac_K2O'] / comp['molar_mass_K2O'])
    n_po2_5 = 2.0 * (comp['mass_frac_P2O5'] / comp['molar_mass_P2O5'])

    total_moles = (n_sio2 + n_tio2 + n_alo1_5 + n_feot + 
                   n_mgo + n_cao + n_nao1_5 + n_ko1_5 + n_po2_5)

    # --- Calculate Mole Fractions ---
    x_alo1_5 = n_alo1_5 / total_moles
    x_feot   = n_feot / total_moles
    x_mgo    = n_mgo / total_moles
    x_cao    = n_cao / total_moles
    x_nao1_5 = n_nao1_5 / total_moles
    x_ko1_5  = n_ko1_5 / total_moles
    x_po2_5  = n_po2_5 / total_moles

    # Iron Speciation based on input ratio
    x_feo1_5 = x_feot * fe3_fet_ratio
    x_feo    = x_feot - x_feo1_5

    # --- Compute log10(fO2) ---
    coeffs = params['planet']['oxygen_fugacity']['borisov_2013']
    T = temp_kelvin
    P = pressure_gpa

    # Base log terms
    log_term = 4.0 * np.log10(x_feo1_5 / x_feo) - coeffs['const_1'] / T + coeffs['const_2']
    
    # Compositional effects
    comp_term = (
        coeffs['coeff_MgO'] * x_mgo / T +
        coeffs['coeff_CaO'] * x_cao / T +
        coeffs['coeff_NaO1_5'] * x_nao1_5 / T +
        coeffs['coeff_KO1_5'] * x_ko1_5 / T +
        coeffs['coeff_AlO1_5'] * x_alo1_5 / T +
        coeffs['coeff_PO2_5'] * x_po2_5 / T +
        coeffs['coeff_Fe_diff'] * (x_feo - x_feo1_5) / T
    )

    # Pressure effects
    p_term_1 = (coeffs['p_term1_a'] / T - coeffs['p_term1_b']) * \
               ((1.0 + coeffs['p_term1_c'] * P) ** 0.75 - 1.0)
    
    p_term_2 = (coeffs['p_term2_a'] / T - coeffs['p_term2_b']) * \
               ((1.0 + coeffs['p_term2_c'] * P) ** 0.75 - 1.0)

    logfO2 = log_term + comp_term + p_term_1 - p_term_2

    return logfO2