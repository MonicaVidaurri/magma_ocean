import numpy as np

def get_XFe2O3(
        temp_kelvin,
        pressure_pa,
        mole_fractions,
        fugacity_O2,
        params):
    """
    Analytically solves for the mole fraction of Fe2O3 in a silicate melt 
    given a target oxygen fugacity (fO2), using the inverse KC91 model.

    Parameters
    ----------
    temp_kelvin : float
        Temperature of the silicate melt (K).
    pressure_pa : float
        Pressure of the environment (Pa).
    mole_fractions : ndarray
        1D array of oxide mole fractions.
    fugacity_O2 : float
        Target oxygen fugacity (Pa).
    params : dict
        Master configuration dictionary.

    Returns
    -------
    frac_Fe2O3 : float
        The mole fraction of Fe2O3 required to produce the target fO2.
    """

    # --- Unpack Parameters ---
    kc = params['planet']['oxygen_fugacity']['kress_carmichael_1991']

    # --- Extract Composition ---
    # Indices: [2]Al2O3, [4]CaO, [5]Na2O, [6]K2O, [8]FeOt
    frac_Al2O3 = mole_fractions[2]
    frac_CaO   = mole_fractions[4]
    frac_Na2O  = mole_fractions[5]
    frac_K2O   = mole_fractions[6]
    frac_FeOt  = mole_fractions[8]

    # --- Static KC91 Thermodynamic Terms ---
    T0 = kc['T0_ref']
    
    term_temp  = kc['temp_coeff'] / temp_kelvin
    term_const = kc['constant_term']
    
    # Composition dependence
    term_comp = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                 kc['coeff_FeOt']  * frac_FeOt  + 
                 kc['coeff_CaO']   * frac_CaO   + 
                 kc['coeff_Na2O']  * frac_Na2O  + 
                 kc['coeff_K2O']   * frac_K2O)
    
    # Temperature deviation correction
    term_temp_corr = kc['temp_correction_coeff'] * (1.0 - (T0 / temp_kelvin) - np.log(temp_kelvin / T0))
    
    # Pressure dependence
    term_pressure = (
        kc['press_coeff_1'] * pressure_pa / temp_kelvin + 
        kc['press_coeff_2'] * (temp_kelvin - T0) * pressure_pa / temp_kelvin + 
        kc['press_coeff_3'] * (pressure_pa**2) / temp_kelvin
    )

    # Sum of all non-iron parameters
    static_c_sum = term_temp + term_const + term_comp + term_temp_corr + term_pressure

    # --- Analytic Solution for Iron Ratio ---
    # Invert the fO2 equation to solve for the ratio: R = X_Fe2O3 / X_FeO
    ln_ferric_ferrous_ratio = kc['scaling_a'] * np.log(fugacity_O2) - static_c_sum
    ratio = np.exp(ln_ferric_ferrous_ratio) 

    # --- Stoichiometric Mass Balance ---
    # Total Fe is conserved: X_FeOt = X_FeO + 2 * X_Fe2O3
    # X_FeOt = X_FeO * (1 + 2 * ratio)
    frac_FeO = frac_FeOt / (1.0 + 2.0 * ratio)
    
    # Final Fe2O3 mole fraction
    frac_Fe2O3 = ratio * frac_FeO

    return frac_Fe2O3
