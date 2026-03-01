import numpy as np

def get_fO2(temp_kelvin, pressure_pa, mole_fractions, params):
    """
    Calculates the oxygen fugacity (fO2) of a silicate melt based on the 
    Kress and Carmichael (1991) empirical parameterization.
    
    Parameters
    ----------
    temp_kelvin : float
        Mantle temperature (K).
    pressure_pa : float
        Local pressure (Pa).
    mole_fractions : ndarray
        1D array of oxide mole fractions.
    params : dict
        Master configuration dictionary containing [planet.oxygen_fugacity.kress_carmichael_1991].

    Returns
    -------
    fugacity_O2 : float
        Oxygen fugacity in Pascals.
    """
    
    # --- Unpack Parameters & Composition ---
    kc_params = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    
    # Indices: [2]Al2O3, [4]CaO, [5]Na2O, [6]K2O, [8]FeOt, [10]FeO, [11]Fe2O3
    frac_Al2O3 = mole_fractions[2]
    frac_CaO   = mole_fractions[4]
    frac_Na2O  = mole_fractions[5]
    frac_K2O   = mole_fractions[6]
    frac_FeOt  = mole_fractions[8]
    frac_FeO   = mole_fractions[10]
    frac_Fe2O3 = mole_fractions[11]

    # --- Iron Ratio ---
    # Numerical safety for log calculations
    safe_Fe2O3 = max(frac_Fe2O3, 1e-20)
    safe_FeO   = max(frac_FeO, 1e-20)
    ln_ferric_ferrous_ratio = np.log(safe_Fe2O3 / safe_FeO)
    
    # --- Model Terms ---
    T0 = kc_params['T0_ref']
    
    # Temperature dependence
    term_temp = kc_params['temp_coeff'] / temp_kelvin
    term_const = kc_params['constant_term']
    
    # Composition dependence
    # Signs are handled by the coefficients in the TOML
    term_comp = (kc_params['coeff_Al2O3'] * frac_Al2O3 + 
                 kc_params['coeff_FeOt']  * frac_FeOt  + 
                 kc_params['coeff_CaO']   * frac_CaO   + 
                 kc_params['coeff_Na2O']  * frac_Na2O  + 
                 kc_params['coeff_K2O']   * frac_K2O)
    
    # Temperature deviation correction
    term_temp_correction = kc_params['temp_correction_coeff'] * (
        1.0 - (T0 / temp_kelvin) - np.log(temp_kelvin / T0)
    )
    
    # Pressure dependence
    term_pressure = (
        kc_params['press_coeff_1'] * pressure_pa / temp_kelvin + 
        kc_params['press_coeff_2'] * (temp_kelvin - T0) * pressure_pa / temp_kelvin + 
        kc_params['press_coeff_3'] * (pressure_pa**2) / temp_kelvin
    )
    
    # --- Final Fugacity ---
    ln_fO2 = (1.0 / kc_params['scaling_a']) * (
        ln_ferric_ferrous_ratio + term_temp + term_const + 
        term_comp + term_temp_correction + term_pressure
    )
    
    return np.exp(ln_fO2)
