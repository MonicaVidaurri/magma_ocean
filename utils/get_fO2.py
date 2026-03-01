import numpy as np

def get_fO2(
        temp_kelvin,
        pressure_pa,
        mole_fractions,
        params):
    """
    Calculates the oxygen fugacity (fO2) of a silicate melt based on its 
    composition, temperature, and pressure using the Kress and Carmichael (1991) 
    empirical parameterization.
    
    Parameters
    ----------
    temp_kelvin : float
        Temperature of the silicate melt in Kelvin.
    pressure_pa : float
        Pressure of the environment in Pascals.
    mole_fractions : ndarray
        1D array of oxide mole fractions. 
        Indices map as follows:
        [0]  SiO2    [1]  TiO2    [2]  Al2O3
        [3]  MgO     [4]  CaO     [5]  Na2O
        [6]  K2O     [7]  P2O5    [8]  FeOt
        [9]  FeO1.5  [10] FeO     [11] Fe2O3
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    fugacity_O2 : float
        Oxygen fugacity in Pascals (linear scale).
    """

    # --- Unpack Parameters ---
    kc = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    
    # --- Unpack Composition Array ---
    # Extract only the oxides actively used in the Kress & Carmichael calibration
    frac_Al2O3 = mole_fractions[2]
    frac_CaO   = mole_fractions[4]
    frac_Na2O  = mole_fractions[5]
    frac_K2O   = mole_fractions[6]
    frac_FeOt  = mole_fractions[8]
    frac_FeO   = mole_fractions[10]  # Ferrous Iron (Fe2+)
    frac_Fe2O3 = mole_fractions[11]  # Ferric Iron (Fe3+)

    # --- Iron Ratio ---
    # Numerical safety: Clamp iron fractions to prevent log(0) -> -inf crashes 
    # in perfectly reduced or perfectly oxidized extreme conditions.
    safe_Fe2O3 = max(frac_Fe2O3, 1e-20)
    safe_FeO   = max(frac_FeO, 1e-20)
    
    ln_ferric_ferrous_ratio = np.log(safe_Fe2O3 / safe_FeO)
    
    # --- Kress & Carmichael (1991) Model Terms ---
    # Reference Temperature (K)
    T0 = kc['temp_ref'] 
    
    # Temperature dependence
    term_temp  = kc['temp_coeff'] / temp_kelvin
    term_const = kc['constant_term']
    
    # Composition dependence (d_i coefficients)
    # Note: Negative coefficients are naturally handled by the TOML values
    term_comp = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                 kc['coeff_FeO']   * frac_FeOt + 
                 kc['coeff_CaO']   * frac_CaO + 
                 kc['coeff_Na2O']  * frac_Na2O + 
                 kc['coeff_K2O']   * frac_K2O)
    
    # Temperature deviation correction
    term_temp_correction = kc['temp_correction_coeff'] * (1.0 - (T0 / temp_kelvin) - np.log(temp_kelvin / T0))
    
    # Pressure dependence (calibrated in Pascals)
    term_pressure = (kc['press_coeff_1'] * pressure_pa / temp_kelvin + 
                     kc['press_coeff_2'] * (temp_kelvin - T0) * pressure_pa / temp_kelvin + 
                     kc['press_coeff_3'] * (pressure_pa**2) / temp_kelvin)
    
    # --- Final Fugacity Calculation ---
    # Combine terms and scale by the empirical 'a' parameter to isolate ln(fO2)
    ln_fO2 = (1.0 / kc['scaling_a']) * (ln_ferric_ferrous_ratio + term_temp + term_const + 
                                        term_comp + term_temp_correction + term_pressure)
    
    fugacity_O2 = np.exp(ln_fO2)
    
    return fugacity_O2
