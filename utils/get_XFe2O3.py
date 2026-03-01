import numpy as np

def get_XFe2O3(
        temp_kelvin,
        pressure_pa,
        mole_fractions,
        fugacity_O2,
        params):
    """
    Analytically solves for the mole fraction of Fe2O3 in a silicate melt 
    given a target oxygen fugacity (fO2), using the inverse of the Kress 
    and Carmichael (1991) empirical parameterization.

    Parameters
    ----------
    temp_kelvin : float
        Temperature of the silicate melt in Kelvin.
    pressure_pa : float
        Pressure of the environment in Pascals.
    mole_fractions : ndarray
        1D array of oxide mole fractions (standard 12-element array from get_comp).
        Indices used:
        [2] Al2O3, [4] CaO, [5] Na2O, [6] K2O, [8] FeOt (Total Iron Oxide)
    fugacity_O2 : float
        Target oxygen fugacity in Pascals.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    frac_Fe2O3 : float
        The mole fraction of Fe2O3 in the melt required to produce the target fO2.
    """

    # --- Extract Parameters ---
    kc = params['planet']['oxygen_fugacity']['kress_carmichael_1991']

    # --- Extract Composition ---
    frac_Al2O3 = mole_fractions[2]
    frac_CaO   = mole_fractions[4]
    frac_Na2O  = mole_fractions[5]
    frac_K2O   = mole_fractions[6]
    frac_FeOt  = mole_fractions[8]  # Total Iron (expressed as FeO)

    # --- Static KC91 Thermodynamic Terms ---
    T0 = kc['temp_ref'] 
    
    term_temp  = kc['temp_coeff'] / temp_kelvin
    term_const = kc['constant_term']
    
    # Note: Negative signs are handled inherently by the TOML coefficients
    term_comp = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                 kc['coeff_FeO']   * frac_FeOt + 
                 kc['coeff_CaO']   * frac_CaO + 
                 kc['coeff_Na2O']  * frac_Na2O + 
                 kc['coeff_K2O']   * frac_K2O)
    
    term_temp_correction = kc['temp_correction_coeff'] * (1.0 - (T0 / temp_kelvin) - np.log(temp_kelvin / T0))
    
    term_pressure = (kc['press_coeff_1'] * pressure_pa / temp_kelvin + 
                     kc['press_coeff_2'] * (temp_kelvin - T0) * pressure_pa / temp_kelvin + 
                     kc['press_coeff_3'] * (pressure_pa**2) / temp_kelvin)

    # Sum of all non-iron components in the parameterization
    static_c_sum = term_temp + term_const + term_comp + term_temp_correction + term_pressure

    # --- Analytic Solution for Iron Ratio ---
    # KC91 forward equation: ln(fO2) = (1/scaling_a) * (ln(X_Fe2O3 / X_FeO) + static_c_sum)
    # Rearranged to solve for the ratio: ln(X_Fe2O3 / X_FeO) = scaling_a * ln(fO2) - static_c_sum
    
    ln_ferric_ferrous_ratio = kc['scaling_a'] * np.log(fugacity_O2) - static_c_sum
    ratio = np.exp(ln_ferric_ferrous_ratio)  # ratio = X_Fe2O3 / X_FeO

    # --- Stoichiometric Mass Balance ---
    # We know the total iron in the system: X_FeOt = X_FeO + 2 * X_Fe2O3
    # Substitute X_Fe2O3 = ratio * X_FeO:
    # X_FeOt = X_FeO + 2 * (ratio * X_FeO)
    # X_FeOt = X_FeO * (1 + 2 * ratio)
    
    # Solve for X_FeO
    frac_FeO = frac_FeOt / (1.0 + 2.0 * ratio)
    
    # Solve for X_Fe2O3
    frac_Fe2O3 = ratio * frac_FeO

    return frac_Fe2O3
