import numpy as np

def get_fO2(T, P, Xi):
    """
    Calculates the oxygen fugacity (fO2) of a silicate melt based on oxide mole fractions.
    
    Parameters
    ----------
    T : float
        Temperature (Kelvin).
    P : float
        Pressure (Pascals).
    Xi : ndarray
        1D array of mole fractions of oxides. 
        Indices map as follows (0-based Python indexing):
        Xi[0]  : SiO2
        Xi[1]  : TiO2
        Xi[2]  : Al2O3
        Xi[3]  : MgO
        Xi[4]  : CaO
        Xi[5]  : Na2O
        Xi[6]  : K2O
        Xi[7]  : P2O5
        Xi[8]  : FeOt
        Xi[9]  : FeO1.5 (intermediate, not used in formula)
        Xi[10] : FeO
        Xi[11] : Fe2O3 (or FeO1.5 equivalent for ratio calculation)

    Returns
    -------
    fO2 : float
        Oxygen fugacity (linear scale, not log).
    """
    
    # --- Index Mapping (MATLAB -> Python) ---
    # Xi(12) -> Xi[11] (Fe2O3)
    # Xi(11) -> Xi[10] (FeO)
    # Xi(3)  -> Xi[2]  (Al2O3)
    # Xi(9)  -> Xi[8]  (FeOt)
    # Xi(5)  -> Xi[4]  (CaO)
    # Xi(6)  -> Xi[5]  (Na2O)
    # Xi(7)  -> Xi[6]  (K2O)

    # --- Calculation ---
    
    # Log ratio of Ferric to Ferrous Iron
    # We add a tiny epsilon to the denominator if Xi[10] is 0 to prevent division by zero, 
    # though valid physical inputs should not be 0.
    term_ratio = np.log(Xi[11] / (Xi[10] + 1e-20))
    
    # Temperature dependent terms
    term_temp = -1.1492e4 / T
    term_const = 6.675
    
    # Composition dependent terms
    term_comp = (2.243 * Xi[2] + 
                 1.828 * Xi[8] - 
                 3.201 * Xi[4] - 
                 5.854 * Xi[5] - 
                 6.215 * Xi[6])
    
    # Correction term for Temperature deviation from reference
    term_correction = 3.36 * (1.0 - 1673.0 / T - np.log(T / 1673.0))
    
    # Pressure dependent terms
    term_pressure = (7.01e-7 * P / T + 
                     1.54e-10 * (T - 1673.0) * P / T - 
                     3.85e-17 * (P**2) / T)
    
    # Combine terms to get ln(fO2)
    # Factor 1/0.196 scales the result (likely related to Fe oxidation parameterization)
    ln_fO2 = (1.0 / 0.196) * (term_ratio + term_temp + term_const + 
                              term_comp + term_correction + term_pressure)
    
    fO2 = np.exp(ln_fO2)
    
    return fO2