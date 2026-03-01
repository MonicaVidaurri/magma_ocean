import numpy as np

def get_comp(params):
    """
    Returns the initial molar oxide composition of the planetary mantle 
    using parameters from a nested TOML dictionary.

    Parameters
    ----------
    params : dict
        Master configuration dictionary containing [planet][oxide_composition].

    Returns
    -------
    mole_fractions : ndarray
        1D array of length 12 containing the normalized mole fractions.
    """
    
    # --- Unpack Sub-dictionary ---
    comp = params['planet']['oxide_composition']

    # --- Calculate Moles ---
    # Convert Mass Fractions (kg/kg) to Moles per kg of bulk rock
    n_SiO2   = comp['mass_frac_SiO2']   / comp['molar_mass_SiO2']
    n_TiO2   = comp['mass_frac_TiO2']   / comp['molar_mass_TiO2']
    n_Al2O3  = comp['mass_frac_Al2O3']  / comp['molar_mass_Al2O3']
    n_MgO    = comp['mass_frac_MgO']    / comp['molar_mass_MgO']
    n_CaO    = comp['mass_frac_CaO']    / comp['molar_mass_CaO']
    n_Na2O   = comp['mass_frac_Na2O']   / comp['molar_mass_Na2O']
    n_K2O    = comp['mass_frac_K2O']    / comp['molar_mass_K2O']
    n_P2O5   = comp['mass_frac_P2O5']   / comp['molar_mass_P2O5']
    n_FeOt   = comp['mass_frac_FeOt']   / comp['molar_mass_FeO']

    # Special handling for single-cation basis (AlO1.5, NaO0.5, etc.) 
    # to maintain consistency with fO2 models
    n_AlO1_5 = 2.0 * n_Al2O3
    n_NaO0_5 = 2.0 * n_Na2O
    n_KO0_5  = 2.0 * n_K2O
    n_PO2_5  = 2.0 * n_P2O5

    # Total bulk moles
    total_moles = (n_SiO2 + n_TiO2 + n_AlO1_5 + n_FeOt + 
                   n_MgO + n_CaO + n_NaO0_5 + n_KO0_5 + n_PO2_5)

    # --- Normalize into 12-element array ---
    mole_fractions = np.zeros(12, dtype=np.float64)

    mole_fractions[0] = n_SiO2   / total_moles
    mole_fractions[1] = n_TiO2   / total_moles
    mole_fractions[2] = n_AlO1_5 / total_moles # Note: Stored as single-Al basis
    mole_fractions[3] = n_MgO    / total_moles
    mole_fractions[4] = n_CaO    / total_moles
    mole_fractions[5] = n_NaO0_5 / total_moles
    mole_fractions[6] = n_KO0_5  / total_moles
    mole_fractions[7] = n_PO2_5  / total_moles
    
    # Iron Speciation 
    x_feot = n_FeOt / total_moles
    mole_fractions[8]  = x_feot                                      # Total Fe
    mole_fractions[9]  = x_feot * comp['ratio_Fe3_to_total_Fe']      # FeO1.5 (Fe3+)
    mole_fractions[10] = mole_fractions[8] - mole_fractions[9]       # FeO (Fe2+)
    mole_fractions[11] = 0.5 * mole_fractions[9]                     # Fe2O3

    return mole_fractions
