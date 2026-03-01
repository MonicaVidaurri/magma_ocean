import numpy as np

def get_comp(composition_model=1):
    """
    Returns the initial molar oxide composition of the planetary mantle.

    This function takes mass fractions of major oxides, converts them to 
    molar abundances using standard molecular weights, and normalizes them 
    into a 12-element mole fraction array. The array is structured to interface 
    directly with the oxygen mass balance and fO2 solvers.

    Parameters
    ----------
    composition_model : int, optional
        Selector for the bulk mantle composition. 
        Currently only `1` (Bulk Silicate Earth) is implemented. Default is 1.

    Returns
    -------
    mole_fractions : ndarray
        A 1D array of length 12 containing the mole fractions of the mantle:
        [0]  SiO2
        [1]  TiO2
        [2]  Al2O3
        [3]  MgO
        [4]  CaO
        [5]  Na2O
        [6]  K2O
        [7]  P2O5
        [8]  FeOt   (Total Iron Oxide)
        [9]  FeO1.5 (Ferric Iron, single-Fe basis)
        [10] FeO    (Ferrous Iron)
        [11] Fe2O3  (Ferric Iron, standard basis)
    """

    # --- Molar Masses (g/mol) ---
    molar_mass_SiO2  = 60.084
    molar_mass_TiO2  = 79.866
    molar_mass_Al2O3 = 101.961
    molar_mass_MgO   = 40.304
    molar_mass_CaO   = 56.078
    molar_mass_Na2O  = 61.979
    molar_mass_K2O   = 94.196
    molar_mass_P2O5  = 141.945
    
    molar_mass_FeO    = 71.845
    molar_mass_FeO1_5 = 159.689 / 2.0   # Equivalent to half of Fe2O3

    # --- Initial Mass Fractions (kg/kg) ---
    if composition_model == 1:
        # Standard Bulk Silicate Earth (BSE) composition
        mass_frac_SiO2  = 0.4597
        mass_frac_TiO2  = 0.012
        mass_frac_Al2O3 = 0.0477
        mass_frac_MgO   = 0.3666
        mass_frac_CaO   = 0.0378
        mass_frac_Na2O  = 0.0035
        mass_frac_K2O   = 0.0004
        mass_frac_P2O5  = 0.002
        
        mass_frac_FeOt = 0.08
        
        # Initial oxidation state (Highly reduced initial magma ocean)
        ratio_Fe3_to_total_Fe = 1e-12
    else:
        raise NotImplementedError(f"Unsupported composition model: {composition_model}. Only model 1 (BSE) is implemented.")

    # --- Convert Mass Fractions to Moles (mol per gram of bulk rock) ---
    moles_SiO2  = mass_frac_SiO2 / molar_mass_SiO2
    moles_TiO2  = mass_frac_TiO2 / molar_mass_TiO2
    moles_Al2O3 = mass_frac_Al2O3 / molar_mass_Al2O3
    moles_MgO   = mass_frac_MgO / molar_mass_MgO
    moles_CaO   = mass_frac_CaO / molar_mass_CaO
    moles_Na2O  = mass_frac_Na2O / molar_mass_Na2O
    moles_K2O   = mass_frac_K2O / molar_mass_K2O
    moles_P2O5  = mass_frac_P2O5 / molar_mass_P2O5
    
    # Total iron is initially treated entirely as FeO for the bulk molar calculation
    moles_FeOt = mass_frac_FeOt / molar_mass_FeO

    total_moles = (
        moles_SiO2 + moles_TiO2 + moles_Al2O3 + moles_FeOt +
        moles_MgO + moles_CaO + moles_Na2O + moles_K2O + moles_P2O5
    )

    # --- Calculate Normalized Mole Fractions ---
    mole_fractions = np.zeros(12, dtype=np.float64)

    mole_fractions[0] = moles_SiO2 / total_moles
    mole_fractions[1] = moles_TiO2 / total_moles
    mole_fractions[2] = moles_Al2O3 / total_moles
    mole_fractions[3] = moles_MgO / total_moles
    mole_fractions[4] = moles_CaO / total_moles
    mole_fractions[5] = moles_Na2O / total_moles
    mole_fractions[6] = moles_K2O / total_moles
    mole_fractions[7] = moles_P2O5 / total_moles
    
    # Iron Speciation 
    mole_fractions[8]  = moles_FeOt / total_moles                   # Total Fe
    mole_fractions[9]  = mole_fractions[8] * ratio_Fe3_to_total_Fe  # FeO1.5 (Fe3+)
    mole_fractions[10] = mole_fractions[8] - mole_fractions[9]      # FeO (Fe2+)
    
    # FIX: Strictly speaking, moles of Fe2O3 = 0.5 * moles of FeO1.5. 
    #  this was not the case in the original matlab code.
    mole_fractions[11] = 0.5 * mole_fractions[9]                                   

    return mole_fractions