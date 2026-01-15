import numpy as np

def get_comp(param):
    '''
    get_comp.m

    Parameters
    ----------
    param : int
        Composition selector (currently only param == 1 implemented)

    Returns
    -------
    Xi : ndarray, shape (12,)
        Mole fractions:
        [0]  XiSiO2
        [1]  XiTiO2
        [2]  XiAl2O3
        [3]  XiMgO
        [4]  XiCaO
        [5]  XiNa2O
        [6]  XiK2O
        [7]  XiP2O5
        [8]  XiFeOt
        [9]  XiFeO1_5
        [10] XiFeO
        [11] XiFe2O3
    '''

    # molecular weights (g/mol)
    MW_SiO2 = 60.084
    MW_TiO2 = 79.866
    MW_Al2O3 = 101.961
    MW_FeO = 71.845
    MW_FeO1_5 = 159.689 / 2.0
    MW_MgO = 40.304
    MW_CaO = 56.078
    MW_Na2O = 61.979
    MW_K2O = 94.196
    MW_P2O5 = 141.945

    if param == 1:
        # Bulk Silicate Earth (mass fractions)
        SiO2 = 0.4597
        TiO2 = 0.012
        Al2O3 = 0.0477
        FeOt = 0.08
        MgO = 0.3666
        CaO = 0.0378
        Na2O = 0.0035
        K2O = 0.0004
        P2O5 = 0.002
        Fe3_Fet = 1e-12
    else:
        raise ValueError('Unsupported composition parameter')

    # moles of oxides
    nSiO2 = SiO2 / MW_SiO2
    nTiO2 = TiO2 / MW_TiO2
    nAl2O3 = Al2O3 / MW_Al2O3
    nFeOt = FeOt / MW_FeO
    nMgO = MgO / MW_MgO
    nCaO = CaO / MW_CaO
    nNa2O = Na2O / MW_Na2O
    nK2O = K2O / MW_K2O
    nP2O5 = P2O5 / MW_P2O5

    total_moles = (
        nSiO2 + nTiO2 + nAl2O3 + nFeOt +
        nMgO + nCaO + nNa2O + nK2O + nP2O5
    )

    # mole fractions
    Xi = np.zeros(12)

    Xi[0] = nSiO2 / total_moles          # XiSiO2
    Xi[1] = nTiO2 / total_moles          # XiTiO2
    Xi[2] = nAl2O3 / total_moles         # XiAl2O3
    Xi[3] = nMgO / total_moles           # XiMgO
    Xi[4] = nCaO / total_moles           # XiCaO
    Xi[5] = nNa2O / total_moles          # XiNa2O
    Xi[6] = nK2O / total_moles           # XiK2O
    Xi[7] = nP2O5 / total_moles          # XiP2O5
    Xi[8] = nFeOt / total_moles          # XiFeOt
    Xi[9] = Xi[8] * Fe3_Fet               # XiFeO1_5
    Xi[10] = Xi[8] - Xi[9]                 # XiFeO
    Xi[11] = Xi[9]                        # XiFe2O3

    return Xi
