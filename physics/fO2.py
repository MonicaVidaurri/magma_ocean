import numpy as np

# temperature (K) and pressure (GPa)
T = 2000.0
P = 0.1

# composition by mass
SiO2 = 0.4597
TiO2 = 0.012
Al2O3 = 0.0477
AlO1_5 = Al2O3 * 2.0
FeOt = 0.0824
MgO = 0.3666
CaO = 0.0378
Na2O = 0.0035
NaO1_5 = Na2O * 2.0
K2O = 0.0004
KO1_5 = K2O * 2.0
P2O5 = 0.001
PO2_5 = P2O5 * 2.0

# Fe3+/FeT ratios
Fe3_Fet = np.concatenate(([0.01], np.arange(0.1, 1.0, 0.1), [0.99]))

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

# moles of oxides
nSiO2 = SiO2 / MW_SiO2
nTiO2 = TiO2 / MW_TiO2
nAl2O3 = Al2O3 / MW_Al2O3
nAlO1_5 = 2.0 * nAl2O3
nFeOt = FeOt / MW_FeO
nMgO = MgO / MW_MgO
nCaO = CaO / MW_CaO
nNa2O = Na2O / MW_Na2O
nNaO1_5 = 2.0 * nNa2O
nK2O = K2O / MW_K2O
nKO1_5 = 2.0 * nK2O
nP2O5 = P2O5 / MW_P2O5
nPO2_5 = 2.0 * nP2O5

# total moles
total_moles = (nSiO2 + nTiO2 + nAlO1_5 + nFeOt + nMgO + nCaO + nNaO1_5 + nKO1_5 + nPO2_5)

# mole fractions
XSiO2 = nSiO2 / total_moles
XTiO2 = nTiO2 / total_moles
XAlO1_5 = nAlO1_5 / total_moles
XFeOt = nFeOt / total_moles

XFeO1_5 = XFeOt * Fe3_Fet
XFeO = XFeOt - XFeO1_5
XFe2O3 = 0.5 * XFeO1_5

XMgO = nMgO / total_moles
XCaO = nCaO / total_moles
XNaO1_5 = nNaO1_5 / total_moles
XKO1_5 = nKO1_5 / total_moles
XPO2_5 = nPO2_5 / total_moles

# log10 oxygen fugacity
logfO2 = (4.0 * np.log10(XFeO1_5 / XFeO) - 28144.0 / T + 13.95
    + 3905.0 * XMgO / T
    - 13359.0 * XCaO / T
    - 14858.0 * XNaO1_5 / T
    - 9805.0 * XKO1_5 / T
    + 10906.0 * XAlO1_5 / T
    + 110971.0 * XPO2_5 / T
    - 11952.0 * (XFeO - XFeO1_5) / T
    + (33122.0 / T - 5.24) * ((1.0 + 0.241 * P) ** (3.0 / 4.0) - 1.0)
    - (39156.0 / T - 6.17) * ((1.0 + 0.132 * P) ** (3.0 / 4.0) - 1.0))

# logfO2 is an array corresponding to Fe3_Fet
