import numpy as np
from scipy.optimize import fsolve

def get_XFeO(Tm, P, Mmo, Rp, g, Xi, MO2_mo):
    '''
    Solve for the oxygen fugacity (fO2) and XFeO in equilibrium with a magma ocean.

    Parameters
    ----------
    Tm : float
        Mantle temperature [K]
    P : float
        Pressure [Pa]
    Mmo : float
        Magma ocean mass [kg]
    Rp : float
        Planet radius [m]
    g : float
        Surface gravity [m/s^2]
    Xi : array_like
        Mole fractions of oxides, 0-based indexing:
        Xi[0] = XSiO2, Xi[1] = XTiO2, Xi[2] = XAl2O3, Xi[3] = XMgO,
        Xi[4] = XCaO, Xi[5] = XNa2O, Xi[6] = XK2O, Xi[7] = XP2O5,
        Xi[8] = XFeOt, Xi[9] = XFeO1_5, Xi[10] = XFeO, Xi[11] = XFe2O3
    MO2_mo : float
        Moles of O2 in magma ocean

    Returns
    -------
    fO2 : float
        Oxygen fugacity [Pa]
    XFeO : float
        Mole fraction of FeO
    '''

    muO2 = 15.9994e-3 * 2  # kg/mol

    PO2 = MO2_mo * g / (4 * np.pi * Rp**2)
    x0 = [PO2 * 0.9, Xi[10]]  # initial guess [fO2, XFeO]

    def get_fun(guess):
        fO2_guess, XFeO_guess = guess
        term1 = (1 / 0.196 * (np.log((Xi[8] - XFeO_guess) / XFeO_guess) - 1.1492e4 / Tm + 6.675
                + 2.243 * Xi[2] + 1.828 * Xi[8] - 3.201 * Xi[4] - 5.854 * Xi[5] - 6.215 * Xi[6]
                + 3.36 * (1 - 1673 / Tm - np.log(Tm / 1673)) + 7.01e-7 * P / Tm
                + 1.54e-10 * (Tm - 1673) * P / Tm - 3.85e-17 * P**2 / Tm))
        fun1 = np.log(fO2_guess) - term1
        fun2 = MO2_mo - (4 * np.pi * Rp**2 * fO2_guess / g) - (Xi[8] - XFeO_guess - Xi[11]) * 0.25 * muO2 * Mmo
        return [fun1, fun2]

    sol = fsolve(get_fun, x0)

    # ensure fO2 is positive and real
    while sol[0] <= 0 or not np.isreal(sol[0]):
        x0 = [x0[0] * 0.8, 0.8 * x0[1]]
        sol = fsolve(get_fun, x0)

    fO2, XFeO = sol[0], sol[1]
    return fO2, XFeO
