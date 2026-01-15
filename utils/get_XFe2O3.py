import numpy as np
from scipy.optimize import fsolve

def get_XFe2O3(T, P, X, fO2, x0):
    '''
    Solve for the mole fraction of Fe2O3 in a silicate melt given fO2.

    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [Pa]
    X : array_like
        Mole fractions of oxides:
            X[0] = XAl2O3
            X[1] = XCaO
            X[2] = XNa2O
            X[3] = XK2O
            X[4] = XFeOt
            X[5] = XFeO1_5
    fO2 : float
        Oxygen fugacity [Pa]
    x0 : float
        Initial guess for XFe2O3

    Returns
    -------
    XFe2O3 : float
        Mole fraction of Fe2O3
    '''

    lnfO2 = np.log(fO2)

    def get_fun(Fe2O3guess):
        term = (
            1/0.196 * (
                np.log(Fe2O3guess)
                - 1.1492e4/T + 6.675
                + 2.243*X[2] + 1.828*X[4]
                - 3.201*X[0] - 5.854*X[5] - 6.215*X[3]
                + 3.36*(1 - 1673/T - np.log(T/1673))
                + 7.01e-7*P/T
                + 1.54e-10*(T - 1673)*P/T
                - 3.85e-17*P**2 / T
            )
        )
        return lnfO2 - term

    XFe2O3, = fsolve(get_fun, x0)
    return XFe2O3
