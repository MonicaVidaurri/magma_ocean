import numpy as np

def get_fO2(T, P, Xi):
    '''
    get_fO2.m

    Parameters
    ----------
    T : float or ndarray
        Temperature (K)
    P : float or ndarray
        Pressure (Pa)  [same assumption as MATLAB code]
    Xi : ndarray, shape (12,)
        Mole fractions of oXiides:
        Xi[0]  = XiSiO2
        Xi[1]  = XiTiO2
        Xi[2]  = XiAl2O3
        Xi[3]  = XiMgO
        Xi[4]  = XiCaO
        Xi[5]  = XiNa2O
        Xi[6]  = XiK2O
        Xi[7]  = XiP2O5
        Xi[8]  = XiFeOt
        Xi[9]  = XiFeO1_5
        Xi[10] = XiFeO
        Xi[11] = XFe2O3

    Returns
    -------
    fO2 : float or ndarray
        Oxygen fugacity
    '''

    lnfO2 = ((1.0 / 0.196) * (np.log(Xi[11] / Xi[10]) - 1.1492e4 / T + 6.675 + 2.243 * Xi[2] + 1.828 * Xi[8]
            - 3.201 * Xi[4] - 5.854 * Xi[5] - 6.215 * Xi[6] + 3.36 * (1.0 - 1673.0 / T - np.log(T / 1673.0))
            + 7.01e-7 * P / T + 1.54e-10 * (T - 1673.0) * P / T - 3.85e-17 * P**2 / T))

    fO2 = np.exp(lnfO2)
    return fO2
