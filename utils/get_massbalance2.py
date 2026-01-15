import numpy as np

def get_massbalance2(Tm, Patm, Mmo, MO_mo, Xi, FeOt, g, Rp, get_fO2):
    '''
    Calculate mass balance for oxygen and Fe3+/Fe ratio.

    Parameters
    ----------
    Tm : float
        Mantle temperature (K)
    Patm : float
        Atmospheric pressure (Pa)
    Mmo : float
        Magma ocean mass (kg)
    MO_mo : float
        Total oxygen mass in magma ocean + atmosphere (kg)
    Xi : ndarray, shape (12,)
        Mole fractions of oxides (from get_comp)
    FeOt : float
        Total Fe fraction in magma ocean
    g : float
        Surface gravity (m/s^2)
    Rp : float
        Planet radius (m)
    get_fO2 : callable
        Function returning fO2 given (T, P, X)

    Returns
    -------
    PO2 : float
        Atmospheric O2 pressure (Pa)
    FFeO1_5 : float
        Mass fraction of FeO1.5 in magma ocean
    mark : int
        Convergence marker (0 = converged, 1 = max iterations reached)
    nFeO1_5 : float
        Moles of FeO1.5
    '''

    muO = 15.9994e-3       # kg/mol
    muFeO1_5 = 159.689e-3 / 2
    muFeO = 71.845e-3

    nFeOt = FeOt * Mmo / muFeO          # moles of total Fe
    nO_t = MO_mo / muO                  # moles of O in system

    # initial bisection bounds
    a = 0.0
    b = 1.0

    # evaluate function at endpoints
    fa = getf(a, Tm, Xi, Patm, nO_t, nFeOt, g, Rp)
    fb = getf(b, Tm, Xi, Patm, nO_t, nFeOt, g, Rp)

    # check if solution is out of range
    if np.sign(fa) * np.sign(fb) > 0:
        nFeO1_5 = nO_t * 2       # put all O into FeO1.5
        nFeO = nFeOt - nFeO1_5
        nO_atm = 0.0
        if nFeO < 0:
            nFeO = 0
            nFeO1_5 = nFeOt
            nO_atm = nO_t - 0.5 * nFeOt

    # bisection root finding
    count = 0
    mark = 0
    while count <= 500:
        p = a + (b - a) / 2
        fp = getf(p, Tm, Xi, Patm, nO_t, nFeOt, g, Rp)

        if fp == 0 or ((b - a)/2 < 1e-17 and fp < 0):
            nFeO1_5 = p * nFeOt
            mark = 0
            break

        count += 1

        if np.sign(fa) * np.sign(fp) > 0:
            a = p
            fa = fp
        else:
            b = p
            fb = fp

    if count >= 500:
        nFeO1_5 = p * nFeOt
        mark = 1

    nO_atm = nO_t - 0.5 * nFeO1_5
    PO2 = (nO_atm * muO) * g / (4 * np.pi * Rp**2)
    FFeO1_5 = nFeO1_5 * muFeO1_5 / Mmo

    if np.isnan(Tm):
        PO2 = 0
        FFeO1_5 = 0

    # compute fO2 (placeholder; might need proper vector)
    X_fO2 = np.concatenate([Xi[:10], [nFeOt - 2*nO_t, 2*nO_t]])
    fo2 = get_fO2(Tm, Patm, X_fO2)

    return PO2, FFeO1_5, mark, nFeO1_5


# nested helper function
def getf(y, Tm, Xi, Patm, nO_t, nFeOt, g, Rp):
    '''
    calculate difference between FeO1.5 equilibrium fO2 and atmospheric O2
    '''
    muO = 15.9994e-3

    # fO2 equilibrium (from original MATLAB code)
    fO2fun = np.exp((np.log(y / (1 - y)) - 1.1492e4 / Tm + 6.675 + 2.243 * Xi[3] + 1.828 * Xi[9]
         - 3.201 * Xi[5] - 5.854 * Xi[6] - 6.215 * Xi[7] + 3.36 * (1 - 1673 / Tm - np.log(Tm / 1673))
         + 7.01e-7 * Patm / Tm + 1.54e-10 * (Tm - 1673) * Patm / Tm - 3.85e-17 * Patm**2 / Tm) / 0.196)

    PO2fun = (nO_t / nFeOt - 0.5 * y) * nFeOt * muO * g / (4 * np.pi * Rp**2)

    return fO2fun - PO2fun
