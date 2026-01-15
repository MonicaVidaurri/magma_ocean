import numpy as np
from scipy.special import gamma
from scipy.integrate import trapezoid, cumulative_trapezoid

def get_OLR_fn(g, Teq, Tsurf, psurf, tog_ana=1, tog_cp0=True, cp_file=None):
    '''
    Calculate outgoing longwave radiation (OLR) for GJ1132b assuming a gray atmosphere.

    Parameters
    ----------
    g : float
        Surface gravity [m/s^2]
    Teq : float
        Planetary equilibrium temperature [K]
    Tsurf : float
        Surface temperature [K]
    psurf : float
        Surface pressure [Pa]
    tog_ana : int, optional
        1 = analytic OLR, 2 = numerical integration, by default 1
    tog_cp0 : bool, optional
        True = constant cp, False = variable cp, by default True
    cp_file : str, optional
        Path to cp_H2O.dat for variable cp [temperature vs cp in kJ/kg/K]

    Returns
    -------
    OLR : float
        Outgoing longwave radiation [W/m^2]
    '''

    # constants
    sigma = 5.67e-8  # Stefan-Boltzmann [W/m^2/K^4]
    Rstar = 8.314462  # J/K/mol
    muH2O = 18.015  # g/mol
    Tskin = Teq / 2**0.25  # skin temperature [K]
    kappa = 1e-5
    cosa = 0.5

    # specific gas constant [J/kg/K]
    R = Rstar / (muH2O / 1e3)
    cp_const = 2000  # J/kg/K constant cp

    tauinf = kappa * psurf / (g * cosa)

    if tog_ana == 1:
        # simple analytic formula
        OLR = sigma * Tsurf**4 * tauinf**(-4*R/cp_const) * gamma(1 + 4*R/cp_const)

    elif tog_ana == 2:
        nLev = int(1e2)

        if tog_cp0:
            # constant cp
            p = np.logspace(0, np.log10(psurf), nLev)[::-1]  # pressure [Pa], descending
            T = Tsurf * (p / psurf)**(R / cp_const)
        else:
            # variable cp
            if cp_file is None:
                raise ValueError('cp_file must be provided for variable cp mode')

            # load cp_H2O.dat
            cp_data = np.loadtxt(cp_file)
            T = np.logspace(1, np.log10(Tsurf), nLev)[::-1]  # descending T
            cp = np.interp(T, cp_data[:,0], cp_data[:,1]) * 1e3  # kJ/kg/K -> J/kg/K

            # numerical integration to get pressure
            lnT = np.log(T)
            integral = cumulative_trapezoid(cp / R, lnT, initial=0)
            p = psurf * np.exp(integral)

        # isothermal stratosphere
        T[T < Tskin] = Tskin
        tau = kappa * p / (g * cosa)
        Trans = np.exp(-tau)

        Bsurf = sigma * Tsurf**4
        B = sigma * T**4

        OLR = Bsurf * Trans[0] + trapezoid(B * Trans, x=None)  # W/m^2

    else:
        raise ValueError('tog_ana must be 1 or 2')

    return OLR
