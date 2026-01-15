import numpy as np

def get_loss(t_flux, Lbol, t, PO2, PH2O, tsat, a, Mp, Rp, LStar, XUV_model=1):
    '''
    Calculate XUV-driven atmospheric escape fluxes of H and O.

    Parameters
    ----------
    t_flux : ndarray
        Time array for XUV flux evolution [yr]
    Lbol : float or ndarray
        Stellar bolometric luminosity (relative to LSun)
    t : float
        Current time [yr]
    PO2 : float
        O2 partial pressure [Pa]
    PH2O : float
        H2O partial pressure [Pa]
    tsat : float
        XUV saturation time [yr]
    a : float
        Semi-major axis of planet orbit [m]
    MPlanet : float
        Planet mass [kg]
    rPlanet : float
        Planet radius [m]
    LStar : float
        Stellar bolometric luminosity [W]
    XUV_model : int
        1 = high XUV
        2 = low XUV

    Returns
    -------
    phi_H : float
        Mass flux of hydrogen [kg/m^2/s]
    phi_O : float
        Mass flux of oxygen [kg/m^2/s]
    '''

    # Constants
    G = 6.673e-11       # m^3/kg/s^2
    sigma = 5.67e-8     # W/m2/K4
    AU = 149597871e3    # m
    NA = 6.022e23       # Avogadro
    muH = 1.008
    muO = 15.9994
    mpr = 1.6726219e-27  # kg (proton mass)

    # Stellar flux at planet
    Fat1AU = LStar * 1366
    F = Fat1AU * (AU / a)**2
    A = 0.25  # albedo factor
    g0 = G * Mp / Rp**2
    ASR = (1 - A) * F / 4
    Teq = (ASR / sigma)**0.25

    # XUV fraction
    f0 = 1e-3
    beta = -1.23
    eff = 0.1

    # Fractional XUV flux
    f = f0 * (t_flux / tsat)**beta

############## ------old stuff------- #################
    # if XUV_model == 1:
    #     # Model A: Ribas+ 2005 decay
    #     f = np.where(t_flux < tsat, f0, f)
    #     L_XUV = f * Lbol * 3.846e26  # convert L☉ → W
    # elif XUV_model == 2:
    #     # Model B: Saturation then zero
    #     f = np.where(t_flux < tsat, f0, 0.0)
    #     #L_XUV = f * Lbol[-1] * 3.846e26 if hasattr(Lbol, '__len__') else f * Lbol * 3.846e26
    #     L_XUV = f * 3.846e26 if hasattr(Lbol, '__len__') else f * Lbol * 3.846e26
    # else:
    #     raise ValueError('XUV_model must be 1 or 2')
#######################################################

    if XUV_model == 1:
        f = np.where(t_flux < tsat, f0, f)
    elif XUV_model == 2:
        f = np.where(t_flux < tsat, f0, 0.0)

    L_XUV = f * Lbol * 3.846e26
    F_XUV = L_XUV / (4 * np.pi * a**2)

    # Interpolate XUV flux at current time
    if t > t_flux[0]:
        Fxuv = np.interp(t, t_flux, F_XUV)
    else:
        Fxuv = F_XUV[0]

    # Base mass escape rate
    Vpot = G * Mp / Rp
    phi = (eff * Fxuv / 4) / Vpot  # kg/m^2/s

    # Escaping region parameters
    Tesc = Teq
    Phi1ref = phi / (muH * mpr)  # molecules/m^2/s
    mu1 = muH
    mu2 = muO

    # Molar fractions of H and O (assuming H2O)
    X1 = 2/3
    X2 = 1.0 - X1

    # Binary diffusion
    b = 4.8e17 * (Tesc)**0.75 * 1e2
    gam = 1 / (1 + X2 * mu2 / (X1 * mu1))
    mu_c_ref = mu1 + (1.38064852e-23 * Tesc * Phi1ref) / (b * g0 * X1 * mpr)
    mu_c = mu2 + gam * (mu_c_ref - mu2)

    # Fluxes
    Phi1 = Phi1ref * (mu_c / mu_c_ref)
    Phi2 = (X2 / X1) * Phi1 * ((mu_c - mu2) / (mu_c - mu1))
    Phi2 = np.maximum(Phi2, 0.0)

    # Critical flux
    Phi1crit = (b * g0 * X1 * mpr) * (mu2 - mu1) / (1.38064852e-23 * Tesc)

    # Convert to kg/m^2/s
    phi_H = Phi1 / NA * muH * 1e-3
    #phi_O = Phi2 / NA * muO * 1e-3 if min(Phi1, Phi1crit) == Phi1crit else 0.0
    if Phi1 >= Phi1crit:
        phi_O = Phi2 / NA * muO * 1e-3
    else:
        phi_O = 0.0

    return phi_H, phi_O
