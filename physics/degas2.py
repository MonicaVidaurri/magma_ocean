import numpy as np

def degas2(Tm, Db, qm, FH2O, Rp, g, Tsurf):
    '''
    degas2.m

    Parameters
    ----------
    Tm : float
        Mantle potential temperature (K)
    Db : float
        Boundary layer thickness (m)
    qm : float
        Heat flux (W/m^2)
    FH2O : float
        Bulk water mass fraction
    Rp : float
        Planetary radius (m)
    g : float
        Gravity (m/s^2)
    Tsurf : float
        Surface temperature (K)

    Returns
    -------
    Dmelt : float
        Melt layer thickness (m)
    rmor : float
        Degassing rate proxy
    meltfrac : float
        Volume-averaged melt fraction
    avgXmelt : float
        Volume-averaged water fraction in melt
    '''

    # bulk distribution coefficient
    D_H2O = 0.01

    # physical constants
    rho_m = 3.3e3
    alpha = 2e-5
    cp = 1.2e3
    chi_d = 1.0
    km = 4.2

    # depth grid (m)
    z = np.arange(0.0, 300e3 + 1e3, 1e3)
    nz = len(z)

    # initialize arrays
    Tadiabat = np.zeros(nz)
    fraction = np.zeros(nz)
    Xmelt = np.zeros(nz)

    # pressure in GPa
    press = rho_m * g * z / 1e9

    # solidus and liquidus
    Tsolidus = np.minimum(104.42 * press + 1420.0,
                           26.53 * press + 1825.0)
    Tliquidus = Tsolidus + 600.0

    # temperature profile
    Tadiabat[0] = Tsurf
    Tp = Tm

    for i in range(1, nz):
        if z[i] < Db:
            Tadiabat[i] = Tadiabat[0] + z[i] * qm / km
        else:
            Tadiabat[i] = Tp + Tp * (alpha * g * z[i] / cp)

    # melt fraction and water partitioning
    for i in range(nz):
        if Tadiabat[i] > Tliquidus[i]:
            fraction[i] = 1.0
            Xmelt[i] = FH2O

        elif Tadiabat[i] > Tsolidus[i]:
            fraction[i] = ((Tadiabat[i] - Tsolidus[i]) /
                           (Tliquidus[i] - Tsolidus[i]))
            Xmelt[i] = FH2O / (D_H2O + fraction[i] * (1.0 - D_H2O))

        else:
            fraction[i] = 0.0
            Xmelt[i] = 0.0

    # indices of molten region
    I = np.where(fraction > 0.0)[0]

    # no melt case
    if I.size < 2:
        meltfrac = 0.0
        avgXmelt = 0.0
        Dmelt = 0.0
        rmor = 0.0
        return Dmelt, rmor, meltfrac, avgXmelt

    # base of magma ocean pressure (kept for completeness)
    if I[-1] < nz - 1:
        exchangeP = press[I[-1] + 1]
    else:
        exchangeP = press[I[-1]]

    # volume-averaged melt and water fraction
    r = Rp - z[I[0]:I[-1] + 1]

    meltfrac = (3.0 *
                np.trapz(fraction[I[0]:I[-1] + 1] * r**2, r) /
                (r[-1]**3 - r[0]**3))

    avgXmelt = (3.0 *
                np.trapz(Xmelt[I[0]:I[-1] + 1] * r**2, r) /
                (r[-1]**3 - r[0]**3))

    # numerical safety
    if meltfrac > 1.0:
        meltfrac = 1.0

    Dmelt = z[I[-1]] - z[I[0]]

    # degassing proxy
    if Dmelt > 0.0:
        rmor = meltfrac * avgXmelt * rho_m
    else:
        rmor = 0.0

    return Dmelt, rmor, meltfrac, avgXmelt
