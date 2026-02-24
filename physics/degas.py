import numpy as np

def degas(Tm, Db, qm, FH2O, Rp, g, Tsurf):
    '''
    degas.m

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
    avgfmelt : float
        Average melt fraction
    avgXmelt : float
        Average water fraction in melt
    '''

    # bulk distribution coefficient
    D_H2O = 0.01

    # physical constants
    rho_m = 3.3e3
    beta = 3 / 2
    K = 43.0
    gamma = 3 / 4
    alpha = 2e-5
    cp = 1.2e3
    chi_d = 1.0
    km = 4.2

    # depth grid (m)
    z = np.arange(0.0, 300e3 + 1e3, 1e3)
    nz = len(z)

    y = np.zeros(nz)        # melt fraction
    Xmelt = np.zeros(nz)    # water in melt
    Tprofile = np.zeros(nz)

    # pressure in GPa
    press = rho_m * g * z / 1e9

    # solidus and liquidus
    Tsolidus = np.minimum(104.42 * press + 1420.0,
                           26.53 * press + 1825.0)
    Tliquidus = Tsolidus + 600.0

    # temperature profile
    Tprofile[0] = Tsurf
    Tp = Tm

    for i in range(1, nz):
        if z[i] < Db:
            # conductive boundary layer
            Tprofile[i] = Tprofile[0] + z[i] * qm / km
        else:
            # mantle adiabat
            Tprofile[i] = Tp + Tp * (alpha * g * z[i] / cp)

    # melt fraction and water partitioning
    for i in range(1, nz):

        if Tprofile[i] >= Tliquidus[i]:
            y[i] = 1.0

        elif Tsolidus[i] <= Tprofile[i] < Tliquidus[i]:
            y[i] = ((Tprofile[i] - Tsolidus[i]) /
                    (Tliquidus[i] - Tsolidus[i]))
            y[i] = np.clip(y[i], 0.0, 1.0)

        else:
            y[i] = 0.0

        if y[i] > 0.0:
            Xmelt[i] = FH2O / (D_H2O + y[i] * (1.0 - D_H2O))
        else:
            Xmelt[i] = 0.0

    # melt layer thickness
    melt_indices = np.where(y > 0.0)[0]

    if melt_indices.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    firstz = melt_indices[0]
    secondz = firstz

    for i in range(firstz, nz):
        if y[i] > 0.0:
            secondz = i
        if i == nz - 1 or y[i + 1] <= 0.0:
            break

    n = secondz + 1
    thirdz = 0
    fourthz = 0

    if n < nz and np.sum(y[n:]) > 0.0:
        y2 = y[n:]
        thirdz = np.where(y2 > 0.0)[0][0] + n
        fourthz = nz - 1

    Dmelt = (z[secondz] - z[firstz]) + (z[fourthz] - z[thirdz])

    r = Rp - z

    # averages over melt region
    if Dmelt > 0.0:
        if fourthz > 0:
            product1 = np.trapezoid(y[firstz:fourthz+1] *
                                 r[firstz:fourthz+1]**2,
                                 r[firstz:fourthz+1])
            product2 = np.trapezoid(Xmelt[firstz:fourthz+1] *
                                 r[firstz:fourthz+1]**2,
                                 r[firstz:fourthz+1])

            denom = (r[secondz]**3 - r[firstz]**3 +
                     r[fourthz]**3 - r[thirdz]**3)

            avgfmelt = 3.0 * product1 / denom
            avgXmelt = 3.0 * product2 / denom

        else:
            product1 = np.trapezoid(y[firstz:secondz+1] *
                                 r[firstz:secondz+1]**2,
                                 r[firstz:secondz+1])
            product2 = np.trapezoid(Xmelt[firstz:secondz+1] *
                                 r[firstz:secondz+1]**2,
                                 r[firstz:secondz+1])

            denom = r[secondz]**3 - r[firstz]**3

            avgfmelt = 3.0 * product1 / denom
            avgXmelt = 3.0 * product2 / denom

        rmor = avgfmelt * avgXmelt * rho_m

    else:
        avgfmelt = 0.0
        avgXmelt = 0.0
        rmor = 0.0

    return Dmelt, rmor, avgfmelt, avgXmelt
