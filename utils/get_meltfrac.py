import numpy as np
from scipy.integrate import trapezoid

def get_meltfrac(g, Tp, Rp, Rc, Mmantle):
    '''
    Gives melt fraction throughout the magma ocean incl.
    the pressure of the base of the magma ocean from the intersection
    of the mantle adiabat with the solidus.

    Parameters
    ----------
    g : float
        Gravity [m/s^2]
    Tp : float
        Mantle potential temperature [K]
    Rp : float
        Planet radius [m]
    Rc : float
        Core radius [m]
    Mmantle : float
        Mantle mass [kg]

    Returns
    -------
    exchangeP : float
        Pressure at the base of the magma ocean [GPa]
    meltfrac : float
        Volume-averaged melt fraction over the magma ocean
    '''

    # physical constants
    alpha = 2e-5     # thermal expansion [1/K]
    cp = 1.2e3       # heat capacity [J/kg/K]

    # calculate mantle density from mass and volume
    rho = Mmantle / (4/3 * np.pi * (Rp**3 - Rc**3))

    # depth grid from surface
    z = np.arange(0, Rp - Rc + 5e3, 5e3)  # meters

    # pressure in GPa
    press = g * rho * z / 1e9

    # solidus and liquidus (linear approximation)
    Tsolidus = np.minimum(104.42 * press + 1420, 26.53 * press + 1825)
    Tliquidus = Tsolidus + 600

    # mantle adiabat
    Tadiabat = Tp + Tp * (alpha * g * z / cp)

    # melt fraction
    fraction = np.zeros_like(z)
    fraction[Tadiabat > Tliquidus] = 1.0
    melt_mask = (Tadiabat > Tsolidus) & (Tadiabat <= Tliquidus)
    fraction[melt_mask] = (Tadiabat[melt_mask] - Tsolidus[melt_mask]) / \
                          (Tliquidus[melt_mask] - Tsolidus[melt_mask])

    # indices of molten layers
    I = np.where(fraction > 0)[0]

    if len(I) < 2:
        meltfrac = 0.0
        exchangeP = 0.0
    else:
        # base pressure
        if I[-1] < len(fraction) - 1:
            exchangeP = press[I[-1] + 1]
        else:
            exchangeP = press[I[-1]]

        # volume-averaged melt fraction over only the magma ocean
        r = Rp - z[:I[-1] + 1]
        meltfrac = 3 * trapezoid(fraction[:I[-1] + 1] * r**2, r) / (r[-1]**3 - r[0]**3)

        meltfrac = min(meltfrac, 1.0)

    return exchangeP, meltfrac
