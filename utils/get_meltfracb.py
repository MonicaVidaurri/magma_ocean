import numpy as np
from scipy.integrate import trapezoid

def get_meltfracb(g, Tp, Rp, Rc, Mmantle):
    '''
    Gives the melt fraction throughout the magma ocean (alternative version)
    and the pressure of the base of the magma ocean.

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
        Volume-averaged melt fraction (using alternative definition)
    '''

    # physical constants
    alpha = 2e-5
    cp = 1.2e3

    # mantle density
    rho = Mmantle / (4/3 * np.pi * (Rp**3 - Rc**3))

    # depth grid
    z = np.arange(0, Rp - Rc + 5e3, 5e3)

    # pressure in GPa
    press = g * rho * z / 1e9

    # solidus and liquidus
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

    # molten layers
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

        # volume-averaged melt fraction (alternative definition)
        r = z - Rp  # note: z - Rp instead of Rp - z
        meltfrac = 3 * trapezoid(fraction * r**2, r) / (Rp**3 - Rc**3)

        # clamp
        meltfrac = min(meltfrac, 1.0)

    return exchangeP, meltfrac
