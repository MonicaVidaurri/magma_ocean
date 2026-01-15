import numpy as np
from get_meltfrac import get_meltfrac  # assuming previous translation

def get_pressure2b(Tm, rs, Mmo, Mmantle, Rp, g, Rc, MH2O):
    '''
    Calculate equilibrium water vapor pressure over a magma ocean (version b).

    Parameters
    ----------
    Tm : float
        Mantle temperature [K]
    rs : float
        Surface radius of the magma ocean? [m]
    Mmo : float
        Mass of the magma ocean [kg]
    Mmantle : float
        Mantle mass [kg]
    Rp : float
        Planet radius [m]
    g : float
        Surface gravity [m/s^2]
    Rc : float
        Core radius [m]
    MH2O : float
        Total water mass [kg]

    Returns
    -------
    PH2O : float
        Water vapor pressure [Pa]
    FH2Ol : float
        Water fraction in liquid
    kH2O : float
        Partition coefficient for H2O
    '''

    # partition coefficients for water between solid and melt
    rho = Mmantle / (4/3 * np.pi * (Rp**3 - Rc**3))

    Tliquidus = 1973
    Tsolidus = 1373

    if Tm > Tsolidus:
        # base pressure of magma ocean in GPa
        baseP = rho * g * (Rp - rs) / 1e9

        # get melt fraction
        _, meltfrac = get_meltfrac(g, Tm, Rp, Rc, Mmantle)
        Mliquid = meltfrac * Mmo
        Msolid = (1 - meltfrac) * Mmo

        kH2O = 0.01

        # Newton-Raphson to solve for water fraction in liquid
        FH2O_0 = MH2O / Mmantle
        for i in range(30):
            f = MH2O - FH2O_0*(kH2O*Msolid + Mliquid) - (4*np.pi*Rp**2 / g) * (FH2O_0 / 3.44e-8)**(1/0.74)
            df = - (kH2O*Msolid + Mliquid) - (1/0.74)*(1/3.44e-8)**(1/0.74) * (4*np.pi*Rp**2 / g) * FH2O_0**((1/0.74) - 1)
            FH2O_new = FH2O_0 - f/df
            if abs(FH2O_new - FH2O_0) < 1e-6:
                FH2Ol = FH2O_new
                break
            FH2O_0 = FH2O_new
        else:
            FH2Ol = FH2O_0  # fallback

        FH2Os = FH2Ol * kH2O
        PH2O = (FH2Ol / 3.44e-8)**(1/0.74)  # Pa

    else:
        FH2Ol = 0.0
        PH2O = MH2O * g / (4 * np.pi * Rp**2)
        kH2O = 0.0

    return PH2O, FH2Ol, kH2O
