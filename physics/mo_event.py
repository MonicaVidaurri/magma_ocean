import numpy as np

from utils.get_meltfrac import get_meltfrac
from physics.mantleheatflux import mantleheatflux


def mo_event(t, y):
    '''
    Phase 1
    '''
    # Constants
    G = 6.67e-11
    sigma = 5.67e-8

    MSun = 1.989e30
    AU = 149597871e3

    MStar = 0.181
    LStar = 0.00438
    Tday = 1.628930

    Omega = 2 * np.pi / (Tday * 86400.0)
    a = (G * MSun * MStar / Omega**2) ** (1 / 3)

    Fat1AU = LStar * 1366.0
    Fstel = Fat1AU * (AU / a) ** 2
    Albedo = 0.75
    ASR = (1 - Albedo) * Fstel / 4

    # Planet properties
    MEarth = 5.97e24
    REarth = 6371e3

    Mp = 1.62 * MEarth
    Rp = 1.16 * REarth
    Rc = 0.54 * Rp
    core_fraction = 0.262

    Mmantle = (1 - core_fraction) * Mp
    Teq = (ASR / sigma) ** 0.25
    gp = G * Mp / Rp**2
    rho = Mmantle / (4 / 3 * np.pi * (Rp**3 - Rc**3))

    # State Tr_variables.txt
    Tm = y[0]
    rs = y[1]
    water_mantle = y[2]
    water_total = y[3]
    Ts = y[6]

    Mmo = 4 / 3 * np.pi * rho * (Rp**3 - rs**3)
    Wmo = water_total
    FH2Os = water_mantle / Mmantle

    _, meltfrac = get_meltfrac(gp, Tm, Rp, Rc, Mmantle)

    q_mantle, Db, uc, Ra, nu = mantleheatflux(Tm, Ts, rs, Rp, Rc, gp, rho, FH2Os, meltfrac)

    # EVENT CONDITION
    return float(Ts - 1420.0)


# Required by SciPy
mo_event.terminal = True
mo_event.direction = 0
