import numpy as np


def mo_event2(t, y):
    '''
    Phase 2.
    Stops when mantle water mass fraction < 1e-9
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

    # planet properties
    MEarth = 5.97e24
    REarth = 6371e3

    Mp = 1.62 * MEarth
    Rp = 1.16 * REarth
    Rc = 0.54 * Rp
    core_fraction = 0.262

    Mmantle = (1 - core_fraction) * Mp

    water_mantle = y[1]  # Tr(2)

    # EVENT CONDITION
    return float(water_mantle / Mmantle - 1e-9)

mo_event2.terminal = True
mo_event2.direction = -1
