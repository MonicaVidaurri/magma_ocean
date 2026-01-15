import numpy as np


def mo_event3(t, y):
    '''
    Phase 3.
    Stops when degassing stops
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

    mantle_temperature = y[0]  # Tr(1)

    # EVENT CONDITION
    return float(mantle_temperature - 1000.0)


# Required by SciPy
mo_event3.terminal = True
mo_event3.direction = 0
