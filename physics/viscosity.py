import numpy as np


def viscosity(Tm, Ts, rho_m, FH2O, meltfrac):
    '''
    Lebrun et al. (2013) parameterization
    '''

    # matlab has this idk it overrides rho so i'll override rho
    rho_m = 3.3e3  # kg/m^3

    # liquid magma dynamic viscosity (Pa/s)
    A = 0.00024  # Pa s
    B = 4600.0   # K
    eta_l = A * np.exp(B / (Tm - 1000.0))

    # solid viscosity (Pa/s)
    alphan = 26.0
    eta_s = 3.7489e9 * np.exp(350e3 / 8.31447 / Tm)

    alpha1 = 6.4e4
    alpha2 = -6.1e6
    nu0 = 2.21e7  # m^2/s

    # melt fraction
    meltfrac = (Tm - 1420.0) / 600.0

    if Tm < 1420.0:
        meltfrac = 0.0
    if meltfrac > 1.0:
        meltfrac = 1.0

    # viscosity regime switch
    if 1.0 - meltfrac < 0.6:
        # liquid-like regime
        eta = eta_l / (1.0 - (1.0 - meltfrac) / (1.0 - 0.4)) ** 2.5
        nu = eta / rho_m
    else:
        # solid-like regime
        eta = eta_s * np.exp(-alphan * meltfrac)
        nu = eta / rho_m

    return nu