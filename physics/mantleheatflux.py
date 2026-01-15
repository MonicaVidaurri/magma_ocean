import numpy as np
from .viscosity import viscosity


def mantleheatflux(Tm, Ts, rs, Rp, Rc, g, rho_m, FH2O, meltfrac):
    # mantle thermal conductivity (W/m/K)
    km = 4.2

    # depth of convective zone
    if meltfrac >= 0.4:
        Z = Rp - rs
    else:
        Z = Rp - Rc

    # coefficient of thermal expansion (1/K)
    alpha = 2e-5

    # mantle heat capacity (J/kg/K)
    cp = 1.2e3

    # mantle thermal diffusivity (m^2 / s)
    kappa = km / rho_m / cp

    # find the viscosity (m^2/s)
    # nu = viscosity(Tm, Ts, rho_m, FH2O, meltfrac)
    nu = viscosity(Tm, Ts, rho_m, FH2O, meltfrac)

    # Rayleigh number
    Ra = (g * alpha * abs(Tm - Ts) * Z**3) / nu / kappa

    # mantle heat flux
    # qm = 0.089 * km * (Tm - Ts) * Ra^(1/3) / Z;
    qm = 0.089 * km * (Tm - Ts) * Ra**(1.0 / 3.0) / Z

    # boundary layer depth (m)
    Db = km * (Tm - Ts) / qm

    # spreading time (s)
    ts = Db**2 / (5.38 * kappa)

    # spreading velocity (m/s)
    uc = Z / ts

    return qm, Db, uc, Ra, nu
