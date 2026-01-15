import numpy as np

def get_flux(Tsurf, Teq, Psurf, Rp, g):
    sigma = 5.67e-8      # Stefan-Boltzmann constant [W/m^2/K^4]
    k0 = 0.01            # absorption coefficient, water, m^2/kg
    p0 = 1.01325e5       # reference pressure, Pa

    Matm = 4.0 * np.pi * Psurf * Rp**2 / g
    tau = (3.0 * Matm) / (8.0 * np.pi * Rp**2) * np.sqrt((k0 * g) / (3.0 * p0))
    emissivity = 2.0 / (tau + 2.0)

    flux = emissivity * sigma * (Tsurf**4 - Teq**4)
    # flux = sigma * (Tsurf**4 - Teq**4)

    return flux