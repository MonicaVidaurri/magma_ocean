import numpy as np

def viscosity2(temp, f_water, g, rho_m, P):
    '''
    Calculate mantle viscosity following Sandu et al. (2011).

    Parameters
    ----------
    temp : float
        Mantle temperature [K]
    f_water : float
        Mass fraction of water in the mantle
    g : float
        Surface gravity [m/s^2] (not used in current calculation)
    rho_m : float
        Mantle density [kg/m^3]
    P : float
        Pressure [Pa]

    Returns
    -------
    nu : float
        Kinematic viscosity [m^2/s]
    '''
    # Olivine composition
    forsterite = 0.9
    fayalite = 0.1
    mfor = 140.0e-3
    mfay = 204.0e-3
    mH2O = 18.02e-3

    # Molecular weight of olivine solid solution
    molv = forsterite * mfor + fayalite * mfay

    # Convert mantle water mass fraction to number of H atoms per 10^6 Si atoms
    C_OH = f_water * molv * 1e6 * 2 / mH2O

    # Li et al. 2008 parameters to get water fugacity
    c0 = -7.9859
    c1 = 4.3559
    c2 = -0.5742
    c3 = 0.0337

    logfH2O = c0 + c1 * np.log(C_OH) + c2 * (np.log(C_OH))**2 + c3 * (np.log(C_OH))**3

    # Viscosity parameters
    eta_0 = 1.24e14  # Pa s, calibration constant
    r = 1.0          # fugacity exponent
    Qa = 335e3       # J/mol, activation energy
    V = 0.0          # m^3/mol, activation volume (no pressure dependence)
    R = 8.31447      # J/mol/K, ideal gas constant

    # Dynamic viscosity
    eta_eff = eta_0 * (np.exp(logfH2O))**(-r) * np.exp((Qa + P * V) / (R * temp))

    # Kinematic viscosity
    nu = eta_eff / rho_m

    return nu
