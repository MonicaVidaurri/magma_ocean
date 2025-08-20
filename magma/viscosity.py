"""Viscosity functions

Originally the matlab files:
- viscosity.m
    - New function viscosity_lebrun
- viscosity2.m
    - New function viscosity_sandu

Original model delevloped by Laura Schaefer (Stanford) ca. ????
Converted to Python by Joe P. Renaud (NASA Goddard) August 2025.
"""

import numpy as np
from numba import njit

@njit
def viscosity_lebrun(Tm, Ts, rho_m, FH2O, meltfrac):
    """
    Calculate the viscosity of the magma ocean following Lebrun et al. (2013).
    
    This function computes kinematic viscosity based on mantle temperature and 
    crystal fraction in the melt. The viscosity behavior transitions between 
    liquid-like (>40% melt fraction) and solid-like (<60% melt fraction) regimes.
    
    Parameters
    ----------
    Tm : float
        Mantle temperature (K)
    Ts : float
        Surface temperature (K) - currently unused in calculations
    rho_m : float
        Mantle density (kg/m³) - overridden internally to 3.3e3 kg/m³
    FH2O : float
        Water fraction - currently unused in main calculation path
    meltfrac : float
        Initial melt fraction - recalculated internally based on temperature
        
    Returns
    -------
    nu : float
        Kinematic viscosity (m²/s)
        
    Notes
    -----
    The melt fraction is recalculated internally as:
    meltfrac = (Tm - 1420) / 600
    
    Viscosity regimes:
    - If solid fraction < 60% (melt fraction > 40%): liquid-like behavior
    - If solid fraction ≥ 60% (melt fraction ≤ 40%): solid-like behavior
    
    The function uses different parameterizations for liquid and solid viscosity:
    - Liquid: eta_l = A * exp(B / (Tm - 1000)) where A=0.00024 Pa·s, B=4600 K
    - Solid: eta_s = 3.7489e9 * exp(350e3 / 8.31447 / Tm) Pa·s
    
    References
    ----------
    Lebrun, T., et al. (2013). Thermal evolution and structure of early Earth's 
    magma ocean. Journal of Geophysical Research: Planets, 118(6), 1155-1176.
    
    Examples
    --------
    >>> nu = viscosity(2000, 1500, 3300, 0.1, 0.5)
    >>> print(f"Kinematic viscosity: {nu:.2e} m²/s")
    """
    rho_m = 3.3e3
    
    # liquid magma dynamic viscosity (Pa s)
    A = 0.00024  # Pa s
    B = 4600     # K
    eta_l = A * np.exp(B / (Tm - 1000))
    
    # melt-fraction dependent dynamic viscosity (Pa s)
    alphan = 26
    # mu = 80  # GPa
    # Am = 5.3e15  # pre-exponential factor
    # h = 1  # mm
    # b = 0.5  # nm
    # n = 2.5  # exponent
    # Ea = 240  # kJ/mol
    # R = 8.31451  # ideal gas constant J/mole
    # eta_s = mu / (2 * Am) * (h / (b*1e-6))**n * np.exp(Ea*1e3 / R / Tm)
    eta_s = 3.7489e9 * np.exp(350e3 / 8.31447 / Tm)
    
    # solid mantle kinematic viscosity McGovern & Schubert (1989)
    alpha1 = 6.4e4
    alpha2 = -6.1e6
    nu0 = 2.21e7  # m2/s
    
    # TODO: Why is this recalculated internally?
    meltfrac = (Tm - 1420) / 600
    if Tm < 1420:
        meltfrac = 0
    if meltfrac > 1:
        meltfrac = 1.0
    
    # viscosity of magma ocean behaves as a liquid for melt fractions greater
    # than 40%, or as a solid for melt fractions smaller than 40%
    if 1 - meltfrac < 0.6:
        eta = eta_l / (1 - (1 - meltfrac) / (1 - 0.4)) ** 2.5
        nu = eta / rho_m
    else:  # if 1 - meltfrac < 0.99
        eta = eta_s * np.exp(-alphan * meltfrac)
        nu = eta / rho_m
    # else:
    #     nu = nu0 * np.exp((alpha1 + alpha2 * FH2O) / Tm)
    
    return nu

@njit
def viscosity_sandu(temp, f_water, g, rho_m, P):
    """
    Calculate the viscosity of the mantle following the parameterization of Sandu et al. (2011).
    
    This function computes mantle viscosity as a function of temperature and water content.
    The calculation involves converting water mass fraction to OH concentration in olivine,
    then using water fugacity relationships to determine effective viscosity. The model
    assumes an olivine solid solution with 90% forsterite and 10% fayalite.
    
    Parameters
    ----------
    temp : float
        Temperature (K)
    f_water : float
        Water abundance as mass fraction of the mantle (dimensionless)
    g : float
        Gravitational acceleration (m/s²) - currently unused in calculations
    rho_m : float
        Mantle density (kg/m³)
    P : float
        Pressure (Pa)
        
    Returns
    -------
    nu : float
        Kinematic viscosity (m²/s)
        
    Notes
    -----
    The water content significantly affects viscosity through the fugacity term,
    with higher water content leading to lower viscosity.
    
    References
    ----------
    Sandu, C., et al. (2011). Convective heat transfer and particle entrainment 
    in experimentally simulated magma oceans. Journal of Geophysical Research, 
    116, B12201.
    
    Li, Z. X. A., et al. (2008). Water contents in mantle xenoliths from the 
    Colorado Plateau and vicinity: Implications for the rheology and hydration-
    induced melting of continental lithosphere. Journal of Geophysical Research, 
    113, B09210.
    
    Warnings
    --------
    The function may encounter numerical issues if C_OH approaches zero or
    becomes very small, as this would cause issues with the logarithmic terms
    in the water fugacity calculation.
    """
    
    # Convert the mass fraction of the mantle into concentration in units of
    # number of atoms of H per 10^6 Si atoms in olivine. Assume an olivine
    # solid solution of (fayalite/forsterite) = 1/9
    forsterite = 0.9
    fayalite = 0.1
    mfor = 140.0e-3
    mfay = 204.0e-3
    mH2O = 18.02e-3
    molv = forsterite * mfor + fayalite * mfay
    # rho_m = 3.3e3  # hold density at surface equal to uncompressed olivine
    
    # P = rho_m * g * z  # pressure in Pa at 100 km (estimate for Db thickness)
    
    # water abundance in OH molecules per 10^6 Si atoms
    C_OH = f_water * molv * 1e6 * 2 / mH2O
    
    # constants relating C_OH to water fugacity (from Li et al. 2008)
    c0 = -7.9859
    c1 = 4.3559
    c2 = -0.5742
    c3 = 0.0337
    
    # calculate ln f_H2O using the water abundance and the above parameters
    # ln f_H2O = c0 + c1 lnCOH + c2 ln^2 COH + c3 ln^3 COH (Li et al. 2008)
    logfH2O = c0 + c1 * np.log(C_OH) + c2 * (np.log(C_OH))**2 + c3 * (np.log(C_OH))**3
    
    # viscosity constants
    # calibration constant P-dependent
    eta_0 = 1.24e14
    #         eta_0 = 5.66e12  # Pa s
    #     eta_0 = 3e17  # P-independent
    # material constant 
    #     A_cre = 90  # MPa^-r / s
    # fugacity exponent
    r = 1.0
    # activation energy for creep 
    Qa = 335e3  # J/mol
    #     # activation volume
    #     V = 4e-6  # no pressure dependence of viscosity
    V = 0.0
    # ideal gas constant
    R = 8.31447  # J/mole/K
    # mantle viscosity
    
    # effective dynamic viscosity (Pa s)
    eta_eff = eta_0 * (np.exp(logfH2O))**(-r) * np.exp((Qa + P*V)/R/temp)
    
    # kinematic viscosity (m^2 / s)
    nu = eta_eff / rho_m
    
    return nu