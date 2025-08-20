"""Thermal functions

Originally the matlab files:
- mantleheatflux.m
    - New function mantle_heat_flux

Helper functions were also added.

Converted to Python by Joe P. Renaud August 2025.
"""

import numpy as np
from numba import njit

from magma.viscosity import viscosity_lebrun, viscosity_sandu

@njit
def calculate_thermal_diffusivity(km, rho_m, cp):
    """
    Calculate thermal diffusivity from thermal properties.
    
    Parameters
    ----------
    km : float
        Thermal conductivity (W/(m·K))
    rho_m : float
        Density (kg/m³)
    cp : float
        Specific heat capacity (J/(kg·K))
        
    Returns
    -------
    kappa : float
        Thermal diffusivity (m²/s)
    """
    return km / (rho_m * cp)

@njit
def calculate_rayleigh_number(g, alpha, delta_T, Z, nu, kappa):
    """
    Calculate the Rayleigh number for thermal convection.
    
    The Rayleigh number characterizes the vigor of convective heat transfer
    and is defined as the ratio of buoyancy forces to viscous and thermal
    diffusion effects.
    
    Parameters
    ----------
    g : float
        Gravitational acceleration (m/s²)
    alpha : float
        Thermal expansion coefficient (K⁻¹)
    delta_T : float
        Temperature difference (K)
    Z : float
        Convection layer thickness (m)
    nu : float
        Kinematic viscosity (m²/s)
    kappa : float
        Thermal diffusivity (m²/s)
        
    Returns
    -------
    Ra : float
        Rayleigh number (dimensionless)
        
    Notes
    -----    
    Critical Rayleigh numbers:
    - Ra_crit ≈ 1708 for onset of convection in infinite layer
    - Ra_crit ≈ 27π⁴/4 ≈ 658 for spherical geometry
    - Vigorous convection typically occurs for Ra > 10⁶
    
    Examples
    --------
    >>> Ra = calculate_rayleigh_number(9.81, 2e-5, 1300, 2.9e6, 1e17, 1e-6)
    >>> print(f"Rayleigh number: {Ra:.2e}")
    """
    return (g * alpha * abs(delta_T) * Z**3) / (nu * kappa)


# @njit
def mantle_heat_flux(Tm, Ts, rs, Rp, Rc, g, rho_m, FH2O, meltfrac,
                     km = 4.2, alpha = 2.0e-5, cp = 1.2e3):
    """
    Calculate the mantle heat flux based on temperature and viscosity.
    
    This function computes mantle heat flux, boundary layer depth, spreading
    velocity, and Rayleigh number for convective heat transfer in planetary
    mantles. The calculation considers different convection regimes based on
    melt fraction and uses appropriate viscosity models.
    
    Parameters
    ----------
    Tm : float
        Mantle temperature (K)
    Ts : float  
        Surface temperature (K)
    rs : float
        Magma surface radius (m)
    Rp : float
        Planetary radius (m)
    Rc : float
        Core radius (m)
    g : float
        Gravitational acceleration (m/s²)
    rho_m : float
        Mantle density (kg/m³)
    FH2O : float
        Water fraction in mantle (dimensionless)
    meltfrac : float
        Melt fraction (dimensionless, 0-1)
    km : float (default, 4.2)
        Thermal conductivity (W/m/K)
    alpha : float (default, 2.0e-5)
        Coefficient of thermal expansion (1/K)
    cp : float (default, 1.2e3)
        Heat capacity (J/kg/K)
    
        
    Returns
    -------
    qm : float
        Mantle heat flux (W/m²)
    Db : float
        Boundary layer thickness (m)
    uc : float
        Spreading velocity (m/s)
    Ra : float
        Rayleigh number (dimensionless)
    nu : float
        Kinematic viscosity (m²/s)
        
    Notes
    -----    
    The scaling law coefficient 0.089 comes from experimental and numerical
    studies of convective heat transfer in high Rayleigh number systems.
    
    References
    ----------
    The scaling relationship and physical approach are based on:
    - Turcotte, D.L. & Schubert, G. (2014). Geodynamics. Cambridge University Press.
    - Schubert, G., Seitz, M.G., & Spohn, T. (1986). Thermal history of Mars 
      and the sulfur content of its core. Journal of Geophysical Research, 91(B2), 2283-2293.
    """
    
    # Calculate thermal diffusivity
    kappa = calculate_thermal_diffusivity(km, rho_m, cp)
    
    # Determine total thickness of the convective zone
    if meltfrac >= 0.4:
        # Convection only occurs within the magma ocean until solidification
        Z = Rp - rs
    else:
        # Full mantle convection from core-mantle boundary
        Z = Rp - Rc

    if Z <= 0.0:
        raise ZeroDivisionError("Unexpected size for convective thickness encountered.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Calculate kinematic viscosity using the viscosity function
    
    # Commented out for now. 
    # if Ts < 1420:
    #     nu = viscosity_sandu(Tm,FH2O,g,rho_m,0)
    # else:
    #     nu = viscosity_lebrun(Tm, Ts, rho_m, FH2O, meltfrac)
    nu = viscosity_lebrun(Tm, Ts, rho_m, FH2O, meltfrac)
    
    # Calculate Rayleigh number
    delta_T = abs(Tm - Ts)
    Ra = calculate_rayleigh_number(g, alpha, delta_T, Z, nu, kappa)
    
    # Calculate mantle heat flux using scaling law
    qm = 0.089 * km * (Tm - Ts) * Ra**(1/3) / Z
    
    # Calculate boundary layer depth [m]
    Db = km * (Tm - Ts) / qm
    
    # Calculate spreading time [s]
    ts = Db**2 / (5.38 * kappa)
    
    # Calculate spreading velocity [m s-1]
    uc = Z / ts
    
    return qm, Db, uc, Ra, nu
