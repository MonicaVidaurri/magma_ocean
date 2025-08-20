""" Degassing calculations for magma oceans.

Originally the matlab files:
- degas.m
    - New function viscosity_lebrun
- degas2.m
    - New function viscosity_sandu

Original model delevloped by Laura Schaefer (Stanford) ca. ????
Converted to Python by Joe P. Renaud (NASA Goddard) August 2025.
"""
from typing import Tuple

import numpy as np
from numba import njit


@njit
def degas_method_1(Tm: float, Db: float, qm: float, FH2O: float, Rp: float, g: float, Tsurf: float,
                   D_H2O: float = 0.01, rho_m: float = 3.3e3, alpha: float = 2.0e-5,
                   cp: float = 1.2e3, km: float = 4.2) -> Tuple[float, float, float, float]:
    """
    Calculate degassing rate using method 1.
    
    This function calculates the degassing rate by determining the melt fraction
    in the melt zone, the weight fraction of water in the melt, and the thickness
    of the melt zone. It handles potentially discontinuous melt regions.
    
    Parameters
    ----------
    Tm : float
        Mantle potential temperature (K)
    Db : float
        Boundary layer depth (m)
    qm : float
        Mantle heat flux (W m-2)
    FH2O : float
        Bulk water content in mantle (mass fraction)
    Rp : float
        Planetary radius (m)
    g : float
        Gravitational acceleration (m s-2)
    Tsurf : float
        Surface temperature (K)
    D_H2O : float (default, 0.01)
        Bulk distribution of water between silicate and melt
    rho_m : float (default, 3.3e3)
        Layer density [kg m-3]
    alpha : float (default, 2.0e-5)
        Coefficient of thermal expansion [K-1]
    cp : float (default, 1.2e3)
        Heat capacity (J/kg/K)
    km : float (default, 4.2)
        Thermal conductivity [W m-1 K-1]
        
    Returns
    -------
    Dmelt : float
        Total thickness of melt zones (m)
    rmor : float
        Mass degassing rate per unit volume (kg m-3)
    avgfmelt : float
        Volume-averaged melt fraction over melt regions
    avgXmelt : float
        Volume-averaged water content in melt (mass fraction)
        
    Notes
    -----    
    The function can handle multiple disconnected melt regions and calculates
    the total degassing contribution from all regions.

    References
    ----------
    Based on the approach described in:
    Sandu, C., et al. (2011). Convective heat transfer and particle entrainment 
    in experimentally simulated magma oceans. JGR, 116, B12201.
    """

    # Depth array from surface to 300 km (following Sandu et al. May need to modify this in future.)
    z = np.arange(0, 300e3 + 1e3, 1e3)  # meters
    n_points = len(z)
    
    # Initialize arrays
    melt_fraction = np.zeros(n_points)  # melt fraction
    water_content_in_melt = np.zeros(n_points)  # water content in melt
    Tprofile = np.zeros(n_points)  # temperature profile
    
    # Pressure in GPa
    press = rho_m * g * z / 1e9
    
    # Solidus and liquidus temperatures
    Tsolidus = np.minimum(104.42 * press + 1420, 26.53 * press + 1825)
    Tliquidus = Tsolidus + 600
    
    # Calculate temperature profile
    Tprofile[0] = Tsurf
    Tp = Tm  # mantle potential temperature
    
    for i in range(1, n_points):
        if z[i] < Db:
            # Boundary layer thermal conductive temperature profile
            Tprofile[i] = Tprofile[0] + z[i] * qm / km
        else:
            # Mantle adiabat
            Tprofile[i] = Tp + Tp * (alpha * g * z[i]) / cp
    
    # Calculate melt fraction and water content
    melt_fraction[0] = 0.0
    water_content_in_melt[0] = 0.0
    
    for i in range(1, n_points):
        if Tprofile[i] >= Tliquidus[i]:
            melt_fraction[i] = 1.0
        elif Tliquidus[i] > Tprofile[i] >= Tsolidus[i]:
            melt_fraction[i] = (Tprofile[i] - Tsolidus[i]) / (Tliquidus[i] - Tsolidus[i])
            melt_fraction[i] = max(0.0, min(1.0, melt_fraction[i]))  # Clamp to [0, 1]
        else:
            melt_fraction[i] = 0.0
        
        if melt_fraction[i] > 0.0:
            water_content_in_melt[i] = FH2O / (D_H2O + melt_fraction[i] * (1 - D_H2O))
        else:
            water_content_in_melt[i] = 0.0
    
    # Find melt regions
    melt_indices = np.where(melt_fraction > 0.0)[0]
    
    if len(melt_indices) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Find continuous melt regions
    firstz = melt_indices[0]
    secondz = firstz
    
    # Find end of first melt region
    for i in range(firstz, n_points - 1):
        if melt_fraction[i] > 0.0:
            secondz = i
        else:
            break
    
    # Check for additional melt regions
    thirdz = secondz + 1
    fourthz = secondz + 1
    
    if secondz + 1 < n_points:
        remaining_melt = np.where(melt_fraction[secondz + 1:] > 0.0)[0]
        if len(remaining_melt) > 0:
            thirdz = remaining_melt[0] + secondz + 1
            fourthz = n_points - 1
            # Find actual end of melt region
            for i in range(thirdz, n_points):
                if melt_fraction[i] > 0.0:
                    fourthz = i
    
    # Calculate total melt layer thickness
    Dmelt = z[secondz] - z[firstz]
    if thirdz < fourthz and thirdz < n_points and fourthz < n_points:
        Dmelt += z[fourthz] - z[thirdz]
    
    # Calculate radius array
    r = Rp - z

    # Calculate volume-averaged properties
    if Dmelt > 0.0:
        if fourthz > thirdz and thirdz < n_points:
            # Two melt regions
            r_vals = r[firstz:fourthz + 1]
            y_vals = melt_fraction[firstz:fourthz + 1]
            Xmelt_vals = water_content_in_melt[firstz:fourthz + 1]
            
            # Integration is occuring backward so we need to negative the results [JPR added 2025-08-20]
            product1 = -np.trapz(y_vals * r_vals**2, r_vals)
            product2 = -np.trapz(Xmelt_vals * r_vals**2, r_vals)
            
            volume_total = (r[firstz]**3 - r[secondz]**3 + 
                          r[thirdz]**3 - r[fourthz]**3)
            
            avgfmelt = 3 * product1 / volume_total
            avgXmelt = 3 * product2 / volume_total
        else:
            # Single melt region
            r_vals = r[firstz:secondz + 1]
            y_vals = melt_fraction[firstz:secondz + 1]
            Xmelt_vals = water_content_in_melt[firstz:secondz + 1]
            
            # Integration is occuring backward so we need to negative the results [JPR added 2025-08-20]
            product1 = -np.trapz(y_vals * r_vals**2, r_vals)
            product2 = -np.trapz(Xmelt_vals * r_vals**2, r_vals)
            
            volume_total = r[firstz]**3 - r[secondz]**3
            
            avgfmelt = 3 * product1 / volume_total
            avgXmelt = 3 * product2 / volume_total
        
        rmor = avgfmelt * avgXmelt * rho_m
    else:
        avgfmelt = 0.0
        avgXmelt = 0.0
        rmor = 0.0
    
    return Dmelt, rmor, avgfmelt, avgXmelt


@njit
def degas_method_2(Tm: float, Db: float, qm: float, FH2O: float,
                   Rp: float, g: float, Tsurf: float,
                   D_H2O: float = 0.01, rho_m: float = 3.3e3, alpha: float = 2.0e-5,
                   cp: float = 1.2e3, km: float = 4.2) -> Tuple[float, float, float, float]:
    """
    Calculate degassing rate using method 2.
    
    This function provides a simplified approach to calculating degassing rates
    by assuming continuous melt regions and using a more straightforward
    volume averaging approach.
    
    Parameters
    ----------
    Tm : float
        Mantle potential temperature (K)
    Db : float
        Boundary layer depth (m)
    qm : float
        Mantle heat flux (W/m²)
    FH2O : float
        Bulk water content in mantle (mass fraction)
    Rp : float
        Planetary radius (m)
    g : float
        Gravitational acceleration (m/s²)
    Tsurf : float
        Surface temperature (K)
    D_H2O : float (default, 0.01)
        Bulk distribution of water between silicate and melt
    rho_m : float (default, 3.3e3)
        Layer density [kg m-3]
    alpha : float (default, 2.0e-5)
        Coefficient of thermal expansion [K-1]
    cp : float (default, 1.2e3)
        Heat capacity (J/kg/K)
    km : float (default, 4.2)
        Thermal conductivity [W m-1 K-1]
        
    Returns
    -------
    Dmelt : float
        Thickness of melt zone (m)
    rmor : float
        Mass degassing rate per unit volume (kg/m³)
    meltfrac : float
        Volume-averaged melt fraction over entire melt region
    avgXmelt : float
        Volume-averaged water content in melt (mass fraction)
        
    Notes
    -----
    This method differs from method 1 in several ways:
    
    1. **Simpler melt region handling**: Assumes a single continuous melt region
    2. **Direct volume averaging**: Uses the full extent from first to last melt
    
    The physical formulations for temperature profiles, solidus/liquidus,
    and water partitioning are identical to method 1.
    
    The volume-averaged melt fraction is calculated as:
    meltfrac = 3 ∫[r₁→r₂] f(r) * r² dr / (r₂³ - r₁³)
    
    where r₁ and r₂ are the inner and outer radii of the melt region.
    
    See Also
    --------
    degas_method_1 : More complex method handling discontinuous melt regions
    """
    # Depth array
    z = np.arange(0, 300e3 + 1e3, 1e3)
    n_points = len(z)
    
    # Initialize arrays
    water_content_in_melt = np.zeros(n_points)
    Tadiabat = np.zeros(n_points)
    fraction = np.zeros(n_points)
    
    # Pressure in GPa
    press = rho_m * g * z / 1e9
    
    # Solidus and liquidus temperatures
    Tsolidus = np.minimum(104.42 * press + 1420, 26.53 * press + 1825)
    Tliquidus = Tsolidus + 600
    
    # Calculate temperature profile
    Tadiabat[0] = Tsurf
    Tp = Tm
    
    for i in range(1, n_points):
        if z[i] < Db:
            Tadiabat[i] = Tadiabat[0] + z[i] * qm / km
        else:
            Tadiabat[i] = Tp + Tp * (alpha * g * z[i]) / cp
    
    # Find melt fraction for each layer
    for i in range(n_points):
        if Tadiabat[i] > Tliquidus[i]:
            fraction[i] = 1.0
            water_content_in_melt[i] = FH2O
        elif Tadiabat[i] > Tsolidus[i]:
            fraction[i] = ((Tadiabat[i] - Tsolidus[i]) / 
                          (Tliquidus[i] - Tsolidus[i]))
            water_content_in_melt[i] = FH2O / (D_H2O + fraction[i] * (1 - D_H2O))
        else:
            fraction[i] = 0.0
            water_content_in_melt[i] = 0.0
    
    # Find indices where there is melt
    I = np.where(fraction > 0)[0]
    
    # If no melt or insufficient melt region
    if len(I) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate melt region properties
    first_idx = I[0]
    last_idx = I[-1]
    
    # Radius array for melt region
    r = Rp - z[first_idx:last_idx + 1]
    
    # Volume-averaged melt fraction and water content
    melt_fractions = fraction[first_idx:last_idx + 1]
    water_contents = water_content_in_melt[first_idx:last_idx + 1]
    
    # Volume averaging using spherical shells
    volume_total = (4.0 / 3.0) * np.pi * (r[0]**3 - r[-1]**3)  # [JPR Modified 2025-08-20]
    
    # Integration is occuring backward so we need to negative the results [JPR added 2025-08-20]
    meltfrac = (3 * -np.trapz(melt_fractions * r**2, r) / volume_total)
    avgXmelt = (3 * -np.trapz(water_contents * r**2, r) / volume_total)
    
    # Ensure melt fraction doesn't exceed 1 due to numerical issues
    meltfrac = min(meltfrac, 1.0)
    
    # Calculate melt layer thickness
    Dmelt = z[last_idx] - z[first_idx]
    
    # Calculate degassing rate
    if Dmelt > 0:
        rmor = meltfrac * avgXmelt * rho_m
    else:
        rmor = 0.0
    
    return Dmelt, rmor, meltfrac, avgXmelt
