import numpy as np
from scipy.integrate import trapezoid

def get_meltfrac(
        gravity,
        temp_potential,
        radius_planet,
        radius_core,
        mass_mantle):
    """
    Calculates the volume-averaged melt fraction of the magma ocean and 
    the pressure at its base, determined by the intersection of the mantle 
    adiabat with the solidus.

    Parameters
    ----------
    gravity : float
        Surface gravitational acceleration in m/s^2.
    temp_potential : float
        Mantle potential temperature in Kelvin.
    radius_planet : float
        Radius of the planet in meters.
    radius_core : float
        Radius of the planetary core in meters.
    mass_mantle : float
        Total mass of the mantle in kg.

    Returns
    -------
    exchange_pressure_gpa : float
        Pressure at the base of the magma ocean in GPa. Represents the boundary 
        between the liquid/mushy magma ocean and the solid inner mantle.
    avg_melt_fraction : float
        Volume-averaged melt fraction over the entire depth of the magma ocean.
    """

    # --- Thermodynamic Constants ---
    thermal_expansion = 2e-5   # alpha (1/K)
    heat_capacity     = 1.2e3  # cp (J/kg/K)

    # --- Planet Geometry & Grid ---
    # Calculate bulk mantle density (kg/m^3)
    volume_mantle  = (4.0 / 3.0) * np.pi * (radius_planet**3 - radius_core**3)
    density_mantle = mass_mantle / volume_mantle

    # Depth grid from surface down to the Core-Mantle Boundary (CMB) in 5km steps
    depths = np.arange(0, radius_planet - radius_core + 5e3, 5e3)

    # Hydrostatic pressure in Pascals and GPa
    pressures_pa  = gravity * density_mantle * depths
    pressures_gpa = pressures_pa / 1e9

    # --- Thermodynamics (Solidus & Liquidus) ---
    # Piecewise linear approximation of the mantle melting curves
    temp_solidus  = np.minimum(104.42 * pressures_gpa + 1420.0, 
                              26.53 * pressures_gpa + 1825.0)
    temp_liquidus = temp_solidus + 600.0

    # Mantle adiabatic temperature profile
    temp_adiabat  = temp_potential + temp_potential * (thermal_expansion * gravity * depths / heat_capacity)

    # --- Melt Fraction Calculation ---
    melt_fraction = np.zeros_like(depths)
    
    # Fully molten regions
    melt_fraction[temp_adiabat >= temp_liquidus] = 1.0
    
    # Partially molten regions (linear interpolation between solidus and liquidus)
    partial_mask = (temp_adiabat > temp_solidus) & (temp_adiabat < temp_liquidus)
    melt_fraction[partial_mask] = (temp_adiabat[partial_mask] - temp_solidus[partial_mask]) / \
                                  (temp_liquidus[partial_mask] - temp_solidus[partial_mask])

    # --- Magma Ocean Boundary Analysis ---
    # Find all depths where melt exists
    idx_melt = np.where(melt_fraction > 0.0)[0]

    # If the mantle is completely solid (or only a single mathematically un-integratable spike)
    if len(idx_melt) < 2:
        return 0.0, 0.0

    # Index of the deepest point of the magma ocean
    idx_base = idx_melt[-1]

    # The exchange pressure is taken at the first grid point of the purely solid mantle
    if idx_base < len(depths) - 1:
        exchange_pressure_gpa = pressures_gpa[idx_base + 1]
    else:
        # Edge case: Entire mantle is molten all the way to the CMB
        exchange_pressure_gpa = pressures_gpa[idx_base]

    # --- Volume-Averaged Melt Fraction ---
    # Isolate the arrays from the surface down to the base of the magma ocean
    z_mo = depths[:idx_base + 1]
    r_mo = radius_planet - z_mo
    f_mo = melt_fraction[:idx_base + 1]

    # Integrate f(z) * r^2 dz over the magma ocean depth
    integral_f_vol = trapezoid(f_mo * r_mo**2, z_mo)
    
    # Exact geometric volume proxy for the spherical shell (divided by 4/3 pi)
    volume_mo_proxy = (r_mo[0]**3 - r_mo[-1]**3) / 3.0

    avg_melt_fraction = integral_f_vol / volume_mo_proxy
    
    # Final numerical safety clamp
    avg_melt_fraction = min(avg_melt_fraction, 1.0)

    return exchange_pressure_gpa, avg_melt_fraction
