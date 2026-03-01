import numpy as np

def get_flux(
        temp_surface,
        temp_equilibrium,
        pressure_surface_pa,
        radius_planet,
        gravity):
    """
    Calculates the net outgoing longwave thermal flux from the planet's surface 
    to space, accounting for the greenhouse effect of a water vapor atmosphere.

    This uses an analytical gray-atmosphere approximation (Eddington approximation) 
    where the optical depth scales with pressure-broadened water vapor absorption.

    Parameters
    ----------
    temp_surface : float
        Surface temperature of the planet in Kelvin.
    temp_equilibrium : float
        Equilibrium temperature of the planet in Kelvin (acts as a proxy for 
        absorbed stellar radiation).
    pressure_surface_pa : float
        Surface atmospheric pressure in Pascals.
    radius_planet : float
        Radius of the planet in meters.
    gravity : float
        Surface gravitational acceleration in m/s^2.

    Returns
    -------
    net_heat_flux : float
        The net thermal flux leaving the surface (W/m^2). 
        Positive values indicate cooling (the surface is losing heat to space).
    """
    
    # --- Constants ---
    stefan_boltzmann_const = 5.67e-8    # sigma (W/m^2/K^4)
    absorption_coeff_water = 0.01       # k0 (m^2/kg) - Reference absorption coeff for H2O
    pressure_reference_pa  = 1.01325e5  # p0 (Pa) - 1 bar reference pressure

    # --- Atmospheric Mass & Optical Depth ---
    surface_area = 4.0 * np.pi * radius_planet**2
    
    # Total mass of the atmosphere (kg)
    mass_atmosphere_kg = (pressure_surface_pa * surface_area) / gravity
    
    # Optical depth (tau) of a pressure-broadened gray atmosphere.
    # Note: Mathematically, this column_mass_factor reduces to 1.5 * (P_surf / g)
    column_mass_factor = (3.0 * mass_atmosphere_kg) / (2.0 * surface_area)
    
    # Pressure broadening scalar
    pressure_broadening_factor = np.sqrt((absorption_coeff_water * gravity) / (3.0 * pressure_reference_pa))
    
    optical_depth = column_mass_factor * pressure_broadening_factor

    # --- Effective Emissivity & Net Flux ---
    # Eddington approximation for the effective emissivity of a gray atmosphere
    effective_emissivity = 2.0 / (optical_depth + 2.0)

    # Net thermal cooling flux (Outgoing Longwave - Absorbed Shortwave proxy)
    net_heat_flux = effective_emissivity * stefan_boltzmann_const * (temp_surface**4 - temp_equilibrium**4)

    return net_heat_flux
