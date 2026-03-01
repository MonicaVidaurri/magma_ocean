import numpy as np

def get_flux(
        temp_surface,
        temp_equilibrium,
        pressure_surface_pa,
        radius_planet,
        gravity,
        params):
    """
    Calculates the net outgoing longwave thermal flux from the planet's surface 
    to space using an analytical gray-atmosphere (Eddington) approximation.

    Parameters
    ----------
    temp_surface : float
        Surface temperature of the planet (K).
    temp_equilibrium : float
        Equilibrium temperature (K).
    pressure_surface_pa : float
        Surface atmospheric pressure (Pa).
    radius_planet : float
        Radius of the planet (m).
    gravity : float
        Surface gravitational acceleration (m/s^2).
    params : dict
        Master configuration dictionary containing [constants] and [planet.atmosphere].

    Returns
    -------
    net_heat_flux : float
        The net thermal flux leaving the surface (W/m^2).
    """
    
    # --- Unpack Parameters ---
    sigma    = params['constants']['stefan_boltzmann']
    p_ref    = params['constants']['pressure_ref']
    k_water  = params['planet']['atmosphere']['absorption_coeff_water']

    # --- Atmospheric Mass & Optical Depth ---
    surface_area = 4.0 * np.pi * radius_planet**2
    
    # Total mass of the atmosphere (kg)
    mass_atmosphere = (pressure_surface_pa * surface_area) / gravity
    
    # Optical depth (tau) calculation
    # Column mass factor simplifies to 1.5 * (P_surf / g)
    column_mass_factor = (3.0 * mass_atmosphere) / (2.0 * surface_area)
    
    # Pressure broadening scalar
    pressure_broadening = np.sqrt((k_water * gravity) / (3.0 * p_ref))
    
    tau = column_mass_factor * pressure_broadening

    # --- Effective Emissivity & Net Flux ---
    # Eddington approximation: epsilon_eff = 2 / (tau + 2)
    emissivity_eff = 2.0 / (tau + 2.0)

    # Stefan-Boltzmann Law for net cooling
    net_heat_flux = emissivity_eff * sigma * (temp_surface**4 - temp_equilibrium**4)

    return net_heat_flux
