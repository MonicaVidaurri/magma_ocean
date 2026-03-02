import numpy as np

def get_flux(
        temp_surface,
        temp_equilibrium,
        pressure_h2o_pa,
        pressure_o2_pa,
        radius_planet,
        gravity,
        params):
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
    pressure_h2o_pa : float
        Surface atmospheric pressure of h2o in Pascals.
    pressure_o2_pa : float
        Surface atmospheric pressure of o2 in Pascals.
    radius_planet : float
        Radius of the planet in meters.
    gravity : float
        Surface gravitational acceleration in m/s^2.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    net_heat_flux : float
        The net thermal flux leaving the surface (W/m^2). 
        Positive values indicate cooling (the surface is losing heat to space).
    """

    # --- Unpack Parameters ---
    c = params['constants']
    atm = params['planet']['atmosphere']
    
    # --- Constants ---
    stefan_boltzmann_const = c['stefan_boltzmann']     # sigma (W/m^2/K^4)
    absorption_coeff_h2o   = atm['absorption_coeff_H2O'] # k0 (m^2/kg) - Reference absorption coeff for H2O
    absorption_coeff_o2    = atm.get('absorption_coeff_O2', 1.0e-5)
    pressure_reference_pa  = c['pressure_ref']         # p0 (Pa) - 1 bar reference pressure

    # --- Atmospheric Mass & Optical Depth ---
    total_pressure_pa = pressure_h2o_pa + pressure_o2_pa

    # If atmosphere is totally stripped, radiate freely
    if total_pressure_pa <= 1e-5:
        return stefan_boltzmann_const * (temp_surface**4 - temp_equilibrium**4)
    
    # Total mass of the atmosphere (kg)
    surface_area = 4.0 * np.pi * radius_planet**2
    mass_atmosphere_kg = (total_pressure_pa * surface_area) / gravity

    # Weighted mean absorption coefficient
    mass_frac_h2o = pressure_h2o_pa / total_pressure_pa
    mass_frac_o2  = pressure_o2_pa / total_pressure_pa
    kappa_mean = (mass_frac_h2o * absorption_coeff_h2o) + (mass_frac_o2 * absorption_coeff_o2)    

    # Optical depth (tau) of a pressure-broadened gray atmosphere.
    # Note: Mathematically, this column_mass_factor reduces to 1.5 * (P_surf / g)
    column_mass_factor = (3.0 * mass_atmosphere_kg) / (2.0 * surface_area)
    
    # Pressure broadening scalar
    pressure_broadening_factor = np.sqrt((kappa_mean * gravity) / (3.0 * pressure_reference_pa))
    
    optical_depth = column_mass_factor * pressure_broadening_factor

    # --- Effective Emissivity & Net Flux ---
    # Eddington approximation for the effective emissivity of a gray atmosphere
    effective_emissivity = 2.0 / (optical_depth + 2.0)

    # Net thermal cooling flux (Outgoing Longwave - Absorbed Shortwave proxy)
    net_heat_flux = effective_emissivity * stefan_boltzmann_const * (temp_surface**4 - temp_equilibrium**4)

    return net_heat_flux
