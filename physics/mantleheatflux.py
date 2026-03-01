import numpy as np
from .viscosity import viscosity

def mantleheatflux(
        temp_mantle,
        temp_surface,
        radius_solid,
        radius_planet,
        radius_core,
        gravity,
        density_mantle,
        mass_frac_water,
        melt_fraction,
        params):
    """
    Calculates the convective heat flux and boundary layer properties of the mantle 
    using parameterized Rayleigh-Bénard convection scaling.

    Parameters
    ----------
    temp_mantle : float
        Potential temperature of the convective mantle (K).
    temp_surface : float
        Surface temperature of the planet (K).
    radius_solid : float
        Radius of the solidification front / base of the magma ocean (m).
    radius_planet : float
        Total radius of the planet (m).
    radius_core : float
        Radius of the planetary core (m).
    gravity : float
        Surface gravitational acceleration (m/s^2).
    density_mantle : float
        Bulk density of the mantle (kg/m^3).
    mass_frac_water : float
        Mass fraction of water in the convecting region.
    melt_fraction : float
        Volume-averaged melt fraction of the convecting region.
    params : dict
        Master configuration dictionary.

    Returns
    -------
    heat_flux_mantle, depth_boundary_layer, velocity_spreading, rayleigh_number, kinematic_viscosity
    """

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']
    conv   = params['planet']['convection']

    thermal_conductivity = thermo['thermal_conductivity']
    thermal_expansion    = thermo['thermal_expansion']
    heat_capacity        = thermo['specific_heat_mantle']
    
    melt_threshold   = conv['melt_fraction_threshold']
    nusselt_coeff    = conv['nusselt_coefficient']
    spreading_factor = conv['spreading_time_factor']

    # --- Convective Geometry ---
    # Rheological transition: If mostly molten, convection is restricted to the liquid layer.
    if melt_fraction >= melt_threshold:
        depth_convective_zone = radius_planet - radius_solid
    else:
        # Solid-state convection throughout the entire mantle
        depth_convective_zone = radius_planet - radius_core

    # --- Fluid Dynamics Properties ---
    thermal_diffusivity = thermal_conductivity / (density_mantle * heat_capacity)
    
    # Calculate viscosity (updated to pass params for its own TOML lookups)
    kinematic_viscosity = viscosity(
        temp_mantle, temp_surface, density_mantle, mass_frac_water, melt_fraction, params
    )

    # --- Rayleigh Number Calculation ---
    temp_difference = abs(temp_mantle - temp_surface)
    
    rayleigh_number = (gravity * thermal_expansion * temp_difference * depth_convective_zone**3) / \
                      (kinematic_viscosity * thermal_diffusivity)

    # --- Heat Flux & Boundary Layer ---
    # Nusselt scaling: Nu = C * Ra^(1/3)
    heat_flux_mantle = (
        nusselt_coeff * thermal_conductivity * temp_difference * 
        (rayleigh_number**(1.0 / 3.0)) / depth_convective_zone
        )

    # Fourier's Law for boundary layer thickness
    depth_boundary_layer = thermal_conductivity * temp_difference / heat_flux_mantle

    # Spreading velocity / characteristic time
    time_spreading = (depth_boundary_layer**2) / (spreading_factor * thermal_diffusivity)
    velocity_spreading = depth_convective_zone / time_spreading

    return heat_flux_mantle, depth_boundary_layer, velocity_spreading, rayleigh_number, kinematic_viscosity
