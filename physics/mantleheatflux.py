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
        melt_fraction):
    """
    Calculates the convective heat flux and boundary layer properties of the mantle 
    using parameterized Rayleigh-Bénard convection scaling.

    This function dynamically adjusts the convective depth based on the rheological 
    state of the mantle (e.g., truncating the convective zone to the liquid magma 
    ocean if the melt fraction exceeds the rheological transition threshold).

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
        Mass fraction of water in the convecting region (dimensionless).
    melt_fraction : float
        Volume-averaged melt fraction of the convecting region (dimensionless).

    Returns
    -------
    heat_flux_mantle : float
        The convective heat flux extracted from the mantle (W/m^2).
    depth_boundary_layer : float
        The thickness of the upper thermal boundary layer (m).
    velocity_spreading : float
        The characteristic convective velocity / surface spreading rate (m/s).
    rayleigh_number : float
        The dimensionless Rayleigh number of the convective system.
    kinematic_viscosity : float
        The kinematic viscosity of the convecting material (m^2/s).
    """

    # --- Thermodynamic Constants ---
    thermal_conductivity = 4.2    # km (W/m/K)
    thermal_expansion    = 2e-5   # alpha (1/K)
    heat_capacity        = 1.2e3  # cp (J/kg/K)

    # --- Convective Geometry ---
    # Rheological transition: If melt fraction > 40%, it is a fluid-supported 
    # magma ocean. Convection is restricted to the liquid layer above the solidus.
    if melt_fraction >= 0.4:
        depth_convective_zone = radius_planet - radius_solid
    else:
        # Solid-state convection throughout the entire mantle
        depth_convective_zone = radius_planet - radius_core

    # --- Fluid Dynamics Properties ---
    # Thermal diffusivity (m^2/s)
    thermal_diffusivity = thermal_conductivity / (density_mantle * heat_capacity)
    
    # Kinematic viscosity (m^2/s)
    kinematic_viscosity = viscosity(temp_mantle, temp_surface, density_mantle, mass_frac_water, melt_fraction)

    # --- Rayleigh Number Calculation ---
    temp_difference = abs(temp_mantle - temp_surface)
    
    rayleigh_number = (gravity * thermal_expansion * temp_difference * depth_convective_zone**3) / (kinematic_viscosity * thermal_diffusivity)

    # --- Heat Flux & Boundary Layer Parameters ---
    # Convective heat flux scaling for hard-turbulent regime (Ra^1/3)
    nusselt_coefficient = 0.089
    heat_flux_mantle = nusselt_coefficient * thermal_conductivity * temp_difference * (rayleigh_number**(1.0 / 3.0)) / depth_convective_zone

    # Thermal boundary layer thickness via Fourier's Law of Conduction (m)
    depth_boundary_layer = thermal_conductivity * temp_difference / heat_flux_mantle

    # Characteristic convective spreading time (s)
    # 5.38 is a standard geometric scaling factor for boundary layer instabilities
    time_spreading = (depth_boundary_layer**2) / (5.38 * thermal_diffusivity)

    # Characteristic surface spreading velocity (m/s)
    velocity_spreading = depth_convective_zone / time_spreading

    return heat_flux_mantle, depth_boundary_layer, velocity_spreading, rayleigh_number, kinematic_viscosity
