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

    The convective depth is set by the caller via radius_solid; this function
    does not determine the phase internally.

    Parameters
    ----------
    temp_mantle : float
        Potential temperature of the convective mantle (K).
    temp_surface : float
        Surface temperature of the planet (K).
    radius_solid : float
        Base of the active convective zone (m). The caller sets this to the
        solidification front during the MO phase, or to radius_core in the
        solid-mantle phase. radius_core is accepted but unused.
    radius_planet : float
        Total radius of the planet (m).
    radius_core : float
        Unused; kept for call-site compatibility.
    gravity : float
        Surface gravitational acceleration (m/s^2).
    density_mantle : float
        Bulk density of the mantle (kg/m^3).
    mass_frac_water : float
        Mass fraction of water in the convecting region (dimensionless).
    melt_fraction : float
        Volume-averaged melt fraction of the convecting region (dimensionless).
    params : dict
        Main configuration dictionary containing TOML parameters.

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

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']
    conv   = params['planet']['convection']

    # --- Thermodynamic Constants ---
    thermal_conductivity = thermo['thermal_conductivity'] # km (W/m/K)
    thermal_expansion    = thermo['thermal_expansion']    # alpha (1/K)
    heat_capacity        = thermo['specific_heat_mantle'] # cp (J/kg/K)

    # --- Convective Geometry ---
    # The caller sets radius_solid to the base of the active convective zone:
    # the solidification front during the MO phase, or radius_core in the solid phase.
    depth_convective_zone = radius_planet - radius_solid

    # --- Fluid Dynamics Properties ---
    # Thermal diffusivity (m^2/s)
    thermal_diffusivity = thermal_conductivity / (density_mantle * heat_capacity)
    
    # Kinematic viscosity (m^2/s)
    kinematic_viscosity = viscosity(temp_mantle, temp_surface, density_mantle, mass_frac_water, melt_fraction, params)

    # --- Rayleigh Number Calculation ---
    temp_difference = abs(temp_mantle - temp_surface)
    
    rayleigh_number = (gravity * thermal_expansion * temp_difference * depth_convective_zone**3) / (kinematic_viscosity * thermal_diffusivity)

    # --- Heat Flux & Boundary Layer Parameters ---
    # Convective heat flux scaling for hard-turbulent regime (Ra^1/3)
    nusselt_coefficient = conv['nusselt_coefficient']
    heat_flux_mantle = nusselt_coefficient * thermal_conductivity * temp_difference * (rayleigh_number**(1.0 / 3.0)) / depth_convective_zone

    # Thermal boundary layer thickness via Fourier's Law of Conduction (m)
    depth_boundary_layer = thermal_conductivity * temp_difference / heat_flux_mantle

    # Characteristic convective spreading time (s)
    spreading_factor = conv['spreading_time_factor']
    time_spreading = (depth_boundary_layer**2) / (spreading_factor * thermal_diffusivity)

    # Characteristic surface spreading velocity (m/s)
    velocity_spreading = depth_convective_zone / time_spreading

    return heat_flux_mantle, depth_boundary_layer, velocity_spreading, rayleigh_number, kinematic_viscosity
