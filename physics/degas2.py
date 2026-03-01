import numpy as np

def degas2(temp_mantle_potential, depth_boundary_layer, heat_flux_mantle, 
           mass_frac_water_bulk, radius_planet, gravity, temp_surface, params):
    """
    Calculates the mantle degassing rate and melt zone properties based on 
    a 1D interior temperature profile and pressure-dependent solidus.

    Parameters
    ----------
    temp_mantle_potential : float
        The potential temperature of the convective mantle in Kelvin.
    depth_boundary_layer : float
        The thickness of the conductive lithospheric boundary layer in meters.
    heat_flux_mantle : float
        The upward heat flux from the mantle in W/m^2.
    mass_frac_water_bulk : float
        The bulk mass fraction of water in the solid mantle (dimensionless).
    radius_planet : float
        The radius of the planet in meters.
    gravity : float
        The surface gravitational acceleration in m/s^2.
    temp_surface : float
        The surface temperature of the planet in Kelvin.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    thickness_melt_layer : float
        The total vertical thickness of all actively melting layers in meters.
    degassing_rate_proxy : float
        A proxy for the degassing rate, representing the mass of volatiles 
        released per unit volume of the melt region (kg/m^3).
    avg_melt_fraction : float
        The volume-averaged melt fraction across all melting zones.
    avg_water_in_melt : float
        The volume-averaged mass fraction of water partitioned into the melt phase.
    """
    
    # --- Unpack Parameters ---
    mat    = params['planet']['material']
    thermo = params['planet']['thermodynamics']
    vols   = params['planet']['volatiles']
    num    = params['numerical']

    # --- Constants & Partitioning ---
    partition_coeff_water = vols['partition_coeff_H2O']   # D_H2O
    density_mantle        = mat['density_mantle']         # rho_m (kg/m^3)
    thermal_expansion     = thermo['thermal_expansion']   # alpha (1/K)
    heat_capacity         = thermo['specific_heat_mantle']# cp (J/kg/K)
    thermal_conductivity  = thermo['thermal_conductivity']# km (W/m/K)
    
    max_depth = num['degas_max_depth']
    step_size = num['degas_step_size']
    
    slope_low_p  = thermo['solidus_slope_low_p']
    int_low_p    = thermo['solidus_intercept_low_p']
    slope_high_p = thermo['solidus_slope_high_p']
    int_high_p   = thermo['solidus_intercept_high_p']
    liq_offset   = thermo['liquidus_offset']

    # --- Vectorized Depth & Pressure Grid ---
    depths = np.arange(0.0, max_depth + step_size, step_size)
    
    # Sanity Check: Ensure we don't calculate deeper than the planet's center
    depths = depths[depths < radius_planet]
    
    pressures_pa  = density_mantle * gravity * depths
    pressures_gpa = pressures_pa / 1e9

    # --- Thermodynamics (Solidus & Liquidus) ---
    temp_solidus = np.minimum(slope_low_p * pressures_gpa + int_low_p,
                              slope_high_p * pressures_gpa + int_high_p)
    temp_liquidus = temp_solidus + liq_offset

    # Conductive profile (Crust) vs Adiabatic profile (Deep Mantle)
    temp_conductive = temp_surface + depths * heat_flux_mantle / thermal_conductivity
    temp_adiabatic  = temp_mantle_potential + temp_mantle_potential * (thermal_expansion * gravity * depths / heat_capacity)
    
    # Stitch them together at the boundary layer
    temp_profile = np.where(depths < depth_boundary_layer, temp_conductive, temp_adiabatic)

    # --- Melt Fraction & Water Partitioning ---
    # Calculate melt fraction (0.0 to 1.0)
    melt_fraction = (temp_profile - temp_solidus) / (temp_liquidus - temp_solidus)
    melt_fraction = np.clip(melt_fraction, 0.0, 1.0)
    
    # Strict thermodynamic bounds
    melt_fraction[temp_profile < temp_solidus] = 0.0
    melt_fraction[temp_profile >= temp_liquidus] = 1.0

    # Calculate water concentrated in the melt (Batch Melting)
    water_in_melt   = np.zeros_like(depths)
    melting_indices = melt_fraction > 0.0
    
    if np.any(melting_indices):
        # Above liquidus (melt fraction == 1.0), all water is in the melt
        water_in_melt[melting_indices] = mass_frac_water_bulk / (
            partition_coeff_water + melt_fraction[melting_indices] * (1.0 - partition_coeff_water)
        )

    # --- Robust Integration of Melt Zones ---
    # Find all distinct, contiguous zones of melting
    idx_melt = np.where(melting_indices)[0]
    
    if len(idx_melt) < 2:
        return 0.0, 0.0, 0.0, 0.0

    # Split into contiguous blocks (handles solid gaps between melt zones)
    step_diffs   = np.diff(idx_melt)
    split_points = np.where(step_diffs > 1)[0] + 1
    melt_blocks  = np.split(idx_melt, split_points)

    radii = radius_planet - depths
    
    thickness_melt_layer = 0.0
    total_volume_proxy   = 0.0
    integral_f_melt      = 0.0
    integral_X_melt      = 0.0

    for block in melt_blocks:
        # Ignore single-point melt spikes (un-integratable)
        if len(block) < 2:
            continue
            
        i_start, i_end = block[0], block[-1]
        
        # Accumulate total thickness
        thickness_melt_layer += (depths[i_end] - depths[i_start])
        
        # Exact geometric volume of this spherical shell (divided by 4/3 pi)
        # Note: top is smaller depth = larger radius
        vol_shell = radii[i_start]**3 - radii[i_end]**3
        total_volume_proxy += vol_shell
        
        # Extract blocks for integration
        z_block = depths[i_start:i_end+1]
        r_block = radii[i_start:i_end+1]
        f_block = melt_fraction[i_start:i_end+1]
        X_block = water_in_melt[i_start:i_end+1]
        
        # Integrate over depth (dz is strictly positive)
        integral_f_melt += np.trapezoid(f_block * r_block**2, z_block)
        integral_X_melt += np.trapezoid(X_block * r_block**2, z_block)

    # --- Final Averages ---
    if total_volume_proxy == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    avg_melt_fraction = 3.0 * integral_f_melt / total_volume_proxy
    avg_water_in_melt = 3.0 * integral_X_melt / total_volume_proxy

    # Degassing rate proxy (kg/m^3)
    degassing_rate_proxy = avg_melt_fraction * avg_water_in_melt * density_mantle

    return thickness_melt_layer, degassing_rate_proxy, avg_melt_fraction, avg_water_in_melt
