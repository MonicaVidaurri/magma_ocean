import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_meltfrac import get_meltfrac
from utils.get_meltfracb import get_meltfracb
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_flux import get_flux
from utils.get_loss import get_loss
from physics.degas2 import degas2
from physics.radiogenics import get_radiogenic_heat
from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def moODE_unified(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, Ts, Ps, OLR, ASR, 
        t_flux, Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, Mp, LStar, Rh, Mh, 
        tides_on_flag, params
    ):
    """ 
    Unified ODE handling Magma Ocean, Solidification, and Tectonic Degassing. 
    The physics regime dynamically branches based on the crustal thickness.
    """
    
    # =====================================================================
    # --- Unpack Parameters ---
    # =====================================================================
    constants = params['constants']
    mat       = params['planet']['material']
    atm       = params['planet']['atmosphere']
    thermo    = params['planet']['thermodynamics']
    comp      = params['planet']['oxide_composition']

    molar_mass_O      = constants['molar_mass_O']
    molar_mass_H2O    = constants['molar_mass_H2O']
    molar_mass_H      = constants['molar_mass_H']
    molar_mass_FeO1_5 = comp['molar_mass_FeO1_5']
    
    heat_capacity_mantle     = thermo['specific_heat_mantle']
    heat_capacity_water      = thermo['specific_heat_H2O']
    density_crust            = mat['density_mantle']
    latent_heat_fusion       = thermo['latent_heat_fusion']
    latent_heat_vaporization = thermo['latent_heat_vaporization']
    thermal_expansion        = thermo['thermal_expansion']
    crit_temp_water = atm['critical_temp_H2O']
    vapor_a         = atm['vapor_press_a']
    vapor_b         = atm['vapor_press_b']

    surface_area = 4.0 * np.pi * Rp**2

    # =====================================================================
    # --- Unpack State Vector ---
    # =====================================================================    
    semi_a       = Tr[0]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)
    eccentricity = Tr[1]
    spin_freq_h  = Tr[2]
    spin_freq_p  = Tr[3]
    temp_mantle  = Tr[4]
    
    # Clamp radius_solid to strictly physical bounds for the math
    radius_solid = Tr[5]
    if radius_solid < Rc:
        radius_solid = Rc
    elif radius_solid > Rp:
        radius_solid = Rp

    mass_water_solid      = max(0.0, Tr[6])
    mass_frac_water_solid = mass_water_solid / Mmantle

    # Protect against negative volatile masses from solver undershoots
    mass_water_atm  = max(Tr[7], 0.0)
    mass_oxygen_atm = max(Tr[8], 0.0)
    temp_surface = Tr[10]

    # Mantle O2 Mass is currently a one way sink so it is unused. 
    # NOTE: in the future we may want to access this O2 again for further outgassing
    mass_O2_in_mantle = Tr[9]

    # Assume mantle is always at least 1 degree hotter than surface.
    temp_mantle = max(temp_mantle, temp_surface + 1.0)

    # =====================================================================
    # --- Phase Branching Logic ---
    # =====================================================================
    is_magma_ocean = (radius_solid < Rp - 0.01)
    mass_magma_ocean = max(0.0, (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - radius_solid**3))

    # Thermodynamics (B_coeff controls melting/solidifying rate)
    if (Rp - radius_solid) > (405e9 / 77.89 / rho_mantle / g):
        Tsol_a = thermo['solidus_slope_high_p'] * 1e-9 
        Tsol_b = thermo['solidus_intercept_high_p']
    else:
        Tsol_a = thermo['solidus_slope_low_p'] * 1e-9 
        Tsol_b = thermo['solidus_intercept_low_p']

    B_coeff = (
        (heat_capacity_mantle * (Tsol_b * thermal_expansion - Tsol_a * rho_mantle * heat_capacity_mantle)) / 
        (g * (Tsol_a * rho_mantle * heat_capacity_mantle - thermal_expansion * temp_mantle)**2)
    )

    # --- Regime-Specific Properties ---
    if is_magma_ocean:
        _, meltfrac = get_meltfrac(g, temp_mantle, Rp, Rc, Mmantle, params)
        
        if mass_water_atm > 0.0 and mass_magma_ocean > 0.0:
            mass_frac_water_melt = min(mass_water_atm / mass_magma_ocean, 1.0)
            pressure_H2O, mass_frac_water_melt, partition_coeff_H2O = get_pressure2(
                temp_mantle, radius_solid, mass_magma_ocean, Mmantle, Rp, g, Rc, mass_water_atm, params
            )
        else:
            pressure_H2O = 0.0
            mass_frac_water_melt = 0.0
            partition_coeff_H2O = 0.0

        if mass_oxygen_atm > 0.0 and mass_magma_ocean > 0.0:
            pressure_O2, mass_frac_FeO1_5, _, _ = get_massbalance4(
                temp_mantle, pressure_H2O, mass_magma_ocean, mass_oxygen_atm, Xi, FeOt, g, Rp, params
            )
            if pressure_O2 < 0.0:
                pressure_O2 = mass_oxygen_atm * g / surface_area
                mass_frac_FeO1_5 = 0.0
        else:
            pressure_O2 = 0.0
            mass_frac_FeO1_5 = 0.0
            
        Db_phys = radius_solid
        heatflux_water_frac = mass_frac_water_melt
        
    else:
        if temp_mantle > thermo['solidus_intercept_low_p']:
            _, meltfrac = get_meltfracb(g, temp_mantle, Rp, Rc, Mmantle, params) 
        else:
            meltfrac = 0.0
            
        mass_frac_water_melt = 0.0
        partition_coeff_H2O = 0.0
        mass_frac_FeO1_5 = 0.0
        
        if temp_surface > crit_temp_water:
            pressure_H2O = mass_water_atm * g / surface_area
        else:
            pressure_H2O = 10 ** (vapor_a - vapor_b / temp_surface) * 1e5
            if (pressure_H2O * surface_area / g) > mass_water_atm:
                pressure_H2O = mass_water_atm * g / surface_area
        
        pressure_O2 = mass_oxygen_atm * g / surface_area
        
        Db_phys = Rc # Convective zone covers the entire solid mantle
        heatflux_water_frac = mass_frac_water_solid

    # =====================================================================
    # --- Energy Fluxes & Loss Rates ---
    # =====================================================================
    flux_to_space            = get_flux(temp_surface, Teq, pressure_H2O, pressure_O2, Rp, g, params)
    flux_loss_H, flux_loss_O = get_loss(t_flux, Lbol, t_sec, pressure_O2, pressure_H2O, tsat, semi_a, Mp, Rp, LStar, params)
    radiogenic_heating_watts = get_radiogenic_heat(t_sec, Mmantle, params)

    q_mantle, Db, uc, Ra, nu = mantleheatflux(
        temp_mantle, temp_surface, Db_phys, Rp, Rc, g, rho_mantle, heatflux_water_frac, meltfrac, params
    )

    if is_magma_ocean:
        tidal_radius = radius_solid
    else:
        tidal_radius = Rp
    da_dt, de_dt, dspin_dt_h, dspin_dt_p, _, tidal_heating_p, _, _, _ = calculate_tidal_dissipation(
        eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
        temp_mantle, Rc, nu, rho_mantle, meltfrac, tidal_radius, tides_on_flag, params
    )

    # =====================================================================
    # --- Intermediate Rate Calculations ---
    # =====================================================================
    mantle_cooling_watts       = surface_area * q_mantle
    total_mantle_heating_watts = radiogenic_heating_watts + tidal_heating_p
    
    if is_magma_ocean:
        thermal_inertia_mantle = (heat_capacity_mantle * Mmantle) - \
            (4.0 * np.pi * rho_mantle * latent_heat_fusion * radius_solid**2) * B_coeff
    else:
        thermal_inertia_mantle = heat_capacity_mantle * Mmantle

    dT_m_dt = (-mantle_cooling_watts + total_mantle_heating_watts) / thermal_inertia_mantle
    
    rate_radius_solidified = B_coeff * dT_m_dt
    # Clamp crustal growth once solid
    if not is_magma_ocean and rate_radius_solidified > 0.0:
        rate_radius_solidified = 0.0

    mass_solidified = 4.0 * np.pi * rho_mantle * radius_solid**2 * rate_radius_solidified

    # Mass Loss & Degassing
    total_mass_loss_H           = surface_area * flux_loss_H
    total_mass_loss_O           = surface_area * flux_loss_O
    water_loss_to_space         = total_mass_loss_H * (molar_mass_H2O / (2.0 * molar_mass_H))
    oxygen_generated_from_water = total_mass_loss_H * (molar_mass_O / (2.0 * molar_mass_H))

    # NOTE FOR FUTURE: To prevent volatiles from being reabsorbed into the magma ocean upon 
    # remelting (rate_radius_solidified < 0), we strictly clamp the partitioning mass 
    # to >= 0. To allow reabsorption, remove this max() clamp.
    safe_mass_solidified = max(mass_solidified, 0.0)

    if is_magma_ocean:
        water_partitioned_to_solid  = partition_coeff_H2O * mass_frac_water_melt * safe_mass_solidified if mass_frac_water_melt > 0 else 0.0
        oxygen_partitioned_to_solid = mass_frac_FeO1_5 * 0.5 * (molar_mass_O / molar_mass_FeO1_5) * safe_mass_solidified
        total_degassing_rate_kg_s = 0.0
    else:
        water_partitioned_to_solid  = 0.0
        oxygen_partitioned_to_solid = 0.0
        if mass_frac_water_solid > 1e-9:
            _, degas_rate, _, _ = degas2(temp_mantle, Db, q_mantle, mass_frac_water_solid, Rp, g, temp_surface, params) 
            total_degassing_rate_kg_s = degas_rate * surface_area * uc
        else:
            total_degassing_rate_kg_s = 0.0

    # Surface Thermal Properties
    net_surface_power = surface_area * (q_mantle - flux_to_space)
    latent_heat_capacity = 0.0
    if mass_water_atm > 0:
        # Calculate the theoretical condensation temperature (T_sat) for the current water mass
        P_actual = mass_water_atm * g / surface_area
        P_crit = 10**(vapor_a - vapor_b / crit_temp_water) * 1e5
        
        # Cap T_sat at the critical point (water cannot condense above 647 K)
        if P_actual >= P_crit:
            T_sat = crit_temp_water
        else:
            # Invert Clausius-Clapeyron to find condensation temperature
            T_sat = vapor_b / (vapor_a - np.log10(P_actual / 1e5))
        
        # ODE Smoothing: Use a hyperbolic tangent to create a smooth activation curve.
        # It equals 0.0 when T_surf > T_sat, and smoothly ramps to 1.0 when T_surf < T_sat.
        # We blend it over a 2.0 Kelvin window so the solver doesn't hit a mathematical wall.
        transition_width = 2.0 
        activation = 0.5 * (1.0 + np.tanh((T_sat - temp_surface) / transition_width))
        
        # Apply latent heat only if the switch is actively blending
        if activation > 1e-4:
            P_sat = 10**(vapor_a - vapor_b / temp_surface) * 1e5
            dP_sat_dT = P_sat * np.log(10) * vapor_b / (temp_surface**2)
            latent_heat_capacity = latent_heat_vaporization * (surface_area / g) * dP_sat_dT * activation
    
    heat_cap_atm   = heat_capacity_water * (pressure_H2O * surface_area / g)
    if is_magma_ocean:
        crust_depth_phys = radius_solid
    else:
        crust_depth_phys = Db
    heat_cap_crust = heat_capacity_mantle * density_crust * ((4.0 / 3.0) * np.pi * (Rp**3 - (Rp - crust_depth_phys)**3))
    
    total_surface_heat_capacity = heat_cap_atm + heat_cap_crust + latent_heat_capacity

    # =====================================================================
    # --- DIFFERENTIAL EQUATIONS ---
    # =====================================================================
    dTr_dt = np.zeros(11, dtype=np.float64)
    dTr_dt[0] = da_dt
    dTr_dt[1] = de_dt
    dTr_dt[2] = dspin_dt_h
    dTr_dt[3] = dspin_dt_p
    dTr_dt[4] = dT_m_dt
    dTr_dt[5] = rate_radius_solidified
    
    # Solid Mantle Water
    dTr_dt[6] = water_partitioned_to_solid - total_degassing_rate_kg_s
    
    # Atmosphere/Magma Water
    if is_magma_ocean:
        dTr_dt[7] = -water_partitioned_to_solid - water_loss_to_space if mass_water_atm > 0 else 0.0
    else:
        dTr_dt[7] = total_degassing_rate_kg_s - water_loss_to_space

    # Atmosphere/Magma Oxygen
    if is_magma_ocean:
        if mass_water_atm > 0 and pressure_O2 > 0:
            dTr_dt[8] = oxygen_generated_from_water - total_mass_loss_O - oxygen_partitioned_to_solid
        elif mass_water_atm > 0 and pressure_O2 <= 0:
            dTr_dt[8] = oxygen_generated_from_water - oxygen_partitioned_to_solid
        elif mass_water_atm <= 0 and pressure_O2 > 0:
            dTr_dt[8] = -oxygen_partitioned_to_solid
        else:
            dTr_dt[8] = 0.0
    else:
        dTr_dt[8] = oxygen_generated_from_water - total_mass_loss_O

    # Solid Mantle Oxygen
    dTr_dt[9] = oxygen_partitioned_to_solid
    dTr_dt[10] = net_surface_power / total_surface_heat_capacity

    return dTr_dt
