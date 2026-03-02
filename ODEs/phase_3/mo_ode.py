import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_meltfracb import get_meltfracb
from utils.get_flux import get_flux
from utils.get_loss import get_loss
from physics.tides import calculate_tidal_dissipation
from physics.radiogenics import get_radiogenic_heat
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def moODE_phase3(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, Ts, Ps, OLR, ASR, 
        t_flux, Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, FH2O, Mp, LStar, Rh, 
        Mh, tides_on_flag, params
    ):
    """ Tectonic phase / last time phase. Stable sub-solidus cooling. """

    # =====================================================================
    # --- Unpack Parameters ---
    # =====================================================================
    c      = params['constants']
    atm    = params['planet']['atmosphere']
    mat    = params['planet']['material']
    thermo = params['planet']['thermodynamics']

    # =====================================================================
    # --- Constants & Molar Masses ---
    # =====================================================================
    molar_mass_O   = c['molar_mass_O']
    molar_mass_H2O = c['molar_mass_H2O']
    molar_mass_H   = c['molar_mass_H']
    
    heat_capacity_mantle = thermo['specific_heat_mantle']
    heat_capacity_water  = thermo['specific_heat_H2O']
    density_crust        = mat['density_mantle'] # Standardized to density_mantle
    crit_temp_water      = atm['critical_temp_H2O']
    
    vapor_a = atm['vapor_press_a']
    vapor_b = atm['vapor_press_b']
    
    surface_area = 4.0 * np.pi * Rp**2

    # =====================================================================
    # --- Unpack State Vector ---
    # =====================================================================
    dTr_dt = np.zeros(8, dtype=np.float64)
    
    semi_a       = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h  = Tr[2]
    spin_freq_p  = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    temp_mantle     = Tr[4]
    mass_water_atm  = max(Tr[5], 0.0)
    mass_oxygen_atm = max(Tr[6], 0.0)
    temp_surface    = Tr[7]

    # =====================================================================
    # --- Mantle Melt Fraction & Properties ---
    # =====================================================================
    # Reference solidus intercept for trigger
    if temp_mantle > thermo['solidus_intercept_low_p']:
        _, meltfrac = get_meltfracb(g, temp_mantle, Rp, Rc, Mmantle, params) 
    else:
        meltfrac = 0.0

    # =====================================================================
    # --- Atmospheric Pressures ---
    # =====================================================================
    if temp_surface > crit_temp_water:
        pressure_H2O = mass_water_atm * g / surface_area
    else:
        # P = 10**(A - B/T) * 1e5 [Pa]
        pressure_H2O = 10**(vapor_a - vapor_b / temp_surface) * 1e5
        if (pressure_H2O * surface_area / g) > mass_water_atm:
            pressure_H2O = mass_water_atm * g / surface_area

    pressure_O2 = mass_oxygen_atm * g / surface_area

    # =====================================================================
    # --- Energy Fluxes & Loss Rates ---
    # =====================================================================
    flux_to_space = get_flux(
        temp_surface,
        Teq, 
        pressure_H2O,
        pressure_O2,
        Rp,
        g,
        params
    )
    
    flux_loss_H, flux_loss_O = get_loss(
        t_flux, Lbol, t_sec, pressure_O2, pressure_H2O, tsat, semi_a, Mp, Rp, LStar, params
    )
    
    radiogenic_heating_watts = get_radiogenic_heat(t_sec, Mmantle, params)

    q_mantle, Db, uc, Ra, nu = mantleheatflux(
        temp_mantle, temp_surface, Rp, Rp, Rc, g, rho_mantle, FH2O, meltfrac, params
    )

    da_dt, de_dt, dspin_dt_h, dspin_dt_p, _, tidal_heating_p = calculate_tidal_dissipation(
        eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
        temp_mantle, Rc, nu, rho_mantle, meltfrac, tides_on_flag, params
    )

    # =====================================================================
    # --- Intermediate Rate Calculations ---
    # =====================================================================
    
    # Mantle Heat & Thermal Inertia
    mantle_cooling_watts       = surface_area * q_mantle
    total_mantle_heating_watts = radiogenic_heating_watts + tidal_heating_p
    thermal_inertia_mantle     = heat_capacity_mantle * Mmantle

    # Mass Loss & Generation
    if temp_surface > crit_temp_water:
        total_mass_loss_H_kg_s = surface_area * flux_loss_H
        total_mass_loss_O_kg_s = surface_area * flux_loss_O
        
        water_loss_to_space_kg_s         = total_mass_loss_H_kg_s * (molar_mass_H2O / (2.0 * molar_mass_H))
        oxygen_generated_from_water_kg_s = total_mass_loss_H_kg_s * (molar_mass_O / (2.0 * molar_mass_H))
    else:
        water_loss_to_space_kg_s         = 0.0
        oxygen_generated_from_water_kg_s = 0.0
        total_mass_loss_O_kg_s           = 0.0

    # Surface Thermal Properties
    net_surface_power = surface_area * (q_mantle - flux_to_space)
    
    heat_cap_atm   = heat_capacity_water * (pressure_H2O * surface_area / g)
    heat_cap_crust = heat_capacity_mantle * density_crust * ((4.0 / 3.0) * np.pi * (Rp**3 - (Rp - Db)**3))
    total_surface_heat_capacity = heat_cap_atm + heat_cap_crust

    # =====================================================================
    # --- DIFFERENTIAL EQUATIONS ---
    # =====================================================================
    dTr_dt[0] = da_dt
    dTr_dt[1] = de_dt
    dTr_dt[2] = dspin_dt_h
    dTr_dt[3] = dspin_dt_p

    dTr_dt[4] = (-mantle_cooling_watts + total_mantle_heating_watts) / thermal_inertia_mantle

    dTr_dt[5] = -water_loss_to_space_kg_s
    dTr_dt[6] = oxygen_generated_from_water_kg_s - total_mass_loss_O_kg_s

    dTr_dt[7] = net_surface_power / total_surface_heat_capacity

    return dTr_dt
