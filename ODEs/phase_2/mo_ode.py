import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_meltfracb import get_meltfracb
from utils.get_flux import get_flux
from utils.get_loss import get_loss
from physics.degas2 import degas2
from physics.radiogenics import get_radiogenic_heat
from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def moODE_phase2(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, 
        t_flux, Lbol, tsat, Mp, LStar, Rh, Mh, 
        params, tides_on_flag
    ):
    """ 
    Phase 2: Solid-state mantle evolution and volcanic degassing. 
    Driven by the master 'params' dictionary.
    """

    # --- Unpack Constants from TOML ---
    univ = params['constants']
    mat  = params['planet']['material']
    atm_params = params['planet']['atmosphere']
    thermo = params['planet']['thermodynamics']

    mu_O    = univ['molar_mass_O']
    mu_H2O  = univ['molar_mass_h2o']
    mu_H    = univ['molar_mass_H']
    
    cp_mantle = mat['specific_heat_mantle']
    cp_water  = mat['specific_heat_h2o']
    
    crit_temp_water = atm_params['water_critical_temp']
    vp_A = atm_params['vapor_press_A']
    vp_B = atm_params['vapor_press_B']

    surface_area = 4.0 * np.pi * Rp**2

    # --- Unpack State Vector ---
    dTr_dt = np.zeros(9, dtype=np.float64)
    
    semi_a       = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h  = Tr[2]
    spin_freq_p  = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    temp_mantle           = Tr[4]
    mass_frac_water_solid = Tr[5] / Mmantle
    mass_water_atm        = max(Tr[6], 0.0)
    mass_oxygen_atm       = max(Tr[7], 0.0)
    temp_surface          = Tr[8]

    # --- Mantle Melt Fraction ---
    # Solidus check using TOML intercept
    if temp_mantle > thermo['solidus_intercept_low_p']:
        _, meltfrac = get_meltfracb(g, temp_mantle, Rp, Rc, Mmantle, params) 
    else:
        meltfrac = 0.0

    # --- Atmospheric Pressures & Condensation ---
    if temp_surface > crit_temp_water:
        # Steam atmosphere: all water is vapor
        pressure_atm = mass_water_atm * g / surface_area
    else:
        # Liquid water present: vapor pressure limited by temp
        pressure_atm = 10**(vp_A - vp_B / temp_surface) * 1e5
        # Safety: cannot have more vapor than total water
        pressure_atm = min(pressure_atm, mass_water_atm * g / surface_area)

    pressure_O2 = mass_oxygen_atm * g / surface_area

    # --- Energy Fluxes & Loss Rates ---
    flux_space = get_flux(temp_surface, Teq, pressure_atm, Rp, g, params)
    
    f_loss_H, f_loss_O = get_loss(
        t_flux, Lbol, t_sec, pressure_O2, pressure_atm, tsat, semi_a, Mp, Rp, LStar, params
    )
    
    Q_radio = get_radiogenic_heat(t_sec, Mmantle, params)

    # Use mantleheatflux (Db = boundary layer depth, uc = convective velocity)
    q_mantle, Db, uc, _, nu = mantleheatflux(
        temp_mantle, temp_surface, Rp, Rp, Rc, g, rho_mantle, mass_frac_water_solid, meltfrac, params
    )

    # Calculate degassing rate (rmor = proxy for volatiles per unit volume)
    if mass_frac_water_solid > 1e-9:
        _, rmor, _, _ = degas2(temp_mantle, Db, q_mantle, mass_frac_water_solid, Rp, g, temp_surface, params) 
    else:
        rmor = 0.0

    da, de, dsh, dsp, _, Q_tidal = calculate_tidal_dissipation(
        eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
        temp_mantle, Rc, nu, rho_mantle, meltfrac, params, tides_on_flag
    )

    # --- Intermediate Rate Calculations ---
    mantle_net_watts = Q_radio + Q_tidal - (surface_area * q_mantle)
    
    # Total volcanic degassing rate (kg/s)
    degassing_rate = rmor * surface_area * uc

    # Atmospheric loss rates
    total_H_loss = surface_area * f_loss_H
    total_O_loss = surface_area * f_loss_O
    
    H2O_loss_rate = total_H_loss * (mu_H2O / (2.0 * mu_H))
    O_gen_rate    = total_H_loss * (mu_O / (2.0 * mu_H))

    # Surface thermal balance
    heat_cap_atm   = cp_water * (pressure_atm * surface_area / g)
    heat_cap_crust = cp_mantle * 3000.0 * ((4.0 / 3.0) * np.pi * (Rp**3 - (Rp - Db)**3))
    
    net_surface_power = surface_area * (q_mantle - flux_space)

    # --- Final State Vector Assembly ---
    dTr_dt[0], dTr_dt[1], dTr_dt[2], dTr_dt[3] = da, de, dsh, dsp

    # Cooling
    dTr_dt[4] = mantle_net_watts / (cp_mantle * Mmantle)
    
    # Mass Balance
    dTr_dt[5] = -degassing_rate
    dTr_dt[6] = degassing_rate - H2O_loss_rate
    dTr_dt[7] = O_gen_rate - total_O_loss
    
    # Surface Temp
    dTr_dt[8] = net_surface_power / (heat_cap_atm + heat_cap_crust)

    return dTr_dt
