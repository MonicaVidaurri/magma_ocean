import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_meltfrac import get_meltfrac
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_flux import get_flux
from utils.get_loss import get_loss
from physics.radiogenics import get_radiogenic_heat
from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def moODE_phase1(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, 
        t_flux, Lbol, Xi, FeOt, tsat, Mp, LStar, Rh, Mh, 
        params, tides_on_flag
    ):
    """ 
    First time phase (Magma Ocean). 
    All physics modules are driven by the master 'params' dictionary.
    """
    
    # --- Unpack Constants from TOML ---
    univ = params['constants']
    mat  = params['planet']['material']
    thermo = params['planet']['thermodynamics']
    comp = params['planet']['oxide_composition']

    mu_O      = univ['molar_mass_O']
    mu_H2O    = univ['molar_mass_h2o']
    mu_H      = univ['molar_mass_H']
    mu_FeO1_5 = comp['molar_mass_FeO1_5']
    mu_FeO    = comp['molar_mass_FeO']
    
    cp_mantle = mat['specific_heat_mantle']
    cp_water  = mat['specific_heat_h2o']
    dHf       = mat['latent_heat_fusion']
    alpha     = thermo['thermal_expansion']
    
    # Grid/Boundary Constants for B_coeff calculation
    # (Matches the logic used in get_meltfrac)
    Tsol_slope_low  = thermo['solidus_slope_low_p']
    Tsol_int_low    = thermo['solidus_intercept_low_p']
    Tsol_slope_high = thermo['solidus_slope_high_p']
    Tsol_int_high   = thermo['solidus_intercept_high_p']

    surface_area = 4.0 * np.pi * Rp**2

    # --- Unpack State Vector ---
    dTr_dt = np.zeros(11, dtype=np.float64)
    
    semi_a       = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h  = Tr[2]
    spin_freq_p  = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    temp_mantle  = Tr[4]
    radius_solid = Tr[5]
    
    # Magma Ocean Geometry
    mass_magma_ocean = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - radius_solid**3)
    if radius_solid > Rp:
        radius_solid = Rp
        mass_magma_ocean = 0.0

    mass_water_mo_atm = Tr[7]
    mass_frac_water_melt = mass_water_mo_atm / mass_magma_ocean if mass_magma_ocean > 0 else 0.0
    mass_oxygen_mo_atm = Tr[8]
    temp_surface       = Tr[10]

    # --- Mantle Melt Fraction & Piecewise B-Coeff ---
    # Determine which solidus regime we are in based on depth
    # Boundary derived from the intersection of the two linear solidus lines
    p_boundary_gpa = (Tsol_int_high - Tsol_int_low) / (Tsol_slope_low - Tsol_slope_high)
    depth_limit = (p_boundary_gpa * 1e9) / (rho_mantle * g)

    if (Rp - radius_solid) > depth_limit:
        Tsol_a, Tsol_b = Tsol_slope_high * 1e-9, Tsol_int_high
    else:
        Tsol_a, Tsol_b = Tsol_slope_low * 1e-9, Tsol_int_low

    # B_coeff handles the relationship between radius change and temperature change
    denom = (g * (Tsol_a * rho_mantle * cp_mantle - alpha * temp_mantle)**2)
    B_coeff = ((cp_mantle * (Tsol_b * alpha - Tsol_a * rho_mantle * cp_mantle)) / denom)

    _, meltfrac = get_meltfrac(g, temp_mantle, Rp, Rc, Mmantle, params)

    # --- Partitioning & Pressures ---
    if mass_water_mo_atm > 0.0:
        pressure_atm, mass_frac_water_melt, kH2O = get_pressure2(
            temp_mantle, radius_solid, mass_magma_ocean, Mmantle, Rp, g, Rc, mass_water_mo_atm, params
        )
    else:
        pressure_atm, kH2O = 0.0, 0.0

    if mass_oxygen_mo_atm > 0.0:
        pressure_O2, f_FeO1_5, _, _ = get_massbalance4(
            temp_mantle, pressure_atm, mass_magma_ocean, mass_oxygen_mo_atm, Xi, FeOt, g, Rp, params
        )
        pressure_O2 = max(pressure_O2, mass_oxygen_mo_atm * g / surface_area) if pressure_O2 <= 0 else pressure_O2
    else:
        pressure_O2, f_FeO1_5 = 0.0, 0.0

    # --- Energy Fluxes & Dissipation ---
    flux_space = get_flux(temp_surface, Teq, pressure_atm, Rp, g, params)
    
    f_loss_H, f_loss_O = get_loss(
        t_flux, Lbol, t_sec, pressure_O2, pressure_atm, tsat, semi_a, Mp, Rp, LStar, params
    )
    
    Q_radio = get_radiogenic_heat(t_sec, Mmantle, params)

    q_mantle, Db, _, _, nu = mantleheatflux(
        temp_mantle, temp_surface, radius_solid, Rp, Rc, g, rho_mantle, mass_frac_water_melt, meltfrac, params
    )

    da, de, dsh, dsp, _, Q_tidal = calculate_tidal_dissipation(
        eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
        temp_mantle, Rc, nu, rho_mantle, meltfrac, params, tides_on_flag
    )

    # --- Rate Calculations ---
    mantle_net_watts = Q_radio + Q_tidal - (surface_area * q_mantle)
    
    if radius_solid < Rp:
        thermal_inertia = (cp_mantle * Mmantle) - (4.0 * np.pi * rho_mantle * dHf * radius_solid**2) * B_coeff
    else:
        thermal_inertia = cp_mantle * Mmantle

    # ODE Derivatives for Heat and Solidification
    dTr_dt[4] = mantle_net_watts / thermal_inertia
    dTr_dt[5] = B_coeff * dTr_dt[4] if radius_solid < Rp else 0.0
    
    mass_sol_rate = 4.0 * np.pi * rho_mantle * radius_solid**2 * dTr_dt[5]

    # Mass Transport (Water/Oxygen)
    total_H_loss = surface_area * f_loss_H
    total_O_loss = surface_area * f_loss_O
    
    # Stoichiometric relations
    H2O_loss_rate = total_H_loss * (mu_H2O / (2.0 * mu_H))
    O_gen_rate    = total_H_loss * (mu_O / (2.0 * mu_H))

    water_to_solid  = kH2O * mass_frac_water_melt * mass_sol_rate if mass_frac_water_melt > 0 else 0.0
    oxygen_to_solid = f_FeO1_5 * 0.5 * (mu_O / mu_FeO1_5) * mass_sol_rate

    # --- Final State Vector Assembly ---
    dTr_dt[0], dTr_dt[1], dTr_dt[2], dTr_dt[3] = da, de, dsh, dsp
    dTr_dt[6] = water_to_solid
    dTr_dt[7] = -water_to_solid - H2O_loss_rate if mass_water_mo_atm > 0 else 0.0
    dTr_dt[9] = oxygen_to_solid
    
    # Oxygen atmosphere logic
    if mass_water_mo_atm > 0:
        dTr_dt[8] = O_gen_rate - total_O_loss - oxygen_to_solid
    else:
        dTr_dt[8] = -oxygen_to_solid if pressure_O2 > 0 else 0.0

    # Surface Temperature (Thermal balance between mantle flux and space loss)
    heat_cap_atm   = cp_water * (pressure_atm * surface_area / g)
    heat_cap_crust = cp_mantle * 3000.0 * ((4.0 / 3.0) * np.pi * (Rp**3 - (Rp - Db)**3))
    
    dTr_dt[10] = (surface_area * (q_mantle - flux_space)) / (heat_cap_atm + heat_cap_crust)

    return dTr_dt
