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
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, Ts, Ps, OLR, ASR, 
        t_flux, Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, Mp, LStar, Rh, Mh, 
        tides_on_flag
    ):
    """ First time phase; There is a magma ocean. Mantle is partially/fully molten. """
    
    # =====================================================================
    # --- Constants & Molar Masses ---
    # =====================================================================
    molar_mass_O      = 15.9994e-3  # kg/mole
    molar_mass_H2O    = 18.015e-3   # kg/mole
    molar_mass_H      = 1.008e-3    # kg/mole
    molar_mass_FeO1_5 = 159.689e-3 / 2.0 
    
    heat_capacity_mantle = 1.2e3    # cp (J/kg/K)
    heat_capacity_water  = 2e3      # cpH2O (J/kg/K)
    density_crust        = 3000.0   # kg/m^3
    latent_heat_fusion   = 4e5      # dHf (J/kg)
    thermal_expansion    = 2e-5     # alpha (1/K)

    surface_area = 4.0 * np.pi * Rp**2

    # =====================================================================
    # --- Unpack State Vector ---
    # =====================================================================
    dTr_dt = np.zeros(11, dtype=np.float64)
    
    semi_a       = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h  = Tr[2]
    spin_freq_p  = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    temp_mantle  = Tr[4]
    radius_solid = Tr[5]
    
    mass_water_solid      = Tr[6]
    mass_frac_water_solid = mass_water_solid / Mmantle

    # Magma Ocean Properties
    mass_magma_ocean = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - radius_solid**3)
    if radius_solid > Rp:
        radius_solid = Rp
        mass_magma_ocean = 0.0

    mass_water_mo_atm = Tr[7]
    if mass_water_mo_atm > 0.0:
        mass_frac_water_melt = mass_water_mo_atm / mass_magma_ocean
    else:
        mass_water_mo_atm = 0.0
        mass_frac_water_melt = 0.0

    mass_oxygen_mo_atm = Tr[8]
    temp_surface       = Tr[10]

    # =====================================================================
    # --- Mantle Melt Fraction & Properties ---
    # =====================================================================
    if (Rp - radius_solid) > 405e9 / 77.89 / rho_mantle / g:
        Tsol_a = 26.53e-9
        Tsol_b = 1825.0
    else:
        Tsol_a = 104.42e-9
        Tsol_b = 1420.0

    B_coeff = (
        (heat_capacity_mantle * (Tsol_b * thermal_expansion - Tsol_a * rho_mantle * heat_capacity_mantle)) / 
        (g * (Tsol_a * rho_mantle * heat_capacity_mantle - thermal_expansion * temp_mantle)**2)
    )

    _, meltfrac = get_meltfrac(g, temp_mantle, Rp, Rc, Mmantle)

    # =====================================================================
    # --- Atmospheric Pressures ---
    # =====================================================================
    if mass_water_mo_atm > 0.0:
        pressure_atm, mass_frac_water_melt, partition_coeff_H2O = get_pressure2(
            temp_mantle, radius_solid, mass_magma_ocean, Mmantle, Rp, g, Rc, mass_water_mo_atm
        )
    else:
        pressure_atm = 0.0
        partition_coeff_H2O = 0.0

    if mass_oxygen_mo_atm > 0.0:
        pressure_O2, mass_frac_FeO1_5, _, _ = get_massbalance4(
            temp_mantle, pressure_atm, mass_magma_ocean, mass_oxygen_mo_atm, Xi, FeOt, g, Rp
        )
        if pressure_O2 < 0.0:
            pressure_O2 = mass_oxygen_mo_atm * g / surface_area
            mass_frac_FeO1_5 = 0.0
    else:
        pressure_O2 = 0.0
        mass_frac_FeO1_5 = 0.0

    # =====================================================================
    # --- Energy Fluxes & Loss Rates ---
    # =====================================================================
    flux_to_space = get_flux(temp_surface, Teq, pressure_atm, Rp, g)
    
    flux_loss_H, flux_loss_O = get_loss(
        t_flux, Lbol, t_sec, pressure_O2, pressure_atm, tsat, semi_a, Mp, Rp, LStar
    )
    
    radiogenic_heating_watts = get_radiogenic_heat(t_sec, Mmantle)

    q_mantle, Db, uc, Ra, nu = mantleheatflux(
        temp_mantle, temp_surface, radius_solid, Rp, Rc, g, rho_mantle, mass_frac_water_melt, meltfrac
    )

    da_dt, de_dt, dspin_dt_h, dspin_dt_p, _, tidal_heating_p = calculate_tidal_dissipation(
        eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
        temp_mantle, Rc, nu, rho_mantle, meltfrac, tides_on_flag
    )

    # =====================================================================
    # --- Intermediate Rate Calculations ---
    # =====================================================================
    
    # Mantle Heat & Thermal Inertia
    mantle_cooling_watts       = surface_area * q_mantle
    total_mantle_heating_watts = radiogenic_heating_watts + tidal_heating_p
    
    if radius_solid < Rp:
        thermal_inertia_mantle = (heat_capacity_mantle * Mmantle) - \
            (4.0 * np.pi * rho_mantle * latent_heat_fusion * radius_solid**2) * B_coeff
    else:
        thermal_inertia_mantle = heat_capacity_mantle * Mmantle

    # Crystallization 
    rate_radius_solidified_m_s = B_coeff * (
        (-mantle_cooling_watts + total_mantle_heating_watts) / thermal_inertia_mantle
    ) if radius_solid < Rp else 0.0
    
    mass_solidified_kg_s = 4.0 * np.pi * rho_mantle * radius_solid**2 * rate_radius_solidified_m_s

    # Mass Loss & Partitioning
    total_mass_loss_H_kg_s = surface_area * flux_loss_H
    total_mass_loss_O_kg_s = surface_area * flux_loss_O
    
    water_loss_to_space_kg_s         = total_mass_loss_H_kg_s * (molar_mass_H2O / (2.0 * molar_mass_H))
    oxygen_generated_from_water_kg_s = total_mass_loss_H_kg_s * (molar_mass_O / (2.0 * molar_mass_H))

    water_partitioned_to_solid_kg_s = partition_coeff_H2O * mass_frac_water_melt * mass_solidified_kg_s if mass_frac_water_melt > 0 else 0.0
    oxygen_partitioned_to_solid_kg_s = mass_frac_FeO1_5 * 0.5 * (molar_mass_O / molar_mass_FeO1_5) * mass_solidified_kg_s

    # Surface Thermal Properties
    net_surface_power = surface_area * (q_mantle - flux_to_space)
    
    heat_cap_atm   = heat_capacity_water * (pressure_atm * surface_area / g)
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
    dTr_dt[5] = rate_radius_solidified_m_s

    # Solid Mantle Water
    dTr_dt[6] = water_partitioned_to_solid_kg_s

    # Magma/Atmosphere Water
    dTr_dt[7] = -water_partitioned_to_solid_kg_s - water_loss_to_space_kg_s if mass_water_mo_atm > 0 else 0.0

    # Solid Mantle Oxygen
    dTr_dt[9] = oxygen_partitioned_to_solid_kg_s

    # Magma/Atmosphere Oxygen
    if mass_water_mo_atm > 0 and pressure_O2 > 0:
        dTr_dt[8] = oxygen_generated_from_water_kg_s - total_mass_loss_O_kg_s - oxygen_partitioned_to_solid_kg_s
    elif mass_water_mo_atm > 0 and pressure_O2 <= 0:
        dTr_dt[8] = oxygen_generated_from_water_kg_s - oxygen_partitioned_to_solid_kg_s
    elif mass_water_mo_atm <= 0 and pressure_O2 > 0:
        dTr_dt[8] = -oxygen_partitioned_to_solid_kg_s
    else:
        dTr_dt[8] = 0.0

    dTr_dt[10] = net_surface_power / total_surface_heat_capacity

    return dTr_dt
