import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_melt_fractions import get_melt_fractions
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_flux import get_flux
from utils.get_loss import get_loss
from physics.degas2 import degas2
from physics.viscosity import viscosity
from physics.radiogenics import get_radiogenic_heat
from physics.shear_modulus import calc_shear_modulus
from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion


# ---------------------------------------------------------------------------
# Shared state-vector layout (same for both phase ODEs)
#   [0]  semi_major_axis   [1]  eccentricity
#   [2]  spin_freq_host    [3]  spin_freq_planet
#   [4]  temp_mantle       [5]  radius_solid
#   [6]  mass_water_solid  [7]  mass_water_atm
#   [8]  mass_oxygen_atm   [9]  mass_O2_solid
#   [10] temp_surface
# ---------------------------------------------------------------------------


def _unpack_header(Tr, Mmantle, Rp, Rc, Mh, Mp):
    """Unpack & guard-clip the state vector. Returns commonly needed scalars."""
    semi_a       = Tr[0]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)
    eccentricity = Tr[1]
    spin_freq_h  = Tr[2]
    spin_freq_p  = Tr[3]
    temp_mantle  = Tr[4]
    radius_solid = float(np.clip(Tr[5], Rc, Rp))

    mass_water_solid      = max(0.0, Tr[6])
    mass_frac_water_solid = mass_water_solid / Mmantle
    mass_water_atm        = max(0.0, Tr[7])
    mass_oxygen_atm       = max(0.0, Tr[8])
    temp_surface          = Tr[10]

    temp_mantle = max(temp_mantle, temp_surface + 1.0)

    return (semi_a, orbital_freq, eccentricity, spin_freq_h, spin_freq_p,
            temp_mantle, radius_solid,
            mass_water_solid, mass_frac_water_solid,
            mass_water_atm, mass_oxygen_atm, temp_surface)


def _b_coeff(Rp, radius_solid, rho_mantle, g, temp_mantle, thermo):
    """Solidification rate coefficient dRs/dT (m/K).

    Tanh blending across the low-P / high-P solidus branch crossover removes
    the discontinuity in B (and thus in dT_m_dt) when radius_solid passes
    through the crossover depth.
    """
    Cp    = thermo['specific_heat_mantle']
    alpha = thermo['thermal_expansion']

    D_cross = 405e9 / 77.89 / rho_mantle / g        # crossover depth (m)
    depth   = Rp - radius_solid
    w       = thermo.get('solidus_crossover_width_km', 30.0) * 1e3  # half-width (m)
    blend   = 0.5 * (1.0 + np.tanh((depth - D_cross) / w))          # 0=low-P, 1=high-P

    Tsol_a = ((1.0 - blend) * thermo['solidus_slope_low_p']
              + blend       * thermo['solidus_slope_high_p']) * 1e-9
    Tsol_b = ((1.0 - blend) * thermo['solidus_intercept_low_p']
              + blend       * thermo['solidus_intercept_high_p'])

    return (
        (Cp * (Tsol_b * alpha - Tsol_a * rho_mantle * Cp)) /
        (g * (Tsol_a * rho_mantle * Cp - alpha * temp_mantle)**2)
    )


def _calc_pressure_H2O_smooth(temp_surface, mass_water_atm, g, surface_area,
                               crit_temp_water, vapor_a, vapor_b):
    """
    Atmospheric water vapor pressure with a smooth transition near the critical
    temperature (647 K), replacing the hard if/else that caused a discontinuous
    spike in surface temperature.

    Above crit_temp_water: all water is supercritical vapor → p = p_full.
    Below crit_temp_water: vapor/liquid equilibrium → p = min(p_sat, p_full).
    Near crit_temp_water: smooth tanh blend between the two regimes.

    The blend width of 20 K is tight enough to be physically accurate but wide
    enough to eliminate the numerical discontinuity.
    """
    p_full = mass_water_atm * g / surface_area
    p_sat  = 10.0 ** (vapor_a - vapor_b / temp_surface) * 1e5
    p_below_crit = min(p_sat, p_full)

    blend = 0.5 * (1.0 + np.tanh((temp_surface - crit_temp_water) / 20.0))
    return blend * p_full + (1.0 - blend) * p_below_crit


def _surface_temp_ode(q_mantle, flux_to_space, Rp, pressure_H2O, g,
                      heat_capacity_water, heat_capacity_mantle, density_crust,
                      crust_depth_phys, latent_heat_vaporization,
                      mass_water_atm, crit_temp_water, vapor_a, vapor_b,
                      temp_surface):
    """Surface temperature ODE term (shared by both phase ODEs)."""
    surface_area = 4.0 * np.pi * Rp**2
    net_surface_power = surface_area * (q_mantle - flux_to_space)

    # Latent heat of condensation/vaporization slows the surface temperature
    # evolution near the dew point.  T_sat is found from the Antoine equation
    # inverted at the actual atmospheric column pressure, capped at crit_temp_water.
    latent_heat_capacity = 0.0
    if mass_water_atm > 0:
        P_actual = mass_water_atm * g / surface_area
        P_crit   = 10**(vapor_a - vapor_b / crit_temp_water) * 1e5
        T_sat    = (crit_temp_water if P_actual >= P_crit
                    else vapor_b / (vapor_a - np.log10(P_actual / 1e5)))
        activation = 0.5 * (1.0 + np.tanh((T_sat - temp_surface) / 20.0))
        if activation > 1e-4:
            P_sat     = 10**(vapor_a - vapor_b / temp_surface) * 1e5
            dP_sat_dT = P_sat * np.log(10) * vapor_b / (temp_surface**2)
            latent_heat_capacity = (latent_heat_vaporization
                                    * (surface_area / g) * dP_sat_dT * activation)

    heat_cap_atm   = heat_capacity_water * (pressure_H2O * surface_area / g)
    heat_cap_crust = (heat_capacity_mantle * density_crust
                      * (4.0 / 3.0) * np.pi * (Rp**3 - (Rp - crust_depth_phys)**3))
    return net_surface_power / (heat_cap_atm + heat_cap_crust + latent_heat_capacity)


# ===========================================================================
# MAGMA OCEAN ODE
# Active when bulk melt fraction >= melt_fraction_threshold.
# ===========================================================================
def moODE_magma_ocean(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, Ts, Ps, OLR, ASR,
        t_flux, Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, Mp, LStar, Rh, Mh,
        tides_on_flag, params):
    """
    ODE for the magma ocean phase.

    Handles dissolved-volatile partitioning, MO heat flux driven from the
    solidification front, and latent-heat thermal inertia. Terminated by
    event_mo_ends when meltfrac_bulk drops below the threshold.
    """
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
    crit_temp_water = atm['critical_temp_H2O']
    vapor_a         = atm['vapor_press_a']
    vapor_b         = atm['vapor_press_b']

    surface_area = 4.0 * np.pi * Rp**2

    (semi_a, orbital_freq, eccentricity, spin_freq_h, spin_freq_p,
     temp_mantle, radius_solid,
     mass_water_solid, mass_frac_water_solid,
     mass_water_atm, mass_oxygen_atm, temp_surface) = _unpack_header(
        Tr, Mmantle, Rp, Rc, Mh, Mp)

    # --- Melt fractions ---
    _, meltfrac_bulk, meltfrac_mo = get_melt_fractions(temp_mantle, params['_grid'])
    if temp_mantle <= thermo['solidus_intercept_low_p']:
        meltfrac_bulk = 0.0
        meltfrac_mo   = 0.0

    mass_magma_ocean = max(0.0, (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - radius_solid**3))

    # --- Solidification rate coefficient ---
    B = _b_coeff(Rp, radius_solid, rho_mantle, g, temp_mantle, thermo)

    # --- Rheology ---
    solid_shear_modulus = calc_shear_modulus(meltfrac_bulk, params)
    nu_solid_bulk = viscosity(temp_mantle, temp_surface, rho_mantle,
                              Tr[6] / Mmantle, meltfrac_bulk, params)

    # --- Water & Oxygen pressures (dissolved equilibrium) ---
    if mass_water_atm > 0.0 and mass_magma_ocean > 0.0:
        pressure_H2O, mass_frac_water_melt, partition_coeff_H2O = get_pressure2(
            temp_mantle, radius_solid, mass_magma_ocean, Mmantle,
            Rp, g, Rc, mass_water_atm, params
        )
    else:
        pressure_H2O, mass_frac_water_melt, partition_coeff_H2O = 0.0, 0.0, 0.0

    if mass_oxygen_atm > 0.0 and mass_magma_ocean > 0.0:
        pressure_O2, mass_frac_FeO1_5, _, _ = get_massbalance4(
            temp_mantle, pressure_H2O, mass_magma_ocean,
            mass_oxygen_atm, Xi, FeOt, g, Rp, params
        )
        if pressure_O2 < 0.0:
            pressure_O2      = mass_oxygen_atm * g / surface_area
            mass_frac_FeO1_5 = 0.0
    else:
        pressure_O2, mass_frac_FeO1_5 = 0.0, 0.0

    # --- Heat flux (MO convection from solidification front) ---
    q_mantle, Db, uc, Ra, nu_ = mantleheatflux(
        temp_mantle, temp_surface, radius_solid, Rp, Rc, g, rho_mantle,
        mass_frac_water_melt, meltfrac_mo, params
    )

    # --- Energy budget ---
    flux_to_space            = get_flux(temp_surface, Teq, pressure_H2O, pressure_O2, Rp, g, params)
    flux_loss_H, flux_loss_O = get_loss(t_flux, Lbol, t_sec, pressure_O2, pressure_H2O,
                                        tsat, semi_a, Mp, Rp, LStar, temp_surface, params)
    radiogenic_heating_watts = get_radiogenic_heat(t_sec, Mmantle, params)

    # --- Tidal dissipation (to solidification front) ---
    visc_solid = nu_solid_bulk * rho_mantle
    da_dt, de_dt, dspin_dt_h, dspin_dt_p, _, tidal_heating_p, _, _, _ = (
        calculate_tidal_dissipation(
            eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
            Rc, visc_solid, solid_shear_modulus, radius_solid, tides_on_flag, params
        )
    )

    # --- Thermal evolution ---
    mantle_cooling_watts = surface_area * q_mantle

    # Fix B: tidal heating dissipates in the solid layer only — it is NOT added to the
    # liquid layer's thermal budget.  tidal_heating_p is still used above for da/dt, de/dt.
    #
    # Fix C: credit only the liquid layer's proportional share of radiogenic heating.
    # The solid layer beneath radius_solid retains its own share; as the MO shrinks,
    # the liquid layer heats only from radioactive elements in the melt.
    frac_molten = mass_magma_ocean / Mmantle
    total_mantle_heating_watts = radiogenic_heating_watts * frac_molten

    # Latent heat reduces effective thermal inertia during solidification.
    # Only the convecting magma ocean mass participates — not the already-solid interior.
    thermal_inertia = ((heat_capacity_mantle * mass_magma_ocean)
                       - (4.0 * np.pi * rho_mantle * latent_heat_fusion * radius_solid**2) * B)

    dT_m_dt = (-mantle_cooling_watts + total_mantle_heating_watts) / thermal_inertia

    rate_rs = B * dT_m_dt
    if radius_solid >= Rp and rate_rs > 0.0:
        rate_rs = 0.0
    elif radius_solid <= Rc and rate_rs < 0.0:
        rate_rs = 0.0

    mass_solidified      = 4.0 * np.pi * rho_mantle * radius_solid**2 * rate_rs
    safe_mass_solidified = max(mass_solidified, 0.0)

    # --- Volatile rates ---
    total_mass_loss_H           = surface_area * flux_loss_H
    total_mass_loss_O           = surface_area * flux_loss_O
    water_loss_to_space         = total_mass_loss_H * (molar_mass_H2O / (2.0 * molar_mass_H))
    oxygen_generated_from_water = total_mass_loss_H * (molar_mass_O  / (2.0 * molar_mass_H))

    water_partitioned  = (partition_coeff_H2O * mass_frac_water_melt * safe_mass_solidified
                          if mass_frac_water_melt > 0 else 0.0)
    oxygen_partitioned = (mass_frac_FeO1_5 * 0.5
                          * (molar_mass_O / molar_mass_FeO1_5) * safe_mass_solidified)

    # --- Assemble ODE ---
    dTr_dt = np.zeros(11, dtype=np.float64)
    dTr_dt[0] = da_dt
    dTr_dt[1] = de_dt
    dTr_dt[2] = dspin_dt_h
    dTr_dt[3] = dspin_dt_p
    dTr_dt[4] = dT_m_dt
    dTr_dt[5] = rate_rs
    # solid water: gains
    dTr_dt[6] = water_partitioned
    # atm water: drains
    dTr_dt[7] = (-water_partitioned - water_loss_to_space
                 if mass_water_atm > 0.0 else 0.0)
    dTr_dt[8] = oxygen_generated_from_water - total_mass_loss_O - oxygen_partitioned
    # solid O2 sink
    dTr_dt[9] = oxygen_partitioned 
    dTr_dt[10] = _surface_temp_ode(
        q_mantle, flux_to_space, Rp, pressure_H2O, g,
        heat_capacity_water, heat_capacity_mantle, density_crust,
        Rp - radius_solid,        # MO column acts as "crust" for surface heat cap
        latent_heat_vaporization, mass_water_atm,
        crit_temp_water, vapor_a, vapor_b, temp_surface
    )
    return dTr_dt


# ===========================================================================
# SOLID MANTLE ODE
# Active when bulk melt fraction < melt_fraction_threshold.
# ===========================================================================
def moODE_solid(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, Ts, Ps, OLR, ASR,
        t_flux, Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, Mp, LStar, Rh, Mh,
        tides_on_flag, params):
    """
    ODE for the solid mantle phase.

    Handles vapour-pressure water equilibrium, solid-state degassing, and
    whole-mantle convection. radius_solid still evolves so remelting episodes
    feed back into meltfrac_bulk correctly. Terminated by event_mo_starts
    when meltfrac_bulk rises above the threshold.
    """
    constants = params['constants']
    mat       = params['planet']['material']
    atm       = params['planet']['atmosphere']
    thermo    = params['planet']['thermodynamics']

    molar_mass_O   = constants['molar_mass_O']
    molar_mass_H2O = constants['molar_mass_H2O']
    molar_mass_H   = constants['molar_mass_H']

    heat_capacity_mantle     = thermo['specific_heat_mantle']
    heat_capacity_water      = thermo['specific_heat_H2O']
    density_crust            = mat['density_mantle']
    latent_heat_vaporization = thermo['latent_heat_vaporization']
    crit_temp_water = atm['critical_temp_H2O']
    vapor_a         = atm['vapor_press_a']
    vapor_b         = atm['vapor_press_b']

    surface_area = 4.0 * np.pi * Rp**2

    (semi_a, orbital_freq, eccentricity, spin_freq_h, spin_freq_p,
     temp_mantle, radius_solid,
     mass_water_solid, mass_frac_water_solid,
     mass_water_atm, mass_oxygen_atm, temp_surface) = _unpack_header(
        Tr, Mmantle, Rp, Rc, Mh, Mp)

    # --- Bulk melt fraction ---
    _, meltfrac_bulk, _ = get_melt_fractions(temp_mantle, params['_grid'])
    if temp_mantle <= thermo['solidus_intercept_low_p']:
        meltfrac_bulk = 0.0

    # --- Solidification rate coefficient (needed for remelting feedback) ---
    B = _b_coeff(Rp, radius_solid, rho_mantle, g, temp_mantle, thermo)

    # --- Rheology ---
    solid_shear_modulus = calc_shear_modulus(meltfrac_bulk, params)
    nu_solid_bulk = viscosity(temp_mantle, temp_surface, rho_mantle,
                              Tr[6] / Mmantle, meltfrac_bulk, params)

    # --- Water & Oxygen pressures (vapour equilibrium) ---
    pressure_H2O = _calc_pressure_H2O_smooth(
        temp_surface, mass_water_atm, g, surface_area,
        crit_temp_water, vapor_a, vapor_b)

    pressure_O2 = mass_oxygen_atm * g / surface_area

    # --- Heat flux (convection over the full solid mantle) ---
    q_mantle, Db, uc, Ra, nu_ = mantleheatflux(
        temp_mantle, temp_surface, Rc, Rp, Rc, g, rho_mantle,
        mass_frac_water_solid, meltfrac_bulk, params
    )

    # --- Energy budget ---
    flux_to_space            = get_flux(temp_surface, Teq, pressure_H2O, pressure_O2, Rp, g, params)
    flux_loss_H, flux_loss_O = get_loss(t_flux, Lbol, t_sec, pressure_O2, pressure_H2O,
                                        tsat, semi_a, Mp, Rp, LStar, temp_surface, params)
    radiogenic_heating_watts = get_radiogenic_heat(t_sec, Mmantle, params)

    # --- Tidal dissipation (full planet radius) ---
    visc_solid = nu_solid_bulk * rho_mantle
    da_dt, de_dt, dspin_dt_h, dspin_dt_p, _, tidal_heating_p, _, _, _ = (
        calculate_tidal_dissipation(
            eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
            Rc, visc_solid, solid_shear_modulus, Rp, tides_on_flag, params
        )
    )

    # --- Thermal evolution (no latent heat) ---
    mantle_cooling_watts       = surface_area * q_mantle
    total_mantle_heating_watts = radiogenic_heating_watts + tidal_heating_p

    dT_m_dt = (-mantle_cooling_watts + total_mantle_heating_watts) / (heat_capacity_mantle * Mmantle)

    # radius_solid evolves to capture remelting episodes that trigger event_mo_starts
    rate_rs = B * dT_m_dt
    if radius_solid >= Rp and rate_rs > 0.0:
        rate_rs = 0.0
    elif radius_solid <= Rc and rate_rs < 0.0:
        rate_rs = 0.0

    # --- Volatile rates ---
    total_mass_loss_H           = surface_area * flux_loss_H
    total_mass_loss_O           = surface_area * flux_loss_O
    water_loss_to_space         = total_mass_loss_H * (molar_mass_H2O / (2.0 * molar_mass_H))
    oxygen_generated_from_water = total_mass_loss_H * (molar_mass_O  / (2.0 * molar_mass_H))

    if mass_frac_water_solid > 1e-9:
        _, degas_rate, _, _ = degas2(temp_mantle, Db, q_mantle, mass_frac_water_solid,
                                     Rp, g, temp_surface, params)
        total_degassing = degas_rate * surface_area * uc
    else:
        total_degassing = 0.0

    # --- Assemble ODE ---
    dTr_dt = np.zeros(11, dtype=np.float64)
    dTr_dt[0] = da_dt
    dTr_dt[1] = de_dt
    dTr_dt[2] = dspin_dt_h
    dTr_dt[3] = dspin_dt_p
    dTr_dt[4] = dT_m_dt
    dTr_dt[5] = rate_rs
    # solid water: drains
    dTr_dt[6] = -total_degassing
    # atm water
    dTr_dt[7] = total_degassing - water_loss_to_space
    # atm O2
    dTr_dt[8] = oxygen_generated_from_water - total_mass_loss_O
    # solid O2: no active sink or source
    dTr_dt[9] = 0.0 
    dTr_dt[10] = _surface_temp_ode(
        q_mantle, flux_to_space, Rp, pressure_H2O, g,
        heat_capacity_water, heat_capacity_mantle, density_crust,
        Db,          # boundary layer depth as "crust" for surface heat cap
        latent_heat_vaporization, mass_water_atm,
        crit_temp_water, vapor_a, vapor_b, temp_surface
    )
    return dTr_dt


# ===========================================================================
# DRY SOLID MANTLE ODE
# Active when melt fraction < threshold AND solid water mass fraction < dry_solid_threshold.
# ===========================================================================
def moODE_dry_solid(
        t_sec, Tr, Rp, Rc, Mmantle, Teq, rho_mantle, g, Ts, Ps, OLR, ASR,
        t_flux, Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, Mp, LStar, Rh, Mh,
        tides_on_flag, params):
    """
    ODE for the dry solid mantle phase.

    Identical to moODE_solid but without solid-state degassing — the volatile
    source in the solid is exhausted. radius_solid still evolves so tidal
    remelting episodes can trigger event_mo_starts.

    Phase transition rules (enforced by run_model loop, not events):
      dry_solid → MO      : allowed (tidal/radiogenic remelting)
      dry_solid → wet_solid: NEVER — once the solid is desiccated there is no
                             mechanism to re-wet it. When MO refreezes after a
                             dry_solid episode the loop checks solid water mass
                             and dispatches back to dry_solid, not wet_solid.
    """
    constants = params['constants']
    mat       = params['planet']['material']
    atm       = params['planet']['atmosphere']
    thermo    = params['planet']['thermodynamics']

    molar_mass_O   = constants['molar_mass_O']
    molar_mass_H2O = constants['molar_mass_H2O']
    molar_mass_H   = constants['molar_mass_H']

    heat_capacity_mantle     = thermo['specific_heat_mantle']
    heat_capacity_water      = thermo['specific_heat_H2O']
    density_crust            = mat['density_mantle']
    latent_heat_vaporization = thermo['latent_heat_vaporization']
    crit_temp_water = atm['critical_temp_H2O']
    vapor_a         = atm['vapor_press_a']
    vapor_b         = atm['vapor_press_b']

    surface_area = 4.0 * np.pi * Rp**2

    (semi_a, orbital_freq, eccentricity, spin_freq_h, spin_freq_p,
     temp_mantle, radius_solid,
     _, mass_frac_water_solid,
     mass_water_atm, mass_oxygen_atm, temp_surface) = _unpack_header(
        Tr, Mmantle, Rp, Rc, Mh, Mp)

    # --- Bulk melt fraction ---
    _, meltfrac_bulk, _ = get_melt_fractions(temp_mantle, params['_grid'])
    if temp_mantle <= thermo['solidus_intercept_low_p']:
        meltfrac_bulk = 0.0

    # --- Solidification rate coefficient (needed for remelting feedback) ---
    B = _b_coeff(Rp, radius_solid, rho_mantle, g, temp_mantle, thermo)

    # --- Rheology ---
    solid_shear_modulus = calc_shear_modulus(meltfrac_bulk, params)
    nu_solid_bulk = viscosity(temp_mantle, temp_surface, rho_mantle,
                              mass_frac_water_solid, meltfrac_bulk, params)

    # --- Water & Oxygen pressures (vapour equilibrium, no dissolved phase) ---
    pressure_H2O = _calc_pressure_H2O_smooth(
        temp_surface, mass_water_atm, g, surface_area,
        crit_temp_water, vapor_a, vapor_b)

    pressure_O2 = mass_oxygen_atm * g / surface_area

    # --- Heat flux (convection over the full solid mantle) ---
    q_mantle, Db, _, _, _ = mantleheatflux(
        temp_mantle, temp_surface, Rc, Rp, Rc, g, rho_mantle,
        mass_frac_water_solid, meltfrac_bulk, params
    )

    # --- Energy budget ---
    flux_to_space            = get_flux(temp_surface, Teq, pressure_H2O, pressure_O2, Rp, g, params)
    flux_loss_H, flux_loss_O = get_loss(t_flux, Lbol, t_sec, pressure_O2, pressure_H2O,
                                        tsat, semi_a, Mp, Rp, LStar, temp_surface, params)
    radiogenic_heating_watts = get_radiogenic_heat(t_sec, Mmantle, params)

    # --- Tidal dissipation (full planet radius) ---
    visc_solid = nu_solid_bulk * rho_mantle
    da_dt, de_dt, dspin_dt_h, dspin_dt_p, _, tidal_heating_p, _, _, _ = (
        calculate_tidal_dissipation(
            eccentricity, orbital_freq, spin_freq_p, spin_freq_h, Rp, Rh, Mp, Mh,
            Rc, visc_solid, solid_shear_modulus, Rp, tides_on_flag, params
        )
    )

    # --- Thermal evolution (no latent heat) ---
    mantle_cooling_watts       = surface_area * q_mantle
    total_mantle_heating_watts = radiogenic_heating_watts + tidal_heating_p

    dT_m_dt = (-mantle_cooling_watts + total_mantle_heating_watts) / (heat_capacity_mantle * Mmantle)

    # radius_solid still evolves to capture remelting episodes
    rate_rs = B * dT_m_dt
    if radius_solid >= Rp and rate_rs > 0.0:
        rate_rs = 0.0
    elif radius_solid <= Rc and rate_rs < 0.0:
        rate_rs = 0.0

    # --- Volatile rates (no degassing) ---
    total_mass_loss_H           = surface_area * flux_loss_H
    total_mass_loss_O           = surface_area * flux_loss_O
    water_loss_to_space         = total_mass_loss_H * (molar_mass_H2O / (2.0 * molar_mass_H))
    oxygen_generated_from_water = total_mass_loss_H * (molar_mass_O  / (2.0 * molar_mass_H))

    # --- Assemble ODE ---
    dTr_dt = np.zeros(11, dtype=np.float64)
    dTr_dt[0] = da_dt
    dTr_dt[1] = de_dt
    dTr_dt[2] = dspin_dt_h
    dTr_dt[3] = dspin_dt_p
    dTr_dt[4] = dT_m_dt
    dTr_dt[5] = rate_rs
    # solid water: source exhausted
    dTr_dt[6] = 0.0
    # atm water: escape only
    dTr_dt[7] = -water_loss_to_space
    # atm O2
    dTr_dt[8] = oxygen_generated_from_water - total_mass_loss_O
    # solid O2: no active sink or source
    dTr_dt[9] = 0.0
    dTr_dt[10] = _surface_temp_ode(
        q_mantle, flux_to_space, Rp, pressure_H2O, g,
        heat_capacity_water, heat_capacity_mantle, density_crust,
        Db,
        latent_heat_vaporization, mass_water_atm,
        crit_temp_water, vapor_a, vapor_b, temp_surface
    )
    return dTr_dt


# ===========================================================================
# PHASE TRANSITION EVENTS
# ===========================================================================
def make_phase_events(params, Mmantle):
    """
    Return terminal event functions for all three phase transitions.

    event_mo_ends     — MO → wet solid; meltfrac_bulk decreasing through threshold.
    event_mo_starts   — any solid → MO; meltfrac_bulk increasing through threshold.
    event_solid_dries — wet solid → dry solid; solid water mass fraction
                        decreasing through dry_solid_threshold.

    All functions receive *dimensional* (t_sec, y_dim) because
    StateScaler.wrap_event unscales before calling.
    """
    mf_threshold      = params['planet']['convection']['melt_fraction_threshold']
    mf_width          = params['planet']['convection']['phase_transition_width']
    solidus_low_p_int = params['planet']['thermodynamics']['solidus_intercept_low_p']
    dry_threshold     = params['planet']['thermodynamics'].get('dry_solid_threshold', 1e-9)

    # Asymmetric thresholds prevent chattering at the phase boundary.
    # The MO→solid and solid→MO crossings are separated by 2*mf_width so
    # neither event can fire at the exact state left by the other.
    mo_end_threshold   = mf_threshold - mf_width   # MO ends below this
    mo_start_threshold = mf_threshold + mf_width   # MO restarts above this

    def event_mo_ends(t_sec, y_dim):
        temp_mantle = max(y_dim[4], y_dim[10] + 1.0)
        _, meltfrac_bulk, _ = get_melt_fractions(temp_mantle, params['_grid'])
        if temp_mantle <= solidus_low_p_int:
            meltfrac_bulk = 0.0
        return meltfrac_bulk - mo_end_threshold

    event_mo_ends.terminal  = True
    event_mo_ends.direction = -1   # fires only when meltfrac is decreasing

    def event_mo_starts(t_sec, y_dim):
        temp_mantle = max(y_dim[4], y_dim[10] + 1.0)
        _, meltfrac_bulk, _ = get_melt_fractions(temp_mantle, params['_grid'])
        if temp_mantle <= solidus_low_p_int:
            meltfrac_bulk = 0.0
        return meltfrac_bulk - mo_start_threshold

    event_mo_starts.terminal  = True
    event_mo_starts.direction = +1  # fires only when meltfrac is increasing

    def event_solid_dries(_, y_dim):
        mass_water_solid = max(0.0, y_dim[6])
        return mass_water_solid / Mmantle - dry_threshold

    event_solid_dries.terminal  = True
    event_solid_dries.direction = -1  # fires only when solid water is decreasing

    return event_mo_ends, event_mo_starts, event_solid_dries
