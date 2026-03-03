import numpy as np

from utils.get_melt_fractions import get_melt_fractions
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4

from physics.mantleheatflux import mantleheatflux
from physics.radiogenics import get_radiogenic_heat
from physics.tides import calculate_tidal_dissipation
from physics.shear_modulus import calc_shear_modulus
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion


def postprocess_magma_ocean(
        sol_unified, Rp, Rc, Mmantle, gp, OLR_interp, ASR, Teq, Xi, FeOt,
        t_flux, Lbol, tsat, a, Mp, LStar, params):

    c      = params['constants']
    thermo = params['planet']['thermodynamics']

    surface_area  = 4.0 * np.pi * Rp**2
    volume_mantle = (4.0 / 3.0) * np.pi * (Rp**3 - Rc**3)
    rho_mantle    = Mmantle / volume_mantle

    Mh            = params['star']['mass_star_relative'] * c['mass_sun']
    Rh            = params['star']['host_radius']
    tides_on_flag = params['simulation']['tides_on']

    mf_threshold      = params['planet']['convection']['melt_fraction_threshold']
    solidus_low_p_int = thermo['solidus_intercept_low_p']

    # Grid is pre-built in run_model and stored in params
    grid = params['_grid']

    t_sec   = sol_unified.t
    Tr      = sol_unified.y
    n_steps = Tr.shape[1]

    Mmo         = np.zeros(n_steps)
    Wmo         = np.zeros(n_steps)
    meltfrac    = np.zeros(n_steps)
    Patm        = np.zeros(n_steps)
    PO2         = np.zeros(n_steps)
    Q_rad       = np.zeros(n_steps)
    Q_tid       = np.zeros(n_steps)
    tidal_scale = np.zeros(n_steps)
    tidal_shear = np.zeros(n_steps)
    tidal_visc  = np.zeros(n_steps)

    for i in range(n_steps):
        temp_mantle     = Tr[4, i]
        radius_solid    = Tr[5, i]
        mass_water_atm  = max(Tr[7, i], 0.0)
        mass_oxygen_atm = max(Tr[8, i], 0.0)
        temp_surface    = Tr[10, i]

        # --- Phase determination (matches ODE criterion) ---
        _, meltfrac_bulk, meltfrac_mo = get_melt_fractions(temp_mantle, grid)
        if temp_mantle <= solidus_low_p_int:
            meltfrac_bulk = 0.0
            meltfrac_mo   = 0.0

        is_magma_ocean = (meltfrac_bulk >= mf_threshold)
        meltfrac[i]    = meltfrac_bulk

        Mmo[i] = max((4.0 / 3.0) * np.pi * (Rp**3 - radius_solid**3) * rho_mantle, 0.0)
        Wmo[i] = mass_water_atm if is_magma_ocean else 0.0

        if is_magma_ocean:
            if mass_water_atm > 0.0 and Mmo[i] > 0.0:
                Patm[i], _, _ = get_pressure2(
                    temp_mantle, radius_solid, Mmo[i], Mmantle,
                    Rp, gp, Rc, mass_water_atm, params
                )
            if mass_oxygen_atm > 0.0 and Mmo[i] > (0.01 * Mmantle):
                PO2[i], _, _, _ = get_massbalance4(
                    temp_mantle, Patm[i], Mmo[i], mass_oxygen_atm,
                    Xi, FeOt, gp, Rp, params
                )
                if PO2[i] < 0.0:
                    PO2[i] = mass_oxygen_atm * gp / surface_area
            else:
                PO2[i] = mass_oxygen_atm * gp / surface_area

            heatflux_water = min(mass_water_atm / Mmo[i], 1.0) if Mmo[i] > 0 else 0.0
            mf_heat        = meltfrac_mo
            Db_phys        = radius_solid
            tidal_rad      = radius_solid
        else:
            Patm[i] = mass_water_atm * gp / surface_area
            PO2[i]  = mass_oxygen_atm * gp / surface_area

            heatflux_water = Tr[6, i] / Mmantle
            mf_heat        = meltfrac_bulk
            Db_phys        = Rc
            tidal_rad      = Rp

        Q_rad[i] = get_radiogenic_heat(t_sec[i], Mmantle, params)

        _, _, _, _, nu = mantleheatflux(
            temp_mantle, temp_surface, Db_phys, Rp, Rc, gp, rho_mantle,
            heatflux_water, mf_heat, params
        )
        visc  = nu * rho_mantle
        solid_shear_modulus = calc_shear_modulus(meltfrac_bulk, params)

        orbital_freq = semi_a2orbital_motion(Tr[0, i], Mh, Mp)
        _, _, _, _, _, Q_tid[i], tidal_scale[i], tidal_shear[i], tidal_visc[i] = (
            calculate_tidal_dissipation(
                Tr[1, i], orbital_freq, Tr[3, i], Tr[2, i], Rp, Rh, Mp, Mh,
                Rc, visc, solid_shear_modulus,
                tidal_rad, tides_on_flag, params
            )
        )

    # Event time estimates (zero-crossings of meltfrac_bulk - threshold)
    mf_diff        = meltfrac - mf_threshold
    crossings_mf   = np.where(np.diff(np.sign(mf_diff)))[0]
    t_crust        = t_sec[crossings_mf] / c['seconds_per_year']

    wf_diff        = (Tr[6, :] / Mmantle) - 1e-9
    crossings_wf   = np.where(np.diff(np.sign(wf_diff)))[0]
    t_degas        = t_sec[crossings_wf] / c['seconds_per_year']

    return {
        't': t_sec, 'Tr': Tr, 'Mmo': Mmo, 'Wmo': Wmo, 'meltfrac': meltfrac,
        'Patm': Patm, 'PO2': PO2, 'Q_rad': Q_rad, 'Q_tid': Q_tid,
        'tidal_scale': tidal_scale, 'tidal_shear': tidal_shear, 'tidal_visc': tidal_visc,
        't_crust': t_crust, 't_degas': t_degas, 'melt_fraction': meltfrac,
    }
