import numpy as np

from utils.get_meltfrac import get_meltfrac
from utils.get_meltfracb import get_meltfracb
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4

from physics.mantleheatflux import mantleheatflux
from physics.radiogenics import get_radiogenic_heat
from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def postprocess_magma_ocean(
        sol_unified, Rp, Rc, Mmantle, gp, OLR_interp, ASR, Teq, Xi, FeOt, 
        t_flux, Lbol, tsat, a, Mp, LStar, params):
    
    c = params['constants']
    thermo = params['planet']['thermodynamics']
    
    surface_area  = 4.0 * np.pi * Rp**2
    volume_mantle = (4.0 / 3.0) * np.pi * (Rp**3 - Rc**3)
    rho_mantle    = Mmantle / volume_mantle

    Mh = params['star']['mass_star_relative'] * c['mass_sun']
    Rh = params['star']['host_radius']
    tides_on_flag = params['simulation']['tides_on']
    solidus_low_p_int = thermo['solidus_intercept_low_p']

    t_sec = sol_unified.t
    Tr    = sol_unified.y
    n_steps = Tr.shape[1]

    # Derived arrays
    Mmo      = np.zeros(n_steps)
    Wmo      = np.zeros(n_steps)
    meltfrac = np.zeros(n_steps)
    Patm     = np.zeros(n_steps)
    PO2      = np.zeros(n_steps)
    Q_rad    = np.zeros(n_steps)
    Q_tid    = np.zeros(n_steps)
    tidal_scale = np.zeros(n_steps)
    tidal_shear = np.zeros(n_steps)
    tidal_visc  = np.zeros(n_steps)

    for i in range(n_steps):
        temp_mantle = Tr[4, i]
        radius_solid = Tr[5, i]
        mass_water_atm = max(Tr[7, i], 0.0)
        mass_oxygen_atm = max(Tr[8, i], 0.0)
        temp_surface = Tr[10, i]

        is_magma_ocean = (radius_solid < Rp - 0.01)
        Mmo[i] = max((4.0 / 3.0) * np.pi * (Rp**3 - radius_solid**3) * rho_mantle, 0.0)
        Wmo[i] = mass_water_atm if is_magma_ocean else 0.0

        if is_magma_ocean:
            _, meltfrac[i] = get_meltfrac(gp, temp_mantle, Rp, Rc, Mmantle, params)
            if mass_water_atm > 0.0 and Mmo[i] > 0.0:
                Patm[i], _, _ = get_pressure2(temp_mantle, radius_solid, Mmo[i], Mmantle, Rp, gp, Rc, mass_water_atm, params)
            if mass_oxygen_atm > 0.0 and Mmo[i] > (0.01 * Mmantle):
                PO2[i], _, _, _ = get_massbalance4(temp_mantle, Patm[i], Mmo[i], mass_oxygen_atm, Xi, FeOt, gp, Rp, params)
                if PO2[i] < 0.0:
                    PO2[i] = mass_oxygen_atm * gp / surface_area
            else:
                PO2[i] = mass_oxygen_atm * gp / surface_area
                
            Db_phys = radius_solid
            heatflux_water = min(mass_water_atm / Mmo[i], 1.0) if Mmo[i] > 0 else 0.0
            tidal_rad = radius_solid
        else:
            if temp_mantle > solidus_low_p_int:
                _, meltfrac[i] = get_meltfracb(gp, temp_mantle, Rp, Rc, Mmantle, params)
            Patm[i] = mass_water_atm * gp / surface_area
            PO2[i]  = mass_oxygen_atm * gp / surface_area
            
            Db_phys = Rc
            heatflux_water = Tr[6, i] / Mmantle
            tidal_rad = Rp

        Q_rad[i] = get_radiogenic_heat(t_sec[i], Mmantle, params)
        _, _, _, _, nu = mantleheatflux(temp_mantle, temp_surface, Db_phys, Rp, Rc, gp, rho_mantle, heatflux_water, meltfrac[i], params)
        
        orbital_freq = semi_a2orbital_motion(Tr[0, i], Mh, Mp)
        _, _, _, _, _, Q_tid[i], tidal_scale[i], tidal_shear[i], tidal_visc[i] = calculate_tidal_dissipation(
            Tr[1, i], orbital_freq, Tr[3, i], Tr[2, i], Rp, Rh, Mp, Mh,
            temp_mantle, Rc, nu, rho_mantle, meltfrac[i], tidal_rad, tides_on_flag, params
        )

    # Calculate zero-crossings to pass plot lines back to run_model
    # 1. Crust Forms/Melts
    rs_diff = Tr[5, :] - (Rp - 0.01)
    crossings_rs = np.where(np.diff(np.sign(rs_diff)))[0]
    t_crust = t_sec[crossings_rs] / c['seconds_per_year']

    # 2. Degassing Exhausts/Starts
    wf_diff = (Tr[6, :] / Mmantle) - 1e-9
    crossings_wf = np.where(np.diff(np.sign(wf_diff)))[0]
    t_degas = t_sec[crossings_wf] / c['seconds_per_year']

    return {
        't': t_sec, 'Tr': Tr, 'Mmo': Mmo, 'Wmo': Wmo, 'meltfrac': meltfrac,
        'Patm': Patm, 'PO2': PO2, 'Q_rad': Q_rad, 'Q_tid': Q_tid,
        'tidal_scale': tidal_scale, 'tidal_shear': tidal_shear, 'tidal_visc': tidal_visc,
        't_crust': t_crust, 't_degas': t_degas
    }
