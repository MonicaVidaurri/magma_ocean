import numpy as np
import pandas as pd
import tomllib
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from scipy.integrate import solve_ivp as scisolve_ivp

# Try to import CyRK if available, otherwise fallback to SciPy
try:
    from CyRK import pysolve_ivp
    USE_CYRK = True
    solve_ivp = pysolve_ivp
except ImportError:
    USE_CYRK = False
    solve_ivp = scisolve_ivp

# Physics & Utility Imports
from ODEs import (moEvent_phase1, moEvent_phase2, moEvent_phase3, 
                  moODE_phase1, moODE_phase2, moODE_phase3)
from utils.postprocess import postprocess_magma_ocean
from utils.get_comp import get_comp
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

CRASH_IF_SOL_FAILS = False
def monitor_ode(fun, t_span, y0, phase_name, method, solver_func, seconds_per_year, **kwargs):
    """Wraps the solver with a progress bar in Years."""
    t0, tf = t_span
    total_years = (tf - t0) / seconds_per_year
    
    with tqdm(total=total_years, unit="yr", desc=phase_name, unit_scale=True) as pbar:
        state = {'last_t': t0}

        def wrapped_fun(t, y):
            dt = t - state['last_t']
            if dt > 0:
                pbar.update(dt / seconds_per_year)
                state['last_t'] = t
            return fun(t, y)

        sol = solver_func(wrapped_fun, t_span, y0, method=method, **kwargs)
        if sol.status == 0 and sol.t[-1] >= tf:
            pbar.numerical_params = pbar.total
            pbar.refresh()
    if not sol.success and CRASH_IF_SOL_FAILS:
        raise Exception(f"Integration failed at t={sol.t[-1]} for {phase_name}: {sol.message}.")
    return sol

def run_simulation(config_path='params.toml'):
    # --- Load Configuration ---
    with open(config_path, 'rb') as config_file:
        params = tomllib.load(config_file)
    constants        = params['constants']
    star_params      = params['star']
    planet_params    = params['planet']
    orbit_params     = params['planet']['orbit']
    numerical_params = params['numerical']
    thermo_params    = params['planet']['thermodynamics']
    material_params  = params['planet']['material']

    # --- Derived Physical Parameters (MKS) ---
    MStar = star_params['MStar_relative'] * constants['MSun']
    LStar = star_params['LStar_relative'] * constants['LSun']
    Rp = planet_params['Rp_relative'] * constants['REarth']
    Rc = planet_params['Rc_relative'] * Rp
    Mp = planet_params['Mp_relative'] * constants['MEarth']
    Mmantle = (1.0 - planet_params['core_fraction']) * Mp
    rho_mantle = Mmantle / ((4.0/3.0) * np.pi * (Rp**3 - Rc**3))
    gp = constants['G'] * Mp / Rp**2
    semi_a = orbit_params['a_relative'] * constants['AU']
    MH2O = planet_params['Earth_oceans'] * constants['Earth_ocean_mass']
    surface_area = 4.0 * np.pi * Rp**2
    
    # Radiative Prep
    Fstel = LStar / (4.0 * np.pi * semi_a**2)
    ASR = (1.0 - planet_params['albedo']) * Fstel / 4.0
    Teq = (ASR / constants['stefan_boltzmann']) ** 0.25

    # Solver choice
    solver = solve_ivp
    if USE_CYRK and numerical_params['method'].lower() in ('rk23', 'rk45', 'dop853'):
        solver = scisolve_ivp

    # --- Initial State Vector (Phase 1) ---
    Xi = get_comp(params)
    Tr0 = np.zeros(11)
    orb_freq = semi_a2orbital_motion(semi_a, MStar, Mp)
    
    Tr0[0], Tr0[1] = semi_a, orbit_params['initial_eccentricity']
    Tr0[2] = 2.0 * np.pi / (orbit_params['tday_host'] * 86400.0)
    Tr0[3] = orb_freq * orbit_params['tday_planet_multiplier']
    Tr0[4] = planet_params['mantle_temp_initial']
    
    # Initial solid radius
    depth_0 = (Tr0[4] - thermo_params['solidus_intercept_low_p']) * material_params['specific_heat_mantle'] / \
              (thermo_params['solidus_slope_low_p'] * rho_mantle * gp * material_params['specific_heat_mantle'] - \
               thermo_params['thermal_expansion'] * gp * Tr0[4])
    Tr0[5] = max(Rp - depth_0, Rc)
    
    Mmo0 = (4.0/3.0) * np.pi * rho_mantle * (Rp**3 - Tr0[5]**3)
    Tr0[7] = (MH2O / Mmantle) * Mmo0
    Tr0[6] = MH2O - Tr0[7]
    Tr0[8] = planet_params['FeOt'] * planet_params['ratio_Fe3_to_total_Fe_initial'] * Mmo0 * (constants['molar_mass_O'] / (2.0 * material_params['molar_mass_FeO1_5']))
    Tr0[9] = planet_params['FeOt'] * planet_params['ratio_Fe3_to_total_Fe_initial'] * (Mmantle - Mmo0) * (constants['molar_mass_O'] / (2.0 * material_params['molar_mass_FeO1_5']))
    Tr0[10] = Tr0[4] - 1.0

    # Load External Tracks
    stellar = pd.read_csv('data/solardata.txt', sep='\t')
    t_span = (numerical_params['start_time_years'] * constants['seconds_per_year'], numerical_params['end_time_years'] * constants['seconds_per_year'])

    # =====================================================================
    # --- PHASE 1: Magma Ocean ---
    # =====================================================================
    p1_ode = partial(moODE_phase1, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
                     t_flux=stellar['tbol'].values, Lbol=stellar['Lbol'].values, Xi=Xi, FeOt=planet_params['FeOt'],
                     tsat=star_params['tsat_years'], Mp=Mp, LStar=LStar, Rh=star_params['host_radius'], Mh=MStar, 
                     params=params, tides_on_flag=numerical_params['tides_on'])

    sol1 = monitor_ode(p1_ode, t_span, Tr0, 'Phase 1', numerical_params['method'], solver, constants['seconds_per_year'],
                       events=moEvent_phase1, rtol=numerical_params['rtol'], atol=numerical_params['atol'])

    # =====================================================================
    # --- PHASE 2: Solidification / Degassing ---
    # =====================================================================
    Mmo_e = (4.0/3.0) * np.pi * rho_mantle * (Rp**3 - sol1.y[5, -1]**3)
    Wmo_e = max(sol1.y[7, -1], 0.0)
    
    Patm_e, _, _ = get_pressure2(sol1.y[4, -1], sol1.y[5, -1], Mmo_e, Mmantle, Rp, gp, Rc, Wmo_e, params)
    PO2_e, _, _, _ = get_massbalance4(sol1.y[4, -1], Patm_e, Mmo_e, sol1.y[8, -1], Xi, planet_params['FeOt'], gp, Rp, params)

    Tr0_2 = np.zeros(9)
    Tr0_2[0] = sol1.y[0, -1]
    Tr0_2[1] = sol1.y[1, -1]
    Tr0_2[2] = sol1.y[2, -1]
    Tr0_2[3] = sol1.y[3, -1]
    Tr0_2[4] = sol1.y[4, -1]
    Tr0_2[5] = sol1.y[6, -1] + sol1.y[7, -1] - (Patm_e * surface_area / gp)
    Tr0_2[6] = Patm_e * surface_area / gp
    Tr0_2[7] = PO2_e * surface_area / gp
    Tr0_2[8] = sol1.y[10, -1]

    p2_ode = partial(moODE_phase2, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
                     t_flux=stellar['tbol'].values, Lbol=stellar['Lbol'].values, tsat=star_params['tsat_years'], 
                     Mp=Mp, LStar=LStar, Rh=star_params['host_radius'], Mh=MStar, params=params, tides_on_flag=numerical_params['tides_on'])

    sol2 = monitor_ode(p2_ode, (sol1.t[-1], t_span[1]), Tr0_2, 'Phase 2', numerical_params['method'], solver, constants['seconds_per_year'],
                       events=moEvent_phase2, rtol=numerical_params['rtol'], atol=numerical_params['atol'])

    # =====================================================================
    # --- PHASE 3: Tectonic Phase ---
    # =====================================================================
    Tr0_3 = np.zeros(8)
    Tr0_3[0] = sol2.y[0, -1]
    Tr0_3[1] = sol2.y[1, -1]
    Tr0_3[2] = sol2.y[2, -1]
    Tr0_3[3] = sol2.y[3, -1]
    Tr0_3[4] = sol2.y[4, -1]
    Tr0_3[5] = sol2.y[6, -1]
    Tr0_3[6] = sol2.y[7, -1]
    Tr0_3[7] = sol2.y[8, -1]

    p3_ode = partial(moODE_phase3, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
                     t_flux=stellar['tbol'].values, Lbol=stellar['Lbol'].values, tsat=star_params['tsat_years'], 
                     FH2O=Tr0_3[1]/Mmantle, Mp=Mp, LStar=LStar, Rh=star_params['host_radius'], Mh=MStar, 
                     params=params, tides_on_flag=numerical_params['tides_on'])

    sol3 = monitor_ode(p3_ode, (sol2.t[-1], t_span[1]), Tr0_3, 'Phase 3', numerical_params['method'], solver, constants['seconds_per_year'],
                       rtol=numerical_params['rtol'], atol=numerical_params['atol'])

    # --- Post-processing & Output ---
    results = postprocess_magma_ocean(
        sol1, sol2, sol3, Rp, Rc, Mmantle,
        Mp, gp,
        Teq, a, LStar, Xi, FeOt, t_flux,
        stellar['Lbol'].values,
        tsat=1e9, params=params)

    # Pad arrays so they're all the same size
    def pad_array(arr, target_rows):
        n_rows, n_cols = arr.shape
        if n_rows < target_rows:
            padded = np.full((target_rows, n_cols), np.nan)
            padded[:n_rows, :] = arr
            return padded
        return arr

    max_rows = max(sol1.y.shape[0], sol2.y.shape[0], sol3.y.shape[0])
    Tr1_padded = pad_array(sol1.y, max_rows)
    Tr2_padded = pad_array(sol2.y, max_rows)
    Tr3_padded = pad_array(sol3.y, max_rows)

    Tr_all = np.concatenate([Tr1_padded, Tr2_padded, Tr3_padded], axis=1)
    print("\nPost-processing and saving results...")
    t_tot_sec = np.concatenate([sol1.t, sol2.t, sol3.t])
    t_tot_years = t_tot_sec / SEC_PER_YEAR

    semia_tot = np.concatenate([sol1.y[0,:], sol2.y[0,:], sol3.y[0,:]])
    eccen_tot = np.concatenate([sol1.y[1,:], sol2.y[1,:], sol3.y[1,:]])
    spin_host_tot = np.concatenate([sol1.y[2,:], sol2.y[2,:], sol3.y[2,:]])
    spin_planet_tot = np.concatenate([sol1.y[3,:], sol2.y[3,:], sol3.y[3,:]])

    mantle_T_tot = np.concatenate([sol1.y[4,:], sol2.y[4,:], sol3.y[4,:]])
    surf_T_tot = np.concatenate([sol1.y[-1,:], sol2.y[-1,:], sol3.y[-1,:]])

    PO2_tot = np.concatenate([results['phase1']['PO2'], results['phase2']['PO2'], results['phase3']['PO2']])
    Patm_tot = np.concatenate([results['phase1']['Patm'], results['phase2']['Patm'], results['phase3']['Patm']])


    results_out = {
        'time_yr': t_tot_years,
        'Mantle_T': mantle_T_tot,
        'Surf_T': surf_T_tot,
        'PO2': PO2_tot,
        'Patm': Patm_tot,
        'semia': semia_tot,
        'eccen': eccen_tot,
        'spin_host': spin_host_tot,
        'spin_planet': spin_planet_tot
    }
    df_results = pd.DataFrame(results_out)
    df_results.to_csv('results.txt', sep='\t', index=False)
    
    return results_out, t_tot_years, constants['AU']

def plot_results(res, t_yr, au_scale):
    mantle_T_tot = res['Mantle_T']
    surf_T_tot = res['Surf_T']
    PO2_tot = res['PO2']
    Patm_tot = res['Patm']
    semia_tot = res['semia']
    eccen_tot = res['eccen']
    spin_host_tot = res['spin_host']
    spin_planet_tot = res['spin_planet']

    # Temperature Plot
    fig_tmp, ax_tmp = plt.subplots(figsize=(8, 5))
    ax_tmp.plot(t_years, mantle_T_tot, label='Mantle T', color='black')
    ax_tmp.plot(t_years, surf_T_tot, label='Surface T', color='orange')
    ax_tmp.set_xlabel('Time [yr]')
    ax_tmp.set_ylabel('Temperature [K]')
    ax_tmp.set_title('Mantle and Surface Temperature Evolution')
    ax_tmp.set_xscale('log')
    ax_tmp.set_yscale('log')
    ax_tmp.legend()
    ax_tmp.grid(True)
    fig_tmp.tight_layout()

    # Atmosphere Plot
    fig_atm, ax_atm = plt.subplots(figsize=(8, 5))
    ax_atm.plot(t_years, PO2_tot, label='PO2', color='blue')
    ax_atm.plot(t_years, Patm_tot, label='Patm', color='black')
    ax_atm.set_xlabel('Time [yr]')
    ax_atm.set_ylabel('Pressure [Pa]')
    ax_atm.set_title('PO2 and Patm Evolution')
    ax_atm.set_xscale('log')
    ax_atm.set_yscale('log')
    ax_atm.legend()
    ax_atm.grid(True)
    fig_atm.tight_layout()

    # Orbit plot
    fig_orb, ax_orb = plt.subplots(figsize=(8, 5))
    ax_orb.set_xscale('log')
    ax2_orb = ax_orb.twinx()
    ax_orb.plot(t_years, semia_tot / AU, color='red')  # Convert back to AU for plotting
    ax2_orb.plot(t_years, eccen_tot, label='Eccentricity', color='blue')

    ax_orb.set_xlabel('Time [yr]')
    ax_orb.set_ylabel('Semi Major Axis [Au]', constants='red')
    ax2_orb.set_ylabel('Eccentricity', constants='blue')
    ax_orb.spines['left'].set_color('red')
    ax2_orb.spines['right'].set_color('blue')
    ax_orb.set_title('Orbital Evolution')
    ax_orb.set_xscale('log')
    ax_orb.set_yscale('log')
    ax2_orb.set_yscale('log')
    ax_orb.grid(True)
    fig_orb.tight_layout()
    ax_orb.tick_params(axis='y', color='red', labelcolor='red')
    ax2_orb.tick_params(axis='y', color='blue', labelcolor='blue')

    # Spin Plot
    spin_host_period = (2 * np.pi / spin_host_tot) / 86400.0
    spin_planet_period = (2 * np.pi / spin_planet_tot) / 86400.0
    
    fig_spin, ax_spin = plt.subplots(figsize=(8, 5))
    ax_spin.plot(t_years, spin_planet_period, color='blue', label='Planet')
    ax_spin.plot(t_years, spin_host_period, color='red', label='Star')

    ax_spin.set_xlabel('Time [yr]')
    ax_spin.set_ylabel('Spin Period [days]')
    ax_spin.set_title('Spin Evolution')
    ax_spin.set_xscale('log')
    ax_spin.set_yscale('log')
    ax_spin.grid(True)
    ax_spin.legend()
    fig_spin.tight_layout()
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    results, t_years, AU = run_simulation()
    plot_results(results, t_years, AU)