import sys
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration loader (built-in for Python 3.11+)
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from scipy.integrate import solve_ivp as scisolve_ivp

# Try to import CyRK if available, otherwise fallback to SciPy
try:
    from CyRK import pysolve_ivp
    USE_CYRK = True
except ImportError:
    USE_CYRK = False

if USE_CYRK:
    solve_ivp = pysolve_ivp
else:
    solve_ivp = scisolve_ivp

# Physics & Utility Imports
from ODEs import moEvent_phase1, moEvent_phase2, moEvent_phase3, moODE_phase1, moODE_phase2, moODE_phase3
from utils.postprocess import postprocess_magma_ocean
from utils.get_comp import get_comp
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.general_utils import merge_dicts
from utils.nondim_scales import StateScaler
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

#########################################
############ INITIALIZATION #############
#########################################

'''!!!!!! BEFORE YOU RUN THE MODEL !!!!!!
Ensure your 'params.toml' file is in the same directory.
Scroll all the way to the bottom of this file where it says
"Post-processing" to change how you want your results and plots 
named and saved.

Otherwise, things will be saved as a tab-delimited results.txt file.
There will be one plot with mantle and surface temps, and 
another plot with PO2 and Patm, both plots showing all 3 time stages.'''

# Load Main configuration
with open("baseline_config.toml", "rb") as f:
    params = tomllib.load(f)

# Load specific configs
simulation_config = 'trappist1e'
with open(f'{simulation_config}.toml', "rb") as f:
    specific_params = tomllib.load(f)

# Merge Params: the specific_config will override defaults from baseline_config.
params = merge_dicts(specific_params, params)

# Load stellar and OLR data
stellar = pd.read_csv('data/solardata.txt', sep='\t')
olr_data = np.load('data/OLRdatab.npz')
Temp_K = olr_data['Temp_K']
P_Pa = olr_data['P_Pa']
OLR = olr_data['OLR']
Ts, Ps = np.meshgrid(Temp_K, P_Pa)
t_flux = stellar['tbol'].values
Lbol = stellar['Lbol'].values

# =====================================================================
# --- TOML Parameter Unpacking ---
# =====================================================================
constants = params['constants']
star_params = params['star']
planet_params = params['planet']
orbit_params = params['planet']['orbit']
simulation_params = params['simulation']
material_params = params['planet']['material']
thermo_params = params['planet']['thermodynamics']
comp_params = params['planet']['oxide_composition']

# --- Integration parameters ---
integration_method = simulation_params['method']
integration_rtol = simulation_params['rtol']
integration_atol = simulation_params['atol']
start_time_sec = simulation_params['start_time_years'] * constants['seconds_per_year']
end_time_sec = simulation_params['end_time_years'] * constants['seconds_per_year']
tides_on_flag = simulation_params['tides_on']

if integration_method.lower() not in ('rk23', 'rk45', 'dop853'):
    solve_ivp = scisolve_ivp # CyRK only supports RK-like methods.

# --- Star properties ---
MStar = star_params['mass_star_relative'] * constants['mass_sun']
LStar = star_params['lum_star_relative'] * constants['lum_sun']
host_radius = star_params['host_radius']
tsat_sec = star_params['xuv']['tsat_years'] * constants['seconds_per_year'] # Converted to sec

# --- Planet properties ---
Rp = planet_params['radius_planet_relative'] * constants['radius_earth']
Rc = planet_params['radius_core_relative'] * Rp
Mp = planet_params['mass_planet_relative'] * constants['mass_earth']
core_fraction = planet_params['core_mass_fraction']
Mmantle = (1.0 - core_fraction) * Mp
rho_mantle = Mmantle / ((4.0 / 3.0) * np.pi * (Rp**3 - Rc**3))
gp = constants['G'] * Mp / Rp**2

# --- Orbital & Radiative properties ---
a = orbit_params['initial_semi_major_axis'] * constants['au']
Fstel = LStar / (4.0 * np.pi * a**2)
ASR = (1.0 - planet_params['albedo']) * Fstel / 4.0
Teq = (ASR / constants['stefan_boltzmann']) ** 0.25

# --- Composition & Thermodynamics ---
MH2O = planet_params['ocean_mass_multiplier'] * constants['mass_ocean_earth']
FH2O = MH2O / Mmantle
FeOt = comp_params['mass_frac_FeO_total']
Fe3_Fet = comp_params['ratio_Fe3_to_total_Fe']

Xi = get_comp(params) 

# Solidus params mapped to match original integration logic
Tsol1 = thermo_params['solidus_slope_low_p'] * 1e-9 # Extracted slope in SI
Tsol2 = thermo_params['solidus_intercept_low_p']
Cp = thermo_params['specific_heat_mantle']
alpha_therm = thermo_params['thermal_expansion']

muO = constants['molar_mass_O']
muFeO1_5 = comp_params['molar_mass_FeO1_5']



#########################################
# Execution Wrappers
#########################################
CRASH_IF_SOL_FAILS = False
def monitor_ode(fun, t_span, y0, phase_name, **kwargs):
    """ Wraps solve_ivp with a tqdm progress bar tracking years. """
    t0, tf = t_span
    total_years = (tf - t0) / constants['seconds_per_year']
    
    with tqdm(total=total_years, unit="yr", desc=phase_name, unit_scale=True) as pbar:
        state = {'last_t': t0}

        def wrapped_fun(t, y):
            dt = t - state['last_t']
            if dt > 0:
                pbar.update(dt / constants['seconds_per_year'])
                state['last_t'] = t
            return fun(t, y)

        sol = solve_ivp(wrapped_fun, t_span, y0, **kwargs)
        
        if sol.status == 0 and sol.t[-1] >= tf:
            pbar.n = pbar.total
            pbar.refresh()

    if not sol.success and CRASH_IF_SOL_FAILS:
        raise Exception(f"Integration failed at t={sol.t[-1]} for {phase_name}: {sol.message}.")
    return sol

# =====================================================================
# --- Initial Conditions Setup ---
# =====================================================================
initial_semi_a = a
initial_orbital_freq = semi_a2orbital_motion(initial_semi_a, MStar, Mp)
initial_eccentricity = orbit_params['initial_eccentricity']
initial_spin_host = 2.0 * np.pi / (86400.0 * star_params['spin_days'])
initial_spin_planet = planet_params['initial_spin_multiplier'] * initial_orbital_freq

Tr0 = np.zeros(11, dtype=np.float64)
Tr0[0] = initial_semi_a                       # Semi Major Axis [m]
Tr0[1] = initial_eccentricity                 # Eccentricity
Tr0[2] = initial_spin_host                    # Star Spin Rate [Rad s-1]
Tr0[3] = initial_spin_planet                  # Planet Spin Rate [Rad s-1]
Tr0[4] = planet_params['initial_mantle_temp'] # Planet Mantle Temperature [K]

# initial depth of base of magma ocean
base_depth = (Tr0[4] - Tsol2) * Cp / (Tsol1 * rho_mantle * gp * Cp - alpha_therm * gp * Tr0[4])
Tr0[5] = max(Rp - base_depth, Rc) 

Mmo0 = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - Tr0[5]**3)

Tr0[6] = MH2O - FH2O * Mmo0 # initial abundance of H2O in solid phase
Tr0[7] = FH2O * Mmo0       # total initial water abundance in magma ocean
Tr0[8] = FeOt * Fe3_Fet * Mmo0 * (muO / 2.0 / muFeO1_5) # oxygen in melt FeO(1.5)
Tr0[9] = FeOt * Fe3_Fet * (Mmantle - Mmo0) * (muO / 2.0 / muFeO1_5) # oxygen in solid FeO(1.5)
Tr0[10] = Tr0[4] - 1.0 # surface temperature


#########################################
# Phase 1 - Magma Ocean
#########################################
phase1_ode = partial(
    moODE_phase1, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Ts, Ps=Ps, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol, Xi=Xi, FeOt=FeOt, 
    Temp_K=Temp_K, P_Pa=P_Pa, tsat=tsat_sec, Mp=Mp, LStar=LStar, Rh=host_radius, 
    Mh=MStar, params=params, tides_on_flag=tides_on_flag
)
scaler_phase1 = StateScaler(phase=1, a0=initial_semi_a, Rp=Rp, M_ocean=MH2O)

sol1 = monitor_ode(phase1_ode, t_span=(start_time_sec, end_time_sec), y0=Tr0, 
                   phase_name='Phase 1', method=integration_method, 
                   scaler=scaler_phase1, events=[moEvent_phase1], 
                   rtol=integration_rtol, atol=integration_atol)

#########################################
# Phase 2 - Solidification
#########################################
Mmo_end = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - sol1.y[5, -1]**3)
Wmo_end = max(sol1.y[7, -1], 0.0)

# Calculate Patm
if Wmo_end > 0:
    Patm_end, _, _ = get_pressure2(sol1.y[4, -1], sol1.y[5, -1], Mmo_end, Mmantle, Rp, gp, Rc, Wmo_end, params)
else:
    Patm_end = 0.0

# Calculate PO2
MO_mo_end = sol1.y[8, -1]
if MO_mo_end > 0.0:
    PO2_end, _, _, _ = get_massbalance4(sol1.y[4, -1], Patm_end, Mmo_end, MO_mo_end, Xi, FeOt, gp, Rp, params)
    if PO2_end < 0.0:
        PO2_end = MO_mo_end * gp / (4.0 * np.pi * Rp**2)
else:
    PO2_end = 0.0

# Setup state vector for Phase 2
Tr0_2 = np.zeros(9, dtype=np.float64)
Tr0_2[0] = sol1.y[0, -1]  # Semi Major Axis [m]
Tr0_2[1] = sol1.y[1, -1]  # Eccentricity
Tr0_2[2] = sol1.y[2, -1]  # Star Spin Rate [Rad s-1]
Tr0_2[3] = sol1.y[3, -1]  # Planet Spin Rate [Rad s-1]
Tr0_2[4] = sol1.y[4, -1]  # Planet Mantle Temperature [K]
Tr0_2[5] = sol1.y[6, -1] + sol1.y[7, -1] - (Patm_end * 4.0 * np.pi * Rp**2 / gp) # Water mass in mantle.
Tr0_2[6] = Patm_end * 4.0 * np.pi * Rp**2 / gp # Water mass in atm
Tr0_2[7] = PO2_end * 4.0 * np.pi * Rp**2 / gp  # O2 mass in atm
Tr0_2[8] = sol1.y[10, -1] # Surface temp

# Kickstart Solidification
Tr0_2[5] += 1e-12 * Mmantle

phase2_ode = partial(
    moODE_phase2, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Ts, Ps=Ps, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol, Xi=Xi, FeOt=FeOt, 
    Temp_K=Temp_K, P_Pa=P_Pa, tsat=tsat_sec, Mp=Mp, LStar=LStar, Rh=host_radius, 
    Mh=MStar, params=params, tides_on_flag=tides_on_flag
)

phase2_event = partial(moEvent_phase2, core_mass_fraction=core_fraction, planet_mass=Mp)
scaler_phase2 = StateScaler(phase=2, a0=initial_semi_a, Rp=Rp, M_ocean=MH2O)

sol2 = monitor_ode(phase2_ode, t_span=(sol1.t[-1], end_time_sec), y0=Tr0_2, 
                   phase_name='Phase 2', method=integration_method, 
                   scaler=scaler_phase2, events=[phase2_event], 
                   rtol=integration_rtol, atol=integration_atol)


#########################################
# Phase 3 - Sub-solidus Tectonics
#########################################
Tr0_3 = np.zeros(8, dtype=np.float64)
Tr0_3[0] = sol2.y[0, -1] # Semi Major Axis [m]
Tr0_3[1] = sol2.y[1, -1] # Eccentricity
Tr0_3[2] = sol2.y[2, -1] # Star Spin Rate [Rad s-1]
Tr0_3[3] = sol2.y[3, -1] # Planet Spin Rate [Rad s-1]
Tr0_3[4] = sol2.y[4, -1] # Planet Mantle Temperature [K]
FH2O3    = sol2.y[5, -1] / Mmantle
Tr0_3[5] = sol2.y[6, -1] # Water mass in Atmo
Tr0_3[6] = sol2.y[7, -1] # O2 mass in Atmo
Tr0_3[7] = sol2.y[8, -1] # surface temperature

phase3_ode = partial(
    moODE_phase3, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Ts, Ps=Ps, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol, Xi=Xi, FeOt=FeOt, 
    Temp_K=Temp_K, P_Pa=P_Pa, tsat=tsat_sec, FH2O=FH2O3, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MStar, params=params, tides_on_flag=tides_on_flag
)
scaler_phase3 = StateScaler(phase=3, a0=initial_semi_a, Rp=Rp, M_ocean=MH2O)

sol3 = monitor_ode(phase3_ode, t_span=(sol2.t[-1], end_time_sec), y0=Tr0_3, 
                   phase_name='Phase 3', method=integration_method, 
                   scaler=scaler_phase3, events=None, 
                   rtol=integration_rtol, atol=integration_atol)

#==========================================================================
# Post-processing & Output
#==========================================================================
results = postprocess_magma_ocean(sol1, sol2, sol3, Rp, Rc, Mmantle, gp, OLR, ASR,
                                  Teq, Xi, FeOt, t_flux, Lbol, tsat_sec, a, Mp, LStar, params)

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

# Time arrays
t_tot_sec = np.concatenate([sol1.t, sol2.t, sol3.t])
t_tot_years = t_tot_sec / constants['seconds_per_year']

# State arrays
semia_tot = np.concatenate([sol1.y[0,:], sol2.y[0,:], sol3.y[0,:]])
eccen_tot = np.concatenate([sol1.y[1,:], sol2.y[1,:], sol3.y[1,:]])
spin_host_tot = np.concatenate([sol1.y[2,:], sol2.y[2,:], sol3.y[2,:]])
spin_planet_tot = np.concatenate([sol1.y[3,:], sol2.y[3,:], sol3.y[3,:]])

mantle_T_tot = np.concatenate([sol1.y[4,:], sol2.y[4,:], sol3.y[4,:]])
surf_T_tot = np.concatenate([sol1.y[10,:], sol2.y[8,:], sol3.y[7,:]])

# Derived arrays from post-processor
PO2_tot = np.concatenate([results['phase1']['PO2'], results['phase2']['PO2'], results['phase3']['PO2']])
Patm_tot = np.concatenate([results['phase1']['Patm'], results['phase2']['Patm'], results['phase3']['Patm']])
Q_tidal = np.concatenate([results['phase1']['Q_tid'], results['phase2']['Q_tid'], results['phase3']['Q_tid']])
Q_radiogenic = np.concatenate([results['phase1']['Q_rad'], results['phase2']['Q_rad'], results['phase3']['Q_rad']])
print(f"Total Radiogenic Heating = {np.sum(Q_radiogenic)/1e12:0.3e} TW.")
print(f"Total Tidal Heating = {np.sum(Q_tidal)/1e12:0.3e} TW.")

# Save to CSV
df_results = pd.DataFrame({
    'time_yr': t_tot_years,
    'Mantle_T': mantle_T_tot,
    'Surf_T': surf_T_tot,
    'PO2': PO2_tot,
    'Patm': Patm_tot,
})

df_results.to_csv('results.txt', sep='\t', index=False)


def plot_magma_ocean():
    phase_line_color = 'k'

    # Adjust heating so it can be log plotted.
    Q_tidal[Q_tidal == 0.0] = np.nan
    Q_radiogenic[Q_radiogenic == 0.0] = np.nan

    # Temperature Plot
    fig_tmp, ax_tmp = plt.subplots(figsize=(8, 5))
    ax_tmp.plot(t_tot_years, mantle_T_tot, label='Mantle T', color='black')
    ax_tmp.plot(t_tot_years, surf_T_tot, label='Surface T', color='orange')
    ax_tmp.set_xlabel('Time [yr]')
    ax_tmp.set_ylabel('Temperature [K]')
    ax_tmp.set_title('Mantle and Surface Temperature Evolution')
    ax_tmp.set_xscale('log')
    ax_tmp.set_yscale('linear')
    ax_heat = ax_tmp.twinx()
    ax_tmp.axvline(x=sol1.t[-1]/constants['seconds_per_year'], ls=':', c=phase_line_color)
    ax_tmp.axvline(x=sol2.t[-1]/constants['seconds_per_year'], ls='-.', c=phase_line_color)
    ax_heat.plot(t_tot_years, Q_tidal/1e12, label='Tidal', color='red')
    ax_heat.plot(t_tot_years, Q_radiogenic/1e12, label='Radiogenic', color='green')
    ax_heat.set_yscale('log')
    ax_heat.set_ylabel('Heating [TW]')
    ax_heat.legend(loc='lower right')
    ax_tmp.legend(loc='lower left')
    ax_tmp.grid(True)
    fig_tmp.tight_layout()

    # Atmosphere Plot
    fig_atm, ax_atm = plt.subplots(figsize=(8, 5))
    ax_atm.plot(t_tot_years, PO2_tot, label='PO2', color='blue')
    ax_atm.plot(t_tot_years, Patm_tot, label='Patm', color='black')
    ax_atm.set_xlabel('Time [yr]')
    ax_atm.set_ylabel('Pressure [Pa]')
    ax_atm.set_title('PO2 and Patm Evolution')
    ax_atm.set_xscale('log')
    ax_atm.set_yscale('log')
    ax_atm.axvline(x=sol1.t[-1]/constants['seconds_per_year'], ls=':', c=phase_line_color)
    ax_atm.axvline(x=sol2.t[-1]/constants['seconds_per_year'], ls='-.', c=phase_line_color)
    ax_atm.legend()
    ax_atm.grid(True)
    fig_atm.tight_layout()

    # Orbit Plot
    fig_orb, ax_orb = plt.subplots(figsize=(8, 5))
    ax_orb.set_xscale('log')
    ax2_orb = ax_orb.twinx()
    ax_orb.plot(t_tot_years, semia_tot / constants['au'], color='red') 
    ax2_orb.plot(t_tot_years, eccen_tot, label='Eccentricity', color='blue')
    ax_orb.set_xlabel('Time [yr]')
    ax_orb.set_ylabel('Semi Major Axis [Au]', color='red')
    ax2_orb.set_ylabel('Eccentricity', color='blue')
    ax_orb.spines['left'].set_color('red')
    ax2_orb.spines['right'].set_color('blue')
    ax_orb.set_title('Orbital Evolution')
    ax_orb.set_xscale('log')
    ax_orb.set_yscale('linear')
    ax2_orb.set_yscale('linear')
    ax_orb.axvline(x=sol1.t[-1]/constants['seconds_per_year'], ls=':', c=phase_line_color)
    ax_orb.axvline(x=sol2.t[-1]/constants['seconds_per_year'], ls='-.', c=phase_line_color)
    ax_orb.grid(True)
    ax_orb.tick_params(axis='y', colors='red')
    ax2_orb.tick_params(axis='y', colors='blue')
    fig_orb.tight_layout()

    # Spin Plot
    spin_host_period = (2 * np.pi / spin_host_tot) / 86400.0
    spin_planet_period = (2 * np.pi / spin_planet_tot) / 86400.0
    
    fig_spin, ax_spin = plt.subplots(figsize=(8, 5))
    ax_spin.plot(t_tot_years, spin_planet_period, color='blue', label='Planet')
    ax_spin.plot(t_tot_years, spin_host_period, color='red', label='Star')
    ax_spin.set_xlabel('Time [yr]')
    ax_spin.set_ylabel('Spin Period [days]')
    ax_spin.set_title('Spin Evolution')
    ax_spin.set_xscale('log')
    ax_spin.set_yscale('linear')
    ax_spin.axvline(x=sol1.t[-1]/constants['seconds_per_year'], ls=':', c=phase_line_color)
    ax_spin.axvline(x=sol2.t[-1]/constants['seconds_per_year'], ls='-.', c=phase_line_color)
    ax_spin.grid(True)
    ax_spin.legend()
    fig_spin.tight_layout()
    
    plt.show()

# Execute plot generator
plot_magma_ocean()
