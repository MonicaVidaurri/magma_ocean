import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

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

from ODEs import moEvent_phase1, moEvent_phase2, moEvent_phase3, moODE_phase1, moODE_phase2, moODE_phase3
from utils.postprocess import postprocess_magma_ocean
from utils.get_comp import get_comp
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

#########################################
############ USER INPUT HERE ############
# Load stellar and OLR data
stellar = pd.read_csv('data/solardata.txt', sep='\t')
# Set params
params = {
####### all TRAPPIST values taken from Agol et al. (2021)
####### all Proxima values taken from Faria et al. (2022)

####### star_params
    #TRAPPIST LStar = 0.000553, Mstar = 0.0898
    'LStar': 1, # in units of LSun
    'MStar': 1, # units of MSun
####### planet_params
    #TRAPPIST 1e: Rp = 0.92, Rc = 0.19, Mp = 0.692, core_fraction = 0.236
    #       mantle_temp = 3000, a = 2.925e-2, tday = 6.1
    #TRAPPIST 1f: Rp = 1.045, Rc = 0.429, Mp = 1.039, core_fraction = 0.192
    #        mantle_temp = 3500, a = 3.849e-2, tday = 9.2
    #Proxima b: Rp = 1.03, Rc = 0.429, Mp = 1.07, core_fraction = 0.2
    #        mantle_temp = lol idk, a =4.8e-2, tday = 11.2
    #Earth: Rc = 0.19, core_fraction = 0.16, choose your own adventure for
    #        mantle_temp. I typically stay at 3000
    'Rp': 0.92 , # units of REarth
    'Rc': 0.19 , # will be multiplied by REarth
    'Mp': 0.692 , # units of MEarth
    'a': 2.925e-2,  # will be multipled by Earth AU
    'tday': 6.1,
    'core_fraction': 0.236,

    # 'Rp': 1 , # units of REarth
    # 'Rc': 0.19 , # will be multiplied by REarth
    # 'Mp': 1 , # units of MEarth
    'Albedo': 0.2,
    # 'core_fraction': 0.16,
    'Earth_oceans': 2, # will be multiplied by 1 Earth ocean, 1.39e21 kg
    'mantle_temp': 3000, #[k]
    'FeOt': 0.08,
    # 'a': 1,  # will be multipled by Earth AU
    # 'tday': 1, # in Earth days
####### define XUV model; 1 = high, 2 = low; MANUALLY CHANGE IN GET_LOSS FOR NOW
    'XUV': 2
    #^ girl...did you change this in utils/get_loss???
}

'''!!!!!! BEFORE YOU RUN THE MODEL !!!!!!
Scroll all the way to the bottom of this file where it says
"Post-processing" to change how you want your results and plots 
named and saved and all that good stuff :)

Otherwise, things will be saved as a tab-delimited results.txt file.
There will be one plot with mantle and surface temps, and 
another plot with PO2 and Patm, both plots showing all 3 time stages.'''
#########################################
#########################################

#============================== begin model ==============================
# --- Constants & Conversions ---
G = 6.6743e-11
sigma = 5.67e-8
SEC_PER_YEAR = 3.15569e7

# --- Star properties ---
MSun = 1.989e30 # [kg]
LSun = 3.846e26 # [W]
MStar = params['MStar'] * MSun  # stellar mass [kg]
LStar = params['LStar'] * LSun  # stellar luminosity [W]

# Need the host star radius to calculate change in spin rate.
# TODO: Connect to params dict! Hardcoded for now
host_radius = 1.0 * 6.957e8

# --- Integration parameters (Now strictly in Seconds) ---
integration_method = 'BDF' 
integration_rtol = 1.0e-4
integration_atol = 1.0e-6

if integration_method.lower() not in ('rk23', 'rk45', 'dop853'):
    # CyRK only supports RK-like methods.
    solve_ivp = scisolve_ivp

start_time_sec = 1.0 * SEC_PER_YEAR
end_time_sec = 8.0e9 * SEC_PER_YEAR

# --- Read from star and OLR files ---
olr_data = np.load('data/OLRdatab.npz')
Temp_K = olr_data['Temp_K']
P_Pa = olr_data['P_Pa']
OLR = olr_data['OLR']
Ts, Ps = np.meshgrid(Temp_K, P_Pa)
t_flux = stellar['tbol']
Lbol = stellar['Lbol']

# --- Planet properties ---
MEarth = 5.97e24 # [kg]
REarth =  6371e3 # [m]
AU = 1.496e11    # [m]

Rp = params['Rp'] * REarth
Rc = params['Rc'] * Rp
Mp = params['Mp'] * MEarth
core_fraction = params['core_fraction']
Mmantle = (1 - core_fraction) * Mp

# Orbital distance
a = params['a'] * AU  # meters

# Surface fluxes
Albedo = params['Albedo']
Fat1AU = LStar * 1366
Fstel = LStar / (4 * np.pi * a**2)   # W/m^2
ASR = (1 - Albedo) * Fstel / 4
Teq = (ASR / sigma) ** 0.25

# Mantle properties
rho_mantle = Mmantle / ((4.0 / 3.0) * np.pi * (Rp**3 - Rc**3))
gp = G * Mp / Rp**2

# Water content
Earth_ocean = 1.39e21
MH2O = params['Earth_oceans'] * Earth_ocean
FH2O = MH2O / Mmantle
FeOt = params['FeOt']
Fe3_Fet = 1e-12

# Solidus parameters
Tsol1 = 26.53e-9
Tsol2 = 1373
Cp = 1.2e3
alpha_therm = 2e-5

# Oxide fraction for magma (1 = BSE)
Xi = get_comp(1) 
muO = 15.9994e-3
muFeO1_5 = 159.689e-3 / 2.0
muFeO = 71.845e-3

# Tide flag
tides_on_flag = False

# --- Initial orbital and spin conditions ---
# TODO: add these to params. Most are pulled from trappist1
initial_semi_a = a
initial_orbital_freq = semi_a2orbital_motion(initial_semi_a, MStar, Mp)
initial_eccentricity = 0.1
initial_spin_host = 2 * np.pi / (86400.0 * 3.295)
initial_spin_planet = 10 * initial_orbital_freq  # Start it out spinning much faster than orbital motion

# --- Initial conditions for phase 1 ---
Tr0 = np.zeros(11, dtype=np.float64)
Tr0[0] = initial_semi_a
Tr0[1] = initial_eccentricity
Tr0[2] = initial_spin_host
Tr0[3] = initial_spin_planet

Tr0[4] = params['mantle_temp']  # mantle temperature

# initial depth of base of magma ocean
Tr0[5] = max(Rp - (Tr0[4] - Tsol2) * Cp / (Tsol1 * rho_mantle * gp * Cp - alpha_therm * gp * Tr0[4]), Rc) 
Mmo0 = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - Tr0[5]**3) # initial mass of magma ocean

Tr0[6] = MH2O - FH2O * Mmo0 # initial abundance of H2O in solid phase
Tr0[7] = FH2O * Mmo0 # total initial water abundance in magma ocean
Tr0[8] = FeOt * Fe3_Fet * Mmo0 * (muO / 2.0 / muFeO1_5) # total initial mass of O in FeO(1.5)
Tr0[9] = FeOt * Fe3_Fet * (Mmantle - Mmo0) * (muO / 2.0 / muFeO1_5) # mass of O stored in solid FeO(1.5)
Tr0[10] = Tr0[4] - 1.0  # initial surface temperature is equal to initial potential temp

#########################################
# Phase 1 - bigass ball of magma baby!!!!
#########################################
CRASH_IF_SOL_FAILS = False

def monitor_ode(fun, t_span, y0, phase_name, **kwargs):
    """
    Wraps solve_ivp with a tqdm progress bar. Displays progress in YEARS 
    even though the underlying math evaluates in SECONDS.
    """
    t0, tf = t_span
    
    # Initialize tqdm (convert bounds to years for display)
    total_years = (tf - t0) / SEC_PER_YEAR
    with tqdm(total=total_years, unit="yr", desc=phase_name, unit_scale=True) as pbar:
        
        state = {'last_t': t0}

        def wrapped_fun(t, y):
            dt = t - state['last_t']
            
            # Only update if time moved forward (ignore solver backtracking)
            if dt > 0:
                # Update progress bar in years
                pbar.update(dt / SEC_PER_YEAR)
                state['last_t'] = t
                
            return fun(t, y)

        sol = solve_ivp(wrapped_fun, t_span, y0, **kwargs)
        
        # Ensure the bar hits 100% if the solver finished successfully at tf
        if sol.status == 0 and sol.t[-1] >= tf:
            pbar.n = pbar.total
            pbar.refresh()

    if not sol.success and CRASH_IF_SOL_FAILS:
        raise Exception(f"Integration failed at t={sol.t[-1]} for {phase_name}: {sol.message}.")
    return sol


phase1_ode = partial(
    moODE_phase1, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MSun, tides_on_flag=tides_on_flag)

sol1 = monitor_ode(phase1_ode, t_span=(start_time_sec, end_time_sec), y0=Tr0, 
                   phase_name='Phase 1', method=integration_method, 
                   events=[moEvent_phase1], rtol=integration_rtol, atol=integration_atol)


#########################################
# Phase 2 - starting to solidify......
#########################################
# Extract final state variables from Phase 1
Mmo_end = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - sol1.y[5, -1]**3)
Wmo_end = max(sol1.y[7, -1], 0.0)

# Calculate Patm
if Wmo_end > 0:
    Patm_end, _, _ = get_pressure2(sol1.y[4, -1], sol1.y[5, -1], Mmo_end, Mmantle, Rp, gp, Rc, Wmo_end)
else:
    Patm_end = 0.0

# Calculate PO2
MO_mo_end = sol1.y[8, -1]
if MO_mo_end > 0.0:
    PO2_end, _, _, _ = get_massbalance4(sol1.y[4, -1], Patm_end, Mmo_end, MO_mo_end, Xi, FeOt, gp, Rp)
    if PO2_end < 0.0:
        PO2_end = MO_mo_end * gp / (4.0 * np.pi * Rp**2)
else:
    PO2_end = 0.0
# -------------------------------------------------------------

Tr0_2 = np.zeros(9, dtype=np.float64)
Tr0_2[0] = sol1.y[0, -1]
Tr0_2[1] = sol1.y[1, -1]
Tr0_2[2] = sol1.y[2, -1]
Tr0_2[3] = sol1.y[3, -1]

Tr0_2[4] = sol1.y[4, -1]  # mantle temp

# Extract atmospheric mass to partition water
Tr0_2[5] = sol1.y[6, -1] + sol1.y[7, -1] - (Patm_end * 4.0 * np.pi * Rp**2 / gp) # Water in solid mantle
Tr0_2[6] = Patm_end * 4.0 * np.pi * Rp**2 / gp  # water in atmosphere
Tr0_2[7] = PO2_end * 4.0 * np.pi * Rp**2 / gp   # O2 in atm
Tr0_2[8] = sol1.y[10, -1] # surface temp

phase2_ode = partial(
    moODE_phase2, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MSun, tides_on_flag=tides_on_flag)

# Kickstart Solidification
Tr0_2[5] += 1e-12 * Mmantle

phase2_event = partial(moEvent_phase2, core_mass_fraction=core_fraction, planet_mass=Mp)

sol2 = monitor_ode(phase2_ode, t_span=(sol1.t[-1], end_time_sec), y0=Tr0_2, 
                   phase_name='Phase 2', method=integration_method, 
                   events=[phase2_event], rtol=integration_rtol, atol=integration_atol)


#########################################
# Phase 3 - mantle + plate tectonics + passive outgassing
#########################################
FH2O3 = Tr0_2[1] / Mmantle
Tr0_3 = np.zeros(8, dtype=np.float64)
Tr0_3[0] = sol2.y[0, -1]
Tr0_3[1] = sol2.y[1, -1]
Tr0_3[2] = sol2.y[2, -1]
Tr0_3[3] = sol2.y[3, -1]

Tr0_3[4] = sol2.y[4, -1]  # mantle temp
Tr0_3[5] = sol2.y[6, -1]  # water in atmosphere
Tr0_3[6] = sol2.y[7, -1]  # O2 in atm
Tr0_3[7] = sol2.y[8, -1]  # surface temp

phase3_ode = partial(
    moODE_phase3, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, FH2O=FH2O3, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MSun, tides_on_flag=tides_on_flag)

phase_3_events = None  # Turn off phase 3 events for now. 
sol3 = monitor_ode(phase3_ode, t_span=(sol2.t[-1], end_time_sec), y0=Tr0_3, 
                   phase_name='Phase 3', method=integration_method, 
                   events=phase_3_events, rtol=integration_rtol, atol=integration_atol)


#==========================================================================
# Post-processing
#==========================================================================
results = postprocess_magma_ocean(sol1, sol2, sol3, Rp, Rc, Mmantle, gp,
                                  OLR, ASR, Teq, Xi, FeOt, t_flux,
                                  Lbol, tsat=1e9, a=a, Mp=Mp, LStar=LStar)

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

# --- CONVERT TIME BACK TO YEARS FOR PLOTTING & CSV ---
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

df_results = pd.DataFrame({
    'time_yr': t_tot_years,
    'Mantle_T': mantle_T_tot,
    'Surf_T': surf_T_tot,
    'PO2': PO2_tot,
    'Patm': Patm_tot,
})

df_results.to_csv('results.txt', sep='\t', index=False)


def plot_magma_ocean():
    # Temperature Plot
    fig_tmp, ax_tmp = plt.subplots(figsize=(8, 5))
    ax_tmp.plot(t_tot_years, mantle_T_tot, label='Mantle T', color='black')
    ax_tmp.plot(t_tot_years, surf_T_tot, label='Surface T', color='orange')
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
    ax_atm.plot(t_tot_years, PO2_tot, label='PO2', color='blue')
    ax_atm.plot(t_tot_years, Patm_tot, label='Patm', color='black')
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
    ax_orb.plot(t_tot_years, semia_tot / AU, color='red')  # Convert back to AU for plotting
    ax2_orb.plot(t_tot_years, eccen_tot, label='Eccentricity', color='blue')

    ax_orb.set_xlabel('Time [yr]')
    ax_orb.set_ylabel('Semi Major Axis [Au]', c='red')
    ax2_orb.set_ylabel('Eccentricity', c='blue')
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
    ax_spin.plot(t_tot_years, spin_planet_period, color='blue', label='Planet')
    ax_spin.plot(t_tot_years, spin_host_period, color='red', label='Star')

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

# Execute plot generator
plot_magma_ocean()
