import numpy as np


import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp as scisolve_ivp
from CyRK import pysolve_ivp

from ODEs import moEvent_phase1, moEvent_phase2, moEvent_phase3, moODE_phase1, moODE_phase2, moODE_phase3
from utils.postprocess import postprocess_magma_ocean
from utils.get_comp import get_comp

from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

USE_CYRK = False
if USE_CYRK:
    solve_ivp = pysolve_ivp
else:
    solve_ivp = scisolve_ivp

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
# Constants
G = 6.6743e-11
sigma = 5.67e-8
# Star properties
MSun = 1.989e30 # [kg]
LSun = 3.846e26 # [W]
MStar = params['MStar'] * MSun  # stellar mass [Msun]
LStar = params['LStar'] * LSun  # stellar luminosity [Lsun]

# Need the host star radius to calculate change in spin rate.
# TODO: Connect to params dict! Hardcoded for now
host_radius = 1.0 * 6.957e8

# Integration parameters
integration_method = 'RK45' # 'BDF'
integration_rtol = 1.0e-5
integration_atol = 1.0e-8

start_time = 1.0
end_time = 8.0e9 # 7.6e9

# Read from star and OLR files
olr_data = np.load('data/OLRdatab.npz')
Temp_K = olr_data['Temp_K']
P_Pa = olr_data['P_Pa']
OLR = olr_data['OLR']
Ts, Ps = np.meshgrid(Temp_K, P_Pa)
t_flux = stellar['tbol']
Lbol = stellar['Lbol']
# Planet properties
MEarth = 5.97e24 # [kg]
REarth =  6371e3 # [m]
AU = 1.496e11 # [m]
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
rho = Mmantle / (4 / 3 * np.pi * (Rp ** 3 - Rc ** 3))
gp = G * Mp / Rp ** 2
# Water content
Earth_ocean = 1.39e21
MH2O = params['Earth_oceans'] * Earth_ocean
FH2O = MH2O / Mmantle
FeOt = 0.08
Fe3_Fet = 1e-12
# Solidus parameters
Tsol1 = 26.53e-9
Tsol2 = 1373
Cp = 1.2e3
alpha = 2e-5
# Oxide fraction for magma
Xi = get_comp(1) # Mole fraction of the oxides, 1 = BSE, 2 = BSMars;  only model=1 is implemented right now
muO = 15.9994e-3
muFeO1_5 = 159.689e-3 / 2
muFeO = 71.845e-3

# Tide flag
tides_on_flag = False

# Initial orbital and spin conditions
# TODO: add these to params. Most are pulled from trappist1
# TODO: explore parameter space
initial_semi_a = a
initial_orbital_freq = semi_a2orbital_motion(initial_semi_a, MStar, Mp)
initial_eccentricity = 0.1
initial_spin_host = 2 * np.pi / (86400.0 * 3.295)
initial_spin_planet = 10 * initial_orbital_freq  # Start it out spinning much faster than orbital motion

# Initial conditions for phase 1
Tr0 = np.zeros(11, dtype=np.float64)
Tr0[0] = initial_semi_a
Tr0[1] = initial_eccentricity
Tr0[2] = initial_spin_host
Tr0[3] = initial_spin_planet

Tr0[4] = params['mantle_temp']  # mantle temperature
Tr0[5] = max(Rp - (Tr0[4] - Tsol2) * Cp / (Tsol1 * rho * gp * Cp - alpha * gp * Tr0[4]), Rc) #initial depth of base of magma ocean
Mmo0 = 4 / 3 * np.pi * rho * (Rp**3 - Tr0[5]**3) #initial mass of magma ocean
Tr0[6] = MH2O - FH2O * Mmo0 #initial abundance of H2O in solid phase
Tr0[7] = FH2O * Mmo0 #total initial water abundance
Tr0[8] = FeOt * Fe3_Fet * Mmo0 * (muO / 2 / muFeO1_5) #total initial mass of O in FeO(1.5)
Tr0[9] = FeOt * Fe3_Fet * (Mmantle - Mmo0) * (muO / 2 / muFeO1_5) #mass of O stored in solid FeO(1.5)
Tr0[10] = Tr0[4] - 1  #initial surface temperature is equal to initial potential temp

#########################################
# Phase 1 - bigass ball of magma baby!!!!
#########################################
def monitor_ode(fun, t_span, y0, phase_name, **kwargs):
    '''Print phase and time in Myr as the model runs.'''
    t0, tf = t_span
    last_print = t0  # keep track to avoid flooding terminal
    print_interval = (tf - t0) / 50  # print ~50 times per phase

    def wrapped_fun(t, y):
        nonlocal last_print
        # Print only every print_interval
        if t - last_print >= print_interval or t == t0:
            pct = 100 * (t - t0) / (tf - t0)
            print(f'{phase_name}: t = {t/1e6:.2f} Myr ({pct:.1f}%); y = {y}')
            last_print = t
        return fun(t, y)

    sol = solve_ivp(wrapped_fun, t_span, y0, **kwargs)
    return sol

phase1_ode = partial(
    moODE_phase1, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho=rho, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MSun, tides_on_flag=tides_on_flag)

sol1 = monitor_ode(phase1_ode, t_span=(start_time, end_time), y0=Tr0, phase_name='Phase 1', method=integration_method, events=[moEvent_phase1], rtol=integration_rtol, atol=integration_atol)


#########################################
# Phase 2 - starting to solidify......
#########################################
Tr0_2 = np.zeros(9)
Tr0_2[0] = sol1.y[0, -1]
Tr0_2[1] = sol1.y[1, -1]
Tr0_2[2] = sol1.y[2, -1]
Tr0_2[3] = sol1.y[3, -1]

Tr0_2[4] = sol1.y[4, -1]  # mantle temp

Tr0_2[5] = sol1.y[6, -1] + sol1.y[7, -1] - sol1.y[10, -1] * 4 * np.pi * Rp ** 2 / gp
Tr0_2[6] = sol1.y[10, -1] * 4 * np.pi * Rp ** 2 / gp  # water in atmosphere
Tr0_2[7] = sol1.y[8, -1] * gp / (4 * np.pi * Rp ** 2)  # O2 in atm
Tr0_2[8] = sol1.y[10, -1]  # surface temp

phase2_ode = partial(
    moODE_phase2, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho=rho, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MSun, tides_on_flag=tides_on_flag)

Tr0_2 = Tr0_2.copy()
Tr0_2[5] += 1e-12 * Mmantle

phase2_event = partial(moEvent_phase2, core_mass_fraction=core_fraction, planet_mass=Mp)

sol2 = monitor_ode(phase2_ode, t_span=(sol1.t[-1], end_time), y0=Tr0_2, phase_name='Phase 2', method=integration_method, events=[phase2_event], rtol=integration_rtol, atol=integration_atol)

#########################################
# Phase 3 - mantle + plate tectonics + passive outgassing
#########################################
FH2O3 = Tr0_2[1] / Mmantle
Tr0_3 = np.zeros(8)
Tr0_3[0] = sol2.y[0, -1]
Tr0_3[1] = sol2.y[1, -1]
Tr0_3[2] = sol2.y[2, -1]
Tr0_3[3] = sol2.y[3, -1]

Tr0_3[4] = sol2.y[4, -1]  # mantle temp
Tr0_3[5] = sol2.y[6, -1]  # water in atmosphere
Tr0_3[6] = sol2.y[7, -1]  # O2 in atm
Tr0_3[7] = sol2.y[8, -1]  # surface temp

phase3_ode = partial(
    moODE_phase3, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho=rho, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, FH2O=FH2O3, Mp=Mp, LStar=LStar,
    Rh=host_radius, Mh=MSun, tides_on_flag=tides_on_flag)

sol3 = monitor_ode(phase3_ode, t_span=(sol2.t[-1], end_time), y0=Tr0_3, phase_name='Phase 3', method=integration_method, events=[moEvent_phase3], rtol=integration_rtol, atol=integration_atol)
#==========================================================================


# JPR Left off 2026-02-17

# Post-processing
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
results = postprocess_magma_ocean(sol1, sol2, sol3, Rp, Rc, Mmantle, gp,
                                  OLR, ASR, Teq, Xi, FeOt, t_flux,
                                  Lbol, tsat=1e9, a=a, Mp=Mp, LStar=LStar)
# First, pad arrays so that they're all the same size and the code doesn't bitch at you
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

# Now you have what you need to gather all your data:
Tr_all = np.concatenate([Tr1_padded, Tr2_padded, Tr3_padded], axis=1)
t_tot = np.concatenate([sol1.t, sol2.t, sol3.t])

semia_tot = np.concatenate([sol1.y[0,:], sol2.y[0,:], sol3.y[0,:]])
eccen_tot = np.concatenate([sol1.y[1,:], sol2.y[1,:], sol3.y[1,:]])
spin_host_tot = np.concatenate([sol1.y[2,:], sol2.y[2,:], sol3.y[2,:]])
spin_planet_tot = np.concatenate([sol1.y[3,:], sol2.y[3,:], sol3.y[3,:]])


mantle_T_tot = np.concatenate([sol1.y[4,:], sol2.y[4,:], sol3.y[4,:]])
surf_T_tot = np.concatenate([sol1.y[-1,:], sol2.y[-1,:], sol3.y[-1,:]])
PO2_tot = np.concatenate([results['phase1']['PO2'],results['phase2']['PO2'],results['phase3']['PO2']])
Patm_tot = np.concatenate([results['phase1']['Patm'],results['phase2']['Patm'],results['phase3']['Patm']])

df_results = pd.DataFrame({
    'time': t_tot,
    'Mantle_T': mantle_T_tot,
    'Surf_T': surf_T_tot,
    'PO2': PO2_tot,
    'Patm': Patm_tot,
    # 'Mmo': Mmo,
    # 'Wmo': Wmo,
    # 'meltfrac': meltfrac,
    # 'FH2O_arr': FH2O_arr,
    # 'kH2O_arr': kH2O_arr,
    # 'flux_arr': flux_arr,
    # 'FFeO1_5': FFeO1_5,
    # 'nFeO1_5': nFeO1_5,
    # 'nFeOt': nFeOt,
    # 'fo2': fo2,
    # 'phi_H': phi_H,
    # 'phi_O': phi_O
})

pd.DataFrame.to_csv(df_results,'results.txt', sep='\t', index=False)

def plot_magma_ocean(results):
#     #Temperature Plot
    fig_tmp, ax_tmp = plt.subplots(figsize=(8, 5))
    ax_tmp.plot(t_tot, mantle_T_tot, label='Mantle T, Python', color='black')
    ax_tmp.plot(t_tot, surf_T_tot, label='Surface T, Python', color='orange')
    ax_tmp.set_xlabel('Time [yr]')
    ax_tmp.set_ylabel('Temperature [K]')
    ax_tmp.set_title('Mantle and Surface Temperature Evolution')
    ax_tmp.set_xscale('log')
    ax_tmp.set_yscale('log')
    ax_tmp.legend()
    ax_tmp.grid(True)
    fig_tmp.tight_layout()

#    #Atmosphere Plot
    fig_atm, ax_atm = plt.subplots(figsize=(8, 5))
    ax_atm.plot(t_tot, PO2_tot, label='PO2', color='blue')
    ax_atm.plot(t_tot, Patm_tot, label='Patm', color='black')
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
    ax_orb.plot(t_tot, semia_tot/1.496e+11, color='blue')
    ax2_orb.plot(t_tot, eccen_tot, label='Eccentricity', color='red')

    ax_orb.set_xlabel('Time [yr]')
    ax_orb.set_ylabel('Semi Major Axis [Au]')
    ax2_orb.set_ylabel('Eccentricity')
    ax_orb.set_title('Orbital Evolution')
    ax_orb.set_xscale('log')
    ax_orb.set_yscale('log')
    ax2_orb.set_yscale('log')
    ax_orb.grid(True)
    fig_orb.tight_layout()

    # Spin Plot
    spin_host_period = (2 * np.pi / spin_host_tot)/86400.0
    spin_planet_period = (2 * np.pi / spin_planet_tot)/86400.0
    fig_spin, ax_spin = plt.subplots(figsize=(8, 5))
    ax_spin.plot(t_tot, spin_planet_period, color='blue', label='Planet')
    ax_spin.plot(t_tot, spin_host_period, color='red', label='Star')

    ax_spin.set_xlabel('Time [yr]')
    ax_spin.set_ylabel('Spin Period [days]')
    ax_spin.set_title('Spin Evolution')
    ax_spin.set_xscale('log')
    ax_spin.set_yscale('log')
    ax_spin.grid(True)
    fig_spin.tight_layout()
    
    # Show all plots
    plt.show()


plot_magma_ocean(results)

