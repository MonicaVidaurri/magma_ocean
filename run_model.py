import numpy as np
from scipy.integrate import solve_ivp

from ODEs import moODEb, moODEb2, moODE3
from physics.mo_event import mo_event
from physics.mo_event2 import mo_event2
from physics.mo_event3 import mo_event3
from utils.postprocess import postprocess_magma_ocean
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

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
    'Rp': 1 , # units of REarth
    'Rc': 0.19 , # will be multiplied by REarth
    'Mp': 1 , # units of MEarth
    'Albedo': 0.2,
    'core_fraction': 0.16,
    'Earth_oceans': 2, # will be multiplied by 1 Earth ocean, 1.39e21 kg
    'mantle_temp': 3000, #[k]
    'FeOt': 0.08,
    'a': 1,  # will be multipled by Earth AU
    'tday': 1, # in Earth days
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
Xi = np.ones(10)  # placeholder for get_comp(1)
muO = 15.9994e-3
muFeO1_5 = 159.689e-3 / 2
muFeO = 71.845e-3

# Initial conditions for phase 1
Tr0 = np.zeros(7)
Tr0[0] = params['mantle_temp']  # mantle temperature
Tr0[1] = max(Rp - (Tr0[0] - Tsol2) * Cp / (Tsol1 * rho * gp * Cp - alpha * gp * Tr0[0]), Rc) #initial depth of base of magma ocean
Mmo0 = 4 / 3 * np.pi * rho * (Rp ** 3 - Tr0[1] ** 3) #initial mass of magma ocean
Tr0[2] = MH2O - FH2O * Mmo0 #initial abundance of H2O in solid phase
Tr0[3] = FH2O * Mmo0 #total initial water abundance
Tr0[4] = FeOt * Fe3_Fet * Mmo0 * (muO / 2 / muFeO1_5) #total initial mass of O in FeO(1.5)
Tr0[5] = FeOt * Fe3_Fet * (Mmantle - Mmo0) * (muO / 2 / muFeO1_5) #mass of O stored in solid FeO(1.5)
Tr0[6] = Tr0[0] - 1  #initial surface temperature is equal to initial potential temp

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
            print(f'{phase_name}: t = {t/1e6:.2f} Myr ({pct:.1f}%)')
            last_print = t
        return fun(t, y)

    sol = solve_ivp(wrapped_fun, t_span, y0, **kwargs)
    return sol

phase1_ode = partial(
    moODEb.moODEb, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho=rho, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, a=a, Mp=Mp, LStar=LStar)

sol1 = solve_ivp(phase1_ode, t_span=[1, 7.6e9], y0=Tr0, method='BDF', events=[mo_event])

sol1 = monitor_ode(phase1_ode, t_span=[1, 7.6e9], y0=Tr0, phase_name='Phase 1', method='BDF', events=[mo_event])
print("Stage 1 Success:", sol1.success)

#########################################
# Phase 2 - starting to solidify......
#########################################
Tr0_2 = np.zeros(5)
Tr0_2[0] = sol1.y[0, -1]  # mantle temp
Tr0_2[1] = sol1.y[2, -1] + sol1.y[3, -1] - sol1.y[6, -1] * 4 * np.pi * Rp ** 2 / gp
Tr0_2[2] = sol1.y[6, -1] * 4 * np.pi * Rp ** 2 / gp  # water in atmosphere
Tr0_2[3] = sol1.y[4, -1] * gp / (4 * np.pi * Rp ** 2)  # O2 in atm
Tr0_2[4] = sol1.y[6, -1]  # surface temp

phase2_ode = partial(
    moODEb2.moODEb2, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho=rho, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, a=a, Mp=Mp, LStar=LStar)

Tr0_2 = Tr0_2.copy()
Tr0_2[1] += 1e-12 * Mmantle
sol2 = solve_ivp(phase2_ode, t_span=[sol1.t[-1], 7.6e9], y0=Tr0_2, method='BDF', events=[mo_event2])

sol2 = monitor_ode(phase2_ode, t_span=[sol1.t[-1], 7.6e9], y0=Tr0_2, phase_name='Phase 2', method='BDF', events=[mo_event2])
print("Stage 2 Success:", sol1.success)

#########################################
# Phase 3 - mantle + plate tectonics + passive outgassing
#########################################
FH2O3 = Tr0_2[1] / Mmantle
Tr0_3 = np.zeros(4)
Tr0_3[0] = sol2.y[0, -1]  # mantle temp
Tr0_3[1] = sol2.y[2, -1]  # water in atmosphere
Tr0_3[2] = sol2.y[3, -1]  # O2 in atm
Tr0_3[3] = sol2.y[4, -1]  # surface temp

phase3_ode = partial(
    moODE3.moODE3, Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho=rho, g=gp,
    Ts=Temp_K, Ps=P_Pa, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol,
    Xi=Xi, FeOt=FeOt, Temp_K=Temp_K, P_Pa=P_Pa, tsat=1e9, FH2O=FH2O3, a=a, Mp=Mp, LStar=LStar)

sol3 = solve_ivp(phase3_ode, t_span=[sol2.t[-1], 7.6e9], y0=Tr0_3, method='BDF', events=[mo_event3])

sol3 = monitor_ode(phase3_ode, t_span=[sol2.t[-1], 7.6e9], y0=Tr0_3, phase_name='Phase 3', method='BDF', events=[mo_event3])
print("Stage 3 Success:", sol1.success)
#==========================================================================


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
mantle_T_tot = np.concatenate([sol1.y[0,:], sol2.y[0,:], sol3.y[0,:]])
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
    plt.figure(figsize=(8, 5))
    plt.plot(t_tot, mantle_T_tot, label='Mantle T, Python', color='black')
    plt.plot(t_tot, surf_T_tot, label='Surface T, Python', color='orange')
    plt.xlabel('Time [yr]')
    plt.ylabel('Temperature [K]')
    plt.title('Mantle and Surface Temperature Evolution')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#    #Atmosphere Plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_tot, PO2_tot, label='PO2', color='blue')
    plt.plot(t_tot, Patm_tot, label='Patm', color='black')
    plt.xlabel('Time [yr]')
    plt.ylabel('Pressure [Pa]')
    plt.title('PO2 and Patm Evolution')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_magma_ocean(results)
