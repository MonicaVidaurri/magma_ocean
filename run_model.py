import sys
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import root_scalar
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from scipy.integrate import solve_ivp as scisolve_ivp
try:
    from CyRK import pysolve_ivp
    USE_CYRK = True
except ImportError:
    USE_CYRK = False

USE_CYRK = False  # Override and only use scipy for now
if USE_CYRK:
    solve_ivp = pysolve_ivp
else:
    solve_ivp = scisolve_ivp

from types import SimpleNamespace
from ODEs.combined_ode import moODE_magma_ocean, moODE_solid, moODE_dry_solid, make_phase_events
from utils.postprocess import postprocess_magma_ocean
from utils.get_comp import get_comp
from utils.general_utils import merge_dicts
from utils.nondim_scales import StateScaler
from utils.mantle_grid import build_mantle_grid
from utils.get_melt_fractions import get_melt_fractions
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

# =====================================================================
# --- INITIALIZATION & TOML UNPACKING ---
# =====================================================================
with open("baseline_config.toml", "rb") as f:
    params = tomllib.load(f)

simulation_config = 'trappist1e'
with open(f'{simulation_config}.toml', "rb") as f:
    specific_params = tomllib.load(f)

params = merge_dicts(specific_params, params)
simulation_version = params['simulation']['version']
save_name = f'{simulation_config}_{simulation_version}'

constants         = params['constants']
star_params       = params['star']
planet_params     = params['planet']
orbit_params      = params['planet']['orbit']
simulation_params = params['simulation']
thermo_params     = params['planet']['thermodynamics']
comp_params       = params['planet']['oxide_composition']

integration_method = simulation_params['method']
integration_rtol   = simulation_params['rtol']
integration_atol   = simulation_params['atol']
# integration_rtol = np.array([
#     1e-4,   # 0: Semi-major axis (Needs extremely tight relative precision for long-term orbit)
#     1e-4,   # 1: Eccentricity
#     1e-4,   # 2: Star Spin Rate 
#     1e-4,   # 3: Planet Spin Rate 
#     1e-3,   # 4: Mantle Temp (5 significant figures is plenty for bulk thermodynamics)
#     1e-7,   # 5: Solid Radius (Allows the phase boundary to step faster)
#     1e-6,   # 6: Mass Water Solid (Slightly tighter to preserve strict mass conservation)
#     1e-6,   # 7: Mass Water MO/Atm 
#     1e-6,   # 8: Mass Oxygen MO/Atm 
#     1e-6,   # 9: Mass Oxygen Solid
#     1e-3    # 10: Surface Temp 
# ], dtype=np.float64)
integration_atol = np.array([
    1e-6,   # 0: Semi-major axis (Needs extremely tight relative precision for long-term orbit)
    1e-5,   # 1: Eccentricity
    1e-6,   # 2: Star Spin Rate 
    1e-6,   # 3: Planet Spin Rate 
    1e-8,   # 4: Mantle Temp (5 significant figures is plenty for bulk thermodynamics)
    1e-6,   # 5: Solid Radius (Allows the phase boundary to step faster)
    1e-6,   # 6: Mass Water Solid (Slightly tighter to preserve strict mass conservation)
    1e-6,   # 7: Mass Water MO/Atm 
    1e-6,   # 8: Mass Oxygen MO/Atm 
    1e-6,   # 9: Mass Oxygen Solid
    1e-6    # 10: Surface Temp 
], dtype=np.float64)

start_time_sec     = simulation_params['start_time_years'] * constants['seconds_per_year']
end_time_sec       = simulation_params['end_time_years'] * constants['seconds_per_year']
tides_on_flag      = simulation_params['tides_on']

MStar = star_params['mass_star_relative'] * constants['mass_sun']
LStar = star_params['lum_star_relative'] * constants['lum_sun']
host_radius = star_params['host_radius']
tsat_sec    = star_params['xuv']['tsat_years'] * constants['seconds_per_year']
stellar = pd.read_csv('data/solardata.txt', sep='\t')
olr_data = np.load('data/OLRdatab.npz')
Temp_K, P_Pa, OLR = olr_data['Temp_K'], olr_data['P_Pa'], olr_data['OLR']
Ts, Ps = np.meshgrid(Temp_K, P_Pa)
t_flux, Lbol = stellar['tbol'].values, stellar['Lbol'].values

Rp = planet_params['radius_planet_relative'] * constants['radius_earth']
Rc = planet_params['radius_core_relative'] * Rp
Mp = planet_params['mass_planet_relative'] * constants['mass_earth']
Mmantle = (1.0 - planet_params['core_mass_fraction']) * Mp
rho_mantle = Mmantle / ((4.0 / 3.0) * np.pi * (Rp**3 - Rc**3))
gp = constants['G'] * Mp / Rp**2
params['_grid'] = build_mantle_grid(Rp, Rc, gp, Mmantle, params)

initial_semi_major_axis = orbit_params['initial_semi_major_axis'] * constants['au']
Fstel = LStar / (4.0 * np.pi * initial_semi_major_axis**2)
ASR = (1.0 - planet_params['albedo']) * Fstel / 4.0
Teq = ((1.0 - planet_params['albedo']) * Fstel / 4.0 / constants['stefan_boltzmann']) ** 0.25

MH2O = planet_params['ocean_mass_multiplier'] * constants['mass_ocean_earth']
FH2O = MH2O / Mmantle
FeOt = comp_params['mass_frac_FeO_total']
Fe3_Fet = comp_params['ratio_Fe3_to_total_Fe']
Xi = get_comp(params) 

Tsol1 = thermo_params['solidus_slope_low_p'] * 1e-9 
Tsol2 = thermo_params['solidus_intercept_low_p']
Cp = thermo_params['specific_heat_mantle']
alpha_therm = thermo_params['thermal_expansion']
muO, muFeO1_5 = constants['molar_mass_O'], comp_params['molar_mass_FeO1_5']

# =====================================================================
# --- INITIAL CONDITIONS ---
# =====================================================================
Tr0 = np.zeros(11, dtype=np.float64)
Tr0[0] = initial_semi_major_axis
Tr0[1] = orbit_params['initial_eccentricity']
Tr0[2] = 2.0 * np.pi / (86400.0 * star_params['spin_days'])
Tr0[3] = planet_params['initial_spin_multiplier'] * semi_a2orbital_motion(initial_semi_major_axis, MStar, Mp)
Tr0[4] = planet_params['initial_mantle_temp']

# Find initial magma ocean depth based on temperature.
max_mantle_depth = Rp - Rc
def temp_difference(z):
    """ Finds the exact intersection of the adiabat and piecewise solidus. """
    T_ad = Tr0[4] + Tr0[4] * (alpha_therm * gp * z / Cp)
    P_gpa = (rho_mantle * gp * z) / 1e9
    
    T_sol_low = thermo_params['solidus_slope_low_p'] * P_gpa + thermo_params['solidus_intercept_low_p']
    T_sol_high = thermo_params['solidus_slope_high_p'] * P_gpa + thermo_params['solidus_intercept_high_p']
    T_sol = min(T_sol_low, T_sol_high)
    
    return T_ad - T_sol

try:
    res = root_scalar(temp_difference, bracket=[0.0, Rp - Rc], method='brentq')
    base_depth = res.root
except ValueError:
    # If the bracket fails, the mantle is hotter than the solidus at the CMB
    base_depth = Rp - Rc
# Original method:
# base_depth = (Tr0[4] - Tsol2) * Cp / (Tsol1 * rho_mantle * gp * Cp - alpha_therm * gp * Tr0[4])
# base_depth = min(base_depth, 900.0e3) 
Tr0[5] = max(Rp - base_depth, Rc)
print(f"Initial Magma Ocean Depth {base_depth/1e3:0.2f} km.")

Mmo0 = (4.0 / 3.0) * np.pi * rho_mantle * (Rp**3 - Tr0[5]**3)

Tr0[6] = MH2O - FH2O * Mmo0
Tr0[7] = FH2O * Mmo0 
muFeO = comp_params['molar_mass_FeO']
Tr0[8] = FeOt * Fe3_Fet * Mmo0 * (muO / 2.0 / muFeO)
Tr0[9] = FeOt * Fe3_Fet * (Mmantle - Mmo0) * (muO / 2.0 / muFeO)
Tr0[10] = Tr0[4] - 1.0

# =====================================================================
# --- INTEGRATION ---
# =====================================================================
CRASH_IF_SOL_FAILS = False
def monitor_ode(fun, t_span, y0, scaler=None, events=None, **kwargs):
    """ Wraps solve_ivp with dimensional translation and a progress bar. """
    
    if scaler:
        # Scale initial conditions and time into Dimensionless Space
        t_span_calc = (scaler.scale_time(t_span[0]), scaler.scale_time(t_span[1]))
        y0_calc = scaler.scale_state(y0)
        wrapped_fun = scaler.wrap_ode(fun)
        wrapped_events = [scaler.wrap_event(e) for e in events] if events else None
    else:
        t_span_calc = t_span
        y0_calc = y0
        wrapped_fun = fun
        wrapped_events = events

    # Setup progress bar
    t0, tf = t_span
    total_years = (tf - t0) / constants['seconds_per_year']
    
    with tqdm(total=total_years, unit="yr", desc="Simulating", unit_scale=True) as pbar:
        state = {'last_t_sec': t0}

        def track_progress(t_calc, y_calc):
            # Calculate dimensional time purely for the progress bar
            t_sec = scaler.unscale_time(t_calc) if scaler else t_calc
            dt = t_sec - state['last_t_sec']
            if dt > 0:
                pbar.update(dt / constants['seconds_per_year'])
                state['last_t_sec'] = t_sec
            return wrapped_fun(t_calc, y_calc)

        # Execute Solver
        sol = solve_ivp(track_progress, t_span_calc, y0_calc, events=wrapped_events, **kwargs)
        
        t_final_sec = scaler.unscale_time(sol.t[-1]) if scaler else sol.t[-1]
        if sol.status == 0 and t_final_sec >= tf:
            pbar.n = pbar.total
            pbar.refresh()

    if not sol.success and CRASH_IF_SOL_FAILS:
        t_fail = scaler.unscale_time(sol.t[-1]) if scaler else sol.t[-1]
        raise Exception(f"Integration failed at t={t_fail}: {sol.message}.")
    
    # Re-dimensionalize the solution object before returning it to the user
    if scaler:
        sol.t = scaler.unscale_time(sol.t)
        for i in range(len(sol.y)):
            sol.y[i, :] = sol.y[i, :] * scaler.y_scales[i]
            
        if sol.t_events:
            sol.t_events = [scaler.unscale_time(te) if te is not None else te for te in sol.t_events]
        if sol.y_events:
            for ev_idx in range(len(sol.y_events)):
                if sol.y_events[ev_idx] is not None and len(sol.y_events[ev_idx]) > 0:
                    for i in range(len(sol.y_events[ev_idx][0])):
                        sol.y_events[ev_idx][:, i] = sol.y_events[ev_idx][:, i] * scaler.y_scales[i]

    return sol

# Frozen keyword args shared by both phase ODEs (identical signatures)
_frozen = dict(
    Rp=Rp, Rc=Rc, Mmantle=Mmantle, Teq=Teq, rho_mantle=rho_mantle, g=gp,
    Ts=Ts, Ps=Ps, OLR=OLR, ASR=ASR, t_flux=t_flux, Lbol=Lbol, Xi=Xi, FeOt=FeOt,
    Temp_K=Temp_K, P_Pa=P_Pa, tsat=tsat_sec, Mp=Mp, LStar=LStar, Rh=host_radius,
    Mh=MStar, params=params, tides_on_flag=tides_on_flag
)
mo_ode    = partial(moODE_magma_ocean, **_frozen)
solid_ode = partial(moODE_solid,       **_frozen)
dry_ode   = partial(moODE_dry_solid,   **_frozen)

event_mo_ends, event_mo_starts, event_solid_dries = make_phase_events(params, Mmantle)

scaler = StateScaler(a0=initial_semi_major_axis, Rp=Rp, M_ocean=MH2O)

# Determine which phase we start in
mf_threshold  = params['planet']['convection']['melt_fraction_threshold']
dry_threshold = params['planet']['thermodynamics'].get('dry_solid_threshold', 1e-9)
_, meltfrac_init, _ = get_melt_fractions(Tr0[4], params['_grid'])
if meltfrac_init >= mf_threshold:
    phase = 'mo'
elif Tr0[6] / Mmantle < dry_threshold:
    phase = 'dry_solid'
else:
    phase = 'wet_solid'
print(f"Starting phase: {phase} (meltfrac_bulk = {meltfrac_init:.3f})")

# =====================================================================
# --- PHASE-CHAINING INTEGRATION LOOP ---
# =====================================================================
# Each solver segment handles exactly one phase (MO or solid).  A terminal
# event flips the phase flag and the loop restarts.  This lets each solver
# work on a continuous, smooth problem and naturally handles freeze-remelt.

_PHASE_LABELS = {'mo': 'Magma Ocean', 'wet_solid': 'Wet Solid', 'dry_solid': 'Dry Solid'}

MAX_PHASES = 20
events_encountered = dict()
t_list     = []
y_list     = []
current_y  = Tr0
t_current  = start_time_sec
run_success = True

total_years = (end_time_sec - start_time_sec) / constants['seconds_per_year']

for _phase_idx in range(MAX_PHASES):
    print(f"Working on phase {_phase_idx}:: {_PHASE_LABELS[phase]}")

    if t_current >= end_time_sec:
        break
    elif t_current != start_time_sec:
        events_encountered[t_current / constants['seconds_per_year']] = phase

    if phase == 'mo':
        ode_fun      = mo_ode
        phase_events = [event_mo_ends]
    elif phase == 'wet_solid':
        ode_fun      = solid_ode
        phase_events  = [event_mo_starts, event_solid_dries]
    else:  # 'dry_solid'
        ode_fun      = dry_ode
        phase_events = [event_mo_starts]

    segment = monitor_ode(
        ode_fun, (t_current, end_time_sec), current_y,
        scaler=scaler, events=phase_events,
        method=integration_method,
        rtol=integration_rtol,
        atol=integration_atol,
    )

    if not segment.success:
        print(f"\nIntegration failed at "
              f"t = {segment.t[-1] / constants['seconds_per_year']:.3e} yr: {segment.message}")
        run_success = False
        break

    t_current = segment.t[-1]
    current_y = segment.y[:, -1]
    t_list.append(segment.t)
    y_list.append(segment.y)

    prev_phase = phase
    if phase == 'mo':
        # After MO freezes out, check whether the solid mantle is still wet.
        # If dry_solid was reached before this MO episode, atmospheric water may
        # be depleted and solid water will be negligible — go straight to dry_solid
        # rather than wet_solid to avoid an unphysical wet-solid re-entry.
        if current_y[6] / Mmantle >= dry_threshold:
            phase = 'wet_solid'
        else:
            phase = 'dry_solid'
    elif phase == 'wet_solid':
        # t_events[0] = event_mo_starts, t_events[1] = event_solid_dries
        phase = 'mo' if len(segment.t_events[0]) > 0 else 'dry_solid'
    else:  # 'dry_solid'
        phase = 'mo'

    print(f"\n Completed Phase {_phase_idx}:: '{_PHASE_LABELS[prev_phase]}' "
          f"at t = {t_current / constants['seconds_per_year']:.3e} yr  →  {_PHASE_LABELS[phase]}")

# Concatenate all segments into a single solution-like object
t_all = np.concatenate(t_list)
y_all = np.concatenate(y_list, axis=1)
sol   = SimpleNamespace(
    t       = t_all,
    y       = y_all,
    success = run_success,
    message = f"Phase-chained integration ({len(t_list)} segment(s)).",
)

print("\nSimulation Complete:")
print(f"\t Success  = {sol.success}.")
print(f"\t Message  = {sol.message}.")
print(f"\t End time = {sol.t[-1] / constants['seconds_per_year']:0.3e} Years.")

# =====================================================================
# --- POST-PROCESSING ---
# =====================================================================
results = postprocess_magma_ocean(sol, Rp, Rc, Mmantle, gp, OLR, ASR,
                                  Teq, Xi, FeOt, t_flux, Lbol, tsat_sec, initial_semi_major_axis, Mp, LStar, params)

t_tot_years = results['t'] / constants['seconds_per_year']


orb_freq_array = np.empty_like(results['Tr'][0,:])
for i, a_ in enumerate(results['Tr'][0,:]):
    orb_freq_array[i] = semi_a2orbital_motion(a_, MStar, Mp)

# ---------------------------------------------------------------------------
# Shared state-vector layout (same for both phase ODEs)
#   [0]  semi_major_axis   [1]  eccentricity
#   [2]  spin_freq_host    [3]  spin_freq_planet
#   [4]  temp_mantle       [5]  radius_solid
#   [6]  mass_water_solid  [7]  mass_water_atm
#   [8]  mass_oxygen_atm   [9]  mass_O2_solid
#   [10] temp_surface
# ---------------------------------------------------------------------------

# Saving results can be slow and take up disk space. If doing quick adjustments turn off saving.
SAVE_DATA = True

if SAVE_DATA:
    df_results = pd.DataFrame({
        'time_yr': t_tot_years,
        'semi_major_axis': results['Tr'][0,:] / constants['au'],
        'orbital_freq': orb_freq_array,
        'eccentricity': results['Tr'][1,:],
        'spin_ratio_host': results['Tr'][2,:] / orb_freq_array,
        'spin_ratio_planet': results['Tr'][3,:] / orb_freq_array,
        'temp_mantle': results['Tr'][4,:],
        'radius_solid': results['Tr'][5,:] / 1e3,
        'mass_water_solid': results['Tr'][6,:] / MH2O,
        'mass_water_atm': results['Tr'][7,:] / MH2O,
        'mass_oxygen_atm': results['Tr'][8,:] / MH2O,
        'mass_O2_solid': results['Tr'][9,:] / MH2O,
        'temp_surface': results['Tr'][10,:],
        'PO2': results['PO2'],
        'Patm': results['Patm'],
        'tidal_shear': results['tidal_shear']/1e9,
        'tidal_visc': results['tidal_visc'],
        'tidal_scale': results['tidal_scale'] * 100,
        'Q_tid': results['Q_tid'],
        'Q_rad': results['Q_rad'],
        'meltfrac': results['meltfrac'] * 100
    })
    df_results.to_csv(f'{save_name}_results.txt', sep=',', index=False)

def plot_magma_ocean():
    Q_tidal = results['Q_tid']
    Q_radiogenic = results['Q_rad']
    Q_tidal[Q_tidal == 0.0] = np.nan
    Q_radiogenic[Q_radiogenic == 0.0] = np.nan

    def add_event_lines(axis):
        phase_ls = {
            'mo': '--',
            'wet_solid': ':',
            'dry_solid': '-.'
        }
        for phase_time, phase_type in events_encountered.items():
            axis.axvline(x=phase_time, ls=phase_ls[phase_type], c='k')

    # Temperature Plot
    fig_tmp, ax_tmp = plt.subplots(figsize=(8, 5))
    ax_tmp.plot(t_tot_years, results['Tr'][4,:], label='Mantle T', color='black')
    ax_tmp.plot(t_tot_years, results['Tr'][10,:], label='Surface T', color='orange')
    ax_tmp.set_xlabel('Time [yr]')
    ax_tmp.set_ylabel('Temperature [K]')
    ax_tmp.set_title('Mantle and Surface Temperature Evolution')
    ax_tmp.set_xscale('log')
    add_event_lines(ax_tmp)
    
    ax_heat = ax_tmp.twinx()
    ax_heat.plot(t_tot_years, Q_tidal/1e12, label='Tidal', color='red')
    ax_heat.plot(t_tot_years, Q_radiogenic/1e12, label='Radiogenic', color='green')
    ax_heat.set_yscale('log')
    ax_heat.set_ylabel('Heating [TW]')
    ax_heat.legend(loc='center right')
    ax_tmp.legend(loc='center left')
    ax_tmp.grid(True)
    fig_tmp.tight_layout()
    fig_tmp.savefig(f"{save_name}_temperature_heat.png")

    # Atmosphere Plot
    fig_atm, ax_atm = plt.subplots(figsize=(8, 5))
    ax_atm.plot(t_tot_years, results['PO2'], label='$P_{O2}$', color='blue')
    ax_atm.plot(t_tot_years, results['Patm'], label='$P_{H2O}$', color='black')
    ax_atm.set_xlabel('Time [yr]')
    ax_atm.set_ylabel('Pressure [Pa]')
    ax_atm.set_title('O2 and H2O Atm. Evolution')
    ax_atm.set_xscale('log')
    ax_atm.set_yscale('log')
    add_event_lines(ax_atm)
    ax_atm.legend()
    ax_atm.grid(True)
    fig_atm.tight_layout()
    fig_atm.savefig(f"{save_name}_atmosphere.png")

    # Orbit Plot
    fig_orb, ax_orb = plt.subplots(figsize=(8, 5))
    ax2_orb = ax_orb.twinx()
    ax_orb.plot(t_tot_years, results['Tr'][0,:] / constants['au'], color='red') 
    ax2_orb.plot(t_tot_years, results['Tr'][1,:], label='Eccentricity', color='blue')
    ax_orb.set_xlabel('Time [yr]')
    ax_orb.set_ylabel('Semi Major Axis [Au]', color='red')
    ax2_orb.set_ylabel('Eccentricity', color='blue')
    ax_orb.spines['left'].set_color('red')
    ax2_orb.spines['right'].set_color('blue')
    ax_orb.set_title('Orbital Evolution')
    ax_orb.set_xscale('log')
    add_event_lines(ax_orb)
    ax_orb.grid(True)
    ax_orb.tick_params(axis='y', colors='red')
    ax2_orb.tick_params(axis='y', colors='blue')
    fig_orb.tight_layout()
    fig_orb.savefig(f"{save_name}_orbit.png")

    # Spin Plot
    spin_host_frac = (results['Tr'][2,:]/orb_freq_array)
    spin_planet_frac = (results['Tr'][3,:]/orb_freq_array)
    fig_spin, ax_spin = plt.subplots(figsize=(8, 5))
    ax_spin.plot(t_tot_years, spin_planet_frac, color='blue', label='Planet')
    ax_spin.plot(t_tot_years, spin_host_frac, color='red', label='Star')
    ax_spin.set_xlabel('Time [yr]')
    ax_spin.set_ylabel('Spin / Orbital Motion')
    ax_spin.set_title('Spin Evolution')
    ax_spin.set_xscale('log')
    ax_spin.set_yscale('linear')
    add_event_lines(ax_orb)
    ax_spin.grid(True)
    ax_spin.legend()
    fig_spin.tight_layout()
    fig_spin.savefig(f"{save_name}_spin.png")

    # Tidal Susceptibility Plot
    fig_susp, ax_susp = plt.subplots(figsize=(11, 6)) 
    # Left=0.25 leaves 25% space on the left. Right=0.75 leaves 25% space on the right.
    fig_susp.subplots_adjust(left=0.2, right=0.8) 
    ax_susp.plot(t_tot_years, results['tidal_shear']/1e9, color='red')
    ax_susp.set_xlabel('Time [yr]')
    ax_susp.set_ylabel('Shear Modulus [GPa]', color='red')
    ax_susp.set_yscale('linear')
    ax_susp.set_xscale('log')
    ax_susp.tick_params(axis='y', labelcolor='red')
    ax_susp.spines['left'].set_color('red')
    ax_susp.grid(True)
    ax_visc = ax_susp.twinx()
    ax_visc.spines['left'].set_position(('axes', -0.15)) # Move outward by 20%
    ax_visc.spines['left'].set_visible(True)
    ax_visc.spines['right'].set_visible(False)
    ax_visc.yaxis.set_label_position('left')
    ax_visc.yaxis.set_ticks_position('left')
    ax_visc.plot(t_tot_years, results['tidal_visc'], color='blue')
    ax_visc.set_ylabel('Viscosity', color='blue')
    ax_visc.set_yscale('log')
    ax_visc.tick_params(axis='y', labelcolor='blue')
    ax_visc.spines['left'].set_color('blue')
    ax_tidal = ax_susp.twinx()
    ax_tidal.spines['right'].set_visible(True)
    ax_tidal.spines['left'].set_visible(False)
    ax_tidal.plot(t_tot_years, results['tidal_scale'] * 100, color='green')
    ax_tidal.set_ylabel('Tidal Scale [%]', color='green')
    ax_tidal.set_yscale('linear')
    ax_tidal.tick_params(axis='y', labelcolor='green')
    ax_tidal.spines['right'].set_color('green')
    ax_mf = ax_susp.twinx() 
    ax_mf.spines['right'].set_position(('axes', 1.15)) # Move outward by 20%
    ax_mf.spines['right'].set_visible(True)
    ax_mf.spines['left'].set_visible(False)
    ax_mf.plot(t_tot_years, 100 * results['meltfrac'], color='black') 
    ax_mf.set_ylabel('Solid Mantle Melt Fraction [%]', color='black')
    ax_mf.set_yscale('linear')
    ax_mf.tick_params(axis='y', labelcolor='black')
    ax_mf.spines['right'].set_color('black')
    
    ax_susp.set_title('Tidal Susceptibility Evolution')
    add_event_lines(ax_susp) 
    fig_susp.savefig(f"{save_name}_tidal_suscept.png")

    plt.show()

plot_magma_ocean()
