"""
Run the magma ocean model for one planet configuration.

Command line::

    python run_model.py trappist1e.toml
    python run_model.py trappist1e.toml --tides on --end-time 1e8 --output-dir output/test
    python run_model.py trappist1e.toml --set planet.orbit.initial_eccentricity=0.05 --no-plot

From Python::

    from run_model import run
    results = run('trappist1e.toml', overrides={'simulation.tides_on': True}, output_dir='output/test')
"""
import argparse
import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ODEs.combined_ode import MagmaOceanModel
from utils.config import load_config
from utils.logger import get_logger, setup_logging
from utils.nondim_scales import StateScaler
from utils.plotting import plot_run
from utils.postprocess import postprocess_magma_ocean

log = get_logger('run_model')


# ======================================================================================================================
# Integration
# ======================================================================================================================
def absolute_tolerances(model, scaler):
    """
    Per-component absolute tolerances in scaled units.

    Every component uses ``simulation.atol`` except the fluid water, whose tolerance must resolve the escape taper
    (W_taper, the water column of ``taper_pressure``). A coarser tolerance lets the solver step back and forth across
    the taper's linear zone once the water is gone, which stalls the integration.
    """
    params    = model.params
    tolerance = np.full(scaler.y_scales.size, params['simulation']['atol'], dtype=np.float64)
    water_taper = (params['planet']['atmosphere']['escape']['taper_pressure'] * model.planet.surface_area
                   / model.planet.gravity)
    tolerance[6] = min(tolerance[6], 0.1 * water_taper / scaler.y_scales[6])
    return tolerance


def finite_difference_jacobian(rhs, atol):
    """
    Forward-difference Jacobian with a bounded step, for the implicit solvers.

    scipy's default estimator (`num_jac`) grows a column's step tenfold whenever the column's effect looks negligible,
    with no upper limit. The solid inventories affect the rates only weakly, so over a long run their steps overflowed
    and the solver passed an infinite state to the right-hand side. Here each step is sqrt(machine epsilon) times the
    larger of the component's magnitude and its absolute tolerance.

    Parameters
    ----------
    rhs : callable
        ``rhs(t, y)`` in the solver's (scaled) units.
    atol : ndarray
        Per-component absolute tolerances (scaled units); they set the smallest step.

    Returns
    -------
    callable
        ``jacobian(t, y) -> ndarray, shape (n, n)``.
    """
    relative_step = np.sqrt(np.finfo(np.float64).eps)
    step_floor = np.asarray(atol, dtype=np.float64)

    def jacobian(t, y):
        rates = rhs(t, y)
        matrix = np.empty((y.size, y.size), dtype=np.float64)
        for column in range(y.size):
            perturbed = y.copy()
            perturbed[column] += relative_step * max(abs(y[column]), step_floor[column])
            step = perturbed[column] - y[column]  # The step actually taken after rounding
            matrix[:, column] = (rhs(t, perturbed) - rates) / step
        return matrix

    return jacobian


def integrate(model, show_progress=True):
    """
    Integrate the model from the configured start to end time.

    Parameters
    ----------
    model : MagmaOceanModel
    show_progress : bool, optional
        Show a tqdm progress bar in simulated years.

    Returns
    -------
    t_sec : ndarray
        Accepted solver times (s).
    states : ndarray, shape (num_states, n)
        States at those times (SI units).
    success : bool
    message : str
    """
    params         = model.params
    sim            = params['simulation']
    seconds_per_yr = params['constants']['seconds_per_year']
    start_sec      = sim['start_time_years'] * seconds_per_yr
    end_sec        = sim['end_time_years'] * seconds_per_yr

    initial_state = model.initial_state()
    # Volatile reservoirs are scaled by the initial water; a dry planet falls back to one Earth ocean so the scaling
    # (and the absolute tolerance it implies) stays finite.
    volatile_scale = model.planet.mass_water_initial or params['constants']['mass_ocean_earth']
    scaler         = StateScaler(a0=initial_state[0], M_ocean=volatile_scale)
    scaled_rhs     = scaler.wrap_ode(model.rhs)

    total_years = (end_sec - start_sec) / seconds_per_yr
    with tqdm(total=total_years, unit='yr', desc='Simulating', unit_scale=True, disable=not show_progress) as pbar:
        def tracked_rhs(t_scaled, y_scaled):
            # Set the bar position directly (summing increments drifts past the total in floating point). Trial steps
            # can probe past the end time; the bar stops there.
            elapsed_years = (min(scaler.unscale_time(t_scaled), end_sec) - start_sec) / seconds_per_yr
            if elapsed_years > pbar.n:
                pbar.n = elapsed_years
                pbar.update(0)
            return scaled_rhs(t_scaled, y_scaled)

        atol = absolute_tolerances(model, scaler)
        implicit = sim['method'] in ('BDF', 'Radau', 'LSODA')
        solution = solve_ivp(
            tracked_rhs, (scaler.scale_time(start_sec), scaler.scale_time(end_sec)),
            scaler.scale_state(initial_state), method=sim['method'], rtol=sim['rtol'], atol=atol,
            **({'jac': finite_difference_jacobian(scaled_rhs, atol)} if implicit else {}))

    t_sec  = scaler.unscale_time(solution.t)
    states = scaler.unscale_state(solution.y)
    if solution.success:
        log.info(f"Integration finished at {t_sec[-1] / seconds_per_yr:0.3e} yr ({solution.t.size} steps).")
    else:
        log.error(f"Integration failed at {t_sec[-1] / seconds_per_yr:0.3e} yr: {solution.message}")
    return t_sec, states, bool(solution.success), str(solution.message)


# ======================================================================================================================
# Output
# ======================================================================================================================
LABELED_KEYS = ('simulation.tides_on', 'simulation.end_time_years', 'planet.orbit.initial_eccentricity',
                'planet.initial_spin_multiplier')


def run_label(params):
    """
    File-name label built from the parameters that define a run.

    Overrides other than those spelled out in the label add a short digest, so runs with different overrides do not
    overwrite each other's files.
    """
    sim   = params['simulation']
    orbit = params['planet']['orbit']
    tides = 'on' if sim['tides_on'] else 'off'
    label = (f"{params['_meta']['config_name']}_v{sim['version']}_tides{tides}"
             f"_e{orbit['initial_eccentricity']:.4g}_sp{params['planet']['initial_spin_multiplier']:.4g}"
             f"_t{sim['end_time_years']:.0e}")
    other = {key: value for key, value in params['_meta']['overrides'].items() if key not in LABELED_KEYS}
    if other:
        digest = hashlib.sha1(json.dumps(other, sort_keys=True, default=str).encode()).hexdigest()[:8]
        label += f'_o{digest}'
    return label


def results_table(results, params, water_scale):
    """
    Output table. The first columns keep the names and units of the pre-2026-09 output so older plotting scripts
    (auto_plot.py, auto_plot_v2.py) still work; new columns follow.
    """
    constants = params['constants']
    orbital_freq = results['orbital_freq']
    return pd.DataFrame({
        'time_yr': results['t'] / constants['seconds_per_year'],
        'semi_major_axis': results['semi_major_axis'] / constants['au'],
        'orbital_freq': orbital_freq,
        'eccentricity': results['eccentricity'],
        'spin_ratio_host': results['spin_freq_host'] / orbital_freq,
        'spin_ratio_planet': results['spin_freq_planet'] / orbital_freq,
        'temp_mantle': results['temp_mantle'],
        'radius_solid': results['radius_solid'] / 1e3,
        'mass_water_solid': results['mass_water_solid'] / water_scale,
        'mass_water_atm': results['mass_water_fluid'] / water_scale,
        'mass_oxygen_atm': results['mass_oxygen_fluid'] / water_scale,
        'mass_O2_solid': results['mass_oxygen_solid'] / water_scale,
        'temp_surface': results['temp_surface'],
        'PO2': results['PO2'],
        'Patm': results['Patm'],
        'tidal_shear': results['tidal_shear'] / 1e9,
        'tidal_visc': results['tidal_visc'],
        'tidal_scale': results['tidal_scale'] * 100,
        'Q_tid': results['Q_tid'],
        'Q_rad': results['Q_rad'],
        'meltfrac': results['meltfrac'] * 100,
        # --- Added 2026-09 ---
        'Q_surface': results['Q_surface'],
        'temp_equilibrium': results['temp_equilibrium'],
        'radius_rheological': results['radius_rheological'] / 1e3,
        'meltfrac_surface': results['meltfrac_surface'] * 100,
        'meltfrac_melt_region': results['meltfrac_melt_region'] * 100,
        'meltfrac_solid_layer': results['meltfrac_solid_layer'] * 100,
        'mass_water_dissolved': results['mass_water_dissolved'] / water_scale,
        'mass_water_vapor': results['mass_water_vapor'] / water_scale,
        'mass_water_ocean': results['mass_water_ocean'] / water_scale,
        'degassing_rate': results['degassing_rate'],
        'water_loss_rate': results['water_loss_rate'],
    })


# ======================================================================================================================
# Public Entry Point
# ======================================================================================================================
def run(config_path, overrides=None, output_dir='output', save=True, plot=True, show_progress=True, log_file=None,
        log_level='INFO'):
    """
    Build, integrate, and post-process one model run.

    Parameters
    ----------
    config_path : str or path-like
        Planet TOML (merged on top of baseline_config.toml).
    overrides : dict, optional
        ``{dotted_key: value}`` configuration overrides, e.g. ``{'simulation.tides_on': True}``.
    output_dir : str or path-like, optional
        Directory for the output table, summary, and figures.
    save : bool, optional
        Write the tab-delimited output table and the JSON summary.
    plot : bool, optional
        Write the standard figures.
    show_progress : bool, optional
        Show a progress bar.
    log_file : str or path-like, optional
        Also write log messages to this file.
    log_level : str, optional
        Logging level.

    Returns
    -------
    results : dict
        Post-processed results (see `utils.postprocess.postprocess_magma_ocean`) plus ``'success'``, ``'message'``,
        ``'label'``, ``'params'``, and ``'output_paths'``.
    """
    setup_logging(level=log_level, log_file=log_file)
    params = load_config(config_path, overrides=overrides)
    label  = run_label(params)
    log.info(f"Run '{label}' (config {params['_meta']['config_path']}).")

    wall_start = time.perf_counter()
    model = MagmaOceanModel(params)
    t_sec, states, success, message = integrate(model, show_progress=show_progress)
    results = postprocess_magma_ocean(model, t_sec, states)
    results.update(success=success, message=message, label=label, params=params, output_paths=[])
    results['summary'].update(success=success, message=message, label=label,
                              wall_time_s=time.perf_counter() - wall_start)

    water_scale = model.planet.mass_water_initial
    if save or plot:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / label
    if save:
        table_path = f'{prefix}_output.txt'
        results_table(results, params, water_scale).to_csv(table_path, sep='\t', index=False)
        summary_path = f'{prefix}_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            json.dump({'summary': results['summary'], 'meta': params['_meta']}, summary_file, indent=2)
        results['output_paths'] += [table_path, summary_path]
    if plot:
        results['output_paths'] += plot_run(results, str(prefix), water_scale)

    summary = results['summary']
    log.info(f"Success = {success}. Liquid layer first ends at {summary['time_liquid_layer_first_end_years']:0.3e} "
             f"yr; final PO2 = {summary['PO2_final_pa']:0.3e} Pa; water lost = "
             f"{100 * summary['water_lost_fraction_final']:0.1f}%.")
    return results


# ======================================================================================================================
# Command Line
# ======================================================================================================================
def _parse_override(text):
    """Parse ``key=value`` where value is a TOML literal (numbers, booleans, quoted strings)."""
    if '=' not in text:
        raise argparse.ArgumentTypeError(f"Override '{text}' must look like key=value.")
    key, value_text = text.split('=', 1)
    try:
        value = tomllib.loads(f'value = {value_text}')['value']
    except tomllib.TOMLDecodeError:
        value = value_text
    return key.strip(), value


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the magma ocean model for one planet configuration.')
    parser.add_argument('config', help='Planet TOML file (merged on top of baseline_config.toml).')
    parser.add_argument('--tides', choices=('on', 'off'), help='Override simulation.tides_on.')
    parser.add_argument('--end-time', type=float, help='Override simulation.end_time_years.')
    parser.add_argument('--set', dest='overrides', action='append', type=_parse_override, default=[],
                        metavar='KEY=VALUE', help='Override any configuration value (repeatable).')
    parser.add_argument('--output-dir', default='output', help='Directory for outputs (default: output).')
    parser.add_argument('--no-plot', action='store_true', help='Skip figures.')
    parser.add_argument('--no-save', action='store_true', help='Skip the output table and summary.')
    parser.add_argument('--log-file', help='Also write the log to this file.')
    parser.add_argument('--log-level', default='INFO', help='Logging level (default: INFO).')
    args = parser.parse_args(argv)

    overrides = dict(args.overrides)
    if args.tides is not None:
        overrides['simulation.tides_on'] = args.tides == 'on'
    if args.end_time is not None:
        overrides['simulation.end_time_years'] = args.end_time

    results = run(args.config, overrides=overrides, output_dir=args.output_dir, save=not args.no_save,
                  plot=not args.no_plot, log_file=args.log_file, log_level=args.log_level)
    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
