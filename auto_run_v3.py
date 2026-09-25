"""
Sweep initial orbital states of one planet configuration with tides on and off.

Every case is a normal `run_model.run` call with configuration overrides, so the TOML files are never edited. Cases
run in parallel, each in its own directory with its own output table, JSON summary, and log. Finished cases are
skipped on re-runs (use --force to redo them). A combined table, ``sweep_summary.tsv``, pairs every tides-on case with
its tides-off baseline for auto_plot_v3.py.

Without tides the orbit and spins never change and do not affect the planet, so tides-off runs ignore the keys in
TIDES_ONLY_KEYS: one tides-off run is made per combination of the remaining sweep keys.

Usage::

    python auto_run_v3.py
    python auto_run_v3.py --config trappist1e.toml --workers 8 --end-time 1e9
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_model import run
from utils.logger import get_logger, setup_logging

# ======================================================================================================================
# Sweep Definition (edit here)
# ======================================================================================================================
CONFIG = 'trappist1e.toml'

# {dotted configuration key: list of values}. Every combination is run.
SWEEP = {
    'planet.orbit.initial_eccentricity': [0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
    'planet.initial_spin_multiplier': [1.0, 3.0, 10.0],
}

# Applied to every case, e.g. {'simulation.end_time_years': 1.0e9, 'planet.atmosphere.escape.xuv_model': 1}.
FIXED_OVERRIDES = {}

# Keys that only matter when tides are on (see module docstring).
TIDES_ONLY_KEYS = ('planet.orbit.initial_eccentricity', 'planet.initial_spin_multiplier', 'star.spin_days')

DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'results'

log = get_logger('auto_run_v3')


# ======================================================================================================================
# Case Construction
# ======================================================================================================================
def _short_key(dotted_key):
    """Last component of a dotted configuration key, used in case names."""
    return dotted_key.split('.')[-1]


def _case_id(tides_on, values):
    """Directory-safe case name from the tides flag and the case's sweep values."""
    parts = [f"{_short_key(key)}={value:.4g}" if isinstance(value, float) else f"{_short_key(key)}={value}"
             for key, value in values.items()]
    return f"tides{'on' if tides_on else 'off'}" + ('__' + '__'.join(parts) if parts else '')


def build_cases(sweep, fixed_overrides):
    """
    All cases of the sweep.

    Returns
    -------
    list of dict
        Each with ``case_id``, ``tides_on``, ``baseline_id`` (the matching tides-off case), ``sweep_values``, and the
        full ``overrides`` for `run_model.run`.
    """
    keys = list(sweep)
    shared_keys = [key for key in keys if key not in TIDES_ONLY_KEYS]
    cases = {}
    for combination in itertools.product(*(sweep[key] for key in keys)):
        values = dict(zip(keys, combination))
        shared = {key: values[key] for key in shared_keys}
        baseline_id = _case_id(False, shared)
        for tides_on in (True, False):
            case_values = values if tides_on else shared
            case_id = _case_id(tides_on, case_values)
            if case_id in cases:
                continue
            overrides = dict(fixed_overrides)
            overrides.update(case_values)
            overrides['simulation.tides_on'] = tides_on
            cases[case_id] = dict(case_id=case_id, tides_on=tides_on, baseline_id=baseline_id,
                                  sweep_values=case_values, overrides=overrides)
    return list(cases.values())


# ======================================================================================================================
# Execution
# ======================================================================================================================
def run_case(config, case, output_root, plot_each):
    """Run one case in a worker process. Failures are returned as records rather than raised, so the sweep goes on."""
    case_dir = Path(output_root) / case['case_id']
    case_dir.mkdir(parents=True, exist_ok=True)
    record = dict(case_id=case['case_id'], tides_on=case['tides_on'], baseline_id=case['baseline_id'],
                  case_dir=str(case_dir), **case['sweep_values'])
    try:
        results = run(config, overrides=case['overrides'], output_dir=case_dir, save=True, plot=plot_each,
                      show_progress=False, log_file=case_dir / 'run.log')
    except Exception as error:
        # Keep the traceback in the case's log so the failure can be diagnosed; the sweep records it as failed.
        get_logger('auto_run_v3').exception(f"Case {case['case_id']} raised.")
        record.update(success=False, message=f'{type(error).__name__}: {error}')
        return record
    record.update(results['summary'])
    record['output_table'] = next((path for path in results['output_paths'] if path.endswith('_output.txt')), '')
    return record


def _load_finished(case, output_root):
    """Summary record of a case finished successfully by an earlier invocation, or None (failed runs are redone)."""
    case_dir = Path(output_root) / case['case_id']
    summaries = sorted(case_dir.glob('*_summary.json'))
    tables = sorted(case_dir.glob('*_output.txt'))
    if not summaries or not tables:
        return None
    with open(summaries[-1], encoding='utf-8') as summary_file:
        summary = json.load(summary_file)['summary']
    if not summary.get('success', False):
        return None
    record = dict(case_id=case['case_id'], tides_on=case['tides_on'], baseline_id=case['baseline_id'],
                  case_dir=str(case_dir), **case['sweep_values'])
    record.update(summary)
    record['output_table'] = str(tables[-1])
    return record


def main(argv=None):
    """Command-line entry point: build the cases, run the unfinished ones in parallel, write sweep_summary.tsv."""
    parser = argparse.ArgumentParser(description='Tides on/off sweep of initial orbital states.')
    parser.add_argument('--config', default=CONFIG, help=f'Planet TOML (default: {CONFIG}).')
    parser.add_argument('--output-root', help='Sweep directory (default: results/tides_sweep_<config>).')
    parser.add_argument('--workers', type=int, default=max((os.cpu_count() or 4) - 2, 1),
                        help='Parallel processes (default: CPU count - 2).')
    parser.add_argument('--end-time', type=float, help='Override simulation.end_time_years for every case.')
    parser.add_argument('--plot-each', action='store_true', help='Also save the standard figures for every case.')
    parser.add_argument('--force', action='store_true', help='Re-run cases that already have results.')
    args = parser.parse_args(argv)

    output_root = Path(args.output_root) if args.output_root else (
        DEFAULT_OUTPUT_ROOT / f'tides_sweep_{Path(args.config).stem}')
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_root / 'sweep.log')

    fixed = dict(FIXED_OVERRIDES)
    if args.end_time is not None:
        fixed['simulation.end_time_years'] = args.end_time
    cases = build_cases(SWEEP, fixed)
    log.info(f"{len(cases)} cases ({sum(case['tides_on'] for case in cases)} with tides) -> {output_root}")

    with open(output_root / 'sweep_meta.json', 'w', encoding='utf-8') as meta_file:
        json.dump({'config': args.config, 'sweep': SWEEP, 'fixed_overrides': fixed,
                   'tides_only_keys': TIDES_ONLY_KEYS, 'started': time.strftime('%Y-%m-%d %H:%M:%S')},
                  meta_file, indent=2)

    records, pending = [], []
    for case in cases:
        finished = None if args.force else _load_finished(case, output_root)
        if finished is not None:
            records.append(finished)
        else:
            pending.append(case)
    log.info(f"{len(records)} cases already finished; running {len(pending)} with {args.workers} workers.")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, args.config, case, output_root, args.plot_each): case for case in pending}
        for count, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                record = future.result()
            except BrokenProcessPool as error:
                # A worker died hard (e.g., out of memory). Record the case as failed; other cases still report.
                record = dict(case_id=case['case_id'], tides_on=case['tides_on'], baseline_id=case['baseline_id'],
                              success=False, message=f'worker crashed: {error}', **case['sweep_values'])
            records.append(record)
            status = 'ok' if record.get('success') else f"FAILED ({record.get('message', '')})"
            log.info(f"[{count}/{len(pending)}] {record['case_id']}: {status}")

    table = pd.DataFrame(records).sort_values(['tides_on', 'case_id'], ascending=[False, True])
    table_path = output_root / 'sweep_summary.tsv'
    table.to_csv(table_path, sep='\t', index=False)
    num_failed = int((~table['success'].astype(bool)).sum())
    log.info(f"Wrote {table_path} ({len(table)} cases, {num_failed} failed).")
    return 1 if num_failed else 0


if __name__ == '__main__':
    sys.exit(main())
