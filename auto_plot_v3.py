"""
Figures comparing tides-on and tides-off runs from an auto_run_v3.py sweep.

Figures (saved in <sweep>/plots):

- ``metrics_vs_eccentricity.png``: magma ocean lifetime, water-loss timing, peak outgassing time, and the time to
  degas the water stored in the solid mantle, versus initial eccentricity, one line per initial spin, with the
  tides-off result as a dashed reference.
- ``timeseries_spin<S>.png``: time series for each initial spin S, one line per eccentricity plus tides off.
- ``lifetime_ratio.png``: magma ocean lifetime with tides divided by without, over the eccentricity-spin grid.

Usage::

    python auto_plot_v3.py results/tides_sweep_trappist1e
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logger import get_logger, setup_logging
from utils.plotting import INK_PRIMARY, INK_SECONDARY, SERIES_COLORS, style_axis

log = get_logger('auto_plot_v3')

# Sweep columns in sweep_summary.tsv are the dotted configuration keys used in auto_run_v3.SWEEP.
ECC_KEY  = 'planet.orbit.initial_eccentricity'
SPIN_KEY = 'planet.initial_spin_multiplier'

# Sequential blue ramp (light to dark) for ordered values such as eccentricity; starts at a step that stays visible
# on a white background.
ORDINAL_BLUES = ('#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
                 '#0d366b')

METRICS = (
    # column, axis label, log scale
    ('time_bulk_melt_below_critical_years', 'Magma ocean lifetime [yr]\n(bulk melt < critical)', True),
    ('time_water_lost_50pct_years', 'Time to lose 50% of water [yr]', True),
    ('time_peak_surface_water_years', 'Time of peak surface water [yr]', True),
    ('time_solid_water_half_degassed_years', 'Time to degas half of the\nsolid-mantle water [yr]', True),
)


def ordinal_colors(count):
    """`count` colors spread along the ordinal blue ramp."""
    if count == 1:
        return [ORDINAL_BLUES[4]]
    indices = np.linspace(0, len(ORDINAL_BLUES) - 1, count).round().astype(int)
    return [ORDINAL_BLUES[index] for index in indices]


def _style_plain(axis, ylabel, log_y):
    """Axis styling for plots whose x-axis is not time."""
    axis.set_ylabel(ylabel, color=INK_PRIMARY)
    if log_y:
        axis.set_yscale('log')
    axis.grid(True, color='#d9d8d4', linewidth=0.6)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)


def _metric_or_end(frame, column):
    """
    Metric values, with events that never happened (NaN) replaced by the run end time.

    Returns the values and a mask of which ones were replaced, so they can be drawn as lower limits.
    """
    values = frame[column].to_numpy(dtype=np.float64)
    missing = ~np.isfinite(values)
    if column.endswith('_years'):
        values = np.where(missing, frame['end_time_years'].to_numpy(dtype=np.float64), values)
    return values, missing


# ======================================================================================================================
# Figures
# ======================================================================================================================
def plot_metrics(summary, plot_dir):
    """
    Summary metrics versus initial eccentricity, one line per initial spin, tides-off as a dashed reference.

    Parameters
    ----------
    summary : pandas.DataFrame
        Rows of one baseline group: tides-on cases plus their tides-off baseline.
    plot_dir : pathlib.Path
        Output directory.

    Returns
    -------
    pathlib.Path
    """
    tides_on  = summary[summary['tides_on']]
    tides_off = summary[~summary['tides_on']]
    spins = sorted(tides_on[SPIN_KEY].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (column, label, log_y) in zip(axes.flat, METRICS):
        _style_plain(axis, label, log_y)
        for index, spin in enumerate(spins):
            subset = tides_on[tides_on[SPIN_KEY] == spin].sort_values(ECC_KEY)
            values, missing = _metric_or_end(subset, column)
            color = SERIES_COLORS[index]
            axis.plot(subset[ECC_KEY], values, color=color, linewidth=2.0, marker='o', markersize=6,
                      label=f'tides on, spin = {spin:g} n')
            if missing.any():
                # Open markers: the event had not happened by the end of the run (value is a lower limit).
                axis.plot(subset[ECC_KEY][missing], values[missing], linestyle='none', marker='o', markersize=9,
                          markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
        if len(tides_off):
            baseline, _ = _metric_or_end(tides_off, column)
            axis.axhline(baseline[0], color=INK_PRIMARY, linestyle='--', linewidth=1.5, label='tides off')
    for axis in axes[-1]:
        axis.set_xlabel('Initial eccentricity', color=INK_PRIMARY)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Tides on vs off (open markers: not reached by the end of the run)', color=INK_PRIMARY)
    fig.tight_layout()
    path = plot_dir / 'metrics_vs_eccentricity.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_timeseries(summary, plot_dir):
    """
    Time series for each initial spin: one line per eccentricity (ordinal blues) plus the tides-off run (dashed).

    Parameters
    ----------
    summary : pandas.DataFrame
        Rows of one baseline group (see `plot_metrics`).
    plot_dir : pathlib.Path
        Output directory.

    Returns
    -------
    list of pathlib.Path
    """
    tides_on  = summary[summary['tides_on']]
    tides_off = summary[~summary['tides_on']]
    baseline_table = pd.read_csv(tides_off['output_table'].iloc[0], sep='\t') if len(tides_off) else None
    paths = []

    panels = (
        # column(s), label, log, scale
        ('temp_mantle', 'Mantle potential T [K]', False),
        ('meltfrac', 'Bulk melt fraction [%]', False),
        ('Q_tid', 'Tidal heating [W]', True),
        (('mass_water_vapor', 'mass_water_ocean'), 'Surface water / initial', True),
        ('PO2', 'O$_2$ pressure [Pa]', True),
    )

    def column_values(table, column):
        if isinstance(column, tuple):
            return sum(table[name] for name in column)
        return table[column]

    for spin in sorted(tides_on[SPIN_KEY].unique()):
        subset = tides_on[tides_on[SPIN_KEY] == spin].sort_values(ECC_KEY)
        colors = ordinal_colors(len(subset))
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.3 * len(panels) + 0.6), sharex=True)
        for axis, (column, label, log_y) in zip(axes, panels):
            style_axis(axis, label, log_y=log_y)
            for color, (_, row) in zip(colors, subset.iterrows()):
                table = pd.read_csv(row['output_table'], sep='\t')
                values = column_values(table, column).to_numpy(dtype=np.float64)
                if log_y:
                    values = np.where(values > 0.0, values, np.nan)
                axis.plot(table['time_yr'], values, color=color, linewidth=1.8, label=f'e = {row[ECC_KEY]:g}')
            if baseline_table is not None:
                values = column_values(baseline_table, column).to_numpy(dtype=np.float64)
                if log_y:
                    values = np.where(values > 0.0, values, np.nan)
                axis.plot(baseline_table['time_yr'], values, color=INK_PRIMARY, linestyle='--', linewidth=1.5,
                          label='tides off')
            if log_y:
                axis.set_ylim(bottom=1e-8 if column != 'Q_tid' else 1e10)
        axes[0].legend(fontsize=8, frameon=False, ncol=2)
        axes[-1].set_xlabel('Time [yr]', color=INK_PRIMARY)
        fig.suptitle(f'Initial spin = {spin:g} x orbital motion', color=INK_PRIMARY)
        fig.tight_layout()
        path = plot_dir / f'timeseries_spin{spin:g}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_lifetime_ratio(summary, plot_dir):
    """
    Heat map of magma ocean lifetime with tides divided by without, over initial eccentricity and spin.

    Parameters
    ----------
    summary : pandas.DataFrame
        Rows of one baseline group (see `plot_metrics`).
    plot_dir : pathlib.Path
        Output directory.

    Returns
    -------
    pathlib.Path or None
        None if the group has no tides-off baseline.
    """
    tides_on  = summary[summary['tides_on']]
    tides_off = summary[~summary['tides_on']]
    if not len(tides_off):
        return None
    column = 'time_bulk_melt_below_critical_years'
    baseline, _ = _metric_or_end(tides_off, column)
    eccentricities = sorted(tides_on[ECC_KEY].unique())
    spins = sorted(tides_on[SPIN_KEY].unique())
    ratio = np.full((len(spins), len(eccentricities)), np.nan)
    lower_limit = np.zeros_like(ratio, dtype=bool)
    for _, row in tides_on.iterrows():
        values, missing = _metric_or_end(row.to_frame().T, column)
        i, j = spins.index(row[SPIN_KEY]), eccentricities.index(row[ECC_KEY])
        ratio[i, j] = values[0] / baseline[0]
        lower_limit[i, j] = missing[0]

    fig, axis = plt.subplots(figsize=(1.3 * len(eccentricities) + 2.5, 1.0 * len(spins) + 2.0))
    image = axis.imshow(np.log10(ratio), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'blues', ('#f0efec',) + ORDINAL_BLUES), aspect='auto', origin='lower')
    for i in range(len(spins)):
        for j in range(len(eccentricities)):
            if np.isfinite(ratio[i, j]):
                text = f"{'>' if lower_limit[i, j] else ''}{ratio[i, j]:.3g}"
                shade = np.log10(ratio[i, j]) > 0.6 * np.nanmax(np.log10(ratio))
                axis.text(j, i, text, ha='center', va='center', fontsize=8,
                          color='white' if shade else INK_PRIMARY)
    axis.set_xticks(range(len(eccentricities)), [f'{value:g}' for value in eccentricities])
    axis.set_yticks(range(len(spins)), [f'{value:g}' for value in spins])
    axis.set_xlabel('Initial eccentricity', color=INK_PRIMARY)
    axis.set_ylabel('Initial spin / orbital motion', color=INK_PRIMARY)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label('log$_{10}$(lifetime with tides / without)', color=INK_PRIMARY)
    axis.set_title('Magma ocean lifetime ratio (">": still molten at end)', color=INK_PRIMARY)
    fig.tight_layout()
    path = plot_dir / 'lifetime_ratio.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    """Command-line entry point: read sweep_summary.tsv and write the figures for each tides-off baseline group."""
    parser = argparse.ArgumentParser(description='Plot an auto_run_v3.py tides on/off sweep.')
    parser.add_argument('sweep_dir', help='Directory holding sweep_summary.tsv.')
    args = parser.parse_args(argv)
    setup_logging()

    sweep_dir = Path(args.sweep_dir)
    summary = pd.read_csv(sweep_dir / 'sweep_summary.tsv', sep='\t')
    summary['tides_on'] = summary['tides_on'].astype(bool)
    failed = summary[~summary['success'].astype(bool)]
    if len(failed):
        log.warning(f"Skipping {len(failed)} failed case(s): {', '.join(failed['case_id'])}")
    summary = summary[summary['success'].astype(bool)]

    # Each tides-on case is compared with its own tides-off baseline (they differ when SWEEP holds keys that are
    # not tides-only). One set of figures per baseline; a subdirectory per baseline when there are several.
    baseline_ids = sorted(summary.loc[summary['tides_on'], 'baseline_id'].unique())
    for baseline_id in baseline_ids:
        group = summary[(summary['tides_on'] & (summary['baseline_id'] == baseline_id))
                        | (summary['case_id'] == baseline_id)]
        plot_dir = sweep_dir / 'plots' if len(baseline_ids) == 1 else sweep_dir / 'plots' / baseline_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        saved = [plot_metrics(group, plot_dir), *plot_timeseries(group, plot_dir),
                 plot_lifetime_ratio(group, plot_dir)]
        for path in saved:
            if path is not None:
                log.info(f'Saved {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
