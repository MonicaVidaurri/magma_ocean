import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.ticker as ticker

dirname = 'low_values/OG/autotides_t1e_highXUV_tideson'
plotdir = 'results/plots/newNEW_PO2/tidesON/proxb_high'

rootdir = '/Users/mvidaurr/Desktop/PycharmProjects/magma_ocean'

# ecc_range = np.linspace(0.02, 0.6, num=5)
# spin_range = np.linspace(start=-100, stop=100, num=15)
time_range = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
time_range = [1e5, 1e6, 1e7, 1e8, 1e9]

ecc_range = np.linspace(0.0001, 0.2, 20)
spin_range = np.arange(-10,11,dtype=np.float64)

# PO2 spans many orders of magnitude, so the color scale is logarithmic.
# Runs that never outgas sit at ~1e-56; clip them into the lowest color bin.
po2_floor = 1e-20
use_log_color = True


def load_po2(time, ecc, spin):
    """PO2 for the run started at (ecc, spin), evaluated at `time` years.

    The grid coordinates come from the directory name, not from the CSV
    columns: `eccentricity` and `spin_ratio_planet` inside the file are the
    evolved values at each output step, which is not what we grid against.
    """
    pathstr = ('results/' + dirname +
               '/t=' + '{:.0e}'.format(time) +
               '/ecc=' + '{:.3}'.format(ecc) +
               '/spin=' + '{:.3}'.format(spin))
    matches = [f for f in glob.glob(pathstr + '/*') if 'output' in f]
    if not matches:
        return np.nan

    df = pd.read_csv(matches[0], sep='\t', skipinitialspace=True)
    df = df.dropna(subset=['PO2'])
    if df.empty:
        return np.nan

    # Each run terminates at its own t=..., so the last output row is the
    # requested time step. If the file carries an explicit time column, take
    # the row closest to `time` instead.
    time_col = next((c for c in ('time', 'time_years', 'years') if c in df.columns), None)
    if time_col is not None:
        return df.loc[(df[time_col] - time).abs().idxmin(), 'PO2']
    return df['PO2'].iloc[-1]


def build_grid(time):
    """PO2 on the (initial spin) x (initial ecc) grid, shaped for imshow."""
    grid = np.full((len(spin_range), len(ecc_range)), np.nan)
    for i, spin in enumerate(spin_range):
        for j, ecc in enumerate(ecc_range):
            grid[i, j] = load_po2(time, ecc, spin)
    return grid


def cell_extent(values):
    """Edges of the imshow cells so each block is centered on its grid value."""
    step = (values[-1] - values[0]) / (len(values) - 1)
    return values[0] - step / 2, values[-1] + step / 2


extent = (*cell_extent(ecc_range), *cell_extent(spin_range))

grids = {t: build_grid(t) for t in time_range}

all_po2 = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
if use_log_color:
    vmin = max(np.nanmin(all_po2), po2_floor)
    vmax = np.nanmax(all_po2)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
else:
    norm = colors.Normalize(vmin=np.nanmin(all_po2), vmax=np.nanmax(all_po2))


def plot_time(time, ax=None, save=True):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()

    grid = np.clip(grids[time], po2_floor, None) if use_log_color else grids[time]

    im = ax.imshow(grid, origin='lower', extent=extent, aspect='auto',
                   cmap='viridis', norm=norm, interpolation='nearest')

    ax.set_xlabel('Initial eccentricity')
    ax.set_ylabel('Initial spin')
    ax.set_title('t = ' + '{:.0e}'.format(time) + ' years')

    if standalone:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
        cbar.set_label('PO2')
        if save:
            fig.savefig(plotdir + '/t_' + '{:.0e}'.format(time) + '.png', dpi=150)
        plt.show()

    return im


def plot_all(save=True):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for time, ax in zip(time_range, axes.ravel()):
        im = plot_time(time, ax=ax)
    for ax in axes.ravel()[len(time_range):]:
        ax.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, label='PO2')
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    if save:
        fig.savefig(plotdir + '/PO2_vs_initial_ecc_spin.png', dpi=150)
    plt.show()


plot_all()