"""Standard figures for a single model run."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots in fixed order (validated CVD-safe order for line charts), plus text/grid inks.
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_GRID      = '#d9d8d4'

# Lower limits of the log axes; below these the values are numerical residue once the water is gone.
FLOOR_OUTGASSING_KG_S = 1.0e-3
FLOOR_PRESSURE_PA     = 1.0e-10


def style_axis(axis, ylabel, log_y=False):
    """Recessive grid and spines, log-time x-axis, and a y label."""
    axis.set_xscale('log')
    if log_y:
        axis.set_yscale('log')
    axis.set_ylabel(ylabel, color=INK_PRIMARY)
    axis.grid(True, color=INK_GRID, linewidth=0.6)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        axis.spines[spine].set_color(INK_SECONDARY)


def _new_figure(num_panels, title):
    fig, axes = plt.subplots(num_panels, 1, figsize=(8.5, 2.6 * num_panels + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(title, color=INK_PRIMARY)
    return fig, axes


def _twin(axis, ylabel, log_y=False):
    """Right-hand y-axis sharing the panel's time axis."""
    twin = axis.twinx()
    if log_y:
        twin.set_yscale('log')
    twin.set_ylabel(ylabel, color=INK_PRIMARY)
    twin.tick_params(colors=INK_SECONDARY, labelsize=9)
    twin.spines['top'].set_visible(False)
    for spine in ('left', 'right', 'bottom'):
        twin.spines[spine].set_color(INK_SECONDARY)
    return twin


def _positive(values):
    """Hide non-positive values on log axes (gaps, not invented values)."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0.0, values, np.nan)


def _legend(*axes):
    """One legend for the lines of a panel and its twin."""
    handles = [handle for axis in axes for handle in axis.get_lines() if not handle.get_label().startswith('_')]
    axes[0].legend(handles, [handle.get_label() for handle in handles], fontsize=8, frameon=False,
                   labelcolor=INK_PRIMARY)


def outgassing_rate(results):
    """
    Water flux from the interior (melt and solid mantle) to the surface (kg/s): the growth of the vapor and ocean
    inventories plus what escape removes from them.
    """
    water_surface = results['mass_water_vapor'] + results['mass_water_ocean']
    return np.gradient(water_surface, results['t']) + results['water_loss_rate']


def plot_run(results, save_prefix, water_scale=None):
    """
    Save the orbit, outgassing, and heating figures of one run.

    Parameters
    ----------
    results : dict
        Output of `utils.postprocess.postprocess_magma_ocean`.
    save_prefix : str
        Path prefix for the PNG files.
    water_scale : float, optional
        Unused; kept for the existing call signature.

    Returns
    -------
    list of str
        Paths of the saved figures.
    """
    t_years = results['t'] / 3.15569e7
    saved   = []
    solid, dotted = dict(linewidth=2.0), dict(linewidth=2.0, linestyle=':')

    def finish(fig, axes, name):
        axes[-1].set_xlabel('Time [yr]', color=INK_PRIMARY)
        fig.tight_layout()
        path = f'{save_prefix}_{name}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # --- Figure 1: orbit and spin ---
    fig, axes = _new_figure(2, 'Orbit and spin')
    style_axis(axes[0], 'Semi-major axis [au]')
    axes[0].plot(t_years, results['semi_major_axis'] / 1.496e11, color=SERIES_COLORS[0], label='a', **solid)
    twin = _twin(axes[0], 'Eccentricity')
    twin.plot(t_years, results['eccentricity'], color=SERIES_COLORS[1], label='e', **dotted)
    _legend(axes[0], twin)
    style_axis(axes[1], 'Spin / orbital motion')
    axes[1].plot(t_years, results['spin_freq_planet'] / results['orbital_freq'], color=SERIES_COLORS[0], **solid)
    finish(fig, axes, 'orbit')

    # --- Figure 2: outgassing ---
    fig, axes = _new_figure(2, 'Outgassing')
    style_axis(axes[0], 'Outgassing rate [kg/s]', log_y=True)
    axes[0].plot(t_years, _positive(outgassing_rate(results)), color=SERIES_COLORS[0], label='Interior total',
                 **solid)
    axes[0].plot(t_years, _positive(results['degassing_rate']), color=SERIES_COLORS[2], label='Solid mantle',
                 **solid)
    axes[0].set_ylim(bottom=FLOOR_OUTGASSING_KG_S)
    twin = _twin(axes[0], 'Bulk melt fraction')
    twin.plot(t_years, results['meltfrac'], color=SERIES_COLORS[1], label='Melt fraction', **dotted)
    twin.set_ylim(0.0, 1.0)
    _legend(axes[0], twin)
    style_axis(axes[1], 'Atmospheric pressure [Pa]', log_y=True)
    axes[1].plot(t_years, _positive(results['Patm'] + results['PO2']), color=SERIES_COLORS[0], label='Total',
                 **solid)
    axes[1].plot(t_years, _positive(results['Patm']), color=SERIES_COLORS[2], label='H$_2$O', linewidth=1.2)
    axes[1].plot(t_years, _positive(results['PO2']), color=SERIES_COLORS[3], label='O$_2$', linewidth=1.2)
    axes[1].set_ylim(bottom=FLOOR_PRESSURE_PA)
    _legend(axes[1])
    finish(fig, axes, 'outgassing')

    # --- Figure 3: heating and the solid layer's rheology ---
    fig, axes = _new_figure(2, 'Heating and rheology')
    style_axis(axes[0], 'Heating [W]', log_y=True)
    axes[0].plot(t_years, _positive(results['Q_tid']), color=SERIES_COLORS[0], label='Tidal', **solid)
    axes[0].plot(t_years, _positive(results['Q_rad']), color=SERIES_COLORS[1], label='Radiogenic', **solid)
    _legend(axes[0])
    style_axis(axes[1], 'Effective viscosity [Pa s]', log_y=True)
    axes[1].plot(t_years, _positive(results['tidal_visc']), color=SERIES_COLORS[0], label='Viscosity', **solid)
    twin = _twin(axes[1], 'Shear modulus [GPa]')
    twin.plot(t_years, results['tidal_shear'] / 1e9, color=SERIES_COLORS[1], label='Shear modulus', **dotted)
    _legend(axes[1], twin)
    finish(fig, axes, 'heating')

    return saved
