"""Melt distribution of an adiabatic mantle, evaluated on the pre-computed MantleGrid."""
from collections import namedtuple

import numpy as np

from utils.mantle_grid import solidus_temperature

_NEWTON_MAX_ITERS = 20
_NEWTON_DEPTH_TOL = 1e-6  # m

MeltState = namedtuple('MeltState', (
    'melt_fraction_surface',       # Melt fraction at the top of the adiabat (0 to 1)
    'melt_fraction_bulk',          # Melt mass / mantle mass (0 to 1)
    'mass_melt',                   # Total mass of melt in the mantle (kg)
    'dmass_melt_dtemp',            # d(mass_melt)/d(T_pot) (kg/K)
    'radius_solidus',              # Radius where the adiabat meets the solidus; base of the melt region (m)
    'dradius_solidus_dtemp',       # d(radius_solidus)/d(T_pot) (m/K)
    'mass_melt_region',            # Mass of the melt-bearing region above radius_solidus (kg)
    'mass_below_solidus',          # Mass of the mantle below radius_solidus (kg); exactly 0 when it reaches the core
    'melt_fraction_melt_region',   # Melt mass / mass_melt_region (0 to 1)
    'radius_rheological',          # Radius where the melt fraction equals the critical value (m)
    'melt_fraction_solid_layer',   # Mean melt fraction of the solid-like layer below radius_rheological (0 to 1)
))


def _level_crossing(temp_potential, level, grid):
    """
    Depth where the adiabat exceeds the solidus by `level` kelvin, and its derivative with respect to temperature.

    The grid nodes bracket the crossing; Newton's method on the analytic solidus then locates it exactly, so the
    crossing depth and its temperature derivative are smooth in temperature (not just continuous).

    Parameters
    ----------
    temp_potential : float
        Mantle potential temperature (K).
    level : float
        Required excess of the adiabat over the solidus (K): 0 for the solidus, liq_offset for the liquidus.
    grid : MantleGrid

    Returns
    -------
    depth : float
        Crossing depth (m). 0 if the adiabat is below the level at the surface, the maximum depth if it never drops
        below.
    ddepth_dtemp : float
        Derivative of the crossing depth with respect to potential temperature (m/K). 0 when pinned at a boundary.
    """
    depths = grid.depths
    excess = temp_potential * grid.adiabat_factor - grid.solidus - level
    below  = excess < 0.0
    if not below.any():
        return depths[-1], 0.0
    index = int(np.argmax(below))
    if index == 0:
        return 0.0, 0.0

    top, bottom = depths[index - 1], depths[index]
    depth = top + excess[index - 1] / (excess[index - 1] - excess[index]) * (bottom - top)
    for _ in range(_NEWTON_MAX_ITERS):
        solidus, dsolidus = solidus_temperature(depth, grid.gravity, grid.density, grid.thermo)
        residual  = temp_potential * (1.0 + grid.adiabat_slope * depth) - solidus - level
        dresidual = temp_potential * grid.adiabat_slope - dsolidus
        step = residual / dresidual
        # The residual decreases with depth inside the bracket; stay inside it.
        depth = min(max(depth - step, top), bottom)
        if abs(step) < _NEWTON_DEPTH_TOL:
            break

    adiabat_factor = 1.0 + grid.adiabat_slope * depth
    _, dsolidus = solidus_temperature(depth, grid.gravity, grid.density, grid.thermo)
    return depth, -adiabat_factor / (temp_potential * grid.adiabat_slope - dsolidus)


def _integral_to(depth, depths, integrand, cumulative):
    """
    Integral from the surface to `depth` of the piecewise-linear interpolant of nodal `integrand`.

    Exact for the interpolant, so it varies continuously as `depth` moves inside a grid cell.
    """
    if depth <= 0.0:
        return 0.0
    if depth >= depths[-1]:
        return cumulative[-1]
    index      = int(np.searchsorted(depths, depth, side='right')) - 1
    offset     = depth - depths[index]
    cell_width = depths[index + 1] - depths[index]
    value_at   = integrand[index] + (integrand[index + 1] - integrand[index]) * offset / cell_width
    return cumulative[index] + 0.5 * (integrand[index] + value_at) * offset


def calc_melt_state(temp_potential, grid, critical_melt_fraction):
    """
    Melt distribution of the mantle for a given potential temperature.

    The mantle temperature follows the adiabat ``T_pot * adiabat_factor(z)``. The local melt fraction is linear between
    the solidus and liquidus and is assumed to decrease with depth (checked by ``grid.max_monotonic_temperature``).
    Every crossing depth and integral is interpolated inside grid cells, so all outputs vary continuously with
    temperature.

    Parameters
    ----------
    temp_potential : float
        Mantle potential temperature (K).
    grid : MantleGrid
        Pre-computed static mantle grid (from build_mantle_grid).
    critical_melt_fraction : float
        Melt fraction at the rheological transition (1 - critical crystal fraction).

    Returns
    -------
    MeltState
    """
    depths     = grid.depths
    liq_offset = grid.liq_offset
    density    = grid.density
    r_planet   = grid.radius_planet
    r_core     = grid.radius_core

    # Adiabat temperature above the solidus; the local melt fraction is clip(excess / liq_offset, 0, 1).
    solidus_excess = temp_potential * grid.adiabat_factor - grid.solidus

    depth_liquidus, _                   = _level_crossing(temp_potential, liq_offset, grid)
    depth_solidus, ddepth_solidus_dtemp = _level_crossing(temp_potential, 0.0, grid)
    depth_rheological, _                = _level_crossing(temp_potential, critical_melt_fraction * liq_offset, grid)

    # Unclipped melt fraction times r^2; valid between the liquidus and solidus crossings.
    partial_integrand  = solidus_excess * grid.r2 / liq_offset
    partial_cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (partial_integrand[1:] + partial_integrand[:-1]) * np.diff(depths))))

    def partial_integral(top, bottom):
        if bottom <= top:
            return 0.0
        return (_integral_to(bottom, depths, partial_integrand, partial_cumulative)
                - _integral_to(top, depths, partial_integrand, partial_cumulative))

    four_pi_rho = 4.0 * np.pi * density
    fully_molten_integral = _integral_to(depth_liquidus, depths, grid.r2, grid.cum_r2)
    mass_melt = four_pi_rho * (fully_molten_integral + partial_integral(depth_liquidus, depth_solidus))
    mass_melt = min(max(mass_melt, 0.0), grid.mass_mantle)

    latent_integral = (_integral_to(depth_solidus, depths, grid.latent_integrand, grid.cum_latent)
                       - _integral_to(depth_liquidus, depths, grid.latent_integrand, grid.cum_latent))
    dmass_melt_dtemp = four_pi_rho * max(latent_integral, 0.0)

    # Melt region: everything above the solidus crossing. When the crossing reaches the core-mantle boundary the solid
    # mantle is set to exactly zero; r_planet - depth would leave a floating-point sliver of solid.
    if depth_solidus >= depths[-1]:
        radius_solidus     = r_core
        mass_below_solidus = 0.0
    else:
        radius_solidus     = r_planet - depth_solidus
        mass_below_solidus = (4.0 / 3.0) * np.pi * density * (radius_solidus**3 - r_core**3)
    mass_melt_region = grid.mass_mantle - mass_below_solidus
    if mass_melt_region > 0.0:
        melt_fraction_melt_region = min(max(mass_melt / mass_melt_region, 0.0), 1.0)
    else:
        melt_fraction_melt_region = 0.0

    # Solid-like layer: everything below the rheological crossing. All of its melt lies above the solidus crossing.
    radius_rheological = r_planet - depth_rheological
    volume_below       = (4.0 / 3.0) * np.pi * (radius_rheological**3 - r_core**3)
    melt_below         = four_pi_rho * partial_integral(max(depth_rheological, depth_liquidus), depth_solidus)
    if volume_below > 0.0:
        melt_fraction_solid_layer = min(max(melt_below / (density * volume_below), 0.0), critical_melt_fraction)
    else:
        melt_fraction_solid_layer = min(max(solidus_excess[-1] / liq_offset, 0.0), critical_melt_fraction)

    return MeltState(
        melt_fraction_surface=min(max(solidus_excess[0] / liq_offset, 0.0), 1.0),
        melt_fraction_bulk=mass_melt / grid.mass_mantle,
        mass_melt=mass_melt,
        dmass_melt_dtemp=dmass_melt_dtemp,
        radius_solidus=radius_solidus,
        dradius_solidus_dtemp=-ddepth_solidus_dtemp,
        mass_melt_region=mass_melt_region,
        mass_below_solidus=mass_below_solidus,
        melt_fraction_melt_region=melt_fraction_melt_region,
        radius_rheological=radius_rheological,
        melt_fraction_solid_layer=melt_fraction_solid_layer,
    )
