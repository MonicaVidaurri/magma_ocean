"""Diagnostics and summary metrics from an integrated solution."""
import numpy as np

from ODEs.combined_ode import STATE_NAMES


def postprocess_magma_ocean(model, t_sec, states):
    """
    Evaluate every diagnostic of the model at each output time.

    The diagnostics come from the same function the integrator used (`MagmaOceanModel.evaluate`), so the reported
    heating, pressures, and melt fractions are exactly those that drove the evolution.

    Parameters
    ----------
    model : MagmaOceanModel
        The model that produced the solution.
    t_sec : ndarray, shape (n,)
        Output times (s).
    states : ndarray, shape (num_states, n)
        State vectors at the output times.

    Returns
    -------
    results : dict
        ``'t'`` (s), ``'Tr'`` (the state array), one array per state name, one array per diagnostic, and
        ``'summary'`` (see `summarize`).
    """
    num_steps = states.shape[1]
    results = {'t': np.asarray(t_sec), 'Tr': states}
    for index, name in enumerate(STATE_NAMES):
        results[name] = states[index, :]

    for step in range(num_steps):
        derivatives, diagnostics = model.evaluate(t_sec[step], states[:, step])
        if step == 0:
            for name in diagnostics:
                results[name] = np.empty(num_steps, dtype=np.float64)
            results['dtemp_mantle_dt'] = np.empty(num_steps, dtype=np.float64)
        for name, value in diagnostics.items():
            results[name][step] = value
        results['dtemp_mantle_dt'][step] = derivatives[4]

    results['summary'] = summarize(model, results)
    return results


def _first_time(times, condition):
    """First time where `condition` is true, or NaN."""
    indices = np.flatnonzero(condition)
    return float(times[indices[0]]) if indices.size else float('nan')


def _last_time(times, condition):
    """Last time where `condition` is true, or NaN."""
    indices = np.flatnonzero(condition)
    return float(times[indices[-1]]) if indices.size else float('nan')


def summarize(model, results):
    """
    Scalar metrics describing the timing and amount of outgassing, used to compare runs (e.g., tides on vs off).

    Times are in years and are NaN when the event never happens within the run. "Liquid layer" means a surface layer
    whose melt fraction exceeds the rheological transition (the magma ocean in the rheological sense). "Bulk melt below
    critical" is the old model's magma-ocean end criterion (bulk melt fraction below the rheological transition).
    """
    props          = model.planet
    seconds_per_yr = model.params['constants']['seconds_per_year']
    t_years        = results['t'] / seconds_per_yr

    has_liquid_layer = results['radius_rheological'] < props.radius
    bulk_is_liquid   = results['meltfrac'] >= model.critical_melt_fraction
    has_melt         = results['mass_melt'] > 0.0
    water_total      = results['mass_water_solid'] + results['mass_water_fluid']
    water_surface    = results['mass_water_vapor'] + results['mass_water_ocean']
    water_initial    = water_total[0]
    water_lost       = water_initial - water_total

    def water_loss_time(fraction):
        if water_initial <= 0.0:
            return float('nan')
        return _first_time(t_years, water_lost >= fraction * water_initial)

    peak_index = int(np.argmax(water_surface))

    # Water stored in the solid mantle (cumulates) reaches the surface late, through solid-state degassing once the
    # liquid layer has thinned. Time for that reservoir to fall to half of its peak.
    water_solid      = results['mass_water_solid']
    solid_peak_index = int(np.argmax(water_solid))
    after_solid_peak = np.arange(water_solid.size) > solid_peak_index
    solid_half_time  = _first_time(t_years, after_solid_peak & (water_solid < 0.5 * water_solid[solid_peak_index]))
    return {
        'end_time_years': float(t_years[-1]),
        # The pre-2026-09 model's magma ocean ended when the bulk melt fraction fell below the rheological transition.
        'time_bulk_melt_below_critical_years': _first_time(t_years, ~bulk_is_liquid),
        'time_liquid_layer_first_end_years': _first_time(t_years, ~has_liquid_layer),
        'time_liquid_layer_last_years': _last_time(t_years, has_liquid_layer),
        'liquid_layer_at_end': bool(has_liquid_layer[-1]),
        'time_melt_first_gone_years': _first_time(t_years, ~has_melt),
        'melt_at_end': bool(has_melt[-1]),
        'peak_surface_water_kg': float(water_surface[peak_index]),
        'time_peak_surface_water_years': float(t_years[peak_index]),
        'time_water_lost_50pct_years': water_loss_time(0.5),
        'time_water_lost_90pct_years': water_loss_time(0.9),
        'water_lost_fraction_final': float(water_lost[-1] / water_initial) if water_initial > 0.0 else float('nan'),
        'peak_solid_water_kg': float(water_solid[solid_peak_index]),
        'time_solid_water_half_degassed_years': solid_half_time,
        'water_solid_final_kg': float(results['mass_water_solid'][-1]),
        'water_surface_final_kg': float(water_surface[-1]),
        'PO2_final_pa': float(results['PO2'][-1]),
        'PO2_max_pa': float(np.max(results['PO2'])),
        'Patm_final_pa': float(results['Patm'][-1]),
        'temp_mantle_final_k': float(results['temp_mantle'][-1]),
        'meltfrac_final': float(results['meltfrac'][-1]),
        'eccentricity_final': float(results['eccentricity'][-1]),
        'tidal_heating_max_w': float(np.max(results['Q_tid'])),
    }
