import numpy as np

from run_model import run


def test_rhs_continuous_across_melting(trappist_model):
    """Mantle heating/cooling rate must vary smoothly through the rheological transition (no phase switches)."""
    state = trappist_model.initial_state()
    state[3] = 1.0 * trappist_model.evaluate(0.0, state)[1]['orbital_freq']
    temperatures = np.arange(2000.0, 2800.0, 1.0)
    rates = []
    for temp in temperatures:
        state[4] = temp
        rates.append(trappist_model.rhs(3.15e10, state)[4])
    rates = np.array(rates)
    steps = np.abs(np.diff(rates))
    assert np.max(steps) < 0.05 * np.ptp(rates)


def test_short_run_conserves_water_and_stays_physical(tmp_path):
    results = run('trappist1e.toml', overrides={'simulation.tides_on': True, 'simulation.end_time_years': 2.0e4},
                  output_dir=tmp_path, save=True, plot=False, show_progress=False)
    assert results['success']
    for name in ('mass_water_solid', 'mass_water_fluid', 'mass_oxygen_fluid', 'mass_oxygen_solid'):
        assert np.all(results[name] >= -1e-6 * results['mass_water_fluid'][0]), name
    assert np.all(results['eccentricity'] >= -1e-9)

    # Water only leaves the planet by escape.
    water_total = results['mass_water_solid'] + results['mass_water_fluid']
    assert np.all(np.diff(water_total) <= 1e-9 * water_total[0])

    # The water reservoirs partition the fluid inventory.
    parts = results['mass_water_dissolved'] + results['mass_water_vapor'] + results['mass_water_ocean']
    assert np.allclose(parts, results['mass_water_fluid'], rtol=1e-6)

    assert any(path.endswith('_output.txt') for path in results['output_paths'])
    assert any(path.endswith('_summary.json') for path in results['output_paths'])


def test_hot_start_above_full_melting(tmp_path):
    """Regression: starting with the solidus crossing at the core-mantle boundary crashed the integrator."""
    results = run('trappist1e.toml', overrides={'planet.initial_mantle_temp': 3300.0,
                                                'simulation.end_time_years': 1.0e3},
                  output_dir=tmp_path, save=False, plot=False, show_progress=False)
    assert results['success']


def test_dry_planet_runs(tmp_path):
    """Regression: zero initial water divided by zero in the state scaling."""
    results = run('trappist1e.toml', overrides={'planet.ocean_mass_multiplier': 0.0,
                                                'simulation.end_time_years': 1.0e4},
                  output_dir=tmp_path, save=False, plot=False, show_progress=False)
    assert results['success']
    assert np.all(results['mass_water_fluid'] == 0.0)


def test_no_solid_state_degassing_beneath_deep_liquid_layer(trappist_model):
    """Solid-state degassing only draws on melt in the solid-like layer, not on the magma ocean itself."""
    state = trappist_model.initial_state()
    diagnostics = trappist_model.evaluate(3.15e7, state)[1]
    liquid_depth = trappist_model.planet.radius - diagnostics['radius_rheological']
    assert liquid_depth > trappist_model.params['numerical']['degas_max_depth']
    assert diagnostics['degassing_rate'] == 0.0


def test_fixed_eccentricity_holds_e_but_not_a(trappist_params):
    import copy
    from ODEs.combined_ode import MagmaOceanModel
    params = copy.deepcopy(trappist_params)
    params['planet']['orbit']['fixed_eccentricity'] = True
    model = MagmaOceanModel(params)
    state = model.initial_state()
    state[4] = 2500.0  # Solid-like layer present, so the planet dissipates
    state[3] = model.evaluate(3.15e10, state)[1]['orbital_freq']  # Synchronous: eccentricity tides shrink the orbit
    dstate_dt, diagnostics = model.evaluate(3.15e10, state)
    assert dstate_dt[1] == 0.0
    assert diagnostics['Q_tid'] > 0.0
    assert dstate_dt[0] < 0.0


def test_fixed_spin_holds_spin(trappist_params):
    import copy
    from ODEs.combined_ode import MagmaOceanModel
    params = copy.deepcopy(trappist_params)
    params['planet']['fixed_spin'] = True
    model = MagmaOceanModel(params)
    state = model.initial_state()
    state[4] = 2500.0
    dstate_dt, diagnostics = model.evaluate(3.15e10, state)
    assert dstate_dt[3] == 0.0
    assert diagnostics['Q_tid'] > 0.0
