import numpy as np

from utils.get_melt_fractions import calc_melt_state

TEMPERATURES = np.arange(1300.0, 3200.0, 0.5)


def _states(model):
    return [calc_melt_state(temp, model.grid, model.critical_melt_fraction) for temp in TEMPERATURES]


def test_grid_ends_at_core_mantle_boundary(trappist_model):
    grid = trappist_model.grid
    assert np.isclose(grid.radii[-1], grid.radius_core)
    assert np.isclose(grid.depths[0], 0.0)


def test_melt_state_is_continuous(trappist_model):
    """No quantity may jump between adjacent 0.5 K samples by more than 0.5% of its range."""
    states = _states(trappist_model)
    for name in ('mass_melt', 'radius_solidus', 'mass_melt_region', 'melt_fraction_melt_region', 'radius_rheological',
                 'melt_fraction_solid_layer', 'dmass_melt_dtemp'):
        values = np.array([getattr(state, name) for state in states])
        max_step = np.max(np.abs(np.diff(values)))
        assert max_step <= 5e-3 * np.ptp(values), name


def test_melt_derivatives_match_finite_differences(trappist_model):
    states = _states(trappist_model)
    mass_melt = np.array([state.mass_melt for state in states])
    dmass     = np.array([state.dmass_melt_dtemp for state in states])
    interior  = (TEMPERATURES > 1440.0) & (TEMPERATURES < 2990.0)
    finite    = np.gradient(mass_melt, TEMPERATURES)
    assert np.max(np.abs(finite - dmass)[interior] / dmass.max()) < 0.01


def test_melt_state_bounds_and_ordering(trappist_model):
    grid = trappist_model.grid
    for state in _states(trappist_model):
        assert 0.0 <= state.melt_fraction_bulk <= 1.0
        assert 0.0 <= state.melt_fraction_melt_region <= 1.0
        assert 0.0 <= state.melt_fraction_solid_layer <= trappist_model.critical_melt_fraction
        # The rheological transition lies above (at larger radius than) the solidus crossing.
        assert grid.radius_core <= state.radius_solidus <= state.radius_rheological <= grid.radius_planet
        assert state.mass_melt <= state.mass_melt_region * (1.0 + 1e-9)


def test_no_melt_below_surface_solidus(trappist_model):
    state = calc_melt_state(1400.0, trappist_model.grid, trappist_model.critical_melt_fraction)
    assert state.mass_melt == 0.0
    assert state.radius_solidus == trappist_model.grid.radius_planet


def test_solidus_radius_derivative_is_smooth(trappist_model):
    """Regression: grid-interpolated crossings made d(r_s)/dT step at every node inside the solidus crossover."""
    temperatures = np.arange(1900.0, 2000.0, 0.05)
    derivative = np.array([calc_melt_state(temp, trappist_model.grid, trappist_model.critical_melt_fraction)
                           .dradius_solidus_dtemp for temp in temperatures])
    assert np.max(np.abs(np.diff(derivative))) < 0.01 * np.ptp(derivative)


def test_solid_mass_is_exactly_zero_when_fully_molten(trappist_model):
    """Regression: a floating-point sliver of solid mantle made the solid water concentration explode above ~3008 K."""
    state = calc_melt_state(3300.0, trappist_model.grid, trappist_model.critical_melt_fraction)
    assert state.mass_below_solidus == 0.0
    assert state.radius_solidus == trappist_model.grid.radius_core
