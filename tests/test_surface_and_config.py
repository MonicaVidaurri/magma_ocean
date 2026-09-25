import numpy as np
import pytest

from utils.config import load_config, set_by_path
from utils.get_flux import get_flux
from utils.surface_temperature import solve_surface_temperature, water_vapor_pressure


# ======================================================================================================================
# Surface temperature
# ======================================================================================================================
@pytest.mark.parametrize('temp_mantle, pressure_H2O', [(3000.0, 5e6), (2200.0, 3e6), (1500.0, 1e4), (1800.0, 0.0)])
def test_surface_temperature_balances_fluxes(trappist_model, temp_mantle, pressure_H2O):
    props  = trappist_model.planet
    params = trappist_model.params
    coefficient = 1.0e-2
    surface = solve_surface_temperature(temp_mantle, coefficient, 300.0, pressure_H2O, 10.0, props.radius,
                                        props.gravity, params)
    assert 300.0 <= surface.temp_surface <= temp_mantle
    interior = coefficient * (temp_mantle - surface.temp_surface)**(4.0 / 3.0)
    outgoing = get_flux(surface.temp_surface, 300.0, surface.pressure_H2O, 10.0, props.radius, props.gravity, params)
    assert np.isclose(interior, outgoing, rtol=1e-5)


@pytest.mark.parametrize('temp_mantle', [900.0, 1200.0, 1600.0, 2400.0])
def test_surface_temperature_is_hottest_root(trappist_model, temp_mantle):
    """With condensation the balance can have several roots; the solver must return the hottest one."""
    props, params = trappist_model.planet, trappist_model.params
    coefficient, temp_eq, pressure_H2O, pressure_O2 = 1.0e-3, 250.0, 3.0e7, 10.0
    surface = solve_surface_temperature(temp_mantle, coefficient, temp_eq, pressure_H2O, pressure_O2, props.radius,
                                        props.gravity, params)
    atm = params['planet']['atmosphere']
    hotter = np.linspace(surface.temp_surface + 0.05, temp_mantle - 1e-6, 4000)
    residual = [coefficient * (temp_mantle - temp)**(4.0 / 3.0)
                - get_flux(temp, temp_eq, water_vapor_pressure(temp, pressure_H2O, atm), pressure_O2, props.radius,
                           props.gravity, params) for temp in hotter]
    assert np.all(np.array(residual) < 0.0)


def test_vapor_is_capped_by_saturation_when_cold(trappist_params):
    atm = trappist_params['planet']['atmosphere']
    assert water_vapor_pressure(300.0, 1.0e7, atm) < 1.0e4
    assert np.isclose(water_vapor_pressure(1000.0, 1.0e7, atm), 1.0e7)


# ======================================================================================================================
# Configuration
# ======================================================================================================================
def test_planet_config_overrides_baseline(trappist_params):
    assert trappist_params['star']['stellar_track_file'] == 'data/stellar_dataTrappist1.txt'
    assert trappist_params['star']['tides']['moi_factor'] == 0.216
    # Filled from the baseline.
    assert trappist_params['planet']['tides']['love_method'] == 'radial_solver'


def test_overrides_and_bad_section():
    params = load_config('trappist1e.toml', overrides={'planet.orbit.initial_eccentricity': 0.05})
    assert params['planet']['orbit']['initial_eccentricity'] == 0.05
    with pytest.raises(KeyError):
        set_by_path(params, 'planet.orbitt.initial_eccentricity', 0.1)


def test_retired_key_is_rejected(tmp_path):
    config = tmp_path / 'old.toml'
    config.write_text('[planet.convection]\nmelt_fraction_threshold = 0.4\n')
    with pytest.raises(KeyError, match='no longer used'):
        load_config(config)


def test_missing_stellar_track_is_reported(tmp_path):
    config = tmp_path / 'bad_track.toml'
    config.write_text('[star]\nstellar_track_file = "data/does_not_exist.txt"\n')
    with pytest.raises(FileNotFoundError):
        load_config(config)
