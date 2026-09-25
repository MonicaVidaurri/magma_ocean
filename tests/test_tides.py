import copy

import numpy as np
import pytest

from physics.tides import TidalModel, orbital_motion


@pytest.fixture(scope='module', params=['radial_solver', 'homogeneous'])
def tidal_setup(request, trappist_params, trappist_model):
    params = copy.deepcopy(trappist_params)
    params['planet']['tides']['love_method'] = request.param
    props = trappist_model.planet
    grav  = params['constants']['G']
    tidal_model = TidalModel(params, props.radius, props.mass, props.radius_core, props.mass_core,
                             props.density_mantle, props.radius_host, props.mass_host)
    semi_major_axis = params['planet']['orbit']['initial_semi_major_axis'] * 1.496e11
    mean_motion = orbital_motion(semi_major_axis, props.mass_host, props.mass, grav)
    return tidal_model, props, grav, semi_major_axis, mean_motion


CASES = [
    # eccentricity, planet spin / n, host spin (rad/s), rheological radius / planet radius
    (0.2, 10.0, 2.2e-5, 1.0),
    (0.2, 10.0, 2.2e-5, 0.6),
    (0.05, 1.2, 1.0e-4, 1.0),
    (0.0, 3.0, 5.0e-6, 0.9),
    (0.3, 0.9, 2.2e-5, 0.8),
]


@pytest.mark.parametrize('eccentricity, spin_ratio, spin_host, radius_fraction', CASES)
def test_energy_and_angular_momentum_conservation(tidal_setup, eccentricity, spin_ratio, spin_host, radius_fraction):
    """Heat dissipated in both bodies equals the orbital + spin energy lost; total angular momentum is conserved."""
    tidal_model, props, grav, semi_major_axis, mean_motion = tidal_setup
    spin_planet = spin_ratio * mean_motion
    rates = tidal_model.calc_rates(semi_major_axis, eccentricity, spin_host, spin_planet, 1.0e18, 3.7e10,
                                   radius_fraction * props.radius)

    mass_host, mass_planet = props.mass_host, props.mass
    d_orbit_energy = grav * mass_host * mass_planet / (2.0 * semi_major_axis**2) * rates.da_dt
    d_spin_energy  = (tidal_model.moi_planet * spin_planet * rates.dspin_dt_planet
                      + tidal_model.moi_host * spin_host * rates.dspin_dt_host)
    heating = rates.tidal_heating_planet + rates.tidal_heating_host
    assert heating > 0.0
    assert np.isclose(-(d_orbit_energy + d_spin_energy), heating, rtol=1e-8)

    reduced_mass = mass_host * mass_planet / (mass_host + mass_planet)
    total_mass   = mass_host + mass_planet
    d_orbit_momentum = reduced_mass * np.sqrt(grav * total_mass) * (
        0.5 * np.sqrt((1.0 - eccentricity**2) / semi_major_axis) * rates.da_dt
        - np.sqrt(semi_major_axis) * eccentricity / np.sqrt(1.0 - eccentricity**2) * rates.de_dt)
    d_spin_momentum = tidal_model.moi_planet * rates.dspin_dt_planet + tidal_model.moi_host * rates.dspin_dt_host
    assert abs(d_orbit_momentum + d_spin_momentum) <= 1e-8 * abs(d_spin_momentum)


def test_zero_eccentricity_does_not_zero_spin_tides(tidal_setup):
    """Regression: the classic path raised at e = 0 and a bare except zeroed every tidal output."""
    tidal_model, props, _, semi_major_axis, mean_motion = tidal_setup
    rates = tidal_model.calc_rates(semi_major_axis, 0.0, 2.2e-5, 10.0 * mean_motion, 1.0e18, 3.7e10, props.radius)
    assert rates.tidal_heating_planet > 0.0
    assert rates.dspin_dt_planet < 0.0
    assert rates.de_dt == 0.0


def test_no_solid_layer_means_no_planet_dissipation(tidal_setup):
    tidal_model, props, _, semi_major_axis, mean_motion = tidal_setup
    rates = tidal_model.calc_rates(semi_major_axis, 0.2, 2.2e-5, 10.0 * mean_motion, 1.0e18, 3.7e10,
                                   props.radius_core)
    assert rates.tidal_heating_planet == 0.0
    assert rates.tidal_scale_planet == 0.0


def test_heating_shrinks_with_solid_layer(tidal_setup):
    tidal_model, props, _, semi_major_axis, mean_motion = tidal_setup
    heating = [tidal_model.calc_rates(semi_major_axis, 0.2, 2.2e-5, mean_motion, 1.0e18, 3.7e10,
                                      props.radius_core + fraction * (props.radius - props.radius_core)
                                      ).tidal_heating_planet
               for fraction in (1.0, 0.5, 0.1, 0.01)]
    assert all(np.diff(heating) < 0.0)


def test_tides_off_returns_zeros(trappist_params, trappist_model):
    params = copy.deepcopy(trappist_params)
    params['simulation']['tides_on'] = False
    props = trappist_model.planet
    tidal_model = TidalModel(params, props.radius, props.mass, props.radius_core, props.mass_core,
                             props.density_mantle, props.radius_host, props.mass_host)
    rates = tidal_model.calc_rates(4.4e9, 0.2, 2.2e-5, 1.0e-4, 1.0e18, 3.7e10, props.radius)
    assert all(value == 0.0 for value in rates)


def test_planet_moment_of_inertia_is_structural(tidal_setup):
    tidal_model, props, *_ = tidal_setup
    factor = tidal_model.moi_planet / (props.mass * props.radius**2)
    # A dense core makes the planet more centrally condensed than a uniform sphere (0.4).
    assert 0.25 < factor < 0.4


def test_structure_calibration_matches_layer_masses(tidal_setup):
    """The Birch-Murnaghan reference densities reproduce the model's core and mantle masses."""
    tidal_model, props, *_ = tidal_setup
    world = tidal_model._build_planet(1.0e18, 3.7e10, props.radius)
    assert np.isclose(world.get_layer(0).mass, props.mass_core, rtol=1e-6)
    assert np.isclose(world.get_layer(1).mass, props.mass_mantle, rtol=1e-6)
    # Self-compression: the solid mantle is denser at the core-mantle boundary than at the surface.
    assert world.get_density(1.001 * props.radius_core) > 1.1 * world.get_density(0.999 * props.radius)


def _heating_with_contrast(tidal_setup, contrast, radius_fraction=0.85):
    tidal_model, props, _, semi_major_axis, mean_motion = tidal_setup
    saved = tidal_model.magma_density_contrast
    tidal_model.magma_density_contrast = contrast
    try:
        return tidal_model.calc_rates(semi_major_axis, 0.2, 2.2e-5, mean_motion, 1.0e18, 3.7e10,
                                      radius_fraction * props.radius).tidal_heating_planet
    finally:
        tidal_model.magma_density_contrast = saved


def test_static_magma_layer_screens_the_solid(tidal_setup):
    """
    A static magma layer loads the solid below it and cancels part of the tidal forcing; the cancellation weakens as
    the magma gets lighter. Only meaningful for the radial solver (the homogeneous method ignores the magma layer).
    """
    tidal_model = tidal_setup[0]
    if tidal_model.love_method != 'radial_solver':
        pytest.skip('magma loading needs the radial solver')
    heating = [_heating_with_contrast(tidal_setup, contrast) for contrast in (0.01, 0.05, 0.2, 0.5)]
    assert all(np.diff(heating) > 0.0)


def test_thin_magma_layer_is_omitted(tidal_setup):
    tidal_model, props, _, semi_major_axis, mean_motion = tidal_setup
    rates = [tidal_model.calc_rates(semi_major_axis, 0.2, 2.2e-5, mean_motion, 1.0e18, 3.7e10, radius
                                    ).tidal_heating_planet
             for radius in (props.radius, props.radius - 0.5 * tidal_model.min_magma_thickness)]
    assert rates[0] == rates[1]
