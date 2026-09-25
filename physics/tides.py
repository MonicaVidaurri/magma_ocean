"""
Dual-body tidal dissipation between the host star and a layered planet, using the TidalPy `_x` backend.

All TidalPy usage in the model is isolated in this module. TidalPy's classic modules are removed in v0.9.0, and the
`_x` suffixes will be dropped at that point, so only the imports below should need updating.
"""
from collections import namedtuple
import logging
import warnings

import numpy as np

# Importing any TidalPy module announces the classic-to-`_x` transition. This module only uses `_x` modules.
warnings.filterwarnings('ignore', message="TidalPy's backend is changing")

from TidalPy.structures_x.worlds import LayeredWorld, StarWorld
from TidalPy.structures_x.layers import SolidLiquidLayer
from TidalPy.Material_x.eos import make_material_eos
from TidalPy.rheology_x import make_rheology
from TidalPy.viscosity_x import make_viscosity
from TidalPy.Tides_x.classes import make_tide
from TidalPy.dynamics_x import OrbitSolver

log = logging.getLogger(__name__)

TidalRates = namedtuple('TidalRates', (
    'da_dt',                 # Semi-major axis derivative (m/s)
    'de_dt',                 # Eccentricity derivative (1/s)
    'dspin_dt_host',         # Host spin-rate derivative (rad/s^2)
    'dspin_dt_planet',       # Planet spin-rate derivative (rad/s^2)
    'tidal_heating_host',    # Heating in the host (W)
    'tidal_heating_planet',  # Heating in the planet (W)
    'tidal_scale_planet',    # Volume fraction of the planet in the tidally active solid layer
))

_ZERO_RATES = TidalRates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Reference-density calibration of the Birch-Murnaghan structure (see TidalModel._calibrate_structure)
_CALIBRATION_RTOL      = 1.0e-8
_CALIBRATION_MAX_ITERS = 60


def orbital_motion(semi_major_axis, mass_host, mass_planet, gravitational_constant):
    """Keplerian mean motion (rad/s)."""
    return np.sqrt(gravitational_constant * (mass_host + mass_planet) / semi_major_axis**3)


def _birch_murnaghan(reference_density, bulk_modulus, bulk_modulus_derivative, **material):
    """Third-order Birch-Murnaghan material for a TidalPy layer; `material` adds e.g. the static shear modulus."""
    return make_material_eos('birch_murnaghan', {
        'reference_density_kg_m3': reference_density,
        'reference_bulk_modulus_pa': bulk_modulus,
        'bulk_modulus_derivative': bulk_modulus_derivative,
        **material})


class TidalModel:
    """
    Tidal response of the star-planet pair.

    The star uses a constant phase lag (CPL) model with a fixed k2 and Q. The planet is three self-compressed layers
    whose hydrostatic structure comes from TidalPy's EOS solver:

    - a liquid, non-dissipative core;
    - a solid-like mantle between the core and the rheological transition radius, with the configured rheology
      (e.g., Andrade) and a uniform shear modulus and viscosity;
    - a static liquid magma layer above it, omitted when thinner than `min_magma_layer_thickness`.

    Each layer follows a third-order Birch-Murnaghan law. The core and mantle reference densities are calibrated once so
    the solved layer masses equal the model's core and mantle masses. The magma follows the mantle's law with its
    density reduced by `magma_density_contrast` at every pressure, so it is always lighter than the solid beneath it.

    The Love numbers come from `planet.tides.love_method`: ``radial_solver`` integrates the full layered response
    (including the load of the magma layer on the solid below it); ``homogeneous`` is the fast quasi-homogeneous
    approximation, which only sees the solid layer's volume fraction.

    Parameters
    ----------
    params : dict
        Main configuration dictionary containing TOML parameters.
    radius_planet, mass_planet : float
        Planet radius (m) and mass (kg).
    radius_core, mass_core : float
        Core radius (m) and mass (kg).
    density_mantle : float
        Mean mantle density (kg/m^3); the starting guess for the mantle's reference density.
    radius_host, mass_host : float
        Star radius (m) and mass (kg).
    """

    def __init__(self, params, radius_planet, mass_planet, radius_core, mass_core, density_mantle, radius_host,
                 mass_host):
        star_tides   = params['star']['tides']
        planet_tides = params['planet']['tides']
        structure    = planet_tides['structure']

        self.tides_on          = bool(params['simulation']['tides_on'])
        self.grav_constant     = params['constants']['G']
        self.radius_planet     = radius_planet
        self.mass_planet       = mass_planet
        self.radius_core       = radius_core
        self.mass_core         = mass_core
        self.mass_mantle       = mass_planet - mass_core
        self.radius_host       = radius_host
        self.mass_host         = mass_host
        self.rheology_name     = planet_tides['rheology_model']
        self.love_method       = planet_tides['love_method']
        self.min_thickness     = planet_tides['min_solid_layer_thickness']
        self.min_magma_thickness = structure['min_magma_layer_thickness']
        self.obliquity_planet  = planet_tides['obliquity']
        self.obliquity_host    = star_tides['obliquity']
        self.tide_config = dict(
            max_degree_l=planet_tides['max_degree_l'],
            eccentricity_truncation=planet_tides['eccentricity_truncation'],
            obliquity_truncation=planet_tides['obliquity_truncation'],
        )

        # Birch-Murnaghan laws; the reference densities are set by the calibration below.
        density_core = mass_core / ((4.0 / 3.0) * np.pi * radius_core**3)
        self.core_law   = dict(reference_density=density_core, bulk_modulus=structure['core_bulk_modulus'],
                               bulk_modulus_derivative=structure['core_bulk_modulus_derivative'])
        self.mantle_law = dict(reference_density=density_mantle, bulk_modulus=structure['mantle_bulk_modulus'],
                               bulk_modulus_derivative=structure['mantle_bulk_modulus_derivative'])
        self.magma_density_contrast = structure['magma_density_contrast']
        reference_world = self._calibrate_structure()

        # Star: CPL with fixed k2 and Q. A star tidal scale < 1 is folded into Q (dissipation scales with k2 / Q).
        self.star = StarWorld('star', radius_host, mass_host)
        self.star.set_tide_model(make_tide('cpl', {
            'fixed_k': [star_tides['fixed_k2']],
            'fixed_q': [star_tides['fixed_Q'] / star_tides['tidal_scale']],
        }))
        self.star.set_tide_config(**self.tide_config)
        self.moi_host = star_tides['moi_factor'] * mass_host * radius_host**2

        self.orbit_solver = OrbitSolver()

        # The planet's moment of inertia is taken from the fully solid structure; moving the rheological boundary
        # changes it by less than the magma density contrast.
        self.moi_planet = reference_world.get_moment_of_inertia()

    # ==================================================================================================================
    # Structure
    # ==================================================================================================================
    def _calibrate_structure(self):
        """
        Scale the core and mantle reference densities until the solved (fully solid) structure has the model's core
        and mantle masses. Returns the calibrated world.

        Each layer takes a secant step in log space. In a large planet, self-compression makes a layer's mass grow
        faster than its reference density (about twice as fast at 4 Earth masses), so a plain ratio update overshoots
        and cycles.
        """
        laws = (self.core_law, self.mantle_law)
        targets = (self.mass_core, self.mass_mantle)
        previous = None  # (log reference densities, log mass ratios) of the last iteration
        for iteration in range(_CALIBRATION_MAX_ITERS):
            world = LayeredWorld('planet', self.radius_planet, self.mass_planet)
            core = SolidLiquidLayer('core', 0, 0.0, self.radius_core, 0.0, is_tidal=False, is_solid=False)
            core.set_eos(_birch_murnaghan(**self.core_law))
            world.add_layer(core)
            mantle = SolidLiquidLayer('mantle', 1, self.radius_core, self.radius_planet, 0.0)
            mantle.set_eos(_birch_murnaghan(**self.mantle_law))
            world.add_layer(mantle)
            result = world.solve_eos()
            if not result['success']:
                raise RuntimeError(f"TidalPy EOS solve failed while calibrating the planet structure: "
                                   f"{result['message']}")

            ratio_core   = self.mass_core / world.get_layer(0).mass
            ratio_mantle = self.mass_mantle / world.get_layer(1).mass
            if max(abs(ratio_core - 1.0), abs(ratio_mantle - 1.0)) < _CALIBRATION_RTOL:
                log.info(f"Planet structure: core reference density {self.core_law['reference_density']:0.0f} kg/m^3, "
                         f"mantle {self.mantle_law['reference_density']:0.0f} kg/m^3; CMB pressure "
                         f"{world.get_pressure(self.radius_core) / 1e9:0.1f} GPa, central pressure "
                         f"{result['central_pressure'] / 1e9:0.1f} GPa, moment of inertia factor "
                         f"{result['planet_moi'] / (self.mass_planet * self.radius_planet**2):0.4f} "
                         f"({iteration} iterations).")
                return world
            log_densities = [np.log(law['reference_density']) for law in laws]
            log_ratios = [np.log(target / world.get_layer(index).mass) for index, target in enumerate(targets)]
            for index, law in enumerate(laws):
                slope = 1.0  # d ln(mass) / d ln(reference density); exact for an incompressible layer
                if previous is not None and log_densities[index] != previous[0][index]:
                    # log_ratio = ln(target) - ln(mass), so its decrease measures the mass gained
                    slope = ((previous[1][index] - log_ratios[index])
                             / (log_densities[index] - previous[0][index]))
                    slope = min(max(slope, 0.05), 10.0)
                law['reference_density'] = float(np.exp(log_densities[index] + log_ratios[index] / slope))
            previous = (log_densities, log_ratios)

        raise RuntimeError(f"Planet structure calibration did not converge in {_CALIBRATION_MAX_ITERS} iterations "
                           f"(core mass ratio {ratio_core:0.6f}, mantle mass ratio {ratio_mantle:0.6f}). Check "
                           f"planet.tides.structure against the core radius and mass.")

    def _build_planet(self, viscosity_solid, shear_solid, radius_rheological):
        """Build and EOS-solve the layered planet for the current solid-layer properties and extent."""
        world = LayeredWorld('planet', self.radius_planet, self.mass_planet)
        layer_index = 0

        core = SolidLiquidLayer('core', layer_index, 0.0, self.radius_core, 0.0, is_tidal=False, is_solid=False)
        core.set_eos(_birch_murnaghan(**self.core_law))
        world.add_layer(core)
        layer_index += 1

        # The magma layer is dropped when very thin (see min_magma_layer_thickness); the solid layer then reaches the
        # surface. The caller guarantees the solid layer itself is at least min_solid_layer_thickness thick.
        if self.radius_planet - radius_rheological < self.min_magma_thickness:
            radius_rheological = self.radius_planet

        mantle = SolidLiquidLayer('mantle', layer_index, self.radius_core, radius_rheological, 0.0)
        mantle.set_eos(_birch_murnaghan(**self.mantle_law, shear_modulus_static_pa=shear_solid))
        mantle.set_shear_viscosity(make_viscosity('constant', {'reference_viscosity_pas': viscosity_solid}))
        mantle.set_shear_rheology(make_rheology(self.rheology_name))
        world.add_layer(mantle)
        layer_index += 1

        if radius_rheological < self.radius_planet:
            # A static (equilibrium-tide) liquid; its load on the solid below depends on the density contrast.
            magma_law = dict(self.mantle_law)
            magma_law['reference_density'] *= 1.0 - self.magma_density_contrast
            magma = SolidLiquidLayer('magma', layer_index, radius_rheological, self.radius_planet, 0.0,
                                     is_tidal=False, is_solid=False, is_static=True)
            magma.set_eos(_birch_murnaghan(**magma_law))
            world.add_layer(magma)

        world.set_tide_model(make_tide('rheology'))
        world.set_tide_config(love_method=self.love_method, layer_tidal_heating=False, **self.tide_config)
        result = world.solve_eos()
        if not result['success']:
            raise RuntimeError(f"TidalPy EOS solve failed (rheological radius {radius_rheological:0.6e} m): "
                               f"{result['message']}")
        return world

    # ==================================================================================================================
    # Rates
    # ==================================================================================================================
    def calc_rates(self, semi_major_axis, eccentricity, spin_host, spin_planet, viscosity_solid, shear_solid,
                   radius_rheological):
        """
        Tidal heating and orbital and spin derivatives for the current state.

        Parameters
        ----------
        semi_major_axis : float
            Semi-major axis (m).
        eccentricity : float
            Orbital eccentricity. Any value in [0, 1) is valid, including exactly 0.
        spin_host, spin_planet : float
            Spin rates (rad/s).
        viscosity_solid : float
            Dynamic viscosity of the solid-like mantle layer (Pa s).
        shear_solid : float
            Shear modulus of the solid-like mantle layer (Pa).
        radius_rheological : float
            Top of the solid-like mantle layer (m). Below `radius_core + min_solid_layer_thickness` the planet has no
            dissipative layer.

        Returns
        -------
        TidalRates
        """
        if not self.tides_on:
            return _ZERO_RATES

        mass_host   = self.mass_host
        mass_planet = self.mass_planet
        mean_motion = orbital_motion(semi_major_axis, mass_host, mass_planet, self.grav_constant)
        solid_thickness = radius_rheological - self.radius_core

        # dU/dM - dU/dw is passed as its own per-mode sum: at small eccentricity the two separate sums nearly cancel.
        self.star.calc_tides(mean_motion, spin_host, eccentricity, self.obliquity_host, semi_major_axis, mass_planet)
        dUdM_host, dUdw_host, dUdO_host = self.star.get_tidal_potential_derivatives()
        heating_host = self.star.get_tidal_heating()

        da_dt = self.orbit_solver.calc_da_dt(mean_motion, semi_major_axis, eccentricity, mass_host, mass_planet,
                                             dUdM_host)
        de_dt = self.orbit_solver.calc_de_dt(mean_motion, semi_major_axis, eccentricity, mass_host, mass_planet,
                                             dUdM_host, dUdw_host,
                                             dU_dM_minus_dw=self.star.get_tidal_dU_dM_minus_dw())
        dspin_dt_host = mass_planet * dUdO_host / self.moi_host

        if solid_thickness >= self.min_thickness:
            planet = self._build_planet(viscosity_solid, shear_solid, radius_rheological)
            planet.calc_tides(mean_motion, spin_planet, eccentricity, self.obliquity_planet, semi_major_axis,
                              mass_host)
            dUdM_planet, dUdw_planet, dUdO_planet = planet.get_tidal_potential_derivatives()
            heating_planet = planet.get_tidal_heating()
            da_dt += self.orbit_solver.calc_da_dt(mean_motion, semi_major_axis, eccentricity, mass_planet, mass_host,
                                                  dUdM_planet)
            de_dt += self.orbit_solver.calc_de_dt(mean_motion, semi_major_axis, eccentricity, mass_planet, mass_host,
                                                  dUdM_planet, dUdw_planet,
                                                  dU_dM_minus_dw=planet.get_tidal_dU_dM_minus_dw())
            dspin_dt_planet = mass_host * dUdO_planet / self.moi_planet
            tidal_scale_planet = ((radius_rheological**3 - self.radius_core**3) / self.radius_planet**3)
        else:
            heating_planet     = 0.0
            dspin_dt_planet    = 0.0
            tidal_scale_planet = 0.0

        return TidalRates(
            da_dt=float(da_dt),
            de_dt=float(de_dt),
            dspin_dt_host=float(dspin_dt_host),
            dspin_dt_planet=float(dspin_dt_planet),
            tidal_heating_host=float(heating_host),
            tidal_heating_planet=float(heating_planet),
            tidal_scale_planet=float(tidal_scale_planet),
        )
