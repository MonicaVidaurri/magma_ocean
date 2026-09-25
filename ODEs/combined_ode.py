"""
Continuous coupled thermal, volatile, orbital, and rotational evolution of a magma ocean planet.

One right-hand side covers the whole evolution, from a fully molten mantle to a solid one and back, with no phase
switching. Every quantity that depended on the old "magma ocean / wet solid / dry solid" phases is now a continuous
function of the mantle potential temperature through the melt distribution of an adiabatic mantle (see
utils/get_melt_fractions.py and model.tex).
"""
from collections import namedtuple

import numpy as np

from physics.degas2 import degas2
from physics.mantleheatflux import mantleheatflux
from physics.radiogenics import get_radiogenic_heat
from physics.shear_modulus import calc_shear_modulus
from physics.tides import TidalModel, orbital_motion
from physics.viscosity import viscosity
from utils.config import resolve_path
from utils.get_comp import get_comp
from utils.get_loss import get_loss
from utils.get_massbalance4 import get_massbalance4
from utils.get_melt_fractions import calc_melt_state
from utils.get_pressure2 import get_pressure2
from utils.logger import get_logger
from utils.mantle_grid import build_mantle_grid
from utils.stellar import StellarTrack
from utils.surface_temperature import solve_surface_temperature

log = get_logger(__name__)

# ======================================================================================================================
# State Vector Layout
# ======================================================================================================================
STATE_NAMES = (
    'semi_major_axis',     # [0] m
    'eccentricity',        # [1]
    'spin_freq_host',      # [2] rad/s
    'spin_freq_planet',    # [3] rad/s
    'temp_mantle',         # [4] Mantle potential temperature, K
    'mass_water_solid',    # [5] Water in the mantle below the melt region, kg
    'mass_water_fluid',    # [6] Water in the melt region + atmosphere + surface ocean, kg
    'mass_oxygen_fluid',   # [7] Excess oxygen (FeO1.5 in the melt region + atmospheric O2), kg
    'mass_oxygen_solid',   # [8] Excess oxygen locked in FeO1.5 of the solid mantle, kg
)
NUM_STATES = len(STATE_NAMES)

PlanetProperties = namedtuple('PlanetProperties', (
    'radius', 'mass', 'radius_core', 'mass_core', 'mass_mantle', 'density_mantle', 'gravity', 'surface_area',
    'mass_water_initial', 'mass_host', 'radius_host'))


class MagmaOceanModel:
    """
    Physical model of one planet: static setup plus the right-hand side of the evolution equations.

    Parameters
    ----------
    params : dict
        Merged configuration from `utils.config.load_config`.
    """

    def __init__(self, params):
        self.params = params
        constants = params['constants']
        star      = params['star']
        planet    = params['planet']

        radius       = planet['radius_planet_relative'] * constants['radius_earth']
        mass         = planet['mass_planet_relative'] * constants['mass_earth']
        radius_core  = planet['radius_core_relative'] * radius
        mass_core    = planet['core_mass_fraction'] * mass
        mass_mantle  = mass - mass_core
        density      = mass_mantle / ((4.0 / 3.0) * np.pi * (radius**3 - radius_core**3))
        gravity      = constants['G'] * mass / radius**2
        self.planet  = PlanetProperties(
            radius=radius, mass=mass, radius_core=radius_core, mass_core=mass_core, mass_mantle=mass_mantle,
            density_mantle=density, gravity=gravity, surface_area=4.0 * np.pi * radius**2,
            mass_water_initial=planet['ocean_mass_multiplier'] * constants['mass_ocean_earth'],
            mass_host=star['mass_star_relative'] * constants['mass_sun'], radius_host=star['host_radius'])

        self.grid = build_mantle_grid(radius, radius_core, gravity, mass_mantle, params)
        self.critical_melt_fraction = 1.0 - planet['rheology']['critical_crystal_fraction']
        self.fixed_eccentricity = bool(planet['orbit']['fixed_eccentricity'])
        self.fixed_spin = bool(planet['fixed_spin'])
        self.stellar_track = StellarTrack(resolve_path(star['stellar_track_file']), constants['lum_sun'])
        self.tidal_model = TidalModel(params, radius, mass, radius_core, mass_core, density,
                                      self.planet.radius_host, self.planet.mass_host)
        self.composition = get_comp(params)
        self.mass_frac_FeO_total = planet['oxide_composition']['mass_frac_FeO_total']

        if planet['initial_mantle_temp'] > self.grid.max_monotonic_temperature:
            log.warning(f"Initial mantle temperature {planet['initial_mantle_temp']:.0f} K exceeds "
                        f"{self.grid.max_monotonic_temperature:.0f} K, above which melt fraction is no longer "
                        f"monotonic with depth. Melt-region boundaries will be approximate.")

    # ==================================================================================================================
    # Initial Conditions
    # ==================================================================================================================
    def initial_state(self):
        """
        Initial state vector from the configuration.

        Water starts uniformly mixed in the mantle; the melt region's share is fluid, the rest is solid. Initial
        excess oxygen follows the configured Fe3+/Fe ratio in the same proportions.
        """
        params    = self.params
        constants = params['constants']
        planet    = params['planet']
        comp      = planet['oxide_composition']
        props     = self.planet

        semi_major_axis = planet['orbit']['initial_semi_major_axis'] * constants['au']
        mean_motion     = orbital_motion(semi_major_axis, props.mass_host, props.mass, constants['G'])
        temp_mantle     = planet['initial_mantle_temp']

        melt = calc_melt_state(temp_mantle, self.grid, self.critical_melt_fraction)
        mass_melt_region = melt.mass_melt_region
        log.info(f"Initial melt region depth {(props.radius - melt.radius_solidus) / 1e3:0.1f} km, "
                 f"bulk melt fraction {melt.melt_fraction_bulk:0.3f}.")

        water_mass_fraction = props.mass_water_initial / props.mass_mantle
        oxygen_per_mantle_mass = (comp['mass_frac_FeO_total'] * comp['ratio_Fe3_to_total_Fe']
                                  * (constants['molar_mass_O'] / 2.0 / comp['molar_mass_FeO']))

        state = np.zeros(NUM_STATES, dtype=np.float64)
        state[0] = semi_major_axis
        state[1] = planet['orbit']['initial_eccentricity']
        state[2] = 2.0 * np.pi / (86400.0 * params['star']['spin_days'])
        state[3] = planet['initial_spin_multiplier'] * mean_motion
        state[4] = temp_mantle
        state[5] = water_mass_fraction * (props.mass_mantle - mass_melt_region)
        state[6] = water_mass_fraction * mass_melt_region
        state[7] = oxygen_per_mantle_mass * mass_melt_region
        state[8] = oxygen_per_mantle_mass * (props.mass_mantle - mass_melt_region)
        return state

    # ==================================================================================================================
    # Right-Hand Side
    # ==================================================================================================================
    def rhs(self, t_sec, state):
        """Time derivative of the state vector (SI units)."""
        return self.evaluate(t_sec, state)[0]

    def evaluate(self, t_sec, state):
        """
        Time derivative of the state vector plus every diagnostic used to compute it.

        Parameters
        ----------
        t_sec : float
            Time since formation; also the stellar age used for the luminosity track (s).
        state : ndarray
            State vector (see STATE_NAMES).

        Returns
        -------
        dstate_dt : ndarray
        diagnostics : dict
        """
        params    = self.params
        constants = params['constants']
        planet    = params['planet']
        thermo    = planet['thermodynamics']
        comp      = planet['oxide_composition']
        props     = self.planet

        radius       = props.radius
        gravity      = props.gravity
        surface_area = props.surface_area
        density      = props.density_mantle
        mass_mantle  = props.mass_mantle

        semi_major_axis = state[0]
        # The integrator may probe slightly outside physical bounds between accepted steps. Fluid inventories feed
        # nonlinear equilibria that need non-negative input. Solid inventories only enter sinks that are linear in
        # them, so they stay signed: a small overshoot below zero then relaxes back instead of freezing in place.
        eccentricity    = max(state[1], 0.0)
        spin_host       = state[2]
        spin_planet     = state[3]
        temp_mantle     = state[4]
        water_solid     = state[5]
        water_fluid     = max(state[6], 0.0)
        oxygen_fluid    = max(state[7], 0.0)
        oxygen_solid    = state[8]

        # --- Melt distribution ---
        melt = calc_melt_state(temp_mantle, self.grid, self.critical_melt_fraction)
        mass_melt_region  = melt.mass_melt_region
        mass_crystals_mr  = max(mass_melt_region - melt.mass_melt, 0.0)
        mass_solid_mantle = melt.mass_below_solidus
        water_frac_solid  = water_solid / mass_solid_mantle if mass_solid_mantle > 0.0 else 0.0

        # --- Star ---
        age_years       = t_sec / constants['seconds_per_year']
        luminosity_bol  = self.stellar_track.luminosity(age_years)
        instellation    = luminosity_bol / (4.0 * np.pi * semi_major_axis**2)
        temp_equilibrium = ((1.0 - planet['albedo']) * instellation / (4.0 * constants['stefan_boltzmann']))**0.25

        # --- Volatile equilibrium in the melt region ---
        pressure_H2O_eq, water_frac_melt, partition_coeff_H2O = get_pressure2(
            melt.mass_melt, mass_crystals_mr, mass_mantle, radius, gravity, water_fluid, params)

        if oxygen_fluid > 0.0 and mass_melt_region > 0.0:
            pressure_O2, mass_frac_FeO1_5, _, _ = get_massbalance4(
                temp_mantle, pressure_H2O_eq, mass_melt_region, oxygen_fluid, self.composition,
                self.mass_frac_FeO_total, gravity, radius, params)
            if pressure_O2 < 0.0:
                pressure_O2      = oxygen_fluid * gravity / surface_area
                mass_frac_FeO1_5 = 0.0
        else:
            pressure_O2      = oxygen_fluid * gravity / surface_area
            mass_frac_FeO1_5 = 0.0

        # --- Surface temperature (quasi-static flux balance) ---
        # q = c (T_m - T_s)^(4/3). The convective viscosity uses the bulk melt fraction: it is monotonic in T_m and
        # reproduces the old model in both limits (the old magma ocean and solid phases). The mean melt fraction of the
        # melt region is not monotonic in T_m, so using it creates a spurious cooling feedback.
        heat_flux_ref, _, _, _, viscosity_convective = mantleheatflux(
            temp_mantle, 0.0, props.radius_core, radius, props.radius_core, gravity, density, water_frac_solid,
            melt.melt_fraction_bulk, params)
        convective_coefficient = heat_flux_ref / temp_mantle**(4.0 / 3.0)
        surface = solve_surface_temperature(temp_mantle, convective_coefficient, temp_equilibrium, pressure_H2O_eq,
                                            pressure_O2, radius, gravity, params)
        temp_surface = surface.temp_surface
        pressure_H2O = surface.pressure_H2O

        # --- Water after condensation ---
        # Condensation splits the undissolved water between vapor and a surface ocean. The melt stays in equilibrium
        # with the total water load (vapor + ocean), so the dissolved fraction from get_pressure2 is unchanged.
        water_dissolved = water_frac_melt * (melt.mass_melt + partition_coeff_H2O * mass_crystals_mr)
        water_vapor     = pressure_H2O * surface_area / gravity
        water_ocean     = max(water_fluid - water_dissolved - water_vapor, 0.0)

        # --- Solid-layer rheology and tides ---
        melt_frac_solid_layer = melt.melt_fraction_solid_layer
        shear_solid     = calc_shear_modulus(melt_frac_solid_layer, params)
        viscosity_solid = viscosity(temp_mantle, temp_surface, density, water_frac_solid, melt_frac_solid_layer,
                                    params) * density
        tides = self.tidal_model.calc_rates(semi_major_axis, eccentricity, spin_host, spin_planet, viscosity_solid,
                                            shear_solid, melt.radius_rheological)

        # --- Mantle energy balance ---
        radiogenic_heating = get_radiogenic_heat(t_sec, mass_mantle, params)
        surface_heat_loss  = surface_area * surface.heat_flux
        heat_capacity = (thermo['specific_heat_mantle'] * self.grid.mass_adiabat
                         + thermo['latent_heat_fusion'] * melt.dmass_melt_dtemp)
        dtemp_mantle_dt = (radiogenic_heating + tides.tidal_heating_planet - surface_heat_loss) / heat_capacity

        # --- Exchange across the base of the melt region ---
        # Mass leaving the melt region (positive when the solidus front rises during cooling).
        mass_solidifying_rate = (4.0 * np.pi * density * melt.radius_solidus**2 * melt.dradius_solidus_dtemp
                                 * dtemp_mantle_dt)
        if mass_solidifying_rate >= 0.0:
            # New cumulate carries crystal-hosted water and all FeO1.5 of the melt region at its current fraction.
            water_to_solid  = partition_coeff_H2O * water_frac_melt * mass_solidifying_rate
            oxygen_to_solid = (mass_frac_FeO1_5 * 0.5 * (constants['molar_mass_O'] / comp['molar_mass_FeO1_5'])
                               * mass_solidifying_rate)
        elif mass_solid_mantle > 0.0:
            # Remelting returns solid-mantle volatiles at the solid's mean concentration.
            water_to_solid  = water_frac_solid * mass_solidifying_rate
            oxygen_to_solid = (oxygen_solid / mass_solid_mantle) * mass_solidifying_rate
        else:
            water_to_solid  = 0.0
            oxygen_to_solid = 0.0

        # --- Solid-state degassing (convective processing of the solid mantle through its melting zone) ---
        heat_flux_solid, boundary_layer_solid, velocity_solid, _, _ = mantleheatflux(
            temp_mantle, temp_surface, props.radius_core, radius, props.radius_core, gravity, density,
            water_frac_solid, melt_frac_solid_layer, params)
        if water_frac_solid != 0.0:
            # Only melt in the solid-like layer degasses this way; the liquid layer's melt is already equilibrated.
            _, degas_rate, _, _ = degas2(temp_mantle, boundary_layer_solid, heat_flux_solid, water_frac_solid, radius,
                                         gravity, temp_surface, density, radius - melt.radius_rheological, params)
            degassing = degas_rate * surface_area * velocity_solid
        else:
            degassing = 0.0

        # --- Escape ---
        flux_loss_H, flux_loss_O = get_loss(t_sec, luminosity_bol, pressure_O2, pressure_H2O, semi_major_axis,
                                            props.mass, radius, temp_surface, params)
        # Escape cannot remove more water than the planet's fluid reservoir holds. Scale it by W / (|W| + W_taper),
        # where W_taper is the water column of `taper_pressure`: ~1 for any real inventory, linear through zero, and
        # odd, so a small integrator overshoot below zero is restored smoothly rather than left in a dead zone.
        water_taper   = planet['atmosphere']['escape']['taper_pressure'] * surface_area / gravity
        supply_factor = state[6] / (abs(state[6]) + water_taper)
        hydrogen_loss = surface_area * flux_loss_H * supply_factor
        oxygen_loss   = surface_area * flux_loss_O * supply_factor
        water_loss        = hydrogen_loss * (constants['molar_mass_H2O'] / (2.0 * constants['molar_mass_H']))
        oxygen_from_water = hydrogen_loss * (constants['molar_mass_O'] / (2.0 * constants['molar_mass_H']))

        # --- Assemble ---
        dstate_dt = np.empty(NUM_STATES, dtype=np.float64)
        dstate_dt[0] = tides.da_dt
        # A fixed eccentricity stands in for forcing by other planets (e.g., a mean-motion resonance), which is not
        # modeled. The semi-major axis still evolves, so the tidal heat is still drawn from the orbit.
        dstate_dt[1] = 0.0 if self.fixed_eccentricity else tides.de_dt
        dstate_dt[2] = tides.dspin_dt_host
        # A fixed spin stands in for a torque that is not modeled (e.g., atmospheric thermal tides) holding the planet
        # out of tidal spin equilibrium; the energy that torque supplies is not tracked.
        dstate_dt[3] = 0.0 if self.fixed_spin else tides.dspin_dt_planet
        dstate_dt[4] = dtemp_mantle_dt
        dstate_dt[5] = water_to_solid - degassing
        dstate_dt[6] = -water_to_solid + degassing - water_loss
        dstate_dt[7] = oxygen_from_water - oxygen_loss - oxygen_to_solid
        dstate_dt[8] = oxygen_to_solid

        diagnostics = dict(
            temp_surface=temp_surface,
            temp_equilibrium=temp_equilibrium,
            luminosity_bol=luminosity_bol,
            radius_solid=melt.radius_solidus,
            radius_rheological=melt.radius_rheological,
            meltfrac=melt.melt_fraction_bulk,
            meltfrac_surface=melt.melt_fraction_surface,
            meltfrac_melt_region=melt.melt_fraction_melt_region,
            meltfrac_solid_layer=melt_frac_solid_layer,
            mass_melt=melt.mass_melt,
            mass_melt_region=mass_melt_region,
            Patm=pressure_H2O,
            PO2=pressure_O2,
            mass_water_dissolved=water_dissolved,
            mass_water_vapor=water_vapor,
            mass_water_ocean=water_ocean,
            water_frac_melt=water_frac_melt,
            tidal_shear=shear_solid,
            tidal_visc=viscosity_solid,
            tidal_scale=tides.tidal_scale_planet,
            convective_viscosity=viscosity_convective * density,
            Q_tid=tides.tidal_heating_planet,
            Q_tid_host=tides.tidal_heating_host,
            Q_rad=radiogenic_heating,
            Q_surface=surface_heat_loss,
            heat_capacity=heat_capacity,
            degassing_rate=degassing,
            water_to_solid_rate=water_to_solid,
            water_loss_rate=water_loss,
            oxygen_loss_rate=oxygen_loss,
            orbital_freq=orbital_motion(semi_major_axis, props.mass_host, props.mass, constants['G']),
        )
        return dstate_dt, diagnostics
