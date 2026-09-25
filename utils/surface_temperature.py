"""Quasi-static surface temperature from the balance of interior heat flow and outgoing radiation."""
from collections import namedtuple

import numpy as np
from scipy.optimize import brentq

from utils.get_flux import get_flux

SurfaceState = namedtuple('SurfaceState', (
    'temp_surface',        # Surface temperature (K)
    'heat_flux',           # Interior heat flux = net outgoing flux (W/m^2)
    'pressure_H2O',        # Water vapor pressure at the surface after condensation (Pa)
))


def saturation_vapor_pressure(temp_surface, atm_params):
    """Water saturation vapor pressure (Pa) from the Antoine-type relation used by the model."""
    return 10.0 ** (atm_params['vapor_press_a'] - atm_params['vapor_press_b'] / temp_surface) * 1e5


def water_vapor_pressure(temp_surface, pressure_uncondensed, atm_params):
    """
    Water vapor pressure (Pa) after condensation of any excess into a surface ocean.

    Below the critical temperature the vapor is capped at saturation. A tanh blend of 20 K half-width around the
    critical temperature keeps the cap smooth.

    Parameters
    ----------
    temp_surface : float
        Surface temperature (K).
    pressure_uncondensed : float
        Pressure (Pa) that the non-dissolved water would exert if it were all vapor.
    atm_params : dict
        ``params['planet']['atmosphere']``.
    """
    pressure_capped = min(saturation_vapor_pressure(temp_surface, atm_params), pressure_uncondensed)
    blend = 0.5 * (1.0 + np.tanh((temp_surface - atm_params['critical_temp_H2O']) / 20.0))
    return blend * pressure_uncondensed + (1.0 - blend) * pressure_capped


def solve_surface_temperature(temp_mantle, convective_coefficient, temp_equilibrium, pressure_H2O_uncondensed,
                              pressure_O2, radius_planet, gravity, params):
    """
    Find the surface temperature at which interior heat flow equals the net outgoing thermal flux.

    The interior flux follows boundary-layer convection, ``q = c * (T_m - T_s)^(4/3)`` (the Ra^(1/3) Nusselt scaling;
    see physics/mantleheatflux.py). The outgoing flux is the gray-atmosphere expression of utils/get_flux.py, with
    water vapor limited by condensation. The atmosphere and surface are assumed to have negligible heat capacity
    compared with the mantle, as in Schaefer et al. (2016).

    With condensation the balance can have several roots (runaway greenhouse bistability). The hottest root is
    returned: a planet cooling from a molten state stays on the hot branch until that branch disappears, which is the
    hysteresis a surface with heat capacity would follow. Roots are bracketed on a grid in log(T_m - T_s), which
    resolves the thin boundary layers of a liquid magma ocean, and refined with Brent's method.

    Parameters
    ----------
    temp_mantle : float
        Mantle potential temperature (K).
    convective_coefficient : float
        ``c`` in ``q = c * (T_m - T_s)^(4/3)`` (W/m^2/K^(4/3)).
    temp_equilibrium : float
        Radiative equilibrium temperature from instellation (K).
    pressure_H2O_uncondensed : float
        Water pressure (Pa) if all non-dissolved water were vapor.
    pressure_O2 : float
        Oxygen partial pressure (Pa).
    radius_planet, gravity : float
        Planet radius (m) and surface gravity (m/s^2).
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    SurfaceState
    """
    atm_params = params['planet']['atmosphere']
    num_params = params['numerical']

    def state_at(temp_surface):
        pressure_H2O = water_vapor_pressure(temp_surface, pressure_H2O_uncondensed, atm_params)
        outgoing     = get_flux(temp_surface, temp_equilibrium, pressure_H2O, pressure_O2, radius_planet, gravity,
                                params)
        return pressure_H2O, outgoing

    def residual(temp_surface):
        interior = convective_coefficient * max(temp_mantle - temp_surface, 0.0)**(4.0 / 3.0)
        return interior - state_at(temp_surface)[1]

    if temp_mantle <= temp_equilibrium:
        # The mantle cannot lose heat through a surface warmer than itself; the surface sits at T_eq.
        temp_surface = temp_equilibrium
    else:
        # Scan from the hottest candidate (smallest T_m - T_s) toward T_eq and refine the first sign change.
        # residual(T_m) < 0 and residual(T_eq) > 0, so a sign change always exists.
        span = temp_mantle - temp_equilibrium
        smallest_gap = min(num_params['surface_temp_xtol'], 1e-3 * span)
        gaps = np.concatenate(([0.0], np.geomspace(smallest_gap, span, num_params['surface_temp_scan_points'])))
        candidates = temp_mantle - gaps
        candidates[-1] = temp_equilibrium
        previous_temp, previous_residual = candidates[0], residual(candidates[0])
        temp_surface = temp_equilibrium
        for candidate in candidates[1:]:
            candidate_residual = residual(candidate)
            if candidate_residual >= 0.0 > previous_residual:
                temp_surface = brentq(residual, candidate, previous_temp, xtol=num_params['surface_temp_xtol'],
                                      rtol=1e-12)
                break
            previous_temp, previous_residual = candidate, candidate_residual

    pressure_H2O, outgoing = state_at(temp_surface)
    return SurfaceState(temp_surface=temp_surface, heat_flux=outgoing, pressure_H2O=pressure_H2O)
