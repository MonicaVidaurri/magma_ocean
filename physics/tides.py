from math import isclose
import numpy as np

from TidalPy.toolbox import quick_dual_body_tidal_dissipation
from TidalPy.constants import G


def calculate_tidal_dissipation(
        eccentricity, orbital_frequency, spin_freq_planet, spin_freq_host,
        radius_planet, radius_host, mass_planet, mass_host,
        radius_planet_core, solid_viscosity, solid_shear,
        solid_radius_p,
        tides_on_flag, params):

    # --- Unpack Parameters ---
    star_tides = params['star']['tides']
    planet_tides = params['planet']['tides']

    # --- Find Tidal Scales ---
    tidal_scale_h = star_tides['tidal_scale']  # Amount of planet participating (full planet)
    # Amount of planet participating in tides (just mantle)
    volume_planet = (4.0 / 3.0) * np.pi * radius_planet**3
    volume_solid_mantle_p = (4.0 / 3.0) * np.pi * (solid_radius_p**3 - radius_planet_core**3)

    tidal_scale_p = volume_solid_mantle_p / volume_planet

    # # Fix overshoots
    # eccentricity = np.clip(eccentricity, 0.0, 0.8)
    # solid_viscosity = np.clip(solid_viscosity, 1.0, 1.0e35)
    # solid_shear = np.clip(solid_shear, 1.0, 1.0e12)

    if eccentricity < 0.001 and isclose(orbital_frequency, spin_freq_planet):
        spin_freq_planet = orbital_frequency

    if tides_on_flag:

        # Unpack other dependent variables
        g_p = G * mass_planet / (radius_planet**2)
        g_h = G * mass_host / (radius_host**2)
        density_bulk_planet = mass_planet / ((4.0 / 3.0) * np.pi * radius_planet**3)
        density_bulk_host = mass_host / ((4.0 / 3.0) * np.pi * radius_host**3)

        # For now just use basic MOI for now
        moi_h = (2.0 / 5.0) * mass_host * radius_host**2
        moi_p = (2.0 / 5.0) * mass_planet * radius_planet**2

        # Tidally active region viscosity
        visc_h      = star_tides['viscosity']  # star's viscosity (not used)
        shear_mod_h = star_tides['shear_modulus']  # Star shear (not used)
        rheology_h  = star_tides['rheology_model']  # Constant phase lag model
        rheology_p  = planet_tides['rheology_model']
        fixed_k2_h  = star_tides['fixed_k2']
        fixed_k2_p  = planet_tides['fixed_k2']  # unused for target planet due to real rheology
        fixed_Q_h   = star_tides['fixed_Q']
        fixed_Q_p   = planet_tides['fixed_Q'] # unused for target planet due to real rheology

        # Assume obliquity is off for now
        obliquity_h = star_tides['obliquity']
        obliquity_p = planet_tides['obliquity']

        dissipation_results = quick_dual_body_tidal_dissipation(
            radii=(radius_host, radius_planet),
            masses=(mass_host, mass_planet),
            gravities=(g_h, g_p),
            densities=(density_bulk_host, density_bulk_planet),
            mois=(moi_h, moi_p),
            viscosities=(visc_h, solid_viscosity),
            shear_moduli=(shear_mod_h, solid_shear),
            rheologies=(rheology_h, rheology_p),
            complex_compliance_inputs=None,
            obliquities=(obliquity_h, obliquity_p),
            spin_frequencies=(spin_freq_host, spin_freq_planet), 
            tidal_scales=(tidal_scale_h, tidal_scale_p),
            fixed_k2s=(fixed_k2_h, fixed_k2_p),
            fixed_qs=(fixed_Q_h, fixed_Q_p),
            eccentricity=eccentricity,
            orbital_frequency=orbital_frequency,
            max_tidal_order_l=2,
            eccentricity_truncation_lvl=10,
            use_obliquity=(not isclose(obliquity_p, 0.0)),
            da_dt_scale=1.,
            de_dt_scale=1.,
            dspin_dt_scale=1.
            )
        da_dt           = dissipation_results['semi_major_axis_derivative']
        de_dt           = dissipation_results['eccentricity_derivative']
        dspin_dt_h      = dissipation_results['host']['spin_rate_derivative']
        dspin_dt_p      = dissipation_results['secondary']['spin_rate_derivative']
        tidal_heating_h = dissipation_results['host']['tidal_heating']
        tidal_heating_p = dissipation_results['secondary']['tidal_heating']
    else:
        da_dt           = 0.0
        de_dt           = 0.0
        dspin_dt_h      = 0.0
        dspin_dt_p      = 0.0
        tidal_heating_h = 0.0
        tidal_heating_p = 0.0
    
    return (
        da_dt,
        de_dt,
        dspin_dt_h,
        dspin_dt_p,
        tidal_heating_h,
        tidal_heating_p,
        tidal_scale_p,
        solid_shear,
        solid_viscosity
    )
