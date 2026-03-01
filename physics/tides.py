from math import isclose
import numpy as np

from TidalPy.rheology import Andrade, Elastic, Newton
from TidalPy.toolbox import quick_dual_body_tidal_dissipation
from TidalPy.constants import G


def calculate_tidal_dissipation(
        eccentricity, orbital_frequency, spin_freq_planet, spin_freq_host,
        radius_planet, radius_host, mass_planet, mass_host,
        temperature_planet_solid, radius_planet_core, kinematic_viscosity_solid,
        rho_planet_solid, meltfraction_planet_solid,
        tides_on_flag, params):

    if tides_on_flag:

        # --- Unpack Parameters ---
        star_tides = params['star']['tides']
        planet_tides = params['planet']['tides']

        # Unpack other dependent variables
        g_p = G * mass_planet / (radius_planet**2)
        g_h = G * mass_host / (radius_host**2)
        density_bulk_planet = mass_planet / ((4.0 / 3.0) * np.pi * radius_planet**3)
        density_bulk_host = mass_host / ((4.0 / 3.0) * np.pi * radius_host**3)

        # For now just use basic MOI for now
        moi_h = (2.0 / 5.0) * mass_host * radius_host**2
        moi_p = (2.0 / 5.0) * mass_planet * radius_planet**2

        # Tidally active region viscosity
        tidal_scale_h = star_tides['tidal_scale']  # Amount of planet participating (full planet)
        tidal_scale_p = (radius_planet - radius_planet_core) / radius_planet  # Amount of planet participating in tides (just mantle)
        visc_h = star_tides['viscosity']  # star's viscosity (not used)
        visc_p = kinematic_viscosity_solid * rho_planet_solid  # the viscosity above is kinetic; we want dynamic so multiple by the layer's density.'
        shear_mod_h = star_tides['shear_modulus']  # Star shear (not used)
        shear_mod_p = planet_tides['shear_modulus_solid']
        rheology_h = star_tides['rheology_model']  # Constant phase lag model
        rheology_p = planet_tides['rheology_model']
        fixed_k2_h = star_tides['fixed_k2']
        fixed_k2_p = planet_tides['fixed_k2']  # unused for target planet due to real rheology
        fixed_Q_h = star_tides['fixed_Q']
        fixed_Q_p = planet_tides['fixed_Q'] # unused for target planet due to real rheology

        # Partial melting parameters
        shear_mod_p_liquid = planet_tides['shear_modulus_liquid_limit']
        crit_melt_frac = planet_tides['crit_melt_frac']
        crit_melt_frac_width = planet_tides['melt_transition_width']
        
        # Unused in original logic but left intact
        hn_visc_slope_1 = 13.5
        hn_visc_slope_2 = 370.0
        
        hn_shear_param_1 = planet_tides['shear_melt_param_1']
        hn_shear_param_2 = planet_tides['shear_melt_param_2']
        hn_shear_falloff_slope = planet_tides['shear_melt_falloff_slope']

        # TODO: this should be an inverse process to find T from melt_frac via inverting get_meltfraction_planet_solidb()
        #   for now just leave it as halfway between a typical solidus and liquidus
        temp_sol_ref = planet_tides['temp_solidus_ref']
        temp_liq_ref = planet_tides['temp_liquidus_ref']
        break_down_temass_planet = temp_sol_ref + crit_melt_frac * (temp_liq_ref - temp_sol_ref)

        # TODO: Handle partial melting of viscosity? Or is that handled above?
        # Handle partial melting of shear
        if meltfraction_planet_solid <= 0.0:
            # No change from the solid case
            pass
        elif (meltfraction_planet_solid > 0.0) and (meltfraction_planet_solid < crit_melt_frac):
            shear_mod_p = shear_mod_p * np.exp((hn_shear_param_1 / temperature_planet_solid) - hn_shear_param_2)
        elif (meltfraction_planet_solid >= crit_melt_frac) and (meltfraction_planet_solid <= (crit_melt_frac + crit_melt_frac_width)):
            shear_mod_p = shear_mod_p * np.exp((hn_shear_param_1 / break_down_temass_planet) - hn_shear_param_2) * \
                np.exp(-hn_shear_falloff_slope * (meltfraction_planet_solid - crit_melt_frac))
        else:
            # meltfraction_planet_solid > crit_melt_frac + crit_melt_frac_width
            shear_mod_p = shear_mod_p_liquid

        # Check for overshoots
        if shear_mod_p < shear_mod_p_liquid:
            shear_mod_p = shear_mod_p_liquid

        # Assume obliquity is off for now
        obliquity_h = star_tides['obliquity']
        obliquity_p = planet_tides['obliquity']

        dissipation_results = quick_dual_body_tidal_dissipation(
            radii=(radius_host, radius_planet),
            masses=(mass_host, mass_planet),
            gravities=(g_h, g_p),
            densities=(density_bulk_host, density_bulk_planet),
            mois=(moi_h, moi_p),
            viscosities=(visc_h, visc_p),
            shear_moduli=(shear_mod_h, shear_mod_p),
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
        tidal_heating_p
    )
