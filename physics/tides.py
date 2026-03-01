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
            tides_on_flag):

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
        tidal_scale_h = 1.0  # Amount of planet participating (full planet)
        tidal_scale_p = (radius_planet - radius_planet_core) / radius_planet  # Amount of planet participating in tides (just mantle)
        visc_h = 1.0e10  # star's viscosity (not used)
        visc_p = kinematic_viscosity_solid * rho_planet_solid  # the viscosity above is kinetic; we want dynamic so multiple by the layer's density.'
        shear_mod_h = 1.0e7  # Star shear (not used)
        shear_mod_p = 60.0e9
        rheology_h = 'cpl'  # Constant phase lag model
        rheology_p = 'Andrade'
        fixed_k2_h = 0.3
        fixed_k2_p = 0.3  # ukinematic_viscosity_solidsed for target planet due to real radius_hosteology
        fixed_Q_h = 30000000
        fixed_Q_p = 30000000 # ukinematic_viscosity_solidsed for target planet due to real radius_hosteology

        # Partial melting parameters
        shear_mod_p_liquid = 1.0e-5
        crit_melt_frac = 0.5
        crit_melt_frac_width = 0.05
        hn_visc_slope_1 = 13.5
        hn_visc_slope_2 = 370.0
        hn_shear_param_1 = 40000.0
        hn_shear_param_2 = 25.0
        hn_shear_falloff_slope = 700.0

        # TODO: this should be an inverse process to find T from melt_frac via inverting get_meltfraction_planet_solidb()
        #   for now just leave it as halfway between a typical solidus and liquidus
        break_down_temass_planet = 1420.0 + crit_melt_frac * (1825.0 - 1420.0)

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
        obliquity_h = 0.0
        obliquity_p = 0.0

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