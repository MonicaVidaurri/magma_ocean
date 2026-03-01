import numpy as np
from .get_meltfrac import get_meltfrac

def get_pressure2(
        temp_mantle,
        radius_solid,
        mass_magma_ocean,
        mass_mantle,
        radius_planet,
        gravity,
        radius_core,
        mass_water_total,
        params):
    """
    Calculates the equilibrium partition of water between a liquid magma ocean 
    and the overlying steam atmosphere using a Newton-Raphson root finder.

    Parameters
    ----------
    temp_mantle : float
        Mantle potential temperature (K).
    radius_solid : float
        Radius of the solidification front (m).
    mass_magma_ocean : float
        Mass of the active magma ocean (kg).
    mass_mantle : float
        Total mass of the mantle (kg).
    radius_planet : float
        Radius of the planet (m).
    gravity : float
        Surface gravity (m/s^2).
    radius_core : float
        Radius of the core (m).
    mass_water_total : float
        Total mass of water in the system (kg).
    params : dict
        Master configuration dictionary.

    Returns
    -------
    pressure_water_vapor, mass_frac_water_liquid, partition_coeff_water
    """

    # --- Unpack Parameters ---
    thermo = params['planet']['thermodynamics']
    vols   = params['planet']['volatiles']
    num    = params['numerical']

    temp_solidus_static   = thermo['temp_solidus_static']
    partition_coeff_water = vols['partition_coeff_water']
    solubility_coeff      = vols['solubility_coeff_water']
    solubility_exponent   = vols['solubility_exponent_water']

    surface_area    = 4.0 * np.pi * radius_planet**2
    mass_atm_factor = surface_area / gravity

    # --- Phase Partitioning ---
    if temp_mantle > temp_solidus_static:
        
        # Get melt fraction (updated to pass params)
        _, volume_melt_fraction = get_meltfrac(
            gravity, temp_mantle, radius_planet, radius_core, mass_mantle, params
        )

        mass_liquid = volume_melt_fraction * mass_magma_ocean
        mass_solid  = (1.0 - volume_melt_fraction) * mass_magma_ocean

        # --- Newton-Raphson Root Finder ---
        bulk_capacity = (partition_coeff_water * mass_solid) + mass_liquid
        inv_exponent  = 1.0 / solubility_exponent

        # Initial guess
        X_current = mass_water_total / mass_mantle

        for _ in range(num['max_nr_iterations']):
            X_safe = max(X_current, 1e-20)

            # P = (X/s)^(1/n)
            pressure_atm = (X_safe / solubility_coeff) ** inv_exponent

            # F(X) = Total_Water - Water_in_Mantle - Water_in_Atmosphere
            f_val = mass_water_total - (X_safe * bulk_capacity) - (mass_atm_factor * pressure_atm)

            # F'(X) derivative
            dp_dx = inv_exponent * (1.0 / solubility_coeff) * ((X_safe / solubility_coeff) ** (inv_exponent - 1.0))
            df_dx = -bulk_capacity - (mass_atm_factor * dp_dx)

            # Newton step
            X_next = X_safe - (f_val / df_dx)
            X_next = max(X_next, 1e-12)

            # Convergence Check
            if abs(X_next - X_safe) / X_safe < num['nr_solver_tolerance']:
                X_current = X_next
                break

            X_current = X_next

        mass_frac_water_liquid = X_current
        pressure_water_vapor   = (mass_frac_water_liquid / solubility_coeff) ** inv_exponent

    else:
        # Mantle is solid
        mass_frac_water_liquid = 0.0
        pressure_water_vapor   = mass_water_total * gravity / surface_area
        partition_coeff_water  = 0.0

    return pressure_water_vapor, mass_frac_water_liquid, partition_coeff_water
