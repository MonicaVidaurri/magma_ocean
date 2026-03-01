import numpy as np
from .get_meltfrac import get_meltfrac

def get_pressure2b(
        temp_mantle,
        radius_solid,
        mass_magma_ocean,
        mass_mantle,
        radius_planet,
        gravity,
        radius_core,
        mass_water_total):
    """
    Calculates the equilibrium partition of water between a liquid magma ocean 
    and the overlying steam atmosphere using a Newton-Raphson root finder (version b).

    Parameters
    ----------
    temp_mantle : float
        Mantle potential temperature in Kelvin.
    radius_solid : float
        Radius of the solidification front (base of the magma ocean) in meters.
    mass_magma_ocean : float
        Mass of the active magma ocean in kg.
    mass_mantle : float
        Total mass of the mantle in kg.
    radius_planet : float
        Radius of the planet in meters.
    gravity : float
        Surface gravitational acceleration in m/s^2.
    radius_core : float
        Radius of the planetary core in meters.
    mass_water_total : float
        Total mass of water in the magma ocean + atmosphere system in kg.

    Returns
    -------
    pressure_water_vapor : float
        Equilibrium atmospheric water vapor pressure in Pascals.
    mass_frac_water_liquid : float
        Mass fraction of water dissolved in the liquid magma.
    partition_coeff_water : float
        Partition coefficient for water between solid and melt.
    """

    # --- Constants & Solubility Parameters ---
    temp_solidus_static   = 1373.0  # K
    partition_coeff_water = 0.01  # kH2O (distribution between solid and melt)

    # Empirical water solubility law parameters: X_water = s * P^n
    solubility_coeff    = 3.44e-8
    solubility_exponent = 0.74

    surface_area    = 4.0 * np.pi * radius_planet**2
    mass_atm_factor = surface_area / gravity

    # --- Phase Partitioning ---
    if temp_mantle > temp_solidus_static:
        
        # Get melt fraction using the robust integration function
        _, volume_melt_fraction = get_meltfrac(gravity, temp_mantle, radius_planet, radius_core, mass_mantle)

        mass_liquid = volume_melt_fraction * mass_magma_ocean
        mass_solid  = (1.0 - volume_melt_fraction) * mass_magma_ocean

        # --- Newton-Raphson Root Finder ---
        bulk_capacity = (partition_coeff_water * mass_solid) + mass_liquid
        inv_exponent = 1.0 / solubility_exponent

        # Initial guess: assume all water is evenly mixed in the mantle
        X_current = mass_water_total / mass_mantle

        # Using 30 iterations to mimic original version 'b' limits, 
        # but with robust relative convergence.
        for _ in range(30):
            # Protect against negative guesses in fractional powers
            X_safe = max(X_current, 1e-20)

            # Atmospheric pressure: P = (X/s)^(1/n)
            pressure_atm = (X_safe / solubility_coeff) ** inv_exponent

            # Objective Function F(X)
            f_val = mass_water_total - (X_safe * bulk_capacity) - (mass_atm_factor * pressure_atm)

            # Derivative Function F'(X)
            dp_dx = inv_exponent * (1.0 / solubility_coeff) * ((X_safe / solubility_coeff) ** (inv_exponent - 1.0))
            df_dx = -bulk_capacity - (mass_atm_factor * dp_dx)

            # Newton step
            X_next = X_safe - (f_val / df_dx)

            # Prevent negative overshoot (keeps the solver physically bounded)
            X_next = max(X_next, 1e-12)

            # Relative convergence check
            if abs(X_next - X_safe) / X_safe < 1e-10:
                X_current = X_next
                break

            X_current = X_next

        # Final state assignment
        mass_frac_water_liquid = X_current
        pressure_water_vapor  = (mass_frac_water_liquid / solubility_coeff) ** inv_exponent

    else:
        # Mantle is below the solidus; no active magma ocean
        mass_frac_water_liquid = 0.0
        pressure_water_vapor = mass_water_total * gravity / surface_area
        partition_coeff_water = 0.0

    return pressure_water_vapor, mass_frac_water_liquid, partition_coeff_water
