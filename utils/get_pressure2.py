import numpy as np


def get_pressure2(
        mass_melt,
        mass_crystals_melt_region,
        mass_mantle,
        radius_planet,
        gravity,
        mass_water_total,
        params):
    """
    Calculates the equilibrium partition of water between the melt-bearing region of the mantle and the overlying
    steam atmosphere using a Newton-Raphson root finder.

    Water dissolves in the melt following the solubility law X = s * P^n. Crystals within the melt-bearing region hold
    D * X. All water that is not dissolved is treated as vapor; condensation to a surface ocean is applied afterwards
    by the caller.

    Parameters
    ----------
    mass_melt : float
        Mass of melt in the mantle (kg).
    mass_crystals_melt_region : float
        Mass of crystals inside the melt-bearing region, i.e., above the solidus crossing (kg).
    mass_mantle : float
        Total mass of the mantle in kg. Only used for the initial guess.
    radius_planet : float
        Radius of the planet in meters.
    gravity : float
        Surface gravitational acceleration in m/s^2.
    mass_water_total : float
        Total mass of water in the melt region + atmosphere (+ surface ocean) system in kg.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    pressure_water_vapor : float
        Equilibrium atmospheric water vapor pressure in Pascals, assuming no condensation.
    mass_frac_water_liquid : float
        Mass fraction of water dissolved in the melt.
    partition_coeff_water : float
        Partition coefficient for water between solid and melt.
    """

    # --- Unpack Parameters ---
    vols   = params['planet']['volatiles']
    num    = params['numerical']

    # --- Constants & Solubility Parameters ---
    partition_coeff_water = vols['partition_coeff_H2O']
    solubility_coeff      = vols['solubility_coeff_H2O']
    solubility_exponent   = vols['solubility_exponent_H2O']
    max_iters = num['max_nr_iterations']
    tolerance = num['nr_solver_tolerance']

    surface_area    = 4.0 * np.pi * radius_planet**2
    mass_atm_factor = surface_area / gravity

    if mass_melt <= 0.0 or mass_water_total <= 0.0:
        # No melt to dissolve into: all water is in the atmosphere.
        return max(mass_water_total, 0.0) * gravity / surface_area, 0.0, partition_coeff_water

    # --- Newton-Raphson Root Finder ---
    # We solve for X (mass_frac_water_liquid) such that:
    # F(X) = Total_Water - Water_in_Melt_Region - Water_in_Atmosphere == 0
    # Where:
    # Water_in_Melt_Region = X * (D * M_crystals + M_melt)
    # Water_in_Atmosphere = (Area / gravity) * (X / s)^(1/n)

    bulk_capacity = (partition_coeff_water * mass_crystals_melt_region) + mass_melt
    inv_exponent = 1.0 / solubility_exponent

    # Initial guess: assume all water is evenly mixed in the mantle
    X_current = mass_water_total / mass_mantle

    for _ in range(max_iters):
        # Protect against negative guesses in fractional powers
        X_safe = max(X_current, 1e-20)

        # Atmospheric pressure: P = (X/s)^(1/n)
        pressure_atm = (X_safe / solubility_coeff) ** inv_exponent

        # Objective Function F(X)
        f_val = mass_water_total - (X_safe * bulk_capacity) - (mass_atm_factor * pressure_atm)

        # Derivative Function F'(X)
        # dP/dX = (1/n) * (1/s) * (X/s)^((1/n) - 1)
        dp_dx = inv_exponent * (1.0 / solubility_coeff) * ((X_safe / solubility_coeff) ** (inv_exponent - 1.0))
        df_dx = -bulk_capacity - (mass_atm_factor * dp_dx)

        # Newton step
        X_next = X_safe - (f_val / df_dx)

        # Prevent negative overshoot (keeps the solver physically bounded)
        X_next = max(X_next, 1e-20)

        # Relative convergence check
        if abs(X_next - X_safe) / X_safe < tolerance:
            X_current = X_next
            break

        X_current = X_next

    # Final state assignment
    mass_frac_water_liquid = X_current
    pressure_water_vapor   = (mass_frac_water_liquid / solubility_coeff) ** inv_exponent

    return pressure_water_vapor, mass_frac_water_liquid, partition_coeff_water
