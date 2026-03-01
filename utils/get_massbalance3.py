import numpy as np
from .get_massbalance2 import get_massbalance2
from .get_fO2 import get_fO2

def get_massbalance3(
        melt_temp,
        atm_pressure,
        magma_mass,
        total_oxygen_mass,
        composition,
        total_iron_fraction,
        gravity,
        planet_radius,
        params):
    """
    Calculates mass balance for Oxygen using a fixed-point iteration method.
    
    This function attempts to solve for the FeO1.5 content using a fast iterative 
    approach. If convergence is slow or unstable, it falls back to either a 
    stoichiometric approximation (for low oxygen) or the robust bisection method.

    Parameters
    ----------
    melt_temp : float
        Temperature of the silicate melt in Kelvin.
    atm_pressure : float
        Atmospheric surface pressure in Pascals.
    magma_mass : float
        Total mass of the active magma ocean in kg.
    total_oxygen_mass : float
        Total mass of Oxygen in the magma ocean + atmosphere system in kg.
    composition : ndarray
        1D array of background oxide mole fractions in the melt.
    total_iron_fraction : float
        Mass fraction of total Iron (Fe) in the magma ocean.
    gravity : float
        Gravitational acceleration in m/s^2.
    planet_radius : float
        Radius of the planet in meters.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    partial_pressure_o2 : float
        Partial pressure of Oxygen in the atmosphere in Pascals.
    mass_fraction_feo1_5 : float
        Mass fraction of FeO1.5 in the magma ocean.
    iteration_count : int
        Number of iterations performed before convergence or fallback.
    moles_feo1_5 : float
        Total moles of FeO1.5 in the magma ocean.
    """

    # --- Unpack Parameters ---
    c = params['constants']
    comp = params['planet']['oxide_composition']
    kc = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num = params['numerical']

    # --- Physical Constants & Molar Masses ---
    molar_mass_O      = c['molar_mass_O']
    molar_mass_FeO1_5 = comp['molar_mass_FeO1_5']
    molar_mass_FeO    = comp['molar_mass_FeO']

    surface_area   = 4.0 * np.pi * planet_radius**2
    surface_factor = surface_area / (molar_mass_O * gravity)

    # --- Initial Molar Calculations ---
    moles_iron_total = total_iron_fraction * magma_mass / molar_mass_FeO
    moles_oxygen_total = total_oxygen_mass / molar_mass_O

    # --- Pre-compute Static Thermodynamic Terms ---
    frac_Al2O3 = composition[2]
    frac_CaO   = composition[4]
    frac_Na2O  = composition[5]
    frac_K2O   = composition[6]
    frac_FeOt  = composition[8]

    term_temp  = kc['temp_coeff'] / melt_temp
    term_const = kc['constant_term']
    
    # Note: Negative coefficients are naturally handled by the TOML values
    term_comp  = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                  kc['coeff_FeO']   * frac_FeOt + 
                  kc['coeff_CaO']   * frac_CaO + 
                  kc['coeff_Na2O']  * frac_Na2O + 
                  kc['coeff_K2O']   * frac_K2O)
    
    T0 = kc['temp_ref']
    term_correction = kc['temp_correction_coeff'] * (1.0 - T0 / melt_temp - np.log(melt_temp / T0))
    term_pressure = (kc['press_coeff_1'] * atm_pressure / melt_temp + 
                     kc['press_coeff_2'] * (melt_temp - T0) * atm_pressure / melt_temp + 
                     kc['press_coeff_3'] * (atm_pressure**2) / melt_temp)

    static_exponent_sum = term_temp + term_const + term_comp + term_correction + term_pressure

    # --- Iteration Loop ---
    count = 0
    moles_Fe3_current = moles_iron_total * 1e-3  # Initial guess (y_current)
    
    partial_pressure_o2 = 0.0
    mass_fraction_feo1_5 = 0.0
    moles_feo1_5 = 0.0

    max_iters = num['max_fixed_point_iters']
    fallback_threshold = num['fallback_iter_threshold', 50]
    tolerance = num['solver_tolerance', 1e-12]

    while count <= max_iters:
        # Protect against log(<=0) or division by zero
        safe_Fe3 = np.clip(moles_Fe3_current, 1e-20, moles_iron_total - 1e-20)
        
        # Calculate next estimate
        log_ratio = np.log(safe_Fe3 / (moles_iron_total - safe_Fe3))
        exponent  = (log_ratio + static_exponent_sum) / kc['scaling_a']
        
        moles_Fe3_next = 2.0 * (moles_oxygen_total - surface_factor * np.exp(exponent))
        
        # Relative Convergence Check
        # Uses absolute difference divided by the current magnitude to handle massive mole numbers
        relative_error = np.abs(moles_Fe3_next - moles_Fe3_current) / max(moles_Fe3_current, 1e-10)
        
        if relative_error < tolerance and moles_Fe3_next > 0:
            moles_feo1_5 = moles_Fe3_next
            moles_o_atm  = moles_oxygen_total - 0.5 * moles_feo1_5
            
            partial_pressure_o2 = (moles_o_atm * molar_mass_O * gravity) / surface_area
            if magma_mass > 0.0:
                mass_fraction_feo1_5 = moles_feo1_5 * molar_mass_FeO1_5 / magma_mass
            break
        
        count += 1
        
        # Fallback Logic for Slow Convergence
        if count >= fallback_threshold:
            if moles_oxygen_total < 1e-3 * moles_iron_total:
                # Case 1: Low Oxygen - Stoichiometric Approximation
                moles_feo1_5 = moles_oxygen_total * 2.0
                moles_o_atm  = moles_oxygen_total - 0.5 * moles_feo1_5
                
                partial_pressure_o2 = (moles_o_atm * molar_mass_O * gravity) / surface_area
                if magma_mass > 0.0:
                    mass_fraction_feo1_5 = moles_feo1_5 * molar_mass_FeO1_5 / magma_mass
                break
            
            else:
                # Case 2: High Oxygen - Defers to robust bisection solver
                # Note: get_massbalance2 requires total oxygen mass in kg, not moles
                partial_pressure_o2, mass_fraction_feo1_5, _, moles_feo1_5 = get_massbalance2(
                    melt_temp, atm_pressure, magma_mass, total_oxygen_mass, 
                    composition, total_iron_fraction, gravity, planet_radius, params
                )
                break
        
        moles_Fe3_current = moles_Fe3_next

    # --- Post-Processing: "Zero Oxygen" Safety Check ---
    if partial_pressure_o2 <= 0.0:
        # If all oxygen is locked in the rock, the atmospheric pressure is technically zero.
        # We calculate the solid fugacity of the melt using the forced stoichiometric limit.
        moles_FeO = max(moles_iron_total - 2.0 * moles_oxygen_total, 0.0)
        moles_FeO1_5_forced = 2.0 * moles_oxygen_total

        # Construct the modified composition array for get_fO2.
        # Note: This mixes pure mole fractions (indices 0-9) with raw moles (indices 10-11).
        # This is mathematically safe ONLY because get_fO2 takes the ln(ratio) of indices 11 and 10,
        # meaning the absolute magnitudes (moles vs fractions) cancel out perfectly.
        new_composition = np.concatenate([
            composition[:10], 
            [moles_FeO, moles_FeO1_5_forced]
        ])
        
        partial_pressure_o2 = get_fO2(melt_temp, atm_pressure, new_composition, params)

    return partial_pressure_o2, mass_fraction_feo1_5, count, moles_feo1_5
