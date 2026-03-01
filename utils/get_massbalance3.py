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
    Calculates mass balance for Oxygen using a fixed-point iteration method 
    driven by master TOML parameters.
    """

    # --- Unpack Parameters ---
    comp_params = params['planet']['oxide_composition']
    kc_params   = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num         = params['numerical']

    mu_O      = comp_params['molar_mass_O']
    mu_FeO1_5 = comp_params['molar_mass_FeO1_5']
    mu_FeO    = comp_params['molar_mass_FeO']

    surface_area = 4.0 * np.pi * planet_radius**2
    surface_factor = surface_area / (mu_O * gravity)

    # --- Initial Molar Calculations ---
    moles_iron_total = (total_iron_fraction * magma_mass) / mu_FeO
    moles_oxygen_total = total_oxygen_mass / mu_O

    # --- Pre-compute Static Thermodynamic Terms ---
    # Unpack indices: Al2O3[2], CaO[4], Na2O[5], K2O[6], FeOt[8]
    term_temp  = kc_params['temp_coeff'] / melt_temp
    term_const = kc_params['constant_term']
    term_comp  = (kc_params['coeff_Al2O3'] * composition[2] + 
                  kc_params['coeff_FeOt']  * composition[8] + 
                  kc_params['coeff_CaO']   * composition[4] + 
                  kc_params['coeff_Na2O']  * composition[5] + 
                  kc_params['coeff_K2O']   * composition[6])
    
    T0 = kc_params['T0_ref']
    term_corr  = kc_params['temp_correction_coeff'] * (1.0 - T0 / melt_temp - np.log(melt_temp / T0))
    term_press = (kc_params['press_coeff_1'] * atm_pressure / melt_temp + 
                  kc_params['press_coeff_2'] * (melt_temp - T0) * atm_pressure / melt_temp + 
                  kc_params['press_coeff_3'] * (atm_pressure**2) / melt_temp)

    static_exponent_sum = term_temp + term_const + term_comp + term_corr + term_press

    # --- Iteration Loop ---
    count = 0
    moles_Fe3_current = moles_iron_total * 1e-3  # Initial guess
    
    partial_pressure_o2 = 0.0
    mass_fraction_feo1_5 = 0.0
    moles_feo1_5 = 0.0

    while count <= num['max_fixed_point_iters']:
        # Safety clamp
        safe_Fe3 = np.clip(moles_Fe3_current, 1e-20, moles_iron_total - 1e-20)
        
        # Fixed-point update rule
        log_ratio = np.log(safe_Fe3 / (moles_iron_total - safe_Fe3))
        exponent  = (log_ratio + static_exponent_sum) / kc_params['scaling_a']
        
        moles_Fe3_next = 2.0 * (moles_oxygen_total - surface_factor * np.exp(exponent))
        
        # Convergence Check
        relative_error = np.abs(moles_Fe3_next - moles_Fe3_current) / max(moles_Fe3_current, 1e-10)
        
        if relative_error < num['solver_tolerance'] and moles_Fe3_next > 0:
            moles_feo1_5 = moles_Fe3_next
            moles_o_atm  = moles_oxygen_total - 0.5 * moles_feo1_5
            
            partial_pressure_o2 = (moles_o_atm * mu_O * gravity) / surface_area
            if magma_mass > 0.0:
                mass_fraction_feo1_5 = (moles_feo1_5 * mu_FeO1_5) / magma_mass
            break
        
        count += 1
        
        # Fallback Logic
        if count >= num['fallback_iter_threshold']:
            if moles_oxygen_total < num['stoic_approx_threshold'] * moles_iron_total:
                # Low Oxygen Fallback: Stoichiometric limit
                moles_feo1_5 = moles_oxygen_total * 2.0
                moles_o_atm  = moles_oxygen_total - 0.5 * moles_feo1_5
                partial_pressure_o2 = (moles_o_atm * mu_O * gravity) / surface_area
                if magma_mass > 0.0:
                    mass_fraction_feo1_5 = (moles_feo1_5 * mu_FeO1_5) / magma_mass
                break
            else:
                # High Oxygen Fallback: Robust Bisection
                partial_pressure_o2, mass_fraction_feo1_5, _, moles_feo1_5 = get_massbalance2(
                    melt_temp, atm_pressure, magma_mass, total_oxygen_mass, 
                    composition, total_iron_fraction, gravity, planet_radius, params
                )
                break
        
        moles_Fe3_current = moles_Fe3_next

    # --- "Zero Oxygen" Fugacity Check ---
    if partial_pressure_o2 <= 0.0:
        moles_FeO = max(moles_iron_total - 2.0 * moles_oxygen_total, 0.0)
        moles_FeO1_5_forced = 2.0 * moles_oxygen_total

        # Composition array for get_fO2 (Indices 10 and 11 used for ln(ratio))
        new_composition = np.concatenate([
            composition[:10], 
            [moles_FeO, moles_FeO1_5_forced]
        ])
        
        # Re-evaluating fugacity using the same params dictionary
        partial_pressure_o2 = get_fO2(melt_temp, atm_pressure, new_composition, params)

    return partial_pressure_o2, mass_fraction_feo1_5, count, moles_feo1_5
