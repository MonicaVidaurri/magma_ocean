import numpy as np
from .get_fO2 import get_fO2
from .get_massbalance2 import get_massbalance2

def get_massbalance4(
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
    Calculates mass balance for Oxygen using a hybrid solver strategy 
    driven by master TOML parameters.
    """

    # --- Unpack Parameters ---
    univ    = params['constants']
    kc      = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num     = params['numerical']

    mu_O      = univ['molar_mass_O']
    mu_FeO1_5 = univ['molar_mass_FeO1_5']
    mu_FeO    = univ['molar_mass_FeO']

    surface_area   = 4.0 * np.pi * planet_radius**2
    surface_factor = surface_area / (mu_O * gravity)

    # --- Initial Molar Calculations ---
    moles_iron_total   = (total_iron_fraction * magma_mass) / mu_FeO
    moles_oxygen_total = total_oxygen_mass / mu_O
    
    partial_pressure_o2  = 0.0
    mass_fraction_feo1_5 = 0.0
    moles_feo1_5         = 0.0
    count = 0

    # =====================================================================
    # --- Branch 1: Low Oxygen (Fixed-Point Iteration Strategy) ---
    # =====================================================================
    if moles_oxygen_total < num['stoic_approx_threshold'] * moles_iron_total:
        
        # Pre-compute Static Thermodynamic Terms (KC91)
        term_temp  = kc['temp_coeff'] / melt_temp
        term_const = kc['constant_term']
        term_comp  = (kc['coeff_Al2O3'] * composition[2] + 
                      kc['coeff_FeOt']  * composition[8] + 
                      kc['coeff_CaO']   * composition[4] + 
                      kc['coeff_Na2O']  * composition[5] + 
                      kc['coeff_K2O']   * composition[6])
        
        T0 = kc['T0_ref']
        term_correction = kc['temp_correction_coeff'] * (1.0 - T0 / melt_temp - np.log(melt_temp / T0))
        term_pressure   = (kc['press_coeff_1'] * atm_pressure / melt_temp + 
                           kc['press_coeff_2'] * (melt_temp - T0) * atm_pressure / melt_temp + 
                           kc['press_coeff_3'] * (atm_pressure**2) / melt_temp)

        static_exponent_sum = term_temp + term_const + term_comp + term_correction + term_pressure

        # Iteration Loop setup
        moles_Fe3_current = moles_iron_total * 1e-3  # Initial guess
        
        while count <= num['max_fixed_point_iters']:
            safe_Fe3 = np.clip(moles_Fe3_current, 1e-20, moles_iron_total - 1e-20)
            
            # Fixed-point update
            log_ratio = np.log(safe_Fe3 / (moles_iron_total - safe_Fe3))
            exponent  = (log_ratio + static_exponent_sum) / kc['scaling_a']
            
            moles_Fe3_next = 2.0 * (moles_oxygen_total - surface_factor * np.exp(exponent))
            
            # Convergence Check
            rel_err = np.abs(moles_Fe3_next - moles_Fe3_current) / max(moles_Fe3_current, 1e-10)
            
            if rel_err < num['solver_tolerance'] and moles_Fe3_next > 0:
                moles_feo1_5 = moles_Fe3_next
                moles_o_atm  = moles_oxygen_total - 0.5 * moles_feo1_5
                
                partial_pressure_o2 = (moles_o_atm * mu_O * gravity) / surface_area
                if magma_mass > 0.0:
                    mass_fraction_feo1_5 = moles_feo1_5 * mu_FeO1_5 / magma_mass
                break
            
            count += 1
            
            # Fallback for slow convergence
            if count >= num['fallback_iter_threshold']:
                moles_feo1_5 = moles_oxygen_total * 2.0
                moles_o_atm  = moles_oxygen_total - 0.5 * moles_feo1_5
                partial_pressure_o2 = (moles_o_atm * mu_O * gravity) / surface_area
                if magma_mass > 0.0:
                    mass_fraction_feo1_5 = moles_feo1_5 * mu_FeO1_5 / magma_mass
                break
            
            moles_Fe3_current = moles_Fe3_next

    # =====================================================================
    # --- Branch 2: High Oxygen (Robust Bisection Solver) ---
    # =====================================================================
    else:
        partial_pressure_o2, mass_fraction_feo1_5, _, moles_feo1_5 = get_massbalance2(
            melt_temp, atm_pressure, magma_mass, total_oxygen_mass, 
            composition, total_iron_fraction, gravity, planet_radius, params
        )

    # =====================================================================
    # --- Post-Processing: "Zero Oxygen" Safety Check ---
    # =====================================================================
    if partial_pressure_o2 <= 0.0:
        m_FeO = max(moles_iron_total - 2.0 * moles_oxygen_total, 0.0)
        m_FeO1_5 = 2.0 * moles_oxygen_total

        # Pass modified composition to the parameterized fO2 solver
        new_composition = np.concatenate([composition[:10], [m_FeO, m_FeO1_5]])
        partial_pressure_o2 = get_fO2(melt_temp, atm_pressure, new_composition, params)

    return partial_pressure_o2, mass_fraction_feo1_5, count, moles_feo1_5
