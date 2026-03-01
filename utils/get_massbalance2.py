import numpy as np

def get_massbalance2(
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
    Calculates the mass balance for Oxygen in the magma ocean system 
    using parameters from a master TOML dictionary.
    """

    # --- Unpack Parameters ---
    comp_params = params['planet']['oxide_composition']
    kc_params   = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num_params  = params['numerical']

    mu_O      = comp_params['molar_mass_O']
    mu_FeO1_5 = comp_params['molar_mass_FeO1_5']
    mu_FeO    = comp_params['molar_mass_FeO']

    surface_area = 4.0 * np.pi * planet_radius**2

    # --- Initial Molar Calculations ---
    moles_iron_total   = (total_iron_fraction * magma_mass) / mu_FeO
    moles_oxygen_total = total_oxygen_mass / mu_O

    # --- Pre-compute Static Thermodynamic Terms ---
    # Unpack composition-dependent indices (Al2O3[2], CaO[4], Na2O[5], K2O[6], FeOt[8])
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

    static_kc_sum = term_temp + term_const + term_comp + term_corr + term_press

    # --- Internal Objective Function ---
    def calculate_disequilibrium(fraction_ferric):
        # fraction_ferric is the molar ratio Fe3+/FeTotal
        safe_ferric = np.clip(fraction_ferric, 1e-20, 1.0 - 1e-20)

        # Melt Equilibrium Fugacity (fO2)
        log_ferric_ferrous_ratio = np.log(safe_ferric / (1.0 - safe_ferric))
        log_fugacity  = (log_ferric_ferrous_ratio + static_kc_sum) / kc_params['scaling_a']
        fugacity_melt = np.exp(log_fugacity)
        
        # Atmospheric Partial Pressure (pO2) from leftover oxygen
        # Every mole of FeO1.5 uses 0.5 moles of free Oxygen (O)
        moles_oxygen_atm = moles_oxygen_total - 0.5 * safe_ferric * moles_iron_total
        pressure_atm_O2  = (moles_oxygen_atm * mu_O * gravity) / surface_area
        
        return fugacity_melt - pressure_atm_O2

    # --- Bisection Solver Setup ---
    lower_bound, upper_bound = 0.0, 1.0
    f_lower = calculate_disequilibrium(lower_bound)
    f_upper = calculate_disequilibrium(upper_bound)
    
    moles_feo1_5 = 0.0
    convergence_flag = 0
    skip_solver = False

    # Stoichiometric limit check
    if np.sign(f_lower) * np.sign(f_upper) > 0:
        #root is outside [0, 1], so system is stoichiometrically limited
        moles_feo1_5 = min(moles_oxygen_total * 2.0, moles_iron_total)
        skip_solver = True

    # --- Bisection Loop ---
    if not skip_solver:
        count = 0
        p_mid = 0.0
        tol = num_params['solver_tolerance']
        
        while count <= num_params['max_iterations']:
            p_mid = lower_bound + (upper_bound - lower_bound) / 2.0
            f_mid = calculate_disequilibrium(p_mid)
            
            if f_mid == 0.0 or (upper_bound - lower_bound) / 2.0 < tol:
                moles_feo1_5 = p_mid * moles_iron_total
                break
            
            count += 1
            if np.sign(f_lower) * np.sign(f_mid) > 0:
                lower_bound, f_lower = p_mid, f_mid
            else:
                upper_bound, f_upper = p_mid, f_mid
        
        if count > num_params['max_iterations']:
            moles_feo1_5 = p_mid * moles_iron_total
            convergence_flag = 1

    # --- Final Output Calculations ---
    if np.isnan(melt_temp):
        return 0.0, 0.0, convergence_flag, moles_feo1_5

    moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
    partial_pressure_o2 = (moles_o_atm * mu_O * gravity) / surface_area
    
    mass_fraction_feo1_5 = (moles_feo1_5 * mu_FeO1_5) / magma_mass if magma_mass > 0.0 else 0.0

    return partial_pressure_o2, mass_fraction_feo1_5, convergence_flag, moles_feo1_5
