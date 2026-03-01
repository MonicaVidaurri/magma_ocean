import numpy as np
from scipy.optimize import root_scalar

def get_XFeO(
        temp_kelvin, 
        pressure_pa, 
        mass_magma_ocean, 
        radius_planet, 
        gravity, 
        mole_fractions, 
        mass_O2_total,
        params):
    """
    Solves for the oxygen fugacity (fO2) and ferrous iron fraction (XFeO) 
    using Brent's method and master TOML parameters.
    """

    # --- Unpack Parameters ---
    univ = params['constants']
    kc   = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num  = params['numerical']

    molar_mass_O2 = univ['molar_mass_O2']
    surface_area  = 4.0 * np.pi * radius_planet**2
    
    # Extract composition indices: [2]Al2O3, [4]CaO, [5]Na2O, [6]K2O, [8]FeOt, [11]Fe2O3
    frac_Al2O3 = mole_fractions[2]
    frac_CaO   = mole_fractions[4]
    frac_Na2O  = mole_fractions[5]
    frac_K2O   = mole_fractions[6]
    frac_FeOt  = mole_fractions[8]
    frac_Fe2O3_initial = mole_fractions[11]

    # --- Pre-compute Static KC91 Terms ---
    T0 = kc['T0_ref']
    term_temp  = kc['temp_coeff'] / temp_kelvin
    term_const = kc['constant_term']
    
    term_comp = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                 kc['coeff_FeOt']  * frac_FeOt  + 
                 kc['coeff_CaO']   * frac_CaO   + 
                 kc['coeff_Na2O']  * frac_Na2O  + 
                 kc['coeff_K2O']   * frac_K2O)
    
    term_corr = kc['temp_correction_coeff'] * (1.0 - T0 / temp_kelvin - np.log(temp_kelvin / T0))
    
    term_press = (kc['press_coeff_1'] * pressure_pa / temp_kelvin + 
                  kc['press_coeff_2'] * (temp_kelvin - T0) * pressure_pa / temp_kelvin + 
                  kc['press_coeff_3'] * (pressure_pa**2) / temp_kelvin)

    static_kc_sum = term_temp + term_const + term_comp + term_corr + term_press

    # --- Objective Function ---
    def objective_function(guess_XFeO):
        # fO2 from KC91 Empirical Relation
        # Ferric = Total - Ferrous
        log_ratio = np.log((frac_FeOt - guess_XFeO) / guess_XFeO)
        ln_fO2_kc = (log_ratio + static_kc_sum) / kc['scaling_a']
        
        # fO2 from Mass Balance (Atmospheric Pressure)
        # Change in Ferric iron vs initial state determines O2 consumed/released
        mass_O2_melt = (frac_FeOt - guess_XFeO - frac_Fe2O3_initial) * 0.25 * molar_mass_O2 * mass_magma_ocean
        mass_O2_atm  = mass_O2_total - mass_O2_melt
        
        if mass_O2_atm <= 0:
            return 1e6 # High penalty
            
        pressure_O2_atm = mass_O2_atm * gravity / surface_area
        ln_fO2_balance  = np.log(pressure_O2_atm)
        
        return ln_fO2_balance - ln_fO2_kc

    # --- Bounded Root Finding ---
    eps = num['solver_epsilon']
    lower_bound = eps
    upper_bound = frac_FeOt - eps

    f_low  = objective_function(lower_bound)
    f_high = objective_function(upper_bound)

    if np.sign(f_low) == np.sign(f_high):
        # System is saturated at a boundary
        frac_FeO = lower_bound if abs(f_low) < abs(f_high) else upper_bound
    else:
        # Brent's method for fast, bracketed convergence
        result   = root_scalar(objective_function, bracket=[lower_bound, upper_bound], method='brentq')
        frac_FeO = result.root

    # --- Final Outputs ---
    mass_O2_melt_final = (frac_FeOt - frac_FeO - frac_Fe2O3_initial) * 0.25 * molar_mass_O2 * mass_magma_ocean
    mass_O2_atm_final  = mass_O2_total - mass_O2_melt_final
    fugacity_O2        = max(mass_O2_atm_final * gravity / surface_area, 0.0)

    return fugacity_O2, frac_FeO
