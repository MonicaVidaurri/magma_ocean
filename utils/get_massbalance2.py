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
    Calculates the mass balance for Oxygen in the magma ocean system.
    
    This function uses a bisection solver to find the equilibrium oxidation state 
    of Iron (Fe3+/FeTotal) that balances the oxygen fugacity of the silicate melt 
    with the partial pressure of oxygen in the atmosphere.

    Parameters
    ----------
    melt_temp : float
        Temperature of the silicate melt in Kelvin.
    atm_pressure : float
        Atmospheric surface pressure in Pascals.
    magma_mass : float
        Total mass of the active magma ocean in kg.
    total_oxygen_mass : float
        Total mass of Oxygen in the combined magma ocean + atmosphere system in kg.
    composition : ndarray
        1D array of background oxide mole fractions.
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
        Mass fraction of FeO1.5 (Fe3+ equivalent) in the magma ocean.
    convergence_flag : int
        0 if converged or solved stoichiometrically, 1 if max iterations reached.
    moles_feo1_5 : float
        Total moles of FeO1.5 in the magma ocean.
    """

    # --- Unpack Parameters ---
    c    = params['constants']
    comp = params['planet']['oxide_composition']
    kc   = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num  = params['numerical']

    # --- Physical Constants & Molar Masses ---
    molar_mass_O      = c['molar_mass_O']
    molar_mass_FeO1_5 = comp['molar_mass_FeO1_5']
    molar_mass_FeO    = comp['molar_mass_FeO']

    surface_area = 4.0 * np.pi * planet_radius**2

    # --- Initial Molar Calculations ---
    # Total moles of Iron (normalized to an all-Fe2+ basis for counting)
    moles_iron_total   = total_iron_fraction * magma_mass / molar_mass_FeO
    moles_oxygen_total = total_oxygen_mass / molar_mass_O

    # --- Pre-compute Static Thermodynamic Terms ---
    # To optimize the bisection loop, we pre-calculate the Kress & Carmichael (1991) 
    # terms that do not change with the iron oxidation state.
    frac_Al2O3 = composition[2]
    frac_CaO   = composition[4]
    frac_Na2O  = composition[5]
    frac_K2O   = composition[6]
    frac_FeOt  = composition[8]

    term_temp  = kc['temp_coeff'] / melt_temp
    term_const = kc['constant_term']
    
    # Note: the TOML values carry the correct positive/negative signs
    term_comp  = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                  kc['coeff_FeO']   * frac_FeOt + 
                  kc['coeff_CaO']   * frac_CaO + 
                  kc['coeff_Na2O']  * frac_Na2O + 
                  kc['coeff_K2O']   * frac_K2O)
    
    T0 = kc['temp_ref']
    term_corr  = kc['temp_correction_coeff'] * (1.0 - T0 / melt_temp - np.log(melt_temp / T0))
    
    term_press = (kc['press_coeff_1'] * atm_pressure / melt_temp + 
                  kc['press_coeff_2'] * (melt_temp - T0) * atm_pressure / melt_temp + 
                  kc['press_coeff_3'] * (atm_pressure**2) / melt_temp)

    static_kc_sum = term_temp + term_const + term_comp + term_corr + term_press

    # --- Internal Objective Function ---
    def calculate_disequilibrium(fraction_ferric):
        """
        Calculates difference between equilibrium fO2 and atmospheric pO2.
        fraction_ferric is the fraction of total Iron that is Fe3+ (FeO1.5).
        """
        # Clamp to avoid log(0) errors at perfectly reduced/oxidized extremes
        safe_ferric = np.clip(fraction_ferric, 1e-20, 1.0 - 1e-20)

        # Melt Equilibrium Fugacity (fO2)
        log_ferric_ferrous_ratio = np.log(safe_ferric / (1.0 - safe_ferric))
        log_fugacity  = (log_ferric_ferrous_ratio + static_kc_sum) / kc['scaling_a']
        fugacity_melt = np.exp(log_fugacity)
        
        # Atmospheric Partial Pressure (pO2) from leftover oxygen
        moles_oxygen_atm = moles_oxygen_total - 0.5 * safe_ferric * moles_iron_total
        pressure_atm_O2  = (moles_oxygen_atm * molar_mass_O * gravity) / surface_area
        
        return fugacity_melt - pressure_atm_O2

    # --- Bisection Solver Setup ---
    lower_bound = 0.0
    upper_bound = 1.0

    f_lower = calculate_disequilibrium(lower_bound)
    f_upper = calculate_disequilibrium(upper_bound)
    
    moles_feo1_5     = 0.0
    convergence_flag = 0
    skip_solver      = False

    # Stoichiometric Check: Is the root outside [0, 1]?
    if np.sign(f_lower) * np.sign(f_upper) > 0:
        # Default limit: Put all available Oxygen into FeO1.5
        moles_feo1_5 = moles_oxygen_total * 2.0
        moles_feo = moles_iron_total - moles_feo1_5
        
        # If we need more FeO1.5 than total Iron allows, cap it:
        if moles_feo < 0:
            moles_feo1_5 = moles_iron_total
            
        skip_solver = True

    # --- Bisection Loop ---
    if not skip_solver:
        count = 0
        p_mid = 0.0
        
        max_iters = num['max_iterations']
        tolerance = num['solver_tolerance']
        
        while count <= max_iters:
            p_mid = lower_bound + (upper_bound - lower_bound) / 2.0
            f_mid = calculate_disequilibrium(p_mid)
            
            # Check convergence 
            if f_mid == 0.0 or ((upper_bound - lower_bound) / 2.0 < tolerance and f_mid < 0):
                moles_feo1_5 = p_mid * moles_iron_total
                convergence_flag = 0
                break
            
            count += 1
            
            # Narrow bracket
            if np.sign(f_lower) * np.sign(f_mid) > 0:
                lower_bound = p_mid
                f_lower = f_mid
            else:
                upper_bound = p_mid
                f_upper = f_mid
        
        # Max iterations reached
        if count > max_iters:
            moles_feo1_5 = p_mid * moles_iron_total
            convergence_flag = 1

    # --- Final Outputs ---
    # NaN safety check
    if np.isnan(melt_temp):
        return 0.0, 0.0, convergence_flag, moles_feo1_5

    moles_o_atm         = moles_oxygen_total - 0.5 * moles_feo1_5
    partial_pressure_o2 = (moles_o_atm * molar_mass_O * gravity) / surface_area
    
    if magma_mass > 0.0:
        mass_fraction_feo1_5 = (moles_feo1_5 * molar_mass_FeO1_5) / magma_mass
    else:
        mass_fraction_feo1_5 = 0.0

    return partial_pressure_o2, mass_fraction_feo1_5, convergence_flag, moles_feo1_5
