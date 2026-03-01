import numpy as np
from scipy.optimize import root_scalar

def get_XFeO(temp_kelvin, pressure_pa, mass_magma_ocean, radius_planet, 
             gravity, mole_fractions, mass_O2_total, params):
    """
    Solves for the oxygen fugacity (fO2) and the ferrous iron fraction (XFeO) 
    in equilibrium with a magma ocean using a bounded 1D root finder.

    Note: This function performs the same role as `get_massbalance2`, but uses 
    a different root-finding architecture.

    Parameters
    ----------
    temp_kelvin : float
        Mantle temperature in Kelvin.
    pressure_pa : float
        Pressure of the environment in Pascals.
    mass_magma_ocean : float
        Total mass of the magma ocean in kg.
    radius_planet : float
        Radius of the planet in meters.
    gravity : float
        Surface gravitational acceleration in m/s^2.
    mole_fractions : array_like
        1D array of oxide mole fractions (Standard 12-element get_comp array).
    mass_O2_total : float
        Total mass of O2 in the magma ocean + atmosphere system in kg.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    fugacity_O2 : float
        Equilibrium oxygen fugacity in Pascals.
    frac_FeO : float
        Equilibrium mole fraction of FeO in the melt.
    """

    # --- Unpack Parameters ---
    univ = params['constants']
    kc   = params['planet']['oxygen_fugacity']['kress_carmichael_1991']
    num  = params['numerical']

    # --- Constants & Geometry ---
    molar_mass_O2 = univ['molar_mass_O'] * 2.0  # kg/mol
    surface_area  = 4.0 * np.pi * radius_planet**2
    
    # Extract composition
    frac_Al2O3 = mole_fractions[2]
    frac_CaO   = mole_fractions[4]
    frac_Na2O  = mole_fractions[5]
    frac_K2O   = mole_fractions[6]
    frac_FeOt  = mole_fractions[8]
    frac_Fe2O3_initial = mole_fractions[11]

    # --- Pre-compute Static KC91 Thermodynamic Terms ---
    T0         = kc['temp_ref']
    term_temp  = kc['temp_coeff'] / temp_kelvin
    term_const = kc['constant_term']
    
    term_comp  = (kc['coeff_Al2O3'] * frac_Al2O3 + 
                  kc['coeff_FeO']   * frac_FeOt + 
                  kc['coeff_CaO']   * frac_CaO + 
                  kc['coeff_Na2O']  * frac_Na2O + 
                  kc['coeff_K2O']   * frac_K2O)
    
    term_temp_correction = kc['temp_correction_coeff'] * (1.0 - (T0 / temp_kelvin) - np.log(temp_kelvin / T0))
    term_pressure = (kc['press_coeff_1'] * pressure_pa / temp_kelvin + 
                     kc['press_coeff_2'] * (temp_kelvin - T0) * pressure_pa / temp_kelvin + 
                     kc['press_coeff_3'] * (pressure_pa**2) / temp_kelvin)

    static_kc_sum = term_temp + term_const + term_comp + term_temp_correction + term_pressure

    # --- 1D Objective Function ---
    def objective_function(guess_XFeO):
        """
        Evaluates the difference between the KC91 fO2 and Mass Balance fO2.
        """
        # KC91 Empirical Fugacity
        # frac_FeOt - guess_XFeO represents the new XFeO1.5 (Ferric Iron)
        log_ferric_ferrous_ratio = np.log((frac_FeOt - guess_XFeO) / guess_XFeO)
        ln_fO2_kc = (log_ferric_ferrous_ratio + static_kc_sum) / kc['scaling_a']
        
        # Mass Balance Fugacity (Atmospheric Pressure)
        # TODO: WARNING: This retains the original code's dimensional unit mismatch 
        # (multiplying mole fraction by mass).
        mass_O2_melt = (frac_FeOt - guess_XFeO - frac_Fe2O3_initial) * 0.25 * molar_mass_O2 * mass_magma_ocean
        mass_O2_atm  = mass_O2_total - mass_O2_melt
        
        # Prevent negative mass in log calculation
        if mass_O2_atm <= 0:
            return 1e6  # High penalty to force solver back into bounds
            
        pressure_O2_atm     = mass_O2_atm * gravity / surface_area
        ln_fO2_mass_balance = np.log(pressure_O2_atm)
        
        # We want the difference to be zero
        return ln_fO2_mass_balance - ln_fO2_kc

    # --- Bounded Root Finding ---
    # We constrain XFeO to be strictly between 0 and Total Iron.
    # We use a tiny epsilon from params to prevent log(0) at the absolute boundaries.
    eps = num['solver_epsilon']
    lower_bound = eps
    upper_bound = frac_FeOt - eps

    # Check if a root exists in this bracket (signs must be opposite)
    f_low  = objective_function(lower_bound)
    f_high = objective_function(upper_bound)

    if np.sign(f_low) == np.sign(f_high):
        # If no root exists, it means the system is saturated at an extreme.
        # Fall back to the boundary that minimizes the error.
        frac_FeO = lower_bound if abs(f_low) < abs(f_high) else upper_bound
    else:
        # Brent's method (brentq) is incredibly fast and guaranteed to converge 
        # if a root is bracketed.
        result   = root_scalar(objective_function, bracket=[lower_bound, upper_bound], method='brentq')
        frac_FeO = result.root

    # --- Final Fugacity Calculation ---
    mass_O2_melt_final = (frac_FeOt - frac_FeO - frac_Fe2O3_initial) * 0.25 * molar_mass_O2 * mass_magma_ocean
    mass_O2_atm_final  = mass_O2_total - mass_O2_melt_final
    fugacity_O2        = max(mass_O2_atm_final * gravity / surface_area, 0.0)

    return fugacity_O2, frac_FeO
