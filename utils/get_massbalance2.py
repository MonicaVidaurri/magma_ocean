import numpy as np
from .get_fO2 import get_fO2

def get_massbalance2(
        melt_temp, 
        atm_pressure, 
        magma_mass, 
        total_oxygen_mass, 
        composition, 
        total_iron_fraction, 
        gravity, 
        planet_radius
    ):
    """
    Calculates the mass balance for Oxygen in the magma ocean system.
    
    This function solves for the oxidation state of Iron (Fe3+/FeTotal) that 
    satisfies equilibrium between the silicate melt and the atmosphere.

    Parameters
    ----------
    melt_temp : float
        Temperature of the silicate melt (Kelvin).
    atm_pressure : float
        Atmospheric surface pressure (Pascals).
    magma_mass : float
        Total mass of the magma ocean (kg).
    total_oxygen_mass : float
        Total mass of Oxygen in the magma ocean + atmosphere system (kg).
    composition : ndarray
        1D array of oxide mole fractions in the melt.
    total_iron_fraction : float
        Mass fraction of total Iron (Fe) in the magma ocean.
    gravity : float
        Gravitational acceleration (m/s^2).
    planet_radius : float
        Radius of the planet (m).

    Returns
    -------
    partial_pressure_o2 : float
        Partial pressure of Oxygen in the atmosphere (Pascals).
    mass_fraction_feo1_5 : float
        Mass fraction of FeO1.5 (Fe2O3 equivalent) in the magma ocean.
    convergence_flag : int
        0 if converged or solved stoichiometrically, 1 if max iterations reached.
    moles_feo1_5 : float
        Total moles of FeO1.5 in the magma ocean.
    """

    # --- Physical Constants ---
    MOLAR_MASS_O = 15.9994e-3          # kg/mol
    MOLAR_MASS_FEO1_5 = 159.689e-3 / 2 # kg/mol
    MOLAR_MASS_FEO = 71.845e-3         # kg/mol

    # --- Initial Molar Calculations ---
    # Total moles of Iron (normalized to FeO basis)
    moles_iron_total = total_iron_fraction * magma_mass / MOLAR_MASS_FEO
    # Total moles of Oxygen
    moles_oxygen_total = total_oxygen_mass / MOLAR_MASS_O

    # Range of nFeO1_5 / nFeOt values to iterate across (0.0 to 1.0)
    # y = n(FeO1.5) / n(Fe_Total)
    lower_bound = 0.0
    upper_bound = 1.0

    # --- Internal Objective Function ---
    def calculate_disequilibrium(y_ratio):
        """
        Calculates difference between equilibrium fO2 and mass-balance pO2.
        y_ratio is the fraction of Iron that is Fe3+ (FeO1.5).
        """
        # Clamp y to avoid log(0) errors
        y_safe = np.clip(y_ratio, 1e-20, 1.0 - 1e-20)

        # 1. Calculate fO2 from Melt Equilibrium
        # Mapping: MATLAB Xi(3)->[2], Xi(9)->[8], Xi(5)->[4], Xi(6)->[5], Xi(7)->[6]
        log_term = np.log(y_safe / (1.0 - y_safe))
        
        term_temp = -1.1492e4 / melt_temp
        term_const = 6.675
        term_comp = (2.243 * composition[2] + 
                     1.828 * composition[8] - 
                     3.201 * composition[4] - 
                     5.854 * composition[5] - 
                     6.215 * composition[6])
        
        term_corr = 3.36 * (1.0 - 1673.0 / melt_temp - np.log(melt_temp / 1673.0))
        
        term_press = (7.01e-7 * atm_pressure / melt_temp + 
                      1.54e-10 * (melt_temp - 1673.0) * atm_pressure / melt_temp - 
                      3.85e-17 * (atm_pressure**2) / melt_temp)

        log_fugacity = (log_term + term_temp + term_const + term_comp + 
                        term_corr + term_press) / 0.196
        
        fO2_val = np.exp(log_fugacity)
        
        # 2. Calculate pO2 from Mass Balance
        # The atmosphere gets the "leftover" oxygen.
        surface_area = 4 * np.pi * planet_radius**2
        pO2_val = ((moles_oxygen_total / moles_iron_total - 0.5 * y_safe) * moles_iron_total * MOLAR_MASS_O * gravity / surface_area)
        
        return fO2_val - pO2_val
    # -----------------------------------

    # Evaluate at boundaries
    f_lower = calculate_disequilibrium(lower_bound)
    f_upper = calculate_disequilibrium(upper_bound)
    
    moles_feo1_5 = 0.0
    mark = 0
    p = 0.0 # Initialize midpoint
    skip_solver = False

    # --- 1. Stoichiometric Check ---
    # If the function signs are the same, the root is not in [0, 1].
    # This means we are at a limit: either 0% Fe3+ or 100% Fe3+.
    if np.sign(f_lower) * np.sign(f_upper) > 0:
        
        # Default: Put all Oxygen into FeO1.5
        moles_feo1_5 = moles_oxygen_total * 2.0
        moles_feo = moles_iron_total - moles_feo1_5
        
        # If we need more FeO1.5 than total Iron allows:
        if moles_feo < 0:
            moles_feo = 0.0
            moles_feo1_5 = moles_iron_total
            # Remaining oxygen goes to atmosphere (calculated later)
            
        skip_solver = True

    # --- 2. Bisection Solver ---
    if not skip_solver:
        count = 0
        while count <= 500:
            p = lower_bound + (upper_bound - lower_bound) / 2.0
            f_p = calculate_disequilibrium(p)
            
            # Check convergence
            if f_p == 0 or ((upper_bound - lower_bound) / 2.0 < 1e-17 and f_p < 0):
                moles_feo1_5 = p * moles_iron_total
                mark = 0
                break
            
            count += 1
            
            # Narrow bracket
            if np.sign(f_lower) * np.sign(f_p) > 0:
                lower_bound = p
                f_lower = f_p
            else:
                upper_bound = p
                f_upper = f_p
        
        if count >= 500:
            # Did not converge
            moles_feo1_5 = p * moles_iron_total
            mark = 1

    # --- Final Calculation ---
    moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
    
    partial_pressure_o2 = (moles_o_atm * MOLAR_MASS_O * gravity) / (4 * np.pi * planet_radius**2)
    if magma_mass > 0.0:
        mass_fraction_feo1_5 = moles_feo1_5 * MOLAR_MASS_FEO1_5 / magma_mass
    else:
        mass_fraction_feo1_5 = 0.0

    # NaN safety check
    if np.isnan(melt_temp):
        partial_pressure_o2 = 0.0
        mass_fraction_feo1_5 = 0.0

    # Call external fO2 function (as per original script logic, though unused in return)
    # Construct input array: [Xi(1:10), nFeOt-2*nO_t, 2*nO_t]
    # Note: This logic assumes 'composition' has enough elements.
    xi_modified = np.concatenate((
        composition[0:10], 
        [moles_iron_total - 2 * moles_oxygen_total, 2 * moles_oxygen_total]
    ))
    
    # Calculate fo2 (variable unused in return, mimics [fo2] = get_fO2(...) line)
    _ = get_fO2(melt_temp, atm_pressure, xi_modified)

    return partial_pressure_o2, mass_fraction_feo1_5, mark, moles_feo1_5
