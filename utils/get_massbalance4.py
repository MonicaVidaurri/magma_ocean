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
        planet_radius
    ):
    """
    Calculates mass balance for Oxygen using a hybrid solver strategy.
    
    This function switches between a fast fixed-point iteration solver (for low 
    oxygen conditions) and a robust bisection solver (for high oxygen conditions).
    It also handles edge cases where pO2 is zero by calling an external fO2 calculator.

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
        Mass fraction of FeO1.5 in the magma ocean.
    count : int
        Number of iterations performed (in low-oxygen mode) or status flag.
    moles_feo1_5 : float
        Total moles of FeO1.5 in the magma ocean.
    """

    # --- Physical Constants ---
    MOLAR_MASS_O = 15.9994e-3          # kg/mol
    MOLAR_MASS_FEO1_5 = 159.689e-3 / 2 # kg/mol
    MOLAR_MASS_FEO = 71.845e-3         # kg/mol

    # --- Initial Molar Calculations ---
    moles_iron_total = total_iron_fraction * magma_mass / MOLAR_MASS_FEO
    moles_oxygen_total = total_oxygen_mass / MOLAR_MASS_O
    
    # Initialize Outputs
    partial_pressure_o2 = 0.0
    mass_fraction_feo1_5 = 0.0
    moles_feo1_5 = 0.0
    count = 0

    # --- Solver Logic ---
    
    # Condition: Low total oxygen (approx. less than 0.1% of iron moles)
    if moles_oxygen_total < 1e-3 * moles_iron_total:
        
        # Strategy: Fixed-Point Iteration
        y_current = moles_iron_total * 1e-3  # Initial guess for FeO1.5 moles
        
        while count <= 100:
            
            # --- Inner Helper: Calculate next y ---
            # y represents moles of FeO1.5
            # We clip y to avoid log(<=0) or division by zero errors
            y_safe = np.clip(y_current, 1e-20, moles_iron_total - 1e-20)
            
            # Calculate Equilibrium constant term
            # Mapping: MATLAB Xi(3)->[2], Xi(9)->[8], Xi(5)->[4], Xi(6)->[5], Xi(7)->[6]
            log_term = np.log(y_safe / (moles_iron_total - y_safe))
            
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
            
            exponent = (log_term + term_temp + term_const + term_comp + 
                        term_corr + term_press) / 0.196
            
            # Update y based on mass balance
            surface_factor = (4 * np.pi * planet_radius**2) / (MOLAR_MASS_O * gravity)
            y_next = 2 * (moles_oxygen_total - surface_factor * np.exp(exponent))
            # ---------------------------------------

            # Convergence Check
            if np.abs(y_next - y_current) < 1e-12 and y_next > 0:
                moles_feo1_5 = y_next
                moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
                
                partial_pressure_o2 = (moles_o_atm * MOLAR_MASS_O * gravity) / (4 * np.pi * planet_radius**2)
                mass_fraction_feo1_5 = moles_feo1_5 * MOLAR_MASS_FEO1_5 / magma_mass
                break
            
            count += 1
            
            # Fallback for non-convergence in low-oxygen regime
            if count >= 50:
                # Force stoichiometric limit: All Oxygen becomes FeO1.5
                moles_feo1_5 = moles_oxygen_total * 2
                moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
                
                partial_pressure_o2 = (moles_o_atm * MOLAR_MASS_O * gravity) / (4 * np.pi * planet_radius**2)
                mass_fraction_feo1_5 = moles_feo1_5 * MOLAR_MASS_FEO1_5 / magma_mass
                break
            
            y_current = y_next
        
        # Sanity Check (from original MATLAB code)
        # In Python, we might just log this, as printing acts as a side effect.
        if moles_feo1_5 > moles_iron_total:
            print(f"Warning: FeO1.5 moles ({moles_feo1_5}) exceeds Total Fe moles ({moles_iron_total})")

    else:
        # Condition: High Oxygen
        # Strategy: Use Robust Bisection Solver (get_massbalance2 logic)
        (partial_pressure_o2, 
         mass_fraction_feo1_5, 
         mark, 
         moles_feo1_5) = get_massbalance2(
            melt_temp, atm_pressure, magma_mass, total_oxygen_mass, 
            composition, total_iron_fraction, gravity, planet_radius
        )
        
        # Re-calculate atmosphere moles to ensure variable scope consistency
        moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
        
        # Sanity Check
        if moles_feo1_5 > moles_iron_total:
             print(f"Warning: FeO1.5 moles ({moles_feo1_5}) exceeds Total Fe moles ({moles_iron_total})")

    # --- Final Polish: Handle pO2 == 0 using External Function ---
    if partial_pressure_o2 == 0:
        # Construct the modified composition array required by get_fO2
        # MATLAB: [Xi(1:10) nFeOt-2*nO_t 2*nO_t]
        
        # Note: We assume composition has at least 10 elements.
        # We append two new values representing specific Fe/O stoichiometries.
        xi_modified = np.concatenate((
            composition[0:10], 
            [moles_iron_total - 2 * moles_oxygen_total, 2 * moles_oxygen_total]
        ))
        
        partial_pressure_o2 = get_fO2(melt_temp, atm_pressure, xi_modified)

    return partial_pressure_o2, mass_fraction_feo1_5, count, moles_feo1_5
