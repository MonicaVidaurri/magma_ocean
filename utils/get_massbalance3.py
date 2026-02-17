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
        planet_radius
    ):
    """
    Calculates mass balance for Oxygen using a fixed-point iteration method.
    
    This function attempts to solve for the FeO1.5 content using a fast iterative 
    approach. If convergence is slow or unstable (step count > 50), it falls back 
    to either a stoichiometric approximation (for low oxygen) or the robust 
    bisection method (calculate_oxygen_mass_balance).

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
    iteration_count : int
        Number of iterations performed.
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

    # --- Inner Helper Function: Fixed-Point Step ---
    def calculate_next_y(y_current):
        """
        Calculates the next estimate for moles of FeO1.5 (y) based on 
        rearranging the equilibrium mass balance equation.
        """
        # Protect against log(<=0) or division by zero
        # y_current is moles of FeO1.5. (moles_iron_total - y_current) is moles of FeO.
        safe_y = np.clip(y_current, 1e-20, moles_iron_total - 1e-20)
        
        # Calculate term inside exponential (Thermodynamic relation)
        # Mapping indices: MATLAB Xi(3)->[2], Xi(9)->[8], Xi(5)->[4], Xi(6)->[5], Xi(7)->[6]
        log_ratio = np.log(safe_y / (moles_iron_total - safe_y))
        
        term_temp = -1.1492e4 / melt_temp
        term_const = 6.675
        term_comp = (2.243 * composition[2] + 
                     1.828 * composition[8] - 
                     3.201 * composition[4] - 
                     5.854 * composition[5] - 
                     6.215 * composition[6])
        
        term_correction = 3.36 * (1.0 - 1673.0 / melt_temp - np.log(melt_temp / 1673.0))
        
        term_pressure = (7.01e-7 * atm_pressure / melt_temp + 
                         1.54e-10 * (melt_temp - 1673.0) * atm_pressure / melt_temp - 
                         3.85e-17 * (atm_pressure**2) / melt_temp)
        
        exponent = (log_ratio + term_temp + term_const + term_comp + 
                    term_correction + term_pressure) / 0.196
        
        # Calculate new estimate for y
        # Based on: y_new = 2 * (nO_t - nO_atm)
        # Where nO_atm is derived from equilibrium relation
        surface_factor = 4 * np.pi * planet_radius**2 / (MOLAR_MASS_O * gravity)
        return 2 * (moles_oxygen_total - surface_factor * np.exp(exponent))

    # --- Iteration Loop ---
    count = 0
    y_current = moles_iron_total * 1e-3  # Initial guess
    
    # Initialize outputs to ensure they exist if loop breaks early
    partial_pressure_o2 = 0.0
    mass_fraction_feo1_5 = 0.0
    moles_feo1_5 = 0.0

    while count <= 100:
        y_next = calculate_next_y(y_current)
        
        # Check for Convergence
        if np.abs(y_next - y_current) < 1e-12 and y_next > 0:
            moles_feo1_5 = y_next
            moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
            
            partial_pressure_o2 = (moles_o_atm * MOLAR_MASS_O * gravity) / (4 * np.pi * planet_radius**2)
            mass_fraction_feo1_5 = moles_feo1_5 * MOLAR_MASS_FEO1_5 / magma_mass
            break
        
        count += 1
        
        # Fallback Logic for Slow Convergence
        if count >= 50:
            if moles_oxygen_total < 1e-3 * moles_iron_total:
                # Case 1: Low Oxygen - Approximation
                moles_feo1_5 = moles_oxygen_total * 2
                moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
                
                partial_pressure_o2 = (moles_o_atm * MOLAR_MASS_O * gravity) / (4 * np.pi * planet_radius**2)
                mass_fraction_feo1_5 = moles_feo1_5 * MOLAR_MASS_FEO1_5 / magma_mass
                break
            
            else:
                # Case 2: High Oxygen - Use Robust Bisection Solver
                # Calls the function defined in the previous step
                partial_pressure_o2, mass_fraction_feo1_5, _, moles_feo1_5 = get_massbalance2(
                    melt_temp, atm_pressure, magma_mass, moles_oxygen_total * MOLAR_MASS_O, 
                    composition, total_iron_fraction, gravity, planet_radius
                )
                
                # Re-calculate moles_o_atm to ensure consistency
                moles_o_atm = moles_oxygen_total - 0.5 * moles_feo1_5
                break
        
        y_current = y_next

    # --- Post-Processing Safety Checks ---
    # The original code attempts to calculate fO2 using an external function 'get_fO2'
    # if PO2 is non-positive. We include placeholders here.
    
    if partial_pressure_o2 <= 0:
        # Construct the modified composition array required by get_fO2
        # MATLAB: [Xi(1:10) nFeOt-2*nO_t 2*nO_t]
        
        # Note: We assume composition has at least 10 elements.
        # We append two new values representing specific Fe/O stoichiometries.
        new_composition = np.concatenate([composition[:10], 
                                          [moles_iron_total - 2*moles_oxygen_total, 2*moles_oxygen_total]])
        partial_pressure_o2 = get_fO2(melt_temp, atm_pressure, new_composition)

    return partial_pressure_o2, mass_fraction_feo1_5, count, moles_feo1_5