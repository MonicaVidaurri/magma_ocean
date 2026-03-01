import numpy as np

def viscosity2(
        temp_mantle,
        mass_frac_water,
        gravity,
        density_mantle,
        pressure_pa,
        params):
    """
    Calculates the kinematic viscosity of the mantle incorporating water-weakening 
    effects (Sandu et al. 2011).

    Parameters
    ----------
    temp_mantle : float
        Mantle potential temperature (K).
    mass_frac_water : float
        Bulk mass fraction of water in the mantle.
    gravity : float
        Surface gravity (m/s^2).
    density_mantle : float
        Bulk mantle density (kg/m^3).
    pressure_pa : float
        Local pressure (Pa).
    params : dict
        Master configuration dictionary.

    Returns
    -------
    kinematic_viscosity : float (m^2/s)
    """
    
    # --- Unpack Constants & Material ---
    univ   = params['constants']
    mat    = params['planet']['material']
    rheo   = params['planet']['rheology']['sandu_2011']

    gas_constant   = univ['gas_constant']
    molar_mass_H2O = univ['molar_mass_H2O']

    # Calculate bulk molar mass of idealized olivine mantle
    molar_mass_olivine = (mat['fraction_forsterite'] * mat['molar_mass_forsterite'] + 
                          mat['fraction_fayalite']   * mat['molar_mass_fayalite'])

    # --- Water Concentration (ppm H/Si) ---
    safe_water_frac = max(mass_frac_water, 1e-12)
    
    # Convert mass fraction to ppm H/Si (2 H atoms per H2O molecule)
    concentration_OH = (safe_water_frac * molar_mass_olivine * 1.0e6 * 2.0) / molar_mass_H2O
    ln_C_OH = np.log(concentration_OH)

    # --- Water Fugacity (Li et al. 2008) ---
    ln_fugacity_H2O = (rheo['fugacity_poly_c0'] + 
                       rheo['fugacity_poly_c1'] * ln_C_OH + 
                       rheo['fugacity_poly_c2'] * ln_C_OH**2 + 
                       rheo['fugacity_poly_c3'] * ln_C_OH**3)
    
    fugacity_H2O = np.exp(ln_fugacity_H2O)

    # --- Dynamic Viscosity (Pa s) ---
    # Arrhenius term: exp((Qa + P*V) / (R*T))
    energy_term = rheo['activation_energy'] + (pressure_pa * rheo['activation_volume'])
    arrhenius_term = np.exp(energy_term / (gas_constant * temp_mantle))

    # Incorporate fugacity-based weakening
    dynamic_viscosity = (rheo['visc_reference'] * (fugacity_H2O**-rheo['fugacity_exponent']) * arrhenius_term)

    # --- Return Kinematic (m^2/s) ---
    return dynamic_viscosity / density_mantle
