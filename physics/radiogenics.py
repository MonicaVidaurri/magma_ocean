import numpy as np

def get_radiogenic_heat(
        t_sec,
        Mmantle,
        params):
    """
    Calculates the radiogenic heat production of the mantle over time.

    Parameters
    ----------
    t_sec : float
        Current integration time in seconds.
    Mmantle : float
        Mass of the planetary mantle in kg.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    Q : float
        Total radiogenic heat production in Watts.
    """
    # --- Unpack Parameters ---
    c = params['constants']
    rad = params['planet']['radiogenics']

    # Handle the time conversion locally
    sec_per_yr = c['seconds_per_year']
    t_years = t_sec / sec_per_yr
    
    # Age of the solar system in years
    t_ss = rad['age_solar_system'] 

    # Specific heat production (W/kg)
    H_238U  = rad['heat_238U']
    H_235U  = rad['heat_235U']
    H_232Th = rad['heat_232Th']
    H_40K   = rad['heat_40K']

    # Present-day bulk concentrations (kg/kg)
    Uran    = rad['bulk_uranium_concentration']
    C_238U  = rad['frac_238U'] * Uran
    C_235U  = rad['frac_235U'] * Uran
    C_40K   = rad['frac_40K']  * Uran
    C_232Th = rad['frac_232Th'] * Uran

    # Decay constants (1/years)
    l_238U  = rad['decay_const_238U']
    l_235U  = rad['decay_const_235U']
    l_232Th = rad['decay_const_232Th']
    l_40K   = rad['decay_const_40K']

    # Calculate total heat production (Q)
    Q = (
        C_238U * H_238U * np.exp(l_238U * (t_ss - t_years))
        + C_235U * H_235U * np.exp(l_235U * (t_ss - t_years))
        + C_232Th * H_232Th * np.exp(l_232Th * (t_ss - t_years))
        + C_40K * H_40K * np.exp(l_40K * (t_ss - t_years))
    ) * Mmantle

    return Q
