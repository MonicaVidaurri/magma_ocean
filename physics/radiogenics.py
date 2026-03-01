import numpy as np

def get_radiogenic_heat(t_sec, Mmantle, params):
    """
    Calculates the total radiogenic heat production of the mantle in Watts.

    Parameters
    ----------
    t_sec : float
        Current integration time in seconds.
    Mmantle : float
        Mass of the planetary mantle in kg.
    params : dict
        Master configuration dictionary.

    Returns
    -------
    total_heat_watts : float
        Total radiogenic heat production (Q) in Watts.
    """
    # --- Unpack Parameters ---
    radio = params['planet']['radiogenics']
    sec_per_yr = params['constants']['seconds_per_year']
    
    t_years = t_sec / sec_per_yr
    t_ss = radio['age_solar_system']
    uran = radio['bulk_uranium_concentration']

    # Concentrations (kg/kg)
    c_238u  = radio['frac_238U'] * uran
    c_235u  = radio['frac_235U'] * uran
    c_40k   = radio['frac_40K']  * uran
    c_232th = radio['frac_232Th'] * uran

    # --- Calculate Specific Heat Production (W/kg) ---
    # Each term follows: C_iso * H_iso * exp(lambda_iso * (t_ref - t_current))
    q_238u  = c_238u  * radio['heat_238U']  * np.exp(radio['lambda_238U']  * (t_ss - t_years))
    q_235u  = c_235u  * radio['heat_235U']  * np.exp(radio['lambda_235U']  * (t_ss - t_years))
    q_232th = c_232th * radio['heat_232Th'] * np.exp(radio['lambda_232Th'] * (t_ss - t_years))
    q_40k   = c_40k   * radio['heat_40K']   * np.exp(radio['lambda_40K']   * (t_ss - t_years))

    # Total power (Watts)
    total_heat_watts = (q_238u + q_235u + q_232th + q_40k) * Mmantle

    return total_heat_watts
