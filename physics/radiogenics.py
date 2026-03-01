import numpy as np

def get_radiogenic_heat(
        t_sec,
        Mmantle):
    """
    Calculates the radiogenic heat production of the mantle over time.

    Parameters
    ----------
    t_sec : float
        Current integration time in seconds.
    Mmantle : float
        Mass of the planetary mantle in kg.

    Returns
    -------
    Q : float
        Total radiogenic heat production in Watts.
    """
    # Handle the time conversion locally
    sec_per_yr = 3.15569e7
    t_years = t_sec / sec_per_yr
    
    # Age of the solar system in years
    t_ss = 4.6e9 

    # Specific heat production (W/kg)
    H_238U  = 9.37e-5
    H_235U  = 5.69e-4
    H_232Th = 2.69e-5
    H_40K   = 2.79e-5

    # Present-day bulk concentrations (kg/kg)
    Uran    = 21.0e-9
    C_238U  = 0.9927 * Uran
    C_235U  = 0.0072 * Uran
    C_40K   = 1.28   * Uran
    C_232Th = 4.01  * Uran

    # Decay constants (1/years)
    l_238U  = 0.155e-9
    l_235U  = 0.985e-9
    l_232Th = 0.0495e-9
    l_40K   = 0.555e-9

    # Calculate total heat production (Q)
    Q = (
        C_238U * H_238U * np.exp(l_238U * (t_ss - t_years))
        + C_235U * H_235U * np.exp(l_235U * (t_ss - t_years))
        + C_232Th * H_232Th * np.exp(l_232Th * (t_ss - t_years))
        + C_40K * H_40K * np.exp(l_40K * (t_ss - t_years))
    ) * Mmantle

    return Q