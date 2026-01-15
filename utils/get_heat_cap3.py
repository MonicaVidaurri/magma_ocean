def get_heat_cap3(Tm, Mmantle, g, Rp, Rc, Mmo, dMs, get_meltfrac):
    '''
    get_heat_cap3.m

    Parameters
    ----------
    Tm : float
        Mantle temperature (K)
    Mmantle : float
        Mantle mass (kg)
    g : float
        Gravity (m/s^2)
    Rp : float
        Planet radius (m)
    Rc : float
        Core radius (m)
    Mmo : float
        Magma ocean mass (kg)
    deltaMs : float
        Change in melt mass (kg)
    get_meltfrac : function
        Function returning (unused, melt fraction) for given T

    Returns
    -------
    Cp : float
        Mantle heat capacity (J/K)
    '''

    cp_kg = 1.2e3     # J/kg/K
    deltaH_kg = 4e5   # J/kg latent heat per kg

    # call get_meltfrac to... you guessed it... get melt fraction
    _, meltfrac = get_meltfrac(g, Tm, Rp, Rc, Mmantle)

    # calculate heat capacity
    Cp = cp_kg * meltfrac * Mmo + deltaH_kg * dMs  # Joules/K for entire mantle

    return Cp
