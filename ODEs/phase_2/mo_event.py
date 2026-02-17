def moEvent(t, y, core_mass_fraction, planet_mass):
    '''
    Phase 2.
    Stops when mantle water mass fraction < 1e-9
    '''

    Mmantle = (1 - core_mass_fraction) * planet_mass

    water_mantle = y[5] 

    # EVENT CONDITION
    return (water_mantle / Mmantle) - 1e-9

moEvent.terminal = True
moEvent.direction = -1
