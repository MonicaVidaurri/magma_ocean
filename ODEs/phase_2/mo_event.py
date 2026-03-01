def moEvent_phase2(t, y, core_mass_fraction, planet_mass):
    '''
    Phase 2.
    Stops when mantle water mass fraction < 1e-9
    '''

    Mmantle = (1 - core_mass_fraction) * planet_mass

    water_mantle = y[5] 

    # EVENT CONDITION
    return (water_mantle / Mmantle) - 1e-9

moEvent_phase2.terminal = True
moEvent_phase2.direction = -1
