import numpy as np


def moEvent_phase3(t, y):
    '''
    Phase 3.
    Stops when degassing stops
    '''
    
    mantle_temperature = y[4]  # Tr(1)

    # EVENT CONDITION
    return mantle_temperature - 1000.0


# Required by SciPy
moEvent_phase3.terminal = True
moEvent_phase3.direction = 0
