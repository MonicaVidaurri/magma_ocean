def moEvent(t, y):
    '''
    Check for the end of phase one (fully magma ocean world)
    Event is when the planet surface reaches the solidus temperature and therefore radius of solidification reaches
    the surface.

    Event works with ODE: moODEb
    '''
    Ts = y[10]

    # EVENT CONDITION
    return Ts - 1420.0


# Required by SciPy
moEvent.terminal = True
moEvent.direction = 0
