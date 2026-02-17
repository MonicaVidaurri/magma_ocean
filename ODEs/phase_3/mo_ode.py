import numpy as np

from physics.mantleheatflux import mantleheatflux
from utils.get_meltfracb import get_meltfracb
from utils.get_flux import get_flux
from utils.get_loss import get_loss

from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def moODE(t, Tr, Rp, Rc, Mmantle, Teq, rho, g, Ts, Ps, OLR, ASR, t_flux, Lbol, Xi, FeOt,
           Temp_K, P_Pa, tsat, FH2O, Mp, LStar, Rh, Mh, tides_on_flag):
    """ ODE when there is no magma ocean. Tectonic phase / last time phase. """

    # constants
    sigma = 5.67e-8          # Boltzmann constant (W/m2/K4)
    Tsola = 26.53e-9         # solidus parameter (K/Pa)
    Tsolb = 1373             # solidus parameter (K)
    dHf = 4e5                # heat of fusion of silicate (J/kg)
    alpha = 2e-5             # thermal expansion coefficient (K^-1)
    cp = 1.2e3               # heat capacity (J/kg/K)
    muO = 15.9994e-3         # atomic weight of O (kg/mole)
    muH2O = 18.015e-3        # molecular weight of H2O (kg/mole)
    NA = 6.022e23            # Avogadro's number (molecules/mole)
    muH = 1.008e-3           # atomic weight of H (kg/mole)
    cpH2O = 2e3              # J/kg/K

    # mid-ocean ridge length
    Lridge = 2 * np.pi * Rp

    # For this model the state vector equals:
    #   0: Semi major axis
    #   1: Eccentricity
    #   2: Host spin Freq
    #   3: Target spin freq
    #   4: Mantle temperature
    #   5: Mass of water in atmosphere
    #   6: Mass of free O in atmosphere
    #   7: Surface Temperature
    # Build derivative array
    dTr_dt = np.zeros(8, dtype=np.float64)
    # Unpack state vector
    semi_a = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h = Tr[2]
    spin_freq_p = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    Tm = Tr[4]       # mantle temperature
    Watm = Tr[5]     # mass of water in atmosphere
    if Watm < 0:
        Watm = 0.0

    MO_tot = Tr[6]   # mass of free O in atmosphere
    if MO_tot < 0:
        MO_tot = 0.0

    Tsurf = Tr[7]

    #######################################################################
    # heat capacity and melt fraction
    heatcap = cp * Mmantle

    if Tm > 1420:
        _, meltfrac = get_meltfracb(g, Tm, Rp, Rc, Mmantle)
    else:
        meltfrac = 0.0

    #######################################################################
    # atmospheric pressure
    if Tr[7] > 647:
        Patm = Watm * g / (4 * np.pi * Rp**2)
    else:
        Patm = 10**(6.079 - 2261.10 / Tr[7]) * 1e5
        Matm = Patm * (4 * np.pi * Rp**2) / g
        if Matm > Watm:
            Patm = Watm * g / (4 * np.pi * Rp**2)

    #######################################################################
    # oxygen partial pressure
    PO2 = MO_tot * g / (4 * np.pi * Rp**2)

    #######################################################################
    # atmospheric flux
    flux = get_flux(Tsurf, Teq, Patm, Rp, g)

    # # use the following flux if get_flux function is ONLY calculating OSR
    # flux = get_flux(Tsurf, Patm, Rp, g)

    #######################################################################
    # atmospheric mass loss
    phi_H, phi_O = get_loss(t_flux, Lbol, t, PO2, Patm, tsat, semi_a, Mp, Rp, LStar)

    #######################################################################
    # radioactive heat production
    H_238U = 9.37e-5
    H_235U = 5.69e-4
    H_232Th = 2.69e-5
    H_40K  = 2.79e-5

    Uran = 21e-9
    C_238U = 0.9927 * Uran
    C_235U = 0.0072 * Uran
    C_40K  = 1.28   * Uran
    C_232Th = 4.01  * Uran

    l_238U = 0.155e-9
    l_235U = 0.985e-9
    l_232Th = 0.0495e-9
    l_40K  = 0.555e-9

    Q = (
        C_238U * H_238U * np.exp(l_238U * (4.6e9 - t))
        + C_235U * H_235U * np.exp(l_235U * (4.6e9 - t))
        + C_232Th * H_232Th * np.exp(l_232Th * (4.6e9 - t))
        + C_40K * H_40K * np.exp(l_40K * (4.6e9 - t))
    ) * Mmantle

    #######################################################################
    # mantle heat flux
    q_mantle, Db, uc, Ra, nu = mantleheatflux(Tm, Tsurf, Rp, Rp, Rc, g, rho, FH2O, meltfrac)
    # nu == viscosity of magma ocean. 
    # !!! Need solid mantle viscosity for tides.

    
    #######################################################################
    # Tidal Heating
    ## Bottom up solidification -> Solid Core -> Solid Mantle -> Magma Ocean. 
    da_dt, de_dt, dspin_dt_h, dspin_dt_p, tidal_heating_h, tidal_heating_p = calculate_tidal_dissipation(
            eccentricity, orbital_freq, spin_freq_p, spin_freq_h,
            Rp, Rh, Mp, Mh,
            Tm, Rc, nu,
            rho, meltfrac,
            tides_on_flag
    )
    
    # Update mantle heating
    Q += tidal_heating_p

    # Update derivative array.
    dTr_dt[0] = da_dt
    dTr_dt[1] = de_dt
    dTr_dt[2] = dspin_dt_h
    dTr_dt[3] = dspin_dt_p

    #######################################################################
    # differential equations
    dTr_dt[4] = (
        3.15569e7 * (-4 * np.pi * Rp**2 * q_mantle + Q)
        / (cp * Mmantle)
    )

    if Tr[7] > 647:
        dTr_dt[5] = (
            -4 * np.pi * Rp**2 * phi_H * muH2O / 2 / muH
        ) * 3.15569e7
    else:
        dTr_dt[5] = 0.0

    if Tr[7] > 647:
        dTr_dt[6] = (
            (4 * np.pi * Rp**2 * phi_H * muO / 2 / muH)
            - (4 * np.pi * Rp**2 * phi_O)
        ) * 3.15569e7
    else:
        dTr_dt[6] = 0.0

    dTr_dt[7] = (
        4 * np.pi * Rp**2 * 3.15569e7
        * (
            (-flux + q_mantle)
            / (
                cpH2O * Patm * 4 * np.pi * Rp**2 / g
                + cp * 4/3 * np.pi * 3e3 * (Rp**3 - Db**3)
            )
        )
    )

    return dTr_dt
