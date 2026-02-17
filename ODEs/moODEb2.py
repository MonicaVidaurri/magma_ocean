import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_meltfracb import get_meltfracb
from utils.get_flux import get_flux
from utils.get_loss import get_loss
from physics.degas2 import degas2

from physics.tides import calculate_tidal_dissipation
from TidalPy.conversions.conversions_x import semi_a2orbital_motion

def moODEb2(t, Tr, Rp, Rc, Mmantle, Teq, rho, g, Ts, Ps, OLR, ASR, t_flux,
    Lbol, Xi, FeOt, Temp_K, P_Pa, tsat, a, Mp, LStar, Rh, Mh, tides_on_flag):
    """ Second phase / stage 2. Solidification starts. """

    # ------------------------------------------------------------------
    # constants
    sigma = 5.67e-8
    Tsola = 26.53e-9
    Tsolb = 1373
    dHf = 4e5
    alpha = 2e-5
    cp = 1.2e3

    muO = 15.9994e-3
    muH2O = 18.015e-3
    NA = 6.022e23
    muH = 1.008e-3
    cpH2O = 2e3

    # mid-ocean ridge length
    Lridge = 2 * np.pi * Rp

    # For this model the state vector equals:
    #   0: Semi major axis
    #   1: Eccentricity
    #   2: Host spin Freq
    #   3: Target spin freq
    #   4: Mantle temperature
    #   5: Water fraction in the solid mantle
    #   6: Mass of water in atmosphere
    #   7: Mass of free O in atmosphere
    #   8: Surface temperature
    # Build derivative array
    dTr_dt = np.zeros(9, dtype=np.float64)
    # Unpack state vector
    semi_a = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h = Tr[2]
    spin_freq_p = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    # Tr = [Tm, FH2O_solid, Watm, MO_tot, Tsurf]
    Tm = Tr[4]
    FH2O = Tr[5] / Mmantle

    Watm = Tr[6]
    if Watm < 0:
        Watm = 0.0

    MO_tot = Tr[7]
    if MO_tot < 0:
        MO_tot = 0.0

    Tsurf = Tr[8]

    # heat capacity and melt fraction
    heatcap = cp * Mmantle

    if Tm > 1420:
        _, meltfrac = get_meltfracb(g, Tm, Rp, Rc, Mmantle)
    else:
        meltfrac = 0.0

    # atmospheric pressure
    if Tr[8] > 647:
        Patm = Watm * g / (4 * np.pi * Rp**2)
    else:
        Patm = 10 ** (6.079 - 2261.10 / Tr[8]) * 1e5
        Matm = Patm * (4 * np.pi * Rp**2) / g
        if Matm > Watm:
            Patm = Watm * g / (4 * np.pi * Rp**2)

    # oxygen partial pressure
    PO2 = MO_tot * g / (4 * np.pi * Rp**2)

    # atmospheric flux
    flux = get_flux(Tsurf, Teq, Patm, Rp, g)

    # # use the following flux if get_flux function is ONLY calculating OSR
    #flux = get_flux(Tsurf, Patm, Rp, g)

    # atmospheric loss
    phi_H, phi_O = get_loss(
        t_flux, Lbol, t, PO2, Patm, tsat, a, Mp, Rp, LStar
    )

    # radioactive heating
    H_238U = 9.37e-5
    H_235U = 5.69e-4
    H_232Th = 2.69e-5
    H_40K = 2.79e-5

    Uran = 21e-9
    C_238U = 0.9927 * Uran
    C_235U = 0.0072 * Uran
    C_40K = 1.28 * Uran
    C_232Th = 4.01 * Uran

    l_238U = 0.155e-9
    l_235U = 0.985e-9
    l_232Th = 0.0495e-9
    l_40K = 0.555e-9

    Q = (
        C_238U * H_238U * np.exp(l_238U * (4.6e9 - t))
        + C_235U * H_235U * np.exp(l_235U * (4.6e9 - t))
        + C_232Th * H_232Th * np.exp(l_232Th * (4.6e9 - t))
        + C_40K * H_40K * np.exp(l_40K * (4.6e9 - t))
    ) * Mmantle

    # mantle heat flux
    q_mantle, Db, uc, Ra, nu = mantleheatflux(Tm, Tsurf, Rp, Rp, Rc, g, rho, FH2O, meltfrac)

    # degassing flux
    if FH2O > 1e-9:
        _, rmor, _, _ = degas2(
            Tm, Db, q_mantle, FH2O, Rp, g, Tsurf
        )
    else:
        rmor = 0.0

    # spreading rate
    S = 2 * Lridge * uc * 1.5
    Xd = 1.0

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

    # ------------------------------------------------------------------
    # DIFFERENTIAL EQUATIONS

    # mantle temperature
    dTr_dt[4] = (
        3.15569e7
        * (-4 * np.pi * Rp**2 * q_mantle + Q)
        / (cp * Mmantle)
    )

    # solid mantle water
    dTr_dt[5] = (
        -rmor * 4 * np.pi * Rp**2 * uc * Xd * 3.15569e7
    )

    # atmospheric water
    if Tr[6] > 1e3 and Tr[8] > 647:
        dTr_dt[6] = (
            (rmor * 4 * np.pi * Rp**2 * uc * Xd
             - 4 * np.pi * Rp**2 * phi_H * muH2O / 2 / muH)
            * 3.15569e7
        )
    else:
        dTr_dt[6] = (
            rmor * 4 * np.pi * Rp**2 * uc * Xd
        ) * 3.15569e7

    # atmospheric oxygen
    if Tr[6] > 1e3 and Tr[7] > 0 and Tr[8] > 647:
        dTr_dt[7] = (
            (4 * np.pi * Rp**2 * phi_H * muO / 2 / muH
             - 4 * np.pi * Rp**2 * phi_O)
            * 3.15569e7
        )
    elif Tr[6] < 1e3 and Tr[7] > 0 and Tr[8] > 647:
        dTr_dt[7] = (
            -4 * np.pi * Rp**2 * phi_O
        ) * 3.15569e7
    else:
        dTr_dt[7] = 0.0

    # surface temperature
    dTr_dt[8] = (
        4 * np.pi * Rp**2 * 3.15569e7
        * (
            (-flux + q_mantle)
            / (
                cpH2O * Patm * 4 * np.pi * Rp**2 / g
                + cp * (4.0 / 3.0) * np.pi * 3e3 * (Rp**3 - Db**3)
            )
        )
    )

    return dTr_dt
