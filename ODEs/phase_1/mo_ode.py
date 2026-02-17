import numpy as np
from physics.mantleheatflux import mantleheatflux
from utils.get_meltfrac import get_meltfrac
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_flux import get_flux
from utils.get_loss import get_loss
import pandas as pd

from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion


def moODE(t, Tr, Rp, Rc, Mmantle, Teq, rho, g, Ts, Ps, OLR, ASR, t_flux, Lbol, Xi, FeOt,
          Temp_K, P_Pa, tsat, Mp, LStar, Rh, Mh, tides_on_flag):
    """ First time phase; There is a magma ocean. Entire mantle is molten. """
    # ------------------------------------------------------------------
    # constants
    sigma = 5.67e-8
    dHf = 4e5
    alpha = 2e-5
    cp = 1.2e3

    muO = 15.9994e-3
    muH2O = 18.015e-3
    muFeO1_5 = 159.689e-3 / 2
    muFeO = 71.845e-3
    NA = 6.022e23
    muH = 1.008e-3
    cpH2O = 2e3

    # mid-ocean ridge length
    Lridge = 2 * np.pi * Rp * 1.5
    
    # stellar = pd.read_csv('data/stellar_dataTrappist1.txt', sep='\t')
    # Lbol = stellar['Lbol']

    # For this model the state vector equals:
    #   0: Semi major axis
    #   1: Eccentricity
    #   2: Host spin Freq
    #   3: Target spin freq
    #   4: Mantle temperature
    #   5: Solidification Radius (solid mantle below; liquid magma ocean above)
    #   6: Water fraction in the solid mantle
    #   7: Mass of water in the magma ocean
    #   8: Mass of Excess Oxygen in the Magma Ocean
    #   9: Mass of Excess Oxygen in the Solid Mantle
    #   10: Surface temperature
    # Build derivative array
    dTr_dt = np.zeros(11, dtype=np.float64)
    # Unpack state vector
    semi_a = Tr[0]
    eccentricity = Tr[1]
    spin_freq_h = Tr[2]
    spin_freq_p = Tr[3]
    orbital_freq = semi_a2orbital_motion(semi_a, Mh, Mp)

    Tm = Tr[4]
    rs = Tr[5]
    FH2Os = Tr[6] / Mmantle

    Mmo = (4.0 / 3.0) * np.pi * rho * (Rp**3 - rs**3)
    if rs > Rp:
        rs = Rp
        Mmo = 0.0

    Wmo = Tr[7]
    if Wmo > 0.0:
        FH2O = Wmo / Mmo
    else:
        Wmo = 0.0
        FH2O = 0.0

    MO_mo = Tr[8]
    MO_sol = Tr[9]
    Tsurf = Tr[10]

    # solidus parameters
    if (Rp - rs) > 405e9 / 77.89 / rho / g:
        Tsola = 26.53e-9
        Tsolb = 1825
    else:
        Tsola = 104.42e-9
        Tsolb = 1420

    # B coefficient
    B = (cp * (Tsolb * alpha - Tsola * rho * cp)) / (
        g * (Tsola * rho * cp - alpha * Tm) ** 2
    )

    # melt fraction
    _, meltfrac = get_meltfrac(g, Tm, Rp, Rc, Mmantle)

    if meltfrac >= 0.4:
        heatcap = cp * (4.0 / 3.0) * rho * np.pi * (Rp**3 - rs**3)
    else:
        heatcap = cp * Mmantle

    # atmospheric pressure
    if Wmo > 0.0:
        Patm, FH2O, kH2O = get_pressure2(
            Tm, rs, Mmo, Mmantle, Rp, g, Rc, Wmo
        )
    else:
        Patm = 0.0
        FH2O = 0.0
        kH2O = 0.0

    # oxygen mass balance
    if MO_mo > 0.0:
        PO2, FFeO1_5, _, _ = get_massbalance4(
            Tm, Patm, Mmo, MO_mo, Xi, FeOt, g, Rp
        )
        if PO2 < 0.0:
            PO2 = MO_mo * g / (4 * np.pi * Rp**2)
            FFeO1_5 = 0.0
    else:
        PO2 = 0.0
        FFeO1_5 = 0.0

    # atmospheric flux
    flux = get_flux(Tsurf, Teq, Patm, Rp, g)

    # # use the following flux if get_flux function is ONLY calculating OSR
    # flux = get_flux(Tsurf, Patm, Rp, g)

    # atmospheric loss
    phi_H, phi_O = get_loss(t_flux, Lbol, t, PO2, Patm, tsat, semi_a, Mp, Rp, LStar)

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
    q_mantle, Db, uc, Ra, nu = mantleheatflux(Tm, Tsurf, rs, Rp, Rc, g, rho, FH2Os, meltfrac)

    # spreading rate
    S = 2 * Lridge * uc

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

    # dTm/dt
    if rs < Rp:
        dTr_dt[4] = 3.15569e7 * (
            (-4 * np.pi * Rp**2 * q_mantle + Q)
            / (cp * (4.0 / 3.0) * np.pi * rho * (Rp**3 - rs**3)
                - (4 * np.pi * rho * dHf * rs**2) * B))
    else:
        dTr_dt[4] = 3.15569e7 * ((-4 * np.pi * Rp**2 * q_mantle + Q) / (cp * Mmantle))

    # drs/dt
    if rs < Rp:
        dTr_dt[5] = B * dTr_dt[4]
    else:
        dTr_dt[5] = 0.0

    # solid mantle water
    if FH2O > 0:
        dTr_dt[6] = -kH2O * FH2O * (-4 * np.pi * rho * rs**2 * dTr_dt[5])
    else:
        dTr_dt[6] = 0.0

    # magma ocean + atmosphere water
    if Wmo > 0:
        dTr_dt[7] = (-dTr_dt[6] - 3.15569e7 * 4 * np.pi * Rp**2 * phi_H * muH2O / 2 / muH)
    else:
        dTr_dt[7] = 0.0

    # solid mantle oxygen
    dTr_dt[9] = (
        FFeO1_5
        * 4
        * np.pi
        * rho
        * rs**2
        * dTr_dt[5]
        * 0.5
        * muO
        / muFeO1_5
    )

    # magma ocean + atmosphere oxygen
    if Wmo > 0 and PO2 > 0:
        dTr_dt[8] = (
            4
            * np.pi
            * Rp**2
            * 3.15569e7
            * (phi_H * muO / 2 / muH - phi_O)
            - dTr_dt[9]
        )
    elif Wmo > 0 and PO2 <= 0:
        dTr_dt[8] = (
            4
            * np.pi
            * Rp**2
            * 3.15569e7
            * (phi_H * muO / 2 / muH)
            - dTr_dt[9]
        )
    elif Wmo <= 0 and PO2 > 0:
        dTr_dt[8] = -dTr_dt[9]
    else:
        dTr_dt[8] = 0.0

    # surface temperature
    dTr_dt[10] = 3.15569e7 * 4 * np.pi * Rp**2 * (
        (-flux + q_mantle)
        / (
            cpH2O * Patm * 4 * np.pi * Rp**2 / g
            + cp * (4.0 / 3.0) * np.pi * 3e3 * (Rp**3 - Db**3)
        )
    )

    return dTr_dt
