import numpy as np

from physics.mantleheatflux import mantleheatflux
from utils.get_meltfracb import get_meltfracb
from utils.get_flux import get_flux
from utils.get_loss import get_loss

from math import isclose
from TidalPy.rheology import Andrade, Elastic, Newton
from TidalPy.toolbox import quick_dual_body_tidal_dissipation
from TidalPy.constants import G
from TidalPy.conversions.conversions_x import semi_a2orbital_motion

from .tide_parameters import rs_kwargs

def moODE3(t, Tr, Rp, Rc, Mmantle, Teq, rho, g, Ts, Ps, OLR, ASR, t_flux, Lbol, Xi, FeOt,
           Temp_K, P_Pa, tsat, FH2O, a, Mp, LStar, Rh, Mh, Mp):
    """ ODE when there is no magma ocean. Tectonic phase / last time phase. """
    
    # TODO: Inputs needed for tidal model
    
    # Leave constant 
    orbital_freq = ...
    eccentricity = ...
    rotation_rate = ...
    !! core_rho = ...

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

    # unpack state vector (MATLAB → Python indexing)
    Tm = Tr[0]       # mantle temperature
    Watm = Tr[1]     # mass of water in atmosphere
    if Watm < 0:
        Watm = 0.0

    MO_tot = Tr[2]   # mass of free O in atmosphere
    if MO_tot < 0:
        MO_tot = 0.0

    Tsurf = Tr[3]

    dTr_dt = np.zeros(4)

    t  # no-op, preserved from MATLAB

    #######################################################################
    # heat capacity and melt fraction
    heatcap = cp * Mmantle

    if Tm > 1420:
        _, meltfrac = get_meltfracb(g, Tm, Rp, Rc, Mmantle)
    else:
        meltfrac = 0.0

    #######################################################################
    # atmospheric pressure
    if Tr[3] > 647:
        Patm = Watm * g / (4 * np.pi * Rp**2)
    else:
        Patm = 10**(6.079 - 2261.10 / Tr[3]) * 1e5
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
    phi_H, phi_O = get_loss(t_flux, Lbol, t, PO2, Patm, tsat, a, Mp, Rp, LStar)

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
    !!! Need solid mantle viscosity for tides.

    if np.isnan(q_mantle):
        q_mantle
    
    #######################################################################
    # Tidal Heating
    ## Bottom up solidification -> Solid Core -> Solid Mantle -> Magma Ocean. 
    ## Constants / Assumptions
    core_bulk_modulus = 150.0e9  # Pa
    core_shear_modulus = 0.0      # Pa
    mantle_bulk_modulus = 100.0e9 # Pa
    mantle_shear_modulus = 50.0e9 # Pa
    core_bulk_viscosity = 1e30    # Pa s (does not matter under the elastic assumption)
    mantle_bulk_viscosity = 1e30  # Pa s (does not matter under the elastic assumption)
    core_shear_viscosity = 1000.0 # Pa s
    tides_on_flag = True

    if tides_on_flag:

        # Unpack other dependent variables
        eccentricity = Tr[7]
        semi_major_axis = Tr[8]
        spin_freq_p = Tr[9]
        orbital_freq = semi_a2orbital_motion(semi_major_axis, Mh, Mp)

        g_p = G * Mp / (Rp**2)
        g_h = G * Mh / (Rh**2)
        rho_bulk_p = Mp / ((4.0 / 3.0) * np.pi * Rp**3)
        rho_bulk_h = Mh / ((4.0 / 3.0) * np.pi * Rh**3)

        # For now just use basic MOI for now
        moi_h = (2.0 / 5.0) * Mh * Rh**2
        moi_p = (2.0 / 5.0) * Mp * Rp**2

        # Tidally active region viscosity
        tidal_scale_h = 1.0  # Amount of planet participating (full planet)
        tidal_scale_p = (Rp - Rc) / Rp  # Amount of planet participating in tides (just mantle)
        visc_h = 1.0e10  # star's viscosity (not used)
        visc_p = nu * rho_m  # the viscosity above is kinetic; we want dynamic so multiple by the layer's density.'
        shear_mod_h = 1.0e7  # Star shear (not used)
        shear_mod_p = 50.0e9
        rheology_h = 'cpl'  # Constant phase lag model
        rheology_p = 'Andrade'
        fixed_k2_h = 0.3
        fixed_k2_p = 0.3  # unused for target planet due to real rheology
        fixed_Q_h = 30000000
        fixed_Q_p = 30000000 # unused for target planet due to real rheology

        # Orbit/spin state (all these constants for now)
        eccentricity = 0.1 
        obliquity_h = 0.0
        obliquity_p = 0.0
        spin_freq_h = orbital_freq
        spin_freq_p = orbital_freq

        dissipation_results = quick_dual_body_tidal_dissipation(
            radii=(Rh, Rp), masses=(Mh, Mp), gravities=(g_h, g_p), densities=(rho_bulk_h, rho_bulk_p),
            mois=(moi_h, moi_p), viscosities=(visc_h, visc_p), shear_moduli=(shear_mod_h, shear_mod_p),
            rheologies=(rheology_h, rheology_p), obliquities=(obliquity_h, obliquity_p),
            spin_frequencies=(spin_freq_h, spin_freq_p), 
            tidal_scales=(tidal_scale_h, tidal_scale_p),
            fixed_k2s=(fixed_k2_h, fixed_k2_p), fixed_qs=(fixed_Q_h, fixed_Q_p),
            eccentricity=eccentricity, orbital_freq=orbital_freq,
            max_tidal_order_l=2, eccentricity_truncation_lvl=6,
            use_obliquity=(not isclose(obliquity_p, 0.0)),
            da_dt_scale=1., de_dt_scale=1.,
            dspin_dt_scale=1.
            )
        de_dt = dissipation_results['eccentricity_derivative']
        da_dt = dissipation_results['semi_major_axis_derivative']
        dspin_dt_h = dissipation_results['host']['spin_rate_derivative']
        dspin_dt_p = dissipation_results['secondary']['spin_rate_derivative']
        tidal_heating_h = dissipation_results['host']['tidal_heating']
        tidal_heating_p = dissipation_results['host']['tidal_heating']
    else:
        de_dt = 0.0
        da_dt = 0.0
        dspin_dt_h = 0.0
        dspin_dt_p = 0.0
        tidal_heating_h = 0.0
        tidal_heating_p = 0.0
    
    # Update derivative array.
    dTr_dt[]

    #######################################################################
    # differential equations
    dTr_dt[0] = (
        3.15569e7 * (-4 * np.pi * Rp**2 * q_mantle + Q)
        / (cp * Mmantle)
    )

    if Tr[3] > 647:
        dTr_dt[1] = (
            -4 * np.pi * Rp**2 * phi_H * muH2O / 2 / muH
        ) * 3.15569e7
    else:
        dTr_dt[1] = 0.0

    if Tr[3] > 647:
        dTr_dt[2] = (
            (4 * np.pi * Rp**2 * phi_H * muO / 2 / muH)
            - (4 * np.pi * Rp**2 * phi_O)
        ) * 3.15569e7
    else:
        dTr_dt[2] = 0.0

    dTr_dt[3] = (
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
