import numpy as np

# Assuming these are properly imported from your physics/utils modules
from utils.get_meltfrac import get_meltfrac
from utils.get_meltfracb import get_meltfracb
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_fO2 import get_fO2
from utils.get_loss import get_loss
from utils.get_flux import get_flux

from physics.mantleheatflux import mantleheatflux
from physics.radiogenics import get_radiogenic_heat
from physics.tides import calculate_tidal_dissipation
from TidalPy.utilities.conversions.conversions_x import semi_a2orbital_motion

def postprocess_magma_ocean(
        sol1,
        sol2,
        sol3,
        Rp,
        Rc,
        Mmantle,
        gp,
        OLR_interp,
        ASR,
        Teq,
        Xi,
        FeOt,
        t_flux,
        Lbol,
        tsat,
        a,
        Mp,
        LStar,
        params):
    """
    Post-processes the raw ODE solver outputs to reconstruct derived physical 
    variables (pressures, melt fractions, fluxes) that were not explicitly 
    saved in the state vectors.

    Parameters
    ----------
    sol1, sol2, sol3 : scipy.integrate.OdeResult
        Solver output objects for Phase 1, Phase 2, and Phase 3 respectively.
    Rp, Rc : float
        Radius of planet and radius of core in meters.
    Mmantle : float
        Mass of the mantle in kg.
    gp : float
        Surface gravity in m/s^2.
    OLR_interp, ASR, Teq : float/interpolators
        Radiative variables.
    Xi : ndarray
        1D array of initial oxide mole fractions.
    FeOt : float
        Bulk mass fraction of total iron.
    t_flux, Lbol : ndarray
        Stellar evolution tracks (time in years, luminosity relative to Sun).
    tsat : float
        XUV saturation time in years.
    a : float
        Semi-major axis in meters.
    Mp, LStar : float
        Planet mass (kg) and Stellar luminosity (Watts).
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    results : dict
        Nested dictionary containing time arrays, state vectors, and derived 
        atmospheric/magma properties for all three phases.
    """
    
    # --- Constants & Geometry ---
    c = params['constants']
    comp = params['planet']['oxide_composition']
    thermo = params['planet']['thermodynamics']
    
    sec_per_year   = c['seconds_per_year']
    molar_mass_O   = c['molar_mass_O']
    molar_mass_H2O = c['molar_mass_H2O']
    molar_mass_FeO = comp['molar_mass_FeO']
    
    surface_area   = 4.0 * np.pi * Rp**2
    
    volume_mantle = (4.0 / 3.0) * np.pi * (Rp**3 - Rc**3)
    rho_mantle    = Mmantle / volume_mantle

    # Extract stellar and simulation parameters required for tidal calculations
    Mh = params['star']['mass_star_relative'] * c['mass_sun']
    Rh = params['star']['host_radius']
    tides_on_flag = params['simulation']['tides_on']
    solidus_low_p_int = thermo['solidus_intercept_low_p']

    # =====================================================================
    # --- PHASE 1: Magma Ocean ---
    # =====================================================================
    t1_sec = sol1.t
    Tr1    = sol1.y
    n1     = Tr1.shape[1]

    # Initialize derived arrays
    Mmo      = np.zeros(n1)
    Wmo      = np.zeros(n1)
    meltfrac = np.zeros(n1)
    Patm     = np.zeros(n1)
    FH2O_arr = np.zeros(n1)
    kH2O_arr = np.zeros(n1)
    flux_arr = np.zeros(n1)
    PO2      = np.zeros(n1)
    FFeO1_5  = np.zeros(n1)
    nFeO1_5  = np.zeros(n1)
    nFeOt    = np.zeros(n1)
    fo2      = np.zeros(n1)
    phi_H    = np.zeros(n1)
    phi_O    = np.zeros(n1)
    Q_rad_1  = np.zeros(n1)
    Q_tid_1  = np.zeros(n1)

    for i in range(n1):
        temp_mantle = Tr1[4, i]
        radius_solid = Tr1[5, i]
        temp_surface = Tr1[10, i]
        
        # Protect against numerical undershoot
        Wmo[i] = max(Tr1[7, i], 0.0) 
        
        # Mass of active magma ocean
        Mmo[i] = (4.0 / 3.0) * np.pi * (Rp**3 - radius_solid**3) * rho_mantle

        # Melt Fraction
        _, meltfrac[i] = get_meltfrac(gp, temp_mantle, Rp, Rc, Mmantle, params)

        # Water Partitioning (Pressure)
        if Wmo[i] > 0.0:
            Patm[i], FH2O_arr[i], kH2O_arr[i] = get_pressure2(
                temp_mantle, radius_solid, Mmo[i], Mmantle, Rp, gp, Rc, Wmo[i], params
            )
        else:
            Patm[i] = 0.0

        # Radiative Flux
        flux_arr[i]  = get_flux(temp_surface, Teq, Patm[i], Rp, gp, params)

        # Oxygen Partitioning
        mass_oxygen_mo_atm = Tr1[8, i]
        if mass_oxygen_mo_atm > 0.0:
            PO2[i], FFeO1_5[i], _, nFeO1_5[i] = get_massbalance4(
                temp_mantle, Patm[i], Mmo[i], mass_oxygen_mo_atm, Xi, FeOt, gp, Rp, params
            )
            # Failsafe if oxygen is entirely in the atmosphere
            if PO2[i] <= 0.0:
                PO2[i] = mass_oxygen_mo_atm * gp / surface_area
                FFeO1_5[i] = 0.0
                nFeO1_5[i] = 0.0
        else:
            PO2[i] = 0.0

        # Oxygen Fugacity
        nFeOt[i] = (FeOt * Mmo[i]) / molar_mass_FeO
        moles_FeO = max(nFeOt[i] - nFeO1_5[i], 0.0)
        
        # Build exact 12-element array expected by get_fO2
        modified_Xi = np.concatenate([Xi[:10], [moles_FeO, nFeO1_5[i]]])
        fo2[i] = get_fO2(temp_mantle, Patm[i], modified_Xi, params)

        # Atmospheric Escape
        phi_H[i], phi_O[i] = get_loss(t_flux, Lbol, t1_sec[i], PO2[i], Patm[i], tsat, a, Mp, Rp, LStar, params)

        # Heating Diagnostics
        Q_rad_1[i] = get_radiogenic_heat(t1_sec[i], Mmantle, params)
        
        mass_frac_water_melt = Wmo[i] / Mmo[i] if Mmo[i] > 0.0 else 0.0
        _, _, _, _, nu = mantleheatflux(
            temp_mantle, temp_surface, radius_solid, Rp, Rc, gp, rho_mantle, mass_frac_water_melt, meltfrac[i], params
        )
        orbital_freq = semi_a2orbital_motion(Tr1[0, i], Mh, Mp)
        _, _, _, _, _, Q_tid_1[i] = calculate_tidal_dissipation(
            Tr1[1, i], orbital_freq, Tr1[3, i], Tr1[2, i], Rp, Rh, Mp, Mh,
            temp_mantle, Rc, nu, rho_mantle, meltfrac[i], tides_on_flag, params
        )

    # --- Phase 1 Global Balances ---
    # Total water = Solid Water (Tr1[6]) + Magma/Atm Water (Tr1[7])
    total_water = Tr1[6, :] + Tr1[7, :]
    H2O_lost_kg = total_water[0] - total_water
    
    # Stoichiometric oxygen left behind by H2O photolysis
    O_gained_kg = H2O_lost_kg * (molar_mass_O / molar_mass_H2O)
    
    # Theoretical O inventory (ignores O lost to space)
    theoretical_total_O = Tr1[8, 0] + O_gained_kg
    
    # Actual O inventory (Atmosphere/Magma Tr[8] + Solid Tr[9])
    actual_total_O = Tr1[8, :] + Tr1[9, :]


    # =====================================================================
    # --- PHASE 2: Solidification / Degassing ---
    # =====================================================================
    t2_sec = sol2.t
    Tr2    = sol2.y
    n2     = Tr2.shape[1]
    
    # In Phase 2: Tr2[6] is atm water, Tr2[7] is atm oxygen
    Patm_2 = Tr2[6, :] * gp / surface_area
    PO2_2  = Tr2[7, :] * gp / surface_area
    
    Q_rad_2 = np.zeros(n2)
    Q_tid_2 = np.zeros(n2)
    
    for i in range(n2):
        temp_mantle = Tr2[4, i]
        temp_surface = Tr2[8, i]
        mass_frac_water_solid = Tr2[5, i] / Mmantle
        
        mf2 = 0.0
        if temp_mantle > solidus_low_p_int:
            _, mf2 = get_meltfracb(gp, temp_mantle, Rp, Rc, Mmantle, params)
            
        Q_rad_2[i] = get_radiogenic_heat(t2_sec[i], Mmantle, params)
        
        _, _, _, _, nu2 = mantleheatflux(
            temp_mantle, temp_surface, Rp, Rp, Rc, gp, rho_mantle, mass_frac_water_solid, mf2, params
        )
        orbital_freq = semi_a2orbital_motion(Tr2[0, i], Mh, Mp)
        _, _, _, _, _, Q_tid_2[i] = calculate_tidal_dissipation(
            Tr2[1, i], orbital_freq, Tr2[3, i], Tr2[2, i], Rp, Rh, Mp, Mh,
            temp_mantle, Rc, nu2, rho_mantle, mf2, tides_on_flag, params
        )


    # =====================================================================
    # --- PHASE 3: Tectonic Steady-State ---
    # =====================================================================
    t3_sec = sol3.t
    Tr3    = sol3.y
    n3     = Tr3.shape[1]
    
    # In Phase 3: Tr3[5] is atm water, Tr3[6] is atm oxygen
    Patm_3 = Tr3[5, :] * gp / surface_area
    PO2_3  = Tr3[6, :] * gp / surface_area
    
    Q_rad_3 = np.zeros(n3)
    Q_tid_3 = np.zeros(n3)
    
    # Water in solid mantle stays constant through Phase 3, inherited from end of Phase 2
    FH2O3 = sol2.y[5, -1] / Mmantle
    
    for i in range(n3):
        temp_mantle = Tr3[4, i]
        temp_surface = Tr3[7, i]
        
        mf3 = 0.0
        if temp_mantle > solidus_low_p_int:
            _, mf3 = get_meltfracb(gp, temp_mantle, Rp, Rc, Mmantle, params)
            
        Q_rad_3[i] = get_radiogenic_heat(t3_sec[i], Mmantle, params)
        
        _, _, _, _, nu3 = mantleheatflux(
            temp_mantle, temp_surface, Rp, Rp, Rc, gp, rho_mantle, FH2O3, mf3, params
        )
        orbital_freq = semi_a2orbital_motion(Tr3[0, i], Mh, Mp)
        _, _, _, _, _, Q_tid_3[i] = calculate_tidal_dissipation(
            Tr3[1, i], orbital_freq, Tr3[3, i], Tr3[2, i], Rp, Rh, Mp, Mh,
            temp_mantle, Rc, nu3, rho_mantle, mf3, tides_on_flag, params
        )


    # =====================================================================
    # --- Compilation & Return ---
    # =====================================================================
    return {
        'phase1': {
            't': t1_sec, 'Tr': Tr1, 'Mmo': Mmo, 'Wmo': Wmo, 'meltfrac': meltfrac,
            'Patm': Patm, 'FH2O': FH2O_arr, 'kH2O': kH2O_arr, 'flux': flux_arr,
            'PO2': PO2, 'FFeO1_5': FFeO1_5, 'nFeO1_5': nFeO1_5, 'nFeOt': nFeOt, 
            'fo2': fo2, 'phi_H': phi_H, 'phi_O': phi_O, 'H2Olost': H2O_lost_kg, 
            'Ogained': O_gained_kg, 'totalO_theoretical': theoretical_total_O,
            'totalO_actual': actual_total_O,
            'Q_rad': Q_rad_1, 'Q_tid': Q_tid_1
        },
        'phase2': {
            't': t2_sec, 'Tr': Tr2, 'PO2': PO2_2, 'Patm': Patm_2,
            'Q_rad': Q_rad_2, 'Q_tid': Q_tid_2
        },
        'phase3': {
            't': t3_sec, 'Tr': Tr3, 'PO2': PO2_3, 'Patm': Patm_3,
            'Q_rad': Q_rad_3, 'Q_tid': Q_tid_3
        }
    }
