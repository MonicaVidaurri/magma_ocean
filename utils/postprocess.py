import numpy as np

# Assuming these are properly imported from your physics/utils modules
from utils.get_meltfrac import get_meltfrac
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_fO2 import get_fO2
from utils.get_loss import get_loss
from utils.get_flux import get_flux

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
    
    sec_per_year   = c['seconds_per_year']
    molar_mass_O   = c['molar_mass_O']
    molar_mass_H2O = c['molar_mass_H2O']
    molar_mass_FeO = comp['molar_mass_FeO']
    
    surface_area   = 4.0 * np.pi * Rp**2
    
    volume_mantle = (4.0 / 3.0) * np.pi * (Rp**3 - Rc**3)
    rho_mantle    = Mmantle / volume_mantle

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

    for i in range(n1):
        temp_mantle = Tr1[4, i]
        radius_solid = Tr1[5, i]
        
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
        temp_surface = Tr1[10, i]
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
    
    # In Phase 2: Tr2[6] is atm water, Tr2[7] is atm oxygen
    Patm_2 = Tr2[6, :] * gp / surface_area
    PO2_2  = Tr2[7, :] * gp / surface_area


    # =====================================================================
    # --- PHASE 3: Tectonic Steady-State ---
    # =====================================================================
    t3_sec = sol3.t
    Tr3    = sol3.y
    
    Patm_3 = Tr3[5, :] * gp / surface_area
    PO2_3  = Tr3[6, :] * gp / surface_area


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
            'totalO_actual': actual_total_O
        },
        'phase2': {
            't': t2_sec, 'Tr': Tr2, 'PO2': PO2_2, 'Patm': Patm_2
        },
        'phase3': {
            't': t3_sec, 'Tr': Tr3, 'PO2': PO2_3, 'Patm': Patm_3
        }
    }
