import numpy as np

# Updated imports to ensure consistency
from utils.get_meltfrac import get_meltfrac
from utils.get_pressure2 import get_pressure2
from utils.get_massbalance4 import get_massbalance4
from utils.get_fO2 import get_fO2
from utils.get_loss import get_loss
from utils.get_flux import get_flux

def postprocess_magma_ocean(
        sol1, sol2, sol3,
        Rp, Rc, Mmantle, Mp,
        gp, Teq, a, LStar,
        Xi, FeOt,
        t_flux, Lbol, tsat,
        params):
    """
    Post-processes ODE results to reconstruct derived physical variables 
    using master TOML parameters.
    """
    
    # --- Unpack Constants from TOML ---
    univ = params['constants']
    comp = params['planet']['oxide_composition']
    
    sec_per_year = univ['seconds_per_year']
    mu_O         = univ['molar_mass_O']
    mu_H2O       = univ['molar_mass_h2o']
    mu_FeO       = comp['molar_mass_FeO']
    
    surface_area  = 4.0 * np.pi * Rp**2
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

        # Reconstruct Melt Fraction (Pass params)
        _, meltfrac[i] = get_meltfrac(gp, temp_mantle, Rp, Rc, Mmantle, params)

        # Reconstruct Water Partitioning (Pass params)
        if Wmo[i] > 0.0:
            Patm[i], FH2O_arr[i], kH2O_arr[i] = get_pressure2(
                temp_mantle, radius_solid, Mmo[i], Mmantle, Rp, gp, Rc, Wmo[i], params
            )
        else:
            Patm[i] = 0.0

        # Reconstruct Radiative Flux (Pass params)
        temp_surface = Tr1[10, i]
        flux_arr[i]  = get_flux(temp_surface, Teq, Patm[i], Rp, gp, params)

        # Reconstruct Oxygen Partitioning (Pass params)
        mass_oxygen_mo_atm = Tr1[8, i]
        if mass_oxygen_mo_atm > 0.0:
            PO2[i], FFeO1_5[i], _, nFeO1_5[i] = get_massbalance4(
                temp_mantle, Patm[i], Mmo[i], mass_oxygen_mo_atm, Xi, FeOt, gp, Rp, params
            )
            if PO2[i] <= 0.0:
                PO2[i] = mass_oxygen_mo_atm * gp / surface_area
                FFeO1_5[i] = 0.0
                nFeO1_5[i] = 0.0
        else:
            PO2[i] = 0.0

        # Reconstruct Oxygen Fugacity (Pass params)
        nFeOt[i] = (FeOt * Mmo[i]) / mu_FeO
        moles_FeO = max(nFeOt[i] - nFeO1_5[i], 0.0)
        
        # Build exact 12-element array expected by get_fO2
        # Indices 10 and 11 are FeO and FeO1.5
        modified_Xi = np.concatenate([Xi[:10], [moles_FeO, nFeO1_5[i]]])
        fo2[i] = get_fO2(temp_mantle, Patm[i], modified_Xi, params)

        # Reconstruct Atmospheric Escape (Pass params)
        phi_H[i], phi_O[i] = get_loss(
            t_flux, Lbol, t1_sec[i], PO2[i], Patm[i], tsat, a, Mp, Rp, LStar, params
        )

    # --- Phase 1 Global Balances ---
    total_water = Tr1[6, :] + Tr1[7, :]
    H2O_lost_kg = total_water[0] - total_water
    O_gained_kg = H2O_lost_kg * (mu_O / mu_H2O)
    
    theoretical_total_O = Tr1[8, 0] + O_gained_kg
    actual_total_O = Tr1[8, :] + Tr1[9, :]

    # =====================================================================
    # --- PHASE 2: Solidification / Degassing ---
    # =====================================================================
    t2_sec = sol2.t
    Tr2    = sol2.y
    Patm_2 = Tr2[6, :] * gp / surface_area
    PO2_2  = Tr2[7, :] * gp / surface_area

    # =====================================================================
    # --- PHASE 3: Tectonic Steady-State ---
    # =====================================================================
    t3_sec = sol3.t
    Tr3    = sol3.y
    Patm_3 = Tr3[5, :] * gp / surface_area
    PO2_3  = Tr3[6, :] * gp / surface_area

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
