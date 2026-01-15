import numpy as np
from utils import get_meltfrac, get_pressure2, get_massbalance4, get_fO2, get_loss, get_flux


def postprocess_magma_ocean(sol1, sol2, sol3, Rp, Rc, Mmantle, gp, OLR_interp, ASR, Teq, Xi, FeOt, t_flux, Lbol, tsat,
                            a, Mp, LStar):
    #########################################
    # Phase 1
    t1 = sol1.t
    Tr1 = sol1.y #mantle temperature
    n1 = Tr1.shape[1]

    Mmo = np.zeros(n1) #Mass of mantle
    Wmo = np.zeros(n1) #Water content in mantle
    meltfrac = np.zeros(n1)
    Patm = np.zeros(n1) #Atm pressure
    FH2O_arr = np.zeros(n1)
    kH2O_arr = np.zeros(n1)
    flux_arr = np.zeros(n1)
    PO2 = np.zeros(n1)  #O2 partial pressure
    FFeO1_5 = np.zeros(n1)
    nFeO1_5 = np.zeros(n1)
    nFeOt = np.zeros(n1)
    fo2 = np.zeros(n1) #oxygen fugacity
    phi_H = np.zeros(n1)
    phi_O = np.zeros(n1)

    for i in range(n1):
        Tm = Tr1[0, i]
        r_solid = Tr1[1, i]
        Wmo[i] = Tr1[3, i]
        Mmo[i] = 4 / 3 * np.pi * (Rp ** 3 - r_solid ** 3) * (Mmantle / (4 / 3 * np.pi * (Rp ** 3 - Rc ** 3)))

        if Wmo[i] < 0: Wmo[i] = 0.0

        _, meltfrac[i] = get_meltfrac.get_meltfrac(gp, Tm, Rp, Rc, Mmantle)

        if Wmo[i] > 0:
            Patm[i], FH2O_arr[i], kH2O_arr[i] = get_pressure2.get_pressure2(Tr1[0, i], Tr1[1, i], Mmo[i],
                                                              Mmantle, Rp, gp, Rc, Wmo[i])
        else:
            Patm[i] = 0.0
            FH2O_arr[i] = 0.0
            kH2O_arr[i] = 0.0

        flux_arr[i] = get_flux.get_flux(Tr1[6, i], Teq, Patm[i], Rp, gp)

        MO_mo = Tr1[4, i]
        if MO_mo > 0.0:
            PO2[i], FFeO1_5[i], _, nFeO1_5[i] = get_massbalance4.get_massbalance4(Tr1[0, i], Patm[i], Mmo[i],
                                                                 MO_mo, Xi, FeOt, gp, Rp)
            if PO2[i] < 0:
                PO2[i] = MO_mo * gp / (4 * np.pi * Rp ** 2)
                FFeO1_5[i] = 0.0
                nFeO1_5[i] = 0.0
        else:
            PO2[i] = 0.0
            FFeO1_5[i] = 0.0
            nFeO1_5[i] = 0.0

        nFeOt[i] = FeOt * Mmo[i] / 71.845e-3
        fo2[i] = get_fO2.get_fO2(Tr1[0, i], Patm[i], np.concatenate([Xi[:10], [nFeOt[i] - nFeO1_5[i]], [nFeO1_5[i]]]))
        phi_H[i], phi_O[i] = get_loss.get_loss(t_flux, Lbol, t1[i], PO2[i], Patm[i], tsat, a, Mp, Rp, LStar)

    H2Olost = (Tr1[2, :] + Tr1[3, :])[0] - (Tr1[2, :] + Tr1[3, :])
    Ogained = H2Olost * 15.9994e-3 / 18.015e-3
    totalO = Tr1[4, 0] + Ogained

    #########################################
    # Phase 2
    t2 = sol2.t
    Tr2 = sol2.y
    n2 = Tr2.shape[1]
    PO2_2 = Tr2[3, :] * gp / (4 * np.pi * Rp ** 2)
    Patm_2 = Tr2[2, :] * gp / (4 * np.pi * Rp ** 2)

    # Phase 3
    t3 = sol3.t
    Tr3 = sol3.y
    n3 = Tr3.shape[1]
    PO2_3 = Tr3[2, :] * gp / (4 * np.pi * Rp ** 2)
    Patm_3 = Tr3[1, :] * gp / (4 * np.pi * Rp ** 2)

    return {
        'phase1': {'t': t1, 'Tr': Tr1, 'Mmo': Mmo, 'Wmo': Wmo, 'meltfrac': meltfrac,
                   'Patm': Patm, 'FH2O': FH2O_arr, 'kH2O': kH2O_arr, 'flux': flux_arr,
                   'PO2': PO2, 'FFeO1_5': FFeO1_5, 'nFeO1_5': nFeO1_5,
                   'nFeOt': nFeOt, 'fo2': fo2, 'phi_H': phi_H, 'phi_O': phi_O,
                   'H2Olost': H2Olost, 'Ogained': Ogained, 'totalO': totalO},
        'phase2': {'t': t2, 'Tr': Tr2, 'PO2': PO2_2, 'Patm': Patm_2},
        'phase3': {'t': t3, 'Tr': Tr3, 'PO2': PO2_3, 'Patm': Patm_3}
    }

