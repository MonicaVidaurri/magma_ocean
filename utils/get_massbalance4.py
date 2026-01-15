import numpy as np
from .get_massbalance2 import get_massbalance2
from .get_fO2 import get_fO2

def get_massbalance4(Tm, Patm, Mmo, MO_mo, Xi, FeOt, gp, Rp):
    '''
    Mass balance of FeO1.5 / O2 with simplified low-O branch.
    '''

    muO = 15.9994e-3
    muFeO1_5 = 159.689e-3 / 2
    muFeO = 71.845e-3

    nFeOt = FeOt * Mmo / muFeO
    nO_t = MO_mo / muO

    count = 0

    if nO_t < 1e-3 * nFeOt:
        y0 = nFeOt * 1e-3

        while count <= 100:
            y = getf(y0, Tm, Xi, Patm, nO_t, nFeOt, gp, Rp)

            if abs(y - y0) < 1e-12 and y > 0:
                nFeO1_5 = y
                nO_atm = nO_t - 0.5 * nFeO1_5
                PO2 = (nO_atm * muO) * gp / (4 * np.pi * Rp**2)
                FFeO1_5 = nFeO1_5 * muFeO1_5 / Mmo
                break

            count += 1

            if count >= 50:
                nFeO1_5 = nO_t * 2
                nO_atm = nO_t - 0.5 * nFeO1_5
                PO2 = (nO_atm * muO) * gp / (4 * np.pi * Rp**2)
                FFeO1_5 = nFeO1_5 * muFeO1_5 / Mmo
                break

            y0 = y

        # sanity check !!
        if nFeO1_5 > nFeOt:
            print('Warning: nFeO1_5 > nFeOt')

    else:
        PO2, FFeO1_5, mark, nFeO1_5 = get_massbalance2(
            Tm, Patm, Mmo, MO_mo, Xi, FeOt, gp, Rp
        )
        nO_atm = nO_t - 0.5 * nFeO1_5

        if nFeO1_5 > nFeOt:
            print('Warning: nFeO1_5 > nFeOt')

    # fallback for invalid PO2
    if PO2 == 0:
        X_fO2 = np.concatenate([Xi[:10], [nFeOt - 2*nO_t, 2*nO_t]])
        PO2 = get_fO2(Tm, Patm, X_fO2)

    return PO2, FFeO1_5, count, nFeO1_5


def getf(y, Tm, Xi, Patm, nO_t, nFeOt, gp, Rp):
    muO = 15.9994e-3

    term = (np.log(y / (nFeOt - y)) - 1.1492e4 / Tm + 6.675 + 2.243 * Xi[3] + 1.828 * Xi[9]
        - 3.201 * Xi[5] - 5.854 * Xi[6] - 6.215 * Xi[7] + 3.36 * (1 - 1673 / Tm - np.log(Tm / 1673))
        + 7.01e-7 * Patm / Tm + 1.54e-10 * (Tm - 1673) * Patm / Tm - 3.85e-17 * Patm**2 / Tm)

    return 2 * (nO_t - (4 * np.pi * Rp**2 / (muO * gp)) * np.exp(term / 0.196))
