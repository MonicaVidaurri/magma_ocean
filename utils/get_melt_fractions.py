import numpy as np
from scipy.integrate import trapezoid


def get_melt_fractions(temp_potential, grid):
    """
    Single-pass computation of bulk and MO melt fractions using a pre-computed grid.

    Replaces the pair of calls to get_meltfrac + get_meltfracb. Because the depth
    grid, pressure, solidus, and liquidus arrays are all constants (pre-computed in
    MantleGrid), only the adiabat needs to be evaluated on each ODE step.

    Parameters
    ----------
    temp_potential : float
        Mantle potential temperature in Kelvin.
    grid : MantleGrid
        Pre-computed static mantle grid (from build_mantle_grid).

    Returns
    -------
    exchange_pressure_gpa : float
        Pressure at the base of the deepest melt layer (GPa).
    meltfrac_bulk : float
        Volume-averaged melt fraction relative to the entire mantle (0–1).
    meltfrac_mo : float
        Volume-averaged melt fraction within the melt region only (0–1).
        Equals 0.0 when there is no distinct melt region.
    """
    # Only dynamic step: scale the pre-computed adiabat factor by current temperature
    temp_adiabat = temp_potential * grid.adiabat_factor

    # Melt fraction field
    melt_fraction = np.zeros(len(grid.depths))
    fully_molten  = temp_adiabat >= grid.liquidus
    melt_fraction[fully_molten] = 1.0

    partial = (temp_adiabat > grid.solidus) & ~fully_molten
    melt_fraction[partial] = ((temp_adiabat[partial] - grid.solidus[partial])
                              / grid.liq_offset)

    idx_melt = np.where(melt_fraction > 0.0)[0]
    if len(idx_melt) < 2:
        return 0.0, 0.0, 0.0

    idx_base = idx_melt[-1]

    # Exchange pressure at the first solid grid point below the melt base
    if idx_base < len(grid.depths) - 1:
        exchange_pressure_gpa = grid.pressures_gpa[idx_base + 1]
    else:
        exchange_pressure_gpa = grid.pressures_gpa[idx_base]

    # Bulk melt fraction: integrate over the entire mantle
    meltfrac_bulk = min(
        trapezoid(melt_fraction * grid.r2, grid.depths) / grid.vol_proxy_full, 1.0
    )

    # MO melt fraction: integrate over the melt region only
    n = idx_base + 1
    vol_proxy_mo = (grid.radii[0]**3 - grid.radii[idx_base]**3) / 3.0
    if vol_proxy_mo > 0.0:
        meltfrac_mo = min(
            trapezoid(melt_fraction[:n] * grid.r2[:n], grid.depths[:n]) / vol_proxy_mo,
            1.0
        )
    else:
        meltfrac_mo = meltfrac_bulk

    return exchange_pressure_gpa, meltfrac_bulk, meltfrac_mo
