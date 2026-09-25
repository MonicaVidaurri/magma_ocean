"""Stellar bolometric luminosity tracks (e.g., Baraffe et al. 2015)."""
import numpy as np


class StellarTrack:
    """
    Bolometric luminosity of the host star versus age, read from a tab-delimited file.

    The file must have a header row with columns ``Lbol`` (luminosity relative to the Sun) and ``tbol`` (age in
    years); see HOW_TO-make_stellardata.txt. Ages outside the tabulated range return the first or last value.

    Parameters
    ----------
    track_path : str or path-like
        Path to the track file.
    solar_luminosity : float
        Solar luminosity (W) used to convert the relative values.
    """

    def __init__(self, track_path, solar_luminosity):
        table = np.genfromtxt(track_path, names=True, delimiter='\t')
        if 'Lbol' not in table.dtype.names or 'tbol' not in table.dtype.names:
            raise ValueError(f"Stellar track '{track_path}' must have 'Lbol' and 'tbol' columns.")
        order = np.argsort(table['tbol'])
        self.age_years = np.asarray(table['tbol'][order], dtype=np.float64)
        self.luminosity_relative = np.asarray(table['Lbol'][order], dtype=np.float64)
        self.solar_luminosity = solar_luminosity

    def luminosity(self, age_years):
        """Bolometric luminosity (W) at the given age (years), linearly interpolated."""
        return float(np.interp(age_years, self.age_years, self.luminosity_relative)) * self.solar_luminosity
