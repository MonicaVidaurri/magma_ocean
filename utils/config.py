"""Loading, merging, and validating the TOML model configuration."""
import copy
import tomllib
from pathlib import Path

from utils.general_utils import merge_dicts

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG = REPO_ROOT / 'baseline_config.toml'

# Keys that every merged configuration must provide, as (section path, key).
REQUIRED_KEYS = (
    (('simulation',), 'end_time_years'),
    (('simulation',), 'tides_on'),
    (('star',), 'stellar_track_file'),
    (('star',), 'mass_star_relative'),
    (('star',), 'host_radius'),
    (('star', 'xuv'), 'tsat_years'),
    (('star', 'tides'), 'moi_factor'),
    (('planet',), 'radius_planet_relative'),
    (('planet',), 'mass_planet_relative'),
    (('planet', 'orbit'), 'initial_semi_major_axis'),
    (('planet', 'orbit'), 'initial_eccentricity'),
    (('planet', 'tides'), 'love_method'),
)

# Keys removed in the 2026-09 rework. Finding one means the TOML was written for the old phase-based model.
RETIRED_KEYS = (
    (('star', 'xuv'), 'time_saturation_years', "renamed to 'tsat_years'"),
    (('planet', 'convection'), 'melt_fraction_threshold',
     "the rheological transition now comes from planet.rheology.critical_crystal_fraction"),
    (('planet', 'convection'), 'phase_transition_width', 'phases were replaced by one continuous model'),
    (('planet', 'thermodynamics'), 'dry_solid_threshold', 'phases were replaced by one continuous model'),
    (('planet', 'thermodynamics'), 'temp_solidus_static', 'melt is now always taken from the mantle grid'),
)


def _get_section(params, section_path):
    section = params
    for name in section_path:
        section = section.get(name, None)
        if not isinstance(section, dict):
            return None
    return section


def set_by_path(params, dotted_key, value):
    """
    Set a nested configuration value using a dotted key such as ``'planet.orbit.initial_eccentricity'``.

    Parameters
    ----------
    params : dict
        Configuration dictionary, modified in place.
    dotted_key : str
        Dot-separated path to the value.
    value : any
        New value.

    Raises
    ------
    KeyError
        If any intermediate section does not exist. New leaf keys are allowed, new sections are not, so that typos
        in section names fail loudly.
    """
    names = dotted_key.split('.')
    section = params
    for name in names[:-1]:
        if name not in section or not isinstance(section[name], dict):
            raise KeyError(f"Configuration section '{name}' in '{dotted_key}' does not exist.")
        section = section[name]
    section[names[-1]] = value


def resolve_path(path_str):
    """Resolve a path from a TOML file: absolute paths are kept, relative ones are taken from the repo root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_config(config_path, overrides=None, baseline_path=BASELINE_CONFIG):
    """
    Load a planet configuration on top of the baseline configuration.

    Parameters
    ----------
    config_path : str or path-like
        Planet-specific TOML. Relative paths are tried against the working directory first, then the repo root.
    overrides : dict, optional
        ``{dotted_key: value}`` pairs applied after merging (see `set_by_path`).
    baseline_path : str or path-like, optional
        TOML holding every default value.

    Returns
    -------
    params : dict
        Merged configuration. ``params['_meta']`` records the source files.

    Raises
    ------
    FileNotFoundError
        If a configuration or the stellar track file does not exist.
    KeyError
        If a required key is missing or a retired key is present.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with open(baseline_path, 'rb') as baseline_file:
        baseline = tomllib.load(baseline_file)
    with open(config_path, 'rb') as config_file:
        specific = tomllib.load(config_file)

    params = merge_dicts(copy.deepcopy(specific), baseline)

    if overrides:
        for dotted_key, value in overrides.items():
            set_by_path(params, dotted_key, value)

    for section_path, key, reason in RETIRED_KEYS:
        section = _get_section(specific, section_path)
        if section is not None and key in section:
            raise KeyError(f"'{'.'.join(section_path + (key,))}' in '{config_path.name}' is no longer used: {reason}.")

    for section_path, key in REQUIRED_KEYS:
        section = _get_section(params, section_path)
        if section is None or key not in section:
            raise KeyError(f"Required configuration key '{'.'.join(section_path + (key,))}' is missing.")

    track_path = resolve_path(params['star']['stellar_track_file'])
    if not track_path.exists():
        raise FileNotFoundError(f"Stellar track file '{track_path}' (star.stellar_track_file) not found.")

    params['_meta'] = {
        'config_path': str(config_path),
        'config_name': config_path.stem,
        'baseline_path': str(baseline_path),
        'stellar_track_path': str(track_path),
        'overrides': dict(overrides) if overrides else {},
    }
    return params
