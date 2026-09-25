import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config


@pytest.fixture(scope='session')
def trappist_params():
    """TRAPPIST-1e configuration with tides on (fresh copy per session)."""
    return load_config('trappist1e.toml', overrides={'simulation.tides_on': True})


@pytest.fixture(scope='session')
def trappist_model(trappist_params):
    from ODEs.combined_ode import MagmaOceanModel
    return MagmaOceanModel(trappist_params)
