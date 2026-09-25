# DEPENDENCIES
Python 3.10+
TidalPy >= 0.8.0 (uses only the new `_x` backend)
numpy
scipy
pandas
matplotlib
tqdm
pytest, pytest-xdist (tests only)

--------------------------------------------------------------------------
QUICK START

Run one planet (the planet TOML is merged on top of baseline_config.toml):

    python run_model.py trappist1e.toml
    python run_model.py trappist1e.toml --tides on --end-time 1e8 --output-dir output/test
    python run_model.py trappist1e.toml --set planet.orbit.initial_eccentricity=0.05 --no-plot

Example cases (TRAPPIST-1e with tides) are in examples/:

    python run_model.py examples/trappist1e_spin_cascade.toml --output-dir output/examples
    python run_model.py examples/trappist1e_fixed_ecc.toml --output-dir output/examples

Just-formed planets (hot start, 1-day spin): examples/proxima_b.toml, trappist1c.toml, trappist1d.toml,
l98-59c.toml. Each run also saves its three figures in --output-dir (skip with --no-plot).

Or from Python:

    from run_model import run
    results = run('trappist1e.toml', overrides={'simulation.tides_on': True})
    print(results['summary'])

Each run writes, into --output-dir (default: output/):
  <label>_output.txt    tab-delimited time series (column names match the older output, new columns appended)
  <label>_summary.json  scalar metrics (magma ocean lifetime, water-loss times, final PO2, ...)
  <label>_*.png         orbit (a, e, spin), outgassing (rate, melt, pressure), and heating (tidal,
                        radiogenic, viscosity, shear modulus) figures

Tides on vs off parameter sweeps (edit the SWEEP dictionary at the top of the script):

    python auto_run_v3.py --workers 8
    python auto_plot_v3.py results/tides_sweep_trappist1e

Tests:

    pytest -n 6 tests

The XUV model (high/low) is set with planet.atmosphere.escape.xuv_model in the TOML files. The stellar luminosity
track is set with star.stellar_track_file. Instructions for making your own stellar data file are in
HOW_TO-make_stellardata.txt. The model currently contains stellar data for the Sun, TRAPPIST-1, and Proxima Centauri.

The physics is described in model.tex. What changed in the 2026-09 rework is summarized in changes.md.

--------------------------------------------------------------------------
Howdy!!

This model is a python translation of GOOEY, a coupled magma ocean-atmosphere model
designed to track the evolution of oxygen and water in a steam atmosphere undergoing
hydrodynamic escape, developed by Laura Schaefer.

The paper for GOOEY can be found here: http://doi.org/10.3847/0004-637X/829/2/63
and the model can be accessed here: https://purl.stanford.edu/rk050tc3031
For additional reaading + an intercomparison between magma ocean models, check out
what the CHILI team has been cookin: https://arxiv.org/pdf/2511.16142

Feel free to email me at monica.r.vidaurri@nasa.gov (or if that don't work,
monavidaurri@gmail.com) if you've got any questions! I'd highly encourage you to read
Laura's paper first - all the core physics is the same, I just put everything
in python terms. (In other words, I'm not the model expert here lol I'm just
the translator.)

Since then (2026) the model has gained tidal dissipation and orbital/spin evolution
(via TidalPy), and the three-phase structure (magma ocean / wet solid / dry solid) was
replaced by a single continuous model; see changes.md and model.tex.

This model doesn't have a name but I don't like the name GOOEY (you can tell
Laura I said that - she already knows). I haven't come up with anything clever
yet, but maybe you can help :) but I ain't callin it GOOEY that's for
damn sure; I just think we can do better.

Good luck and have fun!
