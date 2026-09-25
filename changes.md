# Changes

## 2026-08 Rework

- Magma ocean froze ~100x too fast: the surface temperature ODE gave the surface a heat capacity equal to the whole magma ocean. Surface temperature is now quasi-static (interior flux = outgoing flux).
- Tidal energy was not conserved: in the magma ocean phase, tides changed the orbit but the heat went nowhere. All tidal heat now goes into the mantle budget (energy and angular momentum conserved to 1e-16).
- The three phases (magma ocean / wet solid / dry solid) were replaced by one continuous model. The old phases treated a mantle up to 43% molten as solid (no melt-atmosphere exchange, no O2 sink), and every RHS term jumped at phase switches.
- The planet's tidal model now has layers: a liquid core, a solid-like mantle with its own melt fraction, and a liquid magma layer. Previously the solid layer used a melt fraction that included the liquid ocean.
- Tides silently switched off when e < 1e-6 (forced e = 0 raised an error that a bare `except` swallowed).
- Ported tides to the TidalPy `_x` backend (classic support ends 2026-12-31).
- Stellar track was hard-coded to Proxima Cen for every planet. It is now `star.stellar_track_file`. Added a real BHAC15 solar track; the old `solardata.txt` held the Sun in a mislabeled column.
- Instellation now follows the stellar luminosity track and the current orbit.
- Config bugs: missing `proximab.toml`, the Sun's radius set to 1 m in `erf.toml`, and a `time_saturation_years` vs `tsat_years` key mismatch.
- `run_model.py` is importable (`run()`) and has a command line. Output names reflect the actual parameters.
- `auto_run.py` edited `run_model.py` by line number and rounded eccentricity to 2 decimals. Replaced by `auto_run_v3.py` / `auto_plot_v3.py` (tides on vs off).
- Remelting now returns solid-mantle water and oxygen to the melt. Escape tapers smoothly instead of driving water negative.
- Fixed float clamps in the iron redox solvers (`1 - 1e-20` rounds to 1.0).
- Added logging, tests (`tests/`), `model.tex`, and removed about 15 unused modules.
- After an independent review:
  - Solidus crossings are now solved exactly (smooth rates).
  - Hot (> 3000 K) and dry starts no longer crash.
  - Solid-state degassing no longer re-degasses the magma ocean.
  - The surface temperature takes the hottest root.
  - The fluid-water tolerance resolves the escape taper (it stalled runs after the water was gone).
  - Added a solid-mantle-water degassing-time metric.

## 2026-09 Radial Solver Tides

- The planet's tidal structure now comes from TidalPy's EOS solver: Birch-Murnaghan core and mantle, calibrated to the model's core and mantle masses (`[planet.tides.structure]`). The old constant-density structure was unstably stratified, so its radial-solver Love numbers were meaningless.
- `love_method = "radial_solver"` is now the default. The static magma layer (5% lighter than the solid) loads the solid below it, which cuts the solid's heating to about 1/4 to 1/14 of the homogeneous estimate. Runs take about a minute instead of a few seconds. `"homogeneous"` is still available.
- The eccentricity rate uses TidalPy's exact `dU/dM - dU/dw` sum (it was formed from two nearly cancelling sums at small e).
- Planet TOMLs use the BDF integrator. LSODA failed intermittently on the near-synchronous spin lock (relaxation time about a day), depending on its step history; BDF gives the same results and is faster.
- New option `planet.orbit.fixed_eccentricity` holds e at its initial value (a stand-in for forcing by other planets, e.g. a mean-motion resonance); a still evolves.
- Single-run figures replaced: orbit (a, e, spin), outgassing (rate, melt fraction, pressure), heating (tidal, radiogenic, viscosity, shear modulus). Example TOMLs in `examples/` (spin-orbit resonance cascade; eccentricity held at 0.05).
- Just-formed example TOMLs: Proxima b, TRAPPIST-1c and 1d (e held at the resonance-forced value), L 98-59 c; new BHAC15 0.3 Msun track `data/stellar_dataL98-59.txt`.
- New option `planet.fixed_spin` holds the planet's spin rate (a stand-in for an unmodeled torque such as atmospheric thermal tides).
- The implicit solvers get a bounded finite-difference Jacobian. scipy's default estimator grew the step for weakly coupled columns (solid water and oxygen) without limit until it overflowed and passed an infinite state to the model (2 of 65 exploration runs failed this way).
- The structure calibration uses a secant step; the plain ratio update cycled for planets of a few Earth masses, where self-compression makes mass grow faster than reference density.
- `model.tex` parameter table listed baseline values (core 0.19 R, 2 oceans) instead of trappist1e's (0.55 R, 1 ocean).
