rs_kwargs = dict(
    degree_l = 2,
    solve_for = None,
    starting_radius = 0.0,
    start_radius_tolerance = 1.0e-5,
    nondimensionalize = True,
    # Shooting method parameters
    use_kamata = False,
    integration_method = 'DOP853',
    integration_rtol = 1.0e-5,
    integration_atol = 1.0e-8,
    scale_rtols_bylayer_type = False,
    max_num_steps = 500_000,
    expected_size = 1000,
    max_ram_MB = 500,
    max_step = 0,
    # Propagation matrix method parameters
    use_prop_matrix = False,
    core_model = 0,
    # Equation of State solver parameters
    eos_method_bylayer = None,
    surface_pressure = 0.0,
    eos_integration_method = 'DOP853',
    eos_rtol = 1.0e-3,
    eos_atol = 1.0e-5,
    eos_pressure_tol = 1.0e-3,
    eos_max_iters = 40,
    # Error and log reporting
    verbose = False,
    warnings = True,
    raise_on_fail = False,
    perform_checks = True,
    log_info = False)
