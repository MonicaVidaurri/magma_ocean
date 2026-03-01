import numpy as np

def get_loss(
        t_flux_years,
        Lbol_relative,
        t_sec,
        pressure_O2,
        pressure_H2O,
        t_sat_years,
        semi_major_axis,
        mass_planet,
        radius_planet,
        LStar_watts,
        params,
        XUV_model=1):
    """
    Calculates XUV-driven hydrodynamic escape fluxes of H and O.
    
    Parameters
    ----------
    t_sec : float
        Current integration time in seconds.
    params : dict
        Master configuration dictionary containing [constants], [star.xuv], 
        and [planet.escape].
    """

    # --- Exhaustion Check ---
    if pressure_H2O < 1e-6:
        return 0.0, 0.0

    # --- Unpack Constants & Parameters ---
    univ = params['constants']
    xuv  = params['star']['xuv']
    esc  = params['planet']['atmosphere']['escape']

    G            = univ['G']
    stefan_boltz = univ['stefan_boltzmann']
    boltzmann    = univ['boltzmann']
    avogadro     = univ['avogadro']
    mass_proton  = univ['mass_proton']
    mu_H         = univ['molar_mass_H']
    mu_O         = univ['molar_mass_O']
    L_sun        = univ['solar_luminosity']
    
    # Time conversion
    t_years = t_sec / univ['seconds_per_year']

    # --- Energy & Temperature ---
    gravity = G * mass_planet / radius_planet**2
    
    # Equilibrium Temperature proxy for escape region
    flux_at_planet = LStar_watts / (4.0 * np.pi * semi_major_axis**2)
    absorbed_rad   = (1.0 - params['planet']['albedo']) * flux_at_planet / 4.0
    temp_eq        = (absorbed_rad / stefan_boltz)**0.25
    temp_escape    = temp_eq

    # --- XUV Flux Evolution ---
    f_sat = xuv['f_sat']
    
    # Fractional XUV luminosity relative to Lbol
    f_xuv_track = f_sat * (t_flux_years / t_sat_years)**xuv['decay_beta']

    if XUV_model == 1:
        f_xuv_track = np.where(t_flux_years < t_sat_years, f_sat, f_xuv_track)
    elif XUV_model == 2:
        f_xuv_track = np.where(t_flux_years < t_sat_years, f_sat, 0.0)

    # Convert to physical flux at orbit
    flux_xuv_orbit = (f_xuv_track * Lbol_relative * L_sun) / (4.0 * np.pi * semi_major_axis**2)

    # Interpolate to current time
    if t_years > t_flux_years[0]:
        current_Fxuv = np.interp(t_years, t_flux_years, flux_xuv_orbit)
    else:
        current_Fxuv = flux_xuv_orbit[0]

    # --- Hydrodynamic Escape (Energy-Limited) ---
    grav_potential = G * mass_planet / radius_planet
    phi_energy_limited = (xuv['heating_efficiency'] * current_Fxuv / 4.0) / grav_potential 

    # --- Diffusion-Limited Drag Model ---
    phi_H_ref_molecules = phi_energy_limited / (mu_H * mass_proton) 

    # Stoichiometry and Diffusion
    X_H = esc['source_frac_H']
    X_O = esc['source_frac_O']
    b_diff = esc['diffusion_scaling'] * (temp_escape)**0.75

    gamma = 1.0 / (1.0 + X_O * mu_O / (X_H * mu_H))
    
    # Critical mass calculations
    mu_c_ref = mu_H + (boltzmann * temp_escape * phi_H_ref_molecules) / \
               (b_diff * gravity * X_H * mass_proton)
    mu_c     = mu_O + gamma * (mu_c_ref - mu_O)

    # Molecule fluxes
    phi_H_molecules = phi_H_ref_molecules * (mu_c / mu_c_ref)
    phi_O_molecules = (X_O / X_H) * phi_H_molecules * ((mu_c - mu_O) / (mu_c - mu_H))
    phi_O_molecules = max(phi_O_molecules, 0.0)

    # Drag threshold
    phi_H_critical = (b_diff * gravity * X_H * mass_proton) * (mu_O - mu_H) / (boltzmann * temp_escape)

    # --- Final Mass Fluxes (kg/m^2/s) ---
    # Particle -> Moles -> kg
    flux_loss_H = (phi_H_molecules / avogadro) * mu_H
    
    if phi_H_molecules >= phi_H_critical:
        flux_loss_O = (phi_O_molecules / avogadro) * mu_O
    else:
        flux_loss_O = 0.0

    return flux_loss_H, flux_loss_O
