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
        stellar_luminosity,
        temp_surface,
        params):
    """
    Calculates the XUV-driven hydrodynamic atmospheric escape fluxes of Hydrogen 
    and Oxygen, assuming a water-dominated upper atmosphere.

    Uses a diffusion-limited escape model (e.g., Luger & Barnes) where Hydrogen 
    drags Oxygen with it if the XUV flux is high enough.

    Parameters
    ----------
    t_flux_years : ndarray
        Time array corresponding to the stellar evolution tracks [years].
    Lbol_relative : float or ndarray
        Stellar bolometric luminosity relative to LSun.
    t_sec : float
        Current integration time [seconds].
    pressure_O2 : float
        Partial pressure of Oxygen in the atmosphere [Pa].
    pressure_H2O : float
        Partial pressure of Water Vapor in the atmosphere [Pa].
    t_sat_years : float
        XUV saturation time for the host star [years].
    semi_major_axis : float
        Semi-major axis of the planet's orbit [m].
    mass_planet : float
        Mass of the planet [kg].
    radius_planet : float
        Radius of the planet [m].
    stellar_luminosity : float
        Static stellar bolometric luminosity [W]. Used for equilibrium temperature.
    params : dict
        Main configuration dictionary containing TOML parameters.

    Returns
    -------
    flux_loss_H : float
        Mass flux of hydrogen escaping to space [kg/m^2/s].
    flux_loss_O : float
        Mass flux of oxygen escaping to space [kg/m^2/s].
    """

    # --- Exhaustion Check ---
    # If there is no water left in the atmosphere, hydrodynamic escape of H shuts off.
    if pressure_H2O < 1e-6:
        return 0.0, 0.0

    # --- Unpack Parameters ---
    c        = params['constants']
    p        = params['planet']
    star_xuv = params['star']['xuv']
    esc      = params['planet']['atmosphere']['escape']

    use_surf_temp_for_escape = params['simulation']['surface_temp_escape']

    # --- Constants & Time Conversion ---
    G            = c['G']                # m^3/kg/s^2
    stefan_boltz = c['stefan_boltzmann'] # W/m2/K4
    avogadro     = c['avogadro']         # molecules/mole
    boltzmann    = c['boltzmann']        # J/K
    mass_proton  = c['mass_proton']      # kg

    # The original logic uses g/mol for molecular flux equations. 
    # We pull the MKS (kg/mol) values from TOML and convert inline to preserve the math.
    molar_mass_H = c['molar_mass_H'] * 1000.0  # g/mol
    molar_mass_O = c['molar_mass_O'] * 1000.0  # g/mol
    
    sec_per_year = c['seconds_per_year']
    
    # Convert input integration time (seconds) to years for stellar track interpolation
    t_years = t_sec / sec_per_year

    # --- Planetary Parameters & Energy ---
    gravity = G * mass_planet / radius_planet**2
    albedo  = p['albedo']
    
    # Equilibrium Temperature (Escape Region Temperature Proxy)
    flux_at_planet       = stellar_luminosity / (4.0 * np.pi * semi_major_axis**2)
    absorbed_stellar_rad = (1.0 - albedo) * flux_at_planet / 4.0
    temp_eq              = (absorbed_stellar_rad / stefan_boltz)**0.25

    # --- XUV Flux Evolution ---
    f_sat              = star_xuv['f_sat']
    decay_beta         = star_xuv['decay_beta']
    heating_efficiency = star_xuv['heating_efficiency']
    XUV_model          = esc['xuv_model']

    # Molar fractions for the source gas
    X_H = esc['source_frac_H']
    X_O = esc['source_frac_O']

    # Fractional XUV luminosity relative to Lbol
    f_xuv = f_sat * (t_flux_years / t_sat_years)**decay_beta

    if XUV_model == 1:
        # Ribas decay: flat during saturation, power-law decay after
        f_xuv = np.where(t_flux_years < t_sat_years, f_sat, f_xuv)
    elif XUV_model == 2:
        # Step function: flat during saturation, zero after
        f_xuv = np.where(t_flux_years < t_sat_years, f_sat, 0.0)

    # Convert relative Lbol back to Watts using master solar luminosity
    luminosity_xuv_watts = f_xuv * Lbol_relative * c['lum_sun']
    flux_xuv_orbit       = luminosity_xuv_watts / (4.0 * np.pi * semi_major_axis**2)

    # Interpolate to current time using the converted t_years
    if t_years > t_flux_years[0]:
        current_Fxuv = np.interp(t_years, t_flux_years, flux_xuv_orbit)
    else:
        current_Fxuv = flux_xuv_orbit[0]

    if use_surf_temp_for_escape:
        # --- Expanded Radius / Atmospheric Inflation ---
        # Calculates how far the XUV absorption level is pushed out by the hot surface.
        total_pressure = pressure_H2O + pressure_O2
        P_xuv = 1e-4  # Typical pressure at the XUV tau=1 level (1 nbar)
        
        if total_pressure > P_xuv:
            # Get MKS molar masses for standard physics equations (kg/mol)
            mu_H_mks = c['molar_mass_H']
            mu_O_mks = c['molar_mass_O']
            mean_molar_mass = (X_H * mu_H_mks + X_O * mu_O_mks) / (X_H + X_O)
            
            # Scale height based on surface temperature
            gas_constant = boltzmann * avogadro
            scale_height = (gas_constant * temp_surface) / (mean_molar_mass * gravity)
            
            # Integrate scale height up to the XUV level
            n_scale_heights = np.log(total_pressure / P_xuv)
            R_xuv = radius_planet + (scale_height * n_scale_heights)
        else:
            R_xuv = radius_planet

        # Geometric amplification factor for the expanded absorbing area and lower gravity
        expansion_factor = (R_xuv / radius_planet)**3

        # --- Hydrodynamic Escape (Energy-Limited Base) ---
        grav_potential = G * mass_planet / radius_planet
        
        # Energy-limited mass flux (kg/m^2/s) scaled by the inflated radius
        phi_energy_limited = ((heating_efficiency * current_Fxuv / 4.0) / grav_potential) * expansion_factor 

        # --- Diffusion-Limited Escape (Hydrogen Dragging Oxygen) ---
        # When the atmosphere is this inflated, the upper atmosphere temperature 
        # should scale partially with the surface, not just the stellar equilibrium.
        temp_escape = max(temp_eq, temp_surface * 0.15) 
        
        # Baseline molecule flux of H
        phi_H_ref_molecules = phi_energy_limited / (molar_mass_H * mass_proton)
    else:
        # --- Hydrodynamic Escape (Energy-Limited Base) ---
        grav_potential = G * mass_planet / radius_planet
        
        # Energy-limited mass flux (kg/m^2/s)
        phi_energy_limited = (heating_efficiency * current_Fxuv / 4.0) / grav_potential 

        # --- Diffusion-Limited Escape (Hydrogen Dragging Oxygen) ---
        temp_escape = temp_eq
        
        # Baseline molecule flux of H
        phi_H_ref_molecules = phi_energy_limited / (molar_mass_H * mass_proton) 

    # Molar fractions assuming a pure H2O source gas being dissociated
    X_H = esc['source_frac_H']
    X_O = esc['source_frac_O']

    # Binary diffusion parameter (b) pulled from TOML
    binary_diff_coeff = esc['diffusion_scaling'] * (temp_escape)**0.75

    gamma = 1.0 / (1.0 + X_O * molar_mass_O / (X_H * molar_mass_H))
    
    # Critical mass calculations for crossover
    mu_c_ref = molar_mass_H + (boltzmann * temp_escape * phi_H_ref_molecules) / \
        (binary_diff_coeff * gravity * X_H * mass_proton)
    mu_c     = molar_mass_O + gamma * (mu_c_ref - molar_mass_O)

    # Actual molecule fluxes
    phi_H_molecules = phi_H_ref_molecules * (mu_c / mu_c_ref)
    
    # Oxygen is dragged if the flow is strong enough
    phi_O_molecules = (X_O / X_H) * phi_H_molecules * ((mu_c - molar_mass_O) / (mu_c - molar_mass_H))
    phi_O_molecules = max(phi_O_molecules, 0.0)

    # Critical flux threshold for dragging Oxygen
    phi_H_critical = (binary_diff_coeff * gravity * X_H * mass_proton) * (molar_mass_O - molar_mass_H) / (boltzmann * temp_escape)

    # --- Convert to Mass Flux (kg/m^2/s) ---
    # Convert from (molecules/m^2/s) -> (moles/m^2/s) -> (g/m^2/s) -> (kg/m^2/s)
    flux_loss_H = (phi_H_molecules / avogadro) * molar_mass_H * 1e-3
    
    if phi_H_molecules >= phi_H_critical:
        flux_loss_O = (phi_O_molecules / avogadro) * molar_mass_O * 1e-3
    else:
        flux_loss_O = 0.0

    return flux_loss_H, flux_loss_O
