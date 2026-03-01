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
        XUV_model=1):
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
    LStar_watts : float
        Static stellar bolometric luminosity [W]. Used for equilibrium temperature.
    XUV_model : int, optional
        1 = Ribas+ 2005 continuous decay model.
        2 = Saturation then immediate zero (cutoff model). Default is 1.

    Returns
    -------
    flux_loss_H : float
        Mass flux of hydrogen escaping to space [kg/m^2/s].
    flux_loss_O : float
        Mass flux of oxygen escaping to space [kg/m^2/s].
    """

    # --- 0. Exhaustion Check ---
    # If there is no water left in the atmosphere, hydrodynamic escape of H shuts off.
    if pressure_H2O < 1e-6:
        return 0.0, 0.0

    # --- Constants & Time Conversion ---
    G            = 6.6743e-11    # m^3/kg/s^2
    stefan_boltz = 5.67e-8       # W/m2/K4
    AU_meters    = 1.496e11      # m
    avogadro     = 6.022e23      # molecules/mole
    boltzmann    = 1.380649e-23  # J/K
    mass_proton  = 1.6726e-27    # kg

    molar_mass_H = 1.008    # g/mol
    molar_mass_O = 15.9994  # g/mol
    
    sec_per_year = 3.15569e7
    
    # Convert input integration time (seconds) to years for stellar track interpolation
    t_years = t_sec / sec_per_year

    # --- Planetary Parameters & Energy ---
    gravity = G * mass_planet / radius_planet**2
    albedo  = 0.25
    
    # Equilibrium Temperature (Escape Region Temperature Proxy)
    flux_at_planet       = LStar_watts / (4.0 * np.pi * semi_major_axis**2)
    absorbed_stellar_rad = (1.0 - albedo) * flux_at_planet / 4.0
    temp_eq              = (absorbed_stellar_rad / stefan_boltz)**0.25

    # --- XUV Flux Evolution ---
    f_sat              = 1e-3
    decay_beta         = -1.23
    heating_efficiency = 0.1

    # Fractional XUV luminosity relative to Lbol
    f_xuv = f_sat * (t_flux_years / t_sat_years)**decay_beta

    if XUV_model == 1:
        # Ribas decay: flat during saturation, power-law decay after
        f_xuv = np.where(t_flux_years < t_sat_years, f_sat, f_xuv)
    elif XUV_model == 2:
        # Step function: flat during saturation, zero after
        f_xuv = np.where(t_flux_years < t_sat_years, f_sat, 0.0)

    # Convert relative Lbol back to Watts (3.828e26 is standard LSun)
    luminosity_xuv_watts = f_xuv * Lbol_relative * 3.828e26
    flux_xuv_orbit       = luminosity_xuv_watts / (4.0 * np.pi * semi_major_axis**2)

    # Interpolate to current time using the converted t_years
    if t_years > t_flux_years[0]:
        current_Fxuv = np.interp(t_years, t_flux_years, flux_xuv_orbit)
    else:
        current_Fxuv = flux_xuv_orbit[0]

    # --- Hydrodynamic Escape (Energy-Limited Base) ---
    grav_potential = G * mass_planet / radius_planet
    
    # Energy-limited mass flux (kg/m^2/s)
    phi_energy_limited = (heating_efficiency * current_Fxuv / 4.0) / grav_potential 

    # --- Diffusion-Limited Escape (Hydrogen Dragging Oxygen) ---
    temp_escape = temp_eq
    
    # Baseline molecule flux of H
    phi_H_ref_molecules = phi_energy_limited / (molar_mass_H * mass_proton) 

    # Molar fractions assuming a pure H2O source gas being dissociated
    X_H = 2.0 / 3.0
    X_O = 1.0 / 3.0

    # Binary diffusion parameter (b)
    binary_diff_coeff = 4.8e17 * (temp_escape)**0.75 * 100.0  # scaled to SI

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
