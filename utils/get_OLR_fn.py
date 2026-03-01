import numpy as np
from scipy.special import gamma
from scipy.integrate import trapezoid, cumulative_trapezoid

def get_OLR_fn(
        gravity,
        temp_eq,
        temp_surface,
        pressure_surface_pa,
        params,
        calculation_mode=1,
        use_constant_cp=True,
        cp_file_path=None):
    """
    Calculates the Outgoing Longwave Radiation (OLR) for a planetary atmosphere 
    using gray-gas radiative transfer and master TOML parameters.
    """

    # --- Unpack Parameters ---
    univ    = params['constants']
    atm     = params['planet']['atmosphere']
    mat     = params['planet']['material']
    num     = params['numerical']

    sigma   = univ['stefan_boltzmann']
    R_univ  = univ['gas_constant']
    mu_h2o  = univ['molar_mass_H2O']
    
    # Specific gas constant (R)
    R_spec = R_univ / mu_h2o

    # Radiative properties
    kappa_ir  = atm['kappa_ir']
    cos_theta = atm['cos_theta']

    # Stratospheric Boundary ---
    # Skin temperature (Eddington approximation)
    temp_skin = temp_eq / (2.0**0.25)

    # Total column optical depth
    tau_infinity = kappa_ir * pressure_surface_pa / (gravity * cos_theta)

    # =====================================================================
    # --- Mode 1: Analytic Solution (Optically Thick) ---
    # =====================================================================
    if calculation_mode == 1:
        cp_const = mat['specific_heat_h2o']
        
        if tau_infinity < num['tau_thick_threshold']:
             # Use a simple print or logging if needed
             pass
            
        # Power term: 4 * R / Cp
        power_term = 4.0 * R_spec / cp_const
        
        # OLR = sigma * T^4 * tau^-beta * Gamma(1 + beta)
        olr_flux = (sigma * temp_surface**4 * tau_infinity**(-power_term) * gamma(1.0 + power_term))

    # =====================================================================
    # --- Mode 2: Numerical Integration (Schwarzschild) ---
    # =====================================================================
    elif calculation_mode == 2:
        n_levels = num['olr_n_levels']

        # --- Build Profile ---
        if use_constant_cp:
            cp_const = mat['specific_heat_h2o']
            # Log-spaced pressure from surface to top (1 Pa)
            pressure_profile = np.logspace(np.log10(pressure_surface_pa), 0, n_levels)
            temp_profile = temp_surface * (pressure_profile / pressure_surface_pa)**(R_spec / cp_const)
            
        else:
            if cp_file_path is None:
                raise ValueError("cp_file_path required for variable Cp mode.")

            cp_data = np.loadtxt(cp_file_path)
            temp_profile = np.logspace(np.log10(temp_surface), 1, n_levels)
            cp_profile   = np.interp(temp_profile, cp_data[:, 0], cp_data[:, 1]) * 1e3
            
            ln_temp = np.log(temp_profile)
            integral_ln_P = cumulative_trapezoid(cp_profile / R_spec, ln_temp, initial=0)
            pressure_profile = pressure_surface_pa * np.exp(integral_ln_P)

        # Apply Stratopause
        temp_profile[temp_profile < temp_skin] = temp_skin

        # --- Radiative Transfer ---
        tau_profile = kappa_ir * pressure_profile / (gravity * cos_theta)
        transmittance = np.exp(-tau_profile)

        # Integrate: B_surf * Trans_surf + Integral( B dTrans )
        bb_surface = sigma * temp_surface**4
        bb_profile = sigma * temp_profile**4
        
        integral_emission = trapezoid(bb_profile, transmittance)
        olr_flux = bb_surface * transmittance[0] + integral_emission

    else:
        raise ValueError("calculation_mode must be 1 (analytic) or 2 (numerical).")

    return olr_flux
