import numpy as np
from scipy.special import gamma
from scipy.integrate import trapezoid, cumulative_trapezoid

def get_OLR_fn(
        gravity,
        temp_eq,
        temp_surface,
        pressure_surface_pa,
        calculation_mode=1,
        use_constant_cp=True,
        cp_file_path=None):
    """
    Calculates the Outgoing Longwave Radiation (OLR) for a planetary atmosphere 
    assuming a gray-gas, two-stream approximation.

    Parameters
    ----------
    gravity : float
        Surface gravitational acceleration in m/s^2.
    temp_eq : float
        Planetary equilibrium temperature in Kelvin.
    temp_surface : float
        Surface temperature in Kelvin.
    pressure_surface_pa : float
        Surface atmospheric pressure in Pascals.
    calculation_mode : int, optional
        1 = Analytic OLR (Assumes optically thick atmosphere).
        2 = Numerical integration of the Schwarzschild equation. Default is 1.
    use_constant_cp : bool, optional
        True = Use constant heat capacity for the adiabat.
        False = Use variable heat capacity (requires cp_file_path). Default is True.
    cp_file_path : str, optional
        Path to a data file for variable cp (format: Temp [K] vs cp [kJ/kg/K]). 
        Required if use_constant_cp is False.

    Returns
    -------
    olr_flux : float
        Outgoing longwave radiation flux at the top of the atmosphere in W/m^2.
    """

    # --- Thermodynamic & Radiative Constants ---
    stefan_boltzmann  = 5.67e-8    # sigma (W/m^2/K^4)
    gas_constant_univ = 8.314462   # Rstar (J/K/mol)
    molar_mass_h2o    = 18.015e-3  # kg/mol
    specific_gas_constant = gas_constant_univ / molar_mass_h2o  # R (J/kg/K)
    
    # Radiative properties (Two-stream approximation)
    kappa_ir  = 1e-5  # Gray infrared absorption coefficient (m^2/kg)
    cos_theta = 0.5   # Diffusivity factor (mu)

    # Stratospheric skin temperature (Eddington approximation boundary)
    temp_skin    = temp_eq / (2.0**0.25)

    # Total column optical depth
    tau_infinity = kappa_ir * pressure_surface_pa / (gravity * cos_theta)

    # =====================================================================
    # --- Mode 1: Analytic Solution (Optically Thick Regime) ---
    # =====================================================================
    if calculation_mode == 1:
        cp_const = 2000.0  # J/kg/K
        
        # The analytic Gamma function solution assumes tau_infinity >> 1.
        if tau_infinity < 10.0:
            print(f"Warning: Analytic OLR used with low optical depth (tau={tau_infinity:.2f}). " 
                  f"Results may overpredict flux. Consider numerical mode.")
            
        power_term = 4.0 * specific_gas_constant / cp_const
        
        olr_flux = (stefan_boltzmann * temp_surface**4 * tau_infinity**(-power_term) * gamma(1.0 + power_term))

    # =====================================================================
    # --- Mode 2: Numerical Integration (Schwarzschild Equation) ---
    # =====================================================================
    elif calculation_mode == 2:
        n_levels = 100

        # --- Build Atmospheric Profile (Surface -> TOA) ---
        if use_constant_cp:
            cp_const = 2000.0  # J/kg/K
            # Log-spaced pressure from surface down to 1 Pa
            pressure_profile = np.logspace(np.log10(pressure_surface_pa), 0, n_levels)
            temp_profile = temp_surface * (pressure_profile / pressure_surface_pa)**(specific_gas_constant / cp_const)
            
        else:
            if cp_file_path is None:
                raise ValueError("cp_file_path must be provided when use_constant_cp is False.")

            # Load cp data and convert from kJ/kg/K to J/kg/K
            cp_data = np.loadtxt(cp_file_path)
            
            # Log-spaced temperature from surface down to 10 K
            temp_profile = np.logspace(np.log10(temp_surface), 1, n_levels)
            cp_profile   = np.interp(temp_profile, cp_data[:, 0], cp_data[:, 1]) * 1e3
            
            # Integrate d(lnP) = (Cp/R) * d(lnT)
            # Because temp_profile is descending, d_ln_T is negative, making the integral correctly negative.
            ln_temp          = np.log(temp_profile)
            integral_ln_P    = cumulative_trapezoid(cp_profile / specific_gas_constant, ln_temp, initial=0)
            pressure_profile = pressure_surface_pa * np.exp(integral_ln_P)

        # --- Apply Isothermal Stratosphere ---
        temp_profile[temp_profile < temp_skin] = temp_skin

        # --- Radiative Transfer ---
        tau_profile   = kappa_ir * pressure_profile / (gravity * cos_theta)
        transmittance = np.exp(-tau_profile)

        # Planck function profiles
        blackbody_surface = stefan_boltzmann * temp_surface**4
        blackbody_profile = stefan_boltzmann * temp_profile**4

        # Integrate: OLR = B_surf * Trans_surf + Integral( B dTrans )
        # Note: Transmittance increases monotonically from surface to TOA, so dz is positive.
        integral_atmospheric_emission = trapezoid(blackbody_profile, transmittance)
        
        olr_flux = blackbody_surface * transmittance[0] + integral_atmospheric_emission

    else:
        raise ValueError("calculation_mode must be 1 (analytic) or 2 (numerical).")

    return olr_flux
