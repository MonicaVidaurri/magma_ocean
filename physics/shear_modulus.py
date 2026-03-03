import numpy as np

def calc_shear_modulus(
        melt_fraction,
        params
    ):
    
    # Store the baseline solid shear modulus from the config
    rheo = params['planet']['rheology']
    shear_modulus_solid   = rheo['solid_shear_modulus']
    shear_modulus_liquid  = rheo['liquid_shear_modulus']
    crit_crystal_frac     = rheo['critical_crystal_fraction']
    crit_melt_frac        = 1.0 - crit_crystal_frac
    melt_transition_width = rheo['melt_transition_width']
    shear_solid_softening = rheo['shear_solid_softening']

    if melt_fraction <= 0.0:
        # Completely Solid
        shear = shear_modulus_solid

    elif melt_fraction < crit_melt_frac:
        # Slow Softening Regime (Solid matrix intact but weakening)
        # Use an exponential decay based purely on melt fraction.
        # 'softening_alpha' controls how much it weakens before the critical threshold.
        # (e.g., an alpha of 10 means shear drops by ~99% by mf=0.45)
        shear = shear_modulus_solid * np.exp(-shear_solid_softening * melt_fraction)

    elif melt_fraction <= (crit_melt_frac + melt_transition_width):
        # Matrix Breakdown Regime (Steep drop-off to liquid)
        # First, calculate what the shear was exactly at the critical fraction to ensure continuous connection
        shear_at_crit = shear_modulus_solid * np.exp(-shear_solid_softening * crit_melt_frac)
        
        # Smooth logarithmic interpolation between the critical shear and the liquid limit
        progress = (melt_fraction - crit_melt_frac) / melt_transition_width
        log_shear = np.log(shear_at_crit) + progress * (np.log(shear_modulus_liquid) - np.log(shear_at_crit))
        shear = np.exp(log_shear)

    else:
        # Liquid / Suspension Regime
        # Small rocks floating in magma; shear cannot support planetary-scale tidal deformation
        shear = shear_modulus_liquid

    # Final safety clamp to prevent floating-point undershoots
    shear = np.clip(shear, shear_modulus_liquid, shear_modulus_solid)

    return shear