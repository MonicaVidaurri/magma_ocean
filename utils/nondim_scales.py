import numpy as np

class StateScaler:
    def __init__(self, phase, a0, Rp, M_ocean):
        """
        Initializes the dimensionless scaling arrays based on the ODE phase.
        
        Parameters:
        phase (int): 1, 2, or 3 corresponding to the ODE phase.
        a0 (float): Initial semi-major axis (meters).
        Rp (float): Planet radius (meters).
        M_ocean (float): Initial total water inventory (kg).
        """
        # Characteristic Time: 1 Megayear (in seconds)
        self.t_scale = 3.15569e13  
        
        if phase == 1:
            # 11 Dependent Variables
            self.y_scales = np.array([
                a0,          # 0: Semi-major axis [m]
                1.0,         # 1: Eccentricity [n/a]
                1e-5,        # 2: Star Spin Rate [rad/s]
                1e-5,        # 3: Planet Spin Rate [rad/s]
                1000.0,      # 4: Mantle Temp [K]
                Rp,          # 5: Solid Radius [m]
                M_ocean,     # 6: Mass Water (Solid) [kg]
                M_ocean,     # 7: Mass Water (MO/Atm) [kg]
                M_ocean,     # 8: Mass Oxygen (MO/Atm) [kg]
                M_ocean,     # 9: Mass Oxygen (Solid) [kg]
                1000.0       # 10: Surface Temp [K]
            ], dtype=np.float64)
            
        elif phase == 2:
            # 9 Dependent Variables
            self.y_scales = np.array([
                a0,          # 0: Semi-major axis [m]
                1.0,         # 1: Eccentricity [n/a]
                1e-5,        # 2: Star Spin Rate [rad/s]
                1e-5,        # 3: Planet Spin Rate [rad/s]
                1000.0,      # 4: Mantle Temp [K]
                M_ocean,     # 5: Mass Water (Mantle) [kg]
                M_ocean,     # 6: Mass Water (Atm) [kg]
                M_ocean,     # 7: Mass Oxygen (Atm) [kg]
                1000.0       # 8: Surface Temp [K]
            ], dtype=np.float64)
            
        elif phase == 3:
            # 8 Dependent Variables
            self.y_scales = np.array([
                a0,          # 0: Semi-major axis [m]
                1.0,         # 1: Eccentricity [n/a]
                1e-5,        # 2: Star Spin Rate [rad/s]
                1e-5,        # 3: Planet Spin Rate [rad/s]
                1000.0,      # 4: Mantle Temp [K]
                M_ocean,     # 5: Mass Water (Atm) [kg]
                M_ocean,     # 6: Mass Oxygen (Atm) [kg]
                1000.0       # 7: Surface Temp [K]
            ], dtype=np.float64)
            
        else:
            raise ValueError("Phase must be 1, 2, or 3.")

    def wrap_ode(self, physics_ode):
        """Wraps the physical ODE to handle non-dimensional IO."""
        def dimensionless_ode(t_nondim, y_nondim, *args, **kwargs):
            # 1. Re-dimensionalize inputs for the physics engine
            t_sec = t_nondim * self.t_scale
            y_dim = y_nondim * self.y_scales
            
            # 2. Calculate physical derivatives
            dydt_dim = physics_ode(t_sec, y_dim, *args, **kwargs)
            
            # 3. Non-dimensionalize derivatives for the solver
            # dy_nondim / dt_nondim = (dy_dim / y_scale) * (t_scale / dt_sec)
            dydt_nondim = dydt_dim * (self.t_scale / self.y_scales)
            
            return dydt_nondim
        return dimensionless_ode

    def wrap_event(self, physics_event):
        """Wraps event functions to handle non-dimensional IO."""
        if physics_event is None: 
            return None
        
        def dimensionless_event(t_nondim, y_nondim, *args, **kwargs):
            t_sec = t_nondim * self.t_scale
            y_dim = y_nondim * self.y_scales
            return physics_event(t_sec, y_dim, *args, **kwargs)
            
        # Preserve SciPy event attributes for terminal/direction flags
        dimensionless_event.terminal = getattr(physics_event, 'terminal', False)
        dimensionless_event.direction = getattr(physics_event, 'direction', 0)
        return dimensionless_event
        
    def scale_state(self, y_dim):
        return y_dim / self.y_scales
        
    def scale_time(self, t_sec):
        return t_sec / self.t_scale
        
    def unscale_time(self, t_nondim):
        return t_nondim * self.t_scale