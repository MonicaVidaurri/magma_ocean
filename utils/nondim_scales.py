import numpy as np


class StateScaler:
    def __init__(self, a0, M_ocean):
        """
        Dimensionless scaling for the state vector (see ODEs.combined_ode.STATE_NAMES).

        Parameters
        ----------
        a0 : float
            Initial semi-major axis (m).
        M_ocean : float
            Initial water inventory (kg); scales all volatile reservoirs.
        """
        # Characteristic Time: 1 Megayear (in seconds)
        self.t_scale = 3.15569e13

        self.y_scales = np.array([
            a0,          # 0: Semi-major axis [m]
            1.0,         # 1: Eccentricity [n/a]
            1e-5,        # 2: Star Spin Rate [rad/s]
            1e-5,        # 3: Planet Spin Rate [rad/s]
            1000.0,      # 4: Mantle Temp [K]
            M_ocean,     # 5: Mass Water (Solid) [kg]
            M_ocean,     # 6: Mass Water (Fluid: melt + atmosphere + ocean) [kg]
            M_ocean,     # 7: Mass Oxygen (Fluid) [kg]
            M_ocean,     # 8: Mass Oxygen (Solid) [kg]
        ], dtype=np.float64)

    def wrap_ode(self, physics_ode):
        """Wraps the physical ODE to handle non-dimensional IO."""
        def dimensionless_ode(t_nondim, y_nondim, *args, **kwargs):
            # 1. Re-dimensionalize inputs for the physics engine
            t_sec = t_nondim * self.t_scale
            y_dim = y_nondim * self.y_scales

            # 2. Calculate physical derivatives
            dydt_dim = physics_ode(t_sec, y_dim, *args, **kwargs)

            # 3. Non-dimensionalize derivatives for the solver
            dydt_nondim = dydt_dim * (self.t_scale / self.y_scales)

            return dydt_nondim
        return dimensionless_ode

    def scale_state(self, y_dim):
        return y_dim / self.y_scales

    def unscale_state(self, y_nondim):
        return y_nondim * self.y_scales[:, np.newaxis] if np.ndim(y_nondim) == 2 else y_nondim * self.y_scales

    def scale_time(self, t_sec):
        return t_sec / self.t_scale

    def unscale_time(self, t_nondim):
        return t_nondim * self.t_scale
