# -*- coding: utf-8 -*-
"""Time-varying LQR design for backward driving of the n-trailer chain.

Kept free of any PyMoskito imports so headless scripts can reuse the
exact same control law that runs inside the GUI.
"""
import numpy as np

from .model_builder import define_symbols, build_equations, explicit_func


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]. Needed before feeding angle errors
    into the state feedback, otherwise a 2*pi jump saturates the inputs."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class TimeVaryingLQR:
    """Finite-horizon time-varying LQR along a reference trajectory.

    Designed for trajectory tracking (in particular the time-reversed
    backward-driving references from ntrailer.spline_trajectory):
    A(t), B(t) are evaluated along the reference and the matrix Riccati
    differential equation

        -dS/dt = A(t)^T S + S A(t) - S B(t) R^-1 B(t)^T S + Q

    is integrated backwards from the terminal condition S(T) = S_T
    (default: 0). The feedback gain K(t) = R^-1 B(t)^T S(t) is stored
    on the reference time grid and interpolated at runtime.
    """

    def __init__(self, n_trailers, l, d, reference, Q=None, R=None,
                 S_T=None):
        self.n = int(n_trailers)
        self.l = np.asarray(l, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.ref = reference

        n_states = 2 + (self.n + 1)
        self.Q = np.eye(n_states) if Q is None else np.diag(np.asarray(Q, float))
        self.R = np.eye(2) if R is None else np.diag(np.asarray(R, float))
        self.R_inv = np.linalg.inv(self.R)

        syms = define_symbols(self.n)
        eqs, order = build_equations(syms, self.n)
        _, self.A_func, self.B_func = explicit_func(
            eqs, order, syms['state_syms'], syms['param_syms'])

        S_T = np.zeros((n_states, n_states)) if S_T is None else np.asarray(S_T, float)
        self._solve_riccati(S_T)

    def _matrices_at(self, t):
        phi = np.concatenate((self.ref.x_ref(t), self.ref.u_ref(t),
                              self.l, self.d))
        A = np.array(self.A_func(*phi), dtype=float)
        B = np.array(self.B_func(*phi), dtype=float)
        return A, B

    def _solve_riccati(self, S_T):
        """Integrate the Riccati ODE backwards over the reference grid."""
        from scipy.integrate import solve_ivp

        n_states = self.Q.shape[0]

        def rhs(t, s_flat):
            S = s_flat.reshape(n_states, n_states)
            S = 0.5 * (S + S.T)  # enforce symmetry against drift
            A, B = self._matrices_at(t)
            dS = -(A.T @ S + S @ A
                   - S @ B @ self.R_inv @ B.T @ S + self.Q)
            return dS.flatten()

        t_grid = self.ref.t
        sol = solve_ivp(rhs, (t_grid[-1], t_grid[0]), S_T.flatten(),
                        t_eval=t_grid[::-1], method='RK45',
                        max_step=float(t_grid[1] - t_grid[0]))
        if not sol.success:
            raise RuntimeError(f"Riccati integration failed: {sol.message}")

        S_traj = sol.y.T[::-1].reshape(len(t_grid), n_states, n_states)
        K_traj = np.empty((len(t_grid), 2, n_states))
        for i, t in enumerate(t_grid):
            _, B = self._matrices_at(t)
            S = 0.5 * (S_traj[i] + S_traj[i].swapaxes(-1, -2))
            K_traj[i] = self.R_inv @ B.T @ S

        from scipy.interpolate import interp1d
        self._K_interp = interp1d(t_grid, K_traj, axis=0,
                                  bounds_error=False,
                                  fill_value=(K_traj[0], K_traj[-1]))

    def gain(self, t):
        """Interpolated feedback gain K(t)."""
        return np.asarray(self._K_interp(t))

    def control(self, t, x):
        """u = u_ref(t) - K(t) (x - x_ref(t)) with wrapped angle errors."""
        x = np.asarray(x, dtype=float).flatten()
        e = x - self.ref.x_ref(t)
        e[2:] = wrap_angle(e[2:])
        return self.ref.u_ref(t) - self.gain(t) @ e
