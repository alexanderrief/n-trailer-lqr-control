# -*- coding: utf-8 -*-
"""Spline reference trajectories with forward rollout and time reversal.

Planning a backward-feasible trajectory directly is hard. Instead, the
path is driven *forward* through the nonlinear model and the recorded
states are reversed in time afterwards:

    x_rev(t) = x(T - t),   u_rev(t) = -u(T - t)

Since the kinematic model is driftless (f(q, u) is linear in u), the
reversed trajectory satisfies the system equations exactly with negated
inputs -- only v1 and omega1 flip sign, the joint angles are unchanged.
The result is a valid reference for backward driving, around which the
time-varying LQR (ntrailer.lqr.TimeVaryingLQR) is designed.

Curve types:
    SplineTrajectory.bezier_s_curve(...)   degree-5 Bezier S-curve with
                                           horizontal start/end tangents
    SplineTrajectory.from_waypoints(...)   cubic spline through waypoints
"""
from math import comb

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, interp1d

from .model_builder import define_symbols, build_equations, explicit_func


class Reference:
    """Sampled reference trajectory (t, x, u) with interpolators."""

    def __init__(self, t, x, u):
        self.t = np.asarray(t, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.u = np.asarray(u, dtype=float)
        # cubic interpolation, clamped to the first/last sample
        self._x_interp = interp1d(self.t, self.x, axis=0, kind='cubic',
                                  bounds_error=False,
                                  fill_value=(self.x[0], self.x[-1]))
        self._u_interp = interp1d(self.t, self.u, axis=0, kind='cubic',
                                  bounds_error=False,
                                  fill_value=(self.u[0], self.u[-1]))

    @property
    def T(self):
        return float(self.t[-1])

    def x_ref(self, t):
        return self._x_interp(t)

    def u_ref(self, t):
        return self._u_interp(t)

    def reversed(self):
        """Time-inverted reference: drive the same path backwards.

        Only the inputs change sign; the recorded states (and with them
        the joint angles) are kept and replayed in reverse order.
        """
        return Reference(self.T -self.t[::-1], self.x[::-1].copy(), -self.u[::-1].copy())


class SplineTrajectory:
    """Planar reference curve r(t) for the tractor axle.

    The tractor inputs follow from the first and second derivative of
    the curve:
        theta1 = atan2(dy, dx)
        v1     = |dr/dt|
        omega1 = (ddy*dx - dy*ddx) / v1^2
    """

    def __init__(self, r_func, dr_func, ddr_func, T):
        self.r = r_func        # t -> (2,)
        self.dr = dr_func      # t -> (2,)
        self.ddr = ddr_func    # t -> (2,)
        self.T = float(T)

    # ---------- constructors ----------
    @classmethod
    def bezier_s_curve(cls, P0=(0.0, 0.0), P5=(2.0, 0.75),
                       h1=1.0, h2=0.3, g1=None, g2=None, T=20.0):
        """Degree-5 Bezier S-curve between two straights.

        Control points are placed along the horizontal tangent
        directions t0 = t1 = (1, 0):
            P1 = P0 + h1*t0,  P2 = P1 + h2*t0
            P4 = P5 - g1*t1,  P3 = P4 - g2*t1
        g1/g2 default to h1/h2 (symmetric curve).
        """
        g1 = h1 if g1 is None else g1
        g2 = h2 if g2 is None else g2
        t0 = np.array([1.0, 0.0])
        P0 = np.asarray(P0, float)
        P5 = np.asarray(P5, float)
        P1 = P0 + h1 * t0
        P2 = P1 + h2 * t0
        P4 = P5 - g1 * t0
        P3 = P4 - g2 * t0
        P = np.vstack([P0, P1, P2, P3, P4, P5])  # (6, 2)
        return cls.from_bezier(P, T)

    @classmethod
    def from_bezier(cls, control_points, T):
        """Bezier curve of arbitrary degree, parametrized over [0, T]."""
        P = np.asarray(control_points, dtype=float)
        deg = len(P) - 1
        dP = deg * np.diff(P, axis=0)            # hodograph (1st der.)
        ddP = (deg - 1) * np.diff(dP, axis=0)    # 2nd derivative

        def bernstein(points, s):
            d = len(points) - 1
            s = float(s)
            return sum(comb(d, i) * (1 - s) ** (d - i) * s ** i * points[i]
                       for i in range(d + 1))

        r = lambda t: bernstein(P, t / T)
        dr = lambda t: bernstein(dP, t / T) / T
        ddr = lambda t: bernstein(ddP, t / T) / T ** 2
        return cls(r, dr, ddr, T)

    @classmethod
    def from_waypoints(cls, points, T, bc_type='clamped'):
        """Cubic spline through waypoints, uniform in time.

        'clamped' end conditions give zero curve velocity at the ends;
        use bc_type='natural' for nonzero boundary speed.
        """
        points = np.asarray(points, dtype=float)
        knots = np.linspace(0.0, T, len(points))
        spline = CubicSpline(knots, points, axis=0, bc_type=bc_type)
        return cls(lambda t: spline(t),
                   lambda t: spline(t, 1),
                   lambda t: spline(t, 2),
                   T)

    # ---------- tractor reference ----------
    def tractor_inputs(self, t):
        """Heading, velocity and yaw rate of the tractor at time t."""
        dr = self.dr(t)
        ddr = self.ddr(t)
        v = float(np.hypot(dr[0], dr[1]))
        theta1 = float(np.arctan2(dr[1], dr[0]))
        if v < 1e-9:
            return theta1, 0.0, 0.0
        omega = float((ddr[1] * dr[0] - dr[1] * ddr[0]) / v ** 2)
        return theta1, v, omega

    # ---------- forward rollout through the nonlinear model ----------
    def rollout(self, n_trailers, l, d, dt=0.02, initial_offsets=None):
        """Drive the curve forward through the nonlinear model.

        The tractor inputs are taken from the curve derivatives; the
        trailer angles are obtained by integrating the full kinematics.
        Returns a Reference with samples every dt.

        initial_offsets: optional trailer angle offsets relative to the
        initial tractor heading (default: aligned chain).
        """
        n = int(n_trailers)
        l = np.asarray(l, dtype=float)
        d = np.asarray(d, dtype=float)

        syms = define_symbols(n)
        eqs, order = build_equations(syms, n)
        F_func, _, _ = explicit_func(eqs, order,
                                     syms['state_syms'], syms['param_syms'])

        def u_of_t(t):
            _, v, omega = self.tractor_inputs(t)
            return np.array([v, omega])

        theta1_0, _, _ = self.tractor_inputs(0.0)
        offsets = (np.zeros(n) if initial_offsets is None
                   else np.asarray(initial_offsets, dtype=float))
        x0 = np.concatenate((self.r(0.0),
                             [theta1_0],
                             theta1_0 + offsets))

        def rhs(t, x):
            p = np.concatenate((x, u_of_t(t), l, d))
            return np.array(F_func(*p), dtype=float).flatten()

        t_eval = np.arange(0.0, self.T + dt / 2, dt)
        sol = solve_ivp(rhs, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                        method='RK45', max_step=dt)
        if not sol.success:
            raise RuntimeError(f"forward rollout failed: {sol.message}")

        u_arr = np.array([u_of_t(t) for t in sol.t])
        return Reference(sol.t, sol.y.T, u_arr)


def rollout_constant_input(n_trailers, l, d, u_const, T, dt=0.02, x0=None):
    """Forward rollout with constant tractor inputs [v1, omega1].

    Used for the 'circle' backward-reference type: drive a steady
    circular path through the nonlinear model, then reverse in time.
    """
    n = int(n_trailers)
    l = np.asarray(l, dtype=float)
    d = np.asarray(d, dtype=float)
    u_const = np.asarray(u_const, dtype=float).flatten()

    syms = define_symbols(n)
    eqs, order = build_equations(syms, n)
    F_func, _, _ = explicit_func(eqs, order,
                                 syms['state_syms'], syms['param_syms'])

    if x0 is None:
        x0 = np.concatenate(([0.0, 0.0, 0.0], [0.0] * n))

    def rhs(t, x):
        p = np.concatenate((x, u_const, l, d))
        return np.array(F_func(*p), dtype=float).flatten()

    t_eval = np.arange(0.0, float(T) + dt / 2, dt)
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                    method='RK45', max_step=dt)
    if not sol.success:
        raise RuntimeError(f"constant-input rollout failed: {sol.message}")

    u_arr = np.tile(u_const, (len(sol.t), 1))
    return Reference(sol.t, sol.y.T, u_arr)


def build_backward_reference(traj_type, n_trailers, l, d, **params):
    """Build a time-reversed reference for backward driving.

    traj_type: 'bezier', 'waypoint', or 'circle'
    params: trajectory-specific settings (P5, h1, h2, T, waypoints, v, omega, ...)
    """
    n = int(n_trailers)
    l = np.asarray(l, dtype=float)
    d = np.asarray(d, dtype=float)
    T = float(params.get('T', 20.0))
    dt = float(params.get('dt', 0.02))
    traj_type = str(traj_type).lower()

    if traj_type == 'bezier':
        P5 = params.get('P5', [2.0, 0.75])
        curve = SplineTrajectory.bezier_s_curve(
            P0=tuple(params.get('P0', (0.0, 0.0))),
            P5=tuple(float(v) for v in P5),
            h1=float(params.get('h1', 1.0)),
            h2=float(params.get('h2', 0.3)),
            T=T)
        fwd = curve.rollout(n, l, d, dt=dt)
    elif traj_type == 'waypoint':
        waypoints = params.get('waypoints', [[0.0, 0.0], [1.0, 0.5], [2.0, 0.75]])
        bc_type = params.get('bc_type', 'clamped')
        curve = SplineTrajectory.from_waypoints(waypoints, T, bc_type=bc_type)
        fwd = curve.rollout(n, l, d, dt=dt)
    elif traj_type == 'circle':
        v = float(params.get('v', 1.0))
        omega = float(params.get('omega', 0.5))
        fwd = rollout_constant_input(n, l, d, [v, omega], T, dt=dt)
    else:
        raise ValueError(f"unknown trajectory type '{traj_type}' "
                         "(expected bezier, waypoint, or circle)")

    return fwd.reversed()


def reference_initial_state(ref, deflection_deg=0.0):
    """Start state on a backward reference with optional joint deflection."""
    x0 = np.asarray(ref.x_ref(0.0), dtype=float).flatten().copy()
    if deflection_deg:
        x0[3:] += np.deg2rad(float(deflection_deg))
    return list(x0)
