# -*- coding: utf-8 -*-
"""Smoke tests: symbolic model, output map, and TV-LQR backward driving.

Independent of scenario.yaml so local edits to the scenario do not
break the tests.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ntrailer.lqr import TimeVaryingLQR, wrap_angle
from ntrailer.model_builder import (build_equations, build_output,
                                    chain_positions, define_symbols,
                                    explicit_func)
from ntrailer.spline_trajectory import (build_backward_reference,
                                        reference_initial_state)

N = 2
L = np.array([0.19, 0.19])
D = np.array([0.08, 0.08])


def _compiled_model(n):
    syms = define_symbols(n)
    eqs, order = build_equations(syms, n)
    return syms, explicit_func(eqs, order, syms['state_syms'],
                               syms['param_syms'])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_model_shapes(n):
    _, (F, A, B) = _compiled_model(n)
    n_states = 3 + n
    args = np.concatenate((np.zeros(n_states), [1.0, 0.1],
                           np.full(n, 0.19), np.full(n, 0.08)))
    assert np.array(F(*args), dtype=float).shape == (n_states, 1)
    assert np.array(A(*args), dtype=float).shape == (n_states, n_states)
    assert np.array(B(*args), dtype=float).shape == (n_states, 2)


def test_output_map_matches_chain_positions():
    syms, _ = _compiled_model(N)
    H, _ = build_output(syms, N)
    x = np.array([0.3, -0.2, 0.4, 0.1, -0.3])
    h = np.array(H(*x, 0.0, 0.0, *L, *D), dtype=float).flatten()
    axles, _ = chain_positions(x, L, D)
    np.testing.assert_allclose(h, axles[-1], atol=1e-12)


def _tracking_error(sol, ref):
    err = [np.r_[x[:2] - ref.x_ref(t)[:2],
                 wrap_angle(x[2:] - ref.x_ref(t)[2:])]
           for t, x in zip(sol.t, sol.y.T)]
    return np.abs(np.array(err)).max(axis=1)


def test_tvlqr_stabilizes_backward_driving():
    """5 deg joint deflection: TV-LQR recovers, open loop jackknifes."""
    ref = build_backward_reference('bezier', N, L, D, T=10.0)
    lqr = TimeVaryingLQR(N, L, D, ref, Q=[10, 10, 1, 1, 1], R=[1, 1])
    _, (F, _, _) = _compiled_model(N)
    x0 = np.array(reference_initial_state(ref, deflection_deg=5.0))

    def simulate(control):
        def rhs(t, x):
            u = control(t, x)
            return np.array(F(*np.concatenate((x, u, L, D)))).flatten()
        return solve_ivp(rhs, (0.0, ref.T), x0, t_eval=ref.t, max_step=0.02)

    closed = _tracking_error(simulate(lqr.control), ref)
    open_ = _tracking_error(simulate(lambda t, x: ref.u_ref(t)), ref)

    assert closed[-1] < 0.25 * closed[0]
    assert open_[-1] > 10 * closed[-1]
