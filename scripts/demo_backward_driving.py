#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward driving along a time-reversed reference (offline quickstart).

Reads vehicle geometry, trajectory, and initial condition from
scenario.yaml in the project root.  Runs a closed-loop TV-LQR simulation
and plots tractor reference, tractor path, and last-trailer position.

Usage:
    python scripts/demo_backward_driving.py [--no-plot]
"""
import argparse
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from ntrailer import scenario                                      # noqa: E402
from ntrailer.lqr import TimeVaryingLQR                                # noqa: E402
from ntrailer.model_builder import (define_symbols, build_equations,
                                    explicit_func, build_output)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plot', action='store_true',
                        help='skip plot generation')
    args = parser.parse_args()

    n = scenario.n_trailers()
    l = scenario.axle_distances()
    d = scenario.hitch_lengths()
    bwd = scenario.get_backward_reference()
    x0 = np.asarray(scenario.get_initial_state(), dtype=float)

    traj_type = scenario.trajectory_type()

    n_states = 2 + n + 1
    q = [10.0, 10.0] + [1.0] * (n_states - 2)
    tvlqr = TimeVaryingLQR(n, l, d, bwd, Q=q, R=[1.0, 1.0])

    syms = define_symbols(n)
    eqs, order = build_equations(syms, n)
    F_func, _, _ = explicit_func(eqs, order,
                                 syms['state_syms'], syms['param_syms'])
    H_func, _ = build_output(syms, n)

    def rhs_cl(t, x):
        u = tvlqr.control(t, x)
        return np.array(F_func(*np.concatenate((x, u, l, d)))).flatten()

    sol_cl = solve_ivp(rhs_cl, (0.0, bwd.T), x0, t_eval=bwd.t, max_step=0.02)

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sol_cl.y[0], sol_cl.y[1], 'C0', lw=1.5, label='tractor')
    if n > 0:
        p_dummy = np.concatenate(([0.0, 0.0], l, d))
        trailer_cl = np.array(
            [np.array(H_func(*x, *p_dummy)).flatten() for x in sol_cl.y.T])
        ax.plot(trailer_cl[:, 0], trailer_cl[:, 1], 'C2', lw=1.5,
                label='last trailer')
    ax.plot(bwd.x[:, 0], bwd.x[:, 1], 'k--', lw=1.5, alpha=0.5, 
            label='tractor reference')
    ax.plot(sol_cl.y[0, 0], sol_cl.y[1, 0], 'C0o', ms=6, label='start')
    ax.plot(sol_cl.y[0, -1], sol_cl.y[1, -1], 'C3o', ms=6, label='end')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f'Backward driving, {traj_type} '
                 f'(n_trailers={n}, deflection={scenario.deflection_deg():.0f}°)')

    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'backward_driving.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"plot saved to {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
