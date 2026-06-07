# -*- coding: utf-8 -*-
"""Symbolic derivation of the n-trailer kinematics.

The model is set up implicitly as E(x, u) * x_dot = b(x, u) and converted
to an explicit ODE plus its Jacobians. All expressions are compiled with
sympy.lambdify, so the simulation itself runs on plain numpy.

Argument order of the compiled functions (do not change, the PyMoskito
models and the controllers rely on it):
    [x1, x2, theta1, ..., theta_{n+1}, v1, omega1, l2, ..., l_{n+1}, d1, ..., dn]
"""
import sympy as sp
from sympy import Matrix


def define_symbols(n):
    """Create all symbols for a chain with n trailers.

    Returns a dict; 'state_syms' and 'param_syms' define the argument
    order expected by every lambdified function in this module.
    """
    syms = {}
    syms['x1'], syms['x2'] = sp.symbols('x1 x2')
    syms['theta'] = sp.symbols(f'theta1:{n + 2}')
    syms['dot_x1'], syms['dot_x2'], syms['dot_theta1'] = sp.symbols(
        'dot_x1 dot_x2 dot_theta1')
    syms['theta_dot'] = sp.symbols(f'theta_dot1:{n + 2}')
    syms['v1'], syms['omega1'] = sp.symbols('v1 omega1')
    syms['l_syms'] = sp.symbols(f'l2:{n + 2}')
    syms['d_syms'] = sp.symbols(f'd1:{n + 1}')

    syms['state_syms'] = [syms['x1'], syms['x2']] + list(syms['theta'])
    syms['param_syms'] = ([syms['v1'], syms['omega1']]
                          + list(syms['l_syms']) + list(syms['d_syms']))
    return syms


def build_equations(syms, n):
    """Kinematic equations of the tractor + n-trailer chain.

    Tractor: unicycle kinematics driven by (v1, omega1).
    Trailer i: yaw rate from the velocity component transmitted through
    the hitch, including the coupling terms of all preceding trailers.

    Returns:
        eqs:   list of sympy.Eq, one per state derivative
        order: derivative symbols in state order
    """
    theta = syms['theta']
    dot_x1, dot_x2 = syms['dot_x1'], syms['dot_x2']
    dot_theta1 = syms['dot_theta1']
    theta_dot = syms['theta_dot']
    v1, omega1 = syms['v1'], syms['omega1']
    l_syms, d_syms = syms['l_syms'], syms['d_syms']

    eqs = [
        sp.Eq(dot_x1, v1 * sp.cos(theta[0])),
        sp.Eq(dot_x2, v1 * sp.sin(theta[0])),
        sp.Eq(dot_theta1, omega1),
    ]
    for i in range(1, n + 1):
        idx = i - 1
        term1 = v1 / l_syms[idx] * sp.sin(theta[0] - theta[i])
        coupling = sum(
            (l_syms[j - 1] + d_syms[j]) / l_syms[idx]
            * theta_dot[j] * sp.cos(theta[j] - theta[i])
            for j in range(1, i)
        )
        term2 = omega1 * d_syms[0] / l_syms[idx] * sp.cos(theta[0] - theta[i])
        eqs.append(sp.Eq(theta_dot[i], term1 - coupling - term2))

    order = [dot_x1, dot_x2, dot_theta1] + [theta_dot[i] for i in range(1, n + 1)]
    return eqs, order


def explicit_func(eqs, xdot_syms, state_syms, param_syms):
    """Convert the implicit system E(x,u) * x_dot = b(x,u) to explicit form.

    Returns lambdified functions:
        F_func(x, u, p): explicit right-hand side f(x, u)
        A_func(x, u, p): df/dx, evaluated at the given point
        B_func(x, u, p): df/du, evaluated at the given point
    """
    # residuals f_i = lhs - rhs
    exprs = [eq.lhs - eq.rhs for eq in eqs]

    # E[i, j] = coefficient of xdot_syms[j] in exprs[i]
    E = Matrix([[expr.coeff(xdot, 1) for xdot in xdot_syms]
                for expr in exprs])

    # remainder b = exprs - E * xdot
    b_vec = Matrix([
        expr - sum(E[i, j] * xdot_syms[j] for j in range(len(xdot_syms)))
        for i, expr in enumerate(exprs)
    ])

    # Jacobians of b w.r.t. states and inputs
    input_syms = param_syms[:2]  # v1, omega1
    A0 = b_vec.jacobian(Matrix(state_syms))
    B0 = b_vec.jacobian(Matrix(input_syms))

    # explicit system (minus sign because expr = lhs - rhs)
    E_inv = E.inv()
    F_impl = -E_inv * b_vec
    A_impl = -E_inv * A0
    B_impl = -E_inv * B0

    # lambdify with duplicate-free argument list
    all_args = state_syms + param_syms
    unique_args = list(dict.fromkeys(all_args))
    F_func = sp.lambdify(unique_args, F_impl, 'numpy')
    A_func = sp.lambdify(unique_args, A_impl, 'numpy')
    B_func = sp.lambdify(unique_args, B_impl, 'numpy')

    return F_func, A_func, B_func


def build_output(syms, n):
    """Output map h(x): position of the last trailer's axle midpoint.

    Returns lambdified functions:
        H_func(x, u, p): output vector [x_n, y_n]
        C_func(x, u, p): dh/dx for the linearized output equation
    """
    x1, x2 = syms['x1'], syms['x2']
    theta = syms['theta']
    l = syms['l_syms']
    d = syms['d_syms']
    state_syms = syms['state_syms']
    param_syms = syms['param_syms']

    # --- walk back along the chain from the tractor to the last trailer:
    #     p_i = p_{i-1} - d_i*e(theta_i) - l_{i+1}*e(theta_{i+1})
    # Note: an earlier version paired the last axle length with
    # theta[n-1] instead of theta[n] (off-by-one). That cancels for an
    # aligned chain but misplaces the last trailer once joint angles
    # build up (~4 cm at 0.2 rad relative angle with l = 0.19 m).
    f_x = x1 - l[n - 1] * sp.cos(theta[n]) - d[0] * sp.cos(theta[0])
    f_y = x2 - l[n - 1] * sp.sin(theta[n]) - d[0] * sp.sin(theta[0])
    for i in range(2, n + 1):
        f_x -= (l[i - 2] + d[i - 1]) * sp.cos(theta[i - 1])
        f_y -= (l[i - 2] + d[i - 1]) * sp.sin(theta[i - 1])

    h_vec = Matrix([f_x, f_y])
    C_mat = h_vec.jacobian(Matrix(state_syms))

    all_args = state_syms + param_syms
    unique_args = list(dict.fromkeys(all_args))
    H_func = sp.lambdify(unique_args, h_vec, modules='numpy')
    C_func = sp.lambdify(unique_args, C_mat, modules='numpy')

    return H_func, C_func


def chain_positions(x, l, d):
    """Numeric axle and hitch positions of the whole chain.

    Plain-numpy counterpart of the symbolic output map: documents the
    drawing convention and is shared by the visualizer and the tests.

    Returns:
        axles:   [tractor, trailer1, ..., trailer_n] as (2,) arrays
        hitches: one (2,) array per trailer
    """
    import numpy as np
    x = np.asarray(x, dtype=float).flatten()
    n = len(x) - 3
    thetas = x[2:]
    axles = [x[:2].copy()]
    hitches = []
    for i in range(n):
        e_prev = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
        e_self = np.array([np.cos(thetas[i + 1]), np.sin(thetas[i + 1])])
        hitch = axles[-1] - d[i] * e_prev
        axles.append(hitch - l[i] * e_self)
        hitches.append(hitch)
    return axles, hitches
