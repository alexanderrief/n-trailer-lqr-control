# -*- coding: utf-8 -*-
"""PyMoskito model of the n-trailer vehicle chain.

Vehicle geometry, trajectory, and initial conditions are read from
scenario.yaml (via ntrailer.scenario).  The .sreg regime file only
selects the Model type.
"""
from collections import OrderedDict

import numpy as np
import pymoskito as pm

from . import scenario
from .model_builder import (define_symbols, build_equations,
                            explicit_func, build_output)


class nonlinearNTrailer(pm.Model):
    """Full kinematic model of a tractor with n trailers.

    The symbolic equations are built once in __init__ and compiled
    with lambdify; state_function() then only evaluates numpy code.
    """
    public_settings = OrderedDict([
        ('initial state', [0]),
    ])

    def __init__(self, settings):
        self.n = scenario.n_trailers()
        self.l = scenario.axle_distances()
        self.d = scenario.hitch_lengths()

        settings['initial state'] = scenario.get_initial_state()
        settings['state_count'] = 2 + self.n + 1
        settings['input_count'] = 2

        settings['output_info'] = self._build_output_info()
        super().__init__(settings)

        syms = define_symbols(self.n)
        eqs, order = build_equations(syms, self.n)
        self.F_func, _, _ = explicit_func(
            eqs, order, syms['state_syms'], syms['param_syms'])
        self.order = order

        if self.n > 0:
            self.H_func, _ = build_output(syms, self.n)

    def _build_output_info(self):
        info = {
            0: {'Name': 'x-position', 'Unit': 'm'},
            1: {'Name': 'y-position', 'Unit': 'm'},
        }
        for i in range(1, self.n + 2):
            info[1 + i] = {'Name': f'theta{i}', 'Unit': 'rad'}
        if self.n > 0:
            info[self.n + 3] = {'Name': 'x-n-position', 'Unit': 'm'}
            info[self.n + 4] = {'Name': 'y-n-position', 'Unit': 'm'}
        return info

    def state_function(self, t, x, args):
        """ODE callback: dx = f(x, u) via the compiled explicit model."""
        y = np.asarray(x, dtype=float).flatten()
        u = np.asarray(args[0], dtype=float).flatten()
        p = np.concatenate((y, u, self.l, self.d))
        dx = np.array(self.F_func(*p), dtype=float).flatten()
        return dx

    def calc_output(self, state_vector):
        """Return [state, x_n, y_n] -- state plus last-trailer position."""
        if self.n == 0:
            return np.asarray(state_vector, dtype=float).flatten()
        y = np.asarray(state_vector, dtype=float).flatten()
        u_dummy = [0.0, 0.0]
        p = np.concatenate((u_dummy, self.l, self.d))
        out = np.array(self.H_func(*y, *p)).flatten()
        return np.concatenate((y, out))

    def root_function(self, x):
        return [False]

    def check_consistency(self, x):
        pass


pm.register_simulation_module(pm.Model, nonlinearNTrailer)
