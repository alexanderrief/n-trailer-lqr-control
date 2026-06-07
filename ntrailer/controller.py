# -*- coding: utf-8 -*-
"""PyMoskito controller for backward driving along reversed references.

BackwardTVLQRController -- time-varying LQR along a reversed spline/circle

Vehicle geometry and trajectory come from scenario.yaml (via ntrailer.scenario).
The .sreg regime file only selects the Controller type.
"""
from collections import OrderedDict

import numpy as np
import pymoskito as pm

from . import scenario
from .lqr import TimeVaryingLQR


class BackwardTVLQRController(pm.Controller):
    """Time-varying LQR on a reversed reference from scenario.yaml.

    Q and R are auto-computed: Q = diag(10, 10, 1, 1, ...), R = I_2.
    """
    public_settings = OrderedDict([
        ('tick divider', 1),
    ])

    def __init__(self, settings):
        settings.update(input_order=0)
        settings.update(input_type='Model_State')
        settings.update(output_info={
            0: {'Name': 'v1', 'Unit': 'm/s'},
            1: {'Name': 'omega1', 'Unit': 'rad/s'},
        })
        super().__init__(settings)

        n = scenario.n_trailers()
        l = scenario.axle_distances()
        d = scenario.hitch_lengths()
        ref = scenario.get_backward_reference()

        n_states = 2 + n + 1
        q = [10.0, 10.0] + [1.0] * (n_states - 2)
        r = [1.0, 1.0]

        self._tvlqr = TimeVaryingLQR(n, l, d, ref, Q=q, R=r)

    def _control(self, time, trajectory_values=None, feedforward_values=None,
                 input_values=None, **kwargs):
        x = np.asarray(input_values, dtype=float).flatten()
        return self._tvlqr.control(time, x)


pm.register_simulation_module(pm.Controller, BackwardTVLQRController)
