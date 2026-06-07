# -*- coding: utf-8 -*-
"""Scenario loader: single source of truth for vehicle geometry,
trajectory shape, and initial conditions.

Reads scenario.yaml from the project root once and caches the parsed
config and the computed backward reference.  Both Model and Controller
import from here so they always see the same data.
"""
import os
from copy import deepcopy

import numpy as np
import yaml

from .spline_trajectory import build_backward_reference, reference_initial_state

_SCENARIO_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, 'scenario.yaml')

_config_cache = None
_ref_cache = {}


def _load_config():
    global _config_cache
    if _config_cache is None:
        with open(_SCENARIO_PATH) as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def reload():
    """Force re-read of scenario.yaml and clear all caches."""
    global _config_cache
    _config_cache = None
    _ref_cache.clear()


def _extend_list(lst, n, default=None):
    """Extend *lst* to length *n* by repeating the last element."""
    if default is None:
        default = lst[-1] if lst else 0.0
    while len(lst) < n:
        lst.append(default)
    return lst[:n]


# --- Geometry accessors -----------------------------------------------------

def n_trailers():
    return int(_load_config().get('n_trailers', 2))


def hitch_lengths():
    """d_1 ... d_n as a numpy array, auto-extended to n_trailers."""
    cfg = _load_config()
    n = n_trailers()
    raw = list(cfg.get('hitch_lengths_m', [0.08]))
    return np.array(_extend_list(raw, n), dtype=float)


def axle_distances():
    """l_2 ... l_{n+1} as a numpy array, auto-extended to n_trailers."""
    cfg = _load_config()
    n = n_trailers()
    raw = list(cfg.get('axle_distances_m', [0.19]))
    return np.array(_extend_list(raw, n), dtype=float)


def deflection_deg():
    return float(_load_config().get('deflection_deg', 0.0))


def trajectory_type():
    return str(_load_config()['trajectory']['type'])


def trajectory_duration():
    """Trajectory duration T [s] from scenario.yaml."""
    return float(_load_config()['trajectory'].get('T', 20.0))


# --- Reference --------------------------------------------------------------

def _traj_cache_key(traj):
    """Hashable key fragment for trajectory settings (nested lists -> tuples)."""
    def freeze(value):
        if isinstance(value, list):
            return tuple(freeze(v) for v in value)
        return value
    return tuple(sorted((k, freeze(v)) for k, v in traj.items()))


def get_backward_reference():
    """Build (or return cached) backward reference for the scenario geometry."""
    n = n_trailers()
    l = axle_distances()
    d = hitch_lengths()
    traj = deepcopy(_load_config()['trajectory'])
    traj_type = traj.pop('type')
    key = (n, tuple(l), tuple(d), traj_type, _traj_cache_key(traj))
    if key not in _ref_cache:
        _ref_cache[key] = build_backward_reference(traj_type, n, l, d, **traj)
    return _ref_cache[key]


def get_initial_state():
    """Initial state on the backward reference with the configured deflection."""
    ref = get_backward_reference()
    return reference_initial_state(ref, deflection_deg())
