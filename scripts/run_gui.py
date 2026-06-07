#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyMoskito GUI entry point.

Importing ntrailer.model, ntrailer.controller and ntrailer.visualization
registers models, controllers and the animation visualizer with PyMoskito;
the regime file then references models and controllers by class name.
The visualizer is selected from the toolbar combo box (not via .sreg).

Usage:
    python scripts/run_gui.py
"""
import os
import sys

from PyQt5.QtWidgets import QApplication

import pymoskito as pm
from pymoskito.simulation_interface import PropertyItem, SimulatorInteractor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

# PyMoskito lists every simulation slot in the property tree by default.
# This project only needs model, solver and controller.
_VISIBLE_MODULES = ("Model", "Solver", "Controller")


def _setup_compact_property_tree(self):
    for sim_module in _VISIBLE_MODULES:
        self.target_model.appendRow([
            PropertyItem(sim_module),
            PropertyItem(None),
        ])
    for row in range(self.target_model.rowCount()):
        self._add_settings(self.target_model.index(row, 0))


SimulatorInteractor._setup_model_items = _setup_compact_property_tree

import ntrailer.model       # noqa: F401, E402  (registers models)
import ntrailer.controller  # noqa: F401, E402  (registers controllers)
import ntrailer.visualization  # noqa: F401, E402  (registers visualizer)

REGIME_FILE = os.path.join(os.path.dirname(__file__), os.pardir,
                           'regimes', 'default.sreg')


if __name__ == '__main__':
    app = QApplication([])
    sim = pm.SimulationGui()
    sim.load_regimes_from_file(os.path.abspath(REGIME_FILE))
    sim.show()
    app.exec_()
