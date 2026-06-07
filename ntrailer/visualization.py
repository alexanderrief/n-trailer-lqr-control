# -*- coding: utf-8 -*-
"""PyMoskito visualizer for the n-trailer chain.

Top-down view with the tractor as a green circle, trailers as rounded
grey boxes (FancyBboxPatch + Affine2D rotation), reference trajectory,
and tractor/last-axle path traces.

The number of trailers is detected from the state vector (n = len(x) - 3).
Geometry (d_i, l_i) is taken from scenario.yaml.
"""
import matplotlib as mpl
import numpy as np
import pymoskito as pm

from .model_builder import chain_positions


class NTrailerVisualizer(pm.MplVisualizer):
    """Top-down view of the tractor + n-trailer chain."""

    DIA = 0.10          # tractor circle diameter
    WHEEL_LEN = 0.025   # half-length of a wheel line
    JOINT_R = 0.01      # hitch joint radius
    TRAILER_SIZE = 0.5  # trailer box = TRAILER_SIZE * DIA

    def __init__(self, q_widget, q_layout):
        pm.MplVisualizer.__init__(self, q_widget, q_layout)
        self.axes.set_aspect('equal')
        self.axes.set_xlim(-0.6, 2.6)
        self.axes.set_ylim(-0.6, 1.6)
        self.axes.grid(True, alpha=0.3)
        self.axes.set_xlabel(r'$\xi_1$ in m')
        self.axes.set_ylabel(r'$\xi_2$ in m')
        self.fig.subplots_adjust(left=0.14, bottom=0.14)

        self.n = None
        self.l = None
        self.d = None
        self._artists = None
        self._ref_line = None
        self._tractor_trace = None
        self._axle_trace = None
        self._tractor_pts = []
        self._axle_pts = []

    def update_theme(self):
        pm.MplVisualizer.update_theme(self)
        self.axes.set_xlabel(r'$\xi_1$ in m')
        self.axes.set_ylabel(r'$\xi_2$ in m')
        self.fig.subplots_adjust(left=0.14, bottom=0.14)

    # -- reference curve -------------------------------------------------------

    def _draw_reference(self):
        if self._ref_line is not None:
            self._ref_line.remove()
            self._ref_line = None
        try:
            from . import scenario
            ref = scenario.get_backward_reference()
            self._ref_line, = self.axes.plot(
                ref.x[:, 0], ref.x[:, 1],
                color='#5599dd', linewidth=1.2, linestyle='--',
                alpha=0.6, zorder=0, label='Reference')
        except Exception:
            pass

    # -- initial state before play ---------------------------------------------

    def update_config(self, config):
        super().update_config(config)
        try:
            from . import scenario
            x0 = scenario.get_initial_state()
            if x0 is not None:
                self.update_scene(x0, [0.0, 0.0])
        except Exception:
            pass

    # -- geometry --------------------------------------------------------------

    def _init_geometry(self, n):
        from . import scenario
        self.n = n
        self.d = list(scenario.hitch_lengths())[:n]
        self.l = list(scenario.axle_distances())[:n]
        dd = self.d[-1] if self.d else 0.08
        dl = self.l[-1] if self.l else 0.19
        while len(self.d) < n:
            self.d.append(dd)
        while len(self.l) < n:
            self.l.append(dl)

    # -- wheel helper ----------------------------------------------------------

    @staticmethod
    def _wheel_endpoints(center, theta, radius, wheel_len):
        ct, st = np.cos(theta), np.sin(theta)
        x, y = center
        x1 = x + st * radius - ct * wheel_len
        y1 = y - ct * radius - st * wheel_len
        x2 = x + st * radius + ct * wheel_len
        y2 = y - ct * radius + st * wheel_len
        return x1, y1, x2, y2

    # -- build / update --------------------------------------------------------

    def _build(self, axles, hitches, thetas):
        """Create all matplotlib artists from scratch."""
        self._draw_reference()

        dia = self.DIA
        r = dia / 2.0
        wl = self.WHEEL_LEN
        tsz = self.TRAILER_SIZE * dia
        half = tsz / 2.0
        artists = {}

        # --- tractor ---
        p = axles[0]
        t = thetas[0]
        artists['sphere'] = mpl.patches.Circle(
            p, r, color='green', zorder=0)
        self.axes.add_patch(artists['sphere'])

        x1a, y1a, x2a, y2a = self._wheel_endpoints(p, t, r, wl)
        artists['tractor_w1'] = self.axes.add_line(
            mpl.lines.Line2D([x1a, x2a], [y1a, y2a],
                             color='k', linewidth=3.0, zorder=1))
        x1b, y1b, x2b, y2b = self._wheel_endpoints(p, t, -r, wl)
        artists['tractor_w2'] = self.axes.add_line(
            mpl.lines.Line2D([x1b, x2b], [y1b, y2b],
                             color='k', linewidth=3.0, zorder=1))

        # --- rods, joints, trailers ---
        rods = []
        joints = []
        trailer_patches = []
        trailer_wheels = []
        trailer_axle_lines = []

        prev_axle = axles[0]
        for i in range(self.n):
            h = hitches[i]
            a = axles[i + 1]
            ti = thetas[i + 1]
            ct, st_ = np.cos(ti), np.sin(ti)

            # rod from previous axle to hitch
            rod_a = self.axes.add_line(mpl.lines.Line2D(
                [prev_axle[0], h[0]], [prev_axle[1], h[1]],
                color='k', linewidth=2.0, zorder=3))
            rods.append(rod_a)

            # rod from hitch to trailer axle
            rod_b = self.axes.add_line(mpl.lines.Line2D(
                [h[0], a[0]], [h[1], a[1]],
                color='k', linewidth=2.0, zorder=3))
            rods.append(rod_b)

            # joint
            j = mpl.patches.Circle(h, self.JOINT_R, color='k', zorder=4)
            self.axes.add_patch(j)
            joints.append(j)

            # trailer box (FancyBboxPatch rotated with Affine2D)
            box = mpl.patches.FancyBboxPatch(
                (a[0] - half, a[1] - half), tsz, tsz,
                boxstyle="round,pad=0.005",
                facecolor='#aaaaaa', edgecolor='#333333',
                linewidth=1.0, zorder=0)
            tr = mpl.transforms.Affine2D().rotate_around(a[0], a[1], ti)
            box.set_transform(tr + self.axes.transData)
            self.axes.add_patch(box)
            trailer_patches.append(box)

            # trailer wheels
            x1a, y1a, x2a, y2a = self._wheel_endpoints(a, ti, r, wl)
            tw1 = self.axes.add_line(mpl.lines.Line2D(
                [x1a, x2a], [y1a, y2a],
                color='k', linewidth=3.0, zorder=1))
            x1b, y1b, x2b, y2b = self._wheel_endpoints(a, ti, -r, wl)
            tw2 = self.axes.add_line(mpl.lines.Line2D(
                [x1b, x2b], [y1b, y2b],
                color='k', linewidth=3.0, zorder=1))
            trailer_wheels.append((tw1, tw2))

            # axle stubs (short lines from box edge to wheel)
            frac = 3.0 / 8.0
            ax1 = self.axes.add_line(mpl.lines.Line2D(
                [a[0] + frac * dia * st_, a[0] + 0.5 * dia * st_],
                [a[1] - frac * dia * ct, a[1] - 0.5 * dia * ct],
                color='k', linewidth=2.0, zorder=1))
            ax2 = self.axes.add_line(mpl.lines.Line2D(
                [a[0] - frac * dia * st_, a[0] - 0.5 * dia * st_],
                [a[1] + frac * dia * ct, a[1] + 0.5 * dia * ct],
                color='k', linewidth=2.0, zorder=1))
            trailer_axle_lines.append((ax1, ax2))

            prev_axle = a

        artists['rods'] = rods
        artists['joints'] = joints
        artists['trailer_patches'] = trailer_patches
        artists['trailer_wheels'] = trailer_wheels
        artists['trailer_axle_lines'] = trailer_axle_lines

        # traces
        self._tractor_trace, = self.axes.plot(
            [], [], color='#cc3333', linewidth=1.0,
            alpha=0.7, zorder=0, label='Tractor')
        self._axle_trace, = self.axes.plot(
            [], [], color='#3366cc', linewidth=1.0,
            alpha=0.7, zorder=0, label='Last trailer')
        self._tractor_pts = []
        self._axle_pts = []

        self.axes.legend(loc='upper left', fontsize='small', framealpha=0.7)
        self._artists = artists

    def _update(self, axles, hitches, thetas):
        """Move existing artists to new positions."""
        a = self._artists
        dia = self.DIA
        r = dia / 2.0
        wl = self.WHEEL_LEN
        tsz = self.TRAILER_SIZE * dia
        half = tsz / 2.0

        # tractor
        p = axles[0]
        t = thetas[0]
        a['sphere'].center = tuple(p)
        x1a, y1a, x2a, y2a = self._wheel_endpoints(p, t, r, wl)
        a['tractor_w1'].set_data([x1a, x2a], [y1a, y2a])
        x1b, y1b, x2b, y2b = self._wheel_endpoints(p, t, -r, wl)
        a['tractor_w2'].set_data([x1b, x2b], [y1b, y2b])

        # trailers
        rod_idx = 0
        prev_axle = axles[0]
        for i in range(self.n):
            h = hitches[i]
            ax = axles[i + 1]
            ti = thetas[i + 1]
            ct, st_ = np.cos(ti), np.sin(ti)

            a['rods'][rod_idx].set_data(
                [prev_axle[0], h[0]], [prev_axle[1], h[1]])
            a['rods'][rod_idx + 1].set_data(
                [h[0], ax[0]], [h[1], ax[1]])
            rod_idx += 2

            a['joints'][i].center = tuple(h)

            # recreate trailer box (transform can't be updated in-place)
            old = a['trailer_patches'][i]
            old.remove()
            box = mpl.patches.FancyBboxPatch(
                (ax[0] - half, ax[1] - half), tsz, tsz,
                boxstyle="round,pad=0.005",
                facecolor='#aaaaaa', edgecolor='#333333',
                linewidth=1.0, zorder=0)
            tr = mpl.transforms.Affine2D().rotate_around(ax[0], ax[1], ti)
            box.set_transform(tr + self.axes.transData)
            self.axes.add_patch(box)
            a['trailer_patches'][i] = box

            # wheels
            x1a, y1a, x2a, y2a = self._wheel_endpoints(ax, ti, r, wl)
            a['trailer_wheels'][i][0].set_data([x1a, x2a], [y1a, y2a])
            x1b, y1b, x2b, y2b = self._wheel_endpoints(ax, ti, -r, wl)
            a['trailer_wheels'][i][1].set_data([x1b, x2b], [y1b, y2b])

            # axle stubs
            frac = 3.0 / 8.0
            a['trailer_axle_lines'][i][0].set_data(
                [ax[0] + frac * dia * st_, ax[0] + 0.5 * dia * st_],
                [ax[1] - frac * dia * ct, ax[1] - 0.5 * dia * ct])
            a['trailer_axle_lines'][i][1].set_data(
                [ax[0] - frac * dia * st_, ax[0] - 0.5 * dia * st_],
                [ax[1] + frac * dia * ct, ax[1] + 0.5 * dia * ct])

            prev_axle = ax

    # -- clear -----------------------------------------------------------------

    def _clear_artists(self):
        if self._artists is None:
            return
        a = self._artists
        a['sphere'].remove()
        a['tractor_w1'].remove()
        a['tractor_w2'].remove()
        for r in a['rods']:
            r.remove()
        for j in a['joints']:
            j.remove()
        for p in a['trailer_patches']:
            p.remove()
        for tw1, tw2 in a['trailer_wheels']:
            tw1.remove()
            tw2.remove()
        for ax1, ax2 in a['trailer_axle_lines']:
            ax1.remove()
            ax2.remove()
        if self._tractor_trace is not None:
            self._tractor_trace.remove()
        if self._axle_trace is not None:
            self._axle_trace.remove()
        self._artists = None
        self._tractor_trace = None
        self._axle_trace = None

    # -- expand limits ---------------------------------------------------------

    def _expand_limits(self, points):
        pts = np.asarray(points)
        x_lo, x_hi = self.axes.get_xlim()
        y_lo, y_hi = self.axes.get_ylim()
        margin = 0.3
        if (pts[:, 0].min() < x_lo or pts[:, 0].max() > x_hi
                or pts[:, 1].min() < y_lo or pts[:, 1].max() > y_hi):
            self.axes.set_xlim(min(x_lo, pts[:, 0].min() - margin),
                               max(x_hi, pts[:, 0].max() + margin))
            self.axes.set_ylim(min(y_lo, pts[:, 1].min() - margin),
                               max(y_hi, pts[:, 1].max() + margin))

    # -- main entry ------------------------------------------------------------

    def update_scene(self, x, _u):
        x = np.asarray(x, dtype=float).flatten()
        n = len(x) - 3
        if n < 0:
            return

        thetas = x[2:]
        if self.n != n:
            self._clear_artists()
            self._init_geometry(n)
            axles, hitches = chain_positions(x, self.l, self.d)
            self._build(axles, hitches, thetas)
        else:
            axles, hitches = chain_positions(x, self.l, self.d)
            self._update(axles, hitches, thetas)

        # traces
        self._tractor_pts.append(axles[0].copy())
        tp = np.asarray(self._tractor_pts)
        self._tractor_trace.set_data(tp[:, 0], tp[:, 1])

        self._axle_pts.append(axles[-1].copy())
        ap = np.asarray(self._axle_pts)
        self._axle_trace.set_data(ap[:, 0], ap[:, 1])

        self._expand_limits(axles + hitches)
        self.canvas.draw()


pm.register_visualizer(NTrailerVisualizer)
