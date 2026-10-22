"""Dual-view PyVista visualization: ECEF frame (textured Earth) and the
chief-centered CW/RIC relative frame.

Left view (ECEF): the Earth is fixed and satellite positions are rotated by
the sidereal angle, so trajectories trace the familiar regressing ground-path
spiral. Because real formation separations (km) are invisible at planetary
scale, deputy offsets from the chief are exaggerated by ``separation_scale``
(annotated on screen) so the individual satellites can be told apart.

Right view (CW/RIC): x radial, y in-track, z cross-track, chief at the
origin, with a grid symmetric about the origin.

Usage:
    viewer = FormationViewer(result)
    viewer.show()          # interactive window: time slider + spacebar play/pause
    viewer.save_gif(path)  # off-screen animation export
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from .constants import OMEGA_EARTH, R_EARTH
from .simulation import SimulationResult


def _load_earth(radius: float) -> tuple[pv.PolyData, pv.Texture | None]:
    """Return an Earth sphere mesh and (if available) a realistic texture.

    Tries the pyvista example textures (downloads on first use); falls back to
    an untextured blue sphere if offline.
    """
    try:
        from pyvista import examples

        try:
            mesh = examples.planets.load_planet(radius=radius)
        except AttributeError:  # pyvista < 0.49
            mesh = examples.planets.load_earth(radius=radius)
        texture = examples.load_globe_texture()
        return mesh, texture
    except Exception:
        return pv.Sphere(
            radius=radius, theta_resolution=90, phi_resolution=90
        ), None


def _polyline(points: np.ndarray) -> pv.PolyData:
    """Build a connected polyline through an (N, 3) array of points."""
    poly = pv.PolyData(points)
    n = len(points)
    cells = np.empty(n + 1, dtype=np.int64)
    cells[0] = n
    cells[1:] = np.arange(n)
    poly.lines = cells
    return poly


def _rot_z(angle: float) -> np.ndarray:
    """Rotation of a vector by -angle about z (ECI -> ECEF for angle = wt)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


class FormationViewer:
    """Side-by-side interactive views of a formation in ECEF and CW/RIC."""

    def __init__(
        self,
        result: SimulationResult,
        separation_scale: float = 500.0,
        marker_radius_ecef: float | None = None,
        marker_radius_cw: float | None = None,
        window_size: tuple[int, int] = (1800, 900),
        off_screen: bool = False,
    ):
        self.result = result
        self.frame = 0
        self.playing = False
        self.sep_scale = separation_scale

        res = result
        orbit_r = res.chief_orbit.radius

        self.marker_r_ecef = marker_radius_ecef or 0.012 * orbit_r
        cw_extent = max(
            np.abs(states[:, :3]).max() for states in res.lvlh.values()
        )
        self.marker_r_cw = marker_radius_cw or max(0.025 * cw_extent, 1.0)

        # Precompute display positions. ECEF: rotate by the sidereal angle;
        # deputy offsets from the chief are exaggerated for visibility.
        thetas = OMEGA_EARTH * res.times
        self._ecef_pos: dict[str, np.ndarray] = {}
        chief = res.chief_eci
        rots = np.array([_rot_z(th) for th in thetas])  # (N, 3, 3)
        self._chief_ecef = np.einsum("nij,nj->ni", rots, chief)
        for name, traj in res.eci.items():
            disp = chief + self.sep_scale * (traj - chief)
            self._ecef_pos[name] = np.einsum("nij,nj->ni", rots, disp)

        self.plotter = pv.Plotter(
            shape=(1, 2), window_size=window_size, off_screen=off_screen
        )
        self._markers_ecef: dict[str, tuple[pv.PolyData, np.ndarray]] = {}
        self._markers_cw: dict[str, tuple[pv.PolyData, np.ndarray]] = {}
        self._labels_ecef: dict[str, object] = {}
        self._labels_cw: dict[str, object] = {}
        self._build_ecef_view()
        self._build_cw_view()
        self._set_frame(0)

    # ------------------------------------------------------------------ build
    def _add_label(self, store: dict, name: str, pos, color) -> None:
        """Attach a movable 3D text label (pv.Label when available)."""
        try:
            label = pv.Label(name, position=np.asarray(pos, dtype=float),
                             size=16)
            label.prop.color = color
            label.prop.bold = True
            self.plotter.add_actor(label)
            store[name] = label
        except AttributeError:  # older pyvista: static label at start point
            self.plotter.add_point_labels(
                [np.asarray(pos, dtype=float)], [name], text_color=color,
                font_size=14, shape=None, show_points=False,
                always_visible=True,
            )
            store[name] = None

    def _build_ecef_view(self) -> None:
        p = self.plotter
        res = self.result
        p.subplot(0, 0)
        p.set_background("black")
        p.add_text("ECEF Frame", font_size=12, color="white")
        if self.sep_scale != 1.0:
            p.add_text(
                f"formation offsets x{self.sep_scale:g} for visibility",
                position="lower_left", font_size=8, color="gray",
            )

        earth, texture = _load_earth(R_EARTH)
        p.add_mesh(earth, texture=texture, smooth_shading=True)

        # Chief path over the simulated window (a regressing spiral in ECEF).
        p.add_mesh(
            _polyline(self._chief_ecef), color="gray", line_width=1,
            opacity=0.5,
        )
        for sat in res.satellites:
            traj = self._ecef_pos[sat.name]
            p.add_mesh(
                _polyline(traj), color=sat.color, line_width=2, opacity=0.6
            )
            marker = pv.Sphere(radius=self.marker_r_ecef)
            self._markers_ecef[sat.name] = (marker, marker.points.copy())
            p.add_mesh(marker, color=sat.color, smooth_shading=True)
            self._add_label(
                self._labels_ecef, sat.name,
                traj[0] + np.array([0.0, 0.0, 2.5 * self.marker_r_ecef]),
                sat.color,
            )

        p.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
        p.camera_position = "iso"

    def _build_cw_view(self) -> None:
        p = self.plotter
        res = self.result
        p.subplot(0, 1)
        p.set_background("#101018")
        p.add_text(
            "CW / RIC Frame (chief at origin)", font_size=12, color="white"
        )

        # Chief location marker at the origin.
        p.add_mesh(
            pv.Sphere(radius=0.6 * self.marker_r_cw),
            color="white",
            opacity=0.9,
        )

        for sat in res.satellites:
            traj = res.lvlh[sat.name][:, :3]
            p.add_mesh(
                _polyline(traj), color=sat.color, line_width=2, opacity=0.6
            )
            marker = pv.Sphere(radius=self.marker_r_cw)
            self._markers_cw[sat.name] = (marker, marker.points.copy())
            p.add_mesh(marker, color=sat.color, smooth_shading=True)
            self._add_label(
                self._labels_cw, sat.name,
                traj[0] + np.array([0.0, 0.0, 2.5 * self.marker_r_cw]),
                sat.color,
            )

        # Grid symmetric about the chief origin, with uncluttered ticks.
        bounds = []
        for axis in range(3):
            ext = max(
                abs(res.lvlh[s.name][:, axis]).max() for s in res.satellites
            )
            ext = 1.1 * max(ext, 100.0)
            bounds += [-ext, ext]
        p.show_grid(
            bounds=bounds,
            xtitle="Radial x [m]",
            ytitle="In-track y [m]",
            ztitle="Cross-track z [m]",
            n_xlabels=3,
            n_ylabels=3,
            n_zlabels=3,
            fmt="%.0f",
            color="gray",
        )
        p.add_axes(xlabel="R", ylabel="I", zlabel="C")
        p.camera_position = "iso"

        self._time_text = p.add_text(
            "", position="lower_right", font_size=10, color="white"
        )

    # ----------------------------------------------------------------- update
    def _set_frame(self, frame: int) -> None:
        res = self.result
        frame = int(np.clip(frame, 0, len(res.times) - 1))
        self.frame = frame
        t = res.times[frame]

        for sat in res.satellites:
            pos_ecef = self._ecef_pos[sat.name][frame]
            marker, base = self._markers_ecef[sat.name]
            marker.points = base + pos_ecef
            if self._labels_ecef[sat.name] is not None:
                self._labels_ecef[sat.name].position = pos_ecef + np.array(
                    [0.0, 0.0, 2.5 * self.marker_r_ecef]
                )

            pos_cw = res.lvlh[sat.name][frame, :3]
            marker, base = self._markers_cw[sat.name]
            marker.points = base + pos_cw
            if self._labels_cw[sat.name] is not None:
                self._labels_cw[sat.name].position = pos_cw + np.array(
                    [0.0, 0.0, 2.5 * self.marker_r_cw]
                )

        self._time_text.set_text(
            "lower_right",
            f"t = {t / 60.0:7.1f} min  ({t / res.chief_orbit.period:.2f} orbits)",
        )

    def _on_slider(self, value: float) -> None:
        self._set_frame(round(value))

    def _on_timer(self, *_args) -> None:
        if not self.playing:
            return
        nxt = self.frame + 1
        if nxt >= len(self.result.times):
            nxt = 0
        self._slider.GetRepresentation().SetValue(nxt)
        self._set_frame(nxt)
        self.plotter.render()

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    # ------------------------------------------------------------------- show
    def show(self) -> None:
        """Open the interactive window (slider scrubs time, space plays/pauses)."""
        p = self.plotter
        p.subplot(0, 1)
        self._slider = p.add_slider_widget(
            self._on_slider,
            rng=[0, len(self.result.times) - 1],
            value=0,
            title="Time step",
            fmt="%.0f",
            style="modern",
            pointa=(0.25, 0.08),
            pointb=(0.95, 0.08),
        )
        p.add_key_event("space", self._toggle_play)
        p.add_text(
            "space: play/pause",
            position="upper_right",
            font_size=8,
            color="gray",
        )
        p.add_timer_event(max_steps=10_000_000, duration=30, callback=self._on_timer)
        p.show()

    def save_gif(self, filename: str, step: int = 5, fps: int = 20) -> None:
        """Render the animation off-screen to a GIF."""
        p = self.plotter
        p.open_gif(filename, fps=fps)
        for frame in range(0, len(self.result.times), step):
            self._set_frame(frame)
            p.write_frame()
        p.close()

    def screenshot(self, filename: str, frame: int | None = None) -> None:
        """Save a single rendered frame to an image file."""
        if frame is not None:
            self._set_frame(frame)
        self.plotter.screenshot(filename)
