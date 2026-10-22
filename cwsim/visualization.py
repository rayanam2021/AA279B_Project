"""Dual-view PyVista visualization: ECI frame (textured Earth) and CW/LVLH frame.

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


class FormationViewer:
    """Side-by-side interactive views of a formation in ECI and CW frames."""

    def __init__(
        self,
        result: SimulationResult,
        marker_radius_eci: float | None = None,
        marker_radius_cw: float | None = None,
        window_size: tuple[int, int] = (1800, 900),
        off_screen: bool = False,
    ):
        self.result = result
        self.frame = 0
        self.playing = False

        res = result
        orbit_r = res.chief_orbit.radius

        # Marker sizes: satellites are km-scale apart on a Mm-scale orbit, so
        # the ECI markers are exaggerated for visibility.
        self.marker_r_eci = marker_radius_eci or 0.018 * orbit_r
        cw_extent = max(
            np.abs(states[:, :3]).max() for states in res.lvlh.values()
        )
        self.marker_r_cw = marker_radius_cw or max(0.025 * cw_extent, 1.0)

        self.plotter = pv.Plotter(
            shape=(1, 2), window_size=window_size, off_screen=off_screen
        )
        self._markers_eci: dict[str, tuple[pv.PolyData, np.ndarray]] = {}
        self._markers_cw: dict[str, tuple[pv.PolyData, np.ndarray]] = {}
        self._build_eci_view()
        self._build_cw_view()
        self._set_frame(0)

    # ------------------------------------------------------------------ build
    def _build_eci_view(self) -> None:
        p = self.plotter
        res = self.result
        p.subplot(0, 0)
        p.set_background("black")
        p.add_text("ECI Frame", font_size=12, color="white")

        earth, texture = _load_earth(R_EARTH)
        self._earth_actor = p.add_mesh(earth, texture=texture, smooth_shading=True)

        # Chief reference orbit (one full revolution).
        t_orbit = np.linspace(0.0, res.chief_orbit.period, 361)
        ring, _ = res.chief_orbit.states_eci(t_orbit)
        p.add_mesh(_polyline(ring), color="gray", line_width=1, opacity=0.5)

        for sat in res.satellites:
            traj = res.eci[sat.name]
            p.add_mesh(
                _polyline(traj), color=sat.color, line_width=2, opacity=0.6
            )
            marker = pv.Sphere(radius=self.marker_r_eci)
            self._markers_eci[sat.name] = (marker, marker.points.copy())
            p.add_mesh(marker, color=sat.color, smooth_shading=True)

        p.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
        p.camera_position = "iso"

    def _build_cw_view(self) -> None:
        p = self.plotter
        res = self.result
        p.subplot(0, 1)
        p.set_background("#101018")
        p.add_text("CW / LVLH Frame (chief at origin)", font_size=12, color="white")

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
            p.add_point_labels(
                [traj[0]],
                [sat.name],
                text_color=sat.color,
                font_size=14,
                shape=None,
                show_points=False,
            )

        p.show_grid(
            xtitle="Radial x [m]",
            ytitle="Along-track y [m]",
            ztitle="Cross-track z [m]",
            color="gray",
        )
        p.add_axes(xlabel="x", ylabel="y", zlabel="z")
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
            marker, base = self._markers_eci[sat.name]
            marker.points = base + res.eci[sat.name][frame]
            marker, base = self._markers_cw[sat.name]
            marker.points = base + res.lvlh[sat.name][frame, :3]

        # Spin the Earth about the ECI z-axis at the sidereal rate.
        self._earth_actor.orientation = (0.0, 0.0, np.degrees(OMEGA_EARTH * t))

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
