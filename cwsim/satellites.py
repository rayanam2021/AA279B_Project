"""Satellite and formation containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Satellite:
    """A deputy satellite defined by its LVLH relative state at t = 0.

    Attributes
    ----------
    name : display name
    x0 : initial LVLH state [x, y, z, xdot, ydot, zdot] (m, m/s)
    color : any pyvista/matplotlib color spec for plotting
    impulses : list of (t_burn [s], dv [m/s] 3-vector in LVLH axes)
    """

    name: str
    x0: np.ndarray
    color: str = "white"
    impulses: list[tuple[float, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float).reshape(6)

    def add_impulse(self, t_burn: float, dv: np.ndarray) -> None:
        """Schedule an impulsive maneuver (dv in LVLH axes, m/s)."""
        self.impulses.append((float(t_burn), np.asarray(dv, dtype=float).reshape(3)))
        self.impulses.sort(key=lambda b: b[0])


def along_track_formation(
    spacing: float,
    count: int = 3,
    colors: list[str] | None = None,
) -> list[Satellite]:
    """Equally spaced along-track formation centered on the chief origin.

    A pure along-track offset with zero relative velocity is an equilibrium of
    the CW equations, so the unperturbed formation holds its spacing.
    """
    if colors is None:
        colors = ["#ff5555", "#55ff88", "#5599ff", "#ffcc44", "#cc66ff"]
    offsets = (np.arange(count) - (count - 1) / 2.0) * spacing
    return [
        Satellite(
            name=f"Sat-{i + 1}",
            x0=np.array([0.0, y, 0.0, 0.0, 0.0, 0.0]),
            color=colors[i % len(colors)],
        )
        for i, y in enumerate(offsets)
    ]
