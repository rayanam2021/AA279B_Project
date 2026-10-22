"""Chief reference orbit: circular two-body orbit propagated analytically in ECI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH, R_EARTH


def _rot1(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot3(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class CircularOrbit:
    """Circular chief orbit defined by altitude and orientation angles [rad].

    Attributes
    ----------
    altitude : orbit altitude above R_EARTH [m]
    inclination : orbit inclination [rad]
    raan : right ascension of ascending node [rad]
    arg_lat0 : argument of latitude at t = 0 [rad]
    """

    altitude: float = 550e3
    inclination: float = 0.0
    raan: float = 0.0
    arg_lat0: float = 0.0

    @property
    def radius(self) -> float:
        return R_EARTH + self.altitude

    @property
    def mean_motion(self) -> float:
        return np.sqrt(MU_EARTH / self.radius**3)

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.mean_motion

    def state_eci(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Chief ECI position [m] and velocity [m/s] at time t [s]."""
        a = self.radius
        n = self.mean_motion
        u = self.arg_lat0 + n * t  # argument of latitude

        # Orbit-plane -> ECI rotation: R3(raan) @ R1(inc)
        dcm = _rot3(self.raan) @ _rot1(self.inclination)
        r_plane = a * np.array([np.cos(u), np.sin(u), 0.0])
        v_plane = a * n * np.array([-np.sin(u), np.cos(u), 0.0])
        return dcm @ r_plane, dcm @ v_plane

    def states_eci(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized chief states: returns (N, 3) positions and velocities."""
        r = np.empty((len(times), 3))
        v = np.empty((len(times), 3))
        for i, t in enumerate(times):
            r[i], v[i] = self.state_eci(float(t))
        return r, v
