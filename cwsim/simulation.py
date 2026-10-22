"""Simulation driver: propagates a formation in LVLH and maps it into ECI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import propagate_cw_with_impulses
from .frames import relative_to_eci
from .orbits import CircularOrbit
from .satellites import Satellite


@dataclass
class SimulationResult:
    """Time histories for every satellite in both frames.

    Attributes
    ----------
    times : (N,) sample times [s]
    chief_eci : (N, 3) chief ECI positions [m]
    lvlh : dict name -> (N, 6) LVLH states
    eci : dict name -> (N, 3) ECI positions [m]
    """

    times: np.ndarray
    chief_eci: np.ndarray
    lvlh: dict[str, np.ndarray]
    eci: dict[str, np.ndarray]
    satellites: list[Satellite]
    chief_orbit: CircularOrbit


@dataclass
class Simulation:
    """CW formation simulation about a circular chief orbit."""

    chief_orbit: CircularOrbit = field(default_factory=CircularOrbit)
    satellites: list[Satellite] = field(default_factory=list)

    def add_satellite(self, sat: Satellite) -> None:
        self.satellites.append(sat)

    def run(self, t_final: float, dt: float = 10.0) -> SimulationResult:
        """Propagate all satellites from t = 0 to t_final with step dt [s]."""
        times = np.arange(0.0, t_final + dt / 2, dt)
        n = self.chief_orbit.mean_motion

        chief_r, chief_v = self.chief_orbit.states_eci(times)

        lvlh: dict[str, np.ndarray] = {}
        eci: dict[str, np.ndarray] = {}
        for sat in self.satellites:
            states = propagate_cw_with_impulses(sat.x0, n, times, sat.impulses)
            lvlh[sat.name] = states

            r_eci = np.empty((len(times), 3))
            for k in range(len(times)):
                r_eci[k], _ = relative_to_eci(states[k], chief_r[k], chief_v[k], n)
            eci[sat.name] = r_eci

        return SimulationResult(
            times=times,
            chief_eci=chief_r,
            lvlh=lvlh,
            eci=eci,
            satellites=list(self.satellites),
            chief_orbit=self.chief_orbit,
        )
