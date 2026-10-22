"""cwsim: Clohessy-Wiltshire formation-flying simulator with dual-frame 3D views."""

from .constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from .dynamics import (
    cw_stm,
    cw_two_impulse,
    propagate_cw,
    propagate_cw_with_impulses,
)
from .frames import eci_to_relative, lvlh_dcm, relative_to_eci
from .orbits import CircularOrbit
from .satellites import Satellite, along_track_formation
from .simulation import Simulation, SimulationResult
from .visualization import FormationViewer

try:  # SCP planner requires cvxpy; keep the core package usable without it.
    from .scp import (
        RecoveryPlan,
        ScpConfig,
        apply_plan,
        baseline_two_impulse_plan,
        plan_recovery,
    )
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "MU_EARTH",
    "OMEGA_EARTH",
    "R_EARTH",
    "cw_stm",
    "cw_two_impulse",
    "propagate_cw",
    "propagate_cw_with_impulses",
    "eci_to_relative",
    "lvlh_dcm",
    "relative_to_eci",
    "CircularOrbit",
    "Satellite",
    "along_track_formation",
    "Simulation",
    "SimulationResult",
    "FormationViewer",
    "RecoveryPlan",
    "ScpConfig",
    "apply_plan",
    "baseline_two_impulse_plan",
    "plan_recovery",
]
