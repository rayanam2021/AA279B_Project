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
from .roe import (
    ClosedFormUnavailable,
    RoeReconfigPlan,
    cw_to_roe,
    cw_to_roe_matrix,
    reachable_dv_min,
    roe_control_matrix,
    roe_impulsive_reconfig,
    roe_stm,
    roe_to_cw,
    roe_to_cw_matrix,
)
from .satellites import Satellite, along_track_formation
from .simulation import Simulation, SimulationResult
from .visualization import FormationViewer

try:  # SCP planner requires cvxpy; keep the core package usable without it.
    from .scp import (
        RecoveryPlan,
        ScpConfig,
        apply_plan,
        baseline_roe_plan,
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
    "ClosedFormUnavailable",
    "RoeReconfigPlan",
    "cw_to_roe",
    "cw_to_roe_matrix",
    "reachable_dv_min",
    "roe_control_matrix",
    "roe_impulsive_reconfig",
    "roe_stm",
    "roe_to_cw",
    "roe_to_cw_matrix",
    "Satellite",
    "along_track_formation",
    "Simulation",
    "SimulationResult",
    "FormationViewer",
    "RecoveryPlan",
    "ScpConfig",
    "apply_plan",
    "baseline_roe_plan",
    "baseline_two_impulse_plan",
    "plan_recovery",
]
