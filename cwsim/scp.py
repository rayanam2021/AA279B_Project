"""Sequential Convex Programming (SCP) formation recovery.

Plans impulsive burns for ALL satellites to restore an equally spaced
formation by time tf, allowing the satellites to swap slots (any ordering that
reproduces the original separations is acceptable).

Problem formulation (per slot assignment, per candidate tf)
-----------------------------------------------------------
Decision variables: impulses dv_i[k] in R^3 for every satellite i at every
grid node t_k = t_delay + k*dt, k = 0..N (burns are only allowed after the
reaction delay), plus states x_i[k] in R^6.

Objective (minimize):
    J =   sum_i sum_k ||dv_i[k]||_2          total constellation delta-v
        + w_balance * max_i fuel_i           fuel balancing (mission life ends
                                             when the FIRST sat runs dry)
        + w_slack * sum(collision slack)     exact-penalty feasibility slack
        + w_tf * tf                          time-of-flight (applied in the
                                             outer search over tf candidates)

The sum-of-L2-norms term is a group-lasso: it naturally drives most dv_i[k]
to exactly zero, so the solver *chooses* how many burns to use ("burn as many
times as necessary").

Constraints:
    x_i[k+1] = Phi(dt) (x_i[k] + B dv_i[k])      exact CW dynamics (linear)
    x_i[N] + B dv_i[N] = slot_{sigma(i)}         terminal slot (pos + vel)
    ||dv_i[k]||_2 <= dv_max                      thruster limit per burn
    sum_k ||dv_i[k]||_2 <= fuel_max              per-satellite fuel budget
    nhat_ij(t)^T (r_i(t) - r_j(t)) >= d_min - s  keep-out sphere, linearized,
    s >= 0                                       enforced in CONTINUOUS time
                                                 (sub-sampled; slack penalized)

The CW dynamics are exactly linear, so the ONLY non-convexity is the pairwise
keep-out constraint ||r_i - r_j|| >= d_min. Its linearization about the
previous iterate's relative position (the supporting halfspace of the keep-out
sphere) is an *inner* approximation of the true feasible set, so every SCP
iterate that satisfies it is collision-free wherever it is imposed; iterating
re-aims the halfspace normal nhat to recover optimality. No trust region is
required for constraint validity.

Collision avoidance is enforced along the whole trajectory, not only at burn
nodes: between nodes the coasting state is x(t_k + tau) = Phi(tau)(x_k +
B dv_k), which is AFFINE in the decision variables, so the keep-out constraint
is imposed at n_sub sub-sampled times per segment (default every dt/n_sub)
at no change in problem class. An optional collision_margin pads d_min to
cover the residual excursion between sub-samples, and a fine-grained
post-check reports the true minimum separation of the executed plan.

Slot assignment: the 3! (n!) slot permutations are ranked with a cheap
closed-form two-impulse cost heuristic and the best few are solved exactly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .dynamics import cw_stm, cw_two_impulse, propagate_cw_with_impulses
from .orbits import CircularOrbit
from .roe import roe_impulsive_reconfig
from .satellites import Satellite

# Impulse input matrix: dv adds to the velocity half of the state.
_B = np.vstack([np.zeros((3, 3)), np.eye(3)])


@dataclass
class ScpConfig:
    """Tuning knobs for the SCP recovery planner (SI units)."""

    dt: float = 90.0            # burn-grid spacing [s]
    dv_max: float = 0.5         # max impulse magnitude per burn [m/s]
    fuel_max: float = 5.0       # per-satellite delta-v budget [m/s]
    d_min: float = 200.0        # pairwise keep-out radius [m]
    n_sub: int = 6              # keep-out samples per segment (every dt/n_sub)
    collision_margin: float = 0.0  # pad on d_min for inter-sample excursion [m]
    w_balance: float = 0.1      # weight on max per-sat fuel (balancing)
    w_tf: float = 0.0           # weight on tf [m/s per s] for the tf trade
    w_slack: float = 1.0        # exact-penalty weight on collision slack [1/m]
    max_iters: int = 8          # SCP iterations
    tol: float = 1e-3           # convergence: change in total dv [m/s]
    burn_floor: float = 1e-5    # discard solved burns smaller than this [m/s]
    check_dt: float = 10.0      # resolution of the post-solve collision check
    solver: str = "CLARABEL"


@dataclass
class RecoveryPlan:
    """Output of the planner: the burn schedule and performance metrics."""

    burns: dict[str, list[tuple[float, np.ndarray]]]  # name -> [(t, dv)]
    assignment: tuple[int, ...]       # satellite i ends in slot assignment[i]
    total_dv: float                   # sum over constellation [m/s]
    per_sat_dv: dict[str, float]
    tf: float
    t_delay: float
    objective: float                  # total_dv + w_tf*tf (+ slack penalty)
    min_separation: float             # fine-grid min pairwise distance [m]
    slack_max: float                  # residual linearized violation [m]
    iterations: int
    converged: bool
    times: np.ndarray                 # fine time grid for the trajectories
    trajectories: dict[str, np.ndarray]  # name -> (M, 6) LVLH states

    def summary(self) -> str:
        per_sat = ", ".join(
            f"{k}: {v:.4f}" for k, v in self.per_sat_dv.items()
        )
        n_burns = sum(len(b) for b in self.burns.values())
        return (
            f"assignment={self.assignment}, total dv={self.total_dv:.4f} m/s "
            f"({per_sat}), {n_burns} burns, tf={self.tf / 60:.1f} min, "
            f"min sep={self.min_separation:.1f} m, "
            f"converged={self.converged} ({self.iterations} iters)"
        )


def apply_plan(satellites: list[Satellite], plan: RecoveryPlan) -> None:
    """Attach the planned burns to the Satellite objects (for simulation)."""
    by_name = {s.name: s for s in satellites}
    for name, burns in plan.burns.items():
        for t, dv in burns:
            by_name[name].add_impulse(t, dv)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _states_at_delay(
    n: float, satellites: list[Satellite], t_delay: float
) -> list[np.ndarray]:
    """Propagate each satellite (with its existing impulses) to t_delay."""
    out = []
    for sat in satellites:
        x = propagate_cw_with_impulses(
            sat.x0, n, np.array([t_delay]), sat.impulses
        )[0]
        out.append(x)
    return out


def _rank_assignments(
    n: float,
    x_delay: list[np.ndarray],
    slots: list[np.ndarray],
    horizon: float,
    n_keep: int,
) -> list[tuple[int, ...]]:
    """Rank slot permutations by a closed-form two-impulse cost heuristic."""
    n_sat = len(x_delay)
    cost = np.zeros((n_sat, n_sat))
    for i in range(n_sat):
        for s in range(n_sat):
            try:
                dv1, dv2 = cw_two_impulse(x_delay[i], n, horizon, slots[s])
                cost[i, s] = np.linalg.norm(dv1) + np.linalg.norm(dv2)
            except np.linalg.LinAlgError:
                cost[i, s] = np.inf
    perms = list(itertools.permutations(range(n_sat)))
    perms.sort(key=lambda p: sum(cost[i, p[i]] for i in range(n_sat)))
    return perms[:n_keep]


def _solve_fixed_assignment(
    n: float,
    x_delay: list[np.ndarray],
    slots_assigned: list[np.ndarray],
    horizon: float,
    cfg: ScpConfig,
    verbose: bool = False,
) -> tuple[list[np.ndarray], float, float, int, bool] | None:
    """SCP solve for one slot assignment over [0, horizon] after the delay.

    Returns (dv arrays [(3, N+1) per sat], total_dv, slack_max, iters,
    converged), or None if the convex subproblem fails.
    """
    n_sat = len(x_delay)
    n_steps = max(2, int(round(horizon / cfg.dt)))
    dt = horizon / n_steps
    phi = cw_stm(n, dt)
    n_sub = max(1, cfg.n_sub)
    # Position rows of the STM at each sub-sample time tau = (j+1) dt / n_sub:
    # r(t_k + tau) = S_j (x_k + B dv_k), affine in the decision variables.
    sub_S = [cw_stm(n, (j + 1) * dt / n_sub)[:3, :] for j in range(n_sub)]
    d_req = cfg.d_min + cfg.collision_margin

    # Reference post-burn states at each node for the first linearization:
    # ballistic coast (no burns).
    ref = np.empty((n_sat, n_steps + 1, 6))
    for i in range(n_sat):
        for k in range(n_steps + 1):
            ref[i, k] = cw_stm(n, k * dt) @ x_delay[i]

    pairs = list(itertools.combinations(range(n_sat), 2))
    prev_total = np.inf
    result = None

    for it in range(cfg.max_iters):
        dv = [cp.Variable((3, n_steps + 1)) for _ in range(n_sat)]
        x = [cp.Variable((6, n_steps + 1)) for _ in range(n_sat)]
        slack = cp.Variable((len(pairs), n_steps * n_sub), nonneg=True)
        fuel_cap = cp.Variable(nonneg=True)
        # Post-burn state at each node, affine in (x, dv).
        post = [x[i] + _B @ dv[i] for i in range(n_sat)]

        cons = []
        fuels = []
        for i in range(n_sat):
            cons.append(x[i][:, 0] == x_delay[i])
            for k in range(n_steps):
                cons.append(x[i][:, k + 1] == phi @ post[i][:, k])
            cons.append(
                x[i][:, n_steps] + _B @ dv[i][:, n_steps]
                == slots_assigned[i]
            )
            norms = cp.hstack(
                [cp.norm(dv[i][:, k]) for k in range(n_steps + 1)]
            )
            cons.append(norms <= cfg.dv_max)
            fuel_i = cp.sum(norms)
            cons.append(fuel_i <= cfg.fuel_max)
            cons.append(fuel_i <= fuel_cap)
            fuels.append(fuel_i)

        # Linearized keep-out spheres (supporting halfspace => conservative),
        # enforced at every sub-sample time of every coast segment: the
        # samples tau = (j+1) dt / n_sub of segment k tile (t_d, t_f].
        for p, (i, j) in enumerate(pairs):
            for k in range(n_steps):
                for jj, S in enumerate(sub_S):
                    d = S @ ref[i, k] - S @ ref[j, k]
                    nrm = np.linalg.norm(d)
                    nhat = d / nrm if nrm > 1e-9 else np.array([0.0, 1.0, 0.0])
                    a = nhat @ S  # row vector: maps post-burn state -> n.r
                    cons.append(
                        a @ (post[i][:, k] - post[j][:, k])
                        >= d_req - slack[p, k * n_sub + jj]
                    )

        obj = (
            cp.sum(cp.hstack(fuels))
            + cfg.w_balance * fuel_cap
            + cfg.w_slack * cp.sum(slack)
        )
        prob = cp.Problem(cp.Minimize(obj), cons)
        try:
            prob.solve(solver=cfg.solver)
        except cp.SolverError:
            return None
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None

        total = sum(
            float(np.linalg.norm(dv[i].value[:, k]))
            for i in range(n_sat)
            for k in range(n_steps + 1)
        )
        slack_max = float(slack.value.max()) if slack.value.size else 0.0
        converged = abs(prev_total - total) < cfg.tol
        result = ([dv[i].value.copy() for i in range(n_sat)], total,
                  slack_max, it + 1, converged)
        if verbose:
            print(
                f"    SCP iter {it + 1}: dv={total:.4f} m/s, "
                f"slack={slack_max:.2e} m"
            )
        if converged:
            break
        prev_total = total
        for i in range(n_sat):
            # Re-linearize about the new iterate's post-burn states.
            ref[i] = (x[i].value + _B @ dv[i].value).T

    return result


def _finalize_plan(
    chief: CircularOrbit,
    satellites: list[Satellite],
    dv_arrays: list[np.ndarray],
    assignment: tuple[int, ...],
    t_delay: float,
    tf: float,
    cfg: ScpConfig,
    slack_max: float,
    iters: int,
    converged: bool,
) -> RecoveryPlan:
    """Extract discrete burns from the node grid, then score the plan."""
    n_steps = dv_arrays[0].shape[1] - 1
    dt = (tf - t_delay) / n_steps

    burns: dict[str, list[tuple[float, np.ndarray]]] = {}
    for i, sat in enumerate(satellites):
        burns[sat.name] = [
            (t_delay + k * dt, dv_arrays[i][:, k].copy())
            for k in range(n_steps + 1)
        ]
    return _score_burns(
        chief, satellites, burns, assignment, t_delay, tf, cfg,
        slack_max, iters, converged,
    )


def _score_burns(
    chief: CircularOrbit,
    satellites: list[Satellite],
    burns: dict[str, list[tuple[float, np.ndarray]]],
    assignment: tuple[int, ...],
    t_delay: float,
    tf: float,
    cfg: ScpConfig,
    slack_max: float,
    iters: int,
    converged: bool,
) -> RecoveryPlan:
    """Drop negligible burns, re-propagate finely, and score the plan.

    Burns may sit at arbitrary times in [t_delay, tf], not only on a grid.
    """
    n = chief.mean_motion
    per_sat_dv: dict[str, float] = {}
    for sat in satellites:
        kept = [(t, np.asarray(dv, dtype=float))
                for t, dv in burns.get(sat.name, [])
                if np.linalg.norm(dv) >= cfg.burn_floor]
        burns[sat.name] = kept
        per_sat_dv[sat.name] = sum(np.linalg.norm(d) for _, d in kept)
    total_dv = sum(per_sat_dv.values())

    # Fine re-propagation (existing impulses + planned burns) and separation
    # check over the full [0, tf] window, including the uncontrolled delay.
    times = np.arange(0.0, tf + cfg.check_dt / 2, cfg.check_dt)
    trajectories = {}
    for i, sat in enumerate(satellites):
        all_imp = list(sat.impulses) + burns[sat.name]
        trajectories[sat.name] = propagate_cw_with_impulses(
            sat.x0, n, times, all_imp
        )
    names = [s.name for s in satellites]
    min_sep = np.inf
    for i, j in itertools.combinations(range(len(names)), 2):
        d = np.linalg.norm(
            trajectories[names[i]][:, :3] - trajectories[names[j]][:, :3],
            axis=1,
        )
        min_sep = min(min_sep, float(d.min()))

    return RecoveryPlan(
        burns=burns,
        assignment=assignment,
        total_dv=total_dv,
        per_sat_dv=per_sat_dv,
        tf=tf,
        t_delay=t_delay,
        objective=total_dv + cfg.w_tf * tf + cfg.w_slack * slack_max,
        min_separation=min_sep,
        slack_max=slack_max,
        iterations=iters,
        converged=converged,
        times=times,
        trajectories=trajectories,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def plan_recovery(
    chief_orbit: CircularOrbit,
    satellites: list[Satellite],
    t_delay: float,
    tf: float | list[float],
    config: ScpConfig | None = None,
    slots: list[np.ndarray] | None = None,
    n_assignments: int = 3,
    verbose: bool = False,
    allow_reslot: bool = True,
) -> RecoveryPlan:
    """Plan the minimum-cost multi-satellite recovery via SCP.

    Parameters
    ----------
    chief_orbit : circular reference orbit (defines mean motion).
    satellites : formation; each satellite's existing impulses (e.g. the
        perturbation at t = 0) are included in the coast to t_delay.
    t_delay : reaction delay [s] — no burns before this time.
    tf : recovery time [s], or a list of candidate times. With a list, each
        candidate is solved and the best objective total_dv + w_tf*tf wins
        (set config.w_tf > 0 to actually trade fuel against recovery time).
    slots : target slot states (default: each satellite's original x0, i.e.
        the pre-perturbation formation geometry). Any permutation of the
        satellites over these slots is allowed.
    n_assignments : how many of the heuristically-ranked slot permutations
        to solve exactly.
    allow_reslot : if False, every satellite must return to its own slot
        (identity assignment only), which matches the single-satellite
        baselines' problem.
    """
    cfg = config or ScpConfig()
    n = chief_orbit.mean_motion
    if slots is None:
        slots = [sat.x0.copy() for sat in satellites]
    tf_candidates = np.atleast_1d(np.asarray(tf, dtype=float))

    best: RecoveryPlan | None = None
    for tf_c in tf_candidates:
        horizon = tf_c - t_delay
        if horizon <= 0:
            raise ValueError("tf must exceed t_delay")
        x_delay = _states_at_delay(n, satellites, t_delay)
        perms = (
            _rank_assignments(n, x_delay, slots, horizon, n_assignments)
            if allow_reslot else [tuple(range(len(satellites)))]
        )
        for perm in perms:
            if verbose:
                print(f"  tf={tf_c / 60:.1f} min, assignment {perm}:")
            res = _solve_fixed_assignment(
                n, x_delay, [slots[perm[i]] for i in range(len(satellites))],
                horizon, cfg, verbose,
            )
            if res is None:
                if verbose:
                    print("    solver failed / infeasible")
                continue
            dv_arrays, _, slack_max, iters, converged = res
            plan = _finalize_plan(
                chief_orbit, satellites, dv_arrays, perm, t_delay, tf_c,
                cfg, slack_max, iters, converged,
            )
            if verbose:
                print(f"    -> {plan.summary()}")
            if best is None or plan.objective < best.objective:
                best = plan
    if best is None:
        raise RuntimeError("SCP failed for every assignment / tf candidate")
    return best


def baseline_two_impulse_plan(
    chief_orbit: CircularOrbit,
    satellites: list[Satellite],
    perturbed: str,
    t_delay: float,
    tf: float,
    config: ScpConfig | None = None,
) -> RecoveryPlan:
    """Baseline controller: two-impulse CW return of ONLY the perturbed
    satellite to its original slot (no re-slotting, no collision avoidance).

    Returned as a RecoveryPlan so it can be benchmarked and visualized
    exactly like the SCP solution.
    """
    cfg = config or ScpConfig()
    n = chief_orbit.mean_motion
    names = [s.name for s in satellites]
    idx = names.index(perturbed)
    sat = satellites[idx]

    x_delay = _states_at_delay(n, satellites, t_delay)[idx]
    dv1, dv2 = cw_two_impulse(x_delay, n, tf - t_delay, sat.x0)

    n_sat = len(satellites)
    dv_arrays = [np.zeros((3, 2)) for _ in range(n_sat)]
    dv_arrays[idx][:, 0] = dv1
    dv_arrays[idx][:, 1] = dv2
    return _finalize_plan(
        chief_orbit, satellites, dv_arrays, tuple(range(n_sat)),
        t_delay, tf, cfg, slack_max=0.0, iters=0, converged=True,
    )


def baseline_roe_plan(
    chief_orbit: CircularOrbit,
    satellites: list[Satellite],
    perturbed: str,
    t_delay: float,
    tf: float,
    config: ScpConfig | None = None,
    candidates: str = "augmented",
) -> RecoveryPlan:
    """Baseline controller: Chernick & D'Amico closed-form ROE reconfiguration
    of ONLY the perturbed satellite back to its original slot (no
    re-slotting, no collision avoidance).

    Same interface as `baseline_two_impulse_plan`, but the burn times and
    count come from `cwsim.roe.roe_impulsive_reconfig` (tangential in-plane
    triplets plus one cross-track burn) rather than being fixed at t_delay
    and tf. Raises `cwsim.roe.ClosedFormUnavailable` if tf - t_delay is too
    short for the closed-form scheme.
    """
    cfg = config or ScpConfig()
    n = chief_orbit.mean_motion
    names = [s.name for s in satellites]
    idx = names.index(perturbed)
    sat = satellites[idx]

    x_delay = _states_at_delay(n, satellites, t_delay)[idx]
    # u0 only rotates the ROE vectors; pass the true argument of latitude so
    # plan.pseudo_state is physically meaningful.
    u0 = chief_orbit.arg_lat0 + n * t_delay
    roe_plan = roe_impulsive_reconfig(
        x_delay, n, tf - t_delay, sat.x0, u0=u0, candidates=candidates
    )
    burns = {s.name: [] for s in satellites}
    burns[perturbed] = [(t_delay + t, dv) for t, dv in roe_plan.burns]
    return _score_burns(
        chief_orbit, satellites, burns, tuple(range(len(satellites))),
        t_delay, tf, cfg, slack_max=0.0, iters=0, converged=True,
    )
