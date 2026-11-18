"""Sequential Convex Programming (SCP) formation recovery.

Plans impulsive burns for ALL satellites to restore an equally spaced
formation by time tf, allowing the satellites to swap slots (any ordering that
reproduces the original separations is acceptable). plan_recovery's
`controlled` argument restricts the burns to a subset of satellites (e.g.
only the perturbed one); the rest coast and keep their own slots.

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
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .dynamics import cw_stm, cw_two_impulse, propagate_cw_with_impulses
from .orbits import CircularOrbit
from .roe import roe_impulsive_reconfig
from .satellites import Satellite

# Impulse input matrix: dv adds to the velocity half of the state.
_B = np.vstack([np.zeros((3, 3)), np.eye(3)])


# --------------------------------------------------------------------------
# Initial guesses (first SCP linearization point)
# --------------------------------------------------------------------------

# An initial guess maps one satellite's transfer problem to impulses on the
# SCP burn grid:  guess(n, x0, x_target, horizon, n_steps) -> dv (3, n_steps+1)
# where column k is the impulse at node t_k = k * horizon / n_steps (relative
# to the reaction delay), x0 is the coasted state at the delay and x_target
# the assigned slot. The guess only sets the first KOZ linearization; it need
# not satisfy the terminal, thrust or fuel constraints.
InitialGuess = Callable[[float, np.ndarray, np.ndarray, float, int], np.ndarray]


def ballistic_guess(
    n: float, x0: np.ndarray, x_target: np.ndarray, horizon: float,
    n_steps: int,
) -> np.ndarray:
    """No burns: linearize about the uncontrolled coast."""
    return np.zeros((3, n_steps + 1))


def two_impulse_guess(
    n: float, x0: np.ndarray, x_target: np.ndarray, horizon: float,
    n_steps: int,
) -> np.ndarray:
    """Closed-form CW two-impulse transfer: burn at the first and last node.

    Falls back to the ballistic coast if the horizon is singular for the
    two-impulse solution.
    """
    dv = np.zeros((3, n_steps + 1))
    try:
        dv[:, 0], dv[:, -1] = cw_two_impulse(x0, n, horizon, x_target)
    except np.linalg.LinAlgError:
        pass
    return dv


def _solve_convex_transfer(
    n: float,
    x0: np.ndarray,
    x_target: np.ndarray,
    horizon: float,
    n_steps: int,
    dv_max: float = np.inf,
    fuel_max: float = np.inf,
    solver: str = "CLARABEL",
) -> tuple[np.ndarray | None, str]:
    """Minimum-fuel single-satellite CW transfer on the burn grid (no KOZ).

    Minimizes sum_k ||dv_k||_2 over impulses at t_k = k * horizon / n_steps
    with the states eliminated (condensed form):

        Phi(horizon) x0 + sum_k Phi(horizon - t_k) B dv_k == x_target

    subject to ||dv_k|| <= dv_max and sum_k ||dv_k|| <= fuel_max.

    Returns (dv (3, n_steps + 1), status), or (None, status) on failure.
    """
    h = horizon / n_steps
    # Column block k maps dv_k (applied at t_k) to the final state.
    gain = np.hstack(
        [cw_stm(n, horizon - k * h) @ _B for k in range(n_steps + 1)]
    )
    rhs = np.asarray(x_target, dtype=float) - cw_stm(n, horizon) @ x0

    dv = cp.Variable((3, n_steps + 1))
    norms = cp.norm(dv, 2, axis=0)
    constraints = [gain @ cp.vec(dv, order="F") == rhs]
    if np.isfinite(dv_max):
        constraints.append(norms <= dv_max)
    if np.isfinite(fuel_max):
        constraints.append(cp.sum(norms) <= fuel_max)
    prob = cp.Problem(cp.Minimize(cp.sum(norms)), constraints)
    try:
        prob.solve(solver=solver)
    except cp.SolverError as exc:
        return None, f"solver error: {exc}"
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, prob.status
    return dv.value.copy(), prob.status


def convex_guess(
    n: float, x0: np.ndarray, x_target: np.ndarray, horizon: float,
    n_steps: int, *, dv_max: float = np.inf, fuel_max: float = np.inf,
    solver: str = "CLARABEL",
) -> np.ndarray:
    """Convex warm start: the satellite's minimum-fuel transfer to its slot
    on the SCP grid, ignoring the keep-out constraints.

    This is the SCP's own subproblem with the KOZ removed, so the first KOZ
    linearization is taken about the fuel optimum rather than the coast.
    Pass the burn limits with functools.partial to match the SCP:

        initial_guess=partial(convex_guess, dv_max=cfg.dv_max,
                              fuel_max=cfg.fuel_max)

    Each satellite is solved on its own, so the w_balance coupling is ignored
    (it only affects the guess, not the SCP). Falls back to the ballistic
    coast if the convex problem fails.
    """
    dv, _ = _solve_convex_transfer(
        n, x0, x_target, horizon, n_steps, dv_max, fuel_max, solver
    )
    return dv if dv is not None else ballistic_guess(
        n, x0, x_target, horizon, n_steps
    )


def _guess_reference(
    n: float,
    x_delay: list[np.ndarray],
    slots_assigned: list[np.ndarray],
    horizon: float,
    n_steps: int,
    initial_guess: InitialGuess,
    active: list[bool] | None = None,
) -> np.ndarray:
    """Post-burn reference states (n_sat, n_steps+1, 6) of the guessed burns.

    Satellites with active[i] False cannot burn, so they always coast."""
    phi = cw_stm(n, horizon / n_steps)
    ref = np.empty((len(x_delay), n_steps + 1, 6))
    for i, (x0, target) in enumerate(zip(x_delay, slots_assigned)):
        guess = initial_guess if active is None or active[i] else ballistic_guess
        dv = np.asarray(
            guess(n, x0, target, horizon, n_steps), dtype=float
        )
        if dv.shape != (3, n_steps + 1):
            raise ValueError(
                f"initial_guess returned shape {dv.shape}, expected "
                f"(3, {n_steps + 1})"
            )
        x = np.asarray(x0, dtype=float)
        for k in range(n_steps + 1):
            ref[i, k] = x + _B @ dv[:, k]
            x = phi @ ref[i, k]
    return ref


@dataclass
class ScpConfig:
    """Tuning knobs for the SCP recovery planner (SI units)."""

    dt: float = 90.0            # burn-grid spacing [s]
    dv_max: float = 10.0         # max impulse magnitude per burn [m/s]
    fuel_max: float = np.inf    # per-satellite delta-v budget [m/s]
    d_min: float = 200.0        # pairwise keep-out radius [m]
    n_sub: int = 6              # keep-out samples per segment (every dt/n_sub)
    collision_margin: float = 0.0  # pad on d_min for inter-sample excursion [m]
    w_balance: float = 0.0      # weight on max per-sat fuel (balancing)
    w_tf: float = 0.0           # weight on tf [m/s per s] for the tf trade
    w_slack: float = 1.0        # exact-penalty weight on collision slack [1/m]
    max_iters: int = 50        # SCP iterations
    tol: float = 1e-3           # convergence: relative decrease in objective J
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
    n_keep: int | None,
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
    return perms[:n_keep] if n_keep is not None else perms


def _solve_fixed_assignment(
    n: float,
    x_delay: list[np.ndarray],
    slots_assigned: list[np.ndarray],
    horizon: float,
    cfg: ScpConfig,
    verbose: bool = False,
    initial_guess: InitialGuess = ballistic_guess,
    collision_avoidance: bool = True,
    active: list[bool] | None = None,
) -> tuple[list[np.ndarray], float, float, int, bool] | None:
    """SCP solve for one slot assignment over [0, horizon] after the delay.

    With collision_avoidance=False the keep-out constraints are dropped, the
    problem is convex, and a single solve is returned (no iteration).

    active : per-satellite flag; a satellite with active[i] False has its
        burns fixed to zero and no terminal-slot constraint (it just coasts),
        but still takes part in the keep-out constraints. None = all active.

    Returns (dv arrays [(3, N+1) per sat], total_dv, slack_max, iters,
    converged), or None if the convex subproblem fails.
    """
    n_sat = len(x_delay)
    active = [True] * n_sat if active is None else list(active)
    n_steps = max(2, int(round(horizon / cfg.dt)))
    dt = horizon / n_steps
    phi = cw_stm(n, dt)

    # We want to evaluate the KOZ at multiple times for each dt, make a phi to do that
    n_subsamples = max(1, cfg.n_sub)
    # Position rows of the STM at each sub-sample time tau = (j+1) dt / n_sub:
    # r(t_k + tau) = S_j (x_k + B dv_k), affine in the decision variables.
    phi_pos_subsample = [cw_stm(n, (j + 1) * dt / n_subsamples)[:3, :] for j in range(n_subsamples)]

    # min koz dist
    d_req = cfg.d_min + cfg.collision_margin

    # Reference post-burn states at each node for the first linearization,
    # from the user-selectable initial guess (default: ballistic coast).
    ref = _guess_reference(
        n, x_delay, slots_assigned, horizon, n_steps, initial_guess, active
    )

    pairs = list(itertools.combinations(range(n_sat), 2))
    n_cols = n_steps * n_subsamples  # one KOZ sample per (segment k, sub-sample j)
    phi_pos = np.stack(phi_pos_subsample)  # (n_sub, 3, 6)
    # (post @ repeat)[:, k * n_sub + j] == post[:, k]: copies each node's
    # post-burn state to the columns of its sub-samples.
    repeat = np.kron(np.eye(n_steps), np.ones((1, n_subsamples)))

    # ---- Convex subproblem. Variables, dynamics, burn and fuel constraints
    # are built once; only the KOZ linearisation is rebuilt each iteration.
    # Every constraint is vectorised (a handful of cvxpy objects instead of
    # thousands of scalar ones), which is what keeps compilation cheap.
    # (A cp.Parameter/DPP version was tried: Clarabel failed numerically on
    # KOZ-active cases, so the KOZ rows are plain constants.)
    dv = [cp.Variable((3, n_steps + 1)) for _ in range(n_sat)]
    x = [cp.Variable((6, n_steps + 1)) for _ in range(n_sat)]
    # KOZ slack per pair, per step & sub-sampled point
    slack = cp.Variable((len(pairs), n_cols), nonneg=True)
    fuel_cap = cp.Variable(nonneg=True)
    # Post-burn state at each node, affine in (x, dv).
    post = [x[i] + _B @ dv[i] for i in range(n_sat)]

    constraints = []
    fuels = []
    for i in range(n_sat):
        constraints += [
            x[i][:, 0] == x_delay[i],                        # initial state
            x[i][:, 1:] == phi @ post[i][:, :-1],            # CW dynamics
        ]
        if active[i]:
            constraints.append(post[i][:, n_steps] == slots_assigned[i])  # final slot
        else:
            constraints.append(dv[i] == 0)                   # may not burn
        norms = cp.norm(dv[i], 2, axis=0)                    # |dv| per node
        fuel_i = cp.sum(norms)
        constraints += [
            norms <= cfg.dv_max,                             # max dv per burn
            fuel_i <= cfg.fuel_max,                          # fuel budget
            fuel_i <= fuel_cap,                              # fuel balancing
        ]
        fuels.append(fuel_i)

    obj = cp.Minimize(
        cp.sum(cp.hstack(fuels))
        + cfg.w_balance * fuel_cap
        + cfg.w_slack * cp.sum(slack)
    )
    # Relative post-burn state of each pair, copied to every sub-sample column.
    rel_post = [(post[i] - post[j])[:, :n_steps] @ repeat for i, j in pairs]

    def koz_constraints(ref: np.ndarray) -> list:
        """Linearized keep-out spheres (supporting halfspace => conservative)
        about the reference post-burn states, enforced at every sub-sample
        time of every coast segment: the samples tau = (j+1) dt / n_sub of
        segment k tile (t_d, t_f]. Column k*n_sub + j of `rows` is the row
        vector nhat^T S_j mapping the pair's relative post-burn state at
        node k to n.r at that sample."""
        out = []
        for p_idx, (i, j) in enumerate(pairs):
            # relative position of sat i from sat j at every (k, sub-sample)
            rel = np.einsum("jcs,ks->kjc", phi_pos,
                            ref[i, :n_steps] - ref[j, :n_steps])
            nrm = np.linalg.norm(rel, axis=2)
            nhat = np.zeros_like(rel)
            nhat[..., 1] = 1.0  # fallback direction for coincident sats
            ok = nrm > 1e-9
            nhat[ok] = rel[ok] / nrm[ok, None]
            rows = np.einsum("kjc,jcs->kjs", nhat, phi_pos).reshape(n_cols, 6).T
            out.append(cp.sum(cp.multiply(rows, rel_post[p_idx]), axis=0)
                       >= d_req - slack[p_idx])
        return out

    J_prev = np.inf
    result = None

    for it in range(cfg.max_iters):
        koz = koz_constraints(ref) if collision_avoidance else []
        prob = cp.Problem(obj, constraints + koz)
        try:
            prob.solve(solver=cfg.solver)
        except cp.SolverError:
            return None
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None

        total = float(sum(np.linalg.norm(dv[i].value, axis=0).sum()
                          for i in range(n_sat)))
        slack_max = float(slack.value.max()) if slack.value.size else 0.0
        # Converge on the full subproblem objective (fuel + balancing + slack
        # penalty): stop once its decrease is within a relative tolerance and
        # the KOZ is met (slack below 1 mm).
        J = prob.value
        converged = (not collision_avoidance
                     or ((J_prev - J) <= cfg.tol * max(1.0, abs(J))
                         and slack_max < 1e-3))  # metres
        J_prev = J
        result = ([dv[i].value.copy() for i in range(n_sat)], total,
                  slack_max, it + 1, converged)
        if verbose:
            print(
                f"    SCP iter {it + 1}: J={J:.4f}, dv={total:.4f} m/s, "
                f"slack={slack_max:.2e} m"
            )
        if converged:
            break

        for i in range(n_sat):
            # Re-linearize about the new iterate's post-burn states for the KOZ
            ref[i] = (x[i].value + _B @ dv[i].value).T

    return result


def _polish_burns(
    n: float,
    x0: np.ndarray,
    target: np.ndarray,
    dv: np.ndarray,
    horizon: float,
    floor: float,
) -> np.ndarray:
    """Zero the burns below `floor`, then apply the least-norm correction to
    the kept burns so the coast from x0 lands exactly on `target`.

    Interior-point solvers leave many tiny nonzero burns on the grid. Dropping
    them alone (as burn_floor does) can leave a cm-level terminal miss after
    a long coast. The correction is O(solver tolerance) and keeps the support.
    """
    n_steps = dv.shape[1] - 1
    h = horizon / n_steps
    keep = np.flatnonzero(np.linalg.norm(dv, axis=0) >= floor)
    out = np.zeros_like(dv)
    if keep.size == 0:
        return out
    out[:, keep] = dv[:, keep]
    gain = np.hstack([cw_stm(n, horizon - k * h) @ _B for k in keep])
    resid = (target - cw_stm(n, horizon) @ x0
             - gain @ out[:, keep].reshape(-1, order="F"))
    corr = np.linalg.lstsq(gain, resid, rcond=None)[0]
    out[:, keep] += corr.reshape(3, -1, order="F")
    return out


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
    polish: bool = False,
) -> RecoveryPlan:
    """Extract discrete burns from the node grid, then score the plan.

    polish : re-target each controlled satellite's kept burns onto its
        assigned slot after dropping sub-burn_floor impulses (see
        `_polish_burns`). Satellites with no burns are left untouched.
    """
    n_steps = dv_arrays[0].shape[1] - 1
    dt = (tf - t_delay) / n_steps

    if polish:
        n = chief.mean_motion
        x_delay = _states_at_delay(n, satellites, t_delay)
        dv_arrays = [
            _polish_burns(n, x_delay[i], satellites[assignment[i]].x0,
                          dv_arrays[i], tf - t_delay, cfg.burn_floor)
            for i in range(len(satellites))
        ]

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
    n_assignments: int | None = None,
    verbose: bool = False,
    allow_reslot: bool = True,
    initial_guess: InitialGuess = ballistic_guess,
    collision_avoidance: bool = True,
    controlled: Collection[str] | None = None,
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
    n_assignments : how many of the heuristically-ranked slot permutations
        to solve exactly.
    allow_reslot : if False, every satellite must return to its own slot
        (identity assignment only), which matches the single-satellite
        baselines' problem.
    initial_guess : per-satellite function giving the burns that define the
        first KOZ linearization, with signature
        ``guess(n, x0, x_target, horizon, n_steps) -> dv (3, n_steps + 1)``.
        Defaults to `ballistic_guess` (no burns); `two_impulse_guess` uses
        the closed-form CW two-impulse transfer to the assigned slot;
        `convex_guess` uses the keep-out-free fuel optimum (bind dv_max /
        fuel_max with functools.partial).
    collision_avoidance : if False, drop the keep-out constraints. The
        problem is then convex and solved once per (assignment, tf): the
        same grid, dynamics, objective and burn limits as the SCP, so it is
        the fuel lower bound for the SCP at the same settings. The kept
        burns are polished onto the slots (see `_polish_burns`).
    controlled : names of the satellites allowed to burn (default: all).
        The others coast on their current trajectory and keep their own
        slot, but the controlled satellites must still avoid them. Pass
        just the perturbed satellite to match the single-satellite
        baselines without letting the SCP move neighbours out of the way.
    """
    cfg = config or ScpConfig()
    n = chief_orbit.mean_motion
    slots = [sat.x0.copy() for sat in satellites]
    names = [sat.name for sat in satellites]
    if controlled is None:
        active = [True] * len(satellites)
    else:
        unknown = set(controlled) - set(names)
        if unknown:
            raise ValueError(f"controlled names not in the formation: {sorted(unknown)}")
        active = [name in controlled for name in names]
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
        # A satellite that can't burn keeps its own slot.
        perms = [p for p in perms
                 if all(p[i] == i for i in range(len(p)) if not active[i])]
        for perm in perms:
            if verbose:
                print(f"  tf={tf_c / 60:.1f} min, assignment {perm}:")
            res = _solve_fixed_assignment(
                n, x_delay, [slots[perm[i]] for i in range(len(satellites))],
                horizon, cfg, verbose, initial_guess, collision_avoidance,
                active,
            )
            if res is None:
                if verbose:
                    print("    solver failed / infeasible")
                continue
            dv_arrays, _, slack_max, iters, converged = res
            plan = _finalize_plan(
                chief_orbit, satellites, dv_arrays, perm, t_delay, tf_c,
                cfg, slack_max, iters, converged,
                polish=not collision_avoidance,
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


def baseline_convex_plan(
    chief_orbit: CircularOrbit,
    satellites: list[Satellite],
    perturbed: str,
    t_delay: float,
    tf: float,
    config: ScpConfig | None = None,
    dt: float | None = None,
) -> RecoveryPlan:
    """Baseline controller: fuel-optimal convex return of ONLY the perturbed
    satellite to its original slot (no re-slotting, no collision avoidance).

    Minimizes sum_k ||dv_k||_2 over impulses on a uniform grid of nodes
    t_k in [t_delay, tf], subject to the per-burn limit config.dv_max and the
    budget config.fuel_max. The states are eliminated (condensed form):

        x(tf) = Phi(tf - t_delay) x(t_delay)
                + sum_k Phi(tf - t_k) B dv_k  ==  slot

    so the SOCP has only 3 (N + 1) variables and 6 equality rows (see
    `_solve_convex_transfer`). This is the SCP subproblem without the
    keep-out constraints, restricted to one satellite, so it is the lower
    bound the SCP is measured against. The cost converges to the
    continuous-time optimum from above as dt -> 0.

    dt : grid spacing [s]; defaults to config.dt (the SCP grid).
    """
    cfg = config or ScpConfig()
    n = chief_orbit.mean_motion
    names = [s.name for s in satellites]
    idx = names.index(perturbed)
    sat = satellites[idx]

    horizon = tf - t_delay
    if horizon <= 0:
        raise ValueError("tf must exceed t_delay")
    n_steps = max(2, int(round(horizon / (dt or cfg.dt))))

    x_delay = _states_at_delay(n, satellites, t_delay)[idx]
    dv, status = _solve_convex_transfer(
        n, x_delay, sat.x0, horizon, n_steps,
        cfg.dv_max, cfg.fuel_max, cfg.solver,
    )
    if dv is None:
        raise RuntimeError(f"convex baseline {status}")

    dv_arrays = [np.zeros((3, n_steps + 1)) for _ in satellites]
    dv_arrays[idx] = dv
    return _finalize_plan(
        chief_orbit, satellites, dv_arrays, tuple(range(len(satellites))),
        t_delay, tf, cfg, slack_max=0.0, iters=1,
        converged=status == "optimal", polish=True,
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
