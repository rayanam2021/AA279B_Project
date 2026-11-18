"""Sweep a recovery planner over disturbance direction, magnitude and reaction delay.

Scenario: 3-satellite along-track formation. PERTURBED is hit by an in-plane
impulse of magnitude dv at angle theta (0 = radial, 90 = along-track) at t = 0.
Nothing burns until t_delay = delay * period; the planner must restore the
formation by tf = t_delay + RECOVERY_ORBITS * period.

Every planner in METHODS returns a cwsim RecoveryPlan, and each case is reduced
to one CSV row of plan metadata (no trajectories). The columns are the same for
every method, so CSVs from different methods can be loaded with load_sweep()
and compared directly. Metrics that don't apply to a method (e.g. iterations
and slack for the closed-form baselines) keep the values the plan reports.

Run:
    python my_trade_studies/configurable_scp_sweep.py --method scp
    python my_trade_studies/configurable_scp_sweep.py --method cw --limit 5 --serial

Output: results/sweep_<method>_<timestamp>.csv plus a .json sidecar with the
settings used.
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # one BLAS thread per worker process

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cvxpy as cp  # noqa: E402

from cwsim import CircularOrbit, along_track_formation, propagate_cw_with_impulses  # noqa: E402
from cwsim.dynamics import cw_stm  # noqa: E402
from cwsim.scp import (  # noqa: E402
    RecoveryPlan,
    ScpConfig,
    _finalize_plan,
    _states_at_delay,
    baseline_roe_plan,
    baseline_two_impulse_plan,
    plan_recovery,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ============================ sweep settings ============================
THETA_DEG = list(range(0, 360, 10))        # 0 = radial, 90 = along-track
DV_MAG = [0.05, 0.10, 0.15, 0.20]          # disturbance magnitude [m/s]
DELAY_ORBITS = [0.10, 0.25, 0.50, 0.75, 1.00]
RECOVERY_ORBITS = 2.15                     # tf - t_delay [orbits]; not a round number (CW singularity)
SPACING_M = 1000.0
N_SATS = 3
PERTURBED = "Sat-2"                        # middle satellite

CONFIG = ScpConfig(
    dv_max=1000.0,        # per-burn limit [m/s]
    fuel_max=1e6,         # per-satellite budget [m/s]
    d_min=200.0,          # keep-out radius [m]
    w_balance=0.0,
)
METHOD = "scp"            # default planner; override with --method
# ========================================================================


# ------------------------------- planners -------------------------------
# Each planner: fn(chief, sats, t_delay, tf) -> RecoveryPlan

def convex_recovery(chief, sats, t_delay, tf, cfg, allow_reslot=False):
    """Single convex solve per slot assignment, with NO keep-out constraints.

    Same grid, dynamics, objective and burn limits as plan_recovery's SCP
    subproblem, minus the collision constraints (so no iteration is needed).
    """
    n = chief.mean_motion
    horizon = tf - t_delay
    x0 = _states_at_delay(n, sats, t_delay)
    slots = [sat.x0.copy() for sat in sats]
    perms = (list(itertools.permutations(range(len(sats)))) if allow_reslot
             else [tuple(range(len(sats)))])
    steps = max(2, int(round(horizon / cfg.dt)))
    phi = cw_stm(n, horizon / steps)
    B = np.vstack([np.zeros((3, 3)), np.eye(3)])

    best = None
    for perm in perms:
        burns = [cp.Variable((3, steps + 1)) for _ in sats]
        states = [cp.Variable((6, steps + 1)) for _ in sats]
        fuel_cap = cp.Variable(nonneg=True)
        constraints, fuels = [], []
        for i in range(len(sats)):
            post = states[i] + B @ burns[i]
            constraints.append(states[i][:, 0] == x0[i])
            constraints += [states[i][:, k + 1] == phi @ post[:, k] for k in range(steps)]
            constraints.append(post[:, steps] == slots[perm[i]])
            norms = cp.norm(burns[i], 2, axis=0)
            fuel = cp.sum(norms)
            constraints += [norms <= cfg.dv_max, fuel <= cfg.fuel_max, fuel <= fuel_cap]
            fuels.append(fuel)
        problem = cp.Problem(cp.Minimize(sum(fuels) + cfg.w_balance * fuel_cap), constraints)
        try:
            problem.solve(solver=cfg.solver)
        except cp.SolverError:
            continue
        if problem.status not in ("optimal", "optimal_inaccurate"):
            continue
        plan = _finalize_plan(chief, sats, [b.value.copy() for b in burns], perm,
                              t_delay, tf, cfg, slack_max=0.0, iters=1,
                              converged=problem.status == "optimal")
        if best is None or plan.objective < best.objective:
            best = plan
    if best is None:
        raise RuntimeError("convex solve failed for every assignment")
    return best


def scp(chief, sats, td, tf):
    return plan_recovery(chief, sats, td, tf, CONFIG, allow_reslot=False)


def scp_reslot(chief, sats, td, tf):
    return plan_recovery(chief, sats, td, tf, CONFIG, allow_reslot=True)


def convex(chief, sats, td, tf):
    return convex_recovery(chief, sats, td, tf, CONFIG, allow_reslot=False)


def cw(chief, sats, td, tf):
    return baseline_two_impulse_plan(chief, sats, PERTURBED, td, tf, CONFIG)


def roe(chief, sats, td, tf):
    return baseline_roe_plan(chief, sats, PERTURBED, td, tf, CONFIG)


METHODS = {
    "scp": scp,
    "scp_reslot": scp_reslot,
    "convex": convex,
    "cw": cw,
    "roe": roe,
}


# ------------------------------- one case -------------------------------
SAT_NAMES = [f"Sat-{i + 1}" for i in range(N_SATS)]
FIELDS = (
    ["method", "theta_deg", "dv_mag", "delay_orbits", "recovery_orbits", "t_delay", "tf",
     "status", "error", "runtime_s",
     "total_dv"] + [f"dv_{name}" for name in SAT_NAMES] +
    ["n_burns", "max_burn_dv", "assignment", "reslotted", "objective",
     "min_separation", "min_sep_recovery", "koz_ok", "slack_max", "iterations", "converged",
     "term_pos_err_m", "term_vel_err_m_s"]
)


def summarise(plan: RecoveryPlan, chief, sats) -> dict:
    """Flatten a RecoveryPlan into CSV-friendly metadata (no trajectories)."""
    n = chief.mean_motion
    burn_norms = [np.linalg.norm(dv) for b in plan.burns.values() for _, dv in b]

    # Separation over the recovery window only (the delay phase is uncontrollable).
    after = plan.times >= plan.t_delay
    min_sep_recovery = min(
        np.linalg.norm(plan.trajectories[a.name][after, :3]
                       - plan.trajectories[b.name][after, :3], axis=1).min()
        for a, b in itertools.combinations(sats, 2))

    # Terminal error vs. assigned slot, evaluated exactly at tf (just after any burn at tf).
    pos_err = vel_err = 0.0
    for i, sat in enumerate(sats):
        x_tf = propagate_cw_with_impulses(sat.x0, n, np.array([plan.tf + 1e-6]),
                                          list(sat.impulses) + plan.burns[sat.name])[0]
        err = x_tf - sats[plan.assignment[i]].x0
        pos_err = max(pos_err, np.linalg.norm(err[:3]))
        vel_err = max(vel_err, np.linalg.norm(err[3:]))

    return {
        "total_dv": plan.total_dv,
        **{f"dv_{name}": dv for name, dv in plan.per_sat_dv.items()},
        "n_burns": len(burn_norms),
        "max_burn_dv": max(burn_norms, default=0.0),
        "assignment": "-".join(map(str, plan.assignment)),
        "reslotted": tuple(plan.assignment) != tuple(range(len(sats))),
        "objective": plan.objective,
        "min_separation": plan.min_separation,
        "min_sep_recovery": min_sep_recovery,
        "koz_ok": min_sep_recovery >= CONFIG.d_min,
        "slack_max": plan.slack_max,
        "iterations": plan.iterations,
        "converged": plan.converged,
        "term_pos_err_m": pos_err,
        "term_vel_err_m_s": vel_err,
    }


def run_case(method: str, theta_deg: float, dv_mag: float, delay_orbits: float) -> dict:
    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    sats = along_track_formation(spacing=SPACING_M, count=N_SATS)
    theta = np.radians(theta_deg)
    next(s for s in sats if s.name == PERTURBED).add_impulse(
        0.0, dv_mag * np.array([np.cos(theta), np.sin(theta), 0.0]))
    td = delay_orbits * chief.period
    tf = td + RECOVERY_ORBITS * chief.period

    row = dict(method=method, theta_deg=theta_deg, dv_mag=dv_mag, delay_orbits=delay_orbits,
               recovery_orbits=RECOVERY_ORBITS, t_delay=td, tf=tf, status="ok", error="")
    start = time.perf_counter()
    try:
        plan = METHODS[method](chief, sats, td, tf)
        row.update(summarise(plan, chief, sats))
    except Exception as exc:  # record the failure and keep sweeping
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    row["runtime_s"] = time.perf_counter() - start
    return row


# ------------------------------- loading -------------------------------
def load_sweep(*csv_paths) -> dict[str, np.ndarray]:
    """Load one or more sweep CSVs into {column: array}.

    Numeric columns become float arrays (failed cases -> NaN), True/False
    columns become bool arrays, everything else stays as strings.
    """
    rows = []
    for path in csv_paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    out = {}
    for col in rows[0]:
        values = [r[col] for r in rows]
        if set(values) == {""}:  # e.g. an error column with no errors
            out[col] = np.array(values)
            continue
        if set(values) <= {"True", "False", ""}:
            out[col] = np.array([v == "True" for v in values])
            continue
        try:
            out[col] = np.array([float(v) if v != "" else np.nan for v in values])
        except ValueError:
            out[col] = np.array(values)
    return out


# --------------------------------- main ---------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, default=METHOD)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    parser.add_argument("--debug", action="store_true",
                        help="run in this process (for debugging / breakpoints)")
    parser.add_argument("--out", type=Path, help="output CSV path")
    args = parser.parse_args()

    cases = list(itertools.product(THETA_DEG, DV_MAG, DELAY_ORBITS))
    if args.limit:
        cases = cases[:args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or RESULTS_DIR / f"sweep_{args.method}_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(dict(
        method=args.method, theta_deg=THETA_DEG, dv_mag=DV_MAG, delay_orbits=DELAY_ORBITS,
        recovery_orbits=RECOVERY_ORBITS, spacing_m=SPACING_M, n_sats=N_SATS,
        perturbed=PERTURBED, config=asdict(CONFIG)), indent=2))
    print(f"{len(cases)} cases, method={args.method} -> {out}", flush=True)

    with out.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        writer.writeheader()

        def record(i, row):
            writer.writerow(row)
            f.flush()
            print(f"[{i}/{len(cases)}] theta={row['theta_deg']:>5} dv={row['dv_mag']:.2f} "
                  f"delay={row['delay_orbits']:.2f}: {row['status']} "
                  f"total_dv={row.get('total_dv', float('nan')):.4f} {row['error']}", flush=True)

        if args.debug:
            for i, case in enumerate(cases, 1):
                record(i, run_case(args.method, *case))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_case, args.method, *case) for case in cases]
                for i, future in enumerate(as_completed(futures), 1):
                    record(i, future.result())
    print(f"Done: {out}", flush=True)


if __name__ == "__main__":
    main()
