"""Edit SETTINGS below, then run with the AA279B project Python environment.
Uses main's plan_recovery(..., allow_reslot=...) without modifying the planner.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

# ====================== EDIT THESE SETTINGS ======================
SETTINGS = {
    'theta_deg': list(range(0, 360, 10)),
    'phi_deg': [0],                       # [0] = in-plane disturbances
    'perturbation_dv': [0.05, 0.10, 0.15, 0.20],  # m/s
    'delay_orbits': [0.10, 0.25, 0.50, 0.75, 1.00],

    'enforce_koz': False,                 # True enables collision constraints
    'allow_reordering': False,           # True lets planner rank slot permutations
    'n_assignments': 6,                   # 6 = all permutations for three satellites
    'koz_radius_m': 200.0,
    'koz_margin_m': 10.0,

    # 'fixed_deadline': final time measured from disturbance.
    # 'fixed_duration': final time = reaction delay + recovery duration.
    'timing_mode': 'fixed_deadline',
    'deadline_orbits': 2.15,
    'recovery_duration_orbits': 1.50,

    'spacing_m': 1000.0,
    'burn_grid_s': 300.0,
    'max_burn_dv': 1000.0,                # m/s; original loose bound
    'fuel_per_satellite': 1e6,            # m/s; original loose bound
    'fuel_balance_weight': 0.0,
    'max_iterations': 100,
    'dv_convergence_tolerance': 1e-3,
    'collision_subsamples': 6,
    'validation_step_s': 5.0,
    'position_tolerance_m': 0.01,
    'velocity_tolerance_m_s': 1e-5,
    'slack_tolerance_m': 1e-6,
    'workers': 6,
}
# ================================================================
DEFAULT_PROJECT = Path('/Users/julianbrown/Stanford/AA 279B (Advanced Space Mechancs)/RPO Project/AA279B_Project')


def validate_settings(s):
    import math
    if s['timing_mode'] not in ('fixed_deadline', 'fixed_duration'):
        raise ValueError('timing_mode must be fixed_deadline or fixed_duration')
    for name in ('theta_deg', 'phi_deg', 'perturbation_dv', 'delay_orbits'):
        if not s[name] or not all(math.isfinite(x) for x in s[name]):
            raise ValueError(f'{name} must be a nonempty list of finite numbers')
    if any(x < 0 for x in s['delay_orbits'] + s['perturbation_dv']):
        raise ValueError('Delays and disturbance magnitudes must be nonnegative')
    for name in ('spacing_m', 'burn_grid_s', 'max_burn_dv', 'fuel_per_satellite',
                 'validation_step_s', 'recovery_duration_orbits', 'deadline_orbits',
                 'dv_convergence_tolerance', 'position_tolerance_m', 'velocity_tolerance_m_s'):
        if not math.isfinite(s[name]) or s[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    for name in ('koz_radius_m', 'koz_margin_m', 'slack_tolerance_m', 'fuel_balance_weight'):
        if not math.isfinite(s[name]) or s[name] < 0:
            raise ValueError(f'{name} must be nonnegative and finite')
    for name in ('workers', 'n_assignments', 'collision_subsamples', 'max_iterations'):
        if not isinstance(s[name], int) or s[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if s['max_iterations'] < 2:
        raise ValueError('At least two iterations are required for the cost-change test')
    if s['timing_mode']=='fixed_deadline' and max(s['delay_orbits']) >= s['deadline_orbits']:
        raise ValueError('Every reaction delay must precede the fixed deadline')


def load_project(project):
    sys.path.insert(0, str(project))
    try:
        from cwsim import CircularOrbit, along_track_formation
        from cwsim.dynamics import propagate_cw_with_impulses
        from cwsim.scp import ScpConfig, plan_recovery
    except ModuleNotFoundError as exc:
        raise RuntimeError(f'Project import failed: {exc}. Ensure your local project includes all files from main (including cwsim/roe.py).') from exc
    return CircularOrbit, along_track_formation, propagate_cw_with_impulses, ScpConfig, plan_recovery


def terminal_errors(states, sats, assignment):
    import numpy as np
    targets = np.asarray([sats[j].x0 for j in assignment])
    residual = states[:, -1, :] - targets
    return (float(np.linalg.norm(residual[:, :3], axis=1).max()),
            float(np.linalg.norm(residual[:, 3:], axis=1).max()))


def convex_recovery(chief, sats, td, tf, cfg, allow_reordering, n_assignments):
    """One convex solve per candidate assignment, with no collision constraints.

    Uses the same CW grid, objective, endpoint constraints and control bounds
    as main's SCP. Main's scoring helper packages the resulting burn schedule.
    """
    import numpy as np
    import cvxpy as cp
    from cwsim.dynamics import cw_stm
    from cwsim.scp import _states_at_delay, _rank_assignments, _finalize_plan
    n = chief.mean_motion
    horizon = tf-td
    if horizon <= 0:
        raise ValueError('Recovery duration must be positive')
    x0 = _states_at_delay(n, sats, td)
    slots = [sat.x0.copy() for sat in sats]
    perms = (_rank_assignments(n, x0, slots, horizon, n_assignments)
             if allow_reordering else [tuple(range(len(sats)))])
    steps = max(2, int(round(horizon/cfg.dt)))
    phi = cw_stm(n, horizon/steps)
    B = np.vstack([np.zeros((3, 3)), np.eye(3)])
    best = None
    for perm in perms:
        burns = [cp.Variable((3, steps+1)) for _ in sats]
        states = [cp.Variable((6, steps+1)) for _ in sats]
        cap = cp.Variable(nonneg=True)
        constraints, fuels = [], []
        for i in range(len(sats)):
            post = states[i] + B @ burns[i]
            constraints.append(states[i][:, 0] == x0[i])
            for k in range(steps):
                constraints.append(states[i][:, k+1] == phi @ post[:, k])
            constraints.append(post[:, steps] == slots[perm[i]])
            norms = cp.hstack([cp.norm(burns[i][:, k], 2) for k in range(steps+1)])
            fuel = cp.sum(norms)
            constraints.extend([norms <= cfg.dv_max, fuel <= cfg.fuel_max, fuel <= cap])
            fuels.append(fuel)
        problem = cp.Problem(cp.Minimize(cp.sum(cp.hstack(fuels))+cfg.w_balance*cap), constraints)
        try:
            problem.solve(solver=cfg.solver, canon_backend=cp.SCIPY_CANON_BACKEND)
        except cp.SolverError:
            continue
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            continue
        plan = _finalize_plan(chief, sats, [b.value.copy() for b in burns],
            perm, td, tf, cfg, slack_max=0., iters=1,
            converged=problem.status == 'optimal')
        if best is None or plan.objective < best.objective:
            best = plan
    if best is None:
        raise RuntimeError('No candidate convex solve returned an optimal or optimal_inaccurate solution')
    return best


def run_case(project, s, case):
    import numpy as np
    from itertools import combinations
    CircularOrbit, formation, propagate, ScpConfig, plan_recovery = load_project(project)
    magnitude, delay, phi_deg, theta_deg = case
    row = dict(perturbation_dv=magnitude, delay_orbits=delay, phi_deg=phi_deg,
        theta_deg=theta_deg, enforce_koz=s['enforce_koz'], allow_reordering=s['allow_reordering'],
        solver_mode='scp_with_koz' if s['enforce_koz'] else 'convex_no_koz',
        timing_mode=s['timing_mode'], final_time_orbits=float('nan'), recovery_duration_orbits=float('nan'),
        scp_total_dv=float('nan'), sat1_dv=float('nan'), sat2_dv=float('nan'), sat3_dv=float('nan'),
        assignment='', converged=False, iterations=0, slack_max=float('nan'),
        terminal_position_error_m=float('nan'), terminal_velocity_error_m_s=float('nan'),
        delay_min_separation_m=float('nan'), recovery_min_separation_m=float('nan'),
        delay_koz_violation='', recovery_koz_violation='', planning_margin_met='',
        max_cross_track_m=float('nan'), max_burn_dv=float('nan'),
        burn_and_fuel_limits_met=False, recovery_requirements_met=False, full_mission_requirements_met=False,
        accepted=False, status='error', error='', runtime_s=0.)
    start = time.perf_counter()
    try:
        chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
        sats = formation(spacing=s['spacing_m'], count=3)
        theta, phi = np.radians([theta_deg, phi_deg])
        dv = magnitude*np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])
        sats[1].add_impulse(0., dv)
        td = delay*chief.period
        tf_orbits = s['deadline_orbits'] if s['timing_mode']=='fixed_deadline' else delay+s['recovery_duration_orbits']
        tf = tf_orbits*chief.period
        row.update(final_time_orbits=tf_orbits, recovery_duration_orbits=tf_orbits-delay)
        cfg = ScpConfig(dt=s['burn_grid_s'], dv_max=s['max_burn_dv'], fuel_max=s['fuel_per_satellite'],
            d_min=s['koz_radius_m'], collision_margin=s['koz_margin_m'],
            w_balance=s['fuel_balance_weight'], w_tf=0.,
            burn_floor=0., max_iters=s['max_iterations'], tol=s['dv_convergence_tolerance'],
            n_sub=s['collision_subsamples'], check_dt=s['validation_step_s'])
        # No case is skipped for proximity or baseline conditioning.
        if s['enforce_koz']:
            plan = plan_recovery(chief, sats, td, tf, cfg,
                allow_reslot=s['allow_reordering'], n_assignments=s['n_assignments'])
        else:
            plan = convex_recovery(chief, sats, td, tf, cfg,
                s['allow_reordering'], s['n_assignments'])
        times = np.unique(np.r_[np.arange(0., tf, s['validation_step_s']), td, tf])
        states = np.array([propagate(sat.x0, chief.mean_motion, times,
            list(sat.impulses)+plan.burns[sat.name]) for sat in sats])
        separations = np.array([np.linalg.norm(states[i, :, :3]-states[j, :, :3], axis=1)
                               for i, j in combinations(range(3), 2)])
        pos_error, vel_error = terminal_errors(states, sats, plan.assignment)
        delay_min = float(separations[:, times <= td].min())
        recovery_min = float(separations[:, times >= td].min())
        costs = [float(sum(np.linalg.norm(d) for _, d in plan.burns[sat.name])) for sat in sats]
        largest = max([float(np.linalg.norm(d)) for sat in sats for _, d in plan.burns[sat.name]] or [0.])
        finite = bool(np.isfinite(states).all() and np.isfinite(costs).all())
        limits_met = bool(largest <= s['max_burn_dv']+1e-6 and max(costs) <= s['fuel_per_satellite']+1e-6)
        endpoint_ok = pos_error <= s['position_tolerance_m'] and vel_error <= s['velocity_tolerance_m_s']
        recovery_ok = bool(finite and limits_met and endpoint_ok and
                           (not s['enforce_koz'] or recovery_min >= s['koz_radius_m']))
        mission_ok = bool(recovery_ok and (not s['enforce_koz'] or delay_min >= s['koz_radius_m']))
        accepted = bool(mission_ok and plan.converged and
                        (not s['enforce_koz'] or plan.slack_max <= s['slack_tolerance_m']))
        row.update(scp_total_dv=sum(costs), sat1_dv=costs[0], sat2_dv=costs[1], sat3_dv=costs[2],
            assignment=str(plan.assignment), converged=bool(plan.converged), iterations=int(plan.iterations),
            slack_max=float(plan.slack_max), terminal_position_error_m=pos_error,
            terminal_velocity_error_m_s=vel_error, delay_min_separation_m=delay_min,
            recovery_min_separation_m=recovery_min, delay_koz_violation=delay_min < s['koz_radius_m'],
            recovery_koz_violation=recovery_min < s['koz_radius_m'],
            planning_margin_met=recovery_min >= s['koz_radius_m']+s['koz_margin_m'],
            max_cross_track_m=float(np.abs(states[:, :, 2]).max()), max_burn_dv=largest,
            burn_and_fuel_limits_met=limits_met, recovery_requirements_met=recovery_ok,
            full_mission_requirements_met=mission_ok, accepted=accepted, status='returned')
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['runtime_s'] = time.perf_counter()-start
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    candidate = Path(__file__).resolve().parents[1]
    parser.add_argument('--project', type=Path, default=candidate if (candidate/'cwsim').is_dir() else DEFAULT_PROJECT)
    parser.add_argument('--limit', type=int, default=0, help='Run only first N cases')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Check settings and display case count without running')
    args = parser.parse_args()
    s = dict(SETTINGS)
    if args.workers is not None: s['workers'] = args.workers
    validate_settings(s)
    if args.limit < 0: parser.error('limit must be nonnegative')
    cases = [(m, d, p, t) for m in s['perturbation_dv'] for d in s['delay_orbits']
             for p in s['phi_deg'] for t in s['theta_deg']]
    if args.limit: cases = cases[:args.limit]
    print(f"{len(cases)} cases | KOZ={s['enforce_koz']} | reordering={s['allow_reordering']} | timing={s['timing_mode']}", flush=True)
    if args.dry_run: return
    load_project(args.project)  # Fail before creating outputs if project dependencies are missing.
    tag = f"scp_koz{int(s['enforce_koz'])}_reorder{int(s['allow_reordering'])}_"+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.output_dir or args.project/'my_trade_studies'/'results'/tag
    out.mkdir(parents=True, exist_ok=False)
    (out/'settings.json').write_text(json.dumps(s, indent=2))
    print(f'Output: {out}', flush=True)
    with (out/'results.csv').open('x', newline='') as f, ProcessPoolExecutor(max_workers=s['workers']) as pool:
        futures = [pool.submit(run_case, str(args.project), s, case) for case in cases]
        writer = None
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); f.flush()
            print(f"[{i}/{len(cases)}] dv={row['perturbation_dv']}, delay={row['delay_orbits']}, theta={row['theta_deg']}: {row['status']}, accepted={row['accepted']} {row['error']}", flush=True)
    print(f'Done: {out}', flush=True)


if __name__ == '__main__':
    main()
