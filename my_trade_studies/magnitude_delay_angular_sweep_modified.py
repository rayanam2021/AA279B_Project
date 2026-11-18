"""Four-dimensional CW recovery sweep with six parallel workers."""

import csv
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from cwsim import CircularOrbit, along_track_formation
from cwsim.dynamics import cw_stm, propagate_cw_with_impulses
from cwsim.scp import ScpConfig, baseline_two_impulse_plan, plan_recovery


# ============================================================
# SIMULATION SETTINGS
# ============================================================

TEST_MODE = False
NUM_WORKERS = 6

THETA_VALUES = np.arange(0, 360, 10)
PHI_VALUES = np.arange(-90, 91, 10)

DV_VALUES = [0.05, 0.10, 0.15, 0.20]
DELAY_VALUES = [0.10, 0.25, 0.50, 0.75, 1.00]

TF_ORBITS = 2.15

KOZ_RADIUS = 200.0
KOZ_MARGIN = 10.0
CHECK_DT = 15.0

MU_EARTH = 3.986004418e14

COND_LIMIT = 1e10
TERMINAL_ERROR_LIMIT = 1e-2

PRINT_SATELLITE_DV = False

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "magnitude_delay_angular_results_comparison.csv"


if TEST_MODE:
    THETA_VALUES = [0, 90, 180, 270]
    PHI_VALUES = [-45, 0, 45]
    DV_VALUES = [0.05, 0.20]
    DELAY_VALUES = [0.25, 0.75]


# ============================================================
# ORBIT AND OPTIMIZATION CONFIGURATION
# ============================================================

chief = CircularOrbit(
    altitude=550e3,
    inclination=np.radians(51.6)
)

cfg = ScpConfig(
    dt=300.0,
    dv_max=1e3,
    fuel_max=1e6,
    d_min=200.0,
    collision_margin=10.0,
    enforce_collision=False,
    w_balance=0.0,
    w_tf=0.0,
    burn_floor=0.0,
    max_iters=100,
    check_dt=CHECK_DT,
)

tf = TF_ORBITS * chief.period


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def minimum_pairwise_separation(states):
    positions = np.asarray(states, dtype=float)[:, :3]

    return min(
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )


def delay_check(sats, n, delay):
    times = np.unique(
        np.r_[np.arange(0, delay, CHECK_DT), delay]
    )

    trajectories = [
        propagate_cw_with_impulses(
            s.x0, n, times, s.impulses
        )
        for s in sats
    ]

    return min(
        minimum_pairwise_separation(
            [traj[k] for traj in trajectories]
        )
        for k in range(len(times))
    )


def bound_orbit_from_cw(state, n):
    radius = (MU_EARTH / n**2)**(1.0 / 3.0)

    r = np.array([
        radius + state[0],
        state[1],
        state[2]
    ])

    v = np.array([
        state[3] - n * state[1],
        state[4] + n * (radius + state[0]),
        state[5]
    ])

    energy = (
        0.5 * np.dot(v, v)
        - MU_EARTH / np.linalg.norm(r)
    )

    return np.isfinite(energy) and energy < 0.0


def terminal_error(sats, plan, n, tf):
    """Return maximum combined six-state terminal residual.

    This metric combines position [m] and velocity [m/s]
    numerically, as in the original simulation.
    """

    errors = []

    for sat in sats:
        impulses = list(sat.impulses) + plan.burns[sat.name]

        final_state = propagate_cw_with_impulses(
            sat.x0,
            n,
            np.array([tf]),
            impulses
        )[0]

        errors.append(
            np.linalg.norm(final_state - sat.x0)
        )

    return max(errors)


# ============================================================
# CSV COLUMNS
# ============================================================

FIELDNAMES = [
    "theta_deg",
    "phi_deg",
    "perturbation_dv",
    "delay_orbits",
    "delay_seconds",
    "dv_R",
    "dv_I",
    "dv_C",
    "scp_total_dv",
    "baseline_total_dv",
    "fuel_savings_percent",
    "min_separation",
    "delay_min_separation",
    "baseline_min_separation",
    "assignment",
    "baseline_condition_number",
    "scp_terminal_error",
    "baseline_terminal_error",
    "status",
]


# ============================================================
# INDIVIDUAL SIMULATION CASE
# ============================================================

def run_case(case):
    """Evaluate one perturbation case in an independent process."""

    dv_mag, delay_orb, phi_deg, theta_deg = case

    t_delay = delay_orb * chief.period

    theta, phi = np.radians([theta_deg, phi_deg])

    perturbation = dv_mag * np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi)
    ])

    row = dict(
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        perturbation_dv=dv_mag,
        delay_orbits=delay_orb,
        delay_seconds=t_delay,
        dv_R=perturbation[0],
        dv_I=perturbation[1],
        dv_C=perturbation[2],
        scp_total_dv=np.nan,
        baseline_total_dv=np.nan,
        fuel_savings_percent=np.nan,
        min_separation=np.nan,
        delay_min_separation=np.nan,
        baseline_min_separation=np.nan,
        assignment="",
        baseline_condition_number=np.nan,
        scp_terminal_error=np.nan,
        baseline_terminal_error=np.nan,
        status="success"
    )

    try:
        # ----------------------------------------------------
        # INITIALIZE SATELLITES
        # ----------------------------------------------------

        sats = along_track_formation(
            spacing=1000.0,
            count=3
        )

        sats[1].add_impulse(0.0, perturbation)

        n = chief.mean_motion

        # ----------------------------------------------------
        # PROPAGATE THROUGH REACTION DELAY
        # ----------------------------------------------------

        x_delay = [
            propagate_cw_with_impulses(
                sat.x0,
                n,
                np.array([t_delay]),
                sat.impulses
            )[0]
            for sat in sats
        ]

        # ----------------------------------------------------
        # CHECK BOUND ORBITS
        # ----------------------------------------------------

        if not all(
            bound_orbit_from_cw(x, n)
            for x in x_delay
        ):
            row["status"] = "skipped_unbound_orbit"
            return row

        # ----------------------------------------------------
        # CHECK UNCONTROLLED-DELAY KOZ
        # ----------------------------------------------------

        row["delay_min_separation"] = delay_check(
            sats, n, t_delay
        )

        if row["delay_min_separation"] < KOZ_RADIUS + KOZ_MARGIN:
            row["status"] = "skipped_delay_koz"
            return row

        # ----------------------------------------------------
        # CHECK TWO-IMPULSE CONDITIONING
        # ----------------------------------------------------

        horizon = tf - t_delay

        phi_h = cw_stm(n, horizon)

        cond = np.linalg.cond(phi_h[:3, 3:6])

        row["baseline_condition_number"] = cond

        if not np.isfinite(cond) or cond > COND_LIMIT:
            row["status"] = "skipped_singular_two_impulse"
            return row

        # ----------------------------------------------------
        # TWO-IMPULSE BASELINE
        # ----------------------------------------------------

        try:
            baseline = baseline_two_impulse_plan(
                chief,
                sats,
                "Sat-2",
                t_delay,
                tf,
                cfg
            )

            row["baseline_total_dv"] = baseline.total_dv
            row["baseline_min_separation"] = baseline.min_separation

            row["baseline_terminal_error"] = terminal_error(
                sats,
                baseline,
                n,
                tf
            )

            if not np.isfinite(baseline.total_dv):
                row["status"] = "invalid_baseline"
                return row

            if (
                not np.isfinite(row["baseline_terminal_error"])
                or row["baseline_terminal_error"] > TERMINAL_ERROR_LIMIT
            ):
                row["status"] = "baseline_terminal_error"
                return row

        except Exception as e:
            row["status"] = f"baseline_error: {e}"
            return row

        # ----------------------------------------------------
        # SCP RECOVERY
        # ----------------------------------------------------

        try:
            scp = plan_recovery(
                chief,
                sats,
                t_delay,
                tf,
                cfg,
                allow_reslot=False
            )

            if PRINT_SATELLITE_DV:
                print("SCP per-satellite ΔV:", scp.per_sat_dv)
                print("Baseline per-satellite ΔV:", baseline.per_sat_dv)

            row["scp_total_dv"] = scp.total_dv
            row["min_separation"] = scp.min_separation
            row["assignment"] = str(scp.assignment)

            row["scp_terminal_error"] = terminal_error(
                sats,
                scp,
                n,
                tf
            )

            if not np.isfinite(scp.total_dv):
                row["status"] = "invalid_scp"
                return row

            if (
                not np.isfinite(row["scp_terminal_error"])
                or row["scp_terminal_error"] > TERMINAL_ERROR_LIMIT
            ):
                row["status"] = "scp_terminal_error"
                return row

            if baseline.total_dv > 0:
                row["fuel_savings_percent"] = 100.0 * (
                    1.0 - scp.total_dv / baseline.total_dv
                )

        except Exception as e:
            row["status"] = f"scp_error: {e}"

    except Exception as e:
        row["status"] = f"case_error: {e}"

    return row


# ============================================================
# PARALLEL SIMULATION
# ============================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        (d, delay, phi, theta)
        for d in DV_VALUES
        for delay in DELAY_VALUES
        for phi in PHI_VALUES
        for theta in THETA_VALUES
    ]

    total_cases = len(cases)

    start_time = time.time()
    counts = {}

    print("=" * 65)
    print("FOUR-DIMENSIONAL PARALLEL RECOVERY COMPARISON")
    print("=" * 65)
    print(f"Total cases: {total_cases}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Test mode: {TEST_MODE}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 65)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=FIELDNAMES
        )
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

            futures = {
                executor.submit(run_case, case): case
                for case in cases
            }

            for case_number, future in enumerate(
                as_completed(futures), start=1
            ):

                case = futures[future]

                try:
                    row = future.result()
                except Exception as e:
                    dv_mag, delay_orb, phi_deg, theta_deg = case

                    row = {name: np.nan for name in FIELDNAMES}
                    row.update(
                        theta_deg=theta_deg,
                        phi_deg=phi_deg,
                        perturbation_dv=dv_mag,
                        delay_orbits=delay_orb,
                        status=f"worker_error: {e}",
                        assignment=""
                    )

                writer.writerow(row)
                csvfile.flush()

                status_key = row["status"].split(":")[0]
                counts[status_key] = counts.get(status_key, 0) + 1

                elapsed = time.time() - start_time
                remaining = (
                    elapsed / case_number
                    * (total_cases - case_number)
                )

                print(
                    f"[{case_number}/{total_cases}] "
                    f"theta={row['theta_deg']:3d}° "
                    f"phi={row['phi_deg']:3d}° "
                    f"dv={row['perturbation_dv']:.3f} m/s "
                    f"delay={row['delay_orbits']:.2f} T "
                    f"SCP={row['scp_total_dv']:.4f} m/s "
                    f"status={row['status']} "
                    f"remaining={remaining/60:.1f} min"
                )

    elapsed = time.time() - start_time

    print("=" * 65)
    print(f"Completed in {elapsed/60:.2f} min")
    print("Status counts:", counts)
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()