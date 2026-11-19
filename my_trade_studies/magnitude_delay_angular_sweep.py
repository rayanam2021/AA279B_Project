import csv
import time
from pathlib import Path

import numpy as np

from cwsim import CircularOrbit, along_track_formation
from cwsim.scp import (
    ScpConfig,
    baseline_two_impulse_plan,
    plan_recovery,
)


# ============================================================
# USER SETTINGS
# ============================================================

# Set True for a small test before running the full sweep
TEST_MODE = False

# Full angular sweep
THETA_VALUES = np.arange(0, 360, 10)
PHI_VALUES = np.arange(-90, 91, 10)

# Perturbation magnitudes [m/s]
DV_VALUES = [0.05, 0.10, 0.15, 0.20]

# Reaction delays [orbital periods]
DELAY_VALUES = [0.10, 0.25, 0.50, 0.75, 1.00]

# Recovery deadline [orbital periods]
TF_ORBITS = 2.15

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "magnitude_delay_angular_results_retry.csv"


# ============================================================
# TEST MODE
# ============================================================

if TEST_MODE:

    THETA_VALUES = [0, 90, 180, 270]
    PHI_VALUES = [-45, 0, 45]

    DV_VALUES = [0.05, 0.20]
    DELAY_VALUES = [0.25, 0.75]


# ============================================================
# SPACECRAFT AND OPTIMIZER CONFIGURATION
# ============================================================

chief = CircularOrbit(
    altitude=550e3,
    inclination=np.radians(51.6),
)

cfg = ScpConfig(
    dt=150.0,
    dv_max=0.5,
    fuel_max=5.0,
    d_min=200.0,
    collision_margin=10.0,
    max_iters=6,
)

tf = TF_ORBITS * chief.period


# ============================================================
# CSV CONFIGURATION
# ============================================================

fieldnames = [
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
    "assignment",
    "status",
]


# ============================================================
# SWEEP
# ============================================================

total_cases = (
    len(THETA_VALUES)
    * len(PHI_VALUES)
    * len(DV_VALUES)
    * len(DELAY_VALUES)
)

case_number = 0
start_time = time.time()

print("=" * 65)
print("FOUR-DIMENSIONAL FORMATION RECOVERY TRADE STUDY")
print("=" * 65)
print(f"Total cases: {total_cases}")
print(f"Test mode: {TEST_MODE}")
print(f"Output: {OUTPUT_CSV}")
print("=" * 65)


with open(OUTPUT_CSV, "w", newline="") as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for dv_mag in DV_VALUES:

        for delay_orb in DELAY_VALUES:

            t_delay = delay_orb * chief.period

            for phi_deg in PHI_VALUES:

                for theta_deg in THETA_VALUES:

                    case_number += 1

                    theta = np.radians(theta_deg)
                    phi = np.radians(phi_deg)

                    # ----------------------------------------
                    # Perturbation vector in RIC coordinates
                    # ----------------------------------------

                    perturbation = dv_mag * np.array([
                        np.cos(phi) * np.cos(theta),
                        np.cos(phi) * np.sin(theta),
                        np.sin(phi),
                    ])

                    # ----------------------------------------
                    # Initialize formation
                    # ----------------------------------------

                    sats = along_track_formation(
                        spacing=1000.0,
                        count=3,
                    )

                    # Apply unexpected impulse to Sat-2
                    sats[1].add_impulse(
                        0.0,
                        perturbation,
                    )

                    # ----------------------------------------
                    # Initialize results
                    # ----------------------------------------

                    scp_dv = np.nan
                    baseline_dv = np.nan
                    fuel_savings = np.nan
                    min_separation = np.nan
                    assignment = ""
                    status = "success"

                    # ----------------------------------------
                    # Baseline recovery
                    # ----------------------------------------

                    try:

                        baseline = baseline_two_impulse_plan(
                            chief,
                            sats,
                            "Sat-2",
                            t_delay,
                            tf,
                            cfg,
                        )

                        baseline_dv = baseline.total_dv

                    except Exception as e:

                        status = f"baseline_error: {e}"

                    # ----------------------------------------
                    # SCP recovery
                    # ----------------------------------------

                    try:

                        scp = plan_recovery(
                            chief,
                            sats,
                            t_delay,
                            tf,
                            cfg,
                        )

                        scp_dv = scp.total_dv
                        min_separation = scp.min_separation
                        assignment = str(scp.assignment)

                    except Exception as e:

                        if status == "success":
                            status = f"scp_error: {e}"
                        else:
                            status += f"; scp_error: {e}"

                    # ----------------------------------------
                    # Fuel savings
                    # ----------------------------------------

                    if (
                        np.isfinite(scp_dv)
                        and np.isfinite(baseline_dv)
                        and baseline_dv > 0
                    ):

                        fuel_savings = (
                            100.0
                            * (1.0 - scp_dv / baseline_dv)
                        )

                    # ----------------------------------------
                    # Save results
                    # ----------------------------------------

                    writer.writerow({
                        "theta_deg": theta_deg,
                        "phi_deg": phi_deg,
                        "perturbation_dv": dv_mag,
                        "delay_orbits": delay_orb,
                        "delay_seconds": t_delay,
                        "dv_R": perturbation[0],
                        "dv_I": perturbation[1],
                        "dv_C": perturbation[2],
                        "scp_total_dv": scp_dv,
                        "baseline_total_dv": baseline_dv,
                        "fuel_savings_percent": fuel_savings,
                        "min_separation": min_separation,
                        "assignment": assignment,
                        "status": status,
                    })

                    # Save incrementally
                    csvfile.flush()

                    # ----------------------------------------
                    # Progress display
                    # ----------------------------------------

                    elapsed = time.time() - start_time

                    avg_time = elapsed / case_number

                    remaining = (
                        total_cases - case_number
                    ) * avg_time

                    print(
                        f"[{case_number}/{total_cases}] "
                        f"theta={theta_deg:3d}° "
                        f"phi={phi_deg:3d}° "
                        f"dv={dv_mag:.3f} m/s "
                        f"delay={delay_orb:.2f} T "
                        f"SCP={scp_dv:.4f} m/s "
                        f"remaining={remaining / 60:.1f} min"
                    )


# ============================================================
# COMPLETION
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
print(f"Cases completed: {case_number}")
print(f"Elapsed time: {elapsed / 60:.2f} minutes")
print(f"Results saved to: {OUTPUT_CSV}")