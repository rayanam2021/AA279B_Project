from pathlib import Path
import csv
import sys
import numpy as np

# Allow imports from the project folder.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cwsim import CircularOrbit, along_track_formation
from cwsim.scp import ScpConfig, plan_recovery

RESULTS = PROJECT / "my_trade_studies" / "results"

# Pick a reproducible nonconverged case from the combined CSV.
with (RESULTS / "combined_sweep_results.csv").open(newline="") as handle:
    cases = [
        row for row in csv.DictReader(handle)
        if row.get("converged", "").strip().lower() == "false"
    ]

if not cases:
    raise SystemExit("No explicitly nonconverged cases found.")

cases.sort(key=lambda row: tuple(
    float(row[name]) for name in (
        "perturbation_dv", "delay_orbits", "phi_deg", "theta_deg"
    )
))
case = cases[0]

magnitude = float(case["perturbation_dv"])
delay = float(case["delay_orbits"])
theta_deg = float(case["theta_deg"])
phi_deg = float(case["phi_deg"])

print("CASE")
print(f"  Magnitude: {magnitude} m/s")
print(f"  Delay: {delay} orbits")
print(f"  Theta: {theta_deg} degrees")
print(f"  Phi: {phi_deg} degrees")
print(f"  Original iterations: {case.get('iterations')}")
print(f"  Original slack: {case.get('slack_max')} m")
print(f"  Original separation: {case.get('min_separation')} m")

summaries = []

for limit in (6, 30):
    print(f"\n{'=' * 60}")
    print(f"RERUN WITH MAX_ITERS = {limit}")
    print(f"{'=' * 60}")

    chief = CircularOrbit(
        altitude=550e3,
        inclination=np.radians(51.6),
    )

    theta, phi = np.radians([theta_deg, phi_deg])
    disturbance = magnitude * np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
    ])

    # Fresh satellites for each rerun.
    satellites = along_track_formation(spacing=1000., count=3)
    satellites[1].add_impulse(0., disturbance)

    cfg = ScpConfig(
        dt=150.,
        dv_max=0.5,
        fuel_max=5.,
        d_min=200.,
        collision_margin=10.,
        max_iters=limit,
    )

    plan = plan_recovery(
        chief,
        satellites,
        delay * chief.period,
        2.15 * chief.period,
        cfg,
        verbose=True,
    )

    print("\nSELECTED PLAN:")
    print(plan.summary())
    print(f"Maximum slack: {plan.slack_max:.8g} m")

    summaries.append({
        "max_iters": limit,
        "assignment": str(plan.assignment),
        "converged": plan.converged,
        "iterations": plan.iterations,
        "total_dv": plan.total_dv,
        "slack_max": plan.slack_max,
        "min_separation": plan.min_separation,
    })

print("\nCOMPARISON")
for row in summaries:
    print(row)