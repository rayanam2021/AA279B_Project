"""Demo: SCP formation recovery vs. the closed-form two-impulse baseline.

Scenario: 3-satellite along-track formation (1 km spacing) at 550 km. Sat-2
is perturbed at t = 0; after a reaction delay the constellation must restore
an equally spaced formation by tf — in ANY slot order — while respecting a
keep-out sphere, per-burn thrust limits, and per-satellite fuel budgets.

Run:  python examples/demo_scp_recovery.py [--screenshot out.png] [--gif out.gif]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    CircularOrbit,
    FormationViewer,
    ScpConfig,
    Simulation,
    along_track_formation,
    apply_plan,
    baseline_two_impulse_plan,
    plan_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", help="save a static screenshot")
    parser.add_argument("--gif", help="save an animation GIF")
    parser.add_argument(
        "--baseline", action="store_true",
        help="visualize the two-impulse baseline instead of the SCP plan",
    )
    args = parser.parse_args()

    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, np.array([0.05, 0.10, 0.03]))  # the anomaly

    t_delay = 0.25 * chief.period
    tf_candidates = [1.2 * chief.period, 1.7 * chief.period, 2.2 * chief.period]
    cfg = ScpConfig(
        dt=90.0, dv_max=0.5, fuel_max=5.0, d_min=200.0,
        w_tf=2e-5,  # m/s per s: ~0.11 m/s per orbit of recovery time
    )

    print("Baseline (two-impulse, perturbed sat back to its own slot):")
    base = baseline_two_impulse_plan(
        chief, sats, "Sat-2", t_delay, tf_candidates[1], cfg
    )
    print(f"  {base.summary()}")

    print("SCP (all sats, re-slotting allowed, collision keep-out "
          f"{cfg.d_min:.0f} m):")
    plan = plan_recovery(
        chief, sats, t_delay, tf_candidates, cfg, verbose=True
    )
    print(f"  BEST: {plan.summary()}")
    saving = base.total_dv - plan.total_dv
    print(
        f"\n  SCP saves {saving:.4f} m/s "
        f"({100 * saving / base.total_dv:.1f}% of baseline), "
        f"baseline min sep = {base.min_separation:.1f} m"
    )

    chosen = base if args.baseline else plan
    apply_plan(sats, chosen)
    sim = Simulation(chief_orbit=chief, satellites=sats)
    result = sim.run(t_final=chosen.tf + 0.3 * chief.period, dt=15.0)

    viewer = FormationViewer(result, off_screen=bool(args.screenshot or args.gif))
    if args.screenshot:
        viewer.screenshot(args.screenshot, frame=len(result.times) // 2)
        print(f"Saved {args.screenshot}")
    elif args.gif:
        viewer.save_gif(args.gif, step=6)
        print(f"Saved {args.gif}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
