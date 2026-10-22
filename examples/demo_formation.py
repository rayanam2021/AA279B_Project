"""Demo: 3-satellite along-track formation with one perturbed satellite.

Scenario (mirrors the project setup):
  - Chief reference: circular LEO at 550 km altitude, 51.6 deg inclination.
  - Three satellites equally spaced along-track (+/- 1 km spacing).
  - At t = 0, Sat-2 (the middle satellite) suffers an unexpected delta-v,
    producing secular along-track drift through its neighbors' slots.
  - After a reaction delay, a closed-form two-impulse CW maneuver returns
    Sat-2 to its original slot by the recovery time t_f.

Run:  python examples/demo_formation.py [--gif out.gif]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    CircularOrbit,
    FormationViewer,
    Simulation,
    along_track_formation,
    cw_two_impulse,
    propagate_cw_with_impulses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", help="save animation to this GIF instead of showing")
    parser.add_argument(
        "--screenshot", help="save a static screenshot instead of showing"
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="show the unforced drift only (no recovery burns)",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------- scenario setup
    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    n = chief.mean_motion
    print(f"Chief orbit: 550 km, period = {chief.period / 60:.1f} min")

    sats = along_track_formation(spacing=1000.0, count=3)
    sat2 = sats[1]  # middle satellite, nominally at the origin slot

    # Unexpected perturbation at t = 0: mostly along-track, some radial/cross.
    dv_perturb = np.array([0.05, 0.10, 0.03])  # m/s in LVLH axes
    sat2.add_impulse(0.0, dv_perturb)
    print(f"Perturbation on {sat2.name}: dv = {dv_perturb} m/s")

    # --------------------------------------------- two-impulse recovery maneuver
    t_delay = 0.5 * chief.period   # reaction delay before the first burn
    t_f = 2.0 * chief.period       # time of the second (arrival) burn
    if not args.no_recovery:
        # State of Sat-2 at the first burn time (drifting, unforced).
        x_at_delay = propagate_cw_with_impulses(
            sat2.x0, n, np.array([t_delay]), sat2.impulses
        )[0]
        # Two-impulse transfer back to the original slot, arriving at t_f.
        dv1, dv2 = cw_two_impulse(x_at_delay, n, t_f - t_delay, x_target=sat2.x0)
        sat2.add_impulse(t_delay, dv1)
        sat2.add_impulse(t_f, dv2)
        total = np.linalg.norm(dv1) + np.linalg.norm(dv2)
        print(
            f"Recovery: burn 1 at {t_delay / 60:.1f} min |dv1| = "
            f"{np.linalg.norm(dv1):.4f} m/s, burn 2 at {t_f / 60:.1f} min "
            f"|dv2| = {np.linalg.norm(dv2):.4f} m/s, total = {total:.4f} m/s"
        )

    # ------------------------------------------------------------------ simulate
    sim = Simulation(chief_orbit=chief, satellites=sats)
    result = sim.run(t_final=3.0 * chief.period, dt=15.0)

    # ------------------------------------------------------------------ visualize
    viewer = FormationViewer(result, off_screen=bool(args.gif or args.screenshot))
    if args.gif:
        viewer.save_gif(args.gif, step=4)
        print(f"Saved {args.gif}")
    elif args.screenshot:
        viewer.screenshot(args.screenshot, frame=len(result.times) // 3)
        print(f"Saved {args.screenshot}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
