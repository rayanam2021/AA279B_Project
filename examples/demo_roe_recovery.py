"""Demo: Chernick & D'Amico closed-form ROE recovery vs. two-impulse CW.

Scenario (same as examples/demo_formation.py): 3-satellite along-track
formation (1 km spacing) at 550 km / 51.6 deg. Sat-2 takes an unplanned
delta-v at t = 0. After a reaction delay it returns to its original slot by
tf, using either
  * the two-impulse CW transfer (burns fixed at t_delay and tf), or
  * the closed-form ROE scheme of Chernick & D'Amico (tangential in-plane
    burns at optimal locations plus one cross-track burn).

The script prints a cost table over several transfer times. It then shows
(or saves) the ROE plan for the chosen transfer time in the 3D viewer.

Run:  python examples/demo_roe_recovery.py [--orbits 2.0] [--screenshot out.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    CircularOrbit,
    ClosedFormUnavailable,
    FormationViewer,
    ScpConfig,
    Simulation,
    along_track_formation,
    apply_plan,
    baseline_roe_plan,
    baseline_two_impulse_plan,
    propagate_cw_with_impulses,
    roe_impulsive_reconfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbits", type=float, default=2.0,
                        help="transfer time tf - t_delay to visualize [orbits]")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="reaction delay [orbits]")
    parser.add_argument("--screenshot", help="save a static screenshot")
    parser.add_argument("--gif", help="save an animation GIF")
    parser.add_argument("--no-view", action="store_true",
                        help="print the comparison only")
    args = parser.parse_args()

    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    n, period = chief.mean_motion, chief.period
    cfg = ScpConfig()

    def formation():
        sats = along_track_formation(spacing=1000.0, count=3)
        sats[1].add_impulse(0.0, np.array([0.05, 0.10, 0.03]))
        return sats

    t_delay = args.delay * period
    print(f"Chief: 550 km, period {period / 60:.1f} min; reaction delay "
          f"{args.delay:.2f} orbits\n")
    print(f"{'transfer':>9} | {'two-impulse':>12} | {'ROE closed form':>15} | "
          f"{'dv_min':>7} | {'dominant':>8} | burns | min sep [m]")
    print("-" * 82)
    for orbits in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
        tf = t_delay + orbits * period
        sats = formation()
        two = baseline_two_impulse_plan(chief, sats, "Sat-2", t_delay, tf, cfg)
        two_txt = f"{two.total_dv:12.4f}" if two.total_dv < 100 else f"{'singular':>12}"
        try:
            roe = baseline_roe_plan(chief, sats, "Sat-2", t_delay, tf, cfg)
        except ClosedFormUnavailable:
            print(f"{orbits:7.2f} o | {two_txt} | {'window too short':>15} |")
            continue
        # Diagnostics (dominance, reachable minimum) from the raw solver.
        x_delay = propagate_cw_with_impulses(
            sats[1].x0, n, np.array([t_delay]), sats[1].impulses)[0]
        raw = roe_impulsive_reconfig(x_delay, n, tf - t_delay, sats[1].x0)
        print(f"{orbits:7.2f} o | {two_txt} | {roe.total_dv:15.4f} | "
              f"{raw.dv_lower_bound:7.4f} | {raw.dominant:>8} | "
              f"{len(roe.burns['Sat-2']):5d} | {roe.min_separation:8.1f}")

    if args.no_view:
        return

    tf = t_delay + args.orbits * period
    sats = formation()
    plan = baseline_roe_plan(chief, sats, "Sat-2", t_delay, tf, cfg)
    print(f"\nVisualizing ROE plan, transfer {args.orbits:.2f} orbits:")
    for t, dv in plan.burns["Sat-2"]:
        print(f"  t = {t / 60:7.1f} min  dv = {np.round(dv, 5)} m/s")
    apply_plan(sats, plan)
    result = Simulation(chief_orbit=chief, satellites=sats).run(
        t_final=tf + 0.3 * period, dt=15.0)
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
