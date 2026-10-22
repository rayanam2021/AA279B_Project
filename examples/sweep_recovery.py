"""Trade study: recovery cost and trajectories vs. perturbation and delay.

Sweeps the perturbation velocity direction (radial / along-track / cross-track),
magnitude, and reaction delay; for each case solves both the two-impulse
baseline and the SCP re-slotting planner, then produces:

    results/sweep_trajectories.png      how the recovery trajectories change
    results/sweep_cost_vs_delay.png     total dv vs reaction delay
    results/sweep_cost_vs_magnitude.png total dv vs perturbation size
    results/sweep_results.csv           raw numbers

Run:  python examples/sweep_recovery.py [--quick]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    CircularOrbit,
    ScpConfig,
    along_track_formation,
    baseline_two_impulse_plan,
    plan_recovery,
)

# Validated categorical palette (first three slots, light mode) + chart chrome.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

DIRECTIONS = {
    "radial": np.array([1.0, 0.0, 0.0]),
    "along-track": np.array([0.0, 1.0, 0.0]),
    "cross-track": np.array([0.0, 0.0, 1.0]),
}


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE_AXIS)


def run_case(chief, direction, mag, delay_orb, tf_orb, cfg):
    """Solve one (direction, magnitude, delay) case; returns (base, scp)."""
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, mag * DIRECTIONS[direction])
    t_delay = delay_orb * chief.period
    tf = tf_orb * chief.period
    base = baseline_two_impulse_plan(chief, sats, "Sat-2", t_delay, tf, cfg)
    scp = plan_recovery(chief, sats, t_delay, tf, cfg, n_assignments=2)
    return base, scp


def plot_trajectories(cases, delays, out_path):
    """Grid of SCP recovery trajectories: rows = direction, cols = delay."""
    dirs = list(DIRECTIONS)
    fig, axes = plt.subplots(
        len(dirs), len(delays), figsize=(13, 9.5), facecolor=SURFACE,
        constrained_layout=True,
    )
    for r, direction in enumerate(dirs):
        # Cross-track perturbations live out-of-plane: plot y-z for that row.
        vert = 2 if direction == "cross-track" else 0
        vlabel = "cross-track z [m]" if vert == 2 else "radial x [m]"
        for c, delay in enumerate(delays):
            ax = axes[r, c]
            style_axis(ax)
            _, scp = cases[(direction, delay)]
            for i, (name, traj) in enumerate(scp.trajectories.items()):
                ax.plot(
                    traj[:, 1], traj[:, vert], color=SLOT[i], linewidth=1.6,
                )
                ax.plot(
                    traj[0, 1], traj[0, vert], "o", color=SLOT[i],
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
                )
                ax.plot(
                    traj[-1, 1], traj[-1, vert], "s", markersize=6,
                    markerfacecolor=SURFACE, markeredgecolor=SLOT[i],
                    markeredgewidth=1.6,
                )
                for t_burn, _ in scp.burns[name]:
                    k = min(np.searchsorted(scp.times, t_burn),
                            len(scp.times) - 1)
                    ax.plot(
                        traj[k, 1], traj[k, vert], "o", color=SLOT[i],
                        markersize=3.5,
                    )
            if r == 0:
                ax.set_title(
                    f"delay = {delay:g} orbit", fontsize=10, color=INK2
                )
            if c == 0:
                ax.set_ylabel(
                    f"{direction} pert.\n{vlabel}", fontsize=9, color=INK2
                )
            if r == len(dirs) - 1:
                ax.set_xlabel("along-track y [m]", fontsize=9, color=INK2)

    handles = [
        plt.Line2D([], [], color=SLOT[i], linewidth=1.6, label=f"Sat-{i + 1}")
        for i in range(3)
    ] + [
        plt.Line2D([], [], linestyle="", marker="o", color=INK2,
                   markersize=6, markeredgecolor=SURFACE, label="start"),
        plt.Line2D([], [], linestyle="", marker="s", markerfacecolor=SURFACE,
                   markeredgecolor=INK2, markersize=6, label="final slot"),
        plt.Line2D([], [], linestyle="", marker="o", color=INK2,
                   markersize=3.5, label="burn"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=6, frameon=False,
        fontsize=9, labelcolor=INK2,
    )
    fig.suptitle(
        "SCP recovery trajectories vs. perturbation direction and reaction "
        "delay  (|δv| = 0.1 m/s, LVLH frame)",
        fontsize=12, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cost_curves(x_vals, costs, x_label, title, out_path, reslotted):
    """Total dv vs a swept variable: color = direction, dash = method."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE,
                           constrained_layout=True)
    style_axis(ax)
    for i, direction in enumerate(DIRECTIONS):
        ax.plot(
            x_vals, costs[(direction, "scp")], color=SLOT[i], linewidth=1.8,
            marker="o", markersize=5, label=f"{direction} — SCP",
        )
        ax.plot(
            x_vals, costs[(direction, "baseline")], color=SLOT[i],
            linewidth=1.4, linestyle=(0, (4, 3)), marker="o", markersize=5,
            markerfacecolor=SURFACE, label=f"{direction} — baseline",
        )
        # Open ring marks SCP cases where the optimizer chose to re-slot.
        for x, y, swapped in zip(
            x_vals, costs[(direction, "scp")], reslotted[direction]
        ):
            if swapped:
                ax.plot(x, y, "o", markersize=9, markerfacecolor="none",
                        markeredgecolor=SLOT[i], markeredgewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=10, color=INK2)
    ax.set_ylabel("total constellation Δv [m/s]", fontsize=10, color=INK2)
    ax.set_title(
        title + "\nsolid = SCP (re-slotting allowed, ring = slot swap chosen), "
        "dashed = two-impulse baseline",
        fontsize=11, color=INK, loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="coarser grid for a fast smoke run")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    cfg = ScpConfig(dt=150.0, dv_max=0.5, fuel_max=5.0, d_min=200.0,
                    collision_margin=10.0, max_iters=6)
    # Recovery time: chosen so no (tf - delay) horizon is a half-period
    # multiple, where the two-impulse baseline's Phi_rv goes singular and its
    # delta-v diverges (horizons: 2.05, 1.90, 1.65, 1.15 orbits).
    tf_orb = 2.15

    if args.quick:
        delays = [0.1, 0.5]
        mags = [0.1]
    else:
        delays = [0.1, 0.25, 0.5, 1.0]
        mags = [0.05, 0.1, 0.2]
    mag_ref, delay_ref = 0.1, 0.25

    cache = {}

    def solve(direction, mag, delay):
        key = (direction, mag, delay)
        if key not in cache:
            print(f"  {direction:>12s}  |dv|={mag:.2f} m/s  "
                  f"delay={delay:g} orb ...", flush=True)
            cache[key] = run_case(chief, direction, mag, delay, tf_orb, cfg)
        return cache[key]

    print("Sweep 1/2: reaction delay (fixed |dv| = 0.1 m/s)")
    for d in DIRECTIONS:
        for delay in delays:
            solve(d, mag_ref, delay)
    print("Sweep 2/2: perturbation magnitude (fixed delay = 0.25 orbit)")
    for d in DIRECTIONS:
        for mag in mags:
            solve(d, mag, delay_ref)

    # ------------------------------------------------------------- figures
    traj_delays = (
        sorted({delays[0], delays[len(delays) // 2], delays[-1]})
        if len(delays) >= 3 else delays
    )
    traj_cases = {
        (d, dl): cache[(d, mag_ref, dl)] for d in DIRECTIONS
        for dl in traj_delays
    }
    plot_trajectories(
        traj_cases, traj_delays, outdir / "sweep_trajectories.png"
    )

    def collect(sweep_vals, fixed_mag, fixed_delay, vary):
        costs, rings = {}, {}
        for d in DIRECTIONS:
            b_list, s_list, ring = [], [], []
            for v in sweep_vals:
                mag = v if vary == "mag" else fixed_mag
                delay = v if vary == "delay" else fixed_delay
                base, scp = cache[(d, mag, delay)]
                b_list.append(base.total_dv)
                s_list.append(scp.total_dv)
                ring.append(scp.assignment != tuple(range(3)))
            costs[(d, "baseline")] = b_list
            costs[(d, "scp")] = s_list
            rings[d] = ring
        return costs, rings

    costs, rings = collect(delays, mag_ref, None, vary="delay")
    plot_cost_curves(
        delays, costs, "reaction delay [orbits]",
        f"Recovery cost vs. reaction delay  (|δv| = {mag_ref} m/s, "
        f"t_f = {tf_orb} orbits)",
        outdir / "sweep_cost_vs_delay.png", rings,
    )
    costs, rings = collect(mags, None, delay_ref, vary="mag")
    plot_cost_curves(
        mags, costs, "perturbation magnitude [m/s]",
        f"Recovery cost vs. perturbation size  (delay = {delay_ref} orbit, "
        f"t_f = {tf_orb} orbits)",
        outdir / "sweep_cost_vs_magnitude.png", rings,
    )

    # ----------------------------------------------------------------- CSV
    csv_path = outdir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "direction", "dv_mag_mps", "delay_orbits", "method",
            "total_dv_mps", "min_separation_m", "assignment", "n_burns",
        ])
        for (d, mag, delay), (base, scp) in sorted(cache.items()):
            for method, plan in (("baseline", base), ("scp", scp)):
                w.writerow([
                    d, mag, delay, method, f"{plan.total_dv:.5f}",
                    f"{plan.min_separation:.1f}", str(plan.assignment),
                    sum(len(b) for b in plan.burns.values()),
                ])
    print(f"\nWrote {outdir}/sweep_trajectories.png, sweep_cost_vs_delay.png,"
          f" sweep_cost_vs_magnitude.png, {csv_path.name}")


if __name__ == "__main__":
    main()
