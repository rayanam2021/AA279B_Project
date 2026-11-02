"""Trade space: two-impulse CW recovery vs. reaction delay and transfer time.

Scenario
    3-satellite along-track formation, 1 km apart, 550 km equatorial orbit.
    Sat-1 (the trailing satellite) is hit by an unplanned impulse at t = 0.
    After a reaction delay it flies a closed-form two-impulse CW transfer back
    to its original slot; the other two satellites hold their slots.
    Keep-out zone d_min = 100 m, per-burn limit 0.5 m/s.

For every (delay, transfer time) pair the total delta-v |dv1| + |dv2| is
computed, then two kinds of region are blocked out:
    * keep-out violation: Sat-1 comes within d_min of another satellite at any
      time in [0, tf] (the uncontrolled drift is included)
    * burn limit exceeded: either impulse is larger than dv_max (this also
      covers the half-orbit transfer times where the CW solution goes singular)

Output (one panel per perturbation direction):
    results/sweep_two_impulse.png
    results/sweep_two_impulse.npz      raw grids

Run:  python examples/sweep.py [--quick] [--mag 0.1]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import CircularOrbit, along_track_formation, cw_stm, cw_two_impulse
from cwsim.scp import ScpConfig

# Sequential blue ramp (steps 100 -> 700) + chart chrome.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
BLOCK_LIMIT = "#e1e0d9"   # burn limit exceeded
BLOCK_KOZ = "#6b6a64"     # keep-out violation
DRIFT = "#eb6834"         # uncontrolled drift leg
TRANSFER = "#2a78d6"      # two-impulse transfer leg

DIRECTIONS = {
    "radial": np.array([1.0, 0.0, 0.0]),
    "along-track": np.array([0.0, 1.0, 0.0]),
    "cross-track": np.array([0.0, 0.0, 1.0]),
}

config = ScpConfig()
SPACING = 1000.0     # formation spacing [m]
D_MIN = config.d_min        # keep-out radius [m]
DV_MAX = config.dv_max         # per-burn limit [m/s]
CHECK_DT = config.check_dt      # separation-check resolution [s]


def cw_positions(x0, n, t):
    """Unforced CW positions (len(t), 3) from state x0, vectorized over t."""
    s, c = np.sin(n * t), np.cos(n * t)
    x, y, z, vx, vy, vz = x0
    return np.column_stack([
        (4.0 - 3.0 * c) * x + s / n * vx + 2.0 * (1.0 - c) / n * vy,
        6.0 * (s - n * t) * x + y - 2.0 * (1.0 - c) / n * vx
        + (4.0 * s - 3.0 * n * t) / n * vy,
        c * z + s / n * vz,
    ])


def min_separation(r, others):
    """Smallest distance from trajectory r (M, 3) to any fixed point."""
    return min(np.linalg.norm(r - p, axis=1).min() for p in others)


def sweep_direction(chief, direction, mag, delays, transfers,
                    directions=DIRECTIONS):
    """Total dv, max single burn, and min separation over the grid.

    Returns three (len(transfers), len(delays)) arrays.
    """
    n = chief.mean_motion
    period = chief.period
    sats = along_track_formation(spacing=SPACING, count=3)
    target = sats[0].x0
    # The unperturbed satellites sit at CW equilibria, so they never move.
    others = [s.x0[:3] for s in sats[1:]]

    x_hit = target.copy()
    x_hit[3:] += mag * directions[direction]

    shape = (len(transfers), len(delays))
    total_dv = np.full(shape, np.nan)
    max_burn = np.full(shape, np.nan)
    min_sep = np.full(shape, np.nan)

    for j, delay_orb in enumerate(delays):
        t_delay = delay_orb * period
        # Uncontrolled drift from the hit to the first burn.
        t_drift = np.linspace(0.0, t_delay,
                              max(int(np.ceil(t_delay / CHECK_DT)) + 1, 2))
        drift_sep = min_separation(cw_positions(x_hit, n, t_drift), others)
        x_delay = cw_stm(n, t_delay) @ x_hit

        for i, transfer_orb in enumerate(transfers):
            horizon = transfer_orb * period
            try:
                dv1, dv2 = cw_two_impulse(x_delay, n, horizon, target)
            except np.linalg.LinAlgError:
                continue
            total_dv[i, j] = np.linalg.norm(dv1) + np.linalg.norm(dv2)
            max_burn[i, j] = max(np.linalg.norm(dv1), np.linalg.norm(dv2))

            x_burn = x_delay.copy()
            x_burn[3:] += dv1
            tau = np.linspace(0.0, horizon,
                              int(np.ceil(horizon / CHECK_DT)) + 1)
            transfer_sep = min_separation(cw_positions(x_burn, n, tau), others)
            min_sep[i, j] = min(drift_sep, transfer_sep)
    return total_dv, max_burn, min_sep


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE_AXIS)


def plot_trade_space(results, delays, transfers, mag, out_path,
                     title="Two-impulse CW recovery of Sat-1",
                     limit_label=f"burn > {DV_MAX:g} m/s",
                     cbar_label="total Δv  |Δv₁| + |Δv₂|  [m/s]",
                     reordered=None):
    """Total-dv heatmap per direction with blocked regions.

    `reordered` optionally maps direction -> boolean grid; those cells get a
    small dot (used by sweep_scp.py to show where SCP swapped slots).
    """
    dirs = list(results)
    fig, axes = plt.subplots(
        1, len(dirs), figsize=(5.2 * len(dirs), 5.4), facecolor=SURFACE,
        constrained_layout=True, squeeze=False,
    )
    axes = axes[0]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    feasible_dv = np.concatenate([
        r["total_dv"][r["feasible"]] for r in results.values()
    ])
    vmin = max(np.nanmin(feasible_dv), 1e-3) if feasible_dv.size else 1e-2
    norm = Normalize(vmin=0.0, vmax=2 * DV_MAX)

    block_cmap = ListedColormap([BLOCK_LIMIT, BLOCK_KOZ])
    for ax, direction in zip(axes, dirs):
        r = results[direction]
        style_axis(ax)
        D, T = np.meshgrid(delays, transfers)

        # Blocked cells: keep-out takes priority over the burn limit.
        blocked = np.where(r["koz"], 2, np.where(r["limit"], 1, 0))
        ax.pcolormesh(D, T, np.ma.masked_equal(blocked, 0), cmap=block_cmap,
                      vmin=1, vmax=2, shading="auto")
        dv = np.ma.masked_where(~r["feasible"], r["total_dv"])
        ax.pcolormesh(D, T, dv, cmap=cmap, norm=norm, shading="auto")

        levels = [lv for lv in (0.05, 0.1, 0.2, 0.5) if lv > vmin]
        if levels and dv.count():
            cs = ax.contour(D, T, dv.filled(np.nan), levels=levels,
                            colors=SURFACE, linewidths=0.8)
            ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7, colors=SURFACE)

        if reordered is not None and reordered[direction].any():
            jj, ii = np.nonzero(reordered[direction].T)
            ax.plot(delays[jj], transfers[ii], ".", color=DRIFT,
                    markersize=3.5, zorder=4)

        # Cheapest feasible recovery.
        if dv.count():
            i, j = np.unravel_index(np.ma.argmin(dv), dv.shape)
            ax.plot(delays[j], transfers[i], "o", markersize=8,
                    markerfacecolor=SURFACE, markeredgecolor=INK,
                    markeredgewidth=1.4, clip_on=False, zorder=5)
            ax.annotate(
                f"min {dv[i, j]:.3f} m/s", (delays[j], transfers[i]),
                xytext=(8, 6), textcoords="offset points", fontsize=8,
                color=INK,
            )

        ax.set_title(f"{direction} hit", fontsize=10, color=INK2, loc="left")
        ax.set_xlabel("reaction delay [orbits]", fontsize=9, color=INK2)
        ax.set_ylabel("transfer time, delay → slot [orbits]", fontsize=9,
                      color=INK2)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        shrink=0.85, pad=0.01)
    cbar.set_label(cbar_label, fontsize=9, color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE_AXIS)

    handles = [
        Patch(facecolor=BLOCK_KOZ,
              label=f"keep-out violated (< {D_MIN:g} m)"),
        Patch(facecolor=BLOCK_LIMIT, edgecolor=BASELINE_AXIS,
              label=limit_label),
        Line2D([], [], linestyle="", marker="o", markersize=8,
               markerfacecolor=SURFACE, markeredgecolor=INK,
               label="cheapest feasible"),
    ]
    if reordered is not None:
        handles.append(Line2D([], [], linestyle="", marker=".",
                              color=DRIFT, markersize=7,
                              label="slots reordered"))
    fig.legend(
        handles=handles, loc="outside lower center", ncol=len(handles),
        frameon=False, fontsize=9, labelcolor=INK2,
    )
    fig.suptitle(
        f"{title}  (|δv| = {mag:g} m/s, "
        f"{SPACING / 1000:g} km spacing, 550 km equatorial)",
        fontsize=12, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_case(chief, direction, mag, delay_orb, transfer_orb, out_path):
    """CW-frame trajectory of one recovery: along-track y vs radial x."""
    n, period = chief.mean_motion, chief.period
    sats = along_track_formation(spacing=SPACING, count=3)
    target = sats[0].x0
    others = {s.name: s.x0[:3] for s in sats[1:]}

    x_hit = target.copy()
    x_hit[3:] += mag * DIRECTIONS[direction]
    t_delay = delay_orb * period
    horizon = transfer_orb * period

    t_drift = np.linspace(0.0, t_delay,
                          max(int(np.ceil(t_delay / CHECK_DT)) + 1, 2))
    r_drift = cw_positions(x_hit, n, t_drift)
    x_delay = cw_stm(n, t_delay) @ x_hit
    dv1, dv2 = cw_two_impulse(x_delay, n, horizon, target)
    x_burn = x_delay.copy()
    x_burn[3:] += dv1
    tau = np.linspace(0.0, horizon, int(np.ceil(horizon / CHECK_DT)) + 1)
    r_transfer = cw_positions(x_burn, n, tau)
    sep = min(min_separation(r_drift, others.values()),
              min_separation(r_transfer, others.values()))

    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE,
                           constrained_layout=True)
    style_axis(ax)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    # Keep-out zones around the satellites that hold their slots.
    for name, p in others.items():
        ax.add_patch(Circle((p[1], p[0]), D_MIN, facecolor=GRID,
                                edgecolor=MUTED, linestyle=(0, (4, 3)),
                                linewidth=1.0, zorder=1))
        ax.plot(p[1], p[0], "s", color=INK2, markersize=7, zorder=3)
        ax.annotate(name, (p[1], p[0]), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK2)

    ax.plot(r_drift[:, 1], r_drift[:, 0], color=DRIFT, linewidth=2,
            linestyle=(0, (4, 2)), zorder=2,
            label=f"drift, {delay_orb:.2f} orbit")
    ax.plot(r_transfer[:, 1], r_transfer[:, 0], color=TRANSFER, linewidth=2,
            zorder=2, label=f"transfer, {transfer_orb:.2f} orbit")

    # Slot / hit point, then both burns.
    ax.plot(target[1], target[0], "o", markersize=9, markerfacecolor=SURFACE,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=4)
    ax.annotate(
        f"Sat-1 slot: hit at t = 0,\nΔv₂ = {np.linalg.norm(dv2):.3f} m/s "
        "on arrival", (target[1], target[0]), xytext=(-12, 8),
        textcoords="offset points", ha="right", fontsize=8, color=INK,
    )
    ax.plot(x_delay[1], x_delay[0], "o", color=TRANSFER, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    ax.annotate(
        f"Δv₁ = {np.linalg.norm(dv1):.3f} m/s", (x_delay[1], x_delay[0]),
        xytext=(0, -16), textcoords="offset points", ha="center",
        fontsize=8, color=INK,
    )

    ax.set_aspect("equal")
    ax.set_ylim(-1.5 * D_MIN, 1.5 * D_MIN)
    ax.invert_xaxis()  # +y to the left (radial up, velocity direction left)
    ax.set_xlabel("along-track y [m]  (+y ←)", fontsize=9, color=INK2)
    ax.set_ylabel("radial x [m]", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower center",
              bbox_to_anchor=(0.62, 0.0))
    ax.set_title(
        f"Cheapest feasible two-impulse recovery, {direction} hit "
        f"(|δv| = {mag:g} m/s)\n"
        f"Δv₁ = {np.linalg.norm(dv1):.3f}, Δv₂ = {np.linalg.norm(dv2):.3f}, "
        f"total {np.linalg.norm(dv1) + np.linalg.norm(dv2):.3f} m/s  ·  "
        f"min separation {sep:.0f} m (keep-out {D_MIN:g} m)",
        fontsize=10, color=INK, loc="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="coarser grid for a fast smoke run")
    parser.add_argument("--mag", type=float, default=0.1,
                        help="perturbation magnitude [m/s]")
    parser.add_argument("--max-delay", type=float, default=1.5,
                        help="largest reaction delay [orbits]")
    parser.add_argument("--max-transfer", type=float, default=2.5,
                        help="largest transfer time [orbits]")
    parser.add_argument("--case-direction", default="radial",
                        choices=list(DIRECTIONS),
                        help="direction whose cheapest case is plotted")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    chief = CircularOrbit(altitude=550e3, inclination=0.0)
    n_d, n_t = (40, 50) if args.quick else (150, 200)
    delays = np.linspace(0.0, args.max_delay, n_d)
    transfers = np.linspace(0.05, args.max_transfer, n_t)

    results = {}
    for direction in DIRECTIONS:
        print(f"  {direction:>12s}  {n_d}x{n_t} grid ...", flush=True)
        total_dv, max_burn, min_sep = sweep_direction(
            chief, direction, args.mag, delays, transfers
        )
        limit = ~(max_burn <= DV_MAX)          # NaN (singular) counts as over
        koz = min_sep < D_MIN
        results[direction] = dict(
            total_dv=total_dv, max_burn=max_burn, min_sep=min_sep,
            limit=limit, koz=koz, feasible=~limit & ~koz,
        )

    plot_trade_space(results, delays, transfers, args.mag,
                     outdir / "sweep_two_impulse.png")
    np.savez(
        outdir / "sweep_two_impulse.npz", delays_orb=delays,
        transfers_orb=transfers, mag=args.mag,
        **{f"{d}_{k}": v for d, r in results.items() for k, v in r.items()},
    )

    r = results[args.case_direction]
    dv = np.ma.masked_where(~r["feasible"], r["total_dv"])
    i, j = np.unravel_index(np.ma.argmin(dv), dv.shape)
    case_path = outdir / f"sweep_two_impulse_best_{args.case_direction}.png"
    plot_case(chief, args.case_direction, args.mag, delays[j], transfers[i],
              case_path)
    print(f"\nWrote {outdir}/sweep_two_impulse.png, sweep_two_impulse.npz,"
          f" {case_path.name}")


if __name__ == "__main__":
    main()
