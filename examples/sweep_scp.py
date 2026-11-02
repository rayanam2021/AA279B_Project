"""Trade space: SCP vs. two-impulse recovery, and where reordering slots pays.

Scenario (same as sweep.py)
    3-satellite along-track formation, 1 km apart, 550 km equatorial orbit.
    Sat-1 (the trailing satellite) is hit by an unplanned impulse at t = 0 and
    nothing burns until the reaction delay has passed. Keep-out d_min and the
    per-burn limit come from ScpConfig.

For every (delay, transfer time) pair three recoveries are compared:
    two-impulse   closed-form CW transfer of Sat-1 back to its own slot
    SCP, fixed    SCP planner, every satellite held to its own slot
    SCP           SCP planner with slot reordering allowed (best of the
                  fixed-order plan and the top-ranked reorderings)

Is reordering really cheaper?  CW dynamics are linear and the unhit
satellites start at rest in their slots, so a reordered plan can always be
folded back into a fixed-order one: Sat-1 flies the summed burns of every
satellite in its slot cycle and ends in its own slot while the rest hold. By
the triangle inequality that never costs more, so reordering can only win on
total delta-v when the folded plan breaks a constraint (it hits a neighbour's
keep-out zone or exceeds the per-burn limit). The folded plan is therefore
included as a fixed-order option, which also guards against the fixed-order
SCP stalling in a poor local optimum (its ballistic first guess can leave
Sat-1 trapped on the far side of a neighbour).

On top of the sweep.py directions a retrograde (-along-track) hit is swept: it
sends Sat-1 forward through Sat-2 and Sat-3, which is where handing out the
slots in a new order beats flying Sat-1 back.

Output
    results/sweep_scp_two_impulse.png   two-impulse trade space (fine grid)
    results/sweep_scp.png               SCP trade space, dots = slots reordered
    results/sweep_scp_reorder.png       delta-v saved by reordering vs. the
                                        best feasible fixed-order plan (SCP,
                                        two-impulse or folded reorder plan)
    results/sweep_scp_best_reorder.png  trajectories of the largest saving
    results/sweep_scp.npz               raw grids (re-plot with --replot)

An SCP solve takes ~1-4 s, so its grid is coarser than the two-impulse one and
cells are solved in parallel.

Run:  python examples/sweep_scp.py [--quick] [--mag 0.1] [--workers 8]
"""

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cwsim import (
    CircularOrbit,
    ScpConfig,
    along_track_formation,
    baseline_two_impulse_plan,
    plan_recovery,
)
from cwsim.scp import _score_burns  # re-scores the folded fixed-order plan
from sweep import (
    BASELINE_AXIS,
    BLOCK_KOZ,
    BLOCK_LIMIT,
    D_MIN,
    DRIFT,
    DV_MAX,
    GRID,
    INK,
    INK2,
    MUTED,
    SEQ_BLUE,
    SPACING,
    SURFACE,
    plot_trade_space,
    style_axis,
    sweep_direction,
)

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]  # Sat-1/2/3: blue, orange, aqua

DIRECTIONS = {
    "radial": np.array([1.0, 0.0, 0.0]),
    "+along-track": np.array([0.0, 1.0, 0.0]),
    "-along-track": np.array([0.0, -1.0, 0.0]),
    "cross-track": np.array([0.0, 0.0, 1.0]),
}

# Coarser burn grid than the ScpConfig default keeps a solve to a few seconds;
# the margin pads d_min for the excursion between keep-out samples.
SCP_CONFIG = ScpConfig(dt=300.0, n_sub=4, max_iters=6, collision_margin=5.0)
N_ASSIGNMENTS = 3   # ranked slot permutations the SCP solves exactly
PERMS = list(itertools.permutations(range(3)))
IDENTITY = PERMS.index((0, 1, 2))


def make_chief():
    return CircularOrbit(altitude=550e3, inclination=0.0)


def hit_formation(direction, mag):
    sats = along_track_formation(spacing=SPACING, count=3)
    sats[0].add_impulse(0.0, mag * DIRECTIONS[direction])
    return sats


def fold_to_fixed_order(plan, sats):
    """Burns of a fixed-order plan equivalent to a reordered one.

    Sat-1 flies the summed burns of every satellite in its slot cycle (burns
    share the planner's time grid); everyone else holds their slot.
    """
    cycle, i = [0], plan.assignment[0]
    while i != 0:
        cycle.append(i)
        i = plan.assignment[i]
    summed = {}
    for i in cycle:
        for t, dv in plan.burns[sats[i].name]:
            summed[t] = summed.get(t, 0.0) + dv
    burns = {s.name: [] for s in sats}
    burns[sats[0].name] = sorted(summed.items(), key=lambda b: b[0])
    return burns


def folded_plan(chief, sats, plan, cfg):
    """Fold a reordered plan back to fixed order and score it."""
    return _score_burns(
        chief, sats, fold_to_fixed_order(plan, sats), (0, 1, 2),
        plan.t_delay, plan.tf, cfg, slack_max=0.0, iters=0, converged=True,
    )


def max_burn(plan):
    return max((np.linalg.norm(dv) for b in plan.burns.values()
                for _, dv in b), default=0.0)


def solve_cell(task):
    """Fixed-order, reordering and folded SCP plans for one grid cell.

    Returns [(total_dv, min_sep, perm index)] for (fixed, reslot) and
    (total_dv, min_sep, max burn) for the reslot plan folded back to fixed
    order; NaN / -1 where there is no such plan.
    """
    direction, mag, delay_orb, transfer_orb, cfg = task
    chief = make_chief()
    sats = hit_formation(direction, mag)
    t_delay = delay_orb * chief.period
    tf = t_delay + transfer_orb * chief.period
    out = []
    plan = None
    for kw in (dict(allow_reslot=False),
               dict(allow_reslot=True, n_assignments=N_ASSIGNMENTS)):
        try:
            plan = plan_recovery(chief, sats, t_delay, tf, cfg, **kw)
        except RuntimeError:
            plan = None
            out.append((np.nan, np.nan, -1))
            continue
        out.append((plan.total_dv, plan.min_separation,
                    PERMS.index(plan.assignment)))
    if plan is None or plan.assignment == (0, 1, 2):
        out.append((np.nan, np.nan, np.nan))
    else:
        folded = folded_plan(chief, sats, plan, cfg)
        out.append((folded.total_dv, folded.min_separation,
                    max_burn(folded)))
    return out


def sweep_scp(directions, mag, delays, transfers, cfg, workers):
    """SCP grids per direction, solved in parallel over all cells."""
    tasks = [(d, mag, dl, tr, cfg) for d in directions
             for tr in transfers for dl in delays]
    shape = (len(directions), len(transfers), len(delays))
    raw = np.empty(shape + (3, 3))
    flat = raw.reshape(-1, 3, 3)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        step = max(1, len(tasks) // 20)
        for k, cell in enumerate(pool.map(solve_cell, tasks, chunksize=2)):
            flat[k] = cell
            if (k + 1) % step == 0 or k + 1 == len(tasks):
                print(f"    {k + 1:5d}/{len(tasks)} cells  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    grids = {}
    for a, d in enumerate(directions):
        for b, key in enumerate(("fixed", "reslot")):
            grids[f"{d}_scp_{key}_dv"] = raw[a, :, :, b, 0]
            grids[f"{d}_scp_{key}_sep"] = raw[a, :, :, b, 1]
            grids[f"{d}_scp_{key}_perm"] = raw[a, :, :, b, 2].astype(int)
        grids[f"{d}_scp_folded_dv"] = raw[a, :, :, 2, 0]
        grids[f"{d}_scp_folded_sep"] = raw[a, :, :, 2, 1]
        grids[f"{d}_scp_folded_burn"] = raw[a, :, :, 2, 2]
    return grids


def two_impulse_grids(chief, directions, mag, delays, transfers, tag):
    """Two-impulse sweep in the npz key layout used for the SCP grids."""
    grids = {}
    for d in directions:
        total_dv, max_burn, min_sep = sweep_direction(
            chief, d, mag, delays, transfers, directions=DIRECTIONS
        )
        grids[f"{d}_{tag}_dv"] = total_dv
        grids[f"{d}_{tag}_burn"] = max_burn
        grids[f"{d}_{tag}_sep"] = min_sep
    return grids


def two_impulse_result(g, d, tag):
    """Grids for one direction in plot_trade_space's format."""
    limit = ~(g[f"{d}_{tag}_burn"] <= DV_MAX)   # NaN (singular) counts as over
    koz = g[f"{d}_{tag}_sep"] < D_MIN
    return dict(total_dv=g[f"{d}_{tag}_dv"], limit=limit, koz=koz,
                feasible=~limit & ~koz)


def scp_result(g, d):
    """Cheapest feasible SCP plan per cell (fixed order or reordered)."""
    dv = {}
    for key in ("fixed", "reslot"):
        ok = g[f"{d}_scp_{key}_sep"] >= D_MIN
        dv[key] = np.where(ok, g[f"{d}_scp_{key}_dv"], np.nan)
    use_reslot = np.isfinite(dv["reslot"]) & ~(dv["fixed"] <= dv["reslot"])
    total_dv = np.where(use_reslot, dv["reslot"], dv["fixed"])
    found = (np.isfinite(g[f"{d}_scp_fixed_dv"])
             | np.isfinite(g[f"{d}_scp_reslot_dv"]))
    feasible = np.isfinite(total_dv)
    reordered = use_reslot & (g[f"{d}_scp_reslot_perm"] != IDENTITY)
    return dict(total_dv=total_dv, limit=~found, koz=found & ~feasible,
                feasible=feasible, reordered=reordered)


def reorder_savings(g, d):
    """Delta-v of the best fixed-order plan minus the reordered SCP plan.

    Fixed order = the cheapest feasible one of SCP-with-order-held, the
    two-impulse transfer, and the reordered plan folded back to fixed order,
    so an SCP local optimum can't inflate the saving.
    Returns (savings, category, fixed_best); category is 0 no feasible plan,
    1 fixed order as cheap or cheaper, 2 feasible only with reordering,
    3 reordering cheaper.
    """
    scp = scp_result(g, d)
    ti = two_impulse_result(g, d, "ti_coarse")
    fixed_scp = np.where(g[f"{d}_scp_fixed_sep"] >= D_MIN,
                         g[f"{d}_scp_fixed_dv"], np.nan)
    fixed_ti = np.where(ti["feasible"], ti["total_dv"], np.nan)
    folded_ok = ((g[f"{d}_scp_folded_sep"] >= D_MIN)
                 & (g[f"{d}_scp_folded_burn"] <= DV_MAX))
    fixed_folded = np.where(folded_ok, g[f"{d}_scp_folded_dv"], np.nan)
    fixed_best = np.fmin(np.fmin(fixed_scp, fixed_ti), fixed_folded)
    reorder_dv = np.where(scp["reordered"], scp["total_dv"], np.nan)

    has_fixed = np.isfinite(fixed_best)
    has_reorder = np.isfinite(reorder_dv)
    category = np.zeros(fixed_best.shape, dtype=int)
    category[has_fixed] = 1
    category[has_reorder & ~has_fixed] = 2
    wins = has_reorder & has_fixed & (reorder_dv < fixed_best)
    category[wins] = 3
    savings = np.where(wins, fixed_best - reorder_dv, np.nan)
    return savings, category, fixed_best


def perm_label(perm):
    """'Sat-1→slot 2, Sat-2→slot 1' for the satellites that change slot."""
    return ", ".join(f"Sat-{i + 1}→slot {s + 1}"
                     for i, s in enumerate(perm) if s != i)


def plot_reorder(g, directions, delays, transfers, mag, out_path):
    """Where reordering beats every fixed-order plan, and by how much."""
    fig, axes = plt.subplots(
        1, len(directions), figsize=(5.2 * len(directions), 5.6),
        facecolor=SURFACE, constrained_layout=True, squeeze=False,
    )
    axes = axes[0]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    per_dir = {d: reorder_savings(g, d) for d in directions}
    norm = Normalize(vmin=0.0, vmax=100.0)
    cat_cmap = ListedColormap([BLOCK_KOZ, BLOCK_LIMIT, DRIFT])
    D, T = np.meshgrid(delays, transfers)

    for ax, d in zip(axes, directions):
        savings, category, fixed_best = per_dir[d]
        style_axis(ax)
        ax.pcolormesh(D, T, np.ma.masked_equal(category, 3), cmap=cat_cmap,
                      vmin=-0.5, vmax=2.5, shading="auto")
        sav = np.ma.masked_invalid(savings)
        ax.pcolormesh(D, T, 100 * sav / fixed_best, cmap=cmap, norm=norm,
                      shading="auto")

        # Why the folded fixed-order plan is unavailable where reordering wins.
        win = category >= 2
        for blocked, marker in (
            (g[f"{d}_scp_folded_sep"] < D_MIN, "x"),
            (g[f"{d}_scp_folded_burn"] > DV_MAX, "^"),
        ):
            ii, jj = np.nonzero(win & blocked)
            ax.plot(delays[jj], transfers[ii], marker, color=INK,
                    markersize=4, markeredgewidth=0.9, fillstyle="none",
                    zorder=4)

        n_win = int(win.sum())
        if sav.count():
            i, j = np.unravel_index(np.ma.argmax(sav), sav.shape)
            perm = PERMS[g[f"{d}_scp_reslot_perm"][i, j]]
            ax.plot(delays[j], transfers[i], "o", markersize=8,
                    markerfacecolor=SURFACE, markeredgecolor=INK,
                    markeredgewidth=1.4, clip_on=False, zorder=5)
            ax.annotate(
                f"saves {sav[i, j]:.3f} m/s "
                f"({100 * sav[i, j] / fixed_best[i, j]:.0f}%)\n"
                f"{perm_label(perm)}",
                (delays[j], transfers[i]), xytext=(8, 6),
                textcoords="offset points", fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE,
                          edgecolor="none", alpha=0.85),
            )
        ax.set_title(
            f"{d} hit · reordering wins {n_win}/{category.size} cells",
            fontsize=10, color=INK2, loc="left",
        )
        ax.set_xlabel("reaction delay [orbits]", fontsize=9, color=INK2)
        ax.set_ylabel("transfer time, delay → slot [orbits]", fontsize=9,
                      color=INK2)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        shrink=0.85, pad=0.01)
    cbar.set_label("Δv saved by reordering  [% of best fixed order]",
                   fontsize=9, color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE_AXIS)

    fig.legend(
        handles=[
            Patch(facecolor=SEQ_BLUE[3], label="reordering cheaper"),
            Patch(facecolor=DRIFT, label="feasible only with reordering"),
            Patch(facecolor=BLOCK_LIMIT, edgecolor=BASELINE_AXIS,
                  label="fixed order as cheap or cheaper"),
            Patch(facecolor=BLOCK_KOZ, label="no feasible plan"),
            Line2D([], [], linestyle="", marker="o", markersize=8,
                   markerfacecolor=SURFACE, markeredgecolor=INK,
                   label="largest saving"),
            Line2D([], [], linestyle="", marker="x", color=INK,
                   markersize=5, label="folded fixed order hits keep-out"),
            Line2D([], [], linestyle="", marker="^", color=INK,
                   fillstyle="none", markersize=5,
                   label=f"folded fixed order needs a burn > {DV_MAX:g} m/s"),
        ],
        loc="outside lower center", ncol=4, frameon=False, fontsize=9,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"When does reordering the slots pay?  SCP with reordering vs. best "
        f"feasible fixed-order plan  (|δv| = {mag:g} m/s, "
        f"{SPACING / 1000:g} km spacing)",
        fontsize=12, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reorder_case(direction, mag, delay_orb, transfer_orb, cfg,
                      out_path):
    """Trajectories of the best fixed-order plan vs. the reordered plan."""
    chief = make_chief()
    sats = hit_formation(direction, mag)
    t_delay = delay_orb * chief.period
    tf = t_delay + transfer_orb * chief.period

    candidates = []
    try:
        candidates.append(("SCP, fixed order", plan_recovery(
            chief, sats, t_delay, tf, cfg, allow_reslot=False)))
    except RuntimeError:
        pass
    try:
        candidates.append(("two-impulse, fixed order",
                           baseline_two_impulse_plan(
                               chief, sats, "Sat-1", t_delay, tf, cfg)))
    except np.linalg.LinAlgError:
        pass
    reorder = plan_recovery(chief, sats, t_delay, tf, cfg,
                            n_assignments=N_ASSIGNMENTS)
    if reorder.assignment != (0, 1, 2):
        candidates.append(("reordered plan folded to fixed order",
                           folded_plan(chief, sats, reorder, cfg)))
    candidates = [c for c in candidates if c[1].min_separation >= D_MIN
                  and max_burn(c[1]) <= DV_MAX]
    panels = ([min(candidates, key=lambda c: c[1].total_dv)]
              if candidates else [])
    panels.append((f"SCP, reordered ({perm_label(reorder.assignment)})",
                   reorder))

    vert = 2 if direction == "cross-track" else 0
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.4 * len(panels)),
                             facecolor=SURFACE, constrained_layout=True,
                             sharex=True, sharey=True, squeeze=False)
    for ax, (label, plan) in zip(axes[:, 0], panels):
        style_axis(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        drift = plan.times <= t_delay
        for i, (name, traj) in enumerate(plan.trajectories.items()):
            ax.plot(traj[drift, 1], traj[drift, vert], color=SLOT[i],
                    linewidth=1.6, linestyle=(0, (4, 2)))
            ax.plot(traj[~drift, 1], traj[~drift, vert], color=SLOT[i],
                    linewidth=1.8, label=name)
            ax.plot(traj[0, 1], traj[0, vert], "o", color=SLOT[i],
                    markersize=6, markeredgecolor=SURFACE)
            ax.plot(traj[-1, 1], traj[-1, vert], "s", markersize=7,
                    markerfacecolor=SURFACE, markeredgecolor=SLOT[i],
                    markeredgewidth=1.6)
            for t_burn, _ in plan.burns[name]:
                k = min(np.searchsorted(plan.times, t_burn),
                        len(plan.times) - 1)
                ax.plot(traj[k, 1], traj[k, vert], "o", color=SLOT[i],
                        markersize=3.5)
        ax.set_title(
            f"{label}: total Δv {plan.total_dv:.3f} m/s, "
            f"min separation {plan.min_separation:.0f} m",
            fontsize=10, color=INK, loc="left",
        )
        ax.set_ylabel("cross-track z [m]" if vert == 2 else "radial x [m]",
                      fontsize=9, color=INK2)
    ax.invert_xaxis()  # +y to the left, as in sweep.py
    ax.set_xlabel("along-track y [m]  (+y ←)", fontsize=9, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3,
                      loc="upper right")
    fig.suptitle(
        f"Largest reordering saving: {direction} hit, |δv| = {mag:g} m/s, "
        f"delay {delay_orb:.2f} orbit, transfer {transfer_orb:.2f} orbit\n"
        "dashed = uncontrolled drift, ○ start, □ final slot, • burn",
        fontsize=11, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="coarser grids for a fast smoke run")
    parser.add_argument("--mag", type=float, default=0.1,
                        help="perturbation magnitude [m/s]")
    parser.add_argument("--max-delay", type=float, default=1.5,
                        help="largest reaction delay [orbits]")
    parser.add_argument("--max-transfer", type=float, default=2.5,
                        help="largest transfer time [orbits]")
    parser.add_argument("--directions", nargs="+", default=list(DIRECTIONS),
                        choices=list(DIRECTIONS))
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="parallel SCP processes")
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from sweep_scp.npz")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    npz_path = outdir / "sweep_scp.npz"
    cfg = SCP_CONFIG

    if args.replot:
        g = dict(np.load(npz_path))
        directions = [str(d) for d in g["directions"]]
        mag = float(g["mag"])
    else:
        directions, mag = args.directions, args.mag
        chief = make_chief()
        n_d, n_t = (40, 50) if args.quick else (150, 200)
        s_d, s_t = (6, 8) if args.quick else (16, 20)
        g = dict(
            directions=np.array(directions), mag=mag,
            delays_orb=np.linspace(0.0, args.max_delay, n_d),
            transfers_orb=np.linspace(0.05, args.max_transfer, n_t),
            scp_delays_orb=np.linspace(0.0, args.max_delay, s_d),
            scp_transfers_orb=np.linspace(0.05, args.max_transfer, s_t),
        )
        print(f"Two-impulse: {n_d}x{n_t} grid per direction")
        g.update(two_impulse_grids(chief, directions, mag, g["delays_orb"],
                                   g["transfers_orb"], "ti"))
        g.update(two_impulse_grids(chief, directions, mag,
                                   g["scp_delays_orb"],
                                   g["scp_transfers_orb"], "ti_coarse"))
        print(f"SCP: {s_d}x{s_t} grid x {len(directions)} directions, "
              f"{args.workers} workers")
        g.update(sweep_scp(directions, mag, g["scp_delays_orb"],
                           g["scp_transfers_orb"], cfg, args.workers))
        np.savez(npz_path, **g)

    delays, transfers = g["delays_orb"], g["transfers_orb"]
    s_delays, s_transfers = g["scp_delays_orb"], g["scp_transfers_orb"]

    plot_trade_space(
        {d: two_impulse_result(g, d, "ti") for d in directions},
        delays, transfers, mag, outdir / "sweep_scp_two_impulse.png",
    )
    scp = {d: scp_result(g, d) for d in directions}
    plot_trade_space(
        scp, s_delays, s_transfers, mag, outdir / "sweep_scp.png",
        title="SCP recovery, slot reordering allowed",
        limit_label="SCP found no plan",
        cbar_label="total constellation Δv  [m/s]",
        reordered={d: r["reordered"] for d, r in scp.items()},
    )
    plot_reorder(g, directions, s_delays, s_transfers, mag,
                 outdir / "sweep_scp_reorder.png")

    # Trajectories of the single largest saving across all directions.
    best = None
    for d in directions:
        savings = reorder_savings(g, d)[0]
        if np.isfinite(savings).any():
            i, j = np.unravel_index(np.nanargmax(savings), savings.shape)
            if best is None or savings[i, j] > best[0]:
                best = (savings[i, j], d, s_delays[j], s_transfers[i])
    written = ["sweep_scp_two_impulse.png", "sweep_scp.png",
               "sweep_scp_reorder.png"]
    if best is not None:
        _, d, delay_orb, transfer_orb = best
        plot_reorder_case(d, mag, delay_orb, transfer_orb, cfg,
                          outdir / "sweep_scp_best_reorder.png")
        written.append("sweep_scp_best_reorder.png")
    else:
        print("Reordering never beat the fixed-order plans on this grid.")
    print(f"\nWrote {outdir}/" + ", ".join(written + [npz_path.name]))


if __name__ == "__main__":
    main()
