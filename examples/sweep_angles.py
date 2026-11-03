"""Trade space: does SCP reorder the slots, vs. the direction of the hit?

Scenario (same as sweep.py / sweep_scp.py)
    3-satellite along-track formation, 1 km apart, 550 km equatorial orbit.
    Sat-1 (the trailing satellite) is hit at t = 0 by an impulse of fixed size
    whose direction is swept:
        in-plane angle      alpha, in the radial / along-track plane, measured
                            from +radial toward +along-track
        out-of-plane angle  beta, elevation toward +cross-track
        dv = |dv| [cos(beta) cos(alpha), cos(beta) sin(alpha), sin(beta)]
    The formation is mirror-symmetric in cross-track, so beta in [0, 90] covers
    every direction. Each reaction delay gets its own panel; the transfer time
    (delay -> slot) is fixed.

For every direction the SCP planner is solved with the order held and with
reordering allowed (sweep_scp.solve_cell); the cheaper feasible plan is the
SCP's decision. As in sweep_scp.py the reordered plan is also folded back to
fixed order: where that folded plan is feasible and no costlier, reordering
was a choice SCP made, not something the constraints forced.

Output
    results/sweep_angles_reorder.png   SCP decision: which slot order it chose
    results/sweep_angles_dv.png        SCP total delta-v, reordered region
                                       outlined
    results/sweep_angles.npz           raw grids (re-plot with --replot)

Run:  python examples/sweep_angles.py [--quick] [--mag 0.1]
          [--delays 0.25 0.5 1.0] [--transfer 1.0] [--workers 8]
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep import (
    BASELINE_AXIS,
    BLOCK_KOZ,
    BLOCK_LIMIT,
    D_MIN,
    DV_MAX,
    INK,
    INK2,
    MUTED,
    SEQ_BLUE,
    SPACING,
    SURFACE,
    style_axis,
)
from sweep_scp import (
    IDENTITY,
    PERMS,
    SCP_CONFIG,
    perm_label,
    scp_result,
    solve_cell,
)

# One color per non-identity slot order, in PERMS order.
REORDERS = [k for k in range(len(PERMS)) if k != IDENTITY]
REORDER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#e0a526"]

ALPHA_TICKS = ([0, 90, 180, 270, 360],
               ["0°\n+R", "90°\n+T", "180°\n−R", "270°\n−T", "360°\n+R"])


def hit_vector(mag, alpha_deg, beta_deg):
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    return mag * np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a),
                           np.sin(b)])


def sweep_angles(mag, alphas, betas, delays, transfer_orb, cfg, workers):
    """SCP grids (len(betas), len(alphas)) per delay, solved in parallel."""
    tasks = [(hit_vector(mag, a, b), dl, transfer_orb, cfg)
             for dl in delays for b in betas for a in alphas]
    raw = np.empty((len(delays), len(betas), len(alphas), 3, 3))
    flat = raw.reshape(-1, 3, 3)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        step = max(1, len(tasks) // 20)
        for k, cell in enumerate(pool.map(solve_cell, tasks, chunksize=2)):
            flat[k] = cell
            if (k + 1) % step == 0 or k + 1 == len(tasks):
                print(f"    {k + 1:5d}/{len(tasks)} cells  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    # Same key layout as sweep_scp.sweep_scp, tagged by delay index.
    grids = {}
    for a in range(len(delays)):
        for b, key in enumerate(("fixed", "reslot")):
            grids[f"d{a}_scp_{key}_dv"] = raw[a, :, :, b, 0]
            grids[f"d{a}_scp_{key}_sep"] = raw[a, :, :, b, 1]
            grids[f"d{a}_scp_{key}_perm"] = raw[a, :, :, b, 2].astype(int)
        grids[f"d{a}_scp_folded_dv"] = raw[a, :, :, 2, 0]
        grids[f"d{a}_scp_folded_sep"] = raw[a, :, :, 2, 1]
        grids[f"d{a}_scp_folded_burn"] = raw[a, :, :, 2, 2]
    return grids


def decision(g, tag):
    """SCP decision per cell and where reordering was optional.

    category: 0 no feasible plan, 1 order kept, 2 + REORDERS.index(perm) for
    a reordering. optional: reordered, yet the plan folded back to fixed
    order is feasible and no costlier.
    """
    r = scp_result(g, tag)
    category = np.where(r["feasible"], 1, 0)
    perm = g[f"{tag}_scp_reslot_perm"]
    for c, k in enumerate(REORDERS):
        category[r["reordered"] & (perm == k)] = 2 + c
    folded_ok = ((g[f"{tag}_scp_folded_sep"] >= D_MIN)
                 & (g[f"{tag}_scp_folded_burn"] <= DV_MAX)
                 & (g[f"{tag}_scp_folded_dv"] <= r["total_dv"] + 1e-6))
    return category, r["reordered"] & folded_ok, r


def angle_axes(n_panels, height=4.6):
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.4 * n_panels, height), facecolor=SURFACE,
        constrained_layout=True, squeeze=False, sharey=True,
    )
    for ax in axes[0]:
        style_axis(ax)
        ax.set_xticks(*ALPHA_TICKS)
        ax.set_yticks([0, 30, 60, 90], ["0°", "30°", "60°", "90°\n+N"])
        ax.set_xlabel("in-plane angle α (from +radial toward +along-track)",
                      fontsize=9, color=INK2)
    axes[0, 0].set_ylabel("out-of-plane angle β", fontsize=9, color=INK2)
    return fig, axes[0]


def plot_decision(g, delays, alphas, betas, transfer_orb, mag, out_path):
    fig, axes = angle_axes(len(delays))
    cmap = ListedColormap([BLOCK_KOZ, BLOCK_LIMIT] + REORDER_COLORS)
    A, B = np.meshgrid(alphas, betas)
    present = set()
    for k, (ax, delay) in enumerate(zip(axes, delays)):
        category, optional, r = decision(g, f"d{k}")
        present |= set(np.unique(category))
        ax.pcolormesh(A, B, category, cmap=cmap, vmin=-0.5,
                      vmax=len(REORDERS) + 1.5, shading="nearest")
        ii, jj = np.nonzero(optional)
        ax.plot(alphas[jj], betas[ii], "o", markersize=3.2,
                markerfacecolor=SURFACE, markeredgecolor=INK,
                markeredgewidth=0.6)
        n_re = int(r["reordered"].sum())
        ax.set_title(
            f"delay {delay:g} orbit · reordered in {n_re}/{category.size}",
            fontsize=10, color=INK2, loc="left",
        )

    handles = [Patch(facecolor=BLOCK_LIMIT, edgecolor=BASELINE_AXIS,
                     label="order kept")]
    handles += [Patch(facecolor=REORDER_COLORS[c],
                      label=f"reordered: {perm_label(PERMS[k])}")
                for c, k in enumerate(REORDERS) if 2 + c in present]
    if 0 in present:
        handles.append(Patch(facecolor=BLOCK_KOZ, label="no feasible plan"))
    handles.append(Line2D(
        [], [], linestyle="", marker="o", markersize=5,
        markerfacecolor=SURFACE, markeredgecolor=INK,
        label="reordering optional (folded fixed-order plan feasible, "
              "no costlier)",
    ))
    fig.legend(handles=handles, loc="outside lower center",
               ncol=min(len(handles), 3), frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle(
        f"Does SCP reorder the slots?  Sat-1 hit direction sweep  "
        f"(|δv| = {mag:g} m/s, transfer {transfer_orb:g} orbit, "
        f"{SPACING / 1000:g} km spacing)",
        fontsize=12, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dv(g, delays, alphas, betas, transfer_orb, mag, out_path):
    fig, axes = angle_axes(len(delays))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    results = [scp_result(g, f"d{k}") for k in range(len(delays))]
    feasible_dv = np.concatenate([r["total_dv"][r["feasible"]]
                                  for r in results])
    vmax = float(feasible_dv.max()) if feasible_dv.size else 2 * DV_MAX
    norm = Normalize(vmin=0.0, vmax=vmax)
    A, B = np.meshgrid(alphas, betas)
    for ax, delay, r in zip(axes, delays, results):
        ax.pcolormesh(A, B, np.ma.masked_where(r["feasible"], np.ones(A.shape)),
                      cmap=ListedColormap([BLOCK_KOZ]), shading="nearest")
        ax.pcolormesh(A, B, np.ma.masked_where(~r["feasible"], r["total_dv"]),
                      cmap=cmap, norm=norm, shading="nearest")
        if r["reordered"].any() and not r["reordered"].all():
            ax.contour(A, B, r["reordered"].astype(float), levels=[0.5],
                       colors=INK, linewidths=1.4)
        ii, jj = np.nonzero(r["reordered"])
        ax.plot(alphas[jj], betas[ii], ".", color=INK, markersize=2.5)
        ax.set_title(f"delay {delay:g} orbit", fontsize=10, color=INK2,
                     loc="left")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        shrink=0.85, pad=0.01)
    cbar.set_label("SCP total constellation Δv  [m/s]", fontsize=9,
                   color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE_AXIS)
    fig.legend(
        handles=[
            Line2D([], [], color=INK, linewidth=1.4, marker=".",
                   markersize=5, label="slots reordered (boundary + cells)"),
            Patch(facecolor=BLOCK_KOZ, label="no feasible plan"),
        ],
        loc="outside lower center", ncol=2, frameon=False, fontsize=9,
        labelcolor=INK2,
    )
    fig.suptitle(
        f"SCP recovery cost vs. hit direction  (|δv| = {mag:g} m/s, "
        f"transfer {transfer_orb:g} orbit)",
        fontsize=12, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="30° angle steps for a fast smoke run")
    parser.add_argument("--mag", type=float, default=0.1,
                        help="perturbation magnitude [m/s]")
    parser.add_argument("--delays", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0],
                        help="reaction delays, one panel each [orbits]")
    parser.add_argument("--transfer", type=float, default=1.0,
                        help="transfer time, delay -> slot [orbits]")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="parallel SCP processes")
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from sweep_angles.npz")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    npz_path = outdir / "sweep_angles.npz"

    if args.replot:
        g = dict(np.load(npz_path))
    else:
        step = 30.0 if args.quick else 10.0
        g = dict(
            mag=args.mag, transfer_orb=args.transfer,
            delays_orb=np.array(args.delays),
            alphas_deg=np.arange(0.0, 360.0, step),
            betas_deg=np.arange(0.0, 90.0 + step / 2, step),
        )
        n_cells = len(args.delays) * g["alphas_deg"].size * g["betas_deg"].size
        print(f"SCP: {g['alphas_deg'].size}x{g['betas_deg'].size} angles x "
              f"{len(args.delays)} delays = {n_cells} cells, "
              f"{args.workers} workers")
        g.update(sweep_angles(args.mag, g["alphas_deg"], g["betas_deg"],
                              args.delays, args.transfer, SCP_CONFIG,
                              args.workers))
        np.savez(npz_path, **g)

    delays, alphas, betas = g["delays_orb"], g["alphas_deg"], g["betas_deg"]
    mag, transfer_orb = float(g["mag"]), float(g["transfer_orb"])
    plot_decision(g, delays, alphas, betas, transfer_orb, mag,
                  outdir / "sweep_angles_reorder.png")
    plot_dv(g, delays, alphas, betas, transfer_orb, mag,
            outdir / "sweep_angles_dv.png")
    print(f"\nWrote {outdir}/sweep_angles_reorder.png, sweep_angles_dv.png, "
          f"{npz_path.name}")


if __name__ == "__main__":
    main()
