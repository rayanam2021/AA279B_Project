"""Run the full configurable_scp_sweep grid and overlay every trajectory in the RT plane.

Every (theta, dv, delay) case from configurable_scp_sweep.py is planned with the
chosen method, and each moving satellite's relative trajectory is drawn in the
radial / along-track plane as a thin, semi-transparent line. A trajectory that
ever enters another satellite's keep-out zone (closer than CONFIG.d_min) is
drawn in red along its whole length. The keep-out circles and the formation
slots are drawn on top.

The figure is sized 16:9 with large black fonts, a white background and a short
title (the planner's slide name) so it drops straight onto a slide.

Run:
    python my_trade_studies/plot_sweep_trajectories.py --method scp_convex
    python my_trade_studies/plot_sweep_trajectories.py --method cw --recovery-only
    python my_trade_studies/plot_sweep_trajectories.py --cache results/trajectories_cw_<stamp>.pkl

Output: results/trajectories_<method>_<stamp>.png plus a .pkl cache of the
trajectories, so the figure can be restyled with --cache without re-planning.
"""
import argparse
import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from tqdm import tqdm

from configurable_scp_sweep import (
    CONFIG, DELAY_ORBITS, DV_MAG, METHOD, METHODS, RESULTS_DIR, THETA_DEG, build_case,
)

# ------------------------------- style -------------------------------
TRAJ_COLOR = "#2a78d6"      # nominal trajectory
KOZ_COLOR = "#d62728"       # trajectory that enters a keep-out zone at any point
SAT_COLOR = "black"
GRID = "#e1e0d9"            # light warm grey, as in the notebook slide figures
TRAJ_ALPHA, TRAJ_LW = 0.18, 0.8
KOZ_ALPHA, KOZ_LW = 0.55, 1.1
MOVE_TOL_M = 1.0            # a satellite that never leaves its slot by more than this isn't drawn

# Slide names for each planner, used as the default title.
METHOD_TITLES = {
    "convex": "Optimal",
    "scp_convex": "Safe",
    "scp_reslot_convex": "Safe + Reslotting",
    "cw": "Two-Impulse CW",
    "roe": "ROE",
}

# Slide styling: black text and axes, large fonts, white background.
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "font.size": 20, "axes.labelsize": 22, "axes.titlesize": 24,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True,
})


# ------------------------------- sweep -------------------------------
def run_case(method: str, theta_deg: float, dv_mag: float, delay_orbits: float) -> dict:
    """Plan one case and keep only what the plot needs (positions in the RT plane)."""
    chief, sats, td, tf = build_case(theta_deg, dv_mag, delay_orbits)
    out = dict(theta_deg=theta_deg, dv_mag=dv_mag, delay_orbits=delay_orbits, t_delay=td,
               slots={s.name: s.x0[:2].copy() for s in sats})
    try:
        plan = METHODS[method](chief, sats, td, tf)
    except Exception as exc:  # record the failure and keep sweeping
        return dict(out, status="error", error=f"{type(exc).__name__}: {exc}")
    return dict(out, status="ok", times=plan.times,
                rt={name: traj[:, :2].astype(np.float32) for name, traj in plan.trajectories.items()})


def run_sweep(method: str, workers: int, limit: int) -> list[dict]:
    cases = list(itertools.product(THETA_DEG, DV_MAG, DELAY_ORBITS))
    if limit:
        cases = cases[:limit]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, method, *c) for c in cases]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="case", dynamic_ncols=True):
            results.append(fut.result())
    results.sort(key=lambda r: (r["theta_deg"], r["dv_mag"], r["delay_orbits"]))
    return results


# ------------------------------- plotting -------------------------------
def axis_limits(boxes: np.ndarray, slots: dict, quantile: float):
    """Frame that fully contains `quantile` of the trajectories along each edge.

    boxes: (n, 4) per-trajectory [tmin, tmax, rmin, rmax] in km. A handful of
    outliers (e.g. one huge drift loop) would otherwise shrink everything else.
    """
    pad = CONFIG.d_min / 1e3
    slot_t = [t / 1e3 for _, t in slots.values()]
    slot_r = [r / 1e3 for r, _ in slots.values()]
    t_lo = min(np.quantile(boxes[:, 0], 1 - quantile), min(slot_t) - pad)
    t_hi = max(np.quantile(boxes[:, 1], quantile), max(slot_t) + pad)
    r_lo = min(np.quantile(boxes[:, 2], 1 - quantile), min(slot_r) - pad)
    r_hi = max(np.quantile(boxes[:, 3], quantile), max(slot_r) + pad)
    mt, mr = 0.04 * (t_hi - t_lo), 0.06 * (r_hi - r_lo)
    return (t_lo - mt, t_hi + mt), (r_lo - mr, r_hi + mr)


def plot(results: list[dict], method: str, out: Path, recovery_only: bool, title: str | None,
         quantile: float = 0.99, xlim=None, ylim=None):
    d_min = CONFIG.d_min
    ok = [r for r in results if r["status"] == "ok"]
    slots = results[0]["slots"]

    nominal, violating, boxes = [], [], []
    n_cases_koz = 0
    for r in ok:
        keep = r["times"] >= r["t_delay"] if recovery_only else np.ones(len(r["times"]), bool)
        rt = {name: xy[keep] for name, xy in r["rt"].items()}
        case_hit = False
        for name, xy in rt.items():
            if np.abs(xy - slots[name]).max() <= MOVE_TOL_M:
                continue  # stationary neighbour: shown as a slot marker only
            dist = np.min([np.linalg.norm(xy - other, axis=1)
                           for o, other in rt.items() if o != name], axis=0)
            inside = dist < d_min
            case_hit |= inside.any()
            tr = xy[:, ::-1] / 1e3  # plot along-track on x, radial on y, in km
            boxes.append([tr[:, 0].min(), tr[:, 0].max(), tr[:, 1].min(), tr[:, 1].max()])
            (violating if inside.any() else nominal).append(tr)
        n_cases_koz += case_hit

    fig, ax = plt.subplots(figsize=(13.33, 7.5), layout="constrained")
    ax.add_collection(LineCollection(nominal, colors=TRAJ_COLOR, alpha=TRAJ_ALPHA,
                                     linewidths=TRAJ_LW, zorder=2))
    ax.add_collection(LineCollection(violating, colors=KOZ_COLOR, alpha=KOZ_ALPHA,
                                     linewidths=KOZ_LW, zorder=3))

    # Formation slots with their keep-out circles: a light fill under the
    # trajectories, the dashed edge and the slot marker on top.
    for j, (r0, t0) in enumerate(slots.values()):
        x, y = t0 / 1e3, r0 / 1e3
        ax.add_patch(Circle((x, y), d_min / 1e3, facecolor=GRID, edgecolor="none",
                            alpha=0.7, zorder=1))
        ax.add_patch(Circle((x, y), d_min / 1e3, facecolor="none", edgecolor=SAT_COLOR,
                            linewidth=1.8, linestyle=(0, (4, 3)), zorder=4))
        ax.plot(x, y, marker="s", ms=13, mfc="white", mec=SAT_COLOR, mew=2.0, zorder=5)
        ax.annotate(f"Slot {j + 1}", (x, y + d_min / 1e3), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    handles = [
        Line2D([], [], color=TRAJ_COLOR, lw=3, label="Trajectory"),
        Line2D([], [], color=KOZ_COLOR, lw=3, label="Enters Keep-Out Zone"),
        Line2D([], [], color=SAT_COLOR, ls="none", marker="s", ms=12, mfc="white",
               mew=2.0, label="Slot"),
        Circle((0, 0), 1, facecolor=GRID, edgecolor=SAT_COLOR, linewidth=1.5,
               linestyle=(0, (4, 3)), label="Keep-Out Zone"),
    ]
    # Legend below the axes, placed before the axes box is measured below.
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False)

    if title is None:
        title = METHOD_TITLES.get(method, method)
    boxes = np.array(boxes)
    auto_x, auto_y = axis_limits(boxes, slots, quantile)
    xlim, ylim = xlim or auto_x, ylim or auto_y
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title or " ")  # reserve the title's space before measuring the axes
    ax.set_xlabel("Along-Track [km]")
    ax.set_ylabel("Radial [km]")
    fig.canvas.draw()
    # Equal scale (true RT geometry, KOZs stay circles): widen the short side so the
    # requested frame fills the axes box rather than leaving empty margins.
    bbox = ax.get_window_extent()
    box_ratio = bbox.height / bbox.width
    dx, dy = np.ptp(xlim), np.ptp(ylim)
    if dy / dx < box_ratio:
        grow = (box_ratio * dx - dy) / 2
        ylim = (ylim[0] - grow, ylim[1] + grow)
    else:
        grow = (dy / box_ratio - dx) / 2
        xlim = (xlim[0] - grow, xlim[1] + grow)
    ax.set_xlim(xlim[::-1])  # along-track positive to the left
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    n_out = int(np.sum((boxes[:, 0] < xlim[0]) | (boxes[:, 1] > xlim[1])
                       | (boxes[:, 2] < ylim[0]) | (boxes[:, 3] > ylim[1])))
    ax.set_title(title)

    fig.savefig(out, dpi=200)
    print(f"{len(ok)}/{len(results)} cases plotted, {n_cases_koz} enter a keep-out zone -> {out}")
    if n_out:
        print(f"{n_out} trajector{'y extends' if n_out == 1 else 'ies extend'} beyond the frame")


# --------------------------------- main ---------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, default=METHOD)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    parser.add_argument("--cache", type=Path, help="re-plot trajectories from an earlier .pkl")
    parser.add_argument("--recovery-only", action="store_true",
                        help="draw only t >= t_delay (skip the uncontrolled drift)")
    parser.add_argument("--title", help="figure title ('' for none)")
    parser.add_argument("--quantile", type=float, default=0.99,
                        help="auto-frame so this fraction of trajectories fits on each edge")
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="along-track limits [km] (overrides --quantile)")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="radial limits [km] (overrides --quantile)")
    parser.add_argument("--out", type=Path, help="output image (.png, .svg, .pdf)")
    args = parser.parse_args()

    if args.cache:
        with args.cache.open("rb") as f:
            saved = pickle.load(f)
        method, results = saved["method"], saved["results"]
        stem = args.cache.with_suffix("")
    else:
        method = args.method
        results = run_sweep(method, args.workers, args.limit)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = RESULTS_DIR / f"trajectories_{method}_{stamp}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        with stem.with_suffix(".pkl").open("wb") as f:
            pickle.dump(dict(method=method, results=results), f)
        for r in results:
            if r["status"] != "ok":
                print(f"theta={r['theta_deg']} dv={r['dv_mag']} delay={r['delay_orbits']}: {r['error']}")

    out = args.out or Path(f"{stem}{'_recovery' if args.recovery_only else ''}.png")
    plot(results, method, out, args.recovery_only, args.title, args.quantile, args.xlim, args.ylim)


if __name__ == "__main__":
    main()
