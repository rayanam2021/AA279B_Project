"""Plot and compare sweep results written by configurable_scp_sweep.py.

Figures (saved next to the first CSV unless --out is given):
    sweep_dv_map.png      total recovery dv over (theta, delay) for each method,
                          one column per disturbance magnitude, shared log colour scale
    sweep_dv_compare.png  method A vs method B total dv for the chosen burn
                          directions (default: along-track, theta = 90),
                          coloured by reaction delay; points below the diagonal
                          mean method B is cheaper

Cases where the uncontrolled drift leaves a satellite inside another's keep-out
sphere at t_delay (the start of the recovery) are planner-independent bad starts:
they are drawn in amber on the dv map, left out of its colour scale, and
excluded from the comparison plot and win summary. The check uses the CSV's
sep_at_delay column, or recomputes it from the scenario for older CSVs.

Each sweep is labelled from its filename, sweep_<label>_<date>_<time>.csv,
so two configurations of the same method (e.g. sweep_scp_*.csv and
sweep_scp_two_imp_*.csv, both method=scp) are kept apart. When two sweeps share
a label, each gets its run date and time appended, e.g.
"scp_reslot_convex (11-18 09:12)". Override with --labels.

Run:
    python my_trade_studies/plot_sweep_comparison.py                      # latest CSV per label
    python my_trade_studies/plot_sweep_comparison.py A.csv B.csv --theta 90
    python my_trade_studies/plot_sweep_comparison.py A.csv B.csv --labels ballistic two-impulse
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from configurable_scp_sweep import CONFIG, RESULTS_DIR, load_sweep, separation_at_delay

# Chart chrome and ordered blue ramp (light -> dark) for delay / magnitude.
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DELAY_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # validated ordinal steps
KOZ_START = "#000000"   # case starts inside the keep-out (excluded)
MISSING = "#f0efec"     # failed / missing case

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sweep_label(path: Path) -> str:
    """Label from a sweep filename: sweep_<label>_<date>_<time>.csv -> <label>."""
    return path.stem.removeprefix("sweep_").rsplit("_", 2)[0]


def sweep_stamp(path: Path) -> str:
    """Run time from a sweep filename, sweep_<label>_YYYYMMDD_HHMMSS.csv -> 'MM-DD HH:MM'."""
    date, time = path.stem.rsplit("_", 2)[1:]
    if len(date) == 8 and len(time) >= 4 and (date + time).isdigit():
        return f"{date[4:6]}-{date[6:]} {time[:2]}:{time[2:4]}"
    return f"{date}_{time}"


def unique_labels(paths: list[Path]) -> list[str]:
    """Filename labels, with the run date/time appended wherever two sweeps share a label."""
    base = [sweep_label(p) for p in paths]
    labels = [f"{b} ({sweep_stamp(p)})" if base.count(b) > 1 else b for b, p in zip(base, paths)]
    # same file (or same minute) twice: number the repeats
    return [lab if labels[:i].count(lab) == 0 else f"{lab} #{labels[:i].count(lab) + 1}"
            for i, lab in enumerate(labels)]


def two_line(label: str) -> str:
    """Put a trailing '(date time)' on its own line, for narrow axis labels."""
    return label.replace(" (", "\n(", 1)


def plain_log_ticks(axis):
    """Log axis ticks at 1, 2, 5 x 10^k labelled as plain numbers (0.05, 0.1, 0.2)."""
    axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def latest_per_method() -> list[Path]:
    """Most recent sweep_<label>_<stamp>.csv for each label, oldest label first."""
    latest = {}
    by_time = sorted(RESULTS_DIR.glob("sweep_*.csv"), key=lambda p: p.stem.rsplit("_", 2)[1:])
    for path in by_time:
        latest[sweep_label(path)] = path
    return list(latest.values())


def starts_in_koz(r) -> np.ndarray:
    """True where a satellite is inside a keep-out sphere at t_delay."""
    sep = r.get("sep_at_delay")
    if sep is None or not np.issubdtype(sep.dtype, np.floating):  # older CSV
        sep = np.array([separation_at_delay(t, m, d) for t, m, d in
                        zip(r["theta_deg"], r["dv_mag"], r["delay_orbits"])])
    return sep < CONFIG.d_min


def case_lookup(r, exclude_koz=False) -> dict[tuple, float]:
    """(theta, dv_mag, delay) -> total_dv, NaN for failed cases (and, with
    exclude_koz, for cases that start inside the keep-out)."""
    dv = np.where(r["koz_start"], np.nan, r["total_dv"]) if exclude_koz else r["total_dv"]
    return {(t, m, d): v for t, m, d, v in
            zip(r["theta_deg"], r["dv_mag"], r["delay_orbits"], dv)}


def plot_dv_maps(sweeps: dict, out: Path):
    """Heatmap of total dv over (theta, delay); rows = methods, columns = dv magnitude."""
    # union of every sweep's grid, so a sweep with extra angles / magnitudes is not cropped
    thetas, mags, delays = (np.unique(np.concatenate([r[k] for r in sweeps.values()]))
                            for k in ("theta_deg", "dv_mag", "delay_orbits"))
    xticks = [t for t in range(0, 360, 45) if thetas.min() <= t <= thetas.max()]
    all_dv = np.concatenate([np.where(r["koz_start"], np.nan, r["total_dv"])
                             for r in sweeps.values()])
    norm = LogNorm(np.nanmin(all_dv[all_dv > 0]), np.nanmax(all_dv))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap.set_bad(MISSING)
    koz_cmap = ListedColormap([KOZ_START])
    koz_cmap.set_bad((0, 0, 0, 0))  # transparent where the case starts clear

    fig, axes = plt.subplots(len(sweeps), len(mags),
                             figsize=(max(3.2 * len(mags) + 2.2, 6.5), 2.4 * len(sweeps) + 1.4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for row, (method, r) in enumerate(sweeps.items()):
        lookup = case_lookup(r, exclude_koz=True)
        koz = {(t, m, d): k for t, m, d, k in
               zip(r["theta_deg"], r["dv_mag"], r["delay_orbits"], r["koz_start"])}
        for col, mag in enumerate(mags):
            grid = np.array([[lookup.get((t, mag, d), np.nan) for t in thetas] for d in delays])
            koz_grid = np.array([[1.0 if koz.get((t, mag, d)) else np.nan for t in thetas]
                                 for d in delays])
            ax = axes[row, col]
            im = ax.pcolormesh(thetas, np.arange(len(delays)), grid, cmap=cmap, norm=norm,
                               shading="nearest", edgecolors=SURFACE, linewidth=0.5)
            ax.pcolormesh(thetas, np.arange(len(delays)), koz_grid, cmap=koz_cmap,
                          shading="nearest", edgecolors=SURFACE, linewidth=0.5)
            ax.set_yticks(range(len(delays)), [f"{d:g}" for d in delays])
            ax.set_xticks(xticks)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 0:
                ax.set_title(f"disturbance {mag:g} m/s")
            if col == 0:
                ax.set_ylabel(f"{two_line(method)}\ndelay [orbits]", fontsize=8.5)
            if row == len(sweeps) - 1:
                ax.set_xlabel("disturbance angle θ [deg]\n(0 radial, 90 along-track)")
    cb = fig.colorbar(im, ax=axes, label="total recovery Δv [m/s]", shrink=0.9, pad=0.02)
    plain_log_ticks(cb.ax.yaxis)
    cb.ax.tick_params(labelsize=8)
    fig.legend(handles=[Patch(color=KOZ_START, label="starts inside keep-out (excluded)"),
                        Patch(color=MISSING, label="failed / missing")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Total recovery Δv across the sweep", color=INK, fontsize=11)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


def plot_comparison(name_a, ra, name_b, rb, thetas, out: Path):
    """Scatter of method A vs method B total dv at the given angles, coloured by delay."""
    la, lb = case_lookup(ra, exclude_koz=True), case_lookup(rb, exclude_koz=True)
    delays = np.unique(ra["delay_orbits"])
    mags = np.unique(ra["dv_mag"])
    shared = [k for k in la if k in lb and np.isfinite(la[k]) and np.isfinite(lb[k])]
    lo = min(min(la[k], lb[k]) for k in shared) / 1.3
    hi = max(max(la[k], lb[k]) for k in shared) * 1.3

    fig, axes = plt.subplots(1, len(thetas), figsize=(max(5.2 * len(thetas), 6.5), 5.9),
                             squeeze=False, layout="constrained")
    for ax, theta in zip(axes[0], thetas):
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(0.97, 0.03, "B cheaper", transform=ax.transAxes, ha="right", va="bottom",
                color=MUTED, fontsize=9)
        ax.text(0.03, 0.97, "A cheaper", transform=ax.transAxes, ha="left", va="top",
                color=MUTED, fontsize=9)
        for delay, color in zip(delays, DELAY_COLORS * 2):
            keys = [(theta, m, delay) for m in mags if (theta, m, delay) in shared]
            if not keys:
                continue
            a = np.array([la[k] for k in keys])
            b = np.array([lb[k] for k in keys])
            order = np.argsort(a)
            ax.plot(a[order], b[order], color=color, lw=1, alpha=0.6, zorder=2)
            ax.scatter(a, b, s=48, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=2,
                       label=f"{delay:g} orbits   B cheaper {np.sum(b < a)}/{len(keys)}")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), aspect="equal",
               xlabel=f"A: {name_a}\ntotal Δv [m/s]", ylabel=f"B: {name_b}\ntotal Δv [m/s]",
               title=f"θ = {theta:g}°  (one point per disturbance magnitude)")
        for axis in (ax.xaxis, ax.yaxis):
            plain_log_ticks(axis)
        ax.grid(color=GRID, lw=0.5, which="major")
        ax.legend(title="reaction delay", frameon=False, fontsize=8, title_fontsize=8,
                  loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle(f"A: {name_a}   vs   B: {name_b}\n"
                 "(cases starting inside the keep-out excluded)", color=INK, fontsize=10)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


def print_win_summary(name_a, ra, name_b, rb):
    """How often each method is cheaper, per delay, over every angle and magnitude."""
    la, lb = case_lookup(ra, exclude_koz=True), case_lookup(rb, exclude_koz=True)
    n_koz = len({k for r in (ra, rb) for k, flag in
                 zip(zip(r["theta_deg"], r["dv_mag"], r["delay_orbits"]), r["koz_start"])
                 if flag})
    print(f"\n{name_b} cheaper than {name_a} (all angles and magnitudes; "
          f"excluded {n_koz} cases that start inside the keep-out):")
    for delay in np.unique(ra["delay_orbits"]):
        keys = [k for k in la if k[2] == delay and k in lb]
        ratio = np.array([lb[k] / la[k] for k in keys])
        ok = np.isfinite(ratio)
        print(f"  delay {delay:5.2f} orbits: {np.sum(ratio[ok] < 1):3d}/{ok.sum()} cases, "
              f"median {name_b}/{name_a} dv ratio = {np.median(ratio[ok]):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="*", type=Path,
                        help="sweep CSVs (default: latest per method); the first two are compared")
    parser.add_argument("--theta", type=float, nargs="+", default=[90],
                        help="disturbance angles for the comparison plot [deg]")
    parser.add_argument("--labels", nargs="+",
                        help="names for the sweeps, one per CSV (default: from the filenames)")
    parser.add_argument("--out", type=Path, help="output directory for the figures")
    args = parser.parse_args()

    paths = args.csvs or latest_per_method()
    not_csv = [p.name for p in paths if p.suffix != ".csv"]
    if not_csv:
        parser.error(f"expected sweep result CSVs, got {', '.join(not_csv)}")
    if args.labels and len(args.labels) != len(paths):
        parser.error(f"--labels needs {len(paths)} names, got {len(args.labels)}")
    labels = args.labels or unique_labels(paths)
    if len(set(labels)) != len(labels):
        parser.error(f"--labels must be distinct, got {labels}")
    sweeps = {}
    for label, path in zip(labels, paths):
        r = load_sweep(path)
        r["koz_start"] = starts_in_koz(r)
        sweeps[label] = r
        print(f"{path.name}: label={label}, method={r['method'][0]}, "
              f"{len(r['total_dv'])} cases, {np.sum(r['status'] != 'ok')} failed, "
              f"{r['koz_start'].sum()} start inside the keep-out")
    out_dir = args.out or paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dv_maps(sweeps, out_dir / "sweep_dv_map.png")
    if len(sweeps) >= 2:
        (name_a, ra), (name_b, rb) = list(sweeps.items())[:2]
        plot_comparison(name_a, ra, name_b, rb, args.theta, out_dir / "sweep_dv_compare.png")
        print_win_summary(name_a, ra, name_b, rb)


if __name__ == "__main__":
    main()
