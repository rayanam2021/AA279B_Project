"""Do the headline results survive stricter exclusions and SCP-quality checks?

The slide figures drop cases where the hit satellite is inside another
satellite's keep-out at the end of the uncontrolled drift (t_delay). This
script recomputes the headline numbers under these rules:

    end       inside a keep-out at t_delay                 (what the figures use)
    any       inside a keep-out at any time in [0, t_delay]
    margin    within MARGIN x d_min at any time in [0, t_delay]
    any+best  'any', and each planner's cost is the cheapest safe, converged
              plan from either SCP start (convex or free drift). A fixed-slot
              plan also counts for the swap planner, since it is a valid plan
              with swaps allowed. This guards against a swap "saving" that is
              really a bad fixed-slot local optimum.
    +<=Rx     'any+best', also dropping cases where the convex-start fixed or
              swap plan costs more than R x the lower bound (--max-ratio).

The drift-phase separation is recomputed here by propagating all satellites
over [0, t_delay]; it depends only on the scenario, not on the planner.

It also reports, per planner, SCP convergence and keep-out violations during
the recovery (from the planners' own 10 s post-check), and the warm-start
comparison with non-converged and unsafe free-drift plans removed. In the
table, the warm-start rows ("free-drift start ...") use all free-drift plans.

Run:
    .venv/bin/python my_trade_studies/keepout_filter_check.py
    .venv/bin/python my_trade_studies/keepout_filter_check.py --margin 2.0 --json out.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from configurable_scp_sweep import CONFIG, SPACING_M, build_case  # noqa: E402
from cwsim import propagate_cw_with_impulses  # noqa: E402
from make_slide_figures import analyse  # noqa: E402

DRIFT_SAMPLE_DT = 5.0  # [s] resolution of the drift-phase separation check
REL_TOL = 1e-3         # another plan only counts as cheaper by more than 0.1% (solver noise)


def drift_min_separation(keys: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Closest pairwise distance over [0, t_delay] [m] and when it happens [orbits]."""
    d_min_out, t_min_out = np.empty(len(keys)), np.empty(len(keys))
    for i, case in enumerate(keys):
        chief, sats, td, _ = build_case(*case)
        t = np.linspace(0.0, td, max(200, int(td / DRIFT_SAMPLE_DT)))
        pos = [propagate_cw_with_impulses(s.x0, chief.mean_motion, t, s.impulses)[:, :3]
               for s in sats]
        sep = np.min([np.linalg.norm(pos[a] - pos[b], axis=1)
                      for a, b in itertools.combinations(range(len(sats)), 2)], axis=0)
        d_min_out[i] = sep.min()
        t_min_out[i] = t[sep.argmin()] / chief.period
    return d_min_out, t_min_out


def drift_rate_per_orbit(keys: list[tuple]) -> np.ndarray:
    """Secular along-track drift of the hit satellite [spacings per orbit].

    A kick dv at angle theta from an equilibrium slot drifts along-track at
    -3 * dv * sin(theta) per unit time; the radial part adds no secular drift.
    """
    chief = build_case(*keys[0])[0]
    return np.array([3 * mag * np.sin(np.radians(th)) * chief.period / SPACING_M
                     for th, mag, _ in keys])


def usable(S: dict, label: str) -> np.ndarray:
    """Plan finished, SCP converged, and keep-out met over the recovery."""
    s = S[label]
    return ((s.text("status") == "ok") & s.flag("converged")
            & (s.num("min_sep_recovery") >= CONFIG.d_min))


def best_of_starts(A: dict) -> dict:
    """Cheapest usable plan per case from either SCP start.

    Fixed slots: convex or free-drift start. Swaps allowed: either swap
    start, or either fixed-slot plan (a fixed-slot plan is feasible with
    swaps allowed, so a swap planner that ends up above it just missed it).
    """
    S = A["S"]

    def cost(label):
        return np.where(usable(S, label), S[label].num("total_dv"), np.inf)

    def cheaper(a, b):  # a beats b by more than the solver-noise tolerance
        return a < (1 - REL_TOL) * b

    fixed_from_free = cheaper(cost("fixed_free"), cost("fixed"))
    fx = np.where(fixed_from_free, cost("fixed_free"), cost("fixed"))
    swap_from_free = cheaper(cost("swap_free"), cost("swap"))
    sw = np.where(swap_from_free, cost("swap_free"), cost("swap"))
    swapped = np.where(swap_from_free, S["swap_free"].flag("reslotted"), S["swap"].flag("reslotted"))
    iters = np.where(swap_from_free, S["swap_free"].num("iterations"), S["swap"].num("iterations"))
    fixed_wins = cheaper(fx, sw)  # a fixed-slot plan beats both swap runs
    sw = np.where(fixed_wins, fx, sw)
    swapped = swapped & ~fixed_wins
    fx, sw = (np.where(np.isinf(x), np.nan, x) for x in (fx, sw))
    return dict(fx=fx, sw=sw, swapped=swapped, swap_iters=iters, fixed_from_free=fixed_from_free,
                swap_from_free=swap_from_free & ~fixed_wins, fixed_beats_swap=fixed_wins)


def headline_stats(A: dict, valid: np.ndarray, rate: np.ndarray, best: dict | None = None) -> dict:
    """The headline numbers over the cases in `valid`.

    best : output of best_of_starts() to use the cheapest plan from either
        SCP start for the fixed and swap planners (default: convex start).
    """
    d_min = CONFIG.d_min
    S = A["S"]
    lb, drift, delay = A["lb"], A["drift"], A["delay"]
    if best is None:
        fx, sw = A["fx"], A["sw"]
        swapped = valid & S["swap"].flag("reslotted")
        swap_iters = S["swap"].num("iterations")
    else:
        fx, sw = best["fx"], best["sw"]
        swapped = valid & best["swapped"]
        swap_iters = best["swap_iters"]
    binds = valid & (S["lb"].num("min_sep_recovery") < d_min)
    prem_fx, prem_sw = fx / lb - 1, sw / lb - 1
    saving = 1 - sw / fx
    with np.errstate(invalid="ignore", divide="ignore"):
        recovered = (fx - sw) / (fx - lb)
    rec_mask = binds & (fx - lb > 1e-3)
    best = int(np.nanargmax(np.where(binds, saving, -np.inf)))
    big = binds & (saving > 0.05)
    cw, cw_sep = S["cw"].num("total_dv"), S["cw"].num("min_sep_recovery")
    ff, ff_status = S["fixed_free"].num("total_dv"), S["fixed_free"].text("status")
    fx_convex = A["fx"]  # the warm-start rows always compare the two starts

    by_delay = {}
    for d in np.unique(delay):
        m = binds & (delay == d)
        by_delay[f"{d:g}"] = dict(n=int(m.sum()),
                                  fixed_pct=round(100 * float(np.nanmedian(prem_fx[m])), 1),
                                  swap_pct=round(100 * float(np.nanmedian(prem_sw[m])), 1))
    th, mag, dl = A["keys"][best]
    return {
        "valid cases": int(valid.sum()),
        "keep-out binds": int(binds.sum()),
        "swaps chosen": int(swapped.sum()),
        "swaps where keep-out doesn't bind": int((swapped & ~binds).sum()),
        "swap plans below the bound": int(np.sum(valid & (sw < lb - 1e-4))),
        "median premium, fixed [%]": round(100 * float(np.nanmedian(prem_fx[binds])), 1),
        "median premium, swap [%]": round(100 * float(np.nanmedian(prem_sw[binds])), 1),
        "premium recovered by swaps [%]": round(100 * float(np.nanmedian(recovered[rec_mask]))),
        "swap cheaper by >0.5%": int(np.sum(binds & (saving > 0.005))),
        "best case": f"{th:g}°/{mag:g}/{dl:g}: {fx[best]:.3f}->{sw[best]:.3f} "
                     f"({100 * saving[best]:.0f}%)",
        "swaps solved by one convex problem": f"{int(np.sum(swapped & (swap_iters == 1)))}/{int(swapped.sum())}",
        "two-impulse unsafe": f"{int(np.sum(valid & (cw_sep < d_min)))}/{int(valid.sum())}",
        "two-impulse cost / bound (median)": round(float(np.nanmedian((cw / lb)[valid])), 2),
        "free-drift start >1% worse": int(np.sum(valid & (ff > 1.01 * fx_convex))),
        "free-drift start failures": int(np.sum(valid & (ff_status != "ok"))),
        "free-drift start worst [m/s]": round(float(np.nanmax(ff[valid])), 1),
        "cases saving >5%": int(big.sum()),
        "  in drift-rate band 1-2 /orbit": int(np.sum(big & (rate >= 1) & (rate < 2))),
        "  in drift-at-delay window 0.75-2": int(np.sum(big & (drift >= 0.75) & (drift < 2))),
        "premium by delay": by_delay,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin", type=float, default=1.5,
                        help="near-miss rule: drop cases within MARGIN x d_min during the drift")
    parser.add_argument("--max-ratio", type=float, default=3.0,
                        help="last column: drop cases whose convex-start fixed or swap plan "
                             "costs more than this many times the lower bound")
    parser.add_argument("--json", type=Path, help="also write the results to this JSON file")
    args = parser.parse_args()

    A, _ = analyse()
    keys, d_min = A["keys"], CONFIG.d_min
    sep_end = A["S"]["swap"].num("sep_at_delay", ok_only=False)
    min_drift, t_min = drift_min_separation(keys)
    rate = drift_rate_per_orbit(keys)

    end_in, any_in = sep_end < d_min, min_drift < d_min
    print(f"inside a keep-out at the end of the drift: {end_in.sum()} of {len(keys)}")
    print(f"inside a keep-out at any time in the drift: {any_in.sum()} "
          f"({np.sum(any_in & ~end_in)} passed through and came back out):")
    for i in np.flatnonzero(any_in & ~end_in):
        th, mag, dl = keys[i]
        print(f"  θ={th:g}, {mag:g} m/s, {dl:g} orbit: closest {min_drift[i]:.0f} m at "
              f"{t_min[i]:.2f} orbits, {sep_end[i]:.0f} m at t_delay")

    # SCP convergence and plan quality per planner (cases clear at t_delay)
    S, v = A["S"], ~end_in
    print("\nSCP convergence (cases clear of the keep-out at t_delay):")
    for label in ("fixed", "swap", "fixed_free", "swap_free"):
        s = S[label]
        ok = s.text("status") == "ok"
        conv = s.flag("converged")
        ratio = s.num("total_dv") / A["lb"]
        print(f"  {label:10s} failed {np.sum(v & ~ok):2d}, not converged {np.sum(v & ok & ~conv):2d}, "
              f"hit max_iters {np.sum(v & (s.num('iterations') >= CONFIG.max_iters)):2d}, "
              f"cost > 3x bound {np.sum(v & (ratio > 3)):2d} (of which not converged "
              f"{np.sum(v & ok & ~conv & (ratio > 3))})")

    # Keep-out during the recovery (t >= t_delay), from the planners' 10 s post-check.
    # Between samples the separation can dip by about (v_rel * check_dt / 2)^2 / (2 d_min),
    # ~0.02 m at 0.5 m/s relative speed: well inside the 0.5 m collision margin.
    print(f"\nKeep-out broken during the recovery (closest approach < {d_min:g} m, "
          f"post-check every {CONFIG.check_dt:g} s):")
    for label in ("fixed", "swap", "fixed_free", "swap_free", "cw", "lb"):
        s = S[label]
        ok = v & (s.text("status") == "ok")
        sep = s.num("min_sep_recovery")
        broken = ok & (sep < d_min)
        line = (f"  {label:10s} {broken.sum():3d} of {ok.sum()}, closest approach "
                f"{np.nanmin(sep[ok]):7.2f} m")
        if label in ("fixed_free", "swap_free") and broken.any():
            line += (f"; these cost {np.nanmin(s.num('total_dv')[broken]):.1f}-"
                     f"{np.nanmax(s.num('total_dv')[broken]):.1f} m/s, converged "
                     f"{np.sum(broken & s.flag('converged'))}/{broken.sum()}")
        print(line)
    best = best_of_starts(A)
    print("\nBest of both SCP starts (cases clear at t_delay):")
    print(f"  free-drift start gives a fixed-slot plan >{100 * REL_TOL:g}% cheaper in "
          f"{np.sum(v & best['fixed_from_free'])} cases (those >1% cheaper:)")
    for i in np.flatnonzero(v & best["fixed_from_free"] & (best["fx"] < 0.99 * A["fx"])):
        th, mag, dl = keys[i]
        print(f"    θ={th:g}, {mag:g} m/s, {dl:g} orbit: fixed {A['fx'][i]:.4f} -> {best['fx'][i]:.4f}, "
              f"swap {A['sw'][i]:.4f} -> {best['sw'][i]:.4f}")
    print(f"  free-drift start gives a cheaper swap plan in {np.sum(v & best['swap_from_free'])} cases; "
          f"a fixed-slot plan beats both swap runs in {np.sum(v & best['fixed_beats_swap'])}")

    # Warm-start comparison with non-converged and unsafe free-drift plans removed
    ff_ok = v & (S["fixed_free"].text("status") == "ok") & S["fixed_free"].flag("converged")
    ff_use = v & usable(S, "fixed_free")
    ff, fx_c = S["fixed_free"].num("total_dv"), A["fx"]
    worse = v & (ff > 1.01 * fx_c)
    print("\nWarm start, fixed slots: free-drift vs convex start")
    for tag, m in (("all plans", v), ("converged plans only", ff_ok),
                   ("converged and safe plans only", ff_use)):
        print(f"  {tag:30s} >1% worse in {np.sum(worse & m):3d} of {np.sum(m & np.isfinite(ff))}, "
              f"worst {np.nanmax(ff[m]):7.2f} m/s ({np.nanmax((ff / A['lb'])[m]):6.1f}x bound), "
              f"better by >1% in {np.sum(m & (ff < 0.99 * fx_c))}")
    print(f"  ({np.sum(v & ~ff_ok)} failed or not converged, "
          f"{np.sum(ff_ok & ~ff_use)} converged but break the keep-out in flight)")

    cheap = ((A["fx"] / A["lb"] <= args.max_ratio) & (A["sw"] / A["lb"] <= args.max_ratio))
    print(f"\nCases whose convex-start fixed or swap plan costs > {args.max_ratio:g}x the bound "
          f"(clear at t_delay):")
    for i in np.flatnonzero(v & ~cheap):
        th, mag, dl = keys[i]
        print(f"  θ={th:g}, {mag:g} m/s, {dl:g} orbit: fixed {A['fx'][i]:.3f} "
              f"({A['fx'][i] / A['lb'][i]:.1f}x), swap {A['sw'][i]:.3f} "
              f"({A['sw'][i] / A['lb'][i]:.1f}x), converged {S['fixed'].flag('converged')[i]}, "
              f"{sep_end[i]:.0f} m from a neighbour at t_delay")
    rules = {
        "end": (~end_in, None),
        "any": (~any_in, None),
        f"margin {args.margin:g}x": (min_drift >= args.margin * d_min, None),
        "any+best": (~any_in, best),
        f"+<={args.max_ratio:g}x": (~any_in & cheap, best),
    }
    results = {name: headline_stats(A, mask, rate, b) for name, (mask, b) in rules.items()}

    names = list(results)
    width = max(len(k) for k in results["end"])
    col = max(14, *(len(str(x)) + 2 for r in results.values() for x in r.values()
                    if not isinstance(x, dict)))
    print(f"\n{'':{width}}  " + "".join(f"{n:{col}}" for n in names))
    for key in results["end"]:
        if key == "premium by delay":
            continue
        print(f"{key:{width}}  " + "".join(f"{str(results[n][key]):{col}}" for n in names))
    print("\nmedian premium by delay, fixed / swap [%] (n = binding cases):")
    for n in names:
        row = "  ".join(f"{d}: {v['fixed_pct']}/{v['swap_pct']} (n={v['n']})"
                        for d, v in results[n]["premium by delay"].items())
        print(f"  {n:10s} {row}")

    if args.json:
        args.json.write_text(json.dumps(dict(
            rules={"end": "inside a keep-out at t_delay",
                   "any": "inside a keep-out at any time in [0, t_delay]",
                   f"margin {args.margin:g}x": f"within {args.margin:g} x d_min in [0, t_delay]",
                   "any+best": "'any', cheapest safe converged plan from either SCP start",
                   f"+<={args.max_ratio:g}x": f"'any+best', minus cases where a convex-start plan "
                                              f"costs > {args.max_ratio:g}x the bound"},
            results=results), indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
