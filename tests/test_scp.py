"""End-to-end SCP recovery test: terminal geometry and keep-out compliance.

Run:  python tests/test_scp.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    CircularOrbit,
    ScpConfig,
    along_track_formation,
    baseline_convex_plan,
    baseline_roe_plan,
    baseline_two_impulse_plan,
    plan_recovery,
    propagate_cw_with_impulses,
)
from cwsim.scp import convex_guess


def test_scp_restores_formation() -> None:
    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    n = chief.mean_motion
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, np.array([0.0, 0.1, 0.0]))  # along-track kick

    t_delay = 0.25 * chief.period
    tf = 1.6 * chief.period
    cfg = ScpConfig(dt=120.0, n_sub=6, collision_margin=25.0, d_min=200.0)
    plan = plan_recovery(chief, sats, t_delay, tf, cfg)
    print(f"plan: {plan.summary()}")

    # Propagate each satellite (perturbation + planned burns) to tf and a
    # settling check point one orbit later.
    t_check = np.array([tf, tf + chief.period])
    finals = {}
    for sat in sats:
        impulses = list(sat.impulses) + plan.burns[sat.name]
        finals[sat.name] = propagate_cw_with_impulses(
            sat.x0, n, t_check, impulses
        )

    # Terminal states must sit on the along-track slot set {-1km, 0, +1km}
    # with zero radial/cross components and zero relative velocity.
    ys = sorted(finals[s.name][0][1] for s in sats)
    assert np.allclose(ys, [-1000.0, 0.0, 1000.0], atol=1e-3), ys
    for sat in sats:
        xf = finals[sat.name][0]
        assert abs(xf[0]) < 1e-3 and abs(xf[2]) < 1e-3, xf
        assert np.linalg.norm(xf[3:]) < 1e-6, xf
        # One orbit later the formation must not have drifted (equilibrium;
        # solver-precision residual velocities amplify through the secular
        # terms, so allow centimeter-level tolerance).
        assert np.allclose(finals[sat.name][1], xf, atol=1e-2)
    print(f"terminal in-track positions: {[f'{y:+.4f}' for y in ys]} m  PASS")

    # Pairwise in-track separations are exactly the 1 km spacing.
    seps = sorted(
        abs(finals[a.name][0][1] - finals[b.name][0][1])
        for a, b in itertools.combinations(sats, 2)
    )
    assert np.allclose(seps, [1000.0, 1000.0, 2000.0], atol=1e-3), seps
    print("pairwise in-track separations 1/1/2 km:   PASS")

    # With continuous-time (sub-sampled) keep-out constraints plus margin,
    # the fine-grid minimum separation must respect d_min strictly.
    assert plan.min_separation >= cfg.d_min, plan.min_separation
    print(f"min separation {plan.min_separation:.1f} m >= "
          f"{cfg.d_min:.0f} m keep-out:          PASS")


def test_baseline_convex_plan() -> None:
    """Convex baseline reaches the slot, moves only the perturbed satellite,
    and is no more expensive than the closed-form baselines."""
    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    n = chief.mean_motion
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, np.array([0.05, 0.10, 0.03]))
    # 2.15-orbit window: whole orbits are singular for the two-impulse CW.
    t_delay, tf = 0.5 * chief.period, 2.65 * chief.period
    cfg = ScpConfig(dt=60.0)

    plan = baseline_convex_plan(chief, sats, "Sat-2", t_delay, tf, cfg)
    impulses = list(sats[1].impulses) + plan.burns["Sat-2"]
    xf = propagate_cw_with_impulses(sats[1].x0, n, np.array([tf]), impulses)[0]
    assert np.abs(xf[:3] - sats[1].x0[:3]).max() < 1e-3, xf
    assert np.abs(xf[3:] - sats[1].x0[3:]).max() < 1e-6, xf
    assert all(t_delay <= t <= tf for t, _ in plan.burns["Sat-2"])
    assert plan.burns["Sat-1"] == [] and plan.burns["Sat-3"] == []

    roe = baseline_roe_plan(chief, sats, "Sat-2", t_delay, tf, cfg).total_dv
    cw = baseline_two_impulse_plan(chief, sats, "Sat-2", t_delay, tf, cfg).total_dv
    # Grid optimum may sit marginally above CD's continuous-time burn times.
    assert plan.total_dv <= 1.01 * roe and plan.total_dv <= cw, (
        plan.total_dv, roe, cw)
    print(f"convex baseline {plan.total_dv:.5f} m/s "
          f"(CD {roe:.5f}, two-impulse {cw:.5f}):  PASS")

    # dv_max is honoured.
    capped = baseline_convex_plan(
        chief, sats, "Sat-2", t_delay, tf, ScpConfig(dt=60.0, dv_max=0.02)
    )
    assert max(np.linalg.norm(d) for _, d in capped.burns["Sat-2"]) <= 0.02 + 1e-6
    print(f"dv_max=0.02 respected ({capped.total_dv:.5f} m/s):        PASS")


def test_scp_controlled_only_moves_perturbed() -> None:
    """With controlled=[perturbed] the neighbours never burn, the perturbed
    satellite still reaches its slot, and the keep-out is respected."""
    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    n = chief.mean_motion
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, np.array([0.0, 0.1, 0.0]))  # drifts toward Sat-1

    t_delay, tf = 0.25 * chief.period, 2.4 * chief.period
    cfg = ScpConfig(collision_margin=10.0, d_min=200.0)
    plan = plan_recovery(chief, sats, t_delay, tf, cfg, allow_reslot=False,
                         initial_guess=convex_guess, controlled=["Sat-2"])
    assert plan.converged
    assert plan.burns["Sat-1"] == [] and plan.burns["Sat-3"] == [], plan.burns
    assert plan.assignment == (0, 1, 2)
    impulses = list(sats[1].impulses) + plan.burns["Sat-2"]
    xf = propagate_cw_with_impulses(sats[1].x0, n, np.array([tf + 1e-6]), impulses)[0]
    assert np.abs(xf[:3] - sats[1].x0[:3]).max() < 1e-2, xf
    assert np.abs(xf[3:] - sats[1].x0[3:]).max() < 1e-5, xf
    assert plan.min_separation >= cfg.d_min, plan.min_separation
    print(f"controlled=[Sat-2]: {plan.summary()}  PASS")

    # Re-slotting keeps uncontrolled satellites in their own slots.
    reslot = plan_recovery(chief, sats, t_delay, tf, cfg, allow_reslot=True,
                           initial_guess=convex_guess, controlled=["Sat-2"])
    assert reslot.assignment == (0, 1, 2)
    assert reslot.burns["Sat-1"] == [] and reslot.burns["Sat-3"] == []


def test_assignment_pruning_matches_exhaustive() -> None:
    """Branch-and-bound assignment pruning returns the same plan as solving
    the SCP for every slot permutation, with far fewer SCP solves."""
    import cwsim.scp as scp

    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    # Sweep-style cases (configurable_scp_sweep.py grid), drawn with a fixed
    # seed; a coarser grid keeps the test quick.
    grid = list(itertools.product(range(0, 180, 10), [0.05, 0.10, 0.15, 0.20],
                                  [0.10, 0.25, 0.50, 0.75, 1.00]))
    rng = np.random.default_rng(0)
    cases = [grid[i] for i in rng.choice(len(grid), 20, replace=False)]
    cfg = ScpConfig(dt=180.0, n_sub=6, dv_max=1000.0, fuel_max=1e6,
                    d_min=200.0, collision_margin=0.5)

    calls = {"n": 0}
    orig = scp._solve_fixed_assignment

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    scp._solve_fixed_assignment = counted
    solves = {True: 0, False: 0}
    try:
        for theta, mag, delay in cases:
            plans = {}
            for prune in (False, True):
                sats = along_track_formation(spacing=1000.0, count=3)
                th = np.radians(theta)
                sats[1].add_impulse(0.0, mag * np.array([np.cos(th), np.sin(th), 0.0]))
                t_delay = delay * chief.period
                calls["n"] = 0
                plans[prune] = plan_recovery(
                    chief, sats, t_delay, t_delay + 2.15 * chief.period, cfg,
                    initial_guess=convex_guess, prune=prune)
                solves[prune] += calls["n"]
            full, pruned = plans[False], plans[True]
            # Same KOZ status, and pruning never does worse (ties within the
            # SCP's relative convergence tolerance may pick either slot).
            assert ((full.min_sep_recovery >= cfg.d_min)
                    == (pruned.min_sep_recovery >= cfg.d_min)), (theta, mag, delay)
            assert pruned.objective <= full.objective * (1 + cfg.tol) + 1e-9, (
                theta, mag, delay, full.summary(), pruned.summary())
    finally:
        scp._solve_fixed_assignment = orig
    assert solves[True] < solves[False]
    print(f"pruning matches exhaustive on {len(cases)} cases; SCP solves "
          f"{solves[True]} vs {solves[False]}:  PASS")


if __name__ == "__main__":
    test_scp_restores_formation()
    test_baseline_convex_plan()
    test_scp_controlled_only_moves_perturbed()
    test_assignment_pruning_matches_exhaustive()
    print("\nSCP test passed.")
