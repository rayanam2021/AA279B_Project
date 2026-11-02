"""Validation of the ROE <-> CW maps and the Chernick & D'Amico solver.

Run:  python tests/test_roe.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    MU_EARTH,
    CircularOrbit,
    ClosedFormUnavailable,
    cw_stm,
    cw_to_roe,
    cw_to_roe_matrix,
    cw_two_impulse,
    propagate_cw_with_impulses,
    relative_to_eci,
    roe_control_matrix,
    roe_impulsive_reconfig,
    roe_stm,
    roe_to_cw,
    roe_to_cw_matrix,
)

CHIEF = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
N = CHIEF.mean_motion
A = CHIEF.radius
P = CHIEF.period


def _wrap(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _kepler_elements(r: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Classical elements (a, i, raan, e_x, e_y, mean arg. of latitude)."""
    h = np.cross(r, v)
    inc = np.arccos(h[2] / np.linalg.norm(h))
    node = np.cross([0.0, 0.0, 1.0], h)
    raan = np.arctan2(node[1], node[0])
    n_hat = node / np.linalg.norm(node)
    m_hat = np.cross(h / np.linalg.norm(h), n_hat)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    a = 1.0 / (2.0 / rn - vn**2 / MU_EARTH)
    e_vec = ((vn**2 - MU_EARTH / rn) * r - (r @ v) * v) / MU_EARTH
    ex, ey = e_vec @ n_hat, e_vec @ m_hat          # e cos(w), e sin(w)
    u_true = np.arctan2(r @ m_hat, r @ n_hat)
    e, w = np.hypot(ex, ey), np.arctan2(ey, ex)
    nu = u_true - w
    ecc_anom = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                                np.sqrt(1 + e) * np.cos(nu / 2))
    mean_anom = ecc_anom - e * np.sin(ecc_anom)
    return dict(a=a, i=inc, raan=raan, ex=ex, ey=ey, u=w + mean_anom)


# --------------------------------------------------------------------------
# Frame / dynamics consistency
# --------------------------------------------------------------------------

def test_roundtrip() -> None:
    for u in (0.0, 0.7, 2.5, -1.9):
        T = roe_to_cw_matrix(A, N, u)
        assert np.allclose(cw_to_roe_matrix(A, N, u) @ T, np.eye(6), atol=1e-12)
    x = np.array([120.0, -800.0, 40.0, 0.03, -0.2, 0.01])
    assert np.allclose(roe_to_cw(cw_to_roe(x, A, N, 1.3), A, N, 1.3), x)
    print("CW <-> ROE roundtrip:                       PASS")


def test_roe_stm_matches_cw_stm() -> None:
    """ROE map + Keplerian ROE STM reproduce the CW STM exactly."""
    x0 = np.array([100.0, -500.0, 50.0, 0.1, -0.2, 0.05])
    u0 = 0.4
    for t in (300.0, 0.37 * P, 2.6 * P):
        lhs = cw_to_roe(cw_stm(N, t) @ x0, A, N, u0 + N * t)
        rhs = roe_stm(N, t) @ cw_to_roe(x0, A, N, u0)
        assert np.allclose(lhs, rhs, rtol=0, atol=1e-14), (lhs - rhs)
    print("ROE STM == CW STM through the ROE map:      PASS")


def test_control_matrix_matches_impulse() -> None:
    """Gamma(u) dv equals the ROE jump from a CW velocity impulse."""
    x = np.array([30.0, 200.0, -10.0, 0.02, 0.01, -0.03])
    dv = np.array([0.013, -0.021, 0.034])
    for u in (0.0, 1.1, 4.0):
        x_plus = x.copy()
        x_plus[3:] += dv
        jump = cw_to_roe(x_plus, A, N, u) - cw_to_roe(x, A, N, u)
        assert np.allclose(jump, roe_control_matrix(A, N, u) @ dv, atol=1e-16)
    print("Control input matrix Gamma (GVE):           PASS")


def test_roe_matches_orbital_elements() -> None:
    """The linear map agrees with ROE built from true Keplerian elements
    ([1] Eq. 4) to first order in separation."""
    t = 1234.0
    r_c, v_c = CHIEF.state_eci(t)
    u_c = CHIEF.arg_lat0 + N * t
    el_c = _kepler_elements(r_c, v_c)
    x_rel = np.array([150.0, -600.0, 80.0, 0.04, -0.3, 0.06])
    r_d, v_d = relative_to_eci(x_rel, r_c, v_c, N)
    el_d = _kepler_elements(r_d, v_d)

    d_raan = _wrap(el_d["raan"] - el_c["raan"])
    true_roe = np.array([
        (el_d["a"] - el_c["a"]) / el_c["a"],
        _wrap(el_d["u"] - u_c) + d_raan * np.cos(el_c["i"]),
        el_d["ex"],                  # chief is circular: e_c = 0
        el_d["ey"],
        el_d["i"] - el_c["i"],
        d_raan * np.sin(el_c["i"]),
    ])
    lin_roe = cw_to_roe(x_rel, A, N, u_c)
    err = np.abs(true_roe - lin_roe)
    # Second-order residual ~ (rho / a)^2 with rho ~ 1 km: ~2e-8.
    assert err.max() < 5e-8, err
    assert err.max() < 5e-3 * np.abs(lin_roe).max()
    print(f"ROE map vs Keplerian elements: max err {err.max():.1e}   PASS")


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

def _arrival(x0, tf, burns):
    return propagate_cw_with_impulses(x0, N, np.array([tf]), burns)[0]


def _case(roe0, roef, orbits, u0=0.0):
    tf = orbits * P
    x0 = roe_to_cw(np.asarray(roe0, float), 1.0, N, u0)
    xt = roe_to_cw(np.asarray(roef, float), 1.0, N, u0 + N * tf)
    return x0, xt, tf


def test_reconfig_hits_target() -> None:
    rng = np.random.default_rng(3)
    for _ in range(25):
        x0, xt, tf = _case(rng.normal(0, 200, 6), rng.normal(0, 200, 6),
                           rng.uniform(1.6, 5.0), u0=rng.uniform(0, 6.3))
        for cand in ("paper", "augmented"):
            plan = roe_impulsive_reconfig(x0, N, tf, xt, candidates=cand)
            xf = _arrival(x0, tf, plan.burns)
            assert np.abs(xf[:3] - xt[:3]).max() < 1e-6, xf - xt
            assert np.abs(xf[3:] - xt[3:]).max() < 1e-9
            assert all(0.0 <= t <= tf for t, _ in plan.burns)
            assert plan.inplane_dv >= plan.dv_min_inplane * (1 - 1e-9)
    print("Closed-form reconfiguration hits target:    PASS")


def test_u0_invariance() -> None:
    """Burn times and dv do not depend on the argument-of-latitude origin."""
    x0 = np.array([50.0, 900.0, 30.0, 0.02, -0.05, 0.04])
    tf = 2.3 * P
    p1 = roe_impulsive_reconfig(x0, N, tf, u0=0.0)
    p2 = roe_impulsive_reconfig(x0, N, tf, u0=2.2)
    assert len(p1.burns) == len(p2.burns)
    for (t1, d1), (t2, d2) in zip(p1.burns, p2.burns):
        assert abs(t1 - t2) < 1e-6 and np.allclose(d1, d2, atol=1e-12)
    print("Invariance to u0:                           PASS")


def test_de_dominant_is_optimal() -> None:
    """[2] Table 11: dominant de reaches the reachable minimum ||a de||/2 n."""
    x0, xt, tf = _case([0, 0, 300, -100, 0, 0], [10, 20, -50, 80, 0, 0], 2.5)
    for cand in ("paper", "augmented"):
        plan = roe_impulsive_reconfig(x0, N, tf, xt, candidates=cand)
        assert plan.dominant == "de" and plan.inplane_optimal
        de = np.hypot(-50 - 300, 80 + 100)
        assert np.isclose(plan.total_dv, N * de / 2, rtol=1e-9)
        # All in-plane burns tangential at u = Ubar + k pi.
        assert all(abs(dv[0]) < 1e-15 for _, dv in plan.burns)
    print(f"Dominant de optimal: dv = {plan.total_dv:.5f} m/s = dv_min   PASS")


def test_dlambda_dominant_suboptimal() -> None:
    """[2] Sec. VI: dominant dlambda is flagged sub-optimal; the augmented
    candidate set is never worse than the paper's."""
    x0, xt, tf = _case([0, 2000, 30, -20, 0, 0], [0, 0, 0, 0, 0, 0], 2.0)
    paper = roe_impulsive_reconfig(x0, N, tf, xt, candidates="paper")
    aug = roe_impulsive_reconfig(x0, N, tf, xt, candidates="augmented")
    assert paper.dominant == "dlambda"
    assert not paper.inplane_optimal and paper.inplane_ratio > 1.0
    assert aug.inplane_dv <= paper.inplane_dv * (1 + 1e-12)
    print(f"Dominant dlambda: paper x{paper.inplane_ratio:.3f}, augmented "
          f"x{aug.inplane_ratio:.3f}   PASS")


def test_out_of_plane() -> None:
    """[1] Eq. 23: single cross-track burn at the phase of di, size n||a di||."""
    x0, xt, tf = _case([0, 0, 0, 0, 40, -30], [0, 0, 0, 0, -20, 50], 1.0)
    plan = roe_impulsive_reconfig(x0, N, tf, xt)
    assert len(plan.burns) == 1
    t, dv = plan.burns[0]
    assert np.isclose(abs(dv[2]), N * np.hypot(-60, 80), rtol=1e-12)
    phase = np.arctan2(80, -60)
    assert abs(np.sin(N * t - phase)) < 1e-9  # u = phase + k pi
    # Short window (< pi) without an aligned location: two edge burns.
    x0, xt, tf = _case([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 30, 30], 0.2, u0=2.0)
    plan = roe_impulsive_reconfig(x0, N, tf, xt, u0=2.0)
    assert [t for t, _ in plan.burns] == [0.0, tf]
    assert np.abs(_arrival(x0, tf, plan.burns) - xt).max() < 1e-9
    print("Out-of-plane single / edge-pair burns:      PASS")


def test_short_window_raises() -> None:
    x0, xt, tf = _case([50, 300, 40, -20, 0, 0], [0, 0, 0, 0, 0, 0], 0.4)
    try:
        roe_impulsive_reconfig(x0, N, tf, xt, candidates="paper")
    except ClosedFormUnavailable:
        print("Short window raises ClosedFormUnavailable:  PASS")
        return
    raise AssertionError("expected ClosedFormUnavailable")


def test_cheaper_than_two_impulse() -> None:
    """Formation-recovery scenario from examples/demo_formation.py."""
    x_hit = np.zeros(6)
    x_hit[3:] = [0.05, 0.10, 0.03]
    x_delay = cw_stm(N, 0.5 * P) @ x_hit
    for orbits in (1.3, 1.7, 2.4):
        tf = orbits * P
        dv1, dv2 = cw_two_impulse(x_delay, N, tf)
        plan = roe_impulsive_reconfig(x_delay, N, tf)
        assert plan.total_dv < np.linalg.norm(dv1) + np.linalg.norm(dv2)
        assert plan.arrival_error < 1e-6
    print("Cheaper than two-impulse CW (demo case):    PASS")


def test_against_convex_optimum() -> None:
    """Cross-check optimality against a fine-grid convex solve with full 3D
    impulses. Where the closed form claims optimality, no grid solution may
    beat it; elsewhere it must stay within the reported bounds."""
    try:
        import cvxpy as cp
    except ImportError:  # pragma: no cover
        print("Convex cross-check:                         SKIP (no cvxpy)")
        return

    def convex_opt(x0, xt, tf, n_grid=600):
        ts = np.linspace(0.0, tf, n_grid + 1)
        B = np.stack([cw_stm(N, tf - t)[:, 3:] for t in ts])  # (K, 6, 3)
        dv = cp.Variable((3, n_grid + 1))
        xf = cw_stm(N, tf) @ x0 + sum(B[k] @ dv[:, k] for k in range(len(ts)))
        obj = cp.sum(cp.norm(dv, axis=0))
        cp.Problem(cp.Minimize(obj), [xf == xt]).solve(solver="CLARABEL")
        return float(obj.value)

    cases = [
        ([0, 0, 300, -100, 0, 0], [10, 20, -50, 80, 0, 0], 2.5),  # de
        ([0, 0, 200, 50, 30, 0], [0, 0, 0, 0, 0, 40], 1.7),       # de + di
        ([0, 2000, 30, -20, 0, 0], [0, 0, 0, 0, 0, 0], 2.0),      # dlambda
    ]
    for roe0, roef, orbits in cases:
        x0, xt, tf = _case(roe0, roef, orbits)
        plan = roe_impulsive_reconfig(x0, N, tf, xt)
        opt = convex_opt(x0, xt, tf)
        assert plan.total_dv >= plan.dv_lower_bound * (1 - 1e-9)
        if plan.inplane_optimal and plan.oop_dv == 0.0:
            assert opt >= plan.total_dv * (1 - 1e-4), (opt, plan.total_dv)
        assert plan.total_dv <= 1.25 * opt, (plan.total_dv, opt)
        print(f"  {plan.dominant:8s}: closed form {plan.total_dv:.5f}, "
              f"convex {opt:.5f}, dv_min {plan.dv_lower_bound:.5f} m/s")
    print("Convex cross-check:                         PASS")


def test_baseline_roe_plan() -> None:
    """scp.baseline_roe_plan returns the perturbed satellite to its slot."""
    try:
        from cwsim import ScpConfig, along_track_formation, baseline_roe_plan
    except ImportError:  # pragma: no cover
        print("baseline_roe_plan:                          SKIP (no cvxpy)")
        return
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, np.array([0.05, 0.10, 0.03]))
    t_delay, tf = 0.5 * P, 2.5 * P
    plan = baseline_roe_plan(CHIEF, sats, "Sat-2", t_delay, tf, ScpConfig())
    impulses = list(sats[1].impulses) + plan.burns["Sat-2"]
    xf = propagate_cw_with_impulses(sats[1].x0, N, np.array([tf]), impulses)[0]
    assert np.abs(xf - sats[1].x0).max() < 1e-6, xf
    assert all(t_delay <= t <= tf for t, _ in plan.burns["Sat-2"])
    assert plan.burns["Sat-1"] == [] and plan.burns["Sat-3"] == []
    print(f"baseline_roe_plan: {plan.total_dv:.4f} m/s, slot reached     PASS")


if __name__ == "__main__":
    test_roundtrip()
    test_roe_stm_matches_cw_stm()
    test_control_matrix_matches_impulse()
    test_roe_matches_orbital_elements()
    test_reconfig_hits_target()
    test_u0_invariance()
    test_de_dominant_is_optimal()
    test_dlambda_dominant_suboptimal()
    test_out_of_plane()
    test_short_window_raises()
    test_cheaper_than_two_impulse()
    test_against_convex_optimum()
    test_baseline_roe_plan()
    print("\nAll ROE tests passed.")
