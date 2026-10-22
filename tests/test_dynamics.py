"""Validation tests: STM vs numerical integration, solver accuracy, frames.

Run:  python tests/test_dynamics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cwsim import (
    CircularOrbit,
    cw_stm,
    cw_two_impulse,
    eci_to_relative,
    propagate_cw_with_impulses,
    relative_to_eci,
)


def cw_derivs(x: np.ndarray, n: float) -> np.ndarray:
    """CW equations of motion (for RK4 cross-check of the STM)."""
    px, py, pz, vx, vy, vz = x
    return np.array([
        vx, vy, vz,
        3.0 * n**2 * px + 2.0 * n * vy,
        -2.0 * n * vx,
        -(n**2) * pz,
    ])


def rk4(x0: np.ndarray, n: float, tf: float, steps: int = 20000) -> np.ndarray:
    dt = tf / steps
    x = x0.copy()
    for _ in range(steps):
        k1 = cw_derivs(x, n)
        k2 = cw_derivs(x + 0.5 * dt * k1, n)
        k3 = cw_derivs(x + 0.5 * dt * k2, n)
        k4 = cw_derivs(x + dt * k3, n)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def test_stm_vs_rk4() -> None:
    chief = CircularOrbit(altitude=550e3)
    n = chief.mean_motion
    x0 = np.array([100.0, -500.0, 50.0, 0.1, -0.2, 0.05])
    tf = 1.5 * chief.period

    x_stm = cw_stm(n, tf) @ x0
    x_rk4 = rk4(x0, n, tf)
    err = np.abs(x_stm - x_rk4)
    assert err[:3].max() < 1e-3, f"STM position error {err[:3].max():.2e} m"
    assert err[3:].max() < 1e-6, f"STM velocity error {err[3:].max():.2e} m/s"
    print(f"STM vs RK4:        max pos err {err[:3].max():.2e} m  PASS")


def test_stm_identity_and_composition() -> None:
    n = CircularOrbit().mean_motion
    assert np.allclose(cw_stm(n, 0.0), np.eye(6))
    # Phi(t1 + t2) = Phi(t2) @ Phi(t1)
    t1, t2 = 700.0, 1900.0
    assert np.allclose(cw_stm(n, t1 + t2), cw_stm(n, t2) @ cw_stm(n, t1))
    print("STM identity/composition:              PASS")


def test_two_impulse_rendezvous() -> None:
    chief = CircularOrbit(altitude=550e3)
    n = chief.mean_motion
    x0 = np.array([200.0, -3000.0, 100.0, 0.05, 0.3, -0.1])
    tf = 0.7 * chief.period
    target = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])

    dv1, dv2 = cw_two_impulse(x0, n, tf, x_target=target)
    xf = propagate_cw_with_impulses(
        x0, n, np.array([tf]), impulses=[(0.0, dv1), (tf, dv2)]
    )[0]
    err = np.abs(xf - target)
    assert err[:3].max() < 1e-6, f"arrival position error {err[:3].max():.2e} m"
    assert err[3:].max() < 1e-9, f"arrival velocity error {err[3:].max():.2e} m/s"
    print(f"Two-impulse rendezvous: arrival err {err[:3].max():.2e} m  PASS")


def test_frames_roundtrip() -> None:
    chief = CircularOrbit(altitude=550e3, inclination=0.9, raan=0.4, arg_lat0=1.1)
    n = chief.mean_motion
    r_c, v_c = chief.state_eci(1234.0)
    x_rel = np.array([150.0, -800.0, 60.0, 0.2, -0.1, 0.05])

    r_d, v_d = relative_to_eci(x_rel, r_c, v_c, n)
    x_back = eci_to_relative(r_d, v_d, r_c, v_c, n)
    assert np.allclose(x_back, x_rel, atol=1e-9)
    print("LVLH <-> ECI roundtrip:                PASS")


def test_along_track_equilibrium() -> None:
    """A pure along-track offset must be stationary under CW dynamics."""
    n = CircularOrbit().mean_motion
    x0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    xf = cw_stm(n, 5000.0) @ x0
    assert np.allclose(xf, x0, atol=1e-9)
    print("Along-track offset equilibrium:        PASS")


if __name__ == "__main__":
    test_stm_identity_and_composition()
    test_stm_vs_rk4()
    test_two_impulse_rendezvous()
    test_frames_roundtrip()
    test_along_track_equilibrium()
    print("\nAll tests passed.")
