"""Clohessy-Wiltshire (CW) relative dynamics.

LVLH frame convention (chief-centered):
    x : radial (away from Earth)
    y : along-track (chief velocity direction for a circular orbit)
    z : cross-track (orbit normal, completes right-handed triad)

State vector: x = [x, y, z, xdot, ydot, zdot]  (m, m/s)
"""

from __future__ import annotations

import numpy as np


def cw_stm(n: float, t: float) -> np.ndarray:
    """Full 3-DOF Clohessy-Wiltshire state transition matrix Phi(t) (6x6).

    x(t) = Phi(t) @ x(0), for a chief on a circular orbit with mean motion n.

    Parameters
    ----------
    n : mean motion of the chief orbit [rad/s]
    t : propagation time [s]
    """
    s = np.sin(n * t)
    c = np.cos(n * t)

    phi_rr = np.array([
        [4.0 - 3.0 * c,        0.0, 0.0],
        [6.0 * (s - n * t),    1.0, 0.0],
        [0.0,                  0.0, c],
    ])
    phi_rv = np.array([
        [s / n,                2.0 * (1.0 - c) / n,       0.0],
        [-2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * n * t) / n, 0.0],
        [0.0,                  0.0,                        s / n],
    ])
    phi_vr = np.array([
        [3.0 * n * s,          0.0, 0.0],
        [-6.0 * n * (1.0 - c), 0.0, 0.0],
        [0.0,                  0.0, -n * s],
    ])
    phi_vv = np.array([
        [c,          2.0 * s,       0.0],
        [-2.0 * s,   4.0 * c - 3.0, 0.0],
        [0.0,        0.0,           c],
    ])

    phi = np.zeros((6, 6))
    phi[:3, :3] = phi_rr
    phi[:3, 3:] = phi_rv
    phi[3:, :3] = phi_vr
    phi[3:, 3:] = phi_vv
    return phi


def propagate_cw(x0: np.ndarray, n: float, times: np.ndarray) -> np.ndarray:
    """Propagate an unforced CW state over an array of times.

    Returns an (N, 6) array of states; times are measured from the epoch of x0.
    """
    x0 = np.asarray(x0, dtype=float).reshape(6)
    return np.stack([cw_stm(n, float(t)) @ x0 for t in np.atleast_1d(times)])


def propagate_cw_with_impulses(
    x0: np.ndarray,
    n: float,
    times: np.ndarray,
    impulses: list[tuple[float, np.ndarray]] | None = None,
) -> np.ndarray:
    """Piecewise-STM propagation with impulsive delta-v maneuvers.

    Parameters
    ----------
    x0 : initial LVLH state at times[0] epoch reference (t = 0)
    n : chief mean motion [rad/s]
    times : (N,) sample times [s], measured from the epoch of x0 (must be sorted)
    impulses : list of (t_burn, dv) with dv a 3-vector [m/s] in LVLH axes.
        An impulse at t_burn applies to all samples with t >= t_burn.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    out = np.empty((times.size, 6))

    # Segment boundaries: epoch state updated at each burn, in time order.
    burns = sorted(impulses or [], key=lambda b: b[0])
    seg_t0 = 0.0
    seg_x0 = np.asarray(x0, dtype=float).reshape(6).copy()
    burn_idx = 0

    for i, t in enumerate(times):
        # Apply any burns that occur at or before this sample time.
        while burn_idx < len(burns) and burns[burn_idx][0] <= t:
            tb, dv = burns[burn_idx]
            seg_x0 = cw_stm(n, tb - seg_t0) @ seg_x0
            seg_x0[3:] += np.asarray(dv, dtype=float).reshape(3)
            seg_t0 = tb
            burn_idx += 1
        out[i] = cw_stm(n, t - seg_t0) @ seg_x0
    return out


def cw_two_impulse(
    x0: np.ndarray,
    n: float,
    tf: float,
    x_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form two-impulse CW transfer (fixed-time rendezvous to a slot).

    Computes dv1 applied at t=0 and dv2 applied at t=tf such that the deputy
    arrives at x_target (default: the origin with zero relative velocity).

    Returns (dv1, dv2), each a 3-vector in LVLH axes [m/s].
    """
    x0 = np.asarray(x0, dtype=float).reshape(6)
    if x_target is None:
        x_target = np.zeros(6)
    x_target = np.asarray(x_target, dtype=float).reshape(6)

    phi = cw_stm(n, tf)
    phi_rr, phi_rv = phi[:3, :3], phi[:3, 3:]
    phi_vr, phi_vv = phi[3:, :3], phi[3:, 3:]

    r0, v0 = x0[:3], x0[3:]
    rf, vf = x_target[:3], x_target[3:]

    # Required post-burn velocity to hit rf at tf (phi_rv is singular when
    # n*tf is a multiple of pi for the in-plane block; caller should avoid).
    v0_plus = np.linalg.solve(phi_rv, rf - phi_rr @ r0)
    dv1 = v0_plus - v0

    # Arrival velocity before the second burn, then match target velocity.
    vf_minus = phi_vr @ r0 + phi_vv @ v0_plus
    dv2 = vf - vf_minus
    return dv1, dv2
