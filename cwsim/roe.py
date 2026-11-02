"""Relative orbital elements (ROE) and the Chernick & D'Amico closed-form
fuel-optimal impulsive reconfiguration solver for near-circular orbits.

ROE state (quasi-nonsingular, non-dimensional; [1] Eq. 4, [2] Eq. 3)
--------------------------------------------------------------------
    da   = (a_d - a_c) / a_c                       relative semi-major axis
    dlam = (u_d - u_c) + (RAAN_d - RAAN_c) cos i_c relative mean longitude
    dex  = e_xd - e_xc,   e_x = e cos(omega)       relative eccentricity
    dey  = e_yd - e_yc,   e_y = e sin(omega)         vector
    dix  = i_d - i_c                               relative inclination
    diy  = (RAAN_d - RAAN_c) sin i_c                 vector

with u = M + omega the mean argument of latitude. The vector is ordered
[da, dlam, dex, dey, dix, diy].

Relation to the CW (Hill / RTN) frame used by cwsim
---------------------------------------------------
For a circular chief, the first-order map from ROE to the rectilinear RTN
state is the general solution of the Clohessy-Wiltshire equations written
with the ROE as integration constants ([3], [4]):

    x / a       = da - dex cos u - dey sin u
    y / a       = dlam + 2 dex sin u - 2 dey cos u
    z / a       = dix sin u - diy cos u
    xdot / (an) = dex sin u - dey cos u
    ydot / (an) = -3/2 da + 2 dex cos u + 2 dey sin u
    zdot / (an) = dix cos u + diy sin u

where u is the chief's argument of latitude at the epoch of the state. The
CW axes match RTN (x radial, y along-track, z cross-track). Under Keplerian
dynamics the ROE are constant except for the drift dlam' = -3/2 n da, so
this map plus the ROE STM (`roe_stm`) reproduces the CW STM exactly (to
machine precision, see tests/test_roe.py).

The ROE are singular for an equatorial chief (i_c = 0) when built from
classical elements, because RAAN is undefined. The CW <-> ROE map above has
no such problem: u can be measured from any fixed in-plane reference
direction, here the chief position at the epoch where u = u0. Because the
CW dynamics are time-invariant, the choice of u0 only rotates the dex/dey
and dix/diy vectors. It never changes the burn times or delta-vs.

Closed-form reconfiguration (Chernick & D'Amico [1], [2])
---------------------------------------------------------
Given an initial state at t = 0 and a target state at t = tf, the required
change is the pseudo-state ([1] Eq. 15, [2] Eq. 2)

    dalpha_bar = dalpha(tf) - Phi(tf, 0) dalpha(0)
               = sum_k Phi(tf, t_k) Gamma(t_k) dv_k

Reachable-set theory splits the problem into three decoupled 2D planes:
(dlam, da), de and di. The lower bound on the cost is set by the dominant
plane ([2] Eq. 8, Tables 6-7):

    dv_min,de   = n ||a de_bar|| / 2
    dv_min,da   = n |a da_bar| / 2
    dv_min,dlam = n |a da_bar / 2 + 2 a dlam_bar / (3 du)|
    dv_min,di   = n ||a di_bar||

with du = n tf. The dlam bound is the paper's Table 6 expression written out
with da0 = 2 and dlam0 = -3 du.

In-plane maneuvers are tangential and placed at the optimal locations

    u_k = atan2(de_bar_y, de_bar_x) + k pi              ([2] Eq. 21, J2 = 0)

so every burn moves de along (k even) or against (k odd) the desired change.
The perpendicular de component then vanishes identically and the in-plane
problem reduces to a 3x3 linear system in the tangential magnitudes
([2] Tables 10-11; [1] Eqs. 27, 31-32 with J2 = 0):

    2 sum_k w_k                 = a da_bar
    -3 sum_k (u_f - u_k) w_k    = a dlam_bar
    2 sum_k (-1)^k w_k          = ||a de_bar||          (w_k = dv_T,k / n)

For dominant de (and usually da), a triplet of locations exists whose
solution costs exactly dv_min. This is the optimal scheme of [2] Table 11.
For dominant dlam, or when the target lies outside the nested reachable
set, no scheme reaches dv_min. The solver then follows the sub-optimal
handling of [2] Sec. VI: solve as if de were dominant and keep the scheme
whose cost is closest to dv_min. The ratio inplane_dv / dv_min_inplane
reports the sub-optimality.

Extension (candidates='augmented', the default): the window edges u0 and
uf, which are the dominant-dlam optimal times of [2] Appendix B (Table 14),
are added to the candidate set together with their half-orbit shifts
u0 + k pi and uf - k pi. Every subset of up to four locations is then solved
against the full 4D in-plane pseudo-state. Because this is a superset of the
paper's candidates it is never worse, and it also works for shorter windows.
Both variants hit the true optimum when dv_min is reachable. When it is
not, the true optimum needs burns at interior locations that no closed form
in [1], [2] provides. Against a fine-grid convex solve of the same
tangential problem, over 60 random 1-4 orbit reconfigurations, the
augmented scheme had a median excess of 0.3 %, a 90th percentile of 3 % and
a worst case of 12 %. On the same set the de-aligned triplets alone
reached 1.8x in the worst case and found no scheme for one short window.
Use candidates='paper' to reproduce the published schemes exactly.

The out-of-plane maneuver is a single cross-track burn at
u = atan2(di_bar_y, di_bar_x) + k pi with |dv_N| = n ||a di_bar||
([1] Eq. 23, [2] Table 10). If the window is shorter than half an orbit and
holds no such location, two cross-track burns at the window edges are used.
This extends the eccentric-orbit "Else" row of [2] Table 8 to the
near-circular case.

Limitations
-----------
* Keplerian near-circular dynamics only, which is exactly the CW model.
  J2 terms from [1] Eq. 8 and [2] Eq. 20 are not included.
* The paper's in-plane schemes need three candidate locations
  u = Ubar + k pi in [u0, uf], which is guaranteed only for du >= 3 pi
  (1.5 orbits). The augmented set usually works from about one orbit.
  Windows with no exact scheme raise `ClosedFormUnavailable`, and the
  caller can fall back to `cw_two_impulse`.
* The (dlam, da) reachable set of [2] Table 6 is the tangential-only
  parallelogram. A radial burn shifts dlam directly by -2 dv_R / (n a)
  ([1] Eq. 10), so the full 3D reachable set reaches slightly further along
  the dlam-dominant edge normal. By a support-function comparison it reaches
  about 8 % further at du = pi and 2 % at du = 2 pi, so dv_min,dlam can
  slightly overstate the true 3D lower bound for short windows (compare
  the radial remark in [1] Sec. IV). In the convex cross-checks, allowing
  radial burns lowered the optimum cost by less than 1 %.

References
----------
[1] M. Chernick and S. D'Amico, "New Closed-Form Solutions for Optimal
    Impulsive Control of Spacecraft Relative Motion," Journal of Guidance,
    Control, and Dynamics, Vol. 41, No. 2, 2018, pp. 301-319,
    doi:10.2514/1.G002848. (Preprint: AIAA SPACE 2016.)
[2] M. Chernick and S. D'Amico, "Closed-Form Optimal Impulsive Control of
    Spacecraft Formations Using Reachable Set Theory," AAS 18-308, 2018;
    journal version "Closed-Form Optimal Impulsive Control of Spacecraft
    Relative Motion Using Reachable Set Theory," JGCD Vol. 44, No. 1, 2021,
    pp. 25-44, doi:10.2514/1.G005218, arXiv:2002.07832. Equation and table
    numbers in this module refer to AAS 18-308 unless stated otherwise.
[3] S. D'Amico, "Autonomous Formation Flying in Low Earth Orbit," Ph.D.
    thesis, TU Delft, 2010 (ROE as integration constants of the HCW
    equations; see also DLR-GSOC TN 05-08, 2005).
[4] G. Gaias and S. D'Amico, "Impulsive Maneuvers for Formation
    Reconfiguration Using Relative Orbital Elements," JGCD, Vol. 38, No. 6,
    2015, pp. 1036-1049, doi:10.2514/1.G000189.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .dynamics import propagate_cw_with_impulses


class ClosedFormUnavailable(ValueError):
    """The closed-form ROE scheme has no solution for this window."""


# --------------------------------------------------------------------------
# CW <-> ROE conversion
# --------------------------------------------------------------------------

def roe_to_cw_matrix(a: float, n: float, u: float) -> np.ndarray:
    """6x6 matrix T(u) with x_cw = T(u) @ roe (circular chief, [3], [4]).

    Parameters
    ----------
    a : chief semi-major axis [m] (use 1.0 for "a-scaled" ROE in meters)
    n : chief mean motion [rad/s]
    u : chief argument of latitude at the epoch of the state [rad]
    """
    s, c = np.sin(u), np.cos(u)
    an = a * n
    return np.array([
        [a,         0.0, -a * c,      -a * s,      0.0,    0.0],
        [0.0,       a,   2 * a * s,   -2 * a * c,  0.0,    0.0],
        [0.0,       0.0, 0.0,         0.0,         a * s,  -a * c],
        [0.0,       0.0, an * s,      -an * c,     0.0,    0.0],
        [-1.5 * an, 0.0, 2 * an * c,  2 * an * s,  0.0,    0.0],
        [0.0,       0.0, 0.0,         0.0,         an * c, an * s],
    ])


def cw_to_roe_matrix(a: float, n: float, u: float) -> np.ndarray:
    """Closed-form inverse of `roe_to_cw_matrix`: roe = T(u)^-1 @ x_cw.

    From the map in the module docstring, with xb = x/a and vb = v/(an):
        da   = 4 xb + 2 vb_y
        dlam = yb - 2 vb_x
        dex  = (3 xb + 2 vb_y) cos u + vb_x sin u
        dey  = (3 xb + 2 vb_y) sin u - vb_x cos u
        dix  = zb sin u + vb_z cos u
        diy  = -zb cos u + vb_z sin u
    """
    s, c = np.sin(u), np.cos(u)
    ia, ian = 1.0 / a, 1.0 / (a * n)
    return np.array([
        [4 * ia,      0.0, 0.0,     0.0,      2 * ian,     0.0],
        [0.0,         ia,  0.0,     -2 * ian, 0.0,         0.0],
        [3 * ia * c,  0.0, 0.0,     ian * s,  2 * ian * c, 0.0],
        [3 * ia * s,  0.0, 0.0,     -ian * c, 2 * ian * s, 0.0],
        [0.0,         0.0, ia * s,  0.0,      0.0,         ian * c],
        [0.0,         0.0, -ia * c, 0.0,      0.0,         ian * s],
    ])


def cw_to_roe(x_cw: np.ndarray, a: float, n: float, u: float = 0.0) -> np.ndarray:
    """Convert a CW/RTN state [x, y, z, xd, yd, zd] to ROE.

    Returns the non-dimensional ROE [da, dlam, dex, dey, dix, diy]. Pass
    a = 1.0 to get the "a-scaled" ROE a*dalpha in meters.
    """
    x_cw = np.asarray(x_cw, dtype=float).reshape(6)
    return cw_to_roe_matrix(a, n, u) @ x_cw


def roe_to_cw(roe: np.ndarray, a: float, n: float, u: float = 0.0) -> np.ndarray:
    """Convert ROE [da, dlam, dex, dey, dix, diy] to a CW/RTN state."""
    roe = np.asarray(roe, dtype=float).reshape(6)
    return roe_to_cw_matrix(a, n, u) @ roe


# --------------------------------------------------------------------------
# Keplerian ROE dynamics (near-circular)
# --------------------------------------------------------------------------

def roe_stm(n: float, t: float) -> np.ndarray:
    """Keplerian ROE state transition matrix Phi(t) (6x6).

    [1] Eq. 8 / [2] Eq. 34 with J2 = 0 (kappa = 0): only the mean-longitude
    drift dlam(t) = dlam(0) - 3/2 n t da(0) remains.
    """
    phi = np.eye(6)
    phi[1, 0] = -1.5 * n * t
    return phi


def roe_control_matrix(a: float, n: float, u: float) -> np.ndarray:
    """Control input matrix Gamma(u) (6x3): d(roe) = Gamma @ [dvR, dvT, dvN].

    Near-circular Gauss variational equations, [1] Eq. 10 / [2] Eq. 36.
    """
    s, c = np.sin(u), np.cos(u)
    return np.array([
        [0.0,  2.0,     0.0],
        [-2.0, 0.0,     0.0],
        [s,    2.0 * c, 0.0],
        [-c,   2.0 * s, 0.0],
        [0.0,  0.0,     c],
        [0.0,  0.0,     s],
    ]) / (n * a)


def reachable_dv_min(
    pseudo_state: np.ndarray, n: float, du: float
) -> dict[str, float]:
    """Reachable delta-v minima per plane ([2] Tables 6-7, near-circular).

    Parameters
    ----------
    pseudo_state : a-scaled pseudo-state a*dalpha_bar [m] (6,)
    n : chief mean motion [rad/s]
    du : reconfiguration window in argument of latitude, n*tf [rad]

    Returns a dict with keys 'da', 'dlambda', 'de', 'di' [m/s]. The in-plane
    minimum is the max of the first three ([2] Eq. 8).
    """
    d = np.asarray(pseudo_state, dtype=float).reshape(6)
    dv_da = 0.5 * abs(d[0])
    dv_de = 0.5 * float(np.hypot(d[2], d[3]))
    # [2] Table 6: |m*dlam - da| / |da0| with da0 = 2, dlam0 = -3 du, which
    # is the gauge of the tangential (dlam, da) parallelogram.
    dv_dl = abs(0.5 * d[0] + 2.0 * d[1] / (3.0 * du)) if du > 0 else np.inf
    dv_di = float(np.hypot(d[4], d[5]))
    return {
        "da": n * dv_da,
        "dlambda": n * dv_dl,
        "de": n * dv_de,
        "di": n * dv_di,
    }


# --------------------------------------------------------------------------
# Closed-form solver
# --------------------------------------------------------------------------

@dataclass
class RoeReconfigPlan:
    """Output of `roe_impulsive_reconfig`.

    Attributes
    ----------
    burns : [(t, dv)] with t in [0, tf] measured from the epoch of x0 and dv
        a 3-vector in CW/RTN axes [m/s]. Same format as Satellite.impulses.
    total_dv : sum of burn magnitudes [m/s]
    dv_min : reachable delta-v minimum per plane [m/s], keys 'da',
        'dlambda', 'de', 'di' ([2] Tables 6-7)
    dv_min_inplane, dv_min_oop : in-plane (max of da/dlambda/de) and
        out-of-plane minima [m/s]
    dominant : dominant in-plane ROE, 'da', 'dlambda' or 'de' ('none' if no
        in-plane change is required)
    inplane_optimal : True if the in-plane cost equals dv_min_inplane
        ([2] Table 11 scheme found); False for the [2] Sec. VI sub-optimal
        scheme
    inplane_dv, oop_dv : cost of the in-plane and out-of-plane burns [m/s]
        before burns at the same instant are merged
    pseudo_state : a-scaled pseudo-state a*dalpha_bar [m]
    arrival_error : |x(tf) - x_target| after applying the burns with the CW
        STM [m, m/s mixed norm], a self-check
    """

    burns: list[tuple[float, np.ndarray]]
    total_dv: float
    dv_min: dict[str, float]
    dv_min_inplane: float
    dv_min_oop: float
    dominant: str
    inplane_optimal: bool
    inplane_dv: float
    oop_dv: float
    pseudo_state: np.ndarray
    arrival_error: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def dv_lower_bound(self) -> float:
        """max over planes of the reachable minima ([2] Eq. 8)."""
        return max(self.dv_min_inplane, self.dv_min_oop)

    @property
    def inplane_ratio(self) -> float:
        """In-plane cost / reachable minimum (1.0 means optimal)."""
        if self.dv_min_inplane <= 0:
            return 1.0
        return self.inplane_dv / self.dv_min_inplane

    def summary(self) -> str:
        times = ", ".join(f"{t:.0f}s" for t, _ in self.burns)
        return (
            f"{len(self.burns)} burns at [{times}], total dv="
            f"{self.total_dv:.5f} m/s (in-plane {self.inplane_dv:.5f}, "
            f"oop {self.oop_dv:.5f}); dominant {self.dominant}, in-plane "
            f"dv_min={self.dv_min_inplane:.5f} m/s "
            f"({'optimal' if self.inplane_optimal else f'x{self.inplane_ratio:.3f}'})"
        )


def _candidate_locations(phase: float, u0: float, uf: float, tol: float):
    """All u = phase + k pi in [u0, uf], with sign (-1)^k relative to phase."""
    k_lo = int(np.ceil((u0 - phase) / np.pi - tol))
    k_hi = int(np.floor((uf - phase) / np.pi + tol))
    out = []
    for k in range(k_lo, k_hi + 1):
        u = min(max(phase + k * np.pi, u0), uf)
        out.append((u, 1.0 if k % 2 == 0 else -1.0))
    return out


# Cap on candidate locations kept for subset enumeration (first/last half);
# C(40, 4) ~ 9e4 small linear solves.
_MAX_CANDIDATES = 40


def _inplane_candidates(
    d: np.ndarray,
    u0: float,
    uf: float,
    rtol: float,
    augment: bool,
    extra_locations: tuple[float, ...] = (),
) -> np.ndarray:
    """Candidate tangential burn locations for the in-plane scheme.

    * u = Ubar + k pi, the optimal times of the dominant-de / dominant-da
      schemes ([2] Eq. 21 with J2 = 0, Table 10; [1] Eqs. 31-32).
    * augment=True also adds the window edges u0 and uf, the optimal times
      of the dominant-dlambda scheme ([2] Appendix B, Table 14), and their
      half-orbit shifts u0 + k pi and uf - k pi. A tangential burn moves de
      along (cos u, sin u) ([1] Eq. 10), so the shifted locations share the
      de direction of the edge burns. That lets the [2] Sec. VI search
      cancel the de side effect of a dlambda-type burn pair. In the same
      mode the cross-track burn locations (extra_locations) are added so
      in-plane and out-of-plane burns can share one impulse. This set
      extends the paper and contains its candidates.
    """
    scale = float(np.max(np.abs(d)))
    de_norm = float(np.hypot(d[2], d[3]))
    phase = float(np.arctan2(d[3], d[2])) if de_norm > rtol * scale else u0
    u = [c[0] for c in _candidate_locations(phase, u0, uf, 1e-9)]
    if augment:
        extra = [u0, uf, *extra_locations]
        k = 1
        while u0 + k * np.pi < uf - 1e-9:
            extra += [u0 + k * np.pi, uf - k * np.pi]
            k += 1
        for ue in extra:
            if all(abs(ue - uk) > 1e-9 for uk in u):
                u.append(ue)
    u = sorted(u)
    if len(u) > _MAX_CANDIDATES:
        half = _MAX_CANDIDATES // 2
        u = u[:half] + u[-half:]
    return np.array(u)


def _solve_inplane(
    d: np.ndarray,
    u0: float,
    uf: float,
    n: float,
    rtol: float,
    augment: bool = True,
    oop_burns: list[tuple[float, float]] = (),
) -> tuple[list[tuple[float, float]], float]:
    """Closed-form tangential scheme for the a-scaled in-plane pseudo-state.

    d = [a da, a dlam, a dex, a dey] [m]. Each tangential burn w_k = dv_T,k/n
    [m] at u_k changes the pseudo-state by ([1] Eqs. 8, 10 with J2 = 0)

        w_k * [2, -3 (u_f - u_k), 2 cos u_k, 2 sin u_k].

    Every subset of up to four candidate locations is solved as a small
    linear system ([2] Step 5: "choose maneuvers ... solve the linear
    system"). The exact solution with the least total delta-v is kept. If its
    in-plane cost equals the reachable minimum, this is the optimal [2]
    Table 11 scheme. Otherwise it is the [2] Sec. VI best sub-optimal scheme.

    The total counts merging with a cross-track burn at the same location
    (oop_burns = [(u, w_N)], w_N = dv_N / n): the combined impulse costs
    sqrt(w_T^2 + w_N^2) rather than |w_T| + |w_N|.

    Returns ([(u_k, w_k)], in-plane cost n * sum |w_k| [m/s]).
    """
    scale = float(np.max(np.abs(d)))
    if scale < 1e-12:
        return [], 0.0
    oop_u = tuple(uo for uo, _ in oop_burns)
    u = _inplane_candidates(d, u0, uf, rtol, augment, oop_u)
    # Cross-track magnitude sharing each candidate location (0 if none).
    o = np.zeros_like(u)
    for uo, wo in oop_burns:
        o[np.abs(u - uo) < 1e-9] = abs(wo)
    cols = np.vstack([
        2.0 * np.ones_like(u), -3.0 * (uf - u), 2.0 * np.cos(u), 2.0 * np.sin(u)
    ])  # (4, K)
    tol = rtol * scale + 1e-12

    best_cost, best_peak, best = np.inf, np.inf, None
    for size in range(1, min(4, len(u)) + 1):
        combos = np.array(list(itertools.combinations(range(len(u)), size)))
        A = np.transpose(cols[:, combos], (1, 0, 2))  # (m, 4, size)
        if size == 4:
            det = np.linalg.det(A)
            ok = np.abs(det) > 1e-10 * np.max(np.abs(A), axis=(1, 2)) ** 4
            combos, A = combos[ok], A[ok]
            if len(A) == 0:
                continue
            w = np.linalg.solve(A, np.broadcast_to(d, (len(A), 4))[..., None])[..., 0]
        else:
            w = np.einsum("mij,j->mi", np.linalg.pinv(A), d)
        resid = np.linalg.norm(np.einsum("mij,mj->mi", A, w) - d, axis=1)
        feas = resid <= tol
        if not np.any(feas):
            continue
        combos, w = combos[feas], w[feas]
        oc = o[combos]
        # Extra total delta-v this subset adds on top of the cross-track burns.
        cost = (np.hypot(w, oc) - oc).sum(axis=1)
        peak = np.abs(w).max(axis=1)
        # Least cost; among equal-cost schemes (dominant de has infinitely
        # many) the smallest peak burn, then the earliest locations.
        order = np.lexsort((peak, np.round(cost / tol)))
        i = order[0]
        if cost[i] < best_cost - tol or (
            cost[i] <= best_cost + tol and peak[i] < best_peak - tol
        ):
            best_cost, best_peak = float(cost[i]), float(peak[i])
            best = (combos[i], w[i])
    if best is None:
        raise ClosedFormUnavailable(
            f"no exact tangential scheme from {len(u)} candidate location(s) "
            f"in a {np.degrees(uf - u0):.0f} deg window; the closed-form "
            "scheme needs a longer window (about one orbit or more)"
        )
    combo, w = best
    burns = [(float(u[i]), float(wk)) for i, wk in zip(combo, w)
             if abs(wk) > 1e-12 * scale]
    return burns, n * float(np.abs(w).sum())


def _solve_oop(
    d: np.ndarray, u0: float, uf: float, n: float, rtol: float, choice: str
) -> list[tuple[float, float]]:
    """Cross-track burns for the a-scaled pseudo-state d = [a dix, a diy].

    Returns [(u_k, w_k)] with w_k = dv_N,k / n [m].
    """
    norm = float(np.hypot(d[0], d[1]))
    if norm < 1e-12:
        return []
    phase = float(np.arctan2(d[1], d[0]))
    cands = _candidate_locations(phase, u0, uf, 1e-9)
    if cands:
        # [1] Eq. 23 / [2] Table 10: one burn at the phase of di_bar (+k pi),
        # sign alternating with k.
        u, s = cands[0] if choice == "first" else cands[-1]
        return [(u, s * norm)]
    # Window < pi holds no aligned location: the hull boundary in that
    # direction is the chord between the window edges, so two burns at u0
    # and uf are optimal (cf. [2] Table 8, "Else" row).
    A = np.array([[np.cos(u0), np.cos(uf)], [np.sin(u0), np.sin(uf)]])
    if abs(np.linalg.det(A)) < 1e-12:
        raise ClosedFormUnavailable("window too short for out-of-plane control")
    w = np.linalg.solve(A, d)
    return [(u0, float(w[0])), (uf, float(w[1]))]


def roe_impulsive_reconfig(
    x0: np.ndarray,
    n: float,
    tf: float,
    x_target: np.ndarray | None = None,
    u0: float = 0.0,
    oop_location: str = "first",
    candidates: str = "augmented",
    rtol: float = 1e-9,
) -> RoeReconfigPlan:
    """Closed-form fuel-optimal impulsive reconfiguration (Chernick & D'Amico).

    Drop-in alternative to `cw_two_impulse`: transfers the CW state x0 at
    t = 0 to x_target (default: origin, at rest) at t = tf. The burn times
    and RTN delta-vs come from the ROE reachable-set solution of [1], [2]
    for Keplerian near-circular motion, which is the CW model.

    Parameters
    ----------
    x0 : initial CW/RTN state [m, m/s]
    n : chief mean motion [rad/s]
    tf : reconfiguration time [s]; burns occur in [0, tf]
    x_target : CW/RTN state to reach at tf (default zeros)
    u0 : chief argument of latitude at t = 0 [rad]. Only rotates the ROE
        vectors; burn times and delta-vs do not depend on it.
    oop_location : 'first' or 'last' admissible cross-track burn location
    candidates : in-plane burn-location set. 'paper' uses only the de-aligned
        locations u = Ubar + k pi of [2] Table 10, which reproduces the
        published schemes. 'augmented' (default) also allows the window
        edges u0, uf (the dominant-dlambda times of [2] Table 14) and their
        half-orbit shifts. It is a superset, so it is never worse, and it is
        markedly cheaper for dlambda-dominant or short-window
        reconfigurations.
    rtol : relative tolerance for accepting a burn subset's solution

    Returns
    -------
    RoeReconfigPlan whose `burns` list can be passed to
    `propagate_cw_with_impulses` or added to a Satellite (offset the times by
    the epoch of x0).

    Raises
    ------
    ClosedFormUnavailable if the window is too short for the scheme.
    """
    x0 = np.asarray(x0, dtype=float).reshape(6)
    x_target = (np.zeros(6) if x_target is None
                else np.asarray(x_target, dtype=float).reshape(6))
    if tf <= 0:
        raise ValueError("tf must be positive")
    du = n * tf
    uf = u0 + du

    # Pseudo-state in a-scaled ROE (a = 1 -> meters): [1] Eq. 15, [2] Eq. 2.
    roe0 = cw_to_roe(x0, 1.0, n, u0)
    roef = cw_to_roe(x_target, 1.0, n, uf)
    d = roef - roe_stm(n, tf) @ roe0

    dv_min = reachable_dv_min(d, n, du)
    inplane_keys = ("da", "dlambda", "de")
    dominant = max(inplane_keys, key=lambda k: dv_min[k])
    dv_min_in = dv_min[dominant]
    if dv_min_in <= 1e-12 * max(1.0, dv_min["di"]):
        dominant = "none"  # no in-plane change required

    if candidates not in ("augmented", "paper"):
        raise ValueError("candidates must be 'augmented' or 'paper'")
    oop_burns = _solve_oop(d[4:], u0, uf, n, rtol, oop_location)
    ip_burns, ip_cost = _solve_inplane(
        d[:4], u0, uf, n, rtol, augment=(candidates == "augmented"),
        oop_burns=oop_burns,
    )
    oop_cost = n * sum(abs(w) for _, w in oop_burns)

    # Build RTN impulses; merge burns that fall at the same instant (a
    # combined impulse never costs more than the two applied separately).
    merged: dict[float, np.ndarray] = {}

    def _add(u: float, dv: np.ndarray) -> None:
        t = float(np.clip((u - u0) / n, 0.0, tf))
        for tk in merged:
            if abs(tk - t) < 1e-6:
                merged[tk] = merged[tk] + dv
                return
        merged[t] = dv

    for u, w in ip_burns:
        _add(u, np.array([0.0, n * w, 0.0]))
    for u, w in oop_burns:
        _add(u, np.array([0.0, 0.0, n * w]))
    burns = sorted(merged.items(), key=lambda b: b[0])
    total = float(sum(np.linalg.norm(dv) for _, dv in burns))

    optimal = ip_cost <= dv_min_in * (1.0 + 1e-6) + 1e-12
    notes = []
    if not optimal:
        notes.append(
            f"in-plane target outside the nested reachable set (dominant "
            f"{dominant}); [2] Sec. VI sub-optimal scheme used, "
            f"cost x{ip_cost / dv_min_in:.4f} of dv_min"
        )

    xf = propagate_cw_with_impulses(x0, n, np.array([tf]), burns)[0]
    err = float(np.linalg.norm(xf - x_target))

    return RoeReconfigPlan(
        burns=[(t, dv) for t, dv in burns],
        total_dv=total,
        dv_min=dv_min,
        dv_min_inplane=dv_min_in,
        dv_min_oop=dv_min["di"],
        dominant=dominant,
        inplane_optimal=bool(optimal),
        inplane_dv=ip_cost,
        oop_dv=oop_cost,
        pseudo_state=d,
        arrival_error=err,
        notes=notes,
    )
