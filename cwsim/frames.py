"""Frame transformations between ECI and the chief-centered RIC frame.

RIC (radial / in-track / cross-track, also called the Hill frame and loosely
LVLH) basis, expressed in ECI:
    x_hat = r / |r|            (radial)
    z_hat = (r x v) / |r x v|  (orbit normal / cross-track)
    y_hat = z_hat x x_hat      (along-track)
"""

from __future__ import annotations

import numpy as np


def lvlh_dcm(r_chief: np.ndarray, v_chief: np.ndarray) -> np.ndarray:
    """Rotation matrix whose columns are the LVLH basis vectors in ECI.

    Maps LVLH components to ECI: vec_eci = R @ vec_lvlh.
    """
    r = np.asarray(r_chief, dtype=float)
    v = np.asarray(v_chief, dtype=float)
    x_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    return np.column_stack([x_hat, y_hat, z_hat])


def relative_to_eci(
    x_rel: np.ndarray,
    r_chief: np.ndarray,
    v_chief: np.ndarray,
    n: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an LVLH relative state to an absolute ECI state.

    Uses the transport theorem with LVLH angular velocity omega = n * z_hat
    (exact for a circular chief orbit).
    """
    x_rel = np.asarray(x_rel, dtype=float).reshape(6)
    rho, rho_dot = x_rel[:3], x_rel[3:]
    dcm = lvlh_dcm(r_chief, v_chief)
    omega = np.array([0.0, 0.0, n])  # LVLH frame rotation rate, in LVLH axes

    r_eci = np.asarray(r_chief) + dcm @ rho
    v_eci = np.asarray(v_chief) + dcm @ (rho_dot + np.cross(omega, rho))
    return r_eci, v_eci


def eci_to_relative(
    r_dep: np.ndarray,
    v_dep: np.ndarray,
    r_chief: np.ndarray,
    v_chief: np.ndarray,
    n: float,
) -> np.ndarray:
    """Convert an absolute deputy ECI state to an LVLH relative state."""
    dcm = lvlh_dcm(r_chief, v_chief)
    omega = np.array([0.0, 0.0, n])

    rho = dcm.T @ (np.asarray(r_dep) - np.asarray(r_chief))
    rho_dot = dcm.T @ (np.asarray(v_dep) - np.asarray(v_chief)) - np.cross(omega, rho)
    return np.concatenate([rho, rho_dot])
