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
    plan_recovery,
    propagate_cw_with_impulses,
)


def test_scp_restores_formation() -> None:
    chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
    n = chief.mean_motion
    sats = along_track_formation(spacing=1000.0, count=3)
    sats[1].add_impulse(0.0, np.array([0.0, 0.1, 0.0]))  # along-track kick

    t_delay = 0.25 * chief.period
    tf = 1.6 * chief.period
    cfg = ScpConfig(dt=120.0, n_sub=6, collision_margin=25.0, d_min=200.0)
    plan = plan_recovery(chief, sats, t_delay, tf, cfg, n_assignments=2)
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


if __name__ == "__main__":
    test_scp_restores_formation()
    print("\nSCP test passed.")
