# AA279B Project — CW Formation-Flying Simulator

A modular Clohessy-Wiltshire (CW) relative-motion simulator for close-proximity
satellite formations in LEO, with side-by-side interactive 3D views of the
**ECEF frame** (textured Earth, labeled satellites with exaggerated formation
offsets) and the **CW/RIC frame** (chief at the origin), built on
[PyVista](https://pyvista.org).

Supports any number of satellites, impulsive maneuvers, and closed-form
two-impulse CW recovery — the baseline controller for the project's formation
recovery study.

## Install

```sh
pip install -r requirements.txt
```

## Quick start

```sh
python examples/demo_formation.py                 # interactive window
python examples/demo_formation.py --no-recovery   # unforced drift only
python examples/demo_formation.py --gif out.gif   # export animation
```

In the interactive window the slider scrubs time and **spacebar** plays/pauses.
The demo sets up three satellites at 550 km / 51.6°, equally spaced 1 km
along-track; the middle satellite is hit with a δv at t = 0, drifts through the
formation, then recovers to its slot with a two-impulse CW maneuver after a
half-orbit reaction delay.

## Package layout (`cwsim/`)

| Module | Contents |
|---|---|
| `constants.py` | μ⊕, R⊕, ω⊕ |
| `dynamics.py` | Full 3-DOF CW STM `cw_stm(n, t)`, unforced + impulsive propagation, closed-form two-impulse solver `cw_two_impulse` |
| `orbits.py` | `CircularOrbit` — analytic circular chief orbit in ECI |
| `frames.py` | RIC ↔ ECI transforms (DCM + transport theorem) |
| `satellites.py` | `Satellite` (initial state, color, scheduled impulses), `along_track_formation` builder |
| `simulation.py` | `Simulation.run()` → time histories in both frames |
| `visualization.py` | `FormationViewer` — dual-view PyVista window, GIF/screenshot export |
| `scp.py` | SCP formation-recovery planner (`plan_recovery`) + two-impulse baseline (`baseline_two_impulse_plan`), requires `cvxpy` |

## SCP formation recovery (`cwsim/scp.py`)

Plans impulsive burns for **all** satellites to restore an equally spaced
formation by `tf`, with slot re-ordering allowed (any permutation that
reproduces the original separations). Objective and constraints:

- **Minimize** total constellation Δv (sum of L2 impulse norms — a group
  lasso, so the solver chooses *how many* burns to use), plus a fuel-balancing
  term `w_balance · maxᵢ fuelᵢ` (mission life ends when the first satellite
  runs dry) and a time-of-flight weight `w_tf · tf` applied in an outer search
  over candidate recovery times.
- **Subject to** exact linear CW dynamics on a burn grid, terminal slot states
  (position *and* velocity), per-burn limit `‖Δv‖ ≤ Δv_max`, per-satellite
  fuel budget `Σ‖Δv‖ ≤ fuel_max`, no burns before the reaction delay, and
  pairwise keep-out spheres `‖rᵢ − rⱼ‖ ≥ d_min`.
- The keep-out constraint is the **only** non-convexity (CW dynamics are
  linear). It is linearized as the supporting halfspace of the keep-out
  sphere about the previous iterate — a conservative *inner* approximation —
  and imposed **continuously in time**: the inter-node coasting state is
  affine in the decision variables, so the constraint is sampled every
  `dt/n_sub` along every coast segment (plus a `collision_margin` pad on
  `d_min` covering the residual inter-sample excursion). A fine-grid
  post-check reports the realized minimum separation.
- Slot assignments are ranked by a closed-form two-impulse cost heuristic;
  the best few permutations are solved exactly.

Beware: the two-impulse **baseline** diverges when its transfer horizon
`tf − t_delay` is a multiple of the half orbit period (Φ_rv singular).

```sh
python examples/demo_scp_recovery.py   # SCP vs baseline + 3D visualization
python examples/sweep_recovery.py      # perturbation/delay trade study -> results/
```

## Conventions

- RIC (Hill) frame, loosely also called LVLH: **x** radial (zenith), **y**
  in-track (along-track), **z** cross-track (orbit normal); state
  `[x, y, z, ẋ, ẏ, ż]` in meters and m/s.
- Chief orbit is circular; mean motion `n = √(μ/a³)`.
- Impulses are instantaneous δv vectors in LVLH axes, applied via piecewise-STM
  propagation (exact, no numerical integration).

## Minimal API example

```python
import numpy as np
from cwsim import CircularOrbit, Satellite, Simulation, FormationViewer

chief = CircularOrbit(altitude=550e3, inclination=np.radians(51.6))
sat = Satellite("Deputy", x0=[0, 1000, 0, 0, 0, 0], color="cyan")
sat.add_impulse(0.0, [0.05, 0.1, 0.0])  # dv in m/s, LVLH axes

sim = Simulation(chief_orbit=chief, satellites=[sat])
result = sim.run(t_final=3 * chief.period, dt=15.0)
FormationViewer(result).show()
```

## Validation

```sh
python tests/test_dynamics.py
```

Checks the STM against RK4 integration of the CW ODEs (agreement to ~1e-12 m),
STM composition `Φ(t₁+t₂) = Φ(t₂)Φ(t₁)`, two-impulse rendezvous arrival error,
LVLH↔ECI round-trip consistency, and the along-track-offset equilibrium.
