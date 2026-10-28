# Orbit reacquisition: unified user guide and technical documentation

**Documentation edition:** 28 October 2026. This is the sole documentation file for the repository. It combines the former root README, technical report, method notes, optional-mode guides, calibration roadmap and handover guides. Source code, example data and generated figures remain separate.

> **Scientific status:** Research prototype. Geometric orbit propagation and station-visibility calculations have basic test coverage. Single-family candidate fractions and global-family importance weights are **not calibrated probabilities of reacquisition**. The robust handover extension supports **known** and **single-family** modes only.

## Reading order

1. [Getting started and project structure](#1-getting-started-and-project-structure)
2. [Tracklets and deterministic orbit estimation](#2-tracklets-and-deterministic-orbit-estimation)
3. [Local Monte Carlo sampling](#3-local-monte-carlo-sampling)
4. [Single-family workflow](#4-single-family-workflow)
5. [Global-family workflow](#5-global-family-workflow)
6. [Known-orbit and approximate-range inputs](#6-known-orbit-and-approximate-range-inputs)
7. [Sky-search maps and telescope pointing](#7-sky-search-maps-and-telescope-pointing)
8. [Ground-station overlap and handover](#8-ground-station-overlap-and-handover)
9. [Calibration, sensitivity and limitations](#9-calibration-sensitivity-and-limitations)
10. [Implementation notes and testing](#10-implementation-notes-and-testing)

---


## 1. Getting started and project structure

The original root README is preserved here as the project overview and command reference. Detailed explanations follow in increasing order of complexity.

### Orbit reacquisition: angles-only orbit determination and sky-search maps

A reproducible **research prototype** for generating synthetic optical tracklets, estimating candidate Earth orbits, exploring provisional alternative orbit families, and producing sky-search maps for a user-defined observing station. The original **single-solution** and newer **global-family** workflows remain separate; the search-map CLI is their shared consumer.

**Current scope:** two-body dynamics, approximate rotating spherical Earth, geometric visibility and a sample-centered circular field-of-view optimizer. **Not** a production satellite-tracking system or calibrated detection-probability engine.

#### Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Generate search maps immediately from included example ensembles:
python sky_search_prototype/sky_search_maps.py --mode single \
  --input examples/mc_fitted_5_min.npz --lat -29.2567 --lon -70.7346 \
  --alt-km 2.4 --name La_Silla --hours 16 --step-minutes 20 \
  --fov-radius 1 --output results/single_chile.png

python sky_search_prototype/sky_search_maps.py --mode global \
  --input examples/stage_b_3_min.npz --lat 43.6045 --lon 1.444 \
  --alt-km 0.15 --name Toulouse --hours 12 --step-minutes 20 \
  --fov-radius 1 --output results/global_toulouse.png
```

**Outputs:** PNG with time-dependent visibility and best one-field mass, family contributions (global mode), and a local tangent-plane sky map; a companion CSV time series. Every plotted symbol has a legend. `PASS POSSIBLE` / `NO PASS` refer to **sampled** geometric visibility above the elevation cutoff, not guaranteed observability or detection.

#### Two independent simulation modes

| Mode | Input | Interpretation |
|---|---|---|
| `single` | `mc_fitted_*.npz` from the original fitter and local Monte Carlo | Equal-weight **candidate fractions** around one fitted solution; *not calibrated posterior probability*. |
| `global` | `stage_b_*.npz` from Stage A/B | **Provisional importance-weighted mass** across sampled candidate families; conditional on the Stage B prior, proposal, and finite sample coverage. |

**Do not concatenate the two ensembles or combine their weights.** A `--legacy-mc` overlay on the global dashboard is diagnostic only.

#### Pipeline and commands

##### Mode 1: original fit + local Monte Carlo

```bash
python demo_mc_fitted.py --quick --output-dir results/single
python demo_mc_fitted.py --draws 2000 --dashboard \
  --dashboard-orbit-tracks 25 --sky-prediction-minutes 120 \
  --output-dir results/single
```

This uses five cases: 3, 5, 10, 15 minutes and two 15-minute passes separated by approximately two hours. The 3-minute free-state fit fails with the current circular-seeded optimizer, so **no local-MC NPZ is generated for that case**. The other cases produce local ensembles.

##### Mode 2: global candidate families (Stage A) and weighted samples (Stage B)

```bash
python demo_stage_a.py --case 3_min --output-dir results/stage_a
python demo_stage_b.py --stage-a results/stage_a/stage_a_3_min.npz \
  --draws 15000 --output-dir results/stage_b
python demo_global_dashboard.py \
  --stage-a results/stage_a/stage_a_3_min.npz \
  --stage-b results/stage_b/stage_b_3_min.npz \
  --prediction-minutes 120 --output results/global_dashboard.png
```

Stage A searches a bounded range/range-rate grid and refines selected candidates. Family IDs are **provisional nearest-seed clusters**, not established disconnected posterior modes. Stage B is **multiple importance sampling**, not MCMC or sequential Monte Carlo. Its prior is data-centered, and low ESS limits inferential reliability.

##### Optional connectivity diagnostic (not used by default)

```bash
python demo_connected_regions.py \
  --stage-a examples/stage_a_3_min.npz \
  --stage-b examples/stage_b_3_min.npz \
  --output-dir results/connected
```

The existing Stage A grid is fixed-attributable, **not profiled** over angular nuisance parameters. Existing 3- and 5-minute tests found only three qualifying grid cells each and failed the minimum-cell criterion. The diagnostic therefore falls back to the original family labels. Do not treat the connectivity plot as proof of family separation.

##### Search maps for any station

```bash
python sky_search_prototype/sky_search_maps.py --mode global \
  --input examples/stage_b_3_min.npz \
  --lat -29.2567 --lon -70.7346 --alt-km 2.4 --name La_Silla \
  --hours 16 --step-minutes 20 --min-elevation 10 \
  --fov-radius 1 --map-radius 12 --output results/global_chile.png
```

Latitude/longitude are degrees (east-positive longitude); altitude is km. The map is a **local gnomonic tangent-plane view**, not a full-sky HEALPix map. A circle is the circular field of view; small cyan dots are individual propagated visible samples, and the cyan cross marks the best sample-centered pointing. The heatmap is smoothed **only for display**; field capture is computed from original weighted samples. The best pointing is optimized over **sample locations**, not all possible sky directions.

#### Repository map

```text
orbit_reacquisition_repo/
├── README.md, requirements.txt, LICENSE, .gitignore
├── orbit_toy.py, tracklet.py                 # Synthetic orbit/station foundations
├── orbit_fit_multipass.py, demo_multipass_fit.py
├── orbit_mc_fitted.py, demo_mc_fitted.py     # Original single-solution mode
├── orbit_dashboard.py, orbit_fitted_dashboard.py
├── orbit_stage_a.py, demo_stage_a.py, orbit_stage_a_dashboard.py
├── orbit_stage_b.py, demo_stage_b.py, orbit_stage_b_dashboard.py
├── orbit_global_dashboard.py, demo_global_dashboard.py
├── orbit_connected_regions.py, demo_connected_regions.py
├── orbit_station_search.py, demo_station_search.py
├── sky_search_prototype/sky_search_maps.py  # Shared CLI, both modes
├── orbit_stage_c.py                         # Reserved stub, NOT implemented
├── docs/                                    # Updated technical report, PDF and notes
├── examples/                                # Small saved datasets and example figures
└── tests/                                   # Smoke tests
```

#### Scientific interpretation and caveats

1. **No exhaustive global guarantee.** Stage A samples a finite range/range-rate grid; Stage B may miss important modes. Stage A's four 3-minute labels are provisional numerical clusters.
2. **Probabilities are conditional on the ensemble.** Single-mode maps are equal candidate fractions. Global-mode maps are provisional Stage B importance-weighted mass, **not calibrated detection probabilities**.
3. **Visibility is geometric only.** No weather, darkness, illumination, magnitude, limiting sensitivity, optical tracking, light-time, Earth flattening, J2 or detailed EOP transforms.
4. **Time resolution matters.** `NO PASS` means no ensemble orbit above the elevation threshold at **sampled times** within the specified horizon; short unsampled passes could be missed.
5. **Synthetic cases are not nested.** The 3-, 5- and 10-minute arcs use different reproducible noise realizations. The known truth is the same.
6. **Numerical reliability.** ESS and replicated posterior runs should be improved before assigning operational significance to global-family weights.
7. **Reference station.** Original observations are synthesized for 46.95° N, 7.44° E, altitude 0.55 km (Bern). Search stations can be specified independently.

See **docs/technical_report.md (included below)** or **docs/technical_report.pdf (included below)** for definitions, mathematical models, validation results, and search-map methods.

#### Tests

```bash
python -m unittest discover -s tests -v
```

This checks imports, example NPZ schema, weight normalization, and basic FoV logic. It does **not** constitute a full physics or statistical validation suite.

#### Additional observation modes (optional, separate)

The original single-solution and global-family workflows are unchanged. A new independent entry point `optional_modes/known_range_search.py` adds `--mode known` (one exact ECI-like state at a stated epoch) and `--mode range` (Gaussian topocentric-range conditioning of an existing orbit ensemble, retaining its prior weights). Both reuse the existing station-search renderer, with explicit sample-weight semantics. See optional_modes/README.md (included below) for commands, assumptions and examples. A range measurement cannot add missing orbit families; effective sample size is reported to flag weight collapse.

#### Standalone angular-speed sensitivity

Run `python analysis/angular_speed_sensitivity.py --output results/angular_speed_sensitivity.png`. This produces a four-panel figure showing angular speed, idealized field crossing time, radial-velocity sensitivity, and range/transverse-velocity degeneracy. This is **not** part of the main tooling. See analysis/README.md (included below).

#### Quick station search with optional approximate range

`search.py` is the single entry point for station search. **Without a range input it
uses the original single-family ensemble**; add `--mode global` to use the Stage B
ensemble. Adding both `--range-km` and `--range-sigma-km` reweights the chosen
ensemble without altering the original NPZ.

```bash
### La Silla, single-family fallback (no range measurement)
python search.py --lat -29.2567 --lon -70.7346 --alt-km 2.4 \
  --name La_Silla --output results/chile_single.png

### La Silla, single-family with hypothetical Bern range at the ensemble epoch
python search.py --lat -29.2567 --lon -70.7346 --alt-km 2.4 \
  --name La_Silla --range-km 20000 --range-sigma-km 1000 \
  --output results/chile_range.png

### La Silla, Stage B global-family posterior with the same range measurement
python search.py --mode global --lat -29.2567 --lon -70.7346 --alt-km 2.4 \
  --name La_Silla --range-km 20000 --range-sigma-km 1000 \
  --output results/chile_global_range.png
```

The ranging station defaults to Bern (46.95°N, 7.44°E, 0.55 km); change it using
`--range-lat`, `--range-lon`, `--range-alt-km`. The search station and ranging
station are independent. Range refers to **topocentric slant distance at the
saved state epoch**, not geocentric radius, altitude, or range at the later
search time. If the range observation was taken at another time, first propagate
the ensemble to that time before conditioning (not yet supported by this quick
entry point). Both range values must be given together.

Outputs: labeled PNG, CSV of time-dependent visible and best-FoV fractions, and
`*_conditioned.npz` when a range measurement is supplied. A low effective sample
size (ESS) warning means importance-weight collapse. This workflow cannot add
missing orbital families or produce calibrated probabilities from the original
local-MC sample. The search uses a finite time grid and candidate-centered
pointings, so it is not a continuous-time or globally optimal search guarantee.

##### Range observation at a different UTC time

The quick-entry `search.py` now supports `--range-epoch`. If omitted, the range
observation is interpreted at the ensemble's stored state epoch, preserving
previous behavior. An ISO 8601 UTC timestamp such as `2026-10-22T20:10:00`
propagates every state from the ensemble epoch to the observation epoch before
computing the topocentric slant range from `--range-lat`, `--range-lon`, and
`--range-alt-km`. The search station specified by `--lat` and `--lon` is separate.

```bash
python search.py --lat -29.2567 --lon -70.7346 --alt-km 2.4 \
  --name La_Silla --range-km 20000 --range-sigma-km 1000 \
  --range-epoch 2026-10-22T20:10:00 \
  --output results/chile_range_timestamped.png
```

The input timestamp is interpreted as UTC, without timezone conversion. The
current two-body propagator is deliberately approximate, and this entry point
rejects range epochs more than seven days from the state epoch. This guard is
not a guarantee of accuracy at shorter intervals. Low effective sample size
(ESS) means the range observation has collapsed the weights onto very few
existing candidates. It cannot create new orbits. If no range is provided,
`search.py` uses the unmodified single-family ensemble by default.

See `docs/probability_calibration_roadmap.md` for the distinction between
conditional ensemble mass and a calibrated detection probability.

#### Experimental posterior sky-coverage validation

A separate research diagnostic is available in `calibration/` (included below).
It re-runs Stage A and Stage B on independent synthetic noisy tracklets,
measures 50/90/95% future sky-direction containment, reports importance-sampling
ESS and failure rates, and exports plots/CSV/JSON. **This is a fixed-truth
repeated-noise experiment, not prior-predictive simulation-based calibration.**
The included 20-trial pilot (1,200 Stage B draws per trial) found empirical
containment 20% / 80% / 80% at nominal 50% / 90% / 95%, with median ESS 5.7.
These are unstable pilot diagnostics, not a calibrated probability estimate.

```bash
python calibration/run_coverage.py --trials 20 --draws 1200 \
  --output-dir calibration/results/my_run
```

#### Two-station direct handover (known-orbit geometric overlap)

Find an approximate **maximum ground-surface separation** for two freely located stations which can both observe the same known orbit for at least 2 minutes:

```bash
python handover/station_overlap.py \
  --orbit examples/mc_fitted_5_min.npz \
  --overlap-min 2 --min-elevation 10 \
  --hours 12 --window-step-min 15 \
  --output-dir results/handover
```

The result includes `summary.json` (station coordinates and UTC overlap window), `overlap_windows.csv` and `overlap_baseline.png`. The solver scans candidate overlap windows and a global station grid, then locally refines the most widely separated visible pair. `--grid-deg`, `--window-step-min`, and `--samples` control resolution. **Not a certified global optimum.** Assumes a known six-state orbit, two-body dynamics, spherical Earth, and geometric elevation only. Inferred orbit ensembles require a separate robust/probabilistic overlap criterion. The separation is the shortest great-circle surface distance, not line-of-sight distance or baseline chord.

#### Fixed-station robust handover (known orbit / single family)

Run `python handover/robust_station_overlap.py --mode known` or `--mode single` to search for the most distant second ground station supporting a scheduled 2-minute simultaneous visibility interval. Produces geographical coverage, separation-vs-coverage, timing plots, elevation margins and midpoint telescope FoV coverage at both stations. See handover/ROBUST_HANDOVER_README.md (included below) for commands, definitions and important sampling limitations. Global-family handover is intentionally excluded.


## 2. Tracklets and deterministic orbit estimation

This is the full technical report on the observation model, deterministic fitting, residuals, orbital elements, test cases, and numerical caveats. Its local Monte Carlo sections provide the foundation for the next chapter.

### Angles-only orbit determination: deterministic fitting and local Monte Carlo validation

**Technical report | 23 October 2026 | Current five-configuration experiment**

**Scope.** This document describes the **currently active** pipeline: synthetic optical observations, circular-orbit initialization, six-parameter deterministic fitting, local Monte Carlo generation of physically admissible measurement-compatible orbits, and diagnostic dashboards. It supersedes the earlier deterministic-only report. The old global Monte Carlo sampler, adaptive importance sampler and MCMC are **not** used.

#### 1. Objective and interpretation

Given a sequence of time-tagged right ascension and declination measurements from a known ground station, estimate the satellite's Earth-centered inertial-like Cartesian state, convert it into classical orbital elements, and investigate how the recovery changes with the length and separation of observation windows.

Two different outputs must be distinguished:

1. **Point estimate:** the best physically admissible orbit found by the nonlinear optimizer.
2. **Local admissible ensemble:** states sampled near that estimate which also satisfy a chosen angular-fit threshold and physical filters.

The second output is **not a globally complete set of compatible orbits** and **not a calibrated Bayesian posterior**. A failed point fit is not proof that the measurements contain insufficient information.

#### 2. Files and dependencies

| File | Role |
|---|---|
| `tracklet.py` | Older attributable utilities; imported by `orbit_toy.py`, though not directly used by the current fitter. |
| `orbit_toy.py` | Physical constants, approximate rotating station state and bound/perigee filtering. |
| `orbit_fit_multipass.py` | Circular seed search, six-dimensional nonlinear least squares, numerical propagation and classical-element conversion. |
| `demo_multipass_fit.py` | Synthetic truth and observation generation, plus the earlier three-case deterministic validation. |
| `orbit_mc_fitted.py` | Jacobian/SVD-based local Monte Carlo proposal and acceptance tests. |
| `demo_mc_fitted.py` | Five-case comparison and data/figure exports. |
| `orbit_dashboard.py` | Multi-panel visualizations of orbital tracks, residuals and orbital elements. |
| `orbit_fitted_dashboard.py` | Adapter between the local Monte Carlo result and the dashboard. |

Use the **latest** `demo_mc_fitted.py` containing the 3-minute case and the latest dashboard adapter supporting `max_orbit_tracks` and `sky_prediction_minutes`. NumPy, SciPy and Matplotlib are required.

#### 3. Notation and units

All vectors below are expressed in the same **approximate Earth-centered inertial (ECI-like)** Cartesian frame unless explicitly marked otherwise. Angles in equations are in radians; plots and file fields labeled `*_deg` use degrees.

| Symbol | Meaning | Units |
|---|---|---|
| \(t\) | Observation time | UTC timestamp |
| \(t_0\) | Epoch at which an orbit state is defined | UTC timestamp |
| \(T_0\) | Common synthetic truth initialization time, 2026-10-22 20:00:00 UTC | UTC timestamp |
| \(t_i\) | Time of observation \(i\), \(i=1,\ldots,N\) | UTC timestamp |
| \(\tau_i=t_i-t_0\) | Propagation interval relative to fit epoch | s |
| \(N\) | Number of paired RA/Dec observations | dimensionless |
| \(\mathbf r(t)\) | Geocentric satellite position | km |
| \(\mathbf v(t)\) | Geocentric satellite velocity | km/s |
| \(\mathbf x=(\mathbf r,\mathbf v)\) | Six-component Cartesian orbit state | km; km/s |
| \(\mathbf r_{\rm st}(t)\) | Station position expressed in the ECI-like frame | km |
| \(\mathbf v_{\rm st}(t)\) | Station velocity due to Earth rotation | km/s |
| \(\mathbf d(t)=\mathbf r(t)-\mathbf r_{\rm st}(t)\) | Topocentric line-of-sight vector | km |
| \(\rho=\|\mathbf d\|\) | Topocentric range | km |
| \(\dot\rho\) | Topocentric radial range rate | km/s |
| \(\hat{\mathbf u}=\mathbf d/\rho\) | Unit line-of-sight vector | dimensionless |
| \(\alpha,\delta\) | Right ascension and declination | rad |
| \(\alpha_i,\delta_i\) | Measured RA/Dec for observation \(i\) | rad |
| \(\hat\alpha_i(\mathbf x),\hat\delta_i(\mathbf x)\) | Predicted RA/Dec from a trial orbit | rad |
| \(\Delta\alpha_i,\Delta\delta_i\) | Wrapped RA and ordinary Dec prediction-minus-observation residuals | rad |
| \(\sigma_\alpha,\sigma_\delta\) | Assumed standard deviations of RA and Dec coordinate errors | arcsec or rad |
| \(\mu\) | Earth's gravitational parameter, 398600.4418 | km³/s² |
| \(R_E\) | Spherical Earth reference radius, approximately 6378.137 | km |
| \(\omega_E\) | Earth's rotation angular speed | rad/s |
| \(\theta_{\rm GMST}(t)\) | Approximate Greenwich mean sidereal angle | rad |
| \(\varphi,\lambda,h_{\rm st}\) | Station latitude, longitude, altitude | deg, deg, km |
| \(\chi^2(\mathbf x)\) | Sum of squared standardized angular residuals | dimensionless |
| \(\hat{\mathbf x}\) | Fitted state with smallest accepted \(\chi^2\) among attempted seeds | km; km/s |
| \(\mathbf x_{\rm circ}\) | Best circular initialization state | km; km/s |
| \(a,e,i,\Omega,\omega,M\) | Semimajor axis, eccentricity, inclination, right ascension of ascending node, argument of periapsis, mean anomaly | km, dimensionless, angles |
| \(r_p=a(1-e)\) | Perigee radius for a bound Keplerian ellipse | km |
| \(h_p=r_p-R_E\) | Perigee altitude | km |
| \(J\) | Jacobian of standardized angular residuals with respect to scaled state corrections | dimensionless in scaled coordinates |
| \(U,S,V^\mathsf T\) | Singular-value decomposition of \(J\) | as defined in §7 |
| \(s_j\) | Singular value \(j\) of \(J\) | dimensionless |
| \(q(\mathbf x)\) | Local Gaussian *proposal* density for candidate states | inverse state-space volume |
| \(n_{\rm draw}\) | Number of proposed Monte Carlo states | dimensionless |
| \(n_{\rm phys}\) | Number passing the physical orbit filter | dimensionless |
| \(n_{\rm acc}\) | Number passing both physical and angular-fit filters | dimensionless |

**Terminology:** The symbol \(\omega\) denotes *argument of periapsis*; \(\omega_E\) denotes *Earth's rotation rate*. These are unrelated. Likewise \(M\) is *mean anomaly*, not the number of measurements \(N\).

#### 4. Dynamics, Earth rotation and observation model

##### 4.1 Two-body dynamics

The fitted orbit is propagated using

\[
\frac{d\mathbf r}{dt}=\mathbf v,\qquad
\frac{d\mathbf v}{dt}=-\mu\frac{\mathbf r}{\|\mathbf r\|^3}.
\]

`propagate_at` uses SciPy's `solve_ivp` with DOP853 and dense output. It integrates from the state epoch forward and backward as needed, including across the approximately two-hour gap between observing windows. The state is **not reset** between windows.

##### 4.2 Moving ground station

The ground station is fixed on the simplified spherical Earth at

\[
(\varphi,\lambda,h_{\rm st})=(46.95^\circ,7.44^\circ,0.55\ {\rm km}).
\]

Its inertial-like position is recomputed at **each** observation time:

\[
\mathbf r_{\rm st}(t)=(R_E+h_{\rm st})
\begin{bmatrix}
\cos\varphi\cos[\theta_{\rm GMST}(t)+\lambda]\\
\cos\varphi\sin[\theta_{\rm GMST}(t)+\lambda]\\
\sin\varphi
\end{bmatrix}.
\]

The corresponding rotational velocity is
\[
\mathbf v_{\rm st}(t)=(0,0,\omega_E)\times\mathbf r_{\rm st}(t).
\]

Therefore, Earth rotation is **included** in the line-of-sight predictions for every tracklet and for the second observing window. The coordinates are ECI-*like*, not a fully rigorous ICRS/GCRS/J2000 transformation. UTC is approximated as UT1, and Earth flattening and detailed Earth orientation are omitted.

##### 4.3 Predicted optical angles

Given \(\mathbf d_i=\mathbf r(t_i)-\mathbf r_{\rm st}(t_i)\),

\[
\hat\alpha_i=\operatorname{atan2}(d_{y,i},d_{x,i})\pmod{2\pi},
\qquad
\hat\delta_i=\operatorname{atan2}
\left(d_{z,i},\sqrt{d_{x,i}^2+d_{y,i}^2}\right).
\]

The forward model does not currently apply light-time, aberration, refraction, optical visibility or illumination corrections.

#### 5. Synthetic orbit and five observation configurations

The **same synthetic satellite** is used in all experiments. At \(T_0\),

\[
\mathbf r(T_0)=\mathbf r_{\rm st}(T_0)+20\,000\ {\rm km}\,\hat{\mathbf u}_{\rm up},
\qquad
\mathbf v(T_0)=3.5\ {\rm km/s}\,\hat{\mathbf e}_{\rm east},
\]

where \(\hat{\mathbf u}_{\rm up}=\mathbf r_{\rm st}(T_0)/\|\mathbf r_{\rm st}(T_0)\|\) and
\(\hat{\mathbf e}_{\rm east}\) is the normalized cross product of the Earth-spin-axis unit vector with \(\hat{\mathbf u}_{\rm up}\).

This produces an **elliptical**, not circular, orbit with

\[
a_{\rm true}=22\,179.7086\ {\rm km},\quad
e_{\rm true}=0.1893162,\quad
i_{\rm true}=46.95^\circ,\quad
\Omega_{\rm true}=248.70256^\circ,\quad
\omega_{\rm true}=270^\circ.
\]

Mean anomaly \(M\) changes with the chosen fit epoch; comparisons of \(M\) are made at the same epoch.

At each observation time the truth is propagated, converted into topocentric RA/Dec, and perturbed by independent normal errors:

\[
\alpha_i=\alpha_{\rm true}(t_i)+\epsilon_{\alpha,i},\quad
\delta_i=\delta_{\rm true}(t_i)+\epsilon_{\delta,i},
\quad
\epsilon_{\alpha,i},\epsilon_{\delta,i}\sim\mathcal N(0,\sigma^2),
\]
where \(\sigma=5\) arcsec **in each angular coordinate**.

| Case | \(N\) | Observation times relative to \(T_0\) | Noise generation |
|---|---:|---|---|
| 3 min | 20 | 0–180 s | Separate reproducible random seed 303 |
| 5 min | 30 | 0–300 s | Separate reproducible random seed 305 |
| 10 min | 45 | 0–600 s | Separate reproducible random seed 310 |
| 15 min | 60 | 0–900 s | Historical sequence from seed 3 |
| Two passes | 120 | 0–900 s and 7200–8100 s | Historical sequence from seed 3, following the earlier 70 s and 15 min draws |

The old **70-second** case had eight observations over 0–70 s. It is retained as a historical comparison, but it is **not one of the five cases** in the latest Monte Carlo demo.

**Controlled-comparison caveat:** The experiments share the true orbit and noise standard deviation, but **do not share an identical nested set of noisy observations**. Measurement counts and noise realizations differ, so changes in fit accuracy cannot be attributed solely to arc duration. The time arrays are rounded to whole seconds, which can introduce small cadence variations.

#### 6. Deterministic orbit fitting

##### 6.1 Circular seed search

A single angular tracklet directly constrains direction and angular motion, but not the topocentric range \(\rho\) and range rate \(\dot\rho\) separately. The algorithm uses the measured angular direction at its chosen fit epoch to form trial positions

\[
\mathbf r=\mathbf r_{\rm st}(t_0)+\rho\hat{\mathbf u}.
\]

For each trial position, it imposes a *temporary* circular-orbit velocity:

\[
\mathbf v_{\rm circ}
=\sqrt{\frac{\mu}{\|\mathbf r\|}}
\bigl(\cos\psi\,\mathbf e_1+\sin\psi\,\mathbf e_2\bigr),
\]

where \(\mathbf e_1,\mathbf e_2\) are orthonormal vectors perpendicular to \(\mathbf r\), and \(\psi\) is a trial in-plane velocity angle. Here \(\psi\) is a **seed-search parameter**, not the orbital argument of periapsis \(\omega\).

The demo uses seven coarse trial ranges, four trial phases, nonlinear two-parameter refinement of promising circular seeds, and up to four subsequent free-state starts. The range bounds are 300–60,000 km. The circular constraint is **removed** in the final fit.

##### 6.2 Free six-state least squares

The optimizer adjusts all components of

\[
\mathbf x(t_0)=(r_x,r_y,r_z,v_x,v_y,v_z)^\mathsf T.
\]

RA residuals are wrapped to \([-\pi,\pi)\):

\[
\Delta\alpha_i=
\operatorname{atan2}\!\left[
\sin(\hat\alpha_i-\alpha_i),
\cos(\hat\alpha_i-\alpha_i)
\right],\qquad
\Delta\delta_i=\hat\delta_i-\delta_i.
\]

The standardized residual vector, ordered RA then Dec for each timestamp, is

\[
\mathbf f(\mathbf x)=
\left(
\frac{\Delta\alpha_1}{\sigma_\alpha},
\frac{\Delta\delta_1}{\sigma_\delta},
\ldots,
\frac{\Delta\alpha_N}{\sigma_\alpha},
\frac{\Delta\delta_N}{\sigma_\delta}
\right)^\mathsf T,
\qquad
\chi^2(\mathbf x)=\mathbf f^\mathsf T\mathbf f.
\]

**Important:** RA errors are *not* multiplied by \(\cos\delta\); this matches the synthetic noise model, which adds noise directly to RA coordinates.

The optimizer is SciPy `least_squares(method="trf")`. For numerical scaling, Cartesian corrections use

\[
\mathbf D=\operatorname{diag}(1000,1000,1000,1,1,1),
\qquad
\mathbf x=\mathbf x_{\rm seed}+\mathbf D\mathbf z,
\]

where \(\mathbf z\) is dimensionless, the first three diagonal values are in km, and the last three are in km/s.

The returned solution must pass the physical filter: a **bound** Keplerian orbit with perigee altitude \(h_p\ge100\) km. This filter is applied **after** each unconstrained optimization, not enforced continuously during it. If no physically admissible free solution is obtained, the routine returns its best circular seed and marks `success=False`.

A low \(\chi^2\) does not by itself prove that the orbital elements are accurate, especially for short arcs.

#### 7. Local Monte Carlo sampling around a successful fit

##### 7.1 Linearized local sensitivity

For a successful deterministic fit \(\hat{\mathbf x}\), define the dimensionless correction \(\mathbf z\) through \(\mathbf x=\hat{\mathbf x}+\mathbf D\mathbf z\). Evaluate the Jacobian

\[
J_{ij}=\left.\frac{\partial f_i}{\partial z_j}\right|_{\mathbf z=\mathbf 0},
\qquad
J=USV^\mathsf T,\qquad
S=\operatorname{diag}(s_1,\ldots,s_6).
\]

The code estimates \(J\) by centered finite differences. Under an ideal locally linear Gaussian-noise model, the unregularized inverse normal matrix would be \((J^\mathsf TJ)^{-1}\). However, weakly constrained directions make that unstable.

The implemented proposal instead regularizes singular values:

\[
\tilde s_j=\max(s_j,s_{\rm floor}),\qquad
s_{\rm floor}=\max(s_1\times10^{-7},10^{-12}),
\]
\[
b_j=\min(1/\tilde s_j,20),\qquad
C_z=V\,\operatorname{diag}(b_j^2)\,V^\mathsf T.
\]

Here \(C_z\) is the **proposal covariance in scaled coordinates**. The maximum standard deviation 20 is a numerical cap, not a physical uncertainty bound.

##### 7.2 Drawing candidate states

For each independent draw \(k\),

\[
\boldsymbol\xi_k\sim\mathcal N(\mathbf0,I_6),\qquad
\mathbf z_k=V\,\operatorname{diag}(b_j)\boldsymbol\xi_k,
\qquad
\mathbf x_k=\hat{\mathbf x}+\mathbf D\mathbf z_k.
\]

The optional `proposal_scale` multiplies the local corrections; its default is 1. These are **independent Gaussian proposal draws**, not a Markov chain. The proposal can be extremely broad in poorly identified directions and may place many samples on nonphysical orbits.

##### 7.3 Physical and angular-fit acceptance

First reject any orbit that is unbound or has \(h_p<100\) km. For the remaining candidates, compute the **full nonlinear** angular residual vector and accept only if

\[
\chi^2(\mathbf x_k)\le
\chi^2(\hat{\mathbf x})+
F^{-1}_{\chi^2_6}(0.95)
\approx\chi^2(\hat{\mathbf x})+12.592,
\]

where \(F^{-1}_{\chi^2_6}(0.95)\) is the 95th percentile of a chi-square distribution with **six** degrees of freedom. This is a **chosen local likelihood-ratio-style threshold**, not a guarantee of 95% frequentist coverage for the actual weakly identified nonlinear problem.

The acceptance rate is \(n_{\rm acc}/n_{\rm draw}\). It is a **proposal-efficiency statistic**, not the probability that the fitted orbit is correct. Accepted samples are **not reweighted** by their likelihood or proposal density; their histograms show the truncated proposal distribution.

The sampler deliberately refuses to run when `fit.success=False`, so the 3-minute case currently has **no** Monte Carlo ensemble.

#### 8. Five-case results

The following values are from the saved quick comparison, using **150 proposal draws per successful fit**. These are observed numerical results for this specific synthetic orbit and noise realization, not general performance statistics.

| Case | \(N\) | Fit success | Fitted \(\chi^2\) | \(a_{\rm fit}\) (km) | \(e_{\rm fit}\) | Physical draws | Accepted draws |
|---|---:|---|---:|---:|---:|---:|---:|
| 3 min | 20 | No | 68.43 (circular fallback) | 28,053.31 (fallback) | ≈0 (fallback) | 0 (not run) | 0 (not run) |
| 5 min | 30 | Yes | 38.96 | 20,496.81 | 0.256108 | 41 | 39 |
| 10 min | 45 | Yes | 75.97 | 26,876.85 | 0.034519 | 143 | 135 |
| 15 min | 60 | Yes | 125.48 | 21,989.74 | 0.196576 | 150 | 138 |
| Two passes | 120 | Yes | 230.26 | 22,181.12 | 0.189264 | 150 | 141 |
| **Truth** | | | | **22,179.71** | **0.189316** | | |

The **3-minute** fit did not yield a physically admissible *free* six-state result from the attempted circular seeds; the tabulated state is the circular fallback and is **not** a recovered orbit. The earlier **70-second** attempt likewise did not recover the truth reliably.

The **5-minute** and **10-minute** fits both found physical solutions but not the true orbital elements accurately. In fact, the 10-minute semimajor axis is farther from the truth than the 5-minute estimate. This nonmonotonic behavior is consistent with the combined effects of weak observability, different noise realizations, and local optimization.

The **15-minute** result is considerably closer, while the **two-window** fit recovers the truth very accurately. In the two-window case:

\[
a_{\rm fit}=22\,181.12\ {\rm km},\quad
e_{\rm fit}=0.189264,\quad
i_{\rm fit}=46.95012^\circ,\quad
\Omega_{\rm fit}=248.70248^\circ,\quad
\omega_{\rm fit}=269.99182^\circ.
\]

The two windows are at 0–15 min and 120–135 min, and Earth rotation is accounted for at each timestamp. The success illustrates the value of a longer temporal baseline and changing geometry **for this particular synthetic case**.

##### 8.1 What “failure” means here

There are three different possible outcomes:

- **Optimization failure:** no acceptable free-orbit solution is returned, as in the 3-minute test.
- **Accurate fit to angles but inaccurate elements:** a compatible orbit is recovered, but its range/velocity and orbital elements differ from truth, as in the 5- and 10-minute tests.
- **Successful state recovery:** fitted orbit and truth agree closely, as in the two-window test.

The 3-minute result does **not** establish a fundamental three-minute observability threshold. To identify such a threshold, we would need stronger global initialization, multiple noise realizations and nested observation sets.

#### 9. Dashboard interpretation

With `--dashboard`, the demo can generate a multi-panel figure for **each successful case**. The panels show:

1. Three-dimensional Earth-centered orbit trajectories, the fitted/representative trajectory and selected alternative accepted states.
2. On-sky RA/Dec tracks from the observer, including optional prediction beyond the observed arc.
3. Angular observations and residuals.
4. Histograms of the six classical orbital elements.
5. Pairwise element-density/correlation panels.
6. A sampling summary.

The orbital-element plots use locally adjusted display ranges so narrow distributions remain visible. Angular variables \(\Omega,\omega,M\) require wraparound handling near 0°/360°. The dashboard can show many nearly overlapping curves during the observed arc **because all accepted orbits were selected to reproduce those observations**. Extrapolating beyond the measurements makes their divergence easier to see.

The optional parameters `--dashboard-tracks`, `--dashboard-orbit-tracks`, and `--sky-prediction-minutes` control sky-track count, 3D-track count, and prediction duration respectively, in the latest dashboard-enabled version.

**Crucial:** Filled contours and histograms show the density of accepted **local proposal draws**, not posterior probability, and do not include disconnected solution families missed by the deterministic fit.

#### 10. Reproduction

Keep the current Python modules listed in §2 in the repository root and install NumPy, SciPy and Matplotlib.

Quick comparison (150 draws per successful case):

```bash
python demo_mc_fitted.py --quick
```

Fuller local ensembles and dashboards:

```bash
python demo_mc_fitted.py \
  --draws 2000 \
  --dashboard \
  --dashboard-tracks 60 \
  --dashboard-orbit-tracks 25 \
  --sky-prediction-minutes 120 \
  --output-dir results
```

Outputs include `mc_fitted_summary.csv`, `mc_fitted_comparison.png`, `mc_fitted_<case>.npz` for successful fits, and `mc_fitted_dashboard_<case>.png` when requested. NPZ files contain accepted six-component states at the saved epoch, their \(\chi^2\) scores, the reference fitted state, the acceptance threshold, and classical orbital elements.

**Code-version note:** Earlier copies of `orbit_fitted_dashboard.py` lack the `max_orbit_tracks` and `sky_prediction_minutes` arguments. Use the latest dashboard adapter and plotting module together with the five-case demo. The older `demo_multipass_fit.py` still writes its own validation outputs to `/mnt/data/`; the newer Monte Carlo demo supports `--output-dir`.

#### 11. Limitations and recommended next tests

1. **Short-arc observability:** A single brief RA/Dec arc may constrain angular motion well while leaving topocentric range and radial velocity poorly determined.
2. **Local optimizer dependence:** The finite circular seed grid and post-fit physical filtering can miss admissible non-circular solutions. Failure of `fit.success` is not a proof of physical non-identifiability.
3. **Local-only sampling:** The Gaussian proposal explores a neighborhood of one fit and does not establish global completeness.
4. **Statistical calibration:** The chosen \(\Delta\chi^2\) cutoff is approximate; accepted proposal draws are not posterior samples or confidence-interval realizations.
5. **Measurement-model consistency:** The current synthetic noise is Gaussian in the RA and Dec **coordinates**, not necessarily the correct tangent-plane error model for real astrometry.
6. **Simplified physics:** Two-body dynamics, spherical Earth, approximate GMST/UTC, and omission of J2, atmospheric effects, light-time and rigorous reference-frame transforms limit real-world fidelity.
7. **Validation design:** Use nested observation sets with identical measurements over shared intervals, many independent noise seeds, and multiple truth orbits before claiming a minimum recoverable arc duration.
8. **Prediction validation:** Compare sky-plane spread against withheld later observations or a second station, not only against the tracklets used in fitting.

**Conclusion.** The current pipeline successfully recovers the known synthetic elliptical orbit from a sufficiently informative two-window optical dataset and constructs local physically admissible orbit ensembles around successful fits. The 3-minute case fails at deterministic initialization/refinement; 5- and 10-minute cases produce compatible but biased element estimates; and the 15-minute and especially two-window cases recover the truth more closely. These are results for one controlled toy-model scenario, not universal observation-duration thresholds.


---

### Supplement: Global orbit families and station sky-search maps (27 October 2026)

#### 12. Relationship between the two independent pipelines

The original **single** mode performs circular-seeded deterministic fitting, then local Monte Carlo around one accepted free-state solution. Its samples are retained with equal display weight. The newer **global** mode performs Stage A bounded range/range-rate exploration, Stage B multiple importance sampling around discovered candidate states plus a broad proposal component, and a separate global-family dashboard. Neither mode alters the other. Both may be supplied to the same sky-search plotting CLI, but their samples must **never be concatenated or jointly renormalized**.

Stage A candidate families are provisional clustering labels; they are not proven topologically disconnected likelihood components. The optional connected-region diagnostic labels 8-connected qualifying cells of the existing fixed-attributable Stage A grid, which is not a profiled likelihood. In tests, only three cells qualified for each of the 3- and 5-minute cases, below the ten-cell minimum; the diagnostic fell back to the original family labels. This diagnostic is disabled by default.

#### 13. Stage B probability model and limitations

Stage B works in transformed coordinates consisting of the four angular attributable parameters, log(topocentric range), and topocentric range rate. Its current prior is Gaussian around the observed attributable (with configurable width), uniform in log-range and range rate within explicit bounds, and truncated by bound-orbit/perigee filters. Because this prior is centered using the observed data, the resulting weights should be described as **provisional prior-dependent importance weights**, not a rigorously calibrated posterior from a data-independent population prior.

For proposal density q(x), prior density p(x), measurement likelihood L(x), and physical admissibility indicator I_phys(x), the unnormalized importance weight is

    w_raw(x) = L(x) p(x) I_phys(x) / q(x),
    L(x) proportional to exp[-chi2(x)/2].

Normalize weights as w_k = w_raw,k / sum_j w_raw,j, with sum_k w_k = 1. Effective sample size is ESS = 1 / sum_k w_k^2. The saved 3-minute Stage B example used 15,000 proposals with ESS about 161; the 5-minute example used 8,000 proposals with ESS about 33.5. Low ESS and incomplete mode coverage mean that reported family percentages are provisional.

#### 14. Propagation to an arbitrary observing station

The station is specified by geodetic-style latitude phi (degrees), east-positive longitude lambda (degrees), and altitude h (km). In the current toy model, the Earth is spherical and the station position r_st(t) is rotated with approximate GMST, as described in Section 4. Each candidate state x_k at the saved common epoch t0 is propagated to requested times t_j under two-body gravity. The station-to-object vector is d_kj = r_k(t_j) - r_st(t_j), and the topocentric unit direction is u_kj = d_kj / ||d_kj||.

The geometric elevation e_kj is the angle between u_kj and the station's local horizon plane. Define V_kj = 1 if e_kj is at least the user-defined minimum elevation e_min, otherwise 0. With normalized ensemble weights w_k, the weighted visible fraction at time t_j is

    P_visible(t_j) = sum_k w_k V_kj.

In **single** mode, w_k = 1/N, so this is an *equal candidate fraction*, not a calibrated probability. In **global** mode, w_k are the provisional Stage B importance weights.

#### 15. Field-of-view capture and sky maps

For a telescope center c (a unit direction vector) and circular field-of-view radius theta (radians), define the angular separation gamma_kj(c) = arccos(clip(u_kj dot c, -1, 1)). The weighted mass captured by one pointing at time t_j is

    P_FOV(c,t_j) = sum_k w_k V_kj 1[gamma_kj(c) <= theta].

The prototype chooses c from the **visible sample directions** and maximizes P_FOV over that finite candidate set. This is not guaranteed to be the continuous-sky optimum. At each sampled time, it computes P_visible and the best sample-centered P_FOV, then selects a time for the local sky map. The map uses a gnomonic tangent-plane projection centered on the selected pointing, with east/north offsets in degrees. The heatmap is smoothed for display only; numerical FoV masses use unsmoothed samples.

**Figure symbols:** Small cyan dots represent individual propagated visible orbit samples; the cyan cross marks the selected pointing center; the cyan circle shows the user-defined circular telescope FoV; background color indicates smoothed weighted sample density; the vertical dotted time marker indicates the map's selected time. In the time-series panel, visibility and best-FoV curves are fractions/masses of the **full ensemble**, not conditional percentages among visible samples. Family-colored curves show each family's weighted visible contribution.

A `PASS POSSIBLE` flag means at least one candidate is above the minimum elevation at a sampled time. `NO PASS` means none meets the threshold at the sampled times, **not** a rigorous continuous-time proof that no pass occurs. A single tiny-weight outlier can trigger `PASS POSSIBLE`. No optical detectability model (weather, illumination, brightness, sensor sensitivity or trailing) is included.

#### 16. Files, reproduction and future validation

The repository includes the original `demo_mc_fitted.py` and `orbit_dashboard.py`, Stage A/B demos and separate global dashboard, optional connectivity diagnostic, and `sky_search_prototype/sky_search_maps.py`. The original station is Bern (46.95 N, 7.44 E, 0.55 km). Example target stations are Toulouse (43.6045 N, 1.444 E, 0.15 km) and La Silla (-29.2567 N, -70.7346 E, 2.4 km).

To reproduce search maps from bundled examples, run `python sky_search_prototype/sky_search_maps.py --mode global --input examples/stage_b_3_min.npz --lat -29.2567 --lon -70.7346 --alt-km 2.4 --hours 16 --step-minutes 20 --fov-radius 1 --output results/global_chile.png`. Replace `--mode global` and its input with `--mode single --input examples/mc_fitted_5_min.npz` to visualize local candidate fractions.

Next steps include nested shared-noise tracklets, stronger global search and independent importance-sampling convergence checks, physically motivated priors, full-sky HEALPix density maps, continuous pointing optimization, multi-exposure scheduling and detection models. These are **not implemented** in the present prototype. `orbit_stage_c.py` remains a reserved stub; the independent sky-search prototype is the current implementation of geometric station reacquisition.


## 3. Local Monte Carlo sampling

The local sampler proposes physically admissible candidate states around a fitted solution. This is not an exhaustive or calibrated posterior.

### Implementation and provenance notes

- Consolidated from the independently developed original, Stage A, Stage B, global dashboard, connectivity and sky-search modules.
- Existing algorithms were not rewritten during consolidation.
- `orbit_stage_c.py` is an unimplemented placeholder; geometric search maps are implemented in `sky_search_prototype/sky_search_maps.py`.
- Existing figures and example NPZ files are bundled as reproducibility fixtures.
- The old technical report is preserved as `technical_report_original.pdf`; the current PDF appends an updated technical supplement.
- Import layout: core modules at repository root; search CLI in `sky_search_prototype/` inserts its parent into the Python import path.
- Original single-mode Monte Carlo and global weighted samples are separate.


## 4. Single-family workflow

The single-family mode uses the fitted state and local candidate ensemble. The optional-mode guide also documents known-orbit and range inputs; those commands are repeated in context in Chapter 6.

### Optional known-orbit and approximate-range modes

These are **additive**: existing `--mode single` and `--mode global` in `sky_search_prototype/sky_search_maps.py` are unchanged.

Run from repository root:

```bash
python optional_modes/known_range_search.py --mode known \
  --state 12000 18000 10000 -1 2.5 1 \
  --epoch 2026-10-22T20:00:00 --lat 43.6045 --lon 1.444 \
  --name Toulouse --hours 12 --output results/known_toulouse.png
```

The state is `[rx ry rz vx vy vz]` in km and km/s, in the project's approximate ECI-like frame. It is a **hypothetical** example, not a validated orbit.

```bash
python optional_modes/known_range_search.py --mode range \
  --input examples/stage_b_3_min.npz \
  --range-km 20000 --range-sigma-km 1000 \
  --range-lat 46.95 --range-lon 7.44 --range-alt-km 0.55 \
  --lat -29.2567 --lon -70.7346 --alt-km 2.4 \
  --name La_Silla --hours 16 --output results/range_chile.png
```

Range is **topocentric** from the specified ranging station at the **input ensemble epoch**, not the future search station and not geocentric altitude. It is assumed independent of the angular data and Gaussian in range. Existing weights are preserved and updated by Bayes' rule. For single-MC input, equal weights are an **assumption about the candidate set**, not a calibrated posterior. The script prints effective sample size after conditioning. If it collapses, the ensemble does not support reliable range conditioning.

The output includes a labeled sky-search PNG, CSV and `_conditioned.npz` file. The search module's `global` rendering path is used for both modes to preserve supplied weights; this does **not** turn a single solution into a global posterior.

`PASS POSSIBLE` is a geometric flag for the sampled time grid and elevation limit; `NO PASS` does not prove the orbit never rises at other times or between grid steps. Optical detection is not modeled.


## 5. Global-family workflow

Stage A searches admissible range/range-rate hypotheses and Stage B importance-weights candidate orbits. The calibration roadmap below explains why family probabilities remain provisional.

### From candidate fractions to calibrated reacquisition probabilities

#### Current status

- **Single-family:** local Monte Carlo candidates with equal base weights are
  not demonstrated to be posterior samples. Their fractions are *candidate
  coverage*, not calibrated probability.
- **Global-family:** Stage B importance-weighted candidate states approximate
  a prior-dependent posterior, but proposal coverage, low ESS, and provisional
  family identification prevent a calibration claim.
- **Range-conditioned:** a Gaussian slant-range likelihood is applied to the
  existing weighted candidates at an independently specified observation time.
  ESS can collapse. No new orbital families are discovered by reweighting.
- **Known orbit:** a delta distribution at the supplied state gives deterministic
  geometric pointing predictions, not a nonzero-error detection probability.

#### Posterior

Let x be a six-dimensional state at a fixed reference epoch, D_a the optical
tracklet, D_r an independent slant-range measurement, and pi(x) an explicit
prior over physically admissible orbits. Then

    p(x | D_a,D_r) proportional to L_a(D_a|x) L_r(D_r|x) pi(x).

For independent, correctly specified Gaussian errors:

    log L = -0.5 * (chi2_angular + ((rho(x,t_r)-rho_obs)/sigma_r)**2) + constant.

Use the correct angular error model (including the RA cos(dec) metric or a
unit-vector tangent-plane residual), observation correlations, timestamps,
station position uncertainties, and any known selection effects. If the range
measurement is derived from the same optical observations, independence fails
and double counting must be avoided.

Importance samples from a proposal q(x) need weights proportional to
L_a L_r pi/q. A naive uniform range/range-rate grid or equal weighting of
local optimization solutions is not automatically a physical prior or posterior.

#### Predictive sky probabilities

Propagate each posterior state to time t and transform to an observer-centered
unit line-of-sight u(x,t). For a telescope pointing c, angular radius alpha,
and visibility condition V(x,t), the geometric posterior mass is

    P_geom(c,t) = integral 1[acos(u dot c)<=alpha] 1[V] p(x|D) dx.

A weighted sample approximates this integral by summing normalized posterior
weights of samples satisfying both indicators. This is a conditional sky-coverage
probability, not a detection probability.

#### Detection probability

Model detection efficiency eta(x,c,t,exposure) in [0,1], accounting for limiting
magnitude, illumination, phase, apparent angular rate and trailing, weather,
background, sensor efficiency, and operational constraints:

    P_detect(c,t) = integral 1[in FoV] 1[visible] eta(x,c,t) p(x|D) dx.

For multiple pointings, calculate the union/conditional sequential probability,
not the sum of overlapping single-pointing probabilities. Update the posterior
with a non-detection likelihood after each failed search.

#### Recommended validation before using probabilities operationally

1. Define the target population and physical prior, including orbit classes,
   altitude/perigee constraints, and sensor selection effects.
2. Verify the angular likelihood and clock/station uncertainties with synthetic
   recovery experiments across LEO, MEO, GEO and eccentric regimes.
3. Replace or supplement sparse importance proposals with adaptive sampling,
   SMC or MCMC that explores all admissible connected regions. Track ESS and
   convergence; ensure the true region is represented.
4. Conduct simulation-based calibration: generate orbits from the stated prior,
   synthesize noisy tracklets and range observations, infer posterior predictive
   sky maps, and check empirical 50%, 90%, 95% region coverage.
5. Validate against held-out real multi-station tracks; quantify coverage,
   false-confidence cases, and ephemeris model errors.
6. Calibrate eta using telescope injections/known targets and actual observing
   conditions. Only then call P_detect a detection probability.

#### Time-tagged range implementation

`search.py --range-epoch 2026-10-22T20:10:00` propagates each original state
from its stored epoch to that UTC timestamp using the existing two-body RK4
model. It computes topocentric range using the specified *ranging* station at
that time, updates normalized weights, and retains original-epoch states for
subsequent sky-search propagation. A missing `--range-epoch` means the state
epoch. A missing range pair means no reweighting at all. The example range
values are hypothetical, not observed data.


## 6. Known-orbit and approximate-range inputs

Known-orbit propagation and Gaussian range conditioning are optional observation modes. A range timestamp can differ from the original tracklet epoch. Angular-speed sensitivity is a separate study, not part of the main search pipeline.

### Technical supplement: known state, range conditioning and angular speed

#### Known-state mode

Input is a Cartesian ECI-like state **x**₀ = (**r**₀, **v**₀), epoch t₀, and a receiving station with latitude φ, longitude λ and altitude h. The two-body propagator computes **r**(t), while the rotating station model supplies **r**ₛ(t). Topocentric direction is **u**(t) = (**r**(t) − **r**ₛ(t))/ρ(t), with ρ(t) = |**r**(t) − **r**ₛ(t)|. Geometric elevation is calculated from **u** and station local up. The known orbit is treated as a delta distribution; field-of-view coverage is either zero or one at each sampled time. It does not model state uncertainty, measurement uncertainty or detection efficiency.

#### Approximate range mode

For an ensemble of states **x**ₖ at common epoch t₀, and a topocentric range observation ρ_obs ± σρ from a specified ranging station at that same epoch, calculate ρₖ = |**r**ₖ − **r**ₛ(t₀)|. Assuming independent Gaussian range error, update weights by

wₖ' = [wₖ exp(−(ρₖ−ρ_obs)²/(2σρ²))] / Σⱼ [wⱼ exp(−(ρⱼ−ρ_obs)²/(2σρ²))].

Effective sample size ESS = 1/Σₖ(wₖ')². This conditioning is exact for the **discrete empirical ensemble** and stated range likelihood, not necessarily a consistent continuous posterior: local Monte Carlo samples were not posterior draws and Stage B weights remain provisional. A range observation at a different epoch requires propagating states to that epoch first; the current CLI intentionally assumes the same epoch. The range observation must not duplicate data already incorporated into the ensemble likelihood.

#### Angular-speed sensitivity

With **d** = **r** − **r**ₛ, ρ = |**d**|, **u** = **d**/ρ and **v**ᵣₑₗ = **v** − **v**ₛ, the angular velocity vector of the line of sight is

**u̇** = (**v**ᵣₑₗ − (**v**ᵣₑₗ·**u**)**u**)/ρ.

Its magnitude ω = |**u̇**| is in rad/s. For an approximately central crossing of a circular field of diameter Θ radians, the crossing time is Δt ≈ Θ/ω, if angular speed is nearly constant and the track is locally straight. For an off-center crossing with impact parameter b in angular units, the path length is 2√[(Θ/2)² − b²], giving Δt ≈ 2√[(Θ/2)² − b²]/ω. Exposure time and smear constraints may therefore depend strongly on angular rate.

A single instantaneous angular speed constrains the ratio v_perp/ρ. It is insensitive to radial velocity at fixed ρ and v_perp, so it does not uniquely identify an orbit. Time variation of the angular rate and track curvature contain extra information, particularly because gravitational acceleration and observer rotation are known. The strength of these effects depends on range, tracklet duration, noise and viewing geometry. The standalone sensitivity figure varies range and assumed velocities, but does **not** estimate formal information matrices or run repeated orbit fits. For quantitative orbit-admissibility improvement, a later controlled study should fit identical noisy arcs over multiple simulated ranges, inclinations and angular speeds, and report Fisher/Jacobian singular values, solution-region area, truth recovery and family counts.

#### Assumptions and limitations

Both modes use the original spherical rotating Earth and two-body model, without illumination, weather, atmosphere, light-time or telescope limiting magnitude. The search-map `NO PASS` flag only applies to the supplied time grid and elevation cutoff. The plotted heatmap is smoothed for display, while coverage uses unsmoothed orbit samples. `PASS POSSIBLE` is not an optical detection guarantee.


## 7. Sky-search maps and telescope pointing

The search-map commands and symbol definitions are in Chapter 1. This short handover overview connects sky coverage to the station-overlap calculations in the next chapter.

### Direct station handover geometry

This tool answers: **How far apart could two freely chosen ground stations be and still simultaneously see a specified satellite for at least two minutes?**

Use `python handover/station_overlap.py --help`. Supply an NPZ with `epoch` and either `reference_state` (preferred) or `states`. Use `--state-index` for a specific candidate in `states`. All units: km, km/s, degrees, minutes.

Example: `python handover/station_overlap.py --orbit examples/mc_fitted_5_min.npz --overlap-min 2 --min-elevation 10 --hours 8 --output-dir results/handover_demo`.

The two station locations are optimized freely over Earth. This is **not** the maximum separation from a fixed existing station, nor a guarantee of tracking continuity if the telescope cannot slew/acquire in time. Elevation is checked at discrete points within each overlap interval, with optional finer sampling. Grid search plus local optimization can miss a better solution, especially near the longitude seam; decreasing the scan steps and grid spacing can improve the estimate. There is no weather, daylight, illumination, atmospheric extinction, terrain, Earth oblateness, mount slew, link budget, or orbital uncertainty. A direct handover needs both stations' tracking capabilities and synchronized timing, in addition to geometric overlap.

For uncertain orbits, one must choose a definition, e.g. a 90% weighted fraction of candidate orbits admitting the **same pair of stations and same two-minute interval**. Maximizing each candidate separately would overstate operational handover feasibility.


## 8. Ground-station overlap and handover

Only the known-orbit and initial single-family modes are supported by the robust fixed-station handover extension. The geographical rings are great-circle distance guides, not visibility boundaries.

### Robust fixed-station handover (known orbit / single-family only)

This optional module does **not** modify the orbit-fitting or station-search pipelines.

#### Quick start

From repository root:

```bash
python handover/robust_station_overlap.py --mode known --orbit examples/mc_fitted_5_min.npz --output-dir results/known
python handover/robust_station_overlap.py --mode single --orbit examples/mc_fitted_5_min.npz --output-dir results/single
```

Defaults: Bern fixed station (46.95 N, 7.44 E, 0.55 km), 2 minutes simultaneous visibility, both elevations >=10 degrees, 8-hour search horizon, 20-minute start-time grid, 10-degree station grid, 5 overlap-time checks, required 90% candidate coverage, 1-degree-radius telescope FoV. All configurable using `--help`.

##### Files

- `handover_map.png`: best *scheduled* two-minute ensemble coverage attainable at each potential receiver location. The legend distinguishes fixed and selected stations.
- `separation_curve.png`: largest station separation found for coverage thresholds 50%, 75%, 90%, 95%, 100%.
- `handover_timeline.png`: maximum attainable ensemble coverage as a function of overlap start time.
- `station_grid.csv`: location, coverage, best window and distance.
- `handover_windows.csv`: maximum coverage per sampled start time.
- `best_handover.json`: selected receiver, common overlap window, distance, elevation margins, and independently optimized single-pointing sky coverage at both stations at the window midpoint.

##### Important caveats

1. **Not a guaranteed maximum**: geographic coordinates are sampled on a grid, and overlap is checked at discrete times. Tighten `--grid-deg`, `--step-min` and increase `--checks` for a more accurate answer. Sampled visibility does not mathematically guarantee continuous visibility between checks.
2. **Known mode** uses `reference_state` in the NPZ (or the first state when absent). It assumes that state is exact; dynamics use a two-body spherical-Earth model.
3. **Single mode** gives equal weight to the stored local Monte Carlo states. Its coverage is a *candidate fraction*, **not a calibrated handover probability**. It may omit entirely different orbit families.
4. **Optical coverage** in JSON is a *snapshot at the handover midpoint*, not a complete telescope acquisition schedule. It ignores tracking slew, exposure time, target brightness, weather and daylight.
5. **Robustness**: the tool reports minimum elevation margins for the selected feasible candidates. It does not yet optimize an independent robustness objective (e.g., longest feasible overlap) or guarantee the full continuous overlap duration.
6. **Geometry**: Earth is spherical and rotating; satellite motion is two-body. No Earth obstruction beyond elevation, terrain, refraction, illumination or Sun/Moon constraints.
7. **No global-family support** is intentionally provided in this module.

##### World map and distance rings

The handover map includes true spherical great-circle surface-distance rings
centered on fixed Station A, in increments of 2,000 km. The rings are drawn
in orange dashed lines and labelled in thousands of kilometres (e.g. `8k`).
The red star marks Station A, the cyan diamond marks the selected Station B,
and the teal solid line shows their shortest great-circle connection.
Colored grid squares show maximum sampled-window geometric ensemble coverage
at each potential receiving station; the colorbar gives the fraction.
The map includes a complete legend. Rings are distance guides, **not**
visibility boundaries. Longitude seam crossings are split to avoid spurious
horizontal segments. Basemap is required to draw the coastlines.


## 9. Calibration, sensitivity and limitations

### Predictive sky-coverage validation (research diagnostic)

Run from the repository root:

```bash
python calibration/run_coverage.py --trials 20 --draws 1200 \
  --output-dir calibration/results/my_run
```

For a longer experiment, increase `--trials`, `--draws`, `--n-range`, `--n-rate`, and `--n-refine`. The default configuration is intentionally a quick smoke/pilot, **not** an adequate calibration study. Optional `--search-lat`, `--search-lon`, `--search-alt-km`, `--prediction-hours`, `--arc-minutes`, `--n-obs`, and `--seed` select the test scenario. Output: per-trial CSV, aggregate JSON and a labeled diagnostic PNG.

#### Exact scope

- Generates fresh independent Gaussian-noise RA/Dec tracklets for **one fixed synthetic truth orbit** from `demo_multipass_fit.py`, and reruns Stage A and Stage B for every trial.
- Predicts sky positions at the requested future epoch for the search station (default La Silla).
- Builds **nested spherical caps** around a sample-centered representative direction, with radii containing nominal 50%, 90%, 95% importance weight; checks truth inclusion.
- Records failure rate, effective sample size, conditional coverage and coverage counting failures as misses.
- No horizon visibility restriction is applied to the credible-cap experiment. The experiment validates *direction prediction*, not observable-region or telescope detection probability.
- This is **not** simulation-based calibration (SBC): the truth is not drawn from the inference prior. The nominal caps are not HEALPix/HPD credible regions. There is no detector model. A frequentist coverage rate for this fixed orbit cannot establish Bayesian calibration across an orbit population.
- Stage B's proposal is sparse; an ESS near single digits invalidates confidence in nominal mass quantiles. Failure to include truth can result from proposal misspecification, prior, numerical approximations, or the heuristic cap geometry.
- The current angular likelihood uses Gaussian noise in RA coordinate angle and Dec, as in the synthetic generator. All orbit propagation is two-body, with simplified Earth rotation. Do not interpret as operational tracking performance.

#### Next validation milestones

1. Specify an actual physical target-population prior and draw truth states from **that same prior**, including orbit selection effects and observing geometry.
2. Improve posterior sampling and demonstrate high ESS, cross-chain or independent-proposal consistency, and coverage of disconnected orbit modes.
3. Implement proper equal-area spherical credible regions (HEALPix or equivalent) and evaluate 50/90/95% containment and PIT/rank calibration across prior-predictive trials.
4. Repeat for different orbital regimes, tracklet durations, error models and range measurements; report binomial uncertainty intervals and failure rates.
5. Only then incorporate visibility, detector sensitivity, magnitude, trailing, weather and sequential search into calibrated detection probabilities.


## 9.1. Standalone angular-speed sensitivity study

### Separate angular-speed sensitivity analysis

```bash
python analysis/angular_speed_sensitivity.py --output results/angular_speed_sensitivity.png --fov-diameter 1
```

The figure explores (i) instantaneous topocentric angular speed versus range for three assumed inertial velocities, (ii) nominal field-of-view crossing time, (iii) radial-velocity sensitivity at fixed transverse velocity and range, and (iv) the range/transverse-velocity degeneracy at a fixed observed angular rate. The geometry is deliberately simple: target initially above a Bern-like station, eastward transverse motion, spherical rotating Earth, no exposure blur or field chord variations. FOV crossing time assumes a central, approximately straight crossing at constant speed.

For line-of-sight vector **d** = **r** − **r_station**, range ρ = |**d**|, and relative velocity **v_rel** = **v** − **v_station**, the instantaneous angular speed is

ω = |**v_rel** − (**v_rel**·**u**)**u**| / ρ, where **u** = **d**/ρ.

Thus an observed angular rate primarily constrains the **ratio** of transverse velocity to range, not either individually. Radial velocity does not affect the instantaneous angular rate at fixed range and transverse velocity, but influences angular acceleration and subsequent track curvature. Earth rotation, gravity, multi-epoch measurements, dynamical constraints and a second station can break the degeneracy. This experiment is **not** an empirical estimate of orbit recovery success rates; that requires simulated noisy tracklets and fitting across multiple truth orbits and geometries.


## 10. Implementation notes and testing

Keep the code and datasets in their existing locations. The consolidation changes documentation only.

### Implementation and provenance notes

- Consolidated from the independently developed original, Stage A, Stage B, global dashboard, connectivity and sky-search modules.
- Existing algorithms were not rewritten during consolidation.
- `orbit_stage_c.py` is an unimplemented placeholder; geometric search maps are implemented in `sky_search_prototype/sky_search_maps.py`.
- Existing figures and example NPZ files are bundled as reproducibility fixtures.
- The old technical report is preserved as `technical_report_original.pdf`; the current PDF appends an updated technical supplement.
- Import layout: core modules at repository root; search CLI in `sky_search_prototype/` inserts its parent into the Python import path.
- Original single-mode Monte Carlo and global weighted samples are separate.
