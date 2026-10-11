# Pre-registration — conformal purity constraints for surrogate-based PSA optimisation

**Status: DRAFT for review by the project lead (2026-10-11). Not frozen.** On approval it will be frozen by a git
tag `prereg-v1` before any step-6 simulation is run. After the freeze, any change is recorded under "Deviations"
(at the end of this file) with its date and reason. Results will not be used to edit sections 1-9.

Model and inputs as of the freeze: the step-4 layered-bed baseline (`docs/validation/step4.md`), design space
`configs/design_space.yaml`, disturbances `configs/disturbances.yaml`, mapping `src/h2psa/design.py`, CSS runner
`scripts/run_batch.py` (step 5, `docs/validation/step5.md`).

## 1. Research questions and hypotheses

* **RQ1.** How often do surrogate-optimal operating points violate the 100-ppm impurity limit (H2 purity >= 99.99 %)
  when re-simulated with the physical model? **H1:** with a point-estimate constraint the violation rate is well
  above 10 %.
* **RQ2.** Do split-conformal bounds keep the violation rate at the nominal level alpha on OPTIMISER-SELECTED points,
  not only on random test points? **H2:** no; selection on the constraint boundary (optimiser's curse) concentrates
  points where the surrogate under-predicts impurity, so the violation rate on selected points exceeds alpha
  although coverage on random test points is >= 1 - alpha.
* **RQ3.** Does a locally adaptive conformal bound (score normalised by a per-point uncertainty) restore the
  violation rate on selected points? **H3:** yes, at least partially.
* **RQ4.** What does each guarantee cost in H2 recovery and productivity? (descriptive; no hypothesis)
* **RQ5.** Are the guarantees robust to the disturbance variables (training and calibration at nominal conditions,
  operation under disturbance = distribution shift)? **H5:** no; the violation rate of the selected points rises
  under disturbance, most for low adsorbent capacity.

## 2. Variables

* **Decision variables (8),** `configs/design_space.yaml`: t_feed [40, 150] s, t_blow [60, 200] s, purge_to_feed
  [0.02, 0.30], P_high [4.0, 8.5] bar, P_low [0.3, 1.5] bar, f_zeolite in {0, 1/30, ..., 1}, n_equalisations
  in {1, 2, 3}, feed_flow [0.5, 1.5] x 5.0e-5 m3/s (STP 298 K, 1 bar). Fixed: t_eq 16 s, t_press 21 s, t_purge 40 s,
  T_feed = T_ambient = 303 K, beta_P = 0.102 s-1, EP gas pure H2.
* **Disturbance variables (3),** `configs/disturbances.yaml`: capacity factor phi [0.70, 1.00] (q_max of both
  layers), y_CH4 and y_CO of the feed (CO2 keeps the anchor's total carbon, N2 fixed, H2 by difference).
  Nominal: phi = 1, Lopes feed (Table 5.4).
* **Unit-cube mapping** (all sampling and optimisation in [0, 1]^d): continuous variables linear;
  f_zeolite = floor(31 u) / 30 (31 equally likely grid values, u = 1 maps to 1); n_equalisations = 1 + floor(3 u)
  (u = 1 maps to 3).
* **Outputs** per CSS run: impurity (ppm, eq. 5.48, feed components only), H2 recovery (eq. 5.49), productivity
  (eq. 5.50); CSS criterion of step 4 (relative change < 1e-4 over 10 plain cycles, impurity change < 1e-5, mole
  balance < 0.1 %), cap 2000 cycles.

## 3. Response modelling and censoring

* **Response:** z = clip(log10(impurity / ppm), -1, 3). The constraint is z <= 2 (100 ppm). Clipping is monotone and
  does not change the event {z <= 2}, so a bound on z is a bound for the constraint.
* **Censored points.** The runner stops a point when, after >= 100 cycles, the impurity exceeds 1000 ppm (10x the
  limit) and has increased over the last 30 cycles; the label is "censored: impurity > X ppm". On the step-4
  histories (231 runs) the rule never stopped a run that converged below 1000 ppm (smallest CSS impurity of a
  stopped run: 1515 ppm; `docs/validation/step5.md`). A censored point therefore has z = 3 exactly, the clipped
  label is fully observed for every point, and the conformal calibration needs no censoring correction. Censored
  points get z = 3 in every surrogate's training set.
* **Objectives of censored points** are not at CSS and are not used: the recovery and productivity surrogates are
  trained on non-censored points only (censored points are infeasible by > 10x, so the optimiser never needs the
  objectives there).
* **Runs at the 2000-cycle cap** without CSS and without censoring: the last-cycle values are used and flagged;
  their number is reported. **Solver failures:** re-run once from the He-filled column; points failing again are
  listed and excluded from training; in calibration / test sets they count as miscovered in a sensitivity analysis;
  in re-simulated selected points they count as violations in a sensitivity analysis.

## 4. Data sets (all nominal unless stated; never reused across roles)

| set | role | design | size | seed |
|---|---|---|---|---|
| D_train | surrogate training and model selection only | scrambled Sobol in the 8 decision variables (`scipy.stats.qmc.Sobol(d=8, scramble=True, seed=20261011)`, first 2^9 points); training sizes = the nested prefixes | 128 / 256 / 512 | 20261011 |
| D_cal | conformal calibration only | i.i.d. uniform, 8 decision variables | 256 | `numpy.random.default_rng(1)` |
| D_test | coverage on random points only | i.i.d. uniform, 8 decision variables | 256 | `default_rng(2)` |
| D_shift | coverage under shift (RQ5) | i.i.d. uniform, 8 decision + 3 disturbance variables (Lopes anchor) | 128 | `default_rng(3)` |
| S_nom | re-simulated selected points | section 6 | 660 | NSGA-II seeds 10, 11, 12 |
| S_dist | selected points under disturbance (RQ5) | section 6, one disturbance draw per point, uniform in the disturbance box | 180 | `default_rng(4)` |

D_cal and D_test are never used for model selection, hyperparameters, early stopping or any choice of section 5
or 6. D_cal enters only the conformal quantile; D_test, D_shift, S_nom and S_dist enter only the metrics.
Warm starts of the CSS runs (nearest finished CSS state) are a numerical device; the step-4 acceleration check
bounds their effect on the labels (impurity within 1e-3 relative).

## 5. Surrogates and constraint variants

Inputs: the unit-cube coordinates of the 8 decision variables (disturbances are not inputs: training is nominal).
Three outputs, modelled separately: z, recovery, productivity.

* **GP:** Gaussian process, constant mean, Matern 5/2 kernel with one length scale per input, white-noise term
  (lower bound 1e-6 on the standardised scale), outputs standardised; hyperparameters by maximum marginal
  likelihood with 10 restarts (scikit-learn `GaussianProcessRegressor`, `random_state = 0`).
* **GBM:** gradient boosting (scikit-learn `HistGradientBoostingRegressor`, `random_state = 0`), squared-error loss
  for the point prediction; quantile loss at level 1 - alpha for CQR. Hyperparameters (learning rate {0.03, 0.1},
  max_leaf_nodes {7, 15, 31}, min_samples_leaf {5, 10, 20}, max_iter {200, 500, 1000}) by 5-fold CV on D_train only
  (squared error, resp. pinball loss).
* Library versions are pinned in `pyproject.toml` at the freeze.

Constraint variants on z (one-sided; s = conformal score on D_cal, q = the ceil((n + 1)(1 - alpha))-th smallest
score, n = |D_cal| = 256):

| variant | score s_i on D_cal | constraint used by the optimiser |
|---|---|---|
| P: point estimate | – | zhat(x) <= 2 |
| G: global split conformal | z_i - zhat(x_i) | zhat(x) + q <= 2 |
| L: locally adaptive conformal | GP: (z_i - zhat(x_i)) / sigma(x_i), sigma the GP posterior std; GBM: CQR, z_i - Qhat_{1-alpha}(x_i) | GP: zhat(x) + q sigma(x) <= 2; GBM: Qhat_{1-alpha}(x) + q <= 2 |

alpha = 0.10 primary, 0.05 secondary.

## 6. Optimiser and selection of points to re-simulate

* NSGA-II (pymoo), maximise (recovery, productivity) predicted by the surrogate, subject to the constraint
  variant; constraint handling by Deb's feasibility rules. Population 100, 250 generations, SBX crossover
  (eta 15, p 0.9), polynomial mutation (eta 20, p 1/8); f_zeolite and n_equalisations rounded by the unit-cube
  mapping (repair) before evaluation. Seeds 10, 11, 12.
* From each final non-dominated feasible set: 10 points nearest to the 0.05, 0.15, ..., 0.95 quantiles of predicted
  productivity (a point is used once; ties to the next nearest). 3 seeds x 10 = **30 points per arm**; duplicates
  after rounding are simulated once and counted once.
* **Arms (S_nom, 660 runs):** primary: 2 surrogates x 3 training sizes x 3 variants (P, G, L) at alpha = 0.10
  = 18 arms x 30 = 540; secondary: alpha = 0.05, variants G and L, n_train = 512, 2 surrogates = 4 arms x 30 = 120.
* **RQ5 (S_dist, 180 runs):** the 30 points of each primary arm at n_train = 512 (2 surrogates x 3 variants = 6 arms)
  re-simulated at one disturbance draw each (paired with their nominal re-simulation in S_nom).
* **Exploratory, outside the budget:** the same 180 points at the NETL dry-basis anchor (P_high <= 8.5 bar);
  a disturbance-aware recalibration. Reported as exploratory only.

## 7. Metrics

* **Violation rate** of an arm: fraction of its re-simulated points with z > 2; Clopper-Pearson 95 % CI. Primary
  estimates pool the 6 arms of a variant (2 surrogates x 3 sizes; n = 180); per-arm values are reported.
* **Coverage** = fraction with z <= bound(x): on D_test (random points) and on the selected points (S_nom), per
  surrogate, size, variant and alpha; the difference is the selection effect.
* **Fronts.** Realised front of an arm = non-dominated set of its re-simulated feasible points. Reference front =
  non-dominated set of all feasible simulated nominal points (D_train, D_cal, D_test, S_nom), i.e. the best known
  feasible front. Objectives normalised by the reference front's ideal and nadir points; hypervolume with reference
  point (-0.1, -0.1) in normalised space; **HV ratio** = HV(realised) / HV(reference). **Recovery loss** at 5
  productivity levels (0.1, 0.3, 0.5, 0.7, 0.9 of the reference front's productivity range): reference recovery
  minus the best realised recovery with at least that productivity (pp; reported as "not reached" when none).
  Uncertainty by bootstrap over the arm's selected points (2000 resamples, `default_rng(5)`).
* **RQ5:** paired violation (nominal vs disturbed) of the same points; coverage on D_shift with the nominal
  bounds, and a logistic regression of miscoverage on (phi, y_CH4, y_CO) scaled to [0, 1].

## 8. Analysis decisions fixed in advance and criteria

* Order: all simulations of D_train, D_cal, D_test and D_shift; surrogate fitting and model selection on D_train;
  conformal quantiles on D_cal; optimisation; selection; then S_nom and S_dist. The analysis script is committed
  before S_nom is simulated. Nothing is refitted after any test-set, D_shift or S result is seen; a bug found later
  is fixed, everything downstream rerun, and the change reported under Deviations.
* Tests are one-sided at 5 %; the four hypotheses H1, H2, H3, H5 are reported with Holm's correction as a family.
* Sanity gate before the selected-point analysis: coverage of G and L on D_test must be consistent with >= 1 - alpha
  (one-sided Clopper-Pearson 95 % upper bound >= 1 - alpha). If it fails, the conformal implementation is checked;
  the gate result is reported either way.

| RQ | confirmed | not confirmed |
|---|---|---|
| RQ1 | pooled violation rate of P (n = 180): CI lower bound > 0.10 | lower bound <= 0.10 |
| RQ2 | pooled violation rate of G at alpha = 0.10 (n = 180): CI lower bound > 0.10, with the D_test sanity gate passed | lower bound <= 0.10 (global conformal suffices on selected points) |
| RQ3 | pooled violation rate of L (n = 180): CI upper bound <= 0.15 AND G - L > 0 (Newcombe 95 % CI of the difference above 0) | neither; "partially" if exactly one holds; descriptive only if RQ2 is not confirmed |
| RQ4 | descriptive: HV ratio and recovery loss per variant and alpha, and relative to P's realised front | – |
| RQ5 | paired violation rate disturbed > nominal over the 6 arms (n = 180, exact McNemar, one-sided p < 0.05) | p >= 0.05 |

Secondary (reported, not decisive): per-arm rates; trends with training size; alpha = 0.05; logistic coefficient of
phi on D_shift miscoverage (sign expected negative).

**Limitations declared now.** The binomial CIs are conditional on one training set and one calibration set per
arm; the points of a front share the surrogate's error and are not independent (a by-seed bootstrap is reported as
a sensitivity check). The reference front is the best known one, not the true optimum. Single-column emulation:
absolute recoveries below multi-bed industrial values (step 4).

## 9. Budget and compute

| block | CSS runs |
|---|---|
| D_train (Sobol, nested 128 / 256 / 512) | 512 |
| D_cal | 256 |
| D_test | 256 |
| D_shift | 128 |
| S_nom (18 primary arms x 30 + 4 secondary x 30) | 660 |
| S_dist (6 arms x 30) | 180 |
| **total** | **1992** (+ <= 5 % reruns of solver failures) |

Compute plan (`docs/validation/step5.md`): 10 cores (Apple M1 Pro, 8 performance + 2 efficiency), 8 workers.
From the step-4 timing distribution (31 layered CSS runs, median 5.9 min, mean 9.1 min) the 2000 runs take about
39 h of wall time; with early stopping (censored runs) about 24 h. These are extrapolations from the step-4
scans; the step-6 design space is wider and the first block (D_train) re-estimates the time. Runs use
`scripts/run_batch.py` under `caffeinate -i` with checkpoint / resume.

## Deviations

None yet.
