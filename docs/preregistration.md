# Pre-registration — conformal purity constraints for surrogate-based PSA optimisation

**Status: FROZEN, version 1 (2026-10-12), git tag `prereg-v1`.** Reviewed by the project lead (draft of
2026-10-11, review of 2026-10-12: RQ3 power, band-matched comparison, cluster bootstrap, Holm on exact p-values,
RQ5 primary analysis, pinned libraries). Frozen before any step-6 simulation is run. After the freeze, any change
is recorded under "Deviations" (at the end of this file) with its date and reason. Results will not be used to edit
sections 1-9.

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
  although coverage on random test points is >= 1 - alpha. Selection is separated from the boundary effect
  (larger error near the constraint, without selection) by the band-matched comparison (sections 7, 8).
* **RQ3.** Does a locally adaptive conformal bound (score normalised by a per-point uncertainty) restore the
  violation rate on selected points? **H3:** yes, at least partially.
* **RQ4.** What does each guarantee cost in H2 recovery and productivity? (descriptive; no hypothesis)
* **RQ5.** Are the guarantees robust to the disturbance variables (training and calibration at nominal conditions,
  operation under disturbance = distribution shift)? **H5:** no; miscoverage of the nominal conformal bounds rises
  as the adsorbent capacity falls (negative phi coefficient, section 7).

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
| S_nom | re-simulated selected points | section 6 | 900 | NSGA-II seeds 10, 11, 12 |
| S_dist | selected points under disturbance (RQ5) | section 6, one disturbance draw per point, uniform in the disturbance box | 240 | `default_rng(4)` |

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
* **Software (pinned in `pyproject.toml` at the freeze):** scikit-learn 1.9.1, pymoo 0.6.2, statsmodels 0.15.0.
  Environment of all step-6 runs and analyses (recorded): Python 3.14.7, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6.

Constraint variants on z (one-sided; s = conformal score on D_cal, q = the ceil((n + 1)(1 - alpha))-th smallest
score, n = |D_cal| = 256). bound(x) denotes the left-hand side of the constraint; every variant requires
bound(x) <= 2:

| variant | score s_i on D_cal | bound(x) |
|---|---|---|
| P: point estimate | – | zhat(x) |
| G: global split conformal | z_i - zhat(x_i) | zhat(x) + q |
| L: locally adaptive conformal | GP: (z_i - zhat(x_i)) / sigma(x_i), sigma the GP posterior std; GBM: CQR, z_i - Qhat_{1-alpha}(x_i) | GP: zhat(x) + q sigma(x); GBM: Qhat_{1-alpha}(x) + q |

alpha = 0.10 primary, 0.05 secondary.

## 6. Optimiser and selection of points to re-simulate

* NSGA-II (pymoo), maximise (recovery, productivity) predicted by the surrogate, subject to the constraint
  variant; constraint handling by Deb's feasibility rules. Population 100, 250 generations, SBX crossover
  (eta 15, p 0.9), polynomial mutation (eta 20, p 1/8); f_zeolite and n_equalisations rounded by the unit-cube
  mapping (repair) before evaluation. Seeds 10, 11, 12.
* From each final non-dominated feasible set: m points nearest to the (k - 0.5) / m quantiles of predicted
  productivity, k = 1, ..., m (a point is used once; ties to the next nearest). **m = 10 for P (30 points per arm),
  m = 15 for G and L (45 points per arm).** If a set has fewer than m distinct points, all are used and the
  shortfall is reported. Duplicates after rounding are simulated once and counted once.
* **Arms (S_nom, 900 runs):** primary, alpha = 0.10, 2 surrogates x 3 training sizes: P 6 arms x 30 = 180,
  G and L 12 arms x 45 = 540; secondary: alpha = 0.05, variants G and L, n_train = 512, 2 surrogates = 4 arms x 45
  = 180.
* **RQ5 (S_dist, 240 runs):** all points of the primary arms at n_train = 512 (2 surrogates x 3 variants = 6 arms:
  2 x 30 + 4 x 45) re-simulated at one disturbance draw each (paired with their nominal re-simulation in S_nom).
* **Exploratory, outside the budget:** the S_dist points at the NETL dry-basis anchor (P_high <= 8.5 bar);
  a disturbance-aware recalibration. Reported as exploratory only.

## 7. Metrics

* **Violation rate** of an arm: fraction of its re-simulated points with z > 2; two-sided Clopper-Pearson 95 % CI.
  Primary estimates pool the 6 arms of a variant at alpha = 0.10 (2 surrogates x 3 sizes; n = 180 for P, 270 for
  G and L); per-arm values are reported.
* **Coverage** = fraction with z <= bound(x): on D_test (random points) and on the selected points (S_nom), per
  surrogate, size, variant and alpha.
* **Band-matched coverage (primary for RQ2, RQ3).** For each fitted model and variant, the D_test points with
  bound(x) in the band [1.5, 2.0] (log10 ppm; the selected points satisfy bound(x) <= 2 and lie on or near the
  constraint). If fewer than 30 D_test points of a model / variant fall in the band, the band of that model /
  variant is widened to [1.25, 2.0]; if still fewer than 30, the band is kept and the count reported (fixed rule).
  For P the band is zhat(x) in [1.5, 2.0]. Reported: the number of D_test points in each band, the band used, the
  fraction of selected points inside the band, the band-matched coverage and the selected-point coverage, pooled
  per variant over its 6 primary arms, with the Newcombe 95 % CI of the difference (band-matched minus selected).
* **Fronts.** Realised front of an arm = non-dominated set of its re-simulated feasible points. Reference front =
  non-dominated set of all feasible simulated nominal points (D_train, D_cal, D_test, S_nom), i.e. the best known
  feasible front. Objectives normalised by the reference front's ideal and nadir points; hypervolume with reference
  point (-0.1, -0.1) in normalised space; **HV ratio** = HV(realised) / HV(reference). **Recovery loss** at 5
  productivity levels (0.1, 0.3, 0.5, 0.7, 0.9 of the reference front's productivity range): reference recovery
  minus the best realised recovery with at least that productivity (pp; reported as "not reached" when none).
  Uncertainty by bootstrap over the arm's selected points (2000 resamples, `default_rng(5)`).
* **RQ5 (primary): logistic regression of miscoverage** 1{z > bound(x)} of the nominal G and L bounds at
  alpha = 0.10, n_train = 512 (2 surrogates x 2 variants), on D_shift (128 points x 4 bounds) and on the G and L
  points of S_dist (4 arms x 45, disturbed run). Model: logit P(miscoverage) = intercept per set (D_shift, S_dist)
  + indicator L + indicator GBM + b_phi u_phi + b_CH4 u_CH4 + b_CO u_CO, the disturbances u scaled to [0, 1] over
  their box (u_phi = (phi - 0.70) / 0.30). Fit by maximum likelihood (statsmodels `GLM`, binomial, logit link);
  Wald 95 % CI with cluster-robust covariance (clusters: each D_shift point; each (arm, NSGA-II seed) of S_dist).
  Reported: b_phi with CI and the odds ratio phi 0.70 vs 1.00 = exp(-b_phi), and b_CH4, b_CO.
* **RQ5 (secondary):** paired violation (nominal S_nom vs disturbed S_dist) of the same points over the 6 arms
  (n = 240), exact one-sided McNemar test; coverage on D_shift with the nominal bounds.

## 8. Analysis decisions fixed in advance and criteria

* Order: all simulations of D_train, D_cal, D_test and D_shift; surrogate fitting and model selection on D_train;
  conformal quantiles on D_cal; optimisation; selection; then S_nom and S_dist. The analysis script is committed
  before S_nom is simulated. Nothing is refitted after any test-set, D_shift or S result is seen; a bug found later
  is fixed, everything downstream rerun, and the change reported under Deviations.
* **Confirmatory family and multiplicity.** RQ1, RQ2, RQ3 and RQ5 each get one one-sided p-value (table below);
  Holm's step-down procedure at family-wise level 0.05 is applied to these four p-values, and an RQ is confirmed
  if its Holm-adjusted p < 0.05. The CIs of section 7 (two-sided 95 %) are the reported effect sizes; they do not
  decide. RQ4 is descriptive.
* **Sanity gate** before the selected-point analysis: coverage of G and L on D_test must be consistent with
  >= 1 - alpha (one-sided Clopper-Pearson 95 % upper bound >= 1 - alpha). If it fails, the conformal implementation
  is checked; the gate result is reported either way.

| RQ | p-value (one-sided, exact unless stated) | confirmed (Holm-adjusted p < 0.05) | not confirmed |
|---|---|---|---|
| RQ1 | binomial, pooled P violations (n = 180), H0: rate <= 0.10 | violation rate of P above 0.10 | otherwise |
| RQ2 | binomial, pooled G violations at alpha = 0.10 (n = 270), H0: rate <= 0.10 | violation rate of G above 0.10, with the D_test sanity gate passed | otherwise (global conformal suffices on selected points) |
| RQ3 | max(p_a, p_b) (intersection-union): p_a binomial, pooled L violations (n = 270), H0: rate >= 0.17; p_b Fisher's exact test, H0: rate_G <= rate_L | both: L rate below 0.17 AND G - L > 0 | "partially" if exactly one of p_a, p_b is below the Holm level of RQ3; "not confirmed" if neither; descriptive only if RQ2 is not confirmed |
| RQ4 | – | descriptive: HV ratio and recovery loss per variant and alpha, and relative to P's realised front | – |
| RQ5 | Wald test of b_phi from the logistic regression of section 7 (cluster-robust, asymptotic), H0: b_phi >= 0 | b_phi < 0 (miscoverage rises as capacity falls) | otherwise |

* **Selection versus boundary effect (RQ2, RQ3).** A violation excess on selected points is interpreted as a
  selection effect only if the selected-point coverage is lower than the band-matched D_test coverage (Newcombe
  95 % CI of band-matched minus selected above 0); otherwise it is reported as a boundary effect (the error near
  the constraint, present without selection).
* **Dependence (mandatory robustness check).** Cluster bootstrap with one generator `default_rng(6)`, 2000 draws
  per RQ, used in the order RQ1, RQ2, RQ3, RQ5. Clusters: (arm, NSGA-II seed), i.e. 18 per pooled variant;
  for RQ3 the (surrogate, n_train, seed) clusters are resampled jointly for G and L (paired); for RQ5 the D_shift
  points and the S_dist (arm, seed) clusters, each resampled within its set. Each decisive quantity (pooled rate;
  G - L; b_phi) gets a one-sided percentile bound at the level that Holm assigned to the RQ (0.05 / (5 - r) for
  the RQ ranked r-th by p-value), and the verdict is recomputed with that bound against the same threshold. If the
  bootstrap verdict differs from the primary verdict for any RQ, that RQ is reported as "not robust".
* **Power (RQ3, component p_a).** Exact binomial, n = 270, H0: rate >= 0.17, probability of rejecting at the true
  L violation rates below, at the Holm levels 0.05 (last step) and 0.0125 (first step) and at 0.025 (equivalent
  to the two-sided 95 % Clopper-Pearson upper bound <= 0.17). The replaced draft design (n = 180, threshold 0.15)
  for comparison. Clustering is ignored (independent points assumed); the cluster bootstrap addresses it. The power
  of p_b depends on the G rate and is not tabulated.

| design | rejection region | true rate 0.05 | 0.08 | 0.10 |
|---|---|---|---|---|
| n = 270, 0.17, level 0.05 | <= 35 violations | 1.000 | 0.998 | 0.953 |
| n = 270, 0.17, level 0.025 | <= 33 | 1.000 | 0.994 | 0.904 |
| n = 270, 0.17, level 0.0125 | <= 32 | 1.000 | 0.990 | 0.867 |
| draft: n = 180, 0.15, level 0.025 | <= 17 | 0.996 | 0.806 | 0.464 |

Secondary (reported, not decisive): per-arm rates; trends with training size; alpha = 0.05; the RQ5 McNemar test
and D_shift coverage; b_CH4 and b_CO.

**Limitations declared now.** The binomial CIs are conditional on one training set and one calibration set per
arm; the points of a front share the surrogate's error and are not independent (cluster bootstrap above; with
18 clusters per pooled variant it is itself coarse). The band-matched comparison pools the same 256 D_test points
over 6 models. The reference front is the best known one, not the true optimum. Single-column emulation: absolute
recoveries below multi-bed industrial values (step 4).

## 9. Budget and compute

| block | CSS runs |
|---|---|
| D_train (Sobol, nested 128 / 256 / 512) | 512 |
| D_cal | 256 |
| D_test | 256 |
| D_shift | 128 |
| S_nom (P 6 x 30, G and L 12 x 45, secondary 4 x 45) | 900 |
| S_dist (2 x 30 + 4 x 45) | 240 |
| **total** | **2292** (+ <= 5 % reruns of solver failures) |

Compute plan (`docs/validation/step5.md`): 10 cores (Apple M1 Pro, 8 performance + 2 efficiency), 8 workers.
From the step-4 timing distribution (31 layered CSS runs, median 5.9 min, mean 9.1 min) the 2292 runs take about
44 h of wall time (p90 46 h); with early stopping (censored runs) about 27.5 h (p90 28 h). These are extrapolations
from the step-4 scans; the step-6 design space is wider and the first block (D_train) re-estimates the time. Runs
use `scripts/run_batch.py` under `caffeinate -i` with checkpoint / resume.

## Deviations

* **2026-10-12 — compute concurrency.** Step 6a uses 6 workers, including the 32-point
  D_train pilot, instead of 8 because another heavy Python modelling job was active
  on the same 10-core machine (6 requested threads; observed CPU use above 300 %).
  This follows the project lead's Step 6a exception; sampling, sizes and numerical
  criteria are unchanged. The pilot supplies the revised time estimate.
* **2026-10-12 — generation infrastructure repairs.** Before any Step 6a simulation,
  the frozen batch runner was extended to persist solver failures, perform the
  registered once-only He-filled-column retry, pause new submissions at the lead's
  2 % failure / 5 % cap thresholds, and monitor without displaying response labels.
  Checkpoints now retain active wall time, attempt/warm-start provenance and the
  impurity history needed by the registered censoring rule across resumes. A
  generation-only driver writes unit-cube coordinates and the required per-point
  CSVs and commits/pushes each completed block. The model, tolerances, CSS/censoring
  criteria, seeds and mappings remain as registered. No additional solver tightening
  is specified in prereg-v1; retries retain the frozen settings, including the
  already tightened impurity CSS tolerance 1e-5. Interrupted work after the last
  saved checkpoint (every 10 cycles) is not included in per-point wall time.
* **2026-10-12 — pilot monitoring denominator.** The lead's pause thresholds are
  also checked against the 32-point pilot execution block before the remaining
  D_train points are submitted (one primary failure or two cap hits pauses intake).
  Subsequent checks use each full registered data-set size. This conservative
  additional check uses only failure/cap counts and changes no analysis decisions.
* **2026-10-12 — CSV-only publication history.** Automatic approval review rejected
  pushing the local infrastructure commit because the lead's publication instruction
  covered generated CSVs. Each completed CSV is therefore committed locally and
  its verified single-file change is cherry-picked into a separate publishing
  checkout based on remote main. The lead subsequently approved infrastructure
  commit 7f25a21 explicitly, and it was pushed separately after the pilot. Every
  later publishing commit is checked to contain only its registered block CSV;
  additional local infrastructure changes are excluded. Local and published commit
  IDs are recorded separately. This changes publication history, not the data.
* **2026-10-13 — publication checkpoint delays.** The CSV-only publication path
  stops the driver after each completed block for the agent to commit and push
  that concrete CSV before launching the next block. These checkpoints added
  about 43 minutes between simulation phases. D_shift and its CSV finished at
  00:21 EDT on 2026-10-13; its commit/push completed at 06:29 EDT when the agent
  continued the session. The final publication checkpoint therefore added a
  further 6.14 hours after generation. This records the delay from deferred
  agent-controlled publication; all 1152 points were generated once, in the
  registered order, and no sampling, numerical or analysis decision changed.
* **2026-10-13 — Step 6b numerical conventions fixed before held-out-label inspection.**
  The frozen text does not specify GP initial kernel values/upper bounds, CV fold
  shuffling, or inverse representatives for repaired discrete unit coordinates.
  GP uses the installed sklearn defaults: unit initial covariance amplitude,
  length scales and white-noise level; amplitude and eight length-scale bounds
  [1e-5, 1e5], and white-noise variance (`noise_level`) bounds [1e-6, 1e5]. `normalize_y=True` implements
  the constant empirical mean and population-standardised target; default
  `alpha=1e-10` is numerical jitter. Ten optimiser restarts are additional to
  the initial fit. Predictive sigma includes the registered white-noise term.
  GBM uses `KFold(5, shuffle=False)` on eligible rows in original prefix order,
  negative mean squared error / pinball loss to choose the best of all 54 grid
  combinations, and `early_stopping=False` to honour each registered max_iter.
  Other estimator settings retain the pinned library defaults. Objective fits
  exclude censored and failed rows; cap rows retain their last-cycle labels.
  Optimiser repair maps bin k for f_zeolite to (k+0.5)/31 and bin j for
  n_equalisations to (j+0.5)/3 before surrogate evaluation, preserving the frozen
  physical mapping. Polynomial mutation probability 1/8 applies per coordinate
  (`prob=1`, `prob_var=1/8` in pymoo). Deterministic order is GP then GBM;
  training sizes 128, 256, 512; primary P/G/L then secondary G/L at 512;
  seeds 10, 11, 12; stable index order resolves equal-distance selection ties.
  These pre-results declarations supply implementation details omitted by v1;
  no held-out labels are used to choose them.
* **2026-10-13 — Step 6b gate/report order.** The lead explicitly requests the
  D_test sanity gate before optimisation and descriptive test RMSE in Step 6b.
  This moves those diagnostics ahead of the section-8 optimisation/selection
  order. All fits, hyperparameters and optimiser settings are frozen first;
  the diagnostics do not change methods, hyperparameters or selected-arm settings.
* **2026-10-13 — duplicate/statistical implementation declarations.** The lead
  confirmed that a shared rounded physical point has one simulation ID and is
  counted once within each arm; cross-arm memberships remain in pooled arm rates.
  Stable first membership supplies the seed cluster for within-arm duplicates.
  Exact p-value ties use the fixed RQ1/RQ2/RQ3/RQ5 order for Holm ranks; bootstrap
  percentiles use NumPy's linear interpolation and original Holm-assigned levels.
  The primary RQ2 gate uses all primary alpha=0.10 G/L arms; secondary alpha=0.05
  diagnostics are reported independently. Undefined logistic/bootstrap fits are
  reported as undefined rather than replaced or omitted from quantiles. Cluster
  covariance uses the pinned statsmodels finite-sample correction. One rng5 stream
  bootstraps fronts in lexical arm order; one rng6 stream bootstraps RQ1, RQ2,
  paired RQ3, then RQ5. These conventions were fixed before test-label inspection.
* **2026-10-13 — Step 6c concurrency.** S_nom and S_dist are simulated with 6 workers instead of 8 because
  other heavy jobs were active on the same 10-core machine at launch (an RStudio session at about 200 % CPU for
  4 days, plus a pytest job; load average about 6). This is the Step 6a exception; the runner, its flags and its
  pause rules are unchanged (`scripts/run_step6c.py` reuses `run_step6a.run_phase`). Publication is left to the
  user (no git from the agent).
* **2026-10-14 — Python environment drift and repair.** Homebrew upgraded its Python 3.14 from 3.14.7 to
  3.14.8 at 2026-10-13 23:47 EDT; the venv followed Homebrew's current Python. The 899 S_nom points finished
  between 12:18 and 18:47 on 2026-10-13. They all came from the single runner process started at 12:15 under the
  3.14.7 framework: the log has one launch, and its shutdown warning names the 3.14.7 path. Pool workers are not
  respawned, and the pinned packages were installed on 10-09 / 10-12. So none of them ran on 3.14.8. Only the
  resumed hung attempt of S_nom_0590 ran on 3.14.8, and it produced no output (see below). The venv was rebuilt
  on the exact pinned CPython 3.14.7, independent of Homebrew (uv-managed standalone build,
  `uv venv --python 3.14.7`). Its packages were installed from the previous venv's `pip freeze`: all 41 versions
  are identical, including numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, pymoo 0.6.2 and
  statsmodels 0.15.0. Check: one cycle from the stored CSS states of S_nom_0000, 0450 and 0899 reproduces the
  stored impurity within 1.3e-6 relative, i.e. within the CSS tolerance. `brew pin python@3.14` prevents further
  Homebrew upgrades.
* **2026-10-14 — per-attempt wall-time limit (new rule, Step 6c).** S_nom_0590 stayed inside its first cycle
  for about 22 h (2026-10-13 16:00 to 2026-10-14 14:30), with no solver error. The registered failure rule could
  not trigger, and the runner had no limit. Rule: an attempt whose active wall time, summed over resumes, exceeds
  2 h is a solver failure (TimeoutError), handled exactly as registered: one retry from the He-filled column
  (cold, no warm start, frozen tolerances); if the retry also exceeds 2 h or fails, the point is "failed"
  (section 3). Scope: S_nom_0590 and all of S_dist, applied uniformly (`run_batch.py --point-timeout 7200`).
  The rule would have changed no earlier result. All 1152 Step 6a points finished in at most 79.3 min (D_train
  41.1, D_cal 57.2, D_test 79.3, D_shift 77.9 min; none resumed, so these times are complete). The 899 finished
  S_nom points took at most 9.7 min. The hung first attempt of S_nom_0590 was recorded as timed out by
  converting its checkpoint to the state the runner's failure branch would have written: attempt 2, cold start,
  error TimeoutError. Its active time after the last checkpoint was not recorded, and the original checkpoint is
  kept outside the repository.
* **2026-10-14 — session interruption.** The Step 6c driver was killed when the agent session ended (S_nom
  899/900). It was resumed from checkpoints at 04:54 EDT without recomputing finished points, detached from the
  session (`nohup`), and then stopped again at 14:30 for the two repairs above. Batch wall time excludes the
  idle gap.

## Addendum A (2026-10-14) — Step 7b model-form robustness (DESCRIPTIVE; designed after the Step 6c results)

Decided by the project lead on 2026-10-14, after the Step 6c results were seen. This addendum leaves sections 1-9
and the Step 6c results unchanged. Its analysis is **descriptive**: no hypothesis, no multiplicity correction and
no confirmatory language. This addendum and its scripts are committed before any Step 7b simulation runs.

* **Question.** How do the violation rate, the coverage of the frozen bounds and the realised fronts of the selected
  points change when the physical model's form changes within the declared uncertainty set carried from steps 3 and 4
  (`configs/design_space.yaml: scenarios_carried_to_step7`)?
* **Scenarios run** (`h2psa.design.SCENARIOS`, set per point by the optional `scenario` key of `design.build`):
  * `ep_inflow_ed_gas`: the EP inflow is the gas of the matching ED step (ED1 -> EP1, ...) instead of pure H2.
    beta_P is unchanged, as in the step-3 scenario `band_ep_gas_h2`.
  * `thermal_range_low` / `thermal_range_high`: both layers' solid heat capacity and the wall heat capacity at the
    ends of their assumed ranges, applied as in step 3 (`scripts/run_psa.py` band_thermal_*). Values:
    - Cps of HY2SEPS-AC: 700 / 1100 J kg-1 K-1 (`hy2seps_ac_lopes2010.yaml`).
    - Cps of HY2SEPS-ZEO: 800 / 1100 J kg-1 K-1 (`hy2seps_zeo_lopes2010.yaml`).
    - Wall heat capacity per length: `wall_heat_capacity(e)` with e = 2 / 5 mm, e_wall kept at 3 mm
      (`compact_column_lopes2010.yaml`).
* **Scenarios dropped**, with reasons:
  * `noncompetitive_h2`: this mixing rule belongs to the step-3 virial isotherm of AC5-KS. The layered adsorbents use
    the multisite Langmuir model, which has no such rule. Defining a new layered model form after seeing the results
    is not justified.
  * `kinetics_table4.10`: Table 4.10 gives the measured kinetics of AC5-KS only. The layered adsorbents already use
    their own measured kinetics (Tables 3.12, 3.15; step 4), so the scenario does not apply.
* **Points:** the 210 selected S_nom points of the GP arms at n_train = 512, with unchanged IDs: P (30), G (45) and L
  (45) at alpha = 0.10, and G (45) and L (45) at alpha = 0.05. The baseline is their Step 6c S_nom labels, not
  rerun.
* **Runs:** 3 scenarios x 210 = 630, plus the spot checks, 3 x 10 = 30. Total 660 CSS runs; estimated 6 h on
  6 workers (range 5 to 9 h).
* **Runner:** `scripts/run_step7b.py --workers 6` under `caffeinate -i`, detached with `nohup`, with
  `HOMEBREW_NO_AUTO_UPDATE=1`, on the uv CPython 3.14.7 venv. It reuses `run_step6a.run_phase` with the Step 6c rules:
  blind monitoring, a once-only He retry, a pause at 2 % failures or 5 % cap hits, and `--point-timeout 7200` per
  attempt. Checkpoint / resume. Order: for each scenario, the 210-point block, then its spot check. Labels go to
  `data/generated/M_<scenario>.csv` and `M_<scenario>_cold.csv`.
* **Warm start:**
  * Each point starts from its own baseline CSS state (`results/step6c/S_nom/css_states`). The runner checks it is
    at distance 0, so it is the nearest state.
  * CSS is declared only after >= 10 plain cycles, as before.
* **Spot check:**
  * Per scenario, 10 points drawn with `numpy.random.default_rng(7)` (one stream, scenarios in the order above) are
    rerun from the He-filled column (`run_batch.py --cold-start`, new flag).
  * If any |z_cold - z_warm| > 0.01, or a run on either side failed, the workflow pauses and is reported before the
    next scenario.
* **Code changes** (tests in `tests/test_step7b.py`):
  * the `scenario` key and `design.scenario_settings`;
  * a `Cps` argument of `column.hy2seps_layer` / `layered_column`;
  * `run_batch.py --cold-start`;
  * a `cold_start` pass-through in `run_step6a.run_phase`.
  * Regression: without a scenario, the layered model rerun from its recorded warm start reproduces the Step 6c
    label of S_nom_0333 and its final state bitwise.
* **Analysis:** `python scripts/analyse_step7b.py [--generated-dir data/generated] [--out results/step7b/analysis]`.
  * Inputs: `S_nom.csv`, `M_<scenario>.csv`, `M_<scenario>_cold.csv`, `S_nom_memberships.csv`, the frozen
    `docs/validation/step6b/predictions/S_nom.csv`, and `results/step7b/workflow.json` (run status).
  * Outputs: `violations.csv`, `paired.csv`, `coverage.csv`, `fronts.csv`, `recovery_loss.csv`,
    `spot_checks.csv`, `statuses.csv` and `REPORT.md`. With the default `--out` these are copied to
    `docs/validation/step7b.md` and `docs/validation/step7b/`.
  * Violation rate (z > 2; censored = 3; failed runs excluded and counted) per arm (variant x alpha) for baseline
    and each scenario, with Clopper-Pearson 95 % CIs.
  * Paired baseline -> scenario changes on the same points, with the exact McNemar two-sided p reported as
    descriptive.
  * Coverage of the frozen nominal bounds (z <= bound; for P, the point prediction).
  * Step-7a fronts: reference point (recovery 0 %, productivity 0). Per arm: feasible points, realised front and
    hypervolume. Recovery at productivity 100, 125, ..., 250 mol/kg/day, interpolated on the realised front; loss
    vs P in the same scenario and change vs the same arm at baseline.
