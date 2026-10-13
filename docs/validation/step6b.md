# Step 6b — frozen surrogates, conformal diagnostics and selection

Registration: `prereg-v1`; no new CSS simulations in this step.

## Step 6a confirmation

| block | n | css | censored | cap | failed |
|---|---:|---:|---:|---:|---:|
| D_train | 512 | 432 | 80 | 0 | 0 |
| D_cal | 256 | 215 | 41 | 0 | 0 |
| D_test | 256 | 214 | 41 | 1 | 0 |
| D_shift | 128 | 112 | 16 | 0 | 0 |

Batch wall time 16.99 h; elapsed through final publication 23.85 h.
The sole cap, D_test_0143 at 2000 cycles, retains last-cycle impurity/z/objectives and status=cap.
Censored z=3; censored objective labels are excluded from fitting and objective RMSE.

## Frozen fitting and calibration

Only D_train nested prefixes are fitted. GP uses marginal likelihood with 10 restarts; GBM uses all 54 combinations and five-fold CV.
Full chosen hyperparameters: [hyperparameters.json](step6b/hyperparameters.json).
All model files are frozen and hashed before D_test is read.
Conformal ranks are 232 (alpha 0.10) and 245 (alpha 0.05), with no interpolation or clipping of q.
All 24 quantiles: [q.csv](step6b/q.csv).

## Sanity gate and context

Failed sanity arms: none.
Per-arm coverage, two-sided CP intervals and one-sided upper gate: [gate.csv](step6b/gate.csv).
Independent arithmetic checks on any failed arms: [gate_audit.json](step6b/gate_audit.json).
Fixed band counts/widening: [band.csv](step6b/band.csv).
Descriptive held-out RMSE only: [accuracy.csv](step6b/accuracy.csv). No method changed from diagnostics.

## Optimisation and selection

Population 100, generations 250, seeds 10/11/12, SBX eta 15/p 0.9, PM eta 20/per-coordinate p 1/8.
Canonical inverse-bin repair precedes predictions. Final feasible nondominated fronts supply actual productivity quantiles.
Short fronts retain all distinct points. Shared physical points have one run ID and count once per arm.
Selection totals: `{'unique_nominal': 900, 'unique_disturbed': 240, 'nominal_memberships': 900, 'within_arm_duplicates': 0, 'global_repeated_memberships': 0, 'front_shortfall': 0}`.
Per-arm counts: [selection_arms.csv](step6b/selection_arms.csv); per-seed shortfalls: [selection_seeds.csv](step6b/selection_seeds.csv).
Repeated memberships: [duplicates.csv](step6b/duplicates.csv); complete memberships are committed alongside point lists.
One rng4 disturbance draw per eligible unique nominal point gives paired S_dist points.

## Future analysis

`scripts/analyse_step6.py` implements the registered metrics; synthetic tests run before its commit.
After Step 6c supplies S_nom.csv and S_dist.csv, run `python scripts/analyse_step6.py`.
It uses the committed memberships and frozen predictions under docs/validation/step6b/predictions.
Analysis does not refit models. The selected lists contain no re-simulation labels.
Step 6c has not started. Deviations/declarations: docs/preregistration.md, 2026-10-13.
Focused synthetic tests passed before publication; independent arithmetic and selection audits passed.

## Quantiles by model

| surrogate | n_train | G, 0.10 | L, 0.10 | G, 0.05 | L, 0.05 |
|---|---:|---:|---:|---:|---:|
| GP | 128 | 0.761873 | 1.362920 | 1.150709 | 1.824376 |
| GP | 256 | 0.662211 | 1.442347 | 0.957189 | 2.168776 |
| GP | 512 | 0.493529 | 1.185329 | 0.788413 | 1.557675 |
| GBM | 128 | 0.940914 | 0.205561 | 1.219819 | 0.000000 |
| GBM | 256 | 0.979886 | 0.153805 | 1.310907 | 0.000000 |
| GBM | 512 | 0.871268 | 0.230179 | 1.180012 | 0.295350 |

## Per-arm test gates and fixed bands

| arm | covered / 256 | coverage | two-sided CP 95% | one-sided upper 95% | gate | band | n initial → final | widened |
|---|---:|---:|---|---:|---|---|---:|---|
| GP_n128_G_a0.10 | 236/256 | 0.9219 | [0.8819, 0.9516] | 0.9476 | pass | [1.25, 2.00] | 20 → 35 | True |
| GP_n128_L_a0.10 | 238/256 | 0.9297 | [0.8912, 0.9578] | 0.9541 | pass | [1.25, 2.00] | 17 → 36 | True |
| GP_n256_G_a0.10 | 237/256 | 0.9258 | [0.8865, 0.9547] | 0.9509 | pass | [1.25, 2.00] | 21 → 32 | True |
| GP_n256_L_a0.10 | 236/256 | 0.9219 | [0.8819, 0.9516] | 0.9476 | pass | [1.25, 2.00] | 21 → 29 | True |
| GP_n512_G_a0.10 | 228/256 | 0.8906 | [0.8458, 0.9261] | 0.9211 | pass | [1.25, 2.00] | 21 → 32 | True |
| GP_n512_L_a0.10 | 234/256 | 0.9141 | [0.8728, 0.9454] | 0.9411 | pass | [1.25, 2.00] | 22 → 33 | True |
| GP_n128_G_a0.05 | 251/256 | 0.9805 | [0.9550, 0.9936] | 0.9923 | pass | [1.25, 2.00] | 28 → 39 | True |
| GP_n128_L_a0.05 | 249/256 | 0.9727 | [0.9445, 0.9889] | 0.9871 | pass | [1.25, 2.00] | 24 → 36 | True |
| GP_n256_G_a0.05 | 246/256 | 0.9609 | [0.9293, 0.9811] | 0.9787 | pass | [1.25, 2.00] | 22 → 30 | True |
| GP_n256_L_a0.05 | 247/256 | 0.9648 | [0.9343, 0.9838] | 0.9815 | pass | [1.25, 2.00] | 19 → 32 | True |
| GP_n512_G_a0.05 | 244/256 | 0.9531 | [0.9196, 0.9755] | 0.9727 | pass | [1.25, 2.00] | 21 → 31 | True |
| GP_n512_L_a0.05 | 241/256 | 0.9414 | [0.9052, 0.9668] | 0.9635 | pass | [1.25, 2.00] | 18 → 32 | True |
| GBM_n128_G_a0.10 | 235/256 | 0.9180 | [0.8773, 0.9485] | 0.9444 | pass | [1.25, 2.00] | 28 → 42 | True |
| GBM_n128_L_a0.10 | 231/256 | 0.9023 | [0.8592, 0.9358] | 0.9312 | pass | [1.50, 2.00] | 33 → 33 | False |
| GBM_n256_G_a0.10 | 240/256 | 0.9375 | [0.9005, 0.9639] | 0.9604 | pass | [1.25, 2.00] | 27 → 44 | True |
| GBM_n256_L_a0.10 | 224/256 | 0.8750 | [0.8281, 0.9129] | 0.9076 | pass | [1.25, 2.00] | 26 → 41 | True |
| GBM_n512_G_a0.10 | 231/256 | 0.9023 | [0.8592, 0.9358] | 0.9312 | pass | [1.25, 2.00] | 21 → 37 | True |
| GBM_n512_L_a0.10 | 231/256 | 0.9023 | [0.8592, 0.9358] | 0.9312 | pass | [1.25, 2.00] | 29 → 40 | True |
| GBM_n128_G_a0.05 | 247/256 | 0.9648 | [0.9343, 0.9838] | 0.9815 | pass | [1.25, 2.00] | 27 → 47 | True |
| GBM_n128_L_a0.05 | 239/256 | 0.9336 | [0.8958, 0.9608] | 0.9572 | pass | [1.50, 2.00] | 36 → 36 | False |
| GBM_n256_G_a0.05 | 250/256 | 0.9766 | [0.9497, 0.9914] | 0.9897 | pass | [1.50, 2.00] | 37 → 37 | False |
| GBM_n256_L_a0.05 | 256/256 | 1.0000 | [0.9857, 1.0000] | 1.0000 | pass | [1.25, 2.00] | 0 → 0 | True |
| GBM_n512_G_a0.05 | 243/256 | 0.9492 | [0.9147, 0.9727] | 0.9697 | pass | [1.25, 2.00] | 28 → 38 | True |
| GBM_n512_L_a0.05 | 244/256 | 0.9531 | [0.9196, 0.9755] | 0.9727 | pass | [1.50, 2.00] | 33 → 33 | False |

P bands (not a conformal gate):

| arm | band | n initial → final | widened |
|---|---|---:|---|
| GP_n128_P_a0.10 | [1.25, 2.00] | 27 → 38 | True |
| GP_n256_P_a0.10 | [1.25, 2.00] | 23 → 34 | True |
| GP_n512_P_a0.10 | [1.25, 2.00] | 21 → 31 | True |
| GBM_n128_P_a0.10 | [1.25, 2.00] | 23 → 41 | True |
| GBM_n256_P_a0.10 | [1.25, 2.00] | 27 → 42 | True |
| GBM_n512_P_a0.10 | [1.50, 2.00] | 33 → 33 | False |

## Descriptive test RMSE

| surrogate | n_train | z | recovery (pp) | productivity (mol/kg/day) |
|---|---:|---:|---:|---:|
| GP | 128 | 0.578684 | 5.772946 | 4.769382 |
| GP | 256 | 0.495709 | 4.091527 | 3.962243 |
| GP | 512 | 0.456252 | 3.763945 | 2.806205 |
| GBM | 128 | 0.781613 | 12.489813 | 16.801862 |
| GBM | 256 | 0.750512 | 12.218295 | 11.906412 |
| GBM | 512 | 0.745694 | 7.247476 | 7.808467 |

## Counts per selected arm

| arm | requested | unique in arm | duplicate memberships | front shortfall |
|---|---:|---:|---:|---:|
| GP_n128_P_a0.10 | 30 | 30 | 0 | 0 |
| GP_n128_G_a0.10 | 45 | 45 | 0 | 0 |
| GP_n128_L_a0.10 | 45 | 45 | 0 | 0 |
| GP_n256_P_a0.10 | 30 | 30 | 0 | 0 |
| GP_n256_G_a0.10 | 45 | 45 | 0 | 0 |
| GP_n256_L_a0.10 | 45 | 45 | 0 | 0 |
| GP_n512_P_a0.10 | 30 | 30 | 0 | 0 |
| GP_n512_G_a0.10 | 45 | 45 | 0 | 0 |
| GP_n512_L_a0.10 | 45 | 45 | 0 | 0 |
| GP_n512_G_a0.05 | 45 | 45 | 0 | 0 |
| GP_n512_L_a0.05 | 45 | 45 | 0 | 0 |
| GBM_n128_P_a0.10 | 30 | 30 | 0 | 0 |
| GBM_n128_G_a0.10 | 45 | 45 | 0 | 0 |
| GBM_n128_L_a0.10 | 45 | 45 | 0 | 0 |
| GBM_n256_P_a0.10 | 30 | 30 | 0 | 0 |
| GBM_n256_G_a0.10 | 45 | 45 | 0 | 0 |
| GBM_n256_L_a0.10 | 45 | 45 | 0 | 0 |
| GBM_n512_P_a0.10 | 30 | 30 | 0 | 0 |
| GBM_n512_G_a0.10 | 45 | 45 | 0 | 0 |
| GBM_n512_L_a0.10 | 45 | 45 | 0 | 0 |
| GBM_n512_G_a0.05 | 45 | 45 | 0 | 0 |
| GBM_n512_L_a0.05 | 45 | 45 | 0 | 0 |
