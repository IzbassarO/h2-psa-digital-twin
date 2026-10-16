# h2-psa-digital-twin

Open digital twin of a layered-bed (activated carbon / zeolite) hydrogen PSA unit, built only from published
adsorption data and validated against laboratory breakthrough and PSA-cycle experiments (Lopes 2010). We use it in a
pre-registered study of 3 357 cyclic-steady-state (CSS) simulations. The study asks whether surrogate-based
optimisation keeps a 99.99 % H2 purity specification (impurity ≤ 100 ppm) on the operating points it selects. It
tests point-estimate constraints and conformal-prediction bounds. One command regenerates every figure, and a second
recomputes every number quoted in the paper, both from the CSVs in this repository.

**Paper.** M. Sissenbay, I. Orynbassar, Y. Gilazhov. *A digital twin of a layered-bed hydrogen PSA unit: when do conformal purity
guarantees survive surrogate-based optimisation?* Submitted to the International Journal of Hydrogen Energy, 2026.
DOI: *to be added*.

## Key results

Violation = re-simulated impurity > 100 ppm at an optimiser-selected point (α = 0.10; 95 % CI Clopper-Pearson).
Every value below is recomputed by `make check` from `data/generated/`.

| Result | Value | Status | Report |
|---|---|---|---|
| RQ1 point-estimate constraint (P) violates | 165/179 = 92.2 % [87.2, 95.7] | pre-registered, confirmed | `docs/validation/step6c.md` |
| RQ2 global conformal bound (G) violates | 104/270 = 38.5 % [32.7, 44.6] | pre-registered, confirmed | `docs/validation/step6c.md` |
| RQ3 local conformal bound (L), pooled | 137/270 = 50.7 % | pre-registered, not confirmed | `docs/validation/step6c.md` |
| RQ5 miscoverage vs adsorbent capacity φ | OR 4.79 (φ 0.70 vs 1.00) | pre-registered, confirmed | `docs/validation/step6c.md` |
| Split by surrogate: GP-L / GBM-L violate | 5/135 / 132/135 | exploratory | `docs/validation/step6d.md` |
| Replication, new seeds: H_B1 GP-L ≤ 17 % | 1/135 = 0.7 % [0.0, 4.1] | pre-registered (Add. B), confirmed | `docs/validation/step8a.md` |
| Replication, new seeds: H_B2 GBM-L > 10 % | 130/135 = 96.3 % [91.6, 98.8] | pre-registered (Add. B), confirmed | `docs/validation/step8a.md` |
| Cost of the GP-L guarantee (n_train 512) | 4.2–9.2 pp H2 recovery at matched productivity | exploratory | `docs/validation/step7a.md` |

## Repository map

| Path | Contents |
|---|---|
| `src/h2psa/` | The model package: isotherms, kinetics, transport correlations, 1D column model (`column.py`), PSA cycle to CSS (`cycle.py`), design-space mapping, surrogates and conformal bounds (`step6.py`), frozen pre-registered metrics (`step6_metrics.py`) |
| `scripts/` | Pipeline steps (`run_*.py`), analyses (`analyse_*.py`, `explore_*.py`), `make_figures.py`, `check_paper_numbers.py`, `check_params.py` |
| `data/params/` | Every physical parameter, transcribed with a page citation (YAML) |
| `data/benchmarks/` | Validation targets: experiments digitised from Lopes (2010) |
| `data/generated/` | All simulation results: labels of the 3 357 CSS runs, selected points, arm memberships, model traces for F5/F6 |
| `data/PROVENANCE.md` | Sources, errata E1–E7 found in them, declared assumptions |
| `configs/` | Design space and disturbance ranges |
| `docs/preregistration.md` | Frozen pre-registration (tag `prereg-v1`), Addenda A and B, deviations |
| `docs/validation/` | Gate report of every step (`stepN.md`) with its tables and figures |
| `tests/` | pytest suite |
| `literature/` | Source PDFs (not redistributed; `literature/README.md` has the links) |
| `results/` | Generated output (git-ignored) |

## Quick start

Python ≥ 3.12; the results were produced with Python 3.14.7 on macOS (Apple M1 Pro). The pre-registration pins
Python 3.14.7 and the analysis libraries. With any other version, `make check` prints a warning but still verifies
the input hashes and every number (e.g. `uv python install 3.14.7` gives the exact interpreter).

```bash
git clone https://github.com/IzbassarO/h2-psa-digital-twin && cd h2-psa-digital-twin
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install --no-deps -e .   # exact versions of the paper
# or: pip install -e ".[dev]"                                       # compatible versions
make figures     # paper figures -> results/figures/
make check       # every paper number recomputed and compared
```

## Reproduce the paper

| Level | Command | What it does | Time (M1 Pro) |
|---|---|---|---|
| 1 | `make figures` | Figures F5–F10, FS7 (PDF + 300 dpi PNG), captions and tables T1–T3 from the committed CSVs, written to `results/figures/` and `results/tables/` | ~1 min |
| 2 | `make check` | `check_params.py` (25 parameter checks), then `check_paper_numbers.py`. The second script recomputes RQ1–RQ5, the 6d split, the 7a cost, 7b and the 8a replication with the frozen analysis code and prints `quantity \| paper \| recomputed \| OK`. It exits with code 1 on any mismatch | ~2.5 min |
| 3 | `make validate` | Re-runs the model validation against Lopes (2010): the thermal calibration (not adopted, reported), breakthrough runs 1–5 (Step 2), and the PSA cycle with its 17 reference simulations and sensitivity cases (Step 3). It rewrites `docs/validation/step2*` and `step3*`, so compare with `git diff docs/validation`; only the runtimes should differ | Step 2: 23 min breakthrough + ~10 min calibration; Step 3: ~8 CPU-h (2–3 h on 4 workers) |
| 4 | `make full` | The full study: 3 357 CSS runs | ~56 h on 6 workers |

The test suite runs with `make test`. `make clean` removes only `results/figures`, `results/tables` and caches.

**Level 4.** The study is a resumable workflow with pre-registered stop points, so `make full` prints these steps
instead of running them. Each simulation step resumes from checkpoints when it is repeated. Wall times are those
recorded in the reports.

```bash
python scripts/run_step6a.py --prepare
caffeinate -i python scripts/run_step6a.py --workers 6 --defer-publication  # D_train/cal/test/shift, 1 152 runs, 17.0 h
python scripts/run_step6b.py all --workers 6                                # surrogates, conformal bounds, NSGA-II selection
caffeinate -i python scripts/run_step6c.py --workers 6                      # S_nom + S_dist, 1 140 runs, 30.5 h
python scripts/analyse_step6.py && python scripts/report_step6c.py          # pre-registered analysis
python scripts/explore_step6d.py && python scripts/explore_step7a.py        # exploratory (6d needs the 6b models)
caffeinate -i python scripts/run_step7b.py --workers 6 && python scripts/analyse_step7b.py   # 660 runs, 4.9 h
python scripts/run_step8a.py select
caffeinate -i python scripts/run_step8a.py simulate --workers 6 && python scripts/analyse_step8a.py  # 405 runs, 3.7 h
make figures check
```

`run_step6a.py` was the registered data-publication workflow. Without `--defer-publication` it commits each data
block and pushes it to `origin`. With the flag it stops after each block so that the block can be committed.

## Pre-registration

The hypotheses, metrics, tests and library versions were frozen before any study simulation in
`docs/preregistration.md`, tagged **`prereg-v1`** (commit `36079a2`). To verify that the frozen text is unchanged:

```bash
git show prereg-v1:docs/preregistration.md | shasum -a 256
# 4f091034c1042dc009f79b147e0b069402ee33c530a909ad668840347ec8fd8e
```

The current `docs/preregistration.md` adds, below the frozen sections, Addendum A (Step 7b, model-form robustness,
descriptive) and Addendum B (Step 8a, confirmatory replication of the Step 6d split, registered before any
replication label existed), plus the dated deviations. `scripts/run_step8a.py` refuses to run unless Addendum B
and the Step 8a scripts are committed unchanged.

## Data source

All model parameters and validation data come from F.V.S. Lopes, *New pressure swing adsorption cycles for hydrogen
purification from steam reforming off-gases*, PhD thesis, University of Porto, 2010, doi:10.34626/ftfe-yr63 (open
access). Seven errata in the thesis tables (E1–E7) and their resolution are documented in `data/PROVENANCE.md`.
The thesis PSA performance figures (81.6 % recovery, 99.98 % purity) are the author's simulations, not
measurements. Validation uses the measured outlet flows, pressures and temperatures (Figs 5.15–5.19).

## License, citation, contact

* Code: MIT (`LICENSE`). Data in `data/`, including the simulation results in `data/generated/`: CC BY 4.0
  (`data/LICENSE-DATA.md`).
* Cite the paper above and this repository (`CITATION.cff`; archived on Zenodo, DOI to be added).
* Corresponding author: Madina Sissenbay (Safi Utebayev Atyrau University of Oil and Gas), m.sisenbay@aogu.edu.kz.
  Izbassar Orynbassar (School of Information Studies, Syracuse University). Yessengaly Gilazhov (Institute of
  Petrochemical Engineering and Ecology named after N.K. Nadirov, Safi Utebayev Atyrau University of Oil and Gas).
