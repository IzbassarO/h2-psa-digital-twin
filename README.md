# h2-psa-digital-twin

Open digital twin of a layered-bed hydrogen PSA unit: verified dynamic model, surrogate optimization
with conformal purity guarantees, and robustness under feed variability and adsorbent degradation.

> **Status:** step 1 of 7 — literature parameters transcribed and verified. Model code starts in step 2.

## Research question

Surrogate-based optimisation of PSA cycles pushes the operating point onto the product-purity
constraint (H₂ ≥ 99.99 %). That is exactly where surrogate error is largest, so "optimal" regimes can
violate the specification when re-checked with the physical model. We quantify how often this happens,
and replace the point-estimate constraint with a conformal-prediction bound that gives a calibrated
feasibility guarantee — and we report what the guarantee costs in H₂ recovery.

## Roadmap

| Step | Content | Gate |
|---|---|---|
| 1 ✅ | Literature parameters → `data/params/*.yaml`, `data/benchmarks/*.csv` | `scripts/check_params.py` passes; independent audit |
| 2 | Column model: mass/energy/momentum balances, bi-LDF, virial / multisite-Langmuir isotherms | Breakthrough curves of Lopes (2010) ch.5 |
| 3 | PSA cycle to cyclic steady state | Experiment 81.6 % / 99.9805 % and 17 reference simulations |
| 4 | Layered bed (activated carbon + zeolite) | Trends of Yang (2008) |
| 5 | Scenarios: industrial feed (NETL 2022), feed variability, adsorbent capacity loss | — |
| 6 | Design of experiments → surrogates → conformal intervals → NSGA-II | Interval coverage on held-out runs |
| 7 | Monte Carlo robustness, figures, manuscript, Zenodo release | One command regenerates every number |

## Layout

```
data/
  params/        transcribed parameters with page-level citations (YAML)
  benchmarks/    validation targets (CSV)
  PROVENANCE.md  sources, errata found in the sources, assumptions
src/h2psa/       package (isotherms, kinetics; model code from step 2)
scripts/         check_params.py and pipeline scripts
tests/           pytest suite
configs/         scenario and run configurations
literature/      source PDFs (git-ignored, not redistributed)
results/         generated outputs
```

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/check_params.py   # 25 consistency checks
pytest
```

## Data sources

Lopes (2010) PhD thesis, U. Porto (open access); NETL (2022) DOE/NETL-2022/3241; Yang et al. (2008).
See `data/PROVENANCE.md`, including six errata found in the sources and how they were resolved.

## Licence

Code: MIT (`LICENSE`). Transcribed data: CC BY 4.0 (`data/LICENSE-DATA.md`).

## Authors

Madina Sissenbay (Atyrau University of Oil and Gas) · Izbassar Orynbassar (Syracuse University)
