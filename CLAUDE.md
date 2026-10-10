# CLAUDE.md

## Goal

Open digital twin of a layered-bed H2 PSA unit; paper for *Computers & Chemical Engineering*.

Core claim: surrogate optimisation pushes operating points onto the 99.99 % H2 purity constraint,
where surrogate error is largest. We replace the point constraint with a conformal-prediction bound.

## Roadmap

The 7 steps in `README.md`. A step's gate must pass before the next step starts.
Every step's final gate report goes into git as `docs/validation/stepN.md` (a copy of the generated
`results/stepN/REPORT.md`) with its figures in `docs/validation/stepN/`; `results/` itself stays ignored.

## Setup

```bash
source .venv/bin/activate          # created with: python3 -m venv .venv && pip install -e ".[dev]"
python scripts/check_params.py     # must end with "25/25 checks passed"
pytest
```

## Data rules

- Every parameter lives in `data/params/*.yaml` with a page citation. Never hardcode physical parameters in code.
- Corrections of source values go in `data/PROVENANCE.md` with evidence.
- Thesis page numbers are printed pages; PDF page = printed + 34.
- Source PDFs are in `literature/` (git-ignored). Never commit them.
- Load YAML only via `h2psa.isotherms.load_yaml` — plain `yaml.safe_load` reads `9.71e5` as a string.

## Measured vs simulated (Lopes 2010, ch. 5)

- The PSA experiment's performance numbers (81.6 % recovery, 99.98054 % purity, 63 ppm CO, 101.1 mol/kg/day)
  are the author's SIMULATED values (p.256, practice stated on p.241). Treat them as the author's 18th simulation
  (verification, with Table 5.10), never as measurements. Earlier step-3 documents called them "experimental" by
  mistake.
- Measured cyclic data: outlet molar flows at the GC sample times, cycles 1 and 37 (Fig. 5.19); pressure,
  cycle 1 (Fig. 5.18a); thermocouples, cycles 1-10 (Fig. 5.18b) and cycles 1, 37 (Fig. 5.19). Validation means
  these.
- No input from the original author is available: EP inflow composition and gas-holdup treatment are declared
  assumptions (`data/PROVENANCE.md`), carried as an uncertainty band.

## Code rules

- Package lives in `src/h2psa`.
- SI units internally: pressure in Pa inside the model; isotherm helpers take bar.
- Every new module gets tests in `tests/`.
- Results go to `results/` (git-ignored).
