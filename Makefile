# Reproduction levels (README.md, "Reproduce the paper"). Run inside the project venv; PY=... to override.
PY ?= python

.PHONY: figures check validate full test clean

# Level 1: figures F5-F10, FS7 and tables T1-T3 from the committed CSVs -> results/figures, results/tables
figures:
	$(PY) scripts/make_figures.py

# Level 2: parameter consistency checks + every number quoted in the paper recomputed from the committed CSVs
check:
	$(PY) scripts/check_params.py
	$(PY) scripts/check_paper_numbers.py

# Level 3: re-run the model validation against Lopes (2010): breakthrough runs 1-5 (Step 2), PSA cycle (Step 3).
# Rewrites docs/validation/step2.md, step3.md and their figures; compare with `git diff docs/validation`.
validate:
	$(PY) scripts/calibrate_thermal.py
	$(PY) scripts/run_breakthrough.py
	$(PY) scripts/run_psa.py --workers 4

# Level 4: the full study is a resumable multi-day workflow with pre-registered stop points, not one command.
full:
	@echo "The full study (3357 CSS runs, ~56 h on 6 workers) runs step by step; see README.md, 'Level 4'."
	@echo "Note: scripts/run_step6a.py commits and pushes each data block by design; use --defer-publication."
	@exit 1

test:
	$(PY) -m pytest

# Removes only what `make figures` / `make test` write. Never touches data/ or the simulation caches in results/.
clean:
	rm -rf results/figures results/tables .pytest_cache
	find src scripts tests -name __pycache__ -type d -prune -exec rm -rf {} +
