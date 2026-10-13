"""Evaluate frozen prereg-v1 metrics after S_nom and S_dist CSS generation.

This command neither fits surrogates nor runs physical simulations. Missing
selected-point labels, memberships, predictions or frozen reports stop it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.step6_metrics import analyse_generated


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    frozen_reports = ROOT / "docs" / "validation" / "step6b"
    parser.add_argument("--generated-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--predictions-dir", type=Path, default=frozen_reports / "predictions")
    parser.add_argument("--report-dir", type=Path, default=frozen_reports)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "step6b" / "metrics")
    args = parser.parse_args(argv)
    # Both registered bootstrap generators use exactly 2000 draws in real analyses.
    results = analyse_generated(args.generated_dir, args.predictions_dir, args.report_dir,
                                require_manifest=args.report_dir.resolve() == frozen_reports.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in results["tables"].items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(json_safe(results["results"]), indent=2, allow_nan=False) + "\n")
    print(f"Frozen metrics written to {args.out}; undefined values are recorded explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
