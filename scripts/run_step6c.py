"""Step 6c: simulate the selected points S_nom (900) then S_dist (240); no git.

    caffeinate -i python scripts/run_step6c.py --workers 6

Same runner, flags and pause rules as Step 6a (`run_step6a.run_phase`: blind monitoring, once-only He retry,
2 % failure / 5 % cap pause), on the frozen point lists of Step 6b (one simulation ID per unique physical point).
Repeating the command resumes from checkpoints. Labels are exported to data/generated/{S_nom,S_dist}.csv for the
user to commit.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_step6a as g  # noqa: E402

WORK = ROOT / "results" / "step6c"
SIZES = {"S_nom": 900, "S_dist": 240}
POINT_TIMEOUT_S = 7200          # 2 h per attempt, uniform for Step 6c (Deviations 2026-10-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(6, 8), default=8)
    args = parser.parse_args()
    g.SIZES.update(SIZES)                      # export_block checks the registered block size
    inputs = WORK / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for block in SIZES:
        points = pd.read_csv(ROOT / "data" / "generated" / f"{block}_points.csv", float_precision="round_trip")
        points.to_csv(inputs / f"{block}_design.csv", index=False)               # joined into the label CSV
        points[["id", *g.DECISIONS, *g.DISTURBANCES, "anchor"]].to_csv(inputs / f"{block}.csv", index=False)
    path = WORK / "workflow.json"
    state = json.loads(path.read_text()) if path.exists() else {"workers": args.workers, "phases": {}}
    if state["workers"] != args.workers:
        raise RuntimeError("Worker change requires a recorded deviation before resuming")
    if state.get("status", "").startswith("paused"):
        raise RuntimeError("Monitoring pause requires project-lead instruction before continuing")
    with (WORK / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for block, size in SIZES.items():
            if state["phases"].get(block, {}).get("complete"):
                continue
            code = g.run_phase(block, block, args.workers, state, WORK, point_timeout=POINT_TIMEOUT_S)
            counts = g.metadata_counts(g.read_metadata(block, WORK))
            if code or counts["n"] != size:
                state["status"] = f"paused {block}: runner exit {code}; {counts}"
                g.atomic_json(path, state)
                print(state["status"], flush=True)
                return code or 1
            csv = g.export_block(block, WORK)
            state["phases"][block].update(complete=True, counts=counts, csv=str(csv.relative_to(ROOT)))
            g.atomic_json(path, state)
            print(f"{block}: {counts}; labels exported to {csv}", flush=True)
    state["status"] = "complete — Step 6c simulations"
    g.atomic_json(path, state)
    print(json.dumps({b: state["phases"][b] for b in SIZES}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
