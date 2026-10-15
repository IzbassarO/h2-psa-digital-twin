"""Step 7b: model-form robustness re-simulation (docs/preregistration.md, Addendum A); no git.

    HOMEBREW_NO_AUTO_UPDATE=1 nohup caffeinate -i .venv/bin/python scripts/run_step7b.py --workers 6 \
        >> results/step7b/driver.log 2>&1 &

Points: the 210 selected S_nom points of the GP arms at n_train = 512 (P, G, L; alpha 0.10 and 0.05), unchanged IDs.
Scenarios: design.SCENARIOS, in that order. Per scenario:
1. block M_<scenario> (210 points), each warm-started from its own baseline CSS state
   (results/step6c/S_nom/css_states; checked to be at distance 0 under design.scaled, so it is the nearest state);
2. spot check M_<scenario>_cold: N_SPOT points drawn with default_rng(SPOT_SEED) (one stream over the scenarios in
   order), run from the He-filled column (--cold-start). Any |z_cold - z_warm| > TOL_DZ, or a failed run on either
   side, pauses the workflow before the next scenario.
Runner, flags and pause rules of Step 6c (`run_step6a.run_phase`: blind monitoring, once-only He retry, 2 % failure /
5 % cap pause, 2 h per attempt). Repeating the command resumes from checkpoints. Labels are exported to
data/generated/M_<scenario>.csv and M_<scenario>_cold.csv for the user to commit.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_batch as rb  # noqa: E402
import run_step6a as g  # noqa: E402
from h2psa import design  # noqa: E402

GEN = ROOT / "data" / "generated"
WORK = ROOT / "results" / "step7b"
BASELINE_STATES = ROOT / "results" / "step6c" / "S_nom" / "css_states"
N_SPOT, SPOT_SEED, TOL_DZ = 10, 7, 0.01
POINT_TIMEOUT_S = 7200


def selected_points() -> pd.DataFrame:
    """The S_nom points of every GP arm at n_train = 512, in S_nom order."""
    m = pd.read_csv(GEN / "S_nom_memberships.csv")
    ids = set(m.loc[(m["surrogate"] == "GP") & (m["n_train"] == 512), "id"])
    points = pd.read_csv(GEN / "S_nom_points.csv", float_precision="round_trip")
    return points.loc[points["id"].isin(ids)].reset_index(drop=True)


def spot_ids(ids) -> dict:
    rng = np.random.default_rng(SPOT_SEED)
    return {s: sorted(rng.choice(sorted(ids), N_SPOT, replace=False).tolist()) for s in design.SCENARIOS}


def write_inputs(block: str, frame: pd.DataFrame):
    inputs = WORK / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frame.to_csv(inputs / f"{block}_design.csv", index=False)                    # joined into the label CSV
    frame[["id", *g.DECISIONS, *g.DISTURBANCES, "anchor", "scenario"]].to_csv(inputs / f"{block}.csv", index=False)


def seed_states(block: str):
    """Copy each unfinished point's own baseline CSS state into the block and check it is at distance 0."""
    states = WORK / block / "css_states"
    states.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(WORK / "inputs" / f"{block}.csv", dtype={"id": str}, float_precision="round_trip")
    for row in rows.to_dict("records"):
        if (WORK / block / "cases" / f"{row['id']}.pkl").exists():
            continue
        target = states / f"{row['id']}.npz"
        if not target.exists():
            shutil.copy(BASELINE_STATES / target.name, target)
        with np.load(target) as d:
            if not np.array_equal(d["design"], design.scaled(rb.point_of(row))):
                raise ValueError(f"{block}/{row['id']}: seeded state is not the point's own baseline state")


def spot_check(scenario: str, generated: Path = GEN) -> pd.DataFrame:
    read = lambda name: pd.read_csv(generated / f"{name}.csv", float_precision="round_trip")  # noqa: E731
    pair = read(f"M_{scenario}_cold")[["id", "status", "z"]].merge(
        read(f"M_{scenario}")[["id", "status", "z"]], on="id", suffixes=("_cold", "_warm"), validate="one_to_one")
    pair["dz"] = pair["z_cold"] - pair["z_warm"]
    pair["ok"] = pair["dz"].abs().le(TOL_DZ)            # NaN (a failed run on either side) is not ok
    return pair.assign(scenario=scenario)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(6, 8), default=6)
    args = parser.parse_args()
    points = selected_points()
    if len(points) != 210:
        raise ValueError(f"expected 210 GP n_train = 512 points, found {len(points)}")
    spots = spot_ids(points["id"])
    blocks = []
    for s in design.SCENARIOS:
        blocks += [(s, f"M_{s}", points, False), (s, f"M_{s}_cold", points.loc[points["id"].isin(spots[s])], True)]
    for scenario, block, frame, _ in blocks:
        write_inputs(block, frame.assign(scenario=scenario))
        g.SIZES[block] = len(frame)                     # export_block checks the block size
    path = WORK / "workflow.json"
    state = json.loads(path.read_text()) if path.exists() else {"workers": args.workers, "phases": {}}
    if state["workers"] != args.workers:
        raise RuntimeError("Worker change requires a recorded deviation before resuming")
    if state.get("status", "").startswith("paused"):
        raise RuntimeError("Monitoring pause requires project-lead instruction before continuing")
    with (WORK / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for scenario, block, frame, cold in blocks:
            if not state["phases"].get(block, {}).get("complete"):
                if not cold:
                    seed_states(block)
                code = g.run_phase(block, block, args.workers, state, WORK, point_timeout=POINT_TIMEOUT_S,
                                   cold_start=cold)
                counts = g.metadata_counts(g.read_metadata(block, WORK))
                if code or counts["n"] != len(frame):
                    state["status"] = f"paused {block}: runner exit {code}; {counts}"
                    g.atomic_json(path, state)
                    print(state["status"], flush=True)
                    return code or 1
                csv = g.export_block(block, WORK)
                state["phases"][block].update(complete=True, counts=counts, csv=str(csv.relative_to(ROOT)))
                g.atomic_json(path, state)
                print(f"{block}: {counts}; labels exported to {csv}", flush=True)
            if cold:
                check = spot_check(scenario)
                check.to_csv(WORK / f"spot_check_{scenario}.csv", index=False)
                state["phases"][block]["spot_check"] = dict(n=len(check), ok=int(check["ok"].sum()),
                                                            max_abs_dz=float(check["dz"].abs().max()))
                if not check["ok"].all():
                    state["status"] = f"paused spot check {scenario}: {state['phases'][block]['spot_check']}"
                    g.atomic_json(path, state)
                    print(state["status"], flush=True)
                    return 4
                g.atomic_json(path, state)
    state["status"] = "complete — Step 7b simulations"
    g.atomic_json(path, state)
    print(json.dumps(state["phases"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
