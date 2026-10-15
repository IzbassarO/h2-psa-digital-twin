"""Step 8a: confirmatory replication of the Step 6d finding (docs/preregistration.md, Addendum B); no git writes.

    python scripts/run_step8a.py select
    HOMEBREW_NO_AUTO_UPDATE=1 nohup caffeinate -i .venv/bin/python scripts/run_step8a.py simulate --workers 6 \
        >> results/step8a/driver.log 2>&1 &

select: the frozen Step 6b surrogates and q, NSGA-II seeds 13/14/15, 9 arms (GP-L, GBM-L, GP-G; n_train 128/256/512;
alpha 0.10) x 3 seeds x 15 points, selection and one-ID-per-physical-point lists of Step 6b. Refuses unless Addendum B
and both Step 8a scripts are committed unchanged; first reruns one Step 6b optimisation and stops unless it is
reproduced exactly. Writes data/generated/S_rep_points.csv and S_rep_memberships.csv. A point whose 8 decision
variables equal an S_nom point's gets that ID in `snom_id` and is not simulated (its Step 6c label is reused).
simulate: the Step 6c runner and rules (`run_step6a.run_phase`: blind monitoring, warm start from the nearest finished
state of this batch, once-only He retry, 2 % failure / 5 % cap pause, --point-timeout 7200) on the points without
snom_id. Refuses unless the point list is committed unchanged. Labels: data/generated/S_rep.csv. Repeating the
command resumes from checkpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_step6a as g  # noqa: E402
import run_step6b as b6  # noqa: E402  (also pins BLAS threads to 1)
from h2psa.selection import CubeRepair, SelectionProblem, build_selected_lists, front_and_selection  # noqa: E402
from h2psa.step6 import DECISIONS, Surrogates, file_hash, registered_arms  # noqa: E402
from h2psa.step6_metrics import PINNED_SOFTWARE  # noqa: E402

GEN = ROOT / "data" / "generated"
WORK = ROOT / "results" / "step8a"
BLOCK = "S_rep"
SEEDS = (13, 14, 15)
GROUPS = (("GP", "L"), ("GBM", "L"), ("GP", "G"))
ARMS = [a for a in registered_arms() if a["alpha"] == .10 and (a["surrogate"], a["variant"]) in GROUPS]
REPRO = ("GP_n128_L_a0.10", 10)                        # Step 6b optimisation rerun before the new seeds
POINT_TIMEOUT_S = 7200
FILES = {name: GEN / f"{BLOCK}_{name}.csv" for name in ("points", "memberships")}


def committed(*paths):
    for path in paths:
        rel = str(Path(path).relative_to(ROOT))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True)
        if tracked.returncode or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode:
            raise RuntimeError(f"{rel} must be committed unchanged first")


def check_environment():
    if platform.python_version() != "3.14.7" or {n: version(n) for n in PINNED_SOFTWARE} != PINNED_SOFTWARE:
        raise RuntimeError("Environment differs from the pinned prereg-v1 versions")


def frozen_quantiles() -> dict:
    state6b = json.loads((b6.WORK / "workflow.json").read_text())
    b6.verify_fit_freeze(state6b)                      # 24 model hashes
    return b6.quantiles(state6b)                       # q.csv hash


def nsga(arm: dict, seed: int, qs: dict) -> dict:
    """selection.optimise_seed with an unrestricted seed (it admits only the Step 6b seeds 10-12)."""
    s = Surrogates(arm["surrogate"], arm["n_train"], b6.WORK)
    bound = lambda X: s.bound(X, arm["variant"], arm["alpha"], qs[arm["arm"]])  # noqa: E731
    algorithm = NSGA2(pop_size=100, crossover=SBX(eta=15, prob=.9), mutation=PM(eta=20, prob=1.0, prob_var=1 / 8),
                      repair=CubeRepair(), eliminate_duplicates=True)
    result = minimize(SelectionProblem(s.predict, bound), algorithm, ("n_gen", 250), seed=seed, verbose=False,
                      save_history=False)
    return front_and_selection(result.pop.get("X"), s.predict, bound, 15)


def optimise(arm: dict, seed: int, qs: dict) -> dict:
    path = WORK / "optimization" / f"{arm['arm']}_seed{seed}.joblib"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(nsga(arm, seed, qs), path.with_suffix(".part"))
        path.with_suffix(".part").replace(path)
    return joblib.load(path)


def reproduces_step6b(qs: dict) -> bool:
    arm = next(a for a in ARMS if a["arm"] == REPRO[0])
    new, old = nsga(arm, REPRO[1], qs), joblib.load(b6.WORK / "optimization" / f"{REPRO[0]}_seed{REPRO[1]}.joblib")
    return all(np.array_equal(new[k], old[k]) for k in ("front_X", "selected_X", "selected_bound"))


def decision_key(row) -> tuple:
    return tuple(float(row[name]) for name in DECISIONS)


def point_lists(outputs: list) -> dict[str, pd.DataFrame]:
    """Step 6b lists with S_rep IDs; S_nom matches get snom_id. The disturbed lists are not used."""
    lists = build_selected_lists(outputs, b6.map_point, b6.map_disturbance)
    points, members = lists["nominal_points"], lists["nominal_memberships"]
    for frame in (points, members):
        frame["id"] = frame["id"].str.replace("S_nom_", f"{BLOCK}_", regex=False)
    snom = pd.read_csv(GEN / "S_nom_points.csv", float_precision="round_trip")
    lookup = {decision_key(r): r["id"] for r in snom.to_dict("records")}
    points["snom_id"] = [lookup.get(decision_key(r)) for r in points.to_dict("records")]
    return {"points": points, "memberships": members, "seeds": lists["shortages"]}


def select(workers: int):
    committed(ROOT / "docs" / "preregistration.md", ROOT / "scripts" / "run_step8a.py",
              ROOT / "scripts" / "analyse_step8a.py")
    check_environment()
    if any(path.exists() for path in FILES.values()):
        raise RuntimeError("Step 8a points already exist; selection is not repeated")
    qs = frozen_quantiles()
    if not reproduces_step6b(qs):
        raise RuntimeError(f"Step 6b optimisation {REPRO} not reproduced; nothing selected")
    print(f"Step 6b optimisation {REPRO} reproduced exactly.", flush=True)
    tasks = [(arm, seed) for arm in ARMS for seed in SEEDS]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(optimise, *zip(*tasks), [qs] * len(tasks)))
    lists = point_lists([({**arm, "seed": seed}, r) for (arm, seed), r in zip(tasks, results)])
    for name, path in FILES.items():
        lists[name].to_csv(path, index=False)
    WORK.mkdir(parents=True, exist_ok=True)
    lists["seeds"].to_csv(WORK / "selection_seeds.csv", index=False)
    members, points = lists["memberships"], lists["points"]
    summary = dict(memberships=len(members), unique_points=len(points),
                   reused_from_S_nom=int(points["snom_id"].notna().sum()),
                   to_simulate=int(points["snom_id"].isna().sum()),
                   within_arm_duplicates=int((~members["counted_in_arm"]).sum()),
                   front_shortfall=int(lists["seeds"]["shortfall"].sum()))
    state = {"selection": summary, "selected_sha256": {n: file_hash(p) for n, p in FILES.items()},
             "reproduced": list(REPRO), "phases": {}}
    g.atomic_json(WORK / "workflow.json", state)
    print(json.dumps(summary), flush=True)


def simulate(workers: int):
    committed(*FILES.values())
    check_environment()
    path = WORK / "workflow.json"
    state = json.loads(path.read_text())
    if state["selected_sha256"] != {n: file_hash(p) for n, p in FILES.items()}:
        raise RuntimeError("Point list differs from the one selected")
    if state.setdefault("workers", workers) != workers:
        raise RuntimeError("Worker change requires a recorded deviation before resuming")
    if state.get("status", "").startswith("paused"):
        raise RuntimeError("Monitoring pause requires project-lead instruction before continuing")
    points = pd.read_csv(FILES["points"], float_precision="round_trip")
    run = points.loc[points["snom_id"].isna()].drop(columns="snom_id")
    inputs = WORK / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    run.to_csv(inputs / f"{BLOCK}_design.csv", index=False)                    # joined into the label CSV
    run[["id", *g.DECISIONS, *g.DISTURBANCES, "anchor"]].to_csv(inputs / f"{BLOCK}.csv", index=False)
    g.SIZES[BLOCK] = len(run)                                                 # export_block checks the size
    with (WORK / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = g.run_phase(BLOCK, BLOCK, workers, state, WORK, point_timeout=POINT_TIMEOUT_S)
        counts = g.metadata_counts(g.read_metadata(BLOCK, WORK))
        if code or counts["n"] != len(run):
            state["status"] = f"paused {BLOCK}: runner exit {code}; {counts}"
            g.atomic_json(path, state)
            print(state["status"], flush=True)
            return code or 1
        csv = g.export_block(BLOCK, WORK)
        state["phases"][BLOCK].update(complete=True, counts=counts, csv=str(csv.relative_to(ROOT)))
        state["status"] = "complete — Step 8a simulations"
        g.atomic_json(path, state)
    print(f"{BLOCK}: {counts}; labels exported to {csv}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "simulate"))
    parser.add_argument("--workers", type=int, choices=(6, 8), default=6)
    args = parser.parse_args()
    return (select if args.phase == "select" else simulate)(args.workers) or 0


if __name__ == "__main__":
    raise SystemExit(main())
