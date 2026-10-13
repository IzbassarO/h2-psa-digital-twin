"""Generate prereg-v1 data only; report status/timing, never response distributions.

    python scripts/run_step6a.py --prepare
    caffeinate -i python scripts/run_step6a.py --workers 6

The pilot is the first 32 D_train points, in the same checkpoint directory as the
remaining 480. A monitoring gate or failed push stops the workflow before the next
phase. Repeating the command resumes completed points and completed commits.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa import design
from h2psa.isotherms import load_yaml

DECISIONS = ("t_feed", "t_blow", "purge_to_feed", "P_high", "P_low", "f_zeolite",
             "n_equalisations", "feed_flow")
DISTURBANCES = ("phi", "y_CH4", "y_CO")
SIZES = {"D_train": 512, "D_cal": 256, "D_test": 256, "D_shift": 128}
WORK = ROOT / "results" / "step6a"


def sample_blocks() -> dict[str, pd.DataFrame]:
    """Exactly the registered draws and mappings, independent of any responses."""
    ds = load_yaml(design.CONFIGS / "design_space.yaml")["decision_variables"]
    disturbances = load_yaml(design.CONFIGS / "disturbances.yaml")["variables"]
    cubes = {"D_train": qmc.Sobol(d=8, scramble=True, seed=20261011).random_base2(9),
             "D_cal": np.random.default_rng(1).random((256, 8)),
             "D_test": np.random.default_rng(2).random((256, 8)),
             "D_shift": np.random.default_rng(3).random((128, 11))}
    frames = {}
    for block, cube in cubes.items():
        names = DECISIONS + DISTURBANCES if block == "D_shift" else DECISIONS
        frame = pd.DataFrame(cube, columns=[f"u_{name}" for name in names])
        frame.insert(0, "id", [f"{block}_{i:04d}" for i in range(len(cube))])
        for j, name in enumerate(DECISIONS):
            u = cube[:, j]
            if name == "f_zeolite":
                frame[name] = np.minimum(np.floor(31 * u), 30) / 30
            elif name == "n_equalisations":
                frame[name] = np.minimum(1 + np.floor(3 * u), 3).astype(int)
            else:
                lo, hi = ds[name]["bounds"]
                frame[name] = lo + (hi - lo) * u
        for j, name in enumerate(DISTURBANCES, 8):
            variable = disturbances[name]
            if block == "D_shift":
                lo, hi = variable["bounds"]
                frame[name] = lo + (hi - lo) * cube[:, j]
            else:
                frame[name] = variable["nominal"]
        frame["anchor"] = "lopes"
        frames[block] = frame
    return frames


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def prepare(work_root: Path = WORK):
    inputs = work_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frames = sample_blocks()
    for block, frame in frames.items():
        path = inputs / f"{block}_design.csv"
        if path.exists():
            prior = pd.read_csv(path, float_precision="round_trip")
            pd.testing.assert_frame_equal(prior, frame, check_exact=True)
        else:
            frame.to_csv(path, index=False)
        physical = frame[["id", *DECISIONS, *DISTURBANCES, "anchor"]]
        physical.to_csv(inputs / f"{block}.csv", index=False)
        if block == "D_train":
            physical.iloc[:32].to_csv(inputs / "D_train_pilot.csv", index=False)
    provenance = work_root / "provenance.json"
    if not provenance.exists():
        atomic_json(provenance, {
            "prereg_tag": "prereg-v1",
            "prereg_commit": subprocess.check_output(["git", "rev-parse", "prereg-v1"], cwd=ROOT,
                                                       text=True).strip(),
            "python": sys.version,
            "libraries": {name: importlib.metadata.version(name)
                          for name in ("numpy", "scipy", "pandas", "scikit-learn", "pymoo", "statsmodels")},
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(inputs.glob("*.csv"))},
        })
    print("Prepared registered inputs: D_train=512 (pilot prefix=32), D_cal=256, D_test=256, D_shift=128.",
          flush=True)


def read_metadata(block: str, work_root: Path = WORK) -> pd.DataFrame:
    """Read only columns permitted for generation monitoring."""
    path = work_root / block / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    allowed = {"id", "status", "cycles", "wall_s", "wall_min", "solver_failures", "attempts", "start"}
    return pd.read_csv(path, usecols=lambda name: name in allowed)


def metadata_counts(frame: pd.DataFrame) -> dict:
    counts = {status: int((frame["status"] == status).sum()) if len(frame) else 0
              for status in ("css", "censored", "cap", "failed")}
    failures = int((frame["solver_failures"] > 0).sum()) if "solver_failures" in frame else counts["failed"]
    return {"n": len(frame), "status_counts": counts, "solver_failure_points": failures}


def export_block(block: str, work_root: Path = WORK, generated: Path | None = None) -> Path:
    """Write per-point labels mechanically; do not expose them in monitoring output."""
    generated = ROOT / "data" / "generated" if generated is None else generated
    inputs = pd.read_csv(work_root / "inputs" / f"{block}_design.csv", float_precision="round_trip")
    results = pd.read_csv(work_root / block / "results.csv", float_precision="round_trip")
    expected = set(inputs["id"])
    if len(results) != SIZES[block] or results["id"].duplicated().any() or set(results["id"]) != expected:
        raise ValueError(f"{block}: incomplete or non-unique result IDs")
    fields = ["id", "status", "ppm", "recovery", "productivity", "cycles", "start", "wall_s",
              "solver_failures", "attempts"]
    fields += [name for name in ("original_start", "resume_start", "cycles_total") if name in results]
    frame = inputs.merge(results[fields], on="id", how="left", validate="one_to_one", sort=False)
    frame.rename(columns={"ppm": "impurity_ppm", "start": "warm_start_source", "wall_s": "wall_time_s"},
                 inplace=True)
    frame.rename(columns={"original_start": "initial_warm_start_source"}, inplace=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["z"] = np.clip(np.log10(frame["impurity_ppm"]), -1, 3)
    frame.loc[frame["status"] == "censored", "z"] = 3.0
    frame.loc[frame["status"] == "failed", ["impurity_ppm", "z", "recovery", "productivity"]] = np.nan
    frame["objectives_at_css"] = frame["status"] == "css"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"{block}.csv"
    tmp = path.with_suffix(".csv.part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def commit_and_push(block: str, path: Path):
    relative = str(path.relative_to(ROOT))
    expected = {f"data/generated/{name}.csv": f"Step 6a: {name} ({size} points)"
                for name, size in SIZES.items()}
    if relative != f"data/generated/{block}.csv" or relative not in expected:
        raise RuntimeError("Publication is restricted to registered block CSVs")

    def git_text(arguments, cwd=ROOT):
        return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()

    def verify_commit(commit: str, cwd: Path, required_path: str | None = None):
        names = git_text(["diff-tree", "--no-commit-id", "--name-only", "--root", "-r", commit], cwd).splitlines()
        if len(names) != 1 or names[0] not in expected or (required_path and names[0] != required_path):
            raise RuntimeError("Refusing publication of a commit containing anything beyond one registered CSV")
        if git_text(["show", "-s", "--format=%s", commit], cwd) != expected[names[0]]:
            raise RuntimeError("CSV commit message differs from the registered block message")

    subprocess.run(["git", "add", "--", relative], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=ROOT)
    if changed.returncode == 1:
        staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
        if staged.splitlines() != [relative]:
            raise RuntimeError("Unrelated staged files: refusing to include them in a data commit")
        subprocess.run(["git", "commit", "-m", f"Step 6a: {block} ({SIZES[block]} points)"], cwd=ROOT, check=True)
    elif changed.returncode != 0:
        raise RuntimeError("Could not inspect staged CSV")
    local_commit = git_text(["log", "-1", "--format=%H", "--", relative])
    verify_commit(local_commit, ROOT, relative)
    local_blob = git_text(["rev-parse", f"{local_commit}:{relative}"])

    # Publish CSV commits from a separate history. The simulation checkout may
    # contain local infrastructure changes which must never enter this push.
    subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
    remote = git_text(["rev-parse", "refs/remotes/origin/main"])
    publication_file = WORK / "csv-publish.json"
    publishing = WORK / "csv-publish"
    if publication_file.exists():
        publication = json.loads(publication_file.read_text())
    else:
        publication = {"base": remote, "remote": remote}
        atomic_json(publication_file, publication)
    if not publishing.exists():
        if remote != publication["remote"]:
            raise RuntimeError("Remote main changed unexpectedly before CSV worktree creation")
        subprocess.run(["git", "worktree", "add", "--detach", str(publishing), publication["base"]],
                       cwd=ROOT, check=True)
    head = git_text(["rev-parse", "HEAD"], publishing)
    if remote not in (publication["remote"], head):
        raise RuntimeError("Remote main changed unexpectedly; CSV publication stopped")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", publication["base"], "HEAD"],
                             cwd=publishing)
    if ancestor.returncode != 0:
        raise RuntimeError("CSV publishing worktree no longer descends from its recorded remote base")
    if git_text(["status", "--porcelain"], publishing):
        raise RuntimeError("CSV publishing worktree has unfinished changes")

    def verify_outgoing():
        commits = git_text(["rev-list", "--reverse", f"{publication['base']}..HEAD"], publishing).splitlines()
        for commit in commits:
            verify_commit(commit, publishing)

    verify_outgoing()
    exists = subprocess.run(["git", "cat-file", "-e", f"HEAD:{relative}"], cwd=publishing,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    same_blob = exists.returncode == 0 and git_text(["rev-parse", f"HEAD:{relative}"], publishing) == local_blob
    if not same_blob:
        subprocess.run(["git", "cherry-pick", "--ff", "--no-edit", local_commit], cwd=publishing, check=True)
    published_commit = git_text(["log", "-1", "--format=%H", "--", relative], publishing)
    verify_commit(published_commit, publishing, relative)
    verify_outgoing()
    subprocess.run(["git", "-c", "push.followTags=false", "push", "origin", "HEAD:main"],
                   cwd=publishing, check=True)
    publication["remote"] = git_text(["rev-parse", "HEAD"], publishing)
    atomic_json(publication_file, publication)
    return {"local": local_commit, "published": published_commit}


def save_report(state: dict, work_root: Path = WORK):
    lines = ["# Step 6a — generation status and timing", "", "No response distributions or analyses computed."]
    if "pilot" in state:
        p = state["pilot"]
        lines += ["", f"Pilot: {p['n']}/32; {p['status_counts']}; solver failure points={p['solver_failure_points']}."]
        if "median_s" in p:
            lines += [f"Per-point wall seconds: median={p['median_s']:.1f}, p90={p['p90_s']:.1f}, max={p['max_s']:.1f}.",
                      f"Censored fraction={p['status_counts']['censored'] / p['n']:.3%}; elapsed={p['wall_s']:.1f} s.",
                      f"Remaining estimates (hours, {state['workers']} workers): {state.get('estimate_remaining_h')}."]
    lines += ["", "| block | n | css | censored | cap | failed | solver failure points | wall hours |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for block in SIZES:
        info = metadata_counts(read_metadata(block, work_root))
        c = info["status_counts"]
        wall = sum(p["wall_s"] for name, p in state.get("phases", {}).items()
                   if name == block or (block == "D_train" and name == "pilot"))
        lines.append(f"| {block} | {info['n']} | {c['css']} | {c['censored']} | {c['cap']} | {c['failed']} | "
                     f"{info['solver_failure_points']} | {wall / 3600:.3f} |")
    lines += ["", f"Workflow state: {state.get('status', 'prepared')}",
              f"Generation wall hours: {sum(p['wall_s'] for p in state.get('phases', {}).values()) / 3600:.3f}.",
              "Deviations: see docs/preregistration.md, Deviations (2026-10-12)."]
    (work_root / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_phase(phase: str, block: str, workers: int, state: dict, work_root: Path = WORK):
    inputs = work_root / "inputs" / ("D_train_pilot.csv" if phase == "pilot" else f"{block}.csv")
    out = work_root / block
    out.mkdir(parents=True, exist_ok=True)
    command = ["caffeinate", "-i", sys.executable, str(ROOT / "scripts" / "run_batch.py"), str(inputs), str(out),
               "--workers", str(workers), "--blind", "--retry-failures", "--failure-limit", "0.02",
               "--cap-limit", "0.05", "--block-size", str(32 if phase == "pilot" else SIZES[block])]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    info = state.setdefault("phases", {}).setdefault(phase, {"wall_s": 0.0})
    prior_wall_s = info["wall_s"]
    info.update(started_at=datetime.now(ZoneInfo("America/New_York")).isoformat(), running=True)
    state["status"] = f"running {phase}"
    atomic_json(work_root / "workflow.json", state)
    print(f"Starting {phase} with {workers} workers; monitored runner log: {out / (phase + '.log')}", flush=True)
    log_path = out / (phase + ".log")
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        previous = None
        while process.poll() is None:
            current = metadata_counts(read_metadata(block, work_root))
            if current != previous:
                print(f"{phase}: {current}", flush=True)
                previous = current
            info["wall_s"] = prior_wall_s + time.monotonic() - started
            atomic_json(work_root / "workflow.json", state)
            time.sleep(10)
        returncode = process.wait()
    info["wall_s"] = prior_wall_s + time.monotonic() - started
    info.update(running=False, returncode=returncode)
    atomic_json(work_root / "workflow.json", state)
    return returncode


def generate(args):
    """Run under an exclusive workspace lock, held by main."""
    path = WORK / "workflow.json"
    state = json.loads(path.read_text()) if path.exists() else {"workers": args.workers, "phases": {}}
    if state["workers"] != args.workers:
        raise RuntimeError("Worker change requires a recorded deviation before resuming")
    if state.get("status", "").startswith("paused"):
        raise RuntimeError("Monitoring pause requires project-lead instruction before continuing")
    for phase, block in (("pilot", "D_train"), *((name, name) for name in SIZES)):
        if state["phases"].get(phase, {}).get("complete"):
            continue
        code = run_phase(phase, block, args.workers, state)
        meta = read_metadata(block)
        counts = metadata_counts(meta)
        if phase == "pilot":
            counts["wall_s"] = state["phases"][phase]["wall_s"]
            state["pilot"] = counts
            if len(meta) == 32:
                seconds = meta["wall_s"].to_numpy()
                counts.update(median_s=float(np.median(seconds)), p90_s=float(np.quantile(seconds, .9)),
                              max_s=float(np.max(seconds)))
                estimates = {name: n * float(np.mean(seconds)) / args.workers / 3600
                             for name, n in {"D_train_remaining": 480, "D_cal": 256, "D_test": 256,
                                             "D_shift": 128}.items()}
                estimates["total"] = sum(estimates.values())
                state["estimate_remaining_h"] = estimates
                state["estimated_total_wall_h"] = counts["wall_s"] / 3600 + estimates["total"]
                print(f"Pilot: {counts}; remaining estimate hours={estimates}", flush=True)
        if code or counts["n"] != (32 if phase == "pilot" else SIZES[block]):
            state["status"] = f"paused {phase}: runner exit {code}; {counts}"
            atomic_json(path, state)
            save_report(state)
            print(state["status"], flush=True)
            return code or 1
        if phase != "pilot":
            csv = export_block(block)
            if args.defer_publication:
                state["phases"][phase]["generation_complete"] = True
                state["status"] = f"awaiting CSV publication {block}"
                atomic_json(path, state)
                save_report(state)
                print(f"Ready to commit and push completed block CSV: {csv}", flush=True)
                return 0
            try:
                commits = commit_and_push(block, csv)
            except Exception:
                state["status"] = f"paused {phase}: CSV commit/push failed"
                atomic_json(path, state)
                save_report(state)
                raise
            state.setdefault("commits", {})[block] = commits
        state["phases"][phase]["complete"] = True
        atomic_json(path, state)
        save_report(state)
    state["status"] = "complete — Step 6a only"
    atomic_json(path, state)
    save_report(state)
    print((WORK / "REPORT.md").read_text(), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, choices=(6, 8), default=8)
    parser.add_argument("--defer-publication", action="store_true",
                        help="finish/export one block, then stop for its CSV-only commit and push")
    args = parser.parse_args()
    prepare()
    if args.prepare:
        return 0
    with (WORK / "workflow.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Step 6a workflow is already running") from None
        return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
