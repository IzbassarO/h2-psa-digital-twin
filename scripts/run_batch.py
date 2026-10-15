"""Parallel CSS runner with checkpoint / resume (step 5).

    caffeinate -i python scripts/run_batch.py POINTS.csv OUT_DIR [--workers 8]

POINTS.csv: column `id` and any design / disturbance variable of configs/design_space.yaml and
configs/disturbances.yaml (plus `anchor`); missing ones take their defaults (src/h2psa/design.py).

* A finished point is written atomically to OUT_DIR/cases/<id>.pkl and is never recomputed.
* A point in progress saves its state every 10 cycles to OUT_DIR/ckpt/<id>.npz; after a kill (sleep, reboot,
  Ctrl-C) the same command resumes it from there. The resumed run is a warm start: CSS is still declared only
  after >= 10 plain cycles, and the cycle cap counts the cycles before the kill.
* Warm start of a new point: CSS state of the nearest finished point of this batch (design.scaled distance).
* Clearly infeasible points stop early (css.censoring in design_space.yaml): status "censored", ppm = X.
* OUT_DIR/results.csv: one row per finished point (rewritten after every point).
* --blind suppresses all performance labels in logs; --retry-failures retries a
  primary solver failure exactly once from a fresh He-filled column, using the
  frozen numerical settings. Both attempts remain in checkpoint provenance.
* --point-timeout S: an attempt whose active wall time (summed over resumes) exceeds S seconds is a solver
  failure (TimeoutError), handled like any other: cold retry with --retry-failures, otherwise "failed"
  (Step 6c rule, docs/preregistration.md, Deviations 2026-10-14).
* --failure-limit / --cap-limit pause above fractions of --block-size (or the
  requested size), drain active workers, and exit 3. Submission is bounded by
  --workers so a gate cannot leave the rest of a block queued.
Run long batches under `caffeinate -i` (macOS: no idle sleep while the command runs).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa import design  # noqa: E402
from h2psa.cycle import CycleRunner  # noqa: E402
from h2psa.isotherms import load_yaml  # noqa: E402

CSS = load_yaml(design.CONFIGS / "design_space.yaml")["css"]
CENSOR = (CSS["censoring"]["min_cycles"], float(CSS["censoring"]["limit_ppm"]), CSS["censoring"]["window"])


def _atomic(path: Path, write):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def nearest_state(point, states: Path):
    best = None
    x = design.scaled(point)
    for fn in states.glob("*.npz"):
        with np.load(fn) as d:
            dist = float(np.linalg.norm(d["design"] - x))
            if best is None or dist < best[0]:
                best = (dist, fn.stem, d["y"].copy())
    return best


def point_of(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "id" and not (isinstance(v, float) and np.isnan(v))}


def _same_point(left: dict, right: dict) -> bool:
    """Allow only CSV float round-trip noise when checking a saved input."""
    if left.keys() != right.keys():
        return False
    for key, val in left.items():
        other = right[key]
        if isinstance(val, (int, float)) and isinstance(other, (int, float)):
            if not np.isclose(val, other, rtol=1e-14, atol=0, equal_nan=True):
                return False
        elif val != other:
            return False
    return True


@contextmanager
def time_limit(seconds: float | None):
    """Raise TimeoutError in this (worker main) thread after `seconds` of wall time; None = no limit."""
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise TimeoutError("attempt wall-time limit already used")

    def expire(signum, frame):
        raise TimeoutError("attempt exceeded its wall-time limit")
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_point(pid: str, point: dict, out: Path, max_cycles: int = CSS["max_cycles"], *,
              blind: bool = False, retry_failures: bool = False, timeout_s: float | None = None) -> str:
    """Run/resume a point; optionally retry a primary failure once from a fresh He column.

    The cold retry uses the same frozen column, integrator and CSS tolerances. The
    checkpoint carries attempt provenance, accumulated active wall time, and the
    censor history; performance labels remain private when ``blind`` is true.
    """
    out = Path(out)
    done = out / "cases" / f"{pid}.pkl"
    if done.exists():
        with open(done, "rb") as fh:
            saved = pickle.load(fh)
        if not _same_point(saved["point"], point):
            raise ValueError(f"[{pid}] requested inputs differ from finished case")
        return pid

    def log(s):
        if blind:
            # CycleRunner emits label-bearing progress messages; keep only the
            # cycle number and elapsed time (including resumed cycles).
            match = re.match(r"cycle (\d+):.*\((\d+) s\)$", s)
            if match:
                s = f"cycle {n0 + int(match[1])} ({match[2]} s this segment)"
            else:
                return
        print(f"[{pid}] {s}", flush=True)

    def announce(s):
        print(f"[{pid}] {s}", flush=True)

    ck = out / "ckpt" / f"{pid}.npz"
    n0, start, original_start, y0 = 0, None, None, None
    elapsed_before, attempt, errors, cycles_prior_attempts = 0.0, 1, [], 0
    attempt_wall0 = 0.0                                  # active wall time when the current attempt began
    censor_ppm = []
    attempt_start = "cold"
    resumed = ck.exists()
    if ck.exists():
        with np.load(ck) as d:
            y0, n0, start = d["y"].copy(), int(d["n"]), "checkpoint"
            if "meta" in d:
                meta = json.loads(str(d["meta"].item()))
                if not _same_point(meta["point"], point):
                    raise ValueError(f"[{pid}] requested inputs differ from checkpoint")
                elapsed_before = float(meta.get("wall_s", 0.0))
                attempt = int(meta.get("attempt", 1))
                original_start = meta.get("original_start")
                attempt_start = meta.get("attempt_start", "cold")
                errors = meta.get("errors", [])
                cycles_prior_attempts = int(meta.get("cycles_prior_attempts", 0))
                attempt_wall0 = float(meta.get("attempt_wall0", 0.0))
                censor_ppm = d["censor_ppm"].tolist() if "censor_ppm" in d else []
            else:
                original_start = "unknown (legacy checkpoint)"
                announce("legacy checkpoint: prior wall time and censor history unavailable")
    else:
        seed = nearest_state(point, out / "css_states")
        y0, start = (None, None) if seed is None else (seed[2], seed[1])
        original_start = start
        attempt_start = start or "cold"

    def save_checkpoint(y, n, elapsed):
        meta = dict(point=point, original_start=original_start, attempt_start=attempt_start,
                    attempt=attempt, errors=errors, wall_s=elapsed,
                    cycles_prior_attempts=cycles_prior_attempts, attempt_wall0=attempt_wall0)
        _atomic(ck, lambda fh: np.savez(fh, y=y, n=n, censor_ppm=np.asarray(censor_ppm),
                                       meta=json.dumps(meta, sort_keys=True)))

    while True:
        t0 = time.perf_counter()
        last_n = n0
        history_before = len(censor_ppm)
        cyc, m, w = design.build(point)
        run = CycleRunner(m, cyc, w_ads=w)
        if n0 > max_cycles:
            raise ValueError(f"[{pid}] checkpoint exceeds cycle cap")
        # n=0 on a retry remains a true fresh-column run; neighbour/checkpoint
        # state must never become its initial condition accidentally.
        if attempt == 2 and n0 == 0:
            y0 = None
        announce(f"start from {start or 'He-filled column'}" + (f" (cycle {n0})" if n0 else "") +
                 (f", attempt {attempt}" if attempt > 1 else ""))
        save_checkpoint(m.y0 if y0 is None else y0, n0, elapsed_before)

        def checkpoint(n, y):
            nonlocal last_n
            last_n = n0 + n
            if n % 10 == 0:
                save_checkpoint(y, last_n, elapsed_before + time.perf_counter() - t0)

        try:
            with time_limit(None if timeout_s is None else timeout_s - (elapsed_before - attempt_wall0)):
                res = run.run_to_css(y0=y0, max_cycles=max_cycles - n0, tol=CSS["tol"], tol_ppm=CSS["tol_ppm"],
                                     accel=True, censor=CENSOR, censor_history=censor_ppm,
                                     checkpoint=checkpoint, log=log)
        except Exception as exc:
            elapsed_before += time.perf_counter() - t0
            last_n = max(last_n, n0 + len(censor_ppm) - history_before)
            errors.append(dict(attempt=attempt, error_type=type(exc).__name__, error=str(exc), cycles=last_n))
            announce(f"solver failure on attempt {attempt}: {type(exc).__name__}" if blind else
                     f"solver failure on attempt {attempt}: {exc!r}")
            if retry_failures and attempt == 1:
                cycles_prior_attempts += last_n
                attempt, attempt_start, n0, start, y0, censor_ppm = 2, "cold", 0, None, None, []
                attempt_wall0 = elapsed_before
                # Persist the primary failure before a cold retry so monitoring
                # sees it promptly and an interruption cannot repeat attempt 1.
                save_checkpoint(m.y0, 0, elapsed_before)
                resumed = False
                continue
            res = dict(css=False, censored=None, cycles=max(last_n - n0, 0), runtime_s=time.perf_counter() - t0,
                       perf=dict(ppm=np.nan, recovery_pct=np.nan, productivity=np.nan, purity_pct=np.nan))
            status = "failed"
        else:
            elapsed_before += time.perf_counter() - t0
            status = "css" if res["css"] else "censored" if res["censored"] is not None else "cap"
        res.update(id=pid, point=point, status=status, start=start, original_start=original_start,
                   attempt_start=attempt_start, cycles_before_resume=n0, w_ads=w, species=m.sp,
                   wall_s=elapsed_before, attempts=attempt, solver_failures=int(bool(errors)),
                   attempt_errors=errors, resumed=resumed,
                   cycles_total=cycles_prior_attempts + n0 + res["cycles"])
        _atomic(done, lambda fh: pickle.dump(res, fh))
        if res["css"]:
            _atomic(out / "css_states" / f"{pid}.npz", lambda fh: np.savez(fh, y=res["y_end"], design=design.scaled(point)))
        ck.unlink(missing_ok=True)
        msg = f"done: {status} after {n0 + res['cycles']} cycles, {elapsed_before / 60:.1f} min wall"
        if not blind and status != "failed":
            msg += f", ppm {res['perf']['ppm']:.4g}"
        announce(msg)
        break
    return pid


def summary(out: Path) -> pd.DataFrame:
    rows = []
    for fn in sorted((out / "cases").glob("*.pkl")):
        with open(fn, "rb") as fh:
            r = pickle.load(fh)
        p = r.get("perf", {})
        ppm = r["censored"] if r["status"] == "censored" else p["ppm"]
        wall_s = r.get("wall_s", r["runtime_s"])
        rows.append(dict(id=r["id"], **r["point"], status=r["status"], ppm=ppm, recovery=p["recovery_pct"],
                         productivity=p["productivity"], purity=p["purity_pct"],
                         cycles=r["cycles_before_resume"] + r["cycles"],
                         start=r.get("attempt_start", r["start"]) or "cold",
                         resume_start=r["start"] or "cold",
                         original_start=r.get("original_start", r["start"]) or "cold",
                         attempts=r.get("attempts", 1), solver_failures=r.get("solver_failures", 0),
                         cycles_total=r.get("cycles_total", r["cycles_before_resume"] + r["cycles"]),
                         wall_s=wall_s, wall_min=wall_s / 60))
    columns = ["id", "status", "ppm", "recovery", "productivity", "purity", "cycles", "start",
               "resume_start", "original_start", "attempts", "solver_failures", "cycles_total", "wall_s", "wall_min"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    _atomic(out / "results.csv", lambda fh: fh.write(df.to_csv(index=False).encode()))
    return df


def block_counts(out: Path) -> tuple[int, int]:
    """Primary failures include successful retries and active retry checkpoints."""
    failures, caps, finished = 0, 0, set()
    report = out / "results.csv"
    if report.exists():
        # Read only permitted monitoring counts, never response labels/traces.
        metadata = pd.read_csv(report, dtype={"id": str}, usecols=["id", "status", "solver_failures"])
        finished = set(metadata["id"])
        failures = int(metadata["solver_failures"].fillna(0).sum())
        caps = int((metadata["status"] == "cap").sum())
    for fn in (out / "ckpt").glob("*.npz"):
        if fn.stem in finished:
            continue
        try:
            with np.load(fn) as d:
                if "meta" in d:
                    failures += int(bool(json.loads(str(d["meta"].item())).get("errors")))
        except FileNotFoundError:
            # A worker atomically finished since the glob. Its completed future
            # triggers summary before the next gate check and any submission.
            continue
    return failures, caps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blind", action="store_true", help="log status/timing only; no performance labels")
    ap.add_argument("--retry-failures", action="store_true", help="retry each primary solver failure once from He")
    ap.add_argument("--failure-limit", type=float, help="pause above this primary-failure fraction")
    ap.add_argument("--cap-limit", type=float, help="pause above this cap-hit fraction")
    ap.add_argument("--point-timeout", type=float, help="per-attempt active wall-time limit [s]")
    ap.add_argument("--block-size", type=int, help="threshold denominator (default: number of requested points)")
    a = ap.parse_args(argv)
    if a.workers < 1 or (a.block_size is not None and a.block_size < 1):
        ap.error("workers and block-size must be positive")
    if any(v is not None and not 0 <= v <= 1 for v in (a.failure_limit, a.cap_limit)):
        ap.error("monitoring limits must be fractions in [0, 1]")
    out = Path(a.out)
    for d in ("cases", "ckpt", "css_states"):
        (out / d).mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(a.points, dtype={"id": str}, float_precision="round_trip")
    if "id" not in frame or frame["id"].duplicated().any() or frame["id"].isna().any():
        ap.error("points must have unique nonempty ids")
    pts = frame.to_dict("records")
    if any(Path(r["id"]).name != r["id"] or r["id"] in (".", "..") for r in pts):
        ap.error("point ids must be filenames without path separators")
    for r in pts:
        case = out / "cases" / f"{r['id']}.pkl"
        if case.exists():
            with open(case, "rb") as fh:
                saved = pickle.load(fh)
            if not _same_point(saved["point"], point_of(r)):
                ap.error(f"[{r['id']}] requested inputs differ from finished case")
    # Growing a pilot prefix into its full block is permitted, while existing
    # rows retain their input values for reproducible checkpoint/resume.
    requested = out / "points.csv"
    if requested.exists():
        old = pd.read_csv(requested, dtype={"id": str}, float_precision="round_trip").to_dict("records")
        current = {r["id"]: r for r in pts}
        for r in old:
            if r["id"] in current and not _same_point(point_of(r), point_of(current[r["id"]])):
                ap.error(f"[{r['id']}] requested inputs differ from previous manifest")
        all_requested = {r["id"]: r for r in old}
        all_requested.update(current)
        snapshot = pd.DataFrame(all_requested.values())
    else:
        snapshot = frame
    _atomic(requested, lambda fh: fh.write(snapshot.to_csv(index=False).encode()))
    todo = [r for r in pts if not (out / "cases" / f"{r['id']}.pkl").exists()]
    n_ck = sum((out / "ckpt" / f"{r['id']}.npz").exists() for r in todo)
    print(f"{len(pts)} points, {len(pts) - len(todo)} finished, {len(todo)} to run ({n_ck} from checkpoints)", flush=True)
    # Refresh a crash-interrupted CSV from immutable final cases before checking
    # a resumed block's monitoring limits.
    summary(out)
    denominator = a.block_size or len(pts)
    gated = False

    def gate():
        nonlocal gated
        if a.failure_limit is None and a.cap_limit is None:
            return gated
        failures, caps = block_counts(out)
        exceeded = (a.failure_limit is not None and failures > a.failure_limit * denominator) or \
                   (a.cap_limit is not None and caps > a.cap_limit * denominator)
        if exceeded and not gated:
            gated = True
            print(f"MONITOR PAUSE: {failures} primary solver failures, {caps} cap hits / {denominator} points; "
                  "stopping new submissions and draining active points", flush=True)
        return gated

    with ProcessPoolExecutor(a.workers) as ex:
        pending = iter(todo)
        futs = {}

        def submit_one():
            r = next(pending, None)
            if r is not None:
                futs[ex.submit(run_point, r["id"], point_of(r), out,
                               blind=a.blind, retry_failures=a.retry_failures,
                               timeout_s=a.point_timeout)] = r["id"]
                return True
            return False

        if not gate():
            for _ in range(a.workers):
                if not submit_one():
                    break
        while futs:
            done, _ = wait(futs, timeout=1.0, return_when=FIRST_COMPLETED)
            # A primary failure can occur while its cold retry is still active.
            # Check before filling a freed worker so no extra point slips past.
            gate()
            for f in done:
                pid = futs.pop(f)
                error = f.exception()
                if error is not None:
                    print(f"[{pid}] WORKER ERROR: {type(error).__name__}" if a.blind else
                          f"[{pid}] WORKER ERROR: {error!r}", flush=True)
                    # Infrastructure errors are distinct from persisted solver
                    # failures; leave the checkpoint resumable and halt intake.
                    gated = True
                summary(out)
            if not gate():
                for _ in done:
                    submit_one()
    summary(out)
    return 3 if gated else 0


if __name__ == "__main__":
    sys.exit(main())
