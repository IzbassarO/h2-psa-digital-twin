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
Run long batches under `caffeinate -i` (macOS: no idle sleep while the command runs).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        d = np.load(fn)
        dist = float(np.linalg.norm(d["design"] - x))
        if best is None or dist < best[0]:
            best = (dist, fn.stem, d["y"])
    return best


def point_of(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "id" and not (isinstance(v, float) and np.isnan(v))}


def run_point(pid: str, point: dict, out: Path, max_cycles: int = CSS["max_cycles"]) -> str:
    cyc, m, w = design.build(point)
    run = CycleRunner(m, cyc, w_ads=w)
    log = lambda s: print(f"[{pid}] {s}", flush=True)
    ck = out / "ckpt" / f"{pid}.npz"
    n0, start = 0, None
    if ck.exists():
        d = np.load(ck)
        y0, n0, start = d["y"], int(d["n"]), "checkpoint"
    else:
        seed = nearest_state(point, out / "css_states")
        y0, start = (None, None) if seed is None else (seed[2], seed[1])
    log(f"start from {start or 'He-filled column'}" + (f" (cycle {n0})" if n0 else ""))

    def checkpoint(n, y):
        if n % 10 == 0:
            _atomic(ck, lambda fh: np.savez(fh, y=y, n=n0 + n))

    res = run.run_to_css(y0=y0, max_cycles=max(max_cycles - n0, 1), tol=CSS["tol"], tol_ppm=CSS["tol_ppm"],
                         accel=True, censor=CENSOR, checkpoint=checkpoint, log=log)
    status = "css" if res["css"] else "censored" if res["censored"] is not None else "cap"
    res.update(id=pid, point=point, status=status, start=start, cycles_before_resume=n0, w_ads=w, species=m.sp)
    _atomic(out / "cases" / f"{pid}.pkl", lambda fh: pickle.dump(res, fh))
    if res["css"]:
        _atomic(out / "css_states" / f"{pid}.npz", lambda fh: np.savez(fh, y=res["y_end"], design=design.scaled(point)))
    ck.unlink(missing_ok=True)
    log(f"done: {status} after {n0 + res['cycles']} cycles, ppm {res['perf']['ppm']:.4g}, "
        f"{res['runtime_s'] / 60:.1f} min wall")
    return pid


def summary(out: Path) -> pd.DataFrame:
    rows = []
    for fn in sorted((out / "cases").glob("*.pkl")):
        with open(fn, "rb") as fh:
            r = pickle.load(fh)
        p = r["perf"]
        ppm = r["censored"] if r["status"] == "censored" else p["ppm"]
        rows.append(dict(id=r["id"], **r["point"], status=r["status"], ppm=ppm, recovery=p["recovery_pct"],
                         productivity=p["productivity"], purity=p["purity_pct"],
                         cycles=r["cycles_before_resume"] + r["cycles"], start=r["start"] or "cold",
                         wall_min=r["runtime_s"] / 60))
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    for d in ("cases", "ckpt", "css_states"):
        (out / d).mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(a.points, dtype={"id": str}).to_dict("records")
    todo = [r for r in pts if not (out / "cases" / f"{r['id']}.pkl").exists()]
    n_ck = sum((out / "ckpt" / f"{r['id']}.npz").exists() for r in todo)
    print(f"{len(pts)} points, {len(pts) - len(todo)} finished, {len(todo)} to run ({n_ck} from checkpoints)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_point, r["id"], point_of(r), out): r["id"] for r in todo}
        for f in as_completed(futs):
            if f.exception() is not None:
                print(f"[{futs[f]}] FAILED: {f.exception()!r}", flush=True)
            else:
                summary(out)
    summary(out)


if __name__ == "__main__":
    main()
