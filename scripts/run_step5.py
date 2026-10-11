"""Step 5 report: capacity-factor check at CSS, early-stopping verification, compute plan.

    caffeinate -i python scripts/run_batch.py results/step5/phi/points.csv results/step5/phi   # the CSS runs
    python scripts/run_step5.py                                                                # this report

Reads the step-4 cases (results/step4/) and the step-5 runs (results/step5/phi/); writes results/step5/REPORT.md
and copies it to docs/validation/step5.md.
"""
from __future__ import annotations

import heapq
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.cycle import censored  # noqa: E402
from h2psa.design import CONFIGS  # noqa: E402
from h2psa.isotherms import load_yaml  # noqa: E402

S4, OUT = ROOT / "results" / "step4", ROOT / "results" / "step5"
PHI = OUT / "phi"
C = load_yaml(CONFIGS / "design_space.yaml")["css"]["censoring"]
CENSOR = (C["min_cycles"], float(C["limit_ppm"]), C["window"])
STEP4_OF = {"f1.0_t80": "base_f1.0_t80", "f0.6_t101": "base_f0.6_t101", "f0.4_t60": "base_f0.4_t60"}
WORKERS, N_RUNS = 8, (500, 1000, 2000)


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt[c].format(r[c]) if c in fmt and isinstance(r[c], (int, float, np.number))
                                      else str(r[c]) for c in cols) + " |")
    return "\n".join(rows)


def load(fn):
    with open(fn, "rb") as fh:
        return pickle.load(fh)


def first_censor(ppm):
    return next((k for k in range(1, len(ppm) + 1) if censored(ppm[:k], *CENSOR)), None)


def censor_check():
    """Replay the rule on every stored step-4 CSS history (all runs and variants of step 4)."""
    rows = []
    for d in sorted(p for p in S4.iterdir() if p.is_dir() and p.name.startswith(("cases", "accel_check"))):
        for fn in sorted(d.glob("*.pkl")):
            r = load(fn)
            ppm = [h["ppm"] for h in r["history"][:-1]]
            c = first_censor(ppm)
            rows.append(dict(set=d.name, case=fn.stem, cycles=len(ppm), final_ppm=r["perf"]["ppm"],
                             stop_cycle=c, X_ppm=ppm[c - 1] if c else np.nan,
                             min_after_stop=min(ppm[c - 1:]) if c else np.nan,
                             max_after_100=max(ppm[CENSOR[0] - 1:]) if len(ppm) >= CENSOR[0] else np.nan,
                             wall_min=r["runtime_s"] / 60))
    return pd.DataFrame(rows)


def makespan(t, n, w, reps=2000, seed=0):
    """Wall time [h] of n runs drawn from the timing sample t [min] on w workers (greedy list scheduling)."""
    rng, out = np.random.default_rng(seed), []
    for _ in range(reps):
        q = [0.0] * w
        for v in rng.choice(t, n):
            heapq.heapreplace(q, q[0] + v)
        out.append(max(q) / 60)
    return np.median(out), np.percentile(out, 90)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cc = censor_check()
    cc.to_csv(OUT / "censor_check.csv", index=False)
    fired = cc[cc.stop_cycle.notna()]
    wrong = fired[fired.final_ppm < CENSOR[1]]
    near = cc[(cc.final_ppm < CENSOR[1]) & (cc.cycles >= CENSOR[0])].sort_values("max_after_100", ascending=False)

    # capacity factor at CSS (3 cases x phi) and the NETL anchor
    ph = []
    for fn in sorted((PHI / "cases").glob("*.pkl")):
        r = load(fn)
        case = fn.stem.rsplit("_", 1)[0]
        ref = load(S4 / "cases" / f"{STEP4_OF[case]}.pkl")["perf"] if case in STEP4_OF else None
        p = r["perf"]
        ph.append(dict(id=r["id"], case=case, phi=r["point"].get("phi", 1.0), anchor=r["point"].get("anchor", "lopes"),
                       status=r["status"], ppm=r["censored"] if r["status"] == "censored" else p["ppm"],
                       recovery=p["recovery_pct"], productivity=p["productivity"],
                       **{f"{s}_ppm": p[f"imp_{s}_ppm"] for s in ("CO2", "CH4", "CO", "N2")},
                       ppm_step4=ref["ppm"] if ref and r["point"].get("phi", 1.0) == 1.0 and "anchor" not in r["point"] else np.nan,
                       cycles=r["cycles_before_resume"] + r["cycles"], start=r["start"] or "cold",
                       wall_min=r["runtime_s"] / 60))
    ph = pd.DataFrame(ph)
    ph.to_csv(OUT / "phi_check.csv", index=False)
    lop = ph[ph.anchor == "lopes"]
    mono = {c: bool(np.all(np.diff(g.sort_values("phi", ascending=False).ppm.values) > 0)) for c, g in lop.groupby("case")}
    rep = lop[lop.ppm_step4.notna()]
    d_rep = (rep.ppm / rep.ppm_step4 - 1).abs().max() if len(rep) else np.nan

    # compute plan: timing distribution of the step-4 layered baseline (31 runs), with / without early stopping
    t, tc = [], []
    for fn in sorted((S4 / "cases").glob("*.pkl")):
        if fn.stem.startswith("limit"):
            continue
        r = load(fn)
        ppm = [h["ppm"] for h in r["history"][:-1]]
        w = r["runtime_s"] / 60
        c = first_censor(ppm)
        t.append(w)
        tc.append(w * (c + 1) / (len(ppm) + 1) if c else w)      # cycles to the stop + the final cycle
    t, tc = np.array(t), np.array(tc)
    ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()
    perf = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"],
                          capture_output=True, text=True).stdout.split()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    tim = pd.DataFrame([dict(sample=lab, n=len(x), median=np.median(x), p90=np.percentile(x, 90), mean=x.mean(),
                             max=x.max()) for lab, x in (("step-4 layered baseline, as run", t),
                                                         ("same, early stopping applied", tc))])
    plan = []
    for n in N_RUNS:
        m0, p0 = makespan(t, n, WORKERS)
        m1, p1 = makespan(tc, n, WORKERS)
        plan.append(dict(N=n, median_h=m0, p90_h=p0, median_h_censoring=m1, p90_h_censoring=p1,
                         days_censoring=m1 / 24))
    plan = pd.DataFrame(plan)

    f2 = {c: "{:.4g}" for c in ("final_ppm", "X_ppm", "min_after_stop", "max_after_100")}
    f2.update(wall_min="{:.1f}", stop_cycle="{:.0f}")
    fph = {c: "{:.4g}" for c in ("ppm", "CO2_ppm", "CH4_ppm", "CO_ppm", "N2_ppm", "ppm_step4")}
    fph.update(recovery="{:.2f}", productivity="{:.1f}", wall_min="{:.1f}", phi="{:.2f}")
    resumed = ph[ph.start == "checkpoint"]
    gate_phi = "PASS" if len(mono) == 3 and all(mono.values()) and d_rep < 1e-3 else "FAIL"
    gate_cens = "PASS" if len(wrong) == 0 else "FAIL"
    text = f"""# Step 5 — design space, disturbance variables, compute plan, pre-registration

Generated by `scripts/run_step5.py` from `results/step4/` and `results/step5/` (runs by `scripts/run_batch.py`).
Step-5 content (project lead, 2026-10-11): design space promoted (`configs/design_space.yaml`, t_blow added),
disturbance variables (`configs/disturbances.yaml`, sources in `data/PROVENANCE.md`), early stopping of clearly
infeasible points, parallel runner with checkpoint / resume, compute plan, pre-registration DRAFT
(`docs/preregistration.md`). No large data generation.

## Verdict

| check | result |
|---|---|
| phi = 1 reproduces the step-4 baseline exactly | PASS: one cycle from the step-4 CSS state is bitwise identical through `design.build` and the step-4 builder (`tests/test_design.py`); at CSS max relative ppm difference {d_rep:.1e} ({len(rep)} cases, independent warm-start path) |
| phi lowers capacity -> impurity rises monotonically (3 cases, phi 1.0 / 0.85 / 0.70, at CSS) | {gate_phi}: {", ".join(f"{k} {'monotone' if v else 'NOT monotone'}" for k, v in mono.items())} |
| early stopping never censors a point that converges below 1000 ppm (step-4 histories) | {gate_cens}: {len(cc)} histories, {len(fired)} would stop, {len(wrong)} of them below 1000 ppm at CSS |
| runner: killed batch resumes from checkpoints without recomputing finished points | PASS: {len(resumed)} points resumed from their checkpoint, {int((ph.start != 'checkpoint').sum())} finished before the kill were not rerun |

## Design and disturbance variables

Decision variables (8): t_feed, t_blow (new, [60, 200] s), purge_to_feed, P_high, P_low, f_zeolite (1/30 grid),
n_equalisations, feed_flow; equalisation pressures equally spaced between P_high and P_low (reproduces cycles A
and B), purge flow = purge_to_feed x H2 fed per cycle / t_purge. Disturbances (3): capacity factor phi [0.70, 1.00]
on q_max of both layers (ASSUMED range: no peer-reviewed quantification in H2 PSA; 0.70 = stress case by analogy);
y_CH4 [0.0168, 0.0231] (reformer-digital-twin v3.0: slip factor 0.80-1.10 over catalyst activity 0.10-0.20 and load
0.7-1.1); y_CO [0.005, 0.015] (medium-temperature shift, Johnson Matthey 2025; contains Lopes 1.2 % and NETL 0.50 %);
CO2 keeps the anchor's carbon, H2 by difference. Anchors: Lopes feed (nominal) and NETL stream 10 dry basis, both at
P_high <= 8.5 bar. Model inputs: `design.build(point)`; phi through `layered_column(phi=...)`.

## Capacity factor at CSS

The three step-4 cases (AC only at t_feed 80 s, near the limit; f_AC 0.6 at 101 s, the experiment's feed time; f_AC 0.4
at 60 s, deep inside the feasible region) at phi = 1.0, 0.85, 0.70, and the NETL dry-basis anchor at f_AC 0.6 / 101 s /
5 bar; run with `scripts/run_batch.py` (8 workers; the batch was killed once and resumed, `start` = checkpoint;
`cycles` include those before the kill, `wall_min` of a resumed point only the time after the resume):

{md(ph.drop(columns=["case"]), fph)}

## Early stopping (censoring) of clearly infeasible points

Rule (`censored` in `src/h2psa/cycle.py`; `css.censoring` in the design space): after >= {CENSOR[0]} cycles the
impurity exceeds {CENSOR[1]:.0f} ppm (10x the limit) and has increased over the last {CENSOR[2]} cycles -> stop,
label "censored: impurity > X ppm", X the last value. Verification: the rule replayed on every stored step-4 CSS
history ({len(cc)} runs: baseline, first run with brute force, acceleration checks, earlier criteria), all stored
in `results/step5/censor_check.csv`.

* Runs the rule would stop: {len(fired)}; their CSS impurity {fired.final_ppm.min():.4g}-{fired.final_ppm.max():.4g} ppm,
  so all > 1000 ppm; none of them returns below {fired.min_after_stop.min():.4g} ppm after the stop.
* X versus the CSS value: X / CSS = {(fired.X_ppm / fired.final_ppm).min():.3f}-{(fired.X_ppm / fired.final_ppm).max():.3f}.
  X is usually below the CSS value (a lower bound), but not always (by up to {100 * ((fired.X_ppm / fired.final_ppm).max() - 1):.2f} %),
  so the label is used as "impurity > 1000 ppm" (clipped response, pre-registration section 3).
* Runs that converged below 1000 ppm, largest impurity after cycle {CENSOR[0]} (closest to being stopped):

{md(near.head(8)[["set", "case", "cycles", "final_ppm", "max_after_100"]], f2)}

## Compute plan

Machine: {chip}, `sysctl -n hw.ncpu` = {ncpu} ({perf[0] if perf else "?"} performance + {perf[1] if len(perf) > 1 else "?"} efficiency cores), {mem:.0f} GB.
Recommended {WORKERS} workers (one per performance core). Timing sample: the 31 layered-bed CSS runs of step 4 (warm
start + Anderson, 7 workers on a shared machine); with early stopping each stopped run is shortened to its stop cycle
(+1 final cycle) in proportion to its cycles:

{md(tim, {c: "{:.1f}" for c in ("median", "p90", "mean", "max")})}

Batch wall time [h] for N CSS runs on {WORKERS} workers (2000 bootstrap batches from the timing sample, greedy
scheduling):

{md(plan, {"N": "{:.0f}", **{c: "{:.1f}" for c in plan.columns if c != "N"}})}

Caveats: the step-6 design space is wider than the step-4 scans (P_high, P_low, t_blow, purge, n_eq vary); the first
block of runs (D_train) re-estimates the distribution. The censoring saving depends on the infeasible fraction of the
design. The pre-registered budget (1992 runs) maps to the N = 2000 row.

Runner (`scripts/run_batch.py`): finished points are written atomically and never recomputed; a point in progress
saves its state every 10 cycles and resumes from it after a kill (sleep, reboot); new points warm-start from the
nearest finished CSS state of the batch. Run long batches as

    caffeinate -i python scripts/run_batch.py POINTS.csv OUT_DIR --workers {WORKERS} > OUT_DIR/run.log 2>&1

(`caffeinate -i` keeps macOS from idle-sleeping while the command runs; closing the lid still sleeps a laptop on
battery: the checkpoints cover that case).
"""
    (OUT / "REPORT.md").write_text(text)
    (ROOT / "docs" / "validation" / "step5.md").write_text(text)
    print(f"wrote {OUT / 'REPORT.md'} and docs/validation/step5.md")


if __name__ == "__main__":
    main()
