"""Step 7b analysis (docs/preregistration.md, Addendum A). DESCRIPTIVE: designed after the Step 6c results, so no
hypothesis is tested, no multiplicity correction is applied and nothing here is confirmatory.

    python scripts/analyse_step7b.py [--generated-dir data/generated] [--out results/step7b/analysis]

Inputs (--generated-dir): S_nom.csv (baseline labels), M_<scenario>.csv and M_<scenario>_cold.csv (Step 7b labels),
S_nom_memberships.csv; the frozen S_nom predictions (docs/validation/step6b/predictions/S_nom.csv); run status from
results/step7b/workflow.json when present. Arms: the 5 GP arms at n_train = 512.
Outputs (--out): violations.csv, paired.csv, coverage.csv, fronts.csv, recovery_loss.csv, spot_checks.csv,
statuses.csv and REPORT.md. With the default --out, REPORT.md is copied to docs/validation/step7b.md and the
tables to docs/validation/step7b/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from h2psa.design import SCENARIOS  # noqa: E402
from h2psa.step6_metrics import (_observed, coverage_summary, feasible_front, hypervolume_2d,  # noqa: E402
                                 join_predictions, violation_summary)
from explore_step7a import LEVELS, REFERENCE, recovery_at  # noqa: E402
from report_step6c import md  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "step7b" / "analysis"
PREDICTIONS = ROOT / "docs" / "validation" / "step6b" / "predictions" / "S_nom.csv"
SETS = ("baseline", *SCENARIOS)
ARM_FIELDS = ["arm", "variant", "alpha"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def frames(generated: Path) -> dict[str, pd.DataFrame]:
    """Per set (baseline and each scenario): one row per (id, arm) with labels and the frozen bound."""
    m = read(generated / "S_nom_memberships.csv")
    m = m.loc[(m["surrogate"] == "GP") & (m["n_train"] == 512)]
    ids = set(m["id"])
    predictions = read(PREDICTIONS)
    predictions = predictions.loc[predictions["id"].isin(ids)]
    out = {}
    for name in SETS:
        labels = read(generated / ("S_nom.csv" if name == "baseline" else f"M_{name}.csv"))
        labels = labels.loc[labels["id"].isin(ids)]
        if set(labels["id"]) != ids:
            raise ValueError(f"{name}: labels do not cover the 210 GP n_train = 512 points")
        out[name] = join_predictions(labels, predictions, m)
    return out


def paired(base: pd.DataFrame, scen: pd.DataFrame) -> dict:
    """Violation changes of the same points, baseline -> scenario; failed pairs excluded and counted."""
    pair = base[["id", "status", "z"]].merge(scen[["id", "status", "z"]], on="id", suffixes=("_base", "_scen"),
                                             validate="one_to_one")
    bad = pair["status_base"].eq("failed") | pair["status_scen"].eq("failed")
    ok = pair.loc[~bad]
    zb = ok["z_base"].where(ok["status_base"] != "censored", 3.0)
    zs = ok["z_scen"].where(ok["status_scen"] != "censored", 3.0)
    vb, vs = zb.gt(2.0), zs.gt(2.0)
    up, down = int((~vb & vs).sum()), int((vb & ~vs).sum())
    p = float(binomtest(up, up + down, .5).pvalue) if up + down else 1.0
    return dict(n=len(ok), baseline_k=int(vb.sum()), scenario_k=int(vs.sum()), feasible_to_violating=up,
                violating_to_feasible=down, mcnemar_exact_p_two_sided=p, failed_pairs=int(bad.sum()),
                median_dz=float((zs - zb).median()))


def front_rows(f: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, losses, fronts = [], [], {}
    for name, frame in f.items():
        for arm, g in frame.groupby("arm", sort=True):
            o = _observed(g)
            front = feasible_front(o)
            fronts[name, arm] = front
            rows.append(dict(set=name, arm=arm, n_observed=len(o),
                             n_feasible=int((o["status"].isin(("css", "cap")) & o["z"].le(2.0)).sum()),
                             n_front=len(front), hv=hypervolume_2d(front, REFERENCE),
                             max_productivity=front[:, 1].max() if len(front) else np.nan))
    for (name, arm), front in fronts.items():
        for level in LEVELS:
            r = recovery_at(front, level)
            losses.append(dict(set=name, arm=arm, productivity=level, recovery=r,
                               loss_vs_P=recovery_at(fronts[name, "GP_n512_P_a0.10"], level) - r,
                               change_vs_baseline=r - recovery_at(fronts["baseline", arm], level)))
    return pd.DataFrame(rows), pd.DataFrame(losses)


def analyse(generated: Path) -> dict[str, pd.DataFrame]:
    f = frames(generated)
    violations, coverage, pairs = [], [], []
    for name, frame in f.items():
        for arm, g in frame.groupby("arm", sort=True):
            info = dict(set=name, arm=arm, variant=g.iloc[0]["variant"], alpha=float(g.iloc[0]["alpha"]))
            violations.append({**info, **violation_summary(g)})
            coverage.append({**info, **coverage_summary(g)})
            if name != "baseline":
                pairs.append({**info, **paired(f["baseline"].loc[f["baseline"]["arm"] == arm], g)})
    fronts, losses = front_rows(f)
    spots = []
    for s in SCENARIOS:
        cold = read(generated / f"M_{s}_cold.csv")[["id", "status", "z"]]
        warm = read(generated / f"M_{s}.csv")[["id", "status", "z"]]
        pair = cold.merge(warm, on="id", suffixes=("_cold", "_warm"), validate="one_to_one")
        spots.append(pair.assign(scenario=s, dz=pair["z_cold"] - pair["z_warm"]))
    statuses = pd.DataFrame([dict(set=name, n=len(u), **{k: int(u["status"].eq(k).sum())
                                                         for k in ("css", "censored", "cap", "failed")})
                             for name, u in ((n, frame.drop_duplicates("id")) for n, frame in f.items())])
    return dict(violations=pd.DataFrame(violations), coverage=pd.DataFrame(coverage), paired=pd.DataFrame(pairs),
                fronts=fronts, recovery_loss=losses, spot_checks=pd.concat(spots, ignore_index=True),
                statuses=statuses)


def ci(lo, hi):
    return "–" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{lo:.3f}, {hi:.3f}]"


def report(t: dict, workflow: dict | None) -> str:
    v = t["violations"].assign(rate_ci=lambda d: [f"{k}/{n} = {r:.3f} {ci(a, b)}" for k, n, r, a, b in
                                                   zip(d["k"], d["n"], d["rate"], d["ci_lower"], d["ci_upper"])])
    wide_v = v.pivot(index="arm", columns="set", values="rate_ci")[list(SETS)].reset_index()
    c = t["coverage"].assign(cov=lambda d: [f"{r:.3f} {ci(a, b)}" for r, a, b in zip(d["rate"], d["ci_lower"], d["ci_upper"])])
    wide_c = c.pivot(index="arm", columns="set", values="cov")[list(SETS)].reset_index()
    loss = t["recovery_loss"]
    blocks = []
    for name in SETS:
        sub = loss.loc[loss["set"] == name]
        rec = sub.pivot(index="arm", columns="productivity", values="recovery")
        lvp = sub.loc[sub["arm"] != "GP_n512_P_a0.10"].pivot(index="arm", columns="productivity", values="loss_vs_P")
        chg = sub.pivot(index="arm", columns="productivity", values="change_vs_baseline")
        for frame in (rec, lvp, chg):
            frame.columns = [f"p = {x:g}" for x in frame.columns]
        blocks.append(f"### {name}\n\nRecovery (%) at matched productivity:\n\n{md(rec.reset_index())}\n\n"
                      f"Loss vs P in the same set (pp):\n\n{md(lvp.reset_index())}\n" +
                      ("" if name == "baseline" else
                       f"\nChange vs the same arm at baseline (pp; negative = lower recovery):\n\n{md(chg.reset_index())}\n"))
    runs = "" if workflow is None else md(pd.DataFrame([
        dict(block=b, n=p.get("counts", {}).get("n"), **p.get("counts", {}).get("status_counts", {}),
             wall_h=p.get("wall_s", np.nan) / 3600) for b, p in workflow.get("phases", {}).items()]))
    return f"""# Step 7b — model-form robustness of the GP n_train = 512 selected points (DESCRIPTIVE)

**DESCRIPTIVE analysis, designed after the Step 6c results** (docs/preregistration.md, Addendum A). No hypothesis
is tested, no multiplicity correction is applied and nothing here is confirmatory; McNemar p-values are descriptive.
Points: the 210 selected S_nom points of the 5 GP arms at n_train = 512 (P, G, L at alpha 0.10; G, L at 0.05), each
re-simulated under {", ".join(SCENARIOS)}; baseline = their Step 6c S_nom labels. Bounds: the frozen nominal
bounds of Step 6b. Violation: z > 2; censored labels count as z = 3; failed runs are excluded and counted.
Generated by `scripts/analyse_step7b.py`; tables in `docs/validation/step7b/`.

## Runs

{runs}

Label status of the 210 points per set:

{md(t["statuses"])}

Warm-start spot check (cold start from the He-filled column vs warm start from the point's own baseline CSS state;
tolerance |dz| <= 0.01):

{md(t["spot_checks"][["scenario", "id", "status_warm", "status_cold", "z_warm", "z_cold", "dz"]], 4)}

## Violation rate per arm (variant x alpha) and scenario

Clopper-Pearson 95 % CI.

{md(wide_v)}

## Paired change vs baseline (same points)

McNemar exact two-sided p, descriptive only.

{md(t["paired"][["set", "arm", "n", "baseline_k", "scenario_k", "feasible_to_violating", "violating_to_feasible",
                 "mcnemar_exact_p_two_sided", "median_dz", "failed_pairs"]])}

## Coverage of the frozen bounds (z <= bound)

P: the bound is the point prediction.

{md(wide_c)}

## Fronts and recovery loss (Step 7a definitions)

Reference point (recovery 0 %, productivity 0); feasible = CSS or cap with z <= 2.

{md(t["fronts"])}

{chr(10).join(blocks)}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    tables = analyse(args.generated_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    wf = ROOT / "results" / "step7b" / "workflow.json"
    text = report(tables, json.loads(wf.read_text()) if wf.exists() else None)
    (args.out / "REPORT.md").write_text(text)
    if args.out.resolve() == DEFAULT_OUT.resolve():
        docs = ROOT / "docs" / "validation" / "step7b"
        docs.mkdir(parents=True, exist_ok=True)
        for name in tables:
            shutil.copy(args.out / f"{name}.csv", docs / f"{name}.csv")
        shutil.copy(args.out / "REPORT.md", ROOT / "docs" / "validation" / "step7b.md")
    print(f"Step 7b descriptive tables written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
