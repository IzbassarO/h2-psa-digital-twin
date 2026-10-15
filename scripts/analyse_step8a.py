"""Step 8a analysis (docs/preregistration.md, Addendum B). CONFIRMATORY for H_B1-H_B3 only; everything else here is
descriptive. Designed after Step 6c, 6d and 7b.

    python scripts/analyse_step8a.py [--generated-dir data/generated] [--out results/step8a/analysis]

Inputs (--generated-dir): S_rep_points.csv, S_rep_memberships.csv, S_rep.csv (Step 8a labels); S_nom.csv for the
points that reuse a Step 6c label (snom_id) and, with S_nom_memberships.csv, for the descriptive Step 6c rates.
Pooled group = 3 arms (n_train 128/256/512, alpha 0.10), each point once per arm (first seed); final failures
excluded (sensitivity: counted as violations).
* H_B1: GP-L rate <= 0.17, exact binomial (less); also the one-sided Clopper-Pearson upper bound at the Holm level.
* H_B2: GBM-L rate > 0.10, exact binomial (greater).
* H_B3: GP-G rate > GP-L rate, one-sided Fisher exact.
Holm over the three at 0.05 (ties in the order B1, B2, B3); confirmed iff Holm-adjusted p < 0.05. Robustness flag:
cluster bootstrap over (arm, seed), 2000 draws, one default_rng(8) used in the order B1, B2, B3 (B3 paired over
(n_train, seed)). With the default --out, REPORT.md is copied to docs/validation/step8a.md and the tables to
docs/validation/step8a/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from h2psa.step6_metrics import (_cluster_groups, _percentiles, _resample, cp_upper,  # noqa: E402
                                 pooled_selected, violation_summary)
from report_step6c import md  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "step8a" / "analysis"
HYPOTHESES = ("H_B1", "H_B2", "H_B3")
GROUPS = {"GP_L": ("GP", "L"), "GBM_L": ("GBM", "L"), "GP_G": ("GP", "G")}
DRAWS, BOOT_SEED = 2000, 8


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def labels(generated: Path) -> pd.DataFrame:
    """One label per S_rep ID: simulated in Step 8a, or the Step 6c label of the identical S_nom point."""
    points = read(generated / "S_rep_points.csv")
    reused = points.loc[points["snom_id"].notna(), ["id", "snom_id"]].merge(
        read(generated / "S_nom.csv")[["id", "status", "z"]].rename(columns={"id": "snom_id"}),
        on="snom_id", validate="many_to_one")
    out = pd.concat([read(generated / "S_rep.csv")[["id", "status", "z"]], reused[["id", "status", "z"]]],
                    ignore_index=True)
    if out["id"].duplicated().any() or set(out["id"]) != set(points["id"]):
        raise ValueError("Labels must cover every S_rep point exactly once")
    return out.assign(reused=out["id"].isin(reused["id"]))


def selected(memberships: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Rows (arm, id) counted once per arm (first seed), alpha 0.10, with labels and a group column."""
    frame = pooled_selected(memberships.merge(lab, on="id", how="left", validate="many_to_one"))
    groups = {v: k for k, v in GROUPS.items()}
    frame = frame.assign(group=[groups.get(t) for t in zip(frame["surrogate"], frame["variant"])])
    return frame.loc[frame["group"].notna()]


def p_values(k: dict, n: dict) -> dict:
    return {"H_B1": float(binomtest(k["GP_L"], n["GP_L"], .17, alternative="less").pvalue),
            "H_B2": float(binomtest(k["GBM_L"], n["GBM_L"], .10, alternative="greater").pvalue),
            "H_B3": float(fisher_exact([[k["GP_G"], n["GP_G"] - k["GP_G"]], [k["GP_L"], n["GP_L"] - k["GP_L"]]],
                                       alternative="greater").pvalue)}


def holm(p: dict) -> dict:
    out, previous = {}, 0.0
    for rank, h in enumerate(sorted(HYPOTHESES, key=lambda h: (p[h], HYPOTHESES.index(h)))):
        previous = min(1.0, max(previous, (3 - rank) * p[h]))
        out[h] = dict(p_value=p[h], rank=rank + 1, level=.05 / (3 - rank), adjusted_p=previous,
                      verdict="confirmed" if previous < .05 else "not confirmed")
    return out


def tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pooled = pd.DataFrame([{"group": g, **violation_summary(frame.loc[frame["group"] == g])} for g in GROUPS])
    s = pooled.set_index("group")
    decisions = holm(p_values(s["k"].to_dict(), s["n"].to_dict()))
    sens = p_values(s["sensitivity_k"].to_dict(), s["sensitivity_n"].to_dict())
    rows = []
    for h in HYPOTHESES:
        d = decisions[h]
        rows.append(dict(hypothesis=h, **d, sensitivity_p=sens[h],
                         cp_upper_at_level=cp_upper(int(s.loc["GP_L", "k"]), int(s.loc["GP_L", "n"]), d["level"])
                         if h == "H_B1" else np.nan))
    return pooled, pd.DataFrame(rows), decisions


def bootstrap(frame: pd.DataFrame, decisions: dict, draws: int = DRAWS) -> pd.DataFrame:
    rng, rows = np.random.default_rng(BOOT_SEED), []
    for h, group, upper, threshold in (("H_B1", "GP_L", True, .17), ("H_B2", "GBM_L", False, .10)):
        source = frame.loc[frame["group"] == group]
        groups = _cluster_groups(source, ["arm", "seed"])
        keys = list(groups)
        rates = [violation_summary(_resample(groups, keys, rng.integers(len(keys), size=len(keys)), source))["rate"]
                 for _ in range(draws)]
        level = decisions[h]["level"]
        (bound,), undefined = _percentiles(rates, (1 - level if upper else level,))
        ok = bound <= threshold if upper else bound > threshold
        rows.append(dict(hypothesis=h, statistic=f"{group} rate", side="upper" if upper else "lower", level=level,
                         bound=bound, threshold=threshold, n_clusters=len(keys), undefined_draws=undefined,
                         verdict="undefined" if undefined else "confirmed" if ok else "not confirmed"))
    sub = {g: frame.loc[frame["group"] == g] for g in ("GP_G", "GP_L")}
    groups = {g: _cluster_groups(f, ["n_train", "seed"]) for g, f in sub.items()}
    keys = sorted(set(groups["GP_G"]) | set(groups["GP_L"]))
    differences = []
    for _ in range(draws):
        i = rng.integers(len(keys), size=len(keys))
        r = {g: violation_summary(_resample(groups[g], keys, i, sub[g]))["rate"] for g in sub}
        differences.append(r["GP_G"] - r["GP_L"])
    level = decisions["H_B3"]["level"]
    (bound,), undefined = _percentiles(differences, (level,))
    rows.append(dict(hypothesis="H_B3", statistic="GP_G - GP_L rate", side="lower", level=level, bound=bound,
                     threshold=0.0, n_clusters=len(keys), undefined_draws=undefined,
                     verdict="undefined" if undefined else "confirmed" if bound > 0 else "not confirmed"))
    out = pd.DataFrame(rows)
    out["primary_verdict"] = [decisions[h]["verdict"] for h in out["hypothesis"]]
    out["robustness"] = np.where(out["verdict"] == "undefined", "undefined",
                                 np.where(out["verdict"] == out["primary_verdict"], "robust", "not robust"))
    return out


def per_arm(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{**dict(zip(keys, key)), **violation_summary(g)}
                         for key, g in frame.groupby(keys, sort=False)])


def analyse(generated: Path, draws: int = DRAWS) -> dict[str, pd.DataFrame]:
    lab = labels(generated)
    frame = selected(read(generated / "S_rep_memberships.csv"), lab)
    pooled, hyp, decisions = tests(frame)
    step6c = selected(read(generated / "S_nom_memberships.csv"), read(generated / "S_nom.csv")[["id", "status", "z"]])
    arms = per_arm(frame, ["group", "arm"]).merge(
        per_arm(step6c, ["group", "arm"])[["arm", "n", "k", "rate"]].rename(
            columns={"n": "n_6c", "k": "k_6c", "rate": "rate_6c"}), on="arm", how="left")
    statuses = pd.DataFrame([dict(source="reused S_nom" if reused else "simulated S_rep", n=len(u),
                                  **{s: int(u["status"].eq(s).sum()) for s in ("css", "censored", "cap", "failed")})
                             for reused, u in lab.groupby("reused")])
    return dict(hypotheses=hyp, bootstrap=bootstrap(frame, decisions, draws), pooled=pooled,
                arms=arms, seeds=per_arm(frame, ["group", "arm", "seed"]),
                pooled_6c=pd.DataFrame([{"group": g, **violation_summary(step6c.loc[step6c["group"] == g])}
                                        for g in GROUPS]),
                statuses=statuses)


def report(t: dict, workflow: dict | None) -> str:
    rate = lambda d: d.assign(rate_ci=[f"{k}/{n} = {r:.3f} [{a:.3f}, {b:.3f}]" for k, n, r, a, b in  # noqa: E731
                                       zip(d["k"], d["n"], d["rate"], d["ci_lower"], d["ci_upper"])])
    sel = "" if workflow is None else f"Selection: `{json.dumps(workflow.get('selection'))}`."
    runs = "" if workflow is None else md(pd.DataFrame([
        dict(block=b, wall_h=p.get("wall_s", np.nan) / 3600, **p.get("counts", {}).get("status_counts", {}))
        for b, p in workflow.get("phases", {}).items()]))
    h = t["hypotheses"]
    return f"""# Step 8a — confirmatory replication of the Step 6d finding

Pre-registered in `docs/preregistration.md`, **Addendum B (2026-10-15), designed after Step 6c, 6d and 7b**.
Confirmatory for H_B1-H_B3 only; all other tables are descriptive. Frozen Step 6b surrogates, q and NSGA-II settings;
new seeds 13, 14, 15; 9 arms (GP-L, GBM-L, GP-G; n_train 128/256/512; alpha 0.10) x 45 points. Violation: z > 2
(censored z = 3); final failures excluded (sensitivity: counted as violations). Generated by
`scripts/analyse_step8a.py`; tables in `docs/validation/step8a/`.

## Points and runs

{sel}

{md(t["statuses"])}

{runs}

## Confirmatory tests (Holm over H_B1-H_B3, family-wise 0.05)

* H_B1: pooled GP-L rate <= 0.17 (exact binomial, less; Clopper-Pearson one-sided upper bound at the Holm level).
* H_B2: pooled GBM-L rate > 0.10 (exact binomial, greater).
* H_B3: pooled GP-G - GP-L > 0 (one-sided Fisher exact).

{md(h[["hypothesis", "p_value", "rank", "level", "adjusted_p", "verdict", "cp_upper_at_level", "sensitivity_p"]])}

Pooled violation rates (Clopper-Pearson 95 %):

{md(rate(t["pooled"])[["group", "rate_ci", "failed", "sensitivity_k", "sensitivity_n"]])}

## Robustness flag: cluster bootstrap over (arm, seed)

{md(t["bootstrap"][["hypothesis", "statistic", "side", "level", "bound", "threshold", "n_clusters",
                    "undefined_draws", "verdict", "primary_verdict", "robustness"]])}

## Descriptive: per arm (Step 8a, seeds 13-15) beside Step 6c (seeds 10-12)

{md(rate(t["arms"])[["group", "arm", "rate_ci", "failed", "k_6c", "n_6c", "rate_6c"]])}

Step 6c pooled rates of the same groups:

{md(rate(t["pooled_6c"])[["group", "rate_ci", "failed"]])}

## Descriptive: per arm and seed

{md(t["seeds"][["group", "arm", "seed", "n", "k", "rate", "failed"]])}
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
    wf = ROOT / "results" / "step8a" / "workflow.json"
    (args.out / "REPORT.md").write_text(report(tables, json.loads(wf.read_text()) if wf.exists() else None))
    if args.out.resolve() == DEFAULT_OUT.resolve():
        docs = ROOT / "docs" / "validation" / "step8a"
        docs.mkdir(parents=True, exist_ok=True)
        for name in tables:
            shutil.copy(args.out / f"{name}.csv", docs / f"{name}.csv")
        shutil.copy(args.out / "REPORT.md", ROOT / "docs" / "validation" / "step8a.md")
    print(tables["hypotheses"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
