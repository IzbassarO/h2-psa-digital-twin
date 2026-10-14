"""Step 6c report: formats the frozen analysis output; computes no new statistics.

    python scripts/analyse_step6.py        # registered analysis -> results/step6b/metrics/
    python scripts/report_step6c.py        # this report

Writes results/step6c/REPORT.md, copies it to docs/validation/step6c.md and the metric tables to
docs/validation/step6c/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep   # Newcombe, as in the analysis

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)


def md(df: pd.DataFrame, digits: int = 4) -> str:
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "–" if not np.isfinite(v) else f"{v:.{digits}g}"
        return str(v)
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def ci(lo, hi, digits=3):
    return "–" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "step6b" / "metrics")
    ap.add_argument("--generated", type=Path, default=ROOT / "data" / "generated")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "step6c")
    ap.add_argument("--docs", type=Path, default=ROOT / "docs" / "validation")
    a = ap.parse_args(argv)
    M = a.metrics
    res = json.loads((M / "metrics.json").read_text())
    t = {name: pd.read_csv(M / f"{name}.csv") for name in
         ("violations", "pooled_violations", "coverage", "bands", "pooled_bands", "fronts", "gate",
          "logistic_coefficients", "statuses", "power", "failed_ids")}
    dec, rob, cnt = res["tests"]["decisions"], res["robustness"], res["tests"]["pooled_counts"]
    lg, mc = res["logistic"], res["mcnemar"]
    nan = float("nan")
    f = lambda v: nan if v is None else v                                  # JSON null = undefined

    # runs
    wf_path = a.out / "workflow.json"
    wf = json.loads(wf_path.read_text()) if wf_path.exists() else {"phases": {}}
    run_rows = []
    for block in ("S_nom", "S_dist"):
        lab = pd.read_csv(a.generated / f"{block}.csv")
        s = t["statuses"].set_index("set").loc[block]
        w = lab["wall_time_s"] / 60
        run_rows.append(dict(block=block, n=int(s["n"]), css=int(s["css"]), censored=int(s["censored"]),
                             cap=int(s["cap"]), failed=int(s["failed"]), successful_retries=int(s["successful_retries"]),
                             batch_wall_h=wf["phases"].get(block, {}).get("wall_s", nan) / 3600,
                             point_median_min=w.median(), point_p90_min=w.quantile(.9), point_max_min=w.max()))
    runs = pd.DataFrame(run_rows)

    # primary RQ table
    def rate_row(rq, variant, label):
        c, d, b = cnt[variant], dec[rq], rob[rq]
        bnd = b.get("lower", nan)
        return dict(RQ=rq, quantity=label, estimate=f"{c['k']}/{c['n']} = {c['rate']:.3f}",
                    ci95=ci(c["ci_lower"], c["ci_upper"]), p_one_sided=f(d["p_value"]), holm_rank=d["rank"],
                    holm_level=d["level"], p_adjusted=f(d["adjusted_p"]), verdict=d["verdict"],
                    bootstrap=f"lower {f(bnd):.3f} at {d['level']:.4g}: {b['verdict']}", robustness=b["robustness"])
    rows = [rate_row("RQ1", "P", "violation rate P (H0 <= 0.10)"),
            rate_row("RQ2", "G", "violation rate G, alpha 0.10 (H0 <= 0.10)")]
    g, l, d3, b3 = cnt["G"], cnt["L"], dec["RQ3"], rob["RQ3"]
    lo, hi = confint_proportions_2indep(g["k"], g["n"], l["k"], l["n"], method="newcomb", compare="diff")
    rows.append(dict(RQ="RQ3", quantity="L rate (H0 >= 0.17) and G - L (H0 <= 0)",
                     estimate=f"L {l['k']}/{l['n']} = {l['rate']:.3f}; G - L = {g['rate'] - l['rate']:+.3f}",
                     ci95=f"L {ci(l['ci_lower'], l['ci_upper'])}; G - L {ci(lo, hi)}",
                     p_one_sided=f(d3["p_value"]), holm_rank=d3["rank"], holm_level=d3["level"],
                     p_adjusted=f(d3["adjusted_p"]), verdict=d3["verdict"] + f" (p_a {f(d3['p_a']):.3g}, p_b {f(d3['p_b']):.3g})",
                     bootstrap=f"L upper {f(b3['L_upper']):.3f}, G - L lower {f(b3['difference_lower']):+.3f}: {b3['verdict']}",
                     robustness=b3["robustness"]))
    d5, b5 = dec["RQ5"], rob["RQ5"]
    rows.append(dict(RQ="RQ5", quantity="b_phi, logistic miscoverage (H0 >= 0)",
                     estimate=f"{f(lg.get('b_phi', nan)):.3f} (OR phi 0.70 vs 1.00 = {f(lg.get('odds_ratio', nan)):.2f})",
                     ci95=ci(f(lg.get("b_phi_lower", nan)), f(lg.get("b_phi_upper", nan))),
                     p_one_sided=f(d5["p_value"]), holm_rank=d5["rank"], holm_level=d5["level"],
                     p_adjusted=f(d5["adjusted_p"]), verdict=d5["verdict"],
                     bootstrap=f"upper {f(b5['upper']):.3f}: {b5['verdict']}", robustness=b5["robustness"]))
    rq = pd.DataFrame(rows)

    sens = res["failure_sensitivity_tests"]["decisions"]
    n_failed = int(t["statuses"]["failed"].sum())

    v = t["violations"]
    prim = v[(v.set == "S_nom") & np.isclose(v.alpha, .10)]
    sec = v[(v.set == "S_nom") & np.isclose(v.alpha, .05)]
    dist = v[v.set == "S_dist"]
    vcols = ["arm", "n", "k", "rate", "ci_lower", "ci_upper", "failed", "selection_shortfall"]

    pb = t["pooled_bands"]
    bands = t["bands"][np.isclose(t["bands"].alpha, .10)]
    bcols = ["arm", "lower", "n_band_observed", "band_coverage", "selected_n", "selected_coverage",
             "selected_inside_fraction", "difference", "difference_lower", "difference_upper"]

    cov = t["coverage"]
    ct = cov[cov.set == "D_test"][["arm", "alpha", "n", "rate", "ci_lower", "ci_upper"]].rename(
        columns={"n": "n_test", "rate": "cov_test", "ci_lower": "test_lo", "ci_upper": "test_hi"})
    cs = cov[cov.set == "S_nom"][["arm", "n", "rate", "ci_lower", "ci_upper"]].rename(
        columns={"n": "n_sel", "rate": "cov_selected", "ci_lower": "sel_lo", "ci_upper": "sel_hi"})
    cvt = ct.merge(cs, on="arm")
    cvt["selected_minus_test"] = cvt.cov_selected - cvt.cov_test
    cshift = cov[cov.set == "D_shift"][["arm", "n", "rate", "ci_lower", "ci_upper"]]

    fr = t["fronts"]
    fcols = ["arm", "n_front", "hv_ratio", "hv_ratio_lower", "hv_ratio_upper", "hv_ratio_minus_P",
             *[f"loss_{x:g}" for x in LEVELS]]
    fprim, fsec = fr[np.isclose(fr.alpha, .10)][fcols], fr[np.isclose(fr.alpha, .05)][fcols]

    coef = t["logistic_coefficients"]
    slg = res["failure_sensitivity_logistic"]

    text = f"""# Step 6c — selected-point simulations and pre-registered analysis (prereg-v1)

Generated by `scripts/report_step6c.py` from the output of `scripts/analyse_step6.py`, run unchanged as committed
in 8bd64ee (Step 6b, 2026-10-13, before any S_nom / S_dist simulation). Simulations: `scripts/run_step6c.py`
(the registered runner `scripts/run_batch.py` with the Step 6a flags and pause rules, 6 workers, `caffeinate -i`;
deviation of 2026-10-13). Pre-simulation check of the zero CQR quantiles: `docs/validation/step6b.md`, addendum
(case b, ties at z = 3; no change). Metric tables: `docs/validation/step6c/`.

## Simulations

{md(runs, 3)}

## Pre-registered confirmatory results (section 8)

Holm step-down over RQ1, RQ2, RQ3, RQ5 at family-wise 0.05; CIs are two-sided 95 % (Clopper-Pearson; Newcombe for
G - L; cluster-robust Wald for b_phi) and are effect sizes, not decisions. Cluster bootstrap (2000 draws,
`default_rng(6)`, clusters (arm, NSGA-II seed); RQ5: D_shift points and S_dist (arm, seed)) at the Holm level
of each RQ. Sanity gate (all primary G / L arms on D_test): {"passed" if res["tests"]["gate_pass"] else "FAILED"}.

{md(rq, 3)}

RQ4 is descriptive (fronts below). Final solver failures: {n_failed}{"" if n_failed == 0 else " — failure-sensitivity verdicts: " + ", ".join(f"{k} {sens[k]['verdict']}" for k in ("RQ1", "RQ2", "RQ3", "RQ5"))}.

### Pooled violation rates of the selected points (alpha = 0.10, 6 arms per variant)

{md(t["pooled_violations"][["variant", "n", "k", "rate", "ci_lower", "ci_upper", "failed"]])}

## Band-matched comparison (primary for RQ2, RQ3)

Pooled over the 6 primary arms of each variant. Selection effect if the Newcombe 95 % CI of (band-matched D_test
coverage - selected coverage) lies above 0; otherwise boundary effect.

{md(pb[["variant", "n_band", "band_coverage", "n_selected", "selected_coverage", "selected_inside_fraction", "difference", "difference_lower", "difference_upper", "interpretation"]])}

Per arm (alpha = 0.10; band [lower, 2.0], fixed widening rule of Step 6b):

{md(bands[bcols])}

## Coverage: D_test (random points) vs selected points (S_nom)

{md(cvt[["arm", "n_test", "cov_test", "test_lo", "test_hi", "n_sel", "cov_selected", "sel_lo", "sel_hi", "selected_minus_test"]])}

## Fronts (RQ4): HV ratio and recovery loss

Reference front: feasible non-dominated nominal points of D_train, D_cal, D_test and S_nom
({len(pd.read_csv(M / "reference_front.csv"))} points). HV ratio with bootstrap 95 % interval (2000 resamples,
`default_rng(5)`); recovery loss in pp at 0.1 ... 0.9 of the reference productivity range ("–" = not reached).
Per-level bootstrap intervals: `docs/validation/step6c/fronts.csv`.

{md(fprim)}

## RQ5: logistic regression of miscoverage (primary)

G and L bounds, n_train = 512, alpha = 0.10; rows: D_shift (128 points x 4 bounds) and the G / L points of S_dist.
Status: {lg["status"]}; n = {lg.get("n", "–")}, clusters = {lg.get("n_clusters", "–")}. Odds ratio phi 0.70 vs 1.00:
{f(lg.get("odds_ratio", nan)):.3g} {ci(f(lg.get("odds_ratio_lower", nan)), f(lg.get("odds_ratio_upper", nan)), 2)}.
Failure sensitivity: {slg["status"]}, b_phi = {f(slg.get("b_phi", nan)):.3g}.

{md(coef)}

## Secondary analyses (reported, not decisive)

### Per-arm violation rates, primary arms (alpha = 0.10; trends with training size)

{md(prim[vcols])}

### alpha = 0.05 arms (n_train = 512)

{md(sec[vcols])}

{md(fsec)}

### RQ5 secondary: paired McNemar (nominal S_nom vs disturbed S_dist, 6 arms at n_train = 512)

n = {mc["n"]}, nominal violations {mc["nominal_violations"]}, disturbed {mc["disturbed_violations"]}; feasible to
violating {mc["increase"]}, violating to feasible {mc["decrease"]}; exact one-sided p = {f(mc["p_value"]):.4g}
(failed pairs {mc["failed_pairs"]}).

Per-arm violation rates under disturbance (S_dist):

{md(dist[vcols])}

Coverage on D_shift with the nominal bounds:

{md(cshift)}

### Registered RQ3 power (section 8; ignores clustering)

{md(t["power"])}
"""
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "REPORT.md").write_text(text)
    (a.docs / "step6c.md").write_text(text)
    tables = a.docs / "step6c"
    tables.mkdir(parents=True, exist_ok=True)
    for p in sorted(M.glob("*")):
        shutil.copy2(p, tables / p.name)
    print(f"wrote {a.out / 'REPORT.md'}, {a.docs / 'step6c.md'} and {tables}/")


if __name__ == "__main__":
    main()
