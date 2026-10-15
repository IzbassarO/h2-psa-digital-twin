"""Step 6d: EXPLORATORY diagnostics of the Step 6c results (not pre-registered). No simulations, no refits.

    python scripts/explore_step6d.py

Reads the Step 6c labels, the frozen Step 6b models / predictions / q, and the frozen metric functions
(`h2psa.step6_metrics`, used unchanged). Writes tables to results/step6d/ and docs/validation/step6d/, and the
report to results/step6d/REPORT.md and docs/validation/step6d.md. Nothing of Step 6c is modified.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from h2psa.step6 import INPUTS, Surrogates, conformal_quantile  # noqa: E402
from h2psa.step6_metrics import (_observed, clopper_pearson, feasible_front, front_metrics,  # noqa: E402
                                 hypervolume_2d, join_predictions, newcombe_difference, violation_summary)
from report_step6c import md  # noqa: E402

GEN, FROZEN, MODELS = ROOT / "data" / "generated", ROOT / "docs" / "validation" / "step6b", ROOT / "results" / "step6b"
OUT, DOCS = ROOT / "results" / "step6d", ROOT / "docs" / "validation" / "step6d"
PROBS = (0.10, 0.25, 0.50, 0.75, 0.90)


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def add_components(frame: pd.DataFrame, q: dict) -> pd.DataFrame:
    """Per (id, arm): GP/GBM mean, centre and scale of the frozen bound; checks bound = centre + q * scale."""
    parts, cache = [], {}
    for arm, group in frame.groupby("arm", sort=True):
        info = group.iloc[0]
        key = (info["surrogate"], int(info["n_train"]))
        s = cache.setdefault(key, Surrogates(*key, MODELS))
        X = group[INPUTS].to_numpy(float)
        centre, scale = s.components(X, info["variant"], float(info["alpha"]))
        group = group.assign(mean=s.models["z"].predict(X), centre=centre, scale=scale, q=q.get(arm, 0.0))
        if "bound" in group and not np.allclose(group["centre"] + group["q"] * group["scale"], group["bound"],
                                                rtol=0, atol=1e-9):
            raise ValueError(f"{arm}: frozen bound not reproduced from the models")
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def out_of_sample_q(frame: pd.DataFrame, key: str, alpha: float) -> pd.Series:
    """For each row, the split-conformal quantile of the scores of rows with a different `key`."""
    out = pd.Series(np.nan, index=frame.index)
    for value in frame[key].unique():
        held = frame[key] == value
        out[held] = conformal_quantile(frame.loc[~held, "score"].to_numpy(), alpha)[0]
    return out


def rate(k, n):
    lo, hi = clopper_pearson(k, n)
    return dict(n=n, k=k, rate=k / n if n else np.nan, ci_lower=lo, ci_upper=hi)


def quantiles(values, prefix):
    values = np.asarray(values, float)
    return {f"{prefix}_q{int(p*100):02d}": float(np.quantile(values, p)) for p in PROBS} if len(values) else {}


def breakdowns(sel):
    """Section 1: per-arm and pooled violation rates; G - L by surrogate (and by surrogate x n)."""
    arms = []
    for arm, g in sel.groupby("arm", sort=True):
        i = g.iloc[0]
        arms.append(dict(surrogate=i["surrogate"], n_train=int(i["n_train"]), variant=i["variant"],
                         alpha=float(i["alpha"]), **violation_summary(g)))
    arms = pd.DataFrame(arms).sort_values(["alpha", "surrogate", "n_train", "variant"], ascending=[False, True, True, True])
    primary = sel.loc[np.isclose(sel["alpha"], .10)]
    pooled = []
    for (s, v), g in primary.groupby(["surrogate", "variant"], sort=True):
        pooled.append(dict(surrogate=s, variant=v, **violation_summary(g)))
    contrasts = []
    for keys, g in [((s, "all"), g) for s, g in primary.groupby("surrogate")] + \
                   [((s, n), g) for (s, n), g in primary.groupby(["surrogate", "n_train"])]:
        G, L = violation_summary(g.loc[g["variant"] == "G"]), violation_summary(g.loc[g["variant"] == "L"])
        d, lo, hi = newcombe_difference(G["k"], G["n"], L["k"], L["n"])
        contrasts.append(dict(surrogate=keys[0], n_train=keys[1], G=f"{G['k']}/{G['n']}", L=f"{L['k']}/{L['n']}",
                              G_minus_L=d, ci_lower=lo, ci_upper=hi,
                              direction="L worse" if hi < 0 else "L better" if lo > 0 else "no clear difference"))
    return arms, pd.DataFrame(pooled), pd.DataFrame(contrasts)


def mechanism(sel, test, cal, bands):
    """Section 2: uncertainty scale and error/scale at selected points vs D_test (all and band-matched)."""
    rows = []
    for arm, s in sel.loc[sel["variant"] == "L"].groupby("arm", sort=True):
        t = test.loc[test["arm"] == arm]
        b = bands.loc[bands["arm"] == arm].iloc[0]
        groups = {"D_test": t, "D_test band": t.loc[t["bound"].between(b["lower"], b["upper"])], "selected": s}
        for name, g in groups.items():
            o = _observed(g)
            gp = o.iloc[0]["surrogate"] == "GP"
            width = o["scale"] if gp else o["centre"] - o["mean"]      # GP sigma; GBM raw CQR width Qhat - mean
            err = o["z"] - o["mean"]
            ok = width > 0
            rows.append(dict(arm=arm, set=name, n=len(o), width="sigma" if gp else "Qhat-mean",
                             **quantiles(width, "w"), nonpositive_width=int((~ok).sum()),
                             **quantiles((err.abs() / width)[ok], "abs_err_over_w"),
                             median_signed_err=float(err.median()), frac_underpredicted=float((err > 0).mean()),
                             margin_median=float((o["bound"] - o["mean"]).median()),
                             miscoverage=float((o["z"] > o["bound"]).mean())))
    ties = []
    for arm, s in sel.loc[(sel["variant"] == "L") & (sel["surrogate"] == "GBM")].groupby("arm", sort=True):
        for name, g in (("D_cal", cal.loc[cal["arm"] == arm]), ("D_test", test.loc[test["arm"] == arm]), ("selected", s)):
            o = _observed(g)
            at3 = o["centre"] == 3.0
            ties.append(dict(arm=arm, set=name, n=len(o), qhat_eq_3=int(at3.sum()), fraction=float(at3.mean()),
                             qhat_max=float(o["centre"].max()), zero_scores=int((o["z"] - o["centre"] == 0).sum())))
    return pd.DataFrame(rows), pd.DataFrame(ties)


def fronts(sel, nominal, labels_by_set):
    """Section 3: per-arm feasibility, true (normalised) hypervolume, HV ratio and where the reference comes from."""
    reference = feasible_front(nominal)
    nadir, ideal = reference.min(axis=0), reference.max(axis=0)
    width = ideal - nadir
    ref_hv = hypervolume_2d((reference - nadir) / width)
    rows = []
    for arm, g in sel.groupby("arm", sort=True):
        o, i = _observed(g), g.iloc[0]
        feas = o.loc[o["status"].isin(("css", "cap")) & o["z"].le(2.0)]
        norm = (feas[["recovery", "productivity"]].to_numpy() - nadir) / width
        box = (norm > -0.1).all(axis=1) if len(norm) else np.zeros(0, bool)
        m = front_metrics(o, reference)
        hv = hypervolume_2d(norm) if len(norm) else 0.0
        rows.append(dict(arm=arm, surrogate=i["surrogate"], n_train=int(i["n_train"]), variant=i["variant"],
                         alpha=float(i["alpha"]), n_selected=len(g), n_observed=len(o), n_feasible=len(feas),
                         n_feasible_in_hv_box=int(box.sum()), n_front=m["n_front"],
                         max_recovery=float(feas["recovery"].max()) if len(feas) else np.nan,
                         max_productivity=float(feas["productivity"].max()) if len(feas) else np.nan,
                         hv=hv, hv_ratio=hv / ref_hv, **{k: m[k] for k in m if k.startswith("loss_")}))
    origin = []
    for rec, prod in reference:
        hit = nominal.loc[np.isclose(nominal["recovery"], rec, rtol=0, atol=1e-12) &
                          np.isclose(nominal["productivity"], prod, rtol=0, atol=1e-12)]
        ids = hit["id"].tolist()
        arms = [labels_by_set["S_nom_arm"].get(x, "") for x in ids]
        origin.append(dict(recovery=rec, productivity=prod, id=";".join(ids), arm=";".join(a for a in arms if a)))
    box = dict(nadir_recovery=nadir[0], nadir_productivity=nadir[1], ideal_recovery=ideal[0],
               ideal_productivity=ideal[1], reference_hv=ref_hv,
               recovery_cutoff=nadir[0] - 0.1 * width[0], productivity_cutoff=nadir[1] - 0.1 * width[1])
    return pd.DataFrame(rows), pd.DataFrame(origin), box, reference


def recalibration(sel, reference):
    """Section 4: q from the re-simulated selected points (other seeds / other arms), evaluated on held-out points."""
    gl = _observed(sel.loc[sel["variant"].isin(("G", "L"))]).copy()
    gl["score"] = (gl["z"] - gl["centre"]) / gl["scale"]
    gl["group"] = gl["surrogate"] + "-" + gl["variant"] + np.where(
        (gl["surrogate"] == "GBM") & (gl["variant"] == "L"), "-a" + gl["alpha"].map("{:.2f}".format), "")
    rows, details = [], []
    for scheme in ("seed rotation", "leave one arm out"):
        for arm, g in gl.groupby("arm", sort=True):
            alpha = float(g.iloc[0]["alpha"])
            if scheme == "seed rotation":
                g = g.assign(q_new=out_of_sample_q(g, "seed", alpha))
                q_values = g.groupby("seed")["q_new"].first().tolist()
            else:
                pool = gl.loc[gl["group"] == g.iloc[0]["group"]]
                if pool["arm"].nunique() < 2:
                    rows.append(dict(scheme=scheme, arm=arm, status="undefined: no other arm with the same score"))
                    continue
                q_new = conformal_quantile(pool.loc[pool["arm"] != arm, "score"].to_numpy(), alpha)[0]
                g, q_values = g.assign(q_new=q_new), [q_new]
            new_bound = g["centre"] + g["q_new"] * g["scale"]
            covered = int((g["z"] <= new_bound).sum())
            survive = g.loc[new_bound <= 2.0]
            m0, m1 = front_metrics(g, reference), front_metrics(survive, reference)
            rows.append(dict(scheme=scheme, arm=arm, status="defined", alpha=alpha, q_cal=float(g.iloc[0]["q"]),
                             q_new=";".join(f"{v:.3f}" for v in q_values),
                             coverage_before=float((g["z"] <= g["bound"]).mean()),
                             **{f"coverage_after_{k}": v for k, v in rate(covered, len(g)).items()},
                             n_survive=len(survive), survive_fraction=len(survive) / len(g),
                             survivors_violating=int(survive["z"].gt(2.0).sum()),
                             hv_ratio_before=m0["hv_ratio"], hv_ratio_survivors=m1["hv_ratio"],
                             n_front_before=m0["n_front"], n_front_survivors=m1["n_front"]))
            details.append(g.assign(scheme=scheme, new_bound=new_bound)[
                ["scheme", "arm", "seed", "id", "z", "status", "centre", "scale", "q", "bound", "score", "q_new", "new_bound"]])
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def main():
    q = read(FROZEN / "q.csv").set_index("arm")["q"].to_dict()
    labels = {name: read(GEN / f"{name}.csv") for name in ("D_train", "D_cal", "D_test", "S_nom")}
    sel = join_predictions(labels["S_nom"], read(FROZEN / "predictions" / "S_nom.csv"), read(GEN / "S_nom_memberships.csv"))
    test = join_predictions(labels["D_test"], read(FROZEN / "predictions" / "D_test.csv"))
    arms = test[["arm", "surrogate", "n_train", "variant", "alpha"]].drop_duplicates()
    cal = add_components(labels["D_cal"].merge(arms.loc[arms["variant"] == "L"], how="cross"), q)
    sel, test = add_components(sel, q), add_components(test, q)

    tables = {}
    tables["arm_violations"], tables["pooled_violations"], tables["g_minus_l"] = breakdowns(sel)
    tables["mechanism"], tables["gbm_qhat_ties"] = mechanism(sel, test, cal, read(FROZEN / "band.csv"))
    nominal = pd.concat(labels.values(), ignore_index=True)
    tables["fronts"], tables["reference_origin"], box, reference = fronts(
        sel, nominal, {"S_nom_arm": sel.set_index("id")["arm"].to_dict()})
    tables["recalibration"], tables["recalibration_points"] = recalibration(sel, reference)
    for d in (OUT, DOCS):
        d.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(d / f"{name}.csv", index=False)
    pd.Series(box).to_csv(OUT / "hv_box.csv", header=["value"])
    shutil.copy(OUT / "hv_box.csv", DOCS / "hv_box.csv")
    text = report(tables, box)
    (OUT / "REPORT.md").write_text(text)
    (ROOT / "docs" / "validation" / "step6d.md").write_text(text)
    print(f"wrote {OUT / 'REPORT.md'}, docs/validation/step6d.md and {DOCS}")


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def report(t, box):
    arms, pooled, gl = t["arm_violations"], t["pooled_violations"], t["g_minus_l"]
    mech, ties, fr, origin, rc = t["mechanism"], t["gbm_qhat_ties"], t["fronts"], t["reference_origin"], t["recalibration"]
    arms = arms.assign(ci95=[ci(a, b) for a, b in zip(arms["ci_lower"], arms["ci_upper"])])
    pooled = pooled.assign(ci95=[ci(a, b) for a, b in zip(pooled["ci_lower"], pooled["ci_upper"])])
    gl = gl.assign(ci95=[ci(a, b) for a, b in zip(gl["ci_lower"], gl["ci_upper"])])
    sel = mech.loc[mech["set"] == "selected"]
    gbm, gp = sel.loc[sel["width"] == "Qhat-mean"], sel.loc[sel["width"] == "sigma"]
    test = mech.loc[mech["set"] == "D_test"].set_index("arm")
    origin_arms = origin["arm"].value_counts()
    rc = rc.assign(coverage_after=[f"{k:.0f}/{n:.0f} = {r:.3f} {ci(a, b)}" if s == "defined" else "–"
                                   for k, n, r, a, b, s in zip(rc["coverage_after_k"], rc["coverage_after_n"],
                                                               rc["coverage_after_rate"], rc["coverage_after_ci_lower"],
                                                               rc["coverage_after_ci_upper"], rc["status"])])
    rot, loo = rc.loc[rc["scheme"] == "seed rotation"], rc.loc[rc["scheme"] == "leave one arm out"]
    rc_cols = ["arm", "q_cal", "q_new", "coverage_before", "coverage_after", "n_survive", "survivors_violating",
               "n_front_before", "n_front_survivors", "hv_ratio_before", "hv_ratio_survivors"]
    gbm_l_rot = rot.loc[rot["arm"].str.contains("GBM_n...?_L", regex=True)]
    return f"""# Step 6d — diagnostics and selection-aware recalibration (EXPLORATORY, not pre-registered)

**Everything in this report is EXPLORATORY.** It is not part of the pre-registered analysis (prereg-v1); no p-value
here is confirmatory and no multiplicity correction is applied. The Step 6c results, scripts and report are
unchanged. No new simulations and no refits: inputs are the Step 6c labels (`data/generated/S_nom.csv`, `D_*.csv`),
the frozen Step 6b models (`results/step6b/models`), predictions, `q.csv` and `band.csv`, and the frozen metric
functions of `h2psa.step6_metrics` used unchanged. Generated by `scripts/explore_step6d.py`; tables in
`docs/validation/step6d/`. Every frozen bound is reproduced from the models to 1e-9 before use.
Violation: z > 2 (impurity > 100 ppm); censored labels count as z = 3; the one failed point (S_nom_0590,
GBM_n256_P) is excluded, as in the primary analysis. CIs are Clopper-Pearson (rates) and Newcombe (differences).

## 1. Violation rates by variant x surrogate x n_train x alpha

{md(arms[["surrogate", "n_train", "variant", "alpha", "n", "k", "rate", "ci95", "failed"]])}

Pooled over n_train (alpha = 0.10):

{md(pooled[["surrogate", "variant", "n", "k", "rate", "ci95"]])}

G - L (alpha = 0.10), pooled and per n_train:

{md(gl[["surrogate", "n_train", "G", "L", "G_minus_L", "ci95", "direction"]])}

**Answer.** The pooled RQ3 result (L worse than G) is driven entirely by GBM. For GBM, L is worse than G at every
n_train; L violates at 132/135 points. For GP, L is better than G (5/135 against 51/135), with a clear difference at
n = 128 and 256 and none at 512, where both are near 0. Both registered alpha = 0.05 GP arms and GBM-G have 0/45
violations, while GBM-L at alpha = 0.05 violates at 45/45. The GBM quantile surrogate is the failure, not
conformal locality as such.

## 2. Mechanism for L

Width: GP predictive sigma, or for GBM the raw one-sided CQR width Qhat_(1-alpha)(x) - mean(x), before the
conformal correction q. Error = z - mean(x). "D_test band" is the band-matched D_test subset of Step 6c.
The ratio |error| / width is reported only where the width is > 0.

{md(mech[["arm", "set", "n", "width", "w_q10", "w_q50", "w_q90", "nonpositive_width", "abs_err_over_w_q50",
          "abs_err_over_w_q90", "median_signed_err", "frac_underpredicted", "miscoverage"]], 3)}

GBM-L: points with Qhat exactly 3.0 (D_cal ties of Step 6b) and zero conformity scores:

{md(ties)}

**Answer.**
- **GBM-L: hypothesis supported, in an extreme form.** At every selected point of every GBM-L arm
  ({int(gbm['nonpositive_width'].sum())}/{int(gbm['n'].sum())}), the 1 - alpha quantile model lies *below* the mean
  model (median Qhat - mean from {gbm['w_q50'].min():.2f} to {gbm['w_q50'].max():.2f}). This is quantile crossing.
  On D_test, crossing occurs at only {test.loc[gbm['arm'], 'nonpositive_width'].min():.0f} to
  {test.loc[gbm['arm'], 'nonpositive_width'].max():.0f} of 256 points. NSGA-II finds the regions where the separately
  fitted quantile GBM is spuriously low: there the bound Qhat + q falls below the mean prediction (median margin < 0),
  and the true z lies above the mean at {gbm['frac_underpredicted'].min():.0%} to
  {gbm['frac_underpredicted'].max():.0%} of the points. The constant shift q (0.15 to 0.30) is calibrated on random
  points and cannot repair a width that is negative only where the optimiser looks.
- **The Qhat = 3 ties do not drive this.** No selected point has Qhat = 3 (all selected Qhat <= 1.85, since the
  bound must be <= 2). The ties lie in the censored region of D_cal: 31/256 points (12 %) at n = 512 and alpha = 0.10,
  and 63/256 at alpha = 0.05. There they lower q, but q is small for every GBM-L arm whatever the cause.
- **GP-L: hypothesis not supported.** The optimiser selects points where sigma is *larger* than on D_test
  (median {gp['w_q50'].min():.2f} to {gp['w_q50'].max():.2f}, against {test.loc[gp['arm'], 'w_q50'].min():.2f} to
  {test.loc[gp['arm'], 'w_q50'].max():.2f}). Whether these are extrapolative regions was not checked. The median
  |error| / sigma at the selected points ({gp['abs_err_over_w_q50'].min():.2f} to
  {gp['abs_err_over_w_q50'].max():.2f}) is similar to D_test's and below q at the 90th percentile, so the larger
  sigma covers the larger error. At n = 128 and 256 the mean is biased low at the selected points (z > mean at 91 %),
  but sigma absorbs the bias. At n = 512 the GP-L bound is conservative.

## 3. RQ4 explained: feasibility, true hypervolume and HV ratio

The reference front is the feasible nondominated set of the nominal points of D_train, D_cal, D_test and S_nom.
Its 12 points come from: {', '.join(f'{a} ({c})' for a, c in origin_arms.items())}; none come from D_train,
D_cal or D_test. HV is computed in coordinates normalised to the reference box (recovery
{box['nadir_recovery']:.2f} to {box['ideal_recovery']:.2f} %, productivity {box['nadir_productivity']:.1f} to
{box['ideal_productivity']:.1f}). The reference point is at -0.1 in both normalised coordinates, so a point counts only
if recovery > {box['recovery_cutoff']:.2f} % **and** productivity > {box['productivity_cutoff']:.1f}. Reference
HV = {box['reference_hv']:.4f}. Feasible: CSS or cap with z <= 2.

{md(fr[["arm", "n_selected", "n_observed", "n_feasible", "n_feasible_in_hv_box", "n_front", "max_recovery",
        "max_productivity", "hv", "hv_ratio", "loss_0.1", "loss_0.3", "loss_0.5"]])}

**What "only GP_n512_P has a substantial front" means.** It does **not** mean the other arms have no feasible
points. GP-L arms have 40 to 45/45 feasible points and fronts of 22 to 31 points, and GP-G at n = 512 has 44/45.
It means that the HV metric is anchored to a reference front made almost entirely of GP_n512_P's own 13 feasible
points. These are P points that happened to be feasible while sitting on the predicted constraint, so they have the
highest productivity of all feasible points (221 to 267). GP_n512_P's ratio of 0.98 is therefore close to
tautological. The bounded arms trade productivity for feasibility: their feasible points top out at 150 to
237 productivity, mostly below the reference box ({box['productivity_cutoff']:.1f}), so their HV is 0 and their
recovery losses are "not reached". Only GBM_n512_G (2 points in the box), GP_n512_G (8) and GP_n256_P (1, which is
itself a reference point) reach it. The HV ratio is a ranking of the tail beyond the P arms' lucky feasible points,
not a measure of a usable front. A reference point at fixed, physically chosen levels would be needed to compare
the bounded fronts. That would be a new metric, not done here.

## 4. Selection-aware recalibration

Score per point: (z - centre) / scale with the frozen centre and scale of each bound (G: mean, 1; GP-L: mean, sigma;
GBM-L: Qhat, 1), the same score as on D_cal. New bound = centre + q_new * scale. Coverage is evaluated on held-out
points. "Survive" means the held-out selected point stays predicted-feasible under the new bound (<= 2). The
surviving front uses only the existing selected points; nothing is re-optimised.

- **Seed rotation**: q_new is the split-conformal quantile (frozen `conformal_quantile`, same alpha) of the 30
  points of the other two NSGA-II seeds. It is evaluated on the 15 points of the third seed, for all 3 rotations,
  pooled. q_new lists the three rotations (held-out seed 10, 11, 12).
- **Leave one arm out**: q_new is computed from all selected points of the other arms with the same surrogate and
  the same score definition (G with G, L with L; GBM-L only at the same alpha). It is evaluated on the held-out arm.

Seed rotation:

{md(rot[rc_cols], 3)}

Leave one arm out:

{md(loo[["arm", "status", *rc_cols[1:]]], 3)}

**Answer.**
- **Within an arm, recalibration on the selected points restores nominal coverage.** Pooled held-out coverage is
  {rot['coverage_after_rate'].min():.2f} to {rot['coverage_after_rate'].max():.2f}, against
  {rot['coverage_before'].min():.2f} to {rot['coverage_before'].max():.2f} before. q moves in both directions:
  - GBM-L needs q of about 1.23 to 1.31 instead of 0.15 to 0.30. After that, no GBM-L selected point stays
    predicted-feasible (survivors {int(gbm_l_rot['n_survive'].sum())}/{int(gbm_l_rot['coverage_after_n'].sum())}),
    so the front vanishes and would have to be re-optimised under the new q. The same holds wherever q rises:
    GP-G at n = 128 and 256, and GBM-G at alpha = 0.10 with n = 256 and 512.
  - Some arms are conservative at the selected points: GP-L at n = 256 and 512 (both alphas), GP-G at n = 512
    (both alphas), GBM-G at n = 128, and GBM-G at n = 512 with alpha = 0.05. There q_new < q_cal, every point
    survives and the fronts are unchanged. GP_n128_L is mixed and keeps 15/45 points (11 front points, 3 of the
    15 violating).
- **Across arms, q does not transfer.** Leave-one-arm-out coverage ranges from
  {loo['coverage_after_rate'].min():.2f} to {loo['coverage_after_rate'].max():.2f}. The selection-induced score shift
  depends on the surrogate's n_train, so a q from other arms either under-covers (GBM_n256_G 0.16, GP_n128_G 0.36,
  GBM_n128_L 0.56, GP_n128_L 0.58) or removes the front.
- **Ties.** GBM predictions are piecewise constant and censored labels are clipped at z = 3, so several selected
  GBM-L points share a score. Hence the identical q_new across rotations.

**Limitations.**
- Selected points are a small, dependent sample: 30 calibration points per rotation, and points of one seed come
  from one NSGA-II population. The exchangeability that split-conformal validity needs is not guaranteed between
  seeds, and still less between arms.
- The held-out seed was itself selected with the *old* bound. Coverage of points that NSGA-II would choose under
  the *new* bound is not measured. Doing so needs re-optimisation and new simulations, which are out of scope here.
- The censored labels (z = 3) make some scores lower bounds.
- Results are descriptive and are not a test.
"""


if __name__ == "__main__":
    main()
