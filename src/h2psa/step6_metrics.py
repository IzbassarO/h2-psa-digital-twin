"""Frozen prereg-v1 section 7/8 metrics, evaluated after selected-point simulation.

Implementation declarations fixed before selected labels exist: sorted arm order;
linear NumPy percentiles; stable RQ order for equal Holm p-values; statsmodels'
cluster covariance with its finite-sample correction enabled. Primary exclusions
apply only to final failed runs; successful retries and capped runs are included.
Physical simulations shared across arms retain distinct arm memberships, with
stable first-seed ownership after duplicate (id, arm) memberships are removed.
No undefined fit/draw is replaced with another estimator or silently discarded.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version as package_version
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta, binom, binomtest, fisher_exact, norm
from statsmodels.stats.proportion import confint_proportions_2indep

RQ_ORDER = ("RQ1", "RQ2", "RQ3", "RQ5")
RECOVERY_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
ARM_FIELDS = ("arm", "surrogate", "n_train", "variant", "alpha")
PINNED_SOFTWARE = {"numpy": "2.5.3", "scipy": "1.18.1", "pandas": "3.0.6", "scikit-learn": "1.9.1",
                   "pymoo": "0.6.2", "statsmodels": "0.15.0"}
DECLARATIONS = {
    "percentiles": "numpy linear interpolation",
    "holm_ties": "stable RQ1, RQ2, RQ3, RQ5 order",
    "point_bootstrap_order": "sorted arm IDs, one default_rng(5)",
    "cluster_bootstrap_order": "RQ1, RQ2, RQ3, RQ5; one default_rng(6)",
    "logistic_covariance": "cluster robust, use_correction=True, asymptotic normal Wald",
    "undefined_bootstrap": "any undefined draw leaves percentile bounds undefined; count reported",
    "failure_policy": "exclude final failures primarily; sensitivity treats them as violations/miscoverage",
    "duplicates": "one ID per arm; stable first seed supplies cluster; shared IDs retain contributions across arms",
    "gate_scope": "all primary alpha=0.10 G/L arms; secondary/diagnostic gates reported independently",
    "bootstrap_RQ3_dependency": "descriptive status follows bootstrap RQ2 verdict",
}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not 0 <= k <= n:
        raise ValueError("Binomial counts must satisfy 0 <= k <= n")
    if n == 0:
        return np.nan, np.nan
    return (0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)))


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return np.nan if n == 0 else 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def newcombe_difference(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    if not n1 or not n2:
        return np.nan, np.nan, np.nan
    lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", compare="diff", alpha=0.05)
    return k1 / n1 - k2 / n2, float(lo), float(hi)


def _observed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = frame.loc[frame["status"] != "failed"].copy()
    selected["z"] = pd.to_numeric(selected["z"]).astype(float)
    selected.loc[selected["status"] == "censored", "z"] = 3.0
    if not np.isfinite(selected["z"]).all():
        raise ValueError("Non-failed labels must have finite clipped z")
    return selected


def _rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return dict(n=n, k=k, rate=k / n if n else np.nan, ci_lower=lo, ci_upper=hi)


def _alpha(frame: pd.DataFrame, value: float) -> np.ndarray:
    return np.isclose(pd.to_numeric(frame["alpha"]).to_numpy(dtype=float), value)


def _arm_info(frame: pd.DataFrame) -> dict:
    return {name: frame.iloc[0][name] for name in ARM_FIELDS}


def violation_summary(frame: pd.DataFrame) -> dict:
    observed = _observed(frame)
    failed = int(frame["status"].eq("failed").sum())
    k = int(observed["z"].gt(2.0).sum())
    return {**_rate(k, len(observed)), "failed": failed,
            **{f"sensitivity_{key}": value for key, value in _rate(k + failed, len(frame)).items()}}


def coverage_summary(frame: pd.DataFrame) -> dict:
    if not frame.empty and not np.isfinite(frame["bound"]).all():
        raise ValueError("Coverage bounds must be finite")
    observed = _observed(frame)
    covered = int(observed["z"].le(observed["bound"]).sum())
    failed = int(frame["status"].eq("failed").sum())
    return {**_rate(covered, len(observed)), "failed": failed,
            **{f"sensitivity_{key}": value for key, value in _rate(covered, len(frame)).items()}}


def choose_band(predictions: pd.DataFrame) -> dict:
    bounds = predictions["z_hat"] if predictions.iloc[0]["variant"] == "P" else predictions["bound"]
    lower = 1.5 if bounds.between(1.5, 2.0, inclusive="both").sum() >= 30 else 1.25
    return dict(lower=lower, upper=2.0, widened=lower < 1.5,
                n=int(bounds.between(lower, 2.0, inclusive="both").sum()))


def band_tables(test: pd.DataFrame, selected: pd.DataFrame, registered_bands: pd.DataFrame | None = None):
    rows, pooled = [], []
    selected_by_arm = {arm: group for arm, group in selected.groupby("arm", sort=True)}
    for arm, group in test.groupby("arm", sort=True):
        chosen = choose_band(group)
        if registered_bands is not None:
            supplied = registered_bands.loc[registered_bands["arm"] == arm]
            if len(supplied) != 1 or any(float(supplied.iloc[0][key]) != float(chosen[key])
                                         for key in ("lower", "upper", "n")):
                raise ValueError(f"{arm}: band report differs from the fixed rule")
        bounds = group["z_hat"] if group.iloc[0]["variant"] == "P" else group["bound"]
        band = group.loc[bounds.between(chosen["lower"], chosen["upper"], inclusive="both")]
        sel = selected_by_arm.get(arm, group.iloc[:0])
        selected_bounds = sel["z_hat"] if group.iloc[0]["variant"] == "P" else sel["bound"]
        inside = int(selected_bounds.between(chosen["lower"], chosen["upper"], inclusive="both").sum())
        bc, sc = coverage_summary(band), coverage_summary(sel)
        diff, lo, hi = newcombe_difference(bc["k"], bc["n"], sc["k"], sc["n"])
        rows.append({**_arm_info(group), **chosen, "n_band_observed": bc["n"], "band_covered": bc["k"],
                     "band_coverage": bc["rate"], "selected_n": sc["n"], "selected_covered": sc["k"],
                     "selected_coverage": sc["rate"], "selected_inside_n": inside,
                     "selected_inside_fraction": inside / len(sel) if len(sel) else np.nan,
                     "difference": diff, "difference_lower": lo, "difference_upper": hi,
                     "band_failed": bc["failed"], "selected_failed": sc["failed"],
                     "selected_simulated": len(sel),
                     "sensitivity_band_n": bc["sensitivity_n"],
                     "sensitivity_selected_n": sc["sensitivity_n"]})
    table = pd.DataFrame(rows)
    for variant, group in table.loc[np.isclose(table["alpha"], 0.10)].groupby("variant", sort=True):
        n1, k1 = int(group["n_band_observed"].sum()), int(group["band_covered"].sum())
        n2, k2 = int(group["selected_n"].sum()), int(group["selected_covered"].sum())
        diff, lo, hi = newcombe_difference(k1, n1, k2, n2)
        pooled.append(dict(variant=variant, n_band=n1, band_covered=k1, band_coverage=k1/n1 if n1 else np.nan,
                           n_selected=n2, selected_covered=k2, selected_coverage=k2/n2 if n2 else np.nan,
                           selected_inside_fraction=(group["selected_inside_n"].sum()/group["selected_simulated"].sum()
                                                     if group["selected_simulated"].sum() else np.nan),
                           difference=diff, difference_lower=lo, difference_upper=hi,
                           interpretation="undefined" if not np.isfinite(lo) else
                           "selection effect" if lo > 0 else "boundary effect"))
        n1s, n2s = int(group["sensitivity_band_n"].sum()), int(group["sensitivity_selected_n"].sum())
        ds, ls, us = newcombe_difference(k1, n1s, k2, n2s)
        pooled[-1].update(sensitivity_difference=ds, sensitivity_difference_lower=ls,
                          sensitivity_difference_upper=us)
    return table, pd.DataFrame(pooled)


def nondominated(points) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    if not len(points):
        return points
    points = points[np.lexsort((-points[:, 1], -points[:, 0]))]
    keep, best_productivity = [], -np.inf
    for point in points:
        if point[1] > best_productivity:
            keep.append(point)
            best_productivity = point[1]
    return np.asarray(keep)


def hypervolume_2d(points, reference=(-0.1, -0.1)) -> float:
    points = nondominated(points)
    points = points[(points > np.asarray(reference)).all(axis=1)]
    points = points[np.argsort(points[:, 0])] if len(points) else points
    previous, total = reference[0], 0.0
    for x, y in points:
        total += (x - previous) * (y - reference[1])
        previous = x
    return float(total)


def feasible_front(frame: pd.DataFrame) -> np.ndarray:
    observed = _observed(frame)
    usable = observed.loc[observed["status"].isin(("css", "cap")) & observed["z"].le(2.0)]
    return nondominated(usable[["recovery", "productivity"]].to_numpy())


def realised_front_coordinates(selected: pd.DataFrame) -> pd.DataFrame:
    """Export each arm's feasible nondominated objective coordinates."""
    rows = []
    for _, group in selected.groupby("arm", sort=True):
        info = _arm_info(group)
        rows.extend({**info, "recovery": float(recovery), "productivity": float(productivity)}
                    for recovery, productivity in feasible_front(group))
    return pd.DataFrame(rows, columns=[*ARM_FIELDS, "recovery", "productivity"])


def front_metrics(frame: pd.DataFrame, reference_front: np.ndarray) -> dict:
    """Recovery labels are percentages: their direct difference is percentage points."""
    front = feasible_front(frame)
    if frame.empty:
        return dict(status="undefined: no selected simulations", hv_ratio=np.nan, n_front=0,
                    **{f"loss_{level:g}": np.nan for level in RECOVERY_LEVELS})
    if not len(reference_front):
        return dict(status="undefined: empty reference", hv_ratio=np.nan,
                    **{f"loss_{level:g}": np.nan for level in RECOVERY_LEVELS})
    nadir, ideal = reference_front.min(axis=0), reference_front.max(axis=0)
    width = ideal - nadir
    degenerate = bool((width <= 0).any())
    hv = np.nan if degenerate else hypervolume_2d((front - nadir) / width)
    reference_hv = np.nan if degenerate else hypervolume_2d((reference_front - nadir) / width)
    result = dict(status="undefined HV: zero objective range" if degenerate else "defined",
                  hv_ratio=hv/reference_hv if not degenerate else np.nan, n_front=len(front))
    for level in RECOVERY_LEVELS:
        threshold = nadir[1] + level * width[1]
        reference_recovery = reference_front[reference_front[:, 1] >= threshold, 0].max()
        attained = front[front[:, 1] >= threshold, 0]
        result[f"productivity_{level:g}"] = threshold
        result[f"loss_{level:g}"] = float(reference_recovery - attained.max()) if len(attained) else np.nan
        result[f"not_reached_{level:g}"] = not len(attained)
    return result


def _percentiles(values, probabilities) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    undefined = int((~np.isfinite(values)).sum())
    return (np.full(len(probabilities), np.nan) if undefined or not len(values)
            else np.quantile(values, probabilities, method="linear")), undefined


def front_tables(selected: pd.DataFrame, nominal: pd.DataFrame, draws: int = 2000,
                 arm_metadata: pd.DataFrame | None = None):
    reference = feasible_front(nominal)
    rng, rows = np.random.default_rng(5), []
    groups = {arm: group for arm, group in selected.groupby("arm", sort=True)}
    metadata = (pd.DataFrame([_arm_info(group) for group in groups.values()]) if arm_metadata is None
                else arm_metadata[list(ARM_FIELDS)].drop_duplicates().sort_values("arm"))
    for _, info in metadata.iterrows():
        group = groups.get(info["arm"], selected.iloc[:0])
        observed = _observed(group)
        metrics = front_metrics(observed, reference)
        bootstrap = {key: [] for key in ("hv_ratio", *[f"loss_{x:g}" for x in RECOVERY_LEVELS])}
        for _ in range(draws):
            resampled = observed.iloc[rng.integers(len(observed), size=len(observed))] if len(observed) else observed
            boot = front_metrics(resampled, reference)
            for key in bootstrap:
                bootstrap[key].append(boot[key])
        result = {**info.to_dict(), **metrics, "bootstrap_draws": draws}
        for key, values in bootstrap.items():
            interval, undefined = _percentiles(values, (.025, .975))
            result.update({f"{key}_lower": interval[0], f"{key}_upper": interval[1],
                           f"{key}_undefined_draws": undefined})
        rows.append(result)
    table = pd.DataFrame(rows)
    if len(table):
        for i, row in table.iterrows():
            baseline = table.loc[(table["surrogate"] == row["surrogate"]) &
                                 (table["n_train"] == row["n_train"]) &
                                 (table["variant"] == "P") & np.isclose(table["alpha"], .10)]
            if len(baseline) == 1:
                for key in ("hv_ratio", *[f"loss_{x:g}" for x in RECOVERY_LEVELS]):
                    table.loc[i, f"{key}_minus_P"] = row[key] - baseline.iloc[0][key]
    return table, pd.DataFrame(reference, columns=["recovery", "productivity"])


def shift_regression_rows(shift: pd.DataFrame, disturbed: pd.DataFrame, failure_sensitivity=False) -> pd.DataFrame:
    frames = []
    for name, frame in (("D_shift", shift), ("S_dist", disturbed)):
        frame = frame.loc[frame["variant"].isin(("G", "L")) &
                          frame["n_train"].eq(512) & _alpha(frame, .10)]
        frame = frame.copy() if failure_sensitivity else _observed(frame).copy()
        frame["set"] = name
        frame["miscovered"] = frame["z"].gt(frame["bound"]).astype(int)
        if failure_sensitivity:
            frame.loc[frame["status"] == "failed", "miscovered"] = 1
        frame["cluster"] = ("D_shift:" + frame["id"].astype(str) if name == "D_shift" else
                            "S_dist:" + frame["arm"].astype(str) + ":" + frame["seed"].astype(str))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fit_shift_logistic(rows: pd.DataFrame, clustered: bool = True) -> dict:
    names = ("intercept_D_shift", "intercept_S_dist", "L", "GBM", "b_phi", "b_CH4", "b_CO")
    if rows.empty:
        return dict(status="undefined: no rows", p_value=np.nan, b_phi=np.nan, coefficients=[])
    design = np.column_stack((rows["set"].eq("D_shift"), rows["set"].eq("S_dist"),
                              rows["variant"].eq("L"), rows["surrogate"].eq("GBM"),
                              rows["u_phi"], rows["u_y_CH4"], rows["u_y_CO"])).astype(float)
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) < len(names):
        return dict(status="undefined: rank-deficient design", p_value=np.nan, b_phi=np.nan, coefficients=[])
    if clustered and rows["cluster"].nunique() < 2:
        return dict(status="undefined: fewer than two clusters", p_value=np.nan, b_phi=np.nan, coefficients=[])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.GLM(rows["miscovered"].to_numpy(), design, family=sm.families.Binomial())
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": rows["cluster"].to_numpy(),
                                                         "use_correction": True}, use_t=False) if clustered else model.fit()
        bad = [w for w in caught if w.category.__name__ in ("PerfectSeparationWarning", "ConvergenceWarning")]
        if bad or not fit.converged or not np.isfinite(fit.params).all():
            raise ValueError("separation or convergence failure")
        info = design.T @ (design * (fit.fittedvalues * (1-fit.fittedvalues))[:, None])
        if np.linalg.matrix_rank(info) < len(names):
            raise ValueError("singular fitted information (possible separation)")
        result = dict(status="defined", n=len(rows), n_clusters=rows["cluster"].nunique(),
                      b_phi=float(fit.params[4]), coefficients=[])
        if not clustered:
            return result
        if not np.isfinite(fit.bse).all() or (fit.bse <= 0).any():
            raise ValueError("undefined cluster covariance")
        for name, value, se in zip(names, fit.params, fit.bse):
            result["coefficients"].append(dict(term=name, estimate=float(value), se=float(se),
                                               lower=float(value-norm.ppf(.975)*se),
                                               upper=float(value+norm.ppf(.975)*se)))
        b, se = fit.params[4], fit.bse[4]
        result.update(p_value=float(norm.cdf(b/se)), b_phi_lower=float(b-norm.ppf(.975)*se),
                      b_phi_upper=float(b+norm.ppf(.975)*se), odds_ratio=float(np.exp(-b)),
                      odds_ratio_lower=float(np.exp(-(b+norm.ppf(.975)*se))),
                      odds_ratio_upper=float(np.exp(-(b-norm.ppf(.975)*se))))
        return result
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return dict(status=f"undefined: {exc}", p_value=np.nan, b_phi=np.nan, coefficients=[])


def mcnemar_exact(nominal: pd.DataFrame, disturbed: pd.DataFrame) -> dict:
    paired = disturbed.merge(nominal[["id", "arm", "status", "z"]].rename(columns={"id": "nominal_id"}),
                             on=["nominal_id", "arm"], how="left", indicator=True,
                             suffixes=("_dist", "_nom"), validate="many_to_one")
    if paired["_merge"].ne("both").any():
        raise ValueError("Every disturbed membership needs its nominal arm pair")
    bad = paired["status_dist"].eq("failed") | paired["status_nom"].eq("failed")

    def calculate(frame, sensitivity=False):
        nom = frame["z_nom"].gt(2.0) | (frame["status_nom"].eq("failed") if sensitivity else False)
        dist = frame["z_dist"].gt(2.0) | (frame["status_dist"].eq("failed") if sensitivity else False)
        increase, decrease = int((~nom & dist).sum()), int((nom & ~dist).sum())
        p = (np.nan if not len(frame) else
             float(binomtest(increase, increase+decrease, .5, alternative="greater").pvalue) if increase+decrease else 1.0)
        return dict(n=len(frame), nominal_violations=int(nom.sum()), disturbed_violations=int(dist.sum()),
                    increase=increase, decrease=decrease, p_value=p)
    return {**calculate(paired.loc[~bad]), "failed_pairs": int(bad.sum()),
            "sensitivity": calculate(paired, True)}


def holm(p_values: dict[str, float]) -> dict:
    ordered = sorted(RQ_ORDER, key=lambda key: (p_values[key] if np.isfinite(p_values[key]) else np.inf,
                                               RQ_ORDER.index(key)))
    result, previous = {}, 0.0
    for rank, rq in enumerate(ordered, 1):
        value = p_values[rq]
        adjusted = np.nan if not np.isfinite(value) else min(1.0, max(previous, (5-rank)*value))
        if np.isfinite(adjusted):
            previous = adjusted
        result[rq] = dict(p_value=value, adjusted_p=adjusted, rank=rank, level=.05/(5-rank))
    return result


def pooled_selected(selected: pd.DataFrame, pool_policy="per_arm") -> pd.DataFrame:
    if pool_policy != "per_arm":
        raise ValueError("The lead-fixed policy counts shared IDs once per arm")
    frame = selected.loc[_alpha(selected, .10)].sort_values(["arm", "seed", "id"], kind="stable")
    return frame.drop_duplicates(["arm", "id"])


def rq_tests(selected: pd.DataFrame, logistic: dict, gate_pass: bool, pool_policy="per_arm") -> dict:
    pooled = pooled_selected(selected, pool_policy)
    totals = {variant: violation_summary(group) for variant, group in pooled.groupby("variant")}
    for variant in ("P", "G", "L"):
        totals.setdefault(variant, violation_summary(pooled.iloc[:0]))
    p = {}
    for rq, variant in (("RQ1", "P"), ("RQ2", "G")):
        k, n = totals[variant]["k"], totals[variant]["n"]
        p[rq] = float(binomtest(k, n, .10, alternative="greater").pvalue) if n else np.nan
    g, l = totals["G"], totals["L"]
    pa = float(binomtest(l["k"], l["n"], .17, alternative="less").pvalue) if l["n"] else np.nan
    pb = float(fisher_exact([[g["k"], g["n"]-g["k"]], [l["k"], l["n"]-l["k"]]],
                           alternative="greater").pvalue) if g["n"] and l["n"] else np.nan
    p["RQ3"] = max(pa, pb) if np.isfinite([pa, pb]).all() else np.nan
    p["RQ5"] = logistic.get("p_value", np.nan)
    decisions = holm(p)
    decisions["RQ3"].update(p_a=pa, p_b=pb)
    for rq in RQ_ORDER:
        row = decisions[rq]
        row["verdict"] = ("undefined" if not np.isfinite(row["p_value"]) else
                          "confirmed" if row["adjusted_p"] < .05 else "not confirmed")
    if not gate_pass and decisions["RQ2"]["verdict"] == "confirmed":
        decisions["RQ2"]["verdict"] = "not confirmed: sanity gate failed"
    if decisions["RQ2"]["verdict"] != "confirmed":
        decisions["RQ3"]["verdict"] = "descriptive only"
    elif np.isfinite([pa, pb]).all() and ((pa < decisions["RQ3"]["level"]) != (pb < decisions["RQ3"]["level"])):
        decisions["RQ3"]["verdict"] = "partially"
    return dict(decisions=decisions, pooled_counts=totals, gate_pass=gate_pass, pool_policy=pool_policy)


def _cluster_groups(frame: pd.DataFrame, keys: list[str]) -> dict:
    return {key: group for key, group in frame.groupby(keys, sort=True, dropna=False)}


def _resample(groups: dict, keys: list, indices, template: pd.DataFrame) -> pd.DataFrame:
    rows = [groups[keys[i]] for i in indices if keys[i] in groups]
    return pd.concat(rows, ignore_index=True) if rows else template.iloc[:0]


def cluster_robustness(selected: pd.DataFrame, shift_rows: pd.DataFrame, tests: dict,
                       draws: int = 2000, pool_policy="per_arm") -> dict:
    rng, output = np.random.default_rng(6), {}
    frame = pooled_selected(selected, pool_policy)
    for rq, variant in (("RQ1", "P"), ("RQ2", "G")):
        source = frame.loc[frame["variant"] == variant]
        groups = _cluster_groups(source, ["arm", "seed"])
        keys, values = list(groups), []
        for _ in range(draws):
            sampled = _resample(groups, keys, rng.integers(len(keys), size=len(keys)), source) if keys else source
            values.append(violation_summary(sampled)["rate"])
        bound, undefined = _percentiles(values, (tests["decisions"][rq]["level"],))
        verdict = "undefined" if undefined or not keys else "confirmed" if bound[0] > .10 else "not confirmed"
        if rq == "RQ2" and not tests["gate_pass"] and verdict == "confirmed":
            verdict = "not confirmed: sanity gate failed"
        output[rq] = dict(lower=float(bound[0]), undefined_draws=undefined, n_clusters=len(keys), verdict=verdict)
    g = frame.loc[frame["variant"] == "G"]
    l = frame.loc[frame["variant"] == "L"]
    groups_g = _cluster_groups(g, ["surrogate", "n_train", "seed"])
    groups_l = _cluster_groups(l, ["surrogate", "n_train", "seed"])
    keys, rates_l, differences = sorted(set(groups_g) | set(groups_l)), [], []
    for _ in range(draws):
        indices = rng.integers(len(keys), size=len(keys)) if keys else []
        bg, bl = _resample(groups_g, keys, indices, g), _resample(groups_l, keys, indices, l)
        rg, rl = violation_summary(bg)["rate"], violation_summary(bl)["rate"]
        rates_l.append(rl)
        differences.append(rg-rl)
    level = tests["decisions"]["RQ3"]["level"]
    upper, missing_l = _percentiles(rates_l, (1-level,))
    lower, missing_difference = _percentiles(differences, (level,))
    components = (upper[0] < .17, lower[0] > 0)
    verdict = ("undefined" if missing_l or missing_difference or not keys else
               "confirmed" if all(components) else "partially" if any(components) else "not confirmed")
    if output["RQ2"]["verdict"] != "confirmed":
        verdict = "descriptive only"
    output["RQ3"] = dict(L_upper=float(upper[0]), difference_lower=float(lower[0]),
                         undefined_draws=max(missing_l, missing_difference), n_clusters=len(keys), verdict=verdict)
    groups_shift = _cluster_groups(shift_rows.loc[shift_rows["set"] == "D_shift"], ["id"])
    groups_dist = _cluster_groups(shift_rows.loc[shift_rows["set"] == "S_dist"], ["arm", "seed"])
    keys_shift, keys_dist, coefficients = list(groups_shift), list(groups_dist), []
    for _ in range(draws):
        bs = _resample(groups_shift, keys_shift, rng.integers(len(keys_shift), size=len(keys_shift)), shift_rows) if keys_shift else shift_rows.iloc[:0]
        bd = _resample(groups_dist, keys_dist, rng.integers(len(keys_dist), size=len(keys_dist)), shift_rows) if keys_dist else shift_rows.iloc[:0]
        coefficients.append(fit_shift_logistic(pd.concat([bs, bd], ignore_index=True), clustered=False)["b_phi"])
    upper, undefined = _percentiles(coefficients, (1-tests["decisions"]["RQ5"]["level"],))
    verdict = "undefined" if undefined else "confirmed" if upper[0] < 0 else "not confirmed"
    output["RQ5"] = dict(upper=float(upper[0]), undefined_draws=undefined,
                         n_shift_clusters=len(keys_shift), n_dist_clusters=len(keys_dist), verdict=verdict)
    for rq in RQ_ORDER:
        row = output[rq]
        row["draws"] = draws
        row["level"] = tests["decisions"][rq]["level"]
        row["primary_verdict"] = tests["decisions"][rq]["verdict"]
        row["robustness"] = ("undefined" if row["verdict"] == "undefined" else
                             "robust" if row["verdict"].split(":")[0] == row["primary_verdict"].split(":")[0]
                             else "not robust")
    return output


def join_predictions(labels: pd.DataFrame, predictions: pd.DataFrame, memberships: pd.DataFrame | None = None):
    if labels["id"].duplicated().any() or predictions.duplicated(["id", "arm"]).any():
        raise ValueError("Labels must be unique by id; predictions by (id, arm)")
    if memberships is None:
        expected_ids = set(labels["id"])
        if any(set(group["id"]) != expected_ids for _, group in predictions.groupby("arm")):
            raise ValueError("Every prediction arm must cover the complete simulated ID set")
        extras = predictions.drop(columns=[name for name in predictions if name != "id" and name in labels])
        result = labels.merge(extras, on="id", validate="one_to_many")
    else:
        membership = memberships.copy()
        if "counted_in_arm" in membership:
            flags = membership["counted_in_arm"].astype(str).str.lower()
            membership = membership.loc[flags.isin(("true", "1"))]
        membership = membership.sort_values(["arm", "seed", "id"], kind="stable").drop_duplicates(["id", "arm"])
        if not set(membership["id"]) <= set(labels["id"]):
            raise ValueError("Selected memberships reference missing simulated IDs")
        common = (set(membership) & set(labels)) - {"id"}
        lookup = labels.set_index("id")
        for name in common:
            left = membership[name].to_numpy()
            right = lookup.loc[membership["id"], name].to_numpy()
            if not ((left == right) | (pd.isna(left) & pd.isna(right))).all():
                raise ValueError(f"Membership/label disagreement for {name}")
        result = membership.merge(labels.drop(columns=list(common)), on="id", validate="many_to_one")
        extras = predictions[[name for name in predictions if name in ("id", "arm") or name not in result]]
        result = result.merge(extras, on=["id", "arm"], how="left", validate="one_to_one")
    if (result.empty and memberships is None) or not set(predictions["id"]) <= set(labels["id"]):
        raise ValueError("Predictions do not match the simulation IDs")
    if result["bound"].isna().any():
        raise ValueError("Missing predictions for selected memberships")
    return result


def analyse_frames(labels: dict[str, pd.DataFrame], predictions: dict[str, pd.DataFrame],
                   memberships: dict[str, pd.DataFrame], gate: pd.DataFrame, bands: pd.DataFrame,
                   draws: int = 2000, pool_policy="per_arm") -> dict:
    expected_primary = {(surrogate, n, variant) for surrogate in ("GP", "GBM")
                        for n in (128, 256, 512) for variant in ("P", "G", "L")}
    for name in ("D_test", "D_shift"):
        metadata = predictions[name][list(ARM_FIELDS)].drop_duplicates()
        if metadata["arm"].duplicated().any():
            raise ValueError(f"{name}: inconsistent arm metadata")
        for _, row in metadata.iterrows():
            if (row["surrogate"] not in ("GP", "GBM") or row["n_train"] not in (128, 256, 512)
                    or row["variant"] not in ("P", "G", "L") or not any(np.isclose(row["alpha"], a) for a in (.05, .10))
                    or row["arm"] != f"{row['surrogate']}_n{int(row['n_train'])}_{row['variant']}_a{row['alpha']:.2f}"):
                raise ValueError(f"{name}: unregistered arm metadata")
        present = {(row["surrogate"], row["n_train"], row["variant"])
                   for _, row in metadata.loc[np.isclose(metadata["alpha"], .10)].iterrows()}
        required_arms = (expected_primary if name == "D_test" else
                         {(surrogate, 512, variant) for surrogate in ("GP", "GBM") for variant in ("G", "L")})
        if not required_arms <= present:
            raise ValueError(f"{name}: missing registered primary prediction arms")
    frames = {name: join_predictions(labels[name], predictions[name], memberships.get(name))
              for name in ("D_test", "D_shift", "S_nom", "S_dist")}
    violations, coverage, pooled = [], [], []
    test_metadata = predictions["D_test"][list(ARM_FIELDS)].drop_duplicates().sort_values("arm")
    registered_nominal = test_metadata.loc[np.isclose(test_metadata["alpha"], .10) |
                                          (test_metadata["n_train"].eq(512) & test_metadata["variant"].isin(("G", "L")))]
    for name in ("S_nom", "S_dist"):
        groups = {arm: group for arm, group in frames[name].groupby("arm", sort=True)}
        expected = (registered_nominal if name == "S_nom" else
                    registered_nominal.loc[registered_nominal["n_train"].eq(512) & np.isclose(registered_nominal["alpha"], .10)])
        if not set(groups) <= set(expected["arm"]):
            raise ValueError(f"{name}: memberships contain unregistered optimization arms")
        for _, info in expected.iterrows():
            group = groups.get(info["arm"], frames[name].iloc[:0])
            requested = 30 if info["variant"] == "P" else 45
            violations.append({"set": name, **info.to_dict(), **violation_summary(group),
                               "requested_n": requested, "simulated_n": len(group),
                               "selection_shortfall": max(requested-len(group), 0)})
    for name, frame in frames.items():
        groups = {arm: group for arm, group in frame.groupby("arm", sort=True)}
        metadata = (registered_nominal if name == "S_nom" else
                    registered_nominal.loc[registered_nominal["n_train"].eq(512) & _alpha(registered_nominal, .10)]
                    if name == "S_dist" else predictions[name][list(ARM_FIELDS)].drop_duplicates().sort_values("arm"))
        for _, info in metadata.iterrows():
            group = groups.get(info["arm"], frame.iloc[:0])
            coverage.append({"set": name, **info.to_dict(), **coverage_summary(group)})
    primary_selected = pooled_selected(frames["S_nom"], pool_policy)
    for variant in ("P", "G", "L"):
        pooled.append({"variant": variant, **violation_summary(primary_selected.loc[primary_selected["variant"] == variant])})
    band, pooled_band = band_tables(frames["D_test"], frames["S_nom"], bands)
    nominal = pd.concat([labels[name] for name in ("D_train", "D_cal", "D_test", "S_nom")], ignore_index=True)
    fronts, reference = front_tables(frames["S_nom"], nominal, draws, registered_nominal)
    shift_rows = shift_regression_rows(frames["D_shift"], frames["S_dist"])
    logistic = fit_shift_logistic(shift_rows)
    sensitivity_logistic = fit_shift_logistic(shift_regression_rows(frames["D_shift"], frames["S_dist"], True))
    tested_gate = []
    for arm, group in frames["D_test"].groupby("arm", sort=True):
        if group.iloc[0]["variant"] == "P":
            continue
        result = coverage_summary(group)
        upper = cp_upper(result["k"], result["n"])
        passed = bool(upper >= 1-float(group.iloc[0]["alpha"]))
        registered = gate.loc[gate["arm"] == arm]
        if len(registered) != 1 or str(registered.iloc[0]["gate_pass"]).lower() not in (str(passed).lower(), str(int(passed))):
            raise ValueError(f"{arm}: supplied sanity gate disagrees with the registered calculation")
        sensitivity_upper = cp_upper(result["k"], result["sensitivity_n"])
        tested_gate.append({**_arm_info(group), "gate_pass": passed, "upper": upper,
                            "sensitivity_upper": sensitivity_upper,
                            "sensitivity_gate_pass": bool(sensitivity_upper >= 1-float(group.iloc[0]["alpha"]))})
    primary_gate = [row for row in tested_gate if np.isclose(row["alpha"], .10)]
    if not primary_gate:
        raise ValueError("Primary G/L sanity gates are required")
    gate_pass = all(row["gate_pass"] for row in primary_gate)
    tests = rq_tests(frames["S_nom"], logistic, gate_pass, pool_policy)
    sensitivity_selected = frames["S_nom"].copy()
    failed_selected = sensitivity_selected["status"] == "failed"
    sensitivity_selected.loc[failed_selected, "status"] = "css"
    sensitivity_selected.loc[failed_selected, "z"] = 3.0
    sensitivity_gate_pass = all(row["sensitivity_gate_pass"] for row in primary_gate)
    sensitivity_tests = rq_tests(sensitivity_selected, sensitivity_logistic, sensitivity_gate_pass, pool_policy)
    paired_nominal = frames["S_nom"].loc[frames["S_nom"]["n_train"].eq(512) & _alpha(frames["S_nom"], .10)]
    paired_disturbed = frames["S_dist"].loc[_alpha(frames["S_dist"], .10)]
    status_rows = []
    failed_ids = []
    for name, source in labels.items():
        status_rows.append({"set": name, "n": len(source),
                            **{status: int(source["status"].eq(status).sum()) for status in ("css", "censored", "cap", "failed")},
                            "successful_retries": int((source.get("attempts", pd.Series(1, index=source.index)).gt(1) &
                                                        source["status"].ne("failed")).sum())})
        failed_ids.extend({"set": name, "id": value} for value in source.loc[source["status"] == "failed", "id"])
    return dict(tables={"violations": pd.DataFrame(violations), "pooled_violations": pd.DataFrame(pooled),
                        "coverage": pd.DataFrame(coverage), "bands": band, "pooled_bands": pooled_band,
                        "fronts": fronts, "reference_front": reference, "gate": pd.DataFrame(tested_gate),
                        "realised_fronts": realised_front_coordinates(frames["S_nom"]),
                        "logistic_coefficients": pd.DataFrame(logistic["coefficients"]),
                        "power": rq3_power(), "statuses": pd.DataFrame(status_rows),
                        "failed_ids": pd.DataFrame(failed_ids, columns=["set", "id"])},
                results={"declarations": {**DECLARATIONS, "pool_policy": pool_policy, "draws": draws},
                         "tests": tests, "failure_sensitivity_tests": sensitivity_tests,
                         "logistic": logistic, "failure_sensitivity_logistic": sensitivity_logistic,
                         "mcnemar": mcnemar_exact(paired_nominal, paired_disturbed),
                         "robustness": cluster_robustness(frames["S_nom"], shift_rows, tests, draws, pool_policy)})


def verify_frozen_manifest(generated_dir: Path, predictions_dir: Path, report_dir: Path,
                           require_manifest: bool = False) -> dict | None:
    """Verify committed frozen inputs/predictions independently of model files."""
    path = report_dir / "provenance.json"
    if not path.exists():
        if require_manifest:
            raise FileNotFoundError(f"Frozen analysis provenance is required: {path}")
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("prereg_tag") != "prereg-v1":
        raise ValueError("Analysis provenance must identify prereg-v1")
    if require_manifest:
        actual = {name: package_version(name) for name in PINNED_SOFTWARE}
        if (manifest.get("python") != "3.14.7" or manifest.get("libraries") != PINNED_SOFTWARE
                or platform.python_version() != "3.14.7" or actual != PINNED_SOFTWARE):
            raise ValueError("Analysis environment or manifest differs from the prereg-v1 pinned versions")
    expected = {
        "input_sha256": ({"D_train", "D_cal", "D_test", "D_shift"}, generated_dir, True),
        "selected_sha256": ({"S_nom_points", "S_dist_points", "S_nom_memberships", "S_dist_memberships"}, generated_dir, True),
        "prediction_sha256": ({"D_test.csv", "D_shift.csv", "S_nom.csv", "S_dist.csv"}, predictions_dir, False),
    }
    frozen_files = []
    for field, (keys, directory, append_extension) in expected.items():
        hashes = manifest.get(field)
        if not isinstance(hashes, dict) or set(hashes) != keys:
            raise ValueError(f"Incomplete frozen manifest field: {field}")
        for name in sorted(keys):
            frozen_files.append((directory / (name + ".csv" if append_extension else name), hashes[name]))
    if not isinstance(manifest.get("q_sha256"), str):
        raise ValueError("Missing frozen conformal quantile hash")
    frozen_files.append((report_dir / "q.csv", manifest["q_sha256"]))
    for filename, expected_hash in frozen_files:
        if not filename.exists():
            raise FileNotFoundError(f"Frozen artifact is missing: {filename}")
        digest = hashlib.sha256()
        with filename.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Frozen artifact hash mismatch: {filename}")
    return manifest


def analyse_generated(generated_dir: Path, predictions_dir: Path, report_dir: Path,
                      draws: int = 2000, pool_policy="per_arm", require_manifest: bool = False) -> dict:
    verify_frozen_manifest(generated_dir, predictions_dir, report_dir, require_manifest)
    required = [generated_dir/f"{name}.csv" for name in ("D_train", "D_cal", "D_test", "D_shift", "S_nom", "S_dist")]
    required += [generated_dir/f"{name}_memberships.csv" for name in ("S_nom", "S_dist")]
    required += [predictions_dir/f"{name}.csv" for name in ("D_test", "D_shift", "S_nom", "S_dist")]
    required += [report_dir/"gate.csv", report_dir/"band.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Selected simulations and frozen reports are required before analysis: " + ", ".join(missing))
    labels = {name: pd.read_csv(generated_dir/f"{name}.csv", float_precision="round_trip")
              for name in ("D_train", "D_cal", "D_test", "D_shift", "S_nom", "S_dist")}
    predictions = {name: pd.read_csv(predictions_dir/f"{name}.csv", float_precision="round_trip")
                   for name in ("D_test", "D_shift", "S_nom", "S_dist")}
    memberships = {name: pd.read_csv(generated_dir/f"{name}_memberships.csv", float_precision="round_trip")
                   for name in ("S_nom", "S_dist")}
    return analyse_frames(labels, predictions, memberships, pd.read_csv(report_dir/"gate.csv"),
                          pd.read_csv(report_dir/"band.csv"), draws, pool_policy)


def rq3_power() -> pd.DataFrame:
    rows = []
    for n, threshold, levels in ((270, .17, (.05, .025, .0125)), (180, .15, (.025,))):
        for level in levels:
            candidates = np.arange(n+1)
            region = candidates[binom.cdf(candidates, n, threshold) < level]
            cutoff = int(region.max()) if len(region) else -1
            rows.append(dict(n=n, null_rate=threshold, level=level, rejection_max=cutoff,
                             **{f"power_{rate:g}": float(binom.cdf(cutoff, n, rate))
                                for rate in (.05, .08, .10)}))
    return pd.DataFrame(rows)
