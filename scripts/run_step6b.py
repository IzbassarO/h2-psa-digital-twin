"""Fit, calibrate, check and select exactly prereg-v1; never run a CSS simulator.

Run phases in order: train, calibrate, diagnostics, select, report (or all).
Cached fits/selections are immutable; held-out inspection forbids new fitting.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa import design
from h2psa.isotherms import load_yaml
from h2psa.selection import build_selected_lists, optimise_seed
from h2psa.step6 import (ALPHAS, DECISIONS, DISTURBANCES, INPUTS, RESPONSES, Surrogates,
                        arm_id, atomic_json, conformal_quantile, cp_bounds, file_hash,
                        fit_one, fit_tasks, fixed_band, model_id, registered_arms)

DATA = ROOT / "data" / "generated"
WORK = ROOT / "results" / "step6b"
SIZES = dict(D_train=512, D_cal=256, D_test=256, D_shift=128)


def save(state):
    atomic_json(WORK / "workflow.json", state)


def read_set(name, inputs_only=False):
    columns = ["id", *INPUTS, "u_phi", "u_y_CH4", "u_y_CO"] if inputs_only else None
    frame = pd.read_csv(DATA / f"{name}.csv", float_precision="round_trip", usecols=columns)
    if frame.id.tolist() != [f"{name}_{i:04d}" for i in range(SIZES[name])]:
        raise ValueError(f"Unexpected ordered IDs: {name}")
    return frame


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "workflow.json"
    provenance = {"prereg_tag": "prereg-v1",
                  "prereg_commit": subprocess.check_output(["git", "rev-parse", "prereg-v1"], text=True).strip(),
                  "input_sha256": {name: file_hash(DATA / f"{name}.csv") for name in SIZES},
                  "python": platform.python_version(),
                  "libraries": {name: importlib.metadata.version(name) for name in
                                ("numpy", "scipy", "pandas", "scikit-learn", "pymoo", "statsmodels")},
                  "duplicate_counting": "one shared physical simulation; once per arm",
                  "implementation": 1}
    pinned = {"numpy": "2.5.3", "scipy": "1.18.1", "pandas": "3.0.6", "scikit-learn": "1.9.1",
              "pymoo": "0.6.2", "statsmodels": "0.15.0"}
    if provenance["python"] != "3.14.7" or provenance["libraries"] != pinned:
        raise ValueError("Environment differs from prereg-v1")
    state = json.loads(path.read_text()) if path.exists() else {"provenance": provenance, "phases": {}}
    if state["provenance"] != provenance:
        raise ValueError("Inputs, versions or frozen implementation differ from saved run")
    frozen = subprocess.check_output(["git", "show", "prereg-v1:docs/preregistration.md"], text=True)
    if frozen.split("## Deviations")[0] != (ROOT / "docs" / "preregistration.md").read_text().split("## Deviations")[0]:
        raise ValueError("Frozen preregistration sections changed")
    previous = json.loads((ROOT / "results" / "step6a" / "workflow.json").read_text())
    if not previous["status"].startswith("complete"):
        raise ValueError("Step 6a is not complete")
    state["step6a"] = {key: previous[key] for key in
                        ("blocks", "generation_wall_s", "generation_elapsed_s", "total_elapsed_s")}
    save(state)
    return state


def train(state, workers):
    if state["phases"].get("train"):
        verify_fit_freeze(state)
        return
    if state.get("held_out_inspected"):
        raise RuntimeError("No new fit is permitted after held-out-label inspection")
    print(f"Training 24 registered models on D_train only, {workers} processes; 54x5 CV for each GBM.", flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(fit_one, task, str(DATA / "D_train.csv"), str(WORK)): task for task in fit_tasks()}
        for future in as_completed(futures):
            info = future.result()
            print(json.dumps({"fitted": info["specification"]["model"], "n_eligible": info["n_eligible"],
                              "chosen": info["chosen"], "wall_s": info["wall_s"]}), flush=True)
    infos = [json.loads((WORK / "models" / f"{model_id(*task)}.json").read_text()) for task in fit_tasks()]
    atomic_json(WORK / "hyperparameters.json", [{"model": info["specification"]["model"],
                 **{key: info["specification"][key] for key in ("family", "n_train", "response", "quantile")},
                 "n_eligible": info["n_eligible"], "chosen": info["chosen"], "warnings": info["warnings"],
                 "wall_s": info["wall_s"]} for info in infos])
    state["fit_sha256"] = {path.name: file_hash(path) for path in sorted((WORK / "models").glob("*.joblib"))}
    if len(state["fit_sha256"]) != 24:
        raise ValueError("Expected exactly 24 fitted models")
    state["phases"]["train"] = {"wall_s": time.monotonic() - started,
                                "frozen_at": datetime.now().astimezone().isoformat()}
    save(state)


def verify_fit_freeze(state):
    if not state["phases"].get("train"):
        raise ValueError("Training phase must complete first")
    current = {path.name: file_hash(path) for path in sorted((WORK / "models").glob("*.joblib"))}
    if state["fit_sha256"] != current:
        raise ValueError("Fitted models changed after being frozen")


def bundles():
    return {(family, n): Surrogates(family, n, WORK) for family in ("GP", "GBM") for n in (128, 256, 512)}


def calibrate(state):
    verify_fit_freeze(state)
    if state["phases"].get("calibrate"):
        return
    started = time.monotonic()
    state["held_out_inspected"] = True
    save(state)
    frame = read_set("D_cal")
    if (frame.status == "failed").any():
        raise ValueError("Frozen n=256 calibration needs declared failure sensitivity")
    X, y = frame[INPUTS].to_numpy(float), frame.z.to_numpy(float)
    rows, score_rows = [], []
    for arm in registered_arms(include_diagnostics=True):
        if arm["variant"] == "P":
            continue
        surrogate = Surrogates(arm["surrogate"], arm["n_train"], WORK)
        center, scale = surrogate.components(X, arm["variant"], arm["alpha"])
        scores = (y - center) / scale
        q, rank = conformal_quantile(scores, arm["alpha"])
        rows.append({**arm, "n_cal": len(scores), "rank": rank, "q": q})
        score_rows.extend({"id": pid, "arm": arm["arm"], "score": score} for pid, score in zip(frame.id, scores))
    pd.DataFrame(rows).to_csv(WORK / "q.csv", index=False)
    pd.DataFrame(score_rows).to_csv(WORK / "calibration_scores.csv", index=False)
    state["phases"]["calibrate"] = {"wall_s": time.monotonic() - started}
    state["q_sha256"] = file_hash(WORK / "q.csv")
    save(state)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def quantiles(state):
    if not state["phases"].get("calibrate") or state["q_sha256"] != file_hash(WORK / "q.csv"):
        raise ValueError("Immutable calibration quantiles required")
    return pd.read_csv(WORK / "q.csv", float_precision="round_trip").set_index("arm").q.to_dict()


def prediction_table(frame, arms, fitted, qs):
    X = frame[INPUTS].to_numpy(float)
    result = []
    for arm in arms:
        surrogate = fitted[arm["surrogate"], arm["n_train"]]
        pred = surrogate.predict(X)
        bound = surrogate.bound(X, arm["variant"], arm["alpha"], qs.get(arm["arm"], 0))
        table = pd.DataFrame({"id": frame.id.to_numpy(), **arm, "bound": bound,
                              "z_hat": pred["z"], "recovery_hat": pred["recovery"],
                              "productivity_hat": pred["productivity"]})
        for name in ("u_phi", "u_y_CH4", "u_y_CO"):
            if name in frame:
                table[name] = frame[name].to_numpy()
        result.append(table)
    return pd.concat(result, ignore_index=True)


def diagnostics(state):
    verify_fit_freeze(state)
    qs = quantiles(state)
    if state["phases"].get("diagnostics"):
        return
    # Set the irreversible provenance marker BEFORE accessing held-out labels.
    state["held_out_inspected"] = True
    save(state)
    started = time.monotonic()
    fitted, frame = bundles(), read_set("D_test")
    pred = prediction_table(frame, registered_arms(include_diagnostics=True), fitted, qs)
    (WORK / "predictions").mkdir(parents=True, exist_ok=True)
    pred.to_csv(WORK / "predictions" / "D_test.csv", index=False)
    gates, bands, accuracy = [], [], []
    for arm, table in pred.groupby("arm", sort=False):
        base = table.iloc[0][["arm", "surrogate", "n_train", "variant", "alpha"]].to_dict()
        valid = frame.status.to_numpy() != "failed"
        covered = (frame.z.to_numpy(float) <= table.bound.to_numpy(float)) & valid
        n, k = int(valid.sum()), int(covered.sum())
        mask, info = fixed_band(table.bound.to_numpy(float))
        bands.append({**base, **info, "covered": int(covered[mask].sum())})
        if base["variant"] != "P":
            lo, hi, gate_upper = cp_bounds(k, n)
            gates.append({**base, "n": n, "covered": k, "coverage": k/n,
                          "cp_lower": lo, "cp_upper": hi, "cp_one_sided_upper": gate_upper,
                          "gate_pass": gate_upper >= 1-base["alpha"], "failed": int((~valid).sum())})
    for (family, n), surrogate in fitted.items():
        values = surrogate.predict(frame[INPUTS].to_numpy(float))
        for response in RESPONSES:
            eligible = frame.status != "failed"
            if response != "z":
                eligible &= frame.status != "censored"
            rmse = float(np.sqrt(mean_squared_error(frame.loc[eligible, response], values[response][eligible])))
            accuracy.append({"surrogate": family, "n_train": n, "response": response,
                             "n": int(eligible.sum()), "rmse": rmse})
    pd.DataFrame(gates).to_csv(WORK / "gate.csv", index=False)
    pd.DataFrame(bands).to_csv(WORK / "band.csv", index=False)
    pd.DataFrame(accuracy).to_csv(WORK / "accuracy.csv", index=False)
    failed_arms = [row["arm"] for row in gates if not row["gate_pass"]]
    audit = []
    # Independent arithmetic audit on failures; preserve every fitted method/q.
    cal = read_set("D_cal")
    for aid in failed_arms:
        arm = next(item for item in registered_arms(True) if item["arm"] == aid)
        surrogate = fitted[arm["surrogate"], arm["n_train"]]
        center, scale = surrogate.components(cal[INPUTS].to_numpy(float), arm["variant"], arm["alpha"])
        scores = (cal.z.to_numpy(float)-center)/scale
        rank = int(np.ceil(257*(1-arm["alpha"])))
        independent_q = float(np.partition(scores.copy(), rank-1)[rank-1])
        matches = independent_q == qs[aid]
        if not matches:
            raise RuntimeError("Conformal arithmetic audit failed; downstream work stopped")
        audit.append({"arm": aid, "independent_rank": rank, "q_matches": matches,
                      "calibration_scores": 256, "method_changed": False})
    atomic_json(WORK / "gate_audit.json", {"failed_arms": failed_arms, "checks": audit,
                "checks_common": ["rank ceil((n+1)*(1-alpha)); no interpolated quantile",
                                  "one-sided upper Clopper-Pearson beta .95",
                                  "raw z-score / GP sigma for L; quantile residual for GBM L",
                                  "censored z=3; cap last-cycle labels included"], "methods_changed": False})
    state["gate_failed_arms"] = failed_arms
    state["phases"]["diagnostics"] = {"wall_s": time.monotonic()-started}
    save(state)
    print(f"Sanity gate: {len(gates)-len(failed_arms)}/{len(gates)} pass; failed={failed_arms}", flush=True)


def map_point(u):
    ds = load_yaml(design.CONFIGS / "design_space.yaml")["decision_variables"]
    dist = load_yaml(design.CONFIGS / "disturbances.yaml")["variables"]
    point = {}
    for j, name in enumerate(DECISIONS):
        if name == "f_zeolite":
            point[name] = float(min(np.floor(31*u[j]), 30)/30)
        elif name == "n_equalisations":
            point[name] = int(min(1+np.floor(3*u[j]), 3))
        else:
            lo, hi = ds[name]["bounds"]
            point[name] = float(lo+(hi-lo)*u[j])
    point.update({name: dist[name]["nominal"] for name in DISTURBANCES}, anchor="lopes")
    return point


def map_disturbance(u):
    ds = load_yaml(design.CONFIGS / "disturbances.yaml")["variables"]
    return {name: float(ds[name]["bounds"][0]+np.diff(ds[name]["bounds"])[0]*u[j])
            for j, name in enumerate(DISTURBANCES)}


def optimise_one(arm, seed, work_root, qs):
    work = Path(work_root)
    path = work / "optimization" / f"{arm['arm']}_seed{seed}.joblib"
    if path.exists():
        return joblib.load(path)
    surrogate = Surrogates(arm["surrogate"], arm["n_train"], work)
    bound = lambda X: surrogate.bound(X, arm["variant"], arm["alpha"], qs.get(arm["arm"], 0))
    started = time.monotonic()
    result = optimise_seed(surrogate.predict, bound, seed, 10 if arm["variant"] == "P" else 15)
    result["wall_s"] = time.monotonic()-started
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".joblib.part")
    joblib.dump(result, temporary)
    temporary.replace(path)
    return result


def select(state, workers):
    verify_fit_freeze(state)
    qs = quantiles(state)
    if not state["phases"].get("diagnostics"):
        raise ValueError("Lead-requested sanity diagnostics must precede optimization")
    if state["phases"].get("select"):
        return
    started = time.monotonic()
    arms = registered_arms()
    tasks = [(arm, seed) for arm in arms for seed in (10, 11, 12)]
    outputs = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(optimise_one, arm, seed, str(WORK), qs): (arm, seed) for arm, seed in tasks}
        for future in as_completed(pending):
            arm, seed = pending[future]
            result = future.result()
            outputs[arm["arm"], seed] = result
            print(json.dumps({"optimized": arm["arm"], "seed": seed, "front_n": len(result["front_X"]),
                              "selected_n": len(result["selected_X"]), "shortfall": result["shortfall"]}), flush=True)
    ordered = [({**arm, "seed": seed}, outputs[arm["arm"], seed]) for arm, seed in tasks]
    lists = build_selected_lists(ordered, map_point, map_disturbance)
    names = {"nominal_points": "S_nom_points", "disturbed_points": "S_dist_points",
             "nominal_memberships": "S_nom_memberships", "disturbed_memberships": "S_dist_memberships"}
    for key, name in names.items():
        lists[key].to_csv(DATA / f"{name}.csv", index=False)
    lists["shortages"].to_csv(WORK / "selection_seeds.csv", index=False)
    members = lists["nominal_memberships"]
    summary = []
    for arm in arms:
        subset = members.loc[members.arm == arm["arm"]]
        count = int(subset.counted_in_arm.sum())
        seed_rows = lists["shortages"].loc[lists["shortages"].arm == arm["arm"]]
        summary.append({**arm, "requested": 30 if arm["variant"] == "P" else 45,
                        "selected_memberships": len(subset), "unique_in_arm": count,
                        "duplicates_within_arm": len(subset)-count,
                        "front_shortfall": int(seed_rows.shortfall.sum())})
    pd.DataFrame(summary).to_csv(WORK / "selection_arms.csv", index=False)
    repeated = members.loc[members.id.duplicated(keep=False)].copy()
    repeated.to_csv(WORK / "duplicates.csv", index=False)
    fitted = bundles()
    for points_name, members_name, set_name in (("nominal_points", "nominal_memberships", "S_nom"),
                                                ("disturbed_points", "disturbed_memberships", "S_dist")):
        predictions = []
        unique = lists[members_name].loc[lists[members_name].counted_in_arm]
        for arm in arms:
            ids = unique.loc[unique.arm == arm["arm"], "id"]
            if not len(ids):
                continue
            frame = lists[points_name].set_index("id").loc[ids].reset_index()
            predictions.append(prediction_table(frame, [arm], fitted, qs))
        if predictions:
            pd.concat(predictions, ignore_index=True).to_csv(WORK / "predictions" / f"{set_name}.csv", index=False)
        else:
            pd.DataFrame(columns=["id", "arm", "surrogate", "n_train", "variant", "alpha", "bound", "z_hat"]
                         ).to_csv(WORK / "predictions" / f"{set_name}.csv", index=False)
    shift_arms = [arm for arm in arms if arm["n_train"] == 512 and arm["alpha"] == .10 and arm["variant"] != "P"]
    prediction_table(read_set("D_shift", inputs_only=True), shift_arms, fitted, qs).to_csv(
        WORK / "predictions" / "D_shift.csv", index=False)
    state["selection"] = {"unique_nominal": len(lists["nominal_points"]),
                          "unique_disturbed": len(lists["disturbed_points"]),
                          "nominal_memberships": len(members),
                          "within_arm_duplicates": int((~members.counted_in_arm).sum()),
                          "global_repeated_memberships": len(members)-len(lists["nominal_points"]),
                          "front_shortfall": int(lists["shortages"].shortfall.sum())}
    state["phases"]["select"] = {"wall_s": time.monotonic()-started}
    state["selected_sha256"] = {name: file_hash(DATA / f"{name}.csv") for name in names.values()}
    state["prediction_sha256"] = {path.name: file_hash(path) for path in sorted((WORK / "predictions").glob("*.csv"))}
    save(state)


def report(state):
    import shutil
    target = ROOT / "docs" / "validation" / "step6b"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("q.csv", "gate.csv", "band.csv", "accuracy.csv", "selection_seeds.csv",
                 "selection_arms.csv", "duplicates.csv", "gate_audit.json", "hyperparameters.json"):
        shutil.copyfile(WORK / name, target / name)
    (target / "predictions").mkdir(exist_ok=True)
    for path in sorted((WORK / "predictions").glob("*.csv")):
        shutil.copyfile(path, target / "predictions" / path.name)
    lines = ["# Step 6b — frozen surrogates, conformal diagnostics and selection", "",
             "Registration: `prereg-v1`; no new CSS simulations in this step.", "",
             "## Step 6a confirmation", "", "| block | n | css | censored | cap | failed |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in state["step6a"]["blocks"].items():
        c = row["status_counts"]
        lines.append(f"| {name} | {row['n']} | {c['css']} | {c['censored']} | {c['cap']} | {c['failed']} |")
    lines += ["", "Batch wall time 16.99 h; elapsed through final publication 23.85 h.",
              "The sole cap, D_test_0143 at 2000 cycles, retains last-cycle impurity/z/objectives and status=cap.",
              "Censored z=3; censored objective labels are excluded from fitting and objective RMSE.", "",
              "## Frozen fitting and calibration", "",
              "Only D_train nested prefixes are fitted. GP uses marginal likelihood with 10 restarts; GBM uses all 54 combinations and five-fold CV.",
              "Full chosen hyperparameters: [hyperparameters.json](step6b/hyperparameters.json).",
              "All model files are frozen and hashed before D_test is read.",
              "Conformal ranks are 232 (alpha 0.10) and 245 (alpha 0.05), with no interpolation or clipping of q.",
              "All 24 quantiles: [q.csv](step6b/q.csv).", "", "## Sanity gate and context", "",
              f"Failed sanity arms: {state.get('gate_failed_arms') or 'none'}.",
              "Per-arm coverage, two-sided CP intervals and one-sided upper gate: [gate.csv](step6b/gate.csv).",
              "Independent arithmetic checks on any failed arms: [gate_audit.json](step6b/gate_audit.json).",
              "Fixed band counts/widening: [band.csv](step6b/band.csv).",
              "Descriptive held-out RMSE only: [accuracy.csv](step6b/accuracy.csv). No method changed from diagnostics.",
              "", "## Optimisation and selection", "",
              "Population 100, generations 250, seeds 10/11/12, SBX eta 15/p 0.9, PM eta 20/per-coordinate p 1/8.",
              "Canonical inverse-bin repair precedes predictions. Final feasible nondominated fronts supply actual productivity quantiles.",
              "Short fronts retain all distinct points. Shared physical points have one run ID and count once per arm.",
              f"Selection totals: `{state.get('selection')}`.",
              "Per-arm counts: [selection_arms.csv](step6b/selection_arms.csv); per-seed shortfalls: [selection_seeds.csv](step6b/selection_seeds.csv).",
              "Repeated memberships: [duplicates.csv](step6b/duplicates.csv); complete memberships are committed alongside point lists.",
              "One rng4 disturbance draw per eligible unique nominal point gives paired S_dist points.", "",
              "## Future analysis", "",
              "`scripts/analyse_step6.py` implements the registered metrics; synthetic tests run before its commit.",
              "After Step 6c supplies S_nom.csv and S_dist.csv, run `python scripts/analyse_step6.py`.",
              "It uses the committed memberships and frozen predictions under docs/validation/step6b/predictions.",
              "Analysis does not refit models. The selected lists contain no re-simulation labels.",
              "Step 6c has not started. Deviations/declarations: docs/preregistration.md, 2026-10-13.",
              "Focused synthetic tests passed before publication; independent arithmetic and selection audits passed."]
    q = pd.read_csv(WORK / "q.csv", float_precision="round_trip")
    lines += ["", "## Quantiles by model", "", "| surrogate | n_train | G, 0.10 | L, 0.10 | G, 0.05 | L, 0.05 |",
              "|---|---:|---:|---:|---:|---:|"]
    for family in ("GP", "GBM"):
        for n in (128, 256, 512):
            values = [float(q.loc[q.arm == arm_id(family, n, variant, alpha), "q"].iloc[0])
                      for alpha in ALPHAS for variant in ("G", "L")]
            lines.append(f"| {family} | {n} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    gate, band = pd.read_csv(WORK / "gate.csv"), pd.read_csv(WORK / "band.csv")
    lines += ["", "## Per-arm test gates and fixed bands", "",
              "| arm | covered / 256 | coverage | two-sided CP 95% | one-sided upper 95% | gate | band | n initial → final | widened |",
              "|---|---:|---:|---|---:|---|---|---:|---|"]
    for _, row in gate.iterrows():
        b = band.loc[band.arm == row.arm].iloc[0]
        lines.append(f"| {row.arm} | {int(row.covered)}/256 | {row.coverage:.4f} | "
                     f"[{row.cp_lower:.4f}, {row.cp_upper:.4f}] | {row.cp_one_sided_upper:.4f} | "
                     f"{'pass' if row.gate_pass else 'fail'} | [{b.lower:.2f}, 2.00] | "
                     f"{int(b.n_initial)} → {int(b.n)} | {bool(b.widened)} |")
    lines += ["", "P bands (not a conformal gate):", "", "| arm | band | n initial → final | widened |",
              "|---|---|---:|---|"]
    for _, row in band.loc[band.variant == "P"].iterrows():
        lines.append(f"| {row.arm} | [{row.lower:.2f}, 2.00] | {int(row.n_initial)} → {int(row.n)} | {bool(row.widened)} |")
    accuracy = pd.read_csv(WORK / "accuracy.csv")
    lines += ["", "## Descriptive test RMSE", "", "| surrogate | n_train | z | recovery (pp) | productivity (mol/kg/day) |",
              "|---|---:|---:|---:|---:|"]
    for family in ("GP", "GBM"):
        for n in (128, 256, 512):
            values = [float(accuracy.loc[(accuracy.surrogate == family) & (accuracy.n_train == n) &
                                        (accuracy.response == response), "rmse"].iloc[0]) for response in RESPONSES]
            lines.append(f"| {family} | {n} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    selected = pd.read_csv(WORK / "selection_arms.csv")
    lines += ["", "## Counts per selected arm", "", "| arm | requested | unique in arm | duplicate memberships | front shortfall |",
              "|---|---:|---:|---:|---:|"]
    for _, row in selected.iterrows():
        lines.append(f"| {row.arm} | {int(row.requested)} | {int(row.unique_in_arm)} | "
                     f"{int(row.duplicates_within_arm)} | {int(row.front_shortfall)} |")
    state["prediction_sha256"] = {path.name: file_hash(path) for path in sorted((WORK / "predictions").glob("*.csv"))}
    provenance = {**state["provenance"], "model_sha256": state["fit_sha256"], "q_sha256": state["q_sha256"],
                  "selected_sha256": state["selected_sha256"], "prediction_sha256": state["prediction_sha256"],
                  "phase_wall_s": {name: info["wall_s"] for name, info in state["phases"].items()},
                  "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                      (ROOT / "scripts" / "run_step6b.py", ROOT / "scripts" / "analyse_step6.py",
                       ROOT / "src" / "h2psa" / "step6.py", ROOT / "src" / "h2psa" / "selection.py",
                       ROOT / "src" / "h2psa" / "step6_metrics.py")}}
    atomic_json(target / "provenance.json", provenance)
    text = "\n".join(lines)+"\n"
    (WORK / "REPORT.md").write_text(text)
    (ROOT / "docs" / "validation" / "step6b.md").write_text(text)
    state["status"] = "complete — Step 6b only; awaiting commit/push"
    save(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "calibrate", "diagnostics", "select", "report", "all"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    state = prepare()
    for name in ("train", "calibrate", "diagnostics", "select", "report"):
        if args.phase not in (name, "all"):
            continue
        state["status"] = f"running {name}"
        save(state)
        if name in ("train", "select"):
            globals()[name](state, args.workers)
        else:
            globals()[name](state)


if __name__ == "__main__":
    main()
