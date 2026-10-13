"""Frozen Step 6 surrogate and conformal methods; no physical simulation calls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import GridSearchCV, KFold
from threadpoolctl import threadpool_limits

DECISIONS = ("t_feed", "t_blow", "purge_to_feed", "P_high", "P_low", "f_zeolite",
             "n_equalisations", "feed_flow")
DISTURBANCES = ("phi", "y_CH4", "y_CO")
INPUTS = [f"u_{name}" for name in DECISIONS]
TRAIN_SIZES = (128, 256, 512)
ALPHAS = (0.10, 0.05)
RESPONSES = ("z", "recovery", "productivity")
GBM_GRID = {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [5, 10, 20], "max_iter": [200, 500, 1000]}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_id(family: str, n_train: int, response: str, quantile: float | None = None):
    suffix = "" if quantile is None else f"_q{quantile:.2f}"
    return f"{family}_n{n_train}_{response}{suffix}"


def arm_id(family: str, n_train: int, variant: str, alpha: float):
    return f"{family}_n{n_train}_{variant}_a{alpha:.2f}"


def registered_arms(include_diagnostics: bool = False):
    """Stable order, with P independent of alpha and secondary optimization at 512."""
    arms = []
    for family in ("GP", "GBM"):
        for n_train in TRAIN_SIZES:
            for variant in ("P", "G", "L"):
                arms.append(dict(arm=arm_id(family, n_train, variant, .10), surrogate=family,
                                 n_train=n_train, variant=variant, alpha=.10))
        sizes = TRAIN_SIZES if include_diagnostics else (512,)
        for n_train in sizes:
            for variant in ("G", "L"):
                arms.append(dict(arm=arm_id(family, n_train, variant, .05), surrogate=family,
                                 n_train=n_train, variant=variant, alpha=.05))
    return arms


def fit_tasks():
    tasks = []
    for n in TRAIN_SIZES:
        for family in ("GP", "GBM"):
            for response in RESPONSES:
                tasks.append((family, n, response, None))
        for alpha in ALPHAS:
            tasks.append(("GBM", n, "z", 1 - alpha))
    return tasks


def training_rows(frame: pd.DataFrame, n_train: int, response: str):
    """The prefix is taken BEFORE excluding ineligible labels."""
    prefix = frame.iloc[:n_train].copy()
    allowed = prefix.status != "failed"
    if response != "z":
        allowed &= prefix.status != "censored"
    selected = prefix.loc[allowed]
    X = selected[INPUTS].to_numpy(float)
    y = selected[response].to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite eligible training values")
    if response == "z" and not (selected.loc[selected.status == "censored", "z"] == 3).all():
        raise ValueError("Censored z must equal 3")
    return X, y, selected.id.tolist()


def make_gp():
    kernel = ConstantKernel() * Matern(np.ones(8), nu=2.5) + WhiteKernel(noise_level_bounds=(1e-6, 1e5))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10,
                                    random_state=0)


def fit_one(task, train_path: str, work_root: str):
    """One independent fit; accesses D_train only. Grid CV is serial within worker."""
    family, n, response, quantile = task
    train_path, work = Path(train_path), Path(work_root)
    name = model_id(family, n, response, quantile)
    model_path, info_path = work / "models" / f"{name}.joblib", work / "models" / f"{name}.json"
    specification = dict(model=name, family=family, n_train=n, response=response,
                         quantile=quantile, train_sha256=file_hash(train_path), implementation=1)
    if model_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info["specification"] != specification:
            raise ValueError(f"Cached fit specification differs: {name}")
        return info
    started = time.monotonic()
    frame = pd.read_csv(train_path, float_precision="round_trip")
    expected = [f"D_train_{i:04d}" for i in range(512)]
    if frame.id.tolist() != expected:
        raise ValueError("D_train IDs must retain their registered prefix order")
    X, y, ids = training_rows(frame, n, response)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "GP":
            if quantile is not None:
                raise ValueError("GP quantile fit not registered")
            estimator = make_gp().fit(X, y)
            chosen = {"kernel": str(estimator.kernel_), "theta": estimator.kernel_.theta.tolist(),
                      "log_marginal_likelihood": float(estimator.log_marginal_likelihood_value_),
                      "target_mean": float(estimator._y_train_mean),
                      "target_scale": float(estimator._y_train_std),
                      "n_restarts_optimizer": 10, "alpha": estimator.alpha}
        elif family == "GBM":
            loss = "squared_error" if quantile is None else "quantile"
            estimator = HistGradientBoostingRegressor(loss=loss, quantile=quantile,
                                                      early_stopping=False, random_state=0)
            scoring = "neg_mean_squared_error" if quantile is None else make_scorer(
                mean_pinball_loss, greater_is_better=False, alpha=quantile)
            search = GridSearchCV(estimator, GBM_GRID, scoring=scoring,
                                  cv=KFold(5, shuffle=False), n_jobs=1, refit=True, error_score="raise")
            search.fit(X, y)
            estimator = search.best_estimator_
            if estimator.n_iter_ != estimator.max_iter:
                raise ValueError("Unexpected GBM early stopping")
            chosen = {**search.best_params_, "loss": loss, "quantile": quantile,
                      "cv_loss": float(-search.best_score_), "best_index": int(search.best_index_),
                      "cv_folds": 5, "grid_combinations": len(search.cv_results_["params"]),
                      "early_stopping": False, "random_state": 0}
            (work / "cv").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(search.cv_results_).to_csv(work / "cv" / f"{name}.csv", index=False)
        else:
            raise ValueError(family)
    info = {"specification": specification, "n_eligible": len(ids), "training_ids": ids,
            "chosen": chosen, "warnings": [str(w.message) for w in caught],
            "wall_s": time.monotonic() - started}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".joblib.part")
    joblib.dump(estimator, temporary)
    temporary.replace(model_path)
    atomic_json(info_path, info)
    return info


@dataclass
class Surrogates:
    family: str
    n_train: int
    work: Path

    def __post_init__(self):
        self.models = {response: joblib.load(self.work / "models" /
                       f"{model_id(self.family, self.n_train, response)}.joblib") for response in RESPONSES}
        self.quantiles = {} if self.family == "GP" else {
            alpha: joblib.load(self.work / "models" /
                              f"{model_id('GBM', self.n_train, 'z', 1-alpha)}.joblib") for alpha in ALPHAS}

    def predict(self, X):
        return {name: model.predict(X) for name, model in self.models.items()}

    def components(self, X, variant: str, alpha: float):
        if variant != "L":
            return self.models["z"].predict(X), np.ones(len(X))
        if self.family == "GP":
            mean, sigma = self.models["z"].predict(X, return_std=True)
            if not np.isfinite(sigma).all() or (sigma <= 0).any():
                raise ValueError("Nonpositive registered GP predictive sigma")
            return mean, sigma
        return self.quantiles[alpha].predict(X), np.ones(len(X))

    def bound(self, X, variant: str, alpha: float, q: float = 0):
        center, scale = self.components(X, variant, alpha)
        return center if variant == "P" else center + q * scale


def conformal_quantile(scores, alpha: float):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite one-dimensional calibration scores required")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    q = float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")
    return q, rank


def cp_bounds(k: int, n: int):
    if not 0 <= k <= n or n == 0:
        raise ValueError("Invalid binomial count")
    lower = 0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1))
    upper = 1.0 if k == n else float(beta.ppf(.975, k+1, n-k))
    gate_upper = 1.0 if k == n else float(beta.ppf(.95, k+1, n-k))
    return lower, upper, gate_upper


def fixed_band(bound):
    bound = np.asarray(bound, float)
    initial = (bound >= 1.5) & (bound <= 2.0)
    widened = int(initial.sum()) < 30
    lower = 1.25 if widened else 1.5
    final = (bound >= lower) & (bound <= 2.0)
    return final, {"lower": lower, "upper": 2.0, "widened": widened,
                   "n_initial": int(initial.sum()), "n": int(final.sum()), "below_30": int(final.sum()) < 30}
