"""Registered NSGA-II selection and simulation lists; this module runs no CSS.

The pre-results numerical declaration uses the midpoint of each categorical
unit-cube bin as its canonical inverse. Training coordinates remain untouched.
Polynomial mutation has probability 1/8 per coordinate (operator probability 1).
Exact quantile-distance ties follow stable final-front order; registries follow
the caller's arm/seed order and the selection order within each result.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

DECISIONS = ("t_feed", "t_blow", "purge_to_feed", "P_high", "P_low", "f_zeolite",
             "n_equalisations", "feed_flow")
DISTURBANCES = ("phi", "y_CH4", "y_CO")
OUTPUTS = ("z", "recovery", "productivity")
MEMBERSHIP = ("arm", "surrogate", "n_train", "variant", "alpha", "seed")


def repair_unit_cube(X) -> np.ndarray:
    """Canonical valid preimages of floor(31*u)/30 and 1+floor(3*u)."""
    X = np.array(X, dtype=float, copy=True)
    if X.ndim not in (1, 2) or X.shape[-1] != 8:
        raise ValueError("Decision coordinates must have shape (8,) or (n, 8)")
    if not np.isfinite(X).all() or np.any((X < 0) | (X > 1)):
        raise ValueError("Decision coordinates must be finite and in [0, 1]")
    X[..., 5] = (np.minimum(np.floor(31 * X[..., 5]), 30) + .5) / 31
    X[..., 6] = (np.minimum(np.floor(3 * X[..., 6]), 2) + .5) / 3
    return X


class CubeRepair(Repair):
    def _do(self, problem, X, **kwargs):
        return repair_unit_cube(X)


def _predictions(predict_fn, bound_fn, X):
    if not len(X):
        return {name: np.empty(0) for name in OUTPUTS}, np.empty(0)
    raw = predict_fn(X)
    values = {name: np.asarray(raw[name], dtype=float).reshape(-1) for name in OUTPUTS}
    bound = np.asarray(bound_fn(X), dtype=float).reshape(-1)
    if any(len(v) != len(X) or not np.isfinite(v).all() for v in (*values.values(), bound)):
        raise ValueError("Prediction and bound callbacks must return finite arrays of length n")
    return values, bound


class SelectionProblem(Problem):
    def __init__(self, predict_fn, bound_fn):
        super().__init__(n_var=8, n_obj=2, n_ieq_constr=1, xl=0.0, xu=1.0)
        self.predict_fn, self.bound_fn = predict_fn, bound_fn

    def _evaluate(self, X, out, *args, **kwargs):
        # The algorithm also repairs its stored individuals before evaluation.
        X = repair_unit_cube(X)
        values, bound = _predictions(self.predict_fn, self.bound_fn, X)
        out["F"] = -np.column_stack((values["recovery"], values["productivity"]))
        out["G"] = (bound - 2)[:, None]


def select_productivity_quantiles(productivity, m: int) -> np.ndarray:
    """Nearest distinct points to actual linear sample quantiles, in k order."""
    productivity = np.asarray(productivity, dtype=float).reshape(-1)
    if m < 1 or not np.isfinite(productivity).all():
        raise ValueError("m must be positive and productivity finite")
    if len(productivity) < m:
        return np.arange(len(productivity), dtype=int)
    targets = np.quantile(productivity, (np.arange(m) + .5) / m, method="linear")
    remaining = np.ones(len(productivity), dtype=bool)
    chosen = []
    for target in targets:
        distances = np.abs(productivity - target)
        distances[~remaining] = np.inf
        index = int(np.argmin(distances))
        chosen.append(index)
        remaining[index] = False
    return np.asarray(chosen, dtype=int)


def front_and_selection(X, predict_fn, bound_fn, m: int) -> dict:
    """Filter a final population to distinct feasible nondominated points."""
    X = repair_unit_cube(X)
    if X.ndim != 2:
        raise ValueError("Final population must have shape (n, 8)")
    if m < 1:
        raise ValueError("m must be positive")
    seen, keep = set(), []
    for i, row in enumerate(X):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    X = X[keep]
    values, bound = _predictions(predict_fn, bound_fn, X)
    feasible = bound <= 2
    X, bound = X[feasible], bound[feasible]
    values = {name: value[feasible] for name, value in values.items()}
    objectives = np.column_stack((values["recovery"], values["productivity"]))
    # Row i dominates row j only with >= on both and > on at least one.
    dominates = np.all(objectives[:, None, :] >= objectives[None, :, :], axis=2) & \
                np.any(objectives[:, None, :] > objectives[None, :, :], axis=2)
    keep = ~dominates.any(axis=0)
    X, bound = X[keep], bound[keep]
    values = {name: value[keep] for name, value in values.items()}
    chosen = select_productivity_quantiles(values["productivity"], m)
    return dict(front_X=X, front_predictions=values, front_bound=bound,
                selected_X=X[chosen], selected_predictions={name: value[chosen] for name, value in values.items()},
                selected_bound=bound[chosen], selected_indices=chosen,
                requested_m=m, shortfall=max(m - len(chosen), 0))


def optimise_seed(predict_fn: Callable, bound_fn: Callable, seed: int, m: int) -> dict:
    """Run one registered seed; default NSGA-II uses Deb's feasibility rules."""
    if seed not in (10, 11, 12) or m not in (10, 15):
        raise ValueError("Registered seeds are 10/11/12 and selections are m=10/15")
    problem = SelectionProblem(predict_fn, bound_fn)
    algorithm = NSGA2(pop_size=100, crossover=SBX(eta=15, prob=.9),
                      mutation=PM(eta=20, prob=1.0, prob_var=1 / 8),
                      repair=CubeRepair(), eliminate_duplicates=True)
    result = minimize(problem, algorithm, ("n_gen", 250), seed=seed, verbose=False, save_history=False)
    population = result.pop.get("X")
    return front_and_selection(population, predict_fn, bound_fn, m)


def _physical_key(point: Mapping) -> tuple:
    """Exact physical identity; do not merge nearby continuous decisions."""
    return tuple((name, value.item() if isinstance(value, np.generic) else value)
                 for name, value in sorted(point.items()))


def build_selected_lists(arm_outputs: Iterable[tuple[Mapping, Mapping]], mapper: Callable,
                         disturbance_mapper: Callable) -> dict[str, pd.DataFrame]:
    """Build unique run lists and retain every arm/seed membership.

    ``arm_outputs`` contains (metadata, optimise_seed_result) pairs in the desired
    fixed order. Metadata has arm, surrogate, n_train, variant, alpha and seed.
    ``mapper(u8)`` returns the physical decision and nominal disturbance fields
    (including anchor); ``disturbance_mapper(u3)`` returns phi, y_CH4 and y_CO.
    Neither callback may run a simulator. One rng4 draw is assigned per unique
    eligible nominal ID, shared across its paired disturbed memberships.
    """
    registry, nominal, memberships, shortages = {}, [], [], []
    counted = set()
    for metadata, result in arm_outputs:
        meta = {name: metadata[name] for name in MEMBERSHIP}
        X = repair_unit_cube(result["selected_X"])
        if X.ndim != 2:
            raise ValueError("Selected decisions must have shape (n, 8)")
        shortages.append({**meta, "front_n": len(result["front_X"]), "selected_n": len(X),
                          "requested_m": result["requested_m"], "shortfall": result["shortfall"]})
        for rank, u in enumerate(X):
            physical = dict(mapper(u.copy()))
            key = _physical_key(physical)
            if key not in registry:
                pid = f"S_nom_{len(nominal):04d}"
                registry[key] = pid
                nominal.append({"id": pid, **dict(zip((f"u_{name}" for name in DECISIONS), u)), **physical})
            pid = registry[key]
            member_key = (meta["arm"], pid)
            memberships.append({"id": pid, **meta, "selection_rank": rank + 1,
                                "counted_in_arm": member_key not in counted,
                                **{f"pred_{name}": result["selected_predictions"][name][rank] for name in OUTPUTS},
                                "pred_bound": result["selected_bound"][rank]})
            counted.add(member_key)

    rng = np.random.default_rng(4)
    nominal_by_id = {row["id"]: row for row in nominal}
    disturbed, paired, disturbed_ids, paired_counted = [], [], {}, set()
    for row in memberships:
        if row["n_train"] != 512 or row["alpha"] != .10:
            continue
        nominal_id = row["id"]
        if nominal_id not in disturbed_ids:
            did = f"S_dist_{len(disturbed):04d}"
            disturbed_ids[nominal_id] = did
            u = rng.random(3)
            disturbance = dict(disturbance_mapper(u.copy()))
            if set(disturbance) != set(DISTURBANCES):
                raise ValueError("Disturbance mapper must return phi, y_CH4 and y_CO")
            disturbed.append({**nominal_by_id[nominal_id], "id": did, "nominal_id": nominal_id,
                              **dict(zip((f"u_{name}" for name in DISTURBANCES), u)), **disturbance})
        did = disturbed_ids[nominal_id]
        member_key = (row["arm"], did)
        paired.append({**row, "id": did, "nominal_id": nominal_id,
                       "counted_in_arm": member_key not in paired_counted})
        paired_counted.add(member_key)
    membership_columns = ["id", *MEMBERSHIP, "selection_rank", "counted_in_arm",
                          *(f"pred_{name}" for name in OUTPUTS), "pred_bound"]
    point_columns = ["id", *(f"u_{name}" for name in DECISIONS), *DECISIONS, *DISTURBANCES, "anchor"]
    return {"nominal_points": pd.DataFrame(nominal) if nominal else pd.DataFrame(columns=point_columns),
            "nominal_memberships": pd.DataFrame(memberships, columns=membership_columns),
            "disturbed_points": pd.DataFrame(disturbed) if disturbed else
                                pd.DataFrame(columns=[*point_columns, "nominal_id", *(f"u_{name}" for name in DISTURBANCES)]),
            "disturbed_memberships": pd.DataFrame(paired, columns=[*membership_columns, "nominal_id"]),
            "shortages": pd.DataFrame(shortages)}
