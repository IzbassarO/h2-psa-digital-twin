"""Synthetic and mocked selection checks; no CSS or actual optimisation runs."""
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.population import Population

from h2psa import selection


def cube(t):
    X = np.full((len(t), 8), .4)
    X[:, 0] = t
    return X


def predict(X):
    return dict(z=np.ones(len(X)), recovery=1 - X[:, 0], productivity=X[:, 0])


def bound(X):
    return np.ones(len(X))


def result(X):
    return selection.front_and_selection(X, predict, bound, m=10)


def test_categorical_midpoints_are_valid_idempotent_inverse_bins():
    X = cube(np.linspace(0, 1, 34))
    X[:, 5] = np.linspace(0, 1, 34)
    X[:, 6] = np.linspace(0, 1, 34)
    repaired = selection.repair_unit_cube(X)
    bins_f = np.minimum(np.floor(31 * X[:, 5]), 30)
    bins_n = np.minimum(np.floor(3 * X[:, 6]), 2)
    np.testing.assert_array_equal(np.floor(31 * repaired[:, 5]), bins_f)
    np.testing.assert_array_equal(np.floor(3 * repaired[:, 6]), bins_n)
    np.testing.assert_array_equal(repaired[:, [0, 1, 2, 3, 4, 7]], X[:, [0, 1, 2, 3, 4, 7]])
    np.testing.assert_array_equal(selection.repair_unit_cube(repaired), repaired)
    assert repaired[-1, 5] == 30.5 / 31
    assert repaired[-1, 6] == 2.5 / 3
    assert X[-1, 5] == 1  # caller's raw coordinates were untouched


@pytest.mark.parametrize("X", [np.zeros((2, 7)), cube([-1]), cube([np.nan]), cube([1.01])])
def test_invalid_unit_cube_is_rejected(X):
    with pytest.raises(ValueError):
        selection.repair_unit_cube(X)


def test_quantile_selection_uses_actual_productivity_quantiles_without_reuse():
    np.testing.assert_array_equal(selection.select_productivity_quantiles([0, 1, 2, 10], 2), [1, 2])
    np.testing.assert_array_equal(selection.select_productivity_quantiles([0, 1, 1, 1, 10], 3), [1, 2, 3])
    np.testing.assert_array_equal(selection.select_productivity_quantiles([2, 1], 3), [0, 1])


def test_final_front_filters_infeasible_dominated_and_physical_duplicates():
    X = cube([.1, .1, .2, .3, .4])
    X[0, 5], X[1, 5] = .400, .405  # same physical category

    def objectives(X):
        values = {1: (9, 1), 2: (8, .5), 3: (7, 3), 4: (20, 20)}
        pairs = np.array([values[round(t * 10)] for t in X[:, 0]])
        return dict(z=np.zeros(len(X)), recovery=pairs[:, 0], productivity=pairs[:, 1])

    final = selection.front_and_selection(X, objectives, lambda X: np.where(X[:, 0] == .4, 2.01, 2), 3)
    np.testing.assert_array_equal(final["front_X"][:, 0], [.1, .3])
    assert final["shortfall"] == 1
    np.testing.assert_array_equal(final["selected_indices"], [0, 1])
    assert len(final["selected_predictions"]["productivity"]) == 2


def test_empty_feasible_front_reports_complete_shortfall():
    final = selection.front_and_selection(cube([.1, .2]), predict, lambda X: np.full(len(X), 3), 10)
    assert final["front_X"].shape == final["selected_X"].shape == (0, 8)
    assert final["shortfall"] == 10


def test_registered_optimizer_settings_and_repair_before_both_callbacks(monkeypatch):
    calls = []
    X = cube([.1, .3, .6])

    def checked_predict(X):
        np.testing.assert_array_equal(X, selection.repair_unit_cube(X))
        calls.append("predict")
        return predict(X)

    def checked_bound(X):
        np.testing.assert_array_equal(X, selection.repair_unit_cube(X))
        calls.append("bound")
        return bound(X)

    def fake_minimize(problem, algorithm, termination, **kwargs):
        assert problem.n_var == 8 and problem.n_obj == 2 and problem.n_ieq_constr == 1
        assert algorithm.pop_size == 100
        assert algorithm.mating.crossover.eta.value == 15
        assert algorithm.mating.crossover.prob.value == .9
        assert algorithm.mating.mutation.eta.value == 20
        assert algorithm.mating.mutation.prob.value == 1
        assert algorithm.mating.mutation.prob_var.value == 1 / 8
        assert isinstance(algorithm.initialization.repair, selection.CubeRepair)
        assert algorithm.survival.filter_infeasible
        assert termination == ("n_gen", 250)
        assert kwargs == {"seed": 11, "verbose": False, "save_history": False}
        out = problem.evaluate(X, return_as_dictionary=True)
        np.testing.assert_allclose(out["F"], -np.column_stack((1 - X[:, 0], X[:, 0])))
        np.testing.assert_array_equal(out["G"], -np.ones((3, 1)))
        return SimpleNamespace(pop=Population.new("X", X))

    monkeypatch.setattr(selection, "minimize", fake_minimize)
    selected = selection.optimise_seed(checked_predict, checked_bound, 11, 10)
    assert selected["shortfall"] == 7
    assert calls == ["predict", "bound", "predict", "bound"]


def test_nsga_tournament_uses_deb_feasibility_before_objectives():
    from pymoo.algorithms.moo.nsga2 import binary_tournament

    algorithm = selection.NSGA2()
    algorithm.random_state = np.random.default_rng(10)
    pop = Population.new("F", np.array([[-100., -100.], [0., 0.], [-200., -200.]]),
                         "CV", np.array([[1.], [0.], [2.]]))
    winners = binary_tournament(pop, np.array([[0, 1], [0, 2]]), algorithm)
    np.testing.assert_array_equal(winners[:, 0], [1, 0])


@pytest.mark.parametrize("seed,m", [(1, 10), (10, 11), (13, 15)])
def test_optimizer_rejects_unregistered_seed_or_count(seed, m):
    with pytest.raises(ValueError, match="Registered"):
        selection.optimise_seed(predict, bound, seed, m)


def metadata(arm, seed=10, n_train=512, variant="P", alpha=.1):
    return dict(arm=arm, surrogate="GP", n_train=n_train, variant=variant, alpha=alpha, seed=seed)


def physical(u):
    row = dict(zip(selection.DECISIONS, u))
    row["f_zeolite"] = np.minimum(np.floor(31 * u[5]), 30) / 30
    row["n_equalisations"] = int(np.minimum(1 + np.floor(3 * u[6]), 3))
    row.update(phi=1., y_CH4=.1, y_CO=.2, anchor="lopes")
    return row


def disturbed(u):
    return dict(phi=.7 + .3 * u[0], y_CH4=.2 * u[1], y_CO=.3 * u[2])


def test_global_registry_preserves_seed_arm_memberships_and_paired_rng4_draws():
    arms = [(metadata("primary_P", seed=10), result(cube([.1, .2]))),
            (metadata("primary_P", seed=11), result(cube([.1]))),
            (metadata("primary_L", variant="L"), result(cube([.2, .3]))),
            (metadata("n256", n_train=256), result(cube([.4]))),
            (metadata("secondary", variant="G", alpha=.05), result(cube([.5])))]
    lists = selection.build_selected_lists(arms, physical, disturbed)
    nominal, members = lists["nominal_points"], lists["nominal_memberships"]
    shift, paired = lists["disturbed_points"], lists["disturbed_memberships"]
    assert len(nominal) == 5 and len(members) == 7
    assert members["id"].tolist()[:5] == ["S_nom_0000", "S_nom_0001", "S_nom_0000", "S_nom_0001", "S_nom_0002"]
    assert members["counted_in_arm"].tolist()[:5] == [True, True, False, True, True]
    assert len(shift) == 3 and len(paired) == 5
    assert set(paired["arm"]) == {"primary_P", "primary_L"}
    assert paired["nominal_id"].tolist() == members["id"].tolist()[:5]
    np.testing.assert_array_equal(shift[[f"u_{name}" for name in selection.DISTURBANCES]],
                                  np.random.default_rng(4).random((3, 3)))
    assert paired.iloc[0]["id"] == paired.iloc[2]["id"]
    assert len(lists["shortages"]) == 5
    assert lists["shortages"]["shortfall"].sum() == 43


def test_registry_dedupes_by_physical_mapping_without_merging_nearby_continuous_values():
    X = cube([.1, np.nextafter(.1, 1)])
    equal_objectives = lambda X: {name: np.ones(len(X)) for name in selection.OUTPUTS}
    final = selection.front_and_selection(X, equal_objectives, bound, m=10)
    lists = selection.build_selected_lists([(metadata("arm"), final)], physical, disturbed)
    assert len(lists["nominal_points"]) == 2


def test_registry_empty_selection_and_bad_disturbance_schema():
    empty = selection.front_and_selection(cube([.2]), predict, lambda X: np.full(len(X), 3), 10)
    lists = selection.build_selected_lists([(metadata("arm"), empty)], physical, disturbed)
    assert lists["nominal_memberships"].empty and lists["disturbed_memberships"].empty
    assert "id" in lists["nominal_points"] and "nominal_id" in lists["disturbed_points"]
    assert lists["shortages"].iloc[0]["shortfall"] == 10
    with pytest.raises(ValueError, match="Disturbance mapper"):
        selection.build_selected_lists([(metadata("arm"), result(cube([.2])))], physical, lambda u: {"phi": 1})
