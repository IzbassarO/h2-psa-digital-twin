"""Step 8a (docs/preregistration.md, Addendum B): arms, S_nom reuse matching, Step 6b reproduction, and the analysis
on synthetic labels. No CSS run."""
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generated"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run8a, an8a = load("run_step8a"), load("analyse_step8a")


def test_arms_are_the_nine_registered_alpha_010_arms():
    assert [a["arm"] for a in run8a.ARMS] == [f"{s}_n{n}_{v}_a0.10" for s in ("GP", "GBM") for n in (128, 256, 512)
                                              for v in ("G", "L") if (s, v) != ("GBM", "G")]
    assert run8a.SEEDS == (13, 14, 15)


def test_selected_point_equal_to_s_nom_reuses_its_id():
    snom = pd.read_csv(GEN / "S_nom_points.csv", float_precision="round_trip")
    u = snom.loc[snom["id"] == "S_nom_0000", [f"u_{n}" for n in run8a.DECISIONS]].to_numpy(float)
    X = np.vstack([u, np.full((1, 8), .5)])
    result = dict(selected_X=X, front_X=X, requested_m=15, shortfall=13, selected_bound=np.ones(2),
                  selected_predictions={k: np.ones(2) for k in ("z", "recovery", "productivity")})
    lists = run8a.point_lists([({**run8a.ARMS[0], "seed": 13}, result), ({**run8a.ARMS[1], "seed": 13}, result)])
    points, members = lists["points"], lists["memberships"]
    assert points["id"].tolist() == ["S_rep_0000", "S_rep_0001"] and len(members) == 4
    assert points.loc[0, "snom_id"] == "S_nom_0000" and pd.isna(points.loc[1, "snom_id"])


@pytest.mark.skipif(not (ROOT / "results" / "step6b" / "optimization").exists(), reason="Step 6b outputs absent")
def test_nsga_reproduces_a_step6b_optimisation():
    assert run8a.reproduces_step6b(run8a.frozen_quantiles())


def synthetic(tmp_path, k):
    """9 arms x 3 seeds x 15 points; k[group] violations spread over the group's 135 points. S_rep_0000 reuses
    S_nom_0000's label (css, 471 ppm: a violation) and S_rep_0001 failed."""
    rows, labels = [], []
    for arm in run8a.ARMS:
        group = f"{arm['surrogate']}_{arm['variant']}"
        for seed in run8a.SEEDS:
            for rank in range(15):
                pid = f"S_rep_{len(rows):04d}"
                rows.append(dict(id=pid, **arm, seed=seed, selection_rank=rank + 1, counted_in_arm=True))
                i = sum(r["surrogate"] == arm["surrogate"] and r["variant"] == arm["variant"] for r in rows) - 1
                labels.append(dict(id=pid, status="css", z=2.5 if i < k[group] else 1.0))
    labels = pd.DataFrame(labels)
    points = labels[["id"]].assign(snom_id=[("S_nom_0000" if i == "S_rep_0000" else None) for i in labels["id"]])
    labels.loc[labels["id"] == "S_rep_0001", ["status", "z"]] = ["failed", np.nan]
    labels = labels.loc[labels["id"] != "S_rep_0000"]
    for name in ("S_nom.csv", "S_nom_memberships.csv"):
        shutil.copy(GEN / name, tmp_path / name)
    points.to_csv(tmp_path / "S_rep_points.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "S_rep_memberships.csv", index=False)
    labels.to_csv(tmp_path / "S_rep.csv", index=False)
    return an8a.analyse(tmp_path, draws=200)


def test_analysis_confirms_a_replication(tmp_path):
    # GP_G arms come first in ARMS: S_rep_0000 (reused, violating) and S_rep_0001 (failed) are GP-G points
    t = synthetic(tmp_path, {"GP_L": 5, "GBM_L": 130, "GP_G": 50})
    pooled = t["pooled"].set_index("group")
    assert pooled.loc["GP_L", ["k", "n"]].tolist() == [5, 135] and pooled.loc["GBM_L", "k"] == 130
    assert pooled.loc["GP_G", ["n", "failed", "sensitivity_k"]].tolist() == [134, 1, 50]
    h = t["hypotheses"].set_index("hypothesis")
    assert (h["verdict"] == "confirmed").all() and sorted(h["rank"]) == [1, 2, 3]
    assert h.loc["H_B1", "cp_upper_at_level"] <= .17
    assert (t["bootstrap"]["robustness"] == "robust").all() and (t["bootstrap"]["n_clusters"] == 9).all()
    assert t["statuses"].set_index("source").loc["reused S_nom", "n"] == 1
    assert len(t["arms"]) == 9 and t["arms"]["n_6c"].eq(45).all() and len(t["seeds"]) == 27


def test_analysis_rejects_when_gp_l_is_not_better(tmp_path):
    h = synthetic(tmp_path, {"GP_L": 40, "GBM_L": 10, "GP_G": 40})["hypotheses"].set_index("hypothesis")
    assert (h["verdict"] == "not confirmed").all()


def test_holm_and_cp_equivalence():
    d = an8a.holm({"H_B1": .04, "H_B2": .001, "H_B3": .02})
    assert [d[h]["rank"] for h in an8a.HYPOTHESES] == [3, 1, 2]
    assert d["H_B2"]["adjusted_p"] == pytest.approx(.003) and d["H_B3"]["adjusted_p"] == pytest.approx(.04)
    assert d["H_B1"]["adjusted_p"] == pytest.approx(.04) and d["H_B1"]["verdict"] == "confirmed"
    assert an8a.holm({"H_B1": .03, "H_B2": .03, "H_B3": .5})["H_B1"]["rank"] == 1          # ties: B1 first
    for k in range(10, 20):                    # CP upper bound <= 0.17  <=>  exact p <= level (no ties here)
        assert (an8a.cp_upper(k, 135, .025) <= .17) == (binom.cdf(k, 135, .17) <= .025)
