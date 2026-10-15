"""Step 7b: model-form scenarios (design.py), cold-start flag (run_batch.py), runner inputs (run_step7b.py),
descriptive analysis (analyse_step7b.py) and the baseline regression (docs/preregistration.md, Addendum A)."""
import importlib.util
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from h2psa import design
from h2psa.column import wall_heat_capacity

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb, run7b, an7b = load("run_batch"), load("run_step7b"), load("analyse_step7b")
POINT = dict(f_zeolite=0.4, t_feed=101.0)


def test_scenario_settings_and_build():
    assert design.scenario_settings(None) == ({}, None, None)
    assert design.scenario_settings("ep_inflow_ed_gas") == ({"ep_gas": "ed"}, None, None)
    assert design.scenario_settings("thermal_range_low") == ({}, {"AC": 700.0, "ZEO": 800.0}, wall_heat_capacity(0.002))
    assert design.scenario_settings("thermal_range_high") == ({}, {"AC": 1100.0, "ZEO": 1100.0}, wall_heat_capacity(0.005))
    with pytest.raises(ValueError):
        design.scenario_settings("noncompetitive_h2")                  # dropped (Addendum A)
    c0, m0, w0 = design.build(POINT)
    assert c0.ep_gas == "h2" and design.build({**POINT, "scenario": "ep_inflow_ed_gas"})[0].ep_gas == "ed"
    _, m1, w1 = design.build({**POINT, "scenario": "thermal_range_low"})
    assert w1 == w0 and set(np.unique(m0.Cps)) == {880.0, 920.0} and set(np.unique(m1.Cps)) == {700.0, 800.0}
    assert m1.rhoCp_w / m0.rhoCp_w == pytest.approx(wall_heat_capacity(0.002) / wall_heat_capacity(0.003), rel=1e-12)
    assert design.scaled({**POINT, "scenario": "thermal_range_low"}).tolist() == design.scaled(POINT).tolist()


def test_cold_start_ignores_neighbour_states(tmp_path):
    for d in ("cases", "ckpt", "css_states"):
        (tmp_path / d).mkdir()
    (tmp_path / "css_states" / "corrupt.npz").write_bytes(b"not an npz")   # read only by a warm start
    rb.run_point("p", POINT, tmp_path, blind=True, timeout_s=0.5, cold_start=True)
    with open(tmp_path / "cases" / "p.pkl", "rb") as fh:
        r = pickle.load(fh)
    assert r["status"] == "failed" and r["attempt_start"] == "cold" and r["original_start"] is None


def test_runner_inputs_are_the_210_gp512_points():
    points = run7b.selected_points()
    m = pd.read_csv(ROOT / "data" / "generated" / "S_nom_memberships.csv").set_index("id")
    assert len(points) == 210 and points["id"].is_unique
    assert (m.loc[points["id"], "surrogate"] == "GP").all() and (m.loc[points["id"], "n_train"] == 512).all()
    spots = run7b.spot_ids(points["id"])
    assert spots == run7b.spot_ids(points["id"]) and list(spots) == list(design.SCENARIOS)
    assert all(len(set(v)) == 10 and set(v) <= set(points["id"]) for v in spots.values())


def test_analysis_on_synthetic_scenario_labels(tmp_path):
    gen = ROOT / "data" / "generated"
    for name in ("S_nom.csv", "S_nom_memberships.csv"):
        shutil.copy(gen / name, tmp_path / name)
    base = pd.read_csv(gen / "S_nom.csv", float_precision="round_trip")
    base = base.loc[base["id"].isin(run7b.selected_points()["id"])]
    shifted = base.assign(z=3.0, status="censored", impurity_ppm=np.nan)    # every point violates
    for s, frame in zip(design.SCENARIOS, (base, shifted, base)):
        frame.assign(scenario=s).to_csv(tmp_path / f"M_{s}.csv", index=False)
        frame.iloc[:10].assign(scenario=s).to_csv(tmp_path / f"M_{s}_cold.csv", index=False)
    t = an7b.analyse(tmp_path)
    p = t["paired"].set_index("set")
    same, worse = p.loc[design.SCENARIOS[0]], p.loc[design.SCENARIOS[1]]
    assert (same["feasible_to_violating"] == 0).all() and (same["violating_to_feasible"] == 0).all()
    assert (worse["scenario_k"] == worse["n"]).all() and (worse["violating_to_feasible"] == 0).all()
    assert (t["spot_checks"]["dz"] == 0).all() and (t["statuses"]["n"] == 210).all()
    v = t["violations"].set_index(["set", "arm"])["k"]
    assert v.loc["baseline"].equals(v.loc[design.SCENARIOS[0]])
    assert "DESCRIPTIVE" in an7b.report(t, None)


SOURCE = ROOT / "results" / "step6c" / "S_nom" / "css_states" / "S_nom_0318.npz"


@pytest.mark.skipif(not SOURCE.exists(), reason="Step 6c CSS states not present")
def test_baseline_labels_reproduced_exactly(tmp_path):
    """Default thermal values, no scenario: S_nom_0333 rerun from its recorded warm start (S_nom_0318) is bitwise
    identical to its Step 6c label and final state (about 1 min)."""
    for d in ("cases", "ckpt", "css_states"):
        (tmp_path / d).mkdir()
    shutil.copy(SOURCE, tmp_path / "css_states" / SOURCE.name)
    points = pd.read_csv(ROOT / "data" / "generated" / "S_nom_points.csv", float_precision="round_trip")
    row = points.loc[points["id"] == "S_nom_0333", ["id", *run7b.g.DECISIONS, *run7b.g.DISTURBANCES, "anchor"]]
    rb.run_point("S_nom_0333", rb.point_of(row.to_dict("records")[0]), tmp_path, blind=True, retry_failures=True)
    with open(tmp_path / "cases" / "S_nom_0333.pkl", "rb") as fh:
        new = pickle.load(fh)
    label = pd.read_csv(ROOT / "data" / "generated" / "S_nom.csv", float_precision="round_trip").set_index("id").loc["S_nom_0333"]
    assert new["start"] == "S_nom_0318" and new["cycles"] == label["cycles"]
    assert (new["perf"]["ppm"], new["perf"]["recovery_pct"], new["perf"]["productivity"]) == \
        (label["impurity_ppm"], label["recovery"], label["productivity"])
    old = ROOT / "results" / "step6c" / "S_nom" / "cases" / "S_nom_0333.pkl"
    if old.exists():
        with open(old, "rb") as fh:
            assert np.array_equal(pickle.load(fh)["y_end"], new["y_end"])
