"""Synthetic-only statistical, front and future-analysis integration checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from h2psa import step6_metrics as metrics


def labels(z, statuses=None, recovery=None, productivity=None, prefix="point"):
    n = len(z)
    return pd.DataFrame(dict(id=[f"{prefix}_{i}" for i in range(n)], z=z,
                             status=statuses or ["css"]*n,
                             recovery=[70.0]*n if recovery is None else recovery,
                             productivity=[30.0]*n if productivity is None else productivity))


def selected_clusters():
    rows = []
    for surrogate in ("GP", "GBM"):
        for seed in (10, 11, 12):
            for variant, violations in (("P", 3), ("G", seed-9), ("L", seed-10)):
                for i in range(5):
                    rows.append(dict(id=f"{surrogate}_{variant}_{seed}_{i}",
                                     arm=f"{surrogate}_n128_{variant}_a0.10", surrogate=surrogate,
                                     n_train=128, variant=variant, alpha=.10, seed=seed,
                                     status="css", z=3.0 if i < violations else 1.0,
                                     bound=2.0, z_hat=1.8, recovery=70.0, productivity=30.0))
    return pd.DataFrame(rows)


def test_exact_binomial_intervals_and_sanity_upper():
    lo, hi = metrics.clopper_pearson(0, 20)
    assert lo == 0 and hi == pytest.approx(1-.025**(1/20))
    assert metrics.cp_upper(0, 20) == pytest.approx(1-.05**(1/20))
    assert metrics.clopper_pearson(20, 20)[1] == 1
    assert np.isnan(metrics.clopper_pearson(0, 0)).all()


def test_newcombe_uses_wilson_square_root_difference():
    def wilson(k, n):
        p, z = k/n, norm.ppf(.975)
        centre = (p+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return centre-half, centre+half
    p1, p2 = .9, .6
    l1, u1 = wilson(90, 100)
    l2, u2 = wilson(30, 50)
    expected = (.3, .3-np.sqrt((p1-l1)**2+(u2-p2)**2), .3+np.sqrt((u1-p1)**2+(p2-l2)**2))
    np.testing.assert_allclose(metrics.newcombe_difference(90,100,30,50), expected)


def test_failures_censoring_caps_and_constraint_equality():
    frame = labels([2.0, 1.0, 3.0, 1.0, np.nan], ["css", "cap", "censored", "css", "failed"])
    frame["bound"] = [2.0, 1.0, 2.0, .9, 2.0]
    violation = metrics.violation_summary(frame)
    coverage = metrics.coverage_summary(frame)
    assert (violation["n"], violation["k"], violation["sensitivity_n"], violation["sensitivity_k"]) == (4,1,5,2)
    assert (coverage["n"], coverage["k"], coverage["sensitivity_n"], coverage["sensitivity_k"]) == (4,2,5,2)
    assert metrics.feasible_front(frame).tolist() == [[70.0,30.0]]


def test_band_rule_widens_once_and_selected_coverage_uses_all_points():
    test = labels([1.0]*40)
    test["bound"] = [1.8]*29+[1.3]*11
    test["z_hat"] = test["bound"]
    for key, value in dict(arm="GP_n128_G_a0.10",surrogate="GP",n_train=128,variant="G",alpha=.10).items():
        test[key] = value
    selected = test.iloc[:4].copy()
    selected["bound"] = [1.9,1.1,1.1,1.1]
    selected["z"] = [1.0,2.0,2.0,2.0]
    bands, pooled = metrics.band_tables(test,selected)
    assert bands.loc[0,"lower"] == 1.25 and bands.loc[0,"widened"]
    assert bands.loc[0,"n"] == 40
    assert bands.loc[0,"selected_inside_fraction"] == .25
    assert bands.loc[0,"selected_n"] == 4 and bands.loc[0,"selected_coverage"] == .25
    assert pooled.loc[0,"n_band"] == 40 and pooled.loc[0,"n_selected"] == 4
    assert pooled.loc[0,"difference_lower"] > 0
    test.loc[29,"bound"] = 1.5
    assert metrics.choose_band(test)["lower"] == 1.5
    selected.loc[0,"status"] = "failed"
    selected.loc[0,"z"] = np.nan
    bands,pooled=metrics.band_tables(test,selected)
    # Band membership uses frozen predictions and includes the failed simulation.
    assert bands.loc[0,"selected_inside_fraction"] == .25
    assert bands.loc[0,"selected_n"] == 3
    assert pooled.loc[0,"selected_inside_fraction"] == .25


def test_nondominated_hypervolume_and_recovery_losses():
    np.testing.assert_array_equal(metrics.nondominated([[80,10],[80,11],[75,10],[60,30],[60,30]]),
                                  [[80,11],[60,30]])
    assert metrics.hypervolume_2d([[0,1],[1,0]]) == pytest.approx(.21)
    reference = metrics.nondominated([[80,10],[75,20],[60,30]])
    realised = labels([1.,1.],recovery=[80.,60.],productivity=[10.,30.])
    result = metrics.front_metrics(realised,reference)
    assert result["hv_ratio"] == pytest.approx(.21/.585)
    # Stored recovery is a percent: 75 - 60 = 15 pp, without a factor of 100.
    assert result["loss_0.1"] == 15 and result["loss_0.9"] == 0
    sparse = metrics.front_metrics(realised.iloc[:1],reference)
    assert sparse["not_reached_0.1"] and np.isnan(sparse["loss_0.1"])
    assert metrics.front_metrics(realised,np.array([[80.,10.]]))["status"].startswith("undefined HV")


def test_realised_front_coordinates_export_all_arms_and_exclude_unusable_points():
    frame=labels([1.,1.,1.,3.,1.,np.nan],statuses=["css","css","css","censored","cap","failed"],
                 recovery=[80.,75.,60.,99.,70.,90.],productivity=[10.,10.,30.,100.,20.,90.])
    frame["arm"]=["A"]*3+["B"]*3
    frame["surrogate"]="GP"
    frame["n_train"]=128
    frame["variant"]="G"
    frame["alpha"]=.10
    fronts=metrics.realised_front_coordinates(frame)
    assert fronts.columns.tolist() == [*metrics.ARM_FIELDS,"recovery","productivity"]
    assert fronts[["arm","recovery","productivity"]].values.tolist() == [
        ["A",80.,10.],["A",60.,30.],["B",70.,20.]]
    assert fronts["surrogate"].eq("GP").all() and fronts["n_train"].eq(128).all()
    assert metrics.realised_front_coordinates(frame.iloc[:0]).empty


def test_holm_is_step_down_with_fixed_rank_levels():
    decisions = metrics.holm(dict(RQ1=.0001,RQ2=.015,RQ3=.04,RQ5=.5))
    assert decisions["RQ1"]["adjusted_p"] == pytest.approx(.0004)
    assert decisions["RQ2"]["adjusted_p"] == pytest.approx(.045)
    assert decisions["RQ3"]["adjusted_p"] == pytest.approx(.08)
    assert decisions["RQ5"]["adjusted_p"] == pytest.approx(.5)
    assert decisions["RQ3"]["level"] == .025


def test_exact_rq_tests_gate_and_paired_cluster_bootstrap(monkeypatch):
    selected = selected_clusters()
    tests = metrics.rq_tests(selected,dict(p_value=.001),True)
    assert tests["decisions"]["RQ1"]["p_value"] == pytest.approx(binom.sf(17,30,.10))
    assert tests["decisions"]["RQ3"]["p_a"] == pytest.approx(binom.cdf(6,30,.17))
    blocked = metrics.rq_tests(selected,dict(p_value=.001),False)
    assert blocked["decisions"]["RQ2"]["verdict"] == "not confirmed: sanity gate failed"
    assert blocked["decisions"]["RQ3"]["verdict"] == "descriptive only"
    regression = pd.DataFrame(dict(set=["D_shift","D_shift","S_dist","S_dist"],id=["s0","s1","d0","d1"],
                                  arm=["A","A","B","B"],seed=[10,10,10,11]))
    seen = []
    def logistic(rows,clustered=True):
        assert clustered is False
        assert set(rows["set"]) == {"D_shift","S_dist"}
        seen.append(rows)
        return dict(b_phi=-.5)
    monkeypatch.setattr(metrics,"fit_shift_logistic",logistic)
    robust = metrics.cluster_robustness(selected,regression,tests,draws=40)
    assert len(seen) == 40
    # Each paired cluster's G rate exceeds its L rate by exactly 1/5.
    assert robust["RQ3"]["difference_lower"] == pytest.approx(.2)
    assert robust["RQ5"]["upper"] == -.5
    repeated = metrics.cluster_robustness(selected,regression,tests,draws=40)
    assert robust == repeated


def test_shared_ids_count_once_per_arm_and_stable_first_seed():
    sims = labels([1.0],prefix="shared")
    members = pd.DataFrame([dict(id="shared_0",arm=arm,surrogate="GP",n_train=128,variant="P",alpha=.10,seed=seed)
                            for arm,seed in (("A",11),("A",10),("B",12))])
    predictions = pd.DataFrame(dict(id=["shared_0","shared_0"],arm=["A","B"],bound=[2.,2.],z_hat=[1.,1.]))
    joined = metrics.join_predictions(sims,predictions,members)
    assert joined["seed"].tolist() == [10,12]
    assert metrics.violation_summary(metrics.pooled_selected(joined))["n"] == 2


def test_join_coalesces_matching_nominal_pair_and_rejects_incomplete_prediction_grid():
    sims = labels([1.0,1.0])
    members = pd.DataFrame(dict(id=["point_0"],arm=["A"],seed=[10],nominal_id=["nominal_0"]))
    sims["nominal_id"] = ["nominal_0","nominal_1"]
    prediction = pd.DataFrame(dict(id=["point_0"],arm=["A"],bound=[2.],z_hat=[1.]))
    joined = metrics.join_predictions(sims,prediction,members)
    assert joined["nominal_id"].tolist() == ["nominal_0"]
    assert "nominal_id_x" not in joined
    with pytest.raises(ValueError,match="complete simulated ID set"):
        metrics.join_predictions(sims,prediction)
    members["nominal_id"] = "different_nominal"
    with pytest.raises(ValueError,match="disagreement for nominal_id"):
        metrics.join_predictions(sims,prediction,members)


def test_one_sided_paired_mcnemar_and_failure_sensitivity():
    nominal = labels([1.,1.,1.,3.,3.,1.],prefix="nom")
    nominal["arm"] = "A"
    disturbed = labels([3.,3.,3.,1.,1.,np.nan],statuses=["css"]*5+["failed"],prefix="dist")
    disturbed["arm"] = "A"
    disturbed["nominal_id"] = nominal["id"]
    result = metrics.mcnemar_exact(nominal,disturbed)
    assert (result["increase"],result["decrease"],result["n"]) == (3,2,5)
    assert result["p_value"] == pytest.approx(binom.sf(2,5,.5))
    assert result["sensitivity"]["increase"] == 4 and result["failed_pairs"] == 1
    disturbed.loc[0,"nominal_id"] = "missing"
    with pytest.raises(ValueError,match="nominal arm pair"):
        metrics.mcnemar_exact(nominal,disturbed)


def test_clustered_logistic_uses_set_intercepts_and_reports_separation():
    rng = np.random.default_rng(8)
    n = 600
    rows = pd.DataFrame(dict(set=np.where(np.arange(n)%2,"D_shift","S_dist"),
                             variant=np.where(np.arange(n)%3,"G","L"),
                             surrogate=np.where(np.arange(n)%5,"GP","GBM"),
                             u_phi=rng.random(n),u_y_CH4=rng.random(n),u_y_CO=rng.random(n),
                             cluster=[f"c{i//3}" for i in range(n)]))
    probability=1/(1+np.exp(-(.5-2*rows["u_phi"]+.3*rows["u_y_CH4"])))
    rows["miscovered"]=(rng.random(n)<probability).astype(int)
    fit=metrics.fit_shift_logistic(rows)
    assert fit["status"] == "defined" and fit["b_phi"] < 0
    assert fit["p_value"] < .01 and fit["odds_ratio"] > 1
    assert {row["term"] for row in fit["coefficients"]} == {
        "intercept_D_shift","intercept_S_dist","L","GBM","b_phi","b_CH4","b_CO"}
    rows["miscovered"] = 0
    separated = metrics.fit_shift_logistic(rows)
    assert separated["status"].startswith("undefined") and np.isnan(separated["p_value"])


def test_undefined_bootstrap_draw_is_not_silently_removed():
    bounds, count = metrics._percentiles([.1,np.nan,.3],(.05,.95))
    assert count == 1 and np.isnan(bounds).all()


def test_registered_rq3_power_table_and_missing_data_guard(tmp_path):
    assert metrics.rq3_power()["rejection_max"].tolist() == [35,33,32,17]
    with pytest.raises(FileNotFoundError,match="Selected simulations"):
        metrics.analyse_generated(tmp_path,tmp_path,tmp_path)


def test_cli_json_serializes_undefined_values_without_nonstandard_nan():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("analyse_step6",root/"scripts/analyse_step6.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.json_safe(dict(value=np.nan,count=np.int64(4))) == dict(value=None,count=4)


@pytest.fixture
def synthetic_manifest(tmp_path):
    import hashlib
    import json

    generated,predictions,reports=(tmp_path/name for name in ("generated","predictions","reports"))
    for folder in (generated,predictions,reports):
        folder.mkdir()
    def hashed(path):
        path.write_text("id,synthetic\nexample,1\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={"prereg_tag":"prereg-v1",
              "python":"3.14.7","libraries":dict(metrics.PINNED_SOFTWARE),
              "input_sha256":{name:hashed(generated/f"{name}.csv")
                              for name in ("D_train","D_cal","D_test","D_shift")},
              "selected_sha256":{name:hashed(generated/f"{name}.csv")
                                 for name in ("S_nom_points","S_dist_points","S_nom_memberships","S_dist_memberships")},
              "prediction_sha256":{f"{name}.csv":hashed(predictions/f"{name}.csv")
                                   for name in ("D_test","D_shift","S_nom","S_dist")},
              "q_sha256":hashed(reports/"q.csv"),
              "model_sha256":{"missing_ignored_model.joblib":"not needed for replay"}}
    (reports/"provenance.json").write_text(json.dumps(manifest))
    return generated,predictions,reports,manifest


def test_manifest_replay_verifies_committed_artifacts_without_models(synthetic_manifest):
    generated,predictions,reports,manifest=synthetic_manifest
    assert metrics.verify_frozen_manifest(generated,predictions,reports,True) == manifest


@pytest.mark.parametrize("location,name",[("generated","D_test.csv"),
                                         ("generated","S_nom_points.csv"),
                                         ("generated","S_dist_memberships.csv"),
                                         ("predictions","S_nom.csv"),("reports","q.csv")])
def test_manifest_rejects_tampered_artifacts_before_label_parsing(synthetic_manifest,monkeypatch,location,name):
    generated,predictions,reports,_=synthetic_manifest
    folder={"generated":generated,"predictions":predictions,"reports":reports}[location]
    (folder/name).write_text("tampered synthetic content\n")
    monkeypatch.setattr(metrics.pd,"read_csv",lambda *a,**k:pytest.fail("No labels should be parsed before verification"))
    with pytest.raises(ValueError,match="hash mismatch"):
        metrics.analyse_generated(generated,predictions,reports,require_manifest=True)


def test_manifest_presence_and_complete_hash_sets_are_enforced(tmp_path,synthetic_manifest):
    import json
    assert metrics.verify_frozen_manifest(tmp_path,tmp_path,tmp_path) is None
    with pytest.raises(FileNotFoundError,match="provenance is required"):
        metrics.verify_frozen_manifest(tmp_path,tmp_path,tmp_path,True)
    generated,predictions,reports,manifest=synthetic_manifest
    del manifest["selected_sha256"]["S_nom_points"]
    (reports/"provenance.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Incomplete frozen manifest"):
        metrics.verify_frozen_manifest(generated,predictions,reports,True)


@pytest.mark.parametrize("changed",["python","library","manifest"])
def test_default_analysis_rejects_environment_drift(synthetic_manifest,monkeypatch,changed):
    import json
    generated,predictions,reports,manifest=synthetic_manifest
    if changed == "python":
        monkeypatch.setattr(metrics.platform,"python_version",lambda:"3.15.0")
    elif changed == "library":
        monkeypatch.setattr(metrics,"package_version",lambda name:"0.0" if name == "statsmodels"
                            else metrics.PINNED_SOFTWARE[name])
    else:
        manifest["libraries"]["numpy"]="0.0"
        (reports/"provenance.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="pinned versions"):
        metrics.verify_frozen_manifest(generated,predictions,reports,True)


def test_cli_defaults_use_committed_artifacts_and_require_manifest(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("analyse_step6_defaults",root/"scripts/analyse_step6.py")
    script=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script,"ROOT",tmp_path)
    captured=[]
    def fake_analysis(generated,predictions,reports,**kwargs):
        captured.append((generated,predictions,reports,kwargs))
        return {"tables":{},"results":{"synthetic":True}}
    monkeypatch.setattr(script,"analyse_generated",fake_analysis)
    assert script.main([]) == 0
    reports=tmp_path/"docs"/"validation"/"step6b"
    assert captured == [(tmp_path/"data"/"generated",reports/"predictions",reports,{"require_manifest":True})]


def test_future_pipeline_with_synthetic_memberships_and_secondary_gate_failures(monkeypatch):
    metadata = []
    for surrogate in ("GP","GBM"):
        for n_train in (128,256,512):
            for variant in ("P","G","L"):
                metadata.append(dict(arm=f"{surrogate}_n{n_train}_{variant}_a0.10",surrogate=surrogate,
                                     n_train=n_train,variant=variant,alpha=.10))
        for variant in ("G","L"):
            metadata.append(dict(arm=f"{surrogate}_n512_{variant}_a0.05",surrogate=surrogate,
                                 n_train=512,variant=variant,alpha=.05))
    nominal_rows, nominal_members, disturbed_rows, disturbed_members = [], [], [], []
    for meta in metadata:
        for seed in (10,11,12):
            for i in range(3):
                pid=f"nom_{len(nominal_rows)}"
                z=2.5 if meta["variant"] != "L" and i == 0 else 1.
                nominal_rows.append(dict(id=pid,status="css",z=z,recovery=60.+5*i,productivity=30.-5*i))
                nominal_members.append(dict(id=pid,seed=seed,**meta))
                if meta["n_train"] == 512 and meta["alpha"] == .10:
                    did=f"dist_{len(disturbed_rows)}"
                    disturbed_rows.append(dict(id=did,status="css",z=2.5 if i == 1 else z,
                                               recovery=55.+5*i,productivity=25.-5*i,nominal_id=pid,
                                               u_phi=.1+.3*i,u_y_CH4=.2+.1*seed/12,u_y_CO=.8-.2*i))
                    disturbed_members.append(dict(id=did,nominal_id=pid,seed=seed,**meta))
    simulated={"D_train":labels([1.,1.,1.],recovery=[90.,70.,80.],productivity=[5.,100.,50.],prefix="train"),
               "D_cal":labels([1.,1.],prefix="cal"),"D_test":labels([1.]*40,prefix="test"),
               "D_shift":labels([1.]*20,prefix="shift"),"S_nom":pd.DataFrame(nominal_rows),
               "S_dist":pd.DataFrame(disturbed_rows)}
    for name in ("D_shift",):
        for j,key in enumerate(("u_phi","u_y_CH4","u_y_CO")):
            simulated[name][key]=np.linspace(.1+.1*j,.8+.1*j,len(simulated[name]))
    memberships={"S_nom":pd.DataFrame(nominal_members),"S_dist":pd.DataFrame(disturbed_members)}
    predictions={}
    for name in ("D_test","D_shift"):
        predictions[name]=pd.DataFrame([dict(id=pid,bound=.5 if meta["alpha"] == .05 else 1.8,z_hat=1.8,**meta)
                                       for meta in metadata for pid in simulated[name]["id"]])
    for name in ("S_nom","S_dist"):
        predictions[name]=memberships[name].drop(columns=[column for column in ("seed","nominal_id")
                                                         if column in memberships[name]]).copy()
        predictions[name]["bound"]=1.8
        predictions[name]["z_hat"]=1.8
    bands=pd.DataFrame([dict(arm=arm,**metrics.choose_band(group))
                        for arm,group in predictions["D_test"].groupby("arm")])
    gate=pd.DataFrame([dict(arm=meta["arm"],gate_pass=meta["alpha"] == .10)
                       for meta in metadata if meta["variant"] != "P"])
    monkeypatch.setattr(metrics,"fit_shift_logistic",lambda rows,clustered=True:
                        dict(status="defined",b_phi=-.5,p_value=.001,coefficients=[]))
    output=metrics.analyse_frames(simulated,predictions,memberships,gate,bands,draws=12)
    assert output["results"]["tests"]["gate_pass"] is True
    assert output["tables"]["gate"]["gate_pass"].eq(False).sum() == 4
    assert len(output["tables"]["violations"]) == 28
    assert output["results"]["mcnemar"]["n"] == 54
    assert output["tables"]["statuses"]["n"].sum() == sum(map(len,simulated.values()))
    assert output["tables"]["failed_ids"].empty
    assert set(output["tables"]["realised_fronts"]["arm"]) == set(memberships["S_nom"]["arm"])
    assert output["results"]["declarations"]["pool_policy"] == "per_arm"
    assert output["tables"]["violations"]["selection_shortfall"].gt(0).all()
    simulated["D_test"].loc[:35,"status"] = "failed"
    simulated["D_test"].loc[:35,"z"] = np.nan
    sensitivity=metrics.analyse_frames(simulated,predictions,memberships,gate,bands,draws=2)
    assert sensitivity["results"]["tests"]["gate_pass"] is True
    assert sensitivity["results"]["failure_sensitivity_tests"]["gate_pass"] is False
    assert len(sensitivity["tables"]["failed_ids"]) == 36
    incomplete={name:frame.copy() for name,frame in predictions.items()}
    missing=metadata[0]["arm"]
    incomplete["D_test"]=incomplete["D_test"].loc[incomplete["D_test"].arm != missing]
    with pytest.raises(ValueError,match="missing registered primary"):
        metrics.analyse_frames(simulated,incomplete,memberships,gate,bands,draws=2)
    no_selection={name:frame.copy() for name,frame in simulated.items()}
    no_predictions={name:frame.copy() for name,frame in predictions.items()}
    no_memberships={name:frame.iloc[:0].astype(object) for name,frame in memberships.items()}
    for name in ("S_nom","S_dist"):
        no_selection[name]=no_selection[name].iloc[:0].astype(object)
        no_predictions[name]=no_predictions[name].iloc[:0].astype(object)
    monkeypatch.setattr(metrics,"fit_shift_logistic",lambda rows,clustered=True:
                        dict(status="undefined: no selected disturbance rows",b_phi=np.nan,p_value=np.nan,coefficients=[]))
    empty=metrics.analyse_frames(no_selection,no_predictions,no_memberships,gate,bands,draws=2)
    assert len(empty["tables"]["violations"]) == 28
    assert empty["tables"]["violations"]["simulated_n"].eq(0).all()
    assert empty["tables"]["violations"]["rate"].isna().all()
    assert empty["tables"]["pooled_bands"]["interpretation"].eq("undefined").all()
    assert empty["results"]["tests"]["decisions"]["RQ1"]["verdict"] == "undefined"
    assert np.isnan(empty["results"]["mcnemar"]["p_value"])
    assert empty["results"]["robustness"]["RQ5"]["undefined_draws"] == 2
