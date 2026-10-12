"""Runner safety and resume checks with synthetic results and mocked solvers."""
import importlib.util
import json
import pickle
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch():
    spec = importlib.util.spec_from_file_location("run_batch", ROOT / "scripts" / "run_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_column(batch, tmp_path, monkeypatch):
    out = tmp_path / "out"
    for name in ("cases", "ckpt", "css_states"):
        (out / name).mkdir(parents=True)
    model = SimpleNamespace(y0=np.array([1.0, 0.0]), sp=["H2", "He"])
    monkeypatch.setattr(batch.design, "build", lambda point: (object(), model, 1.0))
    monkeypatch.setattr(batch.design, "scaled", lambda point: np.zeros(11))
    monkeypatch.setattr(batch, "nearest_state", lambda *a: (0.1, "neighbor_0001", np.array([0.0, 1.0])))
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr(batch, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    return out, model


def result(css=True, cycles=11):
    return dict(css=css, censored=None, cycles=cycles, runtime_s=999.0,
                y_end=np.array([1.0, 0.0]),
                perf=dict(ppm=12345.678, recovery_pct=87.654321,
                          productivity=55.4321, purity_pct=98.7654))


def saved_case(out, pid="point"):
    with (out / "cases" / f"{pid}.pkl").open("rb") as handle:
        return pickle.load(handle)


def test_primary_failure_retries_once_cold_and_blind_logs_hide_labels(batch, fake_column, monkeypatch, capsys):
    out, _ = fake_column
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        assert kwargs["tol"] == 1e-4 and kwargs["tol_ppm"] == 1e-5
        assert kwargs["censor"] == (100, 1000.0, 30)
        assert kwargs["accel"] is True
        kwargs["log"]("cycle 10: ppm 12345.678 rec 87.654321 change 0 closure 0 acc (12 s)")
        if len(calls) == 1:
            np.testing.assert_array_equal(kwargs["y0"], [0.0, 1.0])
            kwargs["checkpoint"](10, np.array([0.4, 0.6]))
            raise RuntimeError("synthetic failure with ppm 12345.678 recovery 87.654321")
        assert kwargs["y0"] is None
        assert kwargs["censor_history"] == []
        return result()

    monkeypatch.setattr(batch, "CycleRunner", lambda *a, **k: SimpleNamespace(run_to_css=solve))
    assert batch.run_point("point", {"t_feed": 50.0}, out, blind=True, retry_failures=True) == "point"
    saved = saved_case(out)
    assert len(calls) == 2
    assert saved["status"] == "css" and saved["attempts"] == 2
    assert saved["solver_failures"] == 1 and len(saved["attempt_errors"]) == 1
    assert saved["original_start"] == "neighbor_0001" and saved["attempt_start"] == "cold"
    assert saved["cycles_total"] == 21 and saved["wall_s"] == 30.0
    assert not (out / "ckpt" / "point.npz").exists()
    summary = batch.summary(out)
    assert summary.loc[0, "start"] == "cold"
    assert summary.loc[0, "original_start"] == "neighbor_0001"
    stdout = capsys.readouterr().out
    assert "solver failure" in stdout and "done: css" in stdout
    assert "12345.678" not in stdout and "87.654321" not in stdout
    assert "ppm" not in stdout and "rec " not in stdout


def test_double_failure_is_persisted_and_never_retried_again(batch, fake_column, monkeypatch):
    out, _ = fake_column
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(batch, "CycleRunner", lambda *a, **k: SimpleNamespace(run_to_css=solve))
    point = {"t_feed": 50.0}
    batch.run_point("point", point, out, blind=True, retry_failures=True)
    batch.run_point("point", point, out, blind=True, retry_failures=True)
    saved = saved_case(out)
    assert len(calls) == 2
    assert calls[1]["y0"] is None
    assert saved["status"] == "failed" and saved["attempts"] == 2
    assert saved["solver_failures"] == 1 and len(saved["attempt_errors"]) == 2
    assert np.isnan(saved["perf"]["ppm"])
    batch.summary(out)
    assert batch.block_counts(out) == (1, 0)


def write_checkpoint(out, model, *, n=30, wall_s=120.0, errors=None, attempt=1):
    meta = dict(point={"t_feed": 50.0}, original_start="neighbor_0001", attempt_start="neighbor_0001",
                attempt=attempt, errors=[] if errors is None else errors,
                wall_s=wall_s, cycles_prior_attempts=0)
    history = [1500.0 + i for i in range(n)]
    np.savez(out / "ckpt" / "point.npz", y=model.y0, n=n, censor_ppm=history, meta=json.dumps(meta))
    return history


def test_resume_preserves_timing_cycles_censor_history_and_warm_source(batch, fake_column, monkeypatch):
    out, model = fake_column
    history = write_checkpoint(out, model)

    def solve(**kwargs):
        np.testing.assert_array_equal(kwargs["y0"], model.y0)
        assert kwargs["max_cycles"] == 1970
        assert kwargs["censor_history"] == history
        kwargs["checkpoint"](10, model.y0)
        with np.load(out / "ckpt" / "point.npz") as checkpoint:
            assert int(checkpoint["n"]) == 40
            assert json.loads(str(checkpoint["meta"].item()))["wall_s"] == 130.0
        return result()

    monkeypatch.setattr(batch, "CycleRunner", lambda *a, **k: SimpleNamespace(run_to_css=solve))
    batch.run_point("point", {"t_feed": 50.0}, out, blind=True, retry_failures=True)
    saved = saved_case(out)
    assert saved["resumed"] and saved["wall_s"] == 140.0
    assert saved["cycles_before_resume"] == 30 and saved["cycles_total"] == 41
    summary = batch.summary(out)
    assert summary.loc[0, "cycles"] == 41
    assert summary.loc[0, "wall_s"] == 140.0
    assert summary.loc[0, "start"] == "neighbor_0001"
    assert summary.loc[0, "resume_start"] == "checkpoint"


def test_resume_at_cap_does_not_run_an_additional_css_loop_iteration(batch, fake_column, monkeypatch):
    out, model = fake_column
    write_checkpoint(out, model, n=2000)

    def solve(**kwargs):
        assert kwargs["max_cycles"] == 0
        return result(css=False, cycles=0)

    monkeypatch.setattr(batch, "CycleRunner", lambda *a, **k: SimpleNamespace(run_to_css=solve))
    batch.run_point("point", {"t_feed": 50.0}, out, blind=True, retry_failures=True)
    saved = saved_case(out)
    assert saved["status"] == "cap" and saved["cycles_total"] == 2000


def test_active_retry_checkpoint_counts_as_one_primary_failure(batch, fake_column):
    out, model = fake_column
    write_checkpoint(out, model, n=0, errors=[{"attempt": 1, "error": "synthetic"}], attempt=2)
    assert batch.block_counts(out) == (1, 0)
    pd.DataFrame({"id": ["point"], "status": ["css"], "solver_failures": [1]}).to_csv(
        out / "results.csv", index=False)
    # An atomically completed case and its not-yet-removed checkpoint count once.
    assert batch.block_counts(out) == (1, 0)


@pytest.mark.parametrize("counts", [(1, 0), (0, 1)])
def test_monitor_gate_stops_new_submissions_and_drains_workers(batch, tmp_path, monkeypatch, counts):
    points = tmp_path / "points.csv"
    pd.DataFrame({"id": [f"point_{i}" for i in range(10)], "t_feed": [50.0] * 10}).to_csv(points, index=False)
    submitted = []

    class FakeExecutor:
        def __init__(self, workers):
            assert workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, pid, *args, **kwargs):
            submitted.append(pid)
            future = Future()
            future.set_result(pid)
            return future

    monkeypatch.setattr(batch, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(batch, "block_counts", lambda out: counts if submitted else (0, 0))
    monkeypatch.setattr(batch, "summary", lambda out: pd.DataFrame())
    assert batch.main([str(points), str(tmp_path / "out"), "--workers", "2", "--blind",
                       "--retry-failures", "--failure-limit", "0.02", "--cap-limit", "0.05",
                       "--block-size", "10"]) == 3
    assert submitted == ["point_0", "point_1"]
