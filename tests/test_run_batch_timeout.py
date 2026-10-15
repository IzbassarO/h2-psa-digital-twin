"""Step 6c per-attempt wall-time limit of scripts/run_batch.py (Deviations 2026-10-14)."""
import importlib.util
import pickle
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_batch", ROOT / "scripts" / "run_batch.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def test_time_limit_interrupts_python_loop_and_clears_timer():
    with pytest.raises(TimeoutError):
        with rb.time_limit(0.2):
            while True:
                pass
    with rb.time_limit(None):
        time.sleep(0.01)
    with rb.time_limit(5):
        pass
    time.sleep(0.05)                                   # no stray alarm after a normal exit
    with pytest.raises(TimeoutError):
        with rb.time_limit(0):
            pass


def test_timed_out_attempts_retry_cold_once_then_fail(tmp_path):
    point = dict(f_zeolite=0.4, t_feed=101.0)
    for d in ("cases", "ckpt", "css_states"):                # created by run_batch.main
        (tmp_path / d).mkdir()
    rb.run_point("p", point, tmp_path, blind=True, retry_failures=True, timeout_s=0.5)
    with open(tmp_path / "cases" / "p.pkl", "rb") as fh:
        r = pickle.load(fh)
    assert r["status"] == "failed" and r["attempts"] == 2 and r["attempt_start"] == "cold"
    assert [e["error_type"] for e in r["attempt_errors"]] == ["TimeoutError", "TimeoutError"]
    assert not (tmp_path / "ckpt" / "p.npz").exists()
