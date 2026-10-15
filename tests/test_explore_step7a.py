"""Step 7a matched-productivity interpolation (scripts/explore_step7a.py)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("explore_step7a", ROOT / "scripts" / "explore_step7a.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def test_recovery_at_interpolates_attains_below_and_stops_above():
    front = np.array([[80.0, 200.0], [70.0, 220.0]])          # (recovery, productivity), unsorted on purpose
    assert ex.recovery_at(front, 210.0) == 75.0
    assert ex.recovery_at(front, 100.0) == 80.0                # lowest-productivity point attains p
    assert np.isnan(ex.recovery_at(front, 221.0)) and np.isnan(ex.recovery_at(np.empty((0, 2)), 1.0))
