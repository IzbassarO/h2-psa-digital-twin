"""Step 6d exploratory recalibration helper (scripts/explore_step6d.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("explore_step6d", ROOT / "scripts" / "explore_step6d.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def test_out_of_sample_q_uses_only_other_groups():
    frame = pd.DataFrame({"seed": np.repeat([10, 11, 12], 15), "score": np.r_[np.zeros(15), np.ones(15), 2 * np.ones(15)]})
    q = ex.out_of_sample_q(frame, "seed", 0.10)
    # 30 calibration scores, rank ceil(31 * 0.9) = 28: the larger of the two other seeds' constant scores
    assert q[frame.seed == 10].eq(2).all() and q[frame.seed == 11].eq(2).all() and q[frame.seed == 12].eq(1).all()
