"""Step 7c: scripts/make_figures.py regenerates every figure and table from committed data, end to end."""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_figures", ROOT / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)


def test_make_figures_end_to_end(tmp_path):
    assert mf.main(["--out", str(tmp_path)]) == 0
    figures = {p.stem for p in (tmp_path / "figures").glob("F*.pdf")}
    assert figures == {p.stem for p in (tmp_path / "figures").glob("F*.png")} and len(figures) == 7
    for name in mf.CAPTIONS:
        assert (tmp_path / "tables" / f"figure_data_{name}.csv").stat().st_size > 0
    for name in ("T1_design_space", "T2_preregistered_results", "T3_model_form_scenarios"):
        assert (tmp_path / "tables" / f"{name}.csv").exists() and "\\bottomrule" in (tmp_path / "tables" / f"{name}.tex").read_text()
    f7 = pd.read_csv(tmp_path / "tables" / "figure_data_F7.csv")
    k = f7.dropna(subset=["arm"]).groupby("variant")["violation"].sum()
    assert dict(k) == {"P": 29, "G": 22, "L": 0}                   # Step 6c, GP n_train = 256, alpha = 0.10
    fs7 = pd.read_csv(tmp_path / "tables" / "figure_data_FS7.csv")
    assert dict(fs7.groupby("variant")["violation"].sum()) == {"P": 17, "G": 1, "L": 0}     # n_train = 512
    t3 = pd.read_csv(tmp_path / "tables" / "T3_model_form_scenarios.csv").set_index("key")
    assert t3.loc["ep_inflow_ed_gas", "P_a0.10_k"] == 30 and (t3["css"] == 210).all()
    rep = f7.loc[f7["group"] == "replication 8a (prereg.)"]
    assert list(zip(rep["surrogate"], rep["variant"], rep["k"])) == [("GP", "G", 51), ("GP", "L", 1), ("GBM", "L", 130)]
    t2 = pd.read_csv(tmp_path / "tables" / "T2_preregistered_results.csv").set_index("RQ")
    assert t2.loc[["H_B1", "H_B2", "H_B3"], "verdict"].eq("confirmed").all()       # Step 8a, Addendum B
