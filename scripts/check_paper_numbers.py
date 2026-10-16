"""Recompute every number quoted in the paper from the committed CSVs and compare with the manuscript.

    python scripts/check_paper_numbers.py        # ~2.5 min (the registered 2000-draw cluster bootstrap of Step 6c)

Inputs: data/generated/*.csv and the frozen Step 6b predictions / reports in docs/validation/step6b/ (their SHA-256
manifest is verified; a Python / library version other than the prereg-v1 pins only prints a warning). No simulations, no refits, nothing is written. The statistics are the frozen functions of
`h2psa.step6_metrics` and of scripts/analyse_step7b.py, analyse_step8a.py and explore_step7a.py, called unchanged.

PAPER holds each value as printed in the manuscript; a recomputed value matches when it prints the same at the
paper's precision (counts "k/n" and verdicts must be identical). Exit code 1 on any mismatch.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from h2psa.step6_metrics import (_observed, analyse_generated, clopper_pearson, feasible_front,  # noqa: E402
                                 hypervolume_2d, join_predictions, newcombe_difference, verify_frozen_manifest)
import analyse_step7b  # noqa: E402
import analyse_step8a  # noqa: E402
from explore_step7a import LEVELS, REFERENCE, recovery_at  # noqa: E402

GEN, FROZEN = ROOT / "data" / "generated", ROOT / "docs" / "validation" / "step6b"

# Values as printed in the manuscript (rates in % where the paper uses %). Section in brackets.
PAPER = {
    # [Abstract] CSS simulations: D_train/cal/test/shift + S_nom + S_dist + Step 7b (M_*) + Step 8a (S_rep)
    "CSS simulations, total": "3357",
    # [4.2, Table 4] RQ1
    "RQ1 P violations": "165/179",
    "RQ1 P rate %": "92.2",
    "RQ1 95% CI lower %": "87.2",
    "RQ1 95% CI upper %": "95.7",
    "RQ1 Holm-adjusted p": "2.2e-145",
    "RQ1 verdict": "confirmed",
    "RQ1 bootstrap": "robust",
    # [4.3, Table 4] RQ2
    "RQ2 G violations": "104/270",
    "RQ2 G rate %": "38.5",
    "RQ2 95% CI lower": "0.327",
    "RQ2 95% CI upper": "0.446",
    "RQ2 Holm-adjusted p": "6.6e-35",
    "RQ2 verdict": "confirmed",
    "RQ2 bootstrap": "robust",
    "band-matched G: band coverage": "0.90",
    "band-matched G: selected coverage": "0.60",
    "band-matched G: difference": "0.30",
    "band-matched G: CI lower": "0.22",
    "band-matched G: CI upper": "0.37",
    # [4.4, Table 4] RQ3
    "RQ3 L violations": "137/270",
    "RQ3 L rate %": "50.7",
    "RQ3 G - L": "-0.122",
    "RQ3 G - L CI lower": "-0.204",
    "RQ3 G - L CI upper": "-0.038",
    "RQ3 verdict": "not confirmed",
    # [Table 4, 4.7] RQ5
    "RQ5 b_phi": "-1.57",
    "RQ5 b_phi CI lower": "-2.44",
    "RQ5 b_phi CI upper": "-0.69",
    "RQ5 OR phi 0.70 vs 1.00": "4.79",
    "RQ5 OR CI lower": "1.99",
    "RQ5 OR CI upper": "11.52",
    "RQ5 Holm-adjusted p": "4.6e-04",
    "RQ5 verdict": "confirmed",
    "RQ5 bootstrap": "robust",
    "S_dist paired: feasible -> off-spec": "29",
    "S_dist paired: off-spec -> feasible": "1",
    "S_dist McNemar p": "3e-08",
    "S_dist GP-L": "0/45",
    "S_dist GP-G": "7/45",
    "S_dist GBM-G": "26/45",
    "S_dist GP-P": "27/30",
    "S_dist GBM-P": "30/30",
    # [4.4, Table 5] Step 6d split by surrogate (exploratory, alpha 0.10, pooled over n_train)
    "6d GP-P": "76/90",
    "6d GP-G": "51/135",
    "6d GP-L": "5/135",
    "6d GP-L rate %": "3.7",
    "6d GP-L CI lower %": "1.2",
    "6d GP-L CI upper %": "8.4",
    "6d GBM-P": "89/89",
    "6d GBM-G": "53/135",
    "6d GBM-L": "132/135",
    "6d GBM-L rate %": "97.8",
    # [4.6] Step 7a cost of the guarantee (GP, n_train 512, loss vs oracle-filtered P, percentage points)
    "7a HV fraction GP-G %": "85",
    "7a HV fraction GP-L %": "78",
    "7a HV fraction GP-P %": "98",
    "7a loss G a0.10 min pp": "2.9",
    "7a loss G a0.10 max pp": "6.7",
    "7a loss L a0.10 min pp": "4.2",
    "7a loss L a0.10 max pp": "9.2",
    "7a loss G a0.05 min pp": "3.3",
    "7a loss G a0.05 max pp": "7.8",
    "7a loss L a0.05 min pp": "5.3",
    "7a loss L a0.05 max pp": "11.0",
    "7a loss GP-L a0.10, abstract range min pp": "4",
    "7a loss GP-L a0.10, abstract range max pp": "9",
    # [4.7] Step 7b model-form robustness (descriptive, GP n_train 512)
    "7b EP=ED gas: median dz (log10 ppm)": "0.55",
    "7b EP=ED gas: P": "30/30",
    "7b baseline: P": "17/30",
    "7b EP=ED gas: G a0.10": "4/45",
    "7b EP=ED gas: G a0.05": "2/45",
    "7b EP=ED gas: L a0.10": "0/45",
    "7b EP=ED gas: L a0.05": "0/45",
    "7b EP=ED gas: G a0.10 coverage": "0.91",
    "7b EP=ED gas: G a0.05 coverage": "0.96",
    "7b thermal low: P": "21/30",
    "7b thermal high: P": "17/30",
    # [4.5] Step 8a confirmatory replication (Addendum B); the manuscript draft still has placeholders here,
    # the values are those of docs/validation/step8a.md
    "8a H_B1 GP-L": "1/135",
    "8a H_B1 CI upper": "0.041",
    "8a H_B1 verdict": "confirmed",
    "8a H_B2 GBM-L": "130/135",
    "8a H_B2 CI lower": "0.916",
    "8a H_B2 verdict": "confirmed",
    "8a H_B3 GP-G": "51/135",
    "8a H_B3 verdict": "confirmed",
    "8a bootstrap (all three)": "robust",
}


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def kn(k, n) -> str:
    return f"{int(k)}/{int(n)}"


def matches(paper: str, value) -> bool:
    if isinstance(value, str):
        return value == paper
    if not np.isfinite(value):
        return False
    if "e" in paper:
        mantissa = paper.split("e")[0]
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return f"{value:.{digits}e}" == f"{float(paper):.{digits}e}"
    digits = len(paper.split(".")[1]) if "." in paper else 0
    return f"{value:.{digits}f}" == paper


def step6() -> dict:
    """RQ1-RQ5 (pre-registered, Step 6c) and the surrogate split (Step 6d) from the frozen analysis."""
    try:                       # the registered analysis also pins Python 3.14.7 and the library versions
        verify_frozen_manifest(GEN, FROZEN / "predictions", FROZEN, require_manifest=True)
    except ValueError as err:
        print(f"WARNING: {err} (requirements-lock.txt, Python 3.14.7); input hashes are still verified.\n")
    r = analyse_generated(GEN, FROZEN / "predictions", FROZEN)
    res, t = r["results"], r["tables"]
    dec, pooled, boot = res["tests"]["decisions"], res["tests"]["pooled_counts"], res["robustness"]
    P, G, L = pooled["P"], pooled["G"], pooled["L"]
    diff, diff_lo, diff_hi = newcombe_difference(G["k"], G["n"], L["k"], L["n"])
    band = t["pooled_bands"].set_index("variant").loc["G"]
    lg, mc = res["logistic"], res["mcnemar"]
    v = t["violations"]
    dist = v.loc[v["set"] == "S_dist"].set_index("arm")
    nom = v.loc[(v["set"] == "S_nom") & np.isclose(v["alpha"], 0.10)]
    split = nom.groupby(["surrogate", "variant"])[["k", "n"]].sum()
    out = {
        "RQ1 P violations": kn(P["k"], P["n"]), "RQ1 P rate %": 100 * P["rate"],
        "RQ1 95% CI lower %": 100 * P["ci_lower"], "RQ1 95% CI upper %": 100 * P["ci_upper"],
        "RQ1 Holm-adjusted p": dec["RQ1"]["adjusted_p"], "RQ1 verdict": dec["RQ1"]["verdict"],
        "RQ1 bootstrap": boot["RQ1"]["robustness"],
        "RQ2 G violations": kn(G["k"], G["n"]), "RQ2 G rate %": 100 * G["rate"],
        "RQ2 95% CI lower": G["ci_lower"], "RQ2 95% CI upper": G["ci_upper"],
        "RQ2 Holm-adjusted p": dec["RQ2"]["adjusted_p"], "RQ2 verdict": dec["RQ2"]["verdict"],
        "RQ2 bootstrap": boot["RQ2"]["robustness"],
        "band-matched G: band coverage": band["band_coverage"],
        "band-matched G: selected coverage": band["selected_coverage"],
        "band-matched G: difference": band["difference"],
        "band-matched G: CI lower": band["difference_lower"], "band-matched G: CI upper": band["difference_upper"],
        "RQ3 L violations": kn(L["k"], L["n"]), "RQ3 L rate %": 100 * L["rate"],
        "RQ3 G - L": diff, "RQ3 G - L CI lower": diff_lo, "RQ3 G - L CI upper": diff_hi,
        "RQ3 verdict": dec["RQ3"]["verdict"],
        "RQ5 b_phi": lg["b_phi"], "RQ5 b_phi CI lower": lg["b_phi_lower"], "RQ5 b_phi CI upper": lg["b_phi_upper"],
        "RQ5 OR phi 0.70 vs 1.00": lg["odds_ratio"],
        "RQ5 OR CI lower": lg["odds_ratio_lower"], "RQ5 OR CI upper": lg["odds_ratio_upper"],
        "RQ5 Holm-adjusted p": dec["RQ5"]["adjusted_p"], "RQ5 verdict": dec["RQ5"]["verdict"],
        "RQ5 bootstrap": boot["RQ5"]["robustness"],
        "S_dist paired: feasible -> off-spec": str(mc["increase"]),
        "S_dist paired: off-spec -> feasible": str(mc["decrease"]), "S_dist McNemar p": mc["p_value"],
    }
    for name, arm in (("GP-L", "GP_n512_L_a0.10"), ("GP-G", "GP_n512_G_a0.10"), ("GBM-G", "GBM_n512_G_a0.10"),
                      ("GP-P", "GP_n512_P_a0.10"), ("GBM-P", "GBM_n512_P_a0.10")):
        out[f"S_dist {name}"] = kn(dist.loc[arm, "k"], dist.loc[arm, "n"])
    for (surrogate, variant), row in split.iterrows():
        out[f"6d {surrogate}-{variant}"] = kn(row["k"], row["n"])
    k, n = split.loc[("GP", "L")]
    lo, hi = clopper_pearson(int(k), int(n))
    out.update({"6d GP-L rate %": 100 * k / n, "6d GP-L CI lower %": 100 * lo, "6d GP-L CI upper %": 100 * hi,
                "6d GBM-L rate %": 100 * split.loc[("GBM", "L"), "k"] / split.loc[("GBM", "L"), "n"]})
    return out


def step7a() -> dict:
    """Cost of the guarantee, GP n_train 512 (explore_step7a.py, same functions and reference point)."""
    labels = {name: read(GEN / f"{name}.csv") for name in ("D_train", "D_cal", "D_test", "S_nom")}
    sel = join_predictions(labels["S_nom"], read(FROZEN / "predictions" / "S_nom.csv"),
                           read(GEN / "S_nom_memberships.csv"))
    best = feasible_front(pd.concat(labels.values(), ignore_index=True))
    fronts = {arm: feasible_front(_observed(g)) for arm, g in sel.groupby("arm")}
    hv = {arm: hypervolume_2d(f, REFERENCE) / hypervolume_2d(best, REFERENCE) for arm, f in fronts.items()}
    out = {f"7a HV fraction GP-{v} %": 100 * hv[f"GP_n512_{v}_a0.10"] for v in "PGL"}
    for v in "GL":
        for a in ("0.10", "0.05"):
            loss = np.array([recovery_at(fronts["GP_n512_P_a0.10"], p) - recovery_at(fronts[f"GP_n512_{v}_a{a}"], p)
                             for p in LEVELS])
            out[f"7a loss {v} a{a} min pp"], out[f"7a loss {v} a{a} max pp"] = np.nanmin(loss), np.nanmax(loss)
    out["7a loss GP-L a0.10, abstract range min pp"] = out["7a loss L a0.10 min pp"]
    out["7a loss GP-L a0.10, abstract range max pp"] = out["7a loss L a0.10 max pp"]
    return out


def step7b() -> dict:
    t = analyse_step7b.analyse(GEN)
    v = t["violations"].set_index(["set", "arm"])
    c = t["coverage"].set_index(["set", "arm"])
    ep = "ep_inflow_ed_gas"
    base, scen = read(GEN / "S_nom.csv")[["id", "z"]], read(GEN / f"M_{ep}.csv")[["id", "z"]]
    dz = scen.merge(base, on="id", suffixes=("", "_base"), validate="one_to_one").eval("z - z_base")
    out = {"7b EP=ED gas: median dz (log10 ppm)": float(dz.median())}
    for name, s, arm in (("EP=ED gas: P", ep, "P_a0.10"), ("baseline: P", "baseline", "P_a0.10"),
                         ("EP=ED gas: G a0.10", ep, "G_a0.10"), ("EP=ED gas: G a0.05", ep, "G_a0.05"),
                         ("EP=ED gas: L a0.10", ep, "L_a0.10"), ("EP=ED gas: L a0.05", ep, "L_a0.05"),
                         ("thermal low: P", "thermal_range_low", "P_a0.10"),
                         ("thermal high: P", "thermal_range_high", "P_a0.10")):
        row = v.loc[(s, f"GP_n512_{arm}")]
        out[f"7b {name}"] = kn(row["k"], row["n"])
    for a in ("0.10", "0.05"):
        out[f"7b EP=ED gas: G a{a} coverage"] = c.loc[(ep, f"GP_n512_G_a{a}"), "rate"]
    return out


def step8a() -> dict:
    t = analyse_step8a.analyse(GEN)
    pooled = t["pooled"].set_index("group")
    hyp = t["hypotheses"].set_index("hypothesis")
    boot = t["bootstrap"]["robustness"]
    out = {"8a bootstrap (all three)": "robust" if boot.eq("robust").all() else ", ".join(boot)}
    for h, group in (("H_B1", "GP_L"), ("H_B2", "GBM_L"), ("H_B3", "GP_G")):
        row = pooled.loc[group]
        out[f"8a {h} {group.replace('_', '-')}"] = kn(row["k"], row["n"])
        out[f"8a {h} verdict"] = hyp.loc[h, "verdict"]
    out["8a H_B1 CI upper"] = pooled.loc["GP_L", "ci_upper"]
    out["8a H_B2 CI lower"] = pooled.loc["GBM_L", "ci_lower"]
    return out


def main() -> int:
    blocks = ["D_train", "D_cal", "D_test", "D_shift", "S_nom", "S_dist", "S_rep",
              *(f"M_{s}{c}" for s in analyse_step7b.SCENARIOS for c in ("", "_cold"))]
    got = {"CSS simulations, total": str(sum(len(read(GEN / f"{b}.csv")) for b in blocks))}
    for part in (step6, step7a, step7b, step8a):
        got.update(part())
    rows = [(name, paper, got.get(name, "MISSING")) for name, paper in PAPER.items()]
    width = max(len(name) for name in PAPER)
    print(f"{'quantity':<{width}} | {'paper':>14} | {'recomputed':>22} | OK")
    print(f"{'-' * width}-|-{'-' * 14}-|-{'-' * 22}-|---")
    bad = 0
    for name, paper, value in rows:
        ok = value != "MISSING" and matches(paper, value)
        bad += not ok
        shown = value if isinstance(value, str) else f"{value:.6g}"
        print(f"{name:<{width}} | {paper:>14} | {shown:>22} | {'OK' if ok else 'MISMATCH'}")
    print(f"\n{len(rows) - bad}/{len(rows)} paper numbers reproduced" + ("" if not bad else f"; {bad} MISMATCH"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
