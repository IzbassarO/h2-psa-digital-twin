"""Step 7a: EXPLORATORY cost of the guarantee with a FIXED physical reference point (not pre-registered).

    python scripts/explore_step7a.py

No simulations, no refits. Reads the Step 6c labels and memberships; uses the frozen front functions of
`h2psa.step6_metrics` unchanged. Writes results/step7a/ (tables, REPORT.md), docs/validation/step7a.md and
docs/validation/step7a/.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from h2psa.step6_metrics import _observed, feasible_front, hypervolume_2d, join_predictions  # noqa: E402
from report_step6c import md  # noqa: E402

GEN, FROZEN = ROOT / "data" / "generated", ROOT / "docs" / "validation" / "step6b"
OUT, DOCS = ROOT / "results" / "step7a", ROOT / "docs" / "validation" / "step7a"
REFERENCE = (0.0, 0.0)                         # (recovery %, productivity mol/kg/day): no net H2 product
LEVELS = (100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0)   # productivity, mol H2 / kg adsorbent / day


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def recovery_at(front: np.ndarray, p: float) -> float:
    """Recovery of a realised front at productivity p: linear between consecutive front points; below the
    front's lowest productivity, the recovery of that point (it attains p); above its highest, undefined."""
    if not len(front) or p > front[:, 1].max():
        return np.nan
    f = front[np.argsort(front[:, 1])]
    return float(np.interp(p, f[:, 1], f[:, 0])) if p >= f[0, 1] else float(f[0, 0])


def main():
    labels = {name: read(GEN / f"{name}.csv") for name in ("D_train", "D_cal", "D_test", "S_nom")}
    sel = join_predictions(labels["S_nom"], read(FROZEN / "predictions" / "S_nom.csv"),
                           read(GEN / "S_nom_memberships.csv"))
    best = feasible_front(pd.concat(labels.values(), ignore_index=True))
    best_hv = hypervolume_2d(best, REFERENCE)
    fronts, arms, coords = {}, [], []
    for arm, g in sel.groupby("arm", sort=True):
        o, i = _observed(g), g.iloc[0]
        feas = o.loc[o["status"].isin(("css", "cap")) & o["z"].le(2.0)]
        front = feasible_front(o)
        fronts[arm] = front
        coords += [dict(arm=arm, recovery=r, productivity=p) for r, p in front]
        hv = hypervolume_2d(front, REFERENCE)
        arms.append(dict(arm=arm, surrogate=i["surrogate"], n_train=int(i["n_train"]), variant=i["variant"],
                         alpha=float(i["alpha"]), n_observed=len(o), n_feasible=len(feas), n_front=len(front),
                         max_recovery=front[:, 0].max() if len(front) else np.nan,
                         max_productivity=front[:, 1].max() if len(front) else np.nan,
                         hv=hv, hv_fraction_of_best=hv / best_hv))
    arms = pd.DataFrame(arms)
    losses = []
    for _, a in arms.iterrows():
        p_arm = f"{a['surrogate']}_n{a['n_train']}_P_a0.10"
        for level in LEVELS:
            r = recovery_at(fronts[a["arm"]], level)
            rp, rb = recovery_at(fronts[p_arm], level), recovery_at(best, level)
            losses.append(dict(arm=a["arm"], surrogate=a["surrogate"], n_train=a["n_train"], variant=a["variant"],
                               alpha=a["alpha"], productivity=level, recovery=r, recovery_P=rp,
                               loss_vs_P=rp - r, recovery_best=rb, loss_vs_best=rb - r))
    losses = pd.DataFrame(losses)
    for d in (OUT, DOCS):
        d.mkdir(parents=True, exist_ok=True)
        arms.to_csv(d / "arms.csv", index=False)
        losses.to_csv(d / "recovery_loss.csv", index=False)
        pd.DataFrame(coords).to_csv(d / "realised_fronts.csv", index=False)
        pd.DataFrame(best, columns=["recovery", "productivity"]).to_csv(d / "best_known_front.csv", index=False)
    text = report(arms, losses, best, best_hv)
    (OUT / "REPORT.md").write_text(text)
    shutil.copy(OUT / "REPORT.md", ROOT / "docs" / "validation" / "step7a.md")
    print(text)


def report(arms, losses, best, best_hv):
    def wide(frame, column):
        t = frame.pivot_table(index=["arm"], columns="productivity", values=column, dropna=False, sort=False)
        t.columns = [f"p = {c:g}" for c in t.columns]
        return t.reset_index()

    sections = []
    for s in ("GP", "GBM"):
        a = arms.loc[arms["surrogate"] == s].sort_values(["alpha", "n_train", "variant"], ascending=[False, True, True])
        order = a["arm"].tolist()
        l = losses.loc[losses["surrogate"] == s].set_index("arm").loc[order].reset_index()
        gl = l.loc[l["variant"] != "P"]
        sections.append(f"""## {s}

Per arm: re-simulated points, feasible points (CSS or cap, z <= 2), realised front, hypervolume above the fixed
reference point, and HV as a fraction of the best-known front's HV.

{md(a[["arm", "n_observed", "n_feasible", "n_front", "max_recovery", "max_productivity", "hv", "hv_fraction_of_best"]])}

Realised recovery (%) at matched productivity (mol/kg/day); "–" = not reached:

{md(wide(l, "recovery"))}

H2 recovery loss of G and L vs P with the same surrogate and n_train (pp; positive = G/L recovers less; "–" = one
of the two fronts does not reach that productivity):

{md(wide(gl, "loss_vs_P"))}

Loss vs the best-known feasible front (pp):

{md(wide(l, "loss_vs_best"))}
""")
    return f"""# Step 7a — cost of the guarantee at a fixed physical reference point (EXPLORATORY, not pre-registered)

**Everything in this report is EXPLORATORY.** It is not part of the pre-registered analysis (prereg-v1) and replaces
none of the Step 6c results: the pre-registered HV ratio and recovery loss (section 7, normalised to the best-known
front) stay as reported in `docs/validation/step6c.md`. No simulations, no refits; inputs are the Step 6c labels and
memberships. Generated by `scripts/explore_step7a.py`; tables in `docs/validation/step7a/`.

## Reference point and definitions

* **Reference point: recovery 0 %, productivity 0 mol H2 / kg / day.** Both objectives are net H2 measures
  (eq. 5.49: H2 product minus the H2 used for purge and pressurisation; eq. 5.50: net H2 per kg adsorbent per day).
  The origin is the operating point that delivers no net H2, so every point that delivers on-spec H2 at all
  dominates it. It is fixed by physics, not by any arm's result. The design bounds give no positive floor. Inside
  the design box, net H2 can be negative: purge and pressurisation consume more H2 than the product delivers. On
  the random CSS points of D_train, D_cal and D_test, recovery ranges from -190 to 94 % and productivity from -62 to
  246. At the lowest feed flow (u_feed_flow < 0.1, 84 points) productivity spans -62 to 52. A "productivity at
  the lower design bound" is therefore not a constant of the design space. Unlike the Step 6c normalisation
  (box spanned by the best-known front, 11 of whose 12 points come from GP_n512_P), every feasible point of every
  arm counts. Points with recovery or productivity <= 0 add nothing.
* **Hypervolume** (% x mol/kg/day) of the realised front = non-dominated set of an arm's feasible re-simulated
  points (CSS or cap, z <= 2), using the frozen `hypervolume_2d`. Best-known front: all feasible nominal points of
  D_train, D_cal, D_test and S_nom ({len(best)} points), HV = {best_hv:.0f}.
* **Recovery at matched productivity p:** linear interpolation between consecutive realised-front points. Below the
  front's lowest productivity, the recovery of that point (it attains p). Above its highest productivity, not
  reached (no extrapolation). Levels: {", ".join(f"{v:g}" for v in LEVELS)} mol/kg/day. Loss vs P = recovery of
  P's realised front minus that of G / L at the same p, same surrogate and n_train (alpha = 0.05 arms vs P at
  n = 512).
* **Caveat on P as the baseline.** P's realised front contains only those P points that turned out to be feasible
  on re-simulation (P violates at 92 %). An operator using P cannot tell which they are without simulating. "Loss
  vs P" is therefore the cost relative to an oracle-filtered P, an upper bound on the cost of the guarantee. Loss vs
  the best-known front is the second baseline.
* The interpolated segment between two realised points is not itself a simulated point. Recoveries are
  single-column values (Step 4), below multi-bed industrial values.

## Findings

* **GP.**
  - The bounded fronts keep 68 to 85 % of the best-known HV: G 73 to 85 %, L 68 to 78 %, alpha = 0.05 G 80 % and
    L 74 %. P keeps 0 % at n = 128 (no feasible point), 92 % at n = 256 (one feasible point) and 98 % at n = 512.
  - At n = 512 the cost of the guarantee against the oracle-filtered P is 2.9 to 6.7 pp of H2 recovery for G and
    4.2 to 9.2 pp for L, rising with productivity. Neither reaches 250 mol/kg/day, and L does not reach 225. At
    alpha = 0.05 the cost is 3.3 to 7.8 pp (G) and 5.3 to 11.0 pp (L).
  - At n = 256, G and L recover more than P up to 175 mol/kg/day: P's single feasible point has 77.2 %. At
    n = 128 only the bounded variants deliver any feasible point.
* **GBM.**
  - P has no feasible point at any n_train, so the loss vs P is undefined. The guarantee does not cost recovery
    against P, because P delivers nothing usable.
  - GBM-G keeps 70 to 85 % of the best-known HV, within 4.0 to 12.5 pp of the best-known recovery.
  - GBM-L has no feasible point except at n = 512 and alpha = 0.10 (3 points, 52 %, productivity <= 149). This is
    consistent with the quantile crossing of Step 6d.
* The normalised Step 6c HV ratio was 0 for every G/L arm because their fronts lie below the best-known front's
  productivity box (< 217). At a fixed physical reference point they hold most of the hypervolume. The cost of the
  guarantee lies at the high-productivity end of the front: the bounded fronts stop at 190 to 237 mol/kg/day,
  against 261 to 267 for the lucky feasible P points.

{chr(10).join(sections)}
"""


if __name__ == "__main__":
    main()
