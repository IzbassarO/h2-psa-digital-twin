"""Step 3: the two PSA boundary-condition constants the thesis does not report.

1. beta_P, the rate constant of the exponential pressure law (eqs. 5.39-5.40), fitted ONLY to the measured
   pressure history of cycle 1 (Fig. 5.18a, digitised). This is a valve characteristic: the law is imposed at
   the open end of the column, so the fit needs no column model. One beta_P for ED1-3, Blow and EP3-1 (p.180);
   every step starts from the pressure the law reached at the end of the previous one. Objective: sum over the
   seven steps of the mean squared pressure residual in the step (each step weighs equally).
2. The purge H2 flow, fitted to the H2 molar flow leaving at z = 0 during the purge of cycle 1 (Fig. 5.19 1.a,
   GC samples), with the column model in the baseline configuration and beta_P from 1. Cycle 37 is a check
   (scripts/run_psa.py). No purity or recovery information is used.

Writes results/step3/fit_bcs.csv and prints the `fitted:` block for compact_column_lopes2010.yaml.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from h2psa.cycle import CycleRunner, experiment_cycle, psa_model

ROOT = Path(__file__).resolve().parents[1]
DIG = ROOT / "data" / "benchmarks" / "digitized"
OUT = ROOT / "results" / "step3"
EXP_STEPS = ("ED1", "ED2", "ED3", "Blow", "EP3", "EP2", "EP1")


def read(name):
    return pd.read_csv(DIG / f"lopes2010_{name}.csv", comment="#")


def pressure_law(c, beta, t):
    """Pressure [bar] imposed at the open end over one cycle (laws of cycle.py), start P_high."""
    P, t0, out = c.P_high, 0.0, np.full(len(t), np.nan)
    for name, dur, k in c.steps:
        s = (t >= t0) & (t < t0 + dur)
        tau = t[s] - t0
        if name == "Feed":
            out[s], P = c.P_high, c.P_high
        elif name == "Purge":
            out[s], P = c.P_low, c.P_low
        elif name == "Press":
            out[s], P = P + (c.P_high - P) * tau / dur, c.P_high
        else:
            end = c.P_low if name == "Blow" else c.P_inter[k - 1]
            out[s] = end + (P - end) * np.exp(-beta * tau)
            P = end + (P - end) * np.exp(-beta * dur)
        t0 += dur
    return out / 1e5


def step_of(c, t):
    edges = np.cumsum([0] + [d for _, d, _ in c.steps])
    names = [n for n, _, _ in c.steps]
    return np.array([names[min(np.searchsorted(edges, x, side="right") - 1, len(names) - 1)] for x in t])


def fit_beta(c, t, P, steps=EXP_STEPS):
    st = step_of(c, t)
    sel = np.isin(st, steps)
    w = np.array([1.0 / np.sqrt((st[sel] == s).sum()) for s in st[sel]])

    def res(x):
        return w * (pressure_law(c, np.exp(x[0]), t[sel]) - P[sel])

    grid = np.exp(np.linspace(np.log(1e-2), np.log(30.0), 200))  # global search first: the objective is not convex
    obj = np.array([np.sum(res([np.log(b)]) ** 2) for b in grid])
    f = least_squares(res, [np.log(grid[obj.argmin()])], bounds=([np.log(1e-3)], [np.log(50.0)]))
    best = 2 * f.cost
    # interval: objective within one step-equivalent variance of the minimum (each step = one observation)
    ok = grid[obj <= best * (1 + 1 / max(len(steps) - 1, 1))]
    return np.exp(f.x[0]), (ok.min(), ok.max()), best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = experiment_cycle(beta_P=np.nan, purge_flow=np.nan)
    e = read("psa_c1_P")
    e = e[~((e.time_s > 180) & (e.time_s < 200) & (e.pressure_bar > 4))]   # legend symbol
    t, P = e.time_s.values, e.pressure_bar.values
    beta, (lo, hi), obj = fit_beta(c, t, P)
    rows = [dict(quantity="beta_P", case="all exponential steps", value=beta, lo=lo, hi=hi, objective=obj)]
    print(f"beta_P = {beta:.3f} s-1, interval {lo:.3f}-{hi:.3f} (objective {obj:.4f} bar2)")
    st = step_of(c, t)
    for s in EXP_STEPS:
        b1, _, o1 = fit_beta(c, t, P, steps=(s,))
        rms = np.sqrt(np.mean((pressure_law(c, beta, t[st == s]) - P[st == s]) ** 2))
        rows.append(dict(quantity="beta_P", case=f"{s} alone", value=b1, objective=o1,
                         rms_bar_at_common_beta=rms, n=(st == s).sum()))
        print(f"  {s:5s} alone: beta {b1:.3f} s-1; rms at common beta {rms:.3f} bar (n={(st == s).sum()})")
    a = read("psa_c1_P_authorsim")       # the author's simulated line (stroke-outline vertices), for context
    ab, _, _ = fit_beta(c, a.time_s.values, a.pressure_bar.values)
    rows.append(dict(quantity="beta_P", case="author's simulated line (context only)", value=ab))
    print(f"  author's simulated pressure line: beta {ab:.3f} s-1 (context only)")

    # purge flow from the cycle-1 H2 outflow at z = 0 during the purge
    f = read("psa_c1_flow")
    names = [n for n, _, _ in c.steps]
    t0 = sum(d for _, d, _ in c.steps[:names.index("Purge")])
    h2 = f[(f.series == "H2") & (f.time_s > t0) & (f.time_s < t0 + c.t_purge)]
    cache = {}

    def model_h2(F):
        """Model H2 flow leaving at z = 0 [mmol/s] at the GC sample times of the cycle-1 purge."""
        if F not in cache:
            cyc = experiment_cycle(beta_P=beta, purge_flow=F)
            m = psa_model(cyc)
            tr = CycleRunner(m, cyc).run_cycle(m.y0, trace=True, dt_trace=0.5, until="Purge").trace
            s = tr["step"] == "Purge"
            cache[F] = np.interp(h2.time_s.values, tr["t"][s], -tr["in_0"][m.sp.index("H2"), s] * 1e3)
        return cache[F]

    g = least_squares(lambda x: model_h2(float(x[0]) * 1e-4) - h2.molar_flow_mmol_s.values, [2.0],
                      bounds=([0.1], [20.0]), diff_step=0.05)
    F = float(g.x[0]) * 1e-4
    J = g.jac[:, 0] / 1e-4
    sdF = np.sqrt(2 * g.cost / max(len(h2) - 1, 1) / (J @ J))
    mod = model_h2(F)
    print(f"purge flow = {F * 1e3:.4f} +- {sdF * 1e3:.4f} mmol/s H2 (n = {len(h2)} GC samples; "
          f"measured {np.round(h2.molar_flow_mmol_s.values, 3)}, model {np.round(mod, 3)})")
    rows.append(dict(quantity="purge_flow", case="cycle-1 purge H2 outflow at z = 0", value=F, sd=sdF,
                     objective=2 * g.cost, n=len(h2)))
    rows.append(dict(quantity="purge_flow", case="mean measured H2 outflow during purge (context)",
                     value=h2.molar_flow_mmol_s.mean() * 1e-3))
    pd.DataFrame(rows).to_csv(OUT / "fit_bcs.csv", index=False)
    today = datetime.date.today().isoformat()
    print(f"""
fitted:                          # step 3 ({today}); scripts/fit_psa_bcs.py. Not used: purity, recovery, Table 5.10
  pressure_decay_rate: {{value: {beta:.4g}, interval: [{lo:.3g}, {hi:.3g}], units: s-1,
                        data: "Fig. 5.18a cycle-1 pressure history (digitised), steps ED1-3, Blow, EP3-1",
                        objective: "sum over the 7 steps of the mean squared pressure residual (equal step weights); interval: objective <= min x (1 + 1/6)",
                        note: "fitted alone, ED1-3 give 4.5-9.5 s-1 and Blow, EP3-1 0.08-0.12 s-1: one constant (p.180) cannot follow both"}}
  purge_flow:          {{value: {F:.4g}, sd: {sdF:.2g}, units: mol s-1,
                        data: "Fig. 5.19 1.a, H2 molar flow leaving at z = 0 during the cycle-1 purge ({len(h2)} GC samples)",
                        objective: "least squares, baseline model (E7 kinetics, taqvi_levan, assumed thermal, group 1-4, N = 30)"}}""")


if __name__ == "__main__":
    main()
