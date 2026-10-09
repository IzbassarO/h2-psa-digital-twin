"""Step 2: simulate the compact-unit breakthrough runs 1-5 (Lopes 2010) and compare with experiment.

    python scripts/run_breakthrough.py        # ~5 min; writes results/step2/

Outputs: run{1..5}.png, sim_run{k}.csv, breakthrough_times.csv, temperature_peaks.csv, h2_flow.csv,
numerical_gates.csv, diagnostics.csv, REPORT.md.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.column import ColumnModel, compact_breakthrough  # noqa: E402
from h2psa.isotherms import load_yaml  # noqa: E402
from h2psa.kinetics import arrhenius_fit  # noqa: E402
from h2psa.validation import (TC_Z, NoncompetitiveH2, breakthrough_times, exp_flow,  # noqa: E402
                              exp_temperature_peaks, linear_isothermal_model, load_exp, stoichiometric_time)

OUT = ROOT / "results" / "step2"
N, N2, T_END = 60, 120, 6000.0
RUNS = [1, 2, 3, 4, 5]
TOL_T, TOL_RISE, TOL_TPEAK = 0.10, 0.30, 0.20   # gate: breakthrough time; qualitative T: peak rise, peak time
COLOUR = {"H2": "#2a78d6", "CO2": "#eb6834", "CO": "#1baf7a", "CH4": "#eda100", "N2": "#e87ba4"}
MARKER = {"H2": "^", "CO2": "s", "CO": "x", "CH4": "+", "N2": "D"}     # as in the thesis figures
TC_COLOUR = {"Bottom": "#6da7ec", "Middle": "#2a78d6", "Top": "#104281"}  # ordinal along the bed


def simulate(m: ColumnModel):
    return m.solve(T_END, n_out=int(T_END) + 1)


def feed_flows(m):
    return dict(zip(m.sp, m.feed.flow * m.y_feed))


def with_layer(m: ColumnModel, **kw) -> ColumnModel:
    lay = replace(m.col.layers[0], **kw)
    return ColumnModel(replace(m.col, layers=[lay]), m.feed, m.sp, N=m.N, y0={"He": 1.0})


def table410_kinetics(m: ColumnModel) -> ColumnModel:
    """Diagnostic: micropore kinetics refitted to the measured Dc/rc2 of Table 4.10 instead of Table 5.7."""
    k = load_yaml(ROOT / "data" / "params" / "ac5ks_lopes2010.yaml")["micropore_kinetics_breakthrough"]
    fits = {s: arrhenius_fit(v["Dc_over_rc2"]) for s, v in k.items() if s != "units"}
    return with_layer(m, Dc0={s: f[0] for s, f in fits.items()}, Ea={s: f[1] for s, f in fits.items()})


def err(a, b):
    return (a - b) / b if np.isfinite(a) and np.isfinite(b) else np.nan


# ---------------------------------------------------------------------------- analysis pieces
def bt_rows(run, m, r, label="model"):
    rows = []
    F0 = feed_flows(m)
    for i, s in enumerate(m.sp):
        if s in ("H2", "He"):
            continue
        ts = breakthrough_times(r.t, r.outlet_flow[i] / F0[s])
        e = exp_flow(run, s)
        te = breakthrough_times(e.time_s.values, e.flow.values / F0[s])
        for lev, a, b in (("t10", ts[0], te[0]), ("t50", ts[1], te[1])):
            rows.append(dict(run=run, case=label, species=s, level=lev, sim_s=a, exp_s=b, rel_err=err(a, b)))
    return rows


def h2_metrics(run, m, r, label="model"):
    """H2 outlet flow relative to the H2 feed flow, in windows defined by the measured CO2 front."""
    F0 = feed_flows(m)["H2"]
    e = exp_flow(run, "H2")
    c = exp_flow(run, "CO2")
    t10, t50 = breakthrough_times(c.time_s.values, c.flow.values / feed_flows(m)["CO2"])
    fs = r.outlet_flow[m.sp.index("H2")] / F0
    out = dict(run=run, case=label, co2_t10_exp=t10)
    for name, lo, hi in (("before", 0.0, t10 - 30), ("peak", t10 - 150, t50 + 300), ("late", 2500.0, T_END)):
        w = (e.time_s >= lo) & (e.time_s <= hi)
        ws = (r.t >= lo) & (r.t <= hi)
        fe = e.flow.values[w] / F0
        if name == "peak":
            out["exp_peak"], out["sim_peak"] = fe.max(), fs[ws].max()
        else:
            out[f"exp_{name}"] = fe.mean()
            out[f"sim_{name}"] = np.interp(e.time_s.values[w], r.t, fs).mean()
        out[f"n_{name}"] = int(w.sum())
    return out


def t_rows(run, m, r):
    pk = exp_temperature_peaks(run)
    rows = []
    for tc, z in TC_Z.items():
        T = r.T_at(z)
        k = int(np.argmax(T))
        Te, te = pk[tc]
        rise_s, rise_e = T[k] - m.feed.T, Te - m.feed.T
        rows.append(dict(run=run, tc=tc, z_m=z, sim_peak_K=T[k], exp_peak_K=Te, sim_rise_K=rise_s, exp_rise_K=rise_e,
                         rise_rel_err=err(rise_s, rise_e), sim_time_s=r.t[k], exp_time_s=te, time_rel_err=err(r.t[k], te)))
    return rows


def author_rows(run, m, r):
    """Our model vs the author's gPROMS simulation lines (same equations, Table 5.7 parameters)."""
    F0 = feed_flows(m)
    rows = []
    for i, s in enumerate(m.sp):
        if s in ("H2", "He"):
            continue
        panel = "zoom" if s != "CO2" and load_exp(run, "zoom") is not None else "flow"
        a = load_exp(run, f"{panel}_authorsim")
        a = a[a.series == s]
        f = a.flow.values / F0[s]
        ta = [a.time_s.values[f >= lev].min() if (f >= lev).any() else np.nan for lev in (0.1, 0.5)]
        ts = breakthrough_times(r.t, r.outlet_flow[i] / F0[s])
        for lev, x, y in zip(("t10", "t50"), ts, ta):
            rows.append(dict(run=run, item=f"{s} {lev} [s]", ours=x, author=y, rel_diff=err(x, y)))
    a = load_exp(run, "T_authorsim")
    for tc, z in TC_Z.items():
        g = a[a.series == tc]
        k = int(np.argmax(g.temperature_K.values))
        T = r.T_at(z)
        rows.append(dict(run=run, item=f"T peak {tc} [K]", ours=T.max(), author=g.temperature_K.values[k],
                         rel_diff=err(T.max() - m.feed.T, g.temperature_K.values[k] - m.feed.T)))
    return rows


def thermal_sensitivity(au: pd.DataFrame):
    """Runs 1 and 3: thermocouple peaks and CO2 front under the assumed-value ranges (diagnostic, not adopted)."""
    rows = []
    for run in (1, 3):
        base = compact_breakthrough(run, N=N)
        thermal = lambda cps, e: ColumnModel(replace(base.col, e_wall=e, layers=[replace(base.col.layers[0], Cps=cps)]),
                                             base.feed, base.sp, N=N, y0={"He": 1.0})
        cases = [("model (Cps 880, wall 3 mm)", base), ("Cps 700 (lower end of assumed range)", thermal(700.0, 0.003)),
                 ("wall 2 mm (lower end of assumed range)", thermal(880.0, 0.002)),
                 ("wall 0.5 mm (~no wall)", thermal(880.0, 0.0005)), ("Cps 700 + wall 0.5 mm", thermal(700.0, 0.0005))]
        for label, m in cases:
            r = m.solve(2000.0, n_out=2001)
            t10, t50 = breakthrough_times(r.t, r.outlet_flow[m.sp.index("CO2")] / feed_flows(m)["CO2"])
            rows.append({"run": run, "case": label, **{f"T {tc} [K]": r.T_at(z).max() for tc, z in TC_Z.items()},
                         "CO2 t10 [s]": t10, "CO2 t50 [s]": t50})
        a = au[au.run == run].set_index("item").author
        rows.append({"run": run, "case": "author's gPROMS simulation", **{f"T {tc} [K]": a[f"T peak {tc} [K]"] for tc in TC_Z},
                     "CO2 t10 [s]": a["CO2 t10 [s]"], "CO2 t50 [s]": a["CO2 t50 [s]"]})
        pk, c = exp_temperature_peaks(run), exp_flow(run, "CO2")
        t10, t50 = breakthrough_times(c.time_s.values, c.flow.values / feed_flows(base)["CO2"])
        rows.append({"run": run, "case": "experiment", **{f"T {tc} [K]": pk[tc][0] for tc in TC_Z},
                     "CO2 t10 [s]": t10, "CO2 t50 [s]": t50})
    return pd.DataFrame(rows)


def plot_run(run, m, r, path):
    F0 = feed_flows(m)
    minor = [s for s in m.sp if s in ("CO", "CH4", "N2")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for i, s in enumerate(m.sp):
        if s == "He":
            continue
        for a in ax[:2]:
            a.plot(r.t, r.outlet_flow[i] * 1e3, color=COLOUR[s], lw=1.8, label=f"{s} model")
        e_main, e = load_exp(run, "flow"), exp_flow(run, s)
        e_main = e_main[e_main.series == s]
        kw = dict(color=COLOUR[s], marker=MARKER[s], ls="none", ms=6, mfc="none", mew=1.3)
        ax[0].plot(e_main.time_s, e_main.flow * 1e3, label=f"{s} exp.", **kw)
        ax[1].plot(e.time_s, e.flow * 1e3, **kw)
        for a_, pan in ((ax[0], "flow"), (ax[1], "zoom" if minor else "flow")):
            au = load_exp(run, f"{pan}_authorsim")
            au = au[au.series == s]
            a_.plot(au.time_s, au.flow * 1e3, ".", ms=0.8, color=COLOUR[s], alpha=0.4)
    ax[0].set(xlim=(0, T_END), ylim=(0, None), title=f"Run {run}: outlet molar flow")
    if minor:
        ax[1].set(xlim=(0, 1500), ylim=(0, 0.16), title="Zoom: low flow range (as Fig. 5.16/5.17)")
    else:
        ax[1].set(xlim=(0, 1500), ylim=(0, None), title="Zoom: first 1500 s")
    for a in ax[:2]:
        a.set(xlabel="time [s]", ylabel="molar flow [mmol/s]")
    ax[0].legend(fontsize=8, ncol=2, frameon=False)
    d = load_exp(run, "T")
    for tc, z in TC_Z.items():
        g = d[d.series == tc]
        ax[2].plot(g.time_s, g.temperature_K, ".", ms=1.5, color=TC_COLOUR[tc], alpha=0.35)
        ax[2].plot(r.t, r.T_at(z), color=TC_COLOUR[tc], lw=1.8, label=f"{tc} z={z} m")
        au = load_exp(run, "T_authorsim")
        au = au[au.series == tc]
        ax[2].plot(au.time_s, au.temperature_K, ".", ms=0.8, color="#888780", alpha=0.5)
    ax[2].set(xlim=(0, T_END), xlabel="time [s]", ylabel="gas temperature [K]",
              title="Thermocouple temperatures (grey: author model)")
    fig.suptitle("symbols: experiment · lines: this model · fine dots: author's gPROMS simulation", fontsize=9)
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bt, tp, h2, gates, au, runtime = [], [], [], [], [], []

    for run in RUNS:
        m = compact_breakthrough(run, N=N)
        r = simulate(m)
        print(f"run {run}: {r.runtime_s:.1f} s, closure {r.closure['total']:.1e}", flush=True)
        runtime.append(dict(run=run, N=N, runtime_s=r.runtime_s, nfev=r.nfev))
        pd.DataFrame({"time_s": r.t, **{f"{s}_mol_s": r.outlet_flow[i] for i, s in enumerate(m.sp)},
                      **{f"T_{tc}_K": r.T_at(z) for tc, z in TC_Z.items()}}).to_csv(OUT / f"sim_run{run}.csv", index=False)
        bt += bt_rows(run, m, r)
        tp += t_rows(run, m, r)
        h2.append(h2_metrics(run, m, r))
        au += author_rows(run, m, r)
        plot_run(run, m, r, OUT / f"run{run}.png")
        for k, v in r.closure.items():
            gates.append(dict(gate="mole balance", run=run, item=k, value=v, limit=1e-3, passed=abs(v) < 1e-3))

        # grid convergence N -> 2N
        m2 = compact_breakthrough(run, N=N2)
        r2 = simulate(m2)
        runtime.append(dict(run=run, N=N2, runtime_s=r2.runtime_s, nfev=r2.nfev))
        F0 = feed_flows(m)
        for i, s in enumerate(m.sp):
            if s in ("H2", "He"):
                continue
            a = breakthrough_times(r.t, r.outlet_flow[i] / F0[s])[1]
            b = breakthrough_times(r2.t, r2.outlet_flow[i] / F0[s])[1]
            gates.append(dict(gate="grid t50 N=60->120", run=run, item=s, value=err(b, a), limit=0.01,
                              passed=abs(err(b, a)) < 0.01))

        # diagnostics (not adopted): Table 4.10 kinetics; non-competitive H2 for the binaries
        rk = simulate(table410_kinetics(m))
        bt += bt_rows(run, m, rk, label="diag: Table 4.10 kinetics")
        h2.append(h2_metrics(run, m, rk, label="diag: Table 4.10 kinetics"))
        if run <= 3:
            rn = simulate(with_layer(m, iso=NoncompetitiveH2(m.col.layers[0].iso)))
            bt += bt_rows(run, m, rn, label="diag: non-competitive H2")
            h2.append(h2_metrics(run, m, rn, label="diag: non-competitive H2"))

    lm = linear_isothermal_model(N)
    t_st = stoichiometric_time(lm)
    rl = lm.solve(3 * t_st, n_out=3001)
    fl = rl.outlet_flow[0] / (lm.feed.flow * lm.y_feed[0])
    for item, v in (("first moment", np.trapezoid(1 - fl, rl.t)), ("t50", breakthrough_times(rl.t, fl)[1])):
        gates.append(dict(gate="linear isotherm vs t_st", run="-", item=item, value=err(v, t_st), limit=0.01,
                          passed=abs(err(v, t_st)) < 0.01))

    bt, tp, h2, au = pd.DataFrame(bt), pd.DataFrame(tp), pd.DataFrame(h2), pd.DataFrame(au)
    th = thermal_sensitivity(au)
    au.to_csv(OUT / "vs_author_model.csv", index=False)
    th.to_csv(OUT / "thermal_sensitivity.csv", index=False)
    gates, runtime = pd.DataFrame(gates), pd.DataFrame(runtime)
    bt.to_csv(OUT / "breakthrough_times.csv", index=False)
    tp.to_csv(OUT / "temperature_peaks.csv", index=False)
    h2.to_csv(OUT / "h2_flow.csv", index=False)
    gates.to_csv(OUT / "numerical_gates.csv", index=False)
    runtime.to_csv(OUT / "runtime.csv", index=False)
    write_report(bt, tp, h2, gates, runtime, t_st, au, th)
    print(f"wrote {OUT}")


def md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[k]
            cells.append(fmt[c].format(v) if c in fmt and isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                         else ("–" if isinstance(v, float) and not np.isfinite(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(bt, tp, h2, gates, runtime, t_st, au, th):
    base = bt[bt.case == "model"].copy()
    base["within_10pct"] = base.rel_err.abs() <= TOL_T
    n_ok, n_all = int(base.within_10pct.sum()), len(base)
    tp = tp.copy()
    order_ok = tp.groupby("run").apply(lambda g: bool((np.diff(g.sim_time_s) > 0).all() == (np.diff(g.exp_time_s) > 0).all()),
                                       include_groups=False)
    tp["qualitative_ok"] = (tp.rise_rel_err.abs() <= TOL_RISE) & (tp.time_rel_err.abs() <= TOL_TPEAK)
    num = gates.groupby("gate").agg(worst=("value", lambda v: v.abs().max()), limit=("limit", "first"),
                                    passed=("passed", "all")).reset_index()
    rt = runtime.groupby("N").runtime_s.agg(["mean", "min", "max"]).reset_index()
    diag = bt[bt.case != "model"].pivot_table(index=["run", "species", "level"], columns="case", values="rel_err").reset_index()

    pct = {"rel_err": "{:+.1%}", "sim_s": "{:.0f}", "exp_s": "{:.0f}"}
    fails = base[~base.within_10pct]
    g = lambda run, case, col: float(h2[(h2.run == run) & (h2.case == case)][col].iloc[0])
    au_flow = au[~au["item"].str.startswith("T")]
    au_minor = au_flow[au_flow["item"].str.match(r"(CO|N2) ")]
    au_co2 = au_flow[au_flow["item"].str.startswith("CO2")]
    th_c = lambda run, col: float(th[(th.run == run) & (th.case == "Cps 700 + wall 0.5 mm")][col].iloc[0])
    th_a = lambda run, col: float(th[(th.run == run) & (th.case == "author's gPROMS simulation")][col].iloc[0])
    th1 = th[th.run == 1].set_index("case")
    ch4 = lambda run, case, lev: float(bt[(bt.run == run) & (bt.case == case) & (bt.species == "CH4") & (bt.level == lev)].rel_err.iloc[0])
    au_ch4 = au[au["item"] == "CH4 t10 [s]"].iloc[0]
    status = "PASS" if num.passed.all() else "FAIL"
    findings = f"""## Summary

* Numerical gates: **{status}** (mole balance, grid convergence, linear-isotherm limit).
* Breakthrough times vs experiment: **{n_ok}/{n_all} within 10 %**; gate **{"PASS" if n_ok == n_all else "FAIL"}**.
  Failing: {", ".join(f"run {r_.run} {r_.species} {r_.level} ({r_.rel_err:+.0%})" for r_ in fails.itertuples()) or "none"}.
* Temperature peaks: positions right (ordering correct in all runs, peak times mostly within 10 %),
  heights low: the model's rise above 323 K is {tp.rise_rel_err.median():+.0%} (median) vs measured,
  range {tp.rise_rel_err.min():+.0%} to {tp.rise_rel_err.max():+.0%}.
* Implementation check against the author's own gPROMS curves (same equations, Table 5.7): CO and N2 fronts
  within {au_minor.rel_diff.abs().max():.0%}; CO2 fronts {au_co2.rel_diff.min():.0%} to {au_co2.rel_diff.max():.0%} later and temperature rises lower,
  both explained by unreported thermal inputs (problem 3: with Cps 700 J/kg/K and a thin wall our run-1/run-3
  bottom peaks are {th_c(1, "T Bottom [K]"):.1f}/{th_c(3, "T Bottom [K]"):.1f} K vs the author's {th_a(1, "T Bottom [K]"):.1f}/{th_a(3, "T Bottom [K]"):.1f} K and CO2 t10
  {th_c(1, "CO2 t10 [s]"):.0f}/{th_c(3, "CO2 t10 [s]"):.0f} s vs {th_a(1, "CO2 t10 [s]"):.0f}/{th_a(3, "CO2 t10 [s]"):.0f} s). CH4 is the exception (problem 1).
* Runtime: {rt[rt.N == N]["mean"].iloc[0]:.0f} s per breakthrough (6000 s process time) at N = {N} on one core.

## Problems found (not tuned — decisions needed)

**1. CH4 kinetics (errata E3) — gate failure, not a coding bug.** With the micropore constants printed in
Table 5.7 (= Table 5.9), CH4 has Dc/rc2 = 1.9e-3 s-1 at 323 K, 75x below the value measured on the same adsorbent
(Table 4.10, 0.144 s-1). The model then leaks CH4 early through a smeared front: t10 {ch4(5, "model", "t10"):+.0%},
t50 {ch4(5, "model", "t50"):+.0%} vs experiment. The author's own simulated CH4 curve (Fig. 5.17) has a sharp front
(t10 = {au_ch4.author:.0f} s, ours {au_ch4.ours:.0f} s), so the author's simulation cannot have used these constants as
printed. With the Table 4.10 kinetics (diagnostic) CH4 becomes t10 {ch4(5, "diag: Table 4.10 kinetics", "t10"):+.0%},
t50 {ch4(5, "diag: Table 4.10 kinetics", "t50"):+.0%}, but CO and N2 t10 move to +17-19 %. Neither printed set passes
for all species. This feeds directly into the E3 decision planned for step 3.
Options: (a) keep Table 5.7 and accept the CH4 failure; (b) use Table 4.10 (measured) kinetics for all species;
(c) Table 5.7 for all species except CH4 from Table 4.10 (only CH4 is incompatible with the author's curve);
(d) defer to step 3 and choose the set that reproduces the PSA experiment (81.6 % / 99.9805 %).

**2. H2 co-adsorption in the virial mixture model (open issue in PROVENANCE.md).** H2 breakthrough times cannot
be tested (first GC sample after H2 breakthrough). What the data do constrain is the H2 flow excess while the CO2
front moves through the bed: H2 displaced from the gas phase and from the adsorbent leaves the column.
Run 1 (30 % CO2): measured {g(1, "model", "exp_before"):.3f} x feed, virial (Taqvi-LeVan) {g(1, "model", "sim_before"):.3f},
non-competitive H2 bound {g(1, "diag: non-competitive H2", "sim_before"):.3f}. The measurement sits between the two, but its own
flow offset ({g(1, "model", "exp_late") - 1:+.1%} at the late plateau, where 1.000 is exact) is half the gap between them.
Run 2: measured {g(2, "model", "exp_before"):.3f}, virial {g(2, "model", "sim_before"):.3f}, bound {g(2, "diag: non-competitive H2", "sim_before"):.3f} (2 samples).
Run 3 (80 % CO2): measured {g(3, "model", "exp_before"):.3f}, virial {g(3, "model", "sim_before"):.3f}, bound {g(3, "diag: non-competitive H2", "sim_before"):.3f}:
both overpredict; H2 co-adsorption explains at most {abs(g(3, "model", "sim_before") - g(3, "diag: non-competitive H2", "sim_before")) / abs(g(3, "model", "sim_before") - g(3, "model", "exp_before")):.0%} of the gap.
Not reproduced by either variant: the H2 roll-up peak at the CO2 front in run 1 (measured {g(1, "model", "exp_peak"):.3f}, virial
{g(1, "model", "sim_peak"):.3f}); in run 3 the measured H2 flow drops below the feed value after CO2 breakthrough,
which no isotherm can produce (H2 hardly adsorbs) and points to a flow-measurement bias at high CO2 fraction.
Size of the error: the virial model gets the H2 flow before breakthrough within {abs(g(1, "model", "sim_before") - g(1, "model", "exp_before")):.1%}
of feed in run 1 and {abs(g(2, "model", "sim_before") - g(2, "model", "exp_before")):.1%} in run 2, comparable to the measurement offset,
and misses the run-1 roll-up peak by {abs(g(1, "model", "sim_peak") - g(1, "model", "exp_peak")):.1%}. The breakthrough data therefore cannot
confirm or reject the near-zero H2 co-adsorption of the Taqvi-LeVan rule (0.001 mol/kg vs 0.157 mol/kg for pure H2 at
5 bar, 323 K); they only bound its effect on the outlet flow to a few per cent.
Options: (a) keep the published Taqvi-LeVan virial model and carry the non-competitive bound as a sensitivity case
into the recovery analysis (steps 3 and 7); (b) replace the mixing rule by IAST on the pure virial isotherms
(thermodynamically consistent, no new parameters) and compare; (c) the geometric-mean mixing rule the thesis mentions
on p.184; (d) decide in step 3 on the PSA recovery (81.6 %), which is far more sensitive to H2 co-adsorption than
breakthrough. Recommendation: (a) now, (b) as a cross-check in step 3.

**3. Temperature rise 20-40 % low.** Peak positions match; heights are low in our model and, by less, in the
author's model ({au[au["item"].str.startswith("T")].rel_diff.median():+.0%} median difference in rise vs the author). The
relevant inputs are not reported (solid heat capacity, wall thickness, wall heat capacity — `assumed:` block). Run-1
sensitivity (bottom thermocouple): Cps 880 -> 700 raises the peak by {th1.loc["Cps 700 (lower end of assumed range)", "T Bottom [K]"] - th1.loc["model (Cps 880, wall 3 mm)", "T Bottom [K]"]:.1f} K,
wall 3 -> 2 mm by {th1.loc["wall 2 mm (lower end of assumed range)", "T Bottom [K]"] - th1.loc["model (Cps 880, wall 3 mm)", "T Bottom [K]"]:.1f} K, both together with a 0.5 mm wall by
{th1.loc["Cps 700 + wall 0.5 mm", "T Bottom [K]"] - th1.loc["model (Cps 880, wall 3 mm)", "T Bottom [K]"]:.1f} K, which reproduces the author's model
({th1.loc["author's gPROMS simulation", "T Bottom [K]"]:.1f} K) but still not the measurement ({th1.loc["experiment", "T Bottom [K]"]:.1f} K).
The same change moves the CO2 front onto the author's (run 1 t10 {th_c(1, "CO2 t10 [s]"):.0f} s vs {th_a(1, "CO2 t10 [s]"):.0f} s), i.e. the
author evidently used a smaller thermal mass than our assumed values; the CO2 t50 then moves away from the experiment. The early dip of the Top
thermocouple (heater switching off, p.191) is an experimental artefact the model does not contain.
Options: (a) keep the assumed values and report the bias; (b) fit Cps and wall heat capacity to the temperature
histories of runs 1-3 and validate on runs 4-5 (a calibration step, requires your approval).

**4. Shape of the CO2 front.** Our CO2 rises less steeply after breakthrough than measured (run 1 t10
{float(base[(base.run == 1) & (base.species == "CO2") & (base.level == "t10")].rel_err.iloc[0]):+.1%}, t50
{float(base[(base.run == 1) & (base.species == "CO2") & (base.level == "t50")].rel_err.iloc[0]):+.1%}). The author's model is steeper
(run 1 t50 {th_a(1, "CO2 t50 [s]"):.0f} s, measured {float(th1.loc["experiment", "CO2 t50 [s]"]):.0f} s) and the thermal change of problem 3 does not
close this gap (t50 {th_c(1, "CO2 t50 [s]"):.0f} s). The CO2 isotherm, kinetic constants and transport parameters are identical in both models,
so the remaining difference lies in an unreported model detail (candidates: a_p, loading-dependent heat of
adsorption, gas-phase heat capacity treatment). It is within the 10 % gate but is listed so it is not forgotten.

"""
    text = f"""# Step 2 — column model validated on breakthrough curves

Generated by `scripts/run_breakthrough.py` (do not edit by hand; rerun the script).
Model: `src/h2psa/column.py` (Lopes 2010 eqs 5.3-5.16), N = {N} finite volumes, BDF.
Experiment: Lopes (2010) Figs 5.15-5.17, digitised in `data/benchmarks/digitized/`.
Transport: Table 5.7 (per run); micropore kinetics Table 5.7 (= Table 5.9, errata E3).

{findings}## Numerical gates

{md(num, {"worst": "{:.2e}", "limit": "{:.0e}"})}

Mole balance: worst |closure| over runs 1-5, per component and overall. Grid: t50 of every adsorbing
species, N = {N} vs {N2}. Linear limit: single tracer (1 % in He), isothermal, q = K P, t_st = {t_st:.1f} s.

## Breakthrough times vs experiment (gate: |error| <= {TOL_T:.0%})

{n_ok} of {n_all} breakthrough times are within {TOL_T:.0%}. H2 is not in the table: the first H2 sample
of every run is taken after H2 has already broken through (the GC sampling starts at 140-490 s),
so no measured H2 breakthrough time exists.

{md(base[["run", "species", "level", "sim_s", "exp_s", "rel_err", "within_10pct"]], pct)}

## Temperature peaks (qualitative gate)

"Qualitatively right" is defined here as: peak rise above the 323 K feed within {TOL_RISE:.0%}, peak time within
{TOL_TPEAK:.0%}, and the same Bottom -> Middle -> Top ordering of peak times. Peak ordering correct in all runs:
{bool(order_ok.all())}. {int(tp.qualitative_ok.sum())} of {len(tp)} thermocouple peaks meet the rise/time criteria.

{md(tp[["run", "tc", "sim_peak_K", "exp_peak_K", "rise_rel_err", "sim_time_s", "exp_time_s", "time_rel_err", "qualitative_ok"]],
    {"sim_peak_K": "{:.1f}", "exp_peak_K": "{:.1f}", "rise_rel_err": "{:+.0%}", "sim_time_s": "{:.0f}",
     "exp_time_s": "{:.0f}", "time_rel_err": "{:+.0%}"})}

## H2 outlet flow and the virial mixture model (open issue)

H2 outlet flow / H2 feed flow, averaged over the samples before the measured CO2 front (`before`), maximum
around the CO2 front (`peak`), and mean after 2500 s (`late`, = 1 in a perfect flow measurement).

{md(h2[["run", "case", "exp_before", "sim_before", "exp_peak", "sim_peak", "exp_late", "sim_late", "n_before"]],
    {c: "{:.3f}" for c in ["exp_before", "sim_before", "exp_peak", "sim_peak", "exp_late", "sim_late"]})}

## Our model vs the author's gPROMS simulation (implementation check)

Same equations and Table 5.7 parameters; the author's curves are vector-extracted from Figs 5.15-5.17
(`*_authorsim.csv`, ±0.3 pt line half-width ≈ ±12 s on the 0-6000 s axes, ±3 s on the zoomed axes).
For temperatures `rel_diff` compares the rise above 323 K.

{md(au, {"ours": "{:.1f}", "author": "{:.1f}", "rel_diff": "{:+.1%}"})}

## Thermal sensitivity, runs 1 and 3 (assumed values, diagnostic)

{md(th, {c: "{:.1f}" for c in th.columns if c != "case"})}

## Diagnostics (not adopted — for the decision only)

Relative breakthrough-time error per case: `model` = Table 5.7 kinetics, Taqvi-LeVan virial;
`Table 4.10 kinetics` = micropore Dc/rc2 refitted to the measured values of Table 4.10;
`non-competitive H2` = H2 loading from its pure isotherm at its partial pressure (runs 1-3).

{md(diag, {c: "{:+.1%}" for c in diag.columns if c not in ("run", "species", "level")})}

## Runtime per breakthrough simulation (6000 s of process time)

{md(rt, {"mean": "{:.1f} s", "min": "{:.1f} s", "max": "{:.1f} s"})}

## Figures

""" + "\n".join(f"![run {k}](run{k}.png)" for k in RUNS) + "\n"
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
