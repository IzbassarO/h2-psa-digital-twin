"""Step 2: simulate the compact-unit breakthrough runs 1-5 (Lopes 2010) and compare with experiment.

    python scripts/run_breakthrough.py        # ~15 min; writes results/step2/
    (the thermal calibration itself is scripts/calibrate_thermal.py; its result is read from the YAML)

Outputs: run{1..5}.png, sim_run{k}.csv, breakthrough_times.csv, temperature_peaks.csv, objective.csv,
h2_flow.csv, h2_loading.csv, purity.csv, numerical_gates.csv, runtime.csv, vs_author_model.csv, REPORT.md;
REPORT.md and the figures are copied to docs/validation/step2.md and docs/validation/step2/ (committed).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.column import ColumnModel, compact_breakthrough  # noqa: E402
from h2psa.isotherms import Virial, load_yaml  # noqa: E402
from h2psa.validation import (FIT_RUNS, T_FIT, TC_Z, breakthrough_times, exp_flow, exp_temperature_peaks,  # noqa: E402
                              linear_isothermal_model, load_exp, rebuild, sigma_thermocouple, stoichiometric_time,
                              thermal_residuals)

OUT = ROOT / "results" / "step2"
DOCS = ROOT / "docs" / "validation"
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


def err(a, b):
    return (a - b) / b if np.isfinite(a) and np.isfinite(b) else np.nan


# ---------------------------------------------------------------------------- analysis pieces
def bt_rows(run, m, r, label="final"):
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


def h2_metrics(run, m, r, label="final"):
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


def t_rows(run, m, r, label="final"):
    """Thermocouple peaks and the RMS error of each history over 0-T_FIT s."""
    pk, d = exp_temperature_peaks(run), load_exp(run, "T")
    rows = []
    for tc, z in TC_Z.items():
        T = r.T_at(z)
        k = int(np.argmax(T))
        Te, te = pk[tc]
        rise_s, rise_e = T[k] - m.feed.T, Te - m.feed.T
        g = d[(d.series == tc) & (d.time_s >= 0) & (d.time_s <= T_FIT)]
        rms = float(np.sqrt(np.mean((np.interp(g.time_s, r.t, T) - g.temperature_K.values) ** 2)))
        rows.append(dict(run=run, case=label, tc=tc, z_m=z, sim_peak_K=T[k], exp_peak_K=Te, sim_rise_K=rise_s,
                         exp_rise_K=rise_e, rise_rel_err=err(rise_s, rise_e), sim_time_s=r.t[k], exp_time_s=te,
                         time_rel_err=err(r.t[k], te), rms_K=rms))
    return rows


def objective_row(run, m, r, sT, label):
    """Calibration objective of one run (same definition as scripts/calibrate_thermal.py)."""
    res = thermal_residuals(m, run, sT, r=r)
    return dict(run=run, case=label, role="fit" if run in FIT_RUNS else "held out",
                temperature_term=float(res[:-2] @ res[:-2]), co2_time_term=float(res[-2:] @ res[-2:]),
                objective=float(res @ res))


def purity_rows(run, m, r, label):
    """He-free outlet purity: first time it drops below 99.99 % H2 (once H2 flow > half its feed value) and the
    impurity shares at that time."""
    F = r.outlet_flow
    iH, keep = m.sp.index("H2"), [i for i, s in enumerate(m.sp) if s != "He"]
    tot = F[keep].sum(0)
    ok = F[iH] > 0.5 * m.feed.flow * m.y_feed[iH]
    imp = np.where(ok, 1 - F[iH] / np.maximum(tot, 1e-300), 0.0)
    k = int(np.argmax(imp > 1e-4))
    shares = {s: F[i, k] / (tot[k] - F[iH, k]) for i, s in enumerate(m.sp) if s not in ("H2", "He")}
    return dict(run=run, case=label, t_below_9999_s=r.t[k], limiting=max(shares, key=shares.get),
                **{f"share_{s}": v for s, v in shares.items()})


def h2_loadings(run):
    """H2 loading at the feed (5 bar, 323 K) under each mixture rule; IAST species past psi_max."""
    m = compact_breakthrough(run, N=4)
    P = np.array([[m.feed.y.get(n, 0.0) * m.feed.P_out / 1e5 for n in m.col.layers[0].iso.names]])
    T = np.array([m.feed.T])
    out = dict(run=run)
    for mix in Virial.MIXING:
        v = Virial.from_yaml(ROOT / "data" / "params" / "ac5ks_lopes2010.yaml", mixing=mix)
        out[mix] = float(v.equilibrium(P, T)[0, v.names.index("H2")])
        if mix == "iast":
            psi = float(np.exp(v._s_last[0]))
            lnK, A, B = v._coeffs(T)
            a, b = A[0] / v.S, B[0] / v.S**2
            past = [n for n, ai, bi in zip(v.names, a, b) if bi < 0 and psi > _psi_max(ai, bi)]
            out["iast_psi"] = psi
            out["iast_past_psi_max"] = ", ".join(past) or "none"
    out["pure_H2_at_P_H2"] = out.pop("noncompetitive_h2")
    return out


def _psi_max(a, b):
    q = (-a - np.sqrt(a * a - 3 * b)) / (3 * b)
    return q + a * q * q + b * q**3


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


def plot_run(run, m, r, r0, path):
    """r: final model, r0: model before the step-2 decisions (dashed)."""
    F0 = feed_flows(m)
    minor = [s for s in m.sp if s in ("CO", "CH4", "N2")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for i, s in enumerate(m.sp):
        if s == "He":
            continue
        for a in ax[:2]:
            a.plot(r.t, r.outlet_flow[i] * 1e3, color=COLOUR[s], lw=1.8, label=f"{s} model")
            a.plot(r0.t, r0.outlet_flow[i] * 1e3, color=COLOUR[s], lw=1.0, ls="--", alpha=0.7)
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
        ax[2].plot(r0.t, r0.T_at(z), color=TC_COLOUR[tc], lw=1.0, ls="--", alpha=0.7)
        au = load_exp(run, "T_authorsim")
        au = au[au.series == tc]
        ax[2].plot(au.time_s, au.temperature_K, ".", ms=0.8, color="#888780", alpha=0.5)
    ax[2].set(xlim=(0, T_END), xlabel="time [s]", ylabel="gas temperature [K]",
              title="Thermocouple temperatures (grey: author model)")
    fig.suptitle("symbols: experiment · solid: this model (decisions applied) · dashed: before the decisions · "
                 "fine dots: author's gPROMS simulation", fontsize=9)
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


THERMAL_FIT = "thermal: bounded fit (not adopted)"


def calibration_result():
    """Result of scripts/calibrate_thermal.py (results/step2/calibration_result.csv) or None."""
    f = OUT / "calibration_result.csv"
    return pd.read_csv(f, index_col="item").value if f.exists() else None


def thermal_text(cal, fit, asm, tba, obt, held, held_bt):
    """Summary bullet and report section for decision 3 (adopted calibration, or a fit stopped on a bound)."""
    common = f"""* Parameters: solid heat capacity Cps and wall heat capacity per unit length Cw = rho_w Cp_w pi e (d_wi + e), one
  lumped parameter (the wall terms of eq. 5.11 depend on e only through Cw, up to e/2d_wi <= 5 % in the outer exchange
  term; e stays at 3 mm there). Bounds: Cps 600-1200 J/kg/K; Cw from 0.5-5 mm stainless steel.
* Fit data: runs {", ".join(map(str, FIT_RUNS))}, thermocouple histories 0-{T_FIT:.0f} s and CO2 t10/t50; kinetics, isotherm and transport
  fixed; runs 4-5 and the PSA experiment held out.
* Objective: sum over runs of [sum over the 3 thermocouples of mean((T_sim - T_exp)^2)/sigma_T^2 + sum over CO2 t10, t50
  of ((t_sim - t_exp)/sigma_t)^2]. Weights: each thermocouple history counts as one observation of its mean-square error
  (9 terms), each CO2 crossing time as one observation (6 terms); sigma_T = RMS thermocouple deviation from 323 K after
  4000 s (runs 1-3), sigma_t = half the GC sampling interval bracketing the crossing (+) 2.4 s read-off (5-15 s).
* Assumed values (kept in `assumed:`): Cps {asm["solid_heat_capacity"]["value"]:.0f} J/kg/K, wall {asm["wall_thickness"]["value"] * 1e3:.0f} mm stainless steel."""
    if cal is not None:
        cps, cw = cal["solid_heat_capacity"], cal["wall_heat_capacity_per_length"]
        summary = (f"Thermal calibration adopted: Cps = {cps['value']:.0f} ± {cps['sd']:.0f} J/kg/K, Cw = {cw['value']:.0f} ± {cw['sd']:.0f} J/m/K "
                   f"(correlation {cal['correlation']:+.2f}). Held-out runs 4-5: thermocouple RMS {held.rms_before_K.mean():.2f} -> "
                   f"{held.rms_after_K.mean():.2f} K; breakthrough times within 10 %: {int((held_bt.before.abs() <= TOL_T).sum())} -> "
                   f"{int((held_bt.after.abs() <= TOL_T).sum())} of {len(held_bt)}.")
        return summary, f"""## Thermal calibration (decision 3)

{common}
* Result: Cps = **{cps["value"]:.0f} ± {cps["sd"]:.0f} J/kg/K**, Cw = **{cw["value"]:.0f} ± {cw["sd"]:.0f} J/m/K**, correlation
  {cal["correlation"]:+.2f}; {cal["uncertainty"]}.
"""
    if fit is None:
        return "Thermal calibration: not run (scripts/calibrate_thermal.py).", "## Thermal calibration (decision 3)\n\nNot run.\n"
    on = [n for n in ("Cps", "Cw") if str(fit[f"{n}_at_bound"]) == "True"]
    hf = tba[tba.role == "held out"]
    summary = (f"Thermal calibration **stopped, not adopted**: {' and '.join(on) or 'no parameter'} on a bound "
               f"(Cps = {float(fit.Cps):.0f} J/kg/K, Cw = {float(fit.Cw):.0f} J/m/K = lower bound, i.e. a {float(fit.Cw_equiv_wall_mm):.1f} mm wall). "
               f"The final model keeps the assumed thermal values.")
    sec = f"""## Thermal calibration (decision 3) — stopped: parameter on a bound

{common}
* Result (`scripts/calibrate_thermal.py`, {fit.date}, {int(float(fit.nfev))} evaluations): Cps = {float(fit.Cps):.0f} ± {float(fit.Cps_sd):.0f} J/kg/K,
  **Cw = {float(fit.Cw):.0f} ± {float(fit.Cw_sd):.0f} J/m/K, on its lower bound** ({float(fit.Cw_lo):.0f}; a {float(fit.Cw_equiv_wall_mm):.2f} mm stainless wall),
  correlation {float(fit.correlation):+.2f}. Uncertainties from the Gauss-Newton Hessian, scaled by chi2 per observation-equivalent
  ({float(fit.chi2_per_observation):.1f}: the residual is dominated by model error, not by measurement uncertainty). Objective
  {float(fit.objective_start):.1f} -> {float(fit.objective_final):.1f}.
* Per decision 3 the fit is **not adopted**: no `calibrated:` block was written; the final model uses the `assumed:` values.
* What drives it (sum over runs 1-5): temperature term {obt.loc["before", "temperature_term"]:.1f} -> {obt.loc[THERMAL_FIT, "temperature_term"]:.1f},
  CO2-time term {obt.loc["before", "co2_time_term"]:.1f} -> {obt.loc[THERMAL_FIT, "co2_time_term"]:.1f} (assumed values -> bounded fit). The CO2 term
  is dominated by the run-1 t50 error (front shape, see the particle-area section), which the thermal parameters can only
  partly offset; Cps and Cw trade off against each other (correlation {float(fit.correlation):+.2f}).
* Held-out runs 4-5 with the bounded fit (diagnostic): thermocouple RMS {hf.rms_before_K.mean():.2f} K (assumed) -> {hf.rms_fit_K.mean():.2f} K,
  peak-rise error {hf.rise_err_before.median():+.0%} -> {hf.rise_err_fit.median():+.0%} (median).
* Options for your decision: (a) keep the assumed values and report the temperature bias; (b) widen the Cw bound below
  0.5 mm, i.e. accept a wall that stores almost no heat (physically questionable for a steel column); (c) fit on the
  thermocouple histories only (the CO2 front-shape error then no longer drives the thermal parameters); (d) add a third
  parameter (e.g. hw or U), which Table 5.7 reports from correlations.
"""
    return summary, sec


# ---------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ac = load_yaml(ROOT / "data" / "params" / "ac5ks_lopes2010.yaml")
    Rp, Lp = ac["physical"]["pellet_radius"], ac["physical"]["pellet_length_R2030"]
    all_measured = [k for k in ac["micropore_kinetics_breakthrough"] if k != "units"]
    diagnostics = {   # each built on the final model of the run
        "kinetics (b): Table 4.10 for all species": lambda run, m: compact_breakthrough(run, N=N, measured_kinetics=all_measured),
        "mixing: iast": lambda run, m: compact_breakthrough(run, N=N, mixing="iast"),
        "mixing: noncompetitive_h2": lambda run, m: compact_breakthrough(run, N=N, mixing="noncompetitive_h2"),
        "a_p = 3/R_p (sphere)": lambda run, m: rebuild(m, a_p=3 / Rp),
        "a_p = 2/R_p + 2/L_p (finite cylinder)": lambda run, m: rebuild(m, a_p=2 / Rp + 2 / Lp),
    }
    adopted = "calibrated" in load_yaml(ROOT / "data" / "params" / "compact_column_lopes2010.yaml")
    fit = calibration_result()
    if fit is not None and not adopted:   # decision 3: fit on a bound -> not adopted, shown as a diagnostic
        diagnostics[THERMAL_FIT] = lambda run, m: rebuild(m, col={"Cw": float(fit.Cw)}, Cps=float(fit.Cps))
    sT = sigma_thermocouple(323.0)
    bt, tp, h2, gates, au, runtime, obj, pur = [], [], [], [], [], [], [], []

    for run in RUNS:
        m0 = compact_breakthrough(run, N=N, measured_kinetics=[], thermal="assumed")   # before the decisions
        r0 = simulate(m0)
        m = compact_breakthrough(run, N=N)                                              # final
        r = simulate(m)
        print(f"run {run}: {r.runtime_s:.1f} s, closure {r.closure['total']:.1e}", flush=True)
        for label, mm, rr in (("before", m0, r0), ("final", m, r)):
            runtime.append(dict(run=run, case=label, N=N, runtime_s=rr.runtime_s, nfev=rr.nfev))
            bt += bt_rows(run, mm, rr, label)
            tp += t_rows(run, mm, rr, label)
            obj.append(objective_row(run, mm, rr, sT, label))
            h2.append(h2_metrics(run, mm, rr, label))
        pd.DataFrame({"time_s": r.t, **{f"{s}_mol_s": r.outlet_flow[i] for i, s in enumerate(m.sp)},
                      **{f"T_{tc}_K": r.T_at(z) for tc, z in TC_Z.items()}}).to_csv(OUT / f"sim_run{run}.csv", index=False)
        au += author_rows(run, m, r)
        plot_run(run, m, r, r0, OUT / f"run{run}.png")
        for k, v in r.closure.items():
            gates.append(dict(gate="mole balance", run=run, item=k, value=v, limit=1e-3, passed=abs(v) < 1e-3))
        if run == 5:
            pur.append(purity_rows(run, m, r, "final"))

        # grid convergence N -> 2N
        r2 = simulate(compact_breakthrough(run, N=N2))
        runtime.append(dict(run=run, case="final", N=N2, runtime_s=r2.runtime_s, nfev=r2.nfev))
        F0 = feed_flows(m)
        for i, s in enumerate(m.sp):
            if s in ("H2", "He"):
                continue
            a = breakthrough_times(r.t, r.outlet_flow[i] / F0[s])[1]
            b = breakthrough_times(r2.t, r2.outlet_flow[i] / F0[s])[1]
            gates.append(dict(gate="grid t50 N=60->120", run=run, item=s, value=err(b, a), limit=0.01,
                              passed=abs(err(b, a)) < 0.01))

        for label, build in diagnostics.items():
            md_ = build(run, m)
            rd = simulate(md_)
            runtime.append(dict(run=run, case=label, N=N, runtime_s=rd.runtime_s, nfev=rd.nfev))
            bt += bt_rows(run, m, rd, label)
            if label.startswith("mixing"):
                h2.append(h2_metrics(run, m, rd, label))
            if label == THERMAL_FIT:
                tp += t_rows(run, md_, rd, label)
                obj.append(objective_row(run, md_, rd, sT, label))
            if run == 5 and label.startswith("kinetics"):
                pur.append(purity_rows(run, md_, rd, label))

    lm = linear_isothermal_model(N)
    t_st = stoichiometric_time(lm)
    rl = lm.solve(3 * t_st, n_out=3001)
    fl = rl.outlet_flow[0] / (lm.feed.flow * lm.y_feed[0])
    for item, v in (("first moment", np.trapezoid(1 - fl, rl.t)), ("t50", breakthrough_times(rl.t, fl)[1])):
        gates.append(dict(gate="linear isotherm vs t_st", run="-", item=item, value=err(v, t_st), limit=0.01,
                          passed=abs(err(v, t_st)) < 0.01))

    tables = dict(breakthrough_times=bt, temperature_peaks=tp, h2_flow=h2, numerical_gates=gates, vs_author_model=au,
                  runtime=runtime, objective=obj, purity=pur, h2_loading=[h2_loadings(k) for k in RUNS])
    tables = {k: pd.DataFrame(v) for k, v in tables.items()}
    for k, v in tables.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    write_report(t_st=t_st, sT=sT, **tables)
    # gate report in git (results/ is ignored): docs/validation/step2.md + figures
    (DOCS / "step2").mkdir(parents=True, exist_ok=True)
    for k in RUNS:
        shutil.copy(OUT / f"run{k}.png", DOCS / "step2" / f"run{k}.png")
    (DOCS / "step2.md").write_text((OUT / "REPORT.md").read_text().replace("](run", "](step2/run"))
    print(f"wrote {OUT} and {DOCS / 'step2.md'}")


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


def write_report(breakthrough_times, temperature_peaks, h2_flow, numerical_gates, vs_author_model, runtime, objective,
                 purity, h2_loading, t_st, sT):
    bt, tp, h2, gates, au, obj = breakthrough_times, temperature_peaks, h2_flow, numerical_gates, vs_author_model, objective
    cc = load_yaml(ROOT / "data" / "params" / "compact_column_lopes2010.yaml")
    cal, fit = cc.get("calibrated"), calibration_result()
    asm = cc["assumed"]
    exp_psa = pd.read_csv(ROOT / "data" / "benchmarks" / "lopes2010_psa_experiment.csv").iloc[0]

    def within(df):
        return int((df.rel_err.abs() <= TOL_T).sum()), len(df)

    fin, bef = bt[bt.case == "final"].copy(), bt[bt.case == "before"].copy()
    fin["within_10pct"] = fin.rel_err.abs() <= TOL_T
    n_ok, n_all = within(fin)
    n_ok0, _ = within(bef)
    fails = fin[~fin.within_10pct]
    tpf = tp[tp.case == "final"].copy()
    tpf["qualitative_ok"] = (tpf.rise_rel_err.abs() <= TOL_RISE) & (tpf.time_rel_err.abs() <= TOL_TPEAK)
    order_ok = all((np.diff(g.sim_time_s) > 0).all() == (np.diff(g.exp_time_s) > 0).all() for _, g in tpf.groupby("run"))
    num = gates.groupby("gate").agg(worst=("value", lambda v: v.abs().max()), limit=("limit", "first"),
                                    passed=("passed", "all")).reset_index()
    rt = runtime.groupby(["case", "N"]).runtime_s.agg(["mean", "min", "max"]).reset_index()
    rel = lambda df, run, sp, lev: float(df[(df.run == run) & (df.species == sp) & (df.level == lev)].rel_err.iloc[0])
    role = lambda run: "fit" if run in FIT_RUNS else "held out"

    # before/after tables
    ba = fin[["run", "species", "level", "exp_s"]].copy()
    ba["before"] = [rel(bef, *k) for k in zip(ba.run, ba.species, ba.level)]
    ba["after"] = ba.pop("exp_s") * 0 + fin.rel_err.values
    ba.insert(0, "role", [("fit" if (r_ in FIT_RUNS and s_ == "CO2") else "held out") for r_, s_ in zip(ba.run, ba.species)])
    tb = tp[tp.case == "before"].set_index(["run", "tc"])
    tf = tpf.set_index(["run", "tc"])
    tba = pd.DataFrame({"role": [role(r_) for r_, _ in tf.index], "run": [r_ for r_, _ in tf.index], "tc": [t_ for _, t_ in tf.index],
                        "exp_peak_K": tf.exp_peak_K.values, "peak_before_K": tb.sim_peak_K.values, "peak_after_K": tf.sim_peak_K.values,
                        "rise_err_before": tb.rise_rel_err.values, "rise_err_after": tf.rise_rel_err.values,
                        "rms_before_K": tb.rms_K.values, "rms_after_K": tf.rms_K.values})
    has_fit = THERMAL_FIT in set(tp.case)
    if has_fit:
        tx = tp[tp.case == THERMAL_FIT].set_index(["run", "tc"])
        tba["peak_fit_K"], tba["rise_err_fit"], tba["rms_fit_K"] = tx.sim_peak_K.values, tx.rise_rel_err.values, tx.rms_K.values
    ob = obj.pivot_table(index=["run", "role"], columns="case", values="objective").reset_index()
    ob = ob[["role", "run", "before", "final"] + ([THERMAL_FIT] if has_fit else [])]
    obt = obj.groupby("case")[["temperature_term", "co2_time_term"]].sum()
    held = tba[tba.role == "held out"]
    held_bt = ba[(ba.role == "held out") & (ba.run >= 4)]

    # CH4 / purity
    ch4_t10, ch4_t50 = rel(fin, 5, "CH4", "t10"), rel(fin, 5, "CH4", "t50")
    ch4_b = rel(bt[bt.case.str.startswith("kinetics")], 5, "CH4", "t10")
    au_ch4 = float(au[(au.run == 5) & (au["item"] == "CH4 t10 [s]")].author.iloc[0])
    exp_ch4 = float(fin[(fin.run == 5) & (fin.species == "CH4") & (fin.level == "t10")].exp_s.iloc[0])
    pf = purity[purity.case == "final"].iloc[0]
    t10e = {s: float(fin[(fin.run == 5) & (fin.species == s) & (fin.level == "t10")].exp_s.iloc[0]) for s in ("N2", "CO", "CH4", "CO2")}
    side = {s: (rel(bef, 5, s, "t10"), rel(fin, 5, s, "t10")) for s in ("N2", "CO")}
    shares = ", ".join(f"{k[6:]} {v:.0%}" for k, v in pf.items() if str(k).startswith("share_") and v > 0.005)
    imp_ppm = (100 - exp_psa.h2_purity_pct) * 1e4

    # H2 mixing
    h2c = h2[h2.case.isin(["final", "mixing: iast", "mixing: noncompetitive_h2"])].copy()
    h2c["case"] = h2c.case.replace({"final": "taqvi_levan (final)", "mixing: iast": "iast", "mixing: noncompetitive_h2": "noncompetitive_h2"})
    g = lambda run, case, col: float(h2c[(h2c.run == run) & (h2c.case == case)][col].iloc[0])

    # a_p
    ap_cases = ["final", "a_p = 3/R_p (sphere)", "a_p = 2/R_p + 2/L_p (finite cylinder)"]
    rows = []
    for (run, sp), grp in bt[bt.case.isin(ap_cases)].groupby(["run", "species"]):
        e10, e50 = (float(grp[(grp.case == "final") & (grp.level == l)].exp_s.iloc[0]) for l in ("t10", "t50"))
        row = dict(run=run, species=sp, exp_width_s=e50 - e10)
        for c in ap_cases:
            s10, s50 = (float(grp[(grp.case == c) & (grp.level == l)].sim_s.iloc[0]) for l in ("t10", "t50"))
            short = {"final": "2/Rp", "a_p = 3/R_p (sphere)": "3/Rp", "a_p = 2/R_p + 2/L_p (finite cylinder)": "2/Rp+2/Lp"}[c]
            row[f"width {short}"] = s50 - s10
            row[f"t50 err {short}"] = err(s50, e50)
        rows.append(row)
    ap = pd.DataFrame(rows)
    ap1 = ap[(ap.run == 1) & (ap.species == "CO2")].iloc[0]
    diag = bt[~bt.case.isin(["final", "before"])].pivot_table(index=["run", "species", "level"], columns="case", values="rel_err").reset_index()
    n_diag = {c: within(bt[bt.case == c]) for c in bt.case.unique() if c not in ("final", "before")}

    thermal_summary, thermal_section = thermal_text(cal, fit, asm, tba, obt, held, held_bt)
    status = "PASS" if num.passed.all() else "FAIL"
    pct = lambda v: f"{v:+.0%}"
    text = f"""# Step 2 — column model validated on breakthrough curves

Generated by `scripts/run_breakthrough.py` (do not edit by hand; rerun the script). Thermal calibration:
`scripts/calibrate_thermal.py` ({"adopted, stored in `compact_column_lopes2010.yaml` block `calibrated:`" if cal is not None else "not adopted, see below"}).
Model: `src/h2psa/column.py` (Lopes 2010 eqs 5.3-5.16), N = {N} finite volumes, BDF.
Experiment: Lopes (2010) Figs 5.15-5.17, digitised in `data/benchmarks/digitized/`.

## Decisions applied (project lead, 2026-10-09)

1. **CH4 kinetics, option (c), errata E7.** Table 5.7 micropore constants for H2, CO2, CO, N2; CH4 from the
   Arrhenius refit of the measured Table 4.10 values (`psa_transport.micropore_use_measured: [CH4]`). Rule: a printed
   PSA-simulation constant is replaced only if it contradicts both the measured value on the same adsorbent and the
   author's own simulated curve. Option (b) (Table 4.10 for all species) is kept as a sensitivity case.
2. **H2 co-adsorption.** Mixture rule selectable in YAML (`isotherm.mixing`): `taqvi_levan` (default, published),
   `iast` (new; IAST on the pure virial isotherms), `noncompetitive_h2` (bound). All three are reported; none is
   selected here.
3. **Thermal calibration** of Cps and the lumped wall heat capacity per unit length on runs 1-3 (thermocouples + CO2
   t10/t50 jointly); runs 4-5 held out. {"Adopted." if cal is not None else "**Stopped: a parameter ended on its bound; not adopted.**"}
4. **Particle specific area:** diagnostic only (2/R_p vs 3/R_p vs 2/R_p + 2/L_p); adoption rule: only if it follows
   from the particle geometry in the thesis.
5. Report copied to `docs/validation/step2.md` (committed).

"before" below = the model of the first step-2 report (Table 5.7 kinetics for all species, assumed Cps 880 J/kg/K
and 3 mm wall); "final" = the decisions applied ({"including" if cal is not None else "without"} the thermal calibration, see below).

## Summary

* Numerical gates: **{status}** (mole balance, grid convergence, linear-isotherm limit), final model.
* Breakthrough times vs experiment: **{n_ok}/{n_all} within 10 %** (before: {n_ok0}/{n_all}); gate
  **{"PASS" if n_ok == n_all else "FAIL"}**. Failing: {", ".join(f"run {r_.run} {r_.species} {r_.level} ({r_.rel_err:+.0%})" for r_ in fails.itertuples()) or "none"}.
* {thermal_summary}
* Temperature peaks (final): ordering correct in all runs: {order_ok}; {int(tpf.qualitative_ok.sum())} of {len(tpf)} peaks within
  ±{TOL_RISE:.0%} rise and ±{TOL_TPEAK:.0%} time; rise error median {tpf.rise_rel_err.median():+.0%}, range {tpf.rise_rel_err.min():+.0%} to {tpf.rise_rel_err.max():+.0%}.
* Runtime: {rt[(rt.case == "final") & (rt.N == N)]["mean"].iloc[0]:.0f} s per breakthrough (6000 s) at N = {N} with Taqvi-LeVan,
  {rt[(rt.case == "mixing: iast") & (rt.N == N)]["mean"].iloc[0]:.0f} s with IAST.

## Before / after

Breakthrough-time error vs experiment (`role`: CO2 times of runs 1-3 were in the calibration objective).

{md(ba, {"before": "{:+.1%}", "after": "{:+.1%}"})}

Thermocouples (`rms`: RMS of T_sim - T_exp over 0-{T_FIT:.0f} s; `rise_err`: error of the peak rise above 323 K).

{md(tba, {"exp_peak_K": "{:.1f}", "peak_before_K": "{:.1f}", "peak_after_K": "{:.1f}", "rise_err_before": "{:+.0%}",
          "rise_err_after": "{:+.0%}", "rms_before_K": "{:.2f}", "rms_after_K": "{:.2f}", "peak_fit_K": "{:.1f}",
          "rise_err_fit": "{:+.0%}", "rms_fit_K": "{:.2f}"})}

Calibration objective per run (same definition for all runs; lower is better).

{md(ob, {c: "{:.1f}" for c in ob.columns if c not in ("role", "run")})}

{thermal_section}
## CH4 kinetics (decision 1, errata E7)

With the Table 4.10 refit for CH4 the run-5 CH4 errors are t10 {pct(ch4_t10)}, t50 {pct(ch4_t50)}
({"within" if max(abs(ch4_t10), abs(ch4_t50)) <= TOL_T else "outside"} the 10 % gate; the Table 5.7 constants gave
{pct(rel(bef, 5, "CH4", "t10"))} / {pct(rel(bef, 5, "CH4", "t50"))}). Side effect on the other minor species of run 5 (same run, no other
change): N2 t10 {side["N2"][0]:+.1%} -> {side["N2"][1]:+.1%}{" (now outside the gate)" if abs(side["N2"][1]) > TOL_T >= abs(side["N2"][0]) else ""},
CO t10 {side["CO"][0]:+.1%} -> {side["CO"][1]:+.1%}: with CH4 adsorbing at the measured rate, the N2 and CO fronts arrive later.
{"" if abs(ch4_t10) <= TOL_T else f"""**Known limitation:** CH4 t10 stays outside the gate. For context, the author's own model misses the same quantity
by {err(au_ch4, exp_ch4):+.0%} (t10 {au_ch4:.0f} s vs {exp_ch4:.0f} s measured), and the sensitivity case (b) gives {pct(ch4_b)}.
"""}Which impurity limits product purity:
* Run 5 (breakthrough): N2 breaks through first (measured t10: N2 {t10e["N2"]:.0f} s, CO {t10e["CO"]:.0f} s,
  CH4 {t10e["CH4"]:.0f} s, CO2 {t10e["CO2"]:.0f} s). In the final model the He-free H2 purity drops below 99.99 %
  at {pf.t_below_9999_s:.0f} s, with the impurity made up of {shares}: **{pf.limiting}** limits purity.
  CH4 arrives last of the minor impurities; its kinetics act on purity only indirectly, through the N2 and CO fronts (above).
* PSA experiment (held out, not simulated here): {exp_psa.h2_purity_pct} % H2, i.e. {imp_ppm:.0f} ppm impurities, of which
  the thesis attributes {exp_psa.co_ppm:.0f} ppm to CO (p.191); the remaining {imp_ppm - exp_psa.co_ppm:.0f} ppm are not
  attributed. In the author's simulations of the same unit the product impurities are "CH4, CO and, mainly N2" (p.199).
  CO is the specification-relevant impurity (fuel cells, < 10 ppm, p.158) and N2 the first to break through; CH4 is
  neither.

## H2 co-adsorption: three mixture rules (decision 2)

H2 loading at the feed (5 bar, 323 K) [mol/kg] under each rule, and the IAST spreading pressure (psi, mol/kg) with the
species whose pure virial isotherm cannot reach it:

{md(h2_loading, {"taqvi_levan": "{:.2e}", "iast": "{:.2e}", "pure_H2_at_P_H2": "{:.2e}", "iast_psi": "{:.2f}"})}

IAST implementation (`Virial(mixing="iast")`): reduced spreading pressure of the pure virial isotherm
psi = q + (A/S) q^2 + (B/S^2) q^3 in closed form; safeguarded Newton on ln psi with a nested Newton for each pure
loading. Tests: exact reduction to the pure isotherm (< 1e-9) and the Gibbs adsorption isotherm d psi/d ln P_i = q_i
(< 1e-5) for binary, five-component and trace states (`tests/test_isotherms.py`). **Limitation found:** for CO, CH4
and N2 the pure virial fits turn over (P(q) has a maximum at 11-25 bar, beyond the 0-7 bar fit), so psi cannot exceed
psi_max (N2 2.33, CO 3.95, CH4 5.47 mol/kg at 323 K). IAST needs these hypothetical pure states whenever CO2 is loaded;
above psi_max the pure isotherm is continued at constant loading (saturated), which keeps IAST defined and consistent
but makes its weak-species loadings an extrapolation. Species past psi_max at the feed are listed in the table.

H2 outlet flow / H2 feed flow (windows defined by the measured CO2 front: `before` it, `peak` around it, `late` after
2500 s, where 1.000 is exact):

{md(h2c[["run", "case", "exp_before", "sim_before", "exp_peak", "sim_peak", "exp_late", "sim_late", "n_before"]],
    {c: "{:.3f}" for c in ["exp_before", "sim_before", "exp_peak", "sim_peak", "exp_late", "sim_late"]})}

Run 1 (30 % CO2), H2 flow before the CO2 front: measured {g(1, "taqvi_levan (final)", "exp_before"):.3f}; Taqvi-LeVan
{g(1, "taqvi_levan (final)", "sim_before"):.3f}, IAST {g(1, "iast", "sim_before"):.3f}, bound {g(1, "noncompetitive_h2", "sim_before"):.3f}. Measured late-plateau offset
{g(1, "taqvi_levan (final)", "exp_late") - 1:+.1%}. Run 3 (80 % CO2): measured {g(3, "taqvi_levan (final)", "exp_before"):.3f}; {g(3, "taqvi_levan (final)", "sim_before"):.3f} /
{g(3, "iast", "sim_before"):.3f} / {g(3, "noncompetitive_h2", "sim_before"):.3f}. The three rules span a band that the breakthrough data cannot narrow; in
step 3 the band is carried into purity and recovery.

## Particle specific area (decision 4, diagnostic)

The thesis gives AC5-KS only as cylindrical extrudates with R_p = 1.45 mm (Table 4.5) and uses the infinite-cylinder
LDF factor Omega_p = 8 (Table 4.7); no AC5-KS pellet length is reported. The parent R2030 extrudates (same 2.90 mm
diameter) are 6.45 mm long (Table 3.2), used here for the finite-cylinder case only.

{md(ap, {c: ("{:.0f}" if "width" in c else "{:+.1%}") for c in ap.columns if c not in ("run", "species")})}

Run 1 CO2 front width (t50 - t10): measured {ap1.exp_width_s:.0f} s; 2/R_p {ap1["width 2/Rp"]:.0f} s, 3/R_p {ap1["width 3/Rp"]:.0f} s,
2/R_p + 2/L_p {ap1["width 2/Rp+2/Lp"]:.0f} s. **Not adopted:** 3/R_p is the sphere value and contradicts the cylindrical
extrudates; the end-face term needs a pellet length the thesis does not give for AC5-KS, and the thesis's own particle
model is the infinite cylinder. a_p = 2/R_p stays (`assumed:` block). The effect on the front width is small in either
case, so a_p does not explain the steeper measured CO2 front.

## Numerical gates (final model)

{md(num, {"worst": "{:.2e}", "limit": "{:.0e}"})}

Mole balance: worst |closure| over runs 1-5, per component and overall. Grid: t50 of every adsorbing species, N = {N}
vs {N2}. Linear limit: single tracer (1 % in He), isothermal, q = K P, t_st = {t_st:.1f} s.

## Breakthrough times vs experiment, final model (gate: |error| <= {TOL_T:.0%})

H2 is not in the table: the first H2 sample of every run is taken after H2 has already broken through.

{md(fin[["run", "species", "level", "sim_s", "exp_s", "rel_err", "within_10pct"]], {"rel_err": "{:+.1%}", "sim_s": "{:.0f}", "exp_s": "{:.0f}"})}

Sensitivity and diagnostic cases (relative error; within 10 %: {", ".join(f"{k} {v[0]}/{v[1]}" for k, v in n_diag.items())}):

{md(diag, {c: "{:+.1%}" for c in diag.columns if c not in ("run", "species", "level")})}

## Temperature peaks, final model (qualitative gate)

"Qualitatively right": peak rise above 323 K within {TOL_RISE:.0%}, peak time within {TOL_TPEAK:.0%}, same Bottom -> Middle -> Top
ordering. sigma_T = {sT:.2f} K (RMS thermocouple deviation from 323 K after 4000 s, runs 1-3). The early dip of the Top
thermocouple (heater switching off, p.191) is an experimental artefact the model does not contain.

{md(tpf[["run", "tc", "sim_peak_K", "exp_peak_K", "rise_rel_err", "sim_time_s", "exp_time_s", "time_rel_err", "rms_K", "qualitative_ok"]],
    {"sim_peak_K": "{:.1f}", "exp_peak_K": "{:.1f}", "rise_rel_err": "{:+.0%}", "sim_time_s": "{:.0f}",
     "exp_time_s": "{:.0f}", "time_rel_err": "{:+.0%}", "rms_K": "{:.2f}"})}

## Our model vs the author's gPROMS simulation (final model)

Author's curves vector-extracted from Figs 5.15-5.17 (±12 s on the 0-6000 s axes, ±3 s zoomed). Temperatures compare
the rise above 323 K.

{md(au, {"ours": "{:.1f}", "author": "{:.1f}", "rel_diff": "{:+.1%}"})}

## Runtime per breakthrough simulation (6000 s of process time)

{md(rt, {"mean": "{:.1f} s", "min": "{:.1f} s", "max": "{:.1f} s"})}

## Figures

Symbols: experiment; solid: final model; dashed: before the decisions; fine dots: author's simulation.

""" + "\n".join(f"![run {k}](run{k}.png)" for k in RUNS) + "\n"
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
