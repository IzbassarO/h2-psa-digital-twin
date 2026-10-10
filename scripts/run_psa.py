"""Step 3: single-column PSA cycle to CSS, validated on the PSA experiment (Lopes 2010, pp.191-200).

    python scripts/run_psa.py [--workers 4] [--force] [--report-only] [--only baseline,sim03,...]

Every case runs from the He-filled column (p.166) to cyclic steady state and is cached in
results/step3/cases/<case>.pkl (reused unless --force). Then the pre-declared gates are evaluated and
results/step3/REPORT.md, figures and CSVs are written; REPORT.md and figures are copied to
docs/validation/step3.md and docs/validation/step3/ (committed).
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.column import PARAMS, wall_heat_capacity  # noqa: E402
from h2psa.cycle import CycleRunner, experiment_cycle, psa_model, simulation_cycle  # noqa: E402
from h2psa.isotherms import load_yaml  # noqa: E402

OUT = ROOT / "results" / "step3"
CASES_DIR = OUT / "cases"
DOCS = ROOT / "docs" / "validation"
BENCH = ROOT / "data" / "benchmarks"
DIG = BENCH / "digitized"
N = 30
RECORD = tuple(range(1, 11)) + (37,)          # cycles 1-10 (Fig. 5.18b) and 37 (Fig. 5.19)
COLOUR = {"H2": "#2a78d6", "CO2": "#eb6834", "CO": "#1baf7a", "CH4": "#eda100", "N2": "#e87ba4"}
MARKER = {"H2": "^", "CO2": "s", "CO": "x", "CH4": "+", "N2": "D"}
TC = ["Bottom", "Middle", "Top"]
TC_COLOUR = {"Bottom": "#6da7ec", "Middle": "#2a78d6", "Top": "#104281"}
# pre-declared gates (step-3 request; not to be changed after seeing results)
GATE = dict(css_tol=1e-4, closure=1e-3, grid_rec_pp=0.3, grid_ppm_rel=0.10, rec_pp=3.0, ppm_factor=2.0,
            co_factor=3.0, rho_rec=0.9, rho_ppm=0.8)
AUTHOR_CHECK = (1, 3, 4, 7, 13)               # Table 5.10 sims rerun with pure-H2 EP inflow (diagnostic)
# pairwise effects in Table 5.10: (from sim, to sim, what changes)
PAIRS = [(1, 2, "purge 30->45 s (A, 5 bar)"), (3, 4, "purge 30->45 s (B, 5 bar)"),
         (1, 3, "1 -> 3 equalisations (purge 30 s, 5 bar)"), (2, 4, "1 -> 3 equalisations (purge 45 s, 5 bar)"),
         (5, 7, "1 -> 3 equalisations (feed 100 s, 10 bar)"), (6, 8, "1 -> 3 equalisations (feed 120 s, 10 bar)"),
         (1, 5, "5 -> 10 bar (A)"), (3, 7, "5 -> 10 bar (B)"),
         (5, 6, "feed 100->120 s (A, 10 bar)"), (7, 8, "feed 100->120 s (B, 10 bar)"),
         (8, 9, "blow 200->150 s"), (9, 10, "purge 30->20 s"), (10, 11, "feed 120->110 s"),
         (11, 12, "blow 150->100 s"), (12, 13, "blow 100->70 s"), (13, 14, "feed 110->120 s"),
         (7, 15, "303 -> 323 K (100/200/30 s)"), (14, 16, "303 -> 323 K (120/70/20 s)"),
         (16, 17, "purge 20->35 s (323 K)")]


# ---------------------------------------------------------------------------- cases
def cases():
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    ac = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")
    asm, fit = cc["assumed"], cc["fitted"]
    cps, ew = asm["solid_heat_capacity"]["range"], asm["wall_thickness"]["range"]
    bl, bh = fit["pressure_decay_rate"]["interval"]
    out = {
        "baseline": dict(),
        "grid_2N": dict(N=2 * N),
        "band_noncompetitive_h2": dict(opts=dict(mixing="noncompetitive_h2")),
        "band_kinetics_table4.10": dict(opts=dict(measured_kinetics=[k for k in ac["micropore_kinetics_breakthrough"] if k != "units"])),
        "band_ep_gas_h2": dict(cycle=dict(ep_gas="h2")),
        "band_thermal_low": dict(opts=dict(thermal={"Cps": cps[0], "Cw": wall_heat_capacity(ew[0])})),
        "band_thermal_high": dict(opts=dict(thermal={"Cps": cps[1], "Cw": wall_heat_capacity(ew[1])})),
        "diag_betaP_low": dict(cycle=dict(beta_P=bl)),
        "diag_betaP_high": dict(cycle=dict(beta_P=bh)),
    }
    for _, row in pd.read_csv(BENCH / "lopes2010_psa_simulations.csv").iterrows():
        out[f"sim{int(row['sim']):02d}"] = dict(sim=row.to_dict(), group=row["transport_group"])
        if int(row["sim"]) in AUTHOR_CHECK:     # what EP gas did the author use? (diagnostic vs Table 5.10, not the experiment)
            out[f"diag_sim{int(row['sim']):02d}_ep_gas_h2"] = dict(sim=row.to_dict(), group=row["transport_group"],
                                                                 cycle=dict(ep_gas="h2"))
    return out


def run_case(name, spec):
    base = experiment_cycle(**spec.get("cycle", {}))
    c = simulation_cycle(spec["sim"], base) if "sim" in spec else base
    m = psa_model(c, N=spec.get("N", N), group=spec.get("group", "1-4"), **spec.get("opts", {}))
    run = CycleRunner(m, c)
    log = lambda s: print(f"[{name}] {s}", flush=True)
    res = run.run_to_css(record=RECORD if name in ("baseline", "grid_2N") else (37,), log=log)
    res.update(name=name, spec=spec, species=m.sp, cycle=c, N=m.N, z_tc=run.z_tc)
    with open(CASES_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    log(f"done: CSS {res['css']} after {res['cycles']} cycles, {res['runtime_s'] / 60:.1f} min wall, "
        f"{res['cpu_s'] / 60:.1f} min CPU")
    return name


def load_case(name):
    with open(CASES_DIR / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------- comparisons
def read(name):
    return pd.read_csv(DIG / f"lopes2010_{name}.csv", comment="#")


def outlet(tr, sp):
    """Molar flow leaving the bed [mmol/s] at whichever end is open (Fig. 5.19 'bed outlet')."""
    return {s: 1e3 * (np.clip(tr["out_L"][i], 0, None) + np.clip(-tr["in_0"][i], 0, None)) for i, s in enumerate(sp)}


def perf_at(res, cycle):
    h = pd.DataFrame(res["history"])
    return h[h.cycle == cycle].iloc[0].to_dict() if (h.cycle == cycle).any() else None


def flow_errors(res, cyc):
    tr, rows = res["traces"].get(cyc), []
    if tr is None:
        return rows
    mod = outlet(tr, res["species"])
    for panel in ("flow", "zoom"):
        e = read(f"psa_c{cyc}_{panel}")
        for s, g in e.groupby("series"):
            m = np.interp(g.time_s, tr["t"], mod[s])
            rows.append(dict(cycle=cyc, panel=panel, species=s, n=len(g), mean_exp=g.molar_flow_mmol_s.mean(),
                             mean_model=m.mean(), rms_err=np.sqrt(np.mean((m - g.molar_flow_mmol_s) ** 2)),
                             max_abs_err=np.max(np.abs(m - g.molar_flow_mmol_s))))
    return rows


def temperature_trace(res, cycles):
    """Thermocouple temperatures over consecutive recorded cycles, time from the start of the first."""
    T0 = res["cycle"].time
    t = np.concatenate([res["traces"][k]["t"] + (k - cycles[0]) * T0 for k in cycles])
    T = np.concatenate([res["traces"][k]["T"] for k in cycles], axis=1)
    return t, T


def temperature_errors(res):
    rows = []
    for label, name, cycles in (("cycles 1-10", "psa_c1-10_T", list(range(1, 11))), ("cycle 1", "psa_c1_T", [1]),
                                ("cycle 37", "psa_c37_T", [37])):
        if not all(k in res["traces"] for k in cycles):
            continue
        t, T = temperature_trace(res, cycles)
        e = read(name)
        for k, s in enumerate(TC):
            g = e[e.series == s]
            m = np.interp(g.time_s, t, T[k])
            rows.append(dict(window=label, thermocouple=s, n=len(g), rms_K=np.sqrt(np.mean((m - g.temperature_K) ** 2)),
                             bias_K=np.mean(m - g.temperature_K), swing_exp_K=g.temperature_K.max() - g.temperature_K.min(),
                             swing_model_K=m.max() - m.min()))
    return rows


def pressure_errors(res):
    tr = res["traces"][1]
    e = read("psa_c1_P")
    e = e[~((e.time_s > 180) & (e.time_s < 200) & (e.pressure_bar > 4))]
    P_open = np.where(np.isin(tr["step"], ["Blow", "Purge"]), tr["P0"], tr["PL"]) / 1e5
    m = np.interp(e.time_s, tr["t"], P_open)
    edges = np.cumsum([0] + [d for _, d, _ in res["cycle"].steps])
    names = [n for n, _, _ in res["cycle"].steps]
    st = np.array([names[min(np.searchsorted(edges, x, side="right") - 1, len(names) - 1)] for x in e.time_s])
    return [dict(step=s, n=int((st == s).sum()), rms_bar=np.sqrt(np.mean((m[st == s] - e.pressure_bar.values[st == s]) ** 2)))
            for s in names if (st == s).any()]


# ---------------------------------------------------------------------------- figures
def fig_pressure(res, path):
    tr = res["traces"][1]
    e, a = read("psa_c1_P"), read("psa_c1_P_authorsim")
    e = e[~((e.time_s > 180) & (e.time_s < 200) & (e.pressure_bar > 4))]
    P_open = np.where(np.isin(tr["step"], ["Blow", "Purge"]), tr["P0"], tr["PL"]) / 1e5
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(a.time_s, a.pressure_bar, ",", color="0.6", label="author's simulation (Fig. 5.18a)")
    ax.plot(e.time_s, e.pressure_bar, "o", ms=3, mfc="none", color="0.2", label="experiment (Fig. 5.18a)")
    ax.plot(tr["t"], P_open, color="#2a78d6", lw=1.8, label=f"model, beta_P = {res['cycle'].beta_P:.2f} 1/s")
    for b in np.cumsum([d for _, d, _ in res["cycle"].steps])[:-1]:
        ax.axvline(b, color="0.85", lw=0.6, zorder=0)
    ax.set(xlabel="time in cycle 1 [s]", ylabel="pressure at the open end [bar]", xlim=(0, 459))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_flows(res, path):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for col, cyc in enumerate((1, 37)):
        tr = res["traces"].get(cyc)
        if tr is None:
            continue
        mod = outlet(tr, res["species"])
        for row, panel in enumerate(("flow", "zoom")):
            ax = axs[row, col]
            e = read(f"psa_c{cyc}_{panel}")
            for s in COLOUR:
                if panel == "zoom" and s == "H2":
                    continue
                ax.plot(tr["t"], mod[s], color=COLOUR[s], lw=1.5, label=f"{s}")
                g = e[e.series == s]
                ax.plot(g.time_s, g.molar_flow_mmol_s, MARKER[s], color=COLOUR[s], ms=5, mfc="none")
            ax.set(xlim=(0, 459) if panel == "flow" else (160, 400), ylim=(0, 2.8) if panel == "flow" else (0, 0.45),
                   xlabel="time in cycle [s]", ylabel="outlet molar flow [mmol/s]",
                   title=f"cycle {cyc}" + (" (low range)" if panel == "zoom" else ""))
            for b in np.cumsum([d for _, d, _ in res["cycle"].steps])[:-1]:
                ax.axvline(b, color="0.88", lw=0.6, zorder=0)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, title="lines: model, symbols: experiment", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_temperature(res, path):
    fig, axs = plt.subplots(3, 3, figsize=(13, 7.5), sharex="col")
    for col, (name, cycles, title) in enumerate((("psa_c1-10_T", list(range(1, 11)), "cycles 1-10 (Fig. 5.18b)"),
                                                 ("psa_c1_T", [1], "cycle 1 (Fig. 5.19 1.b)"),
                                                 ("psa_c37_T", [37], "cycle 37 (Fig. 5.19 2.b)"))):
        if not all(k in res["traces"] for k in cycles):
            continue
        t, T = temperature_trace(res, cycles)
        e = read(name)
        for k, s in enumerate(reversed(TC)):       # Top first, as in the thesis
            ax = axs[k, col]
            kk = TC.index(s)
            g = e[e.series == s]
            ax.plot(g.time_s, g.temperature_K, "o", ms=2, color=TC_COLOUR[s], alpha=0.6)
            ax.plot(t, T[kk], color="0.15", lw=1.2)
            ax.set_ylabel(f"{s} [K]")
            if k == 0:
                ax.set_title(title, fontsize=10)
        axs[2, col].set_xlabel("time [s]")
    fig.suptitle("thermocouples: symbols experiment, line model", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_css(results, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, res in results.items():
        if not (name == "baseline" or name.startswith("band_")):
            continue
        h = pd.DataFrame(res["history"])
        kw = dict(lw=2.0, color="0.1") if name == "baseline" else dict(lw=1.0)
        axs[0].plot(h.cycle, h.ppm, label=name, **kw)
        axs[1].plot(h.cycle, h.recovery_pct, label=name, **kw)
    axs[0].axhline(195, color="#d23f3f", ls="--", lw=1)
    axs[1].axhline(81.6, color="#d23f3f", ls="--", lw=1)
    axs[0].set(yscale="log", xlabel="cycle", ylabel="product impurities [ppm]", ylim=(1, None))
    axs[1].set(xlabel="cycle", ylabel="H2 recovery [%]", ylim=(70, 90))
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sims(sims, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (a, b, lab, log) in zip(axs, (("rec_author", "rec_model", "H2 recovery [%]", False),
                                          ("ppm_author", "ppm_model", "product impurities [ppm]", True))):
        x, y = sims[a].clip(lower=0.1), sims[b].clip(lower=0.1)
        ax.plot(x, y, "o", color="#2a78d6")
        for _, r in sims.iterrows():
            ax.annotate(int(r.sim), (max(r[a], 0.1), max(r[b], 0.1)), fontsize=7, xytext=(3, 2), textcoords="offset points")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8)
        if log:
            ax.set(xscale="log", yscale="log")
        ax.set(xlabel=f"author (Table 5.10): {lab}", ylabel=f"model: {lab}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
FINDINGS = """## Findings (written after the results; the numbers are in the tables below)

1. **Numerics are sound (gate 1 PASS).** CSS after 178 cycles from the He-filled column (thesis: 240), cycle
   mole balance within 4e-6 in every case, N = 30 -> 60 changes recovery by 0.02 pp and impurities by 1.2 %.
2. **Recovery is predicted, purity is not (gate 2 FAIL).** Recovery at CSS is 81.86 % (experiment 81.6 %;
   productivity 101.0 vs 101.1 mol/kg/day). Impurities are 1825 ppm (9.4x the measured 195 ppm) and CO 675 ppm
   (10.7x the measured 63 ppm); 55 % of the impurity is N2, 37 % CO. Cycle 37 gives the same picture (1750 ppm).
   The model is **pessimistic** on purity, the opposite of the bias expected from step 2 (N2 released 10-13 %
   late in breakthrough run 5). Outlet flows of cycles 1 and 37 and the bottom thermocouple are reproduced;
   middle and top swings are about half the measured ones.
3. **The same excess appears against the author's own model (gate 3).** Ranks agree (Spearman 0.98 for
   recovery, 0.84 for impurity) and 37 of 38 pairwise effects have the author's sign; the exception is blow
   200 -> 150 s (sims 8 -> 9), where the author's impurity rises by 5 ppm and ours falls by 19 ppm (both < 2 %).
   But our impurity is 7-1000x the author's in every simulation, so gate 3 fails on the strict sign rule and the
   absolute differences show a systematic difference between the two implementations, not only between model and
   experiment.
4. **The EP inflow composition is the largest single unknown.** With pure-H2 EP gas the experiment case gives
   352 ppm (CO 154 ppm) instead of 1825 ppm; recovery is unchanged (eq. 5.49 does not count EP gas). Against the
   author's simulations, pure-H2 EP cuts our impurity 2.5-8x but still leaves it 2-150x above Table 5.10, so it
   explains part of the gap, not all. Evidence on what the experiment did, not used for any choice: it is a
   single column with no partner bed, and in Fig. 5.18a the pressure rises along one straight line through
   EP3-EP1 and Press (as a mass-flow controller would give), not in the exponential steps of the ED side.
5. **Other band members move impurity less:** Table 4.10 kinetics 1266 ppm, thermal ends 1687-2017 ppm,
   noncompetitive H2 2076 ppm (and recovery 79.2 %); beta_P at the ends of its interval 1064-3024 ppm. The single
   beta_P is a poor description of the measured pressure (ED steps 0.45-0.53 bar RMS).
6. **Cause: no coding bug found, not excluded.** Checked: mole balance per step and cycle, eqs. 5.48-5.50
   against the step integrals (tests), pressure laws, EP gas bookkeeping, grid, and cycle 1 (clean bed) against
   both the measured and the author's simulated outlet flows (Fig. 5.19), which agree. The excess builds up over
   the first ~20 cycles as N2 and CO accumulate at the product end. Per the instructions, I stopped here and did
   not tune anything.

### Options for the project lead

* **(a) EP inflow = pure H2 for the experiment** (documented evidence in 4). The experiment case then sits at
  352 ppm / 154 ppm CO / 81.9 % recovery, inside the pre-declared factors, but the choice is made after seeing
  results, and the gap to the author's simulations remains.
* **(b) Find the remaining difference to the author's model before any validation claim** (recommended).
  Reproduce simulation 13 at CSS against everything the thesis shows for it: pressure and outlet flows
  (Fig. 5.20), loading profiles at the end of every step (Fig. 5.21), purity/recovery vs cycle (Fig. 5.22) and
  the N2 gas profile at the end of the feed (Fig. 5.23b). That locates where the impurity fronts diverge
  (candidates: EP gas, the pressure laws, heat effects, film/axial terms) without using the experiment.
* **(c) Per-step-type pressure constants** (ED fast, Blow and EP slow), fitted to Fig. 5.18a only, as a
  better description of the measured valve behaviour; a sensitivity on its own.
* **(d) Accept gate 2 as failed and carry a known pessimistic purity bias** into steps 4-7. Not recommended:
  the paper's claim is about the 99.99 % purity constraint, where this bias matters most.
* Gate 3 sign failure (sims 8 -> 9): accept as noise, or keep it as a failure (both effects are below 2 %)."""

def md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[k]
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                cells.append("–")
            elif c in fmt and isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                cells.append(fmt[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ok(b):
    return "PASS" if b else "FAIL"


def analyse():
    names = list(cases())
    results = {n: load_case(n) for n in names if (CASES_DIR / f"{n}.pkl").exists()}
    base = results["baseline"]
    exp = pd.read_csv(BENCH / "lopes2010_psa_experiment.csv").iloc[0]
    exp_ppm = 1e6 * (1 - exp.h2_purity_pct / 100)
    t = {}

    # ---- gate 1: numerical
    hb = pd.DataFrame(base["history"])
    g1 = [dict(check="CSS reached (all CSS quantities < 1e-4 relative over 10 cycles)", value=f"{base['cycles']} cycles",
               limit="criterion met", result=ok(base["css"]))]
    clo = max(abs(pd.DataFrame(r["history"]).closure).max() for r in results.values())
    g1.append(dict(check="per-cycle mole balance, worst cycle of all cases", value=f"{clo:.1e}", limit="< 1e-3",
                   result=ok(clo < GATE["closure"])))
    if "grid_2N" in results:
        p1, p2 = base["perf"], results["grid_2N"]["perf"]
        drec, dppm = abs(p2["recovery_pct"] - p1["recovery_pct"]), abs(p2["ppm"] / p1["ppm"] - 1)
        g1 += [dict(check=f"grid N = {N} -> {2 * N}: recovery", value=f"{drec:.3f} pp", limit="< 0.3 pp",
                    result=ok(drec < GATE["grid_rec_pp"])),
               dict(check=f"grid N = {N} -> {2 * N}: impurity ppm", value=f"{100 * dppm:.1f} %", limit="< 10 %",
                    result=ok(dppm < GATE["grid_ppm_rel"]))]
    t["gate1"] = pd.DataFrame(g1)

    # ---- gate 2: experiment
    rows = []
    for label, p in (("CSS", base["perf"]), ("cycle 37", perf_at(base, 37))):
        if p is None:
            continue
        rows.append(dict(cycle=label, recovery_pct=p["recovery_pct"], ppm=p["ppm"], co_ppm=p["co_ppm"],
                         n2_ppm=p.get("imp_N2_ppm"), ch4_ppm=p.get("imp_CH4_ppm"), co2_ppm=p.get("imp_CO2_ppm"),
                         purity_pct=p["purity_pct"], productivity=p["productivity"],
                         rec_ok=ok(abs(p["recovery_pct"] - exp.h2_recovery_pct) <= GATE["rec_pp"]),
                         ppm_ok=ok(1 / GATE["ppm_factor"] <= p["ppm"] / exp_ppm <= GATE["ppm_factor"]),
                         co_ok=ok(1 / GATE["co_factor"] <= p["co_ppm"] / exp.co_ppm <= GATE["co_factor"])))
    rows.append(dict(cycle="experiment", recovery_pct=exp.h2_recovery_pct, ppm=exp_ppm, co_ppm=exp.co_ppm,
                     purity_pct=exp.h2_purity_pct, productivity=exp.productivity_mol_kg_day))
    t["gate2"] = pd.DataFrame(rows)
    t["flow_errors"] = pd.DataFrame(flow_errors(base, 1) + flow_errors(base, 37))
    t["temperature_errors"] = pd.DataFrame(temperature_errors(base))
    t["pressure_errors"] = pd.DataFrame(pressure_errors(base))

    # ---- band (report, do not select)
    band = []
    for n, r in results.items():
        if n == "baseline" or (n.startswith(("band_", "diag_")) and not n.startswith("diag_sim")):
            p, p37 = r["perf"], perf_at(r, 37)
            band.append(dict(case=n, recovery_pct=p["recovery_pct"], ppm=p["ppm"], co_ppm=p["co_ppm"],
                             n2_ppm=p["imp_N2_ppm"], recovery_c37=p37["recovery_pct"], ppm_c37=p37["ppm"],
                             cycles=r["cycles"], css=r["css"]))
    t["band"] = pd.DataFrame(band)

    # ---- gate 3: author's simulations
    sim = pd.read_csv(BENCH / "lopes2010_psa_simulations.csv")
    srows = []
    for _, s in sim.iterrows():
        n = f"sim{int(s.sim):02d}"
        if n not in results:
            continue
        p = results[n]["perf"]
        srows.append(dict(sim=int(s.sim), cycle=s.cycle, P_high=s.P_high_bar, T=s.T_feed_K, rec_author=s.h2_recovery_pct,
                          rec_model=p["recovery_pct"], ppm_author=1e6 * (1 - s.h2_purity_pct / 100), ppm_model=p["ppm"],
                          co_author=s.co_ppm, co_model=p["co_ppm"], prod_author=s.productivity_mol_kg_day,
                          prod_model=p["productivity"], cycles=results[n]["cycles"]))
    sims = pd.DataFrame(srows)
    t["sims"] = sims
    g3, pairs = [], []
    if len(sims) == len(sim):
        rr = spearmanr(sims.rec_author, sims.rec_model)[0]
        rp = spearmanr(sims.ppm_author, sims.ppm_model)[0]     # ranks: same for ppm and ln(ppm)
        g3 += [dict(check="Spearman rank correlation, recovery", value=f"{rr:.3f}", limit=">= 0.9", result=ok(rr >= GATE["rho_rec"])),
               dict(check="Spearman rank correlation, ln(impurity ppm)", value=f"{rp:.3f}", limit=">= 0.8",
                    result=ok(rp >= GATE["rho_ppm"]))]
        S = sims.set_index("sim")
        for a, b, what in PAIRS:
            for q in ("rec", "ppm"):
                da, dm = S.loc[b, f"{q}_author"] - S.loc[a, f"{q}_author"], S.loc[b, f"{q}_model"] - S.loc[a, f"{q}_model"]
                pairs.append(dict(pair=f"{a} -> {b}", change=what, quantity="recovery [pp]" if q == "rec" else "impurity [ppm]",
                                  author=da, model=dm, same_sign=ok(np.sign(da) == np.sign(dm))))
        pairs = pd.DataFrame(pairs)
        nbad = (pairs.same_sign == "FAIL").sum()
        g3.append(dict(check="pairwise effects with the same sign", value=f"{len(pairs) - nbad}/{len(pairs)}",
                       limit="all", result=ok(nbad == 0)))
    t["gate3"], t["pairs"] = pd.DataFrame(g3), pd.DataFrame(pairs)
    ac = []
    for k in AUTHOR_CHECK:
        n = f"diag_sim{k:02d}_ep_gas_h2"
        if n in results and len(sims):
            s = sims.set_index("sim").loc[k]
            p = results[n]["perf"]
            ac.append(dict(sim=k, rec_author=s.rec_author, rec_ep_ed=s.rec_model, rec_ep_h2=p["recovery_pct"],
                           ppm_author=s.ppm_author, ppm_ep_ed=s.ppm_model, ppm_ep_h2=p["ppm"],
                           co_author=s.co_author, co_ep_ed=s.co_model, co_ep_h2=p["co_ppm"]))
    t["author_check"] = pd.DataFrame(ac)

    # ---- runtime
    t["runtime"] = pd.DataFrame([dict(case=n, N=r["N"], cycles=r["cycles"], wall_min=r["runtime_s"] / 60, cpu_min=r["cpu_s"] / 60,
                                      cpu_s_per_cycle=r["cpu_s"] / (r["cycles"] + 1)) for n, r in results.items()])
    return results, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    todo = {n: s for n, s in cases().items()
            if (not a.only or n in a.only.split(",")) and (a.force or not (CASES_DIR / f"{n}.pkl").exists())}
    if todo and not a.report_only:
        order = sorted(todo, key=lambda n: (n != "grid_2N", n))          # longest first
        with ProcessPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(run_case, n, todo[n]) for n in order]):
                f.result()
    results, t = analyse()
    for k, v in t.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    base = results["baseline"]
    fig_pressure(base, OUT / "pressure.png")
    fig_flows(base, OUT / "flows.png")
    fig_temperature(base, OUT / "temperature.png")
    fig_css(results, OUT / "css.png")
    if len(t["sims"]):
        fig_sims(t["sims"], OUT / "sims.png")
    write_report(results, t)
    (DOCS / "step3").mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        shutil.copy(f, DOCS / "step3" / f.name)
    text = (OUT / "REPORT.md").read_text()
    for f in OUT.glob("*.png"):
        text = text.replace(f"]({f.name})", f"](step3/{f.name})")
    (DOCS / "step3.md").write_text(text)
    print(f"wrote {OUT} and {DOCS / 'step3.md'}")


def verdict(df):
    return "PASS" if len(df) and (df.result == "PASS").all() else "FAIL"


def write_report(results, t):
    base = results["baseline"]
    c = base["cycle"]
    fit = load_yaml(PARAMS / "compact_column_lopes2010.yaml")["fitted"]
    g2 = t["gate2"]
    css = g2[g2.cycle == "CSS"].iloc[0]
    gate2_pass = all(css[k] == "PASS" for k in ("rec_ok", "ppm_ok", "co_ok"))
    f2 = {"recovery_pct": "{:.2f}", "ppm": "{:.0f}", "co_ppm": "{:.1f}", "n2_ppm": "{:.0f}", "ch4_ppm": "{:.1f}",
          "co2_ppm": "{:.2f}", "purity_pct": "{:.4f}", "productivity": "{:.1f}"}
    rt = t["runtime"]
    rb = rt[rt.case == "baseline"].iloc[0]
    text = f"""# Step 3 — single-column PSA cycle to CSS, validated on the PSA experiment

Generated by `scripts/run_psa.py` (cases cached in `results/step3/cases/`). Thesis pages are printed pages.

## Verdict

| gate | result |
|---|---|
| 1. Numerical (CSS, mole balance, grid) | {verdict(t["gate1"])} |
| 2. Experiment (recovery, impurity ppm, CO ppm at CSS) | {"PASS" if gate2_pass else "FAIL"} |
| 3. Author's 17 simulations (rank correlation, effect signs) | {verdict(t["gate3"])} |

{FINDINGS}

## Model and configuration (fixed before any comparison)

* `src/h2psa/cycle.py` on `column.py`: cycles A and B (Figs 5.7, 5.9), boundary conditions eqs. 5.36-5.47,
  performance eqs. 5.48-5.50 (recovery subtracts the H2 entering in purge and pressurisation). Column initially
  full of He at T_feed and P_high (p.166).
* Baseline: E7 kinetics, `taqvi_levan`, assumed thermal values, transport group "1-4" (Table 5.9), N = {N} cells,
  BDF rtol 1e-6.
* **Pressure laws.** ED1-3 and Blow: exponential law of eqs. 5.39-5.40 from the pressure actually reached, one
  rate constant beta_P for all of them (p.180). EP3-1: eq. 5.45 states no pressure law; the author's simulated EP
  pressures are exponential-shaped (Fig. 5.20a, p.197), so the same law and beta_P are used. Press: linear,
  eq. 5.47, from the step-start pressure to P_high.
* **EP inflow composition.** Not stated (eq. 5.45 leaves C_inlet open; no text). Used: the gas that left in the
  matching ED step of the same cycle, well mixed (ED1->EP1, ED2->EP2, ED3->EP3: a tank standing in for the partner
  bed). This is consistent with eq. 5.49, which does not charge EP gas to the product. Pure-H2 inflow is the
  sensitivity case. All inflows enter at the feed temperature.
* **Purge and pressurisation flows.** Not reported. Pressurisation (EP, Press) follows from the pressure laws
  (the inflow is whatever the imposed pressure requires). The purge H2 flow was fitted to the H2 flow leaving at
  z = 0 during the cycle-1 purge (3 GC samples, Fig. 5.19 1.a): **{1e3 * fit["purge_flow"]["value"]:.3f}
  +- {1e3 * fit["purge_flow"]["sd"]:.3f} mmol/s** (`scripts/fit_psa_bcs.py`). No purity or recovery information
  is used. The author's simulated purge outflow in Figs 5.19 and 5.20b is about 0.19 mmol/s.
* **beta_P** was fitted ONLY to the measured cycle-1 pressure history (Fig. 5.18a, digitised from the vector
  graphics, 448 points): **{fit["pressure_decay_rate"]["value"]:.2f} s-1**, interval
  {fit["pressure_decay_rate"]["interval"][0]}-{fit["pressure_decay_rate"]["interval"][1]} s-1. It is a valve
  characteristic: a different observable from the validation targets (purity, recovery). One constant cannot
  follow the record: fitted step by step, ED1-3 give 4.5-9.5 s-1 (plateau reached in about 1 s) and Blow and
  EP3-1 give 0.08-0.12 s-1 (`fit_bcs.csv`). The author's own simulated line in Fig. 5.18a corresponds to about
  0.18 s-1. The two ends of the interval are run as diagnostics.
* **Cycle of the experimental performance.** Not stated. The thesis shows cycles 1 and 37 (Fig. 5.19) and says
  the simulation needs 240 cycles to CSS (p.191). Compared at cycle 37 and at CSS.
* PSA simulations at 323 K (Table 5.10, sims 15-17): heater (ambient side of the wall) at the feed temperature.
  The 10-bar simulations take H2 to 7.9 bar partial pressure, slightly beyond the 0-7 bar isotherm fits.

![pressure](pressure.png)

## Gate 1 — numerical

{md(t["gate1"])}

CSS criterion: impurity ppm, recovery and the bed-averaged loading of every species change by less than 1e-4
(relative) over the last 10 cycles, and every one of those 10 cycles closes its mole balance within 0.1 %.
The thesis needed 240 cycles from the He-filled column (p.191); the baseline here needs {base["cycles"]}.

## Gate 2 — experiment (held out until now)

{md(g2, f2)}

Limits: recovery within +-3 percentage points of 81.6 %, total impurities within a factor of 2 of 195 ppm, CO
within a factor of 3 of 63 ppm. The thesis attributes 63 of the 195 ppm to CO; the remaining 132 ppm are not
broken down (p.199: the impurities are CH4, CO and "mainly N2").

![CSS approach](css.png)

### Shape checks (stated errors, no pass/fail)

Outlet molar flows at the GC sample times, cycles 1 and 37 (Fig. 5.19; 'flow' = main panel, 'zoom' = low range
panel, 160-400 s):

{md(t["flow_errors"], {"mean_exp": "{:.3f}", "mean_model": "{:.3f}", "rms_err": "{:.3f}", "max_abs_err": "{:.3f}"})}

![outlet flows](flows.png)

Thermocouples (Fig. 5.18b cycles 1-10, Fig. 5.19 cycles 1 and 37):

{md(t["temperature_errors"], {"rms_K": "{:.2f}", "bias_K": "{:+.2f}", "swing_exp_K": "{:.1f}", "swing_model_K": "{:.1f}"})}

![temperatures](temperature.png)

Pressure at the open end, cycle 1 (Fig. 5.18a), RMS per step:

{md(t["pressure_errors"], {"rms_bar": "{:.3f}"})}

## Uncertainty band for the experiment (reported, not selected from)

{md(t["band"], {"recovery_pct": "{:.2f}", "ppm": "{:.0f}", "co_ppm": "{:.1f}", "n2_ppm": "{:.0f}", "recovery_c37": "{:.2f}", "ppm_c37": "{:.0f}"})}

Cases: `noncompetitive_h2` (H2 bound); Table 4.10 kinetics for all species (option b); pure-H2 EP inflow; thermal
parameters at the ends of their assumed ranges (low: Cps 700 J/kg/K with a 2 mm wall; high: 1100 J/kg/K with a
5 mm wall). Diagnostics outside the band: beta_P at the ends of its fit interval.

## Gate 3 — the author's 17 simulations (implementation check)

Same model, boundary laws, beta_P, purge flow and EP inflow rule as the baseline; step times, pressures,
temperature and transport group from Tables 5.8-5.10.

{md(t["gate3"])}

{md(t["sims"], {"P_high": "{:.0f}", "T": "{:.0f}", "rec_author": "{:.2f}", "rec_model": "{:.2f}", "ppm_author": "{:.1f}", "ppm_model": "{:.1f}", "co_author": "{:.3g}", "co_model": "{:.3g}", "prod_author": "{:.1f}", "prod_model": "{:.1f}"})}

Impurity ppm of the author = 1e6 (1 - purity); sim 5 is printed as 100.0000 % (below 0.5 ppm), taken as 0 for
the ranking.

![simulations](sims.png)

**Which EP gas did the author use?** Five of the simulations rerun with pure-H2 EP inflow (a diagnostic
against the author's model, Table 5.10, not against the experiment):

{md(t["author_check"], {"rec_author": "{:.2f}", "rec_ep_ed": "{:.2f}", "rec_ep_h2": "{:.2f}", "ppm_author": "{:.1f}", "ppm_ep_ed": "{:.0f}", "ppm_ep_h2": "{:.1f}", "co_author": "{:.3g}", "co_ep_ed": "{:.3g}", "co_ep_h2": "{:.3g}"})}

Pairwise effects (change from the first to the second simulation):

{md(t["pairs"], {"author": "{:+.3g}", "model": "{:+.3g}"})}

## Runtime

{md(rt, {"wall_min": "{:.1f}", "cpu_min": "{:.1f}", "cpu_s_per_cycle": "{:.2f}"})}

Baseline: {rb.cpu_min:.1f} min CPU ({rb.wall_min:.1f} min wall) for {int(rb.cycles)} cycles, {rb.cpu_s_per_cycle:.2f} s
CPU per cycle. Wall times were measured on a shared machine (load average 40-55 on 10 cores from another
project), so CPU time is the comparable figure. Speed-ups done in this step: batched finite-difference Jacobian
(one rhs call for all 49 column groups) and a closed-form rank-2 Newton step for the Taqvi-LeVan isotherm (together
about 3.5x). Further options for step 6: warm start from a neighbouring CSS (allowed; cuts most of the
{base["cycles"]} cycles), parallel runs.
"""
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
