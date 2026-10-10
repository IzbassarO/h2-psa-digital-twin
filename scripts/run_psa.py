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
# measured cyclic data (step-3 decision 2026-10-10, declared before running): H2 per step and CO2 front +-10 %,
# H2 recovery at cycle 37 +-3 pp, thermocouple peak rise +-30 % and peak time +-20 % (step-2 criteria)
GATE_MEAS = dict(h2_rel=0.10, co2_rel=0.10, rec_pp=3.0, rise_rel=0.30, tpeak_rel=0.20)
# verification ladder (step-3 follow-up): one hypothesis at a time on the author's simulations
VARIANTS = {"ep_h2": dict(ep_gas="h2"), "ep_product": dict(ep_gas="product"), "ep_ed_direct": dict(ep_gas="ed_direct"),
            "ep_ed_reversed": dict(ep_gas="ed_reversed"), "beta_steps": "per_step", "beta_author": "author",
            "betaA_ep_h2": "author+h2", "betaA_ep_product": "author+product"}
LADDER_A = (1, 2)
PROFILE_CYCLES = {1: (1, 5, 10, 20), 13: (1, 5, 6, 10, 11, 16, 20, 21, 51, 101, 161, 221, 296)}   # Fig. 5.23 cycles
# corrected model (step-3 decision): author's beta_P (Fig. 5.20a) and pure-H2 EP inflow; sims 1, 2, 13 already ran
COR = dict(beta_P=load_yaml(ROOT / "data" / "params" / "compact_column_lopes2010.yaml")["fitted"]
           ["pressure_decay_rate_author_sims"]["value"], ep_gas="h2")
COR_ALIAS = {1: "ver_sim01_betaA_ep_h2", 2: "ver_sim02_betaA_ep_h2", 13: "dg_sim13_betaA_ep_h2"}
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
    per_step = {k: v for k, v in fit["pressure_decay_rate_per_step"].items() if k in ("ED", "Blow", "EP")}
    for _, row in pd.read_csv(BENCH / "lopes2010_psa_simulations.csv").iterrows():
        out[f"sim{int(row['sim']):02d}"] = dict(sim=row.to_dict(), group=row["transport_group"])
        if int(row["sim"]) in LADDER_A:
            for v, cyc in VARIANTS.items():
                bA = fit["pressure_decay_rate_author_sims"]["value"]
                cyc = {"per_step": dict(beta_steps=per_step), "author": dict(beta_P=bA),
                       "author+h2": dict(beta_P=bA, ep_gas="h2"),
                       "author+product": dict(beta_P=bA, ep_gas="product")}.get(cyc, cyc) if isinstance(cyc, str) else cyc
                out[f"ver_sim{int(row['sim']):02d}_{v}"] = dict(sim=row.to_dict(), group=row["transport_group"], cycle=cyc)
        k = int(row["sim"])
        if k in PROFILE_CYCLES:                  # diagnostics: profiles and step balances (sims 1 and 13)
            base_spec = dict(sim=row.to_dict(), group=row["transport_group"], profiles=PROFILE_CYCLES[k])
            out[f"dg_sim{k:02d}"] = base_spec
            if k == 13:
                out["dg_sim13_beta_author"] = dict(base_spec, cycle=dict(beta_P=fit["pressure_decay_rate_author_sims"]["value"]))
                out["dg_sim13_ep_h2"] = dict(base_spec, cycle=dict(ep_gas="h2"))
                bA = fit["pressure_decay_rate_author_sims"]["value"]
                out["dg_sim13_betaA_ep_h2"] = dict(base_spec, cycle=dict(beta_P=bA, ep_gas="h2"))
                out["dg_sim13_betaA_ep_product"] = dict(base_spec, cycle=dict(beta_P=bA, ep_gas="product"))
                out["dg_sim13_betaA_epsp035"] = dict(base_spec, cycle=dict(beta_P=bA), opts=dict(eps_p=0.35))
        if int(row["sim"]) in AUTHOR_CHECK:     # what EP gas did the author use? (diagnostic vs Table 5.10, not the experiment)
            out[f"diag_sim{int(row['sim']):02d}_ep_gas_h2"] = dict(sim=row.to_dict(), group=row["transport_group"],
                                                                 cycle=dict(ep_gas="h2"))
        if k not in COR_ALIAS:                    # rung c: corrected model on every Table 5.10 simulation
            out[f"cor_sim{k:02d}"] = dict(sim=row.to_dict(), group=row["transport_group"], cycle=COR)
    # corrected model on the experiment cycle: the author's 18th simulation (p.256) and the measured cyclic data
    out["cor_exp"] = dict(cycle=COR, record=RECORD)
    out["cor_exp_grid_2N"] = dict(cycle=COR, N=2 * N)
    out["cor_exp_ep_ed"] = dict(cycle=dict(COR, ep_gas="ed"), record=RECORD)
    for n, o in (("noncompetitive_h2", out["band_noncompetitive_h2"]), ("kinetics_table4.10", out["band_kinetics_table4.10"]),
                 ("thermal_low", out["band_thermal_low"]), ("thermal_high", out["band_thermal_high"])):
        out[f"cor_exp_{n}"] = dict(o, cycle=COR, record=RECORD)
    return out


def cor_case(k):
    """Case holding simulation k with the corrected model (reuses the identical ladder/diagnostic runs)."""
    return COR_ALIAS.get(k, f"cor_sim{k:02d}")


def run_case(name, spec):
    base = experiment_cycle(**spec.get("cycle", {}))
    c = simulation_cycle(spec["sim"], base) if "sim" in spec else base
    m = psa_model(c, N=spec.get("N", N), group=spec.get("group", "1-4"), **spec.get("opts", {}))
    run = CycleRunner(m, c)
    log = lambda s: print(f"[{name}] {s}", flush=True)
    res = run.run_to_css(record=spec.get("record", RECORD if name in ("baseline", "grid_2N") else (37,)), log=log,
                         profiles=spec.get("profiles", ()))
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


def sim13_author(tag):
    return pd.read_csv(DIG / f"lopes2010_sim13_{tag}_authorsim.csv", comment="#")


def front_z(z, f):
    """Position of the descending front: last z where the profile is still above half its maximum."""
    f = np.asarray(f)
    k = int(np.argmax(f))
    below = np.flatnonzero(f[k:] < 0.5 * f[k])
    if not len(below):
        return np.nan
    j = k + below[0]
    return float(np.interp(0.5 * f[k], [f[j], f[j - 1]], [z[j], z[j - 1]]))


def sim13_compare(results):
    """Rung b: our simulation-13 CSS profiles vs Figs 5.20, 5.21, 5.23 (author's model)."""
    rows = []
    for name, res in results.items():
        if not name.startswith("dg_sim13"):
            continue
        last = max(res["profiles"])
        pf = res["profiles"][last]
        sp = res["species"]
        for spc in ("CO2", "CH4", "CO", "N2"):
            a = sim13_author(f"q{spc}")
            for step in ("Feed", "Purge"):
                g = a[a.series == step]
                q = pf[step]["q"][sp.index(spc)]
                rows.append(dict(case=name, quantity=f"q {spc} front z, end of {step} [m]",
                                 author=front_z(g.z_m.values, g.value.values), model=front_z(pf[step]["z"], q)))
                rows.append(dict(case=name, quantity=f"q {spc} max, end of {step} [mol/kg]", author=g.value.max(), model=q.max()))
        a = sim13_author("CN2_feedend")
        g = a[a.series == "cycle 296"]
        C = pf["Feed"]["C"][sp.index("N2")]
        rows.append(dict(case=name, quantity="C N2 front z, end of feed [m]", author=front_z(g.z_m.values, g.value.values),
                         model=front_z(pf["Feed"]["z"], C)))
        rows.append(dict(case=name, quantity="C N2 at z = 0.25 m, end of feed [mol/m3]",
                         author=float(np.interp(0.25, g.z_m, g.value)), model=float(np.interp(0.25, pf["Feed"]["z"], C))))
        rows.append(dict(case=name, quantity="impurities [ppm]", author=20.1, model=res["perf"]["ppm"]))
        rows.append(dict(case=name, quantity="recovery [%]", author=72.88, model=res["perf"]["recovery_pct"]))
    return pd.DataFrame(rows)


def fig_sim13(res, path):
    last = max(res["profiles"])
    pf, tr, sp = res["profiles"][last], res["traces"][last], res["species"]
    fig, axs = plt.subplots(3, 4, figsize=(16, 10.5))
    # Fig. 5.20: pressure and outlet flows
    P_open = np.where(np.isin(tr["step"], ["Blow", "Purge"]), tr["P0"], tr["PL"]) / 1e5
    a = sim13_author("pressure")
    axs[0, 0].plot(a.time_s, a.value, "o", ms=2, color="0.5", label="author (Fig. 5.20a)")
    axs[0, 0].plot(tr["t"], P_open, color="#2a78d6", label="model")
    axs[0, 0].set(title="pressure at the open end [bar]", xlabel="cycle time [s]")
    axs[0, 0].legend(frameon=False, fontsize=7)
    mod = outlet(tr, sp)
    a = sim13_author("H2flow")
    axs[0, 1].plot(a.time_s, a.value, "o", ms=2, color="0.5")
    axs[0, 1].plot(tr["t"], mod["H2"], color=COLOUR["H2"])
    axs[0, 1].set(title="H2 outlet flow [mmol/s]", xlabel="cycle time [s]")
    a = sim13_author("flows")
    for spc in ("CO2", "CH4", "CO", "N2"):
        g = a[(a.series == f"{spc}_mmol_s") & (a.time_s > 150)]
        ax = axs[0, 2] if spc == "CO2" else axs[0, 3]
        ax.plot(g.time_s, g.value, "o", ms=2, color=COLOUR[spc], alpha=0.5)
        ax.plot(tr["t"], mod[spc], color=COLOUR[spc], label=spc)
    axs[0, 2].set(title="CO2 outlet flow [mmol/s]", xlabel="cycle time [s]", xlim=(140, 310))
    axs[0, 3].set(title="CH4, CO, N2 outlet flow [mmol/s]", xlabel="cycle time [s]", xlim=(140, 310), ylim=(0, 0.3))
    axs[0, 3].legend(frameon=False, fontsize=7)
    # Fig. 5.21: loadings at the end of feed and purge
    for j, spc in enumerate(("CO2", "CH4", "CO", "N2")):
        ax = axs[1, j]
        a = sim13_author(f"q{spc}")
        for step, cl in (("Feed", "#d23f3f"), ("Blow", "#7a7a7a"), ("Purge", "#2a78d6"), ("Press", "#1baf7a")):
            g = a[a.series == step]
            ax.plot(g.z_m, g.value, "o", ms=2, color=cl, alpha=0.5)
            ax.plot(pf[step]["z"], pf[step]["q"][sp.index(spc)], color=cl, label=f"end of {step}")
        ax.set(title=f"q {spc} [mol/kg] (Fig. 5.21)", xlabel="z [m]")
    axs[1, 0].legend(frameon=False, fontsize=7, title="symbols author, lines model", title_fontsize=7)
    # Fig. 5.23: end of feed over the cycles
    cyc = [k for k in (1, 6, 11, 21, 51, 101) if k in res["profiles"]] + [last]
    aT, aN = sim13_author("Tg_feedend"), sim13_author("CN2_feedend")
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(cyc):
        cl = cmap(i / max(len(cyc) - 1, 1))
        lab = f"cycle {k}" if k != last else f"CSS (cycle {k})"
        auth = f"cycle {k}" if k != last else "cycle 296"
        for ax, a, key, idx in ((axs[2, 0], aT, "Tg", None), (axs[2, 1], aN, "C", sp.index("N2"))):
            g = a[a.series == auth]
            ax.plot(g.z_m, g.value, "o", ms=2, color=cl, alpha=0.5)
            v = res["profiles"][k]["Feed"][key]
            ax.plot(res["profiles"][k]["Feed"]["z"], v if idx is None else v[idx], color=cl, label=lab)
    axs[2, 0].set(title="Tg at the end of feed [K] (Fig. 5.23a)", xlabel="z [m]")
    axs[2, 1].set(title="C N2 at the end of feed [mol/m3] (Fig. 5.23b)", xlabel="z [m]")
    axs[2, 1].legend(frameon=False, fontsize=7)
    a = sim13_author("qH2")
    for step, cl in (("Feed", "#d23f3f"), ("Blow", "#7a7a7a"), ("Press", "#1baf7a")):
        g = a[a.series == step]
        axs[2, 2].plot(g.z_m, g.value, "o", ms=2, color=cl, alpha=0.5)
        axs[2, 2].plot(pf[step]["z"], pf[step]["q"][sp.index("H2")], color=cl, label=f"end of {step}")
    axs[2, 2].set(title="q H2 [mol/kg] (Fig. 5.21a)", xlabel="z [m]")
    h = pd.DataFrame(res["history"])
    axs[2, 3].plot(h.cycle, h.ppm, color="0.2")
    axs[2, 3].axhline(20.1, color="#d23f3f", ls="--", lw=1)
    axs[2, 3].set(yscale="log", title="impurities [ppm] vs cycle (author: 20.1 at CSS)", xlabel="cycle")
    fig.suptitle(f"simulation 13 at CSS: {res['name']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def h2_release(res):
    """H2 leaving the bed per step [mmol]: author (Fig. 5.20b, integrated) vs model."""
    a = sim13_author("H2flow").sort_values("time_s")
    tr = res["traces"][max(res["traces"])]
    mod = outlet(tr, res["species"])["H2"]
    edges = np.cumsum([0] + [d for _, d, _ in res["cycle"].steps])
    rows = []
    for i, (n, _, _) in enumerate(res["cycle"].steps):
        if n not in ("Feed", "ED1", "ED2", "ED3", "Blow"):
            continue
        tt = np.linspace(edges[i], edges[i + 1], 500)
        sel = (tr["t"] >= edges[i]) & (tr["t"] <= edges[i + 1])
        A = np.trapezoid(np.interp(tt, a.time_s, a.value), tt)
        M = np.trapezoid(mod[sel], tr["t"][sel])
        rows.append(dict(step=n, author_mmol=A, model_mmol=M, ratio=M / A))
    return pd.DataFrame(rows)


def holdup(res):
    """Gas-holdup diagnostic (decision item 2): H2 released per step [mmol] split into interparticle gas,
    macropore gas and adsorbed phase, from one extra cycle at CSS; plus the release with the pore gas excluding the
    adsorbed-phase volume (q absolute, eps_p total porosity; p.31), evaluated on the same states (a posteriori)."""
    c = res["cycle"]
    m = psa_model(c, N=res["N"], group=res["spec"].get("group", "1-4"), **res["spec"].get("opts", {}))
    run = CycleRunner(m, c)
    run.ref = (lambda t: c.P_high, c.t_press)                  # state at the end of Press (linear law, ends at P_high)
    cr = run.run_cycle(res["y_end"].copy(), profiles=True)
    iso = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")["isotherm"]["adsorbed_phase_density"]
    M = load_yaml(PARAMS / "gas_properties_nist.yaml")["species"]
    vads = np.array([M[s]["M"] / iso[s] if s in iso else 0.0 for s in m.sp])      # m3 per mol adsorbed
    iH, V, e = m.sp.index("H2"), m.A * m.dz, m.eps

    def phases(p):
        ep_free = m.eps_p - m.rho_p * (vads[:, None] * p["q"]).sum(0)
        return np.array([1e3 * (V * e * p["C"][iH]).sum(), 1e3 * (V * (1 - e) * m.eps_p * p["Cm"][iH]).sum(),
                         1e3 * (V * (1 - e) * m.rho_p * p["q"][iH]).sum(), 1e3 * (V * (1 - e) * ep_free * p["Cm"][iH]).sum(),
                         (m.eps_p[0] - ep_free.min()) / m.eps_p[0]])
    steps = [n for n, _, _ in c.steps]
    prev = phases(cr.profiles[steps[-1]])
    auth = h2_release(res).set_index("step").author_mmol if res["name"].startswith("dg_sim13") else {}
    rows = []
    for n in ("Feed", "ED1", "ED2", "ED3", "Blow"):
        h = phases(cr.profiles[n])
        d = prev - h if n != "Feed" else None
        prev = h
        if d is None:
            continue
        a = auth.get(n, np.nan)
        rows.append(dict(step=n, interparticle=d[0], macropore=d[1], adsorbed=d[2], total=d[:3].sum(),
                         author=a, ratio=d[:3].sum() / a, total_excl_ads_volume=d[0] + d[3] + d[2],
                         author_gas_scale=(a - d[2]) / (d[0] + d[1]), max_pore_fraction_filled=h[4]))
    cap = dict(eps=float(e[0]), eps_p=float(m.eps_p[0]), gas_capacity=float(e[0] + (1 - e[0]) * m.eps_p[0]),
               rho_b=float((1 - e[0]) * m.rho_p[0]))
    return pd.DataFrame(rows), cap


def step_type(name):
    return name.rstrip("123")


def pressure_by_type(res):
    """Gate 3a: RMS of the open-end pressure vs Fig. 5.18a (measured, cycle 1) per step type [bar]."""
    pe = pd.DataFrame(pressure_errors(res))
    pe["type"] = pe.step.map(step_type)
    return pe.groupby("type", sort=False).apply(
        lambda g: pd.Series(dict(n=int(g.n.sum()), rms_bar=np.sqrt((g.n * g.rms_bar ** 2).sum() / g.n.sum()))),
        include_groups=False).reset_index()


def step_windows(res):
    edges = np.cumsum([0] + [d for _, d, _ in res["cycle"].steps])
    return [(n, edges[i], edges[i + 1]) for i, (n, _, _) in enumerate(res["cycle"].steps)]


def h2_step_integrals(res, cyc, tol=GATE_MEAS["h2_rel"]):
    """Gate 3b: H2 leaving the bed per step [mmol], measured (GC samples, Fig. 5.19) vs model, over the window
    spanned by the samples of that step; both integrated by the trapezoid rule on the sample times (the model also
    exactly). Steps with fewer than 2 samples are not evaluable."""
    tr = res["traces"][cyc]
    mod = outlet(tr, res["species"])["H2"]
    e = read(f"psa_c{cyc}_flow")
    e = e[e.series == "H2"].sort_values("time_s")
    rows = []
    for n, a, b in step_windows(res):
        g = e[(e.time_s >= a) & (e.time_s <= b)]
        if len(g) < 2:
            rows.append(dict(cycle=cyc, step=n, n=len(g), window="–", measured_mmol=np.nan, model_mmol=np.nan,
                             model_exact_mmol=np.nan, rel_err=np.nan, result="not evaluable"))
            continue
        t = g.time_s.values
        ms = np.interp(t, tr["t"], mod)
        sel = (tr["t"] >= t[0]) & (tr["t"] <= t[-1]) & (tr["step"] == n)
        E, Ms = np.trapezoid(g.molar_flow_mmol_s.values, t), np.trapezoid(ms, t)
        rows.append(dict(cycle=cyc, step=n, n=len(g), window=f"{t[0]:.0f}-{t[-1]:.0f} s", measured_mmol=E, model_mmol=Ms,
                         model_exact_mmol=np.trapezoid(mod[sel], tr["t"][sel]), rel_err=Ms / E - 1,
                         result=ok(abs(Ms / E - 1) <= tol)))
    return rows


def co2_feed_front(res, cyc):
    """Gate 3b: CO2 front at z = L in the feed step. The front time is the first sample above 10 % of the CO2 feed
    flow; returns measured and model maxima if no front appears."""
    tr = res["traces"][cyc]
    c = res["cycle"]
    thr = 0.1 * 1e3 * c.feed_flow * c.y_feed["CO2"]
    e = read(f"psa_c{cyc}_flow")
    g = e[(e.series == "CO2") & (e.time_s <= c.t_feed)]
    sel = tr["step"] == "Feed"
    m = 1e3 * np.clip(tr["out_L"][res["species"].index("CO2")][sel], 0, None)
    te = g.time_s[g.molar_flow_mmol_s > thr].min() if (g.molar_flow_mmol_s > thr).any() else np.nan
    tm = tr["t"][sel][m > thr].min() if (m > thr).any() else np.nan
    if np.isnan(te) and np.isnan(tm):
        res_ = "no front in either (timing not evaluable)"
    elif np.isnan(te) or np.isnan(tm):
        res_ = "FAIL (front in one only)"
    else:
        res_ = ok(abs(tm / te - 1) <= GATE_MEAS["co2_rel"])
    return dict(cycle=cyc, threshold_mmol_s=thr, exp_max_mmol_s=g.molar_flow_mmol_s.max(), model_max_mmol_s=m.max(),
                exp_front_s=te, model_front_s=tm, result=res_)


def h2_partial_balance(res, cyc=37):
    """Gate 3c: H2 streams of cycle `cyc` [mmol]; measured where GC samples exist (sampled windows only)."""
    c = res["cycle"]
    sp, tr = res["species"], res["traces"][cyc]
    iH = sp.index("H2")
    integ = {r["step"]: r for r in h2_step_integrals(res, cyc)}
    full = {}
    for n, a, b in step_windows(res):
        sel = tr["step"] == n
        out_L, in_0 = 1e3 * tr["out_L"][iH][sel], 1e3 * tr["in_0"][iH][sel]
        full[n] = (np.trapezoid(np.clip(out_L, 0, None), tr["t"][sel]), np.trapezoid(np.clip(-out_L, 0, None), tr["t"][sel]),
                   np.trapezoid(np.clip(-in_0, 0, None), tr["t"][sel]))       # out at L, in at L, out at 0
    feed = 1e3 * c.feed_flow * c.y_feed["H2"] * c.t_feed
    rows = [dict(stream="H2 fed (z = 0, Feed)", measured="no — set point (Table 5.4 flow x composition)",
                 measured_mmol=feed, model_window_mmol=feed, model_step_mmol=feed)]
    for n, lab, k in (("Feed", "product (z = L, Feed)", 0), ("ED1", "ED gas (z = L, ED1)", 0), ("ED2", "ED gas (z = L, ED2)", 0),
                      ("ED3", "ED gas (z = L, ED3)", 0), ("Blow", "blowdown (z = 0)", 2), ("Purge", "purge outflow (z = 0)", 2),
                      ("Purge", "purge H2 in (z = L)", 1), ("EP3", "EP gas in (z = L, EP3-1)", 1), ("Press", "pressurisation H2 in (z = L)", 1)):
        r = integ.get(n)
        meas = (k != 1) and r is not None and r["n"] >= 2
        val = full["EP3"][1] + full["EP2"][1] + full["EP1"][1] if n == "EP3" else full[n][k]
        rows.append(dict(stream=lab, measured=f"yes, {r['n']} samples, {r['window']}" if meas else "no (no samples, no set point)",
                         measured_mmol=r["measured_mmol"] if meas else np.nan, model_window_mmol=r["model_mmol"] if meas else np.nan,
                         model_step_mmol=val))
    return pd.DataFrame(rows)


def temperature_peaks(res, cycles=range(1, 11)):
    """Gate 3d: thermocouple peak per cycle (cycles 1-10, Fig. 5.18b): rise above T_feed, peak time, ordering.
    Window of cycle k: from mid-blowdown of cycle k-1 to mid-blowdown of cycle k, where all three signals are flat,
    so that each peak (feed front or pressurisation heating) is an interior maximum; peak time from the window start."""
    c = res["cycle"]
    t, T = temperature_trace(res, list(range(1, max(cycles) + 1)))
    e = read("psa_c1-10_T")
    T0, Tf = c.time, c.T_feed
    t_mid = c.t_feed + 3 * c.t_eq + c.t_blow / 2
    rows = []
    for k in cycles:
        a, b = (k - 1) * T0 + t_mid - T0, (k - 1) * T0 + t_mid
        for i, s in enumerate(TC):
            g = e[(e.series == s) & (e.time_s >= a) & (e.time_s < b)]
            sel = (t >= a) & (t < b)
            je, jm = g.temperature_K.values.argmax(), T[i][sel].argmax()
            re_, rm = g.temperature_K.values[je] - Tf, T[i][sel][jm] - Tf
            te, tm = g.time_s.values[je] - a, t[sel][jm] - a
            rows.append(dict(cycle=k, tc=s, exp_rise_K=re_, model_rise_K=rm, rise_rel_err=rm / re_ - 1,
                             exp_time_s=te, model_time_s=tm, time_err_s=tm - te, time_rel_err=tm / te - 1))
    df = pd.DataFrame(rows)
    df["ok"] = (df.rise_rel_err.abs() <= GATE_MEAS["rise_rel"]) & (df.time_rel_err.abs() <= GATE_MEAS["tpeak_rel"])
    dead = np.median(np.diff(np.sort(e[e.series == "Top"].time_s.values)))      # record spacing: closer = simultaneous

    def same_order(g):
        sg = lambda x: np.sign(np.where(np.abs(x) <= dead, 0.0, x))
        te, tm = g.exp_time_s.values, g.model_time_s.values
        return all(sg(te[j] - te[i]) == sg(tm[j] - tm[i]) for i in range(3) for j in range(i + 1, 3))
    return df, df.groupby("cycle").apply(same_order, include_groups=False)


def measured_gates(res):
    """Gates 3a-3d of the step-3 decision for one case (cor_exp or a band member)."""
    pr = pressure_by_type(res)
    h2 = pd.DataFrame(h2_step_integrals(res, 1) + h2_step_integrals(res, 37))
    co2 = pd.DataFrame([co2_feed_front(res, 1), co2_feed_front(res, 37)])
    bal = h2_partial_balance(res)
    tp, order = temperature_peaks(res)
    ev = h2[h2.result != "not evaluable"]
    summary = dict(case=res["name"], p_rms_bar=np.sqrt((pr.n * pr.rms_bar ** 2).sum() / pr.n.sum()),
                   h2_steps_pass=f"{(ev.result == 'PASS').sum()}/{len(ev)}", h2_max_abs_err=ev.rel_err.abs().max(),
                   product_window_err=bal.iloc[1].model_window_mmol / bal.iloc[1].measured_mmol - 1,
                   T_peaks_ok=f"{int(tp.ok.sum())}/{len(tp)}", T_order_ok=f"{int(order.sum())}/{len(order)}",
                   rise_err_median=tp.rise_rel_err.median(), ppm_css=res["perf"]["ppm"], rec_css=res["perf"]["recovery_pct"],
                   rec_c37=perf_at(res, 37)["recovery_pct"])
    return dict(pressure=pr, h2=h2, co2=co2, balance=bal, tpeaks=tp, order=order, summary=summary)


def rung_c(results):
    """Rung c: the corrected model on all 18 of the author's simulations (17 of Table 5.10 + the experiment, p.256)."""
    sim = pd.read_csv(BENCH / "lopes2010_psa_simulations.csv")
    exp = pd.read_csv(BENCH / "lopes2010_psa_experiment.csv").iloc[0]
    rows = []
    for _, s in sim.iterrows():
        rows.append((int(s.sim), cor_case(int(s.sim)), s))
    rows.append((18, "cor_exp", exp))
    out = []
    for k, n, s in rows:
        if n not in results:
            continue
        p = results[n]["perf"]
        pa = 1e6 * (1 - s.h2_purity_pct / 100)
        out.append(dict(sim=k, case=n, cycle=s.cycle, P_high=s.P_high_bar, T=s.T_feed_K, rec_author=s.h2_recovery_pct, rec_model=p["recovery_pct"],
                        d_rec_pp=p["recovery_pct"] - s.h2_recovery_pct, ppm_author=pa, ppm_model=p["ppm"],
                        ppm_ratio=p["ppm"] / pa if pa > 0 else np.nan, co_author=s.co_ppm, co_model=p["co_ppm"],
                        cycles=results[n]["cycles"], css=results[n]["css"],
                        impurity=ok(0.5 <= p["ppm"] / pa <= 2) if pa > 0 else "– (author 0 ppm)",
                        recovery=ok(abs(p["recovery_pct"] - s.h2_recovery_pct) <= 2)))
    df = pd.DataFrame(out).sort_values("sim").reset_index(drop=True)
    g, pairs = [], pd.DataFrame()
    if len(df) == 18:
        for lab, d in (("18 simulations", df), ("17 simulations (Table 5.10)", df[df.sim <= 17])):
            rr = spearmanr(d.rec_author, d.rec_model)[0]
            rp = spearmanr(d.ppm_author, d.ppm_model)[0]
            g += [dict(check=f"Spearman, recovery ({lab})", value=f"{rr:.3f}", limit=">= 0.9", result=ok(rr >= GATE["rho_rec"])),
                  dict(check=f"Spearman, ln(impurity ppm) ({lab})", value=f"{rp:.3f}", limit=">= 0.8", result=ok(rp >= GATE["rho_ppm"]))]
        S = df.set_index("sim")
        pr = []
        for a, b, what in PAIRS:
            for q in ("rec", "ppm"):
                da, dm = S.loc[b, f"{q}_author"] - S.loc[a, f"{q}_author"], S.loc[b, f"{q}_model"] - S.loc[a, f"{q}_model"]
                pr.append(dict(pair=f"{a} -> {b}", change=what, quantity="recovery [pp]" if q == "rec" else "impurity [ppm]",
                               author=da, model=dm, diff=dm - da, same_sign=ok(np.sign(da) == np.sign(dm))))
        pairs = pd.DataFrame(pr)
        nbad = (pairs.same_sign == "FAIL").sum()
        g.append(dict(check="pairwise effects with the same sign (Table 5.10 pairs)", value=f"{len(pairs) - nbad}/{len(pairs)}",
                      limit="all", result=ok(nbad == 0)))
        g.append(dict(check="impurity within 2x of the author", value=f"{(df.impurity == 'PASS').sum()}/{(df.impurity != '– (author 0 ppm)').sum()}",
                      limit="all", result=ok((df.impurity != "FAIL").all())))
        g.append(dict(check="recovery within 2 pp of the author", value=f"{(df.recovery == 'PASS').sum()}/{len(df)}",
                      limit="all", result=ok((df.recovery == "PASS").all())))
    return df, pd.DataFrame(g), pairs


def decision(results):
    """Tables of the step-3 decision (2026-10-10): rung c, holdup diagnostic, measured-data gates, band."""
    t = {}
    t["c_sims"], t["c_gate3"], t["c_pairs"] = rung_c(results)
    cor = {n: r for n, r in results.items() if n.startswith("cor_")}
    g1 = [dict(check="CSS reached, all corrected-model cases", value=f"{sum(r['css'] for r in cor.values())}/{len(cor)}",
               limit="all", result=ok(all(r["css"] for r in cor.values()))),
          dict(check="per-cycle mole balance, worst cycle of all corrected-model cases",
               value=f"{max(abs(pd.DataFrame(r['history']).closure).max() for r in cor.values()):.1e}", limit="< 1e-3",
               result=ok(max(abs(pd.DataFrame(r['history']).closure).max() for r in cor.values()) < GATE["closure"]))]
    if "cor_exp" in results and "cor_exp_grid_2N" in results:
        p1, p2 = results["cor_exp"]["perf"], results["cor_exp_grid_2N"]["perf"]
        drec, dppm = abs(p2["recovery_pct"] - p1["recovery_pct"]), abs(p2["ppm"] / p1["ppm"] - 1)
        g1 += [dict(check=f"grid N = {N} -> {2 * N} (experiment cycle): recovery", value=f"{drec:.3f} pp", limit="< 0.3 pp",
                    result=ok(drec < GATE["grid_rec_pp"])),
               dict(check=f"grid N = {N} -> {2 * N} (experiment cycle): impurity ppm", value=f"{100 * dppm:.1f} %", limit="< 10 %",
                    result=ok(dppm < GATE["grid_ppm_rel"]))]
    t["c_gate1"] = pd.DataFrame(g1)
    if "dg_sim13_betaA_ep_h2" in results:
        t["holdup"], t["holdup_capacity"] = holdup(results["dg_sim13_betaA_ep_h2"])
    if "cor_exp" in results:
        mg = measured_gates(results["cor_exp"])
        t.update({f"m_{k}": v for k, v in mg.items() if k not in ("summary", "order")})
        t["m_order"] = mg["order"]
        band = [mg["summary"]] + [measured_gates(results[n])["summary"] for n in sorted(cor)
                                  if n.startswith("cor_exp_") and n != "cor_exp_grid_2N"]
        t["m_band"] = pd.DataFrame(band)
    return t


def step_balance(res):
    """Per-step, per-component moles [mmol] over the last (CSS) cycle: + enters the bed, - leaves it."""
    sp = res["species"]
    rows = []
    for st, mo in res["step_moles"].items():
        net = mo[0] - mo[1]                       # in at z = 0 minus out at z = L
        rows.append(dict(step=st, **{s: 1e3 * net[i] for i, s in enumerate(sp) if s != "He"}))
    return pd.DataFrame(rows)


def fig_profile_evolution(res, path, species=("N2", "CO")):
    """Gas-phase impurity profiles at the end of every step for the cycles kept (diagnostic)."""
    sp, prof = res["species"], res["profiles"]
    cycles = sorted(prof)
    steps = [n for n, _, _ in res["cycle"].steps]
    fig, axs = plt.subplots(len(species), len(steps), figsize=(2.2 * len(steps), 3.0 * len(species)), sharex=True,
                            sharey="row", squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, spc in enumerate(species):
        for j, st in enumerate(steps):
            ax = axs[i, j]
            for k, c in enumerate(cycles):
                p = prof[c][st]
                ax.plot(p["z"], p["C"][sp.index(spc)], color=cmap(k / max(len(cycles) - 1, 1)),
                        label=f"cycle {c}" if c != cycles[-1] else f"CSS ({c})")
            ax.set_yscale("symlog", linthresh=1e-3)
            if i == 0:
                ax.set_title(f"end of {st}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"C {spc} [mol/m3]")
            if i == len(species) - 1:
                ax.set_xlabel("z [m]")
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{res['name']}: gas-phase impurity profiles (z = L is the product end)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
FINDINGS = """## First experiment comparison: findings at the time (baseline fixed beforehand)

**First experiment comparison = FAIL (1825 ppm impurities, 675 ppm CO, 81.86 % recovery) with the baseline fixed
before any comparison.** Any later experiment result is "after corrections found by reproducing the author's model"
and is reported as such. (Since then: the experiment's purity, CO and recovery turn out to be the author's
simulated values, p.256; see the follow-up section.)

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

    # ---- verification ladder (follow-up): rung a, rung b, diagnostics
    sim = pd.read_csv(BENCH / "lopes2010_psa_simulations.csv").set_index("sim")
    lad = []
    label = {"": "baseline (ED gas mixed, beta_P 0.20)", "ep_h2": "(i) EP = pure H2", "ep_product": "(i) EP = product gas",
             "ep_ed_direct": "(i) EP = ED gas, first out first in", "ep_ed_reversed": "(i) EP = ED gas, last out first in",
             "beta_steps": "(ii) beta_P per step type (Fig. 5.18a)", "beta_author": "beta_P of the author (0.102, Fig. 5.20a)",
             "betaA_ep_h2": "author beta_P + EP = pure H2", "betaA_ep_product": "author beta_P + EP = product gas"}
    for k in (1, 2, 13):
        for v, lab in label.items():
            n = (f"sim{k:02d}" if not v else f"ver_sim{k:02d}_{v}") if k != 13 else \
                {"": "dg_sim13", "ep_h2": "dg_sim13_ep_h2", "beta_author": "dg_sim13_beta_author",
                 "betaA_ep_h2": "dg_sim13_betaA_ep_h2", "betaA_ep_product": "dg_sim13_betaA_ep_product"}.get(v)
            if n not in results:
                continue
            p, a = results[n]["perf"], sim.loc[k]
            pa = 1e6 * (1 - a.h2_purity_pct / 100)
            lad.append(dict(sim=k, case=lab, ppm=p["ppm"], ppm_author=pa, ppm_ratio=p["ppm"] / pa, co_ppm=p["co_ppm"],
                            co_author=a.co_ppm, recovery=p["recovery_pct"], d_recovery_pp=p["recovery_pct"] - a.h2_recovery_pct,
                            prod=p["productivity"], prod_author=a.productivity_mol_kg_day,
                            target=ok(1 / 2 <= p["ppm"] / pa <= 2 and abs(p["recovery_pct"] - a.h2_recovery_pct) <= 2)))
    t["ladder"] = pd.DataFrame(lad)
    s13 = {n: r for n, r in results.items() if n.startswith("dg_sim13")}
    t["sim13_profiles"] = sim13_compare(s13) if s13 else pd.DataFrame()
    t["balance_sim01"] = step_balance(results["dg_sim01"]) if "dg_sim01" in results else pd.DataFrame()
    if "dg_sim13_betaA_ep_h2" in results:
        t["balance_sim13"] = step_balance(results["dg_sim13_betaA_ep_h2"])
        t["h2_release_sim13"] = h2_release(results["dg_sim13_betaA_ep_h2"])

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
        order = sorted(todo, key=lambda n: ("grid_2N" not in n, n))          # longest first
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(run_case, n, todo[n]): n for n in order}
            for f in as_completed(futs):
                if f.exception() is not None:
                    print(f"[{futs[f]}] FAILED: {f.exception()!r}", flush=True)
    results, t = analyse()
    t.update(decision(results))
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(OUT / f"{k}.csv", index=False)
    if "cor_exp" in results:
        cor = results["cor_exp"]
        fig_pressure(cor, OUT / "pressure_cor.png")
        fig_flows(cor, OUT / "flows_cor.png")
        fig_temperature(cor, OUT / "temperature_cor.png")
    if len(t["c_sims"]):
        fig_sims(t["c_sims"], OUT / "sims_cor.png")
    base = results["baseline"]
    fig_pressure(base, OUT / "pressure.png")
    fig_flows(base, OUT / "flows.png")
    fig_temperature(base, OUT / "temperature.png")
    fig_css(results, OUT / "css.png")
    for n in ("dg_sim13", "dg_sim13_betaA_ep_h2"):
        if n in results:
            fig_sim13(results[n], OUT / f"{n}.png")
    if "dg_sim01" in results:
        fig_profile_evolution(results["dg_sim01"], OUT / "profiles_sim01.png")
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


def followup(t):
    """Step-3 follow-up: reproduce the author's model (verification) before touching the experiment again."""
    lad = t["ladder"]
    f_l = {"ppm": "{:.1f}", "ppm_author": "{:.1f}", "ppm_ratio": "{:.2f}", "co_ppm": "{:.2f}", "co_author": "{:.3g}",
           "recovery": "{:.2f}", "d_recovery_pp": "{:+.2f}", "prod": "{:.1f}", "prod_author": "{:.1f}"}
    sp = t["sim13_profiles"]
    sp = sp[sp.case.isin(["dg_sim13", "dg_sim13_beta_author", "dg_sim13_betaA_ep_h2"])] if len(sp) else sp
    return f"""## Follow-up: reproducing the author's model (verification, project lead's decision (b))

**Principle.** The author's 17 simulations (Table 5.10, Figs 5.20-5.23) are a verification target for the
implementation; inferring from them what the author did is legitimate. The PSA experiment is not used to choose
anything. At the time, rung c and the experiment rerun were **not** done: rung a did not meet its targets (below);
both were run after the step-3 decision (section above).

### 1. What the thesis says (quoted)

* p.166: "A PSA cycle of 10 elementary steps was employed (see Figure 5.7): a co-current feed, three pressure
  equalizations-depressurizations, a counter-current blowdown, a counter-current purge with hydrogen, three pressure
  equalizations-pressurizations and a counter-current pressurization with hydrogen."
* p.191: "In this single-column experiment, the pressure equalization was divided into three different steps to
  simulate the behaviour of an industrial PSA unit where several pressure equalization steps are required to
  increase H2 recovery."
* pp.178-180, eqs 5.44-5.45 (EP) and 5.46-5.47 (Press): the inlet condition is
  u0,inlet C_inlet,i = u0 C_g,i - eps Dax C_g,T dy_i/dz with u0,inlet C_inlet,T = u0 C_g,T; **C_inlet is not defined
  for EP**, and no pressure law is given for EP. Press: P = P_inter + (P_high - P_inter) t/t_press (eq. 5.47).
* p.180: "The simulations of the depressurization steps were performed using an exponential decay function until a
  pre-defined pressure (Pinter) is achieved (see equation (5.39)). When three successive pressure equalizations take
  place, three different pressure intervals are defined using exponential functions with the same exponential decay
  rate constant (betaP)."
* Fig. 5.5 (p.164): two columns, one tank, mass-flow controllers on the feed lines, a mass-flow meter and a
  back-pressure regulator on the product line, a vacuum pump. The text never says what the tank or column 2 did in
  the single-column experiment.
* pp.221-223 (chapter 6, same unit): equalisation between columns 1 and 2 reached "2.68, 2.28 and 1.85 bar instead
  of 3, 2.5 and 2 bar" because of dead volumes; "Since the dead volumes may result in significant changes to the
  experimental results, in this chapter, only experiments involving the fast cycling of a single-column PSA are
  reported."
* **p.256: "it was concluded that a hydrogen recovery of 81.6% was obtained in the PSA experiment with a H2 purity of
  99.98054% contaminated with 63 ppm of CO (simulated values) and a process productivity of 101.1 molH2 kgads-1
  day-1."** Chapter 6 states the practice (p.241): "Due to the small number of experimental points, report
  experimental values of these performance parameters may not be very accurate. Since the simulation predicts with
  good accuracy the experimental results obtained, the performance parameters based on simulated results were
  reported." **The experiment's purity, CO and recovery used in gate 2 are therefore the author's simulation of the
  experiment, not measurements.** The measured data of the experiment are the outlet flows at the GC sampling times
  (Fig. 5.19), the pressure (Fig. 5.18a) and the temperatures (Figs 5.18b, 5.19). Decision needed (below).

### 2. Rung a (cycle A, simulations 1 and 2) and simulation 13 — one hypothesis at a time

Target: impurity within a factor of 2 and recovery within 2 pp of Table 5.10.

{md(lad, f_l)}

* **(i) EP inflow.** Pure H2 lowers impurity 2.5-5x; product gas slightly less; ED gas (mixed, first-out-first-in or
  last-out-first-in) gives the highest impurity. Recovery does not depend on the EP gas (eq. 5.49 does not count it).
* **(ii) beta_P per step type**, fitted only to Fig. 5.18a: ED 7.33, Blow 0.078, EP 0.087 s-1 (objective 0.048 vs
  1.20 bar2 for one constant; RMS per step 0.03-0.16 bar, `fit_bcs_beta.csv`). It describes the experiment's valves
  much better but moves the model away from the author (impurity 15-110x).
* **The author's own beta_P.** Fitted to the author's simulated sim-13 pressure (Fig. 5.20a, digitised): **0.102 s-1**, each
  exponential step alone 0.09-0.11 s-1 (EP included). With it our sim-13 pressure, loading profiles and N2 front
  coincide with the author's (table below).
* **(iii) Purge flow.** Not the cause. If the extra purge time costs only its H2 (eq. 5.49), the author's recovery
  changes imply 0.24 (sims 1->2), 0.26 (3->4), 0.23 (9->10) and 0.23 mmol/s (16->17), the same as our 0.261 mmol/s
  fitted to Fig. 5.19.
* Not adopted, diagnostic only: pellet porosity 0.35 instead of 0.627 (Table 4.5) with the author's beta_P on sim 13:
  recovery +5.8 pp but impurity 30x worse — lower gas holdup alone does not explain both gaps.

### 3. Simulation 13 against Figs 5.20-5.23 (rung b, as a diagnostic)

{md(sp, {"author": "{:.4g}", "model": "{:.4g}"}) if len(sp) else ""}

![sim 13, author beta_P + EP = H2](dg_sim13_betaA_ep_h2.png)
![sim 13, baseline](dg_sim13.png)

H2 leaving the bed per step, sim 13 (author's beta_P, EP = H2), author's line in Fig. 5.20b integrated:

{md(t["h2_release_sim13"], {"author_mmol": "{:.1f}", "model_mmol": "{:.1f}", "ratio": "{:.2f}"}) if "h2_release_sim13" in t else ""}

At the end of every step our pressure (open end and interior) and H2 loadings coincide with the author's within
1 % (Fig. 5.21a), yet our column releases 26-43 % more H2 in each ED step and 25 % more in the blowdown, and needs
about 20 % more H2 to pressurise (from eq. 5.49 with the author's recovery and productivity). Equations 5.3 and 5.6 include
the macropore gas exactly as we do, so the thesis does not explain the difference. The extra blowdown loss
(about 12 mmol per cycle) is the 4 pp recovery deficit.

### 4. Localising the excess impurity (diagnostics, sim 1 baseline)

Per-step, per-component moles over the CSS cycle [mmol; + enters, - leaves the bed]:

{md(t["balance_sim01"], {"H2": "{:.3f}", "CO2": "{:.4f}", "CH4": "{:.4f}", "CO": "{:.4f}", "N2": "{:.4f}"}) if len(t["balance_sim01"]) else ""}

Blowdown removes 77 % and the purge 21 % of the N2 fed; 3.4 % leaves with the product (0.049 mmol/cycle). The EP
step returns 0.050 mmol N2 per cycle into the product end with the ED gas — as much as leaves in the product. The
profiles show the mechanism: the purge cleans only the last few cm of the product end, EP re-contaminates them, and
after ~10 cycles the impurity tail spans the bed:

![impurity profiles, sim 1](profiles_sim01.png)

Purity calculation checked against eq. 5.48: feed step only, z = L, He excluded (He was in the denominator before;
its effect at CSS is below 0.3 ppm). Tests cover eqs 5.48-5.50 against the step integrals.

### 5. Changes adopted, and why (A = supported by the thesis, B = needed to reproduce the author's simulations
without using the experiment)

| change | basis | used for |
|---|---|---|
| purity excludes He (eq. 5.48 sums the feed components) | A | all runs |
| beta_P = 0.102 s-1, one constant for ED, Blow and EP | B (Fig. 5.20a) + A (p.180: same constant) | the author's simulations |
| EP inflow = pure H2 | B (sims 1, 2, 13; Figs 5.21, 5.23b) — C_inlet is not given (eq. 5.45) | the author's simulations |

Not adopted: ED gas variants, product gas, per-step beta_P (worse against the author), pellet porosity (contradicts
Table 4.5). The experiment case keeps the pre-declared baseline; nothing has been re-run against it.

### 6. Remaining gap and options

With both corrections: impurity 1.2x (sim 1), 2.4x (sim 2), 1.2x (sim 13) the author's; CO 2.7-8.7x;
**recovery 4.0-4.6 pp below the author in all three** (target 2 pp). Rung a fails on recovery (and narrowly on
sim-2 impurity), so I stopped here.

Most likely causes of the recovery gap, ranked:
1. **Gas released or taken up per step at a given pressure change is 15-40 % larger in our model** (ED, blowdown,
   pressurisation) although pressures and loadings coincide. Something in the author's implementation holds less
   gas than eqs 5.3/5.6 with eps = 0.404 and eps_p = 0.627 imply (a different voidage or porosity in the code, or
   macropore gas not exchanging with the bulk on the step time scale). Not resolvable from the thesis text.
2. Feed flow basis: the author's productivity/recovery ratios imply 0.6-1.3 % more feed H2 per cycle (1 atm instead of
   1 bar?); worth < 0.5 pp.
3. Purge flow: excluded (above).

Options for the project lead:
* **(a) Accept a systematic recovery offset of about -4 pp against the author** (impurity reproduced within ~2x
  with the two B corrections), run rung c (17 simulations: ranks, signs, absolute differences) and then, if it
  passes, the single post-correction experiment comparison.
* **(b) Pursue cause 1** with targeted implementation variants (e.g. macropore gas lumped differently, voidage
  definitions) judged only on the sim-13 step releases (Fig. 5.20b) and Table 5.10 — risks becoming reverse
  engineering of an undocumented code.
* **(c) Contact the author / LSRE** for the gPROMS input (EP C_inlet, porosities, beta_P) — the cleanest fix.
* **Decide how to treat the experiment's purity/CO/recovery** now that they are known to be simulated values
  (p.256): as an 18th author simulation (verification), with validation restricted to the measured outlet flows,
  pressure and temperatures; or keep gate 2 as defined.
"""


def decision_text(t):
    """Report section of the step-3 decision (2026-10-10)."""
    cs, g3, pairs = t["c_sims"], t["c_gate3"], t["c_pairs"]
    d = cs.d_rec_pp
    rp = pairs[pairs.quantity == "recovery [pp]"] if len(pairs) else pairs
    h2, co2, bal, tp, order = t["m_h2"], t["m_co2"], t["m_balance"], t["m_tpeaks"], t["m_order"]
    ev = h2[h2.result != "not evaluable"]
    hold, cap = t.get("holdup", pd.DataFrame()), t.get("holdup_capacity", {})
    f_c = {"P_high": "{:.0f}", "T": "{:.0f}", "rec_author": "{:.2f}", "rec_model": "{:.2f}", "d_rec_pp": "{:+.2f}", "ppm_author": "{:.2f}", "ppm_model": "{:.2f}",
           "ppm_ratio": "{:.2f}", "co_author": "{:.3g}", "co_model": "{:.3g}"}
    f_h = {c: "{:.2f}" for c in ("interparticle", "macropore", "adsorbed", "total", "author", "ratio", "total_excl_ads_volume",
                                 "author_gas_scale", "max_pore_fraction_filled")}
    imp_ok = (cs.impurity == "PASS").sum()
    imp_n = (cs.impurity != "– (author 0 ppm)").sum()
    bad = cs[cs.impurity == "FAIL"]
    big = rp[rp.author.abs() > 1.0]
    amp = big.model / big.author
    byc = cs.groupby(["cycle", "P_high"]).d_rec_pp.agg(["min", "max", "size"])
    off_rows = "; ".join(f"cycle {c} at {pp:.0f} bar ({r['size']:.0f} sims): {r['min']:+.1f} to {r['max']:+.1f} pp"
                         for (c, pp), r in byc.iterrows())
    nfail = pairs[pairs.same_sign == "FAIL"]
    return f"""## Step-3 decision (2026-10-10): reframing, rung c, gas holdup, measured cyclic data

### Reframing: what is measured and what is simulated

* p.256: "it was concluded that a hydrogen recovery of 81.6% was obtained in the PSA experiment with a H2 purity
  of 99.98054% contaminated with 63 ppm of CO (simulated values) and a process productivity of 101.1 molH2
  kgads-1 day-1." p.241 (chapter 6) states the practice: "Due to the small number of experimental points, report
  experimental values of these performance parameters may not be very accurate. Since the simulation predicts with
  good accuracy the experimental results obtained, the performance parameters based on simulated results were
  reported."
* **The PSA experiment's purity, CO, recovery and productivity are the author's simulated values.** Earlier versions
  of this report (and the step-3 request) called them "experimental" or "measured" **by mistake**: gate 2 of the
  first comparison compared our model with the author's model, not with measurements. They are now treated as the
  author's **18th simulation** (verification, together with Table 5.10).
* **Measured cyclic data** (validation from now on): outlet molar flows per species at the GC sample times,
  cycles 1 and 37 (Fig. 5.19); pressure, cycle 1 (Fig. 5.18a); thermocouples, cycles 1-10 (Fig. 5.18b) and
  cycles 1 and 37 (Fig. 5.19). Feed flow, composition, step times and pressures are set points (Table 5.4).
  Purge flow, pressurisation flow and EP gas were not reported.

**Corrected model** (all results in this section): pre-declared baseline (E7 kinetics, `taqvi_levan`, assumed
thermal values, transport group per Table 5.10) + beta_P = 0.102 s-1 (author's simulated Fig. 5.20a) + EP inflow =
pure H2. Both corrections come from the author's simulations (basis B, `data/PROVENANCE.md`); no measured data and
no performance value were used to set anything.

### Verdict

| check | result |
|---|---|
| Gate 1, numerical (corrected model) | {verdict(t["c_gate1"])} |
| Verification, 18 author simulations: impurity within 2x | {imp_ok}/{imp_n} (sim 5: author 0 ppm, not rated) — {"PASS" if imp_ok == imp_n else "FAIL"} |
| Verification: recovery within 2 pp | {(cs.recovery == "PASS").sum()}/{len(cs)} — **FAIL**, offset {d.min():+.1f} to {d.max():+.1f} pp (mean {d.mean():+.2f}, sd {d.std():.2f}) — accepted as a documented discrepancy |
| Verification: Spearman, pairwise signs | {"; ".join(f"{r.check.replace('Spearman, ', '')}: {r.value} {r.result}" for _, r in g3.iterrows() if r.check.startswith(("Spearman", "pairwise")))} |
| 3a. Pressure, cycle 1 (Fig. 5.18a) | RMS {t["m_band"].iloc[0].p_rms_bar:.2f} bar overall, per step type below (reported, no limit declared) |
| 3b. H2 outflow per step, cycles 1 and 37, +-10 % | {(ev.result == "PASS").sum()}/{len(ev)} evaluable steps pass — **{"PASS" if (ev.result == "PASS").all() else "FAIL"}**; ED/EP/Press: no samples |
| 3b. CO2 front in the feed step, +-10 % | not evaluable: no CO2 front reaches z = L during feed, in the experiment or the model |
| 3c. H2 recovery at cycle 37 from measured streams, +-3 pp | **not evaluable**: the measured streams do not close the H2 balance (partial balance below) |
| 3d. Thermocouples, cycles 1-10 (rise 30 %, peak time 20 %, ordering) | {int(tp.ok.sum())}/{len(tp)} peaks, ordering {int(order.sum())}/{len(order)} cycles — **{"PASS" if tp.ok.all() and order.all() else "FAIL"}** |

**Can step 3 be closed?** Not on the declared gates. Numerics pass, and the verification reproduces the author's
ranking (Spearman {g3.value.iloc[0]} recovery, {g3.value.iloc[1]} impurity) and {imp_ok} of {imp_n} impurities within 2x; the
recovery offset is accepted as a documented discrepancy. But two of the measured-data gates fail (3b H2 per step
{(ev.result == "PASS").sum()}/{len(ev)}, 3d thermocouples {int(tp.ok.sum())}/{len(tp)}) and 3c cannot be evaluated from the measured streams. None of the
failures traces to a coding bug (causes in sections 1 and 3). Starting step 4 needs the project lead to accept them
as documented limitations, or to decide otherwise.

### Gate 1 — numerical, corrected model

{md(t["c_gate1"])}

### 1. Code-to-code verification: rung c, all 18 author simulations

Simulation 18 is the experiment cycle (author's simulated values, p.256). Sims 1, 2 and 13 are the identical runs of
the follow-up ladder (`ver_sim01/02_betaA_ep_h2`, `dg_sim13_betaA_ep_h2`).

{md(cs.drop(columns=["case"]), f_c)}

{md(g3)}

![rung c](sims_cor.png)

Pairwise effects (Table 5.10 pairs; `diff` = model effect - author effect):

{md(pairs, {"author": "{:+.3g}", "model": "{:+.3g}", "diff": "{:+.3g}"}) if len(pairs) else ""}

**Is the recovery offset constant?** No: {off_rows} (all 18: mean {d.mean():+.2f}, sd {d.std():.2f} pp). It grows with
P_high and is smallest in cycle B. Recovery trends are preserved in direction and order: Spearman {g3.value.iloc[0]}, and
all {len(rp)} recovery effects have the author's sign. Their size is overstated by a factor of {amp.median():.2f} (median;
range {amp.min():.2f}-{amp.max():.2f} over the {len(big)} effects above 1 pp). An optimiser on this model would move in the same
directions as on the author's model but see slightly steeper recovery trade-offs. A constant offset would keep the
trade-offs exactly; this one does not.

Impurity: {imp_ok}/{imp_n} within 2x. The {len(bad)} failures are sims {", ".join(str(k) for k in bad.sim)} (author
{", ".join(f"{v:.1f}" for v in bad.ppm_author)} ppm; ours {", ".join(f"{v:.1f}" for v in bad.ppm_model)} ppm), all below 6 ppm on the author's side; Table 5.10
prints purity to 1e-5 %, i.e. 0.1 ppm. The experiment cycle (sim 18) gives {cs.set_index("sim").loc[18, "ppm_model"]:.0f} ppm and
{cs.set_index("sim").loc[18, "co_model"]:.1f} ppm CO against the author's simulated 195 and 63 ppm. Sign failures: {"; ".join(f"{r.pair} ({r.change}, {r.quantity}: author {r.author:+.1f}, ours {r.model:+.1f})" for _, r in nfail.iterrows())}.
Both are blowdown-time reductions: the author's impurity rises by 5-6 ppm (25-48 %), ours barely moves (< 5 %).

### 2. Gas-holdup diagnostic (time-boxed)

Gas capacity per unit bed volume used: eps + (1 - eps) eps_p = {cap.get("eps", np.nan):.3f} + (1 - {cap.get("eps", np.nan):.3f}) x
{cap.get("eps_p", np.nan):.3f} = **{cap.get("gas_capacity", np.nan):.3f}** (Table 5.4 bed porosity, Table 4.5 pellet porosity); nothing
else stores gas (no dead volume, no micropore gas term; the adsorbed phase is the absolute loading q, eq. 2.2).
Bulk density (1 - eps) rho_p = {cap.get("rho_b", np.nan):.1f} kg/m3.

H2 released per step, simulation 13 at CSS (corrected model) [mmol], split by phase from the step-end states;
`author` = the author's Fig. 5.20b line integrated; `author_gas_scale` = factor on our gas holdup that would reproduce the
author's release with our adsorbed-phase change; `total_excl_ads_volume` = the same states with the pore gas
excluding the adsorbed-phase volume; `max_pore_fraction_filled` = largest share of eps_p occupied by the adsorbed
phase at the end of the step:

{md(hold, f_h) if len(hold) else ""}

What the thesis supports, quoted:
* eq. 5.3 (p.169): accumulation eps dC/dt in the bulk and the film flux (1 - eps) a_p k_f (C - C_s); eq. 5.6 (p.170):
  d<C_m>/dt = (Omega_m D_p / R_p^2)(C_s - <C_m>) - (rho_p/eps_p) d<q>/dt, "eps_p ... the particle ... porosity".
* Table 4.5 (p.122): eps_p = 0.627, rho_p = 786, rho_s = 2110 kg/m3 (eps_p = 1 - rho_p/rho_s: total porosity; rho_s
  "determined by helium pycnometry", p.36).
* p.42: "The value obtained from the gravimetric unit is called excess amount adsorbed, q_exc. In order to transform
  this value to the real amount adsorbed, buoyancy corrections with equation (2.2) were performed in all the
  experiments." p.31: "we have assumed that the density of the adsorbed phase is equal to the density of the liquid
  at its boiling point at 1 atm" (CO2 1277.78, CH4 422.53, N2 808.60, H2 78, CO 788.6 kg/m3). Nomenclature (p.203):
  q_i "absolute adsorbed phase concentration".

The only holdup variant with textual support is therefore: absolute loadings + total porosity count the gas in the
volume the adsorbed phase occupies twice, so exclude that volume from the pore gas. It changes the H2 released per
step by **less than 1.5 %** (column `total_excl_ads_volume`) — far from the 27-50 % gap. The factor needed on our
gas holdup is not one number either ({hold.author_gas_scale.min():.2f}-{hold.author_gas_scale.max():.2f} across the steps), so no single
porosity or voidage reproduces the author's step releases, and the eps_p = 0.35 diagnostic that recovers the author's
recovery makes impurity 30x worse (follow-up section 2). **Not adopted; our holdup (eqs 5.3, 5.6 as printed) is
kept.** The holdup treatment is a declared assumption (`data/PROVENANCE.md`); the adsorbed-volume variant is below
1.5 % and is not carried as a separate band member.

### 3. Validation against the measured cyclic data (gates declared before running)

**a. Pressure, cycle 1** (Fig. 5.18a measured; beta_P from the author's simulated curve, so this is a check):

{md(t["m_pressure"], {"rms_bar": "{:.3f}"})}

![pressure, corrected model](pressure_cor.png)

**b. H2 leaving the bed per step** (GC samples; window = span of the samples in the step; trapezoid rule on the
sample times for both, `model_exact_mmol` = model integrated on its own time grid):

{md(h2, {"measured_mmol": "{:.2f}", "model_mmol": "{:.2f}", "model_exact_mmol": "{:.2f}", "rel_err": "{:+.1%}"})}

CO2 at z = L in the feed step (front = first sample above 10 % of the CO2 feed flow):

{md(co2, {"threshold_mmol_s": "{:.3f}", "exp_max_mmol_s": "{:.4f}", "model_max_mmol_s": "{:.2e}", "exp_front_s": "{:.0f}", "model_front_s": "{:.0f}"})}

The measured CO2 values in the feed step (0.004 mmol/s) are the plotted zero line; no front, so its timing cannot be
compared.

Failures of 3b, read from the samples (not tuned): cycle-1 feed — the H2 front that displaces the initial He reaches
z = L before 69 s in the experiment and between 69 and 81 s in the model; the single 69-s sample (1.34 vs
0.10 mmol/s) makes the -41 %. Cycle-37 blowdown — the model's H2 flow decays faster than measured after 169 s,
consistent with beta_P = 0.102 s-1 (author's simulations) against about 0.08 s-1 fitted to the measured blowdown
pressure (follow-up section 2). Cycle-37 purge — the model's H2 outflow is 28 % above the measured one with the
purge flow fitted to the cycle-1 purge (where it passes, +6 %).

![outlet flows, corrected model](flows_cor.png)

**c. H2 balance at cycle 37.** Measured: the product-end outflow during feed (4 samples, 40-97 s; none in 0-40 s),
the feed-end outflow during blowdown (7 samples from 169 s; the first 20 s of the step are not sampled) and during
the purge (3 samples). Not measured and without a reported set point: ED gas, EP inflow, purge H2 inflow,
pressurisation H2. The feed H2 is the set point (Table 5.4). Eq. 5.49 needs the purge and pressurisation H2, so an
experimental recovery cannot be formed; nothing was filled in by fitting. Partial balance [mmol per cycle]:

{md(bal, {"measured_mmol": "{:.2f}", "model_window_mmol": "{:.2f}", "model_step_mmol": "{:.2f}"})}

Model recovery at cycle 37 (eq. 5.49): {t["m_band"].iloc[0].rec_c37:.2f} % (author's simulated value 81.6 %).

**d. Thermocouples, cycles 1-10** (Fig. 5.18b). Window of each cycle: mid-blowdown of the previous cycle to
mid-blowdown of this one (signals flat there, so each peak — feed front or pressurisation heating — is an interior
maximum); rise above T_feed = 303 K; peak time from the window start; ordering = sign of the peak-time difference
of every thermocouple pair, peaks closer than the record spacing counting as simultaneous. A first implementation
with windows starting at the feed step put the Middle and Top peaks on the window boundary (pressurisation heating
at the end of the cycle) and was replaced before any corrected-model result was evaluated.

{md(tp, {"exp_rise_K": "{:.2f}", "model_rise_K": "{:.2f}", "rise_rel_err": "{:+.0%}", "exp_time_s": "{:.0f}", "model_time_s": "{:.0f}",
         "time_err_s": "{:+.0f}", "time_rel_err": "{:+.0%}"})}

![temperatures, corrected model](temperature_cor.png)

### 4. Uncertainty band of the cyclic validation (reported, not selected from)

{md(t["m_band"], {"p_rms_bar": "{:.3f}", "h2_max_abs_err": "{:.0%}", "product_window_err": "{:+.1%}", "rise_err_median": "{:+.0%}",
                  "ppm_css": "{:.1f}", "rec_css": "{:.2f}", "rec_c37": "{:.2f}"})}

Members: corrected model (`cor_exp`); EP inflow = ED gas; `noncompetitive_h2`; Table 4.10 kinetics for all species;
thermal parameters at the ends of their ranges. `product_window_err` = model / measured H2 product over the sampled
window of cycle 37 - 1.

### Limitations (draft)

* **EP inflow composition (declared assumption).** Eq. 5.45 leaves C_inlet open and the text does not say how a
  single column was equalised. Pure H2 reproduces the author's simulations; ED gas (tank emulation) is the physical
  alternative. Both are in the band: they change the experiment-cycle impurity {t["m_band"].set_index("case").loc["cor_exp", "ppm_css"]:.0f} vs
  {t["m_band"].set_index("case").loc["cor_exp_ep_ed", "ppm_css"]:.0f} ppm at the same recovery; the measured flows cannot distinguish them.
* **Gas holdup / recovery offset (declared assumption).** Our model follows eqs 5.3 and 5.6 with the Table 4.5
  porosity; at the same pressures and loadings it releases 27-50 % more H2 per depressurisation step than the
  author's model, which the thesis does not explain. Consequence: recovery {d.min():+.1f} to {d.max():+.1f} pp against all 18 of the
  author's simulations.
* **Pressure law.** One beta_P (from the author's simulations) cannot follow the measured ED valves (plateau in about
  1 s; RMS 0.7 bar in ED). The per-step-type fit to Fig. 5.18a is available but not used.
* **Measured data are sparse.** No GC samples in ED, EP or Press; blowdown sampled from 20 s after its start; no
  purge or pressurisation flow reported. Purity and recovery of the experiment were never measured.
* **Thermal.** Top thermocouple rise 52-68 % low in every cycle (Middle -51 to +57 %), with assumed wall and solid
  heat capacities (step 2); the Bottom thermocouple passes in all 10 cycles.
"""


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
    text = f"""# Step 3 — single-column PSA cycle to CSS: verification against the author, validation on measured cyclic data

Generated by `scripts/run_psa.py` (cases cached in `results/step3/cases/`). Thesis pages are printed pages.

{decision_text(t) if "m_band" in t and len(t["c_sims"]) == 18 else ""}

## Earlier record: first comparison (pre-declared baseline) and follow-up

The gates in this part are those of the **first** comparison. Gate 2 there compared with the author's simulated
performance of the experiment, called "experimental" at the time by mistake (see the reframing above).

| gate (first comparison) | result |
|---|---|
| 1. Numerical (CSS, mole balance, grid) | {verdict(t["gate1"])} |
| 2. "Experiment" = author's simulated performance (recovery, impurity ppm, CO ppm at CSS) | {"PASS" if gate2_pass else "FAIL"} |
| 3. Author's 17 simulations (rank correlation, effect signs) | {verdict(t["gate3"])} |

{followup(t) if len(t.get("ladder", ())) else ""}

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

## Gate 2 — the author's simulated performance of the experiment (first comparison; called "experiment" then)

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
