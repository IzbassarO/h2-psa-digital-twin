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
# verification ladder (step-3 follow-up): one hypothesis at a time on the author's simulations
VARIANTS = {"ep_h2": dict(ep_gas="h2"), "ep_product": dict(ep_gas="product"), "ep_ed_direct": dict(ep_gas="ed_direct"),
            "ep_ed_reversed": dict(ep_gas="ed_reversed"), "beta_steps": "per_step", "beta_author": "author",
            "betaA_ep_h2": "author+h2", "betaA_ep_product": "author+product"}
LADDER_A = (1, 2)
PROFILE_CYCLES = {1: (1, 5, 10, 20), 13: (1, 5, 6, 10, 11, 16, 20, 21, 51, 101, 161, 221, 296)}   # Fig. 5.23 cycles
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
    return out


def run_case(name, spec):
    base = experiment_cycle(**spec.get("cycle", {}))
    c = simulation_cycle(spec["sim"], base) if "sim" in spec else base
    m = psa_model(c, N=spec.get("N", N), group=spec.get("group", "1-4"), **spec.get("opts", {}))
    run = CycleRunner(m, c)
    log = lambda s: print(f"[{name}] {s}", flush=True)
    res = run.run_to_css(record=RECORD if name in ("baseline", "grid_2N") else (37,), log=log,
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
        order = sorted(todo, key=lambda n: (n != "grid_2N", n))          # longest first
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(run_case, n, todo[n]): n for n in order}
            for f in as_completed(futs):
                if f.exception() is not None:
                    print(f"[{futs[f]}] FAILED: {f.exception()!r}", flush=True)
    results, t = analyse()
    for k, v in t.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
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
anything. Rung c and the experiment rerun were **not** done: rung a does not meet its targets (below).

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
* **The author's own beta_P.** Fitted to his simulated sim-13 pressure (Fig. 5.20a, digitised): **0.102 s-1**, each
  exponential step alone 0.09-0.11 s-1 (EP included). With it our sim-13 pressure, loading profiles and N2 front
  coincide with his (table below).
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
about 20 % more H2 to pressurise (from eq. 5.49 with his recovery and productivity). Equations 5.3 and 5.6 include
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
2. Feed flow basis: his productivity/recovery ratios imply 0.6-1.3 % more feed H2 per cycle (1 atm instead of
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

Status after the follow-up: verification against the author's model is **incomplete** — impurity is reproduced
within ~2x with two corrections inferred from the author's figures, recovery stays 4.0-4.6 pp low (target 2 pp).
Rung c and the post-correction experiment comparison were not run. The gates below are those of the **first**
comparison (pre-declared baseline).

| gate | result |
|---|---|
| 1. Numerical (CSS, mole balance, grid) | {verdict(t["gate1"])} |
| 2. Experiment (recovery, impurity ppm, CO ppm at CSS) | {"PASS" if gate2_pass else "FAIL"} |
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
