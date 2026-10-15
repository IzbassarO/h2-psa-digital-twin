"""Step 7c: paper figures F5-F10 and tables T1-T3 from committed data only (no simulations, no refits).

    python scripts/make_figures.py [--out results]
    python scripts/make_figures.py --export-traces      # once: steps 2-3 model traces -> data/generated (see below)

Writes <out>/figures/F*.pdf (vector) and F*.png (300 dpi), <out>/figures/captions.md, <out>/tables/T*.csv and T*.tex
(booktabs), and <out>/tables/figure_data_F*.csv: every number plotted in a figure (the T*.csv hold the numbers of
the tables at full precision). Style: scripts/figures.mplstyle (one font, Okabe-Ito colours: P = C0, G = C1, L = C2
in every figure), widths 90 mm and 190 mm.

Inputs: data/generated/*.csv (labels, memberships, model traces), docs/validation/step*/ (frozen predictions and the
metric tables of Steps 6b-7b), configs/*.yaml, data/params/*.yaml, data/benchmarks/digitized/*.csv.
The model curves of F5 / F6 are outputs of Steps 2 and 3 that lived only in the git-ignored results/: `--export-traces`
copies them (no new simulation) from results/step2/sim_run{1..5}.csv (final step-2 model) and
results/step3/cases/cor_exp.pkl (frozen baseline, experiment cycle) to data/generated/model_breakthrough_step2.csv
and data/generated/model_psa_cycle_step3.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.isotherms import R, load_yaml  # noqa: E402
from h2psa.step6_metrics import _observed, clopper_pearson, join_predictions, newcombe_difference  # noqa: E402

GEN, DIG = ROOT / "data" / "generated", ROOT / "data" / "benchmarks" / "digitized"
VAL = ROOT / "docs" / "validation"
FROZEN, C6, D6, A7, B7 = (VAL / s for s in ("step6b", "step6c", "step6d", "step7a", "step7b"))
BT_TRACES, CYCLE_TRACES = GEN / "model_breakthrough_step2.csv", GEN / "model_psa_cycle_step3.csv"
STYLE = Path(__file__).with_name("figures.mplstyle")
MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
VARIANT = {"P": "C0", "G": "C1", "L": "C2"}
SHORT_NAME = {"P": "point estimate", "G": "global CP", "L": "local CP"}
SPECIES = {"H2": "C3", "CO2": "C4", "CO": "C5", "CH4": "C6", "N2": "0.55"}
MARKER = {"H2": "^", "CO2": "s", "CO": "x", "CH4": "+", "N2": "D"}            # as in the thesis figures
TC = {"Top": "C3", "Middle": "C4", "Bottom": "C5"}                              # z = 0.239, 0.151, 0.059 m
SCENARIO_NAME = {"baseline": "baseline", "ep_inflow_ed_gas": "EP = ED gas", "thermal_range_low": "thermal low",
                 "thermal_range_high": "thermal high"}
PHI_BINS = (0.70, 0.775, 0.85, 0.925, 1.00)
CAPTIONS = {
    "F5": "Model validation on the breakthrough runs 1-5 of the compact unit (Lopes 2010, Figs 5.15-5.17): outlet "
          "molar flow per species normalised by its feed flow (top) and thermocouple temperatures (bottom); lines: "
          "this model (Step 2, final), symbols: digitised experiment; first 3000 s of the 6000-s runs shown "
          "(later points lie on the feed plateau).",
    "F6": "PSA cycle validation (experiment cycle, frozen baseline model): pressure at the open column end in "
          "cycle 1 and thermocouple temperatures over cycles 1-10 and in cycle 37; lines: model, symbols: measured "
          "(Lopes 2010, Figs 5.18-5.19).",
    "F7": "Predicted bound vs re-simulated impurity z = log10(ppm) of the optimiser-selected points (GP, n_train = 512, "
          "alpha = 0.10) for P, G and L (CP: conformal prediction); dashed: 100 ppm limit, dotted: median surrogate mean "
          "prediction; filled: violations. Right: violation "
          "rate with Clopper-Pearson 95 % CI, pooled per variant (Step 6c, pre-registered) and per surrogate "
          "(Step 6d, exploratory).",
    "F8": "Mechanism: (a) GP-L predictive sigma and (b) GBM-L quantile-minus-mean gap Qhat_(1-alpha) - mean at the "
          "selected points vs the random D_test points (alpha = 0.10; Step 6d, exploratory).",
    "F9": "Cost of the guarantee (Step 7a, exploratory, fixed reference point (0, 0)): (a) realised fronts of P, G and "
          "L (GP, n_train = 512, alpha = 0.10) and the best-known front; (b) H2 recovery loss of G and L vs the "
          "oracle-filtered P front at matched productivity.",
    "F10": "Robustness: (a) violation rate vs adsorbent capacity factor phi on S_dist (Step 6c; binned, CP 95 % CI) "
           "with the RQ5 logistic miscoverage fit; (b) violation rate of the GP n_train = 512 selected points under the "
           "Step 7b model-form scenarios (descriptive).",
}


def read(path, **kw):
    return pd.read_csv(path, float_precision="round_trip", **kw)


def dig(name):
    return pd.read_csv(DIG / f"lopes2010_{name}.csv", comment="#")


def cp(k, n):
    lo, hi = clopper_pearson(int(k), int(n))
    return dict(n=int(n), k=int(k), rate=k / n if n else np.nan, ci_lower=lo, ci_upper=hi)


def selected():
    """Optimiser-selected S_nom points joined with their frozen bounds; censored as z = 3, failed dropped."""
    return _observed(join_predictions(read(GEN / "S_nom.csv"), read(FROZEN / "predictions" / "S_nom.csv"),
                                      read(GEN / "S_nom_memberships.csv")))


def save(fig, name, out):
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}")
    plt.close(fig)


def label(ax, text):
    ax.set_title(text, loc="left")


def rate_points(ax, x, rows, colour, marker="o", filled=True, **kw):
    r = pd.DataFrame(rows)
    ax.errorbar(x, r["rate"], yerr=[r["rate"] - r["ci_lower"], r["ci_upper"] - r["rate"]], fmt=marker, color=colour,
                mfc=colour if filled else "white", ms=3.5, lw=0.8, **kw)


# ---------------------------------------------------------------------------- F5, F6: model validation
def export_traces():
    parts = []
    for run in range(1, 6):
        s = read(ROOT / "results" / "step2" / f"sim_run{run}.csv")
        keep = np.where(s["time_s"] < 2000, s["time_s"] % 2 == 0, s["time_s"] % 10 == 0)     # 2 s / 10 s steps
        parts.append(s.loc[keep].drop(columns="He_mol_s").assign(run=run))
    bt = pd.concat(parts, ignore_index=True)
    bt[["run", *[c for c in bt if c != "run"]]].to_csv(BT_TRACES, index=False)
    with open(ROOT / "results" / "step3" / "cases" / "cor_exp.pkl", "rb") as fh:
        res = pickle.load(fh)
    tcs = ("Bottom", "Middle", "Top")                                         # order of res["z_tc"]
    rows = []
    for k in (*range(1, 11), 37):
        tr = res["traces"][k]
        p_open = np.where(np.isin(tr["step"], ["Blow", "Purge"]), tr["P0"], tr["PL"]) / 1e5   # as run_psa.py
        rows.append(pd.DataFrame(dict(cycle=k, time_s=tr["t"], step=tr["step"], P_open_bar=p_open,
                                      **{f"T_{s}_K": tr["T"][i] for i, s in enumerate(tcs)})))
    pd.concat(rows, ignore_index=True).to_csv(CYCLE_TRACES, index=False)
    print(f"wrote {BT_TRACES.relative_to(ROOT)} and {CYCLE_TRACES.relative_to(ROOT)}")


def feed_flows(run):
    """Species feed flows [mol/s] of breakthrough run `run`, as h2psa.column.compact_breakthrough."""
    bt = load_yaml(ROOT / "data" / "params" / "compact_column_lopes2010.yaml")["breakthrough_runs"]
    std, y = bt["std_conditions"], bt["runs"][run]["y"]
    flow = std["P_bar"] * 1e5 * bt["flow_rate_std"] / (R * std["T"])
    return {s: flow * v / sum(y.values()) for s, v in y.items() if v > 0}


def f5(out):
    model = read(BT_TRACES)
    fig, axs = plt.subplots(2, 5, figsize=(DOUBLE, 92 * MM), sharex=True, sharey="row")
    data = []
    for j, run in enumerate(range(1, 6)):
        F0, m = feed_flows(run), model.loc[model["run"] == run]
        exp = pd.concat([dig(f"run{run}_flow"), *([dig(f"run{run}_zoom")] if run in (4, 5) else [])])
        ax = axs[0, j]
        for s, f in F0.items():
            y = m[f"{s}_mol_s"] / f
            ax.plot(m["time_s"], y, color=SPECIES[s], lw=0.8, label=f"{s} model")
            g = exp.loc[exp["series"] == s]
            ye = g["molar_flow_mmol_s"] * 1e-3 / f
            ax.plot(g["time_s"], ye, MARKER[s], color=SPECIES[s], mfc="none", ms=2.5, mew=0.5, label=f"{s} exp.")
            data += [pd.DataFrame(dict(panel=f"flow run {run}", source="model", series=s, x=m["time_s"], y=y)),
                     pd.DataFrame(dict(panel=f"flow run {run}", source="experiment", series=s, x=g["time_s"], y=ye))]
        label(ax, f"({'abcde'[j]}) run {run}")
        ax = axs[1, j]
        exp = dig(f"run{run}_T")
        for s, c in TC.items():
            g = exp.loc[exp["series"] == s]
            ax.plot(g["time_s"], g["temperature_K"], ".", color=c, ms=0.8, alpha=0.35, rasterized=True)
            ax.plot(m["time_s"], m[f"T_{s}_K"], color=c, lw=0.8, label=s)
            data += [pd.DataFrame(dict(panel=f"T run {run}", source="model", series=s, x=m["time_s"], y=m[f"T_{s}_K"])),
                     pd.DataFrame(dict(panel=f"T run {run}", source="experiment", series=s, x=g["time_s"],
                                       y=g["temperature_K"]))]
        ax.set_xlabel("time (s)")
        if run == 1:
            ax.legend(title="thermocouple (line: model)", loc="upper right")
    axs[0, 0].set_ylabel("outlet / feed molar flow (-)")
    axs[1, 0].set_ylabel("temperature (K)")
    axs[0, 0].set_xlim(0, 3000)
    fig.legend([Line2D([], [], color=SPECIES[s], marker=MARKER[s], mfc="none", ms=3, lw=0.8) for s in SPECIES],
               list(SPECIES), loc="outside upper center", ncol=5, title="lines: model, symbols: experiment")
    save(fig, "F5_breakthrough_validation", out)
    return pd.concat(data, ignore_index=True)


def step_edges(cycle):
    t, step = cycle["time_s"].to_numpy(), cycle["step"].to_numpy()
    return t[1:][step[1:] != step[:-1]]


def f6(out):
    model = read(CYCLE_TRACES)
    c1 = model.loc[model["cycle"] == 1]
    t_cycle = c1["time_s"].max()
    fig = plt.figure(figsize=(DOUBLE, 100 * MM))
    gs = fig.add_gridspec(3, 3, width_ratios=(1.15, 2.0, 1.0))
    ax = fig.add_subplot(gs[:, 0])
    exp = dig("psa_c1_P")
    exp = exp.loc[~(exp["time_s"].between(180, 200, inclusive="neither") & (exp["pressure_bar"] > 4))]  # run_psa.py
    for b in step_edges(c1):
        ax.axvline(b, color="0.88", lw=0.5, zorder=0)
    ax.plot(exp["time_s"], exp["pressure_bar"], "o", color="0.35", mfc="none", ms=1.8, mew=0.4, label="measured")
    ax.plot(c1["time_s"], c1["P_open_bar"], color="C3", lw=0.9, label="model")
    ax.set(xlabel="time in cycle 1 (s)", ylabel="pressure at the open end (bar)", xlim=(0, t_cycle))
    ax.legend(loc="upper right")
    label(ax, "(a) pressure, cycle 1")
    data = [pd.DataFrame(dict(panel="pressure cycle 1", source="experiment", series="P", x=exp["time_s"],
                              y=exp["pressure_bar"])),
            pd.DataFrame(dict(panel="pressure cycle 1", source="model", series="P", x=c1["time_s"], y=c1["P_open_bar"]))]
    c10 = model.loc[model["cycle"] <= 10]
    c37 = model.loc[model["cycle"] == 37]
    e10, e37 = dig("psa_c1-10_T"), dig("psa_c37_T")
    for k, (s, c) in enumerate(TC.items()):
        for col, (m, t, e, name) in enumerate(((c10, c10["time_s"] + (c10["cycle"] - 1) * t_cycle, e10, "cycles 1-10"),
                                               (c37, c37["time_s"], e37, "cycle 37"))):
            ax = fig.add_subplot(gs[k, col + 1])
            g = e.loc[e["series"] == s]
            ax.plot(g["time_s"], g["temperature_K"], ".", color=c, ms=1.5, alpha=0.5, rasterized=True)
            ax.plot(t, m[f"T_{s}_K"], color=c, lw=0.8)
            ax.set_ylabel(f"{s} T (K)")
            if k == 0:
                label(ax, f"({'bc'[col]}) thermocouples, {name}")
            if k == 2:
                ax.set_xlabel("time (s)" if col == 0 else "time in cycle 37 (s)")
            else:
                ax.tick_params(labelbottom=False)
            data += [pd.DataFrame(dict(panel=f"T {name}", source="experiment", series=s, x=g["time_s"],
                                       y=g["temperature_K"])),
                     pd.DataFrame(dict(panel=f"T {name}", source="model", series=s, x=t, y=m[f"T_{s}_K"]))]
    save(fig, "F6_psa_cycle_validation", out)
    return pd.concat(data, ignore_index=True)


# ---------------------------------------------------------------------------- F7: key figure
def f7(out):
    sel = selected()
    fig = plt.figure(figsize=(DOUBLE, 58 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 1.5))
    data, first = [], None
    xlim, ylim = (1.975, 2.003), (0.6, 3.0)       # every selected bound lies in [1.979, 2.000]
    for i, v in enumerate("PGL"):
        g = sel.loc[sel["arm"] == f"GP_n512_{v}_a0.10"]
        bad = g["z"] > 2.0
        ax = fig.add_subplot(gs[i], sharex=first, sharey=first)
        first = first or ax
        ax.axhline(2.0, ls="--", color="0.25", lw=0.7)
        ax.axvline(2.0, ls="--", color="0.25", lw=0.7)
        if v != "P":                                # P: the bound is the surrogate mean itself
            ax.axhline(g["z_hat"].median(), ls=":", color="0.45", lw=0.8)
            ax.text(xlim[0] + 0.001, g["z_hat"].median() + 0.03, "median surrogate mean", fontsize=5.5, color="0.45")
        ax.scatter(g.loc[~bad, "bound"], g.loc[~bad, "z"], s=11, facecolors="none", edgecolors=VARIANT[v], lw=0.6,
                   label="feasible")
        ax.scatter(g.loc[bad, "bound"], g.loc[bad, "z"], s=13, color=VARIANT[v], edgecolors="k", lw=0.4,
                   label="violation")
        ax.text(0.04, 0.96, f"{int(bad.sum())}/{len(g)} violate", transform=ax.transAxes, va="top")
        ax.text(xlim[0] + 0.001, 2.03, "100 ppm", va="bottom", fontsize=6, color="0.25")
        ax.set(xlim=xlim, ylim=ylim, xlabel="predicted bound, log10 ppm", xticks=(1.98, 1.99, 2.0))
        ax.set_ylabel("re-simulated z, log10 ppm" if i == 0 else "")
        if i:
            ax.tick_params(labelleft=False)
        label(ax, f"({'abc'[i]}) {v}, {SHORT_NAME[v]}")
        data.append(g[["id", "arm", "variant", "bound", "z_hat", "z", "status"]].assign(panel=f"scatter {v}",
                                                                                          violation=bad))
    ax = fig.add_subplot(gs[3])
    rows = []
    for x, (_, r) in enumerate(read(C6 / "pooled_violations.csv").iterrows()):
        rows.append(dict(panel="rates", group="6c pooled (pre-registered)", surrogate="pooled", variant=r["variant"],
                         x=x, **cp(r["k"], r["n"])))
    d6 = read(D6 / "pooled_violations.csv")
    for j, s in enumerate(("GP", "GBM")):
        for i, v in enumerate("PGL"):
            r = d6.loc[(d6["surrogate"] == s) & (d6["variant"] == v)].iloc[0]
            rows.append(dict(panel="rates", group="6d by surrogate (exploratory)", surrogate=s, variant=v,
                             x=3.7 + 3.4 * j + i, **cp(r["k"], r["n"])))
    rates = pd.DataFrame(rows)
    ax.axhline(0.10, ls="--", color="0.4", lw=0.6)
    ax.text(9.6, 0.11, r"$\alpha$ = 0.10", ha="right", va="bottom", fontsize=6, color="0.4")
    for _, r in rates.iterrows():
        rate_points(ax, [r["x"]], [r], VARIANT[r["variant"]], marker="D" if r["surrogate"] == "pooled" else
                    "o" if r["surrogate"] == "GP" else "s")
    ax.set(xticks=rates["x"], xticklabels=rates["variant"], ylim=(-0.02, 1.02), ylabel="violation rate of selected points",
           xlim=(-0.6, 10.4))
    for x, text in ((1, "pooled\n6c (prereg.)"), (4.7, "GP\n6d (expl.)"), (8.1, "GBM\n6d (expl.)")):
        ax.text(x, -0.13, text, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6)
    for x in (2.85, 6.25):
        ax.axvline(x, color="0.85", lw=0.5)
    label(ax, "(d) violation rate, alpha = 0.10")
    save(fig, "F7_bound_vs_resimulated", out)
    return pd.concat([*data, rates], ignore_index=True)


# ---------------------------------------------------------------------------- F8: mechanism
def widths():
    """GP-L sigma = (bound - zhat) / q and GBM-L Qhat - mean = bound - q - zhat per point; checked against the
    quantiles of docs/validation/step6d/mechanism.csv."""
    q = read(FROZEN / "q.csv").set_index("arm")["q"]
    test = _observed(join_predictions(read(GEN / "D_test.csv"), read(FROZEN / "predictions" / "D_test.csv")))
    parts = []
    for name, f in (("D_test", test), ("selected", selected())):
        f = f.loc[(f["variant"] == "L") & np.isclose(f["alpha"], 0.10)]
        qq = f["arm"].map(q)
        w = np.where(f["surrogate"] == "GP", (f["bound"] - f["z_hat"]) / qq, f["bound"] - qq - f["z_hat"])
        parts.append(f[["id", "arm", "surrogate", "n_train", "z"]].assign(set=name, width=w))
    w = pd.concat(parts, ignore_index=True)
    mech = read(D6 / "mechanism.csv").set_index(["arm", "set"])
    for (arm, name), g in w.groupby(["arm", "set"]):
        ref = mech.loc[(arm, name), ["w_q10", "w_q50", "w_q90"]].to_numpy(float)
        if not np.allclose(np.quantile(g["width"], (0.1, 0.5, 0.9)), ref, rtol=0, atol=1e-9):
            raise ValueError(f"{arm} {name}: widths do not reproduce docs/validation/step6d/mechanism.csv")
    return w


def f8(out):
    w = widths()
    fig, axs = plt.subplots(2, 1, figsize=(SINGLE, 100 * MM))
    rng = np.random.default_rng(0)                          # jitter only
    for ax, s, ylabel, title in ((axs[0], "GP", r"GP predictive $\sigma$ (log10 ppm)", "(a) GP-L: sigma"),
                                 (axs[1], "GBM", "$\\hat Q_{1-\\alpha}$ - mean (log10 ppm)\n< 0: quantile crossing",
                                  "(b) GBM-L: quantile minus mean")):
        for i, n in enumerate((128, 256, 512)):
            for j, (name, colour) in enumerate((("D_test", "0.55"), ("selected", VARIANT["L"]))):
                v = w.loc[(w["surrogate"] == s) & (w["n_train"] == n) & (w["set"] == name), "width"]
                x = 3 * i + j
                ax.boxplot(v, positions=[x], widths=0.6, showfliers=False, medianprops=dict(color="k", lw=0.8),
                           boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
                ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=2, color=colour, alpha=0.6, lw=0,
                           label=name if i == 0 else None, rasterized=True)
        if s == "GBM":
            ax.axhline(0, color="0.3", lw=0.6, ls="--")
        ax.set(xticks=[0.5, 3.5, 6.5], xticklabels=[f"n = {n}" for n in (128, 256, 512)], ylabel=ylabel,
               xlim=(-0.7, 7.7))
        label(ax, title)
    axs[1].set_xlabel("training set size")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, markerscale=2)
    save(fig, "F8_mechanism", out)
    summary = w.groupby(["arm", "set"])["width"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).reset_index()
    return pd.concat([w.assign(panel="points"), summary.assign(panel="summary")], ignore_index=True)


# ---------------------------------------------------------------------------- F9: cost of the guarantee
def f9(out):
    fronts, arms, best = read(A7 / "realised_fronts.csv"), read(A7 / "arms.csv"), read(A7 / "best_known_front.csv")
    loss = read(A7 / "recovery_loss.csv")
    sel = selected()
    fig, axs = plt.subplots(2, 1, figsize=(SINGLE, 105 * MM))
    ax = axs[0]
    b = best.sort_values("productivity")
    ax.plot(b["productivity"], b["recovery"], "--", color="0.5", lw=0.8, label="best-known front")
    data = [b.assign(panel="fronts", series="best-known")]
    p_bad = sel.loc[(sel["arm"] == "GP_n512_P_a0.10") & (sel["z"] > 2.0)]
    ax.plot(p_bad["productivity"], p_bad["recovery"], "x", color=VARIANT["P"], ms=2.5, mew=0.5, alpha=0.6,
            label="P, violating (not usable)")
    data.append(p_bad[["id", "recovery", "productivity"]].assign(panel="fronts", series="P violating"))
    for v in "PGL":
        arm = f"GP_n512_{v}_a0.10"
        f = fronts.loc[fronts["arm"] == arm].sort_values("productivity")
        hv = arms.set_index("arm").loc[arm, "hv_fraction_of_best"]
        ax.plot(f["productivity"], f["recovery"], "o-", color=VARIANT[v], ms=2.5, lw=0.9,
                label=f"{v} front (HV {hv:.2f} of best)")
        data.append(f.assign(panel="fronts", series=v, hv_fraction_of_best=hv))
    ax.set(xlabel=r"productivity (mol H$_2$ kg$^{-1}$ day$^{-1}$)", ylabel=r"H$_2$ recovery (%)")
    ax.legend(loc="lower left")
    label(ax, "(a) realised fronts, GP n = 512, alpha = 0.10")
    ax = axs[1]
    for v in "GL":
        for a, ls, filled in ((0.10, "-", True), (0.05, "--", False)):
            arm = f"GP_n512_{v}_a{a:.2f}"
            g = loss.loc[loss["arm"] == arm]
            ax.plot(g["productivity"], g["loss_vs_P"], ls, marker="o", color=VARIANT[v], ms=3,
                    mfc=VARIANT[v] if filled else "white", label=f"{v}, alpha = {a:.2f}")
            data.append(g.assign(panel="loss vs P", series=arm))
    ax.set(xlabel=r"productivity (mol H$_2$ kg$^{-1}$ day$^{-1}$)", ylabel="recovery loss vs P (pp)", ylim=(0, None))
    ax.legend(loc="upper left")
    label(ax, "(b) recovery loss at matched productivity")
    save(fig, "F9_cost_of_guarantee", out)
    return pd.concat(data, ignore_index=True)


# ---------------------------------------------------------------------------- F10: robustness
def f10(out):
    sd = _observed(join_predictions(read(GEN / "S_dist.csv"), read(FROZEN / "predictions" / "S_dist.csv"),
                                    read(GEN / "S_dist_memberships.csv")))
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 72 * MM), gridspec_kw=dict(width_ratios=(1, 1.15)))
    ax = axs[0]
    rows = []
    sd["bin"] = pd.cut(sd["phi"], PHI_BINS, include_lowest=True)
    for i, v in enumerate("PGL"):
        for interval, g in sd.loc[sd["variant"] == v].groupby("bin", observed=False):
            rows.append(dict(panel="S_dist binned", variant=v, phi_lower=interval.left, phi_upper=interval.right,
                             x=interval.mid + (i - 1) * 0.012, **cp((g["z"] > 2.0).sum(), len(g))))
    binned = pd.DataFrame(rows)
    for v in "PGL":
        b = binned.loc[binned["variant"] == v]
        rate_points(ax, b["x"], b.to_dict("records"), VARIANT[v], label=v)
    coef = read(C6 / "logistic_coefficients.csv").set_index("term")["estimate"]
    lo, hi = load_yaml(ROOT / "configs" / "disturbances.yaml")["variables"]["phi"]["bounds"]
    gl = sd.loc[sd["variant"].isin(("G", "L"))]
    phi = np.linspace(lo, hi, 61)
    curves = []
    for s, ls in (("GP", "-"), ("GBM", ":")):
        for v in "GL":
            eta = (coef["intercept_S_dist"] + coef["L"] * (v == "L") + coef["GBM"] * (s == "GBM") +
                   coef["b_phi"] * (phi - lo) / (hi - lo) + coef["b_CH4"] * gl["u_y_CH4"].mean() +
                   coef["b_CO"] * gl["u_y_CO"].mean())
            ax.plot(phi, expit(eta), ls, color=VARIANT[v], lw=0.9, label=f"{v}-{s} fit")
            curves.append(pd.DataFrame(dict(panel="RQ5 logistic", variant=v, surrogate=s, x=phi, rate=expit(eta))))
    ax.set(xlabel=r"adsorbent capacity factor $\varphi$ (-)", ylabel="rate", ylim=(-0.02, 1.02), xlim=(0.69, 1.01))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, columnspacing=1.0,
              title="symbols: z > 2, binned; lines: RQ5 fit of P(z > bound)")
    label(ax, "(a) feed / capacity disturbance (S_dist)")
    ax = axs[1]
    v7 = read(B7 / "violations.csv")
    sets = list(SCENARIO_NAME)
    offsets = {"GP_n512_P_a0.10": -0.3, "GP_n512_G_a0.10": -0.15, "GP_n512_G_a0.05": 0.0, "GP_n512_L_a0.10": 0.15,
               "GP_n512_L_a0.05": 0.3}
    v7["x"] = v7["set"].map(sets.index) + v7["arm"].map(offsets)
    for arm in offsets:
        g = v7.loc[v7["arm"] == arm]
        i = g.iloc[0]
        rate_points(ax, g["x"], g.to_dict("records"), VARIANT[i["variant"]], filled=np.isclose(i["alpha"], 0.10),
                    label=rf"{i['variant']}, $\alpha$ = {i['alpha']:.2f}")
    ax.axhline(0.10, ls="--", color="0.4", lw=0.6)
    ax.set(xticks=range(len(sets)), xticklabels=[SCENARIO_NAME[s] for s in sets], ylim=(-0.02, 1.02),
           ylabel="violation rate of selected points")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, columnspacing=1.0)
    label(ax, "(b) model-form scenarios (Step 7b, descriptive)")
    save(fig, "F10_robustness", out)
    return pd.concat([binned, *curves, v7.assign(panel="Step 7b")], ignore_index=True)


# ---------------------------------------------------------------------------- tables
def tex_cell(v):
    if isinstance(v, float) and not np.isfinite(v):
        return "--"
    s = v if isinstance(v, str) else f"{v:.4g}"
    return s if s.startswith("$") else re.sub(r"([%&_#{}])", r"\\\1", s).replace("<=", r"$\le$").replace(">=", r"$\ge$")


def tex(frame, path, caption, label_):
    lines = [r"\begin{table}", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label_}}}", r"\small",
             rf"\begin{{tabular}}{{{'l' * frame.shape[1]}}}", r"\toprule",
             " & ".join(tex_cell(c) for c in frame.columns) + r" \\", r"\midrule",
             *(" & ".join(tex_cell(v) for v in row) + r" \\" for row in frame.itertuples(index=False)),
             r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def t1():
    from h2psa.cycle import experiment_cycle
    c = experiment_cycle()
    ref = dict(t_feed=c.t_feed, t_blow=c.t_blow, P_high=c.P_high / 1e5, P_low=c.P_low / 1e5, f_zeolite=0.4,
               n_equalisations=len(c.P_inter), feed_flow=1.0,
               purge_to_feed=c.purge_flow * c.t_purge / (c.feed_flow * c.y_feed["H2"] * c.t_feed))
    symbol = dict(t_feed=r"$t_\mathrm{feed}$", t_blow=r"$t_\mathrm{blow}$", purge_to_feed=r"$r_\mathrm{purge}$",
                  P_high=r"$P_\mathrm{H}$", P_low=r"$P_\mathrm{L}$", f_zeolite=r"$f_\mathrm{ZEO}$",
                  n_equalisations=r"$n_\mathrm{eq}$", feed_flow=r"$F/F_0$", phi=r"$\varphi$",
                  y_CH4=r"$y_\mathrm{CH_4}$", y_CO=r"$y_\mathrm{CO}$")
    rows = []
    for kind, path, key in (("decision", "design_space.yaml", "decision_variables"),
                            ("disturbance", "disturbances.yaml", "variables")):
        for name, v in load_yaml(ROOT / "configs" / path)[key].items():
            lo, hi = (min(v["values"]), max(v["values"])) if "values" in v else v["bounds"]
            rows.append(dict(type=kind, variable=name, symbol=symbol[name], lower=lo, upper=hi,
                             levels=("{" + ", ".join(f"{x:g}" for x in v["values"]) + "}" if "values" in v else
                                     f"grid 1/{v['grid']}" if "grid" in v else "continuous"),
                             units=v["units"], reference=v.get("nominal", ref.get(name))))
    return pd.DataFrame(rows)


def t2():
    m = json.loads((C6 / "metrics.json").read_text())
    dec, cnt, rob, lg = m["tests"]["decisions"], m["tests"]["pooled_counts"], m["robustness"], m["logistic"]
    hv = read(C6 / "fronts.csv").set_index("arm")["hv_ratio"]
    g, l = cnt["G"], cnt["L"]
    d, d_lo, d_hi = newcombe_difference(g["k"], g["n"], l["k"], l["n"])
    def row(rq, h0, est, lo, hi, boot, p_key="p_value"):
        d = dec.get(rq[:3], {})                    # RQ3a / RQ3b: the two parts of RQ3 (p_a, p_b; RQ3 = both)
        return dict(RQ=rq, H0=h0, estimate=est, ci_lower=lo, ci_upper=hi, p_one_sided=d.get(p_key, np.nan),
                    holm_level=d.get("level", np.nan), p_adjusted=d.get("adjusted_p", np.nan),
                    verdict=d.get("verdict", "descriptive"), cluster_bootstrap=boot)
    return pd.DataFrame([
        row("RQ1", "violation rate P <= 0.10", f"{cnt['P']['k']}/{cnt['P']['n']} = {cnt['P']['rate']:.3f}",
            cnt["P"]["ci_lower"], cnt["P"]["ci_upper"], f"lower {rob['RQ1']['lower']:.3f}: {rob['RQ1']['verdict']}"),
        row("RQ2", "violation rate G <= 0.10", f"{g['k']}/{g['n']} = {g['rate']:.3f}", g["ci_lower"], g["ci_upper"],
            f"lower {rob['RQ2']['lower']:.3f}: {rob['RQ2']['verdict']}"),
        row("RQ3a", "violation rate L >= 0.17", f"{l['k']}/{l['n']} = {l['rate']:.3f}", l["ci_lower"], l["ci_upper"],
            f"upper {rob['RQ3']['L_upper']:.3f}: {rob['RQ3']['verdict']}", "p_a"),
        row("RQ3b", "G - L <= 0", f"{d:+.3f}", d_lo, d_hi,
            f"lower {rob['RQ3']['difference_lower']:+.3f}: {rob['RQ3']['verdict']}", "p_b"),
        row("RQ4", "none (descriptive)", "HV ratio GP n = 512: P {:.3f}, G {:.3f}, L {:.3f}".format(
            *(hv[f"GP_n512_{v}_a0.10"] for v in "PGL")), np.nan, np.nan, ""),
        row("RQ5", "b_phi >= 0", f"b_phi = {lg['b_phi']:.3f} (OR phi 0.70 vs 1.00 = {lg['odds_ratio']:.2f})",
            lg["b_phi_lower"], lg["b_phi_upper"], f"upper {rob['RQ5']['upper']:.3f}: {rob['RQ5']['verdict']}"),
    ])


def t3():
    v = read(B7 / "violations.csv")
    paired = read(B7 / "paired.csv").set_index(["set", "arm"])
    fronts = read(B7 / "fronts.csv").set_index(["set", "arm"])
    spots = read(B7 / "spot_checks.csv").assign(abs_dz=lambda d: d["dz"].abs()).groupby("scenario")["abs_dz"].max()
    base = read(GEN / "S_nom.csv").set_index("id")
    ids = sorted(set(read(GEN / "S_nom_memberships.csv").query("surrogate == 'GP' and n_train == 512")["id"]))
    rows = []
    for s in SCENARIO_NAME:
        lab = base.loc[ids] if s == "baseline" else read(GEN / f"M_{s}.csv").set_index("id")
        dz = lab["z"] - base.loc[lab.index, "z"]
        row = dict(scenario=SCENARIO_NAME[s], key=s, n=len(lab), css=int(lab["status"].eq("css").sum()),
                   median_dz=dz.median(), spot_check_max_abs_dz=spots.get(s, np.nan))
        for _, r in v.loc[v["set"] == s].iterrows():
            a = r["arm"].removeprefix("GP_n512_")
            row.update({f"{a}_k": r["k"], f"{a}_n": r["n"], f"{a}_rate": r["rate"],
                        f"{a}_ci_lower": r["ci_lower"], f"{a}_ci_upper": r["ci_upper"],
                        f"{a}_hv": fronts.loc[(s, r["arm"]), "hv"],
                        f"{a}_feasible_to_violating": paired["feasible_to_violating"].get((s, r["arm"]), np.nan),
                        f"{a}_mcnemar_p": paired["mcnemar_exact_p_two_sided"].get((s, r["arm"]), np.nan)})
        rows.append(row)
    return pd.DataFrame(rows)


def t3_tex(t):
    arms = [a for a in ("P_a0.10", "G_a0.10", "G_a0.05", "L_a0.10", "L_a0.05")]
    out = pd.DataFrame({"": ["runs at CSS", "spot check max $|\\Delta z|$", "median $\\Delta z$ vs baseline",
                             *[f"violations {a.replace('_a', ', $\\alpha$ = ')}" for a in arms],
                             "P: feasible $\\to$ violating (McNemar p)"]})
    for _, r in t.iterrows():
        mc = "--" if r["key"] == "baseline" else f"{r['P_a0.10_feasible_to_violating']:.0f} ({r['P_a0.10_mcnemar_p']:.2g})"
        out[r["scenario"]] = [f"{r['css']}/{r['n']}", "--" if not np.isfinite(r["spot_check_max_abs_dz"]) else
                              f"{r['spot_check_max_abs_dz']:.1e}", f"{r['median_dz']:+.3f}",
                              *[f"{r[f'{a}_k']:.0f}/{r[f'{a}_n']:.0f}" for a in arms], mc]
    return out


# ---------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--export-traces", action="store_true")
    args = parser.parse_args(argv)
    if args.export_traces:
        export_traces()
        return 0
    for d in ("figures", "tables"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    tables = args.out / "tables"
    with plt.style.context(STYLE):
        for name, make in (("F5", f5), ("F6", f6), ("F7", f7), ("F8", f8), ("F9", f9), ("F10", f10)):
            make(args.out).to_csv(tables / f"figure_data_{name}.csv", index=False)
    a = t1()
    a.to_csv(tables / "T1_design_space.csv", index=False)
    tex(a.assign(range=[r.levels if r.levels.startswith("{") else f"[{r.lower:g}, {r.upper:g}]" +
                        (f", step {r.levels[5:]}" if r.levels.startswith("grid") else "") for r in a.itertuples()])[["type", "variable", "symbol", "range", "units", "reference"]],
        tables / "T1_design_space.tex", "Decision variables and disturbance ranges (configs/design\\_space.yaml, "
        "configs/disturbances.yaml); reference: experiment set point (Lopes 2010, Table 5.4) or nominal disturbance.",
        "tab:design_space")
    b = t2()
    b.to_csv(tables / "T2_preregistered_results.csv", index=False)
    fmt = lambda x: "--" if not np.isfinite(x) else f"{x:.3g}"  # noqa: E731
    tex(b.assign(ci95=[("--" if not np.isfinite(lo) else f"[{lo:.3f}, {hi:.3f}]")
                       for lo, hi in zip(b["ci_lower"], b["ci_upper"])],
                 p=b["p_one_sided"].map(fmt), p_adj=b["p_adjusted"].map(fmt), level=b["holm_level"].map(fmt))
        [["RQ", "H0", "estimate", "ci95", "p", "level", "p_adj", "verdict", "cluster_bootstrap"]],
        tables / "T2_preregistered_results.tex", "Pre-registered results of Step 6c (Holm step-down over RQ1, RQ2, RQ3, "
        "RQ5 at family-wise 0.05; RQ3 requires both RQ3a and RQ3b; RQ4 descriptive).", "tab:preregistered")
    c = t3()
    c.to_csv(tables / "T3_model_form_scenarios.csv", index=False)
    tex(t3_tex(c), tables / "T3_model_form_scenarios.tex", "Model-form scenarios of Step 7b (descriptive, designed "
        "after the Step 6c results): 210 selected points of the GP arms at $n_\\mathrm{train}$ = 512.",
        "tab:model_form")
    (args.out / "figures" / "captions.md").write_text("".join(f"* **{k}.** {v}\n" for k, v in CAPTIONS.items()))
    print(f"figures in {args.out / 'figures'}, tables in {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
