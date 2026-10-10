"""Step 4: layered bed (HY2SEPS-AC at the feed end + HY2SEPS-ZEO at the product end).

    python scripts/run_layered.py [--workers 7] [--force] [--report-only] [--only case1,case2]

Cases run from the He-filled column to CSS with the frozen step-3 cycle (cycle B, beta_P of the author's
simulations, EP = pure H2) and are cached in results/step4/cases/<case>.pkl. Then the pre-declared gates are
evaluated and results/step4/REPORT.md, figures and CSVs are written; REPORT.md and figures are copied to
docs/validation/step4.md and docs/validation/step4/.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.column import (PARAMS, Column, ac5ks_column, hy2seps_transport, layered_column,  # noqa: E402
                          wall_heat_capacity)
from h2psa.cycle import CycleRunner, experiment_cycle, psa_model  # noqa: E402
from h2psa.isotherms import R, load_yaml  # noqa: E402
from h2psa.transport import Gas, coefficients  # noqa: E402

OUT = ROOT / "results" / "step4"
CASES_DIR = OUT / "cases"
DOCS = ROOT / "docs" / "validation"
N = 30
SPECIES = ["H2", "CO2", "CH4", "CO", "N2"]
# ---- pre-declared gates (step-4 request; not to be changed after seeing results)
GATE = dict(closure=1e-3, grid_rec_pp=0.3, grid_ppm_rel=0.10, limit_rec_pp=0.1, limit_ppm_rel=0.02, corr_factor=1.5,
            purity_ppm=100.0, yang_rec=(60.0, 80.0))
# ---- trend scans (declared before running): AC fraction, feed time, feed flow; both Dax correlations
F_AC = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)          # multiples of 1/N and 1/(2N): interfaces on cell faces
T_FEED = (60.0, 80.0, 101.0, 125.0)             # s (101 s = the experiment, Table 5.4)
FLOW = (0.8, 1.2)                               # x the experiment's feed flow, f_ac = 0.6, t_feed = 101 s
DAX = {"corr": "eq5.20", "corr318": "eq3.18"}   # corr = pre-declared (eqs 5.17-5.35); corr318 = ch.3 eq. 3.18
F_REF = 0.6                                     # layered design used for the grid test and trends a, b, d
STEP3_BASELINE = ROOT / "results" / "step3" / "cases" / "cor_exp.pkl"


# ---------------------------------------------------------------------------- configurations
def compact_geometry():
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    e = cc["assumed"]["wall_thickness"]["value"]
    return dict(L=cc["column"]["bed_height"], d_wi=cc["column"]["internal_diameter"], e_wall=e, Cw=wall_heat_capacity(e))


def yang_setup(u_cm_s):
    """Yang (2008) geometry and feed with the declared single-column emulation (`assumed_step4`)."""
    y = load_yaml(PARAMS / "layered_4bed_yang2008.yaml")
    a, col = y["assumed_step4"], y["column"]
    P = y["feed"]["adsorption_pressure_atm"] * 101325.0
    T = a["T_feed_K"]
    d = col["inside_diameter_m"]
    flow = u_cm_s / 100 * np.pi * d**2 / 4 * P / (R * T)
    yf = y["feed"]["composition"]
    t = a["times_s"]
    P_low = a["P_low_bar"] * 1e5
    P_inter = [P - k * (P - P_low) / 3 for k in (1, 2)]
    feed_h2 = flow * yf["H2"] * t["Feed"]
    cyc = replace(experiment_cycle(), kind="Y", t_feed=t["Feed"], t_blow=t["Blow"], t_purge=t["Purge"], t_eq=t["ED"],
                  t_press=t["Press"], P_high=P, P_low=P_low, P_inter=P_inter, T_feed=T, y_feed=yf, feed_flow=flow,
                  purge_flow=a["purge_to_feed_h2"] * feed_h2 / t["Purge"])
    L = sum(l["length_m"] for l in col["layers"])
    e = a["wall_thickness_m"]
    geo = dict(L=L, d_wi=d, e_wall=e, Cw=col["wall_density"] * col["wall_heat_capacity"] * np.pi * e * (d + e))
    return cyc, geo, col["layers"][0]["length_m"] / L


def cases():
    out = {}
    for tag in DAX:
        for f in F_AC:
            for tf in T_FEED:
                out[f"{tag}_f{f:.1f}_t{tf:.0f}"] = dict(dax=tag, f_ac=f, t_feed=tf)
        for q in FLOW:
            out[f"{tag}_f{F_REF:.1f}_t101_flow{q:.1f}"] = dict(dax=tag, f_ac=F_REF, t_feed=101.0, flow=q)
        out[f"{tag}_f{F_REF:.1f}_t101_grid2N"] = dict(dax=tag, f_ac=F_REF, t_feed=101.0, N=2 * N)
        for u in load_yaml(PARAMS / "layered_4bed_yang2008.yaml")["assumed_step4"]["velocities_cm_s"]:
            out[f"{tag}_yang_u{u:.2f}"] = dict(dax=tag, yang=u, N=34)          # 1.00 m | 0.70 m on a face of dz = 5 cm
    out["limit_ac5ks"] = dict(ac5ks=1)          # gate 2: AC5-KS through the layered representation
    out["limit_ac5ks_split"] = dict(ac5ks=2)    # ... and split into two identical AC5-KS layers (0.6 | 0.4)
    return out


def build(spec):
    if "yang" in spec:
        cyc, geo, f = yang_setup(spec["yang"])
    else:
        cyc = experiment_cycle()
        cyc = replace(cyc, t_feed=spec.get("t_feed", cyc.t_feed), feed_flow=cyc.feed_flow * spec.get("flow", 1.0))
        geo, f = compact_geometry(), spec.get("f_ac")
    sp = list(cyc.y_feed) + ["He"]
    if "ac5ks" in spec:                         # AC5-KS with Table 5.9 group 1-4 carried on the layer itself
        tr = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")["psa_transport"]["groups"]["1-4"]
        base = ac5ks_column(tr, cyc.T_feed)
        l = replace(base.layers[0], Dax=tr["Dax"], lam=tr["lambda"], hw=tr["hw"], U=tr["U"])
        L = l.length
        layers = [l] if spec["ac5ks"] == 1 else [replace(l, length=0.6 * L), replace(l, length=0.4 * L)]
        col = replace(base, layers=layers)
    else:
        col = layered_column(f, cyc.y_feed, cyc.T_feed, cyc.P_high, cyc.feed_flow, sp, T_amb=cyc.T_feed,
                             dax=DAX[spec["dax"]], **geo)
    m = psa_model(cyc, N=spec.get("N", N), column=col)
    w = float((m.A * m.dz * (1 - m.eps) * m.rho_p).sum())
    return cyc, m, w


def run_case(name, spec):
    cyc, m, w = build(spec)
    run = CycleRunner(m, cyc, w_ads=w)
    log = lambda s: print(f"[{name}] {s}", flush=True)
    res = run.run_to_css(log=log)
    res.update(name=name, spec=spec, species=m.sp, cycle=cyc, N=m.N, w_ads=w, layer_of_cell=m.layer_of_cell,
               z=m.z, dz=m.dz)
    with open(CASES_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    log(f"done: CSS {res['css']} after {res['cycles']} cycles, {res['runtime_s'] / 60:.1f} min wall, "
        f"{res['cpu_s'] / 60:.1f} min CPU")
    return name


def load_case(name):
    with open(CASES_DIR / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------- gate 3: correlations
def correlation_check():
    """Recompute Dax, lambda, kf, hf, hw, U, Dp for the compact-unit AC5-KS runs (Tables 5.7 and 5.9)."""
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    ac = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")
    col, bt, asm = cc["column"], cc["breakthrough_runs"], cc["assumed"]
    std = bt["std_conditions"]
    flow = std["P_bar"] * 1e5 * bt["flow_rate_std"] / (R * std["T"])
    g = Gas(SPECIES)
    geo = dict(dp=2 * ac["physical"]["pellet_radius"], eps=col["bed_porosity"], d_wi=col["internal_diameter"],
               e_wall=asm["wall_thickness"]["value"], L=col["bed_height"], tau=ac["physical"]["tortuosity"])
    conds = [(f"Table 5.7 run {k}", r["y"], bt["feed_temperature"], bt["pressure_bar"] * 1e5, r) for k, r in bt["runs"].items()]
    grp = ac["psa_transport"]["groups"]
    for name, P, T in (("1-4", 5.0, 303.0), ("5-14", 10.0, 303.0), ("15-17", 10.0, 323.0)):
        conds.append((f"Table 5.9 group {name}", cc["feed"]["composition"], T, P * 1e5, grp[name]))
    rows = []
    for label, y, T, P, tab in conds:
        u0 = flow * R * T / (P * np.pi * col["internal_diameter"] ** 2 / 4)
        yv = [y.get(s, 0.0) for s in SPECIES]
        c = coefficients(g, yv, T, P, u0, **geo)
        c318 = coefficients(g, yv, T, P, u0, dax="eq3.18", **geo)
        row = dict(case=label, P_bar=P / 1e5, T_K=T, Re=c["Re"])
        for k, lab in (("Dax", "Dax (5.20)"), ("lambda", "lambda (5.21)"), ("kf", "kf (5.23)"), ("hf", "hf (5.22)"),
                       ("hw", "hw (5.31)"), ("U", "U (5.29)")):
            row[lab] = c[k] / tab[k]
        row["Dax (3.18)"] = c318["Dax"] / tab["Dax"]
        for s, v in tab["Dp"].items():
            row[f"Dp {s} (5.17)"] = c["Dp"][s] / v
        rows.append(row)
    df = pd.DataFrame(rows)
    ratio_cols = [c for c in df.columns if "(" in c]
    summ = []
    for c in ratio_cols:
        v = df[c].dropna()
        summ.append(dict(coefficient=c, n=len(v), min_ratio=v.min(), max_ratio=v.max(),
                         result=("alternative, not gated" if c == "Dax (3.18)" else
                                 ("PASS" if ((v <= GATE["corr_factor"]) & (v >= 1 / GATE["corr_factor"])).all() else "FAIL"))))
    return df, pd.DataFrame(summ)


def hy2seps_coefficients():
    """Transport coefficients actually used for the HY2SEPS layers (experiment feed, 5 bar, 303 K)."""
    cyc, geo = experiment_cycle(), compact_geometry()
    u0 = cyc.feed_flow * R * cyc.T_feed / (cyc.P_high * np.pi * geo["d_wi"] ** 2 / 4)
    rows = []
    for kind in ("AC", "ZEO"):
        for tag, dax in DAX.items():
            t = hy2seps_transport(kind, cyc.y_feed, cyc.T_feed, cyc.P_high, u0, geo["d_wi"], geo["e_wall"], geo["L"],
                                  SPECIES + ["He"], dax)
            rows.append(dict(layer=kind, dax_rule=dax, Dax=t["Dax"], **{"lambda": t["lambda"]}, kf=t["kf"], hf=t["hf"],
                             hw=t["hw"], U=t["U"], **{f"Dp {s}": t["Dp"][s] for s in SPECIES}, Re=t["Re"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- analysis helpers
def perf_row(name, r):
    p = r["perf"]
    return dict(case=name, **{k: r["spec"].get(k) for k in ("dax", "f_ac", "t_feed", "flow", "yang", "N")},
                recovery=p["recovery_pct"], ppm=p["ppm"], purity=p["purity_pct"], productivity=p["productivity"],
                **{f"{s}_ppm": p[f"imp_{s}_ppm"] for s in ("CO2", "CH4", "CO", "N2") if f"imp_{s}_ppm" in p},
                cycles=r["cycles"], css=r["css"], retries=r.get("retries", 0), wall_min=r["runtime_s"] / 60, cpu_min=r["cpu_s"] / 60)


def interp_ppm_at_recovery(g, rec):
    """ln(ppm) interpolated at recovery `rec` along a feed-time scan (sorted by recovery); nan outside."""
    g = g.sort_values("recovery")
    if not (g.recovery.min() <= rec <= g.recovery.max()):
        return np.nan
    return float(np.exp(np.interp(rec, g.recovery, np.log(np.maximum(g.ppm, 1e-6)))))


def recovery_at_ppm(g, ppm):
    """Recovery at impurity `ppm` along a feed-time scan (ln ppm vs recovery, monotone part); nan if not reached."""
    g = g.sort_values("t_feed")
    lp, rec = np.log(np.maximum(g.ppm.values, 1e-6)), g.recovery.values
    if lp.min() > np.log(ppm):
        return np.nan                                         # target never reached
    if lp.max() < np.log(ppm):
        return float(rec.max())                               # reached at every feed time: best recovery in the scan
    k = np.flatnonzero((lp[:-1] <= np.log(ppm)) & (lp[1:] > np.log(ppm)))
    if not len(k):
        return np.nan
    k = k[-1]
    return float(np.interp(np.log(ppm), lp[k:k + 2], rec[k:k + 2]))


def trends(T, tag):
    """Gate 4 a-e for one Dax rule. T: table of all cases (perf_row)."""
    S = T[(T.dax == tag) & T.yang.isna() & T.flow.isna() & (T.N.isna())]
    rows = []
    # a. equal recovery: every layered fraction (0.2-0.8) has a lower impurity than AC only
    ac = S[S.f_ac == 1.0]
    det = []
    ok_a = True
    for f in (0.8, 0.6, 0.4, 0.2):
        g = S[S.f_ac == f]
        lo, hi = max(ac.recovery.min(), g.recovery.min()), min(ac.recovery.max(), g.recovery.max())
        if lo > hi:
            det.append(f"f_AC {f}: no common recovery range")
            ok_a = False
            continue
        rec = 0.5 * (lo + hi)
        pa, pl = interp_ppm_at_recovery(ac, rec), interp_ppm_at_recovery(g, rec)
        det.append(f"f_AC {f}: {pl:.3g} vs AC only {pa:.3g} ppm at {rec:.1f} %")
        ok_a &= pl < pa
    rows.append(dict(trend="a. layered purer than AC only at equal recovery", result="reproduced" if ok_a else "not reproduced",
                     detail="; ".join(det)))
    # b. limiting impurity in AC only is N2 or CO (t_feed 101 s); f_AC = 0.6 reduces it
    a101 = S[(S.f_ac == 1.0) & (S.t_feed == 101.0)].iloc[0]
    l101 = S[(S.f_ac == F_REF) & (S.t_feed == 101.0)].iloc[0]
    imp = {s: a101[f"{s}_ppm"] for s in ("CO2", "CH4", "CO", "N2")}
    lim = max(imp, key=imp.get)
    ok_b = lim in ("N2", "CO") and l101[f"{lim}_ppm"] < a101[f"{lim}_ppm"]
    rows.append(dict(trend="b. limiting impurity of AC only is N2 or CO; zeolite reduces it",
                     result="reproduced" if ok_b else "not reproduced",
                     detail=f"AC only (t_feed 101 s): " + ", ".join(f"{s} {v:.3g}" for s, v in imp.items())
                     + f" ppm -> limiting {lim}; f_AC {F_REF}: {lim} {l101[f'{lim}_ppm']:.3g} ppm"))
    # c. interior optimum of recovery at 99.99 % purity
    rc = {f: recovery_at_ppm(S[S.f_ac == f], GATE["purity_ppm"]) for f in F_AC}
    reach = {f: v for f, v in rc.items() if np.isfinite(v)}
    if not reach:
        ok_c, txt = False, "no fraction reaches 99.99 % in the feed-time scan (not evaluable)"
    else:
        fbest = max(reach, key=reach.get)
        ends = [rc[1.0], rc[0.0]]
        ok_c = 0.0 < fbest < 1.0 and all(not np.isfinite(e) or reach[fbest] > e for e in ends)
        txt = "; ".join(f"f_AC {f}: {'not reached' if not np.isfinite(v) else f'{v:.1f} %'}" for f, v in rc.items())
    rows.append(dict(trend="c. interior optimum zeolite fraction (recovery at 99.99 %)", result="reproduced" if ok_c else "not reproduced",
                     detail=txt))
    # d. longer feed / higher velocity: lower purity, higher recovery (every fraction; flow at f_AC 0.6)
    bad = []
    for f in F_AC:
        g = S[S.f_ac == f].sort_values("t_feed")
        if not (np.all(np.diff(g.ppm) > 0) and np.all(np.diff(g.recovery) > 0)):
            bad.append(f"f_AC {f}")
    Fq = T[(T.dax == tag) & T.flow.notna()].sort_values("flow")
    base = S[(S.f_ac == F_REF) & (S.t_feed == 101.0)]
    seq = pd.concat([Fq[Fq.flow < 1], base, Fq[Fq.flow > 1]])
    flow_ok = len(seq) == 3 and np.all(np.diff(seq.ppm) > 0) and np.all(np.diff(seq.recovery) > 0)
    rows.append(dict(trend="d. longer feed / higher feed velocity: lower purity, higher recovery",
                     result="reproduced" if not bad and flow_ok else "not reproduced",
                     detail=("feed time monotone for all fractions" if not bad else "feed time not monotone: " + ", ".join(bad))
                     + f"; feed flow x0.8/1.0/1.2 (f_AC {F_REF}): ppm " + "/".join(f"{v:.3g}" for v in seq.ppm)
                     + ", recovery " + "/".join(f"{v:.1f}" for v in seq.recovery)))
    # e. Yang geometry and feed
    Y = T[(T.dax == tag) & T.yang.notna()].sort_values("yang")
    hit = Y[(Y.ppm <= GATE["purity_ppm"]) & (Y.recovery >= GATE["yang_rec"][0]) & (Y.recovery <= GATE["yang_rec"][1])]
    rows.append(dict(trend="e. Yang 2008: >= 99.99 % with 60-80 % recovery at 3.1-5.2 cm/s", result="reproduced" if len(hit) else "not reproduced",
                     detail="; ".join(f"{u:.2f} cm/s: {p:.3g} ppm, {r:.1f} %" for u, p, r in zip(Y.yang, Y.ppm, Y.recovery))))
    return pd.DataFrame(rows), rc


# ---------------------------------------------------------------------------- figures
def fig_scan(T, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, tag in zip(axs, DAX):
        S = T[(T.dax == tag) & T.yang.isna() & T.flow.isna() & T.N.isna()]
        for i, f in enumerate(F_AC):
            g = S[S.f_ac == f].sort_values("t_feed")
            ax.plot(g.recovery, g.ppm, "o-", color=cmap(i / (len(F_AC) - 1)), label=f"f_AC = {f:.1f}")
        ax.axhline(GATE["purity_ppm"], color="#d23f3f", ls="--", lw=1)
        ax.set(yscale="log", xlabel="H2 recovery [%]", title=f"Dax {DAX[tag]} (feed time 60-125 s)")
    axs[0].set_ylabel("product impurity [ppm]")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_profiles(r, path):
    pf = r["profiles"][max(r["profiles"])]
    sp, z = r["species"], r["z"]
    zi = r["dz"] * np.flatnonzero(np.diff(r["layer_of_cell"]))[0] + r["dz"] if len(set(r["layer_of_cell"])) > 1 else None
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, s in zip(axs, ("CO2", "CO", "N2")):
        for st, cl in (("Feed", "#d23f3f"), ("Blow", "#7a7a7a"), ("Purge", "#2a78d6"), ("Press", "#1baf7a")):
            ax.plot(z, pf[st]["C"][sp.index(s)], color=cl, label=f"end of {st}")
        if zi:
            ax.axvline(zi, color="0.3", ls=":", lw=1)
        ax.set(yscale="symlog", xlabel="z [m] (feed end at 0)", title=f"gas-phase {s} [mol/m3]")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{r['name']}: CSS profiles (dotted: AC | ZEO interface)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
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
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def ok(b):
    return "PASS" if b else "FAIL"


def analyse():
    names = list(cases())
    R_ = {n: load_case(n) for n in names if (CASES_DIR / f"{n}.pkl").exists()}
    t = {}
    T = pd.DataFrame([perf_row(n, r) for n, r in R_.items()])
    t["all"] = T
    # gate 1
    g1 = [dict(check="CSS reached (step-3 criterion), all cases", value=f"{sum(r['css'] for r in R_.values())}/{len(R_)}",
               limit="all", result=ok(all(r["css"] for r in R_.values())))]
    clo = max(abs(pd.DataFrame(r["history"]).closure).max() for r in R_.values())
    g1.append(dict(check="per-cycle mole balance, worst cycle of all cases", value=f"{clo:.1e}", limit="< 1e-3",
                   result=ok(clo < GATE["closure"])))
    for tag in DAX:
        a, b = f"{tag}_f{F_REF:.1f}_t101", f"{tag}_f{F_REF:.1f}_t101_grid2N"
        if a in R_ and b in R_:
            p1, p2 = R_[a]["perf"], R_[b]["perf"]
            drec, dppm = abs(p2["recovery_pct"] - p1["recovery_pct"]), abs(p2["ppm"] / p1["ppm"] - 1)
            g1 += [dict(check=f"grid N {N} -> {2 * N}, f_AC {F_REF}, Dax {DAX[tag]}: recovery", value=f"{drec:.3f} pp",
                        limit="< 0.3 pp", result=ok(drec < GATE["grid_rec_pp"])),
                   dict(check=f"grid N {N} -> {2 * N}, f_AC {F_REF}, Dax {DAX[tag]}: impurity", value=f"{100 * dppm:.1f} %",
                        limit="< 10 %", result=ok(dppm < GATE["grid_ppm_rel"]))]
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_layered.py")],
                        capture_output=True, text=True, cwd=ROOT)
    last = pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else pt.stderr[-200:]
    g1.append(dict(check="interface tests (tests/test_layered.py)", value=last, limit="all pass", result=ok(pt.returncode == 0)))
    t["gate1"] = pd.DataFrame(g1)
    nc = []                                     # cases stopped at max_cycles: drift and geometric extrapolation
    for n, r in R_.items():
        if not r["css"]:
            H = pd.DataFrame(r["history"])
            p, keys = H.ppm.values, ["ppm", "recovery_pct"] + [k for k in H if k.startswith("q_")]
            chg = (H[keys].iloc[-1] - H[keys].iloc[-11]).abs() / H[keys].iloc[-1].abs()     # run_to_css rule
            d1, d2 = p[-501] - p[-1001], p[-1] - p[-501]
            rho = d2 / d1 if d1 else np.nan
            rest = d2 * rho / (1 - rho) if 0 < rho < 1 else np.nan
            nc.append(dict(case=n, cycles=r["cycles"], ppm=p[-1], css_change=chg.max(), css_change_by=chg.idxmax(),
                           ppm_drift_last_500=d2, ppm_extrapolated=p[-1] + rest, extrapolated_rel=rest / p[-1]))
    t["gate1_noncss"] = pd.DataFrame(nc)
    # gate 2
    g2 = []
    if STEP3_BASELINE.exists():
        with open(STEP3_BASELINE, "rb") as fh:
            ref = pickle.load(fh)["perf"]
        for n in ("limit_ac5ks", "limit_ac5ks_split"):
            if n in R_:
                p = R_[n]["perf"]
                drec, dppm = p["recovery_pct"] - ref["recovery_pct"], p["ppm"] / ref["ppm"] - 1
                g2.append(dict(case=n, recovery=p["recovery_pct"], recovery_step3=ref["recovery_pct"], d_rec_pp=drec,
                               ppm=p["ppm"], ppm_step3=ref["ppm"], d_ppm_rel=dppm,
                               result=ok(abs(drec) <= GATE["limit_rec_pp"] and abs(dppm) <= GATE["limit_ppm_rel"])))
    t["gate2"] = pd.DataFrame(g2)
    # gate 3
    t["corr_ratios"], t["gate3"] = correlation_check()
    t["hy2seps_coefficients"] = hy2seps_coefficients()
    # gate 4
    for tag in DAX:
        if (T.dax == tag).sum() >= len(F_AC) * len(T_FEED):
            t[f"trends_{tag}"], t[f"rec_at_purity_{tag}"] = trends(T, tag)
    yh = []                                     # trend e diagnostic: where the feed H2 goes (per mol H2 fed)
    for n, r in R_.items():
        if "yang" in r["spec"]:
            sm, i = r["step_moles"], r["species"].index("H2")
            f = sm["Feed"][0][i]
            yh.append(dict(case=n, product=sm["Feed"][1][i] / f,
                           ED_to_partner=sum(v[1][i] for k, v in sm.items() if k.startswith("ED")) / f,
                           blow_purge_out=-(sm["Blow"][0][i] + sm["Purge"][0][i]) / f,
                           purge_in=-sm["Purge"][1][i] / f, press_in=-sm["Press"][1][i] / f,
                           recovery=r["perf"]["recovery_pct"] / 100))
    t["yang_h2"] = pd.DataFrame(yh).sort_values("case") if yh else pd.DataFrame()
    return R_, t


SUMMARY = """**Status: stopped for a decision (gates 1, 3 and 4 fail; none of the causes found is a coding bug).**

* **Gate 2 passes exactly** (AC5-KS through the layered code = step-3 baseline, also split into two layers), the
  mole balance closes to 1e-6, the grid test and the interface tests pass: the layered implementation is verified.
* **Gate 1 fails on CSS only:** 4 of 64 cases stop at 2000 cycles, all with Dax eq. 3.18, low AC fraction or long
  feed, and impurity 570-1900 ppm (far from 100 ppm). The drift is monotone and geometric; extrapolated it adds
  0.7-4.2 % to their ppm (table below), which changes no trend verdict. Cause: the bed-averaged CO2 loading is still
  rising (slow CO2 uptake in the zeolite layer), not a numerical fault. The same physics makes CSS slow near the constraint (f_AC 0.2, 26 ppm:
  1144 cycles, 50 min), so the <= 10 min runtime target holds for the median (7.1 min) but not for every case.
  The two 3.14 cm/s Yang cases first crashed on a BDF step-size underflow; with the per-step rtol retry
  (`CycleRunner._solve`, 1-3 steps re-solved) they reach CSS.
* **Gate 3 fails on Dax, h_w and U; not a coding bug.** Eq. 5.20 is coded as printed (checked against p.172) and
  gives 51-55x the tabulated Dax at every condition, including Table 5.7 run 3 (Re = 6.6, inside the validity range
  3 < Re < 10000 stated on p.172); the PSA groups (Re 2.0-2.1) and the HY2SEPS layers (Re 1.2-1.7) lie below that
  range. The ch.3 correlation (eq. 3.18) reproduces the tabulated Dax within 1.15-1.24x, so the author's tables
  are not eq. 5.20 output. The tabulated
  h_w = 55 and U = 50 W/m2/K are identical in all runs and groups (set values, not correlation output).
  Dp, lambda, kf and hf pass (0.96-1.21).
* **Gate 4 depends on Dax.** With the pre-declared eq. 5.20 the bed is dispersion-dominated: no case gets below
  1.4e4 ppm at the Lopes feed, so a, b, c fail (d holds). With eq. 3.18: a, c, d reproduced (zeolite lowers impurity
  at equal recovery; interior optimum f_AC 0.4, about 3.6 pp more recovery at 99.99 % than AC only), b not (the AC-only
  limiting impurity is CH4, not N2 or CO; HY2SEPS-AC is not the Calgon carbon of the literature), e not (99.99 %
  is reached at 3.1-5.2 cm/s but with 26-55 % recovery: in the declared single-column emulation, pressurisation
  with product takes 42-70 % of the feed H2 and the bed is under-used, 0.1-80 ppm; see the H2 split below).

**Options for the project lead**

1. Dax for the HY2SEPS layers: (A, recommended) adopt eq. 3.18 as the baseline: it reproduces the author's tables
   (eq. 5.20 does not, and the PSA conditions are outside its stated Re range); eq. 5.20 becomes a scenario carried to step 7 (it moves
   impurity by two orders of magnitude, the same kind of model-form uncertainty as the EP gas). (B) Keep eq. 5.20:
   the compact unit never reaches 99.99 % and step 6 has no active constraint.
2. h_w and U: (A, recommended) use the Table 5.9 set values (55, 50 W/m2/K) for every layer of the compact column,
   since they belong to the apparatus and the step-3 baseline uses them; this needs a rerun of the eq. 3.18 scan
   (several hours on 7 workers). (B) Keep the correlation values (h_w 333, U 3.4 W/m2/K) and report the bias.
3. CSS: (A) accept the 4 capped cases with their extrapolated drift (all > 500 ppm). (B) Raise the cap to 5000
   cycles for them (about 4 h each). (C) Add CSS acceleration before step 6, where slow CSS near the constraint sets
   the cost of every training point.
4. Trends b and e stay recorded as not reproduced (no tuning). Optional diagnostic for e only: rerun the Yang cases
   with a declared longer feed time, stated beforehand as a different emulation, not a fit."""

CORR_TEXT = """Reading of gate 3. Eq. 5.20 is coded as printed on p.172 (eps Dax / Dm = 20 + 0.5 Sc Re). The text there
gives its validity as 3 < Re < 10000; the compact-unit conditions have Re = 1.4-6.6 (Table 5.7) and 2.0-2.1
(Table 5.9), the HY2SEPS layers 1.2-1.7. At these Re the constant 20 dominates (0.5 Sc Re < 4) and Dax is about
21 Dm / eps, i.e. 51-55x the tabulated values, also for run 3 inside the stated range; the ch.3 correlation
(eq. 3.18) reproduces them within 1.15-1.24x. h_w and U in
Tables 5.7 / 5.9 do not vary with flow, pressure or temperature (55 and 50 W/m2/K throughout), so they are set
values and the correlations cannot be expected to reproduce them. Per the pre-declared rule the HY2SEPS layers use
the correlations anyway (eq. 5.20 Dax, eq. 5.31 h_w, eq. 5.29 U); expected bias: Dax about 50x too high relative
to what the author would have used, h_w about 6x higher and U about 15x lower than the apparatus values (a more
adiabatic column). Eq. 3.18 is carried as the sensitivity."""


def write_report(R_, t):
    T = t["all"]
    f_all = {"recovery": "{:.2f}", "ppm": "{:.3g}", "purity": "{:.5f}", "productivity": "{:.1f}", "CO2_ppm": "{:.3g}",
             "CH4_ppm": "{:.3g}", "CO_ppm": "{:.3g}", "N2_ppm": "{:.3g}", "wall_min": "{:.1f}", "cpu_min": "{:.1f}",
             "f_ac": "{:.1f}", "t_feed": "{:.0f}", "flow": "{:.1f}", "yang": "{:.2f}"}
    lay = T[T.yang.isna() & T.case.str.startswith(("corr_", "corr318_"))]
    rt = lay.wall_min
    tr = {tag: t.get(f"trends_{tag}") for tag in DAX}
    text = f"""# Step 4 — layered bed (HY2SEPS-AC + HY2SEPS-ZEO)

Generated by `scripts/run_layered.py` (cases cached in `results/step4/cases/`). Thesis pages are printed pages.

## Verdict

| gate | result |
|---|---|
| 1. Numerical (CSS, mole balance, grid N -> 2N, interface) | {ok(len(t["gate1"]) and (t["gate1"].result == "PASS").all())} |
| 2. Limit case: AC5-KS through the layered code = frozen step-3 baseline | {ok(len(t["gate2"]) and (t["gate2"].result == "PASS").all())} |
| 3. Correlations vs Tables 5.7 / 5.9 (factor 1.5) | {ok((t["gate3"][t["gate3"].result != "alternative, not gated"].result == "PASS").all())} |
| 4. Qualitative trends a-e, Dax eq. 5.20 (pre-declared correlations) | {"; ".join(f"{r.trend[:2]} {r.result}" for _, r in tr["corr"].iterrows()) if tr["corr"] is not None else "–"} |
| 4'. Same trends with Dax eq. 3.18 (sensitivity) | {"; ".join(f"{r.trend[:2]} {r.result}" for _, r in tr["corr318"].iterrows()) if tr["corr318"] is not None else "–"} |
| Runtime per CSS, layered bed (target <= 10 min wall) | median {rt.median():.1f} min, max {rt.max():.1f} min |

{SUMMARY}

## Model

* Column: the step-2/3 finite-volume model with per-cell adsorbent properties. Layers carry their own bed porosity,
  particle properties, isotherm, kinetics and transport coefficients (Dax, lambda, kf, hf, hw, U, Dp); at a layer
  interface the dispersion and conduction coefficients take the harmonic mean of the two cells. Interfaces must lie
  on a cell face (checked); the face flux is single-valued, so gas concentration, temperature and flux are
  continuous by construction (tests in `tests/test_layered.py`).
* HY2SEPS-AC (feed end) and HY2SEPS-ZEO (product end): multisite Langmuir extended to mixtures (Tables 2.7, 2.8);
  micropore/crystal Dc/rc2 refitted to the measured values (Tables 3.12, 3.15; errata E4, E5 convention); particle
  properties of Tables 3.10 / 3.13; bed porosity of the ch.3 breakthrough columns (0.394 / 0.391, assumed to hold in
  the compact column); LDF factors Omega_m = 8 (cylindrical extrudates, eq. 3.17), Omega_c = 3 (AC, p.98) and 15
  (ZEO, p.104); tortuosity 2 (pp.96, 102). Solid heat capacities are not reported: 880 (AC) and 920 J/kg/K (ZEO),
  assumed.
* Transport: thesis correlations at the feed conditions, per layer (`src/h2psa/transport.py`). The macropore
  diffusivities quoted on pp.97/103 are molecular diffusion in He at 1 bar and 303 K divided by tau = 2
  (Chapman-Enskog reproduces them within 3-14 %, `tests/test_transport.py`); at PSA conditions Dp follows from
  eq. 5.17 with the same tau (Knudsen neglected as on p.96).
* Cycle: the frozen step-3 baseline (cycle B of the experiment, beta_P = 0.102 s-1, EP = pure H2, purge
  0.261 mmol/s), compact column (0.267 m x 5 cm), Lopes feed at 5 bar and 303 K; feed time and feed flow varied for
  the trends. Productivity uses the adsorbent mass of the layered bed.

## Gate 1 — numerical

{md(t["gate1"])}

Cases stopped at 2000 cycles without meeting the CSS criterion (relative change < 1e-4 over 10 cycles). The drift is
monotone and decays geometrically (no oscillation, mole balance closed); the extrapolation sums the remaining
geometric series of the per-500-cycle ppm increments:

{md(t["gate1_noncss"], {"ppm": "{:.1f}", "css_change": "{:.1e}", "ppm_drift_last_500": "{:.1f}", "ppm_extrapolated": "{:.0f}", "extrapolated_rel": "{:+.1%}"}) if len(t["gate1_noncss"]) else "none"}

## Gate 2 — limit case

AC5-KS (Table 5.9 group 1-4 transport carried on the layer, adsorbent mass from the bed) as one layer and split into
two identical layers (0.6 | 0.4), against the frozen step-3 baseline (`results/step3/cases/cor_exp.pkl`):

{md(t["gate2"], {"recovery": "{:.3f}", "recovery_step3": "{:.3f}", "d_rec_pp": "{:+.4f}", "ppm": "{:.2f}", "ppm_step3": "{:.2f}", "d_ppm_rel": "{:+.2%}"}) if len(t["gate2"]) else "not run"}

## Gate 3 — correlations vs the author's tables (AC5-KS, compact unit)

Ratio recomputed / tabulated (Table 5.7: breakthrough runs at 5 bar, 323 K; Table 5.9: PSA groups; flow
5.0e-5 m3/s at 298 K and 1 bar; scalar coefficients use the mole-fraction mean of the species' mixture
diffusivities, eq. 5.18; U with T_w - T_inf = 10 K, air properties at 300 K, AISI 304 wall):

{md(t["gate3"], {"min_ratio": "{:.2f}", "max_ratio": "{:.2f}"})}

{md(t["corr_ratios"], {c: "{:.2f}" for c in t["corr_ratios"].columns if c not in ("case",)})}

{CORR_TEXT}

Coefficients used for the HY2SEPS layers (experiment feed, 5 bar, 303 K, compact column):

{md(t["hy2seps_coefficients"], {c: "{:.3g}" for c in t["hy2seps_coefficients"].columns if c not in ("layer", "dax_rule")})}

## Gate 4 — qualitative trends (no tuning)

Declared before running: a — every layered fraction 0.2-0.8 has a lower impurity than AC only at the midpoint of
the common recovery range of the feed-time scans; b — the largest impurity of AC only (t_feed 101 s) is N2 or CO
and is lower at f_AC = {F_REF}; c — recovery at 99.99 % purity (interpolated in ln ppm along the feed-time scan) is
highest at an interior fraction and higher than at both ends (an end that never reaches 99.99 % counts as worse);
d — impurity and recovery both rise monotonically with feed time for every fraction, and with feed flow
(x0.8, 1, 1.2) at f_AC = {F_REF}; e — Yang geometry/feed: at least one velocity gives <= 100 ppm with 60-80 %
recovery.

### Dax eq. 5.20 (pre-declared correlation set)

{md(tr["corr"]) if tr["corr"] is not None else "not run"}

### Dax eq. 3.18 (sensitivity: the ch.3 correlation that reproduces Tables 5.7/5.9)

{md(tr["corr318"]) if tr["corr318"] is not None else "not run"}

Trend e diagnostic — where the feed H2 goes in the Yang emulation (mol per mol H2 fed, CSS cycle; purge_in and
press_in are product H2 returned to the column):

{md(t["yang_h2"], {c: "{:.3f}" for c in t["yang_h2"].columns if c != "case"}) if len(t["yang_h2"]) else "not run"}

![feed-time scans](step4_scan.png)

![profiles](profiles_corr318.png)

![profiles](profiles_corr.png)

### All cases

{md(T.drop(columns=["css"]), f_all)}

## Runtime

Layered-bed CSS: median {rt.median():.1f} min wall ({lay.cpu_min.median():.1f} min CPU), max {rt.max():.1f} min, on a shared
machine. Speed-up done in this step: multisite Langmuir solved by Newton in ln(s) with warm start (about 5x per
cycle).

## Step-6 preparation (draft, not run): `configs/design_space.yaml`

```yaml
{(ROOT / "configs" / "design_space.yaml").read_text().strip()}
```
"""
    (OUT / "REPORT.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    todo = {n: s for n, s in cases().items()
            if (not a.only or n in a.only.split(",")) and (a.force or not (CASES_DIR / f"{n}.pkl").exists())}
    if todo and not a.report_only:
        order = sorted(todo, key=lambda n: ("grid2N" not in n and "yang" not in n, n))     # longest first
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(run_case, n, todo[n]): n for n in order}
            for f in as_completed(futs):
                if f.exception() is not None:
                    print(f"[{futs[f]}] FAILED: {f.exception()!r}", flush=True)
    R_, t = analyse()
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(OUT / f"{k}.csv", index=False)
    if "trends_corr" in t or "trends_corr318" in t:
        fig_scan(t["all"], OUT / "step4_scan.png")
    for tag in DAX:
        n = f"{tag}_f{F_REF:.1f}_t101"
        if n in R_:
            fig_profiles(R_[n], OUT / f"profiles_{tag}.png")
    write_report(R_, t)
    (DOCS / "step4").mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        shutil.copy(f, DOCS / "step4" / f.name)
    text = (OUT / "REPORT.md").read_text()
    for f in OUT.glob("*.png"):
        text = text.replace(f"]({f.name})", f"](step4/{f.name})")
    (DOCS / "step4.md").write_text(text)
    print(f"wrote {OUT} and {DOCS / 'step4.md'}")


if __name__ == "__main__":
    main()
