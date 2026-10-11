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
from h2psa.transport import Gas, coefficients, overall_U  # noqa: E402

OUT = ROOT / "results" / "step4"
CASES_DIR = OUT / "cases"
DOCS = ROOT / "docs" / "validation"
N = 30
SPECIES = ["H2", "CO2", "CH4", "CO", "N2"]
# ---- pre-declared gates (step-4 request; not to be changed after seeing results)
GATE = dict(closure=1e-3, grid_rec_pp=0.3, grid_ppm_rel=0.10, limit_rec_pp=0.1, limit_ppm_rel=0.02, corr_factor=1.5,
            purity_ppm=100.0, yang_rec=(60.0, 80.0), accel_ppm_rel=1e-3, accel_rec_pp=0.05)
TOL_PPM = 1e-5    # CSS: impurity ppm change over the final 10 plain cycles, tightened after the acceleration check
# ---- trend scans (declared before running): AC fraction, feed time, feed flow; both Dax correlations
F_AC = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)          # multiples of 1/N and 1/(2N): interfaces on cell faces
T_FEED = (60.0, 80.0, 101.0, 125.0)             # s (101 s = the experiment, Table 5.4)
FLOW = (0.8, 1.2)                               # x the experiment's feed flow, f_ac = 0.6, t_feed = 101 s
# first step-4 run (Dax eq. 3.18 and eq. 5.20, correlation walls, brute-force CSS), commit b23e20a: brute-force
# references for the CSS-acceleration check
CASES_V1 = OUT / "cases_v1"
V1_CHECK = {"f1.0_t80": (1.0, 80.0), "f1.0_t101": (1.0, 101.0), "f1.0_t125": (1.0, 125.0), "f0.8_t80": (0.8, 80.0),
            "f0.8_t101": (0.8, 101.0), "f0.8_t125": (0.8, 125.0), "f0.6_t80": (0.6, 80.0), "f0.6_t101": (0.6, 101.0),
            "f0.4_t80": (0.4, 80.0), "f0.4_t101": (0.4, 101.0), "f0.2_t80": (0.2, 80.0), "f0.0_t60": (0.0, 60.0)}
V1_CAPPED = {"f0.6_t125": (0.6, 125.0, 1.0), "f0.4_t125": (0.4, 125.0, 1.0), "f0.2_t101": (0.2, 101.0, 1.0),
             "f0.6_t101_flow1.2": (0.6, 101.0, 1.2)}
ACCEL_DIR = OUT / "accel_check"
ACCEL_TOL4, CASES_TOL4 = OUT / "accel_check_tol4", OUT / "cases_tol4"   # before the ppm condition (TOL_PPM)
EXT_DIR = OUT / "cases_v1_ext"                  # brute-force references continued with a 10x tighter criterion
STATES = OUT / "css_states"                     # converged CSS states of this run (warm starts)
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
    for f in F_AC:
        for tf in T_FEED:
            out[f"base_f{f:.1f}_t{tf:.0f}"] = dict(f_ac=f, t_feed=tf)
    for q in FLOW:
        out[f"base_f{F_REF:.1f}_t101_flow{q:.1f}"] = dict(f_ac=F_REF, t_feed=101.0, flow=q)
    out[f"base_f{F_REF:.1f}_t101_grid2N"] = dict(f_ac=F_REF, t_feed=101.0, N=2 * N)
    for u in load_yaml(PARAMS / "layered_4bed_yang2008.yaml")["assumed_step4"]["velocities_cm_s"]:
        out[f"base_yang_u{u:.2f}"] = dict(yang=u, N=34)          # 1.00 m | 0.70 m on a face of dz = 5 cm
    out["limit_ac5ks"] = dict(ac5ks=1)          # gate 2: AC5-KS through the layered representation
    out["limit_ac5ks_split"] = dict(ac5ks=2)    # ... and split into two identical AC5-KS layers (0.6 | 0.4)
    return out


def accel_check_cases():
    """CSS-acceleration verification: configurations of the first step-4 run (Dax eq. 3.18, correlation walls)
    whose brute-force CSS is cached in CASES_V1 (20-2000 ppm, incl. the slowest), run cold with acceleration;
    plus the 4 cases that hit the 2000-cycle cap there."""
    out = {f"v1_{k}": dict(f_ac=f, t_feed=t, wall="corr") for k, (f, t) in V1_CHECK.items()}
    out["v1_f0.6_t101_flow0.8"] = dict(f_ac=0.6, t_feed=101.0, flow=0.8, wall="corr")
    out["v1_yang_u5.20"] = dict(yang=5.2, N=34, wall="corr")
    for k, (f, t, q) in V1_CAPPED.items():
        out[f"v1_{k}"] = dict(f_ac=f, t_feed=t, flow=q, wall="corr")
    return out


def wall_values(spec):
    """(h_w, U) [W m-2 K-1] from each geometry's own source (step-4 decision): Table 5.9 for the compact column,
    Yang (2008) Table 2 for theirs (inside film 38.5, outside film 14.2 combined with the wall by eq. 5.29).
    None: the correlations (eqs 5.31, 5.29; first step-4 run)."""
    if spec.get("wall") == "corr":
        return None
    if "yang" in spec:
        y = load_yaml(PARAMS / "layered_4bed_yang2008.yaml")["column"]
        e = y["outside_diameter_m"] / 2 - y["inside_diameter_m"] / 2
        k = load_yaml(PARAMS / "gas_transport_lopes2010.yaml")["external"]["wall_conductivity"]["value"]
        return y["internal_htc"], overall_U(y["internal_htc"], y["external_htc"], y["inside_diameter_m"], e, k)
    tr = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")["psa_transport"]["groups"]["1-4"]
    return tr["hw"], tr["U"]


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
                             dax="eq3.18", wall=wall_values(spec), **geo)
    m = psa_model(cyc, N=spec.get("N", N), column=col)
    w = float((m.A * m.dz * (1 - m.eps) * m.rho_p).sum())
    return cyc, m, w


def family(spec):
    """Cases whose CSS states can seed each other: same geometry, grid and adsorbents."""
    return ("ac5ks" if "ac5ks" in spec else "yang" if "yang" in spec else "compact") + f"_N{spec.get('N', N)}"


def design(spec):
    """Scaled design variables for the nearest-neighbour warm start (ranges of configs/design_space.yaml)."""
    if "yang" in spec:
        return np.array([spec["yang"] / 5.2])
    return np.array([spec.get("f_ac", 1.0), spec.get("t_feed", 101.0) / 110.0, spec.get("flow", 1.0)])


def nearest_state(spec):
    """CSS state of the nearest converged case of the same family in STATES (None if there is none)."""
    best = None
    for fn in STATES.glob("*.npz"):
        d = np.load(fn, allow_pickle=True)
        if str(d["family"]) != family(spec):
            continue
        dist = float(np.linalg.norm(d["design"] - design(spec)))
        if best is None or dist < best[0]:
            best = (dist, fn.stem, d["y"])
    return best


def run_case(name, spec, out_dir=None, warm=True, accel=True):
    out_dir = out_dir or CASES_DIR
    cyc, m, w = build(spec)
    run = CycleRunner(m, cyc, w_ads=w)
    log = lambda s: print(f"[{name}] {s}", flush=True)
    seed = nearest_state(spec) if warm else None
    if seed:
        log(f"warm start from {seed[1]} (distance {seed[0]:.3f})")
    res = run.run_to_css(y0=None if seed is None else seed[2], log=log, accel=accel, tol_ppm=TOL_PPM)
    res.update(name=name, spec=spec, species=m.sp, cycle=cyc, N=m.N, w_ads=w, layer_of_cell=m.layer_of_cell,
               z=m.z, dz=m.dz, warm_from=None if seed is None else seed[1], accel=accel)
    with open(out_dir / f"{name}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    if res["css"] and out_dir == CASES_DIR:
        tmp = STATES / f"{name}.tmp.npz"
        np.savez(tmp, y=res["y_end"], family=family(spec), design=design(spec))
        tmp.rename(STATES / f"{name}.npz")
    log(f"done: CSS {res['css']} after {res['cycles']} cycles ({res['accelerated_cycles']} accelerated), "
        f"{res['runtime_s'] / 60:.1f} min wall, {res['cpu_s'] / 60:.1f} min CPU")
    return name


def extend_brute(name, spec):
    """Brute-force reference continued: plain cycles from the cached first-run CSS state with a 10x tighter
    criterion (relative change < 1e-5 over 10 cycles), at most 3000 more cycles."""
    cyc, m, w = build(spec)
    run = CycleRunner(m, cyc, w_ads=w)
    with open(CASES_V1 / f"corr318_{name[3:]}.pkl", "rb") as fh:
        y0 = pickle.load(fh)["y_end"]
    log = lambda s: print(f"[{name} ext] {s}", flush=True)
    res = run.run_to_css(y0=y0, tol=1e-5, max_cycles=3000, log=log)
    res.update(name=name, spec=spec)
    with open(EXT_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    log(f"done: tol 1e-5 {res['css']} after {res['cycles']} more cycles, {res['runtime_s'] / 60:.1f} min wall")
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
        c = coefficients(g, yv, T, P, u0, dax="eq5.20", **geo)
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
                         result=("PASS (adopted)" if c == "Dax (3.18)" and ((v <= GATE["corr_factor"]) & (v >= 1 / GATE["corr_factor"])).all() else
                                 "FAIL (rejected)" if c == "Dax (5.20)" else
                                 ("PASS" if ((v <= GATE["corr_factor"]) & (v >= 1 / GATE["corr_factor"])).all() else "FAIL"))))
    return df, pd.DataFrame(summ)


def hy2seps_coefficients():
    """Transport coefficients used for the HY2SEPS layers (experiment feed, 5 bar, 303 K, compact column):
    correlations with Dax eq. 3.18; h_w and U from Table 5.9 (step-4 decision)."""
    cyc, geo = experiment_cycle(), compact_geometry()
    u0 = cyc.feed_flow * R * cyc.T_feed / (cyc.P_high * np.pi * geo["d_wi"] ** 2 / 4)
    hw, U = wall_values({})
    rows = []
    for kind in ("AC", "ZEO"):
        t = hy2seps_transport(kind, cyc.y_feed, cyc.T_feed, cyc.P_high, u0, geo["d_wi"], geo["e_wall"], geo["L"],
                              SPECIES + ["He"], "eq3.18")
        rows.append(dict(layer=kind, Dax=t["Dax"], **{"lambda": t["lambda"]}, kf=t["kf"], hf=t["hf"], hw=hw, U=U,
                         **{f"Dp {s}": t["Dp"][s] for s in SPECIES}, Re=t["Re"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- analysis helpers
def perf_row(name, r):
    p = r["perf"]
    return dict(case=name, **{k: r["spec"].get(k) for k in ("f_ac", "t_feed", "flow", "yang", "N")},
                recovery=p["recovery_pct"], ppm=p["ppm"], purity=p["purity_pct"], productivity=p["productivity"],
                **{f"{s}_ppm": p[f"imp_{s}_ppm"] for s in ("CO2", "CH4", "CO", "N2") if f"imp_{s}_ppm" in p},
                cycles=r["cycles"], acc=r.get("accelerated_cycles", 0), warm=r.get("warm_from") or "–", css=r["css"],
                retries=r.get("retries", 0), wall_min=r["runtime_s"] / 60, cpu_min=r["cpu_s"] / 60)


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


def trends(T):
    """Gate 4 a-e. T: table of the baseline cases (perf_row)."""
    S = T[T.yang.isna() & T.flow.isna() & (T.N.isna())]
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
    Fq = T[T.flow.notna()].sort_values("flow")
    base = S[(S.f_ac == F_REF) & (S.t_feed == 101.0)]
    seq = pd.concat([Fq[Fq.flow < 1], base, Fq[Fq.flow > 1]])
    flow_ok = len(seq) == 3 and np.all(np.diff(seq.ppm) > 0) and np.all(np.diff(seq.recovery) > 0)
    rows.append(dict(trend="d. longer feed / higher feed velocity: lower purity, higher recovery",
                     result="reproduced" if not bad and flow_ok else "not reproduced",
                     detail=("feed time monotone for all fractions" if not bad else "feed time not monotone: " + ", ".join(bad))
                     + f"; feed flow x0.8/1.0/1.2 (f_AC {F_REF}): ppm " + "/".join(f"{v:.3g}" for v in seq.ppm)
                     + ", recovery " + "/".join(f"{v:.1f}" for v in seq.recovery)))
    # e. Yang geometry and feed
    Y = T[T.yang.notna()].sort_values("yang")
    hit = Y[(Y.ppm <= GATE["purity_ppm"]) & (Y.recovery >= GATE["yang_rec"][0]) & (Y.recovery <= GATE["yang_rec"][1])]
    rows.append(dict(trend="e. Yang 2008: >= 99.99 % with 60-80 % recovery at 3.1-5.2 cm/s", result="reproduced" if len(hit) else "not reproduced",
                     detail="; ".join(f"{u:.2f} cm/s: {p:.3g} ppm, {r:.1f} %" for u, p, r in zip(Y.yang, Y.ppm, Y.recovery))))
    return pd.DataFrame(rows), rc


# ---------------------------------------------------------------------------- figures
def fig_scan(T, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    cmap = plt.get_cmap("viridis")
    S = T[T.yang.isna() & T.flow.isna() & T.N.isna()]
    for i, f in enumerate(F_AC):
        g = S[S.f_ac == f].sort_values("t_feed")
        ax.plot(g.recovery, g.ppm, "o-", color=cmap(i / (len(F_AC) - 1)), label=f"f_AC = {f:.1f}")
    ax.axhline(GATE["purity_ppm"], color="#d23f3f", ls="--", lw=1)
    ax.set(yscale="log", xlabel="H2 recovery [%]", ylabel="product impurity [ppm]",
           title="Baseline (Dax eq. 3.18, Table 5.9 walls), feed time 60-125 s")
    ax.legend(frameon=False, fontsize=8)
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


def extrapolated(r):
    """CSS impurity ppm of a capped brute-force run: remaining geometric series of the per-500-cycle increments."""
    p = pd.DataFrame(r["history"]).ppm.values
    d1, d2 = p[-501] - p[-1001], p[-1] - p[-501]
    rho = d2 / d1 if d1 else np.nan
    return p[-1] + (d2 * rho / (1 - rho) if 0 < rho < 1 else np.nan)


def timing(x):
    x = np.asarray(x, float)
    return dict(n=len(x), median=np.median(x), p90=np.percentile(x, 90), max=x.max())


def accel_check(adir=ACCEL_DIR):
    """Accelerated (cold start) vs brute-force CSS of the first step-4 run, same configurations. Brute-force
    reference: the first-run CSS continued with a 10x tighter criterion (EXT_DIR) when available; for the capped
    runs the geometric extrapolation of their drift."""
    rows = []
    for name in accel_check_cases():
        fa, fb, fe = adir / f"{name}.pkl", CASES_V1 / f"corr318_{name[3:]}.pkl", EXT_DIR / f"{name}.pkl"
        if not (fa.exists() and fb.exists()):
            continue
        with open(fa, "rb") as fh:
            A = pickle.load(fh)
        with open(fb, "rb") as fh:
            B = pickle.load(fh)
        E = None
        if fe.exists():
            with open(fe, "rb") as fh:
                E = pickle.load(fh)
        capped = not B["css"]
        ref = E["perf"] if E is not None else B["perf"]
        ref_ppm = extrapolated(B) if capped else ref["ppm"]
        dppm = A["perf"]["ppm"] / ref_ppm - 1
        drec = A["perf"]["recovery_pct"] - ref["recovery_pct"]
        rows.append(dict(case=name[3:], brute_cycles=B["cycles"], brute_min=B["runtime_s"] / 60,
                         ext_cycles=E["cycles"] if E is not None else np.nan,
                         accel_cycles=A["cycles"], accelerated=A["accelerated_cycles"], accel_min=A["runtime_s"] / 60,
                         ppm_brute_stop=B["perf"]["ppm"], ppm_brute_ref=ref_ppm, ppm_accel=A["perf"]["ppm"],
                         d_ppm_vs_stop=A["perf"]["ppm"] / B["perf"]["ppm"] - 1, d_ppm_rel=dppm,
                         rec_brute_ref=ref["recovery_pct"], rec_accel=A["perf"]["recovery_pct"], d_rec_pp=drec,
                         result=("capped brute force: vs extrapolation, not gated" if capped else
                                 "no continued reference" if E is None else
                                 ok(A["css"] and abs(dppm) <= GATE["accel_ppm_rel"] and abs(drec) <= GATE["accel_rec_pp"]))))
    return pd.DataFrame(rows)


def analyse():
    names = list(cases())
    R_ = {n: load_case(n) for n in names if (CASES_DIR / f"{n}.pkl").exists()}
    t = {}
    T = pd.DataFrame([perf_row(n, r) for n, r in R_.items()])
    t["all"] = T
    # gate 1
    g1 = [dict(check="CSS reached (step-3 criterion + impurity change < 1e-5, >= 10 plain cycles), all cases",
               value=f"{sum(r['css'] for r in R_.values())}/{len(R_)}", limit="all",
               result=ok(all(r["css"] for r in R_.values())))]
    clo = max(abs(pd.DataFrame(r["history"]).closure).max() for r in R_.values())
    g1.append(dict(check="per-cycle mole balance, worst cycle of all cases", value=f"{clo:.1e}", limit="< 1e-3",
                   result=ok(clo < GATE["closure"])))
    a, b = f"base_f{F_REF:.1f}_t101", f"base_f{F_REF:.1f}_t101_grid2N"
    if a in R_ and b in R_:
        p1, p2 = R_[a]["perf"], R_[b]["perf"]
        drec, dppm = abs(p2["recovery_pct"] - p1["recovery_pct"]), abs(p2["ppm"] / p1["ppm"] - 1)
        g1 += [dict(check=f"grid N {N} -> {2 * N}, f_AC {F_REF}: recovery", value=f"{drec:.3f} pp", limit="< 0.3 pp",
                    result=ok(drec < GATE["grid_rec_pp"])),
               dict(check=f"grid N {N} -> {2 * N}, f_AC {F_REF}: impurity", value=f"{100 * dppm:.1f} %", limit="< 10 %",
                    result=ok(dppm < GATE["grid_ppm_rel"]))]
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_layered.py"),
                         str(ROOT / "tests" / "test_cycle.py")], capture_output=True, text=True, cwd=ROOT)
    last = pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else pt.stderr[-200:]
    g1.append(dict(check="interface and CSS-acceleration tests (tests/test_layered.py, test_cycle.py)", value=last,
                   limit="all pass", result=ok(pt.returncode == 0)))
    t["accel_check"] = ac = accel_check()
    t["accel_tol4"] = accel_check(ACCEL_TOL4)
    if len(ac):
        g = ac[ac.result.isin(("PASS", "FAIL"))]
        g1.append(dict(check="accelerated vs brute-force CSS (ppm within 1e-3 rel., recovery within 0.05 pp)",
                       value=f"{(g.result == 'PASS').sum()}/{len(g)}, ppm {g.ppm_brute_ref.min():.0f}-{g.ppm_brute_ref.max():.0f}",
                       limit="all, >= 10 cases, 20-2000 ppm",
                       result=ok(len(g) >= 10 and (g.result == "PASS").all())))
    t["gate1"] = pd.DataFrame(g1)
    v1 = [pickle.load(open(f, "rb")) for f in sorted(CASES_V1.glob("corr318_*.pkl"))]
    lay = T[T.case.str.startswith("base_")]
    t["timing"] = pd.DataFrame([
        dict(run="first step-4 run, brute force (Dax eq. 3.18 cases)", **timing([r["runtime_s"] / 60 for r in v1])),
        dict(run="this run, warm start + Anderson (all baseline cases)", **timing(lay.wall_min)),
        *([dict(run="acceleration check, brute force to the step-3 criterion (same configurations)", **timing(ac.brute_min)),
           dict(run="acceleration check, cold start + Anderson", **timing(ac.accel_min))] if len(ac) else []),
        *([dict(run="before the ppm condition: all baseline cases", **timing(
              [pickle.load(open(f, "rb"))["runtime_s"] / 60 for f in sorted(CASES_TOL4.glob("base_*.pkl"))])),
           dict(run="before the ppm condition: acceleration check", **timing(t["accel_tol4"].accel_min))]
          if len(t["accel_tol4"]) else [])])
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
    if len(lay[lay.f_ac.notna() & lay.flow.isna() & lay.N.isna()]) >= len(F_AC) * len(T_FEED):
        t["trends"], t["rec_at_purity"] = trends(lay)
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


FINAL_DECISIONS = """## Final decisions (project lead, 2026-10-10)

1. **Axial dispersion: eq. 3.18 is the baseline for all layers and geometries. Eq. 5.20 is REJECTED** (not carried
   as a scenario). Evidence: (i) eq. 3.18 is the thesis's own correlation (Table 4.7, p.123) and reproduces the Dax
   of Tables 5.7/5.9 within 1.15-1.24x (gate 3 table below, kept); (ii) step 2 used the tabulated Dax and matched
   the MEASURED breakthrough curves (15/18 breakthrough times within 10 %), so the measured data support the
   eq. 3.18 magnitude and contradict a 51-55x larger dispersion; (iii) eq. 5.20 (Wakao-Funazkri form) is applied
   here at Re 1-2, below and at the edge of its stated validity (3 < Re < 10000, p.172).
2. **Wall heat transfer from each geometry's own source:** compact Lopes column h_w = 55, U = 50 W/m2/K (Table 5.9)
   in every layer, as in the step-3 baseline; Yang (2008) geometry their internal 38.5 and external 14.2 W/m2/K
   (`layered_4bed_yang2008.yaml`; combined into U by eq. 5.29). Note for step 5: industrial-scale columns are close
   to adiabatic.
3. **CSS acceleration, required before step 6** (near-constraint points are the most important and the slowest to
   converge; unconverged CSS would be label noise exactly at the constraint): warm start from the nearest converged
   CSS state plus Anderson acceleration of the cycle map; same CSS criterion, never declared on an accelerated
   iterate (>= 10 plain cycles at the end); verified against brute force on >= 10 cases spanning 20-2000 ppm
   (impurity within 1e-3 relative, recovery within 0.05 pp); the 4 capped cases rerun.
4. **Trends b and e: NOT reproduced, no tuning.** b: CH4 limits the AC-only bed, consistent with CH4 having the
   slowest micropore kinetics on HY2SEPS-AC. e: the single-column emulation of Yang's 4-bed 10-step cycle spends
   42-70 % of the feed H2 on product pressurisation, so its absolute recovery is not comparable. **Limitation for
   the paper:** the single-column emulation gives absolute recoveries below industrial (multi-bed) values.
5. Step 4 rerun with the new baseline (eq. 3.18, wall values per item 2, accelerated CSS); all pre-declared gates
   re-evaluated below. The first step-4 run (eq. 5.20 / eq. 3.18 with correlation walls, brute-force CSS) is in
   commit b23e20a; its eq. 3.18 cases are the brute-force references of the acceleration check."""

TIGHT_TEXT = """Before the ppm condition (CSS by the step-3 criterion alone, `results/step4/accel_check_tol4/`), the
slowest cases stopped up to 2.5e-3 short of the reference, because their final plain cycles drift by less than 1e-4
per 10 cycles while still far from the fixed point (the stall rule hands over to plain cycles earlier than a
residual of 1e-6). Per the project lead's instruction the gate was not relaxed; the CSS criterion was tightened
(impurity change < 1e-5 over the final >= 10 plain cycles, all other conditions unchanged) and the check rerun
(table above). The check before tightening:
"""

SUMMARY = """## Summary

* **Gates.** 1 PASS (33/33 cases at CSS; mole balance 6.6e-7; grid N -> 2N 0.007 pp and 7.9 %; tests;
  acceleration vs brute force 14/14). 2 PASS (AC5-KS limit = frozen step-3 baseline within 0.02 %). 3 FAIL as
  pre-declared for eqs 5.20, 5.31, 5.29, resolved by decisions 1-2 (eq. 3.18 adopted, 1.15-1.24x; walls from each
  geometry's source). 4: a, c, d reproduced; **b and e NOT reproduced, no tuning** (decision 4): b, CH4 limits the
  AC-only bed (323 ppm of 550 at t_feed 101 s), consistent with CH4 having the slowest micropore kinetics on
  HY2SEPS-AC; e, product pressurisation takes 42-70 % of the feed H2 in the single-column emulation of Yang's
  4-bed cycle. **Limitation for the paper:** single-column emulation gives absolute recoveries below industrial
  multi-bed values.
* **CSS acceleration** (warm start + safeguarded Anderson). Verification on 14 configurations, 25-1766 ppm, cold
  start, against brute force continued to a 10x tighter criterion: impurity within 8.9e-4 relative (worst
  f0.4_t101), recovery within 0.0006 pp; the 26 ppm / 1144-cycle case (f0.2_t80) +6.5e-5. With the step-3 criterion
  alone one case missed (f0.0_t60, -2.5e-3): the gate was kept and the CSS criterion tightened (impurity change
  < 1e-5 over the final >= 10 plain cycles); cost: baseline p90 10.3 -> 13.8 min, check max 8.0 -> 11.2 min.
  The 4 cases capped at 2000 brute-force cycles now converge in 4.6-12.7 min (baseline) and 6.6-8.3 min (cold
  start), within -4.5e-4 to +1.3e-3 of the extrapolated brute-force limit (not gated).
* **Wall time per CSS** (shared machine, 7 workers): brute force median 14.0, p90 96.3, max 104.0 min; accelerated
  median 5.9, p90 13.8, max 87.1 min. The 10-min target holds for the median, not for p90 / max. The maximum is
  f0.2_t125 (1.7 % impurity, far from the constraint): its CO2 front creeps through the zeolite at constant speed
  for ~1500 cycles, a drift rather than a geometric approach, which extrapolation cannot predict.
"""

CORR_TEXT = """Reading of gate 3. Eq. 5.20 is coded as printed on p.172 (eps Dax / Dm = 20 + 0.5 Sc Re). The text there
gives its validity as 3 < Re < 10000; the compact-unit conditions have Re = 1.4-6.6 (Table 5.7) and 2.0-2.1
(Table 5.9), the HY2SEPS layers 1.2-1.7. At these Re the constant 20 dominates (0.5 Sc Re < 4) and Dax is about
21 Dm / eps, i.e. 51-55x the tabulated values, also for run 3 inside the stated range; the ch.3 correlation
(eq. 3.18) reproduces them within 1.15-1.24x. h_w and U in Tables 5.7 / 5.9 do not vary with flow, pressure or
temperature (55 and 50 W/m2/K throughout), so they are set values of the apparatus. Decisions 1 and 2: eq. 3.18 is
adopted, eq. 5.20 rejected, and h_w, U come from each geometry's own source; the correlation values of eqs 5.31 and
5.29 are no longer used."""


def write_report(R_, t):
    T = t["all"]
    f_all = {"recovery": "{:.2f}", "ppm": "{:.3g}", "purity": "{:.5f}", "productivity": "{:.1f}", "CO2_ppm": "{:.3g}",
             "CH4_ppm": "{:.3g}", "CO_ppm": "{:.3g}", "N2_ppm": "{:.3g}", "wall_min": "{:.1f}", "cpu_min": "{:.1f}",
             "f_ac": "{:.1f}", "t_feed": "{:.0f}", "flow": "{:.1f}", "yang": "{:.2f}", "N": "{:.0f}"}
    lay = T[T.case.str.startswith("base_")]
    tm = timing(lay.wall_min)
    tr = t.get("trends")
    g3 = t["gate3"]
    hw_u = g3[g3.coefficient.isin(("hw (5.31)", "U (5.29)"))]
    ac = t["accel_check"]
    text = f"""# Step 4 — layered bed (HY2SEPS-AC + HY2SEPS-ZEO)

Generated by `scripts/run_layered.py` (cases cached in `results/step4/cases/`). Thesis pages are printed pages.

{FINAL_DECISIONS}

## Verdict (rerun with the decided baseline)

| gate | result |
|---|---|
| 1. Numerical (CSS, mole balance, grid N -> 2N, interface, CSS acceleration vs brute force) | {ok(len(t["gate1"]) and (t["gate1"].result == "PASS").all())} |
| 2. Limit case: AC5-KS through the layered code = frozen step-3 baseline | {ok(len(t["gate2"]) and (t["gate2"].result == "PASS").all())} |
| 3. Correlations vs Tables 5.7 / 5.9 (factor 1.5) | FAIL as pre-declared (Dax eq. 5.20, h_w eq. 5.31, U eq. 5.29); resolved by decisions 1-2: Dax eq. 3.18 {g3[g3.coefficient == "Dax (3.18)"].result.iloc[0]}, h_w / U from the apparatus |
| 4. Qualitative trends a-e (baseline) | {"; ".join(f"{r.trend[:2]} {r.result}" for _, r in tr.iterrows()) if tr is not None else "–"} |
| Runtime per CSS, layered bed (target <= 10 min wall) | median {tm["median"]:.1f}, 90th percentile {tm["p90"]:.1f}, max {tm["max"]:.1f} min |

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
* Transport: Dax by eq. 3.18 (decision 1); lambda, kf, hf and Dp by the thesis correlations (eqs 5.17-5.23) at
  the feed conditions, per layer (`src/h2psa/transport.py`); the macropore diffusivities quoted on pp.97/103 are
  molecular diffusion in He at 1 bar and 303 K divided by tau = 2 (Chapman-Enskog reproduces them within 3-14 %),
  so at PSA conditions Dp follows from eq. 5.17 with the same tau. Wall: h_w = 55, U = 50 W/m2/K (Table 5.9) in the
  compact column; h_w = 38.5, U = 12.2 W/m2/K (Yang's 38.5 and 14.2 through eq. 5.29) in the Yang column.
* Cycle: the frozen step-3 baseline (cycle B of the experiment, beta_P = 0.102 s-1, EP = pure H2, purge
  0.261 mmol/s), compact column (0.267 m x 5 cm), Lopes feed at 5 bar and 303 K; feed time and feed flow varied for
  the trends. Productivity uses the adsorbent mass of the layered bed.
* CSS (`CycleRunner.run_to_css(accel=True)`, decision 3): warm start from the converged CSS state of the nearest
  case of the same geometry, grid and adsorbents (Euclidean distance in f_AC, t_feed / 110 s, feed-flow ratio; Yang:
  u / 5.2 cm/s), from the cases finished when the case starts; then Anderson acceleration (type II, memory 5) of
  the cycle map on the state scaled by the solver's variable scales, with the cumulative mole counters excluded,
  concentrations and loadings clipped at zero, and the memory dropped whenever the residual grows above twice its
  best value. When the max-norm of the scaled residual G(x) - x falls below 1e-6, or has not halved in 20
  accelerated cycles (stagnation: the residual floor set by the integration tolerance, or Anderson stalled in a
  strongly nonlinear breakthrough transition), plain cycles follow, and CSS is
  declared only by the step-3 criterion (relative change of ppm, recovery and bed-averaged loadings < 1e-4 over 10
  cycles, mole balance < 0.1 %) evaluated over >= 10 consecutive plain cycles, tightened by one condition on the
  impurity: its relative change over those cycles < 1e-5 (see the acceleration check); after 30 plain cycles
  without CSS the acceleration restarts with an empty memory.

## Gate 1 — numerical

{md(t["gate1"])}

### CSS acceleration vs brute force

Configurations of the first step-4 run (Dax eq. 3.18, correlation walls), whose brute-force CSS is cached, run again
from the He-filled column (cold start, no warm start) with Anderson acceleration. The step-3 criterion stops the
slow brute-force runs short of the fixed point (the 26 ppm case was still rising 0.03 ppm per 100 cycles at its
stop), so the brute-force reference is the first-run CSS state continued with plain cycles under a 10x tighter
criterion (change < 1e-5 over 10 cycles; `ext_cycles` more cycles); `d_ppm_vs_stop` compares with the original
stop. Gate: impurity within 1e-3 relative and recovery within 0.05 pp of the reference; the 4 cases that hit the
2000-cycle cap are compared with the geometric extrapolation of their brute-force drift (reported, not gated):

{md(ac, {"brute_min": "{:.1f}", "ext_cycles": "{:.0f}", "accel_min": "{:.1f}", "ppm_brute_stop": "{:.3f}", "ppm_brute_ref": "{:.3f}", "ppm_accel": "{:.3f}", "d_ppm_vs_stop": "{:+.1e}", "d_ppm_rel": "{:+.1e}", "rec_brute_ref": "{:.3f}", "rec_accel": "{:.3f}", "d_rec_pp": "{:+.4f}"}) if len(ac) else "not run"}

{TIGHT_TEXT if len(t["accel_tol4"]) else ""}
{md(t["accel_tol4"][["case", "accel_cycles", "accel_min", "ppm_brute_ref", "ppm_accel", "d_ppm_rel", "d_rec_pp", "result"]], {"accel_min": "{:.1f}", "ppm_brute_ref": "{:.3f}", "ppm_accel": "{:.3f}", "d_ppm_rel": "{:+.1e}", "d_rec_pp": "{:+.4f}"}) if len(t["accel_tol4"]) else ""}

Wall time per CSS [min] (shared machine, 7 workers):

{md(t["timing"], {"median": "{:.1f}", "p90": "{:.1f}", "max": "{:.1f}"})}

## Gate 2 — limit case

AC5-KS (Table 5.9 group 1-4 transport carried on the layer, adsorbent mass from the bed) as one layer and split into
two identical layers (0.6 | 0.4), against the frozen step-3 baseline (`results/step3/cases/cor_exp.pkl`):

{md(t["gate2"], {"recovery": "{:.3f}", "recovery_step3": "{:.3f}", "d_rec_pp": "{:+.4f}", "ppm": "{:.2f}", "ppm_step3": "{:.2f}", "d_ppm_rel": "{:+.2%}"}) if len(t["gate2"]) else "not run"}

## Gate 3 — correlations vs the author's tables (AC5-KS, compact unit)

Ratio recomputed / tabulated (Table 5.7: breakthrough runs at 5 bar, 323 K; Table 5.9: PSA groups; flow
5.0e-5 m3/s at 298 K and 1 bar; scalar coefficients use the mole-fraction mean of the species' mixture
diffusivities, eq. 5.18; U with T_w - T_inf = 10 K, air properties at 300 K, AISI 304 wall):

{md(g3, {"min_ratio": "{:.2f}", "max_ratio": "{:.2f}"})}

{md(t["corr_ratios"], {c: "{:.2f}" for c in t["corr_ratios"].columns if c not in ("case",)})}

{CORR_TEXT}

Coefficients used for the HY2SEPS layers (experiment feed, 5 bar, 303 K, compact column):

{md(t["hy2seps_coefficients"], {c: "{:.3g}" for c in t["hy2seps_coefficients"].columns if c != "layer"})}

## Gate 4 — qualitative trends (no tuning)

Declared before running: a — every layered fraction 0.2-0.8 has a lower impurity than AC only at the midpoint of
the common recovery range of the feed-time scans; b — the largest impurity of AC only (t_feed 101 s) is N2 or CO
and is lower at f_AC = {F_REF}; c — recovery at 99.99 % purity (interpolated in ln ppm along the feed-time scan) is
highest at an interior fraction and higher than at both ends (an end that never reaches 99.99 % counts as worse);
d — impurity and recovery both rise monotonically with feed time for every fraction, and with feed flow
(x0.8, 1, 1.2) at f_AC = {F_REF}; e — Yang geometry/feed: at least one velocity gives <= 100 ppm with 60-80 %
recovery.

{md(tr) if tr is not None else "not run"}

Trend e diagnostic — where the feed H2 goes in the Yang emulation (mol per mol H2 fed, CSS cycle; purge_in and
press_in are product H2 returned to the column):

{md(t["yang_h2"], {c: "{:.3f}" for c in t["yang_h2"].columns if c != "case"}) if len(t["yang_h2"]) else "not run"}

![feed-time scans](step4_scan.png)

![profiles](profiles_base.png)

### All cases

{md(T.drop(columns=["css"]), f_all)}

## Runtime

Layered-bed CSS with warm start + Anderson: median {tm["median"]:.1f}, 90th percentile {tm["p90"]:.1f}, max
{tm["max"]:.1f} min wall ({lay.cpu_min.median():.1f} min CPU median), on a shared machine; before (first step-4 run,
brute force) see the timing table under gate 1. Other speed-up of this step: multisite Langmuir solved by Newton in
ln(s) with warm start (about 5x per cycle).

## Step-6 preparation (draft, not run): `configs/design_space.yaml`

```yaml
{(ROOT / "configs" / "design_space.yaml").read_text().strip()}
```
"""
    (OUT / "REPORT.md").write_text(text)


def run_all(todo, out_dir, warm, workers):
    order = sorted(todo, key=lambda n: ("grid2N" not in n and "yang" not in n, n))     # longest first
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_case, n, todo[n], out_dir, warm): n for n in order}
        for f in as_completed(futs):
            if f.exception() is not None:
                print(f"[{futs[f]}] FAILED: {f.exception()!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--accel-check", action="store_true", help="run the CSS-acceleration check cases (cold start)")
    ap.add_argument("--extend-brute", action="store_true", help="continue the brute-force references (tol 1e-5)")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for d in (CASES_DIR, ACCEL_DIR, STATES, EXT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    pick = lambda allc, d: {n: s for n, s in allc.items() if (not a.only or n in a.only.split(","))
                            and (a.force or not (d / f"{n}.pkl").exists())}
    if not a.report_only:
        if a.extend_brute:
            todo = {n: s for n, s in pick(accel_check_cases(), EXT_DIR).items() if n[3:] not in
                    {k for k in V1_CAPPED}}
            with ProcessPoolExecutor(a.workers) as ex:
                futs = {ex.submit(extend_brute, n, todo[n]): n for n in todo}
                for f in as_completed(futs):
                    if f.exception() is not None:
                        print(f"[{futs[f]}] FAILED: {f.exception()!r}", flush=True)
        elif a.accel_check:
            run_all(pick(accel_check_cases(), ACCEL_DIR), ACCEL_DIR, False, a.workers)
        else:
            run_all(pick(cases(), CASES_DIR), CASES_DIR, True, a.workers)
    R_, t = analyse()
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(OUT / f"{k}.csv", index=False)
    for f in list(OUT.glob("*.png")) + list((DOCS / "step4").glob("*.png")):
        f.unlink()                              # figures of earlier runs
    if "trends" in t:
        fig_scan(t["all"], OUT / "step4_scan.png")
    n = f"base_f{F_REF:.1f}_t101"
    if n in R_:
        fig_profiles(R_[n], OUT / "profiles_base.png")
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
