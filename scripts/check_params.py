"""Step-1 consistency checks of the transcribed literature parameters.

Run:  python scripts/check_params.py
Every check prints PASS/FAIL; the script exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

np.seterr(over="ignore", invalid="ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.isotherms import R, MultisiteLangmuir, Virial, VirialSpecies, load_yaml  # noqa: E402
from h2psa.kinetics import arrhenius_fit, dc_over_rc2  # noqa: E402

P = ROOT / "data" / "params"
B = ROOT / "data" / "benchmarks"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ----------------------------------------------------------------- C1 virial isotherms AC5-KS
# Loadings at 7 bar read from Figures 4.9-4.10 (pp.125-126); CO2 303 K value also quoted in text p.125.
FIG = {303: {"CO2": 6.7, "H2": 0.24, "CH4": 3.4, "CO": 2.05, "N2": 1.55},
       323: {"CO2": 5.4, "H2": 0.21, "CH4": 2.6, "CO": 1.52, "N2": 1.16}}
vir = Virial.from_yaml(P / "ac5ks_lopes2010.yaml")
worst = 0.0
for T, tg in FIG.items():
    for sp, qf in tg.items():
        q = vir.loading_pure(sp, 7.0, T)
        worst = max(worst, abs(q - qf) / qf)
check("C1 virial AC5-KS vs Fig.4.9-4.10 (7 bar, 303/323 K)", worst < 0.06, f"max rel. dev {worst:.1%} (figure read-off ±5%)")

# C1b: printed CO A1 sign (Table 4.9/5.5) cannot reproduce Fig. 4.10c -> erratum E2
co = vir.species["CO"]
co_printed = VirialSpecies(co.K_inf, co.minus_dH, co.A0, -co.A1, co.B0, co.B1)
v2 = Virial({**vir.species, "CO": co_printed}, vir.S)
reach = v2.pressure_pure("CO", np.linspace(0, 30, 3001), 303).max()
check("C1b erratum E2: printed CO A1=-105.46 fails", reach < 7.0, f"max pressure reachable with printed sign = {reach:.2f} bar (<7 bar)")

# C1c: B1 scale. Table 5.5 header says x1e-6, Table 4.9 says x1e-11. Only 1e11 reproduces CO2 6.7 mol/kg.
c = vir.species["CO2"]
v3 = Virial({"CO2": VirialSpecies(c.K_inf, c.minus_dH, c.A0, c.A1, c.B0, c.B1 / 1e5)}, vir.S)
try:
    q6 = v3.loading_pure("CO2", 7.0, 303)
except ValueError:
    q6 = float("nan")
check("C1c erratum E1: B1 x1e-6 (Table 5.5 header) fails", not (abs(q6 - 6.7) < 0.3),
      f"q_CO2(303K,7bar) with x1e6 = {q6:.2f} vs 6.7; with x1e11 = {vir.loading_pure('CO2', 7.0, 303):.2f}")

# C1d: monotone (thermodynamically valid) inside the fitted range
mono = all(np.all(np.diff(vir.pressure_pure(sp, np.linspace(1e-6, vir.loading_pure(sp, 7.0, T), 2000), T)) > 0)
           for sp in vir.names for T in (303, 323))
check("C1d virial isotherms monotone on 0-7 bar", mono, "dP/dq > 0 for all species, 303 and 323 K")

# ----------------------------------------------------------------- C2 multisite Langmuir
zeo = MultisiteLangmuir.from_yaml(P / "hy2seps_zeo_lopes2010.yaml")
txt = load_yaml(P / "hy2seps_zeo_lopes2010.yaml")["isotherm"]["text_check_303K_7bar"]
dev = {sp: zeo.loading_pure(sp, 7.0, 303) for sp in txt}
ok = all(abs(dev[s] - txt[s]) <= max(0.1 * txt[s], 0.05) for s in txt)
check("C2 MSL HY2SEPS-ZEO vs text p.54 (303 K, 7 bar)", ok,
      ", ".join(f"{s} {dev[s]:.2f}/{txt[s]}" for s in txt))
for f in ("hy2seps_ac_lopes2010.yaml", "hy2seps_zeo_lopes2010.yaml"):
    m = MultisiteLangmuir.from_yaml(P / f)
    aq = [s.a * s.q_max for s in m.species.values()]
    check(f"C2b a*q_max constant ({f.split('_')[1].upper()})", np.ptp(aq) / np.mean(aq) < 0.01,
          f"a*q_max = {np.mean(aq):.2f} ± {np.ptp(aq):.3f} mol/kg (eq. 2.12)")

# ----------------------------------------------------------------- C3 kinetics Arrhenius consistency
def arr_check(tbl: dict, label: str, tol: float = 0.25):
    bad = []
    for sp, v in tbl.items():
        if sp == "units" or not isinstance(v, dict) or "Dc0_over_rc2" not in v:
            continue
        for T, obs in v["Dc_over_rc2"].items():
            pred = v["Dc0_over_rc2"] * np.exp(-v["Ea"] * 1e3 / (R * float(T)))
            if abs(np.log(pred / obs)) > np.log(1 + tol):
                bad.append(f"{sp}@{T}K pred {pred:.3g} vs {obs:.3g}")
    return bad

def refit_check(tbl: dict, label: str, tol: float = 0.25):
    """Measured Dc/rc2 values must be internally Arrhenius-consistent (our refit convention)."""
    worst, fits = 0.0, {}
    for sp, v in tbl.items():
        if sp == "units" or not isinstance(v, dict):
            continue
        D0, Ea = arrhenius_fit(v["Dc_over_rc2"])
        fits[sp] = (D0, Ea)
        for T, obs in v["Dc_over_rc2"].items():
            worst = max(worst, abs(np.log(dc_over_rc2(D0, Ea, float(T)) / obs)))
    ok = worst < np.log(1 + tol)
    check(f"C3 refit Arrhenius {label} (±{tol:.0%})", ok,
          f"max dev {np.exp(worst) - 1:.0%}; " + ", ".join(f"{s}: D0={d:.3g} s-1, Ea={e:.1f}" for s, (d, e) in fits.items()))

hac = load_yaml(P / "hy2seps_ac_lopes2010.yaml")["micropore_kinetics"]
hzeo = load_yaml(P / "hy2seps_zeo_lopes2010.yaml")["crystal_kinetics"]
ac = load_yaml(P / "ac5ks_lopes2010.yaml")
refit_check(hac, "HY2SEPS-AC")
refit_check(hzeo, "HY2SEPS-ZEO")
refit_check(ac["micropore_kinetics_breakthrough"], "AC5-KS (Table 4.10)")

bad = arr_check(hac, "AC")
check("C3b printed (D0,Ea) reproduce HY2SEPS-AC data", not bad, "; ".join(bad) or "consistent")
orig = {**hac, "CH4": {**hac["CH4"], "Dc_over_rc2": {**hac["CH4"]["Dc_over_rc2"], 323: 1.64e-3}}}
D0, Ea = arrhenius_fit(orig["CH4"]["Dc_over_rc2"])
check("C3c erratum E4: printed CH4 323 K value 1.64e-3 breaks monotonicity",
      not (9.41e-3 < 1.64e-3 < 2.71e-2), "303 K 9.41e-3 < 323 K ? < 343 K 2.71e-2 -> corrected to 1.64e-2")
bad = arr_check(hzeo, "ZEO")
check("C3d erratum E5: printed (D0,Ea) of Table 3.15 do NOT reproduce its own Dc/rc2 (documented)", len(bad) > 0,
      f"{len(bad)}/15 points off by >25%, e.g. {bad[0] if bad else '-'}")
mix = {sp: {"Dc_over_rc2": ac["micropore_kinetics_breakthrough"][sp]["Dc_over_rc2"], **v}
       for sp, v in ac["psa_transport"]["micropore"].items()}
bad = arr_check(mix, "AC5-KS")
check("C3b erratum E3: Table 5.9 Dc0/rc2 vs Table 4.10 (expected to FAIL -> documented)", len(bad) > 0,
      f"{len(bad)} inconsistent points, e.g. {bad[0] if bad else '-'}")

# ----------------------------------------------------------------- C4 column / feed bookkeeping
col = load_yaml(P / "compact_column_lopes2010.yaml")
c4 = col["column"]
rho_b = c4["adsorbent_mass"] / c4["bed_volume"]
rho_b2 = ac["physical"]["particle_density"] * (1 - c4["bed_porosity"])
check("C4 bulk density m/V vs rho_p(1-eps)", abs(rho_b / rho_b2 - 1) < 0.01, f"{rho_b:.1f} vs {rho_b2:.1f} kg/m3")
d = np.sqrt(4 * c4["bed_volume"] / (np.pi * c4["bed_height"]))
check("C4b derived internal diameter", abs(d - c4["internal_diameter"]) < 1e-3, f"{d * 1e3:.2f} mm")
nfeed = 1e5 * col["feed"]["flow_rate_std"] / (R * 298.15)
check("C4c feed H2 molar flow vs Fig.5.19 (~1.6e-3 mol/s)", abs(nfeed * 0.79 - 1.6e-3) < 0.1e-3, f"{nfeed * 0.79:.3e} mol/s")
tsum = sum(col["cycle_B_experiment"]["times_s"].values())
check("C4d cycle time = 459 s (Fig. 5.18)", tsum == col["cycle_B_experiment"]["cycle_time_s"] == 459, f"{tsum} s")

# ----------------------------------------------------------------- C5 productivity recomputed from recovery
def productivity(rec_pct, t_feed, t_cycle, m=c4["adsorbent_mass"]):
    return rec_pct / 100 * nfeed * 0.79 * t_feed * 86400 / t_cycle / m

exp = pd.read_csv(B / "lopes2010_psa_experiment.csv").iloc[0]
pe = productivity(exp.h2_recovery_pct, exp.t_feed_s, exp.cycle_time_s)
check("C5 experiment productivity from recovery", abs(pe / exp.productivity_mol_kg_day - 1) < 0.02,
      f"{pe:.1f} vs reported {exp.productivity_mol_kg_day}")
sims = pd.read_csv(B / "lopes2010_psa_simulations.csv")
n_eq = sims.cycle.map({"A": 2, "B": 6})
tcyc = sims.t_feed_s + sims.t_blow_s + sims.t_purge_s + sims.t_press_s + n_eq * sims.t_eq_s
prod = productivity(sims.h2_recovery_pct, sims.t_feed_s, tcyc)
rel = (prod / sims.productivity_mol_kg_day - 1).abs()
check("C5b 17 simulations: productivity from recovery (±3%)", (rel < 0.03).all(),
      f"max dev {rel.max():.1%} (sim {int(sims.sim[rel.idxmax()])}); median {rel.median():.1%}")

# ----------------------------------------------------------------- C6 NETL stream table
n = load_yaml(P / "industrial_feed_netl2022.yaml")
f, p, t = n["psa_feed"], n["psa_product"], n["psa_tail_gas"]
for nm, s in (("feed", f), ("product", p), ("tail", t)):
    check(f"C6 NETL {nm} mole fractions sum to 1", abs(sum(s["composition"].values()) - 1) < 2e-3,
          f"sum = {sum(s['composition'].values()):.4f}")
tot = abs(f["flow_kmol_h"] - p["flow_kmol_h"] - t["flow_kmol_h"]) / f["flow_kmol_h"]
h2 = f["flow_kmol_h"] * f["composition"]["H2"]
h2out = p["flow_kmol_h"] * p["composition"]["H2"] + t["flow_kmol_h"] * t["composition"]["H2"]
check("C6b NETL PSA mole balance (total, H2)", tot < 0.01 and abs(h2out / h2 - 1) < 0.01,
      f"total closure {tot:.2%}, H2 closure {abs(h2out / h2 - 1):.2%}")
rec = p["flow_kmol_h"] * p["composition"]["H2"] / h2
check("C6c NETL derived recovery", abs(rec - n["derived"]["h2_recovery"]) < 0.005, f"{rec:.3f}")

# ----------------------------------------------------------------- summary
n_fail = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
