"""Step 2, decision 3: calibrate the solid heat capacity Cps and the lumped wall heat capacity per unit length
Cw = rho_w Cp_w pi e (d + e) on breakthrough runs 1-3 (thermocouple histories + CO2 t10/t50 jointly).
Kinetics and isotherms fixed. Runs 4-5 and the PSA experiment are held out.

    python scripts/calibrate_thermal.py      # ~10 min

Writes results/step2/calibration.csv (every evaluation) and calibration_result.csv (read by run_breakthrough.py),
and prints the `calibrated:` YAML block. Decision 3: if a parameter ends on a bound the fit is not adopted
(the block is then not to be copied into the YAML).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h2psa.column import compact_breakthrough  # noqa: E402
from h2psa.isotherms import load_yaml  # noqa: E402
from h2psa.validation import FIT_RUNS, T_FIT, rebuild, sigma_thermocouple, thermal_residuals  # noqa: E402

OUT = ROOT / "results" / "step2"
CPS_BOUNDS = (600.0, 1200.0)       # J kg-1 K-1 (decision 3)
WALL_E_BOUNDS = (0.0005, 0.005)    # m of stainless steel (decision 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cc = load_yaml(ROOT / "data" / "params" / "compact_column_lopes2010.yaml")
    asm, d = cc["assumed"], cc["column"]["internal_diameter"]
    cw = lambda e: asm["wall_density"]["value"] * asm["wall_heat_capacity"]["value"] * np.pi * e * (d + e)
    lo, hi = np.array([CPS_BOUNDS[0], cw(WALL_E_BOUNDS[0])]), np.array([CPS_BOUNDS[1], cw(WALL_E_BOUNDS[1])])
    x0 = np.array([asm["solid_heat_capacity"]["value"], cw(asm["wall_thickness"]["value"])])
    models = {r: compact_breakthrough(r, thermal="assumed") for r in FIT_RUNS}
    sT = sigma_thermocouple(models[1].feed.T)
    log = []

    def fun(p):  # p scaled by x0
        Cps, Cw = p * x0
        res = np.concatenate([thermal_residuals(rebuild(m, col={"Cw": Cw}, Cps=Cps), r, sT) for r, m in models.items()])
        log.append(dict(Cps=Cps, Cw=Cw, objective=float(res @ res)))
        print(f"  Cps {Cps:7.1f}  Cw {Cw:7.1f}  objective {res @ res:9.3f}", flush=True)
        return res

    print(f"sigma_T = {sT:.3f} K; bounds Cps {lo[0]:.0f}-{hi[0]:.0f}, Cw {lo[1]:.0f}-{hi[1]:.0f} J/m/K")
    sol = least_squares(fun, np.ones(2), bounds=(lo / x0, hi / x0), diff_step=3e-3, x_scale=1.0, xtol=1e-4, ftol=1e-6)
    p = sol.x * x0
    J = sol.jac / x0                                  # d r / d (Cps, Cw)
    chi2, dof = float(sol.fun @ sol.fun), sol.fun.size - 2
    cov = np.linalg.inv(J.T @ J)
    n_obs = 3 * len(FIT_RUNS) + 2 * len(FIT_RUNS)     # histories count as one observation each
    s2 = chi2 / (n_obs - 2)
    sd, sd_s = np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov) * s2)
    corr = cov[0, 1] / (sd[0] * sd[1])
    at_bound = [bool(min(v - l, h - v) < 0.01 * (h - l)) for v, l, h in zip(p, lo, hi)]
    pd.DataFrame(log).to_csv(OUT / "calibration.csv", index=False)
    e_eq = np.roots([1, d, -p[1] / (np.pi * asm["wall_density"]["value"] * asm["wall_heat_capacity"]["value"])]).max()
    pd.Series({"Cps": p[0], "Cps_sd": sd_s[0], "Cps_lo": lo[0], "Cps_hi": hi[0], "Cps_at_bound": at_bound[0],
               "Cw": p[1], "Cw_sd": sd_s[1], "Cw_lo": lo[1], "Cw_hi": hi[1], "Cw_at_bound": at_bound[1],
               "Cw_equiv_wall_mm": e_eq * 1e3, "correlation": corr, "objective_start": log[0]["objective"],
               "objective_final": chi2, "chi2_per_observation": s2, "sigma_T": sT, "nfev": sol.nfev,
               "date": date.today().isoformat()}, name="value").rename_axis("item").to_csv(OUT / "calibration_result.csv")
    if any(at_bound):
        print("\nA PARAMETER IS ON ITS BOUND: fit not adopted (decision 3: stop and report). Do not copy the block below.")
    print(f"""
status {sol.status}: {sol.message}; {sol.nfev} evaluations
Cps = {p[0]:.0f} +- {sd_s[0]:.0f} J/kg/K (+-{sd[0]:.0f} unscaled), at bound: {at_bound[0]}
Cw  = {p[1]:.0f} +- {sd_s[1]:.0f} J/m/K (+-{sd[1]:.0f} unscaled), at bound: {at_bound[1]}  (= {e_eq * 1e3:.2f} mm stainless steel)
correlation {corr:+.3f}; objective {chi2:.3f} (start {log[0]['objective']:.3f}); chi2 per observation-equivalent {s2:.3f}

calibrated:                      # step 2 decision 3 ({date.today().isoformat()}); scripts/calibrate_thermal.py
  fit_data: "breakthrough runs {', '.join(map(str, FIT_RUNS))} (Figs 5.15, digitised): thermocouple histories 0-{T_FIT:.0f} s and CO2 t10/t50; runs 4-5 and the PSA experiment held out"
  objective: "sum over runs of [sum over the 3 thermocouples of mean((T_sim - T_exp)^2)/sigma_T^2 + sum over CO2 t10, t50 of ((t_sim - t_exp)/sigma_t)^2]; sigma_T = {sT:.2f} K (RMS deviation from 323 K for t > 4000 s, runs 1-3); sigma_t = half the bracketing GC sampling interval (+) 2.4 s read-off"
  fixed: "kinetics, isotherm (Taqvi-LeVan), transport (Table 5.7), wall thickness 3 mm in the heat-exchange geometry"
  objective_value: {{start: {log[0]['objective']:.3f}, final: {chi2:.3f}}}
  solid_heat_capacity: {{value: {p[0]:.1f}, sd: {sd_s[0]:.1f}, bounds: [{lo[0]:.0f}, {hi[0]:.0f}], units: J kg-1 K-1}}
  wall_heat_capacity_per_length: {{value: {p[1]:.1f}, sd: {sd_s[1]:.1f}, bounds: [{lo[1]:.0f}, {hi[1]:.0f}], units: J m-1 K-1,
                                  note: "rho_w Cp_w pi e (d_wi + e); bounds = 0.5-5 mm stainless steel; equivalent to {e_eq * 1e3:.2f} mm"}}
  correlation: {corr:+.3f}
  uncertainty: "Gauss-Newton Hessian at the optimum, scaled by chi2 per observation-equivalent ({s2:.2f})"
""")


if __name__ == "__main__":
    main()
