# Data provenance and errata

All parameters are transcribed by hand from public sources. Every number is cross-checked twice:
by `scripts/check_params.py` (25 automated checks) and by an independent visual audit of the
rendered PDF pages (~580 values, 2026-10-09).

## Sources

| Key | Reference | Access | Used for |
|---|---|---|---|
| Lopes2010 | F.V.S. Lopes, *New pressure swing adsorption cycles for hydrogen purification from steam reforming off-gases*, PhD thesis, University of Porto, 2010. doi:10.34626/ftfe-yr63, hdl:10216/61875 | Open access | Isotherms, kinetics, column, PSA experiment and 17 reference simulations |
| NETL2022 | NETL, *Comparison of Commercial, State-of-the-Art, Fossil-Based Hydrogen Production Technologies*, DOE/NETL-2022/3241, 2022 | Public | Industrial PSA feed/product/tail gas (Exhibit 3-4) |
| Yang2008 | S. Yang et al., *Hydrogen purification by the four-bed PSA process from SMR off-gas*, KNRES autumn conf. 2008, pp. 383-386 | Open access (KoreaScience) | Layered-bed geometry, cycle, experimental trends |
| NIST | E.W. Lemmon et al., *Thermophysical Properties of Fluid Systems*, NIST Chemistry WebBook, SRD 69, https://webbook.nist.gov/chemistry/fluid/ (retrieved 2026-10-09) | Public | Gas heat capacities and viscosities at 5 bar, 303-383 K (`gas_properties_nist.yaml`) |

Thesis page numbers in the YAML files are **printed** page numbers; PDF page = printed + 34.

## Errata found in the sources

| ID | Location | Printed | Used here | Evidence |
|---|---|---|---|---|
| E1 | Lopes2010 Table 5.5 header, column B1 | "B1 × 10^-6" | B1 × 10^-11 (as in Table 4.9) | Only ×1e11 reproduces the CO2 loading of 6.7 mol/kg at 303 K, 7 bar quoted on p.125 (check C1c) |
| E2 | Lopes2010 Tables 4.9 and 5.5, AC5-KS CO, A1 | −105.46 | **+105.46** | With the printed sign the isotherm cannot exceed 0.55 bar; with the corrected sign all CO points of Fig. 4.10c are reproduced (checks C1, C1b) |
| E3 | Lopes2010 Table 5.9 (Dc0/rc², PSA simulations) vs Table 4.10 | e.g. H2 46.35 s⁻¹ | both kept; resolved per species by E7 | Table 5.9 values do not reproduce the measured Dc/rc² of Table 4.10 (H2 1.58 vs 8.13 s⁻¹ at 303 K). Kinetics are **not** chosen by matching the PSA experiment (the held-out validation target of step 3) |
| E4 | Lopes2010 Table 3.12, HY2SEPS-AC CH4, 323 K | 1.64 × 10⁻³ | 1.64 × 10⁻² | Value must lie between 9.41e-3 (303 K) and 2.71e-2 (343 K); the table's own Arrhenius fit gives 1.68e-2 (check C3c) |
| E5 | Lopes2010 Table 3.15, HY2SEPS-ZEO, Dc0/rc² | e.g. CO2 0.268 s⁻¹ | refitted from measured Dc/rc² | Printed (Dc0, Ea) miss the table's own Dc/rc² values by a factor of ~1.9 (15/15 points). Convention: measured values are primary, Arrhenius refitted (`h2psa.kinetics.arrhenius_fit`) |
| E6 | Lopes2010 Table 4.9, AC5-KS H2, A1 | "0.1994-" | +0.1994 | Table 5.5 prints +0.1994; H2 loadings in Fig. 4.10a reproduced (check C1) |
| E7 | Lopes2010 Tables 5.7 and 5.9, AC5-KS CH4, Dc0/rc² = 35.52 s⁻¹, Ea = 26.4 kJ/mol | Dc/rc² = 1.9e-3 s⁻¹ at 323 K | Arrhenius refit of Table 4.10 (0.144 s⁻¹ at 323 K) for CH4 only (`psa_transport.micropore_use_measured`) | Decision rule (project lead, 2026-10-09): a printed PSA-simulation constant is replaced only if it contradicts **both** the measured value on the same adsorbent (Table 4.10) **and** the author's own simulated curve. Only CH4 meets it: 75× below Table 4.10, and with it our model leaks CH4 at t10 = 197 s while the author's simulated CH4 front (Fig. 5.17) is sharp (t10 = 478 s). H2, CO2, CO, N2 also differ from Table 4.10 but our model reproduces the author's CO and N2 curves with them (within 7 %), so they stay. Sensitivity case: Table 4.10 for all species (step-2 report) |

## Values not reported in any source (assumed; varied in sensitivity analysis)

Listed under `assumed:` in `compact_column_lopes2010.yaml`: solid heat capacity, wall thickness,
wall density and heat capacity, pressure-change rate constant βP (to be fitted to Fig. 5.18a),
ambient temperature (303 K PSA, 323 K breakthrough runs), particle specific area a_p = 2/R_p,
He macropore diffusivity (= H2, compact runs), constant isosteric heat (−ΔH of the virial K_H).
AC5-KS tortuosity (τp = 2) is carried over from chapter 3. The pellet length of the parent R2030 extrudates
(6.45 mm, Table 3.2) is stored for a diagnostic only; no AC5-KS length is reported, so a_p stays 2/R_p.

A calibration of the solid heat capacity and the lumped wall heat capacity on breakthrough runs 1-3 (step 2,
decision 3, 2026-10-09) ended with the wall term on its lower bound (0.5 mm stainless steel), so it was not
adopted: the `assumed:` values are still used and no `calibrated:` block exists. Details in
`docs/validation/step2.md` (`scripts/calibrate_thermal.py`).

PSA boundary conditions (step 3, 2026-10-09; `fitted:` in `compact_column_lopes2010.yaml`, `scripts/fit_psa_bcs.py`):
the pressure-law rate constant beta_P (not reported) is fitted only to the measured cycle-1 pressure history
(Fig. 5.18a) and the purge H2 flow (not reported) only to the H2 flow leaving during the cycle-1 purge (Fig. 5.19);
neither uses purity or recovery. The EP inflow composition is not stated in the thesis (eq. 5.45). The
pre-declared baseline used the gas of the matching ED step (`cycle_B_experiment.ep_inflow: ed`); the corrected model
of the step-3 decision uses pure H2 and the author's beta_P (below; `COR` in `scripts/run_psa.py`).

Inferred modelling choices from reproducing the author's PSA simulations (step 3, 2026-10-09/10; verification
against Table 5.10, Figs 5.20-5.23 and the author's simulation of the experiment (p.256), never against measured
data; basis A = supported by the thesis text, B = needed to reproduce the author's simulations; details in
`docs/validation/step3.md`). Adopted for the corrected model (step-3 decision 2026-10-10):

| choice | value | basis | evidence |
|---|---|---|---|
| beta_P | 0.102 s-1, one constant for ED, Blow and EP | B (+ A, p.180 "same exponential decay rate constant") | fit to the author's simulated sim-13 pressure, Fig. 5.20a (`fitted.pressure_decay_rate_author_sims`); each step alone 0.09-0.11 s-1. Checked, not fitted, against the measured pressure (Fig. 5.18a) |
| EP inflow | pure H2 | B | C_inlet not defined in eq. 5.45; with pure H2 our sims 1, 2, 13 impurity is 1.2-2.4x Table 5.10 vs 7-64x with ED gas; N2 profiles of Figs 5.21e, 5.23b |
| purity | feed components only (He excluded) | A | eq. 5.48 |

Not adopted: per-step-type beta_P fitted to Fig. 5.18a (ED 7.33, Blow 0.078, EP 0.087 s-1; describes the
experiment's valves, moves the model away from the author); ED gas variants for EP (kept in the band).

### Assumptions not resolvable from the source

No input from the original author is available. The following are declared modelling assumptions; both
alternatives of each are carried in the uncertainty band of step 3 (and later steps), and neither is chosen by
matching data.

| assumption | used | alternative in the band | why it cannot be resolved |
|---|---|---|---|
| EP inflow composition | pure H2 (reproduces the author's simulations, basis B) | gas of the matching ED step (tank emulation of the partner bed) | eq. 5.45 leaves C_inlet open; the text never says how a single column was equalised (Fig. 5.5 shows a second column and a tank) |
| gas holdup | interparticle eps C + macropore (1 - eps) eps_p C_m with eps_p = 0.627 (total pellet porosity, Table 4.5) and absolute loadings q (eq. 2.2, p.31), i.e. eqs 5.3 and 5.6 as printed | pore gas net of the adsorbed-phase volume (liquid densities, p.31): changes the H2 released per step by < 1.5 %; and the eps_p = 0.35 diagnostic | at identical pressures and loadings our model releases 27-50 % more H2 per depressurisation step than the author's (Fig. 5.20b); no holdup supported by the text reproduces it, and the implied gas-capacity factor differs per step (0.55-0.70), so it is not a single holdup parameter. Result: our recovery is 4-5 pp below the author's |

### Layered bed (step 4, 2026-10-10)

| item | used | basis / evidence |
|---|---|---|
| HY2SEPS bed porosity in the compact column | 0.394 (AC), 0.391 (ZEO) | ASSUMED: values of the 1.65-cm ch.3 breakthrough columns (Tables 3.10, 3.13); the compact column packing is not reported for these adsorbents |
| HY2SEPS solid heat capacity | 880 (AC), 920 J/kg/K (ZEO) | ASSUMED (not reported); ranges in the YAML |
| LDF factors | Omega_m = 8, Omega_c = 3 (AC), 15 (ZEO) | eq. 3.17 with cylindrical extrudates; p.98 "K_LDF = 3 Dc/rc2 for activated carbons"; p.104 "15 Dc/rc2 for zeolites" |
| HY2SEPS macropore Dp | eq. 5.17 at operating conditions, tau = 2, no Knudsen term | The values quoted on pp.97/103 (e.g. H2 7.21e-5 m2/s) are molecular diffusion in He at 1 bar and 303 K divided by tau = 2 (Chapman-Enskog reproduces them within 3-14 %; p.96: Knudsen neglected, tau = 2). Used literally at 5-10 bar they would be 5-10x too large, so they serve as the check of eq. 5.17, not as model inputs |
| gas thermal conductivity | modified Eucken per species + Bird eq. 9.3-17 mixture rule | ASSUMED: p.171 cites Bird et al. (2002) without the formula |
| scalar Dax and kf of a multicomponent feed | mole-fraction mean of the species' mixture diffusivities (eq. 5.18) | ASSUMED: one Dax and one kf per layer; the thesis does not say which diffusivity enters eqs 5.20/5.23 |
| axial dispersion, all layers and geometries | eq. 3.18 (Table 4.7) | DECIDED (project lead, step 4): eq. 5.20 rejected, see below |
| h_w, U, compact Lopes column (every layer) | Table 5.9: h_w = 55, U = 50 W/m2/K | DECIDED (project lead, step 4): wall heat transfer is a property of the apparatus; the same values as the step-3 baseline |
| h_w, U, Yang (2008) column | h_w = 38.5 (their internal coefficient); U = 12.2 W/m2/K from eq. 5.29 with their external 14.2 W/m2/K and the wall of Table 2 (5.1 mm, AISI 304 conductivity 14.9 W/m/K, ASSUMED) | DECIDED (project lead, step 4): each geometry takes wall heat transfer from its own source. In our model U is the overall wall-ambient coefficient of eq. 5.29, so Yang's two film coefficients are combined through it |
| U by correlation (eq. 5.29, first step-4 run only) | Churchill-Chu with T_w - T_inf = 10 K, air at 300 K, AISI 304 wall (14.9 W/m/K) | ASSUMED inputs (not reported); superseded by the two rows above |
| Yang (2008) emulation | single column, cycle time 200 s (feed 50, ED 10, blow 25, purge 50, press 35 s), P_low 1 bar, 303 K, purge H2 = 10 % of feed H2 | ASSUMED (`layered_4bed_yang2008.yaml: assumed_step4`); trend check e is order-of-magnitude |

**Correlations vs the author's tables (gate 3 of step 4).** Recomputed for the compact-unit AC5-KS runs (Tables 5.7,
5.9): Dp (eq. 5.17), lambda (5.21), hf (5.22) and kf (5.23) reproduce the tabulated values within 0.96-1.21. The
Wakao-Funazkri axial dispersion stated for the PSA model (eq. 5.20) gives 51-55x the tabulated Dax; the ch.3
correlation (eq. 3.18) gives 1.15-1.24x. The tabulated h_w = 55 and U = 50 W/m2/K are the same in every run and
group and are not reproduced by eqs 5.31 (1.8-7.8x) and 5.29 (0.06-0.07x): they look like set values. Per the
pre-declared rule the correlations (eq. 5.20, 5.31, 5.29) were used for the HY2SEPS layers in the first step-4 run
(commit b23e20a), with Dax from eq. 3.18 as the sensitivity.

**Decision (project lead, step 4): eq. 5.20 is rejected, eq. 3.18 is the axial dispersion for every layer and
geometry.** Evidence: (i) eq. 3.18 is the thesis's own correlation (Table 4.7, p.123; eq. 3.18) and reproduces the
Dax of Tables 5.7/5.9 within 1.15-1.24x, whereas eq. 5.20 gives 51-55x; (ii) step 2 used the tabulated Dax and
matched the MEASURED breakthrough curves (breakthrough times 15/18 within 10 %), so the measured data support the
eq. 3.18 magnitude and contradict a 51-55x larger dispersion; (iii) eq. 5.20 (the Wakao-Funazkri form,
eps Dax / Dm = 20 + 0.5 Sc Re) is stated on p.172 for 3 < Re < 10000 and would be applied here at Re 1-2, below and
at the edge of its validity. Eq. 5.20 is not carried as a scenario. Wall heat transfer: see the two DECIDED rows
above. Note for step 5: industrial-scale columns are close to adiabatic.

### Disturbance variables (step 5, 2026-10-11)

Bounds in `configs/disturbances.yaml`; mapping in `src/h2psa/design.py`. [V] = number read in the full text,
[A] = abstract / secondary only (not used for a bound).

| item | used | basis / evidence |
|---|---|---|
| adsorbent capacity factor phi | [0.70, 1.00], multiplies q_max of every species in both layers (K, a kept: loadings scale exactly by phi) | **ASSUMED range.** No peer-reviewed quantification of q_max loss in an H2 PSA bed was found, nor an H2 PSA simulation that reduces q_max by 10-30 %. Basis by analogy: (i) Air Products, *Hydrogen PSA Services and Solutions*, brochure 344-16-011 (2020), p.2: "low level impurities accumulate in adsorbents and cause capacity reduction over time" (qualitative, vendor) [V]; (ii) Azhagapillai et al. (2022), ACS Omega 7:6463, doi:10.1021/acsomega.1c03370, sec. 2.1: water capacity of 5A -8 % after 3 heptane exposures, 4A/5A/13X -62 to -76 % after methanol (harsh TSA conditions) [V]. Reading: 0.90-1.00 normal service; 0.70 end-of-life stress case; losses > 30 % documented only under contamination |
| feed CH4, y_CH4 | [0.0168, 0.0231] (Lopes anchor 0.021 x [0.80, 1.10]) | reformer-digital-twin v3.0 (tag v3.0, commit f356b84; Latham 2011 Plant A, Xu-Froment kinetics), run read-only 2026-10-11 with `rdt.operate` (`run_case` fixed firing, `solve_firing_for_outlet_T` outlet T held at 1105.55 K): reformer-outlet CH4 slip (dry) 6.25-8.62 % over catalyst activity 0.10-0.20 (calibrated base 0.20) x load 0.7-1.1; base 7.81 % (factor 0.80-1.10). Ageing alone at held T_out: 8.11 -> 8.53 %; load at held T_out: flat. The factor is applied to the anchor's CH4: CH4 passes the shift unconverted and the shift dilutes the dry gas uniformly to first order. Cross-check: PSA-inlet CH4 3.0 % (IEAGHG 2017/02, base case stream 7) and 5.13 % (NETL 2022 stream 10) [V] |
| feed CO, y_CO | [0.005, 0.015] | medium-temperature-shift exit 0.5-1.5 mol % CO (Johnson Matthey, *Water gas shift - LTS and MTS*, Americas H2 & Syngas Training Seminar 2025, slide 5; vendor training material) [V]; contains the Lopes feed (1.2 %) and NETL 2022 stream 10 (0.50 %, three shift reactors, conversion up to 97.2 %, p.54) [V]. Not used (different flowsheet, not an operating disturbance): HTS only 2.5-3.5 % "typical", 4.65 % dry in the base-case balance (IEAGHG 2017/02, sec. 2.4, 2.6) [V]; LTS 0.1-0.3 % (JM 2025) [V] |
| composition rule | CO2 = anchor (CH4 + CO + CO2) - y_CH4 - y_CO; N2 of the anchor; H2 by difference | ASSUMED: total carbon of the dry gas kept (CH4 slip and CO slip trade against CO2); H2 then stays at the anchor value |
| second feed anchor | NETL 2022 case 1 stream 10, H2O removed and renormalised (0.7502 H2 / 0.1896 CO2 / 0.0514 CH4 / 0.0050 CO / 0.0038 N2) | run at P_high <= 8.5 bar (design space), NOT at the plant's 24.5 bar: the H2 partial pressure must stay inside the 0-7 bar isotherm fits |

### Simulated vs measured values in Lopes (2010), chapter 5

| quantity | status | source |
|---|---|---|
| PSA experiment: purity 99.98054 %, CO 63 ppm, recovery 81.6 %, productivity 101.1 mol/kg/day | **simulated** (author's model of the experiment) | p.256 "(simulated values)"; practice stated on p.241 |
| PSA experiment: outlet molar flows per species at the GC sample times, cycles 1 and 37 | measured (symbols) | Fig. 5.19, p.193 |
| PSA experiment: pressure, cycle 1 | measured (symbols); line = author's simulation | Fig. 5.18a, p.192 |
| PSA experiment: thermocouples (z = 0.059, 0.151, 0.239 m), cycles 1-10, 1 and 37 | measured (symbols); line = author's simulation | Figs 5.18b, 5.19 |
| PSA experiment: feed flow and composition, step times, pressures | set points | Table 5.4, p.167 |
| PSA experiment: purge flow, pressurisation flow, EP gas | not reported (purge fitted to the measured cycle-1 purge outflow) | — |
| 17 simulations (Table 5.10) and Figs 5.20-5.23 | simulated | pp.194-200 |

The values in `data/benchmarks/lopes2010_psa_experiment.csv` are unchanged; `data/benchmarks/README.md` marks them as
simulated. Earlier step-3 documents called them "experimental" by mistake.

## Known limitations of the data

* Isotherms are fitted on 0–7 bar. In the industrial scenario (NETL, 24.5 bar) the H2 partial pressure
  (~18 bar) is an extrapolation; CO2, CH4, CO and N2 partial pressures stay inside the range.
* The experimental validation (Lopes2010) is a single-adsorbent bed (AC5-KS). The layered bed
  (HY2SEPS-AC + HY2SEPS-ZEO) is validated component-wise, plus qualitative trends from Yang2008.
* Loadings used in check C1 are read off Figures 4.9–4.10 (±5 %).

## Open issues

* **H2 in mixtures (AC5-KS virial).** The H2 virial coefficient B is very large (B1 = 7315.4e11 m⁴ K mol⁻²),
  fitted where pure-H2 loadings are tiny. Through the Taqvi–LeVan mixing rule B_ijk = (B_i+B_j+B_k)/3, it
  pushes predicted H2 co-adsorption close to zero once CO2 is loaded (e.g. 4e-4 mol/kg at 5 bar feed vs
  0.012 mol/kg at 0.5 bar). H2 co-adsorption controls H2 lost in blowdown, so this directly biases recovery.
  Step 2 tested it against the H2–CO2 binary breakthrough curves (runs 1–3): the data cannot distinguish the
  mixture rules (effect on the outlet H2 flow a few per cent, comparable to the flow-measurement offset).
  Decision (2026-10-09): the rule is selectable (`isotherm.mixing`: `taqvi_levan` default, `iast`,
  `noncompetitive_h2`) and all three are carried into step 3 as an uncertainty band; none is selected by
  matching the PSA result.
* **IAST on the virial fits.** The pure virial isotherms of CO, CH4 and N2 have a pressure maximum at
  11–25 bar (beyond the 0–7 bar fit), so their reduced spreading pressure is bounded (N2 2.33, CO 3.95,
  CH4 5.47 mol/kg at 323 K). IAST needs these pure states whenever CO2 is loaded; above the bound the pure
  isotherm is continued at constant (saturated) loading. IAST loadings of the weak species are therefore an
  extrapolation of the fits.

## Licence

Transcribed data: CC BY 4.0. Cite the original sources above when reusing them.
Source PDFs are not redistributed (put them in `literature/`, which is git-ignored).
