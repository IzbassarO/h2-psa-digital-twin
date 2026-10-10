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
