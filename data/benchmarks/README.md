# Benchmarks

* `lopes2010_psa_experiment.csv` — measured single-column PSA run on AC5-KS (thesis p.191). **Primary validation target.**
* `lopes2010_psa_simulations.csv` — the author's 17 gPROMS simulations (Tables 5.8 and 5.10, pp.176 and 195).
  Same column and feed as the experiment (feed 0.79 H2 / 0.17 CO2 / 0.021 CH4 / 0.012 CO / 0.007 N2,
  5.00e-5 m3 s-1 at 298 K and 1 bar). Cycle A = 6 steps, one equalisation (t_eq per step = 45 s);
  cycle B = 10 steps, three equalisations (t_eq per step = 15 s). `P_inter_bar` lists intermediate pressures
  separated by `;`. Note: Table 5.10 footnote c prints "Pinter2" twice; the third value is Pinter3.
* Geometry/trend benchmark for the layered bed: `data/params/layered_4bed_yang2008.yaml`.
