# Benchmarks

* `lopes2010_psa_experiment.csv` — single-column PSA run on AC5-KS (thesis p.191). **The purity, CO, recovery and
  productivity are the author's SIMULATED values for the experiment (p.256: "(simulated values)"; practice stated on
  p.241), not measurements.** Measured data of this run: `digitized/lopes2010_psa_*` (outlet flows at the GC sampling
  times, pressure, temperatures).
* `digitized/lopes2010_sim13_*_authorsim.csv` — the author's simulation 13 at CSS (Figs 5.20, 5.21, 5.23): pressure,
  outlet flows, loading profiles at the end of every step, gas temperature and N2 profiles at the end of the feed.
* `lopes2010_psa_simulations.csv` — the author's 17 gPROMS simulations (Tables 5.8 and 5.10, pp.176 and 195).
  Same column and feed as the experiment (feed 0.79 H2 / 0.17 CO2 / 0.021 CH4 / 0.012 CO / 0.007 N2,
  5.00e-5 m3 s-1 at 298 K and 1 bar). Cycle A = 6 steps, one equalisation (t_eq per step = 45 s);
  cycle B = 10 steps, three equalisations (t_eq per step = 15 s). `P_inter_bar` lists intermediate pressures
  separated by `;`. Note: Table 5.10 footnote c prints "Pinter2" twice; the third value is Pinter3.
* Geometry/trend benchmark for the layered bed: `data/params/layered_4bed_yang2008.yaml`.
