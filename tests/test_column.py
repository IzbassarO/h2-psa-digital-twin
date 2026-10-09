"""Step-2 numerical gates for the column model (plus checks of the vectorised isotherms)."""
from pathlib import Path

import numpy as np

from h2psa.column import compact_breakthrough
from h2psa.isotherms import MultisiteLangmuir, Virial, load_yaml
from h2psa.kinetics import arrhenius_fit
from h2psa.validation import breakthrough_times, linear_isothermal_model, stoichiometric_time

P = Path(__file__).resolve().parents[1] / "data" / "params"


def test_virial_equilibrium_matches_scalar_solver():
    v = Virial.from_yaml(P / "ac5ks_lopes2010.yaml")
    ys = [{"H2": 0.7, "CO2": 0.3}, {"H2": 0.73, "CO2": 0.23, "CH4": 0.021, "CO": 0.012, "N2": 0.007}, {"N2": 1.0}]
    for y in ys:
        for Ptot, T in [(5.0, 323.0), (0.5, 303.0), (5.0, 360.0)]:
            ref = v.loading_mix(y, Ptot, T)
            Pv = np.array([[y.get(n, 0.0) * Ptot for n in v.names]])
            q = v.equilibrium(Pv, np.array([T]))[0]
            for i, n in enumerate(v.names):
                if y.get(n, 0.0) > 0:
                    assert abs(q[i] / ref[n] - 1) < 1e-8, (y, Ptot, T, n)


def test_msl_equilibrium_matches_scalar_solver():
    m = MultisiteLangmuir.from_yaml(P / "hy2seps_ac_lopes2010.yaml")
    y = {"H2": 0.73, "CO2": 0.23, "CH4": 0.021, "CO": 0.012, "N2": 0.007}
    ref = m.loading_mix(y, 5.0, 323.0)
    q = m.equilibrium(np.array([[y[n] * 5.0 for n in m.names]]), np.array([323.0]))[0]
    for i, n in enumerate(m.names):
        assert abs(q[i] / ref[n] - 1) < 1e-8, n


def test_builder_micropore_kinetics_errata_e7():
    """Table 5.7 constants for all species except those listed in micropore_use_measured (CH4: Table 4.10 refit)."""
    ac = load_yaml(P / "ac5ks_lopes2010.yaml")
    lay = compact_breakthrough(5, N=4).col.layers[0]
    assert ac["psa_transport"]["micropore_use_measured"] == ["CH4"]
    assert (lay.Dc0["CH4"], lay.Ea["CH4"]) == arrhenius_fit(ac["micropore_kinetics_breakthrough"]["CH4"]["Dc_over_rc2"])
    for s in ("H2", "CO2", "CO", "N2"):
        assert lay.Dc0[s] == ac["psa_transport"]["micropore"][s]["Dc0_over_rc2"]
    assert compact_breakthrough(5, N=4, measured_kinetics=[]).col.layers[0].Dc0["CH4"] == 35.52


def test_jacobian_sparsity_pattern_is_complete():
    m = compact_breakthrough(5, N=8)
    y = m.y0 + 1e-3 * m.yscale * np.random.default_rng(0).random(m.nv)
    S = m.sparsity().toarray() != 0
    f0 = m.rhs(0.0, y)
    for k in range(m.nv):
        yp = y.copy()
        yp[k] += 1e-6 * m.yscale[k]
        dep = np.abs(m.rhs(0.0, yp) - f0) > 1e-12 * (1 + np.abs(f0))
        assert not (dep & ~S[:, k]).any(), f"column {k}: dependence outside the sparsity pattern"


def test_gate_mole_balance_full_breakthrough():
    """Overall and per-component mole balance closes to < 0.1 % over a full run (5 components + He)."""
    r = compact_breakthrough(5, N=60).solve(6000.0, n_out=61)
    for k, v in r.closure.items():
        assert abs(v) < 1e-3, (k, v)


def test_gate_grid_convergence():
    """CO2 breakthrough time (C/C0 = 0.5) changes < 1 % between N and 2N cells."""
    t50 = []
    for N in (60, 120):
        m = compact_breakthrough(1, N=N)
        r = m.solve(1500.0, n_out=1501)
        i = m.sp.index("CO2")
        t50.append(breakthrough_times(r.t, r.outlet_flow[i] / (m.feed.flow * m.y_feed[i]))[1])
    assert abs(t50[1] / t50[0] - 1) < 0.01, t50


def test_gate_linear_isotherm_stoichiometric_time():
    """Single adsorbing component, isothermal, linear isotherm: breakthrough at the stoichiometric time."""
    m = linear_isothermal_model(N=60)
    t_st = stoichiometric_time(m)
    r = m.solve(3 * t_st, n_out=3001)
    f = r.outlet_flow[0] / (m.feed.flow * m.y_feed[0])
    assert abs(np.trapezoid(1 - f, r.t) / t_st - 1) < 0.01      # first moment
    assert abs(breakthrough_times(r.t, f)[1] / t_st - 1) < 0.01  # C/C0 = 0.5
