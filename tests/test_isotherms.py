"""Mixture rules of the AC5-KS virial isotherm (step-2 decision 2)."""
from pathlib import Path

import numpy as np

from h2psa.isotherms import Virial

YAML = Path(__file__).resolve().parents[1] / "data" / "params" / "ac5ks_lopes2010.yaml"
# H2 CO2 CO CH4 N2 partial pressures [bar]: run-5 feed, run-1 binary, 80 % CO2 (N2 trace beyond its psi_max), PSA feed
STATES = np.array([[3.65, 1.15, 0.06, 0.105, 0.035], [3.5, 1.5, 0, 0, 0], [1.0, 4.0, 0, 0, 0.01],
                   [3.95, 0.85, 0.06, 0.105, 0.035]])
TEMPS = np.array([323.0, 323.0, 323.0, 303.0])


def test_iast_reduces_to_pure_isotherm():
    v = Virial.from_yaml(YAML, mixing="iast")
    for i, n in enumerate(v.names):
        for P in (0.01, 1.0, 7.0):
            Pv = np.zeros((1, len(v.names)))
            Pv[0, i] = P
            assert abs(v.equilibrium(Pv, np.array([323.0]))[0, i] / v.loading_pure(n, P, 323.0) - 1) < 1e-9, (n, P)


def test_iast_is_thermodynamically_consistent():
    """Gibbs adsorption isotherm: d psi / d ln P_i = q_i at constant T (psi = reduced spreading pressure)."""
    v = Virial.from_yaml(YAML, mixing="iast")
    q = v.equilibrium(STATES, TEMPS)
    h = 1e-5
    for i in range(len(v.names)):
        live = STATES[:, i] > 0
        up, dn = STATES.copy(), STATES.copy()
        up[:, i] *= np.exp(h)
        dn[:, i] *= np.exp(-h)
        v.equilibrium(up, TEMPS)
        psi_up = np.exp(v._s_last)
        v.equilibrium(dn, TEMPS)
        dpsi = (psi_up - np.exp(v._s_last)) / (2 * h)
        assert np.allclose(dpsi[live], q[live, i], rtol=1e-5, atol=1e-12), (v.names[i], dpsi, q[:, i])


def test_mixing_rules_agree_in_the_henry_limit_and_bracket_h2():
    tl, ia, nc = (Virial.from_yaml(YAML, mixing=k) for k in Virial.MIXING)
    low = STATES * 1e-6
    for v in (ia, nc):
        assert np.allclose(v.equilibrium(low, TEMPS), tl.equilibrium(low, TEMPS), rtol=1e-4)
    q = {k: v.equilibrium(STATES, TEMPS)[:, 0] for k, v in zip(Virial.MIXING, (tl, ia, nc))}
    pure = [nc.loading_pure("H2", p, T) for p, T in zip(STATES[:, 0], TEMPS)]
    assert np.allclose(q["noncompetitive_h2"], pure, rtol=1e-9)
    assert (q["taqvi_levan"] < q["iast"]).all() and (q["iast"] < q["noncompetitive_h2"]).all()
