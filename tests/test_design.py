"""Step 5: design / disturbance variables -> model (src/h2psa/design.py), capacity factor phi, early stopping."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from h2psa import design
from h2psa.column import hy2seps_layer
from h2psa.cycle import CycleRunner, censored

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "results" / "step4" / "css_states"
need_step4 = pytest.mark.skipif(not (STATES / "base_f0.6_t101.npz").exists(), reason="step-4 CSS states not present")


def test_feed_composition():
    y0 = design.anchor_feed("lopes")
    assert design.feed_composition("lopes") == y0
    y = design.feed_composition("lopes", y_CH4=0.025, y_CO=0.005)
    assert abs(sum(y.values()) - 1) < 1e-15 and y["CH4"] == 0.025 and y["CO"] == 0.005 and y["N2"] == y0["N2"]
    assert abs(y["CH4"] + y["CO"] + y["CO2"] - (y0["CH4"] + y0["CO"] + y0["CO2"])) < 1e-15   # carbon kept
    assert abs(y["H2"] - y0["H2"]) < 1e-15                    # constant carbon and N2: H2 unchanged
    n = design.anchor_feed("netl_dry")
    assert list(n) == list(y0) and abs(sum(n.values()) - 1) < 1e-15 and abs(n["CH4"] - 0.0513 / 0.9975) < 1e-12
    with pytest.raises(ValueError):
        design.feed_composition("lopes", y_CH4=0.5)


def test_phi_scales_capacity():
    tr = dict(kf=0.02, hf=150.0, Dp={}, Dax=1e-5, **{"lambda": 0.8}, hw=55.0, U=50.0)
    P = np.array([[0.5, 1.0, 0.1, 0.06, 0.03]])
    T = np.array([303.0])
    for kind in ("AC", "ZEO"):
        q1, q7 = (hy2seps_layer(kind, 0.1, tr, phi).iso.equilibrium(P, T) for phi in (1.0, 0.7))
        assert np.allclose(q7, 0.7 * q1, rtol=1e-12, atol=0)


def test_cycle_rules():
    c, m, w = design.build(dict(P_high=6.0, P_low=0.4, n_equalisations=2, purge_to_feed=0.1, t_blow=80.0,
                                f_zeolite=0.5, feed_flow=1.2))
    assert np.allclose(c.P_inter, [6e5 - 5.6e5 / 3, 6e5 - 2 * 5.6e5 / 3]) and c.t_blow == 80.0
    assert np.isclose(c.purge_flow * c.t_purge, 0.1 * c.feed_flow * c.y_feed["H2"] * c.t_feed)
    assert set(m.layer_of_cell[:15]) == {0} and set(m.layer_of_cell[15:]) == {1}
    c, _, _ = design.build({})                                 # experiment: cycle B, Table 5.4 pressures
    assert c.P_inter == [3.875e5, 2.75e5, 1.625e5]
    with pytest.raises(ValueError, match="multiple of 1/30"):
        design.build(dict(f_zeolite=0.41))


def test_censoring_rule():
    rising = list(np.linspace(900, 1500, 120))
    assert censored(rising) and not censored(rising[:99])                  # needs >= 100 cycles
    assert not censored(rising[::-1])                                      # falling towards CSS: keep going
    assert not censored([500.0] * 150)                                     # below 10x the limit
    assert not censored([2000.0] * 150)                                    # not increasing


@need_step4
def test_phi_one_is_step4_baseline():
    """phi = 1 through design.build is the step-4 baseline: one cycle from its CSS state is bitwise the cycle of
    the step-4 builder (scripts/run_layered.py), and stays at the step-4 CSS."""
    spec = importlib.util.spec_from_file_location("run_layered", ROOT / "scripts" / "run_layered.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    y0 = np.load(STATES / "base_f0.6_t101.npz")["y"]
    out = []
    for c, m, w in (rl.build(dict(f_ac=0.6, t_feed=101.0)), design.build(dict(f_zeolite=0.4, phi=1.0))):
        out.append(CycleRunner(m, c, w_ads=w).run_cycle(y0))
    assert np.array_equal(out[0].y_end, out[1].y_end)
    ref = rl.load_case("base_f0.6_t101")["perf"]
    assert abs(out[1].perf["ppm"] / ref["ppm"] - 1) < 1e-4 and abs(out[1].perf["recovery_pct"] - ref["recovery_pct"]) < 1e-3


@need_step4
@pytest.mark.parametrize("case,point", [("base_f1.0_t80", dict(f_zeolite=0.0, t_feed=80.0)),
                                        ("base_f0.6_t101", dict(f_zeolite=0.4)),
                                        ("base_f0.4_t60", dict(f_zeolite=0.6, t_feed=60.0))])
def test_phi_monotone_short(case, point):
    """Lower capacity, more impurity: 3 cycles from the step-4 CSS state at phi = 1, 0.85, 0.7 (the CSS-level
    check on the same 3 cases is in docs/validation/step5.md)."""
    y0 = np.load(STATES / f"{case}.npz")["y"]
    ppm = []
    for phi in (1.0, 0.85, 0.7):
        c, m, w = design.build({**point, "phi": phi})
        r, y = CycleRunner(m, c, w_ads=w), y0
        for _ in range(3):
            res = r.run_cycle(y)
            y = res.y_end
        ppm.append(res.perf["ppm"])
    assert ppm[0] < ppm[1] < ppm[2]
