"""Single-column PSA cycle (h2psa.cycle): boundary laws, mole balance, EP gas, eqs. (5.48)-(5.50)."""
import numpy as np
import pytest

from h2psa.cycle import CycleRunner, experiment_cycle, psa_model


@pytest.fixture(scope="module")
def two_cycles():
    c = experiment_cycle()
    m = psa_model(c, N=10)
    run = CycleRunner(m, c)
    r1 = run.run_cycle(m.y0)
    r2 = run.run_cycle(r1.y_end, trace=True)
    return c, m, run, r2


def test_pressure_laws(two_cycles):
    """End pressures follow eqs. (5.39)-(5.40) (exponential, from the step-start pressure) and (5.47)."""
    c, m, _, r = two_cycles
    tr = r.trace
    at_end = lambda step, key: tr[key][tr["step"] == step][-1]
    at_start = lambda step, key: tr[key][tr["step"] == step][0]
    for k, step in enumerate(("ED1", "ED2", "ED3"), start=1):
        P0, P1 = at_start(step, "PL"), at_end(step, "PL")
        law = c.P_inter[k - 1] + (P0 - c.P_inter[k - 1]) * np.exp(-c.beta_P * c.t_eq)
        assert P1 == pytest.approx(law, rel=2e-3)
    P0 = at_start("Blow", "P0")
    assert at_end("Blow", "P0") == pytest.approx(c.P_low + (P0 - c.P_low) * np.exp(-c.beta_P * c.t_blow), rel=2e-3)
    assert at_end("Press", "PL") == pytest.approx(c.P_high, rel=2e-3)
    assert at_end("Purge", "P0") == pytest.approx(c.P_low, rel=2e-3)


def test_cycle_mole_balance(two_cycles):
    assert abs(two_cycles[3].perf["closure"]) < 1e-5


def test_ep_inflow_is_ed_gas(two_cycles):
    """Default EP inflow = well-mixed gas of the matching ED step of the same cycle."""
    _, m, _, r = two_cycles
    for k in (1, 2, 3):
        ed, ep = np.clip(r.steps[f"ED{k}"][1], 0, None), -r.steps[f"EP{k}"][1]   # out / in at z = L
        assert ep.sum() > 0 and np.allclose(ep / ep.sum(), ed / ed.sum(), atol=1e-6)


def test_ep_inflow_h2_option():
    c = experiment_cycle(ep_gas="h2")
    m = psa_model(c, N=10)
    r = CycleRunner(m, c).run_cycle(m.y0)
    ep = -r.steps["EP3"][1]
    assert ep[m.sp.index("H2")] / ep.sum() == pytest.approx(1.0, abs=1e-9)


def test_performance_definitions(two_cycles):
    """Purity, recovery and productivity exactly as eqs. (5.48)-(5.50), from the step mole integrals."""
    c, m, run, r = two_cycles
    iH = m.sp.index("H2")
    prod = r.steps["Feed"][1]
    h2_in_press, h2_in_purge = -r.steps["Press"][1][iH], -r.steps["Purge"][1][iH]
    net = prod[iH] - h2_in_press - h2_in_purge
    assert r.perf["purity_pct"] == pytest.approx(100 * prod[iH] / prod.sum())
    assert r.perf["recovery_pct"] == pytest.approx(100 * net / r.steps["Feed"][0][iH])
    assert r.perf["productivity"] == pytest.approx(net / run.w_ads * 86400 / c.time)
    assert h2_in_press > 0 and h2_in_purge == pytest.approx(c.purge_flow * c.t_purge, rel=1e-6)
    assert r.steps["Feed"][0][iH] == pytest.approx(c.feed_flow * c.y_feed["H2"] * c.t_feed, rel=1e-9)
