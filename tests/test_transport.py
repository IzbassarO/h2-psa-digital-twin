"""Transport correlations (h2psa.transport): gas properties and the coefficients of Tables 5.7 / 5.9."""
from pathlib import Path

import numpy as np
import pytest

from h2psa.isotherms import R, load_yaml
from h2psa.transport import Gas, coefficients, pore_diffusivity

P = Path(__file__).resolve().parents[1] / "data" / "params"


def test_gas_properties_against_references():
    g = Gas(["H2", "CO2", "CH4", "CO", "N2", "He"])
    nist = load_yaml(P / "gas_properties_nist.yaml")
    mu_nist = np.array([np.interp(303.0, nist["T"], nist["species"][s]["viscosity"]) for s in g.sp])
    assert np.allclose(g.viscosity(303.0), mu_nist, rtol=0.03)            # eq. (3.24) vs NIST
    D = g.binary_diffusivity(298.0, 101325.0)
    assert D[0, 4] == pytest.approx(0.78e-4, rel=0.05)                    # H2-N2, Bird et al. Table 17.1-1
    assert D[0, 4] == pytest.approx(D[4, 0])


def test_bosanquet_limits():
    Dm = np.array([1e-5, 2e-5])
    assert np.allclose(pore_diffusivity(Dm, 2.0), Dm / 2)                 # no Knudsen term
    assert np.all(pore_diffusivity(Dm, 2.0, 300.0, np.array([2.0, 44.0]), 1e-9) < Dm / 2)


@pytest.mark.parametrize("kind", ["ac", "zeo"])
def test_hy2seps_text_macropore_diffusivities(kind):
    """pp.97 and 103: Dp at 303 K, 1 bar in He = D_i,He / tau with tau = 2 (Knudsen neglected, p.96)."""
    d = load_yaml(P / f"hy2seps_{kind}_lopes2010.yaml")
    for s, v in d["macropore_diffusivity_303K"].items():
        g = Gas([s, "He"])
        Dm = g.molecular_diffusivity([0.005, 0.995], 303.0, 1e5)[0]
        assert pore_diffusivity(Dm, d["physical"]["tortuosity"]) == pytest.approx(v, rel=0.15)


def test_table_5_7_pore_diffusivity_heat_dispersion_and_film_heat():
    """Breakthrough runs 1-5 (5 bar, 323 K): Dp, lambda and hf reproduced within 5 %."""
    cc = load_yaml(P / "compact_column_lopes2010.yaml")
    bt, col = cc["breakthrough_runs"], cc["column"]
    std = bt["std_conditions"]
    flow = std["P_bar"] * 1e5 * bt["flow_rate_std"] / (R * std["T"])
    T, Pr = bt["feed_temperature"], bt["pressure_bar"] * 1e5
    u0 = flow * R * T / (Pr * np.pi * col["internal_diameter"] ** 2 / 4)
    g = Gas(["H2", "CO2", "CH4", "CO", "N2"])
    for run in bt["runs"].values():
        c = coefficients(g, [run["y"].get(s, 0.0) for s in g.sp], T, Pr, u0, dp=2.9e-3, eps=col["bed_porosity"],
                         d_wi=col["internal_diameter"], e_wall=0.003, L=col["bed_height"], tau=2.0)
        for s, v in run["Dp"].items():
            assert c["Dp"][s] == pytest.approx(v, rel=0.05)
        assert c["lambda"] == pytest.approx(run["lambda"], rel=0.05)
        assert c["hf"] == pytest.approx(run["hf"], rel=0.05)
