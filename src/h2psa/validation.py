"""Helpers shared by scripts/run_breakthrough.py and tests/test_column.py (step-2 validation)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .column import ColumnModel, Feed, compact_breakthrough, crossing_time
from .isotherms import R, Linear

DIGITIZED = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "digitized"
TC_Z = {"Bottom": 0.059, "Middle": 0.151, "Top": 0.239}   # thermocouples, Table 5.3


def load_exp(run: int, panel: str) -> pd.DataFrame | None:
    """Digitised data: panel 'flow' | 'zoom' | 'T'. Flows converted to mol/s. None if absent."""
    f = DIGITIZED / f"lopes2010_run{run}_{panel}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, comment="#")
    if "molar_flow_mmol_s" in d:
        d["flow"] = d.pop("molar_flow_mmol_s") * 1e-3
    return d.sort_values(["series", "time_s"]).reset_index(drop=True)


def exp_flow(run: int, species: str) -> pd.DataFrame:
    """Outlet flow samples of one species; minor species from the zoomed panel (finer resolution)."""
    zoom = load_exp(run, "zoom")
    use_zoom = zoom is not None and species not in ("H2", "CO2") and species in set(zoom.series)
    d = zoom if use_zoom else load_exp(run, "flow")
    return d[d.series == species]


def exp_temperature_peaks(run: int, window: int = 5) -> dict:
    """{tc: (peak T [K], time [s])} after a rolling median (removes single-sample noise)."""
    d = load_exp(run, "T")
    out = {}
    for tc in TC_Z:
        g = d[d.series == tc]
        T = g.temperature_K.rolling(window, center=True, min_periods=1).median().values
        k = int(np.argmax(T))
        out[tc] = (float(T[k]), float(g.time_s.values[k]))
    return out


def breakthrough_times(t, f) -> tuple[float, float]:
    """(t10, t50) of a normalised outlet flow f = F/F_feed."""
    return crossing_time(t, f, 0.1), crossing_time(t, f, 0.5)


# ------------------------------------------------------------------ analytical limit
def linear_isothermal_model(N: int, K: float = 1.0, y_feed: float = 0.01) -> ColumnModel:
    """Compact-unit column, run-1 transport, one adsorbing tracer ('CO2' slot) with a linear isotherm
    q = K P in He, isothermal, fast micropores. Feed dilute so the flow is ~constant."""
    m = compact_breakthrough(1, N=N)
    lay = m.col.layers[0]
    lin = replace(lay, iso=Linear({"CO2": K}), Dc0={"CO2": 1e3}, Ea={"CO2": 0.0},
                  Dp={"CO2": lay.Dp["CO2"], "He": lay.Dp["He"]})
    col = replace(m.col, layers=[lin])
    feed = Feed(y={"CO2": y_feed, "He": 1 - y_feed}, T=m.feed.T, flow=m.feed.flow, P_out=m.feed.P_out)
    return ColumnModel(col, feed, ["CO2", "He"], N=N, y0={"He": 1.0}, isothermal=True)


def stoichiometric_time(m: ColumnModel) -> float:
    """t_st = L [eps + (1-eps)(eps_p + rho_p K R T / 1e5)] C_T / N_in for the linear isotherm of m."""
    lay = m.col.layers[0]
    Kc = lay.iso.K[0] * R * m.feed.T / 1e5             # dq/dC, m3 kg-1
    CT = m.feed.P_out / (R * m.feed.T)
    return lay.length * (lay.eps + (1 - lay.eps) * (lay.eps_p + lay.rho_p * Kc)) * CT / m.N_in


# ------------------------------------------------------------------ diagnostic isotherm (open issue)
class NoncompetitiveH2:
    """Diagnostic only: the base mixture isotherm, but H2 adsorbs as if alone at its partial pressure
    (upper bound on H2 co-adsorption). Used to size the effect of the virial H2 mixing rule."""

    def __init__(self, base):
        self.base, self.names, self.species = base, base.names, base.species
        self.iH2 = base.names.index("H2")

    def equilibrium(self, P, T):
        q = self.base.equilibrium(P, T)
        Ph = np.full_like(P, 1e-14)
        Ph[:, self.iH2] = P[:, self.iH2]
        # separate instance so the warm start of the mixture solve is not disturbed
        if not hasattr(self, "_pure"):
            self._pure = type(self.base)(self.base.species, self.base.S)
        q[:, self.iH2] = self._pure.equilibrium(Ph, T)[:, self.iH2]
        return q
