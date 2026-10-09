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


# ------------------------------------------------------------------ variants of a built model
def rebuild(m: ColumnModel, col: dict | None = None, **layer) -> ColumnModel:
    """Same run with some Column fields (`col`) and/or fields of its single layer replaced."""
    c = replace(m.col, **(col or {}), layers=[replace(m.col.layers[0], **layer)])
    return ColumnModel(c, m.feed, m.sp, N=m.N, y0={"He": 1.0}, isothermal=m.isothermal)


# ------------------------------------------------------------------ thermal calibration (step-2 decision 3)
FIT_RUNS = (1, 2, 3)   # training runs; 4-5 and the PSA experiment are held out
T_FIT = 2500.0         # s; the thermal wave and the CO2 front of runs 1-3 are over by then
T_READ = 2.4           # s, time read-off uncertainty of the digitised figures (CSV headers)


def sigma_thermocouple(T_feed: float, runs=FIT_RUNS, t_late: float = 4000.0) -> float:
    """Thermocouple uncertainty [K]: RMS deviation from the feed temperature once the bed is back at it
    (t > t_late), over all thermocouples of `runs` (offset + noise)."""
    d = pd.concat([load_exp(r, "T") for r in runs])
    return float(np.sqrt(((d[d.time_s > t_late].temperature_K - T_feed) ** 2).mean()))


def exp_crossings(m: ColumnModel, run: int, species: str) -> list[tuple[float, float]]:
    """Measured [(t10, sigma), (t50, sigma)]: sigma = half the sampling interval bracketing the crossing,
    combined with the read-off uncertainty."""
    e = exp_flow(run, species)
    t, f = e.time_s.values, e.flow.values / (m.feed.flow * m.y_feed[m.sp.index(species)])
    out = []
    for lev in (0.1, 0.5):
        k = int(np.argmax(f >= lev))
        out.append((crossing_time(t, f, lev), float(np.hypot((t[k] - t[k - 1]) / 2, T_READ))))
    return out


def thermal_residuals(m: ColumnModel, run: int, sT: float, t_end: float = T_FIT, r=None) -> np.ndarray:
    """Normalised residuals of one run: each thermocouple history (0 <= t <= t_end) as
    (T_sim - T_exp) / (sT sqrt(n)), so a history weighs as one observation of its RMS error, and the
    CO2 t10, t50 as (t_sim - t_exp) / sigma_t. Pass a solved Result `r` to skip the simulation."""
    r = r if r is not None else m.solve(t_end, n_out=int(t_end / 2) + 1)
    d = load_exp(run, "T")
    out = []
    for tc, z in TC_Z.items():
        g = d[(d.series == tc) & (d.time_s >= 0) & (d.time_s <= t_end)]
        out.append((np.interp(g.time_s, r.t, r.T_at(z)) - g.temperature_K.values) / (sT * np.sqrt(len(g))))
    i = m.sp.index("CO2")
    ts = breakthrough_times(r.t, r.outlet_flow[i] / (m.feed.flow * m.y_feed[i]))
    out.append(np.array([(a - te) / st for a, (te, st) in zip(ts, exp_crossings(m, run, "CO2"))]))
    return np.concatenate(out)
