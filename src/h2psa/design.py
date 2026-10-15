"""Design and disturbance variables -> layered-bed PSA model (step 5).

Decision variables: configs/design_space.yaml; disturbances (capacity factor phi, feed CH4 and CO):
configs/disturbances.yaml. A point is a dict with any of those names (pressures in bar, as in the configs);
missing names take the experiment's value (Table 5.4, frozen step-3 cycle) or the nominal disturbance, so
build({"f_zeolite": 0.4}) is the step-4 baseline case base_f0.6_t101.
Optional key "scenario": a Step 7b model-form scenario (SCENARIOS; docs/preregistration.md, Addendum A); absent =
the frozen baseline.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .column import PARAMS, layered_column, wall_heat_capacity
from .cycle import experiment_cycle, psa_model
from .isotherms import load_yaml

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
N_CELLS = 30
SCENARIOS = ("ep_inflow_ed_gas", "thermal_range_low", "thermal_range_high")


def anchor_feed(anchor: str = "lopes") -> dict:
    """Nominal feed (dry basis, species order of the Lopes feed): "lopes" (Table 5.4) or "netl_dry" (NETL 2022
    case 1, stream 10, H2O removed and renormalised)."""
    lopes = load_yaml(PARAMS / "compact_column_lopes2010.yaml")["feed"]["composition"]
    if anchor == "lopes":
        return dict(lopes)
    if anchor == "netl_dry":
        y = load_yaml(PARAMS / "industrial_feed_netl2022.yaml")["psa_feed"]["composition"]
        tot = sum(y[s] for s in lopes)
        return {s: y[s] / tot for s in lopes}
    raise ValueError(anchor)


def feed_composition(anchor: str = "lopes", y_CH4: float | None = None, y_CO: float | None = None) -> dict:
    """Anchor feed with CH4 and/or CO set: CO2 keeps the anchor's total carbon (CH4 + CO + CO2), N2 is kept and
    H2 is by difference. Without y_CH4 / y_CO the anchor itself (unchanged floats)."""
    y = anchor_feed(anchor)
    if y_CH4 is None and y_CO is None:
        return y
    carbon = y["CH4"] + y["CO"] + y["CO2"]
    y["CH4"] = y["CH4"] if y_CH4 is None else y_CH4
    y["CO"] = y["CO"] if y_CO is None else y_CO
    y["CO2"] = carbon - y["CH4"] - y["CO"]
    y["H2"] = 1.0 - sum(v for k, v in y.items() if k != "H2")
    if min(y.values()) < 0:
        raise ValueError(f"negative mole fraction: {y}")
    return y


def scenario_settings(name: str | None) -> tuple[dict, dict | None, float | None]:
    """(cycle overrides, solid heat capacity per layer, wall heat capacity per length) of a Step 7b scenario.
    ep_inflow_ed_gas: EP inflow = gas of the matching ED step (step-3 scenario; beta_P unchanged).
    thermal_range_low / _high: both layers' Cps and the wall at the ends of their assumed ranges, applied as in
    step 3 (Cw of the end wall thickness, e_wall kept; scripts/run_psa.py band_thermal_*). None: baseline."""
    if name is None:
        return {}, None, None
    if name == "ep_inflow_ed_gas":
        return {"ep_gas": "ed"}, None, None
    if name in ("thermal_range_low", "thermal_range_high"):
        end = 0 if name == "thermal_range_low" else 1
        cps = {k: load_yaml(PARAMS / f)["physical"]["solid_heat_capacity"]["range"][end]
               for k, f in (("AC", "hy2seps_ac_lopes2010.yaml"), ("ZEO", "hy2seps_zeo_lopes2010.yaml"))}
        e = load_yaml(PARAMS / "compact_column_lopes2010.yaml")["assumed"]["wall_thickness"]["range"][end]
        return {}, cps, wall_heat_capacity(e)
    raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")


def compact_geometry() -> dict:
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    e = cc["assumed"]["wall_thickness"]["value"]
    return dict(L=cc["column"]["bed_height"], d_wi=cc["column"]["internal_diameter"], e_wall=e, Cw=wall_heat_capacity(e))


def build(point: dict, N: int = N_CELLS):
    """(Cycle, ColumnModel, adsorbent mass [kg]) of a design point (see module docstring)."""
    p = point
    cycle_over, Cps, Cw = scenario_settings(p.get("scenario"))
    c = replace(experiment_cycle(), **cycle_over)
    y = feed_composition(p.get("anchor", "lopes"), p.get("y_CH4"), p.get("y_CO"))
    P_high = p.get("P_high", c.P_high / 1e5) * 1e5
    P_low = p.get("P_low", c.P_low / 1e5) * 1e5
    n = int(p.get("n_equalisations", len(c.P_inter)))
    c = replace(c, kind={1: "A", 3: "B"}.get(n, str(n)), t_feed=p.get("t_feed", c.t_feed), t_blow=p.get("t_blow", c.t_blow),
                P_high=P_high, P_low=P_low, P_inter=[P_high - k * (P_high - P_low) / (n + 1) for k in range(1, n + 1)],
                y_feed=y, feed_flow=c.feed_flow * p.get("feed_flow", 1.0))
    if "purge_to_feed" in p:
        c = replace(c, purge_flow=p["purge_to_feed"] * c.feed_flow * y["H2"] * c.t_feed / c.t_purge)
    fz = p.get("f_zeolite", 0.4)
    k = round(fz * N)
    if abs(fz * N - k) > 1e-9:
        raise ValueError(f"f_zeolite = {fz} is not a multiple of 1/{N} (layer interface off a cell face)")
    tr = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")["psa_transport"]["groups"]["1-4"]   # h_w, U of the apparatus
    sp = list(y) + ["He"]
    geometry = {**compact_geometry(), **({} if Cw is None else {"Cw": Cw})}
    col = layered_column(1.0 - k / N, y, c.T_feed, c.P_high, c.feed_flow, sp, T_amb=c.T_feed, dax="eq3.18",
                         wall=(tr["hw"], tr["U"]), phi=p.get("phi", 1.0), Cps=Cps, **geometry)
    m = psa_model(c, N=N, column=col)
    return c, m, float((m.A * m.dz * (1 - m.eps) * m.rho_p).sum())


def scaled(point: dict) -> np.ndarray:
    """Point in [0, 1]^d over the decision and disturbance bounds (missing names: build's defaults);
    distance measure for nearest-neighbour warm starts."""
    ds = load_yaml(CONFIGS / "design_space.yaml")["decision_variables"]
    dist = load_yaml(CONFIGS / "disturbances.yaml")["variables"]
    c = experiment_cycle()
    default = dict(t_feed=c.t_feed, t_blow=c.t_blow, P_high=c.P_high / 1e5, P_low=c.P_low / 1e5, f_zeolite=0.4,
                   n_equalisations=len(c.P_inter), feed_flow=1.0,
                   purge_to_feed=c.purge_flow * c.t_purge / (c.feed_flow * c.y_feed["H2"] * c.t_feed))
    y0 = anchor_feed(point.get("anchor", "lopes"))
    default.update(phi=1.0, y_CH4=y0["CH4"], y_CO=y0["CO"])
    out = []
    for name, v in {**ds, **dist}.items():
        lo, hi = (min(v["values"]), max(v["values"])) if "values" in v else v["bounds"]
        out.append((point.get(name, default[name]) - lo) / (hi - lo))
    return np.array(out)

