"""Single-column PSA cycle to cyclic steady state (CSS) — Lopes (2010) ch.5.

Cycle A (Feed, ED, Blow, Purge, EP, Press) and cycle B (Feed, ED1-3, Blow, Purge, EP3-1, Press), Figs 5.7
and 5.9; boundary conditions eqs. (5.36)-(5.47) pp.178-180; performance eqs. (5.48)-(5.50) p.180.

* ED, Blow and EP: exponential pressure law of eqs. (5.39)-(5.40), one rate constant beta_P for all of them
  (p.180). Each step starts from the pressure actually reached (the law of the thesis written with the
  step-start pressure, so P is continuous) and decays towards its nominal end pressure.
* Press: linear law of eq. (5.47) from the step-start pressure to P_high.
* Purge: P_low at z = 0, pure H2 entering at z = L at a fixed molar flow.
* EP inflow composition: not given in the thesis (eq. 5.45 leaves C_inlet open). Default "ed": the gas that
  left the column in the matching ED step of the same cycle, well mixed (a tank standing in for the partner
  bed: ED1 -> EP1, ED2 -> EP2, ED3 -> EP3); "h2": pure H2 (sensitivity).
* All inflows enter at the feed temperature.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .column import BC, PARAMS, ColumnModel, Feed, ac5ks_column
from .isotherms import R, load_yaml

DAY = 86400.0


@dataclass
class Cycle:
    kind: str                  # "A" (one equalisation) or "B" (three)
    t_feed: float              # s
    t_blow: float
    t_purge: float
    t_eq: float                # each ED and EP step
    t_press: float
    P_high: float              # Pa
    P_low: float
    P_inter: list              # Pa: [P_inter] (A) or [P_inter1, P_inter2, P_inter3] (B)
    T_feed: float              # K
    y_feed: dict
    feed_flow: float           # mol s-1
    beta_P: float              # s-1
    purge_flow: float          # mol s-1 of H2
    ep_gas: str = "ed"         # "ed" | "h2"

    @property
    def steps(self):
        """[(name, duration, k)]; k = equalisation index (1 = highest pressure)."""
        n = len(self.P_inter)
        return ([("Feed", self.t_feed, 0)] + [(f"ED{k}", self.t_eq, k) for k in range(1, n + 1)]
                + [("Blow", self.t_blow, 0), ("Purge", self.t_purge, 0)]
                + [(f"EP{k}", self.t_eq, k) for k in range(n, 0, -1)] + [("Press", self.t_press, 0)])

    @property
    def time(self):
        return sum(s[1] for s in self.steps)


def _std_flow(cc):
    std = cc["breakthrough_runs"]["std_conditions"]
    return std["P_bar"] * 1e5 * cc["feed"]["flow_rate_std"] / (R * std["T"])


def experiment_cycle(**over) -> Cycle:
    """Cycle B of the PSA experiment (Table 5.4 p.167) with beta_P and the purge flow from `fitted:`."""
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    cb, f, fit = cc["cycle_B_experiment"], cc["feed"], cc.get("fitted", {})
    p, t = cb["pressures_bar"], cb["times_s"]
    c = Cycle(kind="B", t_feed=t["Feed"], t_blow=t["Blow"], t_purge=t["Purge"], t_eq=t["ED1"], t_press=t["Press"],
              P_high=p["P_high"] * 1e5, P_low=p["P_low"] * 1e5,
              P_inter=[p["P_inter1"] * 1e5, p["P_inter2"] * 1e5, p["P_inter3"] * 1e5],
              T_feed=f["temperature"], y_feed=f["composition"], feed_flow=_std_flow(cc),
              beta_P=fit.get("pressure_decay_rate", {}).get("value", np.nan),
              purge_flow=fit.get("purge_flow", {}).get("value", np.nan),
              ep_gas=cb.get("ep_inflow", "ed"))
    return replace(c, **over)


def simulation_cycle(row, base: Cycle | None = None) -> Cycle:
    """One of the author's simulations (Tables 5.8/5.10; a row of lopes2010_psa_simulations.csv). beta_P,
    purge flow, feed and EP gas are those of `base` (default: the experiment)."""
    base = base or experiment_cycle()
    P_int = [float(v) * 1e5 for v in str(row["P_inter_bar"]).split(";")]
    return replace(base, kind=row["cycle"], t_feed=float(row["t_feed_s"]), t_blow=float(row["t_blow_s"]),
                   t_purge=float(row["t_purge_s"]), t_eq=float(row["t_eq_s"]), t_press=float(row["t_press_s"]),
                   P_high=float(row["P_high_bar"]) * 1e5, P_low=float(row["P_low_bar"]) * 1e5, P_inter=P_int,
                   T_feed=float(row["T_feed_K"]))


def psa_model(cycle: Cycle, N: int = 30, group: str = "1-4", **opts) -> ColumnModel:
    """AC5-KS column with transport group `group` of Table 5.9, initially full of He at T_feed and P_high
    (p.166); the heater holds the ambient side at T_feed (`assumed: ambient_temperature`).
    opts: mixing, measured_kinetics, thermal (see column.ac5ks_column)."""
    tr = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")["psa_transport"]["groups"][group]
    col = ac5ks_column(tr, cycle.T_feed, **opts)
    feed = Feed(y=cycle.y_feed, T=cycle.T_feed, flow=cycle.feed_flow, P_out=cycle.P_high)
    return ColumnModel(col, feed, list(cycle.y_feed) + ["He"], N=N, y0={"He": 1.0})


@dataclass
class CycleResult:
    perf: dict                 # purity_pct, ppm, co_ppm, recovery_pct, productivity, closure, loadings
    y_end: np.ndarray
    steps: dict = field(default_factory=dict)   # name -> moles (2, ns): in at z = 0, out at z = L
    trace: dict | None = None  # time series when requested


class CycleRunner:
    def __init__(self, model: ColumnModel, cycle: Cycle, rtol: float = 1e-6):
        self.m, self.c, self.rtol = model, cycle, rtol
        self.iH = model.sp.index("H2")
        self.ads = [i for i, s in enumerate(model.sp) if s != "He"]
        cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
        self.w_ads = cc["column"]["adsorbent_mass"]
        self.z_tc = cc["column"]["thermocouples_from_inlet"]
        self.feed_bc = (BC("flow", flow=cycle.feed_flow, y=cycle.y_feed, T=cycle.T_feed),
                        BC("pressure", P=lambda t: cycle.P_high, dPdt=lambda t: 0.0))
        self.ref = (self.feed_bc[1].P, 0.0)   # reference-pressure function and time of the current state

    # -------------------------------------------------------------- step boundary conditions
    def _bcs(self, name, k, dur, P_start, tank):
        c = self.c
        H2 = {"H2": 1.0}

        def const(P):
            return dict(P=lambda t: P, dPdt=lambda t: 0.0)

        def expo(P_end):
            a, b = P_start - P_end, c.beta_P
            return dict(P=lambda t: P_end + a * np.exp(-b * t), dPdt=lambda t: -b * a * np.exp(-b * t))

        closed = BC("closed")
        if name == "Feed":
            return self.feed_bc
        if name.startswith("ED"):
            return closed, BC("pressure", **expo(c.P_inter[k - 1]))
        if name == "Blow":
            return BC("pressure", **expo(c.P_low)), closed
        if name == "Purge":
            return BC("pressure", **const(c.P_low)), BC("flow", flow=c.purge_flow, y=H2, T=c.T_feed)
        if name.startswith("EP"):
            y = H2 if c.ep_gas == "h2" else tank[k]
            return closed, BC("pressure", y=y, T=c.T_feed, **expo(c.P_inter[k - 1]))
        if name == "Press":
            s = (c.P_high - P_start) / dur
            return closed, BC("pressure", P=lambda t: P_start + s * t, dPdt=lambda t: s, y=H2, T=c.T_feed)
        raise ValueError(name)

    def _pressures(self, y):
        """Cell pressures [Pa] of a state integrated with the current reference."""
        N, ns = self.m.N, self.m.ns
        P_ref, t = self.ref
        return P_ref(t) + y[(ns - 1) * N:ns * N]

    # -------------------------------------------------------------- one cycle
    def run_cycle(self, y, trace: bool = False, dt_trace: float = 1.0, until: str | None = None) -> CycleResult:
        """One cycle from state y. until: stop after this step (partial cycle: no performance)."""
        m, c = self.m, self.c
        ns = m.ns
        tank, moles, tr = {}, {}, []
        inv0 = None
        t_cyc = 0.0
        for name, dur, k in c.steps:
            P = self._pressures(y)
            left, right = self._bcs(name, k, dur, P[0] if name == "Blow" or name == "Purge" else P[-1], tank)
            P_ref_old, t_old = self.ref
            m.set_bc(left, right)
            y = m.rebase(y, t_old, P_ref_old)
            self.ref = (m.P_ref, 0.0)
            if inv0 is None:
                inv0 = m.inventory(y, 0.0)
            n_out = max(int(round(dur / dt_trace)) + 1, 2) if trace else 2
            r = m.solve(dur, n_out=n_out, rtol=self.rtol, y0=y)
            y = r.y_end
            self.ref = (m.P_ref, dur)
            moles[name] = r.moles
            if name.startswith("ED"):
                out = np.clip(r.moles[1], 0.0, None)
                tank[k] = dict(zip(m.sp, out / out.sum()))
            if trace:
                tr.append(dict(step=name, t=t_cyc + r.t, out_L=r.outlet_flow, in_0=r.inlet_flow,
                               P0=r.P[0], PL=r.P[-1], T=np.array([r.T_at(z) for z in self.z_tc])))
            t_cyc += dur
            if name == until:
                return CycleResult({}, y, moles, self._stack(tr) if trace else None)
        inv1 = m.inventory(y, self.ref[1])
        return CycleResult(self._performance(moles, inv0, inv1, y), y, moles,
                           self._stack(tr) if trace else None)

    def _performance(self, mo, inv0, inv1, y):
        iH, c = self.iH, self.c
        prod = mo["Feed"][1]
        h2_net = prod[iH] + mo["Press"][1][iH] + mo["Purge"][1][iH]   # press/purge H2 enter at z = L (< 0)
        feed_h2 = mo["Feed"][0][iH]
        net = sum(v[0] - v[1] for v in mo.values())
        inflow = sum(np.clip(v[0], 0, None).sum() + np.clip(-v[1], 0, None).sum() for v in mo.values())
        q = self.m._unpack(y, self.ref[1])[2].mean(axis=1)
        purity = prod[iH] / prod.sum()
        out = dict(purity_pct=100 * purity, ppm=1e6 * (1 - purity),
                   co_ppm=1e6 * prod[self.m.sp.index("CO")] / prod.sum() if "CO" in self.m.sp else np.nan,
                   recovery_pct=100 * h2_net / feed_h2,                                       # eq. (5.49)
                   productivity=h2_net / self.w_ads * DAY / c.time,                           # eq. (5.50)
                   closure=float((inv1 - inv0 - net).sum() / inflow))
        out.update({f"q_{s}": q[i] for i, s in enumerate(self.m.sp) if s != "He"})
        out.update({f"imp_{s}_ppm": 1e6 * prod[i] / prod.sum() for i, s in enumerate(self.m.sp) if i != iH})
        return out

    @staticmethod
    def _stack(tr):
        cat = lambda key, ax=-1: np.concatenate([d[key] for d in tr], axis=ax)
        return dict(t=cat("t"), out_L=cat("out_L"), in_0=cat("in_0"), P0=cat("P0"), PL=cat("PL"), T=cat("T"),
                    step=np.concatenate([[d["step"]] * len(d["t"]) for d in tr]))

    # -------------------------------------------------------------- to CSS
    def run_to_css(self, y0=None, max_cycles: int = 2000, tol: float = 1e-4, window: int = 10,
                   record=(), log=None):
        """Cycle until every CSS quantity (impurity ppm, recovery, bed-averaged loadings) changed by less than
        `tol` (relative) over the last `window` cycles and the cycle mole balance closes within 0.1 %.
        record: cycle numbers whose time series are kept (the last cycle is always kept)."""
        y = self.m.y0.copy() if y0 is None else y0.copy()
        if y0 is None:
            self.ref = (self.feed_bc[1].P, 0.0)
        hist, traces = [], {}
        keys = None
        t0, c0 = time.perf_counter(), time.process_time()
        n = 0
        css = False
        while n < max_cycles:
            n += 1
            res = self.run_cycle(y, trace=n in record)
            y = res.y_end
            hist.append({"cycle": n, **res.perf})
            if res.trace is not None:
                traces[n] = res.trace
            keys = keys or ["ppm", "recovery_pct"] + [k for k in res.perf if k.startswith("q_")]
            if n > window:
                old = hist[-1 - window]
                rel = max(abs(res.perf[k] - old[k]) / max(abs(res.perf[k]), 1e-300) for k in keys)
                hist[-1]["css_change"] = rel
                if log and n % 10 == 0:
                    log(f"cycle {n}: ppm {res.perf['ppm']:.2f} rec {res.perf['recovery_pct']:.3f} "
                        f"change {rel:.1e} closure {res.perf['closure']:.1e} ({time.perf_counter() - t0:.0f} s)")
                if rel < tol and all(abs(h["closure"]) < 1e-3 for h in hist[-window:]):
                    css = True
                    break
        last = self.run_cycle(y, trace=True)   # one more cycle for the time series (continues the sequence)
        hist.append({"cycle": n + 1, **last.perf})
        traces[n + 1] = last.trace
        return dict(history=hist, css=css, cycles=n, y_end=last.y_end, traces=traces, perf=last.perf,
                    runtime_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0)
