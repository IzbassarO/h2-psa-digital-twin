"""Single-column PSA cycle to cyclic steady state (CSS) — Lopes (2010) ch.5.

Cycle A (Feed, ED, Blow, Purge, EP, Press) and cycle B (Feed, ED1-3, Blow, Purge, EP3-1, Press), Figs 5.7
and 5.9; boundary conditions eqs. (5.36)-(5.47) pp.178-180; performance eqs. (5.48)-(5.50) p.180.

* ED, Blow and EP: exponential pressure law of eqs. (5.39)-(5.40), one rate constant beta_P for all of them
  (p.180). Each step starts from the pressure actually reached (the law of the thesis written with the
  step-start pressure, so P is continuous) and decays towards its nominal end pressure.
* Press: linear law of eq. (5.47) from the step-start pressure to P_high.
* Purge: P_low at z = 0, pure H2 entering at z = L at a fixed molar flow.
* EP inflow composition: not given in the thesis (eq. 5.45 leaves C_inlet open). Options (`ep_gas`), all
  taken from the matching ED step of the same cycle (ED1 -> EP1, ED2 -> EP2, ED3 -> EP3) unless stated:
  "ed" the ED gas well mixed (a tank standing in for the partner bed); "ed_direct" the ED outlet composition
  at the same time into the step (first out, first in: a partner bed connected directly); "ed_reversed" the
  ED outlet composition at t_eq - t (last out, first in); "product" the well-mixed feed-step product of the
  same cycle; "h2" pure H2.
* beta_P: one constant (p.180), or per step type (`beta_steps` = {"ED": .., "Blow": .., "EP": ..}).
* All inflows enter at the feed temperature.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .column import BC, PARAMS, Column, ColumnModel, Feed, ac5ks_column
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
    ep_gas: str = "ed"         # "ed" | "ed_direct" | "ed_reversed" | "product" | "h2"
    beta_steps: dict | None = None   # per step type {"ED", "Blow", "EP"}: overrides beta_P

    def beta(self, name):
        kind = "ED" if name.startswith("ED") else "EP" if name.startswith("EP") else name
        return self.beta_steps.get(kind, self.beta_P) if self.beta_steps else self.beta_P

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
    """Cycle B of the PSA experiment (Table 5.4 p.167) with beta_P and the purge flow from `fitted:`.
    Defaults are the frozen baseline (step-3 closure): beta_P from `cycle_B_experiment.beta_P_source`, EP = H2."""
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    cb, f, fit = cc["cycle_B_experiment"], cc["feed"], cc.get("fitted", {})
    p, t = cb["pressures_bar"], cb["times_s"]
    c = Cycle(kind="B", t_feed=t["Feed"], t_blow=t["Blow"], t_purge=t["Purge"], t_eq=t["ED1"], t_press=t["Press"],
              P_high=p["P_high"] * 1e5, P_low=p["P_low"] * 1e5,
              P_inter=[p["P_inter1"] * 1e5, p["P_inter2"] * 1e5, p["P_inter3"] * 1e5],
              T_feed=f["temperature"], y_feed=f["composition"], feed_flow=_std_flow(cc),
              beta_P=fit.get(cb.get("beta_P_source", "pressure_decay_rate"), {}).get("value", np.nan),
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


def psa_model(cycle: Cycle, N: int = 30, group: str = "1-4", column: Column | None = None, **opts) -> ColumnModel:
    """AC5-KS column with transport group `group` of Table 5.9, initially full of He at T_feed and P_high
    (p.166); the heater holds the ambient side at T_feed (`assumed: ambient_temperature`).
    opts: mixing, measured_kinetics, thermal (see column.ac5ks_column). column: any other Column (layered bed)."""
    if column is None:
        tr = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")["psa_transport"]["groups"][group]
        column = ac5ks_column(tr, cycle.T_feed, **opts)
    col = column
    feed = Feed(y=cycle.y_feed, T=cycle.T_feed, flow=cycle.feed_flow, P_out=cycle.P_high)
    return ColumnModel(col, feed, list(cycle.y_feed) + ["He"], N=N, y0={"He": 1.0})


@dataclass
class CycleResult:
    perf: dict                 # purity_pct, ppm, co_ppm, recovery_pct, productivity, closure, loadings
    y_end: np.ndarray
    steps: dict = field(default_factory=dict)   # name -> moles (2, ns): in at z = 0, out at z = L
    trace: dict | None = None  # time series when requested
    profiles: dict | None = None   # step name -> axial profiles at the end of the step, when requested


class CycleRunner:
    def __init__(self, model: ColumnModel, cycle: Cycle, rtol: float = 1e-6, w_ads: float | None = None):
        """w_ads: adsorbent mass for the productivity [kg] (default: the compact unit's AC5-KS charge, Table 5.3)."""
        self.m, self.c, self.rtol = model, cycle, rtol
        self.iH = model.sp.index("H2")
        self.ads = [i for i, s in enumerate(model.sp) if s != "He"]
        cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
        self.w_ads = cc["column"]["adsorbent_mass"] if w_ads is None else w_ads
        self.retries = 0                       # steps re-solved with another rtol (see _solve)
        self.z_tc = cc["column"]["thermocouples_from_inlet"]
        self.feed_bc = (BC("flow", flow=cycle.feed_flow, y=cycle.y_feed, T=cycle.T_feed),
                        BC("pressure", P=lambda t: cycle.P_high, dPdt=lambda t: 0.0))
        self.ref = (self.feed_bc[1].P, 0.0)   # reference-pressure function and time of the current state

    # -------------------------------------------------------------- step boundary conditions
    def _bcs(self, name, k, dur, P_start, tank):
        c = self.c
        H2 = {"H2": 1.0}
        beta = c.beta(name)

        def const(P):
            return dict(P=lambda t: P, dPdt=lambda t: 0.0)

        def expo(P_end):
            a, b = P_start - P_end, beta
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
            if c.ep_gas in ("ed_direct", "ed_reversed"):
                phi, Y = tank[("hist", k)]
                f = (lambda t: t / dur) if c.ep_gas == "ed_direct" else (lambda t: 1.0 - t / dur)
                y_t = lambda t: np.array([np.interp(f(t), phi, Yi) for Yi in Y])
                return closed, BC("pressure", y=H2, y_t=y_t, T=c.T_feed, **expo(c.P_inter[k - 1]))
            y = {"h2": H2, "product": tank.get("product"), "ed": tank.get(k)}[c.ep_gas]
            return closed, BC("pressure", y=y, T=c.T_feed, **expo(c.P_inter[k - 1]))
        if name == "Press":
            s = (c.P_high - P_start) / dur
            return closed, BC("pressure", P=lambda t: P_start + s * t, dPdt=lambda t: s, y=H2, T=c.T_feed)
        raise ValueError(name)

    @staticmethod
    def _ed_history(t, F, nbin=20):
        """ED gas composition vs the fraction phi of the ED gas already released (bins of equal moles):
        (phi at bin centres, mole fractions (ns, nbin)). EP maps its time onto phi (t/t_EP or 1 - t/t_EP)."""
        cum = np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum(0.5 * (F[:, 1:] + F[:, :-1]) * np.diff(t), axis=1)], axis=1)
        tot = cum.sum(0) / cum.sum(0)[-1]
        edges = np.linspace(0, 1, nbin + 1)
        Ni = np.array([np.interp(edges, tot, ci) for ci in cum])
        dN = np.diff(Ni, axis=1)
        return 0.5 * (edges[1:] + edges[:-1]), dN / np.maximum(dN.sum(0), 1e-300)

    def _solve(self, dur, n_out, y):
        """One step; if BDF stops on a step-size underflow, retry with rtol / 10, then rtol x 10 (counted in
        self.retries; the tolerance changes only for that step)."""
        for rtol in (self.rtol, 0.1 * self.rtol, 10 * self.rtol):
            try:
                return self.m.solve(dur, n_out=n_out, rtol=rtol, y0=y)
            except RuntimeError:
                if rtol == 10 * self.rtol:
                    raise
                self.retries += 1

    def _pressures(self, y):
        """Cell pressures [Pa] of a state integrated with the current reference."""
        N, ns = self.m.N, self.m.ns
        P_ref, t = self.ref
        return P_ref(t) + y[(ns - 1) * N:ns * N]

    # -------------------------------------------------------------- one cycle
    def run_cycle(self, y, trace: bool = False, dt_trace: float = 1.0, until: str | None = None,
                  profiles: bool = False) -> CycleResult:
        """One cycle from state y. until: stop after this step (partial cycle: no performance).
        profiles: keep the axial gas concentration, loading and temperature profiles at the end of every step."""
        m, c = self.m, self.c
        ns = m.ns
        tank, moles, tr, prof = {}, {}, [], {}
        hist = c.ep_gas in ("ed_direct", "ed_reversed")
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
            if hist and name.startswith("ED"):
                n_out = max(n_out, int(round(4 * dur)) + 1)
            r = self._solve(dur, n_out, y)
            y = r.y_end
            self.ref = (m.P_ref, dur)
            moles[name] = r.moles
            if name.startswith("ED"):
                out = np.clip(r.moles[1], 0.0, None)
                tank[k] = dict(zip(m.sp, out / out.sum()))
                tank[("hist", k)] = self._ed_history(r.t, np.clip(r.outlet_flow, 0.0, None))
            if name == "Feed":
                out = np.clip(r.moles[1], 0.0, None)
                tank["product"] = dict(zip(m.sp, out / out.sum()))
            if profiles:
                C, Cm, q, Tg, *_ = m._unpack(y, dur)
                prof[name] = dict(z=m.z, C=C, Cm=Cm, q=q, Tg=Tg)
            if trace:
                tr.append(dict(step=name, t=t_cyc + r.t, out_L=r.outlet_flow, in_0=r.inlet_flow,
                               P0=r.P[0], PL=r.P[-1], T=np.array([r.T_at(z) for z in self.z_tc])))
            t_cyc += dur
            if name == until:
                return CycleResult({}, y, moles, self._stack(tr) if trace else None, prof or None)
        inv1 = m.inventory(y, self.ref[1])
        return CycleResult(self._performance(moles, inv0, inv1, y), y, moles,
                           self._stack(tr) if trace else None, prof or None)

    def _performance(self, mo, inv0, inv1, y):
        iH, c = self.iH, self.c
        prod = mo["Feed"][1]
        h2_net = prod[iH] + mo["Press"][1][iH] + mo["Purge"][1][iH]   # press/purge H2 enter at z = L (< 0)
        feed_h2 = mo["Feed"][0][iH]
        net = sum(v[0] - v[1] for v in mo.values())
        inflow = sum(np.clip(v[0], 0, None).sum() + np.clip(-v[1], 0, None).sum() for v in mo.values())
        q = self.m._unpack(y, self.ref[1])[2].mean(axis=1)
        sp = self.m.sp
        prod = np.where(np.array(sp) == "He", 0.0, prod)        # eq. (5.48): feed components only
        purity = prod[iH] / prod.sum()
        out = dict(purity_pct=100 * purity, ppm=1e6 * (1 - purity),
                   co_ppm=1e6 * prod[self.m.sp.index("CO")] / prod.sum() if "CO" in self.m.sp else np.nan,
                   recovery_pct=100 * h2_net / feed_h2,                                       # eq. (5.49)
                   productivity=h2_net / self.w_ads * DAY / c.time,                           # eq. (5.50)
                   closure=float((inv1 - inv0 - net).sum() / inflow))
        out.update({f"q_{s}": q[i] for i, s in enumerate(self.m.sp) if s != "He"})
        out.update({f"imp_{s}_ppm": 1e6 * prod[i] / prod.sum() for i, s in enumerate(sp) if s not in ("H2", "He")})
        return out

    @staticmethod
    def _stack(tr):
        cat = lambda key, ax=-1: np.concatenate([d[key] for d in tr], axis=ax)
        return dict(t=cat("t"), out_L=cat("out_L"), in_0=cat("in_0"), P0=cat("P0"), PL=cat("PL"), T=cat("T"),
                    step=np.concatenate([[d["step"]] * len(d["t"]) for d in tr]))

    # -------------------------------------------------------------- to CSS
    def run_to_css(self, y0=None, max_cycles: int = 2000, tol: float = 1e-4, window: int = 10,
                   record=(), log=None, profiles=(), accel: bool = False, memory: int = 5, tol_acc: float = 1e-6,
                   stall: int = 20, tol_ppm: float | None = None, censor: tuple | None = None, checkpoint=None):
        """Cycle until every CSS quantity (impurity ppm, recovery, bed-averaged loadings) changed by less than
        `tol` (relative) over the last `window` cycles and the cycle mole balance closes within 0.1 %.
        record: cycle numbers whose time series are kept (the last cycle is always kept).
        profiles: cycle numbers whose end-of-step axial profiles are kept (the last cycle is always kept).
        y0: start state (a CSS state of a neighbouring design: warm start); its pressure row must be referred to
        P_high (as every cycle-end state is).
        tol_ppm: if given, impurity ppm must in addition change by less than `tol_ppm` (relative) over the window
        (a tighter condition on the quantity the purity constraint is evaluated on).
        accel: Anderson acceleration of the cycle map (`Anderson`) until its scaled residual is below `tol_acc`
        or has not halved in `stall` accelerated cycles (stagnation at the integration-noise floor, or a stalled
        nonlinear transition); then plain cycles. CSS is only declared by the criterion above over >= `window` consecutive plain cycles
        (a plain cycle starts from the unmodified end state of the previous one); if 3 x window plain cycles do
        not reach it, acceleration resumes with a fresh memory.
        censor: (min_cycles, ppm_limit, window): stop a clearly infeasible point early (see `censored`); the result
        then has css False and censored = the last impurity [ppm].
        checkpoint: called as checkpoint(n, y) after cycle n with the start state of the next cycle (resume)."""
        y = self.m.y0.copy() if y0 is None else y0.copy()
        self.ref = (self.feed_bc[1].P, 0.0)
        hist, traces, profs = [], {}, {}
        keys = None
        t0, c0 = time.perf_counter(), time.process_time()
        n = 0
        css = False
        nv = self.m.nv - 2 * self.m.ns                    # cumulative mole counters are not part of the state
        aa = Anderson(self.m.yscale[:nv], memory, self._nonneg(), stall=stall) if accel else None
        plain, n_plain, n_acc = False, 0, 0               # plain: this cycle starts from the previous end state
        cens = None
        while n < max_cycles:
            n += 1
            res = self.run_cycle(y, trace=n in record, profiles=n in profiles)
            if res.profiles:
                profs[n] = res.profiles
            n_plain = n_plain + 1 if plain else 0
            hist.append({"cycle": n, "plain": plain, **res.perf})
            if res.trace is not None:
                traces[n] = res.trace
            keys = keys or ["ppm", "recovery_pct"] + [k for k in res.perf if k.startswith("q_")]
            rel = np.nan
            if n > window:
                old = hist[-1 - window]
                rel = max(abs(res.perf[k] - old[k]) / max(abs(res.perf[k]), 1e-300) for k in keys)
                hist[-1]["css_change"] = rel
                ppm_ok = tol_ppm is None or abs(res.perf["ppm"] - old["ppm"]) < tol_ppm * abs(res.perf["ppm"])
                if (n_plain >= window or aa is None) and rel < tol and ppm_ok \
                        and all(abs(h["closure"]) < 1e-3 for h in hist[-window:]):
                    y = res.y_end
                    css = True
                    break
            if censor and censored([h["ppm"] for h in hist], *censor):
                y, cens = res.y_end, res.perf["ppm"]
                break
            if aa is not None and aa.active:
                y_next, r = aa.step(y[:nv], res.y_end[:nv])
                hist[-1]["residual"] = r
                if r < tol_acc or aa.stalled:
                    aa.active, plain, y = False, True, res.y_end
                else:
                    n_acc += 1
                    plain, y = False, np.concatenate([y_next, res.y_end[nv:]])
            else:
                plain, y = True, res.y_end
                if aa is not None and n_plain >= 3 * window:
                    aa.reset()
            if checkpoint:
                checkpoint(n, y)
            if log and n % 10 == 0:
                log(f"cycle {n}: ppm {res.perf['ppm']:.2f} rec {res.perf['recovery_pct']:.3f} change {rel:.1e} "
                    f"closure {res.perf['closure']:.1e}{' acc' if aa is not None and aa.active else ''} "
                    f"({time.perf_counter() - t0:.0f} s)")
        last = self.run_cycle(y, trace=True, profiles=True)   # one more cycle for the time series and profiles
        hist.append({"cycle": n + 1, "plain": True, **last.perf})
        traces[n + 1] = last.trace
        profs[n + 1] = last.profiles
        return dict(history=hist, css=css, censored=cens, cycles=n, accelerated_cycles=n_acc, y_end=last.y_end, traces=traces,
                    perf=last.perf, profiles=profs, step_moles=last.steps, retries=self.retries,
                    runtime_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0)

    def _nonneg(self):
        """Mask of the state entries that must stay >= 0 (gas and macropore concentrations, loadings)."""
        N, ns = self.m.N, self.m.ns
        mask = np.zeros(self.m.nv - 2 * ns, bool)
        mask[:(ns - 1) * N] = True                        # C_i in G (the last row of G is the pressure deviation)
        mask[ns * N:3 * ns * N] = True                    # Cm, q
        return mask


def censored(ppm, min_cycles=100, limit=1000.0, window=30):
    """Early-stopping rule for a clearly infeasible point (step 5): after >= `min_cycles` cycles the impurity
    `ppm[-1]` exceeds `limit` (10x the 100-ppm specification) and has increased over the last `window` cycles.
    ppm: impurity per cycle so far (accelerated iterates included, as in the CSS history)."""
    return len(ppm) >= max(min_cycles, window + 1) and ppm[-1] > limit and ppm[-1] > ppm[-1 - window]


class Anderson:
    """Anderson acceleration (type II, Walker & Ni 2011) of a fixed point x = G(x), in variables scaled by
    `scale`, with `memory` previous differences. Safeguards: the memory restarts when the residual grows by more
    than `grow` over the best one since the last restart; iterates are clipped at zero where `nonneg`; the
    least-squares problem is solved by lstsq with a relative cut-off of 1e-10 on the singular values.
    `stalled` is set when the residual has not halved in `stall` steps since the last reset."""

    def __init__(self, scale, memory=5, nonneg=None, grow=2.0, stall=20):
        self.s, self.m, self.nonneg, self.grow, self.stall = scale, memory, nonneg, grow, stall
        self.active = True
        self.reset()

    def reset(self):
        self.dF, self.dG, self.f, self.g, self.best = [], [], None, None, np.inf
        self.active, self.mark, self.since = True, np.inf, 0

    def step(self, x, gx):
        """x: last iterate, gx = G(x). Returns (next iterate, max-norm of the scaled residual G(x) - x)."""
        u, gu = x / self.s, gx / self.s
        f = gu - u
        r = float(np.abs(f).max())
        self.mark, self.since = (r, 0) if r < 0.5 * self.mark else (self.mark, self.since + 1)
        if r > self.grow * self.best:                     # diverging: drop the memory, take the plain step
            self.dF, self.dG, self.f, self.g, self.best = [], [], None, None, np.inf
        if self.f is not None:
            self.dF.append(f - self.f)
            self.dG.append(gu - self.g)
            self.dF, self.dG = self.dF[-self.m:], self.dG[-self.m:]
        self.f, self.g, self.best = f, gu, min(self.best, r)
        if not self.dF:
            return gx.copy(), r
        F, Gm = np.array(self.dF).T, np.array(self.dG).T
        gamma = np.linalg.lstsq(F, f, rcond=1e-10)[0]
        x_new = (gu - Gm @ gamma) * self.s
        if self.nonneg is not None:
            x_new[self.nonneg] = np.maximum(x_new[self.nonneg], 0.0)
        return x_new, r

    @property
    def stalled(self):
        return self.since >= self.stall
