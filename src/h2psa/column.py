"""Dynamic 1D fixed-bed adsorption column — Lopes (2010) ch.5, eqs (5.3)-(5.16), printed pp.167-171.

* axially dispersed plug flow, ideal gas, Ergun momentum balance (5.4) evaluated at cell faces
* bi-LDF mass transfer (5.6)-(5.7) with film resistance through the Biot number (5.9)-(5.10)
* gas (5.5), solid (5.8) and wall (5.11)-(5.13) energy balances
* finite volumes; convection upwinded with the van Leer limiter
* boundary conditions per end (`BC`): closed, imposed molar inflow (Danckwerts), or imposed pressure p(t)
  with Ergun flow through the half cell next to the end (eqs. 5.36-5.47). Default: feed at z = 0,
  fixed pressure at z = L (breakthrough runs); `cycle.py` switches them step by step
* scipy BDF with a banded sparse Jacobian pattern

SI units inside the model (pressure in Pa); isotherms are called with partial pressures in bar.
An isotherm is any object with `.names`, `.equilibrium(P_bar (n, k), T (n,)) -> q (n, k)` and
`.species[name].minus_dH` [kJ/mol]; each layer of the bed carries its own isotherm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csc_matrix

from .isotherms import R, Virial, load_yaml
from .kinetics import arrhenius_fit

PARAMS = Path(__file__).resolve().parents[2] / "data" / "params"
_DELTA = 1e-10  # van Leer smoothing (Haghpanah et al., 2013)
_P_MIN = 1e-10  # bar


@dataclass
class Layer:
    """One adsorbent layer. Per-species dicts are keyed by species name; missing species do not adsorb."""
    length: float      # m
    eps: float         # bed porosity
    iso: object        # isotherm (see module docstring)
    rho_p: float       # particle density, kg m-3
    eps_p: float       # particle porosity
    Rp: float          # particle radius, m
    Omega_m: float     # macropore LDF factor
    Omega_c: float     # micropore LDF factor
    Cps: float         # solid heat capacity, J kg-1 K-1
    a_p: float         # particle specific area, m-1
    kf: float          # film mass transfer coefficient, m s-1
    hf: float          # film heat transfer coefficient, W m-2 K-1
    Dp: dict           # macropore diffusivity, m2 s-1
    Dc0: dict          # micropore Dc0/rc2, s-1
    Ea: dict           # micropore activation energy, kJ mol-1
    Dax: float | None = None   # axial mass dispersion, m2 s-1 (None: the column value)
    lam: float | None = None   # axial heat dispersion, W m-1 K-1 (None: the column value)
    hw: float | None = None    # gas-wall film coefficient, W m-2 K-1 (None: the column value)
    U: float | None = None     # wall-ambient coefficient, W m-2 K-1 (None: the column value)


@dataclass
class Column:
    d_wi: float        # internal diameter, m
    e_wall: float      # wall thickness, m (heat-exchange geometry only)
    Cw: float          # wall heat capacity per unit length, rho_w Cp_w pi e (d_wi + e), J m-1 K-1
    Dax: float         # axial mass dispersion, m2 s-1
    lam: float         # axial heat dispersion, W m-1 K-1
    hw: float          # gas-wall film coefficient, W m-2 K-1
    U: float           # wall-ambient coefficient, W m-2 K-1
    T_amb: float       # K
    layers: list


@dataclass
class Feed:
    y: dict            # mole fractions
    T: float           # K
    flow: float        # molar flow, mol s-1
    P_out: float       # outlet pressure, Pa


@dataclass
class BC:
    """Boundary condition at one end of the column.
    kind "closed": u = 0. "flow": `flow` mol s-1 enters with composition `y` at `T` (Danckwerts).
    "pressure": p(t) [Pa] imposed at the end (`P`, with derivative `dPdt`, t = time since the step
    started); gas entering takes composition `y` at `T` (None: that of the end cell, i.e. zero gradient).
    Exactly one end of the column carries a pressure condition."""
    kind: str
    flow: float = 0.0
    P: Callable | None = None
    dPdt: Callable | None = None
    y: dict | None = None
    T: float | None = None
    y_t: Callable | None = None   # time-dependent inflow composition: t -> mole fractions in model species order


def _ergun(G, a, b):
    """Superficial velocity from -dP/dz = a u + b |u| u (stable root)."""
    return 2.0 * G / (a + np.sqrt(a * a + 4.0 * b * np.abs(G)))


def _vanleer(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def _faces(X, u):
    """Limited upwind values of X (..., N) at the N-1 interior faces for face velocities u (N-1,).
    First order next to the boundaries."""
    d = np.diff(X, axis=-1)                                   # X[j+1] - X[j] across face j+1/2
    z = np.zeros_like(d[..., :1])
    dl = np.concatenate([z, d[..., :-1]], axis=-1)            # X[j] - X[j-1]
    xp = X[..., :-1] + 0.5 * _vanleer((d + _DELTA) / (dl + _DELTA)) * dl
    dr = np.concatenate([-d[..., 1:], z], axis=-1)            # X[j+1] - X[j+2]
    xn = X[..., 1:] + 0.5 * _vanleer((-d + _DELTA) / (dr + _DELTA)) * dr
    return np.where(u >= 0, xp, xn)


def _colour(S):
    """Greedy column grouping for finite differences: columns in a group share no nonzero row."""
    S = S.tocsc()
    groups = np.empty(S.shape[1], int)
    occupied: list[np.ndarray] = []
    for j in range(S.shape[1]):
        rows = S.indices[S.indptr[j]:S.indptr[j + 1]]
        for g, occ in enumerate(occupied):
            if not occ[rows].any():
                break
        else:
            occupied.append(np.zeros(S.shape[0], bool))
            g = len(occupied) - 1
        occupied[g][rows] = True
        groups[j] = g
    return groups


def crossing_time(t, f, level):
    """First time f rises through level (linear interpolation); nan if it never does."""
    t, f = np.asarray(t), np.asarray(f)
    i = int(np.argmax(f >= level))
    if f[i] < level:
        return np.nan
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - f[i - 1]) * (t[i] - t[i - 1]) / (f[i] - f[i - 1]))


@dataclass
class Result:
    t: np.ndarray            # s
    species: list
    outlet_flow: np.ndarray  # (ns, nt) mol s-1
    z: np.ndarray            # cell centres, m
    Tg: np.ndarray           # (N, nt) K
    Tw: np.ndarray           # (N, nt) K
    P: np.ndarray            # (N, nt) Pa
    closure: dict            # relative mole-balance error per species and 'total'
    runtime_s: float
    nfev: int
    inlet_flow: np.ndarray = None   # (ns, nt) mol s-1 into the bed at z = 0 (negative: leaving)
    y_end: np.ndarray = None        # final state vector
    moles: np.ndarray = None        # (2, ns) mol in at z = 0 and out at z = L over the solve (signed)

    def T_at(self, z):
        """Gas temperature history at axial position z (thermocouple)."""
        return np.array([np.interp(z, self.z, self.Tg[:, k]) for k in range(len(self.t))])


class ColumnModel:
    def __init__(self, col: Column, feed: Feed, species: list, N: int = 60,
                 y0: dict | None = None, isothermal: bool = False):
        self.col, self.feed, self.sp, self.N, self.isothermal = col, feed, list(species), N, isothermal
        ns = self.ns = len(species)
        L = sum(l.length for l in col.layers)
        self.dz = L / N
        self.z = (np.arange(N) + 0.5) * self.dz
        self.A = np.pi * col.d_wi**2 / 4

        # per-cell layer properties; every layer interface must lie on a cell face
        edges = np.cumsum([l.length for l in col.layers])
        k_face = edges[:-1] / self.dz
        if np.any(np.abs(k_face - np.round(k_face)) > 1e-6):
            raise ValueError(f"layer interfaces at {edges[:-1]} m are not on cell faces (dz = {self.dz:.4g} m)")
        lay = np.minimum(np.searchsorted(edges, self.z), len(col.layers) - 1)
        self.layer_of_cell = lay
        cell = lambda f: np.array([f(col.layers[k]) for k in lay])
        or_col = lambda name: cell(lambda l: getattr(col, name) if getattr(l, name) is None else getattr(l, name))
        self.Dax_c, self.lam_c, self.hw_c, self.U_c = or_col("Dax"), or_col("lam"), or_col("hw"), or_col("U")

        spc = lambda f: np.array([[f(col.layers[k], s) for k in lay] for s in species])
        self.eps, self.rho_p, self.eps_p = cell(lambda l: l.eps), cell(lambda l: l.rho_p), cell(lambda l: l.eps_p)
        harm = lambda a: 2 * a[:-1] * a[1:] / (a[:-1] + a[1:])               # face value: harmonic mean
        self.eDax_f, self.lam_f = harm(self.eps * self.Dax_c), harm(self.lam_c)
        self.Rp, self.Cps, self.a_p = cell(lambda l: l.Rp), cell(lambda l: l.Cps), cell(lambda l: l.a_p)
        self.kf, self.hf = cell(lambda l: l.kf), cell(lambda l: l.hf)
        Om_m, self.Om_c = cell(lambda l: l.Omega_m), cell(lambda l: l.Omega_c)
        Dp = spc(lambda l, s: l.Dp[s])
        self.Dc0 = spc(lambda l, s: l.Dc0.get(s, 0.0) if s in l.iso.names else 0.0)
        self.Ea = spc(lambda l, s: l.Ea.get(s, 0.0) * 1e3)
        self.dH = spc(lambda l, s: l.iso.species[s].minus_dH * 1e3 if s in l.iso.names else 0.0)
        self.kmac = Om_m * Dp / self.Rp**2                                  # macropore LDF rate, s-1
        self.Bi = self.a_p * self.kf * self.Rp**2 / (self.eps_p * Om_m * Dp)  # eq. (5.10)
        self.kfa = (1 - self.eps) * self.a_p * self.kf                       # film, per bed volume
        self.dp = 2 * self.Rp
        self.layer_maps = []  # (cell slice, isotherm, species index per isotherm name or -1)
        for k, l in enumerate(col.layers):
            idx = np.flatnonzero(lay == k)
            self.layer_maps.append((slice(idx[0], idx[-1] + 1), l.iso,
                                    np.array([species.index(n) if n in species else -1 for n in l.iso.names])))

        # gas properties at the feed temperature
        g = load_yaml(PARAMS / "gas_properties_nist.yaml")
        self.M = np.array([g["species"][s]["M"] for s in species])
        self.Cp = np.array([np.interp(feed.T, g["T"], g["species"][s]["Cp"]) for s in species])
        self.Cv = self.Cp - R
        self.mu = np.array([np.interp(feed.T, g["T"], g["species"][s]["viscosity"]) for s in species])

        # wall geometry, eqs. (5.12)-(5.13)
        d, e = col.d_wi, col.e_wall
        self.alpha_w = d / (e * (d + e))
        self.alpha_wl = 1 / ((d + e) * np.log((d + e) / d))
        self.rhoCp_w = col.Cw / (np.pi * e * (d + e))   # rho_w Cp_w of eq. (5.11) from the lumped wall term

        yf = np.array([feed.y.get(s, 0.0) for s in species])
        self.y_feed = yf / yf.sum()
        self.N_in = feed.flow / self.A  # mol m-2 s-1
        self.set_bc(BC("flow", flow=feed.flow, y=feed.y, T=feed.T),
                    BC("pressure", P=lambda t: feed.P_out, dPdt=lambda t: 0.0))

        y0 = y0 or {"He": 1.0}
        y0v = np.array([y0.get(s, 0.0) for s in species])
        C0 = np.outer(y0v / y0v.sum(), np.full(N, feed.P_out / (R * feed.T)))
        T0 = np.full(N, feed.T)
        # State layout: [G, Cm, q, Tg, Tp, Tw, n_in, n_out]. G holds C_i for all species but the last and,
        # in its last row, dP = P - p_ref(t) (p_ref: pressure imposed at one end). The pressure drop is ~1e-6 P, so P is carried as a deviation
        # with its own tolerance: with C_T as a state, the integration error alone (rtol x P) would be
        # comparable to the Ergun pressure drop and the face velocities would chatter.
        # n_in, n_out: cumulative moles in/out (mole balance only; no other equation reads them).
        G0 = np.vstack([C0[:-1], np.zeros(N)])
        self.y0 = np.concatenate([G0.ravel(), C0.ravel(), self._qstar(C0[None], T0[None]).ravel(),
                                  T0, T0, T0, np.zeros(2 * ns)])
        self.nv = self.y0.size
        CTf = feed.P_out / (R * feed.T)
        gscale, gatol = np.full((ns, N), CTf), np.full((ns, N), 1e-6)
        gscale[-1], gatol[-1] = 100.0, 1e-3          # dP row [Pa]
        self.yscale = np.concatenate([gscale.ravel(), np.full(ns * N, CTf), np.full(ns * N, 1.0),
                                      np.full(3 * N, feed.T), np.full(2 * ns, feed.flow * 100.0)])
        self.atol = np.concatenate([gatol.ravel(), np.full(ns * N, 1e-6), np.full(ns * N, 1e-8),
                                    np.full(3 * N, 1e-5), np.full(2 * ns, 1e-9)])
        S = self.sparsity().tocoo()
        self._jr, self._jc = S.row, S.col
        self._jg = _colour(self.sparsity())

    # ------------------------------------------------------------------ boundary conditions
    def set_bc(self, left: BC, right: BC):
        """Set the boundary conditions at z = 0 and z = L. The pressure state is P - p_ref(t), with p_ref the
        imposed end pressure, so a state vector belongs to the conditions it was integrated with:
        convert it with `rebase` when the pressure end changes."""
        assert [left.kind, right.kind].count("pressure") == 1, "exactly one end must impose the pressure"
        vec = lambda yy: None if yy is None else np.array([yy.get(s, 0.0) for s in self.sp]) / sum(yy.values())
        self.bc = [(b, vec(b.y)) for b in (left, right)]
        self.p_end = 0 if left.kind == "pressure" else 1
        pb = (left, right)[self.p_end]
        self.P_ref, self.dP_ref = pb.P, pb.dPdt

    def rebase(self, y, t_old, P_ref_old):
        """State y integrated with reference pressure P_ref_old(t_old) -> state for the current conditions at t = 0."""
        y = y.copy()
        N = self.N
        sl = slice((self.ns - 1) * N, self.ns * N)
        y[sl] += P_ref_old(t_old) - self.P_ref(0.0)
        return y

    # ------------------------------------------------------------------ pieces
    # All pieces work on a leading batch axis (B states at once: the Jacobian evaluates its column groups
    # in one call); the single-state wrappers _unpack, _gas and rhs add and strip it.
    def _unpack_b(self, Y, t):
        """Y (B, nv) -> C (B, ns, N) with the last species from the ideal-gas law, Cm, q, Tg (B, N), Tp, Tw,
        dP = P - p_ref(t)."""
        ns, N = self.ns, self.N
        b, B = ns * N, Y.shape[0]
        G = Y[:, :b].reshape(B, ns, N)
        Tg = Y[:, 3 * b:3 * b + N]
        dP = G[:, -1]
        C = np.concatenate([G[:, :-1], ((self.P_ref(t) + dP) / (R * Tg) - G[:, :-1].sum(1))[:, None]], axis=1)
        return (C, Y[:, b:2 * b].reshape(B, ns, N), Y[:, 2 * b:3 * b].reshape(B, ns, N),
                Tg, Y[:, 3 * b + N:3 * b + 2 * N], Y[:, 3 * b + 2 * N:3 * b + 3 * N], dP)

    def _unpack(self, y, t=0.0):
        return tuple(v[0] for v in self._unpack_b(y[None], t))

    def _qstar(self, Cm, Tp):
        """Equilibrium loadings for macropore concentrations Cm (B, ns, N) at Tp (B, N)."""
        q = np.zeros_like(Cm)
        B = Cm.shape[0]
        for sl, iso, idx in self.layer_maps:
            n = sl.stop - sl.start
            Pb = np.zeros((B, n, len(idx)))
            have = idx >= 0
            Pb[:, :, have] = (Cm[:, idx[have], sl] * R * Tp[:, None, sl]).transpose(0, 2, 1) / 1e5
            Pb = Pb.reshape(B * n, len(idx))
            # below P_MIN continue linearly through zero (Henry region): smooth for slightly negative Cm
            Pc = np.maximum(Pb, _P_MIN)
            qq = (iso.equilibrium(Pc, Tp[:, sl].reshape(-1)) * (Pb / Pc)).reshape(B, n, len(idx))
            q[:, idx[have], sl] = qq[:, :, have].transpose(0, 2, 1)
        return q

    def _gas_b(self, C, Tg, dP, t):
        """Face fluxes F (B, ns, N+1) [mol m-2 s-1], face velocities u (B, N+1), inflow temperatures at the
        two end faces (B, 2; nan: no inflow), and mixture properties."""
        B = C.shape[0]
        CT = C.sum(1)
        yfr = C / CT[:, None]
        Pref = self.P_ref(t)
        P = Pref + dP
        eps, dp, dz = self.eps, self.dp, self.dz
        mu, rho = self.mu @ yfr, self.M @ C
        a = 150 * mu * (1 - eps) ** 2 / (eps**3 * dp**2)
        b = 1.75 * (1 - eps) * rho / (eps**3 * dp)
        af, bf = 0.5 * (a[:, :-1] + a[:, 1:]), 0.5 * (b[:, :-1] + b[:, 1:])
        u_int = _ergun(-np.diff(dP, axis=-1) / dz, af, bf)
        J = -self.eDax_f * (0.5 * (CT[:, :-1] + CT[:, 1:]))[:, None] * np.diff(yfr, axis=-1) / dz
        F = np.empty((B, self.ns, self.N + 1))
        F[..., 1:-1] = u_int[:, None] * _faces(C, u_int[:, None]) + J
        u_end, T_in = np.zeros((B, 2)), np.full((B, 2), np.nan)
        for e, (bc, yb) in enumerate(self.bc):
            j, sgn = (0, 1.0) if e == 0 else (-1, -1.0)       # sgn: +1 where inflow means u > 0
            if bc.kind == "closed":
                F[..., j] = 0.0
            elif bc.kind == "flow":                            # Danckwerts, eqs. (5.14), (5.36), (5.43)
                F[..., j] = sgn * bc.flow / self.A * yb
                u_end[:, e] = sgn * bc.flow / self.A / CT[:, j]
                T_in[:, e] = bc.T
            else:                                              # imposed pressure, eqs. (5.37)-(5.42), (5.47)
                if bc.y_t is not None:
                    yb = bc.y_t(t)
                drop = (Pref - P[:, 0]) if e == 0 else (P[:, -1] - Pref)   # -dP/dz x dz/2
                v = _ergun(drop / (0.5 * dz), a[:, j], b[:, j])
                u_end[:, e] = v
                inflow = sgn * v > 0
                F[..., j] = v[:, None] * C[..., j]             # outflow (or zero-gradient inflow): dC/dz = 0
                if yb is not None:
                    F[..., j] = np.where(inflow[:, None], v[:, None] * yb * Pref / (R * bc.T), F[..., j])
                if bc.T is not None:
                    T_in[:, e] = np.where(inflow, bc.T, np.nan)
        u = np.concatenate([u_end[:, :1], u_int, u_end[:, 1:]], axis=1)
        return F, u, CT, yfr, P, T_in

    def _gas(self, C, Tg, dP, t=0.0):
        return tuple(v[0] for v in self._gas_b(C[None], Tg[None], dP[None], t))

    def _rhs_b(self, t, Y):
        ns, N, dz, eps = self.ns, self.N, self.dz, self.eps
        C, Cm, q, Tg, Tp, Tw, dP = self._unpack_b(Y, t)
        F, u, CT, yfr, P, T_in = self._gas_b(C, Tg, dP, t)

        Cs = (self.Bi * C + Cm) / (1 + self.Bi)                                    # eq. (5.9)
        film = self.kfa * (C - Cs)
        dq = self.Om_c * self.Dc0 * np.exp(-self.Ea / (R * Tp[:, None])) * (self._qstar(Cm, Tp) - q)  # (5.7)
        dCm = self.kmac * (Cs - Cm) - self.rho_p / self.eps_p * dq                # (5.6)
        dC = (-np.diff(F, axis=-1) / dz - film) / eps                             # (5.3)

        if self.isothermal:
            dTg = dTp = dTw = np.zeros_like(Tg)
        else:
            col = self.col
            Cp_mix = self.Cp @ yfr
            Tf = np.concatenate([np.where(np.isnan(T_in[:, :1]), Tg[:, :1], T_in[:, :1]), _faces(Tg, u[:, 1:-1]),
                                 np.where(np.isnan(T_in[:, 1:]), Tg[:, -1:], T_in[:, 1:])], axis=1)
            conv = -0.5 * (u[:, :-1] + u[:, 1:]) * CT * Cp_mix * np.diff(Tf, axis=-1) / dz
            z = np.zeros_like(Tg[:, :1])
            Qc = np.concatenate([z, self.lam_f * np.diff(Tg, axis=-1) / dz, z], axis=1)
            hgs = (1 - eps) * self.a_p * self.hf * (Tg - Tp)
            dTg = (np.diff(Qc, axis=-1) / dz + conv + eps * R * Tg * dC.sum(1) - hgs
                   - 4 * self.hw_c / col.d_wi * (Tg - Tw)) / (eps * CT * (Cp_mix - R))  # (5.5)
            cap = (1 - eps) * (self.eps_p * (self.Cv @ Cm) + self.rho_p * (self.Cv @ q) + self.rho_p * self.Cps)
            dTp = ((1 - eps) * self.eps_p * R * Tp * dCm.sum(1)
                   + (1 - eps) * self.rho_p * (self.dH * dq).sum(1) + hgs) / cap      # (5.8)
            dTw = (self.alpha_w * self.hw_c * (Tg - Tw)
                   - self.alpha_wl * self.U_c * (Tw - col.T_amb)) / self.rhoCp_w           # (5.11)

        ddP = R * (Tg * dC.sum(1) + CT * dTg) - self.dP_ref(t)                    # P = C_T R Tg
        B = Y.shape[0]
        return np.concatenate([dC[:, :-1].reshape(B, -1), ddP, dCm.reshape(B, -1), dq.reshape(B, -1), dTg, dTp, dTw,
                               self.A * F[..., 0], self.A * F[..., -1]], axis=1)

    def rhs(self, t, y):
        return self._rhs_b(t, y[None])[0]

    def sparsity(self):
        """Gas variables couple to gas variables of cells j-2..j+2 (limiter stencil); everything else is local."""
        ns, N = self.ns, self.N
        b = ns * N

        def var(kind, i, j):  # kind: 0 C, 1 Cm, 2 q, 3 Tg, 4 Tp, 5 Tw
            return kind * b + i * N + j if kind < 3 else 3 * b + (kind - 3) * N + j

        local = lambda j: [var(k, i, j) for k in range(3) for i in range(ns)] + [var(k, 0, j) for k in (3, 4, 5)]
        gas = lambda j: [var(0, i, j) for i in range(ns)] + [var(3, 0, j)]
        rows, cols = [], []
        for j in range(N):
            loc = local(j)
            nb = [c for jj in range(max(0, j - 2), min(N, j + 3)) for c in gas(jj)]
            for r in loc:
                deps = loc + nb if r in gas(j) else loc
                rows += [r] * len(deps)
                cols += deps
        for r0, cells in ((3 * b + 3 * N, (0, 1)), (3 * b + 3 * N + ns, (N - 2, N - 1))):  # cumulative in / out
            deps = [c for jj in cells for c in gas(jj)]
            for i in range(ns):
                rows += [r0 + i] * len(deps)
                cols += deps
        rows += list(range(self.nv))
        cols += list(range(self.nv))
        return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.nv, self.nv)).tocsr()

    def jac(self, t, y):
        """Sparse forward-difference Jacobian; columns grouped so that no two in a group share a row. The base
        state and all perturbed states are evaluated in one batched call."""
        h = 1.5e-8 * np.maximum(np.abs(y), self.yscale)
        ng = self._jg.max() + 1
        Y = np.repeat(y[None], ng + 1, axis=0)
        Y[1 + self._jg, np.arange(self.nv)] += h
        Fb = self._rhs_b(t, Y)
        eg = self._jg[self._jc]
        data = (Fb[1 + eg, self._jr] - Fb[0, self._jr]) / h[self._jc]
        return csc_matrix((data, (self._jr, self._jc)), shape=(self.nv, self.nv))

    def inventory(self, y, t=0.0):
        """Moles of each species held in the bed (gas + macropores + adsorbed)."""
        C, Cm, q, *_ = self._unpack(y, t)
        e = self.eps
        return self.A * self.dz * (e * C + (1 - e) * (self.eps_p * Cm + self.rho_p * q)).sum(1)

    # ------------------------------------------------------------------ driver
    def solve(self, t_end: float, n_out: int = 601, rtol: float = 1e-6, y0=None) -> Result:
        """Integrate over [0, t_end] from y0 (default: the initial state). The cumulative-flow entries of
        y0 are reset to zero, so `moles` and `closure` refer to this solve only."""
        ns, N = self.ns, self.N
        y0 = (self.y0 if y0 is None else y0).copy()
        y0[-2 * ns:] = 0.0
        t_eval = np.linspace(0.0, t_end, n_out)
        t0 = time.perf_counter()
        sol = solve_ivp(self.rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval,
                        jac=self.jac, rtol=rtol, atol=self.atol)
        runtime = time.perf_counter() - t0
        if not sol.success:
            raise RuntimeError(sol.message)
        Y = sol.y
        flows, fin, Tg, Tw, P = [], [], [], [], []
        for k in range(Y.shape[1]):
            C, _, _, tg, _, tw, dP = self._unpack(Y[:, k], sol.t[k])
            F, _, _, _, p, _ = self._gas(C, tg, dP, sol.t[k])
            flows.append(self.A * F[:, -1])
            fin.append(self.A * F[:, 0])
            Tg.append(tg), Tw.append(tw), P.append(p)
        inv0, inv1 = self.inventory(Y[:, 0], 0.0), self.inventory(Y[:, -1], sol.t[-1])
        n_in, n_out_ = Y[-2 * ns:-ns, -1], Y[-ns:, -1]
        err = inv1 - inv0 - (n_in - n_out_)
        scale = np.maximum(np.maximum(n_in, inv0), 1e-300)
        closure = {s: float(err[i] / scale[i]) for i, s in enumerate(self.sp)}
        closure["total"] = float(err.sum() / max(n_in.sum(), inv0.sum()))
        return Result(sol.t, self.sp, np.array(flows).T, self.z, np.array(Tg).T, np.array(Tw).T,
                      np.array(P).T, closure, runtime, sol.nfev, np.array(fin).T, Y[:, -1].copy(),
                      np.array([n_in, n_out_]))


# ---------------------------------------------------------------------- builders
def ac5ks_column(tr: dict, T_amb: float, *, mixing: str | None = None, measured_kinetics=None,
                 thermal: str | dict | None = None, eps_p: float | None = None) -> Column:
    """The compact-unit column packed with AC5-KS (Lopes 2010, Tables 4.5, 5.3, 5.5) with transport
    parameters `tr` (Dax, lambda, kf, hf, hw, U, Dp per species: Table 5.7 or 5.9).

    mixing: virial mixture rule (None: `isotherm.mixing` in ac5ks_lopes2010.yaml).
    measured_kinetics: species whose micropore constants are the Arrhenius refit of Table 4.10 instead of
        Table 5.7/5.9 (None: `psa_transport.micropore_use_measured`, errata E7).
    thermal: "calibrated" | "assumed" solid and wall heat capacities (None: calibrated if the YAML has them),
        or {"Cps": J kg-1 K-1, "Cw": J m-1 K-1} explicitly (sensitivity cases).
    eps_p: pellet porosity override (diagnostics only; the value of record is Table 4.5).
    """
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    ac = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")
    asm, cg = cc["assumed"], cc["column"]
    ph, tp = ac["physical"], ac["psa_transport"]
    mic = {k: (v["Dc0_over_rc2"], v["Ea"]) for k, v in tp["micropore"].items()}
    for s in tp["micropore_use_measured"] if measured_kinetics is None else measured_kinetics:
        mic[s] = arrhenius_fit(ac["micropore_kinetics_breakthrough"][s]["Dc_over_rc2"])
    d, e = cg["internal_diameter"], asm["wall_thickness"]["value"]
    if isinstance(thermal, dict):
        Cps, Cw = thermal["Cps"], thermal["Cw"]
    elif (thermal or ("calibrated" if "calibrated" in cc else "assumed")) == "calibrated":
        Cps, Cw = cc["calibrated"]["solid_heat_capacity"]["value"], cc["calibrated"]["wall_heat_capacity_per_length"]["value"]
    else:
        Cps = asm["solid_heat_capacity"]["value"]
        Cw = wall_heat_capacity(e)
    layer = Layer(
        length=cg["bed_height"], eps=cg["bed_porosity"], iso=Virial.from_yaml(PARAMS / "ac5ks_lopes2010.yaml", mixing),
        rho_p=ph["particle_density"], eps_p=eps_p or ph["pellet_porosity"], Rp=ph["pellet_radius"],
        Omega_m=ph["ldf_factor_macropore"], Omega_c=ph["ldf_factor_micropore"],
        Cps=Cps, a_p=asm["particle_specific_area"]["value"] / ph["pellet_radius"],
        kf=tr["kf"], hf=tr["hf"], Dp={**tr["Dp"], "He": tr["Dp"]["H2"]},
        Dc0={k: v[0] for k, v in mic.items()}, Ea={k: v[1] for k, v in mic.items()})
    return Column(d_wi=d, e_wall=e, Cw=Cw, Dax=tr["Dax"], lam=tr["lambda"], hw=tr["hw"], U=tr["U"],
                  T_amb=T_amb, layers=[layer])


def wall_heat_capacity(e: float) -> float:
    """Lumped wall heat capacity per unit length [J m-1 K-1] of a stainless-steel wall of thickness e [m]
    (assumed density and heat capacity, compact_column_lopes2010.yaml)."""
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    asm, d = cc["assumed"], cc["column"]["internal_diameter"]
    return asm["wall_density"]["value"] * asm["wall_heat_capacity"]["value"] * np.pi * e * (d + e)


def compact_breakthrough(run: int, N: int = 60, *, mixing: str | None = None, measured_kinetics=None,
                         thermal: str | dict | None = None, **kw) -> ColumnModel:
    """Compact-unit breakthrough run 1-5 on AC5-KS (Lopes 2010, Tables 5.3 and 5.7), bed initially full of He.
    Options as in `ac5ks_column`."""
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    bt = cc["breakthrough_runs"]
    r = bt["runs"][run]
    col = ac5ks_column(r, cc["assumed"]["ambient_temperature_breakthrough"]["value"], mixing=mixing,
                       measured_kinetics=measured_kinetics, thermal=thermal)
    std = bt["std_conditions"]
    flow = std["P_bar"] * 1e5 * bt["flow_rate_std"] / (R * std["T"])
    feed = Feed(y=r["y"], T=bt["feed_temperature"], flow=flow, P_out=bt["pressure_bar"] * 1e5)
    return ColumnModel(col, feed, list(r["y"]) + ["He"], N=N, y0={"He": 1.0}, **kw)


# ---------------------------------------------------------------------- layered bed (step 4)
HY2SEPS = {"AC": "hy2seps_ac_lopes2010.yaml", "ZEO": "hy2seps_zeo_lopes2010.yaml"}


def hy2seps_transport(kind: str, y: dict, T: float, P: float, u0: float, d_wi: float, e_wall: float, L: float,
                      species: list, dax: str = "eq5.20") -> dict:
    """Transport coefficients of a HY2SEPS layer from the thesis correlations (eqs 5.17-5.35) at inlet
    conditions (feed composition y, T [K], P [Pa], superficial velocity u0 [m/s])."""
    from .transport import Gas, coefficients
    ph = load_yaml(PARAMS / HY2SEPS[kind])["physical"]
    g = Gas(species)
    return coefficients(g, [y.get(s, 0.0) for s in species], T, P, u0, dp=ph["pellet_diameter"],
                        eps=ph["bed_porosity_ch3"], d_wi=d_wi, e_wall=e_wall, L=L, tau=ph["tortuosity"], dax=dax)


def hy2seps_layer(kind: str, length: float, tr: dict) -> Layer:
    """HY2SEPS-AC ("AC") or HY2SEPS-ZEO ("ZEO") layer: multisite Langmuir (Tables 2.7, 2.8), micropore/crystal
    kinetics refitted to the measured Dc/rc2 (Tables 3.12, 3.15; errata E5 convention), particle properties
    (Tables 3.10, 3.13), transport coefficients `tr` (Dax, lambda, kf, hf, hw, U, Dp per species)."""
    from .isotherms import MultisiteLangmuir
    d = load_yaml(PARAMS / HY2SEPS[kind])
    ph = d["physical"]
    kin = d["micropore_kinetics" if kind == "AC" else "crystal_kinetics"]
    mic = {s: arrhenius_fit(v["Dc_over_rc2"]) for s, v in kin.items() if s != "units"}
    asm = load_yaml(PARAMS / "compact_column_lopes2010.yaml")["assumed"]
    Rp = ph["pellet_diameter"] / 2
    return Layer(length=length, eps=ph["bed_porosity_ch3"], iso=MultisiteLangmuir.from_yaml(PARAMS / HY2SEPS[kind]),
                 rho_p=ph["particle_density"], eps_p=ph["pellet_porosity"], Rp=Rp,
                 Omega_m=ph["ldf_factor_macropore"], Omega_c=ph["ldf_factor_micropore"],
                 Cps=ph["solid_heat_capacity"]["value"], a_p=asm["particle_specific_area"]["value"] / Rp,
                 kf=tr["kf"], hf=tr["hf"], Dp=tr["Dp"], Dc0={k: v[0] for k, v in mic.items()},
                 Ea={k: v[1] for k, v in mic.items()}, Dax=tr["Dax"], lam=tr["lambda"], hw=tr["hw"], U=tr["U"])


def layered_column(f_ac: float, y: dict, T: float, P: float, flow: float, species: list, *, L: float, d_wi: float,
                   e_wall: float, Cw: float, T_amb: float, dax: str = "eq5.20", transport: dict | None = None) -> Column:
    """HY2SEPS-AC (feed end, fraction f_ac of the bed length L) + HY2SEPS-ZEO (product end). Transport
    coefficients per layer from the correlations at the feed conditions, unless `transport` gives
    {"AC": tr, "ZEO": tr} explicitly. Zero-length layers are dropped."""
    u0 = flow * R * T / (P * np.pi * d_wi**2 / 4)
    layers = []
    for kind, length in (("AC", f_ac * L), ("ZEO", (1 - f_ac) * L)):
        if length <= 1e-12:
            continue
        tr = (transport or {}).get(kind) or hy2seps_transport(kind, y, T, P, u0, d_wi, e_wall, L, species, dax)
        layers.append(hy2seps_layer(kind, length, tr))
    l0 = layers[0]
    return Column(d_wi=d_wi, e_wall=e_wall, Cw=Cw, Dax=l0.Dax, lam=l0.lam, hw=l0.hw, U=l0.U, T_amb=T_amb, layers=layers)
