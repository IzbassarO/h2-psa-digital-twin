"""Dynamic 1D fixed-bed adsorption column — Lopes (2010) ch.5, eqs (5.3)-(5.16), printed pp.167-171.

* axially dispersed plug flow, ideal gas, Ergun momentum balance (5.4) evaluated at cell faces
* bi-LDF mass transfer (5.6)-(5.7) with film resistance through the Biot number (5.9)-(5.10)
* gas (5.5), solid (5.8) and wall (5.11)-(5.13) energy balances
* finite volumes; convection upwinded with the van Leer limiter; Danckwerts inlet, fixed outlet pressure
* scipy BDF with a banded sparse Jacobian pattern

SI units inside the model (pressure in Pa); isotherms are called with partial pressures in bar.
An isotherm is any object with `.names`, `.equilibrium(P_bar (n, k), T (n,)) -> q (n, k)` and
`.species[name].minus_dH` [kJ/mol]; each layer of the bed carries its own isotherm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csc_matrix

from .isotherms import R, Virial, load_yaml

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


@dataclass
class Column:
    d_wi: float        # internal diameter, m
    e_wall: float      # wall thickness, m
    rho_w: float       # wall density, kg m-3
    Cpw: float         # wall heat capacity, J kg-1 K-1
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

        # per-cell layer properties
        edges = np.cumsum([l.length for l in col.layers])
        lay = np.minimum(np.searchsorted(edges, self.z), len(col.layers) - 1)
        cell = lambda f: np.array([f(col.layers[k]) for k in lay])
        spc = lambda f: np.array([[f(col.layers[k], s) for k in lay] for s in species])
        self.eps, self.rho_p, self.eps_p = cell(lambda l: l.eps), cell(lambda l: l.rho_p), cell(lambda l: l.eps_p)
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

        yf = np.array([feed.y.get(s, 0.0) for s in species])
        self.y_feed = yf / yf.sum()
        self.N_in = feed.flow / self.A  # mol m-2 s-1

        y0 = y0 or {"He": 1.0}
        y0v = np.array([y0.get(s, 0.0) for s in species])
        C0 = np.outer(y0v / y0v.sum(), np.full(N, feed.P_out / (R * feed.T)))
        T0 = np.full(N, feed.T)
        # State layout: [G, Cm, q, Tg, Tp, Tw, n_in, n_out]. G holds C_i for all species but the last and,
        # in its last row, dP = P - P_out. The pressure drop is ~1e-6 P, so P is carried as a deviation
        # with its own tolerance: with C_T as a state, the integration error alone (rtol x P) would be
        # comparable to the Ergun pressure drop and the face velocities would chatter.
        # n_in, n_out: cumulative moles in/out (mole balance only; no other equation reads them).
        G0 = np.vstack([C0[:-1], np.zeros(N)])
        self.y0 = np.concatenate([G0.ravel(), C0.ravel(), self._qstar(C0, T0).ravel(),
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

    # ------------------------------------------------------------------ pieces
    def _unpack(self, y):
        """-> C (ns, N) with the last species from the ideal-gas law, Cm, q, Tg, Tp, Tw, dP."""
        ns, N = self.ns, self.N
        b = ns * N
        G = y[:b].reshape(ns, N)
        Tg = y[3 * b:3 * b + N]
        dP = G[-1]
        C = np.vstack([G[:-1], (self.feed.P_out + dP) / (R * Tg) - G[:-1].sum(0)])
        return (C, y[b:2 * b].reshape(ns, N), y[2 * b:3 * b].reshape(ns, N),
                Tg, y[3 * b + N:3 * b + 2 * N], y[3 * b + 2 * N:3 * b + 3 * N], dP)

    def _qstar(self, Cm, Tp):
        q = np.zeros_like(Cm)
        for sl, iso, idx in self.layer_maps:
            Pb = np.zeros((sl.stop - sl.start, len(idx)))
            have = idx >= 0
            Pb[:, have] = (Cm[idx[have], sl] * R * Tp[sl]).T / 1e5
            # below P_MIN continue linearly through zero (Henry region): smooth for slightly negative Cm
            Pc = np.maximum(Pb, _P_MIN)
            qq = iso.equilibrium(Pc, Tp[sl]) * (Pb / Pc)
            q[idx[have], sl] = qq[:, have].T
        return q

    def _gas(self, C, Tg, dP):
        """Face fluxes F (ns, N+1) [mol m-2 s-1], face velocities u (N+1), and mixture properties."""
        CT = C.sum(0)
        yfr = C / CT
        P = self.feed.P_out + dP
        eps, dp, dz = self.eps, self.dp, self.dz
        mu, rho = self.mu @ yfr, self.M @ C
        a = 150 * mu * (1 - eps) ** 2 / (eps**3 * dp**2)
        b = 1.75 * (1 - eps) * rho / (eps**3 * dp)
        af, bf = 0.5 * (a[:-1] + a[1:]), 0.5 * (b[:-1] + b[1:])
        u_int = _ergun(-np.diff(dP) / dz, af, bf)
        u_out = _ergun(dP[-1] / (0.5 * dz), a[-1], b[-1])                      # P(L) = P_out
        eps_f = 0.5 * (eps[:-1] + eps[1:])
        J = -eps_f * self.col.Dax * 0.5 * (CT[:-1] + CT[1:]) * np.diff(yfr, axis=1) / dz
        F = np.empty((self.ns, self.N + 1))
        F[:, 0] = self.N_in * self.y_feed                      # Danckwerts, eq. (5.14)
        F[:, 1:-1] = u_int * _faces(C, u_int) + J
        F[:, -1] = u_out * C[:, -1]                            # dC/dz = 0 at z = L, eq. (5.15)
        u = np.concatenate([[self.N_in / CT[0]], u_int, [u_out]])
        return F, u, CT, yfr, P

    def rhs(self, t, y):
        ns, N, dz, eps = self.ns, self.N, self.dz, self.eps
        C, Cm, q, Tg, Tp, Tw, dP = self._unpack(y)
        F, u, CT, yfr, P = self._gas(C, Tg, dP)

        Cs = (self.Bi * C + Cm) / (1 + self.Bi)                                    # eq. (5.9)
        film = self.kfa * (C - Cs)
        dq = self.Om_c * self.Dc0 * np.exp(-self.Ea / (R * Tp)) * (self._qstar(Cm, Tp) - q)  # (5.7)
        dCm = self.kmac * (Cs - Cm) - self.rho_p / self.eps_p * dq                # (5.6)
        dC = (-np.diff(F, axis=1) / dz - film) / eps                              # (5.3)

        if self.isothermal:
            dTg = dTp = dTw = np.zeros(N)
        else:
            col = self.col
            Cp_mix = self.Cp @ yfr
            Tf = np.concatenate([[self.feed.T], _faces(Tg, u[1:-1]), [Tg[-1]]])
            conv = -0.5 * (u[:-1] + u[1:]) * CT * Cp_mix * np.diff(Tf) / dz
            Qc = np.concatenate([[0.0], col.lam * np.diff(Tg) / dz, [0.0]])
            hgs = (1 - eps) * self.a_p * self.hf * (Tg - Tp)
            dTg = (np.diff(Qc) / dz + conv + eps * R * Tg * dC.sum(0) - hgs
                   - 4 * col.hw / col.d_wi * (Tg - Tw)) / (eps * CT * (Cp_mix - R))  # (5.5)
            cap = (1 - eps) * (self.eps_p * (self.Cv @ Cm) + self.rho_p * (self.Cv @ q) + self.rho_p * self.Cps)
            dTp = ((1 - eps) * self.eps_p * R * Tp * dCm.sum(0)
                   + (1 - eps) * self.rho_p * (self.dH * dq).sum(0) + hgs) / cap      # (5.8)
            dTw = (self.alpha_w * col.hw * (Tg - Tw)
                   - self.alpha_wl * col.U * (Tw - col.T_amb)) / (col.rho_w * col.Cpw)  # (5.11)

        ddP = R * (Tg * dC.sum(0) + CT * dTg)                                     # P = C_T R Tg
        return np.concatenate([dC[:-1].ravel(), ddP, dCm.ravel(), dq.ravel(), dTg, dTp, dTw,
                               self.A * F[:, 0], self.A * F[:, -1]])

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
        out0 = 3 * b + 3 * N + ns
        for i in range(ns):
            deps = [c for jj in (N - 2, N - 1) for c in gas(jj)]
            rows += [out0 + i] * len(deps)
            cols += deps
        rows += list(range(self.nv))
        cols += list(range(self.nv))
        return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.nv, self.nv)).tocsr()

    def jac(self, t, y):
        """Sparse forward-difference Jacobian; columns grouped so that no two in a group share a row."""
        f0 = self.rhs(t, y)
        h = 1.5e-8 * np.maximum(np.abs(y), self.yscale)
        data = np.empty(len(self._jr))
        eg = self._jg[self._jc]
        for g in range(self._jg.max() + 1):
            yp = y.copy()
            cols = self._jg == g
            yp[cols] += h[cols]
            df = self.rhs(t, yp) - f0
            sel = eg == g
            data[sel] = df[self._jr[sel]] / h[self._jc[sel]]
        return csc_matrix((data, (self._jr, self._jc)), shape=(self.nv, self.nv))

    def inventory(self, y):
        """Moles of each species held in the bed (gas + macropores + adsorbed)."""
        C, Cm, q, *_ = self._unpack(y)
        e = self.eps
        return self.A * self.dz * (e * C + (1 - e) * (self.eps_p * Cm + self.rho_p * q)).sum(1)

    # ------------------------------------------------------------------ driver
    def solve(self, t_end: float, n_out: int = 601, rtol: float = 1e-6) -> Result:
        ns, N = self.ns, self.N
        t_eval = np.linspace(0.0, t_end, n_out)
        t0 = time.perf_counter()
        sol = solve_ivp(self.rhs, (0.0, t_end), self.y0, method="BDF", t_eval=t_eval,
                        jac=self.jac, rtol=rtol, atol=self.atol)
        runtime = time.perf_counter() - t0
        if not sol.success:
            raise RuntimeError(sol.message)
        Y = sol.y
        flows, Tg, Tw, P = [], [], [], []
        for k in range(Y.shape[1]):
            C, _, _, tg, _, tw, dP = self._unpack(Y[:, k])
            F, _, _, _, p = self._gas(C, tg, dP)
            flows.append(self.A * F[:, -1])
            Tg.append(tg), Tw.append(tw), P.append(p)
        inv0, inv1 = self.inventory(Y[:, 0]), self.inventory(Y[:, -1])
        n_in, n_out_ = Y[-2 * ns:-ns, -1], Y[-ns:, -1]
        err = inv1 - inv0 - (n_in - n_out_)
        scale = np.maximum(np.maximum(n_in, inv0), 1e-300)
        closure = {s: float(err[i] / scale[i]) for i, s in enumerate(self.sp)}
        closure["total"] = float(err.sum() / max(n_in.sum(), inv0.sum()))
        return Result(sol.t, self.sp, np.array(flows).T, self.z, np.array(Tg).T, np.array(Tw).T,
                      np.array(P).T, closure, runtime, sol.nfev)


# ---------------------------------------------------------------------- builders
def compact_breakthrough(run: int, N: int = 60, **kw) -> ColumnModel:
    """Compact-unit breakthrough run 1-5 on AC5-KS (Lopes 2010, Tables 5.3 and 5.7), bed initially full of He."""
    cc = load_yaml(PARAMS / "compact_column_lopes2010.yaml")
    ac = load_yaml(PARAMS / "ac5ks_lopes2010.yaml")
    bt, asm, cg = cc["breakthrough_runs"], cc["assumed"], cc["column"]
    r = bt["runs"][run]
    ph, mic = ac["physical"], ac["psa_transport"]["micropore"]
    species = list(r["y"]) + ["He"]
    layer = Layer(
        length=cg["bed_height"], eps=cg["bed_porosity"], iso=Virial.from_yaml(PARAMS / "ac5ks_lopes2010.yaml"),
        rho_p=ph["particle_density"], eps_p=ph["pellet_porosity"], Rp=ph["pellet_radius"],
        Omega_m=ph["ldf_factor_macropore"], Omega_c=ph["ldf_factor_micropore"],
        Cps=asm["solid_heat_capacity"]["value"], a_p=asm["particle_specific_area"]["value"] / ph["pellet_radius"],
        kf=r["kf"], hf=r["hf"], Dp={**r["Dp"], "He": r["Dp"]["H2"]},
        Dc0={k: v["Dc0_over_rc2"] for k, v in mic.items()}, Ea={k: v["Ea"] for k, v in mic.items()})
    col = Column(d_wi=cg["internal_diameter"], e_wall=asm["wall_thickness"]["value"],
                 rho_w=asm["wall_density"]["value"], Cpw=asm["wall_heat_capacity"]["value"],
                 Dax=r["Dax"], lam=r["lambda"], hw=r["hw"], U=r["U"],
                 T_amb=asm["ambient_temperature_breakthrough"]["value"], layers=[layer])
    std = bt["std_conditions"]
    flow = std["P_bar"] * 1e5 * bt["flow_rate_std"] / (R * std["T"])
    feed = Feed(y=r["y"], T=bt["feed_temperature"], flow=flow, P_out=bt["pressure_bar"] * 1e5)
    return ColumnModel(col, feed, species, N=N, y0={"He": 1.0}, **kw)
