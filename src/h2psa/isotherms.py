"""Adsorption isotherms used by the PSA digital twin.

Two models, both from Lopes (2010):
* Virial isotherm (pure + Taqvi-LeVan multicomponent extension) for AC5-KS.
* Multisite Langmuir (Nitta) for HY2SEPS-AC / HY2SEPS-ZEO.

All functions take temperature in K and pressure in bar and return loadings in mol/kg.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.optimize import brentq, fsolve

R = 8.314462618  # J mol-1 K-1


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML-1.2 floats such as 9.71e5 (PyYAML treats them as strings)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    __import__("re").compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        __import__("re").X,
    ),
    list("-+0123456789."),
)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_Loader)  # noqa: S506 - SafeLoader subclass


# --------------------------------------------------------------------------- virial
@dataclass(frozen=True)
class VirialSpecies:
    K_inf: float      # mol kg-1 bar-1
    minus_dH: float   # kJ mol-1
    A0: float         # m2 mol-1
    A1: float         # m2 K mol-1
    B0: float         # m4 mol-2
    B1: float         # m4 K mol-2

    def KH(self, T: float) -> float:
        return self.K_inf * np.exp(self.minus_dH * 1e3 / (R * T))

    def A(self, T: float) -> float:
        return self.A0 + self.A1 / T

    def B(self, T: float) -> float:
        return self.B0 + self.B1 / T


class Virial:
    """P_i = q_i/K_H,i exp(2/S sum_j A_ij q_j + 3/(2S^2) sum_jk B_ijk q_j q_k).

    `mixing` selects the mixture rule used by `equilibrium`:
    * "taqvi_levan": A_ij = (A_i+A_j)/2, B_ijk = (B_i+B_j+B_k)/3 (Lopes 2010 eq. 5.1-5.2, published);
    * "iast": ideal adsorbed solution theory on the pure-component virial isotherms;
    * "noncompetitive_h2": Taqvi-LeVan, but H2 adsorbs as if alone at its partial pressure (upper bound).
    """
    MIXING = ("taqvi_levan", "iast", "noncompetitive_h2")

    def __init__(self, species: dict[str, VirialSpecies], S: float, mixing: str = "taqvi_levan"):
        if mixing not in self.MIXING:
            raise ValueError(f"mixing must be one of {self.MIXING}, got {mixing!r}")
        self.species = species
        self.S = S
        self.names = list(species)
        self.mixing = mixing

    @classmethod
    def from_yaml(cls, path: str | Path, mixing: str | None = None) -> "Virial":
        d = load_yaml(path)
        sp = {k: VirialSpecies(**v) for k, v in d["isotherm"]["species"].items()}
        return cls(sp, S=d["physical"]["bet_surface_area"], mixing=mixing or d["isotherm"].get("mixing", "taqvi_levan"))

    def pressure_pure(self, name: str, q: float | np.ndarray, T: float) -> np.ndarray:
        s, S = self.species[name], self.S
        q = np.asarray(q, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            return q / s.KH(T) * np.exp(2 / S * s.A(T) * q + 1.5 / S**2 * s.B(T) * q * q)

    def loading_pure(self, name: str, P: float, T: float, q_max: float = 30.0) -> float:
        """Smallest root of P(q) = P (isotherm is monotone in its fitted range)."""
        grid = np.linspace(0.0, q_max, 30001)
        Pq = self.pressure_pure(name, grid, T)
        idx = np.argmax(Pq >= P)
        if Pq[idx] < P:
            raise ValueError(f"{name}: P={P} bar not reached for q<{q_max}")
        return brentq(lambda x: self.pressure_pure(name, x, T) - P, grid[max(idx - 1, 0)], grid[idx])

    def loading_mix(self, y: dict[str, float], P: float, T: float) -> dict[str, float]:
        """Solve the Taqvi-LeVan virial mixture equations in log space for species with y_i > 0."""
        names = [n for n in self.names if y.get(n, 0.0) > 0.0]
        out = {n: 0.0 for n in self.names}
        if not names:
            return out
        A = np.array([self.species[n].A(T) for n in names])
        B = np.array([self.species[n].B(T) for n in names])
        KH = np.array([self.species[n].KH(T) for n in names])
        Pi = np.array([y[n] * P for n in names])
        Aij = (A[:, None] + A[None, :]) / 2
        Bijk = (B[:, None, None] + B[None, :, None] + B[None, None, :]) / 3
        S = self.S

        def resid(lnq):
            q = np.exp(lnq)
            expo = 2 / S * Aij @ q + 1.5 / S**2 * np.einsum("ijk,j,k->i", Bijk, q, q)
            return lnq - np.log(KH) + expo - np.log(Pi)   # from P_i = q_i/K_H exp(expo)

        # start from pure-component loadings at the partial pressures, scaled down for competition
        q0 = np.array([self.loading_pure(n, p, T) for n, p in zip(names, Pi)])
        lnq = fsolve(resid, np.log(q0 / max(1.0, len(names) ** 0.5)), xtol=1e-12)
        if np.max(np.abs(resid(lnq))) > 1e-8:
            raise RuntimeError(f"virial mixture solve did not converge at P={P}, T={T}, y={y}")
        out.update(zip(names, np.exp(lnq)))
        return out

    def equilibrium(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Vectorised mixture loadings for many states at once (the column model's interface).

        P: (n, k) partial pressures [bar] in self.names order; T: (n,) [K]. Returns q (n, k) [mol/kg].
        The mixture rule is `self.mixing`. Solvers keep the last solution as warm start
        (consecutive calls from the ODE solver differ only slightly).
        """
        lnK, A, B = self._coeffs(T)
        lnP = np.log(np.maximum(P, 1e-14))
        if self.mixing == "iast":
            return self._iast(lnP, lnK, A, B)
        q = self._taqvi_levan(lnP, lnK, A, B)
        if self.mixing == "noncompetitive_h2":
            i = self.names.index("H2")
            q[:, i] = self._pure_loading(lnP[:, i], lnK[:, i], A[:, i], B[:, i])
        return q

    def _coeffs(self, T):
        sp = [self.species[n] for n in self.names]
        T = np.asarray(T, float)[:, None]
        lnK = np.log([s.K_inf for s in sp]) + np.array([s.minus_dH for s in sp]) * 1e3 / (R * T)
        A = np.array([s.A0 for s in sp]) + np.array([s.A1 for s in sp]) / T
        B = np.array([s.B0 for s in sp]) + np.array([s.B1 for s in sp]) / T
        return lnK, A + 0 * T, B + 0 * T

    def _pure_loading(self, lnP, lnK, A, B):
        """Pure-component loading at ln P (any shape): Newton on ln q, monotone in the fitted range."""
        S = self.S
        x = lnK + lnP
        for _ in range(100):
            q = np.exp(x)
            dx = -(x - lnK + 2 * A * q / S + 1.5 * B * q * q / S**2 - lnP) / (1 + 2 * A * q / S + 3 * B * q * q / S**2)
            dx = np.clip(dx, -2.0, 2.0)
            x = x + dx
            if np.abs(dx).max() < 1e-12:
                return np.exp(x)
        raise RuntimeError("virial pure-component loading did not converge")

    def _iast(self, lnP, lnK, A, B):
        """IAST on the pure virial isotherms (Myers & Prausnitz 1965).

        Reduced spreading pressure of the pure isotherm, psi = int_0^P q dlnP = q + a q^2 + b q^3 with
        a = A/S, b = B/S^2 [mol/kg]. Unknown per state: the common psi. For given psi each pure loading q0_i
        follows from psi_i(q0_i) = psi, its pressure P0_i from the pure isotherm, and
        h(psi) = ln sum_i P_i/P0_i(psi) = 0 (h decreases monotonically: d ln P0_i/d psi = 1/q0_i).
        Then x_i = P_i/P0_i, 1/q_T = sum_i x_i/q0_i, q_i = x_i q_T.
        Safeguarded Newton in ln psi (bracket kept per state), nested Newton in ln q0.

        Where b < 0 the pure virial P(q) has a maximum (CO, CH4, N2 at 11-25 bar, beyond the 0-7 bar fit) and
        psi(q) cannot exceed psi_max. Above psi_max the pure isotherm is continued at constant loading
        q_max (saturated): ln P0 = ln P0(q_max) + (psi - psi_max)/q_max, which keeps d ln P0/d psi = 1/q0.
        """
        S = self.S
        a, b = A / S, B / S**2
        turn = b < 0
        qm = np.where(turn, (-a - np.sqrt(np.maximum(a * a - 3 * b, 0.0))) / (3 * np.where(turn, b, -1.0)), 1.0)
        q_max = np.where(turn, qm, np.inf)                 # positive root of 1 + 2aq + 3bq^2
        psi_max = np.where(turn, qm + a * qm**2 + b * qm**3, np.inf)
        lnP_max = np.where(turn, np.log(qm) - lnK + 2 * a * qm + 1.5 * b * qm**2, np.inf)
        s = getattr(self, "_s_last", None)
        if s is None or s.shape != lnP.shape[:1]:
            s = np.log(np.exp(lnK + lnP).sum(1))           # Henry limit: psi = sum K_i P_i
        lo, hi = np.full_like(s, -np.inf), np.full_like(s, np.inf)
        y = getattr(self, "_y_last", None)
        for _ in range(200):
            psi = np.exp(s)[:, None]
            sat = psi >= psi_max
            y = self._q0_of_psi(np.minimum(psi, psi_max), a, b, np.log(q_max), y)
            q0 = np.exp(y)
            lnP0 = np.where(sat, lnP_max + (psi - np.where(sat, psi_max, 0.0)) / qm,
                            y - lnK + 2 * a * q0 + 1.5 * b * q0 * q0)
            x = np.exp(lnP - lnP0)
            sx = x.sum(1)
            h = np.log(sx)
            dh = -psi[:, 0] * (x / q0).sum(1) / sx          # dh / d ln psi
            lo, hi = np.where(h > 0, s, lo), np.where(h <= 0, s, hi)
            sn = s + np.clip(-h / dh, -2.0, 2.0)
            bad = (sn <= lo) | (sn >= hi)
            sn = np.where(bad & np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), sn)
            done = np.abs(sn - s).max() < 1e-12
            s = sn
            if done:
                break
        else:
            raise RuntimeError("IAST did not converge")
        self._s_last, self._y_last = s, y
        x = x / sx[:, None]
        return x / (x / q0).sum(1, keepdims=True)

    @staticmethod
    def _q0_of_psi(psi, a, b, y_max, y=None):
        """ln q0 solving q0 + a q0^2 + b q0^3 = psi <= psi_max on the rising branch (ln q0 <= y_max),
        safeguarded Newton in ln q0."""
        if y is None or y.shape != a.shape:
            y = np.log(psi)
        y = np.minimum(y, y_max)
        lo, hi = np.full_like(a, -np.inf), y_max + 0.0
        for _ in range(300):
            q = np.exp(y)
            g = q + a * q * q + b * q**3 - psi
            dg = q * (1 + 2 * a * q + 3 * b * q * q)
            above = g >= 0
            lo, hi = np.where(above, lo, y), np.where(above, np.minimum(y, hi), hi)
            yn = y + np.clip(-g / np.where(dg > 0, dg, np.inf), -2.0, 2.0)
            bad = (yn <= lo) | (yn >= hi)
            fl, fh = np.isfinite(lo), np.isfinite(hi)
            yn = np.where(bad, np.where(fl & fh, 0.5 * (lo + hi), np.where(fl, lo + 1.0, hi - 1.0)), yn)
            done = np.abs(yn - y).max() < 1e-13
            y = yn
            if done:
                return y
        raise RuntimeError("IAST: pure loading at given spreading pressure did not converge")

    def _taqvi_levan(self, lnP, lnK, A, B):
        """Damped Newton in ln q from the last solution. Cold start (or failure): continuation in the pressure
        level from the Henry limit, because Newton from a poor guess can cycle when H2, with its large B,
        sits many decades below its Henry loading."""
        x = getattr(self, "_x_last", None)
        if x is not None and x.shape == lnP.shape:
            try:
                self._x_last = self._tl_newton(lnP, lnK, A, B, x)
                return np.exp(self._x_last)
            except RuntimeError:
                pass
        x = lnK + lnP - 9.0 * np.log(10)
        for s in np.linspace(-9.0, 0.0, 19) * np.log(10):
            x = self._tl_newton(lnP + s, lnK, A, B, x)
        self._x_last = x
        return np.exp(x)

    def _tl_newton(self, lnP, lnK, A, B, x):
        S = self.S
        eye = np.eye(lnP.shape[1])
        for _ in range(100):
            q = np.exp(x)
            Q = q.sum(1, keepdims=True)
            a = (A * q).sum(1, keepdims=True)
            b = (B * q).sum(1, keepdims=True)
            r = x - lnK + (A * Q + a) / S + (B * Q**2 + 2 * Q * b) / (2 * S**2) - lnP
            J = eye + q[:, None, :] * ((A[:, :, None] + A[:, None, :]) / S
                                       + (B[:, :, None] * Q[:, :, None] + b[:, :, None] + Q[:, :, None] * B[:, None, :]) / S**2)
            dx = np.linalg.solve(J, -r[..., None])[..., 0]
            dx *= 2.0 / np.maximum(np.abs(dx).max(1, keepdims=True), 2.0)  # step limit 2 in ln q
            x = x + dx
            if np.abs(dx).max() < 1e-11:
                return x
        raise RuntimeError("virial equilibrium did not converge")


# --------------------------------------------------------------------------- multisite Langmuir
@dataclass(frozen=True)
class MSLSpecies:
    q_max: float      # mol kg-1
    a: float          # number of sites
    K_inf: float      # kPa-1
    minus_dH: float   # kJ mol-1

    def K(self, T: float) -> float:
        return self.K_inf * np.exp(self.minus_dH * 1e3 / (R * T))


class MultisiteLangmuir:
    """q_i/q_max,i = a_i K_i P_i (1 - sum_j q_j/q_max,j)^a_i  (P in kPa internally)."""

    def __init__(self, species: dict[str, MSLSpecies]):
        self.species = species
        self.names = list(species)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultisiteLangmuir":
        d = load_yaml(path)
        return cls({k: MSLSpecies(**v) for k, v in d["isotherm"]["species"].items()})

    def loading_pure(self, name: str, P: float, T: float) -> float:
        s = self.species[name]
        PkPa = P * 100.0
        f = lambda q: q / s.q_max - s.a * s.K(T) * PkPa * (1 - q / s.q_max) ** s.a
        return brentq(f, 0.0, s.q_max * (1 - 1e-12))

    def loading_mix(self, y: dict[str, float], P: float, T: float) -> dict[str, float]:
        names = self.names
        sp = [self.species[n] for n in names]
        PkPa = np.array([y.get(n, 0.0) * P * 100.0 for n in names])

        def resid(theta):
            free = 1 - theta.sum()
            free = max(free, 1e-15)
            return np.array([theta[i] - s.a * s.K(T) * PkPa[i] * free ** s.a for i, s in enumerate(sp)])

        th = fsolve(resid, np.full(len(names), 1e-3), xtol=1e-13)
        return {n: th[i] * sp[i].q_max for i, n in enumerate(names)}

    def equilibrium(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Vectorised: P (n, k) [bar] in self.names order, T (n,) [K] -> q (n, k) [mol/kg].

        With s = 1 - sum(theta): theta_i = a_i K_i P_i s^a_i, so s solves the monotone scalar equation
        f(s) = s - 1 + sum_i a_i K_i P_i s^a_i = 0 on (0, 1]; safeguarded Newton per state.
        """
        sp = [self.species[n] for n in self.names]
        T = np.asarray(T, float)[:, None]
        a = np.array([s.a for s in sp])
        c = a * np.array([s.K_inf for s in sp]) * np.exp(np.array([s.minus_dH for s in sp]) * 1e3 / (R * T)) \
            * np.maximum(P, 0.0) * 100.0
        lo, hi, s = np.zeros(len(T)), np.ones(len(T)), np.ones(len(T))
        for _ in range(100):
            f = s - 1 + (c * s[:, None] ** a).sum(1)
            lo, hi = np.where(f < 0, s, lo), np.where(f >= 0, s, hi)
            fp = 1 + (c * a * s[:, None] ** (a - 1)).sum(1)
            sn = s - f / fp
            sn = np.where((sn <= lo) | (sn >= hi), 0.5 * (lo + hi), sn)
            if np.abs(sn - s).max() < 1e-14:
                s = sn
                break
            s = sn
        return np.array([s_.q_max for s_ in sp]) * c * s[:, None] ** a


class Linear:
    """q_i = K_i P_i (isothermal test isotherm, K in mol kg-1 bar-1)."""

    def __init__(self, K: dict[str, float], minus_dH: float = 0.0):
        self.names = list(K)
        self.K = np.array(list(K.values()))
        self.species = {n: SimpleNamespace(minus_dH=minus_dH) for n in K}

    def equilibrium(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.K * P
