"""Adsorption isotherms used by the PSA digital twin.

Two models, both from Lopes (2010):
* Virial isotherm (pure + Taqvi-LeVan multicomponent extension) for AC5-KS.
* Multisite Langmuir (Nitta) for HY2SEPS-AC / HY2SEPS-ZEO.

All functions take temperature in K and pressure in bar and return loadings in mol/kg.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """P_i = q_i/K_H,i exp(2/S sum_j A_ij q_j + 3/(2S^2) sum_jk B_ijk q_j q_k)."""

    def __init__(self, species: dict[str, VirialSpecies], S: float):
        self.species = species
        self.S = S
        self.names = list(species)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Virial":
        d = load_yaml(path)
        sp = {k: VirialSpecies(**v) for k, v in d["isotherm"]["species"].items()}
        return cls(sp, S=d["physical"]["bet_surface_area"])

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
