"""Transport coefficients from the correlations of Lopes (2010).

Gas properties: Chapman-Enskog binary diffusivities (eq. 3.9-3.13), viscosities (eq. 3.24-3.25), Wilke mixture
viscosity (eq. 3.23, 3.26) and mixture molecular diffusivity (eq. 3.8 = 5.18), Lennard-Jones parameters of
Table 3.1 (p.76). Coefficients: Bosanquet pore diffusivity (5.17, Knudsen 5.19), Wakao-Funazkri axial
dispersion, heat dispersion, film heat and mass transfer (5.20-5.28), overall wall coefficient (5.29-5.30),
Wasch-Froment h_w (5.31) and Churchill-Chu h_ex (5.32-5.35). The ch.3 axial dispersion correlation (eq. 3.18)
is kept as an alternative (`dax_eq318`).

SI units in and out: T [K], P [Pa], lengths [m], velocities [m s-1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .isotherms import R, load_yaml

PARAMS = Path(__file__).resolve().parents[2] / "data" / "params"


class Gas:
    """Pure and mixture gas properties for the species in `species` (Table 3.1 LJ parameters)."""

    def __init__(self, species):
        lj = load_yaml(PARAMS / "gas_transport_lopes2010.yaml")["lennard_jones"]
        nist = load_yaml(PARAMS / "gas_properties_nist.yaml")
        self.sp = list(species)
        self.M = np.array([lj[s]["M"] for s in self.sp])               # g mol-1
        self.sig = np.array([lj[s]["sigma"] for s in self.sp])         # angstrom
        self.ek = np.array([lj[s]["eps_k"] for s in self.sp])          # K
        self._Tg, self._Cp = nist["T"], {s: nist["species"][s]["Cp"] for s in self.sp}

    def Cp(self, T):
        """Molar heat capacities [J mol-1 K-1] (NIST table)."""
        return np.array([np.interp(T, self._Tg, self._Cp[s]) for s in self.sp])

    def viscosity(self, T):
        """Pure-gas viscosities [Pa s], eqs (3.24)-(3.25)."""
        w = T / self.ek
        om = 1.16145 / w**0.14874 + 0.52487 / np.exp(0.7732 * w) + 2.16178 / np.exp(2.43787 * w)
        return 2.669e-6 * np.sqrt(self.M * T) / (self.sig**2 * om)

    def _phi(self, mu):
        """Phi_ij of eq. (3.26)."""
        Mi, Mj = self.M[:, None], self.M[None, :]
        return (1 + Mi / Mj) ** -0.5 / np.sqrt(8) * (1 + np.sqrt(mu[:, None] / mu[None, :]) * (Mj / Mi) ** 0.25) ** 2

    def mixture_viscosity(self, y, T):
        """Wilke, eq. (3.23)."""
        mu = self.viscosity(T)
        return float(np.sum(y * mu / (self._phi(mu) @ y)))

    def conductivity(self, T):
        """Pure-gas thermal conductivities [W m-1 K-1]: modified Eucken (Bird et al. 2002, eq. 9.3-15)."""
        return (self.Cp(T) + 1.25 * R) * self.viscosity(T) / (self.M * 1e-3)

    def mixture_conductivity(self, y, T):
        """Bird et al. (2002) eq. 9.3-17 with Phi_ij of eq. (3.26)."""
        return float(np.sum(y * self.conductivity(T) / (self._phi(self.viscosity(T)) @ y)))

    def binary_diffusivity(self, T, P):
        """D_ij [m2 s-1], Chapman-Enskog eqs (3.9)-(3.13); diagonal undefined (nan)."""
        Mij = 2.0 / (1 / self.M[:, None] + 1 / self.M[None, :])
        sij = 0.5 * (self.sig[:, None] + self.sig[None, :])
        w = T / np.sqrt(self.ek[:, None] * self.ek[None, :])
        om = (1.06036 / w**0.1561 + 0.193 / np.exp(0.47635 * w) + 1.03587 / np.exp(1.52996 * w)
              + 1.76474 / np.exp(3.89411 * w))
        D = 2.66e-2 * T**1.5 / (P * np.sqrt(Mij) * sij**2 * om)
        np.fill_diagonal(D, np.nan)
        return D

    def molecular_diffusivity(self, y, T, P):
        """Mixture diffusivity of each species [m2 s-1], eq. (5.18) = (3.8). Species absent from the mixture
        (y = 0) get their value at infinite dilution."""
        D = self.binary_diffusivity(T, P)
        y = np.asarray(y, float)
        out = np.empty(len(y))
        for i in range(len(y)):
            o = np.arange(len(y)) != i
            out[i] = (1 - y[i]) / np.sum(y[o] / D[i, o])
        return out


def pore_diffusivity(Dm, tau, T=None, M=None, r_pore=None):
    """Bosanquet, eq. (5.17); Knudsen term (5.19) only if a mean pore radius is given."""
    inv = 1.0 / np.asarray(Dm)
    if r_pore is not None:
        inv = inv + 1.0 / (97.0 * r_pore * np.sqrt(T / M))
    return 1.0 / (tau * inv)


def dax_eq318(Dm, eps, Rp, u0):
    """Axial dispersion of ch.3, eq. (3.18): (0.45 + 0.55 eps) Dm + 0.35 Rp u_i (u_i interstitial)."""
    return (0.45 + 0.55 * eps) * Dm + 0.35 * Rp * u0 / eps


def overall_U(hw, h_ex, d_wi, e_wall, k_wall):
    """Global wall coefficient, eqs (5.29)-(5.30), from the inside film h_w, the outside film h_ex and the wall."""
    d_we = d_wi + 2 * e_wall
    d_ln = (d_we - d_wi) / np.log(d_we / d_wi)                           # (5.30)
    return 1.0 / (1 / hw + e_wall * d_wi / (k_wall * d_ln) + d_wi / (d_we * h_ex))   # (5.29)


def coefficients(gas: Gas, y, T, P, u0, *, dp, eps, d_wi, e_wall, L, tau, r_pore=None, dax="eq3.18"):
    """Transport coefficients of Table 5.7 / 5.9 at inlet conditions.

    y: mole fractions (gas.sp order), u0: superficial velocity, dp: particle diameter, eps: bed porosity,
    d_wi, e_wall, L: column internal diameter, wall thickness, length. The scalar coefficients use the
    mole-fraction mean of the species' mixture diffusivities (one Dax and one kf per bed). dax: "eq3.18"
    (ch.3, the baseline: reproduces Tables 5.7/5.9) or "eq5.20" (Wakao-Funazkri, stated for the PSA model
    on p.172; rejected in step 4, 51-55x the tabulated Dax; kept for the gate-3 comparison only).
    Returns dict with Dax, lambda, kf, hf, hw, U, Dp {species} and the dimensionless groups.
    """
    ext = load_yaml(PARAMS / "gas_transport_lopes2010.yaml")["external"]
    y = np.asarray(y, float) / np.sum(y)
    CT = P / (R * T)
    Mmix = float(y @ gas.M) * 1e-3
    rho, mu, kg = CT * Mmix, gas.mixture_viscosity(y, T), gas.mixture_conductivity(y, T)
    Dmi = gas.molecular_diffusivity(y, T, P)
    Dm = float(y @ Dmi)
    cp_mass = float(y @ gas.Cp(T)) / Mmix
    Re = rho * u0 * dp / mu                                              # (5.24)
    Sc = mu / (rho * Dm)                                                 # (5.25)
    Pr = cp_mass * mu / kg                                               # (5.26)
    Dax = Dm / eps * (20 + 0.5 * Sc * Re) if dax == "eq5.20" else dax_eq318(Dm, eps, dp / 2, u0)   # (5.20)
    lam = kg * (7 + 0.5 * Pr * Re)                                       # (5.21)
    hf = kg / dp * (2.0 + 1.1 * Re**0.6 * Pr ** (1 / 3))                 # (5.22), (5.28)
    kf = Dm / dp * (2.0 + 1.1 * Re**0.6 * Sc ** (1 / 3))                 # (5.23), (5.27)
    hw = kg / d_wi * (140 + 0.013396 * d_wi**2 / (dp * kg) * Re)         # (5.31) as printed
    # outside: Churchill-Chu with properties of air at the film temperature (5.32)-(5.35)
    air, dT = ext["air_300K"], ext["wall_minus_ambient"]["value"]
    beta = 1.0 / (T + 0.5 * dT)                                          # ideal gas at T_film
    Gr = ext["gravity"] * beta * dT * L**3 / (air["nu"] * air["alpha"])  # (5.34)
    Ra = Gr * air["Pr"]                                                  # (5.33)
    hex_ = air["k"] / L * (0.68 + 0.67 * Ra**0.25 / (1 + (0.492 / air["Pr"]) ** (9 / 16)) ** (4 / 9))
    U = overall_U(hw, hex_, d_wi, e_wall, ext["wall_conductivity"]["value"])
    Dp = pore_diffusivity(Dmi, tau, T, gas.M, r_pore)
    return dict(Dax=Dax, **{"lambda": lam}, kf=kf, hf=hf, hw=hw, U=U, Dp=dict(zip(gas.sp, Dp)),
                Re=Re, Sc=Sc, Pr=Pr, kg=kg, mu=mu, rho=rho, Dm=Dm, h_ex=hex_)
