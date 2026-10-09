"""Micropore/crystal diffusion kinetics.

Convention (see data/PROVENANCE.md, errata E3 and E5): the *measured* Dc/rc2 values at each temperature
are treated as primary data, and the Arrhenius law Dc/rc2 = D0 exp(-Ea/(R T)) is refitted to them by
least squares in log space. Printed (D0, Ea) pairs are kept in the YAML for traceability only.
"""
from __future__ import annotations

import numpy as np

from .isotherms import R


def arrhenius_fit(points: dict) -> tuple[float, float]:
    """Return (D0 [s-1], Ea [kJ/mol]) fitted to {T_K: Dc/rc2}."""
    T = np.array([float(t) for t in points])
    y = np.log(np.array([float(v) for v in points.values()]))
    slope, intercept = np.polyfit(1.0 / T, y, 1)
    return float(np.exp(intercept)), float(-slope * R / 1e3)


def dc_over_rc2(D0: float, Ea: float, T: float) -> float:
    return D0 * np.exp(-Ea * 1e3 / (R * T))
