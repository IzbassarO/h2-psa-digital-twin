"""Layered bed (step 4): interface handling of the finite-volume column and the HY2SEPS builders."""
from dataclasses import replace

import numpy as np
import pytest

from h2psa.column import ColumnModel, Feed, compact_breakthrough, layered_column, wall_heat_capacity
from h2psa.cycle import experiment_cycle


def _short(model, t=150.0):
    return model.solve(t, n_out=31)


def test_two_identical_layers_equal_one_layer():
    """Splitting a bed into two identical layers (interface on a face) changes nothing."""
    m1 = compact_breakthrough(5, N=30)
    col = m1.col
    l = col.layers[0]
    half = col.layers[0].length / 2
    col2 = replace(col, layers=[replace(l, length=half), replace(l, length=half)])
    m2 = ColumnModel(col2, m1.feed, m1.sp, N=30, y0={"He": 1.0})
    assert np.array_equal(m1.rhs(0.0, m1.y0), m2.rhs(0.0, m2.y0))
    r1, r2 = _short(m1), _short(m2)           # isotherm warm starts differ per array shape: solver-level noise only
    assert np.abs(r1.outlet_flow - r2.outlet_flow).max() < 1e-5 * np.abs(r1.outlet_flow).max()
    assert np.abs(r1.Tg - r2.Tg).max() < 1e-3          # K


@pytest.fixture(scope="module")
def ac_zeo():
    c = experiment_cycle()
    sp = list(c.y_feed) + ["He"]
    col = layered_column(0.6, c.y_feed, c.T_feed, c.P_high, c.feed_flow, sp, L=0.267, d_wi=0.05, e_wall=0.003,
                         Cw=wall_heat_capacity(0.003), T_amb=c.T_feed, dax="eq3.18")
    feed = Feed(y=c.y_feed, T=c.T_feed, flow=c.feed_flow, P_out=c.P_high)
    return col, feed, sp


def test_interface_conserves_moles(ac_zeo):
    """AC | ZEO interface: the face flux is single-valued, so the bed balance closes to solver accuracy."""
    col, feed, sp = ac_zeo
    m = ColumnModel(col, feed, sp, N=30, y0={"He": 1.0})
    assert len(set(m.layer_of_cell)) == 2 and m.layer_of_cell[17] == 0 and m.layer_of_cell[18] == 1
    r = m.solve(300.0, n_out=31)
    assert abs(r.closure["total"]) < 1e-6
    assert all(abs(v) < 1e-5 for k, v in r.closure.items() if k != "He")


def test_interface_flux_continuity(ac_zeo):
    """Steady inert flow (He into the He-filled AC | ZEO bed): the molar flux is the same at every face,
    including the interface face, and temperature stays uniform; only the pressure gradient changes slope
    (different eps and d_p in the Ergun equation)."""
    col, feed, sp = ac_zeo
    m = ColumnModel(col, replace(feed, y={"He": 1.0}), sp, N=30, y0={"He": 1.0})
    r = m.solve(30.0, n_out=4)
    C, _, _, Tg, _, _, dP = m._unpack(r.y_end, r.t[-1])
    F = m._gas(C, Tg, dP, r.t[-1])[0].sum(0)
    assert np.allclose(F, feed.flow / m.A, rtol=1e-6)
    assert np.allclose(Tg, feed.T, atol=1e-6)
    g = -np.diff(dP) / m.dz
    assert np.all(g > 0) and abs(g[16] - g[15]) < 1e-6 * g[15] and abs(g[19] - g[20]) < 1e-6 * g[20]


def test_interface_off_a_face_is_rejected(ac_zeo):
    col, feed, sp = ac_zeo
    with pytest.raises(ValueError, match="not on cell faces"):
        ColumnModel(col, feed, sp, N=31, y0={"He": 1.0})
