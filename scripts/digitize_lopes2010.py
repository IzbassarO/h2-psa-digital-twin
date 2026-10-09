"""Extract the compact-unit breakthrough data (Lopes 2010, Figs 5.15-5.17) from the thesis PDF.

The figures are vector graphics (Excel EMF -> PDF), so every experimental symbol is a path object.
Marker centres are taken from the path geometry and mapped to data units with a linear fit to the
axis tick labels. No pixel reading is involved. The author's simulation lines (gPROMS, same model
equations) are extracted as well, into *_authorsim.csv, as a cross-check of our implementation.

Requires PyMuPDF (`pip install pymupdf`, not a package dependency) and
literature/Lopes2010_thesis.pdf.  Output: data/benchmarks/digitized/*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf
except ImportError:  # pragma: no cover
    sys.exit("pip install pymupdf  (only needed to re-digitise)")

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "literature" / "Lopes2010_thesis.pdf"
OUT = ROOT / "data" / "benchmarks" / "digitized"
QUANT = 0.06  # pt; all path coordinates in these figures are multiples of 0.06 pt

COLOURS = {  # fill colour -> (series in flow panels, series in temperature panels)
    (0.01, 0.02, 0.02): ("CO2", None),
    (0.93, 0.13, 0.14): ("H2", "Middle"),
    (0.12, 0.66, 0.31): ("CO", "Top"),
    (0.11, 0.28, 0.47): ("CH4", "Bottom"),
    (0.41, 0.2, 0.58): ("N2", None),
}
# figure, printed page, runs, and panel order (frames sorted top-to-bottom, left-to-right)
FIGURES = [
    ("5.15", 188, [("run1", "flow"), ("run1", "T"), ("run2", "flow"), ("run2", "T"), ("run3", "flow"), ("run3", "T")]),
    ("5.16", 189, [("run4", "flow"), ("run4", "zoom"), ("run4", "T")]),
    ("5.17", 190, [("run5", "flow"), ("run5", "zoom"), ("run5", "T")]),
]
LEGEND_WORDS = {"CO2", "H2", "CO", "CH4", "N2", "Bottom", "Middle", "Top"}


def subpaths(dr):
    """Split a PyMuPDF drawing into subpaths (lists of points)."""
    out, cur, last = [], [], None
    for it in dr["items"]:
        if it[0] == "re":
            if cur:
                out.append(cur)
            r = it[1]
            out.append([r.tl, r.tr, r.br, r.bl])
            cur, last = [], None
            continue
        pts = [p for p in it[1:] if isinstance(p, pymupdf.Point)]
        if last is None or abs(pts[0].x - last.x) > 1e-3 or abs(pts[0].y - last.y) > 1e-3:
            if cur:
                out.append(cur)
            cur = [pts[0]]
        cur += pts[1:]
        last = pts[-1]
    if cur:
        out.append(cur)
    return out


def marker_centres(page):
    """Return {colour: {kind: array of (x, y)}} for every symbol on the page.

    Hollow symbols (CO2 squares, H2 triangles) are one small path each ("outline"); filled symbols
    (temperature series) are a "fill" path plus an "outline" path with the same centre, so each
    panel later counts fills if present and outlines otherwise. The x and + symbols are one path
    per series whose subpaths are <=5-point stroke outlines; consecutive strokes with a common
    centre form one symbol ("stroke"). Simulation lines contain long curved subpaths and are skipped.
    """
    pts: dict[tuple, dict[str, list]] = {}
    for dr in page.get_drawings():
        col = tuple(round(v, 2) for v in (dr.get("fill") or ()))
        if col not in COLOURS:
            continue
        r = dr["rect"]
        if max(r.width, r.height) <= 6.5:
            if max(r.width, r.height) >= 2.5:
                kind = "fill" if len(dr["items"]) <= 4 else "outline"
                pts.setdefault(col, {}).setdefault(kind, []).append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
            continue
        if col == (0.01, 0.02, 0.02):
            continue  # black multi-subpath paths are axis ticks; CO2 squares are separate paths
        sps = subpaths(dr)
        if not all(len(s) <= 5 for s in sps):
            continue  # a simulation line
        cs = []
        for s in sps:
            xs, ys = [p.x for p in s], [p.y for p in s]
            cs.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(max(xs) - min(xs), max(ys) - min(ys))))
        # every x / + symbol is exactly two consecutive strokes sharing a centre
        assert len(cs) % 2 == 0, "odd number of strokes"
        cs = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(cs[::2], cs[1::2])
              if max(a[2], b[2]) <= 6.5 and abs(a[0] - b[0]) < 0.6 and abs(a[1] - b[1]) < 0.6]
        pts.setdefault(col, {}).setdefault("stroke", []).extend(cs)
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in pts.items()}


def line_vertices(page):
    """{colour: (x, y) vertices} of the simulation lines. Lines are stroke outlines (filled polygons,
    ~0.6 pt wide), so each vertex lies within half a line width of the plotted curve."""
    out: dict[tuple, list] = {}
    for dr in page.get_drawings():
        col = tuple(round(v, 2) for v in (dr.get("fill") or ()))
        items = dr["items"]
        if col not in COLOURS or len(items) <= 30 or all(it[0] == "re" for it in items):
            continue  # symbols, frames, axis ticks
        sps = subpaths(dr)
        if all(len(sp) <= 5 for sp in sps):
            continue  # x / + symbol groups
        out.setdefault(col, []).extend((p.x, p.y) for sp in sps for p in sp)
    return {c: np.array(v) for c, v in out.items()}


def frames(page):
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        col = tuple(round(v, 2) for v in (dr.get("fill") or ()))
        if col == (0.01, 0.02, 0.02) and r.width > 100 and r.height > 100 and len(dr["items"]) <= 20:
            out.append(r)
    out.sort(key=lambda r: (round(r.y0 / 50), r.x0))
    return out


def axis_fit(words, fr):
    """Linear pixel->value maps from the numeric tick labels next to frame fr."""
    xl, yl = [], []
    for x0, y0, x1, y1, w, *_ in words:
        try:
            v = float(w)
        except ValueError:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if fr.y1 < y0 < fr.y1 + 20 and fr.x0 - 15 < cx < fr.x1 + 15:
            xl.append((cx, v))
        elif fr.x0 - 35 < x1 < fr.x0 + 1 and fr.y0 - 8 < cy < fr.y1 + 8:
            yl.append((cy, v))
    fits = []
    for lab in (xl, yl):
        lab = np.array(lab)
        b, a = np.polyfit(lab[:, 0], lab[:, 1], 1)
        resid = np.max(np.abs(a + b * lab[:, 0] - lab[:, 1]))
        fits.append((a, b, resid, lab[:, 1].min(), lab[:, 1].max()))
    return fits


def legend_mask(words, fr, xy):
    """True for symbols that sit just left of a legend label (legend entries, not data)."""
    bad = np.zeros(len(xy), bool)
    for x0, y0, x1, y1, w, *_ in words:
        if w in LEGEND_WORDS and fr.contains(pymupdf.Point(x0, y0)):
            cy = (y0 + y1) / 2
            bad |= (np.abs(xy[:, 1] - cy) < 4) & (xy[:, 0] < x0 + 1) & (xy[:, 0] > x0 - 12)
    return bad


def main():
    doc = pymupdf.open(PDF)
    OUT.mkdir(parents=True, exist_ok=True)
    for fig, printed, panels in FIGURES:
        page = doc[printed + 34 - 1]
        words = page.get_text("words")
        marks = marker_centres(page)
        lines = line_vertices(page)
        frs = frames(page)
        assert len(frs) == len(panels), (fig, len(frs))
        for (run, kind), fr in zip(panels, frs):
            (ax, bx, rx, x_lo, x_hi), (ay, by, ry, y_lo, y_hi) = axis_fit(words, fr)
            rows = []
            for col, by_kind in marks.items():
                name = COLOURS[col][1 if kind == "T" else 0]
                if name is None:
                    continue
                sel = {}
                for k, xy in by_kind.items():
                    inside = (xy[:, 0] > fr.x0 - 2) & (xy[:, 0] < fr.x1 + 2) & (xy[:, 1] > fr.y0 - 2) & (xy[:, 1] < fr.y1 + 2)
                    sel[k] = xy[inside & ~legend_mask(words, fr, xy)]
                xy = sel.get("fill") if len(sel.get("fill", ())) else sel.get("outline", np.empty((0, 2)))
                xy = np.vstack([xy, sel.get("stroke", np.empty((0, 2)))])
                for x, y in xy:
                    rows.append((name, ax + bx * x, ay + by * y))
            rows.sort()
            ut = max(abs(bx) * QUANT, rx)
            uy = max(abs(by) * QUANT, ry)
            if kind == "T":
                head, unit = "temperature_K", "K"
                names = "Bottom (z=0.059 m), Middle (z=0.151 m), Top (z=0.239 m)"
            else:
                head, unit = "molar_flow_mmol_s", "1e-3 mol s-1"
                names = "outlet molar flow per species"
            fn = OUT / f"lopes2010_{run}_{kind}.csv"
            with open(fn, "w") as fh:
                fh.write(f"# Lopes (2010) Figure {fig}, printed p.{printed}, PDF p.{printed + 34}; compact unit {run}, panel '{kind}'\n")
                fh.write(f"# {names}; experimental symbols only (simulation lines excluded)\n")
                fh.write("# method: vector extraction of symbol centres from the PDF paths (scripts/digitize_lopes2010.py)\n")
                fh.write(f"# axes: time {x_lo:g}-{x_hi:g} s, {head} {y_lo:g}-{y_hi:g} {unit}\n")
                fh.write(f"# read-off uncertainty: time +-{ut:.1f} s, value +-{uy:.2g} {unit} "
                         f"(coordinate quantisation {QUANT} pt and tick-label fit residual); "
                         f"symbols overlapping in the figure cannot be separated\n")
                fh.write(f"series,time_s,{head}\n")
                for name, t, v in rows:
                    fh.write(f"{name},{t:.1f},{v:.5g}\n")
            counts = {n: sum(r[0] == n for r in rows) for n in sorted({r[0] for r in rows})}
            print(f"{fn.name}: {counts}  u_t={ut:.1f}s u_y={uy:.2g}  (label resid {rx:.2g}, {ry:.2g})")

            sim = []
            for col, xy in lines.items():
                name = COLOURS[col][1 if kind == "T" else 0]
                if name is None:
                    continue
                xy = xy[(xy[:, 0] >= fr.x0) & (xy[:, 0] <= fr.x1) & (xy[:, 1] >= fr.y0) & (xy[:, 1] <= fr.y1)]
                sim += [(name, ax + bx * x, ay + by * y) for x, y in np.unique(xy.round(3), axis=0)]
            sim.sort()
            fn = OUT / f"lopes2010_{run}_{kind}_authorsim.csv"
            with open(fn, "w") as fh:
                fh.write(f"# Lopes (2010) Figure {fig}, printed p.{printed}, PDF p.{printed + 34}; compact unit {run}, panel '{kind}'\n")
                fh.write("# the AUTHOR'S SIMULATION lines (gPROMS), not experimental data\n")
                fh.write("# method: vertices of the stroke-outline polygons of each line (scripts/digitize_lopes2010.py);\n")
                fh.write(f"# every vertex lies within half a line width (0.3 pt = {0.3 * abs(bx):.0f} s or "
                         f"{0.3 * abs(by):.2g} {unit}) of the plotted curve; lines leaving the frame are clipped at it,\n")
                fh.write("# which can leave a few vertices along the frame edge (e.g. H2 in the zoomed panels)\n")
                fh.write(f"series,time_s,{head}\n")
                for name, t, v in sim:
                    fh.write(f"{name},{t:.1f},{v:.5g}\n")
            print(f"  {fn.name}: { {n: sum(r[0] == n for r in sim) for n in sorted({r[0] for r in sim})} }")


if __name__ == "__main__":
    main()
