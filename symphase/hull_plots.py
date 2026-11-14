# -*- coding: utf-8 -*-
"""
symphase.hull_plots
===================

Publication-quality convex-hull figures (matplotlib only).

    plot_binary(h)            formation-energy hull curve (2 elements)
    plot_ternary(h)           Gibbs triangle, shaded hull facets (3 elements)
    plot_ternary_surface(h)   3D hull surface E_form(x, y) (3 elements)
    plot_quaternary(h)        tetrahedron with tie-lines (4 elements)
    plot_ehull_bar(h)         sorted E_hull bar chart (any n)
    plot_pseudo_binary(h,A,B) hull along the line A -> B (any n)
    plot_subsystems(h, 2|3)   grid of every binary / ternary sub-hull (any n)

All functions accept ``ax=`` (where meaningful) and ``save="file.png|pdf|svg"``
and return the matplotlib Figure.
"""

import itertools
import math

import numpy as np
import matplotlib

if matplotlib.get_backend().lower() in ("agg", "") or not matplotlib.get_backend():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.collections import PolyCollection, LineCollection  # noqa: E402

from pymatgen.core import Composition  # noqa: E402

from .hull import pretty_formula  # noqa: E402

SQ3 = math.sqrt(3.0)
STABLE_C = "#16325c"     # deep navy
HULL_C = "#16325c"
FILL_C = "#9ec5e8"
EHULL_CMAP = "magma_r"
FORM_CMAP = "Blues_r"


# ----------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        "mathtext.default": "regular",
    })


set_style()


def _finish(fig, save):
    if save:
        fig.savefig(save)
    return fig


def _rows(h):
    return h.table()


def _vmax(rows, vmax):
    if vmax is not None:
        return vmax
    eh = [1000 * r["e_hull"] for r in rows if not r["stable"]]
    if not eh:
        return 100.0
    return max(25.0, float(np.percentile(eh, 90)))


def _ehull_norm(rows, vmax):
    return colors.Normalize(vmin=0.0, vmax=_vmax(rows, vmax))


def _comp_vec(comp, els):
    c = Composition(comp)
    return np.array([c.get_atomic_fraction(e) for e in els])


def _lowest_per_formula(rows):
    best = {}
    for r in rows:
        f = r["formula"]
        if f not in best or r["e_hull"] < best[f]["e_hull"]:
            best[f] = r
    return best


def _hull_edges(h):
    """unique edges (pairs of stable compositions) of the lower hull facets"""
    pd = h.pd
    edges = set()
    for f in pd.facets:
        for i, j in itertools.combinations(sorted(int(x) for x in f), 2):
            edges.add((i, j))
    ents = pd.qhull_entries
    return [(ents[i], ents[j]) for i, j in edges], ents


def _label(ax, x, y, text, stable=True, dx=0, dy=8, fs=9.5, ha="center", **kw):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va="bottom" if dy >= 0 else "top", fontsize=fs,
                color=STABLE_C if stable else "#5a5a5a",
                fontweight="bold" if stable else "normal",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75), **kw)


def _ehull_colorbar(fig, ax, norm, label="E$_{hull}$ (meV/atom)", **kw):
    sm = cm.ScalarMappable(norm=norm, cmap=EHULL_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045, extend="max", **kw)
    cb.set_label(label)
    return cb


# ============================================================================
#  binary
# ============================================================================
def plot_binary(h, ax=None, save=None, label_ehull=0.05, vmax=None, show_unstable=True,
                colorbar=True, title=True, figsize=(7.2, 4.8), corner_labels=True):
    if h.dim != 2:
        raise ValueError("plot_binary needs a 2-element system")
    A, B = h.elements
    rows = _rows(h)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot(111)
    norm = _ehull_norm(rows, vmax)
    xb = lambda r: Composition(r["composition"]).get_atomic_fraction(B)

    st = sorted([r for r in rows if r["stable"]], key=xb)
    xs, ys = [xb(r) for r in st], [r["e_form"] for r in st]
    ax.axhline(0, color="0.55", lw=0.9, ls="--", zorder=1)
    ax.fill_between(xs, ys, 0, color=FILL_C, alpha=0.35, lw=0, zorder=1)
    ax.plot(xs, ys, "-", color=HULL_C, lw=2.2, zorder=3, label="convex hull")

    if show_unstable:
        un = [r for r in rows if not r["stable"]]
        if un:
            ax.scatter([xb(r) for r in un], [r["e_form"] for r in un],
                       c=[1000 * r["e_hull"] for r in un], cmap=EHULL_CMAP, norm=norm,
                       s=46, marker="D", edgecolor="k", linewidth=0.5, zorder=4,
                       label="above hull")
            stable_f = {r["formula"] for r in rows if r["stable"]}
            for r in _lowest_per_formula(un).values():
                if r["e_hull"] <= label_ehull and r["formula"] not in stable_f:
                    _label(ax, xb(r), r["e_form"], pretty_formula(r["formula"]), stable=False,
                           dy=-11, fs=8.5)
    ax.scatter(xs, ys, s=80, color=STABLE_C, edgecolor="white", linewidth=1.4, zorder=5,
               label="stable")
    for x, y, r in zip(xs, ys, st):
        if 0 < x < 1:
            _label(ax, x, y, pretty_formula(r["formula"]), dy=-14)
    ax.set_xlim(-0.02, 1.02)
    lo = min([r["e_form"] for r in rows] + [0])
    hi = max([r["e_form"] for r in rows] + [0])
    pad = 0.12 * (hi - lo + 1e-3)
    ax.set_ylim(lo - 1.6 * pad, hi + pad)
    ax.set_xlabel(f"x in {A}$_{{1-x}}${B}$_x$  (atomic fraction of {B})")
    ax.set_ylabel("formation energy (eV/atom)")
    if corner_labels:
        ax.text(0.0, 1.01, str(A), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontweight="bold")
        ax.text(1.0, 1.01, str(B), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontweight="bold")
    if title:
        ax.set_title(f"{A}–{B} convex hull", pad=18)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=9.5)
    if colorbar and show_unstable and any(not r["stable"] for r in rows):
        _ehull_colorbar(fig, ax, norm)
    return _finish(fig, save)


# ============================================================================
#  ternary
# ============================================================================
def _tri_xy(v):
    """barycentric (a, b, c) -> Cartesian Gibbs-triangle coordinates"""
    v = np.asarray(v, float)
    return np.array([v[..., 1] + 0.5 * v[..., 2], SQ3 / 2 * v[..., 2]]).T


def plot_ternary(h, ax=None, save=None, label_ehull=0.05, vmax=None, show_unstable=True,
                 shade=True, colorbar=True, title=True, figsize=(7.6, 6.6), grid=True):
    if h.dim != 3:
        raise ValueError("plot_ternary needs a 3-element system")
    els = h.elements
    rows = _rows(h)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    norm = _ehull_norm(rows, vmax)
    pd = h.pd
    ents = pd.qhull_entries

    # shaded hull facets coloured by mean formation energy
    if shade:
        polys, vals = [], []
        for f in pd.facets:
            pts = [_tri_xy(_comp_vec(ents[int(i)].composition, els)) for i in f]
            polys.append(pts)
            vals.append(np.mean([pd.get_form_energy_per_atom(ents[int(i)]) for i in f]))
        if polys:
            fn = colors.Normalize(vmin=min(vals), vmax=max(0.0, max(vals)))
            pc = PolyCollection(polys, array=np.array(vals), cmap=FORM_CMAP, norm=fn,
                                alpha=0.55, edgecolor="none", zorder=0)
            ax.add_collection(pc)
            if colorbar:
                cb = fig.colorbar(pc, ax=ax, location="left", pad=0.0, fraction=0.04, shrink=0.7)
                cb.set_label("hull formation energy (eV/atom)")
    # triangle frame + grid
    corners = np.array([[0, 0], [1, 0], [0.5, SQ3 / 2], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], color="k", lw=1.6, zorder=2)
    if grid:
        for t in np.arange(0.2, 1.0, 0.2):
            for a, b in (((t, 1 - t, 0), (t, 0, 1 - t)), ((1 - t, t, 0), (0, t, 1 - t)),
                         ((1 - t, 0, t), (0, 1 - t, t))):
                p = _tri_xy([a, b])
                ax.plot(p[:, 0], p[:, 1], color="0.85", lw=0.6, zorder=1)
    # tie-lines
    edges, _ = _hull_edges(h)
    segs = [[_tri_xy(_comp_vec(a.composition, els)), _tri_xy(_comp_vec(b.composition, els))]
            for a, b in edges]
    ax.add_collection(LineCollection(segs, colors=HULL_C, linewidths=1.3, zorder=3))
    # unstable
    if show_unstable:
        un = [r for r in rows if not r["stable"]]
        if un:
            P = np.array([_tri_xy(_comp_vec(r["composition"], els)) for r in un])
            ax.scatter(P[:, 0], P[:, 1], c=[1000 * r["e_hull"] for r in un], cmap=EHULL_CMAP,
                       norm=norm, s=40, marker="D", edgecolor="k", linewidth=0.5, zorder=4)
            best = _lowest_per_formula(un)
            stable_f = {r["formula"] for r in rows if r["stable"]}
            for r in best.values():
                if r["e_hull"] <= label_ehull and r["formula"] not in stable_f:
                    x, y = _tri_xy(_comp_vec(r["composition"], els))
                    _label(ax, x, y, pretty_formula(r["formula"]), stable=False, dy=-10, fs=8)
            if colorbar:
                _ehull_colorbar(fig, ax, norm, shrink=0.7)
    # stable
    st = [r for r in rows if r["stable"]]
    P = np.array([_tri_xy(_comp_vec(r["composition"], els)) for r in st])
    ax.scatter(P[:, 0], P[:, 1], s=75, color=STABLE_C, edgecolor="white", linewidth=1.4, zorder=5)
    for (x, y), r in zip(P, st):
        if Composition(r["composition"]).is_element:
            continue
        _label(ax, x, y, pretty_formula(r["formula"]), dy=7)
    off = 0.045
    ax.text(-off, -off * 0.6, str(els[0]), ha="right", va="top", fontsize=14, fontweight="bold")
    ax.text(1 + off, -off * 0.6, str(els[1]), ha="left", va="top", fontsize=14, fontweight="bold")
    ax.text(0.5, SQ3 / 2 + off * 0.6, str(els[2]), ha="center", va="bottom", fontsize=14,
            fontweight="bold")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, SQ3 / 2 + 0.1)
    if title:
        ax.set_title(f"{'–'.join(map(str, els))} convex hull", pad=14)
    return _finish(fig, save)


def plot_ternary_surface(h, save=None, vmax=None, elev=22, azim=-58, figsize=(8.5, 7)):
    """3D view: hull facets as a surface E_form(x,y); unstable points float above it"""
    if h.dim != 3:
        raise ValueError("needs a 3-element system")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    els = h.elements
    rows = _rows(h)
    pd = h.pd
    ents = pd.qhull_entries
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    polys, vals = [], []
    for f in pd.facets:
        pts = []
        for i in f:
            e = ents[int(i)]
            x, y = _tri_xy(_comp_vec(e.composition, els))
            pts.append((x, y, pd.get_form_energy_per_atom(e)))
        polys.append(pts)
        vals.append(np.mean([p[2] for p in pts]))
    fn = colors.Normalize(vmin=min(vals), vmax=max(0.0, max(vals)))
    pc = Poly3DCollection(polys, facecolors=matplotlib.colormaps[FORM_CMAP](fn(vals)), edgecolor=HULL_C,
                          linewidths=0.8, alpha=0.75)
    ax.add_collection3d(pc)
    norm = _ehull_norm(rows, vmax)
    un = [r for r in rows if not r["stable"]]
    for r in un:
        x, y = _tri_xy(_comp_vec(r["composition"], els))
        z = r["e_form"]
        ax.plot([x, x], [y, y], [z - r["e_hull"], z], color="0.5", lw=0.7)
    if un:
        P = np.array([[*_tri_xy(_comp_vec(r["composition"], els)), r["e_form"]] for r in un])
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=[1000 * r["e_hull"] for r in un], cmap=EHULL_CMAP,
                   norm=norm, s=28, marker="D", edgecolor="k", linewidth=0.4, depthshade=False)
    st = [r for r in rows if r["stable"]]
    P = np.array([[*_tri_xy(_comp_vec(r["composition"], els)), r["e_form"]] for r in st])
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=55, color=STABLE_C, edgecolor="white", depthshade=False)
    for p, r in zip(P, st):
        if len(Composition(r["composition"]).elements) >= 3 or r["e_form"] == min(q["e_form"] for q in st):
            ax.text(p[0], p[1], p[2] - 0.04, pretty_formula(r["formula"]), fontsize=9,
                    color=STABLE_C, ha="center", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    for i, e in enumerate(els):
        c = [(0, 0), (1, 0), (0.5, SQ3 / 2)][i]
        ax.text(c[0], c[1], 0.05, str(e), fontsize=13, fontweight="bold", ha="center")
    tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, SQ3 / 2, 0], [0, 0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color="k", lw=1.2)
    ax.set_zlabel("E$_{form}$ (eV/atom)", labelpad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, SQ3 / 2, 0.75))
    ax.set_title(f"{'–'.join(map(str, els))} hull surface")
    sm = cm.ScalarMappable(norm=norm, cmap=EHULL_CMAP)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.13, extend="max").set_label("E$_{hull}$ (meV/atom)")
    return _finish(fig, save)


# ============================================================================
#  quaternary
# ============================================================================
_TET = np.array([[0, 0, 0], [1, 0, 0], [0.5, SQ3 / 2, 0], [0.5, SQ3 / 6, math.sqrt(2.0 / 3.0)]])


def _tet_xyz(v):
    return np.asarray(v, float) @ _TET


def plot_quaternary(h, save=None, label_ehull=0.0, vmax=None, show_unstable=True,
                    elev=14, azim=-28, figsize=(9, 8), label_binaries=False):
    if h.dim != 4:
        raise ValueError("plot_quaternary needs a 4-element system")
    els = h.elements
    rows = _rows(h)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for i, j in itertools.combinations(range(4), 2):
        p = _TET[[i, j]]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="k", lw=1.2)
    for i, e in enumerate(els):
        c = _TET[i] + (_TET[i] - _TET.mean(0)) * 0.12
        ax.text(*c, str(e), fontsize=14, fontweight="bold", ha="center", va="center")
    edges, _ = _hull_edges(h)
    for a, b in edges:
        p = np.array([_tet_xyz(_comp_vec(a.composition, els)), _tet_xyz(_comp_vec(b.composition, els))])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=HULL_C, lw=0.7, alpha=0.45)
    norm = _ehull_norm(rows, vmax)
    if show_unstable:
        un = [r for r in rows if not r["stable"]]
        if un:
            P = np.array([_tet_xyz(_comp_vec(r["composition"], els)) for r in un])
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=[1000 * r["e_hull"] for r in un],
                       cmap=EHULL_CMAP, norm=norm, s=30, marker="D", edgecolor="k",
                       linewidth=0.4, depthshade=False)
            for r in _lowest_per_formula(un).values():
                if r["e_hull"] <= label_ehull:
                    p = _tet_xyz(_comp_vec(r["composition"], els))
                    ax.text(*p, " " + pretty_formula(r["formula"]), fontsize=7.5, color="0.35")
            sm = cm.ScalarMappable(norm=norm, cmap=EHULL_CMAP)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.04, extend="max").set_label(
                "E$_{hull}$ (meV/atom)")
    st = [r for r in rows if r["stable"]]
    P = np.array([_tet_xyz(_comp_vec(r["composition"], els)) for r in st])
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, color=STABLE_C, edgecolor="white",
               depthshade=False, zorder=5)
    for p, r in zip(P, st):
        nel = len(Composition(r["composition"]).elements)
        if nel >= 3 or (nel == 2 and label_binaries):
            ax.text(p[0], p[1], p[2] + 0.03, pretty_formula(r["formula"]), fontsize=8.5,
                    color=STABLE_C, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.text2D(0.02, 0.02, "labels: stable ternary/quaternary phases (binaries: see sub-system plots)",
              transform=ax.transAxes, fontsize=8, color="0.45")
    ax.set_axis_off()
    ax.set_box_aspect((1, SQ3 / 2, math.sqrt(2 / 3)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{'–'.join(map(str, els))} convex hull")
    return _finish(fig, save)


# ============================================================================
#  any number of elements
# ============================================================================
def plot_ehull_bar(h, save=None, max_rows=40, vmax=None, figsize=None):
    """horizontal bars of E_hull, lowest first (stable phases marked)"""
    allrows = [r for r in _rows(h) if not Composition(r["composition"]).is_element]
    nstable = sum(r["stable"] for r in allrows)
    hide_stable = nstable > max_rows // 2
    rows = [r for r in allrows if not (hide_stable and r["stable"])][:max_rows]
    if not rows:
        rows = allrows[:max_rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=figsize or (7.5, 0.28 * n + 1.6))
    norm = _ehull_norm(rows, vmax)
    cmap = matplotlib.colormaps[EHULL_CMAP]
    y = np.arange(n)[::-1]
    vals = [1000 * r["e_hull"] for r in rows]
    cols = [STABLE_C if r["stable"] else cmap(norm(v)) for r, v in zip(rows, vals)]
    ax.barh(y, vals, color=cols, edgecolor="k", linewidth=0.4, height=0.72)
    xmax = max(max(vals + [0]) * 1.12, 10.0)
    ax.set_xlim(0, xmax)
    for yi, r, v in zip(y, rows, vals):
        if r["stable"]:
            ax.plot(0, yi, "o", color=STABLE_C, ms=7, mec="white", zorder=5, clip_on=False)
            ax.text(xmax * 0.02, yi, "stable", va="center", fontsize=8.5,
                    color=STABLE_C, fontweight="bold")
        else:
            ax.text(v + xmax * 0.01, yi, f"{v:.1f}", va="center", fontsize=7.5, color="0.35")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pretty_formula(r['formula'])}  ({r['name']})" if r["name"] != r["formula"]
                        else pretty_formula(r["formula"]) for r in rows], fontsize=9)
    ax.tick_params(axis="y", which="both", right=False, length=0)
    ax.minorticks_off()
    ax.set_xlabel("E$_{hull}$ (meV/atom)")
    ax.axvline(0, color="k", lw=1)
    for t, lab in ((25, "kT(300 K)"), (100, "")):
        if t < max(vals + [0]) * 1.1:
            ax.axvline(t, color="0.6", lw=0.8, ls=":")
            if lab:
                ax.text(t, n - 0.3, " " + lab, fontsize=8, color="0.4", va="bottom")
    ttl = f"{h.chemsys}: energy above hull"
    if hide_stable:
        ttl += f"\n({nstable} stable compounds not shown)"
    ax.set_title(ttl, pad=10)
    return _finish(fig, save)


def plot_pseudo_binary(h, comp_a, comp_b, ax=None, save=None, n=301, tol=0.02, vmax=None,
                       figsize=(7.2, 4.6)):
    """hull along A -> B for a system with any number of elements"""
    d = h.pseudo_binary(comp_a, comp_b, n=n, tol=tol)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot(111)
    ax.axhline(0, color="0.55", lw=0.9, ls="--")
    ax.fill_between(d["x"], d["e_hull_line"], 0, color=FILL_C, alpha=0.35, lw=0)
    ax.plot(d["x"], d["e_hull_line"], color=HULL_C, lw=2.2, label="hull along the line")
    if d["points"]:
        pts = np.array([(x, e, 1000 * eh) for x, e, eh, _ in d["points"]])
        norm = colors.Normalize(0, vmax or max(25.0, pts[:, 2].max()))
        on = pts[:, 2] < 0.5
        ax.scatter(pts[~on, 0], pts[~on, 1], c=pts[~on, 2], cmap=EHULL_CMAP, norm=norm, s=46,
                   marker="D", edgecolor="k", linewidth=0.5, zorder=4)
        ax.scatter(pts[on, 0], pts[on, 1], s=80, color=STABLE_C, edgecolor="white", lw=1.4, zorder=5)
        for (x, e, eh, name) in d["points"]:
            st_ = eh < 5e-4
            _label(ax, x, e, pretty_formula(Composition(name).reduced_formula)
                   if _is_formula(name) else name, stable=st_, dy=-14 if st_ else 9, fs=8.5,
                   ha="left" if x < 0.12 else ("right" if x > 0.88 else "center"),
                   dx=4 if x < 0.12 else (-4 if x > 0.88 else 0))
        if (~on).any():
            _ehull_colorbar(fig, ax, norm)
    ax.set_xlim(-0.03, 1.03)
    ys = list(d["e_hull_line"]) + [p[1] for p in d["points"]]
    lo, hi = min(ys), max(ys)
    pad = 0.15 * max(hi - lo, 0.02)
    ax.set_ylim(lo - 1.6 * pad, hi + 1.8 * pad)
    ax.set_xlabel(f"x in ({pretty_formula(d['a'])})$_{{1-x}}$({pretty_formula(d['b'])})$_x$ "
                  "(per atom)")
    ax.set_ylabel("formation energy (eV/atom)")
    ax.set_title(f"pseudo-binary {pretty_formula(d['a'])} – {pretty_formula(d['b'])}", pad=10)
    ax.text(0.01, 0.97, "kinks = phases on the hull; straight segments = two-phase regions",
            transform=ax.transAxes, fontsize=8, color="0.45", va="top")
    return _finish(fig, save)


def _is_formula(s):
    try:
        Composition(s)
        return True
    except Exception:  # noqa: BLE001
        return False


def plot_subsystems(h, order=2, save=None, ncols=None, vmax=None, **kw):
    """grid of all binary (order=2) or ternary (order=3) sub-system hulls"""
    from .hull import HullAnalysis
    combos = list(itertools.combinations(h.elements, order))
    nc = ncols or (min(4, len(combos)) if order == 2 else min(3, len(combos)))
    nr = math.ceil(len(combos) / nc)
    w, hh = (4.0, 3.2) if order == 2 else (4.2, 3.9)
    fig, axs = plt.subplots(nr, nc, figsize=(w * nc, hh * nr + 0.6), squeeze=False)
    rows_all = _rows(h)
    vm = _vmax(rows_all, vmax)
    for ax, combo in zip(axs.ravel(), combos):
        try:
            sub = HullAnalysis(h.entries, elements=[str(e) for e in combo], nproc=1, backend="serial")
        except ValueError as err:
            ax.text(0.5, 0.5, str(err), ha="center", va="center", fontsize=8, wrap=True)
            ax.axis("off")
            continue
        if order == 2:
            plot_binary(sub, ax=ax, colorbar=False, title=False, vmax=vm, label_ehull=0,
                        corner_labels=False)
            a_, b_ = (str(e) for e in sub.elements)
            ax.set_title(f"{a_}$_{{1-x}}${b_}$_x$", fontsize=11)
            ax.set_xlabel(f"x ({b_})")
            ax.get_legend().remove()
            ax.set_ylabel("E$_{form}$ (eV/at)")
        else:
            plot_ternary(sub, ax=ax, colorbar=False, title=False, vmax=vm, label_ehull=0, grid=False)
            ax.set_title("–".join(map(str, combo)), fontsize=11)
    for ax in axs.ravel()[len(combos):]:
        ax.axis("off")
    norm = colors.Normalize(0, vm)
    sm = cm.ScalarMappable(norm=norm, cmap=EHULL_CMAP)
    sm.set_array([])
    fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.6, pad=0.02, extend="max").set_label(
        "E$_{hull}$ (meV/atom)")
    fig.suptitle(f"{h.chemsys}: all {'binary' if order == 2 else 'ternary'} sub-system hulls",
                 fontweight="bold", fontsize=14)
    return _finish(fig, save)
