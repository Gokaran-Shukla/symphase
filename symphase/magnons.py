# -*- coding: utf-8 -*-
"""
symphase.magnons — magnon chirality splitting (collinear magnets)
=================================================================

Linear spin-wave theory (LSWT) for collinear magnets with the Heisenberg model

    H = sum_<ij> J_ij  S_i . S_j  -  K sum_i (S_i . n)^2           (J > 0: antiferromagnetic)

* exchange bonds are grouped into **symmetry classes** of the non-magnetic crystal
  space group (bonds related by any crystal operation share one J).  In an
  altermagnet a class can contain e.g. the [110] bond of sublattice A together with
  the [1-10] bond of sublattice B -> this is what splits magnon chiralities.
* Holstein–Primakoff bosons in local frames, Bogoliubov–Colpa diagonalisation
  (J. H. P. Colpa, Physica A 93, 327 (1978)).
* every mode n carries ΔS_z(n) = -Σ_i σ_i (|T_{i,n}|² + |T_{i+N,n}|²) = ±1
  (σ_i = ±1 sublattice sign): the magnon **chirality** (α: ΔS_z = −1, β: +1).
* chirality splitting Δω(q) = ω_β(q) − ω_α(q); its symmetry is that of the
  non-relativistic spin splitting s(k) of the spin group (d-, g-, i-wave in
  altermagnets; zero in PT- or t-connected antiferromagnets), which is checked.

Command line:  symphase magnons --list-bonds          (show bond classes)
               symphase magnons --J 3.0 1.0 0.5 0.2  (one J per class, meV)
               symphase magnons --exchange J.dat      (i j n1 n2 n3 J  per line)
"""

import math
import os
import sys

import numpy as np

from .parallel import pmap, get_shared

KB_MEV = 0.08617333262   # meV/K


# ----------------------------------------------------------------------------
#  bonds and symmetry classes
# ----------------------------------------------------------------------------
def _image_atom(frac, types, x, typ, tol=2e-3):
    d = frac - x
    sh = np.round(d)
    d = d - sh
    ok = np.where((np.linalg.norm(d, axis=1) < tol) & (np.asarray(types) == typ))[0]
    if len(ok) != 1:
        return None, None
    j = ok[0]
    return j, (x - frac[j]).round().astype(int)


def enumerate_bonds(lat, frac, types, magnetic, rmax):
    """unique bonds (i, j, R) between magnetic sites with |d| <= rmax"""
    A = lat.T
    n = [int(math.ceil(rmax / np.linalg.norm(lat[k]))) + 1 for k in range(3)]
    bonds = []
    for i in magnetic:
        for j in magnetic:
            for R in np.ndindex(2 * n[0] + 1, 2 * n[1] + 1, 2 * n[2] + 1):
                R = np.array(R) - n
                d = A @ (frac[j] + R - frac[i])
                L = np.linalg.norm(d)
                if L < 1e-3 or L > rmax + 1e-6:
                    continue
                if i > j or (i == j and tuple(-R) > tuple(R)):
                    continue
                bonds.append(dict(i=i, j=j, R=R, d=d, L=L))
    return bonds


def bond_classes(lat, frac, types, bonds, rotations, translations):
    """group bonds into orbits of the crystal space group (union-find)"""
    index = {}
    for b_i, b in enumerate(bonds):
        index[(b["i"], b["j"], tuple(b["R"]))] = b_i
        index[(b["j"], b["i"], tuple(-b["R"]))] = b_i
    parent = list(range(len(bonds)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for Rm, t in zip(rotations, translations):
        for b_i, b in enumerate(bonds):
            xi = Rm @ frac[b["i"]] + t
            xj = Rm @ (frac[b["j"]] + b["R"]) + t
            gi, Li = _image_atom(frac, types, xi, types[b["i"]])
            gj, Lj = _image_atom(frac, types, xj, types[b["j"]])
            if gi is None or gj is None:
                continue
            k = index.get((gi, gj, tuple(Lj - Li)))
            if k is not None:
                ra, rb = find(b_i), find(k)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for b_i in range(len(bonds)):
        groups.setdefault(find(b_i), []).append(b_i)
    classes = sorted(groups.values(), key=lambda g: (round(bonds[g[0]]["L"], 4), g[0]))
    for c, g in enumerate(classes):
        for b_i in g:
            bonds[b_i]["cls"] = c
    return classes


# ----------------------------------------------------------------------------
#  LSWT
# ----------------------------------------------------------------------------
def _pairs_from(bonds, Jcls=None, Jexplicit=None):
    pairs = []
    for b in bonds:
        J = Jcls[b["cls"]] if Jcls is not None else None
        if Jexplicit is not None:
            J = Jexplicit.get((b["i"], b["j"], tuple(b["R"])), Jexplicit.get((b["j"], b["i"], tuple(-b["R"]))))
        if J is None or J == 0:
            continue
        pairs.append((b["i"], b["j"], b["d"], float(J)))
    return pairs


def hamiltonian(q, sites, sig, S, pairs, K=0.0):
    """bosonic BdG matrix H(q) (2N x 2N) in meV;  q in 1/Å (Cartesian)"""
    N = len(sites)
    loc = {s: n for n, s in enumerate(sites)}
    A = np.zeros((N, N), complex)
    B = np.zeros((N, N), complex)
    Am = np.zeros((N, N), complex)   # A(-q)
    for i, j, d, J in pairs:
        a, b = loc[i], loc[j]
        sij = sig[a] * sig[b]
        c = J * math.sqrt(S[a] * S[b])
        A[a, a] += -sij * J * S[b]
        A[b, b] += -sij * J * S[a]
        Am[a, a] += -sij * J * S[b]
        Am[b, b] += -sij * J * S[a]
        ph, phm = np.exp(1j * q @ d), np.exp(-1j * q @ d)
        if sij > 0:
            A[a, b] += c * ph
            A[b, a] += c * np.conj(ph)
            Am[a, b] += c * phm
            Am[b, a] += c * np.conj(phm)
        else:
            B[a, b] += c * ph
            B[b, a] += c * np.conj(ph)
    for a in range(N):
        A[a, a] += 2 * K * S[a]
        Am[a, a] += 2 * K * S[a]
    return np.block([[A, B], [B.conj().T, Am.T]])


def colpa(H, eps=1e-9):
    """positive eigen-energies and the first N columns of the paraunitary T (Ψ = T Φ)"""
    n2 = H.shape[0]
    N = n2 // 2
    Hs = 0.5 * (H + H.conj().T) + eps * np.eye(n2)
    try:
        Kc = np.linalg.cholesky(Hs).conj().T          # H = K^† K
    except np.linalg.LinAlgError:
        return None, None
    s3 = np.diag(np.r_[np.ones(N), -np.ones(N)])
    W = Kc @ s3 @ Kc.conj().T
    w, U = np.linalg.eigh(0.5 * (W + W.conj().T))
    order = np.argsort(-w)                            # positive first
    w, U = w[order], U[:, order]
    om = w[:N]
    T = np.linalg.solve(Kc, U[:, :N] * np.sqrt(np.abs(om))[None, :])
    return om, T


def _w_modes(q):
    sites, sig, S, pairs, K = (get_shared(k) for k in ("sites", "sig", "S", "pairs", "K"))
    om, T = colpa(hamiltonian(q, sites, sig, S, pairs, K))
    N = len(sites)
    if om is None:
        return np.full(N, np.nan), np.zeros(N)
    o = np.argsort(om)
    om, T = om[o], T[:, o]
    # S_z change operator in the magnon basis; diagonalise inside degenerate subspaces
    Dm = -(T[:N].conj().T * sig[None, :]) @ T[:N] - (T[N:].conj().T * sig[None, :]) @ T[N:]
    dSz = np.real(np.diag(Dm)).copy()
    tol = 1e-6 * max(1.0, np.abs(om).max())
    k = 0
    while k < N:
        m = k + 1
        while m < N and abs(om[m] - om[k]) < tol:
            m += 1
        if m - k > 1:
            dSz[k:m] = np.sort(np.linalg.eigvalsh(0.5 * (Dm[k:m, k:m] + Dm[k:m, k:m].conj().T)))
        k = m
    return om, dSz


def spectrum(qs, sites, sig, S, pairs, K=0.0, nproc=None):
    res = pmap(_w_modes, list(qs), nproc=nproc, backend="process",
               shared=dict(sites=sites, sig=np.asarray(sig, float), S=np.asarray(S, float),
                           pairs=pairs, K=K), min_items=64)
    om = np.array([r[0] for r in res])
    dS = np.array([r[1] for r in res])
    return om, dS


def chirality_split(om, dS):
    """Δω = (lowest β mode) − (lowest α mode) per q; nan if a chirality is missing"""
    out = np.full(len(om), np.nan)
    for k in range(len(om)):
        a = om[k][dS[k] < -0.5]
        b = om[k][dS[k] > 0.5]
        if len(a) and len(b):
            out[k] = b.min() - a.min()
    return out


# ----------------------------------------------------------------------------
#  paths and maps
# ----------------------------------------------------------------------------
DEFAULT_PATH = [("Γ", (0, 0, 0)), ("(½,0,0)", (0.5, 0, 0)), ("(½,½,0)", (0.5, 0.5, 0)), ("Γ", (0, 0, 0)),
                ("(½,−½,0)", (0.5, -0.5, 0)), ("(0,½,0)", (0, 0.5, 0)), ("Γ", (0, 0, 0)), ("(0,0,½)", (0, 0, 0.5))]


def make_path(lat, points, n_per=60):
    Bm = 2 * np.pi * np.linalg.inv(lat).T            # rows b1,b2,b3
    qs, xs, ticks = [], [], []
    x = 0.0
    for (l0, f0), (l1, f1) in zip(points[:-1], points[1:]):
        q0, q1 = np.array(f0) @ Bm, np.array(f1) @ Bm
        seg = np.linalg.norm(q1 - q0)
        if not ticks:
            ticks.append((x, l0))
        for t in np.linspace(0, 1, n_per, endpoint=False):
            qs.append(q0 + t * (q1 - q0))
            xs.append(x + t * seg)
        x += seg
        ticks.append((x, l1))
    qs.append(np.array(points[-1][1]) @ Bm)
    xs.append(x)
    return np.array(qs), np.array(xs), ticks


# ----------------------------------------------------------------------------
#  analysis driver
# ----------------------------------------------------------------------------
def analyse(lat, frac, types, symbols, mags, rotations, translations, J=None, exchange=None,
            K=0.0, spin=None, rmax=6.5, mmin=0.05, nproc=None, n_per=60, nmap=41, sg=None):
    norms = np.linalg.norm(mags, axis=1)
    magnetic = [i for i in range(len(frac)) if norms[i] > mmin * max(norms.max(), 1e-9)]
    if not magnetic:
        raise ValueError("no magnetic atoms")
    nz = mags[magnetic]
    u, s, vt = np.linalg.svd(nz, full_matrices=False)
    if len(s) > 1 and s[1] / s[0] > 1e-3:
        raise ValueError("magnon module supports collinear magnets only")
    n = vt[0]
    for x in n:                      # canonical axis direction (first non-zero component > 0)
        if abs(x) > 1e-8:
            n = n if x > 0 else -n
            break
    sig = np.sign(nz @ n)
    S = np.array([spin] * len(magnetic)) if spin else np.abs(nz @ n) / 2.0   # S = |m|/2 (g = 2)
    bonds = enumerate_bonds(lat, frac, types, magnetic, rmax)
    classes = bond_classes(lat, frac, types, bonds, rotations, translations)
    loc = {s_: k for k, s_ in enumerate(magnetic)}
    cls_info = []
    for c, g in enumerate(classes):
        b = bonds[g[0]]
        kinds = {("same" if sig[loc[bonds[x]["i"]]] == sig[loc[bonds[x]["j"]]] else "opposite") for x in g}
        cls_info.append(dict(cls=c, L=b["L"], n_bonds=len(g), spins="/".join(sorted(kinds)),
                             rep=(symbols[b["i"]], b["i"] + 1, symbols[b["j"]], b["j"] + 1, b["R"].tolist(),
                                  b["d"].tolist()),
                             members=[(bonds[x]["i"] + 1, bonds[x]["j"] + 1, bonds[x]["R"].tolist()) for x in g]))
    out = dict(axis=n.tolist(), sites=[m + 1 for m in magnetic], sigma=sig.tolist(), S=S.tolist(),
               classes=cls_info, bonds=bonds)
    if J is None and exchange is None:
        return out
    Jexp = None
    if exchange is not None:
        Jexp = {}
        for (i, j, R, val) in exchange:
            Jexp[(i - 1, j - 1, tuple(R))] = val
        missing = [k for k in Jexp if (k[0], k[1], k[2]) not in {(b["i"], b["j"], tuple(b["R"])) for b in bonds}
                   and (k[1], k[0], tuple(-np.array(k[2]))) not in {(b["i"], b["j"], tuple(b["R"])) for b in bonds}]
        if missing:
            raise ValueError(f"{len(missing)} exchange bond(s) outside rmax or not between magnetic atoms")
    Jcls = None
    if J is not None:
        Jcls = list(J) + [0.0] * (len(classes) - len(J))
    pairs = _pairs_from(bonds, Jcls, Jexp)
    qs, xs, ticks = make_path(lat, DEFAULT_PATH, n_per)
    om, dS = spectrum(qs, magnetic, sig, S, pairs, K, nproc)
    dw = chirality_split(om, dS)
    # Δω map in the (b1, b2) plane
    Bm = 2 * np.pi * np.linalg.inv(lat).T
    f = np.linspace(-0.5, 0.5, nmap)
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    Q = (F1.ravel()[:, None] * Bm[0][None, :] + F2.ravel()[:, None] * Bm[1][None, :])
    omm, dSm = spectrum(Q, magnetic, sig, S, pairs, K, nproc)
    dwm = chirality_split(omm, dSm).reshape(F1.shape)
    out.update(J_classes=Jcls, n_pairs=len(pairs), q=qs, x=xs, ticks=ticks, omega=om, dSz=dS, split=dw,
               map_q=Q.reshape(nmap, nmap, 3), map_split=dwm, unstable=bool(np.isnan(om).any()))
    # symmetry check against the spin-group spin-splitting form
    if sg is not None:
        from .spingroup import spin_polarisation_form
        deg, forms = spin_polarisation_form(sg)
        out["sym_degree"], out["sym_form"] = deg, forms
        ok = np.isfinite(dwm)
        amp = np.nanmax(np.abs(dwm)) if ok.any() else 0.0
        out["max_split"] = float(amp)
    return out


# ----------------------------------------------------------------------------
#  plots
# ----------------------------------------------------------------------------
def plot_bands(res, save=None, title=None, figsize=(7.4, 6.2)):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from . import hull_plots  # noqa: F401  (style)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.06))
    om, dS, x = res["omega"], res["dSz"], res["x"]
    cmap = matplotlib.colormaps["coolwarm"]
    for n in range(om.shape[1]):
        pts = np.array([x, om[:, n]]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=matplotlib.colors.Normalize(-1, 1), linewidths=2.2)
        lc.set_array(0.5 * (dS[:-1, n] + dS[1:, n]))
        ax.add_collection(lc)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, np.nanmax(om) * 1.08)
    ax.set_ylabel("magnon energy (meV)")
    for t, lab in res["ticks"]:
        for a in (ax, ax2):
            a.axvline(t, color="0.75", lw=0.7, zorder=0)
    ax.plot([], [], color=cmap(0.0), lw=2.2, label="α  (ΔS$_z$ = −1)")
    ax.plot([], [], color=cmap(1.0), lw=2.2, label="β  (ΔS$_z$ = +1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title or "magnon bands coloured by chirality")
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.fill_between(x, res["split"], 0, color="#c0392b", alpha=0.35, lw=0)
    ax2.plot(x, res["split"], color="#c0392b", lw=1.8)
    ax2.set_ylabel("Δω = ω$_β$ − ω$_α$\n(meV)")
    ax2.set_xticks([t for t, _ in res["ticks"]])
    ax2.set_xticklabels([lab for _, lab in res["ticks"]])
    ax2.minorticks_off()
    if save:
        fig.savefig(save)
    return fig


def plot_split_map(res, save=None, title=None, figsize=(5.6, 4.8)):
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    Q = res["map_q"]
    Z = res["map_split"]
    fig, ax = plt.subplots(figsize=figsize)
    v = np.nanmax(np.abs(Z)) or 1.0
    pc = ax.pcolormesh(Q[..., 0], Q[..., 1], Z, cmap="RdBu_r", vmin=-v, vmax=v, shading="gouraud")
    ax.contour(Q[..., 0], Q[..., 1], Z, levels=[0], colors="k", linewidths=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("q$_x$ (1/Å)")
    ax.set_ylabel("q$_y$ (1/Å)")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("Δω = ω$_β$ − ω$_α$ (meV)")
    ax.set_title(title or "chirality splitting, q$_3$ = 0", fontsize=12)
    if save:
        fig.savefig(save)
    return fig


# ----------------------------------------------------------------------------
#  command line
# ----------------------------------------------------------------------------
def read_exchange_file(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            t = line.split()
            rows.append((int(t[0]), int(t[1]), [int(t[2]), int(t[3]), int(t[4])], float(t[5])))
    return rows


def main(argv=None):
    import argparse
    import spglib
    from .tensors import (read_poscar, read_outcar, read_incar, merge_calc, build_moments, fmt_vec,
                          _g, __version__)
    from . import spingroup as sgm
    ap = argparse.ArgumentParser(
        prog="symphase magnons",
        description="Magnon chirality splitting of collinear magnets (LSWT, symmetry-resolved exchange).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  symphase magnons --list-bonds --rmax 6.5\n"
               "  symphase magnons --J 3.0 2.0 0.5 0.4 0.1 --plot magnons.png\n"
               "  symphase magnons --exchange J.dat --K 0.05\n"
               "J > 0 is antiferromagnetic (H = Σ J S_i·S_j); energies in meV; S = |m|/2 unless --spin.")
    ap.add_argument("-p", "--poscar", default="POSCAR")
    ap.add_argument("-o", "--outcar", default="OUTCAR")
    ap.add_argument("-i", "--incar", default="INCAR")
    ap.add_argument("--magmom", default=None)
    ap.add_argument("--mdir", type=float, nargs=3, default=None)
    ap.add_argument("--mmin", type=float, default=0.05)
    ap.add_argument("--symprec", type=float, default=1e-3)
    ap.add_argument("--mag-symprec", type=float, default=0.05)
    ap.add_argument("--rmax", type=float, default=6.5, help="maximum bond length (Å) [6.5]")
    ap.add_argument("--list-bonds", action="store_true", help="only list the symmetry classes of bonds")
    ap.add_argument("--J", type=float, nargs="+", default=None, help="J per bond class (meV), in listed order")
    ap.add_argument("--exchange", default=None, help="file with lines: i j n1 n2 n3 J(meV) (1-based atoms)")
    ap.add_argument("--K", type=float, default=0.0, help="easy-axis single-ion anisotropy (meV)")
    ap.add_argument("--spin", type=float, default=None, help="spin length S (default |m|/2)")
    ap.add_argument("--nq", type=int, default=60, help="q points per path segment [60]")
    ap.add_argument("--nmap", type=int, default=41, help="grid size of the Δω map [41]")
    ap.add_argument("--plot", default="magnons.png")
    ap.add_argument("-n", "--nproc", type=int, default=-1)
    args = ap.parse_args(argv)

    pos = read_poscar(args.poscar)
    outcar = read_outcar(args.outcar) if os.path.isfile(args.outcar) else None
    incar = read_incar(args.incar) if os.path.isfile(args.incar) else None
    calc = merge_calc(outcar, incar)
    species = pos["species"] or (outcar or {}).get("species")
    symbols = [s for s, c in zip(species, pos["counts"]) for _ in range(c)]
    types = [species.index(s) for s in symbols]
    lat, frac = pos["lattice"], pos["frac"]
    log = []
    args.magmom = args.magmom
    mags, src = build_moments(len(frac), calc, args, log)
    ds = spglib.get_symmetry_dataset((lat, frac, types), symprec=args.symprec)
    rots, trs = _g(ds, "rotations"), _g(ds, "translations")
    sg = sgm.spin_group(lat, frac, types, mags, rots, trs, symprec=args.symprec, mtol=args.mag_symprec)
    cls, why = sgm.classify_magnet(sg)
    print(f"SymPhase {__version__} — magnon chirality analysis")
    print(f"  structure  : {pos['comment']}   ({_g(ds, 'international')}, No. {_g(ds, 'number')})")
    print(f"  moments    : {src}")
    print(f"  magnet     : {cls}")
    deg, forms = sgm.spin_polarisation_form(sg)
    print(f"  symmetry   : magnon chirality splitting has the form of the NR spin splitting -> "
          f"{sgm.wave_label(deg, sg['kind'])}" + (f";  {forms[0]}" if forms else ""))
    exch = read_exchange_file(args.exchange) if args.exchange else None
    res = analyse(lat, frac, types, symbols, mags, rots, trs, J=args.J, exchange=exch, K=args.K,
                  spin=args.spin, rmax=args.rmax, mmin=args.mmin, nproc=args.nproc, n_per=args.nq,
                  nmap=args.nmap, sg=sg)
    print(f"\n  magnetic sites: {res['sites']}   σ = {[int(x) for x in res['sigma']]}   "
          f"S = {[round(x, 3) for x in res['S']]}   axis n = {fmt_vec(res['axis'], 3)}")
    print(f"  bond classes up to {args.rmax} Å (bonds related by a crystal operation share one J):")
    print("   cls   |d| (Å)   #bonds/cell   spins       representative  i -> j + R            d (Å)")
    for c in res["classes"]:
        rep = c["rep"]
        print(f"   J{c['cls']+1:<3d} {c['L']:8.4f}   {c['n_bonds']:6d}        {c['spins']:<11s} "
              f"{rep[0]}{rep[1]} -> {rep[2]}{rep[3]} + {str(rep[4]):<12s} {fmt_vec(rep[5], 3)}")
    if "omega" not in res:
        print("\n  give one J per class with --J (meV, J>0 AFM) or an --exchange file to compute magnons")
        return res
    if res["unstable"]:
        print("\n  WARNING: H(q) not positive definite somewhere: the assumed order is not the LSWT ground state")
    print(f"\n  {res['n_pairs']} bonds used;  max |Δω| (q3 = 0 plane) = {res['max_split']:.4f} meV")
    for (xt, lab) in res["ticks"]:
        k = int(np.argmin(np.abs(res["x"] - xt)))
        print(f"    {lab:<10s} ω = {np.round(res['omega'][k], 3)}  ΔS_z = {np.round(res['dSz'][k], 2)}  "
              f"Δω = {res['split'][k]:+.4f}")
    if args.plot and args.plot.lower() != "none":
        stem, ext = os.path.splitext(args.plot)
        plot_bands(res, save=args.plot, title=f"{cls.lower()}: magnon chirality")
        plot_split_map(res, save=f"{stem}_split_map{ext}")
        np.savetxt(f"{stem}_bands.dat", np.column_stack([res["x"], res["omega"], res["dSz"], res["split"]]),
                   header="x(1/Å)  omega_1..N (meV)  dSz_1..N  split(meV)")
        print(f"\n  written: {args.plot}, {stem}_split_map{ext}, {stem}_bands.dat")
    return res


if __name__ == "__main__":
    main()
