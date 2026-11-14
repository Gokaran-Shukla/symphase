# -*- coding: utf-8 -*-
"""
symphase.supercell — magnetic states that need a supercell
==========================================================

The rotating frame (``--spiral``) describes one planar single-Q spiral in a Hamiltonian
that is invariant under rotations about the spiral axis.  Everything else needs an
explicit magnetic supercell:

* **multi-Q** states (skyrmion crystals, triple-Q, vortex crystals, ...);
* **conical** spirals (field or uniform moment along the spiral axis);
* spirals **distorted** by anisotropy in the spiral plane (higher harmonics 3q, 5q, ...);
* any exchange tensor that is not invariant about one axis.

Workflow (``symphase magnons ... --supercell auto``):

1. Luttinger–Tisza scan: lowest eigenvalue of the 3N x 3N Fourier-transformed coupling
   matrix J(q) gives the ordering vector q_LT and its star {q_s} (crystal rotations).
2. Supercell size: the smallest (diagonal) supercell for which every star member has a
   commensurate approximant whose LT energy is within ``tol`` of the minimum.
3. Supercell model: atoms, exchange tensors, anisotropy and field are mapped exactly.
4. Classical ground state: L-BFGS on unit vectors from many seeds — the user's state,
   ferromagnet along the field, single-Q / conical / multi-Q seeds built from the LT
   eigenvectors, and random states; the lowest energy wins.
5. Characterisation: spin structure factor (uniform weight, number of independent Q,
   harmonics) → class; scalar spin chirality and topological charge (2D, 1 site/cell).
6. Linear spin waves in the supercell (stability, Goldstone modes, folded bands).
7. Files for DFT checks: supercell POSCAR, non-collinear MAGMOM/INCAR, exchange file.

``scan_sizes`` repeats 3–5 for a range of sizes (convergence of E per primitive cell,
best commensurate approximant of an incommensurate state).
"""

import math
import os

import numpy as np

from .magnons import (MUB_MEV, classical_energy, lswt, make_path)


# ----------------------------------------------------------------------------
#  Luttinger–Tisza
# ----------------------------------------------------------------------------
def _bond_R(lat, d, ri, rj):
    return np.round(np.linalg.solve(lat.T, d - (rj - ri))).astype(int)


def lt_matrix(qfrac, lat, cart, sites, pairs, Asite, S):
    """Hermitian 3N x 3N J(q) with E = Σ_q u(q)^† J(q) u(q) (one-site-per-cell normalisation)"""
    N = len(sites)
    loc = {s: k for k, s in enumerate(sites)}
    M = np.zeros((3 * N, 3 * N), complex)
    for i, j, d, J in pairs:
        a, b = loc[i], loc[j]
        R = _bond_R(lat, d, cart[a], cart[b])
        ph = np.exp(2j * np.pi * float(np.dot(qfrac, R)))
        blk = S[a] * S[b] * np.asarray(J) * ph / 2.0
        M[3 * a:3 * a + 3, 3 * b:3 * b + 3] += blk
        M[3 * b:3 * b + 3, 3 * a:3 * a + 3] += blk.conj().T
    for a, A in enumerate(Asite):
        if A is not None:
            M[3 * a:3 * a + 3, 3 * a:3 * a + 3] += S[a] ** 2 * 0.5 * (A + A.T)
    return 0.5 * (M + M.conj().T)


def lt_scan(lat, cart, sites, pairs, Asite, S, qdims=None, ngrid=None):
    """global minimum of the lowest eigenvalue of J(q) (grid + Nelder–Mead)"""
    from scipy.optimize import minimize
    if qdims is None:
        span = np.array([np.linalg.solve(lat.T, p[2]) for p in pairs]) if pairs else np.zeros((0, 3))
        qdims = tuple(k for k in range(3) if span.size and np.abs(span[:, k]).max() > 1e-6) or (0,)
    nd = len(qdims)
    ngrid = ngrid or (96 if nd == 1 else (36 if nd == 2 else 16))

    def lam(x):
        q = np.zeros(3)
        q[list(qdims)] = x
        return float(np.linalg.eigvalsh(lt_matrix(q, lat, cart, sites, pairs, Asite, S))[0])
    grid = np.linspace(-0.5, 0.5, ngrid, endpoint=False)
    pts = np.array(np.meshgrid(*[grid] * nd, indexing="ij")).reshape(nd, -1).T
    vals = np.array([lam(p) for p in pts])
    best = pts[np.argmin(vals)]
    r = minimize(lam, best, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-13, maxiter=4000))
    x = r.x if r.fun <= vals.min() else best
    q = np.zeros(3)
    q[list(qdims)] = x
    q = (q + 0.5) % 1.0 - 0.5
    q[np.abs(q) < 1e-7] = 0.0
    lam0 = lam(np.zeros(nd))
    return dict(q=q, lam=lam(x), lam_q0=lam0, qdims=tuple(qdims))


def q_star(q, rotations, tol=2e-4):
    """symmetry-distinct members of the star of q (q and −q identified)"""
    out = []
    for W in rotations:
        qs = np.asarray(q, float) @ np.linalg.inv(W)       # q' = q W^{-1}  (row vector, r.l.u.)
        qs = (qs + 0.5) % 1.0 - 0.5
        if not any(np.allclose(((qs - o + 0.5) % 1.0) - 0.5, 0, atol=tol) or
                   np.allclose(((qs + o + 0.5) % 1.0) - 0.5, 0, atol=tol) for o in out):
            out.append(qs)
    return out


def suggest_supercell(lt, star, lat, cart, sites, pairs, Asite, S, max_n=24, max_cells=400, tol=0.01):
    """smallest diagonal supercell whose commensurate q-grid contains, for every star member,
    a point with LT energy within tol·scale of the minimum.  Returns (sizes, table)."""
    qdims = lt["qdims"]
    scale = max(1e-9, abs(lt["lam_q0"] - lt["lam"]), max(np.abs(J).max() for *_, J in pairs) * max(S) ** 2 * 0.05)
    if np.linalg.norm(lt["q"]) < 1e-9:
        return (1, 1, 1), [((1, 1, 1), 1, 0.0)]
    cache = {}

    def lam(q):
        k = tuple(np.round(q, 10))
        if k not in cache:
            cache[k] = float(np.linalg.eigvalsh(lt_matrix(np.asarray(q), lat, cart, sites, pairs, Asite, S))[0])
        return cache[k]
    # equal sizes along symmetry-related directions (hexagonal / tetragonal stars)
    cands = []
    for n in range(1, max_n + 1):
        sz = [1, 1, 1]
        for k in qdims:
            sz[k] = n
        cands.append(tuple(sz))
    if len(qdims) > 1 and len(star) == 1:                     # independent sizes only for a 1-member star
        import itertools
        for ns in itertools.product(range(1, max_n + 1), repeat=len(qdims)):
            sz = [1, 1, 1]
            for k, n in zip(qdims, ns):
                sz[k] = n
            if tuple(sz) not in cands:
                cands.append(tuple(sz))
    table = []
    for sz in sorted(cands, key=lambda t: (int(np.prod(t)), t)):
        ncell = int(np.prod(sz))
        if ncell > max_cells:
            continue
        pen = 0.0
        for qs in star:
            near = np.array([round(qs[k] * sz[k]) / sz[k] for k in range(3)])
            pen = max(pen, lam(near) - lt["lam"])
        table.append((sz, ncell, pen / scale))
        if pen / scale <= tol:
            return sz, table
    sz = min(table, key=lambda t: (t[2], t[1]))[0]
    return sz, table


# ----------------------------------------------------------------------------
#  building the supercell model
# ----------------------------------------------------------------------------
def build_supercell(lat, frac, M):
    """supercell lattice (rows = M @ lat), fractional positions and map super -> (prim atom, cell t)"""
    M = np.array(M, int).reshape(3, 3) if np.size(M) == 9 else np.diag(np.array(M, int))
    det = int(round(abs(np.linalg.det(M))))
    if det < 1:
        raise ValueError("singular supercell matrix")
    Minv = np.linalg.inv(M)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ M
    lo, hi = corners.min(0), corners.max(0)
    cells = []
    for t in np.array(np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in range(3)], indexing="ij")).reshape(3, -1).T:
        s = t @ Minv
        if np.all(s > -1e-9) and np.all(s < 1 - 1e-9):
            cells.append(t)
    cells = sorted(cells, key=lambda t: tuple(t @ Minv))
    assert len(cells) == det, (len(cells), det)
    slat = M @ lat
    sfrac, smap = [], []
    for i, f in enumerate(frac):
        for t in cells:
            sfrac.append(((np.asarray(f) + t) @ Minv) % 1.0)
            smap.append((i, np.array(t)))
    return slat, np.array(sfrac), smap, M


def map_model(lat, frac, model, slat, sfrac, smap, M):
    """exchange pairs / anisotropy / field of the primitive model mapped onto the supercell"""
    Minv = np.linalg.inv(M)
    key = {}
    for I, f in enumerate(sfrac):
        key[tuple(np.round(f, 6) % 1.0)] = I
    sites_prim = model["magnetic"]
    loc_p = {s: k for k, s in enumerate(sites_prim)}
    ssites = [I for I, (i, t) in enumerate(smap) if i in loc_p]
    sloc = {I: k for k, I in enumerate(ssites)}
    by_prim = {}
    for I, (i, t) in enumerate(smap):
        by_prim.setdefault(i, []).append((I, t))
    pairs = []
    rows = []
    for i, j, d, J in model["pairs"]:
        Rp = _bond_R(lat, d, lat.T @ frac[i], lat.T @ frac[j])
        for I, t in by_prim[i]:
            tgt = ((np.asarray(frac[j]) + t + Rp) @ Minv)
            fj = np.round(tgt % 1.0, 6) % 1.0
            Jdx = key.get(tuple(fj))
            if Jdx is None:                                          # rounding at the boundary
                d2 = [np.linalg.norm(((sfrac[k] - tgt) + 0.5) % 1.0 - 0.5) for k in range(len(sfrac))]
                Jdx = int(np.argmin(d2))
            Rs = np.round(tgt - sfrac[Jdx]).astype(int)
            pairs.append((I, Jdx, d, J))
            rows.append((I + 1, Jdx + 1, Rs.tolist(), np.asarray(J, float)))
    Asite = [model["Asite"][loc_p[smap[I][0]]] for I in ssites]
    hf = None if model["hfield"] is None else np.array([model["hfield"][loc_p[smap[I][0]]] for I in ssites])
    S = np.array([model["S_arr"][loc_p[smap[I][0]]] for I in ssites])
    return dict(sites=ssites, pairs=pairs, rows=rows, Asite=Asite, hfield=hf, S=S, loc=sloc)


# ----------------------------------------------------------------------------
#  fast classical energy and L-BFGS relaxation
# ----------------------------------------------------------------------------
class ClassicalModel:
    """E(e) = e·Jb·e + Σ S_a² e_a·A_a·e_a + Σ S_a h_a·e_a  with a dense 3N x 3N coupling matrix"""

    def __init__(self, sites, pairs, Asite, S, hfield=None):
        N = len(sites)
        loc = {s: k for k, s in enumerate(sites)}
        Jb = np.zeros((3 * N, 3 * N))
        for i, j, d, J in pairs:
            a, b = loc[i], loc[j]
            Jb[3 * a:3 * a + 3, 3 * b:3 * b + 3] += S[a] * S[b] * np.asarray(J, float)
        for a, A in enumerate(Asite):
            if A is not None:
                Jb[3 * a:3 * a + 3, 3 * a:3 * a + 3] += S[a] ** 2 * np.asarray(A, float)
        self.Js = Jb + Jb.T
        self.Jb = Jb
        self.h = np.zeros(3 * N) if hfield is None else (np.asarray(S)[:, None] * np.asarray(hfield)).ravel()
        self.N = N

    def energy(self, e):
        x = np.asarray(e).ravel()
        return float(x @ self.Jb @ x + self.h @ x)

    def _fg(self, v):
        V = v.reshape(self.N, 3)
        nr = np.linalg.norm(V, axis=1)
        e = V / nr[:, None]
        x = e.ravel()
        E = x @ self.Jb @ x + self.h @ x
        g = (self.Js @ x + self.h).reshape(self.N, 3)
        g = (g - np.sum(g * e, axis=1)[:, None] * e) / nr[:, None]
        return E, g.ravel()

    def relax(self, e0, gtol=1e-10, maxiter=20000):
        from scipy.optimize import minimize
        e0 = np.asarray(e0, float)
        e0 = e0 / np.linalg.norm(e0, axis=1)[:, None]
        r = minimize(self._fg, e0.ravel(), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-16, maxcor=30))
        V = r.x.reshape(self.N, 3)
        e = V / np.linalg.norm(V, axis=1)[:, None]
        return e, self.energy(e), int(r.nit)


# ----------------------------------------------------------------------------
#  seeds
# ----------------------------------------------------------------------------
def _lt_vectors(q, lat, cart, sites, pairs, Asite, S, nvec=2):
    w, U = np.linalg.eigh(lt_matrix(q, lat, cart, sites, pairs, Asite, S))
    return [U[:, k].reshape(-1, 3) for k in range(min(nvec, len(w)))]


def make_seeds(prim, sc, smap, star, lat, cart_prim, start_dirs, field_dir, nrandom=8, rng=None):
    """list of (label, directions) for the supercell sites"""
    rng = rng or np.random.default_rng(0)
    sites, loc_p = sc["sites"], {s: k for k, s in enumerate(prim["magnetic"])}
    pos = np.array([smap[I][1] + 0.0 for I in sites])          # cell index t of each super site
    sub = np.array([loc_p[smap[I][0]] for I in sites])
    nb = field_dir if field_dir is not None else np.array([0.0, 0.0, 1.0])
    seeds = [("start (tiled input order)", np.array([start_dirs[k] for k in sub]))]
    seeds.append(("ferromagnet ∥ " + ("field" if field_dir is not None else "z"), np.tile(nb, (len(sites), 1))))
    P = prim

    def wave(qv, u, phase=0.0):
        z = np.exp(1j * (2 * np.pi * pos @ qv + phase))
        return np.real(u[sub] * z[:, None])

    norm = lambda X: X / np.maximum(np.linalg.norm(X, axis=1), 1e-12)[:, None]   # noqa: E731
    for k, qv in enumerate(star):
        for m, u in enumerate(_lt_vectors(qv, lat, cart_prim, P["magnetic"], P["pairs"], P["Asite"], P["S_arr"])):
            u = u / max(np.abs(u).max(), 1e-12)
            X = wave(qv, u)
            if np.linalg.norm(X) < 1e-9:
                X = wave(qv, u, np.pi / 2)
            seeds.append((f"single-Q q{k + 1} (LT vector {m + 1})", norm(X + 1e-9 * rng.normal(size=X.shape))))
            for c in (0.3, 0.7):
                seeds.append((f"conical q{k + 1}, m∥ = {c}", norm(X / max(np.abs(X).max(), 1e-12) + c * nb)))
    if len(star) > 1:
        u0 = []
        for qv in star:                       # phase: every wave points along −n̂ at the origin (core)
            u = _lt_vectors(qv, lat, cart_prim, P["magnetic"], P["pairs"], P["Asite"], P["S_arr"], 1)[0]
            c0 = u[0] @ nb
            if abs(c0) > 1e-8:
                u = u * (-abs(c0) / c0)
            u0.append(u)
        for c in (0.6, 1.0, 1.5):
            X = sum(wave(qv, u / max(np.abs(u).max(), 1e-12)) for qv, u in zip(star, u0))
            seeds.append((f"multi-Q ({len(star)}Q) core-aligned + {c:.1f}·n̂", norm(X / max(np.abs(X).max(), 1e-12)
                                                                                 + c * nb)))
        for c in (0.0, 0.4, 0.8):
            for sgn in (1, -1):
                X = sum(wave(qv, u / max(np.abs(u).max(), 1e-12)) for qv, u in zip(star, u0))
                seeds.append((f"multi-Q ({len(star)}Q) + {c:.1f}·n̂" + (" (reversed)" if sgn < 0 else ""),
                              norm(sgn * X + c * nb)))
            for t in range(2):
                ph = rng.uniform(0, 2 * np.pi, len(star))
                X = sum(wave(qv, u / max(np.abs(u).max(), 1e-12), p) for qv, u, p in zip(star, u0, ph))
                seeds.append((f"multi-Q ({len(star)}Q), random phases + {c:.1f}·n̂", norm(X + c * nb)))
    for t in range(nrandom):
        seeds.append((f"random {t + 1}", norm(rng.normal(size=(len(sites), 3)))))
    # tiny noise: breaks exact cancellations (zero vectors) and accidental symmetry
    return [(lab, norm(X + 1e-6 * rng.normal(size=X.shape))) for lab, X in seeds]


# ----------------------------------------------------------------------------
#  characterisation
# ----------------------------------------------------------------------------
def structure_factor(e, smap, sites, nsub, sub, M):
    """S(Q) on the commensurate grid of the supercell, averaged over sublattices; weights sum to 1"""
    Minv = np.linalg.inv(M)
    pos = np.array([smap[I][1] for I in sites], float)
    det = int(round(abs(np.linalg.det(M))))
    # Q (r.l.u. of the primitive cell) with Q·(rows of M) integer  <=>  Q = M^{-1} n
    grid = set()
    D = np.diag(M).astype(int) if np.allclose(M, np.diag(np.diag(M))) else None
    if D is not None:
        for a in range(D[0]):
            for b in range(D[1]):
                for c in range(D[2]):
                    grid.add((a / D[0], b / D[1], c / D[2]))
    else:
        for n in np.array(np.meshgrid(*[np.arange(-det, det + 1)] * 3, indexing="ij")).reshape(3, -1).T:
            Q = np.round((Minv @ n) % 1.0, 8) % 1.0
            grid.add(tuple(Q))
    out = []
    for Q in sorted(grid):
        Q = np.array(Q)
        z = np.exp(-2j * np.pi * pos @ Q)
        w = 0.0
        for s in range(nsub):
            msk = sub == s
            m = (e[msk] * z[msk, None]).sum(0) / msk.sum()
            w += float(np.real(np.vdot(m, m)))
        out.append((((Q + 0.5) % 1.0) - 0.5, w / nsub))
    Qs = np.array([o[0] for o in out])
    W = np.array([o[1] for o in out])
    return Qs, W


def _is_multiple(Q, F, nmax=7):
    for n in sorted(range(-nmax, nmax + 1), key=lambda k: (abs(k), -k)):
        if n and np.allclose(((n * F - Q + 0.5) % 1.0) - 0.5, 0, atol=1e-6):
            return n
    return 0


def classify_state(e, smap, sites, sub, nsub, M, thr=0.01, thr_h=1e-4):
    """class of the relaxed supercell state from its structure factor"""
    Qs, W = structure_factor(e, smap, sites, nsub, sub, M)
    w0 = float(W[np.all(np.abs(Qs) < 1e-9, axis=1)].sum())
    peaks = [(Q, w) for Q, w in zip(Qs, W) if w > thr_h and np.linalg.norm(Q) > 1e-9]
    pairs = []
    for Q, w in sorted(peaks, key=lambda t: -t[1]):
        if not any(np.allclose(((Q + P + 0.5) % 1.0) - 0.5, 0, atol=1e-6) for P, _ in pairs):
            twin = [w2 for Q2, w2 in peaks if np.allclose(((Q + Q2 + 0.5) % 1.0) - 0.5, 0, atol=1e-6)]
            pairs.append((Q, w + (twin[0] if twin and not np.allclose(((2 * Q + 0.5) % 1) - .5, 0) else 0)))
    fund, harm = [], []
    wmax = max([w for _, w in pairs], default=0.0)
    for Q, w in pairs:
        n = next((_is_multiple(Q, F) for F, _ in fund if _is_multiple(Q, F)), 0)
        if n:
            harm.append((Q, w, n))
        elif w > 0.2 * wmax and w > thr:
            fund.append((Q, w))
        else:
            harm.append((Q, w, 0))                           # combination peak (multi-Q)
    _, sv, _ = np.linalg.svd(e - 0 * e.mean(0), full_matrices=False)
    sv = sv / sv[0]
    geom = "collinear" if sv[1] < 1e-3 else ("coplanar" if sv[2] < 1e-3 else "non-coplanar")
    nQ = len(fund)
    if nQ == 0:
        cls = "commensurate with the primitive cell (q = 0)"
    elif nQ == 1:
        nh = len([h for h in harm if abs(h[2]) > 1])
        if w0 > thr and geom == "non-coplanar":
            cls = "conical spiral (uniform component + single Q)"
        elif w0 > thr and geom == "coplanar":
            cls = "fan / canted spiral (uniform component + single Q, coplanar)"
        elif geom == "collinear":
            cls = "collinear modulated (single Q" + (", squared-up/soliton" if nh else "") + ")"
        elif nh:
            hw = sum(h[1] for h in harm if abs(h[2]) > 1)
            cls = "distorted single-Q spiral (harmonics " + \
                  ", ".join(sorted({f"{abs(h[2])}q" for h in harm if abs(h[2]) > 1})) + f"; weight {hw:.2g})"
        else:
            cls = "single-Q spiral"
    else:
        cls = f"multi-Q state ({nQ} independent Q)" + (" with uniform component" if w0 > thr else "")
    return dict(cls=cls, w0=w0, fundamentals=[(Q.tolist(), float(w)) for Q, w in fund],
                harmonics=[(Q.tolist(), float(w), int(n)) for Q, w, n in harm], geometry=geom,
                Q=Qs, W=W)


def topological_charge(e, smap, sites, M, dims):
    """Berg–Lüscher lattice skyrmion number (one magnetic atom per primitive cell, 2D)"""
    if len(dims) != 2 or len(sites) != int(round(abs(np.linalg.det(M)))):
        return None
    Minv = np.linalg.inv(M)
    idx = {}
    for k, I in enumerate(sites):
        idx[tuple(np.round((smap[I][1] @ Minv) % 1.0, 6) % 1.0)] = k

    def at(t):
        return idx[tuple(np.round((t @ Minv) % 1.0, 6) % 1.0)]
    a, b = np.zeros(3, int), np.zeros(3, int)
    a[dims[0]] = 1
    b[dims[1]] = 1

    def omega(e1, e2, e3):
        return 2 * math.atan2(float(e1 @ np.cross(e2, e3)), 1 + e1 @ e2 + e2 @ e3 + e3 @ e1)
    tot = 0.0
    for k, I in enumerate(sites):
        t = smap[I][1]
        tot += omega(e[at(t)], e[at(t + a)], e[at(t + b)])
        tot += omega(e[at(t + a)], e[at(t + a + b)], e[at(t + b)])
    return tot / (4 * math.pi)


# ----------------------------------------------------------------------------
#  driver
# ----------------------------------------------------------------------------
def run_supercell(lat, frac, types, symbols, mags, rotations, translations=None, size="auto", restarts=8, max_n=24,
                  max_cells=400, tol=0.01, nproc=None, n_per=40, nmap=21, spectrum=True, seed=0,
                  **model_kw):
    """complete supercell workflow; model_kw are passed to magnons.analyse (J, exchange, K, field, ...)"""
    from .magnons import analyse
    if translations is None:
        translations = np.zeros((len(rotations), 3))
    prim = analyse(lat, frac, types, symbols, mags, rotations, translations, return_model=True, **model_kw)
    if "pairs" not in prim:
        raise ValueError("no exchange given (--J or --exchange)")
    mags_ = np.asarray(mags, float)
    cart = np.array([lat.T @ frac[m] for m in prim["magnetic"]])
    S = prim["S_arr"]
    lt = lt_scan(lat, cart, prim["magnetic"], prim["pairs"], prim["Asite"], S)
    star = q_star(lt["q"], rotations) if np.linalg.norm(lt["q"]) > 1e-9 else []
    if isinstance(size, str) and size == "auto":
        sz, table = suggest_supercell(lt, star or [lt["q"]], lat, cart, prim["magnetic"], prim["pairs"],
                                      prim["Asite"], S, max_n=max_n, max_cells=max_cells, tol=tol)
        M = np.diag(sz)
    else:
        table = None
        M = np.array(size, int).reshape(3, 3) if np.size(size) == 9 else np.diag(np.array(size, int))
    slat, sfrac, smap, M = build_supercell(lat, frac, M)
    sc = map_model(lat, frac, prim, slat, sfrac, smap, M)
    ncell = int(round(abs(np.linalg.det(M))))
    fdir = None
    if prim["hfield"] is not None:
        fdir = -prim["hfield"][0] / np.linalg.norm(prim["hfield"][0])
    cm = ClassicalModel(sc["sites"], sc["pairs"], sc["Asite"], sc["S"], sc["hfield"])
    # star members that fit the supercell (commensurate approximants)
    star_c = []
    for qv in (star or ([lt["q"]] if np.linalg.norm(lt["q"]) > 1e-9 else [])):
        qc = np.array([round(qv[k] * M[k, k]) / M[k, k] if np.allclose(M, np.diag(np.diag(M))) else qv[k]
                       for k in range(3)])
        if np.linalg.norm(qc) > 1e-9:
            star_c.append(qc)
    rng = np.random.default_rng(seed)
    seeds = make_seeds(prim, sc, smap, star_c, lat, cart, prim["dirs0"], fdir, nrandom=restarts, rng=rng)
    results = []
    for lab, e0 in seeds:
        if not np.all(np.isfinite(e0)):
            continue
        e, E, nit = cm.relax(e0)
        if np.isfinite(E):
            results.append(dict(seed=lab, E=E, E_cell=E / ncell, dirs=e, iters=nit))
    results.sort(key=lambda r: r["E"])
    best = results[0]
    nsub = len(prim["magnetic"])
    loc_p = {s: k for k, s in enumerate(prim["magnetic"])}
    sub = np.array([loc_p[smap[I][0]] for I in sc["sites"]])
    ch = classify_state(best["dirs"], smap, sc["sites"], sub, nsub, M)
    dims = lt["qdims"] if len(lt["qdims"]) == 2 else ()
    Qtop = topological_charge(best["dirs"], smap, sc["sites"], M, dims) if dims else None
    if Qtop is not None and abs(Qtop) > 0.5 and len(ch["fundamentals"]) >= 2:
        ch["cls"] = (f"skyrmion crystal ({len(ch['fundamentals'])}Q, {abs(round(Qtop))} skyrmion"
                     f"{'s' if abs(round(Qtop)) != 1 else ''} per supercell)")
    deg = sum(1 for r in results if abs(r["E"] - best["E"]) < 1e-6 * max(1.0, abs(best["E"])))
    out = dict(lt=lt, star=[s.tolist() for s in star], size_table=table, M=M, n_cells=ncell, slat=slat,
               sfrac=sfrac, smap=smap, model=sc, seeds=results, best=best, character=ch, Q_top=Qtop,
               E_cell=best["E_cell"], n_best=deg, prim=prim,
               net_moment_per_site=(best["dirs"] * (2 * sc["S"])[:, None]).sum(0) / len(sc["sites"]))
    # reference energies per primitive cell
    E_in = classical_energy(prim["dirs0"], S, prim["magnetic"], prim["pairs"], prim["Asite"], prim["hfield"])
    out["E_input_cell"] = float(E_in)
    out["E_LT_bound_cell"] = float(lt["lam"] * len(prim["magnetic"])) if prim["hfield"] is None else None
    if spectrum:
        axis = fdir if fdir is not None else (np.mean(best["dirs"], 0) / np.linalg.norm(np.mean(best["dirs"], 0))
                                              if np.linalg.norm(np.mean(best["dirs"], 0)) > 0.05
                                              else np.array([0.0, 0.0, 1.0]))
        out["lswt"] = lswt(slat, sc["sites"], best["dirs"], sc["S"], sc["pairs"], sc["Asite"], axis,
                           nproc=nproc, n_per=n_per, nmap=nmap, hfield=sc["hfield"])
    return out


def scan_sizes(lat, frac, types, symbols, mags, rotations, translations, sizes, restarts=4, **model_kw):
    """E per primitive cell versus supercell size (convergence / best commensurate approximant)"""
    rows = []
    for sz in sizes:
        r = run_supercell(lat, frac, types, symbols, mags, rotations, translations, size=sz, restarts=restarts,
                          spectrum=False, **model_kw)
        rows.append(dict(size=tuple(int(x) for x in np.diag(r["M"])), n_cells=r["n_cells"], E_cell=r["E_cell"],
                         cls=r["character"]["cls"], Q_top=r["Q_top"]))
    return rows


# ----------------------------------------------------------------------------
#  output files (DFT checks, symmetry analysis of the supercell state)
# ----------------------------------------------------------------------------
def write_outputs(res, symbols, mags, outdir, comment="SymPhase magnetic supercell"):
    os.makedirs(outdir, exist_ok=True)
    slat, sfrac, smap = res["slat"], res["sfrac"], res["smap"]
    sc = res["model"]
    ssym = [symbols[i] for i, _ in smap]
    species = []
    for s in ssym:
        if s not in species:
            species.append(s)
    order = sorted(range(len(ssym)), key=lambda I: (species.index(ssym[I]), I))
    newidx = {I: k for k, I in enumerate(order)}
    with open(os.path.join(outdir, "POSCAR"), "w") as fh:
        fh.write(f"{comment}  M = {res['M'].tolist()}\n1.0\n")
        for v in slat:
            fh.write(f"  {v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}\n")
        fh.write("  " + " ".join(species) + "\n  " + " ".join(str(ssym.count(s)) for s in species) + "\nDirect\n")
        for I in order:
            f = sfrac[I]
            fh.write(f"  {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}  {ssym[I]}\n")
    mom = np.zeros((len(ssym), 3))
    norms = np.linalg.norm(np.asarray(mags, float), axis=1)
    for k, I in enumerate(sc["sites"]):
        mom[newidx[I]] = res["best"]["dirs"][k] * norms[smap[I][0]]
    magmom = " ".join(f"{m[0]:.4f} {m[1]:.4f} {m[2]:.4f}" for m in mom)
    with open(os.path.join(outdir, "INCAR"), "w") as fh:
        fh.write("# non-collinear starting moments of the SymPhase supercell ground state\n"
                 "# add your usual ENCUT / EDIFF / KPOINTS and, for DMI, LSORBIT = .TRUE.\n"
                 "LNONCOLLINEAR = .TRUE.\nISYM = 0\n# I_CONSTRAINED_M = 1 ; LAMBDA = 10 ; M_CONSTR = <MAGMOM>  "
                 "(constrained-moment check)\n"
                 f"MAGMOM = {magmom}\n")
    with open(os.path.join(outdir, "J_super.dat"), "w") as fh:
        fh.write("# i j n1 n2 n3 Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz  (meV, supercell POSCAR order)\n")
        for i, j, R, J in sc["rows"]:
            a, b = newidx[i - 1] + 1, newidx[j - 1] + 1
            fh.write(f"{a:4d} {b:4d} {R[0]:3d} {R[1]:3d} {R[2]:3d} " + " ".join(f"{x: .6f}" for x in J.ravel()) + "\n")
    with open(os.path.join(outdir, "spins.dat"), "w") as fh:
        fh.write("# atom(POSCAR)  x y z (Cartesian, Å)  ex ey ez\n")
        for k, I in enumerate(sc["sites"]):
            r = slat.T @ sfrac[I]
            e = res["best"]["dirs"][k]
            fh.write(f"{newidx[I] + 1:5d} {r[0]:10.4f} {r[1]:10.4f} {r[2]:10.4f}  {e[0]: .6f} {e[1]: .6f} {e[2]: .6f}\n")
    with open(os.path.join(outdir, "seeds.dat"), "w") as fh:
        fh.write("# seed   E (meV / primitive cell)   L-BFGS iterations\n")
        for r in res["seeds"]:
            fh.write(f"{r['E_cell']:16.8f}  {r['iters']:6d}   {r['seed']}\n")
    return outdir


def plot_state(res, save=None, figsize=(11.0, 4.6), tile=2):
    """spin texture (sites coloured by the out-of-plane component, in-plane arrows; the supercell
    is repeated tile x tile times) and the strongest structure-factor components"""
    import textwrap
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    sc, slat, sfrac = res["model"], res["slat"], res["sfrac"]
    e = res["best"]["dirs"]
    r0 = np.array([slat.T @ sfrac[I] for I in sc["sites"]])
    span = np.ptp(np.r_[r0, r0 + slat[0], r0 + slat[1]], axis=0)
    chain = sorted(np.argsort(-np.ptp(r0, axis=0))[:2].tolist())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.45, 1]))
    nn = min(np.linalg.norm(d) for *_, d, _J in sc["pairs"]) if sc["pairs"] else 1.0
    if np.ptp(r0[:, chain[1]]) < 1e-6:                      # 1D chain: angle profile
        ax_ = chain[0]
        reps = [r0 + k * slat[np.argmax(np.abs(slat[:, ax_]))] for k in range(tile)]
        x = np.concatenate([rr[:, ax_] for rr in reps])
        E = np.tile(e, (tile, 1))
        o = np.argsort(x)
        a1.quiver(x[o], np.zeros(len(o)), E[o, 0], E[o, 1], color="k", scale=1.6 / nn, scale_units="xy",
                  angles="xy", width=0.005, pivot="mid")
        scat = a1.scatter(x[o], np.zeros(len(o)), c=E[o, 2], cmap="coolwarm", vmin=-1, vmax=1, s=60, zorder=0)
        a1.set_ylim(-nn, nn)
        a1.set_yticks([])
        a1.set_xlabel("position along the chain (Å);  arrows: (s_x, s_y)")
    else:
        i1, i2 = chain
        pts, E = [], []
        for u in range(tile):
            for v in range(tile):
                pts.append(r0 + u * slat[0] + v * slat[1])
                E.append(e)
        P, E = np.concatenate(pts), np.concatenate(E)
        i3 = 3 - i1 - i2
        scat = a1.scatter(P[:, i1], P[:, i2], c=E[:, i3], cmap="coolwarm", vmin=-1, vmax=1,
                          s=max(8.0, 2600.0 / len(P)), marker="h", lw=0, zorder=0)
        a1.quiver(P[:, i1], P[:, i2], E[:, i1], E[:, i2], color="k", pivot="mid", scale=1.25 / nn,
                  scale_units="xy", angles="xy", width=0.0025, headwidth=3.5)
        cell = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ slat[:2, [i1, i2]]
        a1.plot(cell[:, 0], cell[:, 1], color="0.2", lw=1.0, ls="--")
        a1.set_aspect("equal")
        a1.set_xlabel(f"{'xyz'[i1]} (Å)")
        a1.set_ylabel(f"{'xyz'[i2]} (Å)")
    cb = fig.colorbar(scat, ax=a1, fraction=0.040, pad=0.02)
    cb.set_label("s$_z$ (out of plane)")
    ch = res["character"]
    ttl = ch["cls"] + (f";  N$_{{sk}}$ = {res['Q_top']:+.2f} / supercell" if res.get("Q_top") not in (None,)
                       and abs(res["Q_top"]) > 0.5 else "")
    a1.set_title("\n".join(textwrap.wrap(ttl, 52)), fontsize=10)
    Q, W = ch["Q"], ch["W"]
    o = np.argsort(-W)[:8]
    lab = [f"({Q[k][0]:+.3f}, {Q[k][1]:+.3f}, {Q[k][2]:+.3f})" for k in o]
    a2.barh(range(len(o))[::-1], W[o], color=["#7f8c8d" if np.linalg.norm(Q[k]) < 1e-9 else "#2c5d8f" for k in o])
    a2.set_yticks(range(len(o))[::-1])
    a2.set_yticklabels(lab, fontsize=8)
    a2.set_xlabel("structure-factor weight (grey: uniform, Q = 0)")
    a2.set_title("strongest Fourier components, Q in r.l.u.", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


def report(res, symbols, prn=print):
    lt = res["lt"]
    prn("\n  ── supercell workflow (multi-Q / conical / distorted states) ──")
    prn(f"  1. Luttinger–Tisza minimum: q_LT = {np.round(lt['q'], 5)} r.l.u.  (λ = {lt['lam']:.5f} meV, "
        f"λ(q=0) = {lt['lam_q0']:.5f});  star: {len(res['star'])} member(s)")
    for k, s in enumerate(res["star"]):
        prn(f"       q{k + 1} = {np.round(s, 5)}")
    if res["size_table"]:
        prn("  2. supercell search (penalty = LT energy cost of the best commensurate q, relative units):")
        for sz, n, pen in res["size_table"][-6:]:
            prn(f"       {str(tuple(int(x) for x in sz)):<12s} {n:5d} cells   penalty {pen:9.5f}")
    prn(f"     supercell M = {res['M'].tolist()}  ({res['n_cells']} primitive cells, "
        f"{len(res['model']['sites'])} magnetic sites)")
    prn(f"  3. model mapped: {len(res['model']['pairs'])} bonds, anisotropy "
        f"{'yes' if any(a is not None for a in res['model']['Asite']) else 'no'}, field "
        f"{'yes' if res['model']['hfield'] is not None else 'no'}")
    prn(f"  4. {len(res['seeds'])} seeds relaxed (L-BFGS); lowest energies per primitive cell:")
    for r in res["seeds"][:6]:
        prn(f"       {r['E_cell']:14.6f} meV   {r['seed']}")
    prn(f"     ground state reached from {res['n_best']} seed(s);  input state: {res['E_input_cell']:.6f} meV/cell"
        + (f";  LT lower bound: {res['E_LT_bound_cell']:.6f}" if res["E_LT_bound_cell"] is not None else ""))
    ch = res["character"]
    prn(f"  5. state: {ch['cls']}   [{ch['geometry']}]")
    prn(f"       uniform weight w(0) = {ch['w0']:.4f};  net moment/site = {np.round(res['net_moment_per_site'], 4)} μB")
    for Q, w in ch["fundamentals"]:
        prn(f"       fundamental Q = {np.round(Q, 4)}   weight {w:.4f}")
    for Q, w, n in ch["harmonics"][:6]:
        prn(f"       {'harmonic ' + str(abs(n)) + 'Q' if n else 'combination'} {np.round(Q, 4)}   weight {w:.4f}")
    if res["Q_top"] is not None:
        prn(f"       topological charge per supercell: {res['Q_top']:+.4f}")
    if "lswt" in res:
        L = res["lswt"]
        prn(f"  6. spin waves in the supercell: {'UNSTABLE (imaginary modes)' if L['unstable'] else 'stable'};"
            f"  zero modes at Γ: {L['n_goldstone']};  {L['omega'].shape[1]} folded bands")
