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
#  LSWT for arbitrary (collinear or non-collinear) magnetic order
# ----------------------------------------------------------------------------
def dm_matrix(D):
    """3x3 matrix M with  S_i^T M S_j = D . (S_i x S_j)"""
    Dx, Dy, Dz = D
    return np.array([[0.0, Dz, -Dy], [-Dz, 0.0, Dx], [Dy, -Dx, 0.0]])


def local_frames(dirs):
    """u_i = x_i + i y_i (complex, ⊥ spin) and v_i = spin direction, right-handed frames"""
    U, V = [], []
    for e in dirs:
        e = np.asarray(e, float) / np.linalg.norm(e)
        t = np.array([0.0, 0.0, 1.0]) if abs(e[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(t, e)
        x /= np.linalg.norm(x)
        y = np.cross(e, x)
        U.append(x + 1j * y)
        V.append(e)
    return np.array(U), np.array(V)


def _as_matrix(J):
    J = np.asarray(J, float)
    return J * np.eye(3) if J.ndim == 0 else J


def _pairs_from(bonds, Jcls=None, Jexplicit=None):
    """list of (i, j, d, 3x3 J)"""
    pairs = []
    bset = {(b["i"], b["j"], tuple(b["R"])): b for b in bonds}
    if Jexplicit is not None:
        for (i, j, R), J in Jexplicit.items():
            b = bset.get((i, j, R))
            if b is not None:
                pairs.append((i, j, b["d"], _as_matrix(J)))
                continue
            b = bset.get((j, i, tuple(-np.array(R))))
            if b is None:
                continue
            pairs.append((j, i, b["d"], _as_matrix(J).T))   # S_i^T J S_j = S_j^T J^T S_i
        return pairs
    for b in bonds:
        J = Jcls[b["cls"]] if Jcls is not None else 0.0
        if np.any(np.asarray(J) != 0):
            pairs.append((b["i"], b["j"], b["d"], _as_matrix(J)))
    return pairs


def hamiltonian(q, sites, U, V, S, pairs, Asite):
    """bosonic BdG matrix H(q) (2N x 2N, meV) for H = Σ S_i^T J_ij S_j + Σ S_i^T A_i S_i"""
    N = len(sites)
    loc = {s: n for n, s in enumerate(sites)}

    def block(qq):
        A = np.zeros((N, N), complex)
        B = np.zeros((N, N), complex)
        for i, j, d, J in pairs:
            a, b = loc[i], loc[j]
            r = math.sqrt(S[a] * S[b]) / 2.0
            ph = np.exp(1j * qq @ d)
            c1 = r * (U[a] @ J @ np.conj(U[b]))
            c3 = r * (U[a] @ J @ U[b])
            dv = V[a] @ J @ V[b]
            A[a, b] += c1 * ph
            A[b, a] += np.conj(c1) * np.conj(ph)
            B[a, b] += c3 * ph
            B[b, a] += c3 * np.conj(ph)
            A[a, a] += -S[b] * dv
            A[b, b] += -S[a] * dv
        for a in range(N):
            Am = Asite[a]
            if Am is None:
                continue
            A[a, a] += S[a] * (np.real(U[a] @ Am @ np.conj(U[a])) - 2 * V[a] @ Am @ V[a])
            B[a, a] += S[a] * (U[a] @ Am @ U[a])
        return A, B
    A, B = block(q)
    Am, _ = block(-q)
    return np.block([[A, B], [B.conj().T, Am.T]])


def colpa(H, eps=1e-9):
    """positive eigen-energies and the first N columns of the paraunitary T (Ψ = T Φ)"""
    n2 = H.shape[0]
    N = n2 // 2
    H0 = 0.5 * (H + H.conj().T)
    scale = max(1.0, np.abs(H0).max())
    Kc = None
    for e in (eps, 1e-7, 1e-6, 1e-5):                  # Goldstone modes: tiny regulariser
        try:
            Kc = np.linalg.cholesky(H0 + e * scale * np.eye(n2)).conj().T   # H = K^† K
            break
        except np.linalg.LinAlgError:
            continue
    if Kc is None:
        return None, None
    s3 = np.diag(np.r_[np.ones(N), -np.ones(N)])
    W = Kc @ s3 @ Kc.conj().T
    w, Uv = np.linalg.eigh(0.5 * (W + W.conj().T))
    order = np.argsort(-w)
    w, Uv = w[order], Uv[:, order]
    om = w[:N]
    T = np.linalg.solve(Kc, Uv[:, :N] * np.sqrt(np.abs(om))[None, :])
    return om, T


def _w_modes(q):
    sites, U, V, S, pairs, Asite, axis = (get_shared(k) for k in ("sites", "U", "V", "S", "pairs", "Asite", "axis"))
    N = len(sites)
    om, T = colpa(hamiltonian(q, sites, U, V, S, pairs, Asite))
    if om is None:
        return np.full(N, np.nan), np.zeros(N), np.zeros((N, 3))
    o = np.argsort(om)
    om, T = om[o], T[:, o]
    w = np.abs(T[:N]) ** 2 + np.abs(T[N:]) ** 2              # (N sites, N modes)
    dS = -(V.T @ w).T                                          # magnon spin vector per mode
    pv = V @ axis                                              # spin projection of each site
    Dm = -(T[:N].conj().T * pv[None, :]) @ T[:N] - (T[N:].conj().T * pv[None, :]) @ T[N:]
    proj = np.real(np.diag(Dm)).copy()
    tol = 1e-6 * max(1.0, np.abs(om).max())
    k = 0
    while k < N:                                               # pure states in degenerate subspaces
        m = k + 1
        while m < N and abs(om[m] - om[k]) < tol:
            m += 1
        if m - k > 1:
            proj[k:m] = np.sort(np.linalg.eigvalsh(0.5 * (Dm[k:m, k:m] + Dm[k:m, k:m].conj().T)))
        k = m
    return om, proj, dS


def spectrum(qs, sites, U, V, S, pairs, Asite, axis, nproc=None):
    res = pmap(_w_modes, list(qs), nproc=nproc, backend="process",
               shared=dict(sites=sites, U=U, V=V, S=np.asarray(S, float), pairs=pairs, Asite=Asite,
                           axis=np.asarray(axis, float)), min_items=64)
    return (np.array([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[2] for r in res]))


def chirality_split(om, proj, thr=0.1):
    """Δω = (lowest mode with spin projection > thr) − (lowest with < −thr), nan otherwise"""
    out = np.full(len(om), np.nan)
    for k in range(len(om)):
        a = om[k][proj[k] < -thr]
        b = om[k][proj[k] > thr]
        if len(a) and len(b):
            out[k] = b.min() - a.min()
    return out


# ----------------------------------------------------------------------------
#  classical ground state
# ----------------------------------------------------------------------------
def classical_energy(dirs, S, sites, pairs, Asite):
    loc = {s: n for n, s in enumerate(sites)}
    E = 0.0
    for i, j, d, J in pairs:
        a, b = loc[i], loc[j]
        E += S[a] * S[b] * dirs[a] @ J @ dirs[b]
    for a, Am in enumerate(Asite):
        if Am is not None:
            E += S[a] ** 2 * dirs[a] @ Am @ dirs[a]
    return E


def relax_spins(dirs, S, sites, pairs, Asite, steps=20000, tol=1e-10, lr=0.05):
    """projected steepest descent of the classical energy on unit vectors (fixed magnetic cell)"""
    loc = {s: n for n, s in enumerate(sites)}
    e = np.array(dirs, float)
    e /= np.linalg.norm(e, axis=1)[:, None]
    E_old = classical_energy(e, S, sites, pairs, Asite)
    scale = max(1e-9, max(np.abs(J).max() for *_, J in pairs) if pairs else 1.0) * max(S) ** 2
    for it in range(steps):
        g = np.zeros_like(e)
        for i, j, d, J in pairs:
            a, b = loc[i], loc[j]
            g[a] += S[a] * S[b] * (J @ e[b])
            g[b] += S[a] * S[b] * (J.T @ e[a])
        for a, Am in enumerate(Asite):
            if Am is not None:
                g[a] += S[a] ** 2 * (Am + Am.T) @ e[a]
        g -= np.sum(g * e, axis=1)[:, None] * e                 # tangent component
        e = e - lr / scale * g
        e /= np.linalg.norm(e, axis=1)[:, None]
        if it % 50 == 0:
            E = classical_energy(e, S, sites, pairs, Asite)
            if abs(E_old - E) < tol * max(1.0, abs(E)) and np.abs(g).max() < 1e-7 * scale:
                break
            E_old = E
    return e, classical_energy(e, S, sites, pairs, Asite), it


# ----------------------------------------------------------------------------
#  incommensurate single-Q spirals (rotating frame, no supercell)
# ----------------------------------------------------------------------------
def rot_n(n, ang):
    """rotation matrix by angle ang about unit vector n"""
    n = np.asarray(n, float) / np.linalg.norm(n)
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx


def rotation_invariant(M, n, tol=1e-8):
    """True if R_n(φ)^T M R_n(φ) = M for all φ (tested at a generic angle)"""
    if M is None:
        return True
    R = rot_n(n, 1.0)
    return np.allclose(R.T @ M @ R, M, atol=tol * max(1.0, np.abs(M).max()))


def spiral_axis_guess(pairs):
    """common direction of the DM vectors (antisymmetric parts), else z"""
    ds = []
    for *_, J in pairs:
        A = 0.5 * (J - J.T)
        d = np.array([A[1, 2], A[2, 0], A[0, 1]])
        if np.linalg.norm(d) > 1e-10:
            ds.append(d / np.linalg.norm(d))
    if ds and all(abs(abs(d @ ds[0]) - 1) < 1e-6 for d in ds):
        return ds[0] * (1 if ds[0][np.argmax(np.abs(ds[0]))] > 0 else -1)
    return np.array([0.0, 0.0, 1.0])


def _bond_R(lat, d, ri, rj):
    """integer lattice vector R of a bond from its Cartesian vector d = r_j + R − r_i"""
    return np.round(np.linalg.solve(lat.T, d - (rj - ri))).astype(int)


def spiral_pairs(pairs, lat, cart, sites, qfrac, n):
    """bond matrices in the rotating frame: J -> J R_n(2π q·R)"""
    loc = {s: k for k, s in enumerate(sites)}
    out = []
    for i, j, d, J in pairs:
        R = _bond_R(lat, d, cart[loc[i]], cart[loc[j]])
        out.append((i, j, d, J @ rot_n(n, 2 * np.pi * float(np.dot(qfrac, R)))))
    return out


def _plane_basis(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(n, t)
    x /= np.linalg.norm(x)
    return x, np.cross(n, x)


def spiral_energy(qfrac, psi, S, sites, pairs, Asite, lat, cart, n):
    """classical energy per magnetic cell of the planar spiral (phases psi in the plane ⊥ n)"""
    ex, ey = _plane_basis(n)
    dirs = np.array([math.cos(p) * ex + math.sin(p) * ey for p in psi])
    return classical_energy(dirs, S, sites, spiral_pairs(pairs, lat, cart, sites, qfrac, n), Asite), dirs


def find_spiral(S, sites, pairs, Asite, lat, cart, n, qdims=(0, 1, 2), ngrid=None, qfrac=None, seed=0):
    """optimal single-Q planar spiral: (q in r.l.u., phases, directions, E)"""
    from scipy.optimize import minimize
    N = len(sites)
    rng = np.random.default_rng(seed)

    def E_of(x, q=None):
        q_ = np.zeros(3)
        if q is None:
            q_[list(qdims)] = x[:len(qdims)]
            ps = np.r_[0.0, x[len(qdims):]]
        else:
            q_ = np.asarray(q, float)
            ps = np.r_[0.0, x]
        return spiral_energy(q_, ps, S, sites, pairs, Asite, lat, cart, n)[0]

    def best_phases(q):
        if N == 1:
            return np.zeros(0), E_of(np.zeros(0), q)
        best = None
        for t in range(6):
            x0 = rng.uniform(0, 2 * np.pi, N - 1)
            r = minimize(lambda x: E_of(x, q), x0, method="BFGS")
            if best is None or r.fun < best[1]:
                best = (r.x, r.fun)
        return best
    if qfrac is not None:
        ph, E = best_phases(qfrac)
        q = np.asarray(qfrac, float)
    else:
        nd = len(qdims)
        ngrid = ngrid or (41 if nd == 1 else (25 if nd == 2 else 13))
        grid = np.linspace(-0.5, 0.5, ngrid)
        best = None
        for g in np.array(np.meshgrid(*[grid] * nd, indexing="ij")).reshape(nd, -1).T:
            qq = np.zeros(3)
            qq[list(qdims)] = g
            ph, E = best_phases(qq)
            if best is None or E < best[2] - 1e-12:
                best = (qq, ph, E)
        x0 = np.r_[best[0][list(qdims)], best[1]]
        r = minimize(E_of, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-13, maxiter=20000))
        q = np.zeros(3)
        q[list(qdims)] = r.x[:nd]
        ph, E = r.x[nd:], r.fun
        if E > best[2]:
            q, ph, E = best
    q = (q + 0.5) % 1.0 - 0.5
    psi = np.r_[0.0, ph]
    E, dirs = spiral_energy(q, psi, S, sites, pairs, Asite, lat, cart, n)
    return q, psi, dirs, E


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
            K=0.0, spin=None, rmax=6.5, mmin=0.05, nproc=None, n_per=60, nmap=41, sg=None,
            K_axis=None, K_local=False, relax=False, axis=None, spiral=None, spiral_axis=None, qdims=None):
    """bond classes and (if J or exchange given) the LSWT spectrum for any magnetic order.

    J        : list of scalar J per bond class (meV, J > 0 AFM)
    exchange : list of (i, j, R, J) with J a scalar or 3x3 tensor (1-based atom indices)
    K        : single-ion anisotropy −K (S·n)², along K_axis (default: magnetic axis or z),
               or along each spin (K_local=True)
    relax    : minimise the classical energy before LSWT (non-collinear ground states)
    axis     : projection axis for magnon spin / chirality (default: collinear axis, plane
               normal of coplanar orders, net moment of ferromagnets, else z)
    spiral   : None (commensurate, the given cell), "auto" (find the optimal single-Q planar
               spiral) or a q vector in reciprocal-lattice units.  The spiral is treated in the
               rotating frame (J -> J R_n(2π q·R)), so incommensurate q needs NO supercell;
               requires a Hamiltonian invariant under rotations about spiral_axis n
               (Heisenberg, DMI ∥ n, anisotropy along n).
    qdims    : reciprocal directions searched for q (default: those spanned by the bonds)
    """
    norms = np.linalg.norm(mags, axis=1)
    magnetic = [i for i in range(len(frac)) if norms[i] > mmin * max(norms.max(), 1e-9)]
    if not magnetic:
        raise ValueError("no magnetic atoms")
    nz = mags[magnetic]
    dirs0 = nz / np.linalg.norm(nz, axis=1)[:, None]
    _, s_, vt = np.linalg.svd(nz, full_matrices=False)
    s_ = s_ / s_[0]
    kind = "collinear" if (len(s_) < 2 or s_[1] < 1e-3) else ("coplanar" if (len(s_) < 3 or s_[2] < 1e-3)
                                                              else "noncoplanar")

    def canon(v):
        for x in v:
            if abs(x) > 1e-8:
                return v if x > 0 else -v
        return v
    if axis is None:
        net = nz.sum(0)
        if kind == "collinear":
            axis = canon(vt[0])
        elif np.linalg.norm(net) > 0.05 * np.abs(nz).sum() / len(nz):
            axis = net / np.linalg.norm(net)
        elif kind == "coplanar":
            axis = canon(vt[2])
        else:
            axis = np.array([0.0, 0.0, 1.0])
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    S = np.array([spin] * len(magnetic)) if spin else np.linalg.norm(nz, axis=1) / 2.0  # S = |m|/2
    bonds = enumerate_bonds(lat, frac, types, magnetic, rmax)
    classes = bond_classes(lat, frac, types, bonds, rotations, translations)
    loc = {s2: k for k, s2 in enumerate(magnetic)}
    cls_info = []
    for c, g in enumerate(classes):
        b = bonds[g[0]]
        cosang = {round(float(dirs0[loc[bonds[x]["i"]]] @ dirs0[loc[bonds[x]["j"]]]), 3) for x in g}
        kinds = {("same" if c_ > 0.999 else ("opposite" if c_ < -0.999 else f"{math.degrees(math.acos(c_)):.0f}°"))
                 for c_ in cosang}
        cls_info.append(dict(cls=c, L=b["L"], n_bonds=len(g), spins="/".join(sorted(kinds)),
                             rep=(symbols[b["i"]], b["i"] + 1, symbols[b["j"]], b["j"] + 1, b["R"].tolist(),
                                  b["d"].tolist()),
                             members=[(bonds[x]["i"] + 1, bonds[x]["j"] + 1, bonds[x]["R"].tolist()) for x in g]))
    out = dict(kind=kind, axis=axis.tolist(), sites=[m + 1 for m in magnetic], directions=dirs0.tolist(),
               S=S.tolist(), classes=cls_info, bonds=bonds,
               sigma=np.sign(dirs0 @ axis).tolist())
    if J is None and exchange is None:
        return out
    Jexp = None
    if exchange is not None:
        Jexp = {(i - 1, j - 1, tuple(R)): val for (i, j, R, val) in exchange}
        known = {(b["i"], b["j"], tuple(b["R"])) for b in bonds}
        missing = [k for k in Jexp if k not in known and (k[1], k[0], tuple(-np.array(k[2]))) not in known]
        if missing:
            raise ValueError(f"{len(missing)} exchange bond(s) outside rmax or not between magnetic atoms: "
                             f"{[(a + 1, b + 1, R) for a, b, R in missing[:3]]}")
    Jcls = None if J is None else list(J) + [0.0] * (len(classes) - len(J))
    pairs = _pairs_from(bonds, Jcls, Jexp)
    Asite = [None] * len(magnetic)
    if K:
        for a in range(len(magnetic)):
            n_ = dirs0[a] if K_local else (np.asarray(K_axis, float) / np.linalg.norm(K_axis) if K_axis is not None
                                           else (axis if kind == "collinear" else np.array([0.0, 0.0, 1.0])))
            Asite[a] = -K * np.outer(n_, n_)
    dirs = dirs0.copy()
    E0 = classical_energy(dirs, S, magnetic, pairs, Asite)
    if spiral is not None:
        cart = np.array([lat.T @ frac[m] for m in magnetic])
        n_sp = np.asarray(spiral_axis, float) if spiral_axis is not None else spiral_axis_guess(pairs)
        n_sp = n_sp / np.linalg.norm(n_sp)
        bad = [k for k, (*_, J) in enumerate(pairs) if not rotation_invariant(J, n_sp)]
        bad += [f"A{k}" for k, A_ in enumerate(Asite) if not rotation_invariant(A_, n_sp)]
        if bad:
            raise ValueError(f"spiral mode needs a Hamiltonian invariant under rotations about n = {n_sp}; "
                             f"{len(bad)} terms are not (choose --spiral-axis along the DM vectors / anisotropy)")
        if qdims is None:
            span = np.array([np.linalg.solve(lat.T, d) for *_, d, _J in [(p[0], p[1], p[2], p[3]) for p in pairs]])
            qdims = tuple(k for k in range(3) if span.size and np.abs(span[:, k]).max() > 1e-6) or (0, 1, 2)
        if isinstance(spiral, str):
            qf, psi, dirs, Esp = find_spiral(S, magnetic, pairs, Asite, lat, cart, n_sp, qdims=qdims)
        else:
            qf, psi, dirs, Esp = find_spiral(S, magnetic, pairs, Asite, lat, cart, n_sp, qfrac=spiral)
        qcart = qf @ (2 * np.pi * np.linalg.inv(lat).T)
        out.update(spiral=dict(q_rlu=qf.tolist(), q_cart=qcart.tolist(), axis=n_sp.tolist(),
                               period_A=float(2 * np.pi / np.linalg.norm(qcart)) if np.linalg.norm(qcart) > 1e-9
                               else float("inf"), E_spiral=float(Esp), E_start=float(E0),
                               phases_deg=np.degrees(psi).tolist()))
        pairs = spiral_pairs(pairs, lat, cart, magnetic, qf, n_sp)
        axis = n_sp
        out["axis"] = axis.tolist()
        out.update(relaxed=False, E_classical=float(Esp))
        relax = False
    if relax:
        dirs, E1, nit = relax_spins(dirs, S, magnetic, pairs, Asite)
        out.update(relaxed=True, E_classical_start=E0, E_classical=E1, relax_iterations=nit)
        # does an incommensurate single-Q spiral beat the best state that fits this cell?
        try:
            n_sp = spiral_axis_guess(pairs)
            if all(rotation_invariant(J_, n_sp) for *_, J_ in pairs) and \
                    all(rotation_invariant(A_, n_sp) for A_ in Asite):
                cart = np.array([lat.T @ frac[m] for m in magnetic])
                span = np.array([np.linalg.solve(lat.T, p_[2]) for p_ in pairs])
                qd = tuple(k for k in range(3) if span.size and np.abs(span[:, k]).max() > 1e-6) or (0,)
                qf, _, _, Esp = find_spiral(S, magnetic, pairs, Asite, lat, cart, n_sp, qdims=qd,
                                            ngrid=21 if len(qd) == 1 else 11)
                out["spiral_check"] = dict(q_rlu=qf.tolist(), E_spiral=float(Esp), axis=n_sp.tolist(),
                                           lower=bool(Esp < E1 - 1e-6 * max(1.0, abs(E1))))
        except Exception:  # noqa: BLE001
            pass
        if K_local:
            Asite = [-K * np.outer(e, e) if K else None for e in dirs]
    elif spiral is None:
        out.update(relaxed=False, E_classical=E0)
    out["directions"] = dirs.tolist()
    out["net_moment"] = (dirs * (2 * S)[:, None]).sum(0).tolist()
    if spiral is not None and np.linalg.norm(out["spiral"]["q_rlu"]) > 1e-9:
        out["net_moment"] = [0.0, 0.0, 0.0]          # averages to zero over the spiral
    Uf, Vf = local_frames(dirs)
    qs, xs, ticks = make_path(lat, DEFAULT_PATH, n_per)
    om, proj, dS = spectrum(qs, magnetic, Uf, Vf, S, pairs, Asite, axis, nproc)
    omm, _, _ = spectrum(-qs, magnetic, Uf, Vf, S, pairs, Asite, axis, nproc)
    dw = chirality_split(om, proj)
    Bm = 2 * np.pi * np.linalg.inv(lat).T
    f = np.linspace(-0.5, 0.5, nmap)
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    Q = (F1.ravel()[:, None] * Bm[0][None, :] + F2.ravel()[:, None] * Bm[1][None, :])
    om2, pr2, _ = spectrum(Q, magnetic, Uf, Vf, S, pairs, Asite, axis, nproc)
    dwm = chirality_split(om2, pr2).reshape(F1.shape)
    nonrec = float(np.nanmax(np.abs(om - omm))) if np.isfinite(om).any() else float("nan")
    out.update(J_classes=Jcls, n_pairs=len(pairs), q=qs, x=xs, ticks=ticks, omega=om, dSz=proj, magnon_spin=dS,
               split=dw, map_q=Q.reshape(nmap, nmap, 3), map_split=dwm, nonreciprocity=nonrec,
               unstable=bool(np.isnan(om).any()), max_split=float(np.nanmax(np.abs(dwm))) if np.isfinite(dwm).any()
               else 0.0, n_goldstone=int(np.sum(np.abs(om[0]) < 1e-3)) if np.isfinite(om[0]).all() else None)
    if sg is not None:
        from .spingroup import spin_polarisation_form
        out["sym_degree"], out["sym_form"] = spin_polarisation_form(sg)
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
    ax.plot([], [], color=cmap(0.0), lw=2.2, label="α  (ΔS·n = −1)")
    ax.plot([], [], color=cmap(1.0), lw=2.2, label="β  (ΔS·n = +1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title or "magnon bands coloured by chirality")
    if res.get("spiral"):
        ax2.text(0.5, 0.5, "single-Q spiral: modes in the rotating frame", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=9, color="0.4")
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
    """lines:  i j n1 n2 n3 J                      (Heisenberg, meV)
               i j n1 n2 n3 J Dx Dy Dz             (+ Dzyaloshinskii–Moriya vector, D·(S_i×S_j))
               i j n1 n2 n3 Jxx Jxy ... Jzz        (full 3x3 tensor, row major)
       atoms are 1-based in POSCAR order; '#' starts a comment"""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            t = line.split()
            i, j, R = int(t[0]), int(t[1]), [int(t[2]), int(t[3]), int(t[4])]
            v = [float(x) for x in t[5:]]
            if len(v) == 1:
                J = v[0]
            elif len(v) == 4:
                J = v[0] * np.eye(3) + dm_matrix(v[1:])
            elif len(v) == 9:
                J = np.array(v).reshape(3, 3)
            else:
                raise ValueError(f"{path}: expected 1, 4 or 9 exchange values, got {len(v)}: {line}")
            rows.append((i, j, R, J))
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
               "  symphase magnons --exchange J_DM.dat --relax          # non-collinear / canted ground state\n"
               "J > 0 is antiferromagnetic (H = Σ S_i·J_ij·S_j + Σ S_i·A_i·S_i); energies in meV;\n"
               "S = |m|/2 unless --spin.  Collinear and non-collinear orders (moments from OUTCAR/INCAR).")
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
    ap.add_argument("--K-axis", type=float, nargs=3, default=None, help="anisotropy axis (default: spin axis or z)")
    ap.add_argument("--K-local", action="store_true", help="easy axis along each site's spin direction")
    ap.add_argument("--relax", action="store_true", help="minimise the classical energy before LSWT")
    ap.add_argument("--axis", type=float, nargs=3, default=None, help="projection axis for magnon spin")
    ap.add_argument("--spiral", nargs="+", default=None, metavar="auto|q1 q2 q3",
                    help="single-Q spiral in the rotating frame (no supercell): 'auto' finds the optimal q, "
                         "or give q in reciprocal-lattice units")
    ap.add_argument("--spiral-axis", type=float, nargs=3, default=None,
                    help="spiral rotation axis n (default: along the DM vectors, else z)")
    ap.add_argument("--qdims", type=int, nargs="+", default=None, help="reciprocal directions (0 1 2) searched for q")
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
                  nmap=args.nmap, sg=sg, K_axis=args.K_axis, K_local=args.K_local, relax=args.relax,
                  axis=args.axis,
                  spiral=(None if args.spiral is None else ("auto" if args.spiral[0].lower() == "auto"
                                                             else [float(x) for x in args.spiral[:3]])),
                  spiral_axis=args.spiral_axis, qdims=tuple(args.qdims) if args.qdims else None)
    print(f"\n  order: {res['kind']}   magnetic sites: {res['sites']}   S = {[round(x, 3) for x in res['S']]}   "
          f"projection axis = {fmt_vec(res['axis'], 3)}")
    if res["kind"] == "collinear":
        print(f"  σ = {[int(x) for x in res['sigma']]}")
    print(f"  bond classes up to {args.rmax} Å (bonds related by a crystal operation share one J):")
    print("   cls   |d| (Å)   #bonds/cell   spins       representative  i -> j + R            d (Å)")
    for c in res["classes"]:
        rep = c["rep"]
        print(f"   J{c['cls']+1:<3d} {c['L']:8.4f}   {c['n_bonds']:6d}        {c['spins']:<11s} "
              f"{rep[0]}{rep[1]} -> {rep[2]}{rep[3]} + {str(rep[4]):<12s} {fmt_vec(rep[5], 3)}")
    if "omega" not in res:
        print("\n  give one J per class with --J (meV, J>0 AFM) or an --exchange file to compute magnons")
        return res
    if res.get("relaxed"):
        print(f"\n  classical relaxation: E = {res['E_classical_start']:.5f} -> {res['E_classical']:.5f} meV/cell "
              f"({res['relax_iterations']} steps)")
        sc = res.get("spiral_check")
        if sc and sc["lower"]:
            print(f"  NOTE: an incommensurate single-Q spiral q = {np.round(sc['q_rlu'], 5)} r.l.u. has lower energy "
                  f"({sc['E_spiral']:.5f} meV/cell) than any state of this cell -> rerun with --spiral auto")
    if res.get("spiral"):
        sp = res["spiral"]
        print(f"\n  single-Q spiral (rotating frame, magnetic cell kept):  q = {np.round(sp['q_rlu'], 6)} r.l.u.,"
              f"  axis n = {fmt_vec(sp['axis'], 3)}")
        print(f"    period 2π/|q| = {sp['period_A']:.3f} Å;  E_spiral = {sp['E_spiral']:.6f} meV/cell "
              f"(starting configuration {sp['E_start']:.6f});  phases {np.round(sp['phases_deg'], 2)}°")
        print("    magnon energies are given in the rotating frame (neutron intensity at k, k ± q)")
    print("  spin directions used" + (" (cell 0 of the spiral)" if res.get("spiral") else "") + ":")
    for k, (sidx, e) in enumerate(zip(res["sites"], res["directions"])):
        print(f"    {symbols[sidx-1]}{sidx}: {fmt_vec(e, 4)}")
    print(f"  net moment: {fmt_vec(res['net_moment'], 3)} muB/cell")
    if res["unstable"]:
        print("\n  WARNING: H(q) not positive definite somewhere: the order is not a (local) LSWT ground state"
              " -> try --relax or anisotropy")
    print(f"\n  {res['n_pairs']} bonds used;  zero modes at Γ: {res['n_goldstone']};  "
          f"max |Δω| (q3 = 0 plane) = {res["max_split"]:.4f} meV;  non-reciprocity max|ω(q)−ω(−q)| = "
          f"{res['nonreciprocity']:.4f} meV")
    for (xt, lab) in res["ticks"]:
        k = int(np.argmin(np.abs(res["x"] - xt)))
        dws = "n/a" if not np.isfinite(res["split"][k]) else f"{res['split'][k]:+.4f}"
        print(f"    {lab:<10s} ω = {np.round(res['omega'][k], 3)}  ΔS·n = {np.round(res['dSz'][k], 2)}  Δω = {dws}")
    if args.plot and args.plot.lower() != "none":
        stem, ext = os.path.splitext(args.plot)
        ttl = (f"spiral q = {np.round(res['spiral']['q_rlu'], 4).tolist()} (rotating frame)" if res.get("spiral")
               else f"{cls.lower()}: magnon chirality")
        plot_bands(res, save=args.plot, title=ttl)
        if np.isfinite(res["map_split"]).any():
            plot_split_map(res, save=f"{stem}_split_map{ext}")
        else:
            print("  (no chirality splitting map: fewer than two modes of opposite magnon spin)")
        np.savetxt(f"{stem}_bands.dat", np.column_stack([res["x"], res["omega"], res["dSz"], res["split"]]),
                   header="x(1/Å)  omega_1..N (meV)  dSz_1..N  split(meV)")
        print(f"\n  written: {args.plot}" + (f", {stem}_split_map{ext}" if np.isfinite(res["map_split"]).any() else "")
              + f", {stem}_bands.dat")
    return res


if __name__ == "__main__":
    main()
