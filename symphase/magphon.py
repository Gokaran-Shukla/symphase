# -*- coding: utf-8 -*-
"""
symphase.magphon — magnon–phonon coupling (magnon polarons)
===========================================================

Spins and lattice are treated as ONE harmonic system.  Canonical pairs per Bloch vector k:

    spins  : (X_a, Y_a)   with  S_a ≈ v_a (S_a − (X²+Y²)/2) + √S_a (X_a x_a + Y_a y_a),  [X, Y] = i
    lattice: (u_jα, P_jα) with  u in Å,  kinetic energy (ħ²/M_j Å²) P²/2 = (4.18016 meV / M[amu]) P²/2

The classical energy  E = Σ_bonds S_i·J_ij(d)·S_j + Σ S·A·S + Σ h·S + ½ u·Φ·u  is expanded to second
order; the bond-length dependence of the exchange,

    J_ij(d0 + δd) = J_ij + Σ_γ G^γ_ij δd_γ,      δd = u_j(R) − u_i(0),      G^γ = ∂J/∂d_γ  (meV/Å),

gives the spin–lattice cross terms ∂²E/∂X_a∂u_jγ = √S_a S_b x_a·G^γ·v_b (± for the two ends of a bond).
The Bloch Hessian 𝓗(k) (positions X, u | momenta Y, P) is diagonalised symplectically:
𝓗 = K†K, eig(K iJ K†) = ±ω_n(k)  (as in Colpa's method); the eigenvectors give the magnon weight
(symplectic norm on the spin pairs) of every hybrid mode.

Physics built in
  * isotropic exchange striction (G^γ = dJ/dr r̂_γ 𝟙) couples linearly ONLY to non-collinear spins
    (x_a·v_b = 0 for collinear order) -> no magnon–phonon hybridisation in a collinear Heisenberg magnet;
  * anisotropic / DMI derivatives (magnetoelastic B-type terms) couple collinear magnets too
    (Kittel's magnetoelastic mechanism) -> anticrossing gaps ("magnon polarons").

Phonons: Born–von Kármán springs per bond (k_L, k_T in meV/Å²) or phonopy force constants
(phonopy_params.yaml / phonopy.yaml with FORCE_CONSTANTS, same primitive cell and atom order).
"""

import math
import os

import numpy as np

from .magnons import local_frames, make_path, DEFAULT_PATH, _bond_R, enumerate_bonds

HB2_AMU = 4.18016          # ħ² / (1 amu · 1 Å²) in meV


# ----------------------------------------------------------------------------
#  inputs
# ----------------------------------------------------------------------------
def atomic_masses(symbols):
    try:
        from pymatgen.core import Element
        return np.array([float(Element(s).atomic_mass) for s in symbols])
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"atomic masses: {err}")


def read_springs(path):
    """lines: i j n1 n2 n3 kL kT   (1-based atoms, meV/Å²) -> list of (i, j, R, kL, kT)"""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if line:
                t = line.split()
                rows.append((int(t[0]) - 1, int(t[1]) - 1, [int(t[2]), int(t[3]), int(t[4])], float(t[5]),
                             float(t[6]) if len(t) > 6 else 0.0))
    return rows


def read_magnetoelastic(path):
    """lines:  i j n1 n2 n3 dJdr                     isotropic exchange striction (meV/Å)
               i j n1 n2 n3 x|y|z G11 ... G33        ∂J/∂d_γ, full 3x3 tensor (meV/Å)
       several lines for the same bond add up.  Returns {(i, j, R): (3, 3, 3) array G[γ]} or scalars."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            t = line.split()
            key = (int(t[0]) - 1, int(t[1]) - 1, (int(t[2]), int(t[3]), int(t[4])))
            if len(t) == 6:
                out.setdefault(key, []).append(("iso", float(t[5])))
            elif len(t) == 15 and t[5].lower() in "xyz":
                out.setdefault(key, []).append(("xyz".index(t[5].lower()), np.array(t[6:], float).reshape(3, 3)))
            else:
                raise ValueError(f"{path}: expected 'i j n1 n2 n3 dJdr' or 'i j n1 n2 n3 x|y|z 9 values': {line}")
    return out


def _G_of_bond(spec, d):
    """(3, 3, 3) array G[γ] from a list of ('iso', dJdr) / (γ, 3x3) specs; d = bond vector"""
    G = np.zeros((3, 3, 3))
    rhat = np.asarray(d, float) / np.linalg.norm(d)
    for kind, val in spec:
        if kind == "iso":
            for g in range(3):
                G[g] += val * rhat[g] * np.eye(3)
        else:
            G[kind] += val
    return G


def spring_fc(lat, frac, springs):
    """Born–von Kármán bonds -> list of (i, j, d, K 3x3) with E = ½ (u_j − u_i)·K·(u_j − u_i)"""
    out = []
    for i, j, R, kL, kT in springs:
        d = lat.T @ (np.asarray(frac[j]) + np.asarray(R) - np.asarray(frac[i]))
        r = d / np.linalg.norm(d)
        out.append((i, j, d, kL * np.outer(r, r) + kT * (np.eye(3) - np.outer(r, r))))
    return out


class PhonopyFC:
    """force constants from phonopy (needs the same primitive cell / atom order as POSCAR)"""

    def __init__(self, path, lat, frac):
        import phonopy
        self.ph = phonopy.load(path)
        prim = self.ph.primitive
        if len(prim.scaled_positions) != len(frac):
            raise ValueError(f"{path}: phonopy primitive cell has {len(prim.scaled_positions)} atoms, POSCAR {len(frac)}")
        dpos = np.asarray(prim.scaled_positions) - np.asarray(frac)
        if np.abs(dpos - np.round(dpos)).max() > 1e-3 or np.abs(np.asarray(prim.cell) - lat).max() > 1e-3:
            raise ValueError(f"{path}: phonopy primitive cell / atom order differ from POSCAR")
        self.masses = np.asarray(prim.masses, float)
        self.lat = lat

    def phi_k(self, kcart):
        """Φ(k) (3n x 3n, meV/Å²) in the atom-position phase convention"""
        qf = self.lat @ kcart / (2 * np.pi)                      # reduced coordinates
        dm = self.ph.dynamical_matrix
        dm.run(qf)
        D = np.asarray(dm.dynamical_matrix)                      # eV/Å²/amu
        sm = np.sqrt(np.repeat(self.masses, 3))
        return D * np.outer(sm, sm) * 1000.0


# ----------------------------------------------------------------------------
#  Bloch Hessian and symplectic diagonalisation
# ----------------------------------------------------------------------------
class MagnonPhonon:
    """coupled spin–lattice harmonic model

    lat, frac, symbols : primitive cell
    sites              : indices of magnetic atoms;  dirs, S: ground-state directions / spin lengths
    pairs              : magnetic bonds (i, j, d, J 3x3), atom indices
    Asite, hfield      : anisotropy matrices / Zeeman vectors per magnetic site (or None)
    me                 : {(i, j, R): spec} magnetoelastic derivatives (read_magnetoelastic or from dJdr per class)
    springs / phonopy  : Born–von Kármán list from spring_fc, or a PhonopyFC
    """

    def __init__(self, lat, frac, symbols, sites, dirs, S, pairs, Asite=None, hfield=None, me=None,
                 springs=None, phonopy=None, masses=None):
        self.lat, self.frac = np.asarray(lat, float), np.asarray(frac, float)
        self.n = len(frac)
        self.sites = list(sites)
        self.N = len(self.sites)
        self.loc = {s: k for k, s in enumerate(self.sites)}
        self.S = np.asarray(S, float)
        U, V = local_frames(dirs)
        self.x, self.y, self.v = np.real(U), np.imag(U), np.asarray(V, float)
        self.pairs = pairs
        self.Asite = Asite or [None] * self.N
        self.hfield = hfield
        self.springs = springs or []
        self.phonopy = phonopy
        self.masses = (phonopy.masses if phonopy is not None else
                       (np.asarray(masses, float) if masses is not None else atomic_masses(symbols)))
        cart = self.lat.T @ self.frac.T
        self.cart = cart.T
        # magnetoelastic G per magnetic bond
        self.G = []
        me = me or {}
        for i, j, d, J in pairs:
            R = tuple(_bond_R(self.lat, d, self.cart[i], self.cart[j]))
            spec = me.get((i, j, R))
            if spec is not None:
                self.G.append(_G_of_bond(spec, d))
                continue
            rev = me.get((j, i, tuple(-np.array(R))))
            if rev is not None:                     # given for j -> i (vector −d, J' = J^T): G = −(G')^T
                self.G.append(-np.transpose(_G_of_bond(rev, -np.asarray(d)), (0, 2, 1)))
            else:
                self.G.append(None)
        self.M = self.N + 3 * self.n

    # ---------------------------------------------------------------- Hessian
    def hessian(self, k, coupling=True, magnons=True, phonons=True):
        """Bloch Hessian 𝓗(k) in the order (X_1..X_N, u_1x..u_nz | Y_1..Y_N, P_1x..P_nz)"""
        N, n, M = self.N, self.n, self.M
        H = np.zeros((2 * M, 2 * M), complex)
        iX = lambda a: a                      # noqa: E731
        iY = lambda a: M + a                  # noqa: E731
        iu = lambda j, g: N + 3 * j + g       # noqa: E731
        iP = lambda j, g: M + N + 3 * j + g   # noqa: E731
        S, x, y, v = self.S, self.x, self.y, self.v
        k = np.asarray(k, float)
        if magnons:
            for (i, j, d, J), G in zip(self.pairs, self.G):
                a, b = self.loc[i], self.loc[j]
                ph = np.exp(1j * k @ d)
                r = math.sqrt(S[a] * S[b])
                for (pa, ea) in ((iX(a), x[a]), (iY(a), y[a])):
                    for (pb, eb) in ((iX(b), x[b]), (iY(b), y[b])):
                        val = r * (ea @ J @ eb)
                        H[pa, pb] += val * ph
                        H[pb, pa] += val * np.conj(ph)
                dv = v[a] @ J @ v[b]
                for p in (iX(a), iY(a)):
                    H[p, p] += -S[b] * dv
                for p in (iX(b), iY(b)):
                    H[p, p] += -S[a] * dv
            for a, A in enumerate(self.Asite):
                if A is None:
                    continue
                As = 0.5 * (A + A.T)
                vAv = v[a] @ As @ v[a]
                H[iX(a), iX(a)] += 2 * S[a] * (x[a] @ As @ x[a]) - 2 * S[a] * vAv
                H[iY(a), iY(a)] += 2 * S[a] * (y[a] @ As @ y[a]) - 2 * S[a] * vAv
                c = 2 * S[a] * (x[a] @ As @ y[a])
                H[iX(a), iY(a)] += c
                H[iY(a), iX(a)] += c
            if self.hfield is not None:
                for a in range(N):
                    hv = float(self.hfield[a] @ v[a])
                    H[iX(a), iX(a)] += -hv
                    H[iY(a), iY(a)] += -hv
        if phonons:
            if self.phonopy is not None:
                Phi = self.phonopy.phi_k(k)
                H[N:N + 3 * n, N:N + 3 * n] += Phi
            for i, j, d, K in self.springs:
                ph = np.exp(1j * k @ d)
                bi, bj = slice(N + 3 * i, N + 3 * i + 3), slice(N + 3 * j, N + 3 * j + 3)
                H[bi, bi] += K
                H[bj, bj] += K
                H[bi, bj] += -K * ph
                H[bj, bi] += -K.T * np.conj(ph)
            for j in range(n):
                for g in range(3):
                    H[iP(j, g), iP(j, g)] += HB2_AMU / self.masses[j]
        if coupling:
            for (i, j, d, J), G in zip(self.pairs, self.G):
                if G is None:
                    continue
                a, b = self.loc[i], self.loc[j]
                ph = np.exp(1j * k @ d)
                for g in range(3):
                    Gg = G[g]
                    for (pa, ea) in ((iX(a), x[a]), (iY(a), y[a])):
                        c = math.sqrt(S[a]) * S[b] * (ea @ Gg @ v[b])       # spin a  ×  δd = u_j(R) − u_i(0)
                        self._add(H, pa, iu(j, g), c * ph)
                        self._add(H, pa, iu(i, g), -c)
                    for (pb, eb) in ((iX(b), x[b]), (iY(b), y[b])):
                        c = S[a] * math.sqrt(S[b]) * (v[a] @ Gg @ eb)       # spin b (cell R)
                        self._add(H, pb, iu(j, g), c)
                        self._add(H, pb, iu(i, g), -c * np.conj(ph))
        return 0.5 * (H + H.conj().T)

    @staticmethod
    def _add(H, p, q, val):
        H[p, q] += val
        H[q, p] += np.conj(val)

    def modes(self, k, coupling=True, eps=1e-9):
        """frequencies ω_n(k) (meV, ascending) and magnon weight of every mode"""
        H = self.hessian(k, coupling=coupling)
        M, N = self.M, self.N
        # canonical rescaling u -> u/s, P -> s P per atom (balances the blocks, keeps ω and weights)
        t = np.ones(2 * M)
        for j in range(self.n):
            phi = np.real(np.trace(H[N + 3 * j:N + 3 * j + 3, N + 3 * j:N + 3 * j + 3])) / 3.0
            kap = HB2_AMU / self.masses[j]
            sj = (max(phi, 1e-12) / kap) ** 0.25
            t[N + 3 * j:N + 3 * j + 3] = 1.0 / sj
            t[M + N + 3 * j:M + N + 3 * j + 3] = sj
        H = H * np.outer(t, t)
        J = np.block([[np.zeros((M, M)), np.eye(M)], [-np.eye(M), np.zeros((M, M))]])
        scale = max(1.0, np.abs(H).max())
        Kc = None
        for e in (0.0, eps, 1e-8, 1e-7, 1e-6, 1e-5):
            try:
                Kc = np.linalg.cholesky(H + e * scale * np.eye(2 * M)).conj().T
                break
            except np.linalg.LinAlgError:
                continue
        if Kc is None:
            return np.full(M, np.nan), np.full(M, np.nan)
        W = Kc @ (1j * J) @ Kc.conj().T
        w, Z = np.linalg.eigh(0.5 * (W + W.conj().T))
        pos = np.argsort(-w)[:M]
        om = w[pos]
        Xi = (1j * J) @ Kc.conj().T @ Z[:, pos]                   # ξ = iJ K† z
        q, p = Xi[:M], Xi[M:]
        c = -2 * np.imag(np.conj(q) * p)                          # symplectic norm per canonical pair
        tot = c.sum(0)
        wm = np.clip(np.abs(c[:self.N].sum(0) / np.where(np.abs(tot) > 1e-30, tot, 1.0)), 0, 1)
        o = np.argsort(om)
        return om[o], wm[o]

    def bands(self, qs, coupling=True):
        res = [self.modes(q, coupling) for q in qs]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])

    def anticrossings(self, qs, xs, om, wm, om0, wm0, wmin=0.25):
        """hybridisation gaps: where a coupled mode has magnon weight in [wmin, 1-wmin], the gap to the
        nearest mode of complementary character (meV); merged per crossing region"""
        out = []
        for k in range(len(qs)):
            for n in range(om.shape[1]):
                if wmin <= wm[k, n] <= 1 - wmin:
                    others = [m for m in range(om.shape[1]) if m != n and wmin <= wm[k, m] <= 1 - wmin]
                    if others:
                        m = min(others, key=lambda m_: abs(om[k, m_] - om[k, n]))
                        out.append((xs[k], float(min(om[k, n], om[k, m])), float(abs(om[k, m] - om[k, n]))))
        if not out:
            return []
        out.sort()
        groups, cur = [], [out[0]]
        for o in out[1:]:
            if o[0] - cur[-1][0] < 3 * (xs[1] - xs[0] + 1e-12) and abs(o[1] - cur[-1][1]) < 2.0:
                cur.append(o)
            else:
                groups.append(cur)
                cur = [o]
        groups.append(cur)
        return [dict(x=g[int(np.argmin([t[2] for t in g]))][0], omega=g[int(np.argmin([t[2] for t in g]))][1],
                     gap=min(t[2] for t in g)) for g in groups]


# ----------------------------------------------------------------------------
#  driver
# ----------------------------------------------------------------------------
def auto_path(lat, bond_vectors):
    """path through the directions spanned by the bonds: chain Γ–X, hexagonal plane Γ–M–K–Γ,
    other planes Γ–X–M–Γ, 3D: the default path"""
    lat = np.asarray(lat, float)
    span = np.array([np.linalg.solve(lat.T, d) for d in bond_vectors]) if len(bond_vectors) else np.zeros((0, 3))
    dims = [k for k in range(3) if span.size and np.abs(span[:, k]).max() > 1e-6]
    if len(dims) == 1:
        e = np.zeros(3); e[dims[0]] = 0.5
        return [("Γ", (0, 0, 0)), ("X", tuple(e))]
    if len(dims) == 2:
        a, b = dims
        cosg = lat[a] @ lat[b] / np.linalg.norm(lat[a]) / np.linalg.norm(lat[b])
        e1, e2 = np.zeros(3), np.zeros(3)
        if abs(abs(cosg) - 0.5) < 1e-3:                     # hexagonal: K is a corner, |K| = 2|M|/√3
            B = 2 * np.pi * np.linalg.inv(lat).T
            e1[a] = 0.5
            target = 2 * np.linalg.norm(e1 @ B) / math.sqrt(3)
            for c1, c2 in ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, -1 / 3), (1 / 3, -1 / 3)):
                e2 = np.zeros(3)
                e2[a], e2[b] = c1, c2
                if abs(np.linalg.norm(e2 @ B) - target) < 1e-6 * target:
                    break
            return [("Γ", (0, 0, 0)), ("M", tuple(e1)), ("K", tuple(e2)), ("Γ", (0, 0, 0))]
        e1[a] = 0.5
        e2[a] = e2[b] = 0.5
        return [("Γ", (0, 0, 0)), ("X", tuple(e1)), ("M", tuple(e2)), ("Γ", (0, 0, 0))]
    return DEFAULT_PATH
def analyse_mp(lat, frac, types, symbols, mags, rotations, translations, springs=None, phonopy=None,
               magnetoelastic=None, dJdr=None, n_per=80, path=None, relax=False, masses=None, **model_kw):
    """magnon–phonon spectrum along a path.  model_kw -> magnons.analyse (J, exchange, K, field, spin, rmax ...)
    dJdr: isotropic exchange striction per bond class (meV/Å, same order as J1, J2, ...)"""
    from .magnons import analyse, relax_spins
    m = analyse(lat, frac, types, symbols, mags, rotations, translations, return_model=True, **model_kw)
    pairs, Asite, hf, sites, S, dirs = m["pairs"], m["Asite"], m["hfield"], m["magnetic"], m["S_arr"], m["dirs0"]
    if relax:
        dirs, _, _ = relax_spins(dirs, S, sites, pairs, Asite, hfield=hf)
    me = dict(magnetoelastic or {})
    if dJdr is not None:
        bonds = m["bonds"]
        for c, info in enumerate(m["classes"]):
            if c < len(dJdr) and dJdr[c]:
                for (i1, j1, R) in info["members"]:
                    me.setdefault((i1 - 1, j1 - 1, tuple(R)), []).append(("iso", float(dJdr[c])))
        del bonds
    sp = spring_fc(np.asarray(lat, float), np.asarray(frac, float), springs) if springs else None
    mp = MagnonPhonon(lat, frac, symbols, sites, dirs, S, pairs, Asite, hf, me, springs=sp, phonopy=phonopy,
                      masses=masses)
    allvec = [d for _i, _j, d, _J in pairs] + ([d for _i, _j, d, _K in sp] if sp else [])
    qs, xs, ticks = make_path(np.asarray(lat, float), path or auto_path(lat, allvec), n_per)
    om, wm = mp.bands(qs, coupling=True)
    om0, wm0 = mp.bands(qs, coupling=False)
    gaps = mp.anticrossings(qs, xs, om, wm, om0, wm0)
    return dict(model=mp, q=qs, x=xs, ticks=ticks, omega=om, w_magnon=wm, omega0=om0, w_magnon0=wm0,
                gaps=gaps, n_magnon=mp.N, n_phonon=3 * mp.n, dirs=dirs,
                max_gap=max([g["gap"] for g in gaps], default=0.0),
                coupled_bonds=sum(G is not None for G in mp.G))


def plot_mp(res, save=None, title=None, figsize=(7.6, 5.0), ylim=None):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from . import hull_plots  # noqa: F401
    fig, ax = plt.subplots(figsize=figsize)
    x, om, wm = res["x"], res["omega"], res["w_magnon"]
    for n in range(res["omega0"].shape[1]):
        ax.plot(x, res["omega0"][:, n], color="0.6", lw=0.9, ls="--", zorder=1)
    cmap = matplotlib.colormaps["coolwarm"]
    for n in range(om.shape[1]):
        pts = np.array([x, om[:, n]]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=matplotlib.colors.Normalize(0, 1), linewidths=2.4, zorder=2)
        lc.set_array(0.5 * (wm[:-1, n] + wm[1:, n]))
        ax.add_collection(lc)
    for t, _ in res["ticks"]:
        ax.axvline(t, color="0.8", lw=0.7, zorder=0)
    ax.set_xticks([t for t, _ in res["ticks"]])
    ax.set_xticklabels([lab for _, lab in res["ticks"]])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*(ylim or (0, np.nanmax(om) * 1.05)))
    ax.set_ylabel("energy (meV)")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.015)
    cb.set_label("magnon weight (0 = phonon, 1 = magnon)")
    ax.plot([], [], color="0.6", ls="--", lw=0.9, label="uncoupled")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title or f"magnon–phonon bands (max. anticrossing gap {res['max_gap']:.3f} meV)", fontsize=11)
    if save:
        fig.savefig(save)
    return fig


def bond_springs(lat, frac, types, rmax, kL, kT=0.0, rmin=0.0):
    """Born–von Kármán springs (kL, kT in meV/Å²) on every bond rmin < |d| ≤ rmax (all atoms)"""
    bonds = enumerate_bonds(np.asarray(lat, float), np.asarray(frac, float), types, list(range(len(frac))), rmax)
    return [(b["i"], b["j"], list(b["R"]), kL, kT) for b in bonds if b["L"] > rmin]
