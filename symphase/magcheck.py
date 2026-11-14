# -*- coding: utf-8 -*-
"""
symphase.magcheck — independent checks of the non-linear spin-wave machinery (symphase.magint)
===============================================================================================

classical_expansion(hf, L)  : the cubic and quartic Holstein–Primakoff terms on an L-cell supercell versus the λ³ and λ⁴
                              Taylor coefficients of the exact classical energy E(λz) of spins of fixed length
                              (S^z' = S − |z|², S⁺ = z √(2S − |z|²)) — any exchange tensor, anisotropy and field.
BosonCluster(hf, ncell, nmax): exact diagonalisation of the bosonic Hamiltonian H₂ + λ H₃ (or H₂ + λ :H₄:) on a periodic
                              cluster of `ncell` cells along a₁, in momentum sectors; second-order energy shifts of
                              one-magnon states are compared with ε + Σ(ε) of MagnonSelfEnergy on the same k grid.
"""

import itertools
import math

import numpy as np


def classical_expansion(hf, se=None, L=(3, 2, 1), seed=7, lam=0.05):
    """returns dict(cubic=(classical, HP), quartic=(classical, HP)) — equal for a correct expansion.
    Note: the quartic HP terms contain exchange only (single-ion anisotropy quartic terms are not included)."""
    from .magint import MagnonSelfEnergy
    se = se or MagnonSelfEnergy(hf)
    M = se.M
    rng = np.random.default_rng(seed)
    cells = [np.array(c) for c in np.ndindex(*L)]
    Z = {(s, tuple(c)): complex(*rng.normal(size=2)) for s in range(M) for c in cells}
    inv = np.linalg.inv(hf.lat)

    def key(s, x):
        f = (np.asarray(x) - se.tau[s]) @ inv
        return s, tuple(np.round(f).astype(int) % np.array(L))

    def spin(s, z):
        Sp = z * np.sqrt(complex(2 * hf.S[s] - abs(z) ** 2))
        return (hf.S[s] - abs(z) ** 2) * hf.V[s] + np.real(Sp * np.conj(hf.U[s]))

    def E(lm):
        sp = {kk: spin(kk[0], lm * Z[kk]) for kk in Z}
        tot = 0.0
        for c in cells:
            R = c @ hf.lat
            for i, j, d, J in hf.pairs:
                a, b = hf.loc[i], hf.loc[j]
                xi = R + se.tau[a]
                tot += sp[key(a, xi)] @ np.asarray(J, float) @ sp[key(b, xi + np.asarray(d, float))]
            for s in range(M):
                Si = sp[key(s, R + se.tau[s])]
                if hf.Asite[s] is not None:
                    tot += Si @ np.asarray(hf.Asite[s], float) @ Si
                if hf.hfield is not None:
                    tot += np.asarray(hf.hfield[s], float) @ Si
        return tot
    lams = np.linspace(-lam, lam, 21)
    co = np.polyfit(lams, [E(x) for x in lams], 8)[::-1]

    def mono(terms):
        return sum(se.classical_terms_value(terms, lambda s, x, R=c @ hf.lat: Z[key(s, R + x)]) for c in cells).real
    return dict(cubic=(co[3], mono(se.terms3)), quartic=(co[4], mono(se.terms4)))


class BosonCluster:
    """periodic cluster of `ncell` magnetic cells along a₁, Fock space with at most `nmax` bosons"""

    def __init__(self, hf, ncell, nmax):
        import scipy.sparse as sp
        self.sp = sp
        self.hf, self.Nc, self.M = hf, ncell, hf.N
        self.nm = ncell * self.M
        self.basis = [s for tot in range(nmax + 1) for s in self._states(self.nm, tot)]
        self.idx = {s: i for i, s in enumerate(self.basis)}
        D = len(self.basis)
        self.a = []
        for m in range(self.nm):
            r, c, v = [], [], []
            for i, s in enumerate(self.basis):
                if s[m] > 0:
                    t = list(s)
                    t[m] -= 1
                    r.append(self.idx[tuple(t)])
                    c.append(i)
                    v.append(math.sqrt(s[m]))
            self.a.append(sp.csr_matrix((v, (r, c)), shape=(D, D)))
        self.ad = [x.T.tocsr() for x in self.a]
        self.inv = np.linalg.inv(hf.lat)
        self.tau = hf.cart

    def kpoints(self):
        """the k grid of the cluster (use it as qs=... / ks=... in MagnonSelfEnergy)"""
        return np.array([2 * np.pi * c / self.Nc * self.inv[:, 0] for c in range(self.Nc)])

    @staticmethod
    def _states(n, tot):
        if n == 1:
            yield (tot,)
            return
        for k in range(tot + 1):
            for rest in BosonCluster._states(n - 1, tot - k):
                yield (k,) + rest

    def _mode(self, s, x):
        f = (np.asarray(x) - self.tau[s]) @ self.inv
        if np.abs(f - np.round(f)).max() > 1e-6 or np.abs(np.round(f[1:])).max() > 0:
            raise ValueError("BosonCluster supports clusters along a₁ only (bonds within the a₁ chain of cells)")
        return int(np.round(f[0])) % self.Nc * self.M + s

    def operator(self, terms):
        """Σ_cells Σ_terms coef × (normal-ordered product of the fields)"""
        H = None
        for c in range(self.Nc):
            R = np.array([c, 0, 0]) @ self.hf.lat
            for coef, fields in terms:
                m = None
                for s, r, dg in [f for f in fields if f[2]] + [f for f in fields if not f[2]]:
                    o = (self.ad if dg else self.a)[self._mode(s, R + np.asarray(r))]
                    m = o if m is None else m @ o
                H = coef * m if H is None else H + coef * m
        return H.tocsr()

    def quadratic(self):
        """the real-space harmonic Hamiltonian (same content as magnons.hamiltonian)"""
        hf, tau = self.hf, self.tau
        U, V, S = hf.U, hf.V, hf.S
        terms = []
        for i, j, d, J in hf.pairs:
            a, b = hf.loc[i], hf.loc[j]
            J = np.asarray(J, float)
            xi, xj = tuple(tau[a]), tuple(tau[a] + np.asarray(d, float))
            r = math.sqrt(S[a] * S[b]) / 2
            c1, c3, dv = r * (U[a] @ J @ np.conj(U[b])), r * (U[a] @ J @ U[b]), V[a] @ J @ V[b]
            terms += [(c1, [(a, xi, True), (b, xj, False)]), (np.conj(c1), [(b, xj, True), (a, xi, False)]),
                      (c3, [(a, xi, True), (b, xj, True)]), (np.conj(c3), [(a, xi, False), (b, xj, False)]),
                      (-S[b] * dv, [(a, xi, True), (a, xi, False)]), (-S[a] * dv, [(b, xj, True), (b, xj, False)])]
        for x in range(self.M):
            xx = tuple(tau[x])
            if hf.Asite[x] is not None:
                Am = np.asarray(hf.Asite[x], float)
                terms += [(S[x] * (np.real(U[x] @ Am @ np.conj(U[x])) - 2 * V[x] @ Am @ V[x]), [(x, xx, True), (x, xx, False)]),
                          (S[x] * (U[x] @ Am @ U[x]) / 2, [(x, xx, True), (x, xx, True)]),
                          (np.conj(S[x] * (U[x] @ Am @ U[x])) / 2, [(x, xx, False), (x, xx, False)])]
            if hf.hfield is not None:
                terms.append((-float(np.asarray(hf.hfield[x]) @ V[x]), [(x, xx, True), (x, xx, False)]))
        return self.operator(terms)

    def hf_terms(self):
        """the Hartree–Fock quadratic operator of hf (after hf.update) — H₄ − H_HF = :H₄: + const"""
        hf, tau = self.hf, self.tau
        out = []
        for (i, j, d, J), val, o1, o2 in hf.hf:
            a, b = hf.loc[i], hf.loc[j]
            pos = {"i": (a, tuple(tau[a])), "j": (b, tuple(tau[a] + np.asarray(d, float)))}
            out.append((val, [pos[o1[0]] + (o1[1],), pos[o2[0]] + (o2[1],)]))
        return out

    def sectors(self):
        sp = self.sp
        D = len(self.basis)

        def shift(s):
            return tuple(s[(c - 1) % self.Nc * self.M + m] for c in range(self.Nc) for m in range(self.M))
        seen, orbits = set(), []
        for s in self.basis:
            if s in seen:
                continue
            orb, t = [s], shift(s)
            while t != s:
                orb.append(t)
                t = shift(t)
            seen.update(orb)
            orbits.append(orb)
        out = []
        for K in range(self.Nc):
            r, c, v, col = [], [], [], 0
            for orb in orbits:
                L = len(orb)
                if (K * L) % self.Nc:
                    continue
                for t, st in enumerate(orb):
                    r.append(self.idx[st])
                    c.append(col)
                    v.append(np.exp(2j * np.pi * K * t / self.Nc) / math.sqrt(L))
                col += 1
            out.append(sp.csr_matrix((v, (r, c)), shape=(D, col)))
        return out

    def second_order(self, H2, V, targets, lams=(0.02, 0.01)):
        """first- and second-order shifts (relative to the ground state) of the levels closest to E0 + target,
        from exact eigenvalues of H2 ± λV (Richardson-extrapolated)"""
        secs = self.sectors()

        def levels(lm):
            return [np.linalg.eigvalsh((B.conj().T @ ((H2 + lm * V) @ B)).toarray()) for B in secs]
        L0 = levels(0.0)
        E0 = L0[0].min()
        Ls = {lm: (levels(lm), levels(-lm)) for lm in lams}
        out = []
        for tg in targets:
            K, i = min(((K, i) for K in range(self.Nc) for i in range(len(L0[K]))),
                       key=lambda t: abs(L0[t[0]][t[1]] - E0 - tg))
            res = []
            for lm in lams:
                Lp, Lm = Ls[lm]
                ep = Lp[K][np.argmin(abs(Lp[K] - L0[K][i]))]
                em = Lm[K][np.argmin(abs(Lm[K] - L0[K][i]))]
                gp, gm = Lp[0].min(), Lm[0].min()
                res.append((((ep - gp) - (em - gm)) / (2 * lm), ((ep + em - 2 * L0[K][i]) - (gp + gm - 2 * E0)) / (2 * lm ** 2)))
            (f1, s1), (f2, s2) = res
            out.append(dict(level=L0[K][i] - E0, first=f2, second=(4 * s2 - s1) / 3))
        return out
