# -*- coding: utf-8 -*-
"""
symphase.magint — magnon–magnon (four-magnon) interactions: Hartree–Fock renormalised spin waves
===============================================================================================

Holstein–Primakoff to quartic order in local frames,
    S⁺ = √(2S) (a − a†aa/4S),   S⁻ = √(2S) (a† − a†a†a/4S),   Sᶻ' = S − a†a,
gives for every bond S_i·J·S_j the quartic terms
    (v_i·J·v_j) n_i n_j  −  Σ_{αβ=±} (e_i^α·J·e_j^β) √(S_iS_j)/8 · [ L_i^α C_j^β / S_j + C_i^α L_j^β / S_i ],
    e⁺ = u*, e⁻ = u,  L⁺ = a, L⁻ = a†,  C⁺ = a†aa, C⁻ = a†a†a.
They are decoupled with Wick's theorem (Hartree–Fock): O1O2O3O4 → Σ_pairings ⟨O_aO_b⟩ O_cO_d, using the
thermal correlations ⟨a†a⟩, ⟨aa⟩ of the current spin-wave Hamiltonian (BZ sums).  The resulting quadratic
terms are added to the bosonic BdG matrix:
    order = 1  ('first')        : correlations of linear spin-wave theory -> the 1/S correction
                                  (Oguchi's zero-point renormalisation, Dyson's T-dependence);
    order = 'sc'                : iterate to self-consistency (self-consistent spin-wave theory).
Exact to order 1/S for collinear magnets.  For non-collinear orders the cubic terms (same order in 1/S at
second order) are not included; single-ion anisotropy and Zeeman terms enter at harmonic level only.
"""

import math

import numpy as np

from .magnons import colpa, hamiltonian, local_frames

KB_MEV = 0.08617333262


def _bose(e, T):
    if T <= 0:
        return np.zeros_like(e)
    with np.errstate(over="ignore", divide="ignore"):
        return 1.0 / np.expm1(np.clip(e / (KB_MEV * T), 1e-12, 700))


class FourMagnonHF:
    """lat, positions (Cartesian, of the magnetic sites, in `sites` order), sites, dirs, S, pairs (i, j, d, J), Asite,
    hfield — as in magnons.analyse(..., return_model=True)"""

    def __init__(self, lat, cart, sites, dirs, S, pairs, Asite=None, hfield=None):
        self.lat = np.asarray(lat, float)
        self.cart = np.asarray(cart, float)
        self.sites = list(sites)
        self.loc = {s: k for k, s in enumerate(self.sites)}
        self.N = len(self.sites)
        self.S = np.asarray(S, float)
        self.U, self.V = local_frames(dirs)
        self.pairs = pairs
        self.Asite = Asite if Asite is not None else [None] * self.N
        self.hfield = hfield
        _, sv, _ = np.linalg.svd(self.V)
        self.collinear = bool(len(sv) < 2 or sv[1] < 1e-6 * sv[0])
        self.terms = [self._bond_terms(p) for p in pairs]
        self.hf = None                                       # list of quadratic real-space terms

    @classmethod
    def from_model(cls, lat, frac, model, dirs=None):
        mag = model["magnetic"]
        cart = np.array([np.asarray(lat).T @ np.asarray(frac)[m] for m in mag])
        return cls(lat, cart, mag, model["dirs0"] if dirs is None else dirs, model["S_arr"], model["pairs"],
                   model["Asite"], model["hfield"])

    # ---------------------------------------------------------------- quartic terms of one bond
    def _bond_terms(self, pair):
        i, j, d, J = pair
        a, b = self.loc[i], self.loc[j]
        Si, Sj = self.S[a], self.S[b]
        u, v = self.U, self.V
        e = {("i", "+"): np.conj(u[a]), ("i", "-"): u[a], ("j", "+"): np.conj(u[b]), ("j", "-"): u[b]}
        L = {"+": [("F",)], "-": [("T",)]}
        C = {"+": [("T",), ("F",), ("F",)], "-": [("T",), ("T",), ("F",)]}
        out = []
        czz = float(v[a] @ J @ v[b])
        if czz:
            out.append((czz, [("i", True), ("i", False), ("j", True), ("j", False)]))
        for al in "+-":
            for be in "+-":
                w = e[("i", al)] @ J @ e[("j", be)]
                if abs(w) < 1e-14:
                    continue
                r = math.sqrt(Si * Sj)
                ops_LC = [("i", o[0] == "T") for o in L[al]] + [("j", o[0] == "T") for o in C[be]]
                ops_CL = [("i", o[0] == "T") for o in C[al]] + [("j", o[0] == "T") for o in L[be]]
                out.append((-w * r / (8 * Sj), ops_LC))
                out.append((-w * r / (8 * Si), ops_CL))
        return out

    # ---------------------------------------------------------------- spectra and correlations
    def _H(self, k):
        H = hamiltonian(np.asarray(k, float), self.sites, self.U, self.V, self.S, self.pairs, self.Asite, self.hfield)
        if self.hf is not None:
            H = H + self._H_hf(k)
        return H

    def modes(self, k):
        om, T = colpa(self._H(k))
        if om is None:
            raise ValueError(f"spin-wave Hamiltonian not positive definite at k = {np.round(k, 4)}"
                             + (" — the Hartree–Fock renormalisation destroys the order (fluctuations too strong: "
                                "low dimension / high T / small S); no self-consistent spin-wave solution"
                                if self.hf is not None else ""))
        o = np.argsort(om)
        return om[o], T[:, o]

    def kgrid(self, n, dims=None):
        n = list(n) + [1] * (3 - len(n))
        f = [((np.arange(m) + 0.5) / m) if m > 1 else np.zeros(1) for m in n]   # shifted (avoids Γ zero modes)
        F = np.array(np.meshgrid(*f, indexing="ij")).reshape(3, -1).T
        return F @ (2 * np.pi * np.linalg.inv(self.lat).T)

    def correlations(self, ks, T=0.0):
        """⟨Ψ_k Ψ_k†⟩ for every k: (nk, 2N, 2N)"""
        N = self.N
        out = []
        for k in ks:
            ep, Pp = self.modes(k)
            em, Pm = self.modes(-np.asarray(k))
            Tf = np.zeros((2 * N, 2 * N), complex)
            Tf[:N, :N], Tf[N:, :N] = Pp[:N], Pp[N:]
            Tf[:N, N:], Tf[N:, N:] = np.conj(Pm[N:]), np.conj(Pm[:N])
            occ = np.r_[1 + _bose(ep, T), _bose(em, T)]
            out.append((Tf * occ[None, :]) @ Tf.conj().T)
        return np.array(out)

    def _bond_expect(self, Cks, ks, pair):
        """⟨a†_p a_q⟩ (hop) and ⟨a_p a_q⟩ (pair) for p, q ∈ {i, j} of one bond"""
        i, j, d, J = pair
        a, b = self.loc[i], self.loc[j]
        N = self.N
        ph = np.exp(1j * ks @ d)
        C11, C12 = Cks[:, :N, :N], Cks[:, :N, N:]
        hop = {("i", "i"): np.mean(C11[:, a, a] - 1).item(), ("j", "j"): np.mean(C11[:, b, b] - 1).item()}
        hop[("i", "j")] = np.mean(ph * (C11[:, b, a] - (1.0 if a == b else 0.0))).item()
        hop[("j", "i")] = np.conj(hop[("i", "j")])
        pr = {("i", "i"): np.mean(C12[:, a, a]).item(), ("j", "j"): np.mean(C12[:, b, b]).item()}
        pr[("i", "j")] = np.mean(np.conj(ph) * C12[:, a, b]).item()
        pr[("j", "i")] = pr[("i", "j")]
        return hop, pr

    @staticmethod
    def _ev(o1, o2, hop, pr):
        (p, d1), (q, d2) = o1, o2
        if d1 and not d2:
            return hop[(p, q)]
        if not d1 and d2:
            return hop[(q, p)] + (1.0 if p == q else 0.0)
        if not d1 and not d2:
            return pr[(p, q)]
        return np.conj(pr[(q, p)])

    def update(self, ks, T=0.0, mix=1.0):
        """one Hartree–Fock step: correlations of the current Hamiltonian -> new quadratic terms"""
        Cks = self.correlations(ks, T)
        new = []
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        for pair, terms in zip(self.pairs, self.terms):
            hop, pr = self._bond_expect(Cks, ks, pair)
            for coef, ops in terms:
                for (x, y), (z, w) in pairings:
                    for (c1, c2), (r1, r2) in (((x, y), (z, w)), ((z, w), (x, y))):
                        val = coef * self._ev(ops[c1], ops[c2], hop, pr)
                        if abs(val) > 1e-15:
                            new.append((pair, val, ops[r1], ops[r2]))
        if self.hf is None or mix >= 1.0:
            self.hf = new
        else:
            self.hf = [(p, (1 - mix) * 0 + mix * v, o1, o2) for (p, v, o1, o2) in new]   # simple replacement
        return Cks

    def _H_hf(self, k):
        N = self.N

        def block(kk):
            A = np.zeros((N, N), complex)
            B = np.zeros((N, N), complex)
            for (i, j, d, J), val, o1, o2 in self.hf:
                site = {"i": self.loc[i], "j": self.loc[j]}
                pos = {"i": np.zeros(3), "j": np.asarray(d, float)}
                (p, d1), (q, d2) = o1, o2
                if d1 and not d2:
                    A[site[p], site[q]] += val * np.exp(1j * kk @ (pos[q] - pos[p]))
                elif d2 and not d1:
                    A[site[q], site[p]] += val * np.exp(1j * kk @ (pos[p] - pos[q]))
                elif d1 and d2:
                    ph = np.exp(1j * kk @ (pos[q] - pos[p]))
                    B[site[p], site[q]] += val * ph
                    B[site[q], site[p]] += val * np.conj(ph)
            return A, B
        A, B = block(np.asarray(k, float))
        Am, _ = block(-np.asarray(k, float))
        A = 0.5 * (A + A.conj().T)
        Am = 0.5 * (Am + Am.conj().T)
        return np.block([[A, B], [B.conj().T, Am.T]])

    def solve(self, grid=(64, 1, 1), T=0.0, order="first", tol=1e-9, maxiter=200, mix=0.5):
        """'first': 1/S correction with linear-spin-wave correlations; 'sc': self-consistent"""
        ks = self.kgrid(grid)
        self.hf = None
        self.update(ks, T)
        it = 1
        if order == "sc":
            prev = None
            for it in range(2, maxiter + 1):
                old = self.hf
                self.update(ks, T)
                cur = np.array([v for (_, v, _, _) in self.hf])
                if old is not None and mix < 1.0:
                    oldv = np.array([v for (_, v, _, _) in old])
                    if len(oldv) == len(cur):
                        self.hf = [(p, mix * v + (1 - mix) * ov, o1, o2)
                                   for (p, v, o1, o2), ov in zip(self.hf, oldv)]
                        cur = mix * cur + (1 - mix) * oldv
                if prev is not None and len(prev) == len(cur) and np.abs(cur - prev).max() < tol:
                    break
                prev = cur
        self.iterations = it
        return self

    def spectrum(self, qs):
        """renormalised magnon energies along qs (meV)"""
        return np.array([self.modes(q)[0] for q in qs])

    def bare_spectrum(self, qs):
        hf, self.hf = self.hf, None
        try:
            return np.array([self.modes(q)[0] for q in qs])
        finally:
            self.hf = hf

    def sublattice_magnetisation(self, grid=(64, 1, 1), T=0.0):
        """⟨S^z'⟩ per site = S − ⟨a†a⟩ (zero-point and thermal reduction)"""
        ks = self.kgrid(grid)
        Cks = self.correlations(ks, T)
        N = self.N
        return np.array([self.S[a] - np.mean(Cks[:, a, a] - 1).real for a in range(N)])


def analyse_interacting(lat, frac, types, symbols, mags, rotations, translations, T=0.0, order="first",
                        grid=(64, 1, 1), n_per=60, path=None, relax=False, **model_kw):
    """linear vs Hartree–Fock (four-magnon) renormalised magnon bands along a path"""
    from .magnons import analyse, relax_spins, make_path, DEFAULT_PATH
    m = analyse(lat, frac, types, symbols, mags, rotations, translations, return_model=True, **model_kw)
    dirs = m["dirs0"]
    if relax:
        dirs, _, _ = relax_spins(dirs, m["S_arr"], m["magnetic"], m["pairs"], m["Asite"], hfield=m["hfield"])
    hf = FourMagnonHF.from_model(np.asarray(lat, float), np.asarray(frac, float), m, dirs=dirs)
    hf.solve(grid=grid, T=T, order=order)
    if path is None:
        from .magphon import auto_path
        path = auto_path(np.asarray(lat, float), [p[2] for p in m["pairs"]])
    qs, xs, ticks = make_path(np.asarray(lat, float), path, n_per)
    e0 = hf.bare_spectrum(qs)
    e1 = hf.spectrum(qs)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(e0 > 1e-6, e1 / e0, np.nan)
    return dict(model=hf, q=qs, x=xs, ticks=ticks, omega0=e0, omega=e1, ratio=ratio, T=T, order=order,
                collinear=hf.collinear, m_sub=hf.sublattice_magnetisation(grid, T), iterations=hf.iterations)


def plot_interacting(res, save=None, title=None, figsize=(7.4, 4.6)):
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    fig, ax = plt.subplots(figsize=figsize)
    for n in range(res["omega0"].shape[1]):
        ax.plot(res["x"], res["omega0"][:, n], "--", color="0.55", lw=1.3, label="linear spin waves" if n == 0 else None)
        ax.plot(res["x"], res["omega"][:, n], color="#c0392b", lw=2.2,
                label=f"+ four-magnon HF ({res['order']}, T = {res['T']:g} K)" if n == 0 else None)
    for t, _ in res["ticks"]:
        ax.axvline(t, color="0.85", lw=0.7, zorder=0)
    ax.set_xticks([t for t, _ in res["ticks"]])
    ax.set_xticklabels([lab for _, lab in res["ticks"]])
    ax.set_xlim(res["x"][0], res["x"][-1])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("magnon energy (meV)")
    ax.legend(fontsize=9)
    ax.set_title(title or "magnon–magnon interactions (Hartree–Fock)", fontsize=11)
    if save:
        fig.savefig(save)
    return fig
