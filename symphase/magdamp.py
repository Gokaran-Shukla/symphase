# -*- coding: utf-8 -*-
"""
symphase.magdamp — two-magnon (second-order) striction and damping of magnons and phonons
==========================================================================================

1. Three-boson magnon–magnon–phonon vertex.  The bond-length dependence of the exchange,
   J(d0 + δd) = J + Σ_γ G^γ δd_γ, inserted into the BILINEAR (two-magnon) part of the spin Hamiltonian
   gives  H3 = N^{-1/2} Σ_{k,q,ν} (b_qν + b†_{−qν}) Ψ†_{k+q} M_ν(k,q) Ψ_k,   Ψ_k = (a_k, a†_{−k}).
   It exists also for collinear magnets with isotropic striction (where the LINEAR, one-magnon
   hybridisation of symphase.magphon vanishes): magnons scatter by absorbing / emitting phonons and a
   phonon can decay into two magnons.  In the magnon normal-mode basis W = T(k+q)† M T(k).

2. Thermal linewidths (Fermi's golden rule, FWHM in meV, Bose factors n, N at temperature T):
   phonon (q, ν):  γ = 2π/N Σ_k { |S_nm|² (n_m(k) − n_n(k+q)) δ(ω + ε_m(k) − ε_n(k+q))        [scattering]
                                + ½ |P_nm|² (1 + n_n + n_m) δ(ω − ε_n(k+q) − ε_m(−k)) }        [two-magnon decay]
   magnon (k, n):  γ = 2π/N Σ_{q,ν,m} { |S|² (1 + N + n_m) δ(ε_n − ε_m(k−q) − ω)              [phonon emission]
                                      + |S|² (N − n_m) δ(ε_n + ω − ε_m(k+q))                  [phonon absorption]
                                      + |P|² (n_m − N) δ(ε_n + ε_m − ω) }                      [fusion into a phonon]
   δ → Gaussian of width σ.

3. Phenomenological damping of the coupled modes: linearised Landau–Lifshitz–Gilbert equations for the
   spins (Gilbert α: Ẋ = ∂H/∂Y + αẎ, Ẏ = −∂H/∂X − αẊ) and a friction η for the lattice momenta
   (Ṗ = −∂H/∂u − ηP), solved exactly: complex eigenfrequencies ω − iγ/2  (γ = FWHM).
   Ferromagnet: γ = 2αω/(1 + α²) exactly.
"""

import math

import numpy as np

from .magnons import colpa, hamiltonian
from .magphon import HB2_AMU

KB_MEV = 0.08617333262


def bose(e, T):
    e = np.asarray(e, float)
    if T <= 0:
        return np.zeros_like(e)
    with np.errstate(over="ignore", divide="ignore"):
        return 1.0 / np.expm1(np.clip(e / (KB_MEV * T), 1e-12, 700))


def _gauss(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (s * math.sqrt(2 * math.pi))


class MagnonPhononVertex:
    """three-boson vertex and linewidths for a symphase.magphon.MagnonPhonon model"""

    def __init__(self, mp):
        self.mp = mp
        self.N = mp.N
        self.U = mp.x + 1j * mp.y
        self.V = mp.v
        self.lat = mp.lat
        self.cart = mp.cart
        self._cache = {}

    # ---------------------------------------------------------------- magnons
    def magnon(self, k):
        """(ε ascending (N,), P (2N, N) positive-norm Colpa vectors) at Cartesian k"""
        key = tuple(np.round(k, 10))
        if key not in self._cache:
            mp = self.mp
            H = hamiltonian(np.asarray(k, float), mp.sites, self.U, self.V, mp.S, mp.pairs, mp.Asite, mp.hfield)
            om, T = colpa(H)
            if om is None:
                raise ValueError(f"magnon Hamiltonian not positive definite at k = {k}")
            o = np.argsort(om)
            self._cache[key] = (om[o], T[:, o])
        return self._cache[key]

    def T_full(self, k):
        """paraunitary T(k) = [[U_k, V*_{−k}], [V_k, U*_{−k}]]  (Ψ_k = T Φ_k, Φ = (α_k, α†_{−k}))"""
        N = self.N
        _, P = self.magnon(k)
        _, Pm = self.magnon(-np.asarray(k))
        T = np.zeros((2 * N, 2 * N), complex)
        T[:N, :N], T[N:, :N] = P[:N], P[N:]
        T[:N, N:], T[N:, N:] = np.conj(Pm[N:]), np.conj(Pm[:N])
        return T

    # ---------------------------------------------------------------- vertex
    def M(self, kin, q, f):
        """2N x 2N coupling of Ψ†_{kin+q} … Ψ_{kin} for a displacement pattern u_j = f_j e^{iq·r_j} (f: (n_atoms, 3))"""
        mp = self.mp
        N = self.N
        kin, q = np.asarray(kin, float), np.asarray(q, float)
        kout = kin + q
        A = np.zeros((N, N), complex)
        B = np.zeros((N, N), complex)
        C = np.zeros((N, N), complex)
        U, V, S = self.U, self.V, mp.S
        for (i, j, d, J), G in zip(mp.pairs, mp.G):
            if G is None:
                continue
            a, b = mp.loc[i], mp.loc[j]
            r = math.sqrt(S[a] * S[b]) / 2.0
            for g in range(3):
                fa, fb = f[i][g], f[j][g]
                if fa == 0 and fb == 0:
                    continue
                Gg = G[g]
                c1 = r * (U[a] @ Gg @ np.conj(U[b]))
                c3 = r * (U[a] @ Gg @ U[b])
                dv = V[a] @ Gg @ V[b]
                e_out, e_in = np.exp(1j * kout @ d), np.exp(1j * kin @ d)
                ph_ab = fb * e_out - fa * e_in                  # a†_{a,kout} … _{b,kin}
                ph_ba = fb * np.conj(e_in) - fa * np.conj(e_out)  # a†_{b,kout} … _{a,kin}
                A[a, b] += c1 * ph_ab
                A[b, a] += np.conj(c1) * ph_ba
                A[a, a] += -S[b] * dv * (fb * np.exp(1j * q @ d) - fa)
                A[b, b] += -S[a] * dv * (fb - fa * np.exp(-1j * q @ d))
                # a pair term c3 a†_a a†_b (and its partner c3* a_b a_a) is ONE real-space term: it is
                # represented once, as a†_{a,kout} a†_{b,−kin} (resp. a_{a,−kout} a_{b,kin}), summed over kin
                B[a, b] += c3 * ph_ab
                C[a, b] += np.conj(c3) * ph_ab
        Mm = np.zeros((2 * N, 2 * N), complex)
        Mm[:N, :N], Mm[:N, N:], Mm[N:, :N] = A, B, C
        return Mm

    def W(self, kin, q, f):
        kin = np.asarray(kin, float)
        return self.T_full(kin + q).conj().T @ self.M(kin, q, f) @ self.T_full(kin)

    # ---------------------------------------------------------------- phonons
    def phonons(self, q):
        """ω_ν(q) (meV), displacement patterns f_ν (n_atoms, 3) = sqrt(ħ²/2Mω) ε_ν (zero-point amplitudes)"""
        mp = self.mp
        n = mp.n
        H = mp.hessian(np.asarray(q, float), coupling=False, magnons=False)
        Phi = H[mp.N:mp.N + 3 * n, mp.N:mp.N + 3 * n]
        m3 = np.repeat(mp.masses, 3)
        D = Phi / np.sqrt(np.outer(m3, m3))
        w2, E = np.linalg.eigh(0.5 * (D + D.conj().T))
        om = np.sqrt(np.clip(w2 * HB2_AMU, 0, None))
        fs = []
        for nu in range(3 * n):
            if om[nu] < 1e-6:
                fs.append(None)
                continue
            fs.append((np.sqrt(HB2_AMU / (2 * om[nu])) * E[:, nu] / np.sqrt(m3)).reshape(n, 3))
        return om, fs

    # ---------------------------------------------------------------- amplitudes
    def scattering(self, k, q, f):
        """S[n, m]: amplitude of α†_{n,k+q} α_{m,k}"""
        N = self.N
        W1 = self.W(k, q, f)
        W2 = self.W(-np.asarray(k) - q, q, f)
        return W1[:N, :N] + W2[N:, N:].T

    def pair(self, k, q, f):
        """P[n, m]: amplitude of α†_{n,k+q} α†_{m,−k}  (both orderings of the boson pair summed)"""
        N = self.N
        W1 = self.W(k, q, f)
        W2 = self.W(-np.asarray(k) - q, q, f)
        return W1[:N, N:] + W2[:N, N:].T

    # ---------------------------------------------------------------- linewidths
    def kgrid(self, n):
        n = list(n) + [1] * (3 - len(n))
        f = [np.arange(m) / m for m in n]
        F = np.array(np.meshgrid(*f, indexing="ij")).reshape(3, -1).T
        return F @ (2 * np.pi * np.linalg.inv(self.lat).T)

    def phonon_linewidth(self, q, T=300.0, grid=(60, 1, 1), sigma=0.2, modes=None):
        """FWHM (meV) of every phonon at q from scattering and two-magnon decay; returns ω, γ, γ_scatt, γ_pair"""
        q = np.asarray(q, float)
        om, fs = self.phonons(q)
        ks = self.kgrid(grid)
        nk = len(ks)
        gs, gp = np.zeros(len(om)), np.zeros(len(om))
        for nu in (range(len(om)) if modes is None else modes):
            if fs[nu] is None:
                continue
            for k in ks:
                e1, _ = self.magnon(k)
                e2, _ = self.magnon(k + q)
                e3, _ = self.magnon(-k)
                Sm = self.scattering(k, q, fs[nu])
                Pm = self.pair(k, q, fs[nu])
                n1, n2, n3 = bose(e1, T), bose(e2, T), bose(e3, T)
                gs[nu] += np.sum(np.abs(Sm) ** 2 * (n1[None, :] - n2[:, None]) *
                                 _gauss(om[nu] + e1[None, :] - e2[:, None], sigma))
                gp[nu] += 0.5 * np.sum(np.abs(Pm) ** 2 * (1 + n2[:, None] + n3[None, :]) *
                                       _gauss(om[nu] - e2[:, None] - e3[None, :], sigma))
        gs *= 2 * math.pi / nk
        gp *= 2 * math.pi / nk
        return dict(omega=om, gamma=gs + gp, gamma_scatt=gs, gamma_pair=gp)

    def magnon_linewidth(self, k, T=300.0, grid=(60, 1, 1), sigma=0.2):
        """FWHM (meV) of every magnon at k from phonon emission, absorption and fusion"""
        k = np.asarray(k, float)
        ek, _ = self.magnon(k)
        nk_ = bose(ek, T)
        g_em, g_ab, g_fu = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        qs = self.kgrid(grid)
        for q in qs:
            om, fs = self.phonons(q)
            omm, fsm = self.phonons(-q)
            Nq, Nqm = bose(om, T), bose(omm, T)
            ekq, _ = self.magnon(k + q)
            ekmq, _ = self.magnon(k - q)
            for nu in range(len(om)):
                if fs[nu] is not None:
                    # absorption of phonon (q,ν): k -> k+q
                    Sa = self.scattering(k, q, fs[nu])              # [n'(k+q), m(k)]
                    g_ab += np.sum(np.abs(Sa) ** 2 * (Nq[nu] - bose(ekq, T))[:, None] *
                                   _gauss(ek[None, :] + om[nu] - ekq[:, None], sigma), axis=0)
                    # fusion: magnon (n,k) + magnon (m', q−k) -> phonon (q, ν)
                    Pf = self.pair(k - q, q, fs[nu])               # α†_{n,k} α†_{m,q−k}
                    eqk = self.magnon(q - k)[0]
                    g_fu += np.sum(np.abs(Pf) ** 2 * (bose(eqk, T)[None, :] - Nq[nu]) *
                                   _gauss(ek[:, None] + eqk[None, :] - om[nu], sigma), axis=1)
                if fsm[nu] is not None:
                    # emission of phonon (q, ν) = absorption amplitude with −q: k -> k − q
                    Se = self.scattering(k, -q, fsm[nu])            # [n'(k−q), m(k)]
                    g_em += np.sum(np.abs(Se) ** 2 * (1 + Nqm[nu] + bose(ekmq, T))[:, None] *
                                   _gauss(ek[None, :] - omm[nu] - ekmq[:, None], sigma), axis=0)
        f = 2 * math.pi / len(qs)
        return dict(epsilon=ek, gamma=f * (g_em + g_ab + g_fu), gamma_emit=f * g_em, gamma_abs=f * g_ab,
                    gamma_fusion=f * g_fu, n=nk_)

    # ---------------------------------------------------------------- Gilbert + phonon friction
    def damped_modes(self, k, alpha=0.0, eta=0.0, coupling=True):
        """complex frequencies of the linearised LLG + damped lattice: returns ω (meV), γ = FWHM (meV),
        magnon weight, sorted by ω"""
        mp = self.mp
        H = mp.hessian(np.asarray(k, float), coupling=coupling)
        M, N = mp.M, mp.N
        Om = np.block([[np.zeros((M, M)), np.eye(M)], [-np.eye(M), np.zeros((M, M))]])
        L = np.eye(2 * M, dtype=complex)
        for a in range(N):                                   # spins (Gilbert): Ẋ = ∂H/∂Y + αẎ, Ẏ = −∂H/∂X − αẊ
            L[a, M + a] -= alpha
            L[M + a, a] += alpha
        A = np.linalg.solve(L, Om @ H)
        for p in range(N, M):                                # lattice momenta: Ṗ = … − η P
            A[M + p, M + p] -= eta
        w, Z = np.linalg.eig(A)
        lam = 1j * w                                         # ξ ∝ e^{−iΩτ},  ξ̇ = −iΩ ξ  ->  Ω = i w
        pos = [i for i in range(len(lam)) if lam[i].real > 1e-9]
        pos = sorted(pos, key=lambda i: lam[i].real)
        om = np.array([lam[i].real for i in pos])
        gam = np.array([-2 * lam[i].imag for i in pos])
        wm = []
        for i in pos:
            z = Z[:, i]
            s = np.sum(np.abs(z[:N]) ** 2 + np.abs(z[M:M + N]) ** 2)
            tot = np.sum(np.abs(z) ** 2)
            wm.append(s / tot if tot > 0 else 0.0)
        return om, gam, np.array(wm)


def gilbert_first_order(mp, k, alpha):
    """independent check: γ_n = 2α ω_n² Σ_a(|X_a|²+|Y_a|²) / ξ†𝓗ξ  (energy dissipation of each undamped mode)"""
    H = mp.hessian(np.asarray(k, float), coupling=False)
    M, N = mp.M, mp.N
    Om = np.block([[np.zeros((M, M)), np.eye(M)], [-np.eye(M), np.zeros((M, M))]])
    w, Z = np.linalg.eig(Om @ H)
    lam = 1j * w
    out = []
    for i in range(len(lam)):
        if lam[i].real > 1e-9:
            z = Z[:, i]
            num = np.sum(np.abs(z[:N]) ** 2 + np.abs(z[M:M + N]) ** 2)
            if num < 1e-12 * np.sum(np.abs(z) ** 2):
                continue
            out.append((lam[i].real, 2 * alpha * lam[i].real ** 2 * num / np.real(np.conj(z) @ H @ z)))
    return sorted(out)


# ----------------------------------------------------------------------------
#  along a path, plots
# ----------------------------------------------------------------------------
def linewidths_along_path(mp, qs, xs, T=300.0, grid=(200, 1, 1), sigma=0.2, every=4, alpha=0.0, eta=0.0):
    """phonon (two-magnon striction) and magnon (phonon emission/absorption/fusion) FWHM along a path,
    plus Gilbert/friction FWHM of the coupled modes if alpha / eta > 0"""
    vx = MagnonPhononVertex(mp)
    idx = list(range(0, len(qs), every))
    if idx[-1] != len(qs) - 1:
        idx.append(len(qs) - 1)
    out = dict(x=np.asarray(xs)[idx], T=T, phonon_omega=[], phonon_gamma=[], magnon_eps=[], magnon_gamma=[])
    for i in idx:
        q = np.asarray(qs[i], float)
        ph = vx.phonon_linewidth(q, T=T, grid=grid, sigma=sigma)
        mg = vx.magnon_linewidth(q, T=T, grid=grid, sigma=sigma)
        out["phonon_omega"].append(ph["omega"])
        out["phonon_gamma"].append(ph["gamma"])
        out["magnon_eps"].append(mg["epsilon"])
        out["magnon_gamma"].append(mg["gamma"])
    for key in ("phonon_omega", "phonon_gamma", "magnon_eps", "magnon_gamma"):
        out[key] = np.array(out[key])
    if alpha or eta:
        dm = [vx.damped_modes(np.asarray(qs[i], float), alpha=alpha, eta=eta) for i in idx]
        M = mp.M

        def pad(v):
            return np.r_[v, np.full(M - len(v), np.nan)] if len(v) < M else np.asarray(v)[:M]
        out["damped_omega"] = np.array([pad(d[0]) for d in dm])
        out["damped_gamma"] = np.array([pad(d[1]) for d in dm])
        out["damped_wm"] = np.array([pad(d[2]) for d in dm])
    return out


def plot_linewidths(lw, res=None, save=None, scale=1.0, title=None, figsize=(7.6, 5.0)):
    """bands (from magphon.analyse_mp result) with ω ± scale·γ/2 bands for phonons (blue) and magnons (red)"""
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    fig, ax = plt.subplots(figsize=figsize)
    if res is not None:
        for n in range(res["omega0"].shape[1]):
            ax.plot(res["x"], res["omega0"][:, n], color="0.75", lw=0.8)
    x = lw["x"]
    for n in range(lw["phonon_omega"].shape[1]):
        w, g = lw["phonon_omega"][:, n], scale * lw["phonon_gamma"][:, n] / 2
        ax.fill_between(x, w - g, w + g, color="#2c5d8f", alpha=0.35, lw=0)
        ax.plot(x, w, color="#2c5d8f", lw=1.4)
    for n in range(lw["magnon_eps"].shape[1]):
        w, g = lw["magnon_eps"][:, n], scale * lw["magnon_gamma"][:, n] / 2
        ax.fill_between(x, w - g, w + g, color="#c0392b", alpha=0.35, lw=0)
        ax.plot(x, w, color="#c0392b", lw=1.4)
    if res is not None:
        for t, _ in res["ticks"]:
            ax.axvline(t, color="0.85", lw=0.7, zorder=0)
        ax.set_xticks([t for t, _ in res["ticks"]])
        ax.set_xticklabels([lab for _, lab in res["ticks"]])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("energy (meV)")
    ax.plot([], [], color="#2c5d8f", lw=6, alpha=0.4, label="phonons ± γ/2" + (f" (×{scale:g})" if scale != 1 else ""))
    ax.plot([], [], color="#c0392b", lw=6, alpha=0.4, label="magnons ± γ/2" + (f" (×{scale:g})" if scale != 1 else ""))
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title or f"two-magnon striction linewidths at T = {lw['T']:g} K", fontsize=11)
    if save:
        fig.savefig(save)
    return fig
