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
Exact to order 1/S for collinear magnets.  MagnonSelfEnergy adds the cubic terms of non-collinear orders (one-loop
bubble + tadpole: the remaining O(1/S) terms, spontaneous magnon decay) and the second-order four-magnon sunset
(magnon–magnon damping, O(1/S²)).  Single-ion anisotropy enters the quartic terms at harmonic level only.
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
        self.order = order
        return self

    def first_order_shift(self, qs):
        """δε_n(k) = [T†(k) H_HF(k) T(k)]_nn with the linear-spin-wave T: the strict 1/S (first-order) shift.
        Equal to spectrum − bare_spectrum for collinear magnets (H_HF ∝ H); for non-collinear orders it is the
        quantity that combines with the cubic self-energy (Goldstone modes stay gapless)."""
        if self.hf is None:
            raise ValueError("call solve() first")
        hf, self.hf = self.hf, None
        try:
            mods = [self.modes(q) for q in qs]
        finally:
            self.hf = hf
        M = self.N
        out = []
        for q, (om, P) in zip(qs, mods):
            Hh = self._H_hf(np.asarray(q, float))
            out.append(np.real(np.einsum("in,ij,jn->n", np.conj(P), Hh, P))[:M])
        return np.array(out)

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
        """⟨S^z'⟩ per site = S − ⟨a†a⟩ (zero-point and thermal reduction); linear-spin-wave correlations unless
        the Hartree–Fock solution is self-consistent"""
        ks = self.kgrid(grid)
        keep = self.hf
        if getattr(self, "order", "first") != "sc":
            self.hf = None
        try:
            Cks = self.correlations(ks, T)
        finally:
            self.hf = keep
        N = self.N
        return np.array([self.S[a] - np.mean(Cks[:, a, a] - 1).real for a in range(N)])


# =============================================================================
#  cubic (three-magnon) terms and second-order self-energies
# =============================================================================
_SWAP = None


def _swap(M):
    return np.r_[np.arange(M, 2 * M), np.arange(M)]


class MagnonSelfEnergy:
    """Second-order magnon self-energies beyond Hartree–Fock.

    Real-space Holstein–Primakoff terms (fields = (sublattice, Cartesian position in the home cell, dagger)):
      cubic   H3 : from (S_p − n_p) v_p·J·S_q^⊥ on every bond (−n_p L_q and −S_p C_q/4S_q), single-ion anisotropy
                   2(S − n) v·A·S^⊥ and Zeeman h·S^⊥ (C terms) — non-zero only for non-collinear orders;
      quartic H4 : the exchange terms of FourMagnonHF.
    Fourier + Bogoliubov (Ψ_p = T_p Φ_p, Φ_p = (α_p, α†_{−p})) give, for the external magnon X = α_{k,n},
        H_r ⊃ Σ_{ordered internal tuples} c(Φ_2..Φ_{r+1}) X Φ_2…Φ_{r+1},   c = (1/r!) Σ_{slot assignments} Γ,
    and the retarded self-energy (bosonic Wick theorem, any temperature)
        Σ(ω) = r! Σ_tuples |c|² [Π_cr(1+n) Π_an n − Π_cr n Π_an(1+n)] / (ω − Σ_cr ε + Σ_an ε + iη),
    r = 2: decay, absorption and source (cubic one-loop bubble), r = 3: four-magnon sunset (2↔2 scattering, 1↔3).
    The cubic tadpole (linear terms from normal ordering H3 → O(1/S) shift of the classical angles) is added as a
    static shift  Σ_tad = −Σ_m 2 Re(h_m g_m)/ε_{0m}.
    Weights are normalised so that Σ(ω) = Σ w/(ω − e + iη), as in MagnonPhononVertex.self_energy."""

    def __init__(self, hf):
        self.hf = hf
        self.M = hf.N
        self.lat = hf.lat
        self.rec = 2 * np.pi * np.linalg.inv(hf.lat).T
        self.tau = hf.cart
        inv = np.linalg.inv(hf.lat)
        for i, j, d, J in hf.pairs:
            a, b = hf.loc[i], hf.loc[j]
            f = (np.asarray(d, float) - (self.tau[b] - self.tau[a])) @ inv
            if np.abs(f - np.round(f)).max() > 1e-5:
                raise ValueError("site positions inconsistent with the bond vectors")
        self._cache = {}
        self.terms3 = self._cubic_terms()
        self.terms4 = self._quartic_terms()

    # ---------------------------------------------------------------- real-space HP terms
    def _cubic_terms(self):
        hf = self.hf
        U, V, S = hf.U, hf.V, hf.S
        Cd = {"+": (True, False, False), "-": (True, True, False)}          # C+ = a†aa, C− = a†a†a

        def e(x, al):
            return np.conj(U[x]) if al == "+" else U[x]
        out = []
        tau = self.tau
        for i, j, d, J in hf.pairs:
            a, b = hf.loc[i], hf.loc[j]
            J = np.asarray(J, float)
            pos = {"i": (a, tuple(tau[a])), "j": (b, tuple(tau[a] + np.asarray(d, float)))}
            for p, q, Jm in (("i", "j", J), ("j", "i", J.T)):
                sp, rp = pos[p]
                sq, rq = pos[q]
                for al in "+-":
                    w = V[sp] @ Jm @ e(sq, al)
                    if abs(w) < 1e-14:
                        continue
                    r = math.sqrt(S[sq] / 2.0)
                    out.append((-w * r, [(sp, rp, True), (sp, rp, False), (sq, rq, al == "-")]))      # −n_p L_q
                    out.append((-S[sp] * w * r / (4 * S[sq]), [(sq, rq, dg) for dg in Cd[al]]))      # −S_p C_q/4S_q
        for x in range(self.M):
            r = math.sqrt(S[x] / 2.0)
            z3 = tuple(tau[x])
            Am = hf.Asite[x]
            if Am is not None:
                As = 0.5 * (np.asarray(Am, float) + np.asarray(Am, float).T)
                for al in "+-":
                    w = V[x] @ As @ e(x, al)
                    if abs(w) < 1e-14:
                        continue
                    nl = (True, False, False) if al == "+" else (True, True, False)       # n L, normal ordered
                    out.append((-2 * w * r, [(x, z3, dg) for dg in nl]))
                    out.append((-2 * S[x] * w * r / (4 * S[x]), [(x, z3, dg) for dg in Cd[al]]))
            if hf.hfield is not None:
                for al in "+-":
                    w = np.asarray(hf.hfield[x], float) @ e(x, al)
                    if abs(w) < 1e-14:
                        continue
                    out.append((-w * r / (4 * S[x]), [(x, z3, dg) for dg in Cd[al]]))
        return out

    def _quartic_terms(self):
        hf = self.hf
        out = []
        for (i, j, d, J), terms in zip(hf.pairs, hf.terms):
            a = hf.loc[i]
            pos = {"i": (a, tuple(self.tau[a])), "j": (hf.loc[j], tuple(self.tau[a] + np.asarray(d, float)))}
            for coef, ops in terms:
                out.append((coef, [(pos[o][0], pos[o][1], dg) for o, dg in ops]))
        return out

    def classical_terms_value(self, terms, z_of):
        """Σ_terms coef Π z (or z̄) for a callable z_of(sublattice, position) — real-space check helper"""
        tot = 0.0
        for coef, fields in terms:
            v = coef
            for s, r, dg in fields:
                zz = z_of(s, np.asarray(r))
                v = v * (np.conj(zz) if dg else zz)
            tot += v
        return tot

    # ---------------------------------------------------------------- Bogoliubov modes, canonical gauge
    def _modes0(self, k):
        hf = self.hf
        H = hamiltonian(np.asarray(k, float), hf.sites, hf.U, hf.V, hf.S, hf.pairs, hf.Asite, hf.hfield)
        om, T = colpa(H)
        if om is None:
            raise ValueError(f"linear spin-wave Hamiltonian not positive definite at k = {np.round(k, 4)}")
        o = np.argsort(om)
        return om[o], T[:, o]

    def _mset(self, p):
        p = np.asarray(p, float)
        f = self.lat @ p / (2 * np.pi)
        f0 = np.round(f - np.floor(f), 8) % 1.0
        key = tuple(f0)
        if key not in self._cache:
            p0 = f0 @ self.rec
            self._cache[key] = (p0,) + self._modes0(p0)
        p0, om, P = self._cache[key]
        D = np.exp(-1j * (self.tau @ (p - p0)))
        return om, P * np.r_[D, D][:, None]

    def _batch(self, ps):
        """energies and P (2M×M, canonical gauge) for an array of momenta — vectorised over repeated folds"""
        ps = np.atleast_2d(np.asarray(ps, float))
        f = ps @ self.lat.T / (2 * np.pi)
        f0 = np.round(f - np.floor(f), 8) % 1.0
        keys, inv = np.unique(f0, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        om0, P0 = [], []
        for kk in keys:
            key = tuple(kk)
            if key not in self._cache:
                p0 = kk @ self.rec
                self._cache[key] = (p0,) + self._modes0(p0)
            _, o, P = self._cache[key]
            om0.append(o)
            P0.append(P)
        om = np.array(om0)[inv]
        P = np.array(P0)[inv]
        D = np.exp(-1j * ((ps - f0 @ self.rec) @ self.tau.T))                 # (B, M)
        M = self.M
        P[:, :M] *= D[:, :, None]
        P[:, M:] *= D[:, :, None]
        return om, P

    def _T(self, ps):
        """energies at p and −p and the full paraunitary T(p) (Ψ_p = T Φ_p) for an array of momenta"""
        ps = np.atleast_2d(np.asarray(ps, float))
        M = self.M
        ep, P = self._batch(ps)
        em, Pm = self._batch(-ps)
        Ts = np.empty((len(ps), 2 * M, 2 * M), complex)
        Ts[:, :M, :M], Ts[:, M:, :M] = P[:, :M], P[:, M:]
        Ts[:, :M, M:], Ts[:, M:, M:] = np.conj(Pm[:, M:]), np.conj(Pm[:, :M])
        return ep, em, Ts

    # ---------------------------------------------------------------- vertices
    def _coeffs(self, terms, k, n, slots, ncell):
        """c(ν_2..ν_{r+1}) for X = α_{k,n} and internal slot momenta (list of (B, 3) arrays)"""
        from itertools import permutations
        M = self.M
        sw = _swap(M)
        r = len(slots)
        B = len(slots[0])
        allp = [np.atleast_2d(np.asarray(k, float))] + [np.asarray(s, float) for s in slots]
        Tp = [self._T(p)[2] for p in allp]
        Tm = [self._T(-p)[2] for p in allp]
        cache = {}

        def fv(field, j):
            key = (field[0], field[1], field[2], j)
            if key not in cache:
                s, rr, dg = field
                ph = np.exp(1j * (allp[j] @ np.asarray(rr, float)))
                v = np.conj(Tm[j][:, s, :])[:, sw] if dg else Tp[j][:, s, :]
                v = ph[:, None] * v
                cache[key] = v[:, n] if j == 0 else v
            return cache[key]
        letters = "ijkl"[:r]
        spec = ",".join("b" + c for c in letters) + "->b" + letters
        out = np.zeros((B,) + (2 * M,) * r, complex)
        for coef, fields in terms:
            if len(fields) != r + 1:
                continue
            for perm in permutations(range(r + 1)):
                x = coef * fv(fields[perm[0]], 0)
                out += x.reshape((-1,) + (1,) * r) * np.einsum(spec, *[fv(fields[perm[j]], j) for j in range(1, r + 1)])
        return out / math.factorial(r) / ncell ** ((r - 1) / 2.0)

    def _assemble(self, c, slots, T, emin=1e-6):
        M = self.M
        r = len(slots)
        Es, cre = [], np.r_[np.zeros(M, bool), np.ones(M, bool)]
        for p in slots:
            ep, em, _ = self._T(p)
            Es.append(np.concatenate([ep, em], axis=1))                      # ν<M: annihilated α_p; ν≥M: created α_{−p}
        epole = 0.0
        A1, A2 = 1.0, 1.0
        ok = True
        for j, E in enumerate(Es):
            shape = [len(E)] + [1] * r
            shape[j + 1] = 2 * M
            Ej = E.reshape(shape)
            sg = np.where(cre, 1.0, -1.0).reshape([1] + [2 * M if t == j else 1 for t in range(r)])
            nb = _bose(Ej, T)
            cr = cre.reshape([1] + [2 * M if t == j else 1 for t in range(r)])
            epole = epole + sg * Ej
            A1 = A1 * np.where(cr, 1 + nb, nb)
            A2 = A2 * np.where(cr, nb, 1 + nb)
            ok = ok & (Ej > emin)
        w = math.factorial(r) * np.abs(c) ** 2 * (A1 - A2) * ok
        e = np.broadcast_to(epole, w.shape)
        keep = np.abs(w) > 1e-14 * max(np.abs(w).max(), 1e-300)
        return w[keep], e[keep]

    def kgrid(self, grid):
        return self.hf.kgrid(grid)

    def cubic_poles(self, k, n, T=0.0, grid=(64, 1, 1), qs=None):
        """Σ³_n(k, ω) = Σ w/(ω − e + iη): one-loop bubble of the cubic vertex (decay, absorption, source)"""
        k = np.asarray(k, float)
        qs = self.kgrid(grid) if qs is None else np.asarray(qs, float)
        slots = [qs, -k[None, :] - qs]
        c = self._coeffs(self.terms3, k, n, slots, len(qs))
        w, e = self._assemble(c, slots, T)
        return self._mset(k)[0][n], w, e

    def quartic_poles(self, k, n, T=0.0, grid=(24, 1, 1), qs=None, chunk=None):
        """Σ⁴_n(k, ω): second-order four-magnon (sunset) self-energy — 2↔2 scattering, 1↔3 decay, source"""
        k = np.asarray(k, float)
        qs = self.kgrid(grid) if qs is None else np.asarray(qs, float)
        nq = len(qs)
        chunk = chunk or max(1, 40000 // max(nq, 1))
        W, E = [], []
        for i0 in range(0, nq, chunk):
            q1 = np.repeat(qs[i0:i0 + chunk], nq, axis=0)
            q2 = np.tile(qs, (len(qs[i0:i0 + chunk]), 1))
            slots = [q1, q2, -k[None, :] - q1 - q2]
            c = self._coeffs(self.terms4, k, n, slots, nq)
            w, e = self._assemble(c, slots, T)
            W.append(w)
            E.append(e)
        return self._mset(k)[0][n], np.concatenate(W), np.concatenate(E)

    # ---------------------------------------------------------------- tadpole (linear terms)
    def _expect(self, f1, f2, ks, Cks):
        M = self.M
        (s1, r1, d1), (s2, r2, d2) = f1, f2
        dv = np.asarray(r2, float) - np.asarray(r1, float)
        same = (s1 == s2) and np.allclose(dv, 0)
        C11, C12 = Cks[:, :M, :M], Cks[:, :M, M:]

        def hop(sa, sb, dd, eq):      # ⟨a†_x a_y⟩, y − x = dd
            return np.mean(np.exp(1j * ks @ dd) * (C11[:, sb, sa] - (1.0 if sa == sb else 0.0))).item() if not eq \
                else np.mean(C11[:, sa, sa] - 1).item()

        def pr(sa, sb, dd):           # ⟨a_x a_y⟩
            return np.mean(np.exp(-1j * ks @ dd) * C12[:, sa, sb]).item()
        if d1 and not d2:
            return hop(s1, s2, dv, same)
        if not d1 and d2:
            return hop(s2, s1, -dv, same) + (1.0 if same else 0.0)
        if not d1 and not d2:
            return pr(s1, s2, dv)
        return np.conj(pr(s2, s1, -dv))

    def linear_terms(self, T=0.0, grid=(64, 1, 1), ks=None):
        """H1 = √N Σ_ν η_ν Φ_{0,ν} from Wick-decoupling the cubic terms (thermal); returns (η, ε_0)"""
        hf = self.hf
        ks = self.kgrid(grid) if ks is None else np.asarray(ks, float)
        key = (float(T), ks.shape, float(np.round(ks.sum(), 9)), float(np.round(np.abs(ks).sum(), 9)))
        lc = self.__dict__.setdefault("_lin_cache", {})
        if key not in lc:
            lc[key] = self._linear_terms(T, ks)
        return lc[key]

    def _linear_terms(self, T, ks):
        hf = self.hf
        keep, hf.hf = hf.hf, None
        try:
            Cks = hf.correlations(ks, T)
        finally:
            hf.hf = keep
        M = self.M
        lam, mu = np.zeros(M, complex), np.zeros(M, complex)
        for coef, fields in self.terms3:
            for a_, b_, c_ in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                val = coef * self._expect(fields[a_], fields[b_], ks, Cks)
                s, _, dg = fields[c_]
                if dg:
                    mu[s] += val
                else:
                    lam[s] += val
        ep, _, T0 = self._T(np.zeros((1, 3)))
        T0 = T0[0]
        eta = lam @ T0[:M] + mu @ np.conj(T0[:M][:, _swap(M)])
        return eta, ep[0]

    def tadpole(self, k, n, T=0.0, grid=(64, 1, 1), ks=None, emin=1e-5):
        """static shift of mode n at k from the cubic linear terms (O(1/S) correction of the classical angles)"""
        M = self.M
        eta, e0 = self.linear_terms(T, grid, ks)
        k = np.asarray(k, float)
        c = self._coeffs(self.terms3, k, n, [-k[None, :], np.zeros((1, 3))], 1)[0]
        tot = 0.0
        for m in range(M):
            if e0[m] < emin:
                continue
            h, g = eta[m], 2 * c[M + n, M + m]
            tot += -2 * np.real(h * g) / e0[m]
        return tot

    # ---------------------------------------------------------------- on-shell results
    @staticmethod
    def _sum(w, e, omega, eta, sigma=None):
        re = float(np.sum(w * (omega - e) / ((omega - e) ** 2 + eta ** 2)))
        if sigma:
            im = float(np.sum(w * np.exp(-0.5 * ((omega - e) / sigma) ** 2))) / (sigma * math.sqrt(2 * math.pi))
        else:
            im = float(np.sum(w * eta / ((omega - e) ** 2 + eta ** 2))) / math.pi
        return re, 2 * math.pi * im

    def onshell(self, k, T=0.0, grid=(64, 1, 1), eta=0.05, sigma=None, cubic=True, tadpole=True,
                quartic=False, grid4=None):
        """per mode at k: bare ε, Re Σ³(ε), Γ³ = −2 Im Σ³(ε), tadpole shift, Re Σ⁴(ε), Γ⁴ (meV)"""
        e0 = self._mset(np.asarray(k, float))[0]
        M = self.M
        out = {key: np.zeros(M) for key in ("re3", "gamma3", "tad", "re4", "gamma4")}
        out["e0"] = e0
        for n in range(M):
            if cubic:
                _, w, e = self.cubic_poles(k, n, T, grid)
                out["re3"][n], out["gamma3"][n] = self._sum(w, e, e0[n], eta, sigma)
                if tadpole:
                    out["tad"][n] = self.tadpole(k, n, T, grid)
            if quartic:
                _, w, e = self.quartic_poles(k, n, T, grid4 or grid)
                out["re4"][n], out["gamma4"][n] = self._sum(w, e, e0[n], eta, sigma)
        return out


# =============================================================================
#  self-consistent magnon decay (iΓ approximation)
# =============================================================================
class SelfConsistentDecay:
    """Non-perturbative treatment of the cubic (three-magnon) self-energy on a closed (unshifted) k grid.

    For every grid magnon (k, n) the energy E_k and the damping Γ_k, with δ_k = first-order Hartree–Fock + tadpole
    (static, `static=True`) and
        Σ³_k(ω) = Σ_poles v |F| / (ω − e' + i[η + (Γ_1 + Γ_2)/2]),
    e' = Σ_created E − Σ_annihilated E of the two internal magnons.  Modes:
        'onshell' : E = ε + δ + Re Σ(ε), Γ = −2 Im Σ(ε), bare internal lines (= MagnonSelfEnergy.onshell) — Born;
        'sc'      : self-consistent iΓ approximation (Chernyshev & Zhitomirsky, Rev. Mod. Phys. 85, 219 (2013)):
                    the resonant channels (decay, absorption) get ω − ε_q − ε_{k−q} + i(Γ_q + Γ_{k−q})/2 with the
                    current widths, Γ_k = −2 Im Σ_k(E_k) is iterated on the grid to self-consistency; the energy is
                    the 1/S on-shell value E = ε + δ + Re Σ(ε) (bare lines), which keeps the Hartree–Fock/cubic
                    Goldstone cancellation of non-collinear magnets exact.  The threshold singularities of the Born
                    damping are removed by the internal widths;
        'dyson'   : diagnostic only — real off-shell root of ω − ε − δ − Re Σ(ω) with bare lines; mixes orders in 1/S
                    (in non-collinear magnets it violates the Goldstone theorem near the Goldstone modes).
    renormalise=True also feeds the energies E into the internal lines (beyond 1/S; use with care).
    profile: 'gauss' (default) or 'lorentz' (canonical iΓ) line shape of the broadened resonant poles in Γ.
    energy_grid: shifted grid for the 1/S energies (recommended for non-collinear magnets: the Hartree–Fock/cubic
    cancellation near Goldstone modes needs a converged sum; default: the closed grid itself).
    Thermal factors use the bare energies.  Grid values are interpolated (periodic, linear) for off-grid momenta."""

    def __init__(self, se, grid=(16, 16, 1), T=0.0, eta=0.02, static=True, renormalise=False, energy_grid=None,
                 profile="gauss"):
        self.se, self.T, self.eta, self.renormalise = se, float(T), float(eta), renormalise
        self.profile = profile
        self.n = np.array(list(grid) + [1] * (3 - len(grid)))[:3]
        f = [np.arange(m) / m for m in self.n]
        self.kf = np.array(np.meshgrid(*f, indexing="ij")).reshape(3, -1).T
        self.ks = self.kf @ se.rec
        self.M = se.M
        self.e0 = se._batch(self.ks)[0]
        self.emin = 1e-3 * float(self.e0.max())
        self.gold = self.e0 <= self.emin
        self.static = static
        self.delta = self._static(self.ks) if static else np.zeros_like(self.e0)
        self.poles = [self._lines(k) for k in self.ks]
        self.energy_grid = energy_grid
        self.E1S = self._energies(self.ks, self.e0, self.delta)
        self.E = self.E1S.copy()
        self.G = np.zeros_like(self.e0)
        self.Z = np.ones_like(self.e0)
        self.iterations = 0
        self.history = []

    # ---------------------------------------------------------------- set-up
    def _static(self, ks):
        se, hf = self.se, self.se.hf
        if hf.hf is None:
            hf.solve(grid=tuple(self.n), T=self.T)
        out = hf.first_order_shift(ks)
        for i, k in enumerate(ks):
            for n in range(self.M):
                out[i, n] += se.tadpole(k, n, self.T, grid=tuple(self.n))     # shifted grid: no Γ zero modes
        return out

    def _energies(self, ks, e0, delta, poles=None):
        """1/S on-shell energies ε + δ + Re Σ³(ε): on the fine shifted energy_grid if given (the Hartree–Fock/cubic
        cancellation near Goldstone modes needs a well-converged BZ sum), else from the closed grid itself"""
        out = np.zeros_like(e0)
        for i, k in enumerate(ks):
            if self.energy_grid is not None:
                o = self.se.onshell(k, self.T, self.energy_grid, eta=self.eta, tadpole=False)
                out[i] = e0[i] + delta[i] + o["re3"]
            else:
                P = self.poles[i] if poles is None else poles[i]
                for n in range(self.M):
                    L = P["lines"][n]
                    out[i, n] = e0[i, n] + delta[i, n] + np.sum(L["v"] / (e0[i, n] - L["e"] + 1j * self.eta)).real
        return np.where(e0 > self.emin, out, 0.0)

    def _lines(self, k):
        """per external band: pole weights v = 2|c|²·F_bare, the slot-momentum rows and band indices of both lines"""
        se, M = self.se, self.M
        qs = self.ks
        B = len(qs)
        slots = [qs, -k[None, :] - qs]
        fr, eb = [], []
        for p in slots:
            ep, em, _ = se._T(p)
            eb.append(np.concatenate([ep, em], axis=0))                 # rows: +p (b), −p (B + b) ; (2B, M)
            fr.append(np.concatenate([p, -p], axis=0) @ se.lat.T / (2 * np.pi))
        nu = np.arange(2 * M)
        row = lambda b, v: b + B * (v >= M)                             # noqa: E731  ν ≥ M: created α_{−p}
        b_, v2_, v3_ = np.meshgrid(np.arange(B), nu, nu, indexing="ij")
        r2, r3 = row(b_, v2_), row(b_, v3_)
        bd2, bd3 = v2_ % M, v3_ % M
        s2, s3 = np.where(v2_ >= M, 1.0, -1.0), np.where(v3_ >= M, 1.0, -1.0)
        E2, E3 = eb[0][r2, bd2], eb[1][r3, bd3]
        ok = (E2 > self.emin) & (E3 > self.emin)
        n2, n3 = _bose(E2, self.T), _bose(E3, self.T)
        A1 = np.where(s2 > 0, 1 + n2, n2) * np.where(s3 > 0, 1 + n3, n3)
        A2 = np.where(s2 > 0, n2, 1 + n2) * np.where(s3 > 0, n3, 1 + n3)
        out = []
        for n in range(M):
            c = se._coeffs(se.terms3, k, n, slots, B)
            v = 2.0 * np.abs(c) ** 2 * (A1 - A2) * ok
            keep = np.abs(v) > 1e-13 * max(np.abs(v).max(), 1e-300)
            out.append(dict(v=v[keep], r2=r2[keep], r3=r3[keep], b2=bd2[keep], b3=bd3[keep], s2=s2[keep], s3=s3[keep],
                            e=(s2 * E2 + s3 * E3)[keep]))
        return dict(fr=fr, lines=out)

    # ---------------------------------------------------------------- interpolation of grid data
    def _interp(self, data, fr):
        """periodic linear interpolation of (Nk, M) grid data at fractional coordinates fr (P, 3) -> (P, M)"""
        from scipy.ndimage import map_coordinates
        D = data.reshape(tuple(self.n) + (self.M,))
        crd = (np.asarray(fr) * self.n[None, :]).T
        return np.stack([map_coordinates(D[..., m], crd, order=1, mode="grid-wrap") for m in range(self.M)], axis=1)

    def _internal(self, P):
        """renormalised energies and dampings of the internal lines for one external k"""
        if self.renormalise:
            Et = [self._interp(self.E, f) for f in P["fr"]]
        else:
            Et = None
        Gt = [self._interp(self.G, f) for f in P["fr"]]
        return Et, Gt

    # ---------------------------------------------------------------- one (k, n)
    def _solve_one(self, L, eps, delta, Et, Gt, mode, w0, Efix=None):
        """returns (E, Γ, Z, overdamped): the real root of ω − ε − δ − Re Σ(ω) closest to ε + δ (window scan +
        Newton); without a root the minimum of |ω − ε − δ − Σ(ω)| (spectral peak) is used and flagged"""
        if Et is not None and mode == "sc":
            e = L["s2"] * Et[0][L["r2"], L["b2"]] + L["s3"] * Et[1][L["r3"], L["b3"]]
        else:
            e = L["e"]
        g = self.eta
        if mode == "sc":                     # widths on the resonant channels only (decay, absorption), not on the source
            res = (L["s2"] > 0) | (L["s3"] > 0)
            g = self.eta + 0.5 * res * (Gt[0][L["r2"], L["b2"]] + Gt[1][L["r3"], L["b3"]])
        v = L["v"]
        e_b = L["e"]

        def sig_bare(w):
            d = 1.0 / (np.atleast_1d(w)[:, None] - e_b[None, :] + 1j * self.eta)
            return d @ v, -(d * d) @ v

        def sig(w):
            d = 1.0 / (np.atleast_1d(w)[:, None] - e[None, :] + 1j * np.broadcast_to(g, e.shape)[None, :])
            return d @ v, -(d * d) @ v
        if mode == "onshell":
            s0, _ = sig(eps)
            return (eps + delta + s0[0].real) if Efix is None or self.energy_grid is None else Efix, -2 * s0[0].imag, 1.0, False
        c0 = eps + delta
        if mode == "sc":
            # iΓ approximation: 1/S-consistent energy (Re Σ at the bare ε keeps the Hartree–Fock/cubic Goldstone
            # cancellation), damping −2 Im Σ at that energy with damped internal lines
            s_e, ds_e = sig_bare(eps)
            E = c0 + s_e[0].real if Efix is None else Efix
            return E, self._gamma(v, e, g, E if E > 0 else eps), 1.0 / max(1.0 - ds_e[0].real, 1e-6), False
        c1 = c0 + sig(eps)[0][0].real                                       # 1/S (on-shell) estimate
        W = max(0.6 * abs(c1), 0.15 * self.e0.max(), 20 * self.eta)
        ws = np.linspace(max(c1 - W, 0.5 * self.emin), c1 + W, 81)
        s, _ = sig(ws)
        D = ws - c0 - s.real
        idx = np.where(np.sign(D[:-1]) != np.sign(D[1:]))[0]
        if len(idx):
            roots = []
            for j in idx:
                a, b = ws[j], ws[j + 1]
                fa = D[j]
                for _ in range(50):                                         # bisection (robust near poles)
                    m = 0.5 * (a + b)
                    fm = m - c0 - sig(m)[0][0].real
                    if np.sign(fm) == np.sign(fa):
                        a, fa = m, fm
                    else:
                        b = m
                    if b - a < 1e-10 * max(1.0, abs(m)):
                        break
                m = 0.5 * (a + b)
                s0, ds = sig(m)
                if abs(m - c0 - s0[0].real) < 1e-6 * max(1.0, abs(c0)) + 1e-3 * abs(s0[0].imag) + 1e-8:
                    roots.append((abs(m - (c1 if w0 is None else w0)), m, s0[0], ds[0]))   # genuine root, not a pole crossing
            if roots:
                _, w, s0, ds = min(roots)
                return w, -2 * s0.imag, 1.0 / max(1.0 - ds.real, 1e-6), False
        j = int(np.argmin(np.abs(ws - c0 - s) + 1e-9 * np.abs(ws - c1)))
        s0, ds = sig(ws[j])
        return ws[j], -2 * s0[0].imag, 1.0 / max(abs(1.0 - ds[0].real), 1e-6), True

    def _gamma(self, v, e, g, w):
        """Γ = 2π Σ v · δ_g(w − e): Lorentzian (canonical iΓ, long tails) or Gaussian of the same FWHM (default:
        the internal widths broaden only near resonance — no tails from far off-shell poles with large weights)"""
        x = w - e
        if self.profile == "lorentz":
            return float(2 * np.sum(v * g / (x * x + g * g)))
        sg = g / math.sqrt(2 * math.log(2))
        return float(2 * math.pi * np.sum(v * np.exp(-0.5 * (x / sg) ** 2) / (sg * math.sqrt(2 * math.pi))))

    def sweep(self, mode="sc"):
        E, G, Z = self.E.copy(), self.G.copy(), self.Z.copy()
        self.overdamped = np.zeros(self.E.shape, bool)
        for i, P in enumerate(self.poles):
            Et, Gt = self._internal(P) if mode == "sc" else (None, None)
            for n in range(self.M):
                if self.gold[i, n]:
                    E[i, n], G[i, n], Z[i, n] = 0.0, 0.0, 1.0
                    continue
                w0 = self.E[i, n] if mode == "sc" else None
                E[i, n], G[i, n], Z[i, n], self.overdamped[i, n] = self._solve_one(
                    P["lines"][n], self.e0[i, n], self.delta[i, n], Et, Gt, mode, w0, self.E1S[i, n])
        return E, np.maximum(G, 0.0), Z

    def solve(self, mode="sc", maxiter=60, tol=1e-5, mix=0.5, verbose=False):
        """mode 'onshell' | 'dyson' | 'sc' (iterate to self-consistency)"""
        if mode != "sc":
            self.E, self.G, self.Z = self.sweep(mode)
            self.iterations = 1
            self.mode = mode
            return self
        self.E, self.G, self.Z = self.sweep("onshell")                     # start: Born result
        for it in range(1, maxiter + 1):
            E, G, Z = self.sweep("sc")
            dE, dG = np.abs(E - self.E).max(), np.abs(G - self.G).max()
            self.E = mix * E + (1 - mix) * self.E
            self.G = mix * G + (1 - mix) * self.G
            self.Z = Z
            self.history.append((dE, dG))
            if verbose:
                print(f"  iteration {it}: max|ΔE| = {dE:.2e}, max|ΔΓ| = {dG:.2e} meV")
            if max(dE, dG) < tol:
                break
        self.iterations = it
        self.mode = mode
        return self

    def residual(self):
        """max |Γ − (−2 Im Σ(E))| with the current internal widths (and |E − E_1/S|): the self-consistency residual"""
        r = 0.0
        for i, P in enumerate(self.poles):
            Et, Gt = self._internal(P)
            for n in range(self.M):
                if self.gold[i, n] or getattr(self, "overdamped", np.zeros_like(self.gold))[i, n]:
                    continue
                L = P["lines"][n]
                e = (L["s2"] * Et[0][L["r2"], L["b2"]] + L["s3"] * Et[1][L["r3"], L["b3"]]) if Et is not None else L["e"]
                res = (L["s2"] > 0) | (L["s3"] > 0)
                g = self.eta + 0.5 * res * (Gt[0][L["r2"], L["b2"]] + Gt[1][L["r3"], L["b3"]])
                if self.E1S[i, n] <= 0:
                    continue
                r = max(r, abs(self.E[i, n] - self.E1S[i, n]), abs(self.G[i, n] - self._gamma(L["v"], e, g, self.E[i, n])))
        return r

    # ---------------------------------------------------------------- arbitrary momenta
    def at(self, qs, mode=None):
        """(ε, E, Γ, Z) at arbitrary momenta, internal lines from the converged grid solution"""
        mode = mode or getattr(self, "mode", "sc")
        qs = np.atleast_2d(np.asarray(qs, float))
        e0 = self.se._batch(qs)[0]
        dl = self._static(qs) if self.static else np.zeros_like(e0)
        E, G, Z = np.zeros_like(e0), np.zeros_like(e0), np.ones_like(e0)
        Ps = [self._lines(k) for k in qs]
        E1 = self._energies(qs, e0, dl, Ps)
        for i, k in enumerate(qs):
            P = Ps[i]
            Et, Gt = self._internal(P) if mode == "sc" else (None, None)
            for n in range(self.M):
                if e0[i, n] <= self.emin:
                    continue
                E[i, n], G[i, n], Z[i, n], _ = self._solve_one(P["lines"][n], e0[i, n], dl[i, n], Et, Gt, mode, None,
                                                               E1[i, n])
        return e0, E, np.maximum(G, 0), Z

    def spectral(self, k, n, omegas, mode=None, eta0=1e-3):
        """A_n(k, ω) = −(1/π) Im 1/(ω − ε − δ − Σ³(ω)) with the self-consistent internal lines"""
        mode = mode or getattr(self, "mode", "sc")
        k = np.asarray(k, float)
        e0 = self.se._batch(k[None])[0][0, n]
        dl = self._static(k[None])[0, n] if self.static else 0.0
        P = self._lines(k)
        L = P["lines"][n]
        Et, Gt = self._internal(P)
        if mode == "sc":
            e = (L["s2"] * Et[0][L["r2"], L["b2"]] + L["s3"] * Et[1][L["r3"], L["b3"]]) if Et is not None else L["e"]
            g = self.eta + 0.5 * ((L["s2"] > 0) | (L["s3"] > 0)) * (Gt[0][L["r2"], L["b2"]] + Gt[1][L["r3"], L["b3"]])
        else:
            e, g = L["e"], self.eta
        om = np.asarray(omegas, float)
        sig = np.array([np.sum(L["v"] / (w - e + 1j * g)) for w in om])
        return -np.imag(1.0 / (om - e0 - dl - sig + 1j * eta0)) / math.pi


def analyse_interacting(lat, frac, types, symbols, mags, rotations, translations, T=0.0, order="first",
                        grid=(64, 1, 1), n_per=60, path=None, relax=False, cubic=False, quartic=False,
                        se_grid=None, grid4=None, eta=0.05, sigma=None, every=1, sc_decay=None, sc_grid=None,
                        sc_eta=0.02, **model_kw):
    """linear vs Hartree–Fock (four-magnon) renormalised magnon bands along a path; with cubic=True also the
    cubic one-loop + tadpole terms (complete 1/S spectrum ε + δε_HF + Re Σ³ + Σ_tad and the T = 0/thermal decay
    rate Γ³), with quartic=True the second-order four-magnon damping Γ⁴ (and Re Σ⁴), on every `every`-th q;
    sc_decay = 'sc' | 'dyson' | 'onshell': SelfConsistentDecay of the cubic self-energy on the closed grid sc_grid
    (energies E and widths Γ of all internal magnons iterated to self-consistency — no threshold singularities)"""
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
    emin = 1e-3 * float(np.nanmax(e0))                       # Goldstone modes at their zero: on-shell 1/S undefined
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(e0 > emin, e1 / e0, np.nan)
    out = dict(model=hf, q=qs, x=xs, ticks=ticks, omega0=e0, omega=e1, ratio=ratio, T=T, order=order,
               collinear=hf.collinear, m_sub=hf.sublattice_magnetisation(grid, T), iterations=hf.iterations)
    if cubic or quartic:
        se = MagnonSelfEnergy(hf)
        idx = list(range(0, len(qs), max(1, every)))
        if idx[-1] != len(qs) - 1:
            idx.append(len(qs) - 1)
        M = e0.shape[1]
        keys = ("re3", "gamma3", "tad", "re4", "gamma4")
        tab = {kk: np.full((len(qs), M), np.nan) for kk in keys}
        dhf = np.full((len(qs), M), np.nan)
        shift = hf.first_order_shift(qs[idx]) if order == "first" else (e1 - e0)[idx]
        for t, i in enumerate(idx):
            o = se.onshell(qs[i], T, se_grid or grid, eta=eta, sigma=sigma, cubic=cubic, tadpole=cubic,
                           quartic=quartic, grid4=grid4)
            for kk in keys:
                tab[kk][i] = o[kk]
            dhf[i] = shift[t]
        bad = e0 <= emin
        for kk in keys:
            tab[kk][bad] = np.nan
        dhf[bad] = np.nan
        out.update(se=se, idx=np.array(idx), d_hf=dhf, cubic=cubic, quartic=quartic, **tab)
        out["omega_1S"] = e0 + dhf + tab["re3"] + tab["tad"]
        out["omega_int"] = out["omega_1S"] + tab["re4"]
        out["gamma"] = tab["gamma3"] + tab["gamma4"]
        if sc_decay:
            scd = SelfConsistentDecay(se, grid=sc_grid or se_grid or grid, T=T, eta=sc_eta,
                                      energy_grid=se_grid or grid).solve(
                sc_decay, mix=0.3, maxiter=300, tol=1e-6)
            e0p, Ep, Gp, Zp = scd.at(qs[idx])
            om_sc, ga_sc, z_sc = (np.full((len(qs), M), np.nan) for _ in range(3))
            om_sc[idx], ga_sc[idx], z_sc[idx] = Ep, Gp, Zp
            om_sc[bad] = np.nan
            ga_sc[bad] = np.nan
            g4 = np.nan_to_num(tab["gamma4"])
            out.update(scd=scd, omega_sc=om_sc + np.nan_to_num(tab["re4"]), gamma_sc=ga_sc + g4, Z_sc=z_sc, sc_mode=sc_decay)
    return out


def plot_interacting(res, save=None, title=None, figsize=(7.4, 4.6)):
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    if "omega_1S" in res:
        return _plot_selfenergy(res, save, title)
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


def _plot_selfenergy(res, save=None, title=None, figsize=(7.4, 6.4)):
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    i = res["idx"]
    x = res["x"]
    for n in range(res["omega0"].shape[1]):
        ax.plot(x, res["omega0"][:, n], "--", color="0.55", lw=1.3, label="linear spin waves" if n == 0 else None)
        w = res["omega_int"][i, n]
        g = res["gamma"][i, n]
        ax.plot(x[i], w, color="#c0392b", lw=2.0, label="interacting (1/S + Σ)" if n == 0 else None)
        ax.fill_between(x[i], w - g / 2, w + g / 2, color="#c0392b", alpha=0.25, lw=0)
        if res.get("cubic"):
            ax2.plot(x[i], res["gamma3"][i, n], color="#16325c", lw=1.8, label="Γ³ (cubic: decay)" if n == 0 else None)
        if res.get("quartic"):
            ax2.plot(x[i], res["gamma4"][i, n], color="#e67e22", lw=1.8, label="Γ⁴ (four-magnon)" if n == 0 else None)
        if "omega_sc" in res:
            ws_, gs_ = res["omega_sc"][i, n], res["gamma_sc"][i, n]
            ax.plot(x[i], ws_, color="#16325c", lw=1.8, ls="-.", label=f"self-consistent decay ({res['sc_mode']})" if n == 0 else None)
            ax.fill_between(x[i], ws_ - gs_ / 2, ws_ + gs_ / 2, color="#16325c", alpha=0.2, lw=0)
            ax2.plot(x[i], gs_, color="#16325c", lw=1.8, ls="-.", label="Γ self-consistent" if n == 0 else None)
    for a in (ax, ax2):
        for t, _ in res["ticks"]:
            a.axvline(t, color="0.85", lw=0.7, zorder=0)
    ax2.set_xticks([t for t, _ in res["ticks"]])
    ax2.set_xticklabels([lab for _, lab in res["ticks"]])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("magnon energy (meV)")
    ax2.set_ylabel("Γ = −2 Im Σ (meV)")
    ax2.set_ylim(bottom=0)
    ax.legend(fontsize=8.5)
    ax2.legend(fontsize=8.5)
    ax.set_title(title or f"magnon interactions beyond Hartree–Fock (T = {res['T']:g} K)", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig
