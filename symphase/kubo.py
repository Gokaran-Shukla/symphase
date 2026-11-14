# -*- coding: utf-8 -*-
"""
symphase.kubo — magnitudes from Wannier-based Kubo codes, constrained by symmetry
=================================================================================

SymPhase gives the SHAPE of a response tensor (which components exist and how they are related).
Magnitudes come from Kubo calculations on a Wannier Hamiltonian.  This module links the two:

plan     the minimal set of components that a Kubo code has to compute (the pivot components of the
         symmetry-allowed basis) + ready-to-paste postw90 input blocks (berry_task = ahc / shc with
         shc_alpha, shc_beta, shc_gamma) and a check component;
import   read results of postw90 (seedname-ahc-fermiscan.dat, seedname-shc-fermiscan.dat + seedname.win),
         or any table 'component value' (e.g. WannierBerri or in-house OHC/SHC codes);
fit      least-squares fit of the independent parameters → the FULL tensor (missing components filled
         by symmetry) and the symmetry violation of the computed data (convergence / frame check);
compute  a built-in reference Kubo solver on seedname_hr.dat (tight-binding convention, optional Wannier
         centres): intrinsic anomalous Hall σ_ij (Berry curvature) and spin Hall σ^k_ij (spin current
         ½{σ_k, v_i}), only for the components that symmetry requires.

Conventions (as in postw90):  σ_ij = −(e²/ħ) ∫ Σ_n f_n Ω_n,ij,
    Ω_n,ij = −2 Im Σ_{m≠n} ⟨n|∂_i H|m⟩⟨m|∂_j H|n⟩ / ((E_n − E_m)² + η²);
    spin: ∂_i H → ½{σ_k, ∂_i H} and an extra factor ½ (units ħ/e).
Units: 3D S/cm and (ħ/e) S/cm;  2D (dims=2) e²/h and (ħ/e) e²/h.
J_i = σ_ij E_j,  J^{S_k}_i = σ^k_ij E_j  (i flow, j field, k polarisation) = postw90 shc_alpha = i,
shc_beta = j, shc_gamma = k.
"""

import json
import math
import os
import re

import numpy as np

AX = "xyz"
E2_HBAR_S = 2.434134807e-4          # e²/ħ in S
SCM = E2_HBAR_S * 1e8               # (e²/ħ)/Å  in S/cm


# ----------------------------------------------------------------------------
#  symmetry-allowed tensor
# ----------------------------------------------------------------------------
def comp_index(name):
    """'σ^z_xy' / 's^z_xy' / 'sigma^z_xy' / 'z,x,y' -> (rank, index);  'σ_xy' -> rank 2"""
    n = name.strip()
    m = re.search(r"\^\{?([xyz])\}?_\{?([xyz])([xyz])\}?", n)
    if m:
        k, i, j = (AX.index(c) for c in m.groups())
        return 3, k * 9 + i * 3 + j
    m = re.search(r"_\{?([xyz])([xyz])\}?$", n)
    if m:
        i, j = (AX.index(c) for c in m.groups())
        return 2, i * 3 + j
    m = re.fullmatch(r"([xyz]),?([xyz]),?([xyz])", n)
    if m:
        k, i, j = (AX.index(c) for c in m.groups())
        return 3, k * 9 + i * 3 + j
    raise ValueError(f"cannot parse component name {name!r}")


def comp_name(rank, c, sym="σ"):
    if rank == 2:
        return f"{sym}_{AX[c // 3]}{AX[c % 3]}"
    return f"{sym}^{AX[c // 9]}_{AX[(c // 3) % 3]}{AX[c % 3]}"


class SymTensor:
    """allowed subspace: rows of B (n_par x ncomp); tensor = p @ B, p = values of the pivot components"""

    def __init__(self, rank, blocks, kind="spin", source="magnetic point group (SOC)"):
        self.rank, self.kind, self.source = rank, kind, source
        self.ncomp = 9 if rank == 2 else 27
        rows, piv, part = [], [], []
        for lab, (E, pv) in blocks.items():
            for r, p in enumerate(pv):
                rows.append(np.asarray(E[r], float))
                piv.append(int(p))
                part.append(lab)
        self.B = np.array(rows).reshape(len(rows), self.ncomp)
        self.piv, self.part = piv, part

    @classmethod
    def from_analysis(cls, res, kind="spin", parts=("even", "odd"), nr=False):
        """res = analyze_tensors(...) result; nr=True uses the spin group (no SOC)"""
        t = (res["spin_group"]["tensors"] if nr else res["tensors"]).get(kind)
        if t is None or "basis" not in t:
            raise ValueError(f"no {kind} tensor in the analysis (use tensors='charge,spin,orbital')")
        blocks = {p: (t["basis"][p]["E"], t["basis"][p]["piv"]) for p in parts if t["basis"].get(p)}
        return cls(2 if kind == "charge" else 3, blocks, kind,
                   "spin group (no SOC)" if nr else "magnetic point group (SOC)")

    @property
    def n_par(self):
        return len(self.piv)

    def names(self, sym="σ"):
        return [comp_name(self.rank, c, sym + ("" if self.kind != "orbital" else "L")) for c in range(self.ncomp)]

    def minimal_set(self):
        return list(self.piv)

    def check_component(self):
        """a non-zero component that is NOT a pivot (verifies the frame/convergence of a Kubo run)"""
        nz = [c for c in range(self.ncomp) if np.abs(self.B[:, c]).max() > 1e-9 and c not in self.piv]
        return nz[0] if nz else None

    def fit(self, values):
        """values: {component index or name: number}.  Returns dict(params, full, residual, violation, ...)"""
        idx, y = [], []
        for k, v in values.items():
            c = k if isinstance(k, (int, np.integer)) else comp_index(k)[1]
            idx.append(int(c))
            y.append(float(v))
        idx, y = np.array(idx), np.array(y)
        A = self.B[:, idx].T                      # (n_given, n_par)
        rank = np.linalg.matrix_rank(A, tol=1e-8) if A.size else 0
        p, *_ = np.linalg.lstsq(A, y, rcond=None) if A.size else (np.zeros(self.n_par),)
        full = p @ self.B if self.n_par else np.zeros(self.ncomp)
        fit_y = A @ p if A.size else np.zeros_like(y)
        forbidden = [int(c) for c in idx if np.abs(self.B[:, c]).max() < 1e-9 and abs(values_get(values, c)) > 0]
        scale = max(np.abs(y).max() if y.size else 0.0, 1e-30)
        return dict(params=p, full=full, given=idx, residual=y - fit_y,
                    violation=float(np.abs(y - fit_y).max() / scale) if y.size else 0.0,
                    determined=bool(rank == self.n_par), rank=int(rank), forbidden_nonzero=forbidden,
                    even=sum(p[r] * self.B[r] for r in range(self.n_par) if self.part[r] == "even") if self.n_par
                    else np.zeros(self.ncomp),
                    odd=sum(p[r] * self.B[r] for r in range(self.n_par) if self.part[r] == "odd") if self.n_par
                    else np.zeros(self.ncomp))


def values_get(values, c):
    for k, v in values.items():
        cc = k if isinstance(k, (int, np.integer)) else comp_index(k)[1]
        if cc == c:
            return float(v)
    return 0.0


# ----------------------------------------------------------------------------
#  planning: what must a Kubo code compute?
# ----------------------------------------------------------------------------
def plan(st, sym="σ"):
    """text plan + postw90 blocks for the minimal component set"""
    nm = st.names(sym)
    lines = [f"{st.kind} tensor, {st.source}: {st.n_par} independent component(s)"]
    if st.n_par == 0:
        lines.append("  nothing to compute: every component vanishes by symmetry")
        return "\n".join(lines), []
    lines.append("  compute these (pivots):  " + ", ".join(f"{nm[c]} [{p}]" for c, p in zip(st.piv, st.part)))
    chk = st.check_component()
    if chk is not None:
        lines.append(f"  check component (must follow from the pivots): {nm[chk]}")
    blocks = []
    if st.kind == "charge":
        if any(p == "odd" for p in st.part):
            blocks.append("berry = true\nberry_task = ahc        # anomalous Hall vector (σ_yz, σ_zx, σ_xy) in one run")
        if any(p == "even" for p in st.part):
            lines.append("  T-even (symmetric) part: Drude/Boltzmann (e.g. postw90 BoltzWann), not an intrinsic Kubo term")
    elif st.kind == "spin":
        for c in st.piv + ([chk] if chk is not None else []):
            k, i, j = c // 9, (c // 3) % 3, c % 3
            blocks.append(f"berry = true\nberry_task = shc\nshc_alpha = {i + 1}   # flow {AX[i]}\n"
                          f"shc_beta  = {j + 1}   # field {AX[j]}\nshc_gamma = {k + 1}   # spin {AX[k]}"
                          + ("     (check)" if c == chk else ""))
        if any(p == "odd" for p in st.part):
            lines.append("  T-odd spin components (magnetic spin splitter) are Fermi-surface terms ∝ 1/Γ: "
                         "not given by the intrinsic Kubo (Berry-type) formula")
    else:
        lines.append("  orbital Hall: postw90 has no OHC task — use WannierBerri / an in-house OHC code for the "
                     "pivot components and 'symphase kubo import' the results")
    return "\n".join(lines), blocks


# ----------------------------------------------------------------------------
#  readers
# ----------------------------------------------------------------------------
def _data_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
    return rows


def read_postw90_ahc(path):
    """seedname-ahc-fermiscan.dat: E_F, σ_x, σ_y, σ_z  (Hall vector σ_x = σ_yz, ...; S/cm)
    -> {E_F: {component name: value}} with σ_ji = −σ_ij"""
    out = {}
    for r in _data_rows(path):
        if len(r) < 4:
            continue
        ef, hx, hy, hz = r[-4:]
        out[ef] = {"σ_yz": hx, "σ_zy": -hx, "σ_zx": hy, "σ_xz": -hy, "σ_xy": hz, "σ_yx": -hz}
    return out


def read_win_shc(win):
    vals = {}
    with open(win) as fh:
        for line in fh:
            m = re.match(r"\s*shc_(alpha|beta|gamma)\s*[=:]?\s*(\d)", line, re.I)
            if m:
                vals[m.group(1).lower()] = int(m.group(2))
    if len(vals) != 3:
        raise ValueError(f"{win}: shc_alpha/beta/gamma not all found")
    return vals["alpha"], vals["beta"], vals["gamma"]


def read_postw90_shc(path, alpha=None, beta=None, gamma=None, win=None):
    """seedname-shc-fermiscan.dat (rows: [index,] E_F, SHC in (ħ/e) S/cm).  The component is
    σ^gamma_{alpha beta}, from the arguments or seedname.win.  -> {E_F: {name: value}}"""
    if alpha is None:
        if win is None:
            raise ValueError("give shc_alpha/beta/gamma or the seedname.win file")
        alpha, beta, gamma = read_win_shc(win)
    name = comp_name(3, (gamma - 1) * 9 + (alpha - 1) * 3 + (beta - 1))
    out = {}
    for r in _data_rows(path):
        if len(r) >= 2:
            out[r[-2]] = {name: r[-1]}
    return out


def read_components(path):
    """generic: JSON {name: value} or lines 'name value' (e.g. results of WannierBerri / own codes)"""
    if path.endswith(".json"):
        with open(path) as fh:
            return {k: float(v) for k, v in json.load(fh).items()}
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if line:
                t = line.replace(",", " ").split()
                out[t[0]] = float(t[1])
    return out


def merge_scans(*scans, tol=1e-6):
    """merge several {E_F: {name: v}} dicts (e.g. one postw90 SHC run per component)"""
    out = {}
    for sc in scans:
        for ef, comps in sc.items():
            key = next((k for k in out if abs(k - ef) < tol), ef)
            out.setdefault(key, {}).update(comps)
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------
#  reference Kubo solver on a Wannier tight-binding Hamiltonian
# ----------------------------------------------------------------------------
def read_hr(path):
    """Wannier90 seedname_hr.dat -> (R (nR,3) int, H (nR, nw, nw) complex, eV), degeneracies applied"""
    with open(path) as fh:
        fh.readline()
        nw = int(fh.readline())
        nR = int(fh.readline())
        deg = []
        while len(deg) < nR:
            deg += [int(x) for x in fh.readline().split()]
        data = np.loadtxt(fh)
    data = data.reshape(nR, nw * nw, 7)
    R = data[:, 0, :3].astype(int)
    H = np.zeros((nR, nw, nw), complex)
    m = data[:, :, 3].astype(int) - 1
    n = data[:, :, 4].astype(int) - 1
    for r in range(nR):
        H[r, m[r], n[r]] = (data[r, :, 5] + 1j * data[r, :, 6]) / deg[r]
    return R, H


def write_hr(path, R, H, comment="written by SymPhase"):
    nR, nw, _ = H.shape
    with open(path, "w") as fh:
        fh.write(comment + "\n" + f"{nw}\n{nR}\n")
        for k in range(0, nR, 15):
            fh.write(" ".join("1" for _ in range(min(15, nR - k))) + "\n")
        for r in range(nR):
            for n in range(nw):
                for m in range(nw):
                    h = H[r, m, n]
                    fh.write(f"{R[r, 0]:5d}{R[r, 1]:5d}{R[r, 2]:5d}{m + 1:5d}{n + 1:5d}{h.real:16.10f}{h.imag:16.10f}\n")


def read_centres(path):
    """seedname_centres.xyz: Wannier centres ('X x y z' lines, Å)"""
    out = []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if len(t) == 4 and t[0] == "X":
                out.append([float(x) for x in t[1:]])
    return np.array(out)


def pauli(spin_order, nw):
    if nw % 2:
        raise ValueError("spin operators need an even number of Wannier functions (spinors)")
    s = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
    h = nw // 2
    if spin_order == "blocks":                       # all up, then all down
        return [np.kron(p, np.eye(h)) for p in s]
    if spin_order == "interleaved":                  # (orbital 1 ↑, orbital 1 ↓, ...)
        return [np.kron(np.eye(h), p) for p in s]
    raise ValueError("spin_order must be 'blocks' or 'interleaved' (check your Wannier90 projections)")


class KuboTB:
    """intrinsic Kubo conductivities of a Wannier tight-binding model"""

    def __init__(self, lat, R, H, centres=None, dims=3, spin_order=None):
        self.lat = np.asarray(lat, float)
        self.R = np.asarray(R, int)
        self.H = np.asarray(H, complex)
        self.nw = self.H.shape[1]
        self.Rc = self.R @ self.lat                       # Cartesian lattice vectors
        self.tau = np.zeros((self.nw, 3)) if centres is None else np.asarray(centres, float)
        self.dims = dims
        self.sig = pauli(spin_order, self.nw) if spin_order else None

    def _hk(self, ks):
        """H(k), ∂_x,y,z H(k) for a batch of Cartesian k (nk, 3)"""
        ph = np.exp(1j * ks @ self.Rc.T)                  # (nk, nR)
        H0 = np.einsum("kr,rmn->kmn", ph, self.H)
        dH = [np.einsum("kr,rmn->kmn", 1j * ph * self.Rc[:, a][None, :], self.H) for a in range(3)]
        if np.abs(self.tau).max() > 0:
            D = np.exp(1j * ks @ self.tau.T)              # (nk, nw)
            dt = self.tau[None, :, :] - self.tau[:, None, :]   # τ_n − τ_m  (nw, nw, 3)
            Hc = np.conj(D)[:, :, None] * H0 * D[:, None, :]
            dH = [np.conj(D)[:, :, None] * dH[a] * D[:, None, :] + 1j * dt[None, :, :, a] * Hc for a in range(3)]
            H0 = Hc
        return H0, dH

    def grid(self, n):
        n = list(n) + [1] * (3 - len(n))
        if self.dims == 2:
            n[2] = 1
        f = [np.arange(m) / m for m in n]
        F = np.array(np.meshgrid(*f, indexing="ij")).reshape(3, -1).T
        B = 2 * np.pi * np.linalg.inv(self.lat).T
        return F @ B

    def curvature(self, ks, comps, eta=1e-3, chunk=400):
        """band-resolved Ω for comps = [(kind, k, i, j)] with kind 'charge' | 'spin' -> E (nk, nw), Ω (nc, nk, nw)"""
        nk = len(ks)
        Eall = np.zeros((nk, self.nw))
        Om = np.zeros((len(comps), nk, self.nw))
        for s in range(0, nk, chunk):
            kk = ks[s:s + chunk]
            H0, dH = self._hk(kk)
            E, U = np.linalg.eigh(H0)
            Ud = np.conj(np.transpose(U, (0, 2, 1)))
            V = [Ud @ d @ U for d in dH]
            dE = E[:, :, None] - E[:, None, :]
            den = dE ** 2 + eta ** 2
            off = 1.0 - np.eye(self.nw)[None]
            Eall[s:s + chunk] = E
            for c, (kind, k, i, j) in enumerate(comps):
                if kind == "charge":
                    A = V[i]
                else:
                    if self.sig is None:
                        raise ValueError("spin Hall needs spin_order ('blocks' or 'interleaved')")
                    sk = self.sig[k]
                    A = Ud @ (0.5 * (sk @ dH[i] + dH[i] @ sk)) @ U
                Bm = V[j]
                term = A * np.transpose(Bm, (0, 2, 1)) * off / den            # A_nm B_mn
                Om[c, s:s + chunk] = -2.0 * np.imag(term.sum(axis=2))
        return Eall, Om

    def conductivity(self, comps, efs, n=(40, 40, 40), eta=1e-3, kT=0.0):
        """σ for comps at the Fermi energies efs.  3D: S/cm ((ħ/e) S/cm for spin); 2D: e²/h"""
        ks = self.grid(n)
        E, Om = self.curvature(ks, comps, eta)
        nk = len(ks)
        if self.dims == 2:
            A = np.linalg.norm(np.cross(self.lat[0], self.lat[1]))
            pref = -2 * np.pi / (A * nk)
        else:
            V = abs(np.linalg.det(self.lat))
            pref = -SCM / (V * nk)
        out = np.zeros((len(efs), len(comps)))
        for a, ef in enumerate(efs):
            f = (E <= ef).astype(float) if kT <= 0 else 1.0 / (np.exp(np.clip((E - ef) / kT, -60, 60)) + 1.0)
            out[a] = pref * np.einsum("ckn,kn->c", Om, f)
        for c, (kind, *_r) in enumerate(comps):
            if kind == "spin":
                out[:, c] *= 0.5
        return out


def compute_symmetric(model, st, efs, n=(40, 40, 40), eta=1e-3, kT=0.0, with_check=True):
    """compute only the pivot components, fill the tensor by symmetry, then verify with the non-pivot
    component of largest predicted magnitude (an independent Kubo evaluation)"""
    cs = st.minimal_set()

    def comp(c):
        return ("charge", 0, c // 3, c % 3) if st.rank == 2 else ("spin", c // 9, (c // 3) % 3, c % 3)
    vals = model.conductivity([comp(c) for c in cs], efs, n, eta, kT) if cs else np.zeros((len(efs), 0))
    fits = [st.fit({c: vals[a, k] for k, c in enumerate(cs)}) for a in range(len(efs))]
    chk = None
    if with_check and cs:
        pred = np.max([np.abs(f["full"]) for f in fits], axis=0)
        pred[cs] = -1
        if pred.max() > 1e-12:
            chk = int(np.argmax(pred))
        else:
            chk = st.check_component()
    if chk is not None:
        cv = model.conductivity([comp(chk)], efs, n, eta, kT)[:, 0]
    for a, (fit, ef) in enumerate(zip(fits, efs)):
        fit["E_F"] = ef
        if chk is not None:
            pr = float(fit["full"][chk])
            fit["check"] = dict(component=chk, computed=float(cv[a]), predicted=pr,
                                rel_error=float(abs(cv[a] - pr) / max(abs(cv[a]), abs(pr), 1e-12)))
    return fits


def format_fit(st, fit, sym="σ", unit=""):
    nm = st.names(sym)
    lines = []
    if st.rank == 2:
        T = fit["full"].reshape(3, 3)
        lines.append(f"  full tensor {sym}_ij ({unit}):")
        for i in range(3):
            lines.append("    [" + "  ".join(f"{T[i, j]:12.5g}" for j in range(3)) + " ]")
    else:
        T = fit["full"].reshape(3, 3, 3)
        for k in range(3):
            if np.abs(T[k]).max() > 0:
                lines.append(f"  {sym}^{AX[k]}_ij ({unit}):")
                for i in range(3):
                    lines.append("    [" + "  ".join(f"{T[k, i, j]:12.5g}" for j in range(3)) + " ]")
    lines.append("  independent parameters: " + ", ".join(f"{nm[c]} = {p:.5g}" for c, p in zip(st.piv, fit["params"])))
    lines.append(f"  symmetry violation of the input: {100 * fit['violation']:.3g} %"
                 + ("" if fit["determined"] else f"   (UNDERDETERMINED: {fit['rank']}/{st.n_par} parameters fixed — "
                                                  "compute the missing pivots)"))
    if fit["forbidden_nonzero"]:
        lines.append("  WARNING non-zero components that symmetry forbids: "
                     + ", ".join(nm[c] for c in fit["forbidden_nonzero"]) + " (frame / convergence?)")
    if "check" in fit:
        ck = fit["check"]
        lines.append(f"  check {nm[ck['component']]}: computed {ck['computed']:.5g}, from symmetry {ck['predicted']:.5g}"
                     f"  (rel. error {100 * ck['rel_error']:.2g} %)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
#  model Hamiltonians (tests, tutorials)
# ----------------------------------------------------------------------------
def honeycomb_lattice(a=2.46, c=20.0):
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2, a * math.sqrt(3) / 2, 0.0])
    return np.array([a1, a2, [0, 0, c]]), np.array([[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]])


def haldane_hr(t1=1.0, t2=0.1, phi=math.pi / 2, M=0.0):
    """Haldane model on the honeycomb lattice -> (R, H) in Wannier90 hr form (2 orbitals)"""
    terms = {}

    def add(R, m, n, v):
        terms.setdefault(tuple(R), np.zeros((2, 2), complex))
        terms[tuple(R)][m, n] += v
    add((0, 0, 0), 0, 0, M)
    add((0, 0, 0), 1, 1, -M)
    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:          # A(0) -> B(R)
        add(R, 0, 1, t1)
        add(tuple(-np.array(R)), 1, 0, t1)
    # next-nearest neighbours: same sublattice, phase +φ for A along a1, a2−a1, −a2 (counter-clockwise), −φ for B
    for R, s in [((1, 0, 0), 1), ((-1, 1, 0), 1), ((0, -1, 0), 1)]:
        for m, sg in ((0, 1), (1, -1)):
            v = t2 * np.exp(1j * sg * s * phi)
            add(R, m, m, v)
            add(tuple(-np.array(R)), m, m, np.conj(v))
    Rs = sorted(terms)
    return np.array(Rs), np.array([terms[r] for r in Rs])


def kane_mele_hr(t1=1.0, lso=0.06, M=0.0):
    """S_z-conserving Kane–Mele model = Haldane(φ = π/2, t2 = λ_SO) for ↑ and its time reverse for ↓;
    4 orbitals in 'blocks' spin order (A↑, B↑, A↓, B↓)"""
    R, Hu = haldane_hr(t1, lso, math.pi / 2, M)
    _, Hd = haldane_hr(t1, lso, -math.pi / 2, M)
    H = np.zeros((len(R), 4, 4), complex)
    H[:, :2, :2] = Hu
    H[:, 2:, 2:] = Hd
    return R, H


# ----------------------------------------------------------------------------
#  command line
# ----------------------------------------------------------------------------
def main(argv=None):
    import argparse
    from .tensors import analyze_tensors
    ap = argparse.ArgumentParser(prog="symphase kubo", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Magnitudes from Wannier-based Kubo codes, constrained by symmetry.",
                                 epilog="examples:\n"
                                        "  symphase kubo plan --kind spin                      # what postw90 must compute\n"
                                        "  symphase kubo import --kind spin --shc Pt-shc-fermiscan.dat --win Pt.win\n"
                                        "  symphase kubo import --kind charge --ahc Fe-ahc-fermiscan.dat --ef 0\n"
                                        "  symphase kubo import --kind orbital --table ohc_components.txt\n"
                                        "  symphase kubo compute --kind spin --hr kane_mele_hr.dat --centres centres.xyz "
                                        "--spin-order blocks --dims 2 --grid 60 60")
    ap.add_argument("action", choices=("plan", "import", "compute"))
    ap.add_argument("-d", "--dir", default=".", help="VASP folder (POSCAR, INCAR/OUTCAR) [.]")
    ap.add_argument("--kind", choices=("charge", "spin", "orbital"), default="spin")
    ap.add_argument("--nr", action="store_true", help="use the spin group (no SOC) instead of the magnetic group")
    ap.add_argument("--part", choices=("both", "even", "odd"), default=None,
                    help="tensor part to fit [compute: charge odd, spin/orbital even; otherwise both]")
    ap.add_argument("--magmom", default=None)
    ap.add_argument("--ahc", nargs="+", default=[], help="postw90 seedname-ahc-fermiscan.dat")
    ap.add_argument("--shc", nargs="+", default=[], help="postw90 seedname-shc-fermiscan.dat (one per component)")
    ap.add_argument("--win", nargs="+", default=[], help="seedname.win for each --shc file (shc_alpha/beta/gamma)")
    ap.add_argument("--table", nargs="+", default=[], help="'component value' lines or JSON (any Kubo code)")
    ap.add_argument("--ef", type=float, nargs="+", default=None, help="Fermi energies (eV) to report / compute")
    ap.add_argument("--hr", default=None, help="Wannier90 seedname_hr.dat (compute)")
    ap.add_argument("--centres", default=None, help="Wannier centres (seedname_centres.xyz) for the TB convention")
    ap.add_argument("--spin-order", choices=("blocks", "interleaved"), default=None)
    ap.add_argument("--dims", type=int, choices=(2, 3), default=3)
    ap.add_argument("--grid", type=int, nargs="+", default=[40, 40, 40])
    ap.add_argument("--eta", type=float, default=1e-3, help="regulariser (eV) [1e-3]")
    ap.add_argument("--kT", type=float, default=0.0, help="Fermi smearing kT (eV) [0]")
    args = ap.parse_args(argv)
    res = analyze_tensors(args.dir, tensors=args.kind, magmom=args.magmom, ossg="no")
    part = args.part or ("both" if args.action != "compute" else ("odd" if args.kind == "charge" else "even"))
    parts = ("even", "odd") if part == "both" else (part,)
    st = SymTensor.from_analysis(res, args.kind, parts=parts, nr=args.nr)
    sym = "σ"
    print(f"SymPhase kubo — {args.kind} tensor of {res['structure'].get('symbols', [''])[0]}…  "
          f"({'spin group' if args.nr else 'magnetic group ' + res['magnetic']['bns_symbol']}), parts: {', '.join(parts)}")
    txt, blocks = plan(st, sym)
    print(txt)
    if args.action == "plan":
        for b in blocks:
            print("\n  --- postw90 (seedname.win) ---\n" + "\n".join("  " + l for l in b.splitlines()))
        return st
    unit = {"charge": "S/cm", "spin": "(ħ/e) S/cm", "orbital": "(ħ/e) S/cm"}[args.kind]
    if args.action == "import":
        scans = [read_postw90_ahc(f) for f in args.ahc]
        for k, f in enumerate(args.shc):
            scans.append(read_postw90_shc(f, win=args.win[k] if k < len(args.win) else None))
        tab = {}
        for f in args.table:
            tab.update(read_components(f))
        data = merge_scans(*scans) if scans else {}
        if tab:
            data = {0.0: tab} if not data else {e: {**v, **tab} for e, v in data.items()}
        if not data:
            raise SystemExit("nothing to import: give --ahc, --shc or --table")
        efs = sorted(data) if args.ef is None else [min(data, key=lambda e: abs(e - x)) for x in args.ef]
        out = []
        for ef in efs[:3] if args.ef is None and len(efs) > 3 else efs:
            fit = st.fit(data[ef])
            print(f"\n  E_F = {ef:+.4f} eV:  {len(data[ef])} computed component(s)")
            print(format_fit(st, fit, sym, unit))
            out.append(fit)
        if args.ef is None and len(efs) > 3:
            print(f"\n  ({len(efs)} Fermi energies in the scan; showing 3 — select with --ef)")
        return out
    if not args.hr:
        raise SystemExit("compute needs --hr seedname_hr.dat")
    R, H = read_hr(args.hr)
    lat = np.array(res["structure"]["lattice"])
    cen = read_centres(args.centres) if args.centres else None
    model = KuboTB(lat, R, H, centres=cen, dims=args.dims, spin_order=args.spin_order)
    unit = ("e²/h" if args.kind == "charge" else "(ħ/e) e²/h") if args.dims == 2 else unit
    fits = compute_symmetric(model, st, args.ef or [0.0], n=args.grid, eta=args.eta, kT=args.kT)
    for fit in fits:
        print(f"\n  E_F = {fit['E_F']:+.4f} eV  (Kubo on {args.hr}: {model.nw} Wannier functions, grid {args.grid})")
        print(format_fit(st, fit, sym, unit))
    return fits
