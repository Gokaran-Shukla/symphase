#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symphase.tensors
================

Read a VASP calculation (POSCAR/CONTCAR, INCAR, OSZICAR, OUTCAR), perform a full
crystal + magnetic symmetry analysis (relativistic magnetic groups AND
non-relativistic spin groups) and print the symmetry-allowed shape of

    * charge  conductivity  sigma_ij            (rank-2 polar tensor)
    * spin    conductivity  sigma^k_ij  (J^{S_k}_i = sigma^k_ij E_j)
    * orbital conductivity  sigma^k_ij  (J^{L_k}_i = sigma^k_ij E_j)

split into the time-reversal-even (T-even) and time-reversal-odd (T-odd)
parts, following Neumann's principle for magnetic point groups.

Symmetry output
---------------
  * crystal space group (Hermann-Mauguin, number, Hall, point group, Wyckoff)
  * magnetic space group (BNS + OG symbol and number, UNI number, MSG type I-IV)
  * magnetic point group and its unitary subgroup
  * full list of magnetic symmetry operations (rotation, screw, mirror,
    glide, rotoinversion; with or without time reversal T)
  * flags: inversion P, time reversal T, antitranslation T*tau, PT,
    mirror planes, M*T planes, C_n*T axes, polar / chiral, ...

Tensor conventions
------------------
  Cartesian frame = POSCAR frame (default) or spglib standardized frame
  (--frame std).

  charge:   J_i = sigma_ij E_j,       D(g) = R (x) R
  spin:     J^{s_k}_i = sigma^k_ij E_j, D(g) = det(R) R (x) R (x) R
            (spin current {v_i, s_k}/2 is polar(i) x axial(k))
  orbital:  same transformation as spin (L is axial and T-odd)

  For a magnetic operation g = {R|t} or g' = {R|t}T:
      T-even part :  sigma = D(R) sigma            for g and g'
      T-odd  part :  sigma = theta(g) D(R) sigma   theta = +1 (g), -1 (g')
  (decomposition into parts even / odd in the magnetic order, as in e.g.
  Seemann et al., PRB 92, 155138 (2015); Zelezny et al., "symmetr").
  For the charge conductivity the Onsager relation sigma_ij(M)=sigma_ji(-M)
  is imposed as well: T-even part symmetric, T-odd part antisymmetric
  (anomalous Hall).  No Onsager constraint is imposed on spin/orbital
  conductivities (the spin current is not a thermodynamic flux).

  Physical reading (clean, constant-broadening Kubo-Bastin limit):
      T-even spin/orbital: intrinsic SHE/OHE (Fermi sea, Berry-curvature like)
      T-odd  spin/orbital: magnetic SHE/OHE, Fermi-surface (~1/Gamma) part
      T-even charge: Drude-like longitudinal; T-odd charge: anomalous Hall

Magnetic moments
----------------
  priority:  --magmom (user)  >  OUTCAR site moments ('magnetization (x/y/z)')
             >  INCAR MAGMOM  >  MAGMOM echoed in OUTCAR  >  zero (non-magnetic)
  ISPIN, LSORBIT, LNONCOLLINEAR, SAXIS are taken from OUTCAR, else from INCAR,
  so a POSCAR + INCAR pair is enough for a symmetry analysis before running VASP.

Spin groups (--analysis spin|both): see symphase.spingroup — magnet class
(FM / AFM / PT-AFM / altermagnet), spin Laue group, d/g/i-wave spin splitting
s(k), nodal planes, non-relativistic tensors and the components that need SOC.
  collinear runs (ISPIN=2, no SOC):  m_i = M_i * direction,
             direction = --mdir (default: SAXIS from OUTCAR, else z)
  non-collinear runs: OUTCAR moments are in the SAXIS spinor frame and are
             rotated to Cartesian coordinates.
  The magnetic group always treats moments as axial vectors coupled to the
  lattice, i.e. the analysis is the relativistic (spin-orbit coupled) one.
  Without SOC the true symmetry is a (larger) spin group.

Requirements: python >= 3.8, numpy, spglib >= 2.0
Usage:  symphase tensors [-p POSCAR] [-o OUTCAR] [-z OSZICAR] [options]
        symphase tensors -h
        python: from symphase import analyze_tensors
"""

import argparse
import json
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np

try:
    import spglib
except ImportError:  # pragma: no cover
    sys.exit("spglib is required:  pip install spglib")

from . import __version__

AX = "xyz"
SQ3 = math.sqrt(3.0)
EYE = np.eye(3)


# ============================================================================
#  small helpers
# ============================================================================
def _g(obj, key):
    """attribute or dict access (spglib < 2.5 returns dicts)"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def expand_vasp_list(tokens):
    """expand VASP 'N*val' notation"""
    out = []
    for t in tokens:
        if "*" in t:
            n, v = t.split("*")
            out += [float(v)] * int(float(n))
        else:
            out.append(float(t))
    return out


def clean_species(s):
    s = s.split("_")[0].split("/")[0].split(":")[0]
    return s[:1].upper() + s[1:].lower() if s else s


def small_int_vector(v, maxint=12, tol=2e-3):
    """try to express direction v as small integers [uvw]"""
    v = np.asarray(v, float)
    if np.linalg.norm(v) < 1e-12:
        return None
    v = v / np.max(np.abs(v))
    for m in range(1, maxint + 1):
        w = v * m
        r = np.round(w)
        if np.max(np.abs(w - r)) < tol * m and np.any(r != 0):
            r = r.astype(int)
            g = np.gcd.reduce(np.abs(r[r != 0]))
            r = r // g
            # canonical sign: first nonzero positive
            nz = r[np.nonzero(r)[0][0]]
            if nz < 0:
                r = -r
            return r
    return None


def fmt_dir(r, brackets="[]"):
    if r is None:
        return "irrational"
    s = "".join(("-" + str(-x)) if x < 0 else str(x) for x in r)
    if any(abs(x) > 9 for x in r):
        s = " ".join(str(x) for x in r)
    return brackets[0] + s + brackets[1]


def canon_sign(v):
    v = np.asarray(v, float)
    for x in v:
        if abs(x) > 1e-8:
            return v if x > 0 else -v
    return v


def fmt_vec(v, nd=3):
    return "(" + ", ".join(f"{x: .{nd}f}" for x in v) + ")"


def frac_str(x, maxden=48):
    f = Fraction(x).limit_denominator(maxden)
    if abs(float(f) - x) < 1e-5:
        return str(f)
    return f"{x:.4f}"


# ============================================================================
#  file readers
# ============================================================================
def read_poscar(path):
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh]
    comment = lines[0].strip()
    sc = [float(x) for x in lines[1].split()[:3] if _isfloat(x)]
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if len(sc) == 3:
        lat = lat * np.array(sc)[:, None]
        fac = 1.0
    else:
        s = sc[0]
        fac = (abs(s) / abs(np.linalg.det(lat))) ** (1.0 / 3.0) if s < 0 else s
        lat = lat * fac
    tok = lines[5].split()
    if all(_isint(t) for t in tok):
        species, counts, idx = None, [int(t) for t in tok], 6
    else:
        species = [clean_species(t) for t in tok]
        counts = [int(t) for t in lines[6].split()[: len(species)]]
        idx = 7
    if lines[idx].strip()[:1].lower() == "s":
        idx += 1
    mode = lines[idx].strip()[:1].lower()
    idx += 1
    nat = sum(counts)
    raw = np.array([[float(x) for x in lines[idx + i].split()[:3]] for i in range(nat)])
    if mode in ("c", "k"):
        frac = (raw * fac) @ np.linalg.inv(lat)
    else:
        frac = raw
    frac = frac - np.floor(frac + 1e-10)
    return dict(comment=comment, lattice=lat, frac=frac, species=species, counts=counts)


def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def read_outcar(path):
    info = dict(path=path)
    with open(path, errors="replace") as fh:
        txt = fh.read()
    lines = txt.splitlines()

    def last(rx, cast=str, flags=0):
        m = re.findall(rx, txt, flags)
        return cast(m[-1]) if m else None

    def first(rx, cast=str, flags=0):
        m = re.search(rx, txt, flags)
        return cast(m.group(1)) if m else None

    info["nions"] = first(r"NIONS\s*=\s*(\d+)", int)
    info["ispin"] = first(r"ISPIN\s*=\s*(\d)", int)
    info["lnoncollinear"] = first(r"LNONCOLLINEAR\s*=\s*([TF])") == "T"
    info["lsorbit"] = first(r"LSORBIT\s*=\s*([TF])") == "T"
    m = re.search(r"SAXIS\s*=\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)", txt)
    info["saxis"] = np.array([float(m.group(i)) for i in (1, 2, 3)]) if m else None
    info["efermi"] = last(r"E-fermi\s*:\s*([-\d.Ee+]+)", float)
    info["toten"] = last(r"free  energy\s+TOTEN\s*=\s*([-\d.Ee+]+)", float)
    info["version"] = first(r"^\s*(vasp\.[\w.]+)", str, re.M)

    # species
    tit = re.findall(r"TITEL\s*=\s*\S+\s+(\S+)", txt)
    if not tit:
        tit = re.findall(r"VRHFIN\s*=\s*([A-Za-z]+)", txt)
    info["species"] = [clean_species(t) for t in tit] if tit else None
    m = re.search(r"ions per type\s*=\s*([\d\s]+)", txt)
    info["counts"] = [int(x) for x in m.group(1).split()] if m else None

    # MAGMOM from INCAR echo
    m = re.search(r"^\s*MAGMOM\s*=\s*([-\d.\s*Ee+]+)$", txt, re.M)
    info["magmom_incar"] = expand_vasp_list(m.group(1).split()) if m else None

    # total magnetization
    m = re.findall(r"number of electron\s+[-\d.]+\s+magnetization\s+([-\d.\sEe+]+)$", txt, re.M)
    info["mag_total"] = [float(x) for x in m[-1].split()] if m else None

    # site moments: last 'magnetization (x)', '(y)', '(z)' blocks
    site = {}
    orb = {}
    for i, l in enumerate(lines):
        mm = re.match(r"^\s*magnetization \(([xyz])\)\s*$", l)
        mo = re.match(r"^\s*orbital moment \(([xyz])\)\s*$", l)
        if mm or mo:
            comp = (mm or mo).group(1)
            vals = _parse_moment_block(lines, i + 1)
            if vals is not None:
                (site if mm else orb)[comp] = vals
    info["site_moments"] = site or None
    info["orbital_moments"] = orb or None
    return info


def _parse_moment_block(lines, start):
    j = start
    nd = 0
    while j < len(lines) and nd < 1:
        if lines[j].strip().startswith("---"):
            nd += 1
        j += 1
    vals = []
    while j < len(lines):
        s = lines[j].strip()
        if s.startswith("---") or s.startswith("tot") or not s:
            break
        tok = s.split()
        if _isint(tok[0]):
            vals.append(float(tok[-1]))
        j += 1
    return np.array(vals) if vals else None


def read_incar(path):
    """INCAR -> dict of upper-case tags (strings), plus parsed spin-related fields"""
    tags = {}
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = re.split(r"[#!]", raw)[0]
            for part in line.split(";"):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k, v = k.strip().upper(), v.strip()
                if k:
                    tags[k] = v
    info = dict(path=path, tags=tags)

    def lbool(k):
        v = tags.get(k)
        return None if v is None else v.strip(".").upper().startswith("T")
    info["ispin"] = int(float(tags["ISPIN"])) if "ISPIN" in tags else None
    info["lsorbit"] = lbool("LSORBIT")
    info["lnoncollinear"] = lbool("LNONCOLLINEAR")
    if info["lsorbit"]:
        info["lnoncollinear"] = True
    if "SAXIS" in tags:
        try:
            info["saxis"] = np.array([float(x) for x in tags["SAXIS"].split()[:3]])
        except ValueError:
            info["saxis"] = None
    else:
        info["saxis"] = None
    try:
        info["magmom"] = expand_vasp_list(tags["MAGMOM"].split()) if "MAGMOM" in tags else None
    except ValueError:
        info["magmom"] = None
    return info


def merge_calc(outcar, incar):
    """effective calculation flags: OUTCAR first, INCAR fills the gaps"""
    calc = dict(outcar or {})
    if incar:
        for k_out, k_in in (("ispin", "ispin"), ("lsorbit", "lsorbit"),
                            ("lnoncollinear", "lnoncollinear"), ("saxis", "saxis")):
            if calc.get(k_out) is None and incar.get(k_in) is not None:
                calc[k_out] = incar[k_in]
        if outcar is None:
            calc["lsorbit"] = bool(incar.get("lsorbit"))
            calc["lnoncollinear"] = bool(incar.get("lnoncollinear"))
        calc["magmom_incarfile"] = incar.get("magmom")
    return calc


def read_oszicar(path):
    info = dict(path=path, nionic=0)
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    lastl = None
    for l in lines:
        if "F=" in l:
            info["nionic"] += 1
            lastl = l
    if lastl:
        for key in ("F", "E0", "d E"):
            m = re.search(re.escape(key) + r"=\s*([-\d.Ee+]+)", lastl)
            if m:
                info[key.replace(" ", "")] = float(m.group(1))
        m = re.search(r"mag=\s*([-\d.Ee+\s]+)$", lastl.strip())
        if m:
            info["mag"] = [float(x) for x in m.group(1).split()]
    return info


# ============================================================================
#  magnetic moments
# ============================================================================
def saxis_to_cart(saxis):
    """rotation matrix taking spinor-frame vectors (VASP SAXIS convention) to Cartesian"""
    sx, sy, sz = saxis / np.linalg.norm(saxis)
    alpha = math.atan2(sy, sx)
    beta = math.atan2(math.hypot(sx, sy), sz)
    ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    return Rz @ Ry


def build_moments(nat, outcar, args, log):
    """return (nat,3) Cartesian moments and a description of their origin"""
    noncol = bool(outcar and (outcar.get("lnoncollinear") or outcar.get("lsorbit")))
    saxis = outcar.get("saxis") if outcar else None
    if args.mdir is not None:
        mdir = np.array(args.mdir, float)
    elif saxis is not None:
        mdir = saxis.copy()
    else:
        mdir = np.array([0.0, 0.0, 1.0])
    mdir = mdir / np.linalg.norm(mdir)
    Rsax = saxis_to_cart(saxis) if (saxis is not None and noncol) else EYE

    def from_list(vals, src, spinor_frame):
        vals = np.asarray(vals, float)
        if vals.size == nat:
            return vals[:, None] * mdir[None, :], f"{src} (collinear, along {fmt_vec(mdir, 3)})"
        if vals.size == 3 * nat:
            m = vals.reshape(nat, 3)
            if spinor_frame:
                m = m @ Rsax.T
            return m, f"{src} (non-collinear" + (", rotated from SAXIS frame)" if spinor_frame and saxis is not None else ")")
        raise ValueError(f"{src}: got {vals.size} values, expected {nat} or {3*nat}")

    if args.magmom:
        m, src = from_list(expand_vasp_list(args.magmom.replace(",", " ").split()), "--magmom", False)
    elif outcar and outcar.get("site_moments"):
        sm = outcar["site_moments"]
        if all(c in sm for c in "xyz"):
            vals = np.stack([sm["x"][:nat], sm["y"][:nat], sm["z"][:nat]], axis=1).ravel()
            m, src = from_list(vals, "OUTCAR site moments", True)
        else:
            m, src = from_list(sm["x"][:nat], "OUTCAR site moments", False)
    elif outcar and outcar.get("magmom_incarfile") and (outcar.get("ispin") == 2 or noncol):
        m, src = from_list(outcar["magmom_incarfile"], "INCAR MAGMOM (initial guess!)", True)
    elif outcar and outcar.get("magmom_incar") and (outcar.get("ispin") == 2 or noncol):
        m, src = from_list(outcar["magmom_incar"], "OUTCAR MAGMOM (initial guess!)", True)
    else:
        m, src = np.zeros((nat, 3)), "none -> non-magnetic"
    norms = np.linalg.norm(m, axis=1)
    nzero = int(np.sum((norms > 0) & (norms < args.mmin)))
    m[norms < args.mmin] = 0.0
    if nzero:
        log.append(f"{nzero} moment(s) with |m| < {args.mmin} muB set to zero (--mmin)")
    return m, src


# ============================================================================
#  magnetic-space-group label table (BNS / OG), from the Stokes-Campbell
#  tables (as distributed with pymatgen).  key 'BNS1.BNS2' -> (BNS, OG)
# ============================================================================
_MSG_TABLE = None


def msg_labels(bns_number):
    """(BNS symbol, OG symbol) for a BNS number such as '62.448'"""
    global _MSG_TABLE
    if _MSG_TABLE is None:
        _MSG_TABLE = {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "msg_labels.txt")
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("#") or "|" not in line:
                        continue
                    k, b, o = line.rstrip("\n").split("|")
                    _MSG_TABLE[k] = (b, o)
        except OSError:
            pass
    return _MSG_TABLE.get(bns_number, (None, None))


def mpg_from_bns_label(label):
    """magnetic point group symbol from a type-III BNS label"""
    s = re.sub(r"^[PABCIFR]", "", label)
    s = re.sub(r"(\d)_\d", r"\1", s)
    s = re.sub(r"[abcnde]", "m", s)
    return s


# ============================================================================
#  symmetry operations
# ============================================================================
class Op:
    __slots__ = ("R", "t", "theta", "Rc", "tc")

    def __init__(self, R, t, theta, Rc, tc):
        self.R, self.t, self.theta, self.Rc, self.tc = R, t, theta, Rc, tc


def lattice_ops_to_cart(rots, trans, trevs, lat, Q=None):
    A = lat.T  # columns = lattice vectors
    Ai = np.linalg.inv(A)
    ops = []
    for R, t, tr in zip(rots, trans, trevs):
        Rc = A @ R @ Ai
        tc = A @ t
        if Q is not None:
            Rc = Q @ Rc @ Q.T
            tc = Q @ tc
        Rc[np.abs(Rc) < 1e-10] = 0.0
        ops.append(Op(np.array(R, int), np.array(t, float), -1 if tr else 1, Rc, tc))
    return ops


def is_lattice_vec(v, pure, tol=1e-4):
    for p in pure:
        d = v - p
        if np.all(np.abs(d - np.round(d)) < tol):
            return True
    return False


def describe_op(op, lat, pure, Q=None):
    """return dict describing the geometric nature of a magnetic operation"""
    R, Rc = op.R, op.Rc
    det = int(round(np.linalg.det(R)))
    # order
    n = 1
    P = R.copy()
    while not np.array_equal(P, np.eye(3, dtype=int)) and n < 12:
        P = P @ R
        n += 1
    # intrinsic translation
    S = np.zeros(3)
    P = np.eye(3, dtype=int)
    for _ in range(n):
        S += P @ op.t
        P = P @ R
    w = S / n
    intrinsic = not is_lattice_vec(w, pure)
    A = lat.T
    wc = A @ w
    if Q is not None:
        wc = Q @ wc
    Rp = Rc * det  # proper part
    ang = math.acos(max(-1.0, min(1.0, (np.trace(Rp) - 1) / 2)))
    fold = int(round(2 * math.pi / ang)) if ang > 1e-6 else 1
    axis = None
    if fold > 1:
        evals, evecs = np.linalg.eig(Rp)
        k = np.argmin(np.abs(evals - 1))
        axis = canon_sign(np.real(evecs[:, k]))
        axis /= np.linalg.norm(axis)
    sense = ""
    if fold > 2 and axis is not None:
        a = np.array([Rp[2, 1] - Rp[1, 2], Rp[0, 2] - Rp[2, 0], Rp[1, 0] - Rp[0, 1]])
        sense = "+" if np.dot(a, axis) > 0 else "-"

    d = dict(det=det, fold=fold, axis=axis, sense=sense, order=n,
             intrinsic=intrinsic, w=w, wc=wc, theta=op.theta)
    if det == 1:
        if fold == 1:
            if np.allclose(op.t - np.round(op.t), 0, atol=1e-5):
                d["kind"], d["sym"] = "identity", "1"
            else:
                d["kind"], d["sym"] = "translation", "t"
        else:
            d["kind"] = f"{fold}-fold screw" if intrinsic else f"{fold}-fold rotation"
            d["sym"] = f"{fold}{sense}"
    else:
        if fold == 1:
            d["kind"], d["sym"] = "inversion", "-1"
        elif fold == 2:
            d["kind"] = "glide plane" if intrinsic else "mirror plane"
            d["sym"] = "m"
        else:
            d["kind"] = f"-{fold} rotoinversion"
            d["sym"] = f"-{fold}{sense}"
    # direction in lattice basis
    if axis is not None:
        if det == 1 or fold > 2:
            # axis: real-space direction [uvw]
            Ainv = np.linalg.inv(A if Q is None else Q @ A)
            d["lat"] = fmt_dir(small_int_vector(Ainv @ axis), "[]")
        else:
            # mirror: plane normal -> Miller (hkl)
            AA = A if Q is None else Q @ A
            d["lat"] = fmt_dir(small_int_vector(AA.T @ axis), "()")
    return d


# ============================================================================
#  tensor machinery
# ============================================================================
def D_rank2(R):
    return np.kron(R, R)


def D_rank3_axial(R):
    return np.linalg.det(R) * np.kron(np.kron(R, R), R)


def transpose_perm2():
    P = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            P[i * 3 + j, j * 3 + i] = 1
    return P


def nullspace(M, tol=1e-8):
    if M.shape[0] == 0:
        return np.eye(M.shape[1])
    _, s, vh = np.linalg.svd(M, full_matrices=(M.shape[0] < M.shape[1]))
    if len(s) < vh.shape[0]:
        s = np.concatenate([s, np.zeros(vh.shape[0] - len(s))])
    rank = int(np.sum(s > tol * max(1.0, s[0])))
    return vh[rank:].T.copy()


def rref(M, tol=1e-9):
    M = M.copy()
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        p = r + int(np.argmax(np.abs(M[r:, c])))
        if abs(M[p, c]) < tol:
            continue
        M[[r, p]] = M[[p, r]]
        M[r] /= M[r, c]
        for rr in range(rows):
            if rr != r:
                M[rr] -= M[rr, c] * M[r]
        piv.append(c)
        r += 1
    M[np.abs(M) < tol] = 0.0
    return M[:r], piv


def solve_tensor(ops, Dfun, ncomp, parity, extra=None):
    """parity: 'even' or 'odd' (behaviour under primed operations)"""
    rows = []
    I = np.eye(ncomp)
    for op in ops:
        s = 1.0 if (parity == "even" or op.theta == 1) else -1.0
        rows.append(s * Dfun(op.Rc) - I)
    if extra is not None:
        rows.append(extra)
    M = np.vstack(rows) if rows else np.zeros((0, ncomp))
    B = nullspace(M)
    if B.shape[1] == 0:
        return np.zeros((0, ncomp)), []
    E, piv = rref(B.T)
    return E, piv


def coef_str(x):
    if abs(x) < 1e-9:
        return None
    sign = "-" if x < 0 else "+"
    a = abs(x)
    for base, lab in ((1.0, ""), (SQ3, "√3"), (1.0 / SQ3, "/√3"), (math.sqrt(2), "√2")):
        f = Fraction(a / base).limit_denominator(12)
        if f != 0 and abs(float(f) * base - a) < 1e-6:
            num, den = f.numerator, f.denominator
            if lab == "":
                s = "" if (num == 1 and den == 1) else (f"{num}" if den == 1 else f"{num}/{den}")
            elif lab == "/√3":
                s = (f"{num}" if True else "") + ("/√3" if den == 1 else f"/({den}√3)")
            else:
                s = (("" if num == 1 else f"{num}") + lab) + ("" if den == 1 else f"/{den}")
            return sign, s
    return sign, f"{a:.5g}"


def expr(E, piv, c, names):
    terms = []
    for r, p in enumerate(piv):
        cs = coef_str(E[r, c])
        if cs is None:
            continue
        sign, s = cs
        terms.append((sign, (s + "·" + names[p]) if s else names[p]))
    if not terms:
        return "0"
    out = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, t in terms[1:]:
        out += f" {sign} {t}"
    return out


# --- naming -----------------------------------------------------------------
def names_rank2(sym):
    return [f"{sym}_{AX[i]}{AX[j]}" for i in range(3) for j in range(3)]


def names_rank3(sym):
    return [f"{sym}^{AX[k]}_{AX[i]}{AX[j]}" for k in range(3) for i in range(3) for j in range(3)]


def print_matrix(P, rows, title, indent="    "):
    w = max(len(s) for r in rows for s in r)
    w = max(w, 4)
    P(indent + title)
    for i, r in enumerate(rows):
        br = ("⎡", "⎤") if i == 0 else (("⎣", "⎦") if i == 2 else ("⎢", "⎥"))
        P(indent + "  " + br[0] + "  ".join(s.center(w) for s in r) + br[1])


# ============================================================================
#  non-relativistic spin-group report
# ============================================================================
def solve_general(DT, ncomp, parity, extra=None):
    """DT: list of (D matrix, theta).  T-even: chi = D chi ; T-odd: chi = theta D chi"""
    I = np.eye(ncomp)
    rows = [((1.0 if parity == "even" else th) * D) - I for D, th in DT]
    if extra is not None:
        rows.append(extra)
    B = nullspace(np.vstack(rows))
    if B.shape[1] == 0:
        return np.zeros((0, ncomp)), []
    return rref(B.T)


def _nonzero(E, piv, names):
    return {names[c] for c in range(len(names)) if expr(E, piv, c, names) != "0"}


def report_spin_group(P, args, lat, frac, types, symbols, mags, ds, Qstd, which, rel_results, SIG, msg_bns=None):
    from . import spingroup as sgm
    sg = sgm.spin_group(lat, frac, types, mags, _g(ds, "rotations"), _g(ds, "translations"),
                        symprec=args.symprec, mtol=max(args.mag_symprec, 0.02))
    if Qstd is not None:                      # rotate lattice and spin parts to the std frame
        for o in sg["ops"] + sg["spin_only"]:
            o.Rc = Qstd @ o.Rc @ Qstd.T
            o.U = Qstd @ o.U @ Qstd.T
            o.Q = o.theta * o.U
        if sg["axis"] is not None:
            sg["axis"] = Qstd @ sg["axis"]
    kind, axis = sg["kind"], sg["axis"]
    cls, why = sgm.classify_magnet(sg)
    nH = sum(o.coset == "H" for o in sg["ops"])
    nA = sum(o.coset == "A" for o in sg["ops"])
    so_txt = {"none": "O(3)  (all spin rotations and T: spin fully decoupled)",
              "collinear": "∞m1  =  C∞(n) ⋊ {E, C2⊥·T}",
              "coplanar": "m1  =  {E, C2(normal)·T}",
              "noncoplanar": "1  (trivial)"}[kind]

    P.h1("SPIN GROUP  (non-relativistic: no spin-orbit coupling)")
    P(f"  magnetic order        : {kind}" + (f",  axis n = {fmt_vec(axis, 3)}" if kind == "collinear" else
                                            (f",  plane normal = {fmt_vec(axis, 3)}" if kind == "coplanar" else "")))
    P(f"  spin-only group       : {so_txt}")
    P(f"  lattice group G       : {_g(ds, 'international')} (No. {_g(ds, 'number')}), {sg['n_lattice_G']} operations")
    P(f"  spin-group lattice ops: {len(sg['ops'])} of {sg['n_lattice_G']}   "
      f"(H: spin unchanged = {nH},  A: with spin operation = {nA},  incompatible = {sg['n_rejected']})")
    if kind == "collinear" and nH:
        try:
            hs = spglib.get_spacegroup_type_from_symmetry(
                np.array([o.R for o in sg["ops"] if o.coset == "H"]),
                np.array([o.t for o in sg["ops"] if o.coset == "H"]), lat, symprec=args.symprec)
            P(f"  spin-preserving subgroup H : {_g(hs, 'international_short') or _g(hs, 'international')} "
              f"(No. {_g(hs, 'number')}),  index [G:H] = {round(sg['n_lattice_G'] / max(nH, 1))}")
        except Exception:  # noqa: BLE001
            pass
    try:
        ssg = sgm.identify_ssg(lat, frac, types, sg, symprec=args.symprec, mag_symprec=args.mag_symprec)
    except Exception as err:  # noqa: BLE001
        ssg = dict(label=f"(identification failed: {err})", number=None, family="")
    P(f"  spin space group      : {ssg['label']}")
    P(f"                          {ssg.get('family', '')}")
    ossg = None
    if getattr(args, "ossg", "auto") != "no":
        try:
            ossg = sgm.identify_ossg(lat, frac, symbols, mags)
        except Exception as err:  # noqa: BLE001
            ossg = dict(available=False, note=f"FindSpinGroup failed: {err}")
        if ossg.get("available"):
            P(f"  OSSG number (Chen 2024): {ossg['index']}   {ossg['ossg_symbol_linear']}")
            P(f"                          G0 = {ossg['g0_symbol']} ({ossg['g0_number']}),  L0 = {ossg['l0_symbol']} "
              f"({ossg['l0_number']}),  i_t = {ossg['it']},  i_k = {ossg['ik']};  spin point group "
              f"{ossg['spin_space_point_group_hm']};  {ossg['conf']}, {ossg['magnetic_phase']}")
            mine = msg_bns
            chk = "" if mine is None else ("  ✓ = SymPhase" if str(mine) == str(ossg["msg_bns_number"])
                                           else f"  ≠ SymPhase {mine} (check tolerances)")
            ossg["msg_agrees"] = None if mine is None else str(mine) == str(ossg["msg_bns_number"])
            P(f"                          MSG with SOC from FindSpinGroup: {ossg['msg_symbol']} "
              f"(BNS {ossg['msg_bns_number']}){chk}   [findspingroup {ossg['findspingroup_version']}]")
        elif getattr(args, "ossg", "auto") == "yes" or kind in ("coplanar", "noncoplanar"):
            P(f"  OSSG number           : not available — {ossg.get('note')}")
    P(f"  magnet class          : {cls}")
    P(f"                          {why}")
    laue = None
    if kind == "collinear" and "antiferromagnet" not in cls:
        try:
            laue = sgm.spin_laue_symbol(sg, _g(ds, "pointgroup"), _g(ds, "std_lattice"),
                                        _g(ds, "std_rotation_matrix"))
        except Exception:  # noqa: BLE001
            laue = None
        if laue:
            P(f"  spin Laue group       : {laue[0]}   (Laue class {laue[1]};  ¹ = [E||g],  ² = [C2⊥||g])")
    elif kind == "collinear":
        P("  spin Laue group       : contains [C2⊥||E] (via t or P)  ->  every band spin degenerate")
    deg, forms = sgm.spin_polarisation_form(sg)
    wave = sgm.wave_label(deg, kind)
    P(f"  NR spin splitting     : {wave}")
    for f in forms[:4]:
        P(f"      {f}")
    planes = []
    if kind == "collinear" and "ALTERMAGNET" in cls:
        planes = sgm.nodal_planes(sg)
        P(f"  spin-degenerate nodal planes (k·n = 0): {len(planes)}")
        for n in planes:
            P(f"      n = {fmt_vec(n, 3)}")

    if not args.no_ops:
        P.h2("Spin-group operations [U || g]  (lattice part g, spin part U; ·T = contains time reversal)")
        P("    #  coset  lattice operation g                                                 spin operation U")
        A = lat.T
        for i, o in enumerate(sg["ops"]):
            d = describe_op(Op(o.R, o.t, 1, o.Rc, A @ o.t), lat, [np.zeros(3)])
            ax = (" ∥ " + fmt_vec(d["axis"], 2)) if d["axis"] is not None else ""
            tt = " ".join(frac_str(x) for x in (o.t - np.floor(o.t + 1e-8)))
            P(f"  {i+1:3d}   {o.coset:<4s} {(d['kind'] + d['sense'] + ax):<46s} t=({tt:<11s})  {sgm.describe_spin_op(o.U)}")
        for o in sg["spin_only"]:
            P(f"    -   spin-only  identity                                                     {sgm.describe_spin_op(o.U)}"
              + ("   (generic angle: generates the continuous group)" if "R(" in sgm.describe_spin_op(o.U) else ""))

    # ---------------------------------------------------------------- NR tensors
    allops = sg["ops"] + sg["spin_only"]
    seen, uniq = set(), []
    for o in allops:
        key = (tuple(np.round(o.Rc, 6).ravel()), tuple(np.round(o.U, 6).ravel()))
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    out = dict(kind=kind, axis=None if axis is None else list(map(float, axis)), magnet_class=cls,
               ssg=ssg, ossg=ossg,
               reason=why, spin_laue_group=laue[0] if laue else None, spin_splitting=wave,
               splitting_degree=deg, s_k=forms, nodal_planes=[list(map(float, n)) for n in planes],
               n_ops=len(sg["ops"]), n_H=nH, n_A=nA, tensors={})
    P.h1("NON-RELATIVISTIC RESPONSE TENSORS  (spin group, no SOC)")
    P("  spin index rotates with the spin part Q of [U||g] (U = θQ, θ = det U);  T-odd parts pick θ.")
    P("  'SOC-only' = allowed with SOC (magnetic group) but forbidden in the spin group.")
    if "charge" in which:
        Pm = transpose_perm2()
        DT = [(D_rank2(o.Rc), o.theta) for o in uniq]
        Ee, pe = solve_general(DT, 9, "even", extra=np.eye(9) - Pm)
        Eo, po = solve_general(DT, 9, "odd", extra=np.eye(9) + Pm)
        nm = names_rank2(SIG)
        P.h2("CHARGE CONDUCTIVITY (NR)")
        for lab, E, pv in (("T-even", Ee, pe), ("T-odd (anomalous Hall)", Eo, po)):
            if not pv:
                P(f"    {lab}: all components vanish")
            else:
                print_matrix(P, [[expr(E, pv, i * 3 + j, nm) for j in range(3)] for i in range(3)],
                             f"{lab}: {len(pv)} independent")
        nr_odd = _nonzero(Eo, po, nm)
        rel = rel_results.get("charge")
        if rel:
            soc = sorted({k for k, v in rel["odd"].items() if v != "0"} - nr_odd)
            P("    anomalous Hall without SOC: " + ("ALLOWED (non-relativistic AHE)" if po else "forbidden")
              + ("   |  SOC-only Hall components: " + ", ".join(soc) if soc else ""))
        out["tensors"]["charge"] = dict(even={nm[c]: expr(Ee, pe, c, nm) for c in range(9)},
                                        odd={nm[c]: expr(Eo, po, c, nm) for c in range(9)},
                                        n_even=len(pe), n_odd=len(po), ahe_allowed=bool(po))
    for kind_t in ("spin", "orbital"):
        if kind_t not in which:
            continue
        sym = SIG + ("" if kind_t == "spin" else "L")
        nm = names_rank3(sym)
        if kind_t == "spin":
            DT = [(np.kron(np.kron(o.Q, o.Rc), o.Rc), o.theta) for o in uniq]
        else:
            DT = [(np.linalg.det(o.Rc) * np.kron(np.kron(o.Rc, o.Rc), o.Rc), o.theta) for o in uniq]
        lab_op = "S" if kind_t == "spin" else "L"
        P.h2(f"{kind_t.upper()} CONDUCTIVITY (NR)  J^{{{lab_op}_k}}_i = {sym}^k_ij E_j")
        if kind_t == "orbital":
            P("    (L is an orbital/lattice quantity: it does not rotate with the spin part of [U||g])")
        res = {}
        for part in ("even", "odd"):
            E, pv = solve_general(DT, 27, part)
            lab = ("T-even" if part == "even" else "T-odd") + \
                  (" (spin-splitter / NR spin current)" if (part == "odd" and kind_t == "spin") else "")
            nz = _nonzero(E, pv, nm)
            if not pv:
                P(f"    {lab}: all components vanish")
            else:
                P(f"    {lab}: {len(pv)} independent:  " + ", ".join(nm[p] for p in pv))
                for k in range(3):
                    rows = [[expr(E, pv, (k * 3 + i) * 3 + j, nm) for j in range(3)] for i in range(3)]
                    if any(x != "0" for r in rows for x in r):
                        print_matrix(P, rows, f"k = {AX[k]}:", indent="      ")
            rel = rel_results.get(kind_t, {}).get(part)
            soc = sorted({k for k, v in (rel or {}).items() if v != "0"} - nz)
            if rel is not None:
                P(f"      SOC-only ({len(soc)}): " + (", ".join(soc[:18]) + (" ..." if len(soc) > 18 else "")
                                                    if soc else "none"))
            res[part] = {nm[c]: expr(E, pv, c, nm) for c in range(27)}
            res[f"n_{part}"] = len(pv)
            res[f"soc_only_{part}"] = soc
        out["tensors"][kind_t] = res
    return out


# ============================================================================
#  report
# ============================================================================
class Printer:
    def __init__(self, quiet=False):
        self.buf = []
        self.quiet = quiet
        self.mute = False

    def __call__(self, s=""):
        if self.mute:
            return
        self.buf.append(s)
        if not self.quiet:
            print(s)

    def h1(self, s):
        self("")
        self("=" * 78)
        self(s)
        self("=" * 78)

    def h2(self, s):
        self("")
        self("-- " + s + " " + "-" * max(0, 74 - len(s)))


def main(argv=None, quiet=False):
    ap = argparse.ArgumentParser(prog="symphase tensors",
        description="Crystal/magnetic symmetry analysis and symmetry-allowed charge, spin "
                    "and orbital conductivity tensors from VASP POSCAR/OSZICAR/OUTCAR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  symphase tensors                        # POSCAR, OUTCAR, OSZICAR in cwd\n"
               "  symphase tensors -p CONTCAR --mmin 0.1\n"
               "  symphase tensors --magmom '0 0 3  0 0 -3  4*0'   # override moments\n"
               "  symphase tensors --mdir 1 1 0           # collinear moments along [110]\n"
               "  symphase tensors --frame std --json out.json --out report.txt\n")
    ap.add_argument("-p", "--poscar", default="POSCAR")
    ap.add_argument("-o", "--outcar", default="OUTCAR")
    ap.add_argument("-z", "--oszicar", default="OSZICAR")
    ap.add_argument("-i", "--incar", default="INCAR")
    ap.add_argument("--analysis", choices=("both", "msg", "spin"), default="both",
                    help="msg: relativistic magnetic groups (SOC); spin: non-relativistic spin "
                         "groups (altermagnetism, no SOC); both [default]")
    ap.add_argument("--symprec", type=float, default=1e-3, help="position tolerance (Å) [1e-3]")
    ap.add_argument("--angle-tol", type=float, default=-1.0, help="angle tolerance (deg) [-1 = spglib default]")
    ap.add_argument("--mag-symprec", type=float, default=0.05,
                    help="tolerance on magnetic moments (muB) [0.05]")
    ap.add_argument("--mmin", type=float, default=0.05,
                    help="moments smaller than this (muB) are set to zero [0.05]")
    ap.add_argument("--magmom", default=None,
                    help="override moments: N values (collinear) or 3N values (Cartesian), "
                         "VASP N*val notation allowed")
    ap.add_argument("--mdir", type=float, nargs=3, default=None,
                    help="direction (Cartesian) of collinear moments [SAXIS or 0 0 1]")
    ap.add_argument("--frame", choices=("input", "std"), default="input",
                    help="Cartesian frame for tensors: POSCAR frame or spglib standardized frame")
    ap.add_argument("--tensors", default="charge,spin,orbital",
                    help="comma list from charge,spin,orbital [all]")
    ap.add_argument("--no-ops", action="store_true", help="do not list individual operations")
    ap.add_argument("--ossg", choices=("auto", "yes", "no"), default="auto",
                    help="oriented spin-space-group number (Chen et al. 2024) for any order via FindSpinGroup [auto]")
    ap.add_argument("--ascii", action="store_true", help="use 's' instead of 'σ' in component names")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    ap.add_argument("--json", default=None, help="write machine-readable results to this JSON file")
    args = ap.parse_args(argv)

    P = Printer(quiet)
    log = []
    SIG = "s" if args.ascii else "σ"

    # ---------------------------------------------------------------- input
    if not os.path.isfile(args.poscar):
        sys.exit(f"cannot find {args.poscar}")
    pos = read_poscar(args.poscar)
    outcar = read_outcar(args.outcar) if os.path.isfile(args.outcar) else None
    incar = read_incar(args.incar) if os.path.isfile(args.incar) else None
    calc = merge_calc(outcar, incar)
    osz = read_oszicar(args.oszicar) if os.path.isfile(args.oszicar) else None

    species = pos["species"] or (outcar and outcar.get("species"))
    counts = pos["counts"]
    if not species:
        species = [f"X{i+1}" for i in range(len(counts))]
        log.append("no species names found (VASP4 POSCAR and no OUTCAR) - using X1, X2, ...")
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    types = [species.index(s) + 1 for s in symbols]
    nat = len(symbols)
    lat, frac = pos["lattice"], pos["frac"]

    if outcar and outcar.get("nions") and outcar["nions"] != nat:
        log.append(f"WARNING: OUTCAR NIONS={outcar['nions']} differs from POSCAR ({nat})")

    mags, magsrc = build_moments(nat, calc, args, log)
    magnetic = bool(np.any(np.linalg.norm(mags, axis=1) > 0))

    P.h1(f"SymPhase {__version__}  —  symmetry & linear-response tensor analysis")
    P(f"  POSCAR  : {os.path.abspath(args.poscar)}   ({pos['comment']})")
    P(f"  OUTCAR  : {os.path.abspath(args.outcar) if outcar else 'not found'}")
    P(f"  OSZICAR : {os.path.abspath(args.oszicar) if osz else 'not found'}")
    P(f"  INCAR   : {os.path.abspath(args.incar) if incar else 'not found'}")

    # ---------------------------------------------------------------- calculation summary
    P.h2("Calculation")
    if outcar or incar:
        if outcar and outcar.get("version"):
            P(f"  VASP version         : {outcar['version']}")
        P(f"  ISPIN                : {calc.get('ispin')}")
        P(f"  LNONCOLLINEAR/LSORBIT: {calc.get('lnoncollinear')} / {calc.get('lsorbit')}")
        if calc.get("saxis") is not None:
            P(f"  SAXIS                : {fmt_vec(calc['saxis'])}")
    if incar:
        t = incar["tags"]
        keys = [k for k in ("SYSTEM", "GGA", "METAGGA", "ENCUT", "ISMEAR", "SIGMA", "EDIFF", "IBRION",
                            "ISIF", "NSW", "LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "IVDW",
                            "MAGMOM", "ISYM", "LORBIT", "NBANDS", "LWANNIER90") if k in t]
        P("  INCAR tags           : " + ("; ".join(f"{k} = {t[k]}" for k in keys) if keys else "-"))
    if outcar:
        if outcar.get("efermi") is not None:
            P(f"  E-fermi              : {outcar['efermi']:.4f} eV")
        if outcar.get("toten") is not None:
            P(f"  TOTEN (last)         : {outcar['toten']:.6f} eV")
        if outcar.get("mag_total"):
            P(f"  total magnetization  : {' '.join(f'{x:.4f}' for x in outcar['mag_total'])} muB (OUTCAR)")
    if osz:
        s = f"  OSZICAR              : {osz['nionic']} ionic step(s)"
        if "F" in osz:
            s += f", F = {osz['F']:.6f} eV, E0 = {osz.get('E0', float('nan')):.6f} eV"
        if "mag" in osz:
            s += f", mag = {' '.join(f'{x:.4f}' for x in osz['mag'])} muB"
        P(s)

    P.h2("Structure")
    P("  lattice vectors (Å):")
    for v, n in zip(lat, "abc"):
        P(f"    {n} = {fmt_vec(v, 6)}")
    abc = np.linalg.norm(lat, axis=1)
    al = math.degrees(math.acos(np.dot(lat[1], lat[2]) / abc[1] / abc[2]))
    be = math.degrees(math.acos(np.dot(lat[0], lat[2]) / abc[0] / abc[2]))
    ga = math.degrees(math.acos(np.dot(lat[0], lat[1]) / abc[0] / abc[1]))
    P(f"  a b c = {abc[0]:.5f} {abc[1]:.5f} {abc[2]:.5f} Å    α β γ = {al:.3f} {be:.3f} {ga:.3f}°"
      f"    V = {abs(np.linalg.det(lat)):.4f} Å^3")
    P(f"  moments source: {magsrc}")
    P("   #  el      fractional coordinates               m_x      m_y      m_z     |m|  (muB, Cartesian)")
    for i in range(nat):
        m = mags[i]
        P(f"  {i+1:3d} {symbols[i]:<3s} {frac[i,0]:10.6f} {frac[i,1]:10.6f} {frac[i,2]:10.6f}   "
          f"{m[0]:8.3f} {m[1]:8.3f} {m[2]:8.3f} {np.linalg.norm(m):7.3f}")
    if magnetic:
        P(f"  net moment per cell: {fmt_vec(mags.sum(0), 3)} muB")
    if outcar and outcar.get("orbital_moments"):
        om = outcar["orbital_moments"]
        P("  (orbital moments found in OUTCAR: " + ", ".join(
            f"L_{c}: max |{np.max(np.abs(v)):.3f}|" for c, v in om.items()) + ")")

    # ---------------------------------------------------------------- crystal symmetry
    angle = args.angle_tol
    cell = (lat, frac, types)
    ds = spglib.get_symmetry_dataset(cell, symprec=args.symprec, angle_tolerance=angle)
    if ds is None:
        sys.exit("spglib could not determine the space group: " + str(spglib.get_error_message()))
    P.h1("CRYSTAL (non-magnetic) SYMMETRY")
    P(f"  space group          : {_g(ds,'international')}  (No. {_g(ds,'number')})")
    P(f"  Hall symbol          : {_g(ds,'hall')}  (Hall No. {_g(ds,'hall_number')})")
    P(f"  point group (crystal class): {_g(ds,'pointgroup')}")
    nrot = len(_g(ds, "rotations"))
    P(f"  number of operations : {nrot}  (in the input cell)")
    cryst_ops = lattice_ops_to_cart(_g(ds, "rotations"), _g(ds, "translations"),
                                    [False] * nrot, lat)
    cryst_inv = any(np.allclose(o.Rc, -EYE, atol=1e-6) for o in cryst_ops)
    P(f"  centrosymmetric      : {'yes' if cryst_inv else 'no'}")
    wy = _g(ds, "wyckoffs")
    eq = _g(ds, "equivalent_atoms")
    sites = _g(ds, "site_symmetry_symbols")
    P("  Wyckoff positions    :")
    seen = set()
    for i in range(nat):
        if eq[i] in seen:
            continue
        seen.add(eq[i])
        mult = int(np.sum(eq == eq[i]))
        ss = sites[i] if sites is not None else ""
        P(f"     {symbols[i]:<3s} atom {i+1:3d}: {mult}{wy[i]}  site symmetry {ss}")

    # ---------------------------------------------------------------- magnetic symmetry
    mcell = (lat, frac, types, mags)
    mds = spglib.get_magnetic_symmetry_dataset(mcell, symprec=args.symprec, angle_tolerance=angle,
                                               mag_symprec=args.mag_symprec)
    if mds is None:
        sys.exit("spglib could not determine the magnetic space group: " + str(spglib.get_error_message()))
    uni = int(_g(mds, "uni_number"))
    mtype = spglib.get_magnetic_spacegroup_type(uni)
    bns = _g(mtype, "bns_number")
    og = _g(mtype, "og_number")
    msgt = int(_g(mtype, "type"))
    bns_lab, og_lab = msg_labels(bns)
    Q = np.array(_g(mds, "std_rotation_matrix")) if args.frame == "std" else None

    rots, trs, tev = _g(mds, "rotations"), _g(mds, "translations"), _g(mds, "time_reversals")
    ops = lattice_ops_to_cart(rots, trs, tev, lat, Q)
    pure = [o.t for o in ops if np.array_equal(o.R, np.eye(3, dtype=int)) and o.theta == 1]
    if not pure:
        pure = [np.zeros(3)]

    fam_pg = spglib.get_pointgroup(np.array(rots))[0].strip()
    uni_rots = np.array([o.R for o in ops if o.theta == 1])
    unit_pg = spglib.get_pointgroup(uni_rots)[0].strip()
    if msgt == 1:
        mpg = fam_pg
    elif msgt in (2, 4):
        mpg = fam_pg + "1'"
    else:
        mpg = mpg_from_bns_label(bns_lab) if bns_lab else f"{fam_pg} (primed, see ops)"

    type_txt = {1: "I   (colourless / ordinary: no T in any operation)",
                2: "II  (grey: T itself is a symmetry -> paramagnet / non-magnetic)",
                3: "III (black-white, T combined with point operations only)",
                4: "IV  (black-white, T combined with a translation, T·τ)"}[msgt]

    P.h1("MAGNETIC SYMMETRY" + ("" if magnetic else "  (no moments -> grey group)"))
    P(f"  magnetic space group : {bns_lab or '?'}   BNS No. {bns}")
    P(f"                         OG symbol {og_lab or '?'}   OG No. {og}")
    P(f"  UNI number           : {uni}    Litvin No. {_g(mtype,'litvin_number')}")
    P(f"  MSG type             : {type_txt}")
    P(f"  family space group   : No. {_g(mtype,'number')}  (point group {fam_pg})")
    P(f"  magnetic point group : {mpg}")
    P(f"  unitary subgroup     : point group {unit_pg}  (operations without T)")
    P(f"  operations           : {len(ops)}  ({sum(o.theta==1 for o in ops)} unitary, "
      f"{sum(o.theta==-1 for o in ops)} with T)")

    descs = [describe_op(o, lat, pure, Q) for o in ops]

    # flags
    def has(pred):
        return [d for o, d in zip(ops, descs) if pred(o, d)]

    inv = has(lambda o, d: d["kind"] == "inversion" and o.theta == 1)
    invT = has(lambda o, d: d["kind"] == "inversion" and o.theta == -1)
    Tpure = has(lambda o, d: d["kind"] == "identity" and o.theta == -1)
    Ttau = has(lambda o, d: d["kind"] == "translation" and o.theta == -1)
    mir = has(lambda o, d: d["kind"] in ("mirror plane", "glide plane") and o.theta == 1)
    mirT = has(lambda o, d: d["kind"] in ("mirror plane", "glide plane") and o.theta == -1)
    rot = has(lambda o, d: d["det"] == 1 and d["fold"] > 1 and o.theta == 1)
    rotT = has(lambda o, d: d["det"] == 1 and d["fold"] > 1 and o.theta == -1)
    rinv = has(lambda o, d: "rotoinversion" in d["kind"])

    def uniq_dirs(lst, with_kind=True):
        out = {}
        for d in lst:
            key = (d["sym"].rstrip("+-"), tuple(np.round(d["axis"], 4)) if d["axis"] is not None else None)
            kinds = out.setdefault(key, set())
            kinds.add(d["kind"])
            out[key] = kinds
        res = []
        for (sym, ax), kinds in out.items():
            dd = next(d for d in lst if (d["sym"].rstrip("+-"), tuple(np.round(d["axis"], 4)) if d["axis"] is not None else None) == (sym, ax))
            if dd["det"] == -1 and dd["fold"] == 2:
                res.append(f"{'/'.join(sorted(kinds))} ⟂ {fmt_vec(ax,3)} = plane {dd.get('lat','')}")
            else:
                res.append(f"{'/'.join(sorted(kinds))} ∥ {fmt_vec(ax,3)} = {dd.get('lat','')}")
        return res

    P.h2("Symmetry elements present")
    yn = lambda b: "YES" if b else "no"
    P(f"  time reversal T                      : {yn(Tpure)}")
    P(f"  T·τ (antitranslation, T + lattice τ) : {yn(Ttau)}" +
      (f"   τ = {', '.join(fmt_vec(d['w'],3) for d in Ttau[:3])} (fractional)" if Ttau else ""))
    P(f"  inversion centre P (no T)            : {yn(inv)}")
    P(f"  PT (inversion × time reversal)       : {yn(invT)}")
    P(f"  crystal (non-magnetic) inversion     : {yn(cryst_inv)}")
    P(f"  mirror/glide planes (no T)           : {len(uniq_dirs(mir))}")
    for s in uniq_dirs(mir):
        P(f"        {s}")
    P(f"  mirror/glide planes × T  (M·T)       : {len(uniq_dirs(mirT))}" + ("   (every element also × T: grey group)" if (Tpure or Ttau) else ""))
    for s in ([] if (Tpure or Ttau) else uniq_dirs(mirT)):
        P(f"        {s}")
    P(f"  proper rotation/screw axes (no T)    : {len(uniq_dirs(rot))}")
    for s in uniq_dirs(rot):
        P(f"        {s}")
    P(f"  rotation/screw axes × T  (C_n·T)     : {len(uniq_dirs(rotT))}" + ("   (every element also × T: grey group)" if (Tpure or Ttau) else ""))
    for s in ([] if (Tpure or Ttau) else uniq_dirs(rotT)):
        P(f"        {s}")
    if rinv:
        P(f"  rotoinversion axes                   : {len(uniq_dirs(rinv))}")
        for s in uniq_dirs(rinv):
            P(f"        {s}" + "")
    P("")
    centro_mag = bool(inv)
    P(f"  -> magnetically centrosymmetric      : {yn(centro_mag)}")
    P(f"  -> T-odd responses (AHE, MOKE, ...)  : {'FORBIDDEN by T or T·τ' if (Tpure or Ttau) else 'symmetry may allow (see tensors)'}")
    if invT and not inv:
        P("  -> PT-symmetric antiferromagnet: Kramers-degenerate bands, T-odd & P-odd responses allowed")
    if magnetic and not (Tpure or Ttau):
        if np.linalg.norm(mags.sum(0)) < 0.05 * np.max(np.linalg.norm(mags, axis=1)):
            if not invT:
                P("  -> compensated magnet without PT and without T·τ: altermagnet-/non-collinear-AFM-like")
    if (outcar or incar) and magnetic and not calc.get("lsorbit"):
        P("  NOTE: run without SOC. The magnetic group above couples spin and lattice (relativistic).")
        P("        Without SOC the true symmetry is the SPIN GROUP analysed below (--analysis spin).")

    if not args.no_ops:
        P.h2("All magnetic symmetry operations (Cartesian frame: " + ("standardized" if Q is not None else "POSCAR") + ")")
        P("    #   T   type                 axis / plane normal (Cart.)    lattice     intrinsic τ (frac)   full t (frac)")
        for i, (o, d) in enumerate(zip(ops, descs)):
            ax = fmt_vec(d["axis"], 3) if d["axis"] is not None else "-"
            tstr = " ".join(frac_str(x) for x in (o.t - np.floor(o.t + 1e-8)))
            w = " ".join(frac_str(x) for x in d["w"]) if d["intrinsic"] or d["kind"] == "translation" else "-"
            kind = d["kind"] + (d["sense"] if d["sense"] else "")
            P(f"  {i+1:3d}  {'′' if o.theta==-1 else ' '}   {kind:<20s} {ax:<30s} {d.get('lat','-'):<11s} {w:<20s} {tstr}")
        P("  (′ = operation combined with time reversal T)")

    # ---------------------------------------------------------------- tensors
    which = [s.strip() for s in args.tensors.split(",") if s.strip()]
    point_ops = []
    seen = set()
    for o in ops:  # point part only matters for macroscopic tensors
        key = (tuple(o.R.ravel()), o.theta)
        if key not in seen:
            seen.add(key)
            point_ops.append(o)

    results = {}
    frame_txt = "standardized (spglib std_rotation_matrix)" if Q is not None else "POSCAR Cartesian"

    P.mute = args.analysis == "spin"
    P.h1("SYMMETRY-ALLOWED LINEAR-RESPONSE TENSORS  (relativistic: magnetic point group, SOC)")
    P(f"  frame: {frame_txt};  magnetic point group {mpg}  ({len(point_ops)} point operations)")
    P("  T-even: invariant under all g and g' with D(R);  T-odd: picks sign -1 under primed g'")

    if "charge" in which:
        Pm = transpose_perm2()
        Ee, pe = solve_tensor(point_ops, D_rank2, 9, "even", extra=np.eye(9) - Pm)
        Eo, po = solve_tensor(point_ops, D_rank2, 9, "odd", extra=np.eye(9) + Pm)
        nm = names_rank2(SIG)
        P.h2("CHARGE CONDUCTIVITY  J_i = σ_ij E_j")
        for lab, E, pv, note in (("T-even (symmetric, dissipative/Drude)", Ee, pe, ""),
                                 ("T-odd (antisymmetric, anomalous Hall)", Eo, po, "")):
            rows = [[expr(E, pv, i * 3 + j, nm) for j in range(3)] for i in range(3)]
            P("")
            if not pv:
                P(f"    {lab}:  all components vanish")
                continue
            print_matrix(P, rows, f"{lab}:  {len(pv)} independent")
        hall = None
        if po:
            # Hall vector h with σ_ij^odd = ε_ijk h_k
            hv = []
            for k, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
                hv.append(expr(Eo, po, i * 3 + j, nm))
            hall = hv
            P(f"\n    anomalous Hall vector (σ^odd_ij = ε_ijk h_k): h = ({hv[0]}, {hv[1]}, {hv[2]})")
            P("    -> ANOMALOUS HALL EFFECT ALLOWED")
        else:
            P("\n    -> anomalous Hall effect forbidden")
        results["charge"] = dict(
            even={nm[c]: expr(Ee, pe, c, nm) for c in range(9)},
            odd={nm[c]: expr(Eo, po, c, nm) for c in range(9)},
            n_even=len(pe), n_odd=len(po), ahe_allowed=bool(po), hall_vector=hall)

    for kind in ("spin", "orbital"):
        if kind not in which:
            continue
        sym = SIG + ("" if kind == "spin" else "L")
        nm = names_rank3(sym)
        Ee, pe = solve_tensor(point_ops, D_rank3_axial, 27, "even")
        Eo, po = solve_tensor(point_ops, D_rank3_axial, 27, "odd")
        lab_op = "S" if kind == "spin" else "L"
        P.h2(f"{kind.upper()} CONDUCTIVITY  J^{{{lab_op}_k}}_i = {sym}^k_ij E_j"
             f"   (i: flow, j: field, k: {kind} polarisation)")
        if kind == "orbital":
            P("    (L_k transforms like S_k: same symmetry constraints as the spin tensor)")
        res = {}
        for lab, E, pv in (("T-even (intrinsic " + ("SHE" if kind == "spin" else "OHE") + ")", Ee, pe),
                           ("T-odd (magnetic " + ("SHE" if kind == "spin" else "OHE") + ", ∝ 1/Γ)", Eo, po)):
            P("")
            P(f"  {lab}:  {len(pv)} independent component(s)")
            if not pv:
                P("    all 27 components vanish")
                res["even" if E is Ee else "odd"] = {nm[c]: "0" for c in range(27)}
                continue
            P("    independent: " + ", ".join(nm[p] for p in pv))
            for k in range(3):
                rows = [[expr(E, pv, (k * 3 + i) * 3 + j, nm) for j in range(3)] for i in range(3)]
                print_matrix(P, rows, f"k = {AX[k]}  ({lab_op}_{AX[k]} polarisation):")
            # symmetric / antisymmetric character of transverse pairs
            col = lambda k, i, j: E[:, (k * 3 + i) * 3 + j]
            asym, symm, gen = [], [], []
            for k in range(3):
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = col(k, i, j), col(k, j, i)
                        if np.allclose(a, 0) and np.allclose(b, 0):
                            continue
                        tag = f"{AX[k]}:{AX[i]}{AX[j]}"
                        if np.allclose(a + b, 0):
                            asym.append(tag)
                        elif np.allclose(a - b, 0):
                            symm.append(tag)
                        else:
                            gen.append(tag)
            P("    transverse pairs (k:ij)  antisymmetric σ^k_ij=-σ^k_ji (SHE-like): " + (", ".join(asym) or "none"))
            P("                             symmetric σ^k_ij=+σ^k_ji (spin-splitter-like): " + (", ".join(symm) or "none"))
            P("                             independent ij, ji (mixed): " + (", ".join(gen) or "none"))
            # conventional / unconventional
            conv, unconv = [], []
            for c in range(27):
                k, i, j = c // 9, (c // 3) % 3, c % 3
                if expr(E, pv, c, nm) == "0":
                    continue
                if i != j and k != i and k != j:
                    conv.append(nm[c])
                else:
                    unconv.append(nm[c])
            P(f"    k ⟂ i ⟂ j (conventional geometry) nonzero : {', '.join(conv) if conv else 'none'}")
            P(f"    unconventional geometry nonzero          : {', '.join(unconv) if unconv else 'none'}")
            # SOT-style geometry table
            P("    allowed polarisations for transverse flow i ⟂ field j:")
            for i in range(3):
                for j in range(3):
                    if i == j:
                        continue
                    ks = [AX[k] for k in range(3) if expr(E, pv, (k * 3 + i) * 3 + j, nm) != "0"]
                    P(f"      flow {AX[i]}, E ∥ {AX[j]} : {lab_op}_{{{','.join(ks) if ks else '-'}}}")
            res["even" if E is Ee else "odd"] = {nm[c]: expr(E, pv, c, nm) for c in range(27)}
        res["n_even"], res["n_odd"] = len(pe), len(po)
        results[kind] = res

    P.mute = False
    spin_res = None
    if args.analysis in ("both", "spin"):
        spin_res = report_spin_group(P, args, lat, frac, types, symbols, mags, ds, Q, which, results, SIG,
                                      msg_bns=bns)

    if log:
        P.h2("Notes")
        for l in log:
            P("  " + l)
    P("")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(P.buf) + "\n")
    js = dict(
        version=__version__,
        structure=dict(lattice=lat.tolist(), frac=frac.tolist(), symbols=symbols,
                       moments=mags.tolist(), moments_source=magsrc),
        crystal=dict(space_group=_g(ds, "international"), number=int(_g(ds, "number")),
                     hall=_g(ds, "hall"), point_group=_g(ds, "pointgroup"),
                     centrosymmetric=cryst_inv),
        magnetic=dict(bns_symbol=bns_lab, bns_number=bns, og_symbol=og_lab, og_number=og,
                      uni_number=uni, type=msgt, magnetic_point_group=mpg,
                      unitary_point_group=unit_pg, n_operations=len(ops),
                      time_reversal=bool(Tpure), antitranslation=bool(Ttau),
                      inversion=bool(inv), PT=bool(invT),
                      mirrors=uniq_dirs(mir), mirrors_T=uniq_dirs(mirT),
                      rotations=uniq_dirs(rot), rotations_T=uniq_dirs(rotT),
                      operations=[dict(R=o.R.tolist(), t=o.t.tolist(), time_reversal=o.theta == -1,
                                       R_cart=o.Rc.tolist(), kind=d["kind"])
                                  for o, d in zip(ops, descs)]),
        frame=frame_txt, tensors=results, spin_group=spin_res, notes=log,
        incar=(incar or {}).get("tags"))
    js["report"] = "\n".join(P.buf)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({k: v for k, v in js.items() if k != "report"}, fh, indent=1, ensure_ascii=False)
    return js


def analyze_tensors(directory=".", poscar=None, outcar="OUTCAR", oszicar="OSZICAR", incar="INCAR",
                    quiet=True, **options):
    """Python API.  Run the symmetry/tensor analysis on a VASP directory.

    options map to command-line flags, e.g. mdir=(1, 1, 0), magmom="2*2.2",
    frame="std", tensors="charge,spin", mmin=0.1, symprec=1e-3, analysis="both|msg|spin".
    Returns a dict (structure, crystal, magnetic, tensors, spin_group, report).
    """
    if poscar is None:
        poscar = "CONTCAR" if os.path.isfile(os.path.join(directory, "CONTCAR")) and \
            os.path.getsize(os.path.join(directory, "CONTCAR")) > 0 else "POSCAR"
    argv = ["-p", os.path.join(directory, poscar), "-o", os.path.join(directory, outcar),
            "-z", os.path.join(directory, oszicar), "-i", os.path.join(directory, incar)]
    for k, v in options.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False or v is None:
            continue
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return main(argv, quiet=quiet)


if __name__ == "__main__":
    main()
