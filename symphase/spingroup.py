# -*- coding: utf-8 -*-
"""
symphase.spingroup — non-relativistic spin-group analysis
=========================================================

Without spin–orbit coupling spin and lattice transform independently.  A
spin-group element is a pair  [U || g]  with a lattice operation g = {R|t}
of the (non-magnetic) crystal space group G and a spin operation U in O(3);
improper U = (proper rotation) x T, i.e. det(U) = -1 means the element
contains time reversal (antiunitary).

    g belongs to the spin group  <=>  there is U with  m_{g(i)} = U m_i  for every atom i

U is found by an orthogonal Procrustes fit (SVD) and verified.  The spin-only
group (operations [U || E]) depends only on the dimensionality of the order:

    non-magnetic : all of O(3)                     (spin fully decoupled)
    collinear    : C_inf(n)  x  {E, C2_perp T}      ("inf m 1")
    coplanar     : {E, C2_normal T}                 ("m 1")
    non-coplanar : {E}

For every element we write U = theta * Q with theta = det(U) = +-1 and Q in SO(3).
Linear-response tensors (decomposed into parts even/odd in the magnetic order)
then obey, for every element,

    charge  sigma_ij      :  D = R (x) R
    spin    sigma^k_ij    :  D = Q (x) R (x) R            (spin index rotates with Q)
    orbital sigma^k_ij    :  D = det(R) R (x) R (x) R     (L is a lattice quantity)
    T-even part: chi = D chi ;   T-odd part: chi = theta D chi

With SOC (magnetic groups) U is locked to the lattice, U = theta det(R) R, and the
formulas reduce to the relativistic ones used in symphase.tensors.

Classification of collinear magnets (Smejkal, Sinova, Jungwirth, PRX 12, 031042
and 040501 (2022)):
    ferromagnet / ferrimagnet  : no operation exchanges the spin sublattices
    antiferromagnet (Kramers)  : sublattices connected by a translation [C2||t] or by
                                 inversion [C2||P]  ->  no non-relativistic spin splitting
    altermagnet                : sublattices connected only by (screw/glide/proper/improper)
                                 rotations -> d-, g- or i-wave spin splitting

The momentum dependence of the non-relativistic spin polarisation s(k) is obtained by
solving   s(theta R k) = theta Q s(k)   for polynomial vector fields of increasing degree.
"""

import itertools
import math

import numpy as np

AX = "xyz"
SUP = {"1": "¹", "2": "²"}
_WAVE = {0: "s-wave (Zeeman-like, uncompensated)", 1: "p-wave", 2: "d-wave", 3: "f-wave",
         4: "g-wave", 5: "h-wave", 6: "i-wave"}


# ----------------------------------------------------------------------------
def rot_axis_angle(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def perp_vector(n):
    n = np.asarray(n, float)
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(n, t)
    return v / np.linalg.norm(v)


def describe_spin_op(U, tol=1e-6):
    """human-readable spin part of [U||g]"""
    th = 1 if np.linalg.det(U) > 0 else -1
    Q = th * U
    tr = np.trace(Q)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1) / 2))))
    if ang < 1e-3:
        base = "E"
    else:
        w, v = np.linalg.eig(Q)
        ax = np.real(v[:, np.argmin(np.abs(w - 1))])
        ax = ax / np.linalg.norm(ax)
        for x in ax:
            if abs(x) > 1e-8:
                if x < 0:
                    ax = -ax
                break
        n = 360.0 / ang
        lab = f"C{int(round(n))}" if abs(n - round(n)) < 1e-3 else f"R({ang:.1f}°)"
        base = f"{lab}[{ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f}]"
    return base + ("·T" if th < 0 else "")


# ----------------------------------------------------------------------------
def atom_permutation(frac, types, R, t, tol=1e-3):
    """j = perm[i] such that  R x_i + t  ==  x_j  (mod 1, same type), or None"""
    new = (frac @ np.asarray(R).T) + t
    perm = np.empty(len(frac), int)
    for i, x in enumerate(new):
        d = frac - x
        d -= np.round(d)
        dist = np.linalg.norm(d, axis=1)
        cand = np.where((dist < tol) & (np.asarray(types) == types[i]))[0]
        if len(cand) != 1:
            return None
        perm[i] = cand[0]
    return perm


def procrustes(A, B):
    """orthogonal U (det +-1) minimising || U A - B ||  (columns = vectors)"""
    M = B @ A.T
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def classify_order(mags, tol=1e-3):
    """('none'|'collinear'|'coplanar'|'noncoplanar', axis or normal)"""
    nz = mags[np.linalg.norm(mags, axis=1) > tol]
    if len(nz) == 0:
        return "none", None
    u, s, vt = np.linalg.svd(nz, full_matrices=False)
    s = s / s[0]
    def canon(v):
        for x in v:
            if abs(x) > 1e-8:
                return v if x > 0 else -v
        return v
    if len(s) < 2 or s[1] < 1e-4:
        return "collinear", canon(vt[0])
    if len(s) < 3 or s[2] < 1e-4:
        return "coplanar", canon(vt[2])
    return "noncoplanar", None


def spin_only_generators(kind, axis):
    """list of U matrices generating the spin-only group (with lattice part E)"""
    E = np.eye(3)
    if kind == "none":
        return [rot_axis_angle([0, 0, 1], 1.0), rot_axis_angle([1, 0, 0], 1.0), -E]
    if kind == "collinear":
        p = perp_vector(axis)
        return [rot_axis_angle(axis, 1.0), -rot_axis_angle(p, math.pi)]   # C_inf(n), C2_perp T
    if kind == "coplanar":
        return [-rot_axis_angle(axis, math.pi)]                            # C2_normal T
    return []


class SpinOp:
    __slots__ = ("R", "t", "Rc", "U", "theta", "Q", "coset", "lattice_only")

    def __init__(self, R, t, Rc, U, coset, lattice_only=False):
        self.R, self.t, self.Rc, self.U = R, t, Rc, U
        self.theta = 1 if np.linalg.det(U) > 0 else -1
        self.Q = self.theta * U
        self.coset, self.lattice_only = coset, lattice_only


def spin_group(lat, frac, types, mags, rotations, translations, symprec=1e-3, mtol=0.05):
    """return dict describing the non-relativistic spin group"""
    A = lat.T
    Ai = np.linalg.inv(A)
    mags = np.asarray(mags, float)
    kind, axis = classify_order(mags)
    norms = np.linalg.norm(mags, axis=1)
    scale = max(norms.max(), 1e-8)
    lattice_ops, rejected = [], 0
    for R, t in zip(rotations, translations):
        perm = atom_permutation(frac, types, R, t, tol=max(symprec * 5, 1e-3))
        if perm is None:
            rejected += 1
            continue
        Rc = A @ R @ Ai
        if kind == "none":
            U = np.eye(3)
        else:
            src = mags.T
            dst = mags[perm].T          # m_{g(i)}
            if kind == "collinear":
                n = axis
                s_src = mags @ n
                s_dst = mags[perm] @ n
                if np.allclose(s_dst, s_src, atol=mtol * scale):
                    U = np.eye(3)
                elif np.allclose(s_dst, -s_src, atol=mtol * scale):
                    U = rot_axis_angle(perp_vector(n), math.pi)   # proper spin flip C2_perp
                else:
                    rejected += 1
                    continue
            else:
                U = procrustes(src, dst)
                if np.max(np.abs(U @ src - dst)) > mtol * scale:
                    rejected += 1
                    continue
        coset = "H" if np.allclose(U, np.eye(3), atol=1e-6) else "A"
        lattice_ops.append(SpinOp(np.array(R), np.array(t), Rc, U, coset))
    gens = [SpinOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), U, "spin-only", True)
            for U in spin_only_generators(kind, axis)]
    return dict(kind=kind, axis=axis, ops=lattice_ops, spin_only=gens,
                n_lattice_G=len(rotations), n_rejected=rejected,
                net=mags.sum(0), mags=mags)


# ----------------------------------------------------------------------------
def classify_magnet(sg, tol_net=0.02):
    """FM / FiM / compensated FiM / AFM (PT or t) / altermagnet  (collinear) + reason"""
    kind = sg["kind"]
    mags = sg["mags"]
    mmax = max(np.linalg.norm(mags, axis=1).max(), 1e-9)
    comp = np.linalg.norm(sg["net"]) < tol_net * mmax * max(1, len(mags)) ** 0.5
    if kind == "none":
        return "non-magnetic", "no moments: spin fully decoupled, bands spin degenerate"
    if kind != "collinear":
        return f"{kind} magnet", "non-collinear order: see the spin-polarisation form s(k) below"
    A = [o for o in sg["ops"] if o.coset == "A"]
    if not A:
        if comp:
            return "compensated ferrimagnet", "no operation exchanges the spin sublattices (s-wave splitting)"
        return "ferromagnet / ferrimagnet", "no operation exchanges the spin sublattices"
    tr = [o for o in A if np.allclose(o.R, np.eye(3))]
    inv = [o for o in A if np.allclose(o.R, -np.eye(3))]
    if tr:
        return "antiferromagnet (conventional)", \
            f"spin sublattices connected by a translation [C2||t], t = {np.round(tr[0].t, 4)} -> Kramers degenerate"
    if inv:
        return "antiferromagnet (PT-symmetric)", \
            "spin sublattices connected by inversion [C2||P] -> PT symmetry, Kramers degenerate"
    return "ALTERMAGNET", "spin sublattices connected only by rotations / mirrors (no t, no P)"


def laue_class(pointgroup):
    pg = pointgroup.strip()
    table = {
        "-1": ["1", "-1"], "2/m": ["2", "m", "2/m"], "mmm": ["222", "mm2", "mmm"],
        "4/m": ["4", "-4", "4/m"], "4/mmm": ["422", "4mm", "-42m", "-4m2", "4/mmm"],
        "-3": ["3", "-3"], "-3m": ["32", "3m", "-3m", "321", "312", "3m1", "31m", "-3m1", "-31m"],
        "6/m": ["6", "-6", "6/m"], "6/mmm": ["622", "6mm", "-6m2", "-62m", "6/mmm"],
        "m-3": ["23", "m-3"], "m-3m": ["432", "-43m", "m-3m"]}
    for k, v in table.items():
        if pg in v:
            return k
    return None


_TEMPL = {
    "-1": [("-1", None)], "2/m": [("2/m", (0, 1, 0))],
    "mmm": [("m", (1, 0, 0)), ("m", (0, 1, 0)), ("m", (0, 0, 1))],
    "4/m": [("4/m", (0, 0, 1))], "4/mmm": [("4/m", (0, 0, 1)), ("m", (1, 0, 0)), ("m", (1, 1, 0))],
    "-3": [("-3", (0, 0, 1))], "-3m": [("-3", (0, 0, 1)), ("m", "auto")],
    "6/m": [("6/m", (0, 0, 1))], "6/mmm": [("6/m", (0, 0, 1)), ("m", (1, 0, 0)), ("m", (1, -1, 0))],
    "m-3": [("m", (1, 0, 0)), ("-3", (1, 1, 1))],
    "m-3m": [("4/m", (1, 0, 0)), ("-3", (1, 1, 1)), ("m", (1, 1, 0))]}


def _proper_axis_order(Rc):
    P = Rc * np.sign(np.linalg.det(Rc))
    c = max(-1.0, min(1.0, (np.trace(P) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-6:
        return None, 1
    w, v = np.linalg.eig(P)
    ax = np.real(v[:, np.argmin(np.abs(w - 1))])
    return ax / np.linalg.norm(ax), int(round(2 * math.pi / ang))


def spin_laue_symbol(sg, pointgroup, std_lattice, std_rot):
    """Litvin-style spin Laue group, e.g. ²4/¹m²m¹m (collinear magnets)"""
    lc = laue_class(pointgroup)
    if lc is None or sg["kind"] != "collinear":
        return None
    ops = sg["ops"]
    Lc = np.asarray(std_lattice)              # rows: conventional a,b,c in std frame
    Qs = np.asarray(std_rot)

    def direction(uvw):
        d = np.asarray(uvw, float) @ Lc       # std Cartesian
        d = Qs.T @ d                          # back to input Cartesian
        return d / np.linalg.norm(d)

    def status_of(d, order=None):
        best = None
        for o in ops:
            ax, n = _proper_axis_order(o.Rc)
            if ax is None or abs(abs(np.dot(ax, d)) - 1) > 1e-3:
                continue
            if order is not None and n != order:
                continue
            if best is None or n > best[0]:
                best = (n, "1" if o.coset == "H" else "2")
        return best

    out = []
    templ = list(_TEMPL[lc])
    if lc == "-3m":
        cand = [(1, 0, 0), (1, 1, 0), (1, -1, 0), (2, 1, 0)]
        for c in cand:
            if status_of(direction(c), 2):
                templ[1] = ("m", c)
                break
    for sym, uvw in templ:
        if uvw is None or uvw == "auto":
            inv_in_A = any(np.allclose(o.Rc, -np.eye(3)) and o.coset == "A" for o in ops)
            out.append(f"{SUP['2' if inv_in_A else '1']}-1")
            continue
        d = direction(uvw)
        if sym in ("4/m", "6/m", "2/m"):
            n = int(sym[0])
            s_rot = status_of(d, n)
            s_two = status_of(d, 2)
            r = SUP[s_rot[1]] if s_rot else "?"
            m = SUP[s_two[1]] if s_two else "?"
            out.append(f"{r}{n}/{m}m")
        elif sym == "-3":
            s3 = status_of(d, 3)
            out.append(f"{SUP[s3[1]] if s3 else '?'}-3")
        else:  # m  <->  C2 along d in the Laue group
            s2 = status_of(d, 2)
            out.append(f"{SUP[s2[1]] if s2 else '?'}m")
    return "".join(out), lc


def nodal_planes(sg, tol=1e-6):
    """normals of spin-degenerate planes k·n = 0 (A-coset mirrors of the spin Laue group)"""
    normals = []
    for o in sg["ops"]:
        if o.coset != "A":
            continue
        for M in (o.Rc, -o.Rc):            # Laue group contains inversion
            if np.linalg.det(M) < 0 and abs(np.trace(M) - 1) < 1e-6:
                w, v = np.linalg.eig(M)
                n = np.real(v[:, np.argmin(np.abs(w + 1))])
                n /= np.linalg.norm(n)
                if not any(abs(abs(np.dot(n, m)) - 1) < 1e-4 for m in normals):
                    normals.append(n)
    return normals


# ----------------------------------------------------------------------------
def _monomials(d):
    return [e for e in itertools.product(range(d + 1), repeat=3) if sum(e) == d]


def _mono_str(e):
    s = []
    for a, p in zip(("kx", "ky", "kz"), e):
        if p == 1:
            s.append(a)
        elif p > 1:
            s.append(f"{a}{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(p))}")
    return "·".join(s) if s else "1"


def spin_polarisation_form(sg, max_degree=6, nsample=24, seed=0):
    """lowest-degree polynomial s(k) allowed by the spin group; returns (degree, [str,...])"""
    if sg["kind"] == "none":
        return None, []
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(nsample, 3))
    ops = sg["ops"] + sg["spin_only"]
    for d in range(0, max_degree + 1):
        mons = _monomials(d)
        nm = len(mons)
        ev = lambda k: np.array([np.prod(k ** np.array(e)) for e in mons])
        rows = []
        for o in ops:
            th, Q, R = o.theta, o.Q, o.Rc
            for k in K:
                mk = ev(k)
                mrk = ev(th * (R @ k))
                # s_a(theta R k) - theta sum_b Q_ab s_b(k) = 0 ;  s_b = sum_m c[b,m] mon_m
                for a in range(3):
                    row = np.zeros(3 * nm)
                    row[a * nm:(a + 1) * nm] += mrk
                    for b in range(3):
                        row[b * nm:(b + 1) * nm] -= th * Q[a, b] * mk
                    rows.append(row)
        M = np.array(rows)
        _, s, vt = np.linalg.svd(M.T @ M)
        s = np.sqrt(np.abs(s))
        rank = int(np.sum(s > 1e-8 * max(1.0, s[0])))
        B = vt[rank:]
        if len(B) == 0:
            continue
        # reduce for readability
        from .tensors import rref, coef_str
        E, piv = rref(B)
        forms = []
        for r in range(len(piv)):
            comps = []
            for a in range(3):
                terms = []
                for m in range(nm):
                    cs = coef_str(E[r, a * nm + m])
                    if cs is None:
                        continue
                    sign, c = cs
                    terms.append((sign, (c + "·" if c else "") + _mono_str(mons[m])))
                if terms:
                    t = ("-" if terms[0][0] == "-" else "") + terms[0][1] + \
                        "".join(f" {sg_} {tt}" for sg_, tt in terms[1:])
                    comps.append(f"s_{AX[a]} = {t}")
            forms.append(",  ".join(comps))
        return d, forms
    return None, []


def wave_label(degree, kind):
    if degree is None:
        return "no non-relativistic spin splitting (spin-degenerate bands)"
    return _WAVE.get(degree, f"degree-{degree}")


# ----------------------------------------------------------------------------
#  spin-space-group identification
# ----------------------------------------------------------------------------
def identify_ssg(lat, frac, types, sg, symprec=1e-3, mag_symprec=0.05):
    """Identify the spin space group.

    collinear: the non-trivial part of a collinear SSG, {[E||h]} + {[C2perp||a]}, is isomorphic
      to a black-and-white (type I, III or IV) magnetic space group with C2perp -> T.  There are
      230 + 674 + 517 = 1421 of them.  We label the SSG by the BNS number of that group, found
      with spglib from the scalar moments m_i . n, and write it as  <BNS label> x inf-m1.
    non-magnetic: SSG = G x O(3) (G = crystal space group).
    coplanar / non-coplanar: the lattice group G, the spin-preserving subgroup and the image of
      the spin parts (spin point group) are reported; no standard number is assigned.
    """
    import spglib
    from .tensors import msg_labels, _g
    kind, axis = sg["kind"], sg["axis"]
    out = dict(kind=kind)
    if kind == "none":
        out.update(label="G × O(3)  (spin fully decoupled)", number=None, family="non-magnetic")
        return out
    if kind == "collinear":
        s = sg["mags"] @ axis
        ds = spglib.get_magnetic_symmetry_dataset((lat, frac, types, s), symprec=symprec,
                                                  mag_symprec=mag_symprec)
        t = spglib.get_magnetic_spacegroup_type(_g(ds, "uni_number"))
        bns = _g(t, "bns_number")
        lab = msg_labels(bns)[0] or bns
        typ = int(_g(t, "type"))
        out.update(number=bns, bw_symbol=lab, bw_type=typ, uni=int(_g(ds, "uni_number")),
                   label=f"{lab} (BNS {bns}, type {['', 'I', 'II', 'III', 'IV'][typ]}) ⊗ ∞m1",
                   family="collinear (1 of 1421 collinear SSGs; C2⊥ ↔ T correspondence)",
                   n_nontrivial_ops=int(_g(ds, "n_operations")))
        return out
    # non-collinear: spin point group = image of the proper spin parts (+ T)
    Qs = []
    for o in sg["ops"]:
        M = o.U
        if not any(np.allclose(M, q, atol=1e-5) for q in Qs):
            Qs.append(M)
    nH = sum(o.coset == "H" for o in sg["ops"])
    out.update(number=None, family=f"{kind} (no standard number assigned)",
               spin_point_group_order=len(Qs),
               label=f"G = {len(sg['ops'])} lattice ops, spin-preserving subgroup {nH} ops, "
                     f"spin point group of order {len(Qs)} ⊗ {'m1' if kind == 'coplanar' else '1'}")
    return out
