# -*- coding: utf-8 -*-
"""
symphase.twod — MXenes and other 2D materials: structures, relaxation, convex hulls, phase diagrams
==================================================================================================

MXenes  M_{n+1} X_n T_x   (M = transition metal(s), X = C / N, T = surface termination: F, O, OH, Cl, Br, S, ... or none)

    from symphase.twod import build_mxene, mxene_hull
    s = build_mxene("Cr", "C", n=1, T="F")           # Cr2CF2 monolayer (hexagonal, vacuum along z)
    res = mxene_hull(["Cr2CF2"], ...)                 # relax, energies, hull, E_hull, decomposition, phase diagram

Command line (asks for the elements and the passivation when they are not given):

    symphase mxene                                    # interactive: M? X? n? termination? site?
    symphase mxene --M Cr --X C --T F --hull          # Cr2CF2 → relax, hull vs competing bulk phases, plots
    symphase mxene --M Mo --X C --T none --hull       # bare Mo2C
    symphase mxene --M Mo,Ti --X C --n 2 --T O        # ordered double-M MXene Mo2TiC2O2 (outer Mo, inner Ti)

Generic 2D helpers: find_layers, is_layered, add_vacuum, monolayer_from_bulk, exfoliation_energy, relax (in-plane
cell relaxation for slabs), formation_energy, write_vasp (POSCAR + INCAR template with dipole correction).

Energies.  Materials Project / OQMD are often unreachable from compute nodes; the default energy model is the
universal machine-learned potential CHGNet (pip install chgnet; trained on Materials Project PBE/PBE+U relaxations,
energies on the MP2020-compatible scale, so elements, carbides, nitrides and fluorides can be mixed in one hull).
It is a screening tool: replace the energies by your own VASP results (--energies CSV, or write_vasp + symphase hull)
for publication.  Competing phases come from a built-in library of crystallographic prototypes (AFLOW
encyclopedia / ICSD parameters), relaxed with the same model; add your own with extra=[...] / --extra POSCAR.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# =============================================================================
#  crystallographic prototypes of competing bulk phases
# =============================================================================
# each entry: (space group, lattice dict, [(element, (x, y, z)), ...])  — conventional settings of pymatgen
_TAU = 0.25


def _ortho(a, b, c):
    return dict(kind="ortho", a=a, b=b, c=c)


def _hexl(a, c):
    return dict(kind="hex", a=a, c=c)


def _cubic(a):
    return dict(kind="cubic", a=a)


def _tetra(a, c):
    return dict(kind="tetra", a=a, c=c)


def _molecular(el, a, bond):
    """α-N2-type molecular crystal (Pa-3, molecules along ⟨111⟩) — the molecular-solid reference of MP"""
    x = bond / (2 * math.sqrt(3) * a)
    return ("Pa-3", _cubic(a), [(el, (x, x, x))])


PROTOTYPES = {
    # elements --------------------------------------------------------------------------------------------
    "Cr_bcc": ("Im-3m", _cubic(2.88), [("Cr", (0, 0, 0))]),
    "Mo_bcc": ("Im-3m", _cubic(3.15), [("Mo", (0, 0, 0))]),
    "W_bcc": ("Im-3m", _cubic(3.17), [("W", (0, 0, 0))]),
    "V_bcc": ("Im-3m", _cubic(3.03), [("V", (0, 0, 0))]),
    "Nb_bcc": ("Im-3m", _cubic(3.30), [("Nb", (0, 0, 0))]),
    "Ta_bcc": ("Im-3m", _cubic(3.30), [("Ta", (0, 0, 0))]),
    "Fe_bcc": ("Im-3m", _cubic(2.87), [("Fe", (0, 0, 0))]),
    "Ni_fcc": ("Fm-3m", _cubic(3.52), [("Ni", (0, 0, 0))]),
    "Co_hcp": ("P6_3/mmc", _hexl(2.51, 4.07), [("Co", (1 / 3, 2 / 3, 0.25))]),
    "Ti_hcp": ("P6_3/mmc", _hexl(2.95, 4.68), [("Ti", (1 / 3, 2 / 3, 0.25))]),
    "Zr_hcp": ("P6_3/mmc", _hexl(3.23, 5.15), [("Zr", (1 / 3, 2 / 3, 0.25))]),
    "Hf_hcp": ("P6_3/mmc", _hexl(3.19, 5.05), [("Hf", (1 / 3, 2 / 3, 0.25))]),
    "Sc_hcp": ("P6_3/mmc", _hexl(3.31, 5.27), [("Sc", (1 / 3, 2 / 3, 0.25))]),
    "Mn_bcc": ("Im-3m", _cubic(2.86), [("Mn", (0, 0, 0))]),
    "C_graphite": ("P6_3/mmc", _hexl(2.46, 6.71), [("C", (0, 0, 0.25)), ("C", (1 / 3, 2 / 3, 0.25))]),
    "C_diamond": ("Fd-3m", _cubic(3.57), [("C", (0, 0, 0))]),
    "N2_alpha": _molecular("N", 5.66, 1.10),
    "F2_mol": _molecular("F", 5.40, 1.42),
    "Cl2_mol": _molecular("Cl", 6.40, 1.99),
    "O2_mol": _molecular("O", 5.40, 1.21),
    "H2_mol": _molecular("H", 4.50, 0.74),
    "S_mol": _molecular("S", 6.0, 1.89),
    # Cr–C ------------------------------------------------------------------------------------------------
    "Cr3C2_D510": ("Pnma", _ortho(5.5329, 5.5329 * 0.511305102207, 5.5329 * 2.07339731425),
                   [("C", (0.1008, _TAU, 0.2055)), ("C", (0.2432, _TAU, -0.0464)), ("Cr", (0.0157, _TAU, 0.4015)),
                    ("Cr", (0.1808, _TAU, 0.7737)), ("Cr", (0.8691, _TAU, -0.0688))]),
    "Cr7C3_D101": ("Pnma", _ortho(4.526, 4.526 * 1.54882898807, 4.526 * 2.68272205038),
                   [("C", (0.4594, _TAU, 0.5629)), ("Cr", (0.0579, _TAU, 0.6261)), ("Cr", (0.2501, _TAU, 0.2063)),
                    ("Cr", (0.2619, _TAU, 0.4165)), ("C", (0.0288, 0.0291, 0.3428)), ("Cr", (0.0565, 0.0642, 0.8119)),
                    ("Cr", (0.2509, 0.0657, 0.0218))]),
    "Cr23C6_D84": ("Fm-3m", _cubic(10.65), [("Cr", (0, 0, 0)), ("Cr", (0.25, 0.25, 0.25)), ("Cr", (0.6191,) * 3),
                                           ("Cr", (0, 0.6699, 0.6699)), ("C", (0.2765, 0, 0))]),
    # Cr–F, Ni–F (rutile / VF3-type) ------------------------------------------------------------------------
    "CrF2_rutile": ("P4_2/mnm", _tetra(4.73, 3.30), [("Cr", (0, 0, 0)), ("F", (0.30, 0.30, 0))]),
    "CrF3_D012": ("R-3c", _hexl(4.99, 13.21), [("Cr", (0, 0, 0)), ("F", (0.59, 0, 0.25))]),
    "NiF2_rutile": ("P4_2/mnm", _tetra(4.65, 3.08), [("Ni", (0, 0, 0)), ("F", (0.302, 0.302, 0))]),
    "NiF3_D012": ("R-3c", _hexl(4.95, 13.0), [("Ni", (0, 0, 0)), ("F", (0.59, 0, 0.25))]),
    # Ni–N ------------------------------------------------------------------------------------------------
    "Ni3N_epsFe3N": ("P6_322", _hexl(4.62, 4.31), [("Ni", (0.328, 0, 0)), ("N", (1 / 3, 2 / 3, 0.25))]),
    "Ni4N_antiperov": ("Pm-3m", _cubic(3.72), [("Ni", (0, 0, 0)), ("Ni", (0, 0.5, 0.5)), ("N", (0.5, 0.5, 0.5))]),
    # Mo–C ------------------------------------------------------------------------------------------------
    "Mo2C_beta_Pbcn": ("Pbcn", _ortho(4.73, 6.03, 5.20), [("Mo", (0.245, 0.128, 0.0827)), ("C", (0, 0.375, 0.25))]),
    "Mo2C_W2C_P-3m1": ("P-3m1", _hexl(3.00, 4.73), [("Mo", (1 / 3, 2 / 3, 0.25)), ("C", (0, 0, 0))]),
    "MoC_WC": ("P-6m2", _hexl(2.90, 2.81), [("Mo", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0.5))]),
    "MoC_NaCl": ("Fm-3m", _cubic(4.27), [("Mo", (0, 0, 0)), ("C", (0.5, 0.5, 0.5))]),
}


def _lattice(d):
    from pymatgen.core import Lattice
    if d["kind"] == "cubic":
        return Lattice.cubic(d["a"])
    if d["kind"] == "hex":
        return Lattice.hexagonal(d["a"], d["c"])
    if d["kind"] == "tetra":
        return Lattice.tetragonal(d["a"], d["c"])
    return Lattice.orthorhombic(d["a"], d["b"], d["c"])


def prototype(name):
    """pymatgen Structure of a library prototype"""
    from pymatgen.core import Structure
    sg, lat, sites = PROTOTYPES[name]
    return Structure.from_spacegroup(sg, _lattice(lat), [e for e, _ in sites], [c for _, c in sites])


def competing_prototypes(elements):
    """library prototypes whose elements are a subset of `elements`"""
    els = set(elements)
    out = {}
    for name, (_, _, sites) in PROTOTYPES.items():
        if {e for e, _ in sites} <= els:
            out[name] = prototype(name)
    return out


# =============================================================================
#  MXene builder
# =============================================================================
_HEX_SITES = {"A": (0.0, 0.0), "B": (1 / 3, 2 / 3), "C": (2 / 3, 1 / 3)}
_GROUP = {"OH": [("O", 0.0), ("H", 0.97)], "SH": [("S", 0.0), ("H", 1.35)]}


def parse_metals(M, n):
    """'Cr' -> [Cr]*(n+1); 'Mo,Ti' -> outer Mo, inner Ti (ordered double-M MXene)"""
    ms = [m.strip() for m in (M.split(",") if isinstance(M, str) else M) if m.strip()]
    if len(ms) == 1:
        return ms * (n + 1)
    if len(ms) == 2:
        return [ms[0]] + [ms[1]] * (n - 1) + [ms[0]]
    if len(ms) == n + 1:
        return ms
    raise ValueError("give one metal, 'outer,inner', or n+1 metals from top to bottom")


def mxene_formula(M, X, n=1, T=None):
    """'Cr','C',1,'F' -> 'Cr2CF2' ; 'Mo,Ti','C',2,'O' -> 'Mo2TiC2O2'"""
    from collections import Counter
    c = Counter(parse_metals(M, n))
    s = "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in dict.fromkeys(parse_metals(M, n)))
    s += f"{X}{n if n > 1 else ''}"
    if T and str(T).lower() not in ("none", "bare", "-", ""):
        s += f"({T})2" if T in _GROUP else f"{T}2"
    return s


def build_mxene(M, X, n=1, T=None, site="fcc", a=None, vacuum=20.0, dz_mx=None, dz_t=None):
    """M_{n+1} X_n T_2 monolayer in a hexagonal cell (c along z, vacuum `vacuum` Å).

    Stacking of the 2n+1 metal/X layers follows the ABC sequence of the parent MAX phase.  Termination sites
    (both surfaces, mirror-symmetric choice):
      'fcc' : above the hollow not occupied by the X layer below (above the opposite-side metal) — 'model I'
      'hcp' : above the X atom of the outermost M–X–M sandwich                                      — 'model II'
      'top' : on top of the outer metal
    """
    from pymatgen.core import Element, Lattice, Structure
    metals = parse_metals(M, n)
    if a is None:
        r = np.mean([Element(m).metallic_radius or 1.35 for m in metals])
        a = 2.36 * r                                                 # ≈ 3.0–3.3 Å for 3d/4d MXenes
    dz_mx = dz_mx or 1.15
    layers = []                                                      # (element, site letter) top → bottom
    cyc = ["B", "A", "C"]
    for i in range(2 * n + 1):
        layers.append((metals[i // 2] if i % 2 == 0 else X, cyc[i % 3]))
    nl = len(layers)
    zs = [(nl - 1) / 2 * dz_mx - i * dz_mx for i in range(nl)]
    species, coords = [], []
    for (el, st), z in zip(layers, zs):
        species.append(el)
        coords.append((*_HEX_SITES[st], z))
    term = None if T is None or str(T).lower() in ("none", "bare", "-", "") else str(T)
    if term:
        parts = _GROUP.get(term, [(term, 0.0)])
        dt = dz_t or (1.0 if parts[0][0] in ("F", "O", "H") else 1.4)

        def tsite(outer, nxt):
            if site == "top":
                return outer
            if site == "hcp":
                return nxt
            return [s for s in "ABC" if s not in (outer, nxt)][0]
        top = tsite(layers[0][1], layers[1][1])
        bot = tsite(layers[-1][1], layers[-2][1])
        for el, h in parts:
            species.append(el)
            coords.append((*_HEX_SITES[top], zs[0] + dt + h))
            species.append(el)
            coords.append((*_HEX_SITES[bot], zs[-1] - dt - h))
    thick = max(c[2] for c in coords) - min(c[2] for c in coords)
    c = thick + vacuum
    lat = Lattice.hexagonal(a, c)
    frac = [(x, y, 0.5 + z / c) for x, y, z in coords]
    s = Structure(lat, species, frac)
    return s.get_sorted_structure()


def prompt_mxene(inp=input, out=print):
    """ask for M, X, n, termination and site interactively; returns a dict for build_mxene"""
    out("MXene builder  M(n+1) X(n) T2   — e.g. Cr + C + F  →  Cr2CF2,   Ni + N + F → Ni2NF2,   Mo + C + none → Mo2C")
    M = inp("  transition metal M (e.g. Cr, Mo, Ti; ordered double-M as outer,inner e.g. Mo,Ti): ").strip()
    X = inp("  X element (C or N) [C]: ").strip() or "C"
    n = inp("  n in M(n+1)X(n) (1, 2, 3) [1]: ").strip()
    n = int(n) if n else 1
    T = inp("  passivation / termination (F, O, OH, Cl, Br, S, none) [F]: ").strip() or "F"
    site = "auto"
    if T.lower() not in ("none", "bare", "-"):
        site = inp("  termination site (fcc, hcp, top, auto = lowest energy) [auto]: ").strip() or "auto"
    return dict(M=M, X=X, n=n, T=None if T.lower() in ("none", "bare", "-") else T, site=site)


# =============================================================================
#  generic 2D tools
# =============================================================================
def find_layers(structure, gap=2.0, axis=2):
    """group atoms into layers separated (along `axis`, periodic) by empty gaps larger than `gap` Å"""
    c = structure.lattice.abc[axis]
    z = np.sort(np.mod(structure.cart_coords[:, axis] if axis == 2 else structure.frac_coords[:, axis] * c, c))
    order = np.argsort(np.mod(structure.cart_coords[:, axis], c))
    zz = np.mod(structure.cart_coords[order, axis], c)
    d = np.diff(np.r_[zz, zz[0] + c])
    cuts = np.where(d > gap)[0]
    if len(cuts) == 0:
        return [list(range(len(structure)))]
    groups = []
    start = (cuts[-1] + 1) % len(zz)
    idx = list(np.r_[order[start:], order[:start]])
    bounds = sorted(((c_ - start) % len(zz)) + 1 for c_ in cuts)
    prev = 0
    for b in bounds:
        groups.append([int(i) for i in idx[prev:b]])
        prev = b
    return [g for g in groups if g]


def vacuum_thickness(structure, axis=2):
    c = structure.lattice.abc[axis]
    zz = np.sort(np.mod(structure.cart_coords[:, axis], c))
    return float(np.max(np.diff(np.r_[zz, zz[0] + c])))


def is_layered(structure, gap=2.0):
    return len(find_layers(structure, gap)) > 1 or vacuum_thickness(structure) > gap


def add_vacuum(structure, vacuum=20.0):
    """slab with `vacuum` Å of empty space along c (atoms centred); in-plane lattice unchanged"""
    from pymatgen.core import Lattice, Structure
    cart = structure.cart_coords.copy()
    c_old = structure.lattice.matrix[2]
    if abs(c_old[0]) > 1e-6 or abs(c_old[1]) > 1e-6:
        raise ValueError("c must be along z for add_vacuum")
    zz = cart[:, 2]
    thick = zz.max() - zz.min()
    m = structure.lattice.matrix.copy()
    m[2] = [0, 0, thick + vacuum]
    cart[:, 2] += (thick + vacuum) / 2 - (zz.max() + zz.min()) / 2
    return Structure(Lattice(m), structure.species, cart, coords_are_cartesian=True)


def monolayer_from_bulk(bulk, gap=2.0, vacuum=20.0, which=0):
    """isolate one layer of a layered bulk crystal (layers ⟂ c) and add vacuum"""
    from pymatgen.core import Structure
    layers = find_layers(bulk, gap)
    if len(layers) < 2 and vacuum_thickness(bulk) < gap:
        raise ValueError("no van der Waals gap found along c")
    g = layers[which]
    cart = bulk.cart_coords[g]
    c = bulk.lattice.matrix[2][2]
    cart[:, 2] = np.mod(cart[:, 2] - cart[0, 2] + c / 2, c) - c / 2 + cart[0, 2]
    s = Structure(bulk.lattice, [bulk.species[i] for i in g], cart, coords_are_cartesian=True)
    return add_vacuum(s, vacuum)


def exfoliation_energy(e_bulk_per_atom, e_mono_per_atom, n_atoms_mono, area):
    """E_exf = (E_mono − E_bulk) per area (meV/Å²) — both per atom, same functional"""
    return 1000.0 * (e_mono_per_atom - e_bulk_per_atom) * n_atoms_mono / area


def in_plane_area(structure):
    m = structure.lattice.matrix
    return float(np.linalg.norm(np.cross(m[0], m[1])))


# =============================================================================
#  energies (CHGNet) and relaxation
# =============================================================================
_MODEL = {}


def _chgnet():
    if "m" not in _MODEL:
        try:
            from chgnet.model import CHGNet
        except ImportError as err:                                   # pragma: no cover
            raise ImportError("energy model CHGNet not installed: pip install chgnet  (or pass your own energies)") from err
        _MODEL["m"] = CHGNet.load(verbose=False) if "verbose" in CHGNet.load.__code__.co_varnames else CHGNet.load()
    return _MODEL["m"]


def relax(structure, two_d=None, fmax=0.02, steps=600, rattle=0.02, seed=0, relax_cell=True):  # noqa: C901
    """CHGNet relaxation (ASE FIRE + FrechetCellFilter).  2D slabs: only the in-plane cell relaxes (vacuum kept).
    A small rattle breaks the starting symmetry so that, e.g., Jahn–Teller distortions can develop.
    Returns (relaxed Structure, total energy eV)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from chgnet.model.dynamics import CHGNetCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    if two_d is None:
        two_d = vacuum_thickness(structure) > 6.0
    atoms = AseAtomsAdaptor.get_atoms(structure)
    if rattle:
        atoms.rattle(rattle, seed=seed)
    atoms.calc = CHGNetCalculator(model=_chgnet(), use_device="cpu")
    if relax_cell:
        mask = [True, True, False, False, False, True] if two_d else None
        obj = FrechetCellFilter(atoms, mask=mask)
    else:
        obj = atoms
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        FIRE(obj, logfile=None).run(fmax=fmax, steps=steps)
    e = float(atoms.get_potential_energy())
    s = AseAtomsAdaptor.get_structure(atoms)
    for prop in list(s.site_properties):
        s.remove_site_property(prop)
    return s, e


def energy(structure):
    """single-point CHGNet energy (eV, total)"""
    p = _chgnet().predict_structure(structure)
    return float(p["e"]) * len(structure)


# =============================================================================
#  hulls
# =============================================================================
def _cache_load(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _cache_save(path, data):
    if path:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, default=float)


def relaxed_library(elements, cache=None, extra=None, fmax=0.02, verbose=True):
    """relax all library prototypes (+ extra {name: Structure}) in the chemical space; cached by name.
    Returns {name: dict(formula, energy (eV/f.u. as relaxed cell), natoms, structure)}"""
    from pymatgen.core import Structure
    data = _cache_load(cache)
    protos = competing_prototypes(elements)
    protos.update(extra or {})
    out = {}
    for name, s in protos.items():
        key = f"{name}"
        if key not in data:
            if verbose:
                print(f"  relaxing {name:18s} ({s.composition.reduced_formula}, {len(s)} atoms) ...", flush=True)
            rs, e = relax(s, two_d=False, fmax=fmax)
            data[key] = dict(formula=rs.composition.formula.replace(" ", ""), energy=e, natoms=len(rs),
                             structure=rs.as_dict())
            _cache_save(cache, data)
        d = data[key]
        out[name] = dict(formula=d["formula"], energy=d["energy"], natoms=d["natoms"],
                         structure=Structure.from_dict(d["structure"]))
    return out


def relax_mxene(M, X, n=1, T=None, site="auto", vacuum=20.0, fmax=0.02, verbose=True):
    """build and relax a MXene; site='auto' relaxes fcc, hcp and top terminations and keeps the lowest.
    Returns dict(structure, energy, site, sites={site: E/atom}, formula)"""
    sites = ["fcc", "hcp", "top"] if (T and site == "auto") else [site if T else "fcc"]
    best, res = None, {}
    for st in sites:
        s0 = build_mxene(M, X, n, T, site=st, vacuum=vacuum)
        rs, e = relax(s0, two_d=True, fmax=fmax)
        res[st] = e / len(rs)
        if verbose:
            print(f"  {mxene_formula(M, X, n, T):10s} site {st:3s}: E = {e / len(rs):.4f} eV/atom, a = {rs.lattice.a:.3f} Å",
                  flush=True)
        if best is None or e < best[1] - 1e-3 * len(rs):              # degenerate within 1 meV/atom: keep first
            best = (rs, e, st)
    return dict(structure=best[0], energy=best[1], site=best[2], sites=res, formula=mxene_formula(M, X, n, T))


def mxene_hull(targets, elements=None, cache=None, extra=None, energies=None, fmax=0.02, verbose=True):
    """convex hull of 2D targets against the bulk library.

    targets  : {label: (Structure, total energy eV)} (e.g. from relax_mxene)
    energies : optional {formula or name: energy per atom (eV)} that REPLACE the model energies (e.g. your VASP)
    Returns dict(hull=HullAnalysis, rows=[...], library=..., targets=...)."""
    from .hull import HullAnalysis, make_entry
    if elements is None:
        elements = sorted({el.symbol for s, _ in targets.values() for el in s.composition.elements})
    lib = relaxed_library(elements, cache=cache, extra=extra, fmax=fmax, verbose=verbose)
    entries = []
    for name, d in lib.items():
        e = d["energy"]
        if energies and (name in energies or d["formula"] in energies):
            e = energies.get(name, energies.get(d["formula"])) * d["natoms"]
        entries.append(make_entry(d["formula"], e, name=name))
    tentries = {}
    for label, (s, e) in targets.items():
        f = s.composition.formula.replace(" ", "")
        if energies and label in energies:
            e = energies[label] * len(s)
        te = make_entry(f, e, name=label)
        tentries[label] = te
        entries.append(te)
    h = HullAnalysis(entries, elements=elements)
    rows = []
    for label, te in tentries.items():
        dec = h.decomposition(te)
        rows.append(dict(label=label, formula=te.composition.reduced_formula,
                         e_form=float(h.form_energy(te)), e_hull=float(h.e_above_hull(te)),
                         decomposition={(k if isinstance(k, str) else getattr(k, "name", None) or k.composition.reduced_formula):
                                        float(v) for k, v in dec.items()}))
    return dict(hull=h, rows=rows, library=lib, targets=tentries)


def load_hull(directory):
    """rebuild the hull result of `symphase mxene --hull` from its output folder (no relaxation)"""
    from pymatgen.core import Structure
    from .hull import HullAnalysis, make_entry
    lib = json.load(open(os.path.join(directory, "library_chgnet.json")))
    mx = json.load(open(os.path.join(directory, "mxene.json")))
    s = Structure.from_file(os.path.join(directory, "POSCAR"))
    ents = [make_entry(v["formula"], v["energy"], name=k) for k, v in lib.items()]
    t = make_entry(s.composition.formula.replace(" ", ""), mx["energy_eV"], name=mx["formula"])
    h = HullAnalysis(ents + [t])
    rows = json.load(open(os.path.join(directory, "hull.json")))
    return dict(hull=h, rows=rows, library=lib, targets={mx["formula"]: t}, structure=s, mxene=mx)


def formation_energy(result, label):
    return [r for r in result["rows"] if r["label"] == label][0]["e_form"]


def plot_phase_diagram(result, save=None, title=None, ax=None):
    """ternary (or binary) hull with the 2D targets marked (E_hull colour); returns the figure"""
    import matplotlib.pyplot as plt
    from . import hull_plots as hp
    h = result["hull"]
    n = len(h.elements)
    if n == 3:
        fig = hp.plot_ternary(h, ax=ax, show_unstable=True, label_ehull=0.0)
    elif n == 2:
        fig = hp.plot_binary(h, ax=ax, show_unstable=True, label_ehull=0.0)
    else:
        raise ValueError("phase diagrams are drawn for 2 or 3 elements; use symphase hull for more")
    fig = fig if fig is not None else plt.gcf()
    a = ax if ax is not None else fig.axes[0]
    for r in result["rows"]:
        te = result["targets"][r["label"]]
        comp = te.composition.fractional_composition
        if n == 3:
            v = np.array([comp[e] for e in h.elements])
            x, y = hp._tri_xy(v)
        else:
            x, y = comp[h.elements[1]], r["e_form"]
        a.scatter([x], [y], marker="*", s=260, color="#c0392b", edgecolor="k", zorder=10)
        a.annotate(f"{r['label']} (2D)\n+{1000 * r['e_hull']:.0f} meV/atom", (x, y), textcoords="offset points",
                   xytext=(10, 8), fontsize=9, color="#c0392b", fontweight="bold", zorder=11)
    if title:
        a.set_title(title, fontsize=11)
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def termination_energies(M, X, n=1, terminations=("F",), mu0=None, cache=None, site="auto", fmax=0.02, verbose=True):
    """surface (termination) phase diagram of one MXene: for each termination T,
        ΔG_T(Δμ) = [E(M_{n+1}X_n T_2) − E(M_{n+1}X_n) − 2 μ_T⁰ − 2 Δμ_T] / A        (eV/Å², per area of one side ×2)
    μ_T⁰ = energy per atom of the elemental reference (library molecular crystal; OH: ½H2 + ½O2 references).
    The termination is favoured where ΔG_T < 0, i.e. for Δμ_T > Δμ* = [E(T) − E(bare) − 2μ⁰]/2 (per formula unit).
    Returns dict(bare=..., T={...: dict(e, delta_mu_star (eV), area)})."""
    bare = relax_mxene(M, X, n, None, fmax=fmax, verbose=verbose)
    out = dict(bare=bare, T={})
    for T in terminations:
        r = relax_mxene(M, X, n, T, site=site, fmax=fmax, verbose=verbose)
        parts = [el for el, _ in _GROUP.get(T, [(T, 0.0)])]
        if mu0 is None:
            lib = relaxed_library(parts, cache=cache, fmax=fmax, verbose=verbose)
            mu = {}
            for d in lib.values():
                els = {e.symbol for e in __import__("pymatgen.core", fromlist=["Composition"]).Composition(d["formula"]).elements}
                if len(els) == 1:
                    el = els.pop()
                    mu[el] = min(mu.get(el, 1e9), d["energy"] / d["natoms"])
        else:
            mu = dict(mu0)
        nfu_b = len(bare["structure"])
        mu_T = sum(mu[e] for e in parts)
        dE = r["energy"] - bare["energy"] - 2 * mu_T                        # per primitive cell (one f.u.)
        area = in_plane_area(r["structure"])
        out["T"][T] = dict(result=r, dE=dE, delta_mu_star=dE / 2.0, area=area, mu0=mu_T, natoms_bare=nfu_b)
    return out


def plot_termination_diagram(data, save=None, ax=None, mu_range=(-5.0, 0.5), label=None):
    """ΔG_T(Δμ_T) per area for the terminations of one MXene (bare = 0)"""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
    else:
        fig = ax.figure
    mus = np.linspace(*mu_range, 200)
    ax.axhline(0, color="0.4", lw=1.6, label=f"{label or 'bare'} (bare)")
    for T, d in data["T"].items():
        g = (d["dE"] - 2 * mus) / d["area"]
        ax.plot(mus, g, lw=2, label=f"{T}-terminated")
        ax.axvline(d["delta_mu_star"], ls=":", color="k", lw=0.8)
    ax.set_xlabel("Δμ (eV)")
    ax.set_ylabel("ΔG (eV/Å²)")
    ax.legend(fontsize=8)
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def write_vasp(structure, directory, magmom=None, encut=520, kspacing=0.25, two_d=True):
    """POSCAR + INCAR template (PBE, MP-compatible settings; dipole correction and ISIF=4-type in-plane relaxation
    are left as comments for slabs) for a DFT follow-up of the screening"""
    from pymatgen.io.vasp import Poscar
    os.makedirs(directory, exist_ok=True)
    Poscar(structure).write_file(os.path.join(directory, "POSCAR"))
    mag = magmom or [3.0 if el.is_transition_metal else 0.0 for el in structure.species]
    lines = [f"SYSTEM = {structure.composition.reduced_formula}", "PREC = Accurate", f"ENCUT = {encut}",
             "EDIFF = 1E-6", "EDIFFG = -0.01", "ISMEAR = 0 ; SIGMA = 0.05", "IBRION = 2 ; NSW = 200",
             "ISIF = 3" if not two_d else "ISIF = 2          # in-plane lattice: scan a, or use a VASP build with constrained ISIF",
             "ISPIN = 2", "MAGMOM = " + " ".join(f"{m:g}" for m in mag), "LORBIT = 11", f"KSPACING = {kspacing}",
             "LASPH = .TRUE.", "LREAL = .FALSE."]
    if two_d:
        lines += ["IDIPOL = 3 ; LDIPOL = .TRUE.   # dipole correction (asymmetric terminations)",
                  "IVDW = 12                       # D3(BJ) if interlayer/adsorbate vdW matters"]
    with open(os.path.join(directory, "INCAR"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return directory


# =============================================================================
#  CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="symphase mxene",
                                 description="MXenes M(n+1)X(n)T2: build, relax (CHGNet), convex hull and phase diagram. "
                                             "Asks for the elements and the termination if they are not given.")
    ap.add_argument("--M", help="transition metal(s): Cr, Mo, ... or outer,inner (e.g. Mo,Ti)")
    ap.add_argument("--X", help="C or N")
    ap.add_argument("--n", type=int, default=None, help="n in M(n+1)X(n) [1]")
    ap.add_argument("--T", help="termination / passivation: F, O, OH, Cl, Br, S or none")
    ap.add_argument("--site", default=None, choices=("fcc", "hcp", "top", "auto"), help="termination site [auto]")
    ap.add_argument("--vacuum", type=float, default=20.0, help="vacuum (Å) [20]")
    ap.add_argument("--hull", action="store_true", help="convex hull against the bulk competing phases + phase diagram")
    ap.add_argument("--extra", nargs="*", default=[], help="extra competing phases (POSCAR/CIF files)")
    ap.add_argument("--energies", help="CSV formula/name,energy_per_atom replacing the model energies (e.g. your VASP)")
    ap.add_argument("--fmax", type=float, default=0.02, help="force tolerance (eV/Å) [0.02]")
    ap.add_argument("--out", default=None, help="output directory [<formula>]")
    ap.add_argument("--vasp", action="store_true", help="also write VASP input folders for the relaxed MXene")
    ap.add_argument("--no-input", action="store_true", help="never prompt (fail if arguments are missing)")
    args = ap.parse_args(argv)
    if args.M is None or args.X is None or args.T is None:
        if args.no_input or not sys.stdin.isatty():
            ap.error("give --M, --X and --T (or run interactively)")
        p = prompt_mxene()
        args.M, args.X, args.T = args.M or p["M"], args.X or p["X"], args.T or (p["T"] or "none")
        args.n = args.n or p["n"]
        args.site = args.site or p["site"]
    n = args.n or 1
    T = None if str(args.T).lower() in ("none", "bare", "-") else args.T
    site = args.site or "auto"
    f = mxene_formula(args.M, args.X, n, T)
    out = args.out or f.replace("(", "").replace(")", "")
    os.makedirs(out, exist_ok=True)
    print(f"\n  MXene {f}  (M = {args.M}, X = {args.X}, n = {n}, termination = {T or 'none'}, site = {site})")
    r = relax_mxene(args.M, args.X, n, T, site=site, vacuum=args.vacuum, fmax=args.fmax)
    s = r["structure"]
    from pymatgen.io.vasp import Poscar
    Poscar(s).write_file(os.path.join(out, "POSCAR"))
    print(f"  lowest: site {r['site']}, a = {s.lattice.a:.3f} Å, E = {r['energy'] / len(s):.4f} eV/atom → {out}/POSCAR")
    with open(os.path.join(out, "mxene.json"), "w") as fh:
        json.dump(dict(formula=f, M=args.M, X=args.X, n=n, T=T, site=r["site"], sites_eV_per_atom=r["sites"],
                       energy_eV=r["energy"], natoms=len(s), a=s.lattice.a, model="CHGNet"), fh, indent=1, default=float)
    if args.vasp:
        write_vasp(s, os.path.join(out, "vasp"))
        print(f"  VASP inputs: {out}/vasp/")
    if args.hull:
        from pymatgen.core import Structure
        extra = {os.path.basename(p): Structure.from_file(p) for p in args.extra}
        energies = None
        if args.energies:
            import csv
            with open(args.energies) as fh:
                energies = {row[0]: float(row[1]) for row in csv.reader(fh) if row and not row[0].startswith("#")}
        res = mxene_hull({f: (s, r["energy"])}, cache=os.path.join(out, "library_chgnet.json"), extra=extra,
                         energies=energies, fmax=args.fmax)
        h = res["hull"]
        print(f"\n  hull {'-'.join(str(e) for e in h.elements)}: stable phases {', '.join(sorted(h.stable_formulas()))}")
        for row in res["rows"]:
            dec = " + ".join(f"{v:.2f} {k}" for k, v in row["decomposition"].items())
            print(f"  {row['label']}: E_form = {row['e_form']:.3f} eV/atom, E_hull = {1000 * row['e_hull']:.0f} meV/atom"
                  f"  → {dec}")
        with open(os.path.join(out, "hull.json"), "w") as fh:
            json.dump(res["rows"], fh, indent=1)
        if len(h.elements) in (2, 3):
            plot_phase_diagram(res, save=os.path.join(out, "phase_diagram.png"), title=f"{f} (2D) in the {'-'.join(str(e) for e in h.elements)} hull")
            print(f"  written: {out}/phase_diagram.png, {out}/hull.json")


if __name__ == "__main__":
    main()
