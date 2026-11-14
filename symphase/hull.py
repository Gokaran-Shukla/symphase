# -*- coding: utf-8 -*-
"""
symphase.hull
=============

Thermodynamic stability (convex hull / energy above hull) for systems with
ANY number of elements, built on pymatgen's ``PhaseDiagram``.

Energy sources (can be mixed, but only if computed with identical settings!)
  * VASP run directories  (composition from CONTCAR/POSCAR, energy from
    OUTCAR 'energy(sigma->0)', fallback OUTCAR TOTEN, fallback OSZICAR E0)
  * CSV tables            (formula, energy [, name, energy_type])
  * Materials Project     (``mp-api``, needs an API key; optional)

Analysis
  * formation energy per atom, energy above hull E_hull (meV/atom),
    decomposition products, stability margin of stable phases
    (equilibrium reaction energy), hull energy at any composition,
    pseudo-binary hull along any composition line A -> B,
    stability of a candidate structure that is not part of the hull.

Plots (matplotlib, publication style)
  * 2 elements : formation-energy hull curve
  * 3 elements : Gibbs-triangle hull (2D) and 3D hull surface
  * 4 elements : tetrahedral hull (3D)
  * any n      : E_hull bar chart, pseudo-binary hull slices, grid of all
                 binary (and ternary) sub-system hulls
  * optional interactive plotly figure from pymatgen's PDPlotter (n <= 4)

Command line:  symphase hull -h
Python:        from symphase.hull import HullAnalysis, load_csv, scan_vasp
"""

import csv
import json
import math
import os
import re
import sys
import warnings

import numpy as np

try:
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
    from pymatgen.core import Structure
except ImportError:  # pragma: no cover
    sys.exit("symphase.hull needs pymatgen:  pip install pymatgen")

from . import __version__
from .parallel import pmap, get_shared, set_shared, resolve_nproc

SQ3 = math.sqrt(3.0)


# ============================================================================
#  data loading
# ============================================================================
def _read_formula_from_poscar(path):
    from .tensors import read_poscar
    p = read_poscar(path)
    if not p["species"]:
        raise ValueError(f"{path}: POSCAR has no species line (VASP4 format)")
    return Composition({s: c for s, c in zip(p["species"], p["counts"])})


def _vasp_energy(directory):
    """last total energy of a VASP run, eV per cell, and its origin"""
    oc = os.path.join(directory, "OUTCAR")
    if os.path.isfile(oc):
        with open(oc, errors="replace") as fh:
            txt = fh.read()
        m = re.findall(r"energy\(sigma->0\)\s*=\s*([-\d.Ee+]+)", txt)
        if m:
            return float(m[-1]), "OUTCAR energy(sigma->0)"
        m = re.findall(r"free\s+energy\s+TOTEN\s*=\s*([-\d.Ee+]+)", txt)
        if m:
            return float(m[-1]), "OUTCAR TOTEN"
    oz = os.path.join(directory, "OSZICAR")
    if os.path.isfile(oz):
        with open(oz, errors="replace") as fh:
            lines = [l for l in fh if "E0=" in l]
        if lines:
            m = re.search(r"E0=\s*([-\d.Ee+]+)", lines[-1])
            if m:
                return float(m.group(1)), "OSZICAR E0"
    raise ValueError(f"{directory}: no energy found in OUTCAR/OSZICAR")


def entry_from_vasp(directory, name=None):
    """ComputedEntry from one VASP run directory"""
    for f in ("CONTCAR", "POSCAR"):
        p = os.path.join(directory, f)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            comp = _read_formula_from_poscar(p)
            break
    else:
        raise ValueError(f"{directory}: no CONTCAR/POSCAR")
    e, src = _vasp_energy(directory)
    name = name or os.path.basename(os.path.normpath(directory))
    data = dict(name=name, source=f"VASP {src}", path=os.path.abspath(directory))
    try:
        st = Structure.from_file(p)
        return ComputedStructureEntry(st, e, entry_id=name, data=data)
    except Exception:  # noqa: BLE001
        return ComputedEntry(comp, e, entry_id=name, data=data)


def _w_scan(args):
    d, name = args
    try:
        return entry_from_vasp(d, name=name)
    except Exception as err:  # noqa: BLE001
        return f"skipped {d}: {err}"


def find_vasp_dirs(*roots):
    """recursively list VASP run directories (POSCAR/CONTCAR + OUTCAR/OSZICAR)"""
    out = []
    for root in roots:
        for d, _dirs, files in os.walk(root):
            if ({"POSCAR", "CONTCAR"} & set(files)) and ({"OUTCAR", "OSZICAR"} & set(files)):
                name = os.path.relpath(d, root) if os.path.abspath(d) != os.path.abspath(root) \
                    else os.path.basename(os.path.abspath(d))
                out.append((d, name))
    return sorted(out)


def scan_vasp(*roots, nproc=None, backend="auto", verbose=True):
    """read every VASP run below `roots` in parallel (OUTCAR parsing is the bottleneck)"""
    res = pmap(_w_scan, find_vasp_dirs(*roots), nproc=nproc, backend=backend, min_items=8)
    entries = []
    for r in res:
        if isinstance(r, str):
            if verbose:
                print("  " + r)
        else:
            entries.append(r)
    return entries


def load_csv(path):
    """CSV with columns  formula, energy  [, name, energy_type, structure]

    energy_type (per row, or a column default):
        total     energy of the formula unit as written   (default)
        per_atom  energy per atom
    structure (optional): path (relative to the CSV) of a POSCAR/CIF of the phase.
    Lines starting with '#' are comments.
    """
    entries = []
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.lstrip().startswith("#"))]
    for i, r in enumerate(rows):
        r = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
        comp = Composition(r["formula"])
        e = float(r.get("energy") or r.get("energy_per_atom"))
        et = (r.get("energy_type") or ("per_atom" if "energy_per_atom" in r and not r.get("energy")
                                       else "total")).lower()
        if et in ("per_atom", "atom", "eV/atom"):
            e = e * comp.num_atoms
        name = r.get("name") or f"{comp.reduced_formula}-{i+1}"
        data = dict(name=name, source=os.path.basename(path), note=r.get("note", ""))
        sp = r.get("structure")
        if sp:
            sp = sp if os.path.isabs(sp) else os.path.join(os.path.dirname(os.path.abspath(path)), sp)
            st = Structure.from_file(sp)
            if st.composition.reduced_composition != comp.reduced_composition:
                raise ValueError(f"{path}: structure {sp} does not match formula {r['formula']}")
            e = e / comp.num_atoms * st.composition.num_atoms
            entries.append(ComputedStructureEntry(st, e, entry_id=name, data=data))
        else:
            entries.append(ComputedEntry(comp, e, entry_id=name, data=data))
    return entries


def load_mp(chemsys, api_key=None, thermo_type="GGA_GGA+U", verbose=True):
    """entries of a chemical system from the Materials Project (needs mp-api + key)"""
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("Materials Project access needs:  pip install mp-api")
    key = api_key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise ValueError("no Materials Project API key (set MP_API_KEY or pass api_key=...)")
    els = chemsys.split("-") if isinstance(chemsys, str) else list(chemsys)
    with MPRester(key) as mpr:
        entries = mpr.get_entries_in_chemsys(els, additional_criteria={"thermo_types": [thermo_type]})
    for e in entries:
        e.data["name"] = f"{e.composition.reduced_formula} ({e.entry_id})"
        e.data["source"] = f"Materials Project {thermo_type}"
    if verbose:
        print(f"  {len(entries)} entries from Materials Project for {'-'.join(els)}")
    return entries


def make_entry(formula, energy, name=None, per_atom=False):
    """convenience: ComputedEntry from formula and energy (eV per formula or per atom)"""
    comp = Composition(formula)
    e = energy * comp.num_atoms if per_atom else energy
    name = name or comp.reduced_formula
    return ComputedEntry(comp, e, entry_id=name, data=dict(name=name, source="user"))


def entry_name(e):
    return e.data.get("name") or str(e.entry_id or e.composition.reduced_formula)


# ============================================================================
#  formatting helpers
# ============================================================================
_SUB = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉.")


def pretty_formula(f, mode="mathtext"):
    """'Ti2NiSn' -> 'Ti$_2$NiSn' (mathtext) or 'Ti₂NiSn' (unicode)"""
    if hasattr(f, "reduced_formula"):
        f = f.reduced_formula
    if mode == "unicode":
        return re.sub(r"(\d+\.?\d*)", lambda m: m.group(1).translate(_SUB), f)
    return re.sub(r"(\d+\.?\d*)", r"$_{\1}$", f)


def _decomp_str(decomp):
    parts = sorted(decomp.items(), key=lambda kv: -kv[1])
    return " + ".join(f"{a:.3f} {e.composition.reduced_formula}" for e, a in parts)


# ============================================================================
#  parallel workers (module level so that they can be sent to processes)
# ============================================================================
def _row_for(pd, e, stable_ids):
    d, eh = pd.get_decomp_and_e_above_hull(e, allow_negative=True)
    stable = id(e) in stable_ids
    margin = None
    if stable and not e.composition.is_element:
        try:
            margin = pd.get_equilibrium_reaction_energy(e)
        except Exception:  # noqa: BLE001
            margin = None
    return dict(
        name=entry_name(e), formula=e.composition.reduced_formula,
        composition=e.composition.formula, natoms=e.composition.num_atoms,
        energy_per_atom=e.energy_per_atom, e_form=pd.get_form_energy_per_atom(e),
        e_hull=max(eh, 0.0), stable=stable,
        decomposition="stable" if stable else _decomp_str(d),
        stability_margin=margin, source=e.data.get("source", ""))


def _w_row(i):
    pd = get_shared("pd")
    return _row_for(pd, get_shared("entries")[i], get_shared("stable_ids"))


def _w_eval(e):
    pd = get_shared("pd")
    d, eh = pd.get_decomp_and_e_above_hull(e, allow_negative=True)
    return dict(name=entry_name(e), formula=e.composition.reduced_formula,
                composition=e.composition.formula, energy_per_atom=e.energy_per_atom,
                e_form=pd.get_form_energy_per_atom(e), e_hull=eh, stable=eh <= 1e-8,
                decomposition=_decomp_str(d), source=e.data.get("source", ""))


def _w_hullE(comp):
    pd = get_shared("pd")
    comp = Composition(comp)
    eh = pd.get_hull_energy_per_atom(comp)
    return eh - sum(comp.get_atomic_fraction(el) * pd.el_refs[el].energy_per_atom
                    for el in comp.elements)


# ============================================================================
#  analysis
# ============================================================================
class HullAnalysis:
    """convex-hull analysis for any number of elements

    >>> h = HullAnalysis(entries)            # list of pymatgen entries
    >>> h.print_table()
    >>> h.plot(save="hull.png")
    """

    def __init__(self, entries, elements=None, nproc=None, backend="auto"):
        """nproc: cores to use (None/-1 = all available); backend: auto|process|mpi|serial"""
        self.nproc = nproc
        self.backend = backend
        entries = list(entries)
        if not entries:
            raise ValueError("no entries")
        if elements is not None:
            els = {Element(x) for x in elements}
            entries = [e for e in entries if set(e.composition.elements) <= els]
        self.entries = entries
        self.elements = sorted({el for e in entries for el in e.composition.elements},
                               key=lambda el: el.X)
        missing = [el for el in self.elements
                   if not any(e.composition.is_element and e.composition.elements[0] == el
                              for e in entries)]
        if missing:
            raise ValueError("missing elemental reference entries for: "
                             + ", ".join(str(m) for m in missing)
                             + "  (every element needs at least one pure-element calculation)")
        self.pd = PhaseDiagram(entries, elements=self.elements)
        self.dim = len(self.elements)
        self._table = None

    # ------------------------------------------------------------------ core
    @property
    def chemsys(self):
        return "-".join(str(e) for e in self.elements)

    def form_energy(self, entry):
        """formation energy per atom (eV/atom)"""
        return self.pd.get_form_energy_per_atom(entry)

    def e_above_hull(self, entry):
        """energy above hull (eV/atom); works for entries not in the hull too"""
        _, eh = self.pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
        return eh

    def hull_formation_energy(self, composition):
        """formation energy per atom of the convex hull at any composition"""
        comp = Composition(composition)
        eh = self.pd.get_hull_energy_per_atom(comp)
        ref = sum(comp.get_atomic_fraction(el) * self.pd.el_refs[el].energy_per_atom
                  for el in comp.elements)
        return eh - ref

    def decomposition(self, composition_or_entry):
        if isinstance(composition_or_entry, (str, Composition)):
            d = self.pd.get_decomposition(Composition(composition_or_entry))
        else:
            d, _ = self.pd.get_decomp_and_e_above_hull(composition_or_entry, allow_negative=True)
        return {e.composition.reduced_formula: a for e, a in d.items()}

    def stable_formulas(self):
        return sorted({e.composition.reduced_formula for e in self.pd.stable_entries},
                      key=lambda f: [Composition(f).get_atomic_fraction(el) for el in self.elements])

    def evaluate(self, entry):
        """stability of one candidate entry against this hull (candidate not added)"""
        set_shared(pd=self.pd)
        return _w_eval(entry)

    def evaluate_many(self, entries, progress=False):
        """high-throughput screening: E_hull of many candidates, in parallel"""
        return pmap(_w_eval, list(entries), nproc=self.nproc, backend=self.backend,
                    shared={"pd": self.pd}, progress=progress)

    def table(self, progress=False):
        """list of dicts, one per entry, sorted by E_hull (computed in parallel)"""
        if self._table is not None:
            return self._table
        stable_ids = {id(e) for e in self.pd.stable_entries}
        rows = pmap(_w_row, range(len(self.entries)), nproc=self.nproc, backend=self.backend,
                    shared={"pd": self.pd, "entries": self.entries, "stable_ids": stable_ids},
                    progress=progress)
        rows.sort(key=lambda r: (r["e_hull"], r["e_form"]))
        self._table = rows
        return rows

    def print_table(self, file=None, max_rows=None):
        out = file or sys.stdout
        rows = self.table()
        w = max(12, max(len(r["name"]) for r in rows) + 1)
        print(f"\nConvex hull: {self.chemsys}  ({self.dim} elements, {len(self.entries)} entries, "
              f"{len(self.pd.stable_entries)} stable)", file=out)
        print("  elemental references (eV/atom): " + ", ".join(
            f"{el}: {self.pd.el_refs[el].energy_per_atom:.4f}" for el in self.elements), file=out)
        hdr = (f"  {'name':<{w}s} {'formula':<14s} {'E (eV/at)':>10s} {'E_form':>9s} "
               f"{'E_hull':>9s}  {'stable':<6s} decomposition / stability margin")
        print(hdr, file=out)
        print("  " + " " * w + f" {'':<14s} {'':>10s} {'(eV/at)':>9s} {'(meV/at)':>9s}", file=out)
        print("  " + "-" * (len(hdr) + 20), file=out)
        for r in rows[:max_rows]:
            tail = r["decomposition"]
            if r["stable"] and r["stability_margin"] is not None:
                tail = f"stable, margin {1000*r['stability_margin']:.1f} meV/atom"
            print(f"  {r['name']:<{w}s} {r['formula']:<14s} {r['energy_per_atom']:10.4f} "
                  f"{r['e_form']:9.4f} {1000*r['e_hull']:9.1f}  {'yes' if r['stable'] else 'no':<6s} {tail}",
                  file=out)
        print("  (margin = energy gained by forming the phase from its stable neighbours; "
              "negative = more stable)", file=out)

    def to_csv(self, path):
        rows = self.table()
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(dict(version=__version__, chemsys=self.chemsys,
                           elements=[str(e) for e in self.elements],
                           stable=self.stable_formulas(), entries=self.table()), fh, indent=1)

    # ---------------------------------------------------------------- pseudo-binary
    def pseudo_binary(self, comp_a, comp_b, n=301, tol=0.02):
        """hull along the composition line A -> B (per atom, atomic fractions)

        returns dict(x, e_hull_line, points=[(x, e_form, e_hull, name)], a, b)
        x = fraction of B (per atom, normalized compositions).
        Entries within `tol` (in atomic-fraction distance) of the line are projected.
        """
        A = Composition(comp_a).fractional_composition
        B = Composition(comp_b).fractional_composition
        xs = np.linspace(0, 1, n)
        comps = [A * (1 - x) + B * x for x in xs]
        ys = np.array(pmap(_w_hullE, comps, nproc=self.nproc, backend=self.backend,
                           shared={"pd": self.pd}, min_items=400))
        els = self.elements
        va = np.array([A.get_atomic_fraction(e) for e in els])
        vb = np.array([B.get_atomic_fraction(e) for e in els])
        d = vb - va
        pts = []
        for e in self.entries:
            v = np.array([e.composition.get_atomic_fraction(el) for el in els])
            x = float(np.dot(v - va, d) / np.dot(d, d))
            if -1e-6 <= x <= 1 + 1e-6 and np.linalg.norm(va + x * d - v) < tol:
                pts.append((x, self.form_energy(e), max(self.e_above_hull(e), 0), entry_name(e)))
        return dict(x=xs, e_hull_line=ys, points=pts, a=Composition(comp_a).reduced_formula,
                    b=Composition(comp_b).reduced_formula)

    def subsystem(self, elements):
        """HullAnalysis restricted to a subset of elements"""
        return HullAnalysis(self.entries, elements=elements, nproc=self.nproc, backend=self.backend)

    # ---------------------------------------------------------------- plotting
    def plot(self, save=None, **kw):
        """automatic plot for the dimensionality of the system"""
        from . import hull_plots as hp
        if self.dim == 2:
            return hp.plot_binary(self, save=save, **kw)
        if self.dim == 3:
            return hp.plot_ternary(self, save=save, **kw)
        if self.dim == 4:
            kw["label_ehull"] = 0.0
            return hp.plot_quaternary(self, save=save, **kw)
        return hp.plot_subsystems(self, order=2, save=save, **kw)


# ============================================================================
#  command line
# ============================================================================
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="symphase hull",
        description="Convex hull / energy above hull for any number of elements (pymatgen).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  symphase hull --scan calcs/                     # all VASP runs below calcs/\n"
               "  symphase hull --csv energies.csv --plot hull.png\n"
               "  symphase hull --mp Ti-Ni-Sn --target ./TiNiSn_new   # MP hull + my candidate\n"
               "  symphase hull --csv TiZrHfNiSn.csv --slice TiNiSn HfNiSn --subsystems 3\n"
               "\nEnergies from different sources are only comparable if computed with the\n"
               "same functional, PAW set, U values, ENCUT ... (MP: use its compatibility scheme).")
    ap.add_argument("--csv", nargs="+", default=[], help="CSV file(s): formula,energy[,name,energy_type]")
    ap.add_argument("--scan", nargs="+", default=[], help="directories scanned recursively for VASP runs")
    ap.add_argument("--runs", nargs="+", default=[], help="individual VASP run directories")
    ap.add_argument("--mp", default=None, help="Materials Project chemical system, e.g. Li-Fe-O")
    ap.add_argument("--mp-key", default=None, help="Materials Project API key [env MP_API_KEY]")
    ap.add_argument("--elements", default=None, help="restrict to these elements, e.g. Ti-Ni-Sn")
    ap.add_argument("--target", nargs="+", default=[],
                    help="VASP dir(s) of candidate structure(s) evaluated against the hull (not added)")
    ap.add_argument("--plot", default="hull.png", help="main plot file [hull.png]; 'none' to skip")
    ap.add_argument("--slice", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="pseudo-binary hull between compositions A and B (repeatable)")
    ap.add_argument("--subsystems", type=int, default=0, choices=(0, 2, 3),
                    help="also plot all binary (2) or ternary (3) sub-system hulls")
    ap.add_argument("--bar", action="store_true", help="also plot E_hull bar chart")
    ap.add_argument("--surface", action="store_true", help="ternary: also plot 3D hull surface")
    ap.add_argument("--interactive", default=None, help="write pymatgen/plotly interactive HTML (n<=4)")
    ap.add_argument("--label-ehull", type=float, default=50.0,
                    help="label unstable phases with E_hull below this (meV/atom) [50]")
    ap.add_argument("--screen", nargs="+", default=[],
                    help="CSV file(s) of candidates screened against the hull in parallel (not added)")
    ap.add_argument("--screen-scan", nargs="+", default=[],
                    help="directories of candidate VASP runs screened against the hull (not added)")
    ap.add_argument("--screen-out", default="screening.csv", help="output of --screen [screening.csv]")
    ap.add_argument("-n", "--nproc", type=int, default=-1,
                    help="number of cores (-1 = all available, honours SLURM_CPUS_PER_TASK) [-1]")
    ap.add_argument("--backend", choices=("auto", "process", "mpi", "serial"), default="auto",
                    help="parallel backend; 'auto' uses MPI when launched with >1 MPI rank [auto]")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import time
    t0 = time.time()
    from .parallel import mpi_comm
    comm = mpi_comm() if args.backend in ("auto", "mpi") else None
    root = comm is None or comm.Get_rank() == 0
    if not root:  # only rank 0 talks
        sys.stdout = open(os.devnull, "w")
    ncores = comm.Get_size() if comm is not None else resolve_nproc(args.nproc)
    print(f"SymPhase {__version__}  —  convex hull analysis  "
          f"[{'MPI, ' + str(ncores) + ' ranks' if comm is not None else str(ncores) + ' core(s)'}]")
    entries = []
    for f in args.csv:
        es = load_csv(f)
        print(f"  {len(es):4d} entries from {f}")
        entries += es
    for r in args.scan:
        es = scan_vasp(r, nproc=args.nproc, backend=args.backend)
        print(f"  {len(es):4d} VASP runs found below {r}")
        entries += es
    for r in args.runs:
        entries.append(entry_from_vasp(r))
    if args.mp:
        entries += load_mp(args.mp, args.mp_key)
    if not entries:
        ap.error("no entries: give --csv, --scan, --runs and/or --mp")
    srcs = {e.data.get("source", "").split()[0] for e in entries}
    if len(srcs) > 1:
        print("  WARNING: mixing energies from " + ", ".join(sorted(srcs))
              + " - only meaningful with identical DFT settings")

    els = args.elements.split("-") if args.elements else None
    h = HullAnalysis(entries, elements=els, nproc=args.nproc, backend=args.backend)
    h.table(progress=root and len(entries) > 2000)
    print(f"  hull + {len(entries)} stabilities in {time.time()-t0:.1f} s")
    h.print_table(max_rows=None if len(entries) <= 400 else 400)
    if len(entries) > 400:
        print(f"  ... ({len(entries)-400} more rows: use --csv-out)")

    cands = []
    for f in args.screen:
        cands += load_csv(f)
    for r in args.screen_scan:
        cands += scan_vasp(r, nproc=args.nproc, backend=args.backend)
    if cands:
        t1 = time.time()
        res = h.evaluate_many(cands, progress=root and len(cands) > 2000)
        res.sort(key=lambda r: r["e_hull"])
        if root:
            with open(args.screen_out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
                w.writeheader()
                w.writerows(res)
        nst = sum(r["stable"] for r in res)
        print(f"\n  screened {len(res)} candidates in {time.time()-t1:.1f} s: {nst} on/below hull, "
              f"{sum(r['e_hull'] < 0.05 for r in res)} within 50 meV/atom -> {args.screen_out}")
        for r in res[:10]:
            print(f"    {r['name']:<24s} {r['formula']:<14s} E_hull = {1000*r['e_hull']:8.1f} meV/atom")

    for t in args.target:
        r = h.evaluate(entry_from_vasp(t))
        print(f"\n  candidate {r['name']} ({r['formula']}): E_form = {r['e_form']:.4f} eV/atom, "
              f"E_hull = {1000*r['e_hull']:.1f} meV/atom -> "
              + ("ON/BELOW the hull (new stable phase!)" if r["stable"] else f"decomposes to {r['decomposition']}"))

    from . import hull_plots as hp
    if not root:
        return h
    h.backend = "serial" if comm is not None else h.backend   # rank 0 alone from here on
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.plot and args.plot.lower() != "none":
            h.plot(save=args.plot, label_ehull=args.label_ehull / 1000)
            print(f"\n  plot written: {args.plot}")
        stem, ext = os.path.splitext(args.plot if args.plot.lower() != "none" else "hull.png")
        if args.surface and h.dim == 3:
            hp.plot_ternary_surface(h, save=f"{stem}_surface{ext}")
            print(f"  plot written: {stem}_surface{ext}")
        if args.bar or h.dim >= 5:
            hp.plot_ehull_bar(h, save=f"{stem}_ehull_bar{ext}")
            print(f"  plot written: {stem}_ehull_bar{ext}")
        for a, b in args.slice:
            fn = f"{stem}_slice_{Composition(a).reduced_formula}_{Composition(b).reduced_formula}{ext}"
            hp.plot_pseudo_binary(h, a, b, save=fn)
            print(f"  plot written: {fn}")
        if args.subsystems:
            fn = f"{stem}_sub{args.subsystems}{ext}"
            hp.plot_subsystems(h, order=args.subsystems, save=fn)
            print(f"  plot written: {fn}")
        if args.interactive:
            if h.dim > 4:
                print("  interactive plot only for <= 4 elements")
            else:
                from pymatgen.analysis.phase_diagram import PDPlotter
                fig = PDPlotter(h.pd, backend="plotly").get_plot()
                fig.write_html(args.interactive)
                print(f"  interactive plot written: {args.interactive}")
    if args.csv_out:
        h.to_csv(args.csv_out)
    if args.json:
        h.to_json(args.json)
    return h


if __name__ == "__main__":
    main()
