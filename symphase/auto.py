# -*- coding: utf-8 -*-
"""
symphase.auto  —  elements in, everything out
=============================================

    symphase run Ti Ni Sn                # or Ti-Ni-Sn, or an input file
    symphase run symphase.in

The minimal input file is ONE line:

    elements = Ti Ni Sn

Pipeline
  1. collect every known phase of the chemical system and all its
     sub-systems (Materials Project if an API key is available, otherwise
     OQMD - no key needed; or your own VASP runs / a CSV table)
  2. convex hull: formation energies, E_hull, decompositions, stability
     margins  (parallel over all cores / MPI)
  3. figures: hull (2D/ternary/3D surface/tetrahedron), all sub-system
     hulls, E_hull bars, pseudo-binary slices
  4. POSCAR of every phase within `ehull_window` of the hull (optionally
     also its symmetry/tensor analysis: tensors = charge,spin,orbital)
  5. optional: ready-to-run VASP inputs for all those phases with an Ibex
     (SLURM) job-array script; re-run `symphase run` with source = vasp
     afterwards to get the hull from your own, consistent DFT energies
  6. SUMMARY.md + SUMMARY.html (self-contained report with all figures)

All optional keys (defaults shown):

    elements      = Ti Ni Sn          # required
    source        = auto              # auto | mp | oqmd | vasp | csv
    mp_api_key    =                   # or environment variable MP_API_KEY
    mp_thermo     = GGA_GGA+U         # GGA_GGA+U | R2SCAN | GGA_GGA+U_R2SCAN
    csv           =                   # source = csv: formula,energy[,name,structure]
    vasp_dir      = vasp              # source = vasp: directory with finished runs
    outdir        = symphase_<chemsys>
    ehull_window  = 0.10              # eV/atom: phases analysed in detail
    max_phases    = 40                # cap on detailed analyses
    temperatures  =                   # e.g. 300 600 900 1200: finite-T hulls (mixing entropy + vib)
    vib           = none              # none | debye | phonopy | bartel  (finite-T vibrations)
    tensors       =                   # e.g. charge,spin,orbital: also symmetry-analyse
                                      # database structures (default off: use `symphase tensors`
                                      # on your own VASP POSCAR/OUTCAR/OSZICAR instead)
    nproc         = -1                # -1 = all cores
    plots         = yes
    plot_format   = png               # png | pdf | svg
    write_vasp    = no                # yes: write VASP inputs + Ibex script
    vasp_mode     = relax             # relax | scf | soc
    potcar_dir    = /home/shuklagn/potpaw_PBE
    vasp_bin      = /home/shuklagn/v_wan3.1/bin/vasp_std
    slurm_partition = largemem
    slurm_ntasks  = 40
    slurm_time    = 48:00:00
    slurm_mem     = 370G
    slurm_concurrent = 10
    oneapi_setvars = /home/shuklagn/intel/oneapi/setvars.sh
"""

import base64
import itertools
import json
import os
import re
import time
import urllib.parse
import urllib.request
import warnings

import numpy as np

from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

from . import __version__
from .hull import HullAnalysis, load_csv, scan_vasp, entry_name
from .parallel import pmap, resolve_nproc

DEFAULTS = dict(
    elements="", source="auto", mp_api_key="", mp_thermo="GGA_GGA+U", csv="",
    vasp_dir="vasp", outdir="", ehull_window=0.10, max_phases=40,
    tensors="", nproc=-1, temperatures="", vib="none", plots=True, plot_format="png",
    write_vasp=False, vasp_mode="relax",
    potcar_dir="/home/shuklagn/potpaw_PBE",
    vasp_bin="/home/shuklagn/v_wan3.1/bin/vasp_std",
    slurm_partition="largemem", slurm_ntasks=40, slurm_time="48:00:00",
    slurm_mem="370G", slurm_concurrent=10,
    oneapi_setvars="/home/shuklagn/intel/oneapi/setvars.sh",
    oqmd_url="http://oqmd.org/oqmdapi/formationenergy",
)


# ============================================================================
#  input
# ============================================================================
def _to_bool(v):
    return str(v).strip().lower() in ("1", "y", "yes", "true", "t", "on")


def read_input(path_or_tokens):
    """input file  OR  list of element tokens  ->  settings dict"""
    cfg = dict(DEFAULTS)
    toks = path_or_tokens if isinstance(path_or_tokens, (list, tuple)) else [path_or_tokens]
    if len(toks) == 1 and os.path.isfile(toks[0]):
        with open(toks[0]) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                if "=" not in line:
                    raise ValueError(f"{toks[0]}: cannot parse line '{line}' (use key = value)")
                k, v = (x.strip() for x in line.split("=", 1))
                k = k.lower().replace("-", "_")
                if k not in DEFAULTS:
                    raise ValueError(f"unknown key '{k}'. Allowed: {', '.join(DEFAULTS)}")
                cfg[k] = v
    else:
        cfg["elements"] = " ".join(toks)
    els = [e for e in re.split(r"[\s,\-]+", str(cfg["elements"])) if e]
    if not els:
        raise ValueError("no elements given (elements = Ti Ni Sn)")
    els = [e[:1].upper() + e[1:].lower() for e in els]
    for e in els:
        Element(e)  # validates
    cfg["elements"] = sorted(set(els), key=lambda e: Element(e).X)
    for k in ("plots", "write_vasp"):
        cfg[k] = _to_bool(cfg[k])
    for k in ("ehull_window",):
        cfg[k] = float(cfg[k])
    for k in ("max_phases", "nproc", "slurm_ntasks", "slurm_concurrent"):
        cfg[k] = int(cfg[k])
    cfg["chemsys"] = "-".join(cfg["elements"])
    if not cfg["outdir"]:
        cfg["outdir"] = f"symphase_{cfg['chemsys']}"
    return cfg


# ============================================================================
#  data sources
# ============================================================================
def _http_json(url, timeout=120, retries=3):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"symphase/{__version__}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(2 * (k + 1))
    raise ConnectionError(f"cannot reach {url.split('?')[0]}: {last}")


def _oqmd_structure(d):
    """pymatgen Structure from OQMD 'unit_cell' + 'sites' ('Ti @ 0 0 0')"""
    try:
        lat = Lattice(np.array(d["unit_cell"], float))
        sp, fc = [], []
        for s in d["sites"]:
            el, xyz = s.split("@")
            sp.append(el.strip())
            fc.append([float(x) for x in xyz.split()[:3]])
        return Structure(lat, sp, fc)
    except Exception:  # noqa: BLE001
        return None


def load_oqmd(elements, url=DEFAULTS["oqmd_url"], verbose=True, limit=200):
    """all OQMD phases of the chemical system and its sub-systems (no API key)"""
    fields = "name,entry_id,delta_e,stability,unit_cell,sites,spacegroup,natoms,prototype,icsd_id"
    entries = []
    for k in range(1, len(elements) + 1):
        for sub in itertools.combinations(elements, k):
            filt = f"element_set={','.join(sub)} AND ntypes={k}"
            q = urllib.parse.urlencode(dict(fields=fields, filter=filt, limit=limit, offset=0,
                                            format="json"))
            nxt = f"{url}?{q}"
            n0 = len(entries)
            while nxt:
                js = _http_json(nxt)
                for d in js.get("data", []):
                    if d.get("delta_e") is None:
                        continue
                    st = _oqmd_structure(d)
                    comp = st.composition if st is not None else Composition(d["name"])
                    e = float(d["delta_e"]) * comp.num_atoms   # formation energy, elements = 0
                    name = f"{comp.reduced_formula}_oqmd-{d.get('entry_id')}"
                    data = dict(name=name, source="OQMD formation energy", oqmd_id=d.get("entry_id"),
                                spacegroup=d.get("spacegroup"), prototype=d.get("prototype"),
                                icsd_id=d.get("icsd_id"), oqmd_stability=d.get("stability"))
                    if st is not None:
                        entries.append(ComputedStructureEntry(st, e, entry_id=name, data=data))
                    else:
                        entries.append(ComputedEntry(comp, e, entry_id=name, data=data))
                nxt = (js.get("links") or {}).get("next")
            if verbose:
                print(f"    OQMD {'-'.join(sub):<14s} {len(entries)-n0:5d} phases")
    return entries


def load_mp_full(elements, api_key, thermo="GGA_GGA+U", verbose=True):
    """Materials Project entries with structures and (when available) site magmoms"""
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys(list(elements),
                                             additional_criteria={"thermo_types": [thermo]})
        ids = sorted({m.group(1) for e in entries
                      for m in [re.search(r"((?:mp|mvc)-\d+)", str(e.entry_id))] if m})
        mags = {}
        try:
            for doc in mpr.materials.magnetism.search(material_ids=ids,
                                                      fields=["material_id", "magmoms"]):
                mags[str(doc.material_id)] = doc.magmoms
        except Exception:  # noqa: BLE001
            pass
    for e in entries:
        m = re.search(r"((?:mp|mvc)-\d+)", str(e.entry_id))
        mid = m.group(1) if m else str(e.entry_id)
        e.data["name"] = f"{e.composition.reduced_formula}_{mid}"
        e.data["source"] = f"Materials Project {thermo}"
        e.data["material_id"] = mid
        st = getattr(e, "structure", None)
        if st is not None and mid in mags and mags[mid] is not None and len(mags[mid]) == len(st):
            st.add_site_property("magmom", list(mags[mid]))
    if verbose:
        print(f"    Materials Project: {len(entries)} entries ({len(mags)} with magnetic moments)")
    return entries


def collect_entries(cfg):
    src = cfg["source"].lower()
    key = cfg["mp_api_key"] or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if src == "auto":
        src = "mp" if key else "oqmd"
        print(f"  source: {'Materials Project (API key found)' if src == 'mp' else 'OQMD (no MP_API_KEY set)'}")
    cfg["source_used"] = src
    if src == "mp":
        if not key:
            raise ValueError("source = mp needs mp_api_key or MP_API_KEY")
        return load_mp_full(cfg["elements"], key, cfg["mp_thermo"])
    if src == "oqmd":
        return load_oqmd(cfg["elements"], url=cfg["oqmd_url"])
    if src == "vasp":
        return scan_vasp(cfg["vasp_dir"], nproc=cfg["nproc"], backend="process")
    if src == "csv":
        es = load_csv(cfg["csv"])
        return es
    raise ValueError(f"unknown source {src}")


def _cache_path(cfg):
    return os.path.join(cfg["outdir"], "data", f"entries_{cfg['source_used']}.json")


def save_entries(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump([e.as_dict() for e in entries], fh)


def load_entries(path):
    from monty.json import MontyDecoder
    with open(path) as fh:
        return [MontyDecoder().process_decoded(d) for d in json.load(fh)]


# ============================================================================
#  per-phase symmetry / tensors (parallel worker)
# ============================================================================
def _w_symmetry(job):
    d, tensors = job
    from .tensors import analyze_tensors
    try:
        res = analyze_tensors(d, tensors=tensors, no_ops=False, out=os.path.join(d, "symmetry_report.txt"))
        m = res["magnetic"]
        t = res.get("tensors", {})
        ch = t.get("charge", {})
        sp = t.get("spin", {})

        def n_unconv(part):
            n = 0
            for k, v in (sp.get(part) or {}).items():
                mm = re.match(r".*\^(\w)_(\w)(\w)$", k)
                if v != "0" and mm and len({mm.group(1), mm.group(2), mm.group(3)}) < 3:
                    n += 1
            return n
        sgr = res.get("spin_group") or {}
        return dict(dir=d, space_group=f"{res['crystal']['space_group']} ({res['crystal']['number']})",
                    magnet_class=sgr.get("magnet_class"), spin_splitting=sgr.get("spin_splitting"),
                    spin_laue=sgr.get("spin_laue_group"),
                    msg=f"{m['bns_symbol']} ({m['bns_number']})", mpg=m["magnetic_point_group"],
                    msg_type=m["type"], inversion=m["inversion"], PT=m["PT"],
                    T=m["time_reversal"] or m["antitranslation"],
                    ahe=ch.get("ahe_allowed"), n_she_even=sp.get("n_even"), n_she_odd=sp.get("n_odd"),
                    n_unconv_even=n_unconv("even"), n_unconv_odd=n_unconv("odd"), error=None)
    except Exception as err:  # noqa: BLE001
        return dict(dir=d, error=f"{type(err).__name__}: {err}")


def _write_phase_dir(d, e):
    os.makedirs(d, exist_ok=True)
    st = e.structure.copy()
    st.to(filename=os.path.join(d, "POSCAR"), fmt="poscar")
    mag = st.site_properties.get("magmom")
    if mag is not None:
        mvals = []
        for m in mag:
            if hasattr(m, "__len__"):
                mvals += [float(x) for x in m]
            else:
                mvals.append(float(m))
        with open(os.path.join(d, "MAGMOM"), "w") as fh:
            fh.write(" ".join(f"{x:.3f}" for x in mvals) + "\n")
    with open(os.path.join(d, "entry.json"), "w") as fh:
        json.dump(dict(name=entry_name(e), formula=e.composition.reduced_formula,
                       data={k: v for k, v in e.data.items() if isinstance(v, (str, int, float, type(None)))}),
                  fh, indent=1)


# ============================================================================
#  VASP inputs for Ibex
# ============================================================================
# PAW choices recommended by the Materials Project for PBE (subset; others = bare symbol)
_MP_PAW = {
    "Li": "Li_sv", "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv", "Be": "Be_sv",
    "Mg": "Mg_pv", "Ca": "Ca_sv", "Sr": "Sr_sv", "Ba": "Ba_sv", "Sc": "Sc_sv", "Y": "Y_sv",
    "Ti": "Ti_pv", "Zr": "Zr_sv", "Hf": "Hf_pv", "V": "V_pv", "Nb": "Nb_pv", "Ta": "Ta_pv",
    "Cr": "Cr_pv", "Mo": "Mo_pv", "W": "W_pv", "Mn": "Mn_pv", "Tc": "Tc_pv", "Re": "Re_pv",
    "Fe": "Fe_pv", "Ru": "Ru_pv", "Os": "Os_pv", "Co": "Co", "Rh": "Rh_pv", "Ir": "Ir",
    "Ni": "Ni_pv", "Pd": "Pd", "Pt": "Pt", "Cu": "Cu_pv", "Ag": "Ag", "Au": "Au",
    "Zn": "Zn", "Cd": "Cd", "Hg": "Hg", "Ga": "Ga_d", "In": "In_d", "Tl": "Tl_d",
    "Ge": "Ge_d", "Sn": "Sn_d", "Pb": "Pb_d", "Bi": "Bi_d", "Sb": "Sb", "As": "As",
    "La": "La", "Ce": "Ce", "Gd": "Gd", "Eu": "Eu", "Th": "Th", "U": "U",
}


def _default_magmom(el):
    e = Element(el)
    return 5.0 if (e.is_transition_metal or e.is_rare_earth) else 0.6


def write_vasp_inputs(cfg, phase_dirs, entries_by_dir):
    root = os.path.join(cfg["outdir"], "vasp")
    os.makedirs(root, exist_ok=True)
    from pymatgen.io.vasp.inputs import Kpoints
    runs = []
    for d in phase_dirs:
        e = entries_by_dir[d]
        rd = os.path.join(root, os.path.basename(d))
        os.makedirs(rd, exist_ok=True)
        st = e.structure.get_sorted_structure(key=lambda site: site.specie.X)
        st.to(filename=os.path.join(rd, "POSCAR"), fmt="poscar")
        # species order exactly as in POSCAR (sites are grouped by species)
        seen = []
        for s in st.species:
            if str(s) not in seen:
                seen.append(str(s))
        species = seen
        mag = st.site_properties.get("magmom")
        if mag is not None and not any(hasattr(m, "__len__") for m in mag):
            mm = [float(m) for m in mag]
        else:
            mm = [_default_magmom(str(s)) for s in st.species]
        mag_str = " ".join(f"{m:.2f}" for m in mm)
        mode = cfg["vasp_mode"]
        incar = [f"SYSTEM = {e.composition.reduced_formula} ({entry_name(e)})  [SymPhase {__version__}]",
                 "PREC = Accurate", "ENCUT = 520", "EDIFF = 1E-6", "ALGO = Fast", "LREAL = Auto",
                 "ISMEAR = 1", "SIGMA = 0.1", "ISPIN = 2", f"MAGMOM = {mag_str}",
                 "LWAVE = .FALSE.", "LCHARG = .TRUE.", "NELM = 200", "LORBIT = 11"]
        if mode == "relax":
            incar += ["IBRION = 2", "ISIF = 3", "NSW = 99", "EDIFFG = -0.01"]
        elif mode == "scf":
            incar += ["NSW = 0", "IBRION = -1"]
        elif mode == "soc":
            incar = [l for l in incar if not l.startswith(("ISPIN", "MAGMOM"))]
            incar += ["NSW = 0", "IBRION = -1", "LSORBIT = .TRUE.", "LNONCOLLINEAR = .TRUE.",
                      "SAXIS = 0 0 1", "ISYM = -1",
                      "MAGMOM = " + " ".join(f"0 0 {m:.2f}" for m in mm)]
        with open(os.path.join(rd, "INCAR"), "w") as fh:
            fh.write("\n".join(incar) + "\n")
        Kpoints.automatic_density(st, 1000).write_file(os.path.join(rd, "KPOINTS"))
        with open(os.path.join(rd, "POTCAR.spec"), "w") as fh:
            fh.write(" ".join(_MP_PAW.get(s, s) for s in species) + "\n")
        runs.append(rd)
    # job list + Ibex script
    with open(os.path.join(root, "joblist.txt"), "w") as fh:
        fh.write("\n".join(os.path.abspath(r) for r in runs) + "\n")
    script = IBEX_TEMPLATE.format(n=len(runs), last=len(runs) - 1, **cfg)
    sp = os.path.join(root, "submit_ibex.sh")
    with open(sp, "w") as fh:
        fh.write(script)
    os.chmod(sp, 0o755)
    return runs, sp


IBEX_TEMPLATE = r"""#!/bin/bash
# =============================================================================
#  SymPhase: VASP job array for all phases of {chemsys}  ({n} runs)
#  Usage (on the Ibex login node, inside this folder):   ./submit_ibex.sh
#  When all runs have finished:   cd ..; symphase run {elements_str} source=vasp
# =============================================================================
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"; mkdir -p logs
POTDIR={potcar_dir}

# --- build POTCARs on the login node (potpaw may not be visible on compute nodes)
while read -r D; do
  [ -f "$D/POTCAR" ] && continue
  : > "$D/POTCAR"
  for el in $(cat "$D/POTCAR.spec"); do
    [ -f "$POTDIR/$el/POTCAR" ] || {{ echo "missing $POTDIR/$el/POTCAR"; exit 1; }}
    cat "$POTDIR/$el/POTCAR" >> "$D/POTCAR"
  done
  grep TITEL "$D/POTCAR" > "$D/potcar_titles.txt" || true
done < joblist.txt

cat > run_array.slurm <<'EOF'
#!/bin/bash
#SBATCH --partition={slurm_partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={slurm_ntasks}
#SBATCH --time={slurm_time}
#SBATCH --mem={slurm_mem}
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
echo "job ${{SLURM_JOB_ID}} task ${{SLURM_ARRAY_TASK_ID}} on ${{SLURM_JOB_NODELIST}}"
set +e; source {oneapi_setvars} > /dev/null 2>&1; set -e
VASP={vasp_bin}
mapfile -t DIRS < "$SLURM_SUBMIT_DIR/joblist.txt"
D=${{DIRS[$SLURM_ARRAY_TASK_ID]}}
[ -z "$D" ] && exit 0
cd "$D"; echo "running in $D"
[ -f OUTCAR ] && grep -q "General timing" OUTCAR && {{ echo "already done"; exit 0; }}
mpirun -n {slurm_ntasks} "$VASP" > OUT 2>&1
grep -q "General timing" OUTCAR && echo "finished: $(grep 'energy(sigma->0)' OUTCAR | tail -1)"
EOF

sbatch --array=0-{last}%{slurm_concurrent} --job-name="SymPhase_{chemsys}" run_array.slurm
"""


# ============================================================================
#  report
# ============================================================================
def _img_b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def write_summary(cfg, h, figs, sym_rows, t_total):
    rows = h.table()
    md = []
    md.append(f"# SymPhase report — {cfg['chemsys']}\n")
    md.append(f"*SymPhase {__version__}, source: {cfg['source_used']}, {len(rows)} entries, "
              f"{len(h.pd.stable_entries)} stable phases, {resolve_nproc(cfg['nproc'])} cores, "
              f"{t_total:.1f} s*\n")
    if cfg["source_used"] == "oqmd":
        md.append("> Energies are OQMD formation energies (PBE / OQMD fitted references). "
                  "Cite: Saal et al., JOM 65, 1501 (2013); Kirklin et al., npj Comput. Mater. 1, 15010 (2015).\n")
    if cfg["source_used"] == "mp":
        md.append("> Energies: Materials Project, MP2020 compatibility. Cite: Jain et al., APL Mater. 1, 011002 (2013).\n")
    md.append("## Stable phases\n")
    md.append("| phase | E_form (eV/atom) | stability margin (meV/atom) |\n|---|---:|---:|")
    for r in rows:
        if r["stable"]:
            mg = "" if r["stability_margin"] is None else f"{1000*r['stability_margin']:.1f}"
            md.append(f"| {r['formula']} | {r['e_form']:.4f} | {mg} |")
    md.append(f"\n## Phases within {1000*cfg['ehull_window']:.0f} meV/atom of the hull\n")
    md.append("| phase | id | E_form (eV/atom) | E_hull (meV/atom) | decomposition |\n|---|---|---:|---:|---|")
    for r in rows:
        if r["e_hull"] <= cfg["ehull_window"] and not r["stable"]:
            md.append(f"| {r['formula']} | {r['name']} | {r['e_form']:.4f} | {1000*r['e_hull']:.1f} | {r['decomposition']} |")
    if sym_rows:
        md.append("\n## Symmetry and allowed responses (phases near the hull)\n")
        md.append("| phase | space group | magnetic space group (BNS) | MPG | P | PT | T | AHE | SHE T-even / T-odd | unconv. spin even / odd | spin group: class / NR splitting |")
        md.append("|---|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|")
        yn = lambda b: "✓" if b else "–"
        for s in sym_rows:
            if s.get("error"):
                md.append(f"| {os.path.basename(s['dir'])} | error: {s['error']} |||||||||")
                continue
            md.append(f"| {os.path.basename(s['dir'])} | {s['space_group']} | {s['msg']} | {s['mpg']} | "
                      f"{yn(s['inversion'])} | {yn(s['PT'])} | {yn(s['T'])} | {yn(s['ahe'])} | "
                      f"{s['n_she_even']} / {s['n_she_odd']} | {s['n_unconv_even']} / {s['n_unconv_odd']} | "
                      f"{s.get('magnet_class') or ''} / {s.get('spin_splitting') or ''} |")
        md.append("\nFull reports: `structures/<phase>/symmetry_report.txt`. "
                  "Moments: from the database when available, otherwise non-magnetic.")
    md.append("\n## Figures\n")
    for title, f in figs:
        md.append(f"### {title}\n\n![{title}]({os.path.relpath(f, cfg['outdir'])})\n")
    md.append("\n## Files\n- `hull_table.csv`, `hull.json` — all entries\n"
              "- `structures/` — POSCAR, MAGMOM, symmetry_report.txt per phase\n"
              + ("- `vasp/` — VASP inputs + `submit_ibex.sh`\n" if cfg["write_vasp"] else "")
              + "- `data/` — cached database entries (re-runs work offline)\n")
    text = "\n".join(md)
    with open(os.path.join(cfg["outdir"], "SUMMARY.md"), "w") as fh:
        fh.write(text)
    # html
    try:
        import markdown  # optional
        body = markdown.markdown(text, extensions=["tables"])
    except Exception:  # noqa: BLE001
        body = _md_to_html(text)
    for title, f in figs:
        rel = os.path.relpath(f, cfg["outdir"])
        if f.endswith(".png"):
            body = body.replace(f'src="{rel}"', f'src="data:image/png;base64,{_img_b64(f)}"')
    html = HTML_TEMPLATE.replace("{TITLE}", f"SymPhase — {cfg['chemsys']}").replace("{BODY}", body)
    with open(os.path.join(cfg["outdir"], "SUMMARY.html"), "w") as fh:
        fh.write(html)


def _md_to_html(text):
    """tiny markdown -> html (headings, tables, images, blockquotes, lists)"""
    out, table = [], []

    def flush():
        if table:
            hdr = [c.strip() for c in table[0].strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in hdr) + "</tr></thead><tbody>")
            for r in table[2:]:
                out.append("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in r.strip("|").split("|")) + "</tr>")
            out.append("</tbody></table>")
            table.clear()
    for line in text.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush()
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        line = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", line)
        m = re.match(r"!\[(.*)\]\((.*)\)", line)
        if m:
            out.append(f'<figure><img src="{m.group(2)}" alt="{m.group(1)}"></figure>')
        elif line.startswith("### "):
            out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<li>{line[2:]}</li>")
        elif line.strip():
            out.append(f"<p>{line}</p>")
    flush()
    return "\n".join(out)


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>{TITLE}</title>
<style>
body{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;max-width:1100px;margin:2em auto;
padding:0 1.5em;color:#1d2433;line-height:1.5}
h1{color:#16325c;border-bottom:3px solid #16325c;padding-bottom:.2em}
h2{color:#16325c;margin-top:2em;border-bottom:1px solid #d0d7e2}
table{border-collapse:collapse;font-size:.9em;margin:1em 0;width:100%}
th{background:#16325c;color:#fff;padding:6px 8px;text-align:left}
td{padding:5px 8px;border-bottom:1px solid #e3e8ef}tr:nth-child(even) td{background:#f6f8fb}
img{max-width:100%;border:1px solid #e3e8ef;border-radius:6px}
blockquote{background:#eef4fb;border-left:4px solid #16325c;margin:1em 0;padding:.6em 1em}
code{background:#eef1f5;padding:1px 4px;border-radius:3px}
</style></head><body>{BODY}</body></html>"""


# ============================================================================
#  driver
# ============================================================================
def run(cfg_or_tokens, **overrides):
    """run the full pipeline; returns (HullAnalysis, summary dict)"""
    cfg = cfg_or_tokens if isinstance(cfg_or_tokens, dict) else read_input(cfg_or_tokens)
    for k, v in overrides.items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown option {k}")
        cfg[k] = v
    cfg["elements_str"] = " ".join(cfg["elements"])
    t0 = time.time()
    out = cfg["outdir"]
    os.makedirs(out, exist_ok=True)
    print(f"SymPhase {__version__} — automatic analysis of {cfg['chemsys']}  "
          f"({resolve_nproc(cfg['nproc'])} cores)  ->  {out}/")

    # ---- 1. entries (cached)
    print("[1/6] collecting phases")
    src = cfg["source"].lower()
    cached = None
    if src in ("auto", "mp", "oqmd"):
        for s in (["mp", "oqmd"] if src == "auto" else [src]):
            p = os.path.join(out, "data", f"entries_{s}.json")
            if os.path.isfile(p):
                cached, cfg["source_used"] = p, s
                break
    if cached:
        entries = load_entries(cached)
        print(f"  {len(entries)} entries from cache {cached}  (delete it to re-download)")
    else:
        entries = collect_entries(cfg)
        if cfg["source_used"] in ("mp", "oqmd"):
            save_entries(entries, _cache_path(cfg))
    entries = [e for e in entries if set(map(str, e.composition.elements)) <= set(cfg["elements"])]
    print(f"  {len(entries)} entries in {cfg['chemsys']} and its sub-systems")

    # ---- 2. hull
    print("[2/6] convex hull")
    h = HullAnalysis(entries, nproc=cfg["nproc"], backend="process")
    rows = h.table()
    h.print_table(max_rows=60)
    h.to_csv(os.path.join(out, "hull_table.csv"))
    h.to_json(os.path.join(out, "hull.json"))

    # ---- 3. figures
    figs = []
    if cfg["plots"]:
        print("[3/6] figures")
        from . import hull_plots as hp
        import matplotlib.pyplot as plt
        fd = os.path.join(out, "figures")
        os.makedirs(fd, exist_ok=True)
        ext = cfg["plot_format"]

        def add(title, fn, func, *a, **kw):
            path = os.path.join(fd, f"{fn}.{ext}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig = func(*a, save=path, **kw)
                plt.close(fig)
                figs.append((title, path))
                print(f"  {path}")
            except Exception as err:  # noqa: BLE001
                print(f"  ({fn} skipped: {err})")
        add("Convex hull", "hull", lambda save=None, **k: h.plot(save=save, **k))
        if h.dim == 3:
            add("3D hull surface", "hull_surface", hp.plot_ternary_surface, h)
        if h.dim >= 3:
            add("All binary sub-system hulls", "subsystems_binary", hp.plot_subsystems, h, order=2)
        if h.dim >= 4:
            add("All ternary sub-system hulls", "subsystems_ternary", hp.plot_subsystems, h, order=3)
        add("Energy above hull", "ehull_bar", hp.plot_ehull_bar, h)
        if h.dim >= 3:
            st = sorted([r for r in rows if r["stable"] and len(Composition(r["composition"]).elements) >= 2],
                        key=lambda r: r["e_form"])[:3]
            for a, b in itertools.combinations([r["formula"] for r in st], 2):
                add(f"Pseudo-binary {a} – {b}", f"slice_{a}_{b}", hp.plot_pseudo_binary, h, a, b)

    # ---- 4. structures + symmetry
    print("[4/6] structures near the hull")
    sel = [r for r in rows if r["e_hull"] <= cfg["ehull_window"]][: cfg["max_phases"]]
    by_name = {entry_name(e): e for e in entries}
    phase_dirs, entries_by_dir = [], {}
    sd = os.path.join(out, "structures")
    for k, r in enumerate(sel):
        e = by_name.get(r["name"])
        if e is None or getattr(e, "structure", None) is None:
            continue
        safe = re.sub(r"[^\w.-]+", "_", r["name"])
        d = os.path.join(sd, f"{k+1:02d}_{safe}")
        _write_phase_dir(d, e)
        phase_dirs.append(d)
        entries_by_dir[d] = e
    print(f"  {len(phase_dirs)} phases written to {sd}/")
    sym_rows = []
    if phase_dirs and cfg["tensors"]:
        print("[5/6] symmetry + response tensors (parallel)")
        jobs = [(d, cfg["tensors"]) for d in phase_dirs]
        sym_rows = pmap(_w_symmetry, jobs, nproc=cfg["nproc"], min_items=4, backend="process")
        for s in sym_rows:
            if s.get("error"):
                print(f"  {os.path.basename(s['dir'])}: {s['error']}")
            else:
                print(f"  {os.path.basename(s['dir']):<34s} {s['space_group']:<18s} {s['msg']:<22s} "
                      f"AHE {'yes' if s['ahe'] else 'no '}  SHE {s['n_she_even']}/{s['n_she_odd']}  "
                      f"{s.get('magnet_class') or ''}")
    if cfg["write_vasp"] and phase_dirs:
        print("[6/6] VASP inputs + Ibex job script")
        runs, sp = write_vasp_inputs(cfg, phase_dirs, entries_by_dir)
        print(f"  {len(runs)} run directories, submit with:  {sp}")
    if str(cfg.get("temperatures", "")).strip():
        from .thermo import FiniteTHull
        temps = sorted({0.0} | {float(x) for x in re.split(r"[\s,]+", str(cfg["temperatures"]).strip()) if x})
        print(f"[+] finite-temperature hulls at {temps} K (vib: {cfg['vib']})")
        try:
            ft = FiniteTHull(entries, vib=cfg["vib"], mixing=True, nproc=cfg["nproc"])
            ft.scan(temps)
            with open(os.path.join(out, "finite_T.txt"), "w") as fh:
                ft.report(file=fh)
            ft.report()
            if cfg["plots"]:
                path = os.path.join(out, "figures", f"ehull_vs_T.{cfg['plot_format']}")
                plt.close(ft.plot_ehull_vs_T(save=path))
                figs.append(("E_hull versus temperature", path))
        except Exception as err:  # noqa: BLE001
            print(f"  finite-T hull skipped: {err}")
    write_summary(cfg, h, figs, sym_rows, time.time() - t0)
    print(f"\nDone in {time.time()-t0:.1f} s.  Open {os.path.join(out, 'SUMMARY.html')}")
    return h, dict(cfg=cfg, figures=figs, symmetry=sym_rows)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="symphase run",
        description="Elements in, everything out: phases, convex hull, figures, symmetry and "
                    "response tensors, VASP/Ibex inputs, HTML report.",
        epilog="examples:\n  symphase run Ti Ni Sn\n  symphase run Fe-O\n  symphase run symphase.in\n"
               "  symphase run Ti Ni Sn write_vasp=yes ehull_window=0.05\n"
               "  symphase run Ti Ni Sn source=vasp vasp_dir=symphase_Ni-Sn-Ti/vasp",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+", help="elements, or an input file, plus optional key=value overrides")
    args = ap.parse_args(argv)
    toks = [t for t in args.items if "=" not in t]
    kv = dict(t.split("=", 1) for t in args.items if "=" in t)
    cfg = read_input(toks)
    for k, v in kv.items():
        k = k.lower().replace("-", "_")
        if k not in DEFAULTS:
            ap.error(f"unknown option {k}")
        cfg[k] = v
    # re-normalise types
    for k in ("plots", "write_vasp"):
        cfg[k] = _to_bool(cfg[k]) if not isinstance(cfg[k], bool) else cfg[k]
    cfg["ehull_window"] = float(cfg["ehull_window"])
    for k in ("max_phases", "nproc", "slurm_ntasks", "slurm_concurrent"):
        cfg[k] = int(cfg[k])
    if "outdir" not in kv and not (len(toks) == 1 and os.path.isfile(toks[0])):
        cfg["outdir"] = f"symphase_{cfg['chemsys']}"
    run(cfg)


if __name__ == "__main__":
    main()
