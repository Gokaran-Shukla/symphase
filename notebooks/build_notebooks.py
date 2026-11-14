"""
Build the SymPhase tutorial + quiz notebooks and the quiz answer banks.

    python build_notebooks.py            # writes *.ipynb and ../symphase/data/quiz_*.json
    jupyter nbconvert --to notebook --execute --inplace 0*.ipynb   # optional: run them

Hands-on quiz answers were computed with SymPhase itself on the shipped examples.
"""
import os
import nbformat as nbf
from symphase.quiz import build_bank

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "symphase", "data")
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def save(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, os.path.join(HERE, name))
    print("wrote", name)


SETUP = code("""# run this first (works from the repository's notebooks/ folder)
import os, sys
sys.path.insert(0, os.path.abspath(".."))      # use the local package if not pip-installed
EX = os.path.abspath("../examples")
%matplotlib inline""")

# =============================================================================
# 01  symmetry & tensors
# =============================================================================
save([
    md("""# SymPhase tutorial 1 — magnetic symmetry and symmetry-allowed conductivity tensors

**Goal:** from a finished VASP calculation (`POSCAR`, `OUTCAR`, `OSZICAR`) obtain

* the crystal space group and the **magnetic space group** (BNS/OG), magnetic point group, MSG type,
* which symmetry elements are present: inversion **P**, time reversal **T**, **PT**, **T·τ**, mirrors, **M·T**, **C<sub>n</sub>·T** …,
* the symmetry-allowed shape of the **charge** σ<sub>ij</sub>, **spin** σ<sup>k</sup><sub>ij</sub> and **orbital** σ<sup>k</sup><sub>ij</sub> conductivity tensors, split into T-even and T-odd parts.

Command line equivalent: `symphase tensors` inside a VASP folder (`POSCAR`, `INCAR`, `OUTCAR`, `OSZICAR`;
`--analysis msg|spin|both` selects relativistic magnetic groups, non-relativistic spin groups or both)."""),
    md(r"""## 1. Theory in five lines

**Neumann's principle:** a response tensor must be invariant under every operation of the (magnetic) point group.
For an operation $g=\{R|\tau\}$ or $g'=\{R|\tau\}\mathcal{T}$ acting on a tensor $\chi$:

$$\chi^{\text{T-even}} = D(R)\,\chi^{\text{T-even}},\qquad \chi^{\text{T-odd}} = \theta_g\,D(R)\,\chi^{\text{T-odd}},\qquad \theta_g=\begin{cases}+1 & g\\ -1 & g'\end{cases}$$

| response | equation | $D(R)$ |
|---|---|---|
| charge | $J_i=\sigma_{ij}E_j$ | $R\otimes R$ (polar rank 2) |
| spin | $J^{S_k}_i=\sigma^k_{ij}E_j$ | $\det(R)\,R\otimes R\otimes R$ (axial rank 3) |
| orbital | $J^{L_k}_i=\sigma^k_{ij}E_j$ | same as spin ( $\mathbf L$ is axial and T-odd) |

For the charge conductivity Onsager's relation $\sigma_{ij}(\mathbf M)=\sigma_{ji}(-\mathbf M)$ is imposed in addition:
T-even part symmetric (Drude), T-odd part antisymmetric (**anomalous Hall**, Hall vector $h_k$ with $\sigma^{odd}_{ij}=\epsilon_{ijk}h_k$).

SymPhase builds the linear constraints for all operations, finds the null space numerically (SVD) and prints each tensor in terms of its independent components."""),
    SETUP,
    md("## 2. Non-magnetic fcc Pt — the textbook spin Hall metal"),
    code("""from symphase import analyze_tensors
res = analyze_tensors(f"{EX}/vasp/fcc_Pt", quiet=False, no_ops=True, tensors="charge,spin")"""),
    md("""**Read the output:** grey group `Fm-3m1'` (type II, T is a symmetry) ⇒ every T-odd tensor vanishes.
The spin Hall tensor has a single independent component, $\\sigma^z_{xy}=-\\sigma^z_{yx}=\\sigma^x_{yz}=\\dots$ — the familiar $\\epsilon_{ijk}$ form."""),
    md("## 3. Ferromagnetic bcc Fe with **m ∥ z** (SOC, non-collinear OUTCAR)"),
    code("""res = analyze_tensors(f"{EX}/vasp/bcc_Fe_m001", tensors="charge,spin")
m = res["magnetic"]
print("magnetic space group:", m["bns_symbol"], m["bns_number"], "| MPG:", m["magnetic_point_group"])
print("inversion:", m["inversion"], "| PT:", m["PT"], "| T:", m["time_reversal"])
print("Hall vector h =", res["tensors"]["charge"]["hall_vector"])
print("T-odd (magnetic) spin-conductivity components:", res["tensors"]["spin"]["n_odd"])"""),
    md("""The full report is in `res["report"]`. Print just the T-odd spin block:"""),
    code("""rep = res["report"]
print(rep[rep.index("T-odd (magnetic SHE"):rep.index("allowed polarisations", rep.index("T-odd (magnetic SHE"))])"""),
    md("""## 4. RuO₂ — an altermagnet: the Néel vector decides everything

Same crystal (`P4_2/mnm`), three Néel-vector orientations. Collinear moments can be rotated with `mdir`."""),
    code("""cases = {"Néel ∥ [001]": dict(directory=f"{EX}/vasp/RuO2_Neel001_noSOC"),
         "Néel ∥ [100]": dict(directory=f"{EX}/vasp/RuO2_Neel100_SOC"),
         "Néel ∥ [110]": dict(directory=f"{EX}/vasp/RuO2_Neel001_noSOC", mdir=(1, 1, 0))}
print(f"{'case':<14s} {'MSG':<14s} {'MPG':<9s} AHE   Hall vector")
for name, kw in cases.items():
    r = analyze_tensors(tensors="charge,spin", **kw)
    c = r["tensors"]["charge"]
    print(f"{name:<14s} {r['magnetic']['bns_symbol']:<14s} {r['magnetic']['magnetic_point_group']:<9s} "
          f"{'yes' if c['ahe_allowed'] else 'no ':<5s} {c['hall_vector']}")"""),
    md("""* **[001]**: `P4_2'/mnm'` — AHE forbidden, but the T-odd spin tensor contains $\\sigma^z_{xy}=+\\sigma^z_{yx}$: the **spin-splitter** (symmetric) form.
* **[100]** and **[110]**: lower symmetry, anomalous (crystal) Hall effect allowed.

Check the spin-splitter form yourself:"""),
    code("""r = analyze_tensors(f"{EX}/vasp/RuO2_Neel001_noSOC", tensors="spin")
odd = r["tensors"]["spin"]["odd"]
print("σ^z_xy =", odd["σ^z_xy"], "   σ^z_yx =", odd["σ^z_yx"])"""),
    md(r"""## 5. Low symmetry: monoclinic MoO₂ ($P2_1/c$)

MoO₂ is a distorted-rutile, non-magnetic metal (structure from AMCSD 0011769, unique axis **b ∥ y**). Only POSCAR + INCAR are needed.
In point group $2/m$ the 2-fold axis $C_{2y}$ flips $x$ and $z$; for an axial rank-3 tensor ($\det R=+1$) a component
$\sigma^k_{ij}$ survives only if it contains an **odd number of $y$ indices**; inversion gives no further constraint."""),
    code("""r = analyze_tensors(f"{EX}/vasp/MoO2_monoclinic", tensors="charge,spin,orbital")
m, t = r["magnetic"], r["tensors"]
print("MSG:", m["bns_symbol"], "| MPG:", m["magnetic_point_group"])
print("charge: T-even", t["charge"]["n_even"], "independent ->", {k: v for k, v in t["charge"]["even"].items() if v != "0"})
print("spin Hall (T-even):", t["spin"]["n_even"], "independent; orbital Hall:", t["orbital"]["n_even"])
rep = r["report"]; i = rep.index("T-even (intrinsic SHE)"); print(rep[i:rep.index("T-odd (magnetic SHE", i)])"""),
    md("""Unconventional channels appear: e.g. flow $x$ with $E\\parallel y$ carries both $S_z$ (conventional) **and** $S_x$.
Without SOC (spin group) every spin component vanishes, while the 13 orbital components survive — the orbital Hall effect needs no SOC:"""),
    code("""sg = r["spin_group"]["tensors"]
print("NR spin Hall components:", sg["spin"]["n_even"], " | SOC-only:", len(sg["spin"]["soc_only_even"]))
print("NR orbital Hall components:", sg["orbital"]["n_even"])"""),
    md("""## 6. Reading the symmetry elements and operations

`res["magnetic"]` contains the lists of mirrors, M·T planes, rotation axes and every operation (rotation matrix in lattice and Cartesian coordinates, translation, time reversal)."""),
    code("""m = analyze_tensors(f"{EX}/vasp/RuO2_Neel100_SOC")["magnetic"]
print("mirror planes      :", *m["mirrors"], sep="\\n   ")
print("mirror planes × T  :", *m["mirrors_T"], sep="\\n   ")
print("rotations × T      :", *m["rotations_T"], sep="\\n   ")
print("number of operations:", m["n_operations"])"""),
    md("""## 7. Command line, JSON and useful options

```bash
cd my_vasp_run/
symphase tensors                          # POSCAR/OUTCAR/OSZICAR of this folder
symphase tensors -p CONTCAR --mmin 0.1    # final structure, ignore |m| < 0.1 μB
symphase tensors --mdir 1 1 0             # rotate collinear moments to [110]
symphase tensors --magmom "0 0 3  0 0 -3  4*0"   # override moments (3N Cartesian)
symphase tensors --frame std              # tensors in the standardized frame
symphase tensors --json res.json --out report.txt
```

**Caveats**
* The magnetic group couples spin and lattice (relativistic). Without SOC the true symmetry is a larger *spin group*.
* Small induced moments can lower the symmetry: tune `mmin` / `mag_symprec`.
* Tensors are given in the POSCAR Cartesian frame unless `frame="std"`."""),
    md(r"""## 8. Spin groups — non-relativistic altermagnetism (no SOC)

Without spin–orbit coupling spin and lattice transform independently: elements are pairs $[U\,||\,g]$
($U\in O(3)$ acting on spins, improper $U$ = rotation × $\mathcal T$). SymPhase finds, for every crystal operation $g$,
the spin operation $U$ with $\mathbf m_{g(i)} = U\mathbf m_i$, adds the spin-only group, and reports

* the **magnet class**: ferromagnet, conventional AFM ($[C_2\,||\,\tau]$), PT-AFM ($[C_2\,||\,P]$) or **altermagnet**,
* the **spin Laue group** (e.g. $^2 4/^1 m\,^2 m\,^1 m$) and the **spin-splitting form** $\mathbf s(\mathbf k)$: d-, g-, i-wave,
* non-relativistic charge/spin/orbital tensors and the **SOC-only** components (allowed only with spin–orbit coupling).

Moments may come from the **INCAR** alone (`MAGMOM`, `SAXIS`, `LSORBIT`) — useful *before* running VASP."""),
    code("""cases = ["RuO2_Neel001_noSOC", "MnTe_altermagnet_INCAR", "Cr_bcc_AFM_INCAR", "Mn2Au_PT_AFM_INCAR",
         "bcc_Fe_m001", "fcc_Pt"]
print(f"{'example':<24s} {'magnet class':<32s} {'spin Laue':<12s} NR spin splitting")
for c in cases:
    sg = analyze_tensors(f"{EX}/vasp/{c}", tensors="charge,spin")["spin_group"]
    print(f"{c:<24s} {sg['magnet_class']:<32s} {sg['spin_laue_group'] or '-':<12s} {sg['spin_splitting']}"
          + (f"   {sg['s_k'][0]}" if sg['s_k'] else ""))"""),
    md("""The full spin-group report (operations $[U||g]$, nodal planes, NR tensors) for MnTe, read from POSCAR + INCAR only:"""),
    code("""r = analyze_tensors(f"{EX}/vasp/MnTe_altermagnet_INCAR", analysis="spin", tensors="spin", no_ops=True)
rep = r["report"]; print(rep[rep.index("SPIN GROUP"):])"""),
    md("""**Which spin-current components need SOC?** Compare relativistic (magnetic group) and non-relativistic (spin group) results for RuO₂:"""),
    code("""t = analyze_tensors(f"{EX}/vasp/RuO2_Neel001_noSOC", tensors="spin")["spin_group"]["tensors"]["spin"]
print("NR T-odd (spin splitter):", {k: v for k, v in t["odd"].items() if v != "0"})
print("SOC-only T-even (spin Hall):", t["soc_only_even"])"""),
    md("""## 9. Exercise

Rotate the bcc-Fe moment to [111] with `magmom="0.577 0.577 0.577  0.577 0.577 0.577"` (scaled by 2.2 if you like) and determine
(i) the magnetic point group, (ii) the direction of the Hall vector, (iii) the number of independent T-odd spin-conductivity components.
Then continue with **04_quiz_symmetry_tensors.ipynb**."""),
    code("""r = analyze_tensors(f"{EX}/vasp/bcc_Fe_m001", magmom="1.27 1.27 1.27 1.27 1.27 1.27", tensors="charge,spin")
print(r["magnetic"]["bns_symbol"], r["magnetic"]["magnetic_point_group"], r["tensors"]["charge"]["hall_vector"],
      r["tensors"]["spin"]["n_odd"])"""),
], "01_tutorial_symmetry_tensors.ipynb")

# =============================================================================
# 02  convex hull
# =============================================================================
save([
    md("""# SymPhase tutorial 2 — convex hull and energy above hull for any number of elements

> ⚠️ The example energies in `examples/hull/*.csv` are **toy numbers** made for teaching (see `make_toy_data.py`).
> For real systems use `symphase run <elements>` (tutorial 3) or your own VASP runs."""),
    md(r"""## 1. Theory

* **Formation energy** per atom: $E_f(\mathrm{A}_x\mathrm{B}_y\dots)=\dfrac{E_{tot}-\sum_i n_i\,\mu_i^{0}}{\sum_i n_i}$ with $\mu_i^0$ the energy per atom of the most stable elemental phase.
* The **convex hull** is the lower convex envelope of all $(\text{composition},E_f)$ points; phases on it are thermodynamically stable at 0 K.
* **Energy above hull** $E_{hull}\ge 0$: energy per atom released when the phase decomposes into the hull phases at the same composition (the decomposition is given by the lever rule).
* **Stability margin** (equilibrium reaction energy) of a stable phase: how far below the hull of all *other* phases it lies (negative = robustly stable).

For $n$ elements the hull lives in an $n$-dimensional space (composition simplex × energy). SymPhase uses pymatgen's `PhaseDiagram` (Qhull) and parallelises everything that is done *per entry*."""),
    SETUP,
    md("## 2. Binary Ni–Al"),
    code("""from symphase.hull import HullAnalysis, load_csv
h = HullAnalysis(load_csv(f"{EX}/hull/NiAl_binary.csv"))
h.print_table()"""),
    code("fig = h.plot()"),
    md("""Each row gives $E_f$, $E_{hull}$ and the decomposition. Ask for anything directly:"""),
    code("""print("stable:", h.stable_formulas())
print("hull formation energy at Ni2Al:", round(h.hull_formation_energy("Ni2Al"), 4), "eV/atom")
print("Ni2Al decomposes into:", h.decomposition("Ni2Al"))"""),
    md("## 3. Ternary Ti–Ni–Sn (half-Heusler TiNiSn, full-Heusler TiNi₂Sn)"),
    code("""h3 = HullAnalysis(load_csv(f"{EX}/hull/TiNiSn_ternary.csv"))
fig = h3.plot()"""),
    code("""from symphase import hull_plots as hp
fig = hp.plot_ternary_surface(h3)"""),
    md("""Shaded triangles are the hull facets (three-phase regions), lines are tie-lines (two-phase regions).
The 3D view shows the hull as a surface $E_f(x,y)$; metastable phases float above it.

### Pseudo-binary slice
Cut the hull along the line TiNiSn → TiNi₂Sn (the half-/full-Heusler line):"""),
    code("fig = hp.plot_pseudo_binary(h3, 'TiNiSn', 'TiNi2Sn')"),
    md("## 4. Quaternary Ti–Zr–Ni–Sn"),
    code("""h4 = HullAnalysis(load_csv(f"{EX}/hull/TiZrNiSn_quaternary.csv"))
fig = h4.plot()"""),
    code("fig = hp.plot_subsystems(h4, order=3)"),
    md("""## 5. Five components Ti–Zr–Hf–Ni–Sn (high-entropy half-Heusler)

Beyond four elements a single picture is impossible; SymPhase gives (i) the grid of all binary sub-hulls,
(ii) $E_{hull}$ bars, (iii) pseudo-binary slices between any compositions."""),
    code("""h5 = HullAnalysis(load_csv(f"{EX}/hull/TiZrHfNiSn_5component.csv"))
print(h5.chemsys, "| entries:", len(h5.entries), "| stable:", len(h5.pd.stable_entries))
fig = h5.plot()   # grid of binary sub-system hulls"""),
    code("fig = hp.plot_ehull_bar(h5, max_rows=25)"),
    code("fig = hp.plot_pseudo_binary(h5, 'TiNiSn', 'HfNiSn')"),
    md("""## 6. Screening new candidates (not added to the hull)

`evaluate` / `evaluate_many` compute $E_{hull}$ of hypothetical compounds against an existing hull — in parallel."""),
    code("""from symphase.hull import make_entry
from pymatgen.core import Composition
# hypothetical energies: hull energy at that composition + an offset (meV/atom)
def toy(formula, offset_meV, name):
    e_hull = h5.pd.get_hull_energy_per_atom(Composition(formula))
    return make_entry(formula, e_hull + offset_meV / 1000, per_atom=True, name=name)
cands = [toy("TiZrHfNi3Sn3", 1.5, "HE-HH new SQS"),
         toy("Ti2ZrNi3Sn3", 8.0, "Ti-rich HE-HH"),
         toy("TiZrNi2Sn3", -12.0, "new Sn-rich phase?")]
for r in h5.evaluate_many(cands):
    print(f"{r['name']:<12s} E_hull = {1000*r['e_hull']:6.1f} meV/atom -> {r['decomposition']}")"""),
    md("""## 7. Parallel performance

Every per-entry quantity (E_hull, decomposition, stability margin), candidate screening, VASP-folder parsing and
pseudo-binary hull points run on all cores (`nproc=-1`, honours `SLURM_CPUS_PER_TASK`) or across nodes with MPI.
A synthetic 6-element system with 3000 random entries:"""),
    code("""import time, numpy as np
from pymatgen.core import Composition
rng = np.random.default_rng(0); els = ["Ti", "Zr", "Hf", "Ni", "Sn", "Co"]
E = [make_entry(e, 0.0) for e in els]
for k in range(3000):
    n = rng.integers(2, 5); sub = rng.choice(els, n, replace=False); c = rng.integers(1, 5, n)
    f = "".join(f"{a}{b}" for a, b in zip(sub, c))
    E.append(make_entry(f, Composition(f).num_atoms * rng.normal(-0.35, 0.2), name=f"r{k}"))
for n in (1, -1):
    hb = HullAnalysis(E, nproc=n); t = time.time(); hb.table()
    print(f"nproc={n:>2}: {time.time()-t:.2f} s for {len(E)} entries ({len(hb.pd.stable_entries)} stable)")"""),
    md("""(The sandbox that built this notebook has only 2 cores; on a 40-core Ibex node the speed-up for large systems is much larger.)

## 8. Command line

```bash
symphase hull --csv energies.csv --plot hull.png --bar
symphase hull --scan calcs/ -n 40                     # every VASP run below calcs/
symphase hull --csv refs.csv --screen candidates.csv  # high-throughput screening
symphase hull --csv data.csv --slice TiNiSn HfNiSn --subsystems 3
srun -n 160 python -m symphase hull --scan calcs/ --screen big.csv   # MPI, 4 nodes
```
CSV format: `formula,energy[,energy_type,name,structure]` (energy per formula as written, or `energy_type=per_atom`).

Your own VASP runs:"""),
    code("""from symphase.hull import scan_vasp
hv = HullAnalysis(scan_vasp(f"{EX}/hull/vasp_runs_NiAl"))
print(len(hv.entries), "runs,", len(hv.pd.stable_entries), "stable:", hv.stable_formulas())"""),
    md("Continue with **03_tutorial_automatic_workflow.ipynb** and **05_quiz_convex_hull.ipynb**."),
], "02_tutorial_convex_hull.ipynb")

# =============================================================================
# 03  automatic workflow
# =============================================================================
save([
    md("""# SymPhase tutorial 3 — elements in, everything out

```bash
symphase run Ti Ni Sn
```
or an input file `symphase.in` containing just

```
elements = Ti Ni Sn
```

What happens:
1. every known phase of Ti–Ni–Sn **and all sub-systems** is downloaded — Materials Project if `MP_API_KEY` is set, otherwise **OQMD (no key needed)**; cached in `data/`,
2. convex hull, E_hull, decompositions, stability margins (all cores),
3. all figures (hull, 3D surface, sub-system grids, E_hull bars, pseudo-binaries),
4. POSCARs of every phase within `ehull_window` (default 100 meV/atom),
5. optional VASP inputs + **Ibex job-array script** (`write_vasp=yes`),
6. `SUMMARY.html` / `SUMMARY.md` report.

All options: `symphase run -h` or the docstring of `symphase.auto`."""),
    SETUP,
    md("""## 1. Live run (needs internet: OQMD or Materials Project)

On your laptop or the Ibex login node this downloads the real data. The cell falls back gracefully when offline."""),
    code("""from symphase.auto import run
try:
    h, info = run(["Ti", "Ni", "Sn"], outdir="live_Ti-Ni-Sn", ehull_window=0.05)
except ConnectionError as err:
    print("offline here ->", err)"""),
    md("""## 2. Offline demonstration with the toy data set

`source=csv` takes a CSV (here with structures for the key phases)."""),
    code("""h, info = run(["Ti", "Ni", "Sn"], source="csv", csv=f"{EX}/hull/TiNiSn_ternary.csv",
              outdir="demo_Ti-Ni-Sn", write_vasp=True, tensors="charge,spin")"""),
    code("""from IPython.display import Image, Markdown, display
display(Image("demo_Ti-Ni-Sn/figures/hull.png", width=600))
display(Markdown(open("demo_Ti-Ni-Sn/SUMMARY.md").read().split("## Figures")[0]))"""),
    md("""## 3. From database phases to your own DFT on Ibex

`write_vasp=yes` creates `vasp/<phase>/{POSCAR,INCAR,KPOINTS,POTCAR.spec}`, `joblist.txt` and `submit_ibex.sh`
(job array, `largemem`, 40 cores, oneAPI, your VASP binary and POTCAR library — all configurable in the input file):"""),
    code("""print(open("demo_Ti-Ni-Sn/vasp/submit_ibex.sh").read()[:1400])"""),
    md("""On Ibex:

```bash
cd demo_Ti-Ni-Sn/vasp && ./submit_ibex.sh      # builds POTCARs, submits the array
# ... when all jobs are done:
symphase run Ti Ni Sn source=vasp vasp_dir=demo_Ti-Ni-Sn/vasp outdir=TiNiSn_myDFT
```
The second run builds the hull purely from **your** energies (identical settings for every phase).

For the magnetic space group and conductivity tensors of any finished run: `cd vasp/01_TiNiSn...; symphase tensors`.

## 4. Running SymPhase itself as an Ibex job

`jobs/ibex_symphase.slurm` in the repository runs `symphase run` / `symphase hull` on a full node (all 40 cores),
`jobs/ibex_symphase_mpi.slurm` spreads very large screenings over several nodes with MPI."""),
], "03_tutorial_automatic_workflow.ipynb")

# =============================================================================
# 06  spin space groups + magnons
# =============================================================================
save([
    md("""# SymPhase tutorial 4 — spin-space-group numbers and magnon chirality splitting

* **Spin space groups (SSG).** Collinear SSGs are in one-to-one correspondence with the black-and-white magnetic
  groups of type I, III and IV (replace every $[C_{2\\perp}\\|g]$ by $g\\mathcal T$): 230 + 674 + 517 = **1421** groups.
  SymPhase labels a collinear SSG by the BNS number of that group, $\\otimes\\,^{\\infty}m1$ (spin-only group).
* **Magnon chirality.** In a collinear magnet every magnon carries $\\Delta S_z=\\pm1$ (α / β). Without SOC the
  splitting $\\Delta\\omega(\\mathbf q)=\\omega_\\beta-\\omega_\\alpha$ has the **same symmetry as the electronic spin splitting**:
  zero in PT- or $\\tau$-connected antiferromagnets, d/g/i-wave in altermagnets."""),
    SETUP,
    md("## 1. Spin-space-group labels of the examples"),
    code("""from symphase import analyze_tensors
for c in ["RuO2_Neel001_noSOC", "MnTe_altermagnet_INCAR", "Cr_bcc_AFM_INCAR", "Mn2Au_PT_AFM_INCAR",
          "bcc_Fe_m001", "fcc_Pt", "Mn3Sn_allin"]:
    sg = analyze_tensors(f"{EX}/vasp/{c}", tensors="charge")["spin_group"]
    print(f"{c:<24s} {sg['ssg']['label']}")"""),
    md("""The black-and-white symbol and the spin Laue group are consistent: RuO₂ $P4_2'/mn'm$ ↔ $^24/^1m^2m^1m$ (primed = $^2$),
MnTe $P6_3'/m'mc'$ ↔ $^26/^2m^1m^2m$.

## 2. Symmetry classes of exchange bonds in RuO₂

Bonds related by any operation of the non-magnetic crystal share one $J$. In the altermagnet the two 6.35 Å bonds
along [110] and [1$\\bar1$0] fall into **different** classes — the microscopic origin of chirality splitting."""),
    code("""import spglib, numpy as np
from symphase.tensors import read_poscar
from symphase.magnons import analyse, plot_bands, plot_split_map
p = read_poscar(f"{EX}/vasp/RuO2_Neel001_noSOC/POSCAR")
types = [0, 0, 1, 1, 1, 1]; symbols = ["Ru", "Ru", "O", "O", "O", "O"]
mags = np.array([[0, 0, 1.2], [0, 0, -1.2]] + [[0, 0, 0]] * 4)
ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], types))
res = analyse(p["lattice"], p["frac"], types, symbols, mags, ds.rotations, ds.translations, rmax=6.5)
for c in res["classes"]:
    print(f"J{c['cls']+1}: |d| = {c['L']:.3f} Å, {c['n_bonds']} bonds/cell, {c['spins']:<9s} e.g. {c['members'][:2]}")"""),
    md("""## 3. Linear spin-wave theory

Illustrative exchange constants (meV, $J>0$ antiferromagnetic) with $J_7\\neq J_8$ and a small easy-axis anisotropy:"""),
    code("""J = [-0.5, 3.0, -0.2, 0.0, 0.2, 0.0, -0.6, -0.2]
res = analyse(p["lattice"], p["frac"], types, symbols, mags, ds.rotations, ds.translations,
              J=J, K=0.05, rmax=6.5, n_per=50, nmap=41)
fig = plot_bands(res, title="RuO$_2$ (toy J): magnon chirality")"""),
    code("fig = plot_split_map(res)"),
    md("""Δω is odd under $q_x\\to-q_x$ and $q_y\\to-q_y$ and vanishes on both axes: **d-wave, $\\Delta\\omega\\propto q_xq_y$**,
exactly the spin-group form $s_z\\propto k_xk_y$. Setting $J_7=J_8$ removes the splitting:"""),
    code("""res0 = analyse(p["lattice"], p["frac"], types, symbols, mags, ds.rotations, ds.translations,
               J=[-0.5, 3.0, -0.2, 0.0, 0.2, 0.0, -0.4, -0.4], K=0.05, rmax=6.5, n_per=20, nmap=11)
print("max |Δω| with J7 = J8:", np.nanmax(np.abs(res0["map_split"])), "meV")"""),
    md("""## 4. Command line

```bash
cd my_vasp_run/
symphase magnons --list-bonds --rmax 6.5          # bond classes
symphase magnons --J -0.5 3.0 -0.2 0 0.2 0 -0.6 -0.2 --K 0.05 --plot magnons.png
symphase magnons --exchange J.dat                 # i j n1 n2 n3 J(meV), e.g. from magex / TB2J
```
Moments come from OUTCAR or INCAR (`MAGMOM`); $S=|m|/2$ unless `--spin`. Validated against the analytic ferromagnet
and antiferromagnetic-chain dispersions (tests)."""),
], "06_tutorial_spin_space_groups_magnons.ipynb")

# =============================================================================
# 07  finite-temperature hulls
# =============================================================================
save([
    md(r"""# SymPhase tutorial 5 — finite-temperature convex hulls

$$G_i(T)=E_i+F^{\rm vib}_i(T)-T\,S^{\rm conf}_i ,\qquad
S^{\rm conf}=-k_B\frac{\sum_s m_s\sum_a x_{s,a}\ln x_{s,a}}{\sum_s m_s}$$

* **mixing entropy** from a sublattice spec per entry (CSV column `mixing`, e.g. `Ti0.5Zr0.5|Ni|Sn`) or `s_config` (k$_B$/atom),
* **vibrations**: `debye` (column `debye_T`), `phonopy` (`total_dos.dat` or `thermal_properties.yaml`, column `phonons`),
  `bartel` (SISSO descriptor via pymatgen, 300–2000 K), or `none`.

> Toy energies and Debye temperatures (see `examples/hull/make_toy_data.py`)."""),
    SETUP,
    code("""from symphase.hull import load_csv
from symphase.thermo import FiniteTHull, parse_mixing
import math
print("S_conf of (Ti0.5Zr0.5)NiSn:", parse_mixing("Ti0.5Zr0.5|Ni|Sn")[0], "k_B/atom  = ln2/3 =", math.log(2) / 3)
ents = load_csv(f"{EX}/hull/TiZrNiSn_quaternary.csv")
ft = FiniteTHull(ents, vib="none")              # mixing entropy only
ft.scan([0, 200, 300, 400, 600, 900, 1200])
ft.report()"""),
    md(r"""The 5 meV/atom mixing enthalpy of (Ti$_{0.5}$Zr$_{0.5}$)NiSn is overcome when $k_BT\ln2/3=5$ meV, i.e.
$T^*=251$ K — SymPhase finds it by bisection. Adding Debye phonons:"""),
    code("""ftd = FiniteTHull(ents, vib="debye")
ftd.scan([0, 200, 300, 400, 600, 900, 1200]); ftd.report()
fig = ftd.plot_ehull_vs_T()"""),
    code("fig = ftd.plot_hull_at(900)"),
    md("""## Command line

```bash
symphase hull --csv data.csv --temperatures 300 600 900 1200 --vib debye
symphase hull --csv data.csv --temperatures 300 900 --vib phonopy      # 'phonons' column -> phonopy files
symphase hull --scan calcs/ --temperatures 600 1200 --vib bartel        # needs structures
symphase run Ti Zr Ni Sn temperatures="300 600 900 1200" vib=bartel     # automatic pipeline
```"""),
], "07_tutorial_finite_temperature_hull.ipynb")

# =============================================================================
# 08  non-collinear magnets
# =============================================================================
save([
    md(r"""# SymPhase tutorial 6 — non-collinear ferro- and antiferromagnets

**Symmetry.** For non-collinear orders the spin part $U$ of every spin-group element $[U\|g]$ is found by an orthogonal
Procrustes fit ($\mathbf m_{g(i)}=U\mathbf m_i$); the spin-only group is $\{E,\,C_2(\hat{\mathbf n})\mathcal T\}$ for coplanar and
trivial for non-coplanar orders. SymPhase classifies compensated orders as non-collinear antiferromagnets and uncompensated
ones (canted, weak ferromagnets, ferrimagnets) as non-collinear ferro/ferrimagnets, and finds the spin-splitting form $\mathbf s(\mathbf k)$
(odd p-/f-wave forms are possible).

**Magnons.** General linear spin-wave theory in local frames (S. Toth and B. Lake, J. Phys.: Condens. Matter **27**, 166002 (2015)):
each spin direction $\mathbf v_i$ gets $\mathbf u_i=\hat{\mathbf x}_i+i\hat{\mathbf y}_i$, and for $H=\sum\mathbf S_i^{T}J_{ij}\mathbf S_j+\sum\mathbf S_i^{T}A_i\mathbf S_i$
(Heisenberg, DMI $\mathbf D\cdot(\mathbf S_i\times\mathbf S_j)$ or full tensors)
$$A_{ij}(\mathbf k)\ni\tfrac{\sqrt{S_iS_j}}{2}\,\mathbf u_i^{T}J_{ij}\bar{\mathbf u}_j\,e^{i\mathbf k\cdot\mathbf d},\quad
B_{ij}(\mathbf k)\ni\tfrac{\sqrt{S_iS_j}}{2}\,\mathbf u_i^{T}J_{ij}\mathbf u_j\,e^{i\mathbf k\cdot\mathbf d},\quad
A_{ii}\ni-S_j\,\mathbf v_i^{T}J_{ij}\mathbf v_j .$$
Every mode carries a magnon spin vector $\Delta\mathbf S_n=-\sum_i\mathbf v_i\,(|T_{in}|^2+|T_{i+N,n}|^2)$; `--relax` minimises the classical energy first."""),
    SETUP,
    md("## 1. Symmetry of the non-collinear examples"),
    code("""from symphase import analyze_tensors
for c in ["Triangular_120AFM_INCAR", "CantedChain_DMI_INCAR", "Mn3Sn_allin"]:
    sg = analyze_tensors(f"{EX}/vasp/{c}", tensors="charge,spin")["spin_group"]
    print(f"{c:<24s} {sg['magnet_class']:<55s} {sg['spin_splitting']:<12s} {(sg['s_k'] or [''])[0]}")"""),
    md("""## 2. Sanity check: a ferromagnet along [111] is identical to one along z

The local frames make the spectrum independent of the global spin direction; the magnon spin points along −[111]."""),
    code("""import numpy as np, spglib, math
from symphase.magnons import analyse, plot_bands, read_exchange_file
lat, fr = np.eye(3) * 3.0, np.zeros((1, 3))
ds = spglib.get_symmetry_dataset((lat, fr, [0]))
rz = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), ds.rotations, ds.translations, J=[-1.0], rmax=3.1, n_per=20, nmap=3)
rd = analyse(lat, fr, [0], ["Fe"], np.array([[1, 1, 1.]]) * 2 / np.sqrt(3), ds.rotations, ds.translations,
             J=[-1.0], rmax=3.1, n_per=20, nmap=3)
print("max |ω_[111] − ω_z| =", np.abs(rz["omega"] - rd["omega"]).max(), " magnon spin:", np.round(rd["magnon_spin"][10, 0], 3))"""),
    md(r"""## 3. Triangular-lattice 120° antiferromagnet (coplanar)

Three sublattices in a $\sqrt3\times\sqrt3$ cell (POSCAR + INCAR only). Compare with the analytic result
$\omega(\mathbf k)=3JS\sqrt{(1-\gamma_{\mathbf k})(1+2\gamma_{\mathbf k})}$ folded with $\mathbf k\pm\mathbf K$:"""),
    code("""from symphase.tensors import read_poscar, read_incar
import matplotlib.pyplot as plt
d = f"{EX}/vasp/Triangular_120AFM_INCAR"
p = read_poscar(f"{d}/POSCAR"); mags = np.array(read_incar(f"{d}/INCAR")["magmom"]).reshape(-1, 3)
ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0, 0]))
r = analyse(p["lattice"], p["frac"], [0, 0, 0], ["Cr"] * 3, mags, ds.rotations, ds.translations,
            J=[1.0], spin=1.0, rmax=3.1, n_per=40, nmap=3)
a = 3.0; a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * np.sqrt(3) / 2, 0]); K = np.array([4 * np.pi / (3 * a), 0, 0])
w = lambda k: 3 * np.sqrt(max((1 - (g := (np.cos(k @ a1) + np.cos(k @ a2) + np.cos(k @ (a2 - a1))) / 3)) * (1 + 2 * g), 0))
ana = np.array([np.sort([w(q), w(q + K), w(q - K)]) for q in r["q"]])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(r["x"], r["omega"], color="#16325c", lw=2.2)
ax.plot(r["x"][::4], ana[::4], "o", color="#e67e22", ms=4, mfc="none")
ax.set_xticks([t for t, _ in r["ticks"]]); ax.set_xticklabels([l for _, l in r["ticks"]]); ax.set_ylabel("ω (meV)")
ax.set_title("120° triangular AFM: LSWT (lines) vs analytic (circles)")
print("max error:", np.abs(np.sort(r["omega"], axis=1) - ana).max(), "meV;  Goldstone modes at Γ:", r["n_goldstone"])"""),
    md(r"""## 4. Canted antiferromagnet (weak ferromagnet) from staggered DMI

Exchange file `J_DM.dat` (one line per bond: `i j n1 n2 n3 J Dx Dy Dz`). Starting from a collinear guess, `relax=True`
finds the canted ground state, tilted by $\varphi=\arctan(D/J)$ from antiparallel:"""),
    code("""d = f"{EX}/vasp/CantedChain_DMI_INCAR"
print(open(f"{d}/J_DM.dat").read())
p = read_poscar(f"{d}/POSCAR"); ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0]))
r = analyse(p["lattice"], p["frac"], [0, 0], ["Fe", "Fe"], np.array([[0.3, 0, 2.0], [0, 0, -2.0]]), ds.rotations,
            ds.translations, exchange=read_exchange_file(f"{d}/J_DM.dat"), spin=1.0, rmax=3.1, relax=True, n_per=40, nmap=3)
e = np.array(r["directions"])
print("angle between spins:", round(math.degrees(math.acos(e[0] @ e[1])), 3), "°   expected", round(180 - math.degrees(math.atan(0.2)), 3))
print("net moment:", np.round(r["net_moment"], 3), "muB/cell;  stable LSWT:", not r["unstable"])
fig = plot_bands(r, title="canted chain: bands coloured by magnon spin along the net moment")"""),
    md("""## 5. Command line

```bash
cd examples/vasp/Triangular_120AFM_INCAR && symphase magnons --J 1.0 --rmax 3.1
cd examples/vasp/CantedChain_DMI_INCAR   && symphase magnons --exchange J_DM.dat --relax --rmax 3.1
symphase tensors --analysis spin        # spin group, magnet class and s(k) of any non-collinear order
```
Non-reciprocity $\\omega(\\mathbf q)\\neq\\omega(-\\mathbf q)$ (e.g. from uniform DMI) is reported as `nonreciprocity`.
Continue with **09_quiz_spin_groups_magnons_finiteT.ipynb**."""),
], "08_tutorial_noncollinear_magnets.ipynb")

# =============================================================================
# 10  incommensurate spirals
# =============================================================================
save([
    md(r"""# SymPhase tutorial 7 — incommensurate spirals without supercells

A single-Q planar spiral $\mathbf S_j(\mathbf R)=R_{\hat{\mathbf n}}(\mathbf Q\cdot\mathbf R)\,\mathbf S_j(\mathbf 0)$ is **not periodic** in the
crystal cell, and an incommensurate $\mathbf Q$ fits no finite supercell. If the Hamiltonian is invariant under rotations about
$\hat{\mathbf n}$ (Heisenberg exchange, DM vectors $\parallel\hat{\mathbf n}$, anisotropy along $\hat{\mathbf n}$), the order becomes periodic in the
**rotating frame**: every bond matrix is replaced by
$$J_{ij}(\mathbf R)\;\longrightarrow\;J_{ij}(\mathbf R)\,R_{\hat{\mathbf n}}(2\pi\,\mathbf q\cdot\mathbf R),$$
and the general non-collinear LSWT runs on the original (small) magnetic cell. SymPhase

1. finds the optimal $\mathbf q$ by minimising the classical spiral energy $E(\mathbf q,\{\psi_i\})$ (grid + Nelder–Mead), or uses a given $\mathbf q$,
2. checks that the Hamiltonian is rotation-invariant about $\hat{\mathbf n}$ (otherwise it refuses: a supercell is needed),
3. computes the magnons in the rotating frame (neutron intensity appears at $\mathbf k$ and $\mathbf k\pm\mathbf q$),
4. with `--relax` (fixed cell) it warns when an incommensurate spiral would be lower in energy.

Validated against the analytic $J_1$–$J_2$ chain, the DMI helix and an explicit 6-site supercell (tests)."""),
    SETUP,
    md(r"""## 1. Frustrated $J_1$–$J_2$ chain (one atom per cell)

Classically $\cos(2\pi q)=-J_1/(4J_2)$ when $J_2>|J_1|/4$; for $J_1=-1$, $J_2=0.5$ meV: $q=1/6$."""),
    code("""import numpy as np, spglib, math
import matplotlib.pyplot as plt
from symphase.magnons import analyse, read_exchange_file, plot_bands
a = 3.0
lat, fr = np.diag([a, 12, 12]), np.zeros((1, 3))
ds = spglib.get_symmetry_dataset((lat, fr, [0]))
m0 = np.array([[0, 0, 2.0]])
r = analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
            spiral="auto", spiral_axis=[0, 0, 1], n_per=60, nmap=3)
sp = r["spiral"]
print("q =", np.round(sp["q_rlu"], 6), "r.l.u.;  period", round(sp["period_A"], 3), "Å;  E =", sp["E_spiral"],
      "meV/site (ferromagnet:", sp["E_start"], ")")"""),
    code("""Jk = lambda k: -2 * np.cos(k * a) + np.cos(2 * k * a)
Q = 2 * np.pi * abs(sp["q_rlu"][0]) / a
ana = [np.sqrt(max((Jk(k) - Jk(Q)) * ((Jk(k + Q) + Jk(k - Q)) / 2 - Jk(Q)), 0)) for k in r["q"][:, 0]]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(r["x"], r["omega"][:, 0], color="#16325c", lw=2.2, label="SymPhase (rotating frame)")
ax.plot(r["x"][::6], ana[::6], "o", color="#e67e22", mfc="none", label="analytic")
ax.set_xticks([t for t, _ in r["ticks"]]); ax.set_xticklabels([l for _, l in r["ticks"]]); ax.set_ylabel("ω (meV)")
ax.legend(); ax.set_title("J1–J2 spiral: zeros at k = 0 and k = ±q")
print("max |error| =", np.max(np.abs(r["omega"][:, 0] - ana)), "meV")"""),
    md(r"""## 2. Phase diagram: when is the spiral stable?"""),
    code("""J2s = np.linspace(0.0, 1.2, 25)
qs = [abs(analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, J=[-1.0, j2], spin=1.0, rmax=6.1,
                  spiral="auto", spiral_axis=[0, 0, 1], n_per=4, nmap=3)["spiral"]["q_rlu"][0]) for j2 in J2s]
exact = [math.acos(min(1, 1 / (4 * j2))) / (2 * np.pi) if j2 > 0.25 else 0 for j2 in J2s]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(J2s, exact, color="0.6", lw=3, label="arccos(|J1|/4J2)/2π")
ax.plot(J2s, qs, "o", color="#16325c", label="SymPhase")
ax.axvline(0.25, ls=":", color="k"); ax.set_xlabel("J2 / |J1|"); ax.set_ylabel("q (r.l.u.)"); ax.legend()"""),
    md(r"""## 3. Helix from a uniform Dzyaloshinskii–Moriya interaction

Ferromagnetic chain with $\mathbf D\parallel\hat{\mathbf x}$ along the chain: a **Bloch helix** in the plane ⊥ x with
$\tan(2\pi q)=D/|J|$ — here $q=0.0314$ r.l.u., a period of 95.5 Å (31.8 lattice constants): impossible to fit in a small supercell,
trivial in the rotating frame. The sign of $q$ (handedness) is fixed by the sign of $D$."""),
    code("""d = f"{EX}/vasp/Helix_DMI_chain"
print(open(f"{d}/J_DMI.dat").read())
r = analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, exchange=read_exchange_file(f"{d}/J_DMI.dat"),
            spin=1.0, rmax=3.1, spiral="auto", n_per=60, nmap=3)
sp = r["spiral"]
print("q =", round(sp["q_rlu"][0], 6), " expected ±", round(math.atan(0.2) / (2 * np.pi), 6), "; axis", sp["axis"],
      "; period", round(sp["period_A"], 2), "Å =", round(sp["period_A"] / a, 2), "a")"""),
    md(r"""## 4. The fixed-cell trap, now detected

With `relax=True` in the 1-atom cell the best commensurate state is the ferromagnet; SymPhase reports that a spiral is lower:"""),
    code("""r = analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
            relax=True, n_per=4, nmap=3)
print("relaxed (fixed cell):", r["E_classical"], "meV;  spiral check:", r["spiral_check"])"""),
    md("""## 5. Command line

```bash
cd examples/vasp/Spiral_J1J2_chain && symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --spiral auto
cd examples/vasp/Helix_DMI_chain   && symphase magnons --exchange J_DMI.dat --rmax 3.1 --spin 1 --spiral auto
symphase magnons ... --spiral 0.25 0 0 --spiral-axis 0 0 1      # fixed q (r.l.u.) and rotation axis
symphase magnons ... --spiral auto --qdims 0 1                   # search q in the a*-b* plane only
```
Limits: single-Q planar spirals; anisotropy or DM vectors not parallel to $\\hat{\\mathbf n}$ break the rotating-frame symmetry —
SymPhase then refuses; those states (multi-Q, conical, distorted spirals) are handled by the supercell workflow of
**12_tutorial_magnetic_supercells.ipynb**. Continue with **11_quiz_noncollinear_spirals.ipynb**."""),
], "10_tutorial_incommensurate_spirals.ipynb")

# =============================================================================
# quizzes
# =============================================================================
S = [
    dict(type="mc", question="Neumann's principle states that …",
         options={"a": "the symmetry group of a physical property contains the point group of the crystal",
                  "b": "every tensor component is non-zero unless forbidden by translations",
                  "c": "only polar tensors are affected by crystal symmetry",
                  "d": "magnetic order never changes response tensors"},
         answer="a", hint="Symmetry of the cause appears in the effect.",
         explanation="Every response tensor must be invariant under all operations of the (magnetic) point group; components that are not invariant vanish."),
    dict(type="mc", question="How does a spin (or magnetic moment) transform?",
         options={"a": "polar vector, T-even", "b": "axial vector, T-odd", "c": "polar vector, T-odd", "d": "axial vector, T-even"},
         answer="b", explanation="S and L are axial (do not flip under inversion) and odd under time reversal."),
    dict(type="mc", question="The spin current J^{S_k}_i = {v_i, S_k}/2 is …",
         options={"a": "polar in i and k, T-odd", "b": "polar in i, axial in k, T-even",
                  "c": "axial in i, polar in k, T-odd", "d": "a scalar"},
         answer="b", explanation="v is polar & T-odd, S is axial & T-odd; their product is polar(i)×axial(k) and T-even. Hence D(R)=det(R) R⊗R⊗R for σ^k_ij."),
    dict(type="mc", question="A type-II (grey) magnetic space group …",
         options={"a": "contains no operation with T", "b": "contains T itself as a symmetry",
                  "c": "contains T only combined with a translation", "d": "only exists for ferromagnets"},
         answer="b", explanation="Grey groups G1' contain pure time reversal: paramagnets and non-magnetic crystals."),
    dict(type="mc", question="Which symmetry forbids ALL macroscopic T-odd responses (anomalous Hall, magnetic SHE, MOKE …)?",
         options={"a": "inversion P", "b": "a mirror plane", "c": "T or T·τ (time reversal, possibly with a translation)", "d": "PT"},
         answer="c", hint="Point-group part of T·τ is the identity combined with T.",
         explanation="For macroscopic tensors only the point part matters; T or T·τ maps a T-odd tensor onto minus itself."),
    dict(type="mc", question="The anomalous Hall effect corresponds to which part of σ_ij?",
         options={"a": "T-even symmetric", "b": "T-odd antisymmetric", "c": "T-even antisymmetric", "d": "T-odd symmetric"},
         answer="b", explanation="Onsager: σ_ij(M)=σ_ji(−M). The part odd in M is antisymmetric; σ^odd_ij = ε_ijk h_k with Hall vector h."),
    dict(type="mc", question="In a PT-symmetric antiferromagnet the electronic bands are …",
         options={"a": "spin split everywhere", "b": "Kramers (doubly) degenerate at every k", "c": "flat", "d": "gapped"},
         answer="b", explanation="PT maps (k, ↑) to (k, ↓), so every band is doubly degenerate although T is broken."),
    dict(type="mc", question="Which combination characterises an altermagnet?",
         options={"a": "net moment ≠ 0, T present", "b": "compensated, PT present",
                  "c": "compensated, neither PT nor T·τ, non-relativistic spin splitting", "d": "non-magnetic, broken inversion"},
         answer="c", explanation="Altermagnets (e.g. RuO2, CrSb, MnTe) are compensated collinear magnets whose sublattices are related by rotations, not by translation or inversion."),
    dict(type="mc", question="Which magnetic point group allows an anomalous Hall effect?",
         options={"a": "m-3m1'", "b": "4'/mmm'", "c": "4/mm'm'", "d": "all of them"},
         answer="c", hint="Compare the outputs for Pt, RuO2 [001] and Fe [001].",
         explanation="4/mm'm' (bcc Fe, m∥z) allows σ_xy^odd; m-3m1' is grey and 4'/mmm' (RuO2, Néel∥z) forbids the Hall vector."),
    dict(type="text", question="Hands-on: what is the magnetic point group of monoclinic MoO2 (examples/vasp/MoO2_monoclinic)?",
         answer=["2/m1'", "2/m1’"], hint="res['magnetic']['magnetic_point_group']",
         explanation="MoO2 is non-magnetic: grey group P2_1/c1' (BNS 14.76), magnetic point group 2/m1'."),
    dict(type="text", question="Hands-on: what is the BNS symbol of the magnetic space group of examples/vasp/bcc_Fe_m001 (m ∥ z)?",
         answer=["I4/mm'm'"], hint="res['magnetic']['bns_symbol']",
         explanation="I4/mm'm' (BNS 139.537): the 4-fold axis is along m; mirrors containing m are primed."),
    dict(type="mc", question="Hands-on: for bcc Fe with m ∥ z, the anomalous Hall vector h points along …",
         options={"a": "x", "b": "y", "c": "z", "d": "it vanishes"},
         answer="c", explanation="h = (0, 0, σ_xy): the Hall vector is parallel to the magnetisation, as for any ferromagnet with a high-symmetry moment."),
    dict(type="num", question="Hands-on: how many independent T-odd (magnetic) spin-conductivity components does bcc Fe (m ∥ z) have?",
         answer=4, tol=0.1, hint="res['tensors']['spin']['n_odd']",
         explanation="σ^x_xz, σ^x_zx, σ^z_xx and σ^z_zz (plus symmetry-related ones): spin-polarised longitudinal/transverse currents with spin ∥ m."),
    dict(type="mc", question="Hands-on: RuO2 with Néel vector ∥ [001]. Is the anomalous Hall effect allowed?",
         options={"a": "yes, h ∥ z", "b": "yes, h ∥ x", "c": "no"}, answer="c",
         explanation="P4_2'/mnm' (MPG 4'/mmm') forbids any Hall vector: no axial vector is invariant under all its operations."),
    dict(type="mc", question="Hands-on: RuO2 with Néel ∥ [001]: what is the form of the T-odd σ^z block?",
         options={"a": "σ^z_xy = −σ^z_yx (SHE-like)", "b": "σ^z_xy = +σ^z_yx (spin-splitter)", "c": "σ^z_xx only", "d": "it vanishes"},
         answer="b", explanation="The symmetric, T-odd σ^z_xy=σ^z_yx is the spin-splitter effect of altermagnetic RuO2 (spin current with spin ∥ Néel vector)."),
    dict(type="text", question="Hands-on: BNS symbol of RuO2 with the Néel vector along [100] (examples/vasp/RuO2_Neel100_SOC)?",
         answer=["Pnn'm'"], explanation="Pnn'm' (58.398), MPG mm'm': rotating the Néel vector into the plane lowers the symmetry."),
    dict(type="mc", question="Hands-on: for RuO2 with Néel ∥ [100], the Hall vector is along …",
         options={"a": "[100]", "b": "[010]", "c": "[001]", "d": "zero"}, answer="b",
         explanation="h = (0, −σ_xz, 0): the crystal Hall vector is perpendicular to the Néel vector — a hallmark of the crystal Hall effect."),
    dict(type="num", question="Hands-on: how many independent T-even spin Hall components does monoclinic MoO2 have?",
         answer=13, tol=0.1, hint="res['tensors']['spin']['n_even']",
         explanation="13 (cubic Pt has 1). The orbital Hall tensor has the same 13 components; without SOC all spin components vanish but the orbital ones survive."),
    dict(type="mc", question="MoO2 has point group 2/m with unique axis y. Which spin/orbital conductivity components σ^k_ij can be non-zero?",
         options={"a": "only those with k ⊥ i ⊥ j", "b": "those with an odd number of y indices among (k, i, j)",
                  "c": "those with an even number of y indices", "d": "only σ^y_ij"},
         answer="b", hint="C2y maps x → −x, z → −z; the tensor is axial (det R = +1 for C2).",
         explanation="Under C2y a component picks (−1)^(n_x+n_z); invariance needs n_x+n_z even, i.e. n_y odd. Inversion adds nothing for an axial rank-3 tensor. Examples: σ^y_xx, σ^y_yy, σ^x_xy (unconventional) and σ^z_xy (conventional)."),
    dict(type="mc", question="SymPhase analyses a calculation WITHOUT spin–orbit coupling. Which statement is right?",
         options={"a": "the result is exact", "b": "the true symmetry is a larger spin group, so some components allowed by the magnetic group may vanish without SOC",
                  "c": "no magnetic group can be defined", "d": "all spin tensors vanish"},
         answer="b", explanation="Without SOC spin rotations are decoupled from the lattice (spin groups); the relativistic MSG analysis gives the components that SOC can switch on."),
]

H = [
    dict(type="mc", question="The formation energy per atom of a compound is …",
         options={"a": "its total energy per atom", "b": "(E_tot − Σ n_i μ_i⁰) / Σ n_i with elemental reference energies μ_i⁰",
                  "c": "the energy above hull", "d": "the cohesive energy"},
         answer="b", explanation="Energy relative to the elements in their reference states, per atom."),
    dict(type="mc", question="A phase with E_hull = 0 …",
         options={"a": "is on the convex hull (thermodynamically stable at 0 K)", "b": "is unstable", "c": "has zero formation energy", "d": "is an element"},
         answer="a", explanation="E_hull = 0 means no combination of other phases has lower energy at that composition."),
    dict(type="mc", question="The convex hull is constructed from …",
         options={"a": "the upper envelope of (composition, E_f)", "b": "the lower convex envelope of (composition, E_f) points",
                  "c": "a spline through all points", "d": "only binary compounds"},
         answer="b", explanation="Lower convex hull in composition–energy space (Qhull in pymatgen)."),
    dict(type="mc", question="For a metastable phase, E_hull is …",
         options={"a": "its formation energy", "b": "the energy per atom released by decomposing into the hull phases at the same composition",
                  "c": "the barrier for decomposition", "d": "always > 1 eV/atom"},
         answer="b", explanation="It is a thermodynamic driving force, not a kinetic barrier."),
    dict(type="mc", question="Why does SymPhase refuse to build a hull when an element has no reference entry?",
         options={"a": "Qhull needs at least 10 points", "b": "formation energies and the simplex corners require the elemental references",
                  "c": "pymatgen only supports metals", "d": "it does not refuse"},
         answer="b", explanation="Every element must have at least one pure-element calculation to define μ_i⁰ and the hull corners."),
    dict(type="mc", question="Can you put Materials Project energies and your own VASP energies into one hull?",
         options={"a": "always", "b": "only if computed with identical settings (functional, PAW, U, corrections …)",
                  "c": "never, pymatgen forbids it", "d": "yes, if ENCUT is the same"},
         answer="b", explanation="Total energies are only comparable for identical computational settings; MP uses its own correction scheme (MP2020)."),
    dict(type="mc", question="Which statement about E_hull and synthesizability is most accurate?",
         options={"a": "E_hull > 0 can never be synthesised", "b": "many metastable phases with E_hull of a few tens of meV/atom are made; there is no universal cut-off",
                  "c": "everything below 1 eV/atom is stable", "d": "E_hull determines the melting point"},
         answer="b", explanation="Kinetics, entropy and finite temperature matter; typical screening windows are 25–100 meV/atom."),
    dict(type="mc", question="The stability margin (equilibrium reaction energy) of a stable phase is …",
         options={"a": "its E_hull", "b": "how far it lies below the hull formed by all OTHER phases (negative = robust)",
                  "c": "its band gap", "d": "always zero"},
         answer="b", explanation="Removing the phase and recomputing the hull gives the energy it gains relative to its neighbours."),
    dict(type="num", question="Hands-on: load examples/hull/NiAl_binary.csv. How many entries are stable (elements included)?",
         answer=6, tol=0.1, hint="len(h.pd.stable_entries)",
         explanation="Ni, Al, Ni3Al, NiAl, Ni2Al3 and NiAl3 (toy data)."),
    dict(type="num", question="Hands-on (Ni–Al): E_hull of 'Ni5Al3-Pt5Ga3' in meV/atom?",
         answer=42.5, tol=0.6, unit="meV/atom", explanation="It decomposes into 1/2 Ni3Al + 1/2 NiAl, 42.5 meV/atom higher."),
    dict(type="mc", question="Hands-on (Ni–Al): 'NiAl2-hyp' decomposes into …",
         options={"a": "NiAl + Al", "b": "Ni2Al3 + NiAl3", "c": "Ni3Al + Al", "d": "it is stable"},
         answer="b", explanation="0.556 Ni2Al3 + 0.444 NiAl3: the two hull phases bracketing x(Ni)=1/3."),
    dict(type="num", question="Hands-on (Ni–Al): E_hull of the L1₀ polymorph 'NiAl-L10' (meV/atom)?",
         answer=125, tol=0.6, unit="meV/atom", explanation="A polymorph decomposes into the ground state of the same composition: −0.540 − (−0.665) = 125 meV/atom."),
    dict(type="num", question="Hands-on (Ti–Ni–Sn): stability margin of half-Heusler TiNiSn in meV/atom?",
         answer=-97.4, tol=1.0, unit="meV/atom", hint="look at 'stability_margin' in h.table()",
         explanation="−97.4 meV/atom: removing TiNiSn from the hull would raise the energy at its composition by that amount."),
    dict(type="mc", question="Hands-on (Ti–Ni–Sn): 'TiNiSn2-hyp' decomposes into …",
         options={"a": "TiSn2 + Ni3Sn4", "b": "TiNiSn + Sn", "c": "TiNi2Sn + Sn", "d": "it is stable"},
         answer="b", explanation="0.75 TiNiSn + 0.25 Sn, E_hull = 127.5 meV/atom."),
    dict(type="num", question="Hands-on (Ti–Zr–Ni–Sn): E_hull of the (Ti0.5Zr0.5)NiSn SQS in meV/atom?",
         answer=5.0, tol=0.3, unit="meV/atom", explanation="5 meV/atom above the TiNiSn + ZrNiSn tie-line: a small positive mixing energy."),
    dict(type="mc", question="Experimentally (Ti,Zr)NiSn solid solutions exist although the SQS lies above the hull. The main reason is …",
         options={"a": "DFT errors only", "b": "configurational entropy −TΔS_mix lowers the free energy at synthesis temperatures",
                  "c": "pressure", "d": "spin–orbit coupling"},
         answer="b", explanation="A few meV/atom of mixing enthalpy is easily overcome by ideal mixing entropy at 1000 K (next question)."),
    dict(type="num", question="Ideal mixing on ONE sublattice per formula unit of (Ti0.5Zr0.5)NiSn (3 atoms). What is T·S_mix at 1000 K in meV per atom?",
         answer=19.9, tol=0.5, unit="meV/atom", hint="S_mix = k_B ln 2 per mixing site; k_B = 8.617e-5 eV/K",
         explanation="k_B·1000 K·ln2 = 59.7 meV per mixing site = 19.9 meV/atom ≫ 5 meV/atom mixing enthalpy."),
    dict(type="mc", question="Hands-on (Ti–Zr–Hf–Ni–Sn): the high-entropy (TiZrHf)NiSn SQS decomposes into …",
         options={"a": "TiNiSn + ZrNiSn + HfNiSn", "b": "(Zr,Hf)NiSn + TiNiSn", "c": "TiNi2Sn + Sn", "d": "it is stable"},
         answer="b", explanation="2/3 (Zr0.5Hf0.5)NiSn + 1/3 TiNiSn, E_hull ≈ 2.7 meV/atom: in the toy data (Zr,Hf)NiSn is itself on the hull."),
    dict(type="mc", question="How does SymPhase visualise systems with more than four elements?",
         options={"a": "it cannot", "b": "grids of all binary/ternary sub-system hulls, E_hull bar charts and pseudo-binary slices",
                  "c": "a 5D projection", "d": "only text output"},
         answer="b", explanation="plot_subsystems, plot_ehull_bar and plot_pseudo_binary work for any number of elements."),
    dict(type="mc", question="What does `symphase hull -n 40` (or MPI) parallelise?",
         options={"a": "only plotting", "b": "the Qhull construction itself",
                  "c": "per-entry E_hull/decompositions/margins, candidate screening, VASP-folder parsing and pseudo-binary hull points",
                  "d": "nothing, -n is ignored"},
         answer="c", explanation="The hull construction (Qhull) is serial and fast; everything done per entry or per candidate is distributed over processes (one node) or MPI ranks (many nodes)."),
]

build_bank(os.path.join(DATA, "quiz_symmetry.json"), "Quiz 1 — magnetic symmetry & response tensors", S)
build_bank(os.path.join(DATA, "quiz_hull.json"), "Quiz 2 — convex hull & phase stability", H)

M = [
    dict(type="mc", question="How many collinear spin space groups are there (non-trivial parts)?",
         options={"a": "230", "b": "1421", "c": "1651", "d": "uncountably many"}, answer="b",
         explanation="They map one-to-one onto the black-and-white magnetic groups of type I, III and IV: 230 + 674 + 517 = 1421."),
    dict(type="mc", question="SymPhase labels a collinear SSG by the BNS number of a magnetic group. Which replacement defines that group?",
         options={"a": "[C2⊥ || g] → g T", "b": "[E || g] → g T", "c": "g → g P", "d": "spin rotations → lattice rotations"},
         answer="a", explanation="Every element that flips the spins, [C2⊥||g], becomes the primed operation gT; spin-preserving ones stay unprimed."),
    dict(type="mc", question="The spin-only group of a collinear magnet is …",
         options={"a": "trivial", "b": "{E, C2⊥T}", "c": "∞m1 = C∞(n) ⋊ {E, C2⊥T}", "d": "O(3)"},
         answer="c", explanation="All rotations about the spin axis plus C2⊥ combined with T leave a collinear configuration invariant."),
    dict(type="mc", question="Why is the anomalous Hall effect of a collinear magnet forbidden without SOC?",
         options={"a": "because of inversion", "b": "the spin-only element [C2⊥T || E] acts on σ_ij as pure time reversal",
                  "c": "because spins are axial", "d": "it is not forbidden"},
         answer="b", explanation="[C2⊥T||E] leaves the lattice untouched but contains T, so every T-odd charge response vanishes; SOC is required."),
    dict(type="mc", question="In a collinear magnet without SOC every magnon carries …",
         options={"a": "ΔS_z = ±1 (chirality α or β)", "b": "ΔS_z = 0", "c": "ΔS_z = ±1/2", "d": "no defined spin"},
         answer="a", explanation="S_z is conserved, so each mode lowers (α) or raises (β) the total S_z by one."),
    dict(type="mc", question="The symmetry of the magnon chirality splitting Δω(q) in an altermagnet is …",
         options={"a": "always s-wave", "b": "the same as the electronic NR spin splitting s(k) (d, g or i-wave)",
                  "c": "always p-wave", "d": "unrelated to the spin group"},
         answer="b", explanation="Both are fixed by the same spin group: RuO2 is d-wave for electrons and magnons."),
    dict(type="mc", question="Which exchange paths split the magnon chiralities in RuO2?",
         options={"a": "the nearest-neighbour inter-sublattice bonds", "b": "the c-axis bonds",
                  "c": "the same-sublattice 6.35 Å bonds along [110] and [1-10], which are inequivalent", "d": "none"},
         answer="c", explanation="Crystal symmetry (O positions) makes the [110] and [1-10] bonds of one sublattice different and swaps them on the other: J7 ≠ J8 gives Δω ∝ qx·qy."),
    dict(type="num", question="Hands-on: how many symmetry classes of Ru–Ru bonds does SymPhase find in RuO2 up to 6.5 Å?",
         answer=8, tol=0.1, hint="symphase magnons --list-bonds --rmax 6.5 (or res['classes'])",
         explanation="J1 … J8; the last two have identical length 6.35 Å but are not related by any crystal operation."),
    dict(type="text", question="Hands-on: BNS number labelling the collinear spin space group of MnTe (examples/vasp/MnTe_altermagnet_INCAR)?",
         answer=["194.269"], hint="analyze_tensors(...)['spin_group']['ssg']['number']",
         explanation="P6_3'/m'mc' (194.269), consistent with the spin Laue group ²6/²m¹m²m."),
    dict(type="mc", question="Why is there no magnon chirality splitting in bcc Cr (AFM with opposite spins at (0,0,0) and (½,½,½))?",
         options={"a": "the exchange is too small", "b": "the sublattices are connected by a translation [C2⊥||τ]",
                  "c": "Cr is non-magnetic", "d": "there is a splitting"},
         answer="b", explanation="[C2⊥||τ] maps α onto β at the same q, so both chiralities are degenerate everywhere (as for electrons)."),
    dict(type="mc", question="Why does non-collinear LSWT use a local frame (u_i, v_i) for every spin?",
         options={"a": "to make the code faster", "b": "Holstein–Primakoff bosons describe deviations from each spin's own direction v_i",
                  "c": "because DMI requires it", "d": "it is only a plotting convention"},
         answer="b", explanation="The bosons live in the plane ⊥ v_i (u_i = x_i + i y_i); the global spectrum is independent of the in-plane choice of x_i."),
    dict(type="num", question="Hands-on: how many zero-energy (Goldstone) modes does the 120° triangular antiferromagnet have at Γ of its 3-sublattice cell?",
         answer=3, tol=0.1, hint="res['n_goldstone'] for examples/vasp/Triangular_120AFM_INCAR with J = 1",
         explanation="ω(k), ω(k+K), ω(k−K) all vanish at k = 0: the three generators of SO(3) are all broken by the coplanar order."),
    dict(type="mc", question="Which formula is the LSWT dispersion of the nearest-neighbour 120° triangular Heisenberg AFM (γ_k = Σ cos(k·δ)/3)?",
         options={"a": "ω = 2JS(1 − γ_k)", "b": "ω = 3JS √((1 − γ_k)(1 + 2γ_k))", "c": "ω = 6JS |sin k|", "d": "ω = 3JS γ_k"},
         answer="b", explanation="SymPhase reproduces it to 10⁻⁴ meV (tutorial 6 and the tests)."),
    dict(type="num", question="Hands-on: canted chain with J = 1 meV and staggered D = 0.2 meV: by how many degrees are the spins canted away from antiparallel after --relax?",
         answer=11.31, tol=0.05, unit="degrees", hint="180° − angle between the relaxed directions",
         explanation="φ = arctan(D/J) = 11.31°; the canting gives a weak ferromagnetic moment ⊥ to the Néel vector."),
    dict(type="mc", question="To obtain a canted (weak-ferromagnetic) state rather than a spiral in a two-sublattice chain, the DM vectors must be …",
         options={"a": "uniform along the chain (same D for every i → i+1 bond)", "b": "staggered, so that D·(S_A × S_B) has the same sign on both bonds",
                  "c": "parallel to the spins", "d": "zero"},
         answer="b", explanation="A uniform D twists every bond the same way (spiral); the staggered pattern of J_DM.dat favours one fixed canting angle."),
    dict(type="mc", question="Hands-on: which non-relativistic spin-splitting form does SymPhase find for the coplanar 120° triangular antiferromagnet?",
         options={"a": "none", "b": "d-wave", "c": "f-wave (odd in k)", "d": "s-wave"}, answer="c",
         hint="analyze_tensors(...)['spin_group']['spin_splitting']",
         explanation="The coplanar spin-only element C2(n)T forces s_z(−k) = −s_z(k); the lowest allowed form is cubic (f-wave), s_z ∝ kx(kx² − 3ky²)."),
    dict(type="num", question="Configurational entropy (k_B per atom) of a high-entropy half-Heusler with sublattices 'TiZrHf | Ni | Sn'?",
         answer=0.3662, tol=0.002, unit="k_B/atom", hint="parse_mixing('TiZrHf|Ni|Sn')",
         explanation="−(3 · ⅓ ln ⅓)/3 = ln 3 / 3 = 0.366 k_B per atom."),
    dict(type="num", question="Hands-on: mixing entropy only (vib='none'): temperature at which (Ti0.5Zr0.5)NiSn (E_hull = 5 meV/atom) reaches the hull?",
         answer=251, tol=2, unit="K", hint="FiniteTHull(..., vib='none').stabilisation_temperature(...)",
         explanation="k_B T ln2 / 3 = 5 meV → T* = 251 K."),
    dict(type="num", question="Debye model: zero-point vibrational energy per atom for θ_D = 400 K, in meV/atom?",
         answer=38.77, tol=0.2, unit="meV/atom",
         explanation="F_vib(T = 0) = (9/8) k_B θ_D = 1.125 × 0.08617 meV/K × 400 K = 38.8 meV/atom."),
    dict(type="mc", question="Which statement about vib='bartel' is correct?",
         options={"a": "it needs phonopy files", "b": "it uses the SISSO descriptor of Bartel et al. via pymatgen, needs structures, valid 300–2000 K",
                  "c": "it works at 0 K only", "d": "it replaces the mixing entropy"},
         answer="b", explanation="GibbsComputedStructureEntry estimates G(T) from volume and mass; mixing entropy is still added on top."),
]
build_bank(os.path.join(DATA, "quiz_magnons.json"), "Quiz 3 — spin groups, magnons (collinear and non-collinear) & finite-T hulls", M)

SP = [
    dict(type="mc", question="Why can an incommensurate spiral not be simulated exactly with an ordinary supercell?",
         options={"a": "supercells are too slow", "b": "an incommensurate q is not a rational fraction of a reciprocal-lattice vector, so no finite supercell is periodic",
                  "c": "spirals have no magnons", "d": "it can, any supercell works"},
         answer="b", explanation="Only commensurate q = m/N fit an N-cell supercell; incommensurate q never repeats."),
    dict(type="mc", question="What does the rotating-frame method do to every bond matrix?",
         options={"a": "J → J R_n(2π q·R)", "b": "J → J + q", "c": "J → J cos(q·R)", "d": "nothing"},
         answer="a", explanation="Spins of cell R are R_n(2π q·R) times those of cell 0, so the bond coupling seen by the cell-0 frames is J R_n(2π q·R): periodic again."),
    dict(type="mc", question="Which condition must the Hamiltonian satisfy for the rotating-frame treatment?",
         options={"a": "only ferromagnetic J", "b": "invariance under rotations about the spiral axis n (Heisenberg, D ∥ n, anisotropy along n)",
                  "c": "inversion symmetry", "d": "no condition"},
         answer="b", explanation="Then R_n^T J R_n = J for every bond and the transformed Hamiltonian is translation invariant; SymPhase checks this and refuses otherwise."),
    dict(type="mc", question="In the rotating frame, where does neutron scattering see the magnon ω(k)?",
         options={"a": "only at k", "b": "at k and k ± q", "c": "only at q", "d": "nowhere"},
         answer="b", explanation="The lab-frame spin correlations mix k with k ± q because the frame rotates with wavevector q."),
    dict(type="mc", question="For the J1–J2 chain with J1 < 0 (FM) and J2 > 0 (AFM), a spiral appears when …",
         options={"a": "J2 > |J1|/4", "b": "J2 > |J1|", "c": "J2 < |J1|/4", "d": "always"},
         answer="a", explanation="J(q) = 2J1 cos(2πq) + 2J2 cos(4πq) has its minimum away from q = 0 only for J2 > |J1|/4; then cos(2πq) = −J1/(4J2)."),
    dict(type="num", question="Hands-on: J1 = −1 meV, J2 = 0.5 meV (examples/vasp/Spiral_J1J2_chain). Spiral wavevector |q| in r.l.u.?",
         answer=0.16667, tol=0.0005, hint="analyse(..., spiral='auto')['spiral']['q_rlu']",
         explanation="cos(2πq) = 1/2 → q = 1/6 (period 6a = 18 Å)."),
    dict(type="num", question="Hands-on: classical energy gain of that spiral relative to the ferromagnet (meV per site, S = 1)?",
         answer=0.25, tol=0.005, unit="meV/site",
         explanation="FM: J1 + J2 = −0.5; spiral: J1 cos60° + J2 cos120° = −0.75 → gain 0.25 meV/site."),
    dict(type="num", question="Hands-on: for J1 = −1 meV, J2 = 0.2 meV, what is |q| (r.l.u.)?",
         answer=0.0, tol=0.002, explanation="J2 < |J1|/4: the ferromagnet (q = 0) is the ground state."),
    dict(type="num", question="Hands-on: J2 = |J1| = 1 meV (J1 FM). |q| in r.l.u.?",
         answer=0.20978, tol=0.0005, explanation="arccos(1/4)/(2π) = 0.2098 r.l.u."),
    dict(type="mc", question="Besides k = 0, where does the rotating-frame magnon ω(k) of a Heisenberg spiral vanish?",
         options={"a": "nowhere", "b": "at k = ±q", "c": "at the zone boundary", "d": "at k = q/2"},
         answer="b", explanation="ω(k) ∝ √[(J(k) − J(q))(...)]: J(k) = J(q) at k = ±q (rotations of the spiral plane, Goldstone modes)."),
    dict(type="num", question="Hands-on: FM chain with uniform DMI, J = −1 meV, D = 0.2 meV (examples/vasp/Helix_DMI_chain). Helix period in lattice constants?",
         answer=31.83, tol=0.1, hint="res['spiral']['period_A'] / a",
         explanation="tan(2πq) = D/|J| → q = 0.03142 r.l.u. → period 1/q = 31.8 a (95.5 Å)."),
    dict(type="num", question="Same chain with D = 0.4 meV: |q| in r.l.u.?",
         answer=0.06056, tol=0.0003, explanation="arctan(0.4)/(2π) = 0.0606 r.l.u."),
    dict(type="mc", question="A uniform DM vector D ∥ chain direction produces …",
         options={"a": "a Bloch helix rotating in the plane ⊥ chain", "b": "a cycloid in the plane containing the chain",
                  "c": "a canted ferromagnet", "d": "a collinear antiferromagnet"},
         answer="a", explanation="The spins rotate about n = D (the chain axis): a helix; D ⊥ chain (interfacial DMI) gives cycloids."),
    dict(type="mc", question="What decides the handedness (sign of q) of a DMI spiral?",
         options={"a": "the sign of J", "b": "the sign (direction) of D relative to the bond", "c": "the spin length", "d": "nothing, both are degenerate"},
         answer="b", explanation="D·(S_i × S_j) favours one sense of rotation; reversing D reverses q. Without DMI ±q are degenerate."),
    dict(type="mc", question="What does SymPhase do when you use --relax in a fixed cell but an incommensurate spiral would be lower in energy?",
         options={"a": "nothing", "b": "it reports the spiral q and energy and suggests --spiral auto", "c": "it enlarges the cell automatically", "d": "it crashes"},
         answer="b", explanation="'spiral_check' compares the relaxed fixed-cell energy with the optimal single-Q spiral."),
    dict(type="mc", question="Hands-on (concept check): for q = 1/6 the spiral is commensurate. What does the explicit 6-site supercell give?",
         options={"a": "a different energy", "b": "the same energy per site and the folded rotating-frame spectrum", "c": "an unstable state", "d": "no magnons"},
         answer="b", explanation="Tested in SymPhase: E/site = −0.75 meV in both, and the 6 supercell branches equal ω_rot(k + m·2π/6a)."),
    dict(type="mc", question="Single-ion anisotropy −K (S·x)² with the spiral axis n = z is …",
         options={"a": "allowed in the rotating frame", "b": "not rotation-invariant about z: single-Q fails, SymPhase refuses — use a supercell",
                  "c": "irrelevant", "d": "equivalent to DMI"},
         answer="b", explanation="Anisotropy within the spiral plane bunches the spiral (higher harmonics); only anisotropy along n keeps the rotating-frame symmetry."),
    dict(type="mc", question="How does SymPhase find the optimal spiral?",
         options={"a": "Luttinger–Tisza only", "b": "minimising the classical energy over q (grid + Nelder–Mead) and the sublattice phases ψ_i (BFGS)",
                  "c": "from DFT", "d": "it guesses q = 0"},
         answer="b", explanation="E(q, ψ) is evaluated with the rotating-frame bond matrices; the global minimum gives q and the phases."),
    dict(type="mc", question="What is the net magnetic moment of an incommensurate planar spiral?",
         options={"a": "S per site", "b": "zero (it averages out over the spiral)", "c": "S/2", "d": "it depends on the cell"},
         answer="b", explanation="The moments rotate through all in-plane directions; SymPhase reports net moment 0 for q ≠ 0."),
    dict(type="mc", question="Which limitation remains?",
         options={"a": "only collinear magnets", "b": "single-Q planar spirals; multi-Q, conical or anisotropy-distorted spirals need a supercell",
                  "c": "no DMI allowed", "d": "only 1D chains"},
         answer="b", explanation="The rotating frame handles one q with a rotation-invariant Hamiltonian; everything else uses the supercell workflow (--supercell auto, tutorial 12)."),
]
build_bank(os.path.join(DATA, "quiz_spirals.json"), "Quiz 4 — non-collinear magnets and incommensurate spirals", SP)




def quiz_nb(name, title, qs, bank, intro, helper):
    cells = [md(f"# {title}\n\n{intro}\n\n**How to answer:** replace `...` by your answer and run the cell — "
                "a letter for multiple choice, a number or a text otherwise. `q.hint(n)`, `q.reveal(n)`, `q.score()`.\n"
                "Prefer clicking? use `q.widget(n)` (needs `ipywidgets`)."),
             SETUP,
             code(f"from symphase.quiz import Quiz\nq = Quiz('{bank}')\nprint(q.title, '-', len(q), 'questions')"),
             md("### Helper cell for the hands-on questions"), code(helper)]
    for k, qq in enumerate(qs, 1):
        txt = f"### Question {k}\n{qq['question']}"
        if qq.get("options"):
            txt += "\n\n" + "\n".join(f"* **({a})** {b}" for a, b in qq["options"].items())
        if qq["type"] == "num":
            txt += f"\n\n*numerical answer{' in ' + qq['unit'] if qq.get('unit') else ''}*"
        cells += [md(txt), code(f"q.answer({k}, ...)")]
    cells += [md("## Your score"), code("q.score()")]
    save(cells, name)


quiz_nb("04_quiz_symmetry_tensors.ipynb", "Quiz 1 — magnetic symmetry & conductivity tensors (20 questions)",
        S, "symmetry", "Work through **01_tutorial_symmetry_tensors.ipynb** first. Questions 10–18 are hands-on; Q10, Q18 and Q19 use monoclinic MoO₂.",
        """from symphase import analyze_tensors
pt  = analyze_tensors(f"{EX}/vasp/fcc_Pt")
fe  = analyze_tensors(f"{EX}/vasp/bcc_Fe_m001")
rz  = analyze_tensors(f"{EX}/vasp/RuO2_Neel001_noSOC")
rx  = analyze_tensors(f"{EX}/vasp/RuO2_Neel100_SOC")
moo = analyze_tensors(f"{EX}/vasp/MoO2_monoclinic")
# explore e.g.  fe['magnetic'], fe['tensors']['spin']['n_odd'], print(rz['report'])""")
quiz_nb("05_quiz_convex_hull.ipynb", "Quiz 2 — convex hull & phase stability (20 questions)",
        H, "hull", "Work through **02_tutorial_convex_hull.ipynb** first. Questions 9–15 and 18 are hands-on (toy data).",
        """from symphase.hull import HullAnalysis, load_csv
nial = HullAnalysis(load_csv(f"{EX}/hull/NiAl_binary.csv"))
tins = HullAnalysis(load_csv(f"{EX}/hull/TiNiSn_ternary.csv"))
tzns = HullAnalysis(load_csv(f"{EX}/hull/TiZrNiSn_quaternary.csv"))
he   = HullAnalysis(load_csv(f"{EX}/hull/TiZrHfNiSn_5component.csv"))
# explore e.g.  nial.print_table(),  [r for r in tins.table() if r['name']=='TiNiSn-HH']""")

quiz_nb("09_quiz_spin_groups_magnons_finiteT.ipynb",
        "Quiz 3 — spin-space groups, magnons (collinear & non-collinear) and finite-T hulls (20 questions)",
        M, "magnons", "Work through tutorials **06**, **07** and **08** first. Questions 8, 9, 12, 14, 16 and 18 are hands-on.",
        """import numpy as np, spglib, math
from symphase import analyze_tensors
from symphase.tensors import read_poscar, read_incar
from symphase.magnons import analyse, read_exchange_file
from symphase.hull import load_csv
from symphase.thermo import FiniteTHull, parse_mixing
p = read_poscar(f"{EX}/vasp/RuO2_Neel001_noSOC/POSCAR")
ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0, 1, 1, 1, 1]))
ruo2 = analyse(p["lattice"], p["frac"], [0, 0, 1, 1, 1, 1], ["Ru"] * 2 + ["O"] * 4,
               np.array([[0, 0, 1.2], [0, 0, -1.2]] + [[0, 0, 0]] * 4), ds.rotations, ds.translations, rmax=6.5)
mnte = analyze_tensors(f"{EX}/vasp/MnTe_altermagnet_INCAR", tensors="charge")
tri  = analyze_tensors(f"{EX}/vasp/Triangular_120AFM_INCAR", tensors="charge")
# explore e.g.  len(ruo2['classes']),  mnte['spin_group']['ssg'],  tri['spin_group']['spin_splitting']""")

quiz_nb("11_quiz_noncollinear_spirals.ipynb", "Quiz 4 — non-collinear magnets and incommensurate spirals (20 questions)",
        SP, "spirals", "Work through tutorials **08** and **10** first. Questions 6–9, 11, 12 and 16 are hands-on.",
        """import numpy as np, spglib, math
from symphase.magnons import analyse, read_exchange_file, dm_matrix
a = 3.0
lat, fr = np.diag([a, 12, 12]), np.zeros((1, 3))
ds = spglib.get_symmetry_dataset((lat, fr, [0])); m0 = np.array([[0, 0, 2.0]])
def spiral(J1, J2):
    return analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, J=[J1, J2], spin=1.0, rmax=6.1,
                   spiral="auto", spiral_axis=[0, 0, 1], n_per=6, nmap=3)["spiral"]
def helix(D):
    ex = [(1, 1, [1, 0, 0], -1.0 * np.eye(3) + dm_matrix([D, 0, 0]))]
    return analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, exchange=ex, spin=1.0, rmax=3.1,
                   spiral="auto", n_per=6, nmap=3)["spiral"]
# e.g.  spiral(-1.0, 0.5),  helix(0.2)['period_A'] / a""")


# =============================================================================
# 12  magnetic supercells: multi-Q, conical, distorted spirals
# =============================================================================
SC_HELPER = """import numpy as np, spglib, math
import matplotlib.pyplot as plt
from symphase.magnons import analyse, read_exchange_file, plot_bands
from symphase.tensors import read_poscar
from symphase.supercell import run_supercell, scan_sizes, report, plot_state, write_outputs
a = 3.0
lat, fr = np.diag([a, 12, 12]), np.zeros((1, 3))          # 1D chain, one Fe per cell
ds = spglib.get_symmetry_dataset((lat, fr, [0])); m0 = np.array([[0, 0, 2.0]])
def chain(size="auto", J=(-1.0, 0.5), **kw):
    base = dict(J=list(J), spin=1.0, rmax=6.1, restarts=3, spectrum=False); base.update(kw)
    return run_supercell(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, size=size, **base)
sk = f"{EX}/vasp/Skyrmion_triangular"
ps = read_poscar(f"{sk}/POSCAR"); dss = spglib.get_symmetry_dataset((ps["lattice"], ps["frac"], [0]))
exs = read_exchange_file(f"{sk}/J_DMI.dat")
def skx(B, size=(10, 10, 1), **kw):
    base = dict(restarts=2, spectrum=False); base.update(kw)
    return run_supercell(ps["lattice"], ps["frac"], [0], ["Mn"], np.array([[0, 0, 2.0]]), dss.rotations,
                         dss.translations, size=list(size) if size != "auto" else "auto", exchange=exs, spin=1.0,
                         rmax=3.1, field=[0, 0, B], **base)"""

save([
    md(r"""# SymPhase tutorial 12 — magnetic supercells: multi-Q, conical and distorted spirals

The rotating frame (tutorial 10) treats **one planar spiral** in a Hamiltonian invariant about the spiral axis, without a supercell.
Everything else needs an explicit magnetic supercell:

| state | breaks the rotating frame because … | example |
|---|---|---|
| conical spiral | uniform component along the axis (field) | `Conical_J1J2_field` |
| distorted spiral | in-plane anisotropy → harmonics 3q, 5q | `Distorted_spiral_K` |
| multi-Q / skyrmion crystal | several Q at once, DM vectors in several directions | `Skyrmion_triangular` |

`symphase magnons ... --supercell auto` runs the complete workflow (full step-by-step guide: `docs/SUPERCELL_WORKFLOW.md`):

1. **Luttinger–Tisza** scan → ordering vector $q_{LT}$ and its star;
2. **supercell size** from commensurate approximants of all star members;
3. exact **mapping** of exchange, anisotropy and field onto the supercell;
4. **classical relaxation** (L-BFGS) from many seeds: input state, ferromagnet, single-Q / conical / multi-Q seeds built from the LT eigenvectors, random states;
5. **identification**: structure factor, harmonics, uniform weight, topological charge;
6. **spin waves** in the supercell (stability, Goldstone modes, folded bands);
7. **files** for DFT and for the symmetry analysis of the new state."""),
    SETUP,
    code(SC_HELPER),
    md(r"""## 1. Sanity check: the J₁–J₂ chain needs 6 cells

$J_1=-1$, $J_2=0.5$ meV: $q=1/6$ exactly. The automatic size search must find 6 cells and the rotating-frame energy −0.75 meV."""),
    code("""r = chain()
report(r, ["Fe"])
print("\\nsupercell:", np.diag(r["M"]), "  E/cell =", round(r["E_cell"], 8))"""),
    md(r"""## 2. Incommensurate $q$: converge the supercell size

For $J_2=0.4$ the exact $q=\arccos(1/1.6)/2\pi=0.14255$ is not a simple fraction. A supercell of $n$ cells allows only $q=m/n$;
the energy per cell is **not monotonic** in $n$ — it is lowest when $m/n$ is closest to $q$."""),
    code("""exact = analyse(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, J=[-1.0, 0.4], spin=1.0, rmax=6.1,
                spiral="auto", spiral_axis=[0, 0, 1], n_per=4, nmap=3)["spiral"]
rows = scan_sizes(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, [[n, 1, 1] for n in range(4, 16)],
                  restarts=3, J=[-1.0, 0.4], spin=1.0, rmax=6.1)
ns = [r_["size"][0] for r_ in rows]; Es = [r_["E_cell"] for r_ in rows]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(ns, Es, "o-", color="#16325c", label="supercell n × 1 × 1")
ax.axhline(exact["E_spiral"], color="#e67e22", ls="--", label=f"rotating frame (q = {abs(exact['q_rlu'][0]):.5f})")
ax.set_xlabel("supercell length n"); ax.set_ylabel("E (meV / cell)"); ax.legend()
best = min(rows, key=lambda r_: (round(r_["E_cell"], 9), r_["n_cells"]))
print("best n =", best["size"][0], " E =", round(best["E_cell"], 6), " exact", exact["E_spiral"])"""),
    md(r"""## 3. Conical spiral in a magnetic field

A field $\mathbf B\parallel\hat z$ cants the spins out of the spiral plane. Classically $\cos\theta = g\mu_B B / \bigl(2S(J(0)-J(Q))\bigr)$
— SymPhase reproduces it exactly, and identifies the state from the structure factor (uniform weight $w(0)=\cos^2\theta$)."""),
    code("""MUB = 0.05788381806
Bs = [0.5, 1, 2, 3, 4]
cs = [chain(field=[0, 0, B])["best"]["dirs"][:, 2].mean() for B in Bs]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(Bs, [2 * MUB * B / 0.5 for B in Bs], color="0.6", lw=3, label="analytic")
ax.plot(Bs, cs, "o", color="#16325c", label="SymPhase supercell")
ax.set_xlabel("B (T)"); ax.set_ylabel("cos θ"); ax.legend()
rc = chain(field=[0, 0, 2.0], spectrum=True, n_per=30, nmap=5)
print(rc["character"]["cls"], "  w(0) =", round(rc["character"]["w0"], 4), "  Goldstone modes:", rc["lswt"]["n_goldstone"])
plot_state(rc);"""),
    md(r"""Spin waves of the cone in the 6-cell supercell (bands are **folded**: 6 cells × 1 atom = 6 bands; one Goldstone mode = rotation about $\mathbf B$):"""),
    code("""plot_bands(dict(rc["lswt"], spiral=None), title="conical spiral, B = 2 T (folded bands)");"""),
    md(r"""## 4. Spiral distorted by an in-plane easy axis

An easy axis $K$ along $x$ (inside the spiral plane) breaks the rotation symmetry: the spins bunch near $\pm x$, the energy drops below the
undistorted spiral ($-0.75-KS^2/2$), and odd harmonics appear in the structure factor."""),
    code("""Ks = [0.1, 0.2, 0.4, 0.6]
res = [chain(size=[6, 1, 1], K=K, K_axis=[1, 0, 0]) for K in Ks]
for K, r_ in zip(Ks, res):
    h3 = sum(w for _, w, n in r_["character"]["harmonics"] if abs(n) == 3)
    print(f"K = {K}:  E = {r_['E_cell']:.6f}  (undistorted {-0.75 - K / 2:.6f})   3q weight = {h3:.4f}   {r_['character']['cls']}")
plot_state(res[2]);"""),
    md(r"""Should the period change? Scan the size — for $K=0.4$ the 6-cell period stays lowest (lock-in):"""),
    code("""for r_ in scan_sizes(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, [[n, 1, 1] for n in range(4, 10)],
                    restarts=3, J=[-1.0, 0.5], spin=1.0, rmax=6.1, K=0.4, K_axis=[1, 0, 0]):
    print(r_["size"], round(r_["E_cell"], 6), r_["cls"])"""),
    md(r"""## 5. Multi-Q: a skyrmion crystal on the triangular lattice

Ferromagnetic $J=-1$ meV, interfacial DMI $\mathbf D_{ij}=D\,\hat z\times\hat r_{ij}$ with $D=1.8$ meV, field along $z$.
The LT star has **three** members at 120°, so the supercell must be $n\times n$ (a strip cannot hold a triple-Q state)."""),
    code("""r14 = skx(14, size="auto")
report(r14, ["Mn"])
plot_state(r14);"""),
    md(r"""The topological charge $N_{sk}=-3$ per 10×10 cell agrees with the geometry: $|Q|$ fixes a skyrmion lattice whose cell is 1/3 of the supercell.

### Field scan: cycloid → skyrmion crystal → ferromagnet"""),
    code("""Bs = [0, 4, 8, 12, 16, 20, 24, 28]
scan = [skx(B) for B in Bs]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
a1.plot(Bs, [abs(r_["Q_top"]) for r_ in scan], "o-", color="#c0392b"); a1.set_xlabel("B (T)"); a1.set_ylabel("|N$_{sk}$| per supercell")
a2.plot(Bs, [r_["character"]["w0"] for r_ in scan], "s-", color="#16325c"); a2.set_xlabel("B (T)"); a2.set_ylabel("uniform weight w(0)")
for B, r_ in zip(Bs, scan):
    print(f"B = {B:4.0f} T   E = {r_['E_cell']:9.5f} meV/cell   {r_['character']['cls']}")"""),
    md(r"""Near the upper boundary a 10×10 cell may show a state with fewer skyrmions — a **finite-cell effect**. Always compare the energy
per cell in other sizes before claiming a new phase (step 6 of the guide).

### Competing states and stability"""),
    code("""for s_ in r14["seeds"][::8]:
    print(f"{s_['E_cell']:10.5f} meV/cell   {s_['seed']}")
r14s = skx(14, spectrum=True, n_per=8, nmap=5)
print("\\nspin waves:", "UNSTABLE" if r14s["lswt"]["unstable"] else "stable", "; bands:", r14s["lswt"]["omega"].shape[1])"""),
    md(r"""## 6. Files for DFT and the symmetry of the new state

`write_outputs` (CLI: automatic, folder `supercell/`) writes `POSCAR`, a non-collinear `INCAR` (MAGMOM of the ground state),
`J_super.dat`, `spins.dat`, `seeds.dat`. The supercell state can then go through the full symmetry analysis:"""),
    code("""import tempfile
from symphase import analyze_tensors
out = write_outputs(r14, ["Mn"], np.array([[0, 0, 2.0]]), tempfile.mkdtemp())
print(sorted(os.listdir(out)))
t = analyze_tensors(out, tensors="charge", quiet=True)
mg = t["magnetic"]
print("magnetic space group:", mg["bns_symbol"], "(BNS", mg["bns_number"] + ")  MPG", mg["magnetic_point_group"],
      "  T:", mg["time_reversal"], "  PT:", mg["PT"])"""),
    md("""The T-odd antisymmetric conductivity (σ_xy, the Hall response of the skyrmion crystal) is symmetry-allowed in P6m'm'.

## 7. Command line

```bash
cd examples/vasp/Conical_J1J2_field  && symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --field 0 0 2 --supercell auto
cd examples/vasp/Distorted_spiral_K  && symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --K 0.4 --K-axis 1 0 0 --scan 4 13 --supercell auto
cd examples/vasp/Skyrmion_triangular && symphase magnons --exchange J_DMI.dat --rmax 3.1 --spin 1 --field 0 0 14 --supercell auto
symphase tensors -p supercell/POSCAR -i supercell/INCAR
symphase magnons ... --supercell 7 1 1          # fixed size;  --supercell 2 1 0 -1 1 0 0 0 1  (matrix)
symphase magnons ... --restarts 30 --max-cells 600
```
Continue with **13_quiz_magnetic_supercells.ipynb**."""),
], "12_tutorial_magnetic_supercells.ipynb")

SC = [
    dict(type="mc", question="Which of these states can NOT be treated with the rotating frame and needs the supercell workflow?",
         options={"a": "a planar Heisenberg spiral", "b": "a conical spiral in a field along the spiral axis",
                  "c": "a DMI helix with D ∥ n", "d": "a collinear antiferromagnet"},
         answer="b", explanation="The field adds a uniform component: the state is not a planar spiral, so the planar rotating-frame ansatz fails."),
    dict(type="mc", question="Why does an easy axis lying in the spiral plane force a supercell?",
         options={"a": "it makes the spins longer", "b": "it breaks invariance under rotations about n, so R_n^T A R_n ≠ A and the spiral distorts (harmonics)",
                  "c": "it removes all magnons", "d": "it does not"},
         answer="b", explanation="Only anisotropy along n keeps the Hamiltonian rotation invariant; in-plane anisotropy bunches the spins (3q, 5q harmonics)."),
    dict(type="mc", question="What does the Luttinger–Tisza (LT) scan return?",
         options={"a": "the exact ground state in every case", "b": "the q minimising the lowest eigenvalue of J(q): the ordering vector and a lower bound of the energy",
                  "c": "the magnon gap", "d": "the magnetic space group"},
         answer="b", explanation="LT minimises the Fourier-transformed coupling matrix with the spin-length constraint relaxed: a lower bound, exact for simple cases."),
    dict(type="mc", question="What is the 'star' of q_LT used for?",
         options={"a": "plotting", "b": "the set of symmetry-equivalent wave vectors that a multi-Q state can combine; all must fit the supercell",
                  "c": "the k-mesh of VASP", "d": "nothing"},
         answer="b", explanation="Crystal rotations map q_LT to equivalent q's; triple-Q states use the three members of a hexagonal star."),
    dict(type="mc", question="Why does SymPhase keep n × n supercells when the star has several members (triangular lattice)?",
         options={"a": "it is faster", "b": "a strip n1 ≠ n2 cannot contain all three Q's at 120°, so no triple-Q (skyrmion) state could form",
                  "c": "VASP requires it", "d": "for nicer plots"},
         answer="b", explanation="Without equal sizes only one star member is commensurate; the multi-Q state is excluded by construction."),
    dict(type="num", question="Hands-on: the J1–J2 chain with J1 = −1, J2 = 0.5 meV: how many cells does the automatic supercell have? (chain()['M'])",
         answer=6, tol=0.1, hint="np.diag(chain()['M'])",
         explanation="q = 1/6 exactly, so the 6-cell supercell has zero LT penalty."),
    dict(type="num", question="Hands-on: J2 = 0.4 (q_exact = 0.14255). Which supercell length n between 4 and 13 gives the lowest energy per cell?",
         answer=7, tol=0.1, hint="scan_sizes(..., [[n, 1, 1] for n in range(4, 14)], J=[-1.0, 0.4], ...)",
         explanation="1/7 = 0.1429 is the closest m/n to 0.14255 for n ≤ 13; the energy is not monotonic in n."),
    dict(type="num", question="Hands-on: energy per cell (meV) of the best J2 = 0.4 approximant (4 decimals)?",
         answer=-0.7125, tol=2e-4, unit="meV",
         explanation="−0.712498 meV, only 2 × 10⁻⁶ meV above the exact rotating-frame value −0.7125."),
    dict(type="num", question="Hands-on: cone angle: cos θ of the J1–J2 chain (S = 1, g = 2) in B = 2 T along z (3 decimals)?",
         answer=0.463, tol=2e-3, hint="chain(field=[0, 0, 2])['best']['dirs'][:, 2].mean()",
         explanation="cos θ = gμ_B B / (2S(J(0) − J(Q))) = 0.2315 / 0.5 = 0.46307."),
    dict(type="num", question="Hands-on: for that cone, what is the uniform structure-factor weight w(0) (3 decimals)?",
         answer=0.214, tol=2e-3, hint="r['character']['w0']",
         explanation="w(0) = cos²θ = 0.2144; the remaining 0.7856 sits at ±Q."),
    dict(type="num", question="Hands-on: how many zero (Goldstone) modes at Γ does the cone have in its supercell?",
         answer=1, tol=0.1, hint="chain(field=[0,0,2], spectrum=True, n_per=10, nmap=3)['lswt']['n_goldstone']",
         explanation="Only rotations about the field direction remain a continuous symmetry."),
    dict(type="mc", question="Hands-on: which harmonic appears for K = 0.4 meV along x in the 6-cell J1–J2 spiral?",
         options={"a": "2q", "b": "3q", "c": "none", "d": "q/2"},
         answer="b", explanation="A uniaxial anisotropy ∝ cos²φ couples φ to 2φ, producing odd harmonics; 3q is the first (weight ≈ 0.015)."),
    dict(type="mc", question="Why does SymPhase relax from many seeds (LT single-Q, conical, multi-Q, random)?",
         options={"a": "to use more cores", "b": "the energy landscape has several local minima; the lowest of many starts is the ground-state candidate",
                  "c": "to make the plots smoother", "d": "seeds change the Hamiltonian"},
         answer="b", explanation="Different phases (cycloid, cone, skyrmion crystal, FM) are separate minima; reaching the lowest from several seeds is the check."),
    dict(type="num", question="Hands-on: how many skyrmions does the 10 × 10 triangular supercell contain at B = 14 T (|N_sk|)?",
         answer=3, tol=0.1, hint="abs(skx(14)['Q_top'])",
         explanation="N_sk = −3: the skyrmion-lattice cell is one third of the supercell, fixed by |Q|."),
    dict(type="num", question="Hands-on: how many members has the star of q_LT for the triangular skyrmion model (±q identified)?",
         answer=3, tol=0.1, hint="len(skx(14)['star'])",
         explanation="Three wave vectors at 120°: the triple-Q skyrmion crystal combines all of them."),
    dict(type="mc", question="Hands-on: sequence of ground states of the skyrmion model with increasing field (10×10 cell)?",
         options={"a": "FM → skyrmion crystal → cycloid", "b": "cycloid (distorted) → skyrmion crystal → field-polarised FM",
                  "c": "skyrmion crystal at all fields", "d": "cycloid → FM → skyrmion crystal"},
         answer="b", explanation="0–6 T: (distorted) cycloid; ≈ 8–20 T: skyrmion crystal (N_sk = −3); ≥ 24 T: ferromagnet."),
    dict(type="mc", question="Hands-on: the magnetic space group of the skyrmion crystal (symphase tensors on supercell/POSCAR + INCAR)?",
         options={"a": "P6/mmm1'", "b": "P6m'm' (BNS 183.189)", "c": "P-1", "d": "P6_3/mmc"},
         answer="b", explanation="Polar 6m'm': no T, no PT; a T-odd antisymmetric σ_xy (Hall response) is allowed."),
    dict(type="mc", question="How is the lattice topological charge computed?",
         options={"a": "from the net moment", "b": "sum of solid angles Ω of the spin triangles of every lattice triangle, divided by 4π (Berg–Lüscher)",
                  "c": "from the magnon gap", "d": "from the space group number"},
         answer="b", explanation="Ω = 2 atan2(e1·(e2×e3), 1 + e1·e2 + e2·e3 + e3·e1); the sum is an integer for a smooth texture."),
    dict(type="mc", question="A supercell of N primitive cells with one magnetic atom per cell gives how many magnon bands?",
         options={"a": "1", "b": "N (folded into the smaller supercell Brillouin zone)", "c": "2N", "d": "depends on the field"},
         answer="b", explanation="Bands are folded: N sites → N bosonic modes per k of the supercell zone."),
    dict(type="mc", question="Which VASP setup checks a predicted non-collinear supercell texture?",
         options={"a": "vasp_std with ISPIN = 2 and collinear MAGMOM", "b": "vasp_ncl with LNONCOLLINEAR, MAGMOM from supercell/INCAR, constrained moments, LSORBIT if DMI matters",
                  "c": "vasp_gam with ISPIN = 1", "d": "no DFT check is possible"},
         answer="b", explanation="Then compare total energies with competing states (seeds.dat) in the same supercell."),
]
build_bank(os.path.join(DATA, "quiz_supercells.json"), "Quiz 5 — magnetic supercells: multi-Q, conical and distorted spirals", SC)
quiz_nb("13_quiz_magnetic_supercells.ipynb", "Quiz 5 — magnetic supercells: multi-Q, conical and distorted spirals (20 questions)",
        SC, "supercells", "Work through **12_tutorial_magnetic_supercells.ipynb** and `docs/SUPERCELL_WORKFLOW.md` first. "
        "Questions 6–11 and 12–17 are hands-on.", SC_HELPER)
