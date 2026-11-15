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


# =============================================================================
# 14  OSSG numbers + magnon–phonon coupling;  15  QHA + anharmonic;  16 quiz
# =============================================================================
OL_HELPER = 'import numpy as np, spglib, math, copy, warnings\nimport matplotlib.pyplot as plt\nwarnings.simplefilter("ignore")\nfrom symphase import analyze_tensors\nfrom symphase.tensors import read_poscar\nfrom symphase.magphon import analyse_mp, bond_springs, read_magnetoelastic, plot_mp\nfrom symphase.qha import QHA, plot_qha, EV_A3_TO_GPA\nfrom symphase.thermo import FiniteTHull, KB\nfrom symphase.hull import load_csv\na = 3.0\nlat, fr = np.diag([a, 12, 12]), np.zeros((1, 3))\nds = spglib.get_symmetry_dataset((lat, fr, [0])); m0 = np.array([[0, 0, 2.0]])\nmpdir = f"{EX}/vasp/MagnonPhonon_FM_chain"\ndef fm_chain(lam=6.0, dJdr=None, J=-5.0, K=0.3, n_per=120):\n    me = None\n    if lam:\n        G = np.zeros((3, 3)); G[0, 2] = G[2, 0] = lam\n        me = {(0, 0, (1, 0, 0)): [(0, G)]}\n    return analyse_mp(lat, fr, [0], ["Fe"], m0, ds.rotations, ds.translations, springs=[(0, 0, [1, 0, 0], 600.0, 300.0)],\n                      magnetoelastic=me, dJdr=dJdr, J=[J], K=K, K_axis=[0, 0, 1], spin=1.0, rmax=3.1, n_per=n_per)\ntri = f"{EX}/vasp/Triangular_120AFM_INCAR"\ndef triangular(dJdr=1.5):\n    t = analyze_tensors(tri, tensors="charge", ossg="no")\n    p = read_poscar(f"{tri}/POSCAR"); syms = t["structure"]["symbols"]; ty = [0] * len(syms)\n    d3 = spglib.get_symmetry_dataset((p["lattice"], p["frac"], ty))\n    return analyse_mp(p["lattice"], p["frac"], ty, syms, np.array(t["structure"]["moments"]), d3.rotations, d3.translations,\n                      springs=bond_springs(p["lattice"], p["frac"], ty, 3.6, 400.0, 150.0), dJdr=[dJdr], J=[1.0],\n                      spin=1.5, rmax=3.6, n_per=40)\nni = QHA.debye_gruneisen(0.0, 10.90, 186.0, 4.6, 450.0, 1.95)          # toy fcc Ni (Debye–Grüneisen)\nqcsv = f"{EX}/hull/NiAl_qha.csv"\n'

save([
    md(r"""# SymPhase tutorial 14 — spin-space-group numbers for any order, and magnon–phonon coupling

**Part A.** Every magnetic order — collinear, coplanar or non-coplanar — gets the **oriented spin-space-group (OSSG) number** of
X. Chen *et al.*, Phys. Rev. X **14**, 031038 (2024). SymPhase calls their FindSpinGroup package (`pip install findspingroup`)
and cross-checks the magnetic space group it derives against its own spglib result.

**Part B.** Magnons and phonons as **one harmonic system**: spin deviations $(X_a, Y_a)$ and atomic displacements/momenta
$(u_{j\alpha}, P_{j\alpha})$ are canonical pairs; the bond-length dependence of the exchange,
$J_{ij}(\mathbf d_0+\delta\mathbf d)=J_{ij}+\sum_\gamma G^\gamma_{ij}\,\delta d_\gamma$, couples them. The Bloch Hessian is diagonalised
symplectically → hybrid "magnon-polaron" bands with a magnon weight per mode."""),
    SETUP,
    code(OL_HELPER),
    md("## A. OSSG numbers (Chen et al. 2024) for collinear and non-collinear orders"),
    code("""rows = []
for d in ["RuO2_Neel001_noSOC", "MnTe_altermagnet_INCAR", "Mn3Sn_allin", "Triangular_120AFM_INCAR", "CantedChain_DMI_INCAR"]:
    r = analyze_tensors(f"{EX}/vasp/{d}", tensors="charge")
    o = r["spin_group"]["ossg"]
    rows.append((d, o["index"], o["conf"], o["spin_space_point_group_hm"], o["msg_bns_number"], r["magnetic"]["bns_number"]))
print(f"{'example':<26s} {'OSSG index':<16s} {'order':<10s} {'spin PG':<8s} MSG(FSG)  MSG(SymPhase)")
for row in rows:
    print(f"{row[0]:<26s} {row[1]:<16s} {row[2]:<10s} {row[3]:<8s} {row[4]:<9s} {row[5]}")"""),
    md(r"""* The index starts with $G_0.L_0$: $G_0$ is the space group of the spatial parts and $L_0$ the spin-preserving subgroup — it coincides with
  SymPhase's own subgroup $H$ (RuO₂: Cmmm 65, MnTe: P-3m1 164). Then follow the enumeration indices and a suffix for the spin-only
  group (**L** collinear, **P** coplanar, none for non-coplanar). Full definitions: Chen *et al.*, PRX **14**, 031038 (2024).
* Collinear SSGs are also labelled by SymPhase's own scheme (BNS number of the $C_{2\perp}\to T$ black-and-white group, 1421 groups) —
  both are printed by `symphase tensors`.
* The magnetic space group that FindSpinGroup derives for the SOC case agrees with SymPhase's spglib result in every case: an independent check.

The same works for the non-coplanar **skyrmion crystal** of tutorial 12 (written by the supercell workflow):"""),
    code("""from symphase.supercell import run_supercell, write_outputs
from symphase.magnons import read_exchange_file
import tempfile
sk = f"{EX}/vasp/Skyrmion_triangular"; ps = read_poscar(f"{sk}/POSCAR")
dss = spglib.get_symmetry_dataset((ps["lattice"], ps["frac"], [0]))
rsk = run_supercell(ps["lattice"], ps["frac"], [0], ["Mn"], np.array([[0, 0, 2.0]]), dss.rotations, dss.translations,
                    size=[10, 10, 1], restarts=2, spectrum=False, exchange=read_exchange_file(f"{sk}/J_DMI.dat"), spin=1.0,
                    rmax=3.1, field=[0, 0, 14])
out = write_outputs(rsk, ["Mn"], np.array([[0, 0, 2.0]]), tempfile.mkdtemp())
o = analyze_tensors(out, tensors="charge")["spin_group"]["ossg"]
print("skyrmion crystal:", rsk["character"]["cls"], "->  OSSG", o["index"], o["conf"], " MSG", o["msg_symbol"], o["msg_bns_number"])"""),
    md(r"""## B. Magnon–phonon coupling

### B1. Ferromagnetic chain with a magnetoelastic (pseudo-dipolar) term

$J=-5$ meV, easy axis $K=0.3$ meV, $S=1$, Fe masses, springs $k_L=600$, $k_T=300$ meV/Å². The derivative of the bond tensor along the chain
has a symmetric $xz$ element $\lambda=6$ meV/Å (`examples/vasp/MagnonPhonon_FM_chain/magnetoelastic.dat`): a longitudinal displacement tilts
the easy axis → linear coupling to the **longitudinal** acoustic phonon. Colour = magnon weight; dashed = uncoupled."""),
    code("""r = fm_chain()
plot_mp(r, title=f"FM chain: magnon polarons, gap {r['max_gap']:.3f} meV");
for g in r["gaps"]:
    print(f"anticrossing at k = {g['x']:.3f} 1/Å, ω ≈ {g['omega']:.3f} meV, gap = {g['gap']:.4f} meV")"""),
    md(r"""**Analytic check.** One magnon ($\omega_m$) coupled to one phonon ($\omega_p$, $\kappa=\hbar^2/M$) through $H_c = g\,X u$ obeys
$(\omega^2-\omega_m^2)(\omega^2-\omega_p^2)=\omega_m\kappa g^2$ with $g = 2\lambda S^{3/2}|\sin ka|$ here:"""),
    code("""from symphase.magphon import HB2_AMU
k = r["q"][:, 0]; S, J, K, lam = 1.0, -5.0, 0.3, 6.0
wm = 2 * S * abs(J) * (1 - np.cos(k * a)) + 2 * K * S
kap = HB2_AMU / 55.845; wL = np.sqrt(kap * 1200 * (1 - np.cos(k * a))); wT = np.sqrt(kap * 600 * (1 - np.cos(k * a)))
g = 2 * lam * S ** 1.5 * np.abs(np.sin(k * a)); B = wm**2 + wL**2; C = wm**2 * wL**2 - wm * kap * g**2
ref = np.sort(np.c_[np.sqrt((B + np.sqrt(B*B - 4*C)) / 2), np.sqrt(np.maximum((B - np.sqrt(B*B - 4*C)) / 2, 0)), wT, wT], axis=1)
print("max |SymPhase − analytic| =", np.abs(r["omega"][1:] - ref[1:]).max(), "meV")"""),
    md("The gap grows **linearly** with the magnetoelastic constant (first-order level repulsion):"),
    code("""lams = [2, 4, 6, 8]
gaps = [fm_chain(lam=l, n_per=200)["max_gap"] for l in lams]
fig, ax = plt.subplots(figsize=(4.6, 3.2)); ax.plot(lams, gaps, "o-", color="#16325c")
ax.set_xlabel("λ (meV/Å)"); ax.set_ylabel("anticrossing gap (meV)")
print(dict(zip(lams, np.round(gaps, 4))))"""),
    md(r"""### B2. Exchange striction needs non-collinear spins

Isotropic striction $G^\gamma = (\mathrm dJ/\mathrm dr)\,\hat r_\gamma\,\mathbb 1$ enters as $x_a\cdot v_b$ — zero for collinear order. The FM chain is
unchanged, the 120° triangular antiferromagnet couples:"""),
    code("""r0 = fm_chain(lam=0, dJdr=[3.0])
print("collinear FM chain, dJ/dr = 3 meV/Å:  max |ω − ω0| =", np.nanmax(np.abs(r0["omega"] - r0["omega0"])), "meV")
rt = triangular(dJdr=1.5)
print("120° triangular, dJ/dr = 1.5 meV/Å :  max |ω − ω0| =", round(float(np.nanmax(np.abs(rt["omega"] - rt["omega0"]))), 4),
      "meV;", rt["n_magnon"], "magnon +", rt["n_phonon"], "phonon branches")
plot_mp(rt, title="120° triangular AFM + exchange striction", ylim=(0, 12));"""),
    md(r"""### B3. Phonons from phonopy

`symphase magnons ... --phonons phonopy_params.yaml` uses DFT force constants (same primitive cell and atom order as the POSCAR).
SymPhase converts phonopy's dynamical matrix to $\Phi(\mathbf k)$ in its own phase convention — identical to the spring model when the
force constants are those springs (checked to $10^{-12}$ in `tests/`).

### B4. Command line
```bash
cd examples/vasp/MagnonPhonon_FM_chain
symphase magnons --J -5 --K 0.3 --K-axis 0 0 1 --spin 1 --rmax 3.1 --phonons springs.dat --magnetoelastic magnetoelastic.dat
symphase magnons --J 1.0 --rmax 3.6 --springs-nn 400 150 3.6 --dJdr 1.5           # 120° triangular (Triangular_120AFM_INCAR)
symphase magnons ... --phonons phonopy_params.yaml --magnetoelastic dJ.dat          # DFT phonons
```"""),
], "14_tutorial_ossg_magnon_phonon.ipynb")

save([
    md(r"""# SymPhase tutorial 15 — quasi-harmonic and anharmonic free energies, and finite-T hulls

$$G(T,P=0)=\min_V\Big[E_\text{static}(V)+F_\text{vib}(V,T)+F_\text{anh}(V,T)\Big]$$

* $E_\text{static}$: Birch–Murnaghan EOS (fit to E–V data, or $E_0,V_0,B_0,B_0'$);
* $F_\text{vib}$: phonopy `thermal_properties.yaml` at several volumes (the phonopy-QHA workflow) **or** the Debye–Grüneisen model
  $\theta(V)=\theta_0\exp[\gamma_0(1-(V/V_0)^q)/q]$;
* $F_\text{anh}=-\tfrac32 k_B\,a\,T^2$ per atom (intrinsic anharmonicity, $a>0$ lowers $F$; $S_\text{anh}=3k_Ba T$), or a table $F_\text{anh}(T)$ from TDEP / thermodynamic integration.

Outputs: $V(T)$, volumetric $\beta=\mathrm d\ln V/\mathrm dT$ (linear $\alpha=\beta/3$), $B_T(T)$, $C_P(T)$ — and $G(T)$ for the finite-T convex hull."""),
    SETUP,
    code(OL_HELPER),
    md("## 1. Toy fcc Ni in the Debye–Grüneisen model ($V_0=10.9$ Å³, $B_0=186$ GPa, $\\theta_0=450$ K, $\\gamma_0=1.95$)"),
    code("""print(ni.summary())
plot_qha(ni, np.arange(0, 1201, 20), title="toy Ni: quasi-harmonic thermal expansion");
p = ni.properties([300, 600, 1000])
for T, V, b, al, B, cp in zip(p["T"], p["V"], p["beta"], p["alpha_lin"], p["B_T"], p["Cp"]):
    print(f"T = {T:5.0f} K  V = {V:.4f} Å³  β = {b*1e6:5.1f}e-6/K  α = {al*1e6:5.1f}e-6/K  B_T = {B:6.1f} GPa  C_P = {cp:.3f} k_B")
print("zero-point expansion: V(0 K) =", round(ni.equilibrium(0)[0], 4), "Å³ vs static V0 =", ni.V0)"""),
    md(r"""**Checks.** (i) $\gamma_0=0$ ⇒ $\theta$ independent of $V$ ⇒ no thermal expansion; (ii) the Grüneisen relation $\beta B_T V=\gamma C_V$;
(iii) the anharmonic term shifts $G$ by exactly $-\tfrac32k_BaT^2$."""),
    code("""q0 = QHA.debye_gruneisen(0.0, 10.9, 186.0, 4.6, 450.0, 0.0)
print("γ = 0: V(T) =", np.round(q0.properties([10, 600, 1200])["V"], 6))
p = ni.properties([800.0]); gam = 1.95 * p["V"][0] / ni.V0
print("Grüneisen ratio βBV/(γ·3k_B) at 800 K =", round(p["beta"][0] * p["B_T"][0] / EV_A3_TO_GPA * p["V"][0] / (gam * 3 * KB), 4),
      "(→ C_V/3k_B < 1)")
qa = QHA.debye_gruneisen(0.0, 10.9, 186.0, 4.6, 450.0, 0.0, anh_a=2e-5)
print("anharmonic shift at 1000 K:", round((qa.G(1000) - q0.G(1000)) * 1e3, 6), "meV;  −3/2 k_B a T² =",
      round(-1.5 * KB * 2e-5 * 1e6 * 1e3, 6), "meV")"""),
    md(r"""## 2. Phonopy workflow

Put `e-v.dat` (volume and energy per cell) and `thermal_properties-1.yaml … -N.yaml` (one phonopy run per volume) in one folder:
`QHA.from_phonopy_dir(folder, anh_a=...)`. The minimisation is identical to phonopy-qha — tested against `phonopy.PhonopyQHA`
($V$ to $2\times10^{-4}$ Å³, $G$ to $10^{-6}$ eV/atom). Here we build such a folder from the Debye–Grüneisen model and read it back:"""),
    code("""import tempfile, os
from symphase.qha import birch_murnaghan, debye_theta
from symphase.thermo import debye_free_energy
tmp = tempfile.mkdtemp(); nat = 4; EVK = 96.48533212
Vs = ni.V0 * np.linspace(0.94, 1.08, 9); Ts = np.arange(0, 1201, 20.0)
with open(f"{tmp}/e-v.dat", "w") as fh:
    for V in Vs: fh.write(f"{V*nat:.8f} {birch_murnaghan(V, 0.0, ni.V0, ni.B0, ni.Bp)*nat:.10f}\\n")
for k, V in enumerate(Vs):
    with open(f"{tmp}/thermal_properties-{k+1}.yaml", "w") as fh:
        fh.write(f"natom: {nat}\\nthermal_properties:\\n")
        for T in Ts:
            fh.write(f"- temperature: {T}\\n  free_energy: {debye_free_energy(T, debye_theta(V, ni.V0, 450.0, 1.95))*nat*EVK:.10f}\\n")
qp = QHA.from_phonopy_dir(tmp)
for T in (300, 900):
    print(f"T = {T} K:  V from tables {qp.equilibrium(T)[0]:.5f}   analytic model {ni.equilibrium(T)[0]:.5f}  Å³/atom")"""),
    md(r"""## 3. Finite-temperature convex hull: harmonic → quasi-harmonic → anharmonic

`examples/hull/NiAl_qha.csv` (toy data) adds `volume, bulk_modulus, bulk_modulus_prime, gruneisen` and `anh_a` columns. The metastable
polymorph Ni₃Al-D0₁₉ has a slightly lower θ_D, a **larger Grüneisen parameter** and some intrinsic anharmonicity — it softens more on heating."""),
    code("""ents = load_csv(qcsv)
noanh = copy.deepcopy(ents)
for e in noanh: e.data.pop("anh_a", None)
models = [("harmonic Debye", noanh, "debye"), ("quasi-harmonic", noanh, "qha-debye"), ("QHA + anharmonic", ents, "qha-debye")]
Tg = np.arange(0, 1601, 50)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for (lab, E, vib), c in zip(models, ["0.5", "#16325c", "#c0392b"]):
    ft = FiniteTHull(E, vib=vib, mixing=False)
    eh = [{r_["name"]: r_["e_hull"] for r_ in ft.hull_at(T).table()}["Ni3Al-D019"] * 1e3 for T in Tg]
    Ts = ft.stabilisation_temperature("Ni3Al-D019", Tmax=3000)
    ax.plot(Tg, eh, lw=2.2, color=c, label=f"{lab}: T* = {Ts:.0f} K")
    print(f"{lab:18s}  T* = {Ts:7.1f} K")
ax.set_xlabel("T (K)"); ax.set_ylabel("E$_{hull}$(Ni$_3$Al-D0$_{19}$) (meV/atom)"); ax.legend(fontsize=9); ax.set_ylim(bottom=-0.5)"""),
    md(r"""Volume-dependent phonons (QHA) almost halve the transition temperature here; intrinsic anharmonicity lowers it further.

## 4. Command line
```bash
symphase hull --csv examples/hull/NiAl_qha.csv --temperatures 0 600 1200 --vib qha-debye --no-mixing
symphase hull --csv my.csv --temperatures 300 900 --vib qha        # CSV column qha = folder with e-v.dat + thermal_properties-*.yaml
symphase run Ni Al temperatures="300 900" vib=qha-debye             # elements-only workflow (with the extra CSV columns)
```
Continue with **16_quiz_ossg_magnon_phonon_qha.ipynb**."""),
], "15_tutorial_qha_anharmonic.ipynb")

OL = [
    dict(type="mc", question="What does the OSSG number of Chen et al. (2024) add to SymPhase's own collinear SSG label?",
         options={"a": "nothing", "b": "a standard number for every order, including coplanar and non-coplanar ones",
                  "c": "the magnon gap", "d": "only the crystal space group"},
         answer="b", explanation="SymPhase numbers collinear SSGs via the C2⊥→T black-and-white group; FindSpinGroup gives the four-index OSSG for any order."),
    dict(type="mc", question="Hands-on: the OSSG index of Mn3Sn (all-in coplanar 120° order, examples/vasp/Mn3Sn_allin) is …",
         options={"a": "194.164.1.1.L", "b": "194.6.1.1.P", "c": "191.187.3.1.P", "d": "136.65.1.1.L"},
         answer="b", explanation="Coplanar ('P' suffix), G0 = P6_3/mmc (194), spin point group 6/mmm."),
    dict(type="mc", question="Hands-on: the OSSG index of the altermagnet MnTe is …",
         options={"a": "194.164.1.1.L", "b": "194.6.1.1.P", "c": "229.229.1.1.L", "d": "191.6.1.2"},
         answer="a", explanation="Collinear ('L'), spin point group ∞/mm; the same index appears in the FindSpinGroup documentation."),
    dict(type="mc", question="Hands-on: which suffix does the non-coplanar skyrmion crystal of tutorial 12 get (OSSG 191.6.1.2)?",
         options={"a": "L", "b": "P", "c": "none (non-coplanar)", "d": "P2"},
         answer="c", explanation="Collinear orders end in L, coplanar in P; a non-coplanar order has a trivial spin-only group and no suffix."),
    dict(type="mc", question="Why does SymPhase compare the MSG derived by FindSpinGroup with its own spglib result?",
         options={"a": "to pick the larger group", "b": "as an independent consistency check of moments, tolerances and frames",
                  "c": "because spglib cannot find MSGs", "d": "it does not"},
         answer="b", explanation="Two independent codes must agree on the SOC magnetic group; a mismatch flags tolerance or input problems."),
    dict(type="mc", question="Which canonical variables describe one spin in the joint magnon–phonon model?",
         options={"a": "S_x and S_z", "b": "(X, Y) with S⊥ = √S (X x + Y y) and [X, Y] = i", "c": "the angles θ and φ only", "d": "u and P"},
         answer="b", explanation="Holstein–Primakoff to linear order: transverse deviations scaled by √S are a canonical pair, like (u, P) for atoms."),
    dict(type="mc", question="Why does isotropic exchange striction (G = dJ/dr r̂ 𝟙) not hybridise magnons and phonons in a collinear magnet?",
         options={"a": "the phonons are too fast", "b": "the linear coupling ∝ x_a·v_b vanishes when all spins are (anti)parallel",
                  "c": "dJ/dr is always zero", "d": "it does"},
         answer="b", explanation="Only non-collinear order (x_a·v_b ≠ 0) or anisotropic/DMI derivatives give terms linear in both u and the magnon."),
    dict(type="num", question="Hands-on: FM chain of tutorial 14 (λ = 6 meV/Å): largest anticrossing gap in meV (2 decimals)?",
         answer=1.11, tol=0.02, unit="meV", hint="fm_chain()['max_gap']",
         explanation="≈ 1.11 meV where the magnon crosses the longitudinal acoustic phonon."),
    dict(type="mc", question="Hands-on: how does the anticrossing gap scale with the magnetoelastic constant λ?",
         options={"a": "∝ λ² ", "b": "linearly ∝ λ", "c": "independent of λ", "d": "∝ √λ"},
         answer="b", explanation="0.37, 0.74, 1.11, 1.48 meV for λ = 2, 4, 6, 8 meV/Å: first-order level repulsion, gap ≈ 2|coupling|."),
    dict(type="mc", question="In the chain example, which phonon branch hybridises with the magnon?",
         options={"a": "both transverse acoustic", "b": "the longitudinal acoustic phonon", "c": "none", "d": "an optical phonon"},
         answer="b", explanation="Only ∂J/∂d_x (along the chain) is non-zero, so only longitudinal displacements u_x couple."),
    dict(type="num", question="Hands-on: 120° triangular AFM with dJ/dr = 1.5 meV/Å: max |ω − ω_uncoupled| in meV (3 decimals)?",
         answer=0.033, tol=0.003, unit="meV", hint="triangular(1.5)",
         explanation="≈ 0.033 meV: non-collinear spins make isotropic striction couple linearly."),
    dict(type="mc", question="What must phonopy force constants satisfy to be used with --phonons phonopy_params.yaml?",
         options={"a": "nothing", "b": "same primitive cell and atom order as the POSCAR (SymPhase checks it)", "c": "a 1x1x1 supercell", "d": "no magnetism"},
         answer="b", explanation="Φ(k) is mapped atom by atom onto the magnetic model; SymPhase refuses a different cell or order."),
    dict(type="mc", question="How is the magnon weight of a hybrid mode defined?",
         options={"a": "fraction of the energy in spins", "b": "fraction of the symplectic (boson) norm carried by the spin pairs (X, Y)",
                  "c": "1 if ω < 5 meV", "d": "random"},
         answer="b", explanation="For each canonical pair c = −2 Im(q* p); the spin pairs' share of Σc is the magnon weight (1 = pure magnon)."),
    dict(type="mc", question="What does the quasi-harmonic approximation compute?",
         options={"a": "F at the static volume only", "b": "G(T) = min_V [E(V) + F_vib(V, T)] with volume-dependent phonons",
                  "c": "only the zero-point energy", "d": "anharmonic phonon lifetimes"},
         answer="b", explanation="Phonon frequencies depend on V; minimising F over V gives V(T), thermal expansion and G(T)."),
    dict(type="num", question="Hands-on: toy Ni (tutorial 15): volumetric thermal expansion β at 300 K in 10⁻⁶/K (1 decimal)?",
         answer=38.0, tol=1.0, unit="1e-6/K", hint="ni.properties([300])['beta'] * 1e6",
         explanation="β ≈ 38 × 10⁻⁶ /K, i.e. a linear α = β/3 ≈ 12.7 × 10⁻⁶ /K — the right size for Ni."),
    dict(type="num", question="Hands-on: the corresponding LINEAR thermal expansion α at 300 K in 10⁻⁶/K?",
         answer=12.7, tol=0.4, unit="1e-6/K", hint="ni.properties([300])['alpha_lin'] * 1e6",
         explanation="α = β/3 for an isotropic solid."),
    dict(type="mc", question="With a Grüneisen parameter γ0 = 0 the quasi-harmonic model gives …",
         options={"a": "large expansion", "b": "no thermal expansion: θ_D does not depend on V", "c": "negative expansion", "d": "melting"},
         answer="b", explanation="Expansion needs ∂F_vib/∂V ≠ 0, i.e. volume-dependent phonons (β B V = γ C_V)."),
    dict(type="num", question="Hands-on: NiAl_qha.csv, harmonic Debye (anh_a removed): stabilisation temperature of Ni3Al-D019 in K?",
         answer=1393, tol=10, unit="K", hint="FiniteTHull(noanh, vib='debye', mixing=False).stabilisation_temperature('Ni3Al-D019', Tmax=3000)",
         explanation="1393 K harmonically; QHA lowers it to 742 K."),
    dict(type="num", question="Hands-on: with quasi-harmonic + anharmonic free energies (vib='qha-debye', anh_a kept): T* of Ni3Al-D019 in K?",
         answer=667, tol=10, unit="K",
         explanation="667 K: the larger Grüneisen parameter and the intrinsic anharmonicity of D019 soften it on heating."),
    dict(type="mc", question="In SymPhase's convention F_anh = −(3/2) k_B a T² per atom; what is the anharmonic entropy?",
         options={"a": "zero", "b": "S_anh = 3 k_B a T", "c": "S_anh = −3 k_B a T", "d": "S_anh = k_B ln a"},
         answer="b", explanation="S = −∂F/∂T = 3 k_B a T, and C_anh = T ∂S/∂T = 3 k_B a T: a > 0 raises the heat capacity linearly in T."),
]
build_bank(os.path.join(DATA, "quiz_magphon_qha.json"),
           "Quiz 6 — spin-space-group numbers, magnon–phonon coupling, quasi-harmonic & anharmonic hulls", OL)
quiz_nb("16_quiz_ossg_magnon_phonon_qha.ipynb",
        "Quiz 6 — spin-space-group numbers, magnon–phonon coupling, quasi-harmonic & anharmonic hulls (20 questions)",
        OL, "magphon_qha", "Work through tutorials **14** and **15** first. Questions 2–4, 8, 9, 11, 15, 16, 18 and 19 are hands-on.",
        OL_HELPER + "\nents = load_csv(qcsv); noanh = copy.deepcopy(ents)\nfor e in noanh: e.data.pop('anh_a', None)")


# =============================================================================
# 17  Kubo magnitudes;  18  self-consistent anharmonic phonons;  19 quiz
# =============================================================================
KS_HELPER = """import numpy as np, math, os, copy, warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
from symphase import analyze_tensors
from symphase.tensors import read_poscar
from symphase.kubo import (SymTensor, plan, KuboTB, compute_symmetric, format_fit, read_hr, read_centres,
                           read_postw90_ahc, read_postw90_shc, haldane_hr, kane_mele_hr, honeycomb_lattice)
from symphase.qha import quartic_oscillator, AnharmonicF, FreeEnergySurface, QHA
from symphase.thermo import FiniteTHull
from symphase.hull import load_csv
KB = f"{EX}/kubo"
lat, frac = honeycomb_lattice(); cen = frac @ lat
def haldane(M=0.0, phi=math.pi / 2, efs=(0.0,), n=(60, 60)):
    R, H = haldane_hr(M=M, phi=phi)
    return KuboTB(lat, R, H, centres=cen, dims=2).conductivity([("charge", 0, 0, 1)], list(efs), n, 1e-6)[:, 0]
def kane_mele(efs=(0.0,), n=(60, 60)):
    R, H = kane_mele_hr()
    return KuboTB(lat, R, H, centres=np.r_[cen, cen], dims=2, spin_order="blocks").conductivity(
        [("spin", 2, 0, 1), ("charge", 0, 0, 1)], list(efs), n, 1e-6)
scph_csv = f"{EX}/hull/NiAl_scph.csv\""""
KS_HELPER = KS_HELPER.replace('NiAl_scph.csv\\"', 'NiAl_scph.csv"')

save([
    md(r"""# SymPhase tutorial 17 — magnitudes from Wannier-based Kubo codes, constrained by symmetry

Symmetry fixes the **shape** of a response tensor; the **magnitude** needs a Kubo calculation on a Wannier Hamiltonian.
`symphase kubo` links the two:

| step | what | command |
|---|---|---|
| plan | minimal set of components to compute (+ postw90 input blocks, a check component) | `symphase kubo plan --kind spin` |
| import | postw90 `*-ahc-fermiscan.dat`, `*-shc-fermiscan.dat` (+ `.win`), any `component value` table | `symphase kubo import ...` |
| fit | least-squares fit of the independent parameters → full tensor, violation, forbidden non-zero components | automatic |
| compute | built-in reference Kubo solver on `seedname_hr.dat` (AHC, SHC) for the pivot components only | `symphase kubo compute ...` |

Conventions as in postw90: $\sigma_{ij}=-\frac{e^2}{\hbar}\int\!\sum_n f_n\Omega_{n,ij}$,
$\Omega_{n,ij}=-2\,\mathrm{Im}\sum_{m\ne n}\frac{\langle n|\partial_iH|m\rangle\langle m|\partial_jH|n\rangle}{(E_n-E_m)^2+\eta^2}$; spin: $\partial_iH\to\frac12\{\sigma_k,\partial_iH\}$
(units $\hbar/e$). $J^{S_k}_i=\sigma^k_{ij}E_j$ ↔ postw90 `shc_alpha=i, shc_beta=j, shc_gamma=k`."""),
    SETUP,
    code(KS_HELPER),
    md("## 1. Plan: what must postw90 compute for fcc Pt?"),
    code("""st = SymTensor.from_analysis(analyze_tensors(f"{KB}/Pt_postw90", tensors="spin", ossg="no"), "spin")
txt, blocks = plan(st)
print(txt); print("\\n--- first postw90 block ---\\n" + blocks[0])"""),
    md("One number is enough. Import a postw90 SHC Fermi scan of $\\sigma^z_{xy}$ (**toy numbers**) and let symmetry fill the tensor:"),
    code("""scan = read_postw90_shc(f"{KB}/Pt_postw90/Pt-shc-fermiscan.dat", win=f"{KB}/Pt_postw90/Pt.win")
fit = st.fit(scan[0.0])
print(format_fit(st, fit, unit="(ħ/e) S/cm"))
efs = sorted(scan); vals = [st.fit(scan[e])["full"][0 * 9 + 1 * 3 + 2] for e in efs]
fig, ax = plt.subplots(figsize=(4.8, 3)); ax.plot(efs, vals, "o-", color="#16325c")
ax.set_xlabel("E$_F$ (eV)"); ax.set_ylabel("σ$^x_{yz}$ ((ħ/e) S/cm)"); ax.set_title("filled by symmetry from σ$^z_{xy}$", fontsize=10)"""),
    md("## 2. Import with a symmetry check: anomalous Hall of bcc Fe ($\\mathbf m\\parallel z$)\n\nThe toy postw90 file contains small forbidden $\\sigma_{yz},\\sigma_{zx}$ (as from an unconverged k mesh or a wrong frame):"),
    code("""stc = SymTensor.from_analysis(analyze_tensors(f"{KB}/Fe_postw90", tensors="charge", ossg="no"), "charge", parts=("odd",))
f2 = stc.fit(read_postw90_ahc(f"{KB}/Fe_postw90/Fe-ahc-fermiscan.dat")[0.0])
print(format_fit(stc, f2, unit="S/cm"))"""),
    md(r"""## 3. Built-in Kubo solver — analytic checks

**Haldane model** (honeycomb, $t_1=1$, $t_2=0.1$, $\phi=\pi/2$): $|\sigma_{xy}|=e^2/h$ in the topological gap, 0 for a large sublattice mass $M$.
**Kane–Mele** ($S_z$ conserved): quantised spin Hall $|\sigma^z_{xy}|=(\hbar/e)\,e^2/h$, no charge Hall."""),
    code("""Ms = np.linspace(0, 1.0, 11)
sx = [abs(haldane(M=m)[0]) for m in Ms]
fig, ax = plt.subplots(figsize=(5, 3.2)); ax.plot(Ms, sx, "o-", color="#c0392b")
ax.axvline(3 * math.sqrt(3) * 0.1, ls=":", color="k"); ax.set_xlabel("M (eV)"); ax.set_ylabel("|σ$_{xy}$| (e²/h)")
ax.set_title("Haldane: Chern transition at M = 3√3 t₂", fontsize=10)
print("Kane–Mele at E_F = 0: σ^z_xy, σ_xy =", np.round(kane_mele()[0], 6))"""),
    md("Units in 3D: stacking Haldane layers with spacing $c=5$ Å must give $(e^2/h)/c$:"),
    code("""lat3 = lat.copy(); lat3[2, 2] = 5.0; R, H = haldane_hr()
s3 = KuboTB(lat3, R, H, centres=frac @ lat3, dims=3).conductivity([("charge", 0, 0, 1)], [0.0], (60, 60, 1), 1e-6)[0, 0]
print(f"σ_xy = {s3:.3f} S/cm;  (e²/h)/c = {3.874045865e-5 / 5e-8:.3f} S/cm")"""),
    md(r"""## 4. Symmetry-driven Kubo run on a Wannier Hamiltonian

`examples/kubo/Graphene_KaneMele` holds the model as a Wannier90 `kane_mele_hr.dat` + `centres.xyz`. SymPhase computes only the
**pivot** components of the allowed tensor (P6/mmm1′: 3 independent), fills the rest by symmetry, and verifies one **non-pivot**
component with an independent Kubo evaluation:"""),
    code("""d = f"{KB}/Graphene_KaneMele"
stg = SymTensor.from_analysis(analyze_tensors(d, tensors="spin", ossg="no"), "spin", parts=("even",))
R, H = read_hr(f"{d}/kane_mele_hr.dat")
model = KuboTB(read_poscar(f"{d}/POSCAR")["lattice"], R, H, centres=read_centres(f"{d}/centres.xyz"), dims=2, spin_order="blocks")
for fit in compute_symmetric(model, stg, [0.0, 0.5], n=(60, 60), eta=1e-6):
    print(f"E_F = {fit['E_F']} eV"); print(format_fit(stg, fit, unit="(ħ/e) e²/h")); print()"""),
    md("""## 5. Command line
```bash
symphase kubo plan    -d examples/kubo/Pt_postw90 --kind spin
symphase kubo import  -d examples/kubo/Pt_postw90 --kind spin --shc Pt_postw90/Pt-shc-fermiscan.dat --win Pt_postw90/Pt.win --ef 0
symphase kubo import  -d examples/kubo/Fe_postw90 --kind charge --part odd --ahc Fe_postw90/Fe-ahc-fermiscan.dat --ef 0
symphase kubo import  -d MyMaterial --kind orbital --table ohc_components.txt     # any code: 'σ^z_xy 1234.5' lines
symphase kubo compute -d examples/kubo/Graphene_KaneMele --kind spin --hr .../kane_mele_hr.dat --centres .../centres.xyz \\
                      --spin-order blocks --dims 2 --grid 60 60
```
The built-in solver uses the tight-binding convention (Wannier centres, no position-matrix `r.dat` terms) and the intrinsic
(Fermi-sea, Berry-type) formula: T-odd spin and T-even charge components need Fermi-surface (∝ 1/Γ) or Boltzmann codes."""),
], "17_tutorial_kubo_magnitudes.ipynb")

save([
    md(r"""# SymPhase tutorial 18 — self-consistent anharmonic phonons (TDEP / SSCHA) in free energies and hulls

Strongly anharmonic solids need **temperature-dependent effective phonons**. TDEP fits effective force constants to MD at each $T$;
the SSCHA minimises a variational free energy. Both give

$$F(T) = F_\text{ph}[\omega_\text{eff}(T)](T) + \Delta F(T),\qquad \Delta F = \langle V - V_\text{eff}\rangle\ \ (\text{TDEP: } U_0)$$

SymPhase reads them and uses them everywhere a free energy is needed:
* **TDEP folders**: `outfile.free_energy` (T, F_ph, S, C_V; eV/atom) + `outfile.U0` (**second value** = baseline to add), one folder per temperature;
* **tables** `T F` (SSCHA or any code) with units eV | meV | Ry | mRy | Ha | kJ/mol, per atom or per cell;
* `AnharmonicF` → smooth $F(T)$ and $S=-\partial F/\partial T$;  `FreeEnergySurface` → anharmonic QHA over volumes;
* hull: `--vib scph`, CSV columns `scph`, `e_static` (required when $F$ contains the static energy, as TDEP's $U_0$ and SSCHA totals do)."""),
    SETUP,
    code(KS_HELPER),
    md(r"""## 1. Why self-consistent phonons? A quartic oscillator, exactly

$V=\tfrac12kx^2+\lambda x^4$ ($k=2000$ meV/Å², $\lambda=1200$ meV/Å⁴, Fe mass). Exact quantum $F$ by diagonalisation vs the self-consistent
harmonic approximation (SCHA: $M\omega_\text{eff}^2=k+12\lambda\langle x^2\rangle$, $F=F_\text{ph}(\omega_\text{eff})+\langle V-V_\text{eff}\rangle$) vs plain harmonic:"""),
    code("""T = np.arange(0, 1601, 50)
q = quartic_oscillator(2000.0, 1200.0, 55.845, T)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.plot(T, q["scha"] - q["exact"], color="#16325c", lw=2, label="SCHA − exact")
a1.plot(T, q["harmonic"] - q["exact"], color="0.5", lw=2, ls="--", label="harmonic − exact")
a1.axhline(0, color="k", lw=0.6); a1.set_xlabel("T (K)"); a1.set_ylabel("ΔF (meV)"); a1.legend()
a2.plot(T, q["w_eff"], color="#c0392b", lw=2); a2.set_xlabel("T (K)"); a2.set_ylabel("ω$_{eff}$ (meV)")
a2.set_title("phonon hardening with temperature", fontsize=10)
k = int(np.argmin(abs(T - 1000)))
print(f"1000 K: SCHA error {q['scha'][k]-q['exact'][k]:+.3f} meV,  harmonic error {q['harmonic'][k]-q['exact'][k]:+.3f} meV;"
      f"  SCHA ≥ exact everywhere: {bool(np.all(q['scha'] >= q['exact'] - 1e-9))}  (Gibbs–Bogoliubov bound)")"""),
    md(r"""## 2. Reading TDEP folders

`examples/hull/scph/Ni3Al-D019/T####/` contains `outfile.free_energy` and `outfile.U0` for every temperature (toy SCHA data,
written in TDEP's layout). `AnharmonicF.from_tdep` adds the baseline and builds a smooth $F(T)$:"""),
    code("""F = AnharmonicF.from_tdep(f"{EX}/hull/scph/Ni3Al-D019", e_static=-5.52)
print("includes the static energy:", F.includes_static, " temperatures:", F.T.astype(int))
for t in (300, 750, 1300):
    print(f"T = {t:5d} K:  ΔF = F − E_static = {F.delta(t)*1e3:9.3f} meV/atom,  S = {F.entropy(t)/8.617333262e-5:.3f} k_B/atom")"""),
    md(r"""## 3. Anharmonic quasi-harmonic approximation over volumes

With total free energies at several volumes, `FreeEnergySurface` minimises over $V$ at every $T$. Check: built from the
Debye–Grüneisen model it reproduces that model's $V(T)$ and $G(T)$:"""),
    code("""qdg = QHA.debye_gruneisen(-5.0, 11.0, 180.0, 4.5, 450.0, 2.0)
Vs = 11.0 * np.linspace(0.95, 1.07, 9)
surf = FreeEnergySurface(Vs, [(lambda T, V=V: float(qdg.F(V, T))) for V in Vs])
for t in (300, 900):
    print(f"T = {t} K:  V surface {surf.equilibrium(t)[0]:.4f}  model {qdg.equilibrium(t)[0]:.4f} Å³;  "
          f"G {surf.G(t):.6f} / {qdg.G(t):.6f} eV")"""),
    md(r"""## 4. Finite-temperature hull with anharmonic free energies

`examples/hull/NiAl_scph.csv`: the two Ni₃Al polymorphs come from TDEP-style folders, all other phases from SSCHA-style tables.
D0₁₉ is harmonically softer (it would win entropy early) but also more anharmonic: its phonons **harden** on heating."""),
    code("""ft = FiniteTHull(load_csv(scph_csv), vib="scph", mixing=False)
Ts = ft.stabilisation_temperature("Ni3Al-D019", Tmax=1600)
Tg = np.arange(0, 1601, 1.0)
L, D = quartic_oscillator(2050, 500, 50, Tg), quartic_oscillator(1450, 1000, 50, Tg)
dH = 25 + 3 * (D["harmonic"] - L["harmonic"]); dS = 25 + 3 * (D["scha"] - L["scha"])
print(f"T* (SCHA, hull from TDEP folders): {Ts:.0f} K;   same model, harmonic phonons only: {Tg[np.argmax(dH <= 0)]:.0f} K")
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(Tg, dH, color="0.5", ls="--", lw=2, label="harmonic"); ax.plot(Tg, dS, color="#c0392b", lw=2, label="SCHA (anharmonic)")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("T (K)"); ax.set_ylabel("G(D0$_{19}$) − G(L1$_2$) (meV/atom)"); ax.legend()"""),
    md("""Anharmonic hardening delays the transition by ≈ 235 K here — the harmonic approximation would be badly wrong.

## 5. Command line
```bash
symphase hull --csv examples/hull/NiAl_scph.csv --temperatures 0 600 1200 --vib scph --no-mixing
```
CSV columns: `scph` (TDEP folder or table), `e_static` (eV/atom), `scph_unit`, `scph_per` (atom|cell), `scph_natom`, `scph_static`.
Continue with **19_quiz_kubo_anharmonic_phonons.ipynb**."""),
], "18_tutorial_selfconsistent_phonons.ipynb")

KQ = [
    dict(type="mc", question="What does symmetry alone NOT give for a response tensor?",
         options={"a": "which components vanish", "b": "relations between components", "c": "the magnitudes", "d": "T-even/T-odd character"},
         answer="c", explanation="Neumann's principle fixes the shape; magnitudes need a Kubo (or Boltzmann) calculation."),
    dict(type="num", question="Hands-on: how many components must postw90 compute to know the full spin Hall tensor of fcc Pt?",
         answer=1, tol=0.1, hint="SymTensor.from_analysis(...).n_par",
         explanation="Cubic m-3m1′: σ^x_yz = σ^y_zx = σ^z_xy = −σ^x_zy = ... — one independent number."),
    dict(type="mc", question="Which postw90 keywords select σ^z_xy (flow x, field y, spin z)?",
         options={"a": "shc_alpha=3, shc_beta=1, shc_gamma=2", "b": "shc_alpha=1, shc_beta=2, shc_gamma=3",
                  "c": "shc_alpha=2, shc_beta=1, shc_gamma=3", "d": "berry_task=ahc"},
         answer="b", explanation="J^{S_k}_i = σ^k_ij E_j ↔ shc_alpha = i (flow), shc_beta = j (field), shc_gamma = k (spin)."),
    dict(type="num", question="Hands-on: from the toy Pt scan, the value of σ^x_zy at E_F = 0 in (ħ/e) S/cm?",
         answer=-1985.3, tol=0.5, hint="st.fit(scan[0.0])['full'][0*9 + 2*3 + 1]",
         explanation="σ^x_zy = −σ^x_yz = −σ^z_xy = −1985.3 by cubic symmetry."),
    dict(type="num", question="Hands-on: fitted anomalous Hall conductivity σ_xy of the toy bcc Fe file at E_F = 0 (S/cm)?",
         answer=751.6, tol=0.5, explanation="Only h_z is allowed for m ∥ z; the fit keeps σ_xy and discards the forbidden parts."),
    dict(type="mc", question="Hands-on: what does SymPhase report about the toy Fe data?",
         options={"a": "nothing", "b": "non-zero components that symmetry forbids (σ_yz, σ_zx) and a 0.27 % violation",
                  "c": "that Fe is non-magnetic", "d": "a quantised Hall conductance"},
         answer="b", explanation="Forbidden non-zero components flag k-mesh convergence or frame problems in the Kubo run."),
    dict(type="mc", question="Why compute a 'check component' in a symmetry-driven Kubo run?",
         options={"a": "for speed", "b": "an independent Kubo evaluation of a component predicted by symmetry verifies frame, spin order and convergence",
                  "c": "postw90 requires it", "d": "it replaces the pivots"},
         answer="b", explanation="SymPhase picks the non-pivot component with the largest predicted value and compares."),
    dict(type="num", question="Hands-on: |σ_xy| of the Haldane model (t2 = 0.1, φ = π/2, M = 0) in the gap, in e²/h?",
         answer=1, tol=1e-3, hint="haldane()", explanation="Chern number ±1: quantised anomalous Hall conductance."),
    dict(type="mc", question="Hands-on: Haldane model with M = 1.0 eV (> 3√3 t2): σ_xy in the gap is …",
         options={"a": "1 e²/h", "b": "0 (trivial insulator)", "c": "2 e²/h", "d": "undefined"},
         answer="b", explanation="The gap closes at M = 3√3 t2 ≈ 0.52 eV; beyond it the Chern number is 0."),
    dict(type="num", question="Hands-on: Kane–Mele spin Hall conductivity σ^z_xy at E_F = 0.5 eV (in the band) in (ħ/e) e²/h (3 decimals)?",
         answer=0.591, tol=0.003, hint="kane_mele(efs=[0.5])",
         explanation="0.591: outside the gap the spin Hall conductivity is no longer quantised."),
    dict(type="num", question="Hands-on: 3D stack of Haldane layers, c = 5 Å: σ_xy in S/cm (integer)?",
         answer=775, tol=1, explanation="(e²/h)/c = 3.874×10⁻⁵ S / 5×10⁻⁸ cm = 774.8 S/cm — the unit check of the solver."),
    dict(type="mc", question="Which contributions does the built-in intrinsic Kubo solver NOT include?",
         options={"a": "Berry curvature of occupied bands", "b": "Fermi-surface (∝ 1/Γ) terms such as T-odd spin conductivities, and r.dat position terms",
                  "c": "spin Hall of Kane–Mele", "d": "the anomalous Hall effect"},
         answer="b", explanation="It is the Fermi-sea (Berry-type) formula in the tight-binding convention."),
    dict(type="mc", question="In TDEP, what must be added to the phonon free energy F_ph of outfile.free_energy?",
         options={"a": "nothing", "b": "the renormalised baseline U0 (second value of outfile.U0)", "c": "the zero-point energy twice", "d": "k_B T"},
         answer="b", explanation="F = F_ph[ω_eff] + U0; the further values of outfile.U0 are higher-order corrections."),
    dict(type="mc", question="Why does SymPhase require e_static when the TDEP baseline (or an SSCHA total free energy) is used?",
         options={"a": "for plotting", "b": "U0 and SSCHA totals contain the static lattice energy; only F − E_static may be added to the DFT energy of a hull entry",
                  "c": "TDEP needs it", "d": "it does not"},
         answer="b", explanation="Otherwise the static energy would be counted twice."),
    dict(type="mc", question="The SCHA free energy compared with the exact one is …",
         options={"a": "always lower", "b": "always higher or equal (Gibbs–Bogoliubov variational bound)", "c": "equal", "d": "random"},
         answer="b", explanation="A variational (Gaussian) trial density matrix gives an upper bound to F."),
    dict(type="num", question="Hands-on: quartic oscillator (k = 2000, λ = 1200, Fe mass) at 1000 K: harmonic error F_harm − F_exact in meV (1 decimal)?",
         answer=-5.1, tol=0.15, hint="quartic_oscillator(2000, 1200, 55.845, [1000])",
         explanation="−5.07 meV, versus +0.22 meV for SCHA."),
    dict(type="mc", question="With a positive quartic term, how do SCHA phonon frequencies change on heating?",
         options={"a": "they soften", "b": "they harden (k_eff = k + 12λ⟨x²⟩ grows)", "c": "unchanged", "d": "they become imaginary"},
         answer="b", explanation="Larger fluctuations at high T probe the stiffer x⁴ part of the potential."),
    dict(type="num", question="Hands-on: stabilisation temperature of Ni3Al-D019 in the SCHA hull (NiAl_scph.csv) in K?",
         answer=790, tol=10, unit="K", hint="FiniteTHull(load_csv(scph_csv), vib='scph', mixing=False).stabilisation_temperature('Ni3Al-D019', Tmax=1600)",
         explanation="≈ 790 K with anharmonic phonons."),
    dict(type="num", question="Hands-on: the same transition with harmonic phonons only (same k), in K?",
         answer=556, tol=10, unit="K", explanation="≈ 556 K: ignoring anharmonic hardening underestimates T* by ≈ 235 K."),
    dict(type="mc", question="What does FreeEnergySurface do with total free energies at several volumes?",
         options={"a": "averages them", "b": "minimises F(V, T) over V at each T -> V(T), G(T), β, B_T (anharmonic QHA)",
                  "c": "fits a Debye temperature", "d": "nothing"},
         answer="b", explanation="It is the QHA minimisation, with anharmonic (TDEP/SSCHA) free energies instead of harmonic ones."),
]
build_bank(os.path.join(DATA, "quiz_kubo_scph.json"),
           "Quiz 7 — Kubo magnitudes with symmetry, self-consistent anharmonic phonons", KQ)
quiz_nb("19_quiz_kubo_anharmonic_phonons.ipynb",
        "Quiz 7 — Kubo magnitudes with symmetry, self-consistent anharmonic phonons (20 questions)",
        KQ, "kubo_scph", "Work through tutorials **17** and **18** first. Questions 2, 4–6, 8–11, 16, 18 and 19 are hands-on.",
        KS_HELPER + "\nst = SymTensor.from_analysis(analyze_tensors(f'{KB}/Pt_postw90', tensors='spin', ossg='no'), 'spin')\n"
        "scan = read_postw90_shc(f'{KB}/Pt_postw90/Pt-shc-fermiscan.dat', win=f'{KB}/Pt_postw90/Pt.win')")


# =============================================================================
# 20  two-magnon striction and damping;  21 quiz
# =============================================================================
DM_HELPER = 'import numpy as np, spglib, math, warnings\nimport matplotlib.pyplot as plt\nwarnings.simplefilter("ignore")\nfrom symphase.magphon import analyse_mp, MagnonPhonon, plot_mp\nfrom symphase.magdamp import MagnonPhononVertex, gilbert_first_order, linewidths_along_path, plot_linewidths, bose\na = 3.0\nlat, fr = np.diag([a, 12, 12]), np.zeros((1, 3))\nds = spglib.get_symmetry_dataset((lat, fr, [0]))\nGX = [("Γ", (0, 0, 0)), ("X", (0.5, 0, 0))]\ndef fm_chain(dJdr=3.0, J=-5.0, K=0.3, n_per=60):\n    return analyse_mp(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.0]]), ds.rotations, ds.translations,\n                      springs=[(0, 0, [1, 0, 0], 600.0, 300.0)], dJdr=[dJdr], J=[J], K=K, K_axis=[0, 0, 1], spin=1.0,\n                      rmax=3.1, n_per=n_per, path=GX, masses=[55.845])\nlat2, fr2 = np.diag([2 * a, 12, 12]), np.array([[0, 0, 0], [0.5, 0, 0]])\nds2 = spglib.get_symmetry_dataset((lat2, fr2, [0, 0]))\ndef afm_chain(J=4.0, K=0.15, dJdr=3.0, kL=900.0, kT=400.0, n_per=60):\n    return analyse_mp(lat2, fr2, [0, 0], ["Fe", "Fe"], np.array([[0, 0, 3.0], [0, 0, -3.0]]), ds2.rotations,\n                      ds2.translations, springs=[(0, 1, [0, 0, 0], kL, kT), (1, 0, [1, 0, 0], kL, kT)], dJdr=[dJdr],\n                      J=[J], K=K, K_axis=[0, 0, 1], spin=1.5, rmax=3.1, n_per=n_per, path=GX, masses=[55.845, 55.845])\n'

save([
    md(r"""# SymPhase tutorial 20 — two-magnon (second-order) striction and damping of magnons and phonons

Tutorial 14 used the **linear** spin–lattice term ($\partial^2E/\partial X\,\partial u$, one magnon + one phonon): it hybridises bands, but
isotropic exchange striction gives **nothing** for collinear spins. Here the next order:

* **two-magnon striction** — $J(\mathbf d_0+\delta\mathbf d)=J+\sum_\gamma G^\gamma\delta d_\gamma$ inserted into the *bilinear* magnon Hamiltonian:
  $H_3=N^{-1/2}\sum_{\mathbf k,\mathbf q,\nu}(b_{\mathbf q\nu}+b^\dagger_{-\mathbf q\nu})\,\Psi^\dagger_{\mathbf k+\mathbf q}M_\nu(\mathbf k,\mathbf q)\Psi_{\mathbf k}$ — magnon scattering by phonons and
  phonon decay into two magnons, present also for collinear magnets;
* **thermal linewidths** (Fermi's golden rule, FWHM): phonons from magnon scattering and two-magnon decay; magnons from phonon
  emission, absorption and fusion;
* **phenomenological damping**: linearised Landau–Lifshitz–Gilbert equations (Gilbert $\alpha$) + lattice friction $\eta$, solved exactly for
  the coupled magnon–phonon modes.

The vertex is checked entry by entry against the first-order change of a frozen-phonon **supercell** Hamiltonian (`tests/`)."""),
    SETUP,
    code(DM_HELPER),
    md(r"""## 1. FM chain: no linear coupling, but finite phonon linewidths

Isotropic exchange striction $\mathrm dJ/\mathrm dr=3$ meV/Å. The linear hybridisation is exactly zero; the two-magnon process (a phonon absorbed by
scattering a thermal magnon $k\to k+q$) gives a finite width:"""),
    code("""r = fm_chain()
vx = MagnonPhononVertex(r["model"])
print("linear hybridisation max |ω − ω0| =", np.nanmax(np.abs(r["omega"] - r["omega0"])), "meV")
q = np.array([0.9, 0, 0]); om, fs = vx.phonons(q)
nu = int(np.argmax([abs(x[0, 0]) if x is not None else 0 for x in fs]))
Ts = [10, 50, 100, 200, 300, 450, 600]
gs = [vx.phonon_linewidth(q, T=T, grid=(4000, 1, 1), sigma=0.02, modes=[nu])["gamma"][nu] for T in Ts]
for T, g in zip(Ts, gs):
    print(f"T = {T:4d} K   longitudinal phonon ω = {om[nu]:.3f} meV   γ = {g:.5f} meV")
fig, ax = plt.subplots(figsize=(4.8, 3.2)); ax.plot(Ts, gs, "o-", color="#2c5d8f")
ax.set_xlabel("T (K)"); ax.set_ylabel("phonon FWHM (meV)"); ax.set_title("scattering of thermal magnons ∝ n(ε)", fontsize=10)"""),
    md(r"""**Check of the Brillouin-zone sum.** In 1D the energy δ-function can be solved: roots $k^*$ of $\varepsilon(k+q)-\varepsilon(k)=\omega$,
$\gamma = a\sum_{k^*}|S(k^*)|^2\,[n(\varepsilon_{k^*})-n(\varepsilon_{k^*+q})]/|\partial_k(\varepsilon_{k+q}-\varepsilon_k)|$:"""),
    code("""S_, J_, K_ = 1.0, 5.0, 0.3
eps = lambda k: 2 * S_ * J_ * (1 - np.cos(k * a)) + 2 * K_ * S_
s = om[nu] / (4 * S_ * J_ * np.sin(q[0] * a / 2)); ref = 0.0
for base in (math.asin(s), math.pi - math.asin(s)):
    k = (base - q[0] * a / 2) / a
    Sm = vx.scattering(np.array([k, 0, 0]), q, fs[nu])[0, 0]
    ref += a * abs(Sm) ** 2 * (bose(eps(k), 300) - bose(eps(k + q[0]), 300)) / (4 * S_ * J_ * a * np.sin(q[0] * a / 2) * abs(math.cos(k * a + q[0] * a / 2)))
print(f"semi-analytic γ = {ref:.5f} meV   BZ sum = {gs[Ts.index(300)]:.5f} meV")"""),
    md("## 2. Magnon linewidths: phonon emission, absorption, fusion — along Γ–X"),
    code("""X = np.array([np.pi / a, 0, 0])
for T in (0.0, 300.0):
    m = vx.magnon_linewidth(X, T=T, grid=(400, 1, 1), sigma=0.2)
    print(f"T = {T:5.0f} K: magnon at X ε = {m['epsilon'][0]:.2f} meV  γ = {m['gamma'][0]:.3f} "
          f"(emission {m['gamma_emit'][0]:.3f}, absorption {m['gamma_abs'][0]:.3f}, fusion {m['gamma_fusion'][0]:.3f}) meV")
lw = linewidths_along_path(r["model"], r["q"], r["x"], T=300.0, grid=(300, 1, 1), sigma=0.2, every=3)
plot_linewidths(lw, r, title="FM chain, isotropic striction, T = 300 K");"""),
    md(r"""Magnons above the top of the phonon band can **emit** phonons (spontaneously, even at $T=0$): their width grows sharply.

## 3. AFM chain: a phonon decays into two magnons — even at $T = 0$

Bogoliubov pairing in the antiferromagnet produces $\alpha^\dagger\alpha^\dagger b$ terms: an optical phonon above twice the magnon gap
decays spontaneously."""),
    code("""ra = afm_chain(); va = MagnonPhononVertex(ra["model"])
print("magnon gap:", np.round(va.magnon(np.zeros(3))[0], 3), "meV")
for qx in (0.3, 0.6, 0.9):
    qq = np.array([qx, 0, 0]); p = va.phonon_linewidth(qq, T=0.0, grid=(2000, 1, 1), sigma=0.05)
    print(f"q = {qx}:  ω = {np.round(p['omega'], 2)}\\n          γ_pair(T=0) = {np.round(p['gamma_pair'], 4)}")"""),
    md(r"""## 4. Gilbert damping and lattice friction (linearised LLG, exact)

Ferromagnet: $\omega\to\omega/(1+\alpha^2)$, FWHM $\gamma=2\alpha\omega/(1+\alpha^2)$. Antiferromagnet: the width is **exchange-enhanced** and nearly
$k$-independent, $\gamma\approx2\alpha(2J+2K)S$ — far above $2\alpha\omega$ at small $\omega$:"""),
    code("""o, g, wm = MagnonPhononVertex(r["model"]).damped_modes(np.array([0.7, 0, 0]), alpha=0.05, coupling=False)
im = int(np.argmax(wm)); e0 = 2 * 5.0 * (1 - np.cos(0.7 * a)) + 0.6
print(f"FM: ω = {o[im]:.4f} (ε/(1+α²) = {e0/(1+0.05**2):.4f}),  γ = {g[im]:.4f} (2αε/(1+α²) = {2*0.05*e0/(1+0.05**2):.4f}) meV")
for kx in (0.0, 0.4, 0.9):
    o, g, wm = va.damped_modes(np.array([kx, 0, 0]), alpha=1e-3, coupling=False)
    mg = [(x, y) for x, y, w in zip(o, g, wm) if w > 0.5]
    print(f"AFM k = {kx}: ω = {mg[0][0]:.3f} meV, γ = {mg[0][1]:.5f} meV  (2αω = {2e-3*mg[0][0]:.5f};  2α(2J+2K)S = {2e-3*(8+0.3)*1.5:.5f})")
o, g, wm = va.damped_modes(np.array([0.5, 0, 0]), alpha=0.0, eta=0.3, coupling=False)
print("pure lattice friction η = 0.3 meV -> phonon FWHM:", np.round([y for y, w in zip(g, wm) if w < 0.5], 4))"""),
    md("""## 5. Command line
```bash
cd examples/vasp/MagnonPhonon_FM_chain
symphase magnons --J -5 --K 0.3 --K-axis 0 0 1 --spin 1 --rmax 3.1 --phonons springs.dat --dJdr 3 \\
                 --linewidth 300 --lw-grid 300 --lw-sigma 0.2 --alpha 0.01 --phonon-damping 0.05
```
writes `*_linewidths.png/.dat` (ω ± γ/2 for phonons and magnons) and prints the Gilbert/friction widths of the coupled modes.
Continue with **21_quiz_two_magnon_damping.ipynb**."""),
], "20_tutorial_two_magnon_damping.ipynb")

DQ = [
    dict(type="mc", question="Which term does the two-magnon (second-order) striction vertex come from?",
         options={"a": "the anisotropy only", "b": "the bond-length derivative of J inserted into the BILINEAR (two-magnon) spin-wave Hamiltonian",
                  "c": "the Zeeman energy", "d": "the phonon force constants"},
         answer="b", explanation="H3 = (b + b†) Ψ† M Ψ: one phonon operator times two magnon operators."),
    dict(type="mc", question="Hands-on: FM chain with isotropic striction dJ/dr = 3 meV/Å. Linear magnon–phonon hybridisation?",
         options={"a": "large anticrossing gaps", "b": "exactly zero (collinear spins)", "c": "only at Γ", "d": "only at X"},
         answer="b", explanation="The linear vertex ∝ x_a·v_b vanishes for collinear order; the two-magnon vertex does not."),
    dict(type="num", question="Hands-on: FWHM (meV) of the longitudinal phonon at q = 0.9 Å⁻¹ in that FM chain at 300 K (3 decimals)?",
         answer=0.144, tol=0.003, hint="vx.phonon_linewidth(qph, T=300, grid=(4000,1,1), sigma=0.02, modes=[nu])",
         explanation="≈ 0.144 meV from scattering of thermal magnons (k → k+q)."),
    dict(type="mc", question="Hands-on: how does that phonon width behave at low temperature (10 K)?",
         options={"a": "it stays 0.144 meV", "b": "it vanishes (≈ 5×10⁻⁴ meV): no thermal magnons to scatter", "c": "it diverges", "d": "it becomes negative"},
         answer="b", explanation="The scattering channel ∝ n(ε_k) − n(ε_k+q) dies out as the magnon population freezes."),
    dict(type="mc", question="How was the vertex M(k, q) validated?",
         options={"a": "not at all", "b": "entry by entry against the first-order change of a frozen-phonon supercell Hamiltonian built by the independent LSWT code",
                  "c": "by plotting", "d": "against experiments"},
         answer="b", explanation="All blocks (scattering, pair creation, pair annihilation) agree to ~10⁻¹¹ for an AFM with anisotropy and complex patterns."),
    dict(type="mc", question="A bug found that way was …",
         options={"a": "a wrong mass", "b": "double counting of the pair terms (a real-space c3 a†a† term must be represented once)", "c": "a sign of J", "d": "none"},
         answer="b", explanation="Comparing with the supercell showed the pair amplitude was twice too large until the duplicate assignment was removed."),
    dict(type="mc", question="Why can the BZ sum of the 1D linewidth be checked semi-analytically?",
         options={"a": "it cannot", "b": "in 1D the energy δ-function has explicit roots k*, giving γ = a Σ|S|²Δn/|∂(ε_k+q − ε_k)/∂k|",
                  "c": "because σ = 0", "d": "because T = 0"},
         answer="b", explanation="Gaussian-smeared BZ sum and root formula agree to 10⁻⁵ relative."),
    dict(type="num", question="Hands-on: magnon at X of the FM chain at T = 0 — FWHM in meV (2 decimals)?",
         answer=0.24, tol=0.02, hint="vx.magnon_linewidth(X, T=0, grid=(400,1,1), sigma=0.2)",
         explanation="≈ 0.24 meV: spontaneous phonon emission (factor 1 + N + n) survives at T = 0."),
    dict(type="mc", question="Which magnons can decay by emitting a phonon?",
         options={"a": "all magnons", "b": "those for which ε(k) − ε(k−q) = ω(q) has solutions — typically magnons above the phonon band edge region",
                  "c": "none", "d": "only at Γ"},
         answer="b", explanation="Energy and momentum conservation restrict the channel; in the FM chain the width grows sharply above the phonon band."),
    dict(type="mc", question="Hands-on: which phonon channel survives at T = 0 in the AFM chain?",
         options={"a": "magnon scattering", "b": "decay into two magnons (pair creation from Bogoliubov pairing)", "c": "none", "d": "Gilbert damping"},
         answer="b", explanation="(1 + n1 + n2) → 1 at T = 0: spontaneous two-magnon decay of optical phonons above 2× the magnon gap."),
    dict(type="num", question="Hands-on: magnon gap (meV) of the AFM chain at Γ (3 decimals)?",
         answer=3.317, tol=0.005, hint="va.magnon(np.zeros(3))[0]",
         explanation="ω = 2S√(K(2J+K)) type gap ≈ 3.317 meV for J = 4, K = 0.15, S = 1.5."),
    dict(type="mc", question="Gilbert damping of a ferromagnetic magnon in the exact linearised LLG gives …",
         options={"a": "γ = αω", "b": "γ = 2αω/(1+α²) and ω → ω/(1+α²)", "c": "γ = 2α", "d": "no damping"},
         answer="b", explanation="Checked to 10⁻⁹ in the tests."),
    dict(type="num", question="Hands-on: AFM chain, α = 10⁻³: Gilbert FWHM of the Γ magnon in meV (4 decimals)?",
         answer=0.0249, tol=0.0002, hint="va.damped_modes(np.zeros(3), alpha=1e-3, coupling=False)",
         explanation="0.0249 meV = 2α(2J + 2K)S — exchange-enhanced."),
    dict(type="mc", question="Compared with 2αω at the AFM magnon gap (≈ 0.0066 meV), the AFM Gilbert width is …",
         options={"a": "the same", "b": "≈ 3.8 times larger and almost k-independent (exchange enhancement)", "c": "smaller", "d": "zero"},
         answer="b", explanation="Elliptical precession in an AFM dissipates through the exchange field."),
    dict(type="mc", question="How was the AFM Gilbert result checked independently?",
         options={"a": "it was not", "b": "against the energy-dissipation formula γ = 2αω² Σ(|X|²+|Y|²)/ξ†𝓗ξ of the undamped modes",
                  "c": "against the FM formula", "d": "against phonopy"},
         answer="b", explanation="Exact LLG eigenvalues and first-order dissipation agree to 10⁻³ relative at α = 10⁻³."),
    dict(type="num", question="Lattice friction η = 0.3 meV in the equation Ṗ = −∂H/∂u − ηP: FWHM of a pure phonon in meV?",
         answer=0.3, tol=0.001, explanation="The amplitude decays at η/2, so the FWHM equals η."),
    dict(type="mc", question="Which quantity in the golden rule of phonon decay into two magnons accounts for stimulated emission?",
         options={"a": "n1 − n2", "b": "1 + n1 + n2", "c": "N − n", "d": "none"},
         answer="b", explanation="(1+n1)(1+n2) − n1 n2 = 1 + n1 + n2 (forward minus reverse process)."),
    dict(type="mc", question="Phonon absorption by a magnon k → k+q enters the magnon linewidth with the factor …",
         options={"a": "1 + N + n", "b": "N − n(ε_{k+q})", "c": "n − N", "d": "1"},
         answer="b", explanation="N(1 + n') − (1 + N) n' = N − n' for absorption; emission has 1 + N + n'."),
    dict(type="mc", question="Which CLI options switch on the damping analysis in symphase magnons?",
         options={"a": "--relax", "b": "--linewidth T, --lw-grid, --lw-sigma, --alpha, --phonon-damping (with --phonons)", "c": "--spiral auto", "d": "--supercell auto"},
         answer="b", explanation="They write *_linewidths.png/.dat with ω ± γ/2 for phonons and magnons."),
    dict(type="mc", question="Why is the two-magnon process important for experiments on collinear magnets?",
         options={"a": "it is not", "b": "it gives temperature-dependent phonon and magnon linewidths (Raman, inelastic neutron/X-ray) even when linear magnon-polaron gaps vanish",
                  "c": "it changes the crystal structure", "d": "it removes all magnons"},
         answer="b", explanation="Spin–phonon damping appears as T-dependent broadening tied to the magnon population."),
]
build_bank(os.path.join(DATA, "quiz_damping.json"), "Quiz 8 — two-magnon striction and damping", DQ)
quiz_nb("21_quiz_two_magnon_damping.ipynb", "Quiz 8 — two-magnon striction and damping of magnons and phonons (20 questions)",
        DQ, "damping", "Work through **20_tutorial_two_magnon_damping.ipynb** first. Questions 2–4, 8, 10, 11 and 13 are hands-on.",
        DM_HELPER + "\nr = fm_chain(); vx = MagnonPhononVertex(r['model'])\nqph = np.array([0.9, 0, 0]); om, fs = vx.phonons(qph)\n"
        "nu = int(np.argmax([abs(x[0, 0]) if x is not None else 0 for x in fs]))\nX = np.array([np.pi / a, 0, 0])\n"
        "ra = afm_chain(); va = MagnonPhononVertex(ra['model'])")


# =============================================================================
# 22  magnon–magnon interactions + self-energies;  23 quiz
# =============================================================================
IN_HELPER = 'import numpy as np, spglib, math, warnings\nimport matplotlib.pyplot as plt\nwarnings.simplefilter("ignore")\nfrom symphase.magint import FourMagnonHF, analyse_interacting, plot_interacting, _bose\nfrom symphase.magnons import analyse\nfrom symphase.magphon import analyse_mp\nfrom symphase.magdamp import MagnonPhononVertex, magnon_spectral_map, plot_spectral_map\na = 3.0\ndef hf_model(kind, S, J, K=0.0, grid=(400, 1, 1), T=0.0, order="first"):\n    if kind == "fm":\n        lat, fr, mg, ty = np.diag([a, 12, 12]), np.zeros((1, 3)), np.array([[0, 0, 2 * S]]), [0]\n    elif kind == "afm":\n        lat, fr, ty = np.diag([2 * a, 12, 12]), np.array([[0, 0, 0], [0.5, 0, 0]]), [0, 0]\n        mg = np.array([[0, 0, 2 * S], [0, 0, -2 * S]])\n    else:\n        lat, fr, ty = np.array([[a, a, 0], [a, -a, 0], [0, 0, 12.]]), np.array([[0, 0, 0], [0.5, 0.5, 0]]), [0, 0]\n        mg = np.array([[0, 0, 2 * S], [0, 0, -2 * S]])\n    ds = spglib.get_symmetry_dataset((lat, fr, ty))\n    m = analyse(lat, fr, ty, ["Fe"] * len(ty), mg, ds.rotations, ds.translations, J=[J], K=K,\n                K_axis=[0, 0, 1] if K else None, spin=S, rmax=3.1, return_model=True)\n    return FourMagnonHF.from_model(lat, fr, m).solve(grid=grid, T=T, order=order), lat\nlat1, fr1 = np.diag([a, 12, 12]), np.zeros((1, 3)); ds1 = spglib.get_symmetry_dataset((lat1, fr1, [0]))\ndef fm_chain_mp(dJdr=1.0, n_per=60):\n    return analyse_mp(lat1, fr1, [0], ["Fe"], np.array([[0, 0, 2.0]]), ds1.rotations, ds1.translations,\n                      springs=[(0, 0, [1, 0, 0], 600.0, 300.0)], dJdr=[dJdr], J=[-5.0], K=0.3, K_axis=[0, 0, 1], spin=1.0,\n                      rmax=3.1, n_per=n_per, path=[("Γ", (0, 0, 0)), ("X", (0.5, 0, 0))], masses=[55.845])\nX = np.array([np.pi / a, 0, 0])\n'

save([
    md(r"""# SymPhase tutorial 22 — magnon–magnon interactions and self-energies beyond the golden rule

**Part A — four-magnon interactions.** Holstein–Primakoff to quartic order in local frames,
$S^+=\sqrt{2S}(a-a^\dagger aa/4S)$, $S^z{}'=S-a^\dagger a$, gives for every bond the quartic terms
$(\mathbf v_i\cdot J\cdot\mathbf v_j)\,n_in_j-\sum_{\alpha\beta}(\mathbf e_i^\alpha\cdot J\cdot\mathbf e_j^\beta)\frac{\sqrt{S_iS_j}}8\big[L_i^\alpha C_j^\beta/S_j+C_i^\alpha L_j^\beta/S_i\big]$.
They are decoupled with Wick's theorem (**Hartree–Fock**) using the thermal correlations $\langle a^\dagger a\rangle,\langle aa\rangle$:
`order="first"` = the $1/S$ correction (Oguchi, Dyson), `order="sc"` = self-consistent spin-wave theory.

**Part B — self-energies.** From the magnon–magnon–phonon vertex of tutorial 20 the full complex, frequency-dependent
self-energies $\Sigma(\omega)$ and spectral functions $A(\omega)=-\frac1\pi\mathrm{Im}\,[\omega-\omega_0-\Sigma(\omega)]^{-1}$: energy shifts, non-Lorentzian
lineshapes and split-off states that the golden rule (which only gives $-2\,\mathrm{Im}\,\Sigma(\omega_0)$) cannot describe."""),
    SETUP,
    code(IN_HELPER),
    md(r"""## A1. Ferromagnet: no zero-point correction, thermal softening (Dyson)

For a ferromagnet the ground state is exact: at $T=0$ the HF correction vanishes. At finite $T$ the first-order result is Dyson's
$\varepsilon_k(T)=\Delta+(\varepsilon_k^0-\Delta)\,[1-\frac1{NS}\sum_q(1-\cos qa)\,n_q]$:"""),
    code("""hf0, _ = hf_model("fm", 1.5, -5.0, K=0.3, T=0.0); k = np.array([1.0, 0, 0])
print("T = 0: HF − LSWT =", hf0.spectrum([k])[0][0] - hf0.bare_spectrum([k])[0][0], "meV")
Ts = [0, 25, 50, 100, 150, 200]; ratio = []
for T in Ts:
    hf, _ = hf_model("fm", 1.5, -5.0, K=0.3, T=T)
    ks = hf.kgrid((400, 1, 1)); e0 = np.array([hf.bare_spectrum([q])[0][0] for q in ks])
    c = np.mean((1 - np.cos(a * ks[:, 0])) * _bose(e0, T)); E0 = hf.bare_spectrum([k])[0][0]
    E = hf.spectrum([k])[0][0]; ratio.append((E - 0.9) / (E0 - 0.9))
    print(f"T = {T:4d} K: ε(k) HF = {E:8.4f}  Dyson = {0.9 + (E0 - 0.9) * (1 - c / 1.5):8.4f}  LSWT = {E0:.4f} meV")
fig, ax = plt.subplots(figsize=(4.8, 3.2)); ax.plot(Ts, ratio, "o-", color="#c0392b")
ax.set_xlabel("T (K)"); ax.set_ylabel("stiffness D(T)/D(0)"); ax.set_title("thermal magnon softening (1D FM chain)", fontsize=10)"""),
    md(r"""## A2. Antiferromagnet: zero-point fluctuations stiffen the spin waves (Oguchi)

At $T=0$ the HF correction rescales the AFM spectrum by $Z=1+r/2S$, $r=1-\frac1N\sum_k\sqrt{1-\gamma_k^2}$:
$r=1-2/\pi=0.3634$ (chain), $0.1579$ (square lattice; $Z_c=1.158$ for $S=\frac12$)."""),
    code("""for S in (0.5, 1.0, 2.5):
    hf, _ = hf_model("afm", S, 3.0, grid=(800, 1, 1)); kk = np.array([0.3, 0, 0])
    print(f"chain  S = {S}: Z = {hf.spectrum([kk])[0][0] / hf.bare_spectrum([kk])[0][0]:.6f}   1 + (1 − 2/π)/2S = {1 + (1 - 2/np.pi)/(2*S):.6f}")
hf, lat = hf_model("square", 0.5, 1.0, grid=(40, 40, 1))
for kf in ([0.05, 0.02, 0], [0.31, 0, 0]):
    kk = np.array(kf) @ (2 * np.pi * np.linalg.inv(lat).T)
    print(f"square S = 1/2, k = {kf}: Z = {hf.spectrum([kk])[0][0] / hf.bare_spectrum([kk])[0][0]:.5f}  (Oguchi 1.15795)")"""),
    md(r"""### Ordered moment
$\langle S^z\rangle=S-\langle a^\dagger a\rangle$. On the square lattice the zero-point reduction is $\approx0.197$ (grid-converged value; the
log-singular BZ sum converges slowly). **In 1D** it diverges logarithmically with the grid — no long-range order, as it must be:"""),
    code("""for g in (40, 80):
    hf, _ = hf_model("square", 0.5, 1.0, grid=(g, g, 1)); print(f"square S=1/2, grid {g}²: ⟨S^z⟩ = {hf.sublattice_magnetisation((g, g, 1))[0]:.4f}")
for g in (200, 800, 3200):
    hf, _ = hf_model("afm", 0.5, 3.0, grid=(g, 1, 1)); print(f"chain  S=1/2, grid {g}: ⟨S^z⟩ = {hf.sublattice_magnetisation((g, 1, 1))[0]:.4f}")"""),
    md(r"""## A3. Self-consistent spin-wave theory and its breakdown

`order="sc"` iterates the Hartree–Fock correlations. In the 1D ferromagnet thermal fluctuations grow with $T$; at 200 K there is **no**
self-consistent solution — the spin-wave Hamiltonian becomes unstable (Mermin–Wagner: no 1D order at $T>0$):"""),
    code("""for T in (50.0, 100.0, 200.0):
    try:
        h1, _ = hf_model("fm", 1.5, -5.0, K=0.3, T=T); hs, _ = hf_model("fm", 1.5, -5.0, K=0.3, T=T, order="sc")
        print(f"T = {T:5.0f} K: first order {h1.spectrum([k])[0][0]:.4f}, self-consistent {hs.spectrum([k])[0][0]:.4f} meV ({hs.iterations} iterations)")
    except ValueError as err:
        print(f"T = {T:5.0f} K: {str(err)[:150]}...")"""),
    md(r"""## B1. Self-energies: $-2\,\mathrm{Im}\,\Sigma(\omega_0)$ is the golden rule

With the same Lorentzian broadening the identity is exact; the real part is new information (energy shift):"""),
    code("""vx = MagnonPhononVertex(fm_chain_mp(1.0)["model"])
e0, W, E = vx.magnon_poles(X, 0, T=300, grid=(300, 1, 1))
s0 = vx.self_energy(W, E, [e0], 0.05)[0]
g = vx.magnon_linewidth(X, T=300, grid=(300, 1, 1), sigma=0.05, broadening="lorentz")["gamma"][0]
print(f"magnon at X (dJ/dr = 1 meV/Å): −2 Im Σ = {-2*s0.imag:.6f}  golden rule = {g:.6f} meV;  Re Σ = {s0.real:+.4f} meV")
ws = np.linspace(e0 - 15, e0 + 15, 6001); A = vx.spectral(e0, vx.self_energy(W, E, ws, 0.05), ws, 1e-3)
print("∫A dω =", round(np.trapezoid(A, ws), 4), ";  peak at", round(ws[np.argmax(A)], 3), "meV (bare", round(e0, 3), ")")"""),
    md(r"""## B2. Strong coupling: the golden rule breaks down

With $\mathrm dJ/\mathrm dr=3$ meV/Å the zone-boundary magnon sits inside the phonon-emission continuum; Re Σ has a square-root
singularity at the continuum edge (1D) and a **sharp split-off state** appears above it, carrying only part of the weight:"""),
    code("""fig, ax = plt.subplots(figsize=(6.4, 3.6))
for dj, c in ((1.0, "#16325c"), (3.0, "#c0392b")):
    v = MagnonPhononVertex(fm_chain_mp(dj)["model"]); e0, W, E = v.magnon_poles(X, 0, T=300, grid=(300, 1, 1))
    A = v.spectral(e0, v.self_energy(W, E, ws, 0.05), ws, 1e-3); i = int(np.argmax(A))
    print(f"dJ/dr = {dj}: peak {ws[i]:.3f} meV (bare {e0:.2f}); weight within ±1 meV of the peak {np.trapezoid(A[abs(ws-ws[i])<1], ws[abs(ws-ws[i])<1]):.3f}")
    ax.semilogy(ws, np.clip(A, 1e-4, None), color=c, lw=1.8, label=f"dJ/dr = {dj} meV/Å")
ax.axvline(e0, ls=":", color="k"); ax.set_xlabel("ω (meV)"); ax.set_ylabel("A(X, ω)"); ax.legend(); ax.set_xlim(10, 27)"""),
    md("## B3. Spectral map along Γ–X"),
    code("""r = fm_chain_mp(1.0)
sm = magnon_spectral_map(r["model"], r["q"], r["x"], np.linspace(0, 26, 300), T=300.0, grid=(200, 1, 1), eta=0.1, every=3)
plot_spectral_map(sm);"""),
    md("""## Command line
```bash
cd examples/vasp/Square_AFM_HF   && symphase magnons --J 1.0 --rmax 3.1 --spin 0.5 --hf 0 --hf-grid 40 40          # Oguchi
cd examples/vasp/AFM_chain_HF    && symphase magnons --J 3.0 --rmax 3.1 --spin 1 --hf 0 --hf-grid 800
symphase magnons ... --hf 100 --hf-order sc                                                       # self-consistent
cd examples/vasp/MagnonPhonon_FM_chain && symphase magnons --J -5 --K 0.3 --K-axis 0 0 1 --spin 1 --rmax 3.1 \\
        --phonons springs.dat --dJdr 1 --linewidth 300 --spectral
```
Limits: exact to $1/S$ for collinear magnets; cubic magnon terms of non-collinear orders and second-order four-magnon damping are not
included. Continue with **23_quiz_magnon_interactions.ipynb**."""),
], "22_tutorial_magnon_interactions.ipynb")

IQ = [
    dict(type="mc", question="Where do four-magnon interactions come from in spin-wave theory?",
         options={"a": "from the phonons", "b": "from the quartic terms of the Holstein–Primakoff expansion (n_i n_j and the a†aa corrections of S±)",
                  "c": "from the Zeeman term", "d": "they do not exist"},
         answer="b", explanation="S⁺ = √2S (a − a†aa/4S) and Sᶻ = S − n produce quartic boson terms on every bond."),
    dict(type="mc", question="What does the Hartree–Fock (Wick) decoupling do?",
         options={"a": "drops all quartic terms", "b": "replaces O1O2O3O4 by Σ_pairings ⟨O_aO_b⟩ O_cO_d — a renormalised quadratic Hamiltonian",
                  "c": "diagonalises H exactly", "d": "adds phonons"},
         answer="b", explanation="With thermal correlations ⟨a†a⟩, ⟨aa⟩ from the spin-wave spectrum."),
    dict(type="num", question="Hands-on: FM chain (S = 1.5, J = −5 meV) at T = 0: HF − LSWT magnon energy at k = 1 Å⁻¹ (meV)?",
         answer=0.0, tol=1e-6, explanation="The fully polarised ferromagnet is an exact eigenstate: no zero-point correction."),
    dict(type="num", question="Hands-on: same chain at T = 200 K, first-order HF energy at k = 1 Å⁻¹ (meV, 2 decimals)?",
         answer=19.57, tol=0.02, hint="hf_model('fm', 1.5, -5.0, K=0.3, T=200).spectrum([[1,0,0]])",
         explanation="19.57 meV vs 30.75 meV in LSWT — Dyson's thermal softening, reproduced exactly."),
    dict(type="num", question="Hands-on: AFM chain, S = 1/2: renormalisation factor Z = ω_HF/ω_LSWT (4 decimals)?",
         answer=1.3634, tol=2e-4, explanation="Z = 1 + (1 − 2/π)/(2S) = 1.3634."),
    dict(type="num", question="Hands-on: square-lattice AFM, S = 1/2: Z (3 decimals)?",
         answer=1.158, tol=2e-3, explanation="Oguchi's spin-wave velocity renormalisation Z_c = 1 + 0.158/(2S)."),
    dict(type="mc", question="Physically, the Oguchi factor Z_c > 1 means …",
         options={"a": "thermal softening", "b": "zero-point (quantum) fluctuations reduce the ordered moment but stiffen the spin waves",
                  "c": "a phase transition", "d": "phonon hardening"},
         answer="b", explanation="Bond correlations ⟨a_ia_j⟩ from zero-point motion renormalise the exchange upwards by r/2S."),
    dict(type="mc", question="Hands-on: what happens to ⟨S^z⟩ of the S = 1/2 AFM chain as the k grid is refined?",
         options={"a": "it converges to 0.30", "b": "it decreases without bound (logarithmic divergence): no long-range order in 1D",
                  "c": "it equals 1/2", "d": "it oscillates"},
         answer="b", explanation="∫dk/|k| diverges in 1D: spin-wave theory signals the absence of Néel order."),
    dict(type="num", question="Hands-on: FM chain at T = 100 K, SELF-CONSISTENT HF energy at k = 1 Å⁻¹ (meV, 2 decimals)?",
         answer=26.25, tol=0.03, hint="hf_model('fm', 1.5, -5.0, K=0.3, T=100, order='sc')",
         explanation="26.25 meV (first order: 27.28 meV): self-consistency enhances the softening."),
    dict(type="mc", question="Hands-on: self-consistent HF for the FM chain at 200 K …",
         options={"a": "converges quickly", "b": "has no solution — the spin-wave Hamiltonian becomes unstable (no 1D order at T > 0)",
                  "c": "gives negative temperatures", "d": "equals LSWT"},
         answer="b", explanation="SymPhase stops with an explanatory error (Mermin–Wagner physics)."),
    dict(type="mc", question="For which magnets is the HF result the complete 1/S correction?",
         options={"a": "all", "b": "collinear magnets; non-collinear orders also have cubic terms of the same order (not included)",
                  "c": "only ferromagnets", "d": "none"},
         answer="b", explanation="Cubic terms contribute at second order with the same power of 1/S in non-collinear structures."),
    dict(type="mc", question="How is the self-energy related to the golden-rule linewidth γ?",
         options={"a": "γ = Re Σ", "b": "γ = −2 Im Σ(ω0) (exactly, with the same broadening)", "c": "γ = |Σ|", "d": "no relation"},
         answer="b", explanation="Checked to 10⁻¹² in the tests; Re Σ and the ω-dependence are new information."),
    dict(type="num", question="Hands-on: −2 Im Σ of the longitudinal phonon at q = 0.9 Å⁻¹ (dJ/dr = 1, T = 300 K, η = 0.05, grid 600) in meV (4 decimals)?",
         answer=0.0185, tol=5e-4, explanation="≈ 0.0185 meV, identical to the Lorentzian golden rule."),
    dict(type="num", question="Hands-on: Re Σ of the zone-boundary magnon (dJ/dr = 1 meV/Å, 300 K) in meV (3 decimals)?",
         answer=-0.044, tol=0.003, explanation="−0.044 meV: a small downward shift; the spectral peak moves to ≈ 20.57 meV."),
    dict(type="mc", question="Hands-on: dJ/dr = 3 meV/Å — where is the spectral peak of the zone-boundary magnon (bare 20.6 meV)?",
         options={"a": "at 20.6 meV", "b": "at ≈ 23.17 meV: a split-off state above the phonon-emission continuum edge", "c": "at 0", "d": "there is none"},
         answer="b", explanation="Re Σ diverges at the 1D continuum edge (22.85 meV); a sharp state forms above it."),
    dict(type="num", question="Hands-on: spectral weight within ±1 meV of that split-off peak (2 decimals)?",
         answer=0.22, tol=0.02, explanation="≈ 0.22: most of the weight is smeared over the continuum — far from a Lorentzian quasiparticle."),
    dict(type="mc", question="What does the sum rule ∫A(ω)dω = 1 check?",
         options={"a": "the phonon masses", "b": "the analytic structure (causality, normalisation) of the self-energy and the ω window",
                  "c": "the temperature", "d": "nothing"},
         answer="b", explanation="It holds to ≈ 2×10⁻⁴ for the 1D magnon spectral function."),
    dict(type="mc", question="Why are the 'anti-resonant' (all-creation / all-annihilation) terms kept?",
         options={"a": "they dominate Im Σ", "b": "they contribute to Re Σ (energy shift) although they never conserve energy", "c": "they are not kept", "d": "for speed"},
         answer="b", explanation="Their poles lie at negative ω, so they only shift the energy."),
    dict(type="mc", question="Which CLI options compute the four-magnon renormalisation?",
         options={"a": "--relax", "b": "--hf T [--hf-order first|sc] [--hf-grid ...]", "c": "--spiral auto", "d": "--supercell"},
         answer="b", explanation="Writes *_hf.png/.dat with LSWT and HF bands."),
    dict(type="mc", question="Which CLI option draws the magnon spectral function from the magnon–phonon self-energy?",
         options={"a": "--spectral (with --phonons and --linewidth T)", "b": "--plot none", "c": "--hf", "d": "--scan"},
         answer="a", explanation="Writes *_spectral.png (log A(k, ω) along the path)."),
]
build_bank(os.path.join(DATA, "quiz_interactions.json"), "Quiz 9 — magnon–magnon interactions and self-energies", IQ)
quiz_nb("23_quiz_magnon_interactions.ipynb", "Quiz 9 — magnon–magnon interactions and self-energies (20 questions)",
        IQ, "interactions", "Work through **22_tutorial_magnon_interactions.ipynb** first. Questions 3–6, 8–10 and 13–16 are hands-on.",
        IN_HELPER + "\nkq = np.array([1.0, 0, 0])")


# =============================================================================
# 24  cubic magnon terms + four-magnon damping;  25 quiz
# =============================================================================
NL_HELPER = 'import math, numpy as np, spglib\nimport matplotlib.pyplot as plt\nfrom symphase.magint import FourMagnonHF, MagnonSelfEnergy\nfrom symphase.magcheck import BosonCluster, classical_expansion\nfrom symphase.magnons import relax_spins, make_path, analyse\nfrom symphase.tensors import read_poscar, read_incar\na = 3.0\n\ndef canted_chain(h=4.2):\n    """AFM chain, XYZ exchange (Jy = 1.3), field h along −z: canted, gapped — the ED test bed"""\n    J = np.diag([1.0, 1.3, 1.0]); S = np.array([1.0, 1.0])\n    pairs = [(0, 1, np.array([a, 0, 0.]), J), (1, 0, np.array([a, 0, 0.]), J)]\n    hfield = np.array([[0, 0, -h]] * 2)\n    dirs, _, _ = relax_spins(np.array([[0, 1, 0.5], [0, -1, 0.5]]), S, [0, 1], pairs, [None, None], hfield=hfield,\n                             steps=200000, tol=1e-15)\n    return FourMagnonHF(np.diag([2 * a, 12, 12]), np.array([[0, 0, 0], [a, 0, 0.]]), [0, 1], dirs, S, pairs, None, hfield)\n\ndef triangular(S=0.5, G=36):\n    """120° triangular AFM (√3×√3 cell of examples/vasp/Triangular_120AFM_INCAR), J = 1 meV"""\n    d = "../examples/vasp/Triangular_120AFM_INCAR"\n    p = read_poscar(f"{d}/POSCAR"); mags = np.array(read_incar(f"{d}/INCAR")["magmom"]).reshape(-1, 3)\n    ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0, 0]))\n    m = analyse(p["lattice"], p["frac"], [0, 0, 0], ["Cr"] * 3, mags, ds.rotations, ds.translations, J=[1.0], spin=S,\n                rmax=3.1, return_model=True)\n    hf = FourMagnonHF.from_model(np.asarray(p["lattice"], float), np.asarray(p["frac"], float), m).solve(grid=(G, G, 1))\n    return hf, MagnonSelfEnergy(hf)\n\ndef canted_square(hr, S=1.0):\n    """square-lattice Heisenberg AFM (J = 1 meV) in a field h = hr·h_s along z (h_s = 8JS), √2 cell"""\n    lat = np.array([[a, a, 0], [a, -a, 0], [0, 0, 12.]])\n    pairs = [(0, 1, np.array(d, float), np.eye(3)) for d in ([a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0])]\n    s = math.sqrt(1 - hr ** 2)\n    hf = FourMagnonHF(lat, np.array([[0, 0, 0], [a, 0, 0.]]), [0, 1], np.array([[s, 0, hr], [-s, 0, hr]]),\n                      np.array([S, S]), pairs, None, np.array([[0, 0, -8 * hr * S]] * 2))\n    return hf, MagnonSelfEnergy(hf), lat\n\ndef square_fm(J=-2.0, K=0.5, S=1.0):\n    """square-lattice ferromagnet, easy axis z (gap 2KS)"""\n    pairs = [(0, 0, np.array([a, 0, 0.]), J * np.eye(3)), (0, 0, np.array([0, a, 0.]), J * np.eye(3))]\n    hf = FourMagnonHF(np.diag([a, a, 12.]), np.zeros((1, 3)), [0], np.array([[0, 0, 1.]]), np.array([S]), pairs,\n                      [np.diag([0, 0, -K])], None)\n    return hf, MagnonSelfEnergy(hf)\n\ndef ed_check(h=4.2, nmax=8):\n    """exact diagonalisation (3-cell cluster) vs bubble + tadpole and four-magnon sunset; returns rows per mode"""\n    hf = canted_chain(h); se = MagnonSelfEnergy(hf)\n    cl = BosonCluster(hf, 3, nmax); qs = cl.kpoints(); k = qs[1]; e = se._mset(k)[0]\n    H2 = cl.quadratic()\n    ed3 = cl.second_order(H2, cl.operator(se.terms3), e)\n    hf.update(qs, 0.0); ed4 = cl.second_order(H2, cl.operator(se.terms4) - cl.operator(cl.hf_terms()), e); hf.hf = None\n    rows = []\n    for n in range(2):\n        _, w, ee = se.cubic_poles(k, n, 0.0, qs=qs); bub = float(np.sum(w / (e[n] - ee)).real)\n        tad = se.tadpole(k, n, 0.0, ks=qs)\n        _, w, ee = se.quartic_poles(k, n, 0.0, qs=qs); sun = float(np.sum(w / (e[n] - ee)).real)\n        rows.append(dict(n=n, eps=e[n], ed3=ed3[n]["second"], bubble=bub, tadpole=tad, ed4=ed4[n]["second"], sunset=sun))\n    return rows\n\ndef tri_velocity(se, hf, kf=0.03):\n    """2S·δε/ε of the three Goldstone branches at small k (x direction): HF (first order), Re Σ³, total"""\n    k = np.array([kf * 2 * np.pi / a, 0, 0])\n    o = se.onshell(k, 0.0, (36, 36, 1), eta=0.01)\n    dhf = hf.first_order_shift([k])[0]\n    S2 = 2 * hf.S[0]\n    return dict(e0=o["e0"], hf=S2 * dhf / o["e0"], re3=S2 * o["re3"] / o["e0"],\n                total=S2 * (dhf + o["re3"] + o["tad"]) / o["e0"], gamma3=o["gamma3"])\n'

save([
    md(r"""# SymPhase tutorial 24 — cubic magnon interactions and second-order four-magnon damping

Tutorial 22 treated the quartic terms at first order (Hartree–Fock). Two pieces were missing:

**Cubic terms.** In a non-collinear magnet the bond energy contains $(S-n_i)\,\mathbf v_i\!\cdot\!J\!\cdot\!\mathbf S_j^\perp$ — its expansion gives
three-boson terms $-n_iL_j$ and $-S_iC_j/4S_j$ (plus single-ion and Zeeman contributions). They vanish for collinear order.
At second order they give the **one-loop bubble** (decay $\alpha_k\to\alpha_q\alpha_{k-q}$, absorption, source) — the same order $1/S$ as Hartree–Fock —
and a static **tadpole** from the linear terms that normal ordering produces (the $O(1/S)$ shift of the classical canting angles).
Together: the complete $1/S$ spectrum $\varepsilon+\delta\varepsilon_{\rm HF}+\mathrm{Re}\,\Sigma^{(3)}(\varepsilon)+\Sigma_{\rm tad}$ and **spontaneous magnon decay** $\Gamma^{(3)}=-2\,\mathrm{Im}\,\Sigma^{(3)}$ even at $T=0$.

**Four-magnon sunset.** The quartic vertex at second order: $2\leftrightarrow2$ scattering, $1\leftrightarrow3$ decay ($O(1/S^2)$) — the intrinsic thermal magnon–magnon damping of collinear magnets.

Both use the general bosonic formula $\Sigma(\omega)=r!\sum|c|^2\,[\prod_{\rm cr}(1+n)\prod_{\rm an}n-\prod_{\rm cr}n\prod_{\rm an}(1+n)]/(\omega-\sum_{\rm cr}\varepsilon+\sum_{\rm an}\varepsilon+i\eta)$
with the vertex $c$ obtained by Fourier + Bogoliubov transformation of the real-space Holstein–Primakoff terms (`MagnonSelfEnergy`)."""),
    SETUP,
    code(NL_HELPER),
    md(r"""## 1. Validation I — the Holstein–Primakoff terms equal the classical expansion

For random non-collinear spins, random exchange tensors, anisotropy and field, the cubic (and quartic) terms evaluated with complex numbers $z_i$
equal the $\lambda^3$ ($\lambda^4$) Taylor coefficient of the *exact* classical energy with $S^{z'}=S-|z|^2$, $S^+=z\sqrt{2S-|z|^2}$:"""),
    code("""rng = np.random.default_rng(2); M = 3
lat = np.diag([M * a, 4.0, 12]); cart = np.array([[m * a + 0.1 * m, 0.2 * m, 0] for m in range(M)])
pairs = [(m, (m + 1) % M, cart[(m + 1) % M] - cart[m] + (lat[0] if (m + 1) % M == 0 else 0), rng.normal(size=(3, 3))) for m in range(M)]
pairs += [(m, m, lat[1].copy(), rng.normal(size=(3, 3))) for m in range(M)]
dirs = rng.normal(size=(M, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
toy = FourMagnonHF(lat, cart, list(range(M)), dirs, rng.uniform(0.6, 2, M), pairs, [None] * M, None)
r = classical_expansion(toy)
print("cubic   : classical %.8f   Holstein–Primakoff %.8f" % r["cubic"])
print("quartic : classical %.8f   Holstein–Primakoff %.8f" % r["quartic"])"""),
    md(r"""## 2. Validation II — exact diagonalisation

A canted AFM chain (XYZ exchange, field near saturation) on a periodic 3-cell cluster: the bosonic Hamiltonian $H_2+\lambda H_3$ and
$H_2+\lambda\,{:}H_4{:}$ (${:}H_4{:}=H_4-H_{\rm HF}$) are diagonalised exactly in momentum sectors; the $\lambda^2$ coefficient of the one-magnon
energy must equal bubble + tadpole and the sunset on the same 3-point $k$ grid:"""),
    code("""rows = ed_check()
for r in rows:
    print(f"mode {r['n']} (ε = {r['eps']:.4f} meV)")
    print(f"   cubic  : ED {r['ed3']:+.6f}   bubble {r['bubble']:+.6f} + tadpole {r['tadpole']:+.6f} = {r['bubble'] + r['tadpole']:+.6f}")
    print(f"   quartic: ED {r['ed4']:+.6f}   sunset {r['sunset']:+.6f}")"""),
    md(r"""Without the tadpole the cubic result would be off by 0.08 meV: in this canted state the linear terms are finite.

## 3. Triangular-lattice antiferromagnet: the complete $1/S$ spectrum

The 120° state (no linear terms by symmetry). Linear spin-wave ordered moment $S-0.261$; near the three Goldstone modes the first-order
Hartree–Fock shift alone opens a spurious gap that the cubic bubble cancels — the Goldstone theorem holds order by order.
The velocities: $c_\parallel(1-0.115/2S)$, $c_\perp(1+0.083/2S)$ (Chubukov, Sachdev, Senthil 1994)."""),
    code("""hf, se = triangular(S=0.5)
print("⟨S⟩ =", np.round(hf.sublattice_magnetisation((30, 30, 1)), 4), "(S − 0.2535 on this grid; S − 0.261 converged)")
print("linear terms:", np.abs(se.linear_terms(0.0, (12, 12, 1))[0]).max())
v = tri_velocity(se, hf)
for n in range(3):
    print(f"branch {n}: ε = {v['e0'][n]:.4f}   2S·δε/ε:  HF {v['hf'][n]:+9.3f}   Re Σ³ {v['re3'][n]:+9.3f}   total {v['total'][n]:+.3f}")
print("c_∥ branch: −0.118 vs −0.115 (Chubukov et al.); c_⊥ branches: HF ≈ +37 and Re Σ³ ≈ −37 cancel to ≈ +0.09 (literature +0.083)")"""),
    md("Along the Brillouin-zone path (strong downward renormalisation, flat regions and decays — Starykh, Chubukov, Abanov; Chernyshev, Zhitomirsky):"),
    code("""d = "../examples/vasp/Triangular_120AFM_INCAR"
p = read_poscar(f"{d}/POSCAR"); mags = np.array(read_incar(f"{d}/INCAR")["magmom"]).reshape(-1, 3)
ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0, 0]))
from symphase.magint import analyse_interacting, plot_interacting
res = analyse_interacting(p["lattice"], p["frac"], [0, 0, 0], ["Cr"] * 3, mags, ds.rotations, ds.translations, T=0.0,
                          grid=(24, 24, 1), n_per=12, cubic=True, sigma=0.04, every=2, J=[1.0], spin=0.5, rmax=3.1)
plot_interacting(res, title="triangular AFM, S = 1/2: complete 1/S spectrum and T = 0 decay rate");"""),
    md(r"""## 4. Field-induced spontaneous magnon decay (square-lattice AFM)

In a field the spins cant, cubic terms appear, and above $h^*\approx0.76\,h_s$ magnons decay into two magnons at $T=0$
(Zhitomirsky, Chernyshev 1999). Maximum decay rate along Γ–X:"""),
    code("""hrs = [0.6, 0.68, 0.72, 0.76, 0.8, 0.85, 0.9]; gm = []
for hr in hrs:
    hfq, seq, latq = canted_square(hr)
    qq, _, _ = make_path(latq, [("Γ", (0, 0, 0)), ("X", (0.5, 0, 0))], 8)
    gm.append(max(seq.onshell(x, 0.0, (24, 24, 1), sigma=0.04, tadpole=False)["gamma3"].max() for x in qq[1:]))
    print(f"h/h_s = {hr:.2f}: max Γ³ = {gm[-1]:.3f} meV")
fig, ax = plt.subplots(figsize=(4.8, 3.2)); ax.plot(hrs, gm, "o-", color="#c0392b"); ax.axvline(0.76, ls=":", color="k")
ax.set_xlabel("h / h_s"); ax.set_ylabel("max Γ³ (meV)"); ax.set_title("onset of magnon decay", fontsize=10)"""),
    md("Below threshold the small values are smearing tails: they shrink when the grid is refined and σ reduced."),
    md(r"""## 5. Four-magnon damping at finite temperature (square-lattice ferromagnet)

No decays at $T=0$ (number-conserving vertex); thermal $2\leftrightarrow2$ scattering gives a damping that grows steeply with $T$ and is
suppressed at small $k$ (on-shell vertex → 0, Dyson):"""),
    code("""fm, sf = square_fm()
kfs = [0.1, 0.2, 0.3, 0.4]
for T in (0.0, 20.0, 40.0):
    g4 = [sf.onshell(np.array([kf * 2 * np.pi / a, 0, 0]), T, (12, 12, 1), cubic=False, quartic=True, eta=0.2, sigma=0.2)["gamma4"][0]
          for kf in kfs]
    print(f"T = {T:4.0f} K: Γ⁴ =", np.round(g4, 4), "meV")"""),
    md(r"""## 6. Spectral function of a decaying magnon

The pole representation works with the spectral-function helpers of tutorial 20:"""),
    code("""from symphase.magdamp import MagnonPhononVertex as MPV
k = np.array([2 * np.pi / (3 * a) * 1.2, 0.0, 0])
e0, W, E = se.cubic_poles(k, 2, 0.0, (36, 36, 1))
ws = np.linspace(0, 3, 1500); sig = MPV.self_energy(W, E, ws, 0.03)
A = MPV.spectral(e0, sig, ws, 1e-3)
fig, ax = plt.subplots(figsize=(5.4, 3.2)); ax.plot(ws, A, color="#16325c"); ax.axvline(e0, ls=":", color="k")
ax.set_xlabel("ω (meV)"); ax.set_ylabel("A(k, ω)"); ax.set_title("cubic self-energy: broadened, shifted magnon", fontsize=10)"""),
    md("""## Command line
```bash
cd examples/vasp/Triangular_120AFM_INCAR
symphase magnons --J 1 --spin 0.5 --rmax 3.1 --hf 0 --hf-grid 30 30 --cubic --se-sigma 0.03 --nq 16 --se-every 2
symphase magnons ... --hf 40 --damping4 --grid4 12 12          # four-magnon damping Γ⁴
```
Output: `*_hf.png` (bands with Γ shading + linewidth panel), `*_selfenergy.dat` (ε, δε_HF, Re Σ³, Σ_tad, Re Σ⁴, ω, Γ³, Γ⁴).
Limits: on-shell perturbation theory (no self-consistency near decay thresholds); single-ion anisotropy enters the quartic terms at harmonic
level only; Goldstone modes exactly at their zero are excluded. Continue with **25_quiz_nonlinear_spin_waves.ipynb**."""),
], "24_tutorial_cubic_four_magnon_damping.ipynb")

NQ = [
    dict(type="mc", question="Where do cubic (three-magnon) terms come from?",
         options={"a": "from phonons", "b": "from (S − n_i) v_i·J·S_j^⊥ (and single-ion/Zeeman terms) — non-zero only when v_i·J·e_j ≠ 0, i.e. non-collinear order",
                  "c": "from the quartic HP expansion", "d": "from the Zeeman term in collinear magnets only"},
         answer="b", explanation="For collinear spins v_i·J·e_j = 0 (isotropic J), so cubic terms vanish."),
    dict(type="mc", question="At which order in 1/S does the cubic one-loop bubble contribute to the spectrum?",
         options={"a": "1/S² ", "b": "1/S — the same order as Hartree–Fock", "c": "S", "d": "never"},
         answer="b", explanation="(√S)² vertices × 1/(S·energy): the complete 1/S spectrum needs both."),
    dict(type="mc", question="What is the tadpole?",
         options={"a": "a phonon", "b": "the second-order effect of linear terms from normal-ordering H₃ — the O(1/S) shift of the classical angles",
                  "c": "the four-magnon sunset", "d": "a numerical artefact"},
         answer="b", explanation="Σ_tad = −Σ_m 2Re(h_m g_m)/ε_0m; zero in the 120° structure by symmetry, finite in canted states."),
    dict(type="num", question="Hands-on: ed_check() — exact-diagonalisation second-order shift of mode 0 (meV, 3 decimals)?",
         answer=-0.7175, tol=0.002, explanation="−0.7175 meV = bubble (−0.7941) + tadpole (+0.0766)."),
    dict(type="num", question="Hands-on: the tadpole contribution of mode 0 in ed_check() (meV, 3 decimals)?",
         answer=0.0766, tol=0.002, explanation="+0.077 meV: without it the cubic result would disagree with exact diagonalisation."),
    dict(type="num", question="Hands-on: four-magnon sunset of mode 0 in ed_check() (meV, 4 decimals)?",
         answer=-0.0059, tol=2e-4, explanation="−0.00590 meV, equal to exact diagonalisation of H₂ + λ:H₄:."),
    dict(type="mc", question="Why is H₄ − H_HF used in the exact-diagonalisation check of the sunset?",
         options={"a": "for speed", "b": "Wick: H₄ − H_HF = :H₄: + const (normal-ordered in magnon operators), whose 2nd-order shift is the sunset",
                  "c": "to remove the field", "d": "H_HF is zero"},
         answer="b", explanation="Its first-order shift vanishes, as the check confirms."),
    dict(type="num", question="Hands-on: triangular AFM, S = 1/2: linear-spin-wave ordered moment ⟨S⟩ on the 30×30 grid (4 decimals)?",
         answer=0.2465, tol=1e-3, explanation="S − 0.2535 (converges to S − 0.261)."),
    dict(type="num", question="Hands-on: triangular AFM — 2S·δε/ε of the c_∥ Goldstone branch at |k|a = 0.188 (3 decimals)?",
         answer=-0.118, tol=0.006, explanation="≈ −0.118 vs the analytic −0.115 (Chubukov, Sachdev, Senthil)."),
    dict(type="mc", question="Hands-on: near the ±K Goldstone branches, what do the HF and cubic terms do separately?",
         options={"a": "both vanish", "b": "each is ≈ ±37ε (a spurious gap); they cancel — Goldstone modes stay gapless", "c": "both are positive", "d": "HF is zero"},
         answer="b", explanation="The Goldstone theorem holds only for the complete 1/S result."),
    dict(type="mc", question="Why is the first-order HF shift [T†H_HF T]_nn used instead of diagonalising H + H_HF?",
         options={"a": "it is faster", "b": "strict order-by-order 1/S expansion; diagonalising mixes orders and breaks Goldstone modes in non-collinear magnets",
                  "c": "they always agree", "d": "for finite T only"},
         answer="b", explanation="For collinear magnets (H_HF ∝ H) both are identical."),
    dict(type="mc", question="Hands-on: square-lattice AFM in a field — magnons decay at T = 0 …",
         options={"a": "at any field", "b": "only above h* ≈ 0.76 h_s", "c": "never", "d": "only at saturation"},
         answer="b", explanation="Max Γ³ ≈ 0.03 meV (smearing tail) at 0.6 h_s vs ≈ 2 meV at 0.85 h_s (Zhitomirsky, Chernyshev)."),
    dict(type="num", question="Hands-on: square FM (J = −2 meV, K = 0.5 meV, S = 1), k = 0.4·2π/a, T = 40 K, grid 12×12, η = σ = 0.2: Γ⁴ (meV, 2 decimals)?",
         answer=1.30, tol=0.05, explanation="≈ 1.30 meV of thermal magnon–magnon damping."),
    dict(type="num", question="Hands-on: the same ferromagnet at T = 0: Γ⁴ (meV)?",
         answer=0.0, tol=1e-9, explanation="The ferromagnetic quartic vertex conserves magnon number: no decay at T = 0."),
    dict(type="mc", question="Thermal factor of the 2→2 scattering channel X + B → C + D?",
         options={"a": "1 + n_B", "b": "n_B(1+n_C)(1+n_D) − (1+n_B)n_C n_D", "c": "n_C n_D", "d": "1"},
         answer="b", explanation="General rule Π_cr(1+n)Π_an n − Π_cr n Π_an(1+n); positive on shell."),
    dict(type="mc", question="At which order in 1/S is the four-magnon sunset?",
         options={"a": "1/S", "b": "1/S² — the leading damping of collinear magnets", "c": "S⁰", "d": "1/S³"},
         answer="b", explanation="Two quartic vertices (∝ S⁰ each) over an energy ∝ S."),
    dict(type="mc", question="Which CLI options add the cubic terms and the four-magnon damping?",
         options={"a": "--relax --scan", "b": "--hf T --cubic  /  --hf T --damping4 [--grid4 ...]", "c": "--spectral", "d": "--supercell"},
         answer="b", explanation="Output: *_hf.png and *_selfenergy.dat."),
    dict(type="mc", question="Why are Goldstone modes exactly at their zero excluded from the on-shell results?",
         options={"a": "they are unstable", "b": "ε = 0: the on-shell 1/S expansion is singular there (the correction itself must vanish)", "c": "they have no vertex", "d": "they are phonons"},
         answer="b", explanation="Results are reported for ε > 10⁻³ of the bandwidth."),
    dict(type="mc", question="What does symphase.magcheck provide?",
         options={"a": "plots", "b": "independent checks: classical-energy expansion of the HP terms and exact diagonalisation of boson clusters",
                  "c": "DFT inputs", "d": "symmetry tables"},
         answer="b", explanation="Use it to verify vertices for your own models."),
    dict(type="mc", question="Which part is NOT complete at the quartic level?",
         options={"a": "exchange tensors", "b": "single-ion anisotropy (quartic terms harmonic only)", "c": "DMI", "d": "Zeeman"},
         answer="b", explanation="Single-ion quartic terms also carry operator-ordering (S − 1/2) subtleties; they are left out consistently."),
]
build_bank(os.path.join(DATA, "quiz_nonlinear.json"), "Quiz 10 — cubic magnon terms and four-magnon damping", NQ)
quiz_nb("25_quiz_nonlinear_spin_waves.ipynb", "Quiz 10 — cubic magnon terms and four-magnon damping (20 questions)",
        NQ, "nonlinear", "Work through **24_tutorial_cubic_four_magnon_damping.ipynb** first. Questions 4–6, 8, 9, 13 and 14 are hands-on.",
        NL_HELPER)


# =============================================================================
# 26  self-consistent magnon decay;  27 quiz
# =============================================================================
SC_HELPER = 'import math, numpy as np, spglib\nimport matplotlib.pyplot as plt\nfrom symphase.magint import FourMagnonHF, MagnonSelfEnergy\nfrom symphase.magcheck import BosonCluster, classical_expansion\nfrom symphase.magnons import relax_spins, make_path, analyse\nfrom symphase.tensors import read_poscar, read_incar\na = 3.0\n\ndef canted_chain(h=4.2):\n    """AFM chain, XYZ exchange (Jy = 1.3), field h along −z: canted, gapped — the ED test bed"""\n    J = np.diag([1.0, 1.3, 1.0]); S = np.array([1.0, 1.0])\n    pairs = [(0, 1, np.array([a, 0, 0.]), J), (1, 0, np.array([a, 0, 0.]), J)]\n    hfield = np.array([[0, 0, -h]] * 2)\n    dirs, _, _ = relax_spins(np.array([[0, 1, 0.5], [0, -1, 0.5]]), S, [0, 1], pairs, [None, None], hfield=hfield,\n                             steps=200000, tol=1e-15)\n    return FourMagnonHF(np.diag([2 * a, 12, 12]), np.array([[0, 0, 0], [a, 0, 0.]]), [0, 1], dirs, S, pairs, None, hfield)\n\ndef triangular(S=0.5, G=36):\n    """120° triangular AFM (√3×√3 cell of examples/vasp/Triangular_120AFM_INCAR), J = 1 meV"""\n    d = "../examples/vasp/Triangular_120AFM_INCAR"\n    p = read_poscar(f"{d}/POSCAR"); mags = np.array(read_incar(f"{d}/INCAR")["magmom"]).reshape(-1, 3)\n    ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0, 0]))\n    m = analyse(p["lattice"], p["frac"], [0, 0, 0], ["Cr"] * 3, mags, ds.rotations, ds.translations, J=[1.0], spin=S,\n                rmax=3.1, return_model=True)\n    hf = FourMagnonHF.from_model(np.asarray(p["lattice"], float), np.asarray(p["frac"], float), m).solve(grid=(G, G, 1))\n    return hf, MagnonSelfEnergy(hf)\n\ndef canted_square(hr, S=1.0):\n    """square-lattice Heisenberg AFM (J = 1 meV) in a field h = hr·h_s along z (h_s = 8JS), √2 cell"""\n    lat = np.array([[a, a, 0], [a, -a, 0], [0, 0, 12.]])\n    pairs = [(0, 1, np.array(d, float), np.eye(3)) for d in ([a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0])]\n    s = math.sqrt(1 - hr ** 2)\n    hf = FourMagnonHF(lat, np.array([[0, 0, 0], [a, 0, 0.]]), [0, 1], np.array([[s, 0, hr], [-s, 0, hr]]),\n                      np.array([S, S]), pairs, None, np.array([[0, 0, -8 * hr * S]] * 2))\n    return hf, MagnonSelfEnergy(hf), lat\n\ndef square_fm(J=-2.0, K=0.5, S=1.0):\n    """square-lattice ferromagnet, easy axis z (gap 2KS)"""\n    pairs = [(0, 0, np.array([a, 0, 0.]), J * np.eye(3)), (0, 0, np.array([0, a, 0.]), J * np.eye(3))]\n    hf = FourMagnonHF(np.diag([a, a, 12.]), np.zeros((1, 3)), [0], np.array([[0, 0, 1.]]), np.array([S]), pairs,\n                      [np.diag([0, 0, -K])], None)\n    return hf, MagnonSelfEnergy(hf)\n\ndef ed_check(h=4.2, nmax=8):\n    """exact diagonalisation (3-cell cluster) vs bubble + tadpole and four-magnon sunset; returns rows per mode"""\n    hf = canted_chain(h); se = MagnonSelfEnergy(hf)\n    cl = BosonCluster(hf, 3, nmax); qs = cl.kpoints(); k = qs[1]; e = se._mset(k)[0]\n    H2 = cl.quadratic()\n    ed3 = cl.second_order(H2, cl.operator(se.terms3), e)\n    hf.update(qs, 0.0); ed4 = cl.second_order(H2, cl.operator(se.terms4) - cl.operator(cl.hf_terms()), e); hf.hf = None\n    rows = []\n    for n in range(2):\n        _, w, ee = se.cubic_poles(k, n, 0.0, qs=qs); bub = float(np.sum(w / (e[n] - ee)).real)\n        tad = se.tadpole(k, n, 0.0, ks=qs)\n        _, w, ee = se.quartic_poles(k, n, 0.0, qs=qs); sun = float(np.sum(w / (e[n] - ee)).real)\n        rows.append(dict(n=n, eps=e[n], ed3=ed3[n]["second"], bubble=bub, tadpole=tad, ed4=ed4[n]["second"], sunset=sun))\n    return rows\n\ndef tri_velocity(se, hf, kf=0.03):\n    """2S·δε/ε of the three Goldstone branches at small k (x direction): HF (first order), Re Σ³, total"""\n    k = np.array([kf * 2 * np.pi / a, 0, 0])\n    o = se.onshell(k, 0.0, (36, 36, 1), eta=0.01)\n    dhf = hf.first_order_shift([k])[0]\n    S2 = 2 * hf.S[0]\n    return dict(e0=o["e0"], hf=S2 * dhf / o["e0"], re3=S2 * o["re3"] / o["e0"],\n                total=S2 * (dhf + o["re3"] + o["tad"]) / o["e0"], gamma3=o["gamma3"])\n\nfrom symphase.magint import SelfConsistentDecay\n\ndef sc_square(S=1.0, hr=0.85, G=10, eta=0.05):\n    """canted square AFM in a field, self-consistent decay object on a closed G×G grid (η = eta·S)"""\n    hfq, seq, latq = canted_square(hr, S=S)\n    return seq, SelfConsistentDecay(seq, grid=(G, G, 1), eta=eta * S), latq\n\ndef born_vs_sc(sc):\n    """(Born Γ, self-consistent Γ) on the grid (Goldstone points removed)"""\n    sc.solve("onshell"); g0 = sc.G.copy()\n    sc.solve("sc", mix=0.3, maxiter=300, tol=1e-6 * max(1.0, sc.e0.max() / 6.8))\n    m = ~sc.gold\n    return g0[m], sc.G[m].copy()\n'

save([
    md(r"""# SymPhase tutorial 26 — self-consistent magnon decay

The Born (on-shell) decay rate of tutorial 24, $\Gamma_k=-2\,\mathrm{Im}\,\Sigma^{(3)}_k(\varepsilon_k)$ with bare internal magnons, is singular at
decay thresholds (the two-magnon continuum has van Hove edges) and becomes larger than the magnon energy when the coupling is strong.
`SelfConsistentDecay` treats the decay non-perturbatively on a **closed** $k$ grid (all $k$, $q$, $k-q$ on the grid):

* **iΓ approximation** (Chernyshev & Zhitomirsky, RMP 85, 219 (2013)): the internal magnons of the resonant channels carry their own
  damping, $\omega-\varepsilon_q-\varepsilon_{k-q}+i(\Gamma_q+\Gamma_{k-q})/2$, and $\Gamma_k=-2\,\mathrm{Im}\,\Sigma_k(E_k)$ is iterated to self-consistency;
* the energy stays the **$1/S$ on-shell value** $E_k=\varepsilon_k+\delta\varepsilon_{\rm HF}+\mathrm{Re}\,\Sigma^{(3)}(\varepsilon_k)+\Sigma_{\rm tad}$ — evaluating $\mathrm{Re}\,\Sigma$ off-shell would
  mix orders and destroy the Hartree–Fock/cubic Goldstone cancellation (`mode="dyson"` shows this, as a diagnostic);
* the anti-resonant (source) poles are not broadened, and the broadened resonant poles use a short-tailed (Gaussian) line shape of
  FWHM $2\eta+\Gamma_q+\Gamma_{k-q}$ (`profile="lorentz"` for the canonical Lorentzian — see §4 why it fails near Goldstone modes)."""),
    SETUP,
    code(SC_HELPER),
    md("## 1. Born versus self-consistent decay (square AFM at h = 0.85 h_s, S = 1)"),
    code("""se1, sc1, lat1 = sc_square(1.0)
g_born, g_sc = born_vs_sc(sc1)
print(f"Born: max Γ = {g_born.max():.3f}, mean {g_born.mean():.3f} meV;  self-consistent: max {g_sc.max():.3f}, mean {g_sc.mean():.3f} meV")
print(f"{sc1.iterations} iterations, residual {sc1.residual():.1e} meV")
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.scatter(sc1.E1S[~sc1.gold], g_born, s=10, color="0.6", label="Born (on-shell)")
ax.scatter(sc1.E1S[~sc1.gold], g_sc, s=10, color="#16325c", label="self-consistent iΓ")
ax.set_xlabel("E (meV)"); ax.set_ylabel("Γ (meV)"); ax.legend(fontsize=8); ax.set_title("all grid magnons", fontsize=10)"""),
    md(r"""## 2. The Born limit for large S

Interaction effects scale as $1/S$: for $S\to\infty$ the internal widths are negligible on the scale of the continuum and the
self-consistent result must approach the Born one (η scaled with S):"""),
    code("""for S in (1.0, 3.0, 10.0, 100.0):
    _, scS, _ = sc_square(S)
    gb, gs = born_vs_sc(scS)
    print(f"S = {S:5.0f}: ⟨Γ_sc⟩/⟨Γ_Born⟩ = {gs.mean() / gb.mean():.3f}   ({scS.iterations} iterations)")"""),
    md("## 3. Threshold singularities are removed"),
    code("""hrs = [0.72, 0.75, 0.78, 0.8, 0.85, 0.9]; mb, ms = [], []
for hr in hrs:
    _, scH, _ = sc_square(1.0, hr=hr); gb, gs = born_vs_sc(scH); mb.append(gb.max()); ms.append(gs.max())
    print(f"h/h_s = {hr:.2f}: max Γ Born {mb[-1]:.3f}   self-consistent {ms[-1]:.3f} meV")
fig, ax = plt.subplots(figsize=(4.8, 3.2)); ax.plot(hrs, mb, "o-", color="0.55", label="Born"); ax.plot(hrs, ms, "s-", color="#16325c", label="self-consistent")
ax.axvline(0.76, ls=":", color="k"); ax.set_xlabel("h / h_s"); ax.set_ylabel("max Γ (meV)"); ax.legend(fontsize=8)"""),
    md(r"""## 4. Non-collinear magnets: Goldstone modes stay undamped

Triangular AFM, $S=\frac12$: the energies are the converged $1/S$ values (`energy_grid`); compare the line shapes of the internal widths.
The Lorentzian tails act on the very large off-resonant decay weights (the same terms that cancel the ±37ε Hartree–Fock gap) and
damp the Goldstone branches unphysically; the Gaussian keeps them underdamped:"""),
    code("""hf_t, se_t = triangular(S=0.5, G=24)
for prof in ("gauss", "lorentz"):
    scT = SelfConsistentDecay(se_t, grid=(12, 12, 1), eta=0.02, energy_grid=(24, 24, 1), profile=prof)
    scT.solve("onshell"); gB = scT.G.copy()
    scT.solve("sc", mix=0.3, maxiter=300, tol=1e-6)
    sm = (~scT.gold) & (scT.E1S < 0.35)
    print(f"{prof:8s}: near-Goldstone max Γ/E = {(scT.G[sm] / scT.E1S[sm]).max():.3f} (Born {(gB[sm] / scT.E1S[sm]).max():.3f}), "
          f"⟨Γ⟩ = {scT.G[~scT.gold].mean():.3f} meV, residual {scT.residual():.1e}")"""),
    md("## 5. Spectral function with self-consistent internal lines"),
    code("""k5 = np.array([0.35 * 2 * np.pi / 6.0, 0.35 * 2 * np.pi / 6.0, 0])
ws = np.linspace(0.5, 9, 900)
n5 = 1
A_sc = sc1.spectral(k5, n5, ws, mode="sc"); A_b = sc1.spectral(k5, n5, ws, mode="onshell")
fig, ax = plt.subplots(figsize=(5.4, 3.2)); ax.plot(ws, A_b, color="0.55", label="Born lines"); ax.plot(ws, A_sc, color="#16325c", label="self-consistent lines")
ax.set_xlabel("ω (meV)"); ax.set_ylabel("A(k, ω)"); ax.legend(fontsize=8)
print("∫A dω (self-consistent):", round(np.trapezoid(A_sc, ws), 3))"""),
    md("""## Command line
```bash
cd examples/vasp/Triangular_120AFM_INCAR
symphase magnons --J 1 --spin 0.5 --rmax 3.1 --hf 0 --hf-grid 30 30 --cubic --sc-decay --sc-grid 12 12 --se-sigma 0.03 --nq 16 --se-every 2
```
Writes `*_scdecay.dat` (ε, E, Γ_sc, Z) and adds the self-consistent band and widths to `*_hf.png`.
Limits: the energies are the 1/S on-shell values; cubic self-energy only (Γ⁴ is added on top); grid convergence of Γ should be checked
(closed grids are coarse). Continue with **27_quiz_self_consistent_decay.ipynb**."""),
], "26_tutorial_self_consistent_decay.ipynb")

SQ = [
    dict(type="mc", question="Why does the Born (on-shell) decay rate fail near decay thresholds?",
         options={"a": "it is too small", "b": "the two-magnon continuum has van Hove edges: Γ_Born is singular there and can exceed ε",
                  "c": "it ignores temperature", "d": "it is exact"},
         answer="b", explanation="Self-consistency (internal widths) regularises these singularities."),
    dict(type="mc", question="What is the iΓ approximation?",
         options={"a": "Γ = 0", "b": "internal magnons carry their own damping: ω − ε_q − ε_{k−q} + i(Γ_q + Γ_{k−q})/2, iterated to self-consistency",
                  "c": "a Lorentzian fit", "d": "a phonon bath"},
         answer="b", explanation="Chernyshev & Zhitomirsky, Rev. Mod. Phys. 85, 219 (2013)."),
    dict(type="mc", question="Why must the grid be closed (unshifted)?",
         options={"a": "speed", "b": "k, q and k−q must all be grid points, so the internal widths Γ_q are known", "c": "to include temperature", "d": "symmetry"},
         answer="b", explanation="Off-grid momenta (e.g. a band path) use periodic interpolation of the converged grid widths."),
    dict(type="mc", question="Why is the energy kept at the 1/S on-shell value instead of solving the Dyson equation off-shell?",
         options={"a": "it is cheaper", "b": "off-shell Re Σ mixes orders in 1/S and breaks the Hartree–Fock/cubic Goldstone cancellation of non-collinear magnets",
                  "c": "Dyson has no solution", "d": "for finite T"},
         answer="b", explanation="mode='dyson' is kept only as a diagnostic."),
    dict(type="mc", question="Why are the source (all-annihilation) poles not broadened?",
         options={"a": "they are zero", "b": "they are anti-resonant (negative energy): broadening them would leak an unphysical Im Σ at ω > 0", "c": "for speed", "d": "they are broadened"},
         answer="b", explanation="Only decay and absorption channels can be resonant."),
    dict(type="num", question="Hands-on: sc_square(1.0) — Born maximum Γ on the grid (meV, 2 decimals)?",
         answer=2.62, tol=0.03, explanation="2.62 meV (h = 0.85 h_s, S = 1, 10×10 grid)."),
    dict(type="num", question="Hands-on: the same — self-consistent maximum Γ (meV, 2 decimals)?",
         answer=1.30, tol=0.03, explanation="1.30 meV: the self-consistent internal widths halve the extreme Born values."),
    dict(type="num", question="Hands-on: the same — mean self-consistent Γ over the grid magnons (meV, 2 decimals)?",
         answer=0.56, tol=0.02, explanation="0.56 meV vs the Born mean 0.96 meV."),
    dict(type="num", question="Hands-on: ⟨Γ_sc⟩/⟨Γ_Born⟩ for S = 3 (2 decimals)?",
         answer=0.73, tol=0.03, explanation="0.73: still a strong self-consistency correction."),
    dict(type="num", question="Hands-on: ⟨Γ_sc⟩/⟨Γ_Born⟩ for S = 100 (2 decimals)?",
         answer=1.04, tol=0.04, explanation="≈ 1: the Born limit is recovered for large S."),
    dict(type="mc", question="Hands-on: at h = 0.75 h_s (just below the decay threshold) …",
         options={"a": "Born and self-consistent agree", "b": "Born gives max Γ ≈ 1.0 meV (grid/threshold artefact), self-consistent ≈ 0.13 meV", "c": "both vanish exactly", "d": "self-consistent is larger"},
         answer="b", explanation="The threshold singularity of the Born result is removed."),
    dict(type="mc", question="Hands-on: triangular AFM — near-Goldstone modes with the Lorentzian line shape …",
         options={"a": "stay undamped", "b": "become overdamped (Γ/E ≈ 2) — unphysical tails of the large off-resonant weights; the Gaussian keeps Γ/E < 0.01",
                  "c": "vanish", "d": "are unchanged"},
         answer="b", explanation="Hence profile='gauss' is the default."),
    dict(type="mc", question="Which quantity measures self-consistency?",
         options={"a": "the number of iterations", "b": "residual(): max |Γ − (−2 Im Σ(E))| with the converged internal widths", "c": "the energy", "d": "Z"},
         answer="b", explanation="Typically 10⁻⁶ meV after convergence."),
    dict(type="mc", question="What is Z in the output?",
         options={"a": "the magnon number", "b": "the quasiparticle residue 1/(1 − ∂Re Σ/∂ω)", "c": "the coordination number", "d": "the partition function"},
         answer="b", explanation="Z < 1: weight transferred to the two-magnon continuum."),
    dict(type="mc", question="How are widths at off-grid momenta (band paths) obtained?",
         options={"a": "recomputed self-consistently", "b": "periodic linear interpolation of the converged grid widths for the internal lines, then one evaluation", "c": "set to zero", "d": "nearest neighbour"},
         answer="b", explanation="SelfConsistentDecay.at(qs)."),
    dict(type="mc", question="Which CLI option switches on the self-consistent decay?",
         options={"a": "--relax", "b": "--hf T --cubic --sc-decay [--sc-grid ...]", "c": "--damping4", "d": "--spectral"},
         answer="b", explanation="Writes *_scdecay.dat."),
    dict(type="mc", question="What does mode='dyson' do?",
         options={"a": "the recommended method", "b": "a diagnostic: real off-shell root with bare lines (mixes orders; breaks Goldstone near Goldstone modes)", "c": "four-magnon damping", "d": "HF only"},
         answer="b", explanation="Use 'sc' for results."),
    dict(type="mc", question="Which convergence check matters most for self-consistent widths?",
         options={"a": "none", "b": "the closed-grid size (and η): Γ is a BZ sum with discrete poles", "c": "the number of bands", "d": "the temperature"},
         answer="b", explanation="Closed grids are coarse; compare two grid sizes."),
    dict(type="mc", question="Is the four-magnon damping Γ⁴ part of the self-consistency?",
         options={"a": "yes", "b": "no — it is added on top (on-shell); the self-consistency treats the cubic self-energy", "c": "it replaces Γ³", "d": "it is ignored"},
         answer="b", explanation="Γ⁴ is O(1/S²) and smooth."),
    dict(type="mc", question="What happens to the self-consistent result when S → ∞?",
         options={"a": "it diverges", "b": "it approaches the Born result (relative corrections ∝ 1/S)", "c": "it vanishes", "d": "it oscillates"},
         answer="b", explanation="Checked in the tests (S = 100: within 5 %)."),
]
build_bank(os.path.join(DATA, "quiz_scdecay.json"), "Quiz 11 — self-consistent magnon decay", SQ)
quiz_nb("27_quiz_self_consistent_decay.ipynb", "Quiz 11 — self-consistent magnon decay (20 questions)",
        SQ, "scdecay", "Work through **26_tutorial_self_consistent_decay.ipynb** first. Questions 6–12 are hands-on.", SC_HELPER)


# =============================================================================
# 28  MXenes / 2D;  29 quiz
# =============================================================================
MX_HELPER = 'import os, json, shutil, tempfile, numpy as np\nimport matplotlib.pyplot as plt\nfrom symphase import twod\nfrom symphase.twod import (build_mxene, mxene_formula, prompt_mxene, relax_mxene, mxene_hull, load_hull,\n                           plot_phase_diagram, find_layers, vacuum_thickness, monolayer_from_bulk, prototype)\nfrom symphase.hull import HullAnalysis, make_entry\nMX = "../examples/mxene"\n\ndef results(name):\n    """shipped results of `symphase mxene --hull` (Cr2CF2, Ni2NF2, Mo2C)"""\n    return load_hull(os.path.join(MX, name))\n\ndef bulk_formation_energies(name):\n    """formation energies (eV/atom) of the relaxed competing phases of one example"""\n    lib = json.load(open(os.path.join(MX, name, "library_chgnet.json")))\n    ents = {k: make_entry(v["formula"], v["energy"], name=k) for k, v in lib.items()}\n    h = HullAnalysis(list(ents.values()))\n    return {k: (e.composition.reduced_formula, float(h.form_energy(e)), float(h.e_above_hull(e))) for k, e in ents.items()}\n\ndef termination():\n    return json.load(open(os.path.join(MX, "termination_F.json")))\n'

save([
    md(r"""# SymPhase tutorial 28 — MXenes and 2D materials: structures, convex hulls, phase diagrams

`symphase.twod` builds $M_{n+1}X_nT_2$ MXenes (M = transition metal, X = C/N, T = passivation F, O, OH, Cl, … or none), relaxes them,
and places them in the convex hull of the bulk competing phases. On the command line it **asks** for the elements:

```
$ symphase mxene --hull
  transition metal M (e.g. Cr, Mo, Ti; ordered double-M as outer,inner e.g. Mo,Ti): Cr
  X element (C or N) [C]: C
  n in M(n+1)X(n) (1, 2, 3) [1]: 1
  passivation / termination (F, O, OH, Cl, Br, S, none) [F]: F
  termination site (fcc, hcp, top, auto = lowest energy) [auto]: auto
```
Energies: the universal machine-learned potential **CHGNet** (trained on Materials Project PBE/PBE+U relaxations, MP2020-compatible
energy scale), so elements, carbides, nitrides and fluorides enter one hull. It is a screening tool — confirm with VASP
(`--vasp` writes inputs, `--energies` replaces the model energies by yours)."""),
    SETUP,
    code(MX_HELPER),
    md("## 1. Building MXenes — the questions the code asks, answered from a list here"),
    code("""answers = iter(["Cr", "C", "1", "F", "auto"])
spec = prompt_mxene(inp=lambda q: (lambda a: (print(q + a), a)[1])(next(answers)))   # symphase mxene asks the same questions
print(spec)
for args in (("Cr", "C", 1, "F"), ("Ni", "N", 1, "F"), ("Mo", "C", 1, None), ("Mo,Ti", "C", 2, "O"), ("Ti", "C", 1, "OH")):
    st = build_mxene(*args)
    print(f"{mxene_formula(*args):11s} {st.composition.formula:22s} a0 = {st.lattice.a:.3f} Å, layers {len(find_layers(st))}, vacuum {vacuum_thickness(st):.1f} Å")"""),
    md(r"""Termination sites: **fcc** (above the metal of the opposite side, 'model I'), **hcp** (above X, 'model II'), **top** (above the outer metal).

## 2. Relax Cr$_2$CF$_2$ on all sites (CHGNet, in-plane cell + positions)"""),
    code("""r = relax_mxene("Cr", "C", 1, "F", site="auto")
print("lowest:", r["site"], " a =", round(r["structure"].lattice.a, 3), "Å;  hcp − fcc =",
      round(1000 * (r["sites"]["hcp"] - r["sites"]["fcc"]), 1), "meV/atom")"""),
    md("## 3. Convex hull against the bulk Cr–C–F phases (library prototypes, relaxed and cached)"),
    code("""tmp = tempfile.mkdtemp(); cache = os.path.join(tmp, "lib.json")
shutil.copy(os.path.join(MX, "Cr2CF2", "library_chgnet.json"), cache)          # reuse the shipped relaxations
res = mxene_hull({"Cr2CF2": (r["structure"], r["energy"])}, cache=cache, verbose=False)
for row in res["rows"]:
    print(f"{row['label']}: E_form = {row['e_form']:.3f} eV/atom, E_hull = {1000 * row['e_hull']:.0f} meV/atom →",
          " + ".join(f"{v:.2f} {k}" for k, v in row["decomposition"].items()))
plot_phase_diagram(res, title="Cr2CF2 (2D) in the Cr–C–F hull");"""),
    md("## 4. Ni$_2$NF$_2$ and bare Mo$_2$C (shipped results of `symphase mxene --hull`)"),
    code("""for name in ("Ni2NF2", "Mo2C"):
    rr = results(name); row = rr["rows"][0]
    print(f"{name}: a = {rr['mxene']['a']:.3f} Å, E_hull = {1000 * row['e_hull']:.0f} meV/atom → {list(row['decomposition'])}")
    plot_phase_diagram(rr, title=f"{name} (2D)")"""),
    md("## 5. How good are the bulk energies? Formation energies vs experiment"),
    code("""exp = {"CrF2": -2.688, "CrF3": -3.003, "NiF2": -2.250, "Cr3C2": -0.177, "Cr7C3": -0.167, "Cr23C6": -0.147,
       "Mo2C": -0.159, "MoC": -0.145, "Ni3N": 0.002}
for name in ("Cr2CF2", "Ni2NF2", "Mo2C"):
    for k, (f, ef, eh) in bulk_formation_energies(name).items():
        if f in exp:
            print(f"{k:16s} {f:7s} CHGNet {ef:+.3f}   experiment {exp[f]:+.3f} eV/atom   E_hull {1000 * eh:4.0f} meV/atom")"""),
    md(r"""## 6. Termination phase diagram

$\Delta G_F(\Delta\mu_F)=[E(M_2XF_2)-E(M_2X)-2\mu_F^0-2\Delta\mu_F]/A$: F termination wins for $\Delta\mu_F>\Delta\mu_F^*$."""),
    code("""tm = termination()
mus = np.linspace(-5, 0.3, 200)
fig, ax = plt.subplots(figsize=(5.2, 3.5)); ax.axhline(0, color="0.4", label="bare")
for m, d in tm.items():
    ax.plot(mus, (d["dE"] - 2 * mus) / d["area"], lw=2, label=f"{m}F2 (Δμ* = {d['delta_mu_star']:.2f} eV, {d['site']})")
ax.set_xlabel("Δμ_F (eV)"); ax.set_ylabel("ΔG (eV/Å²)"); ax.legend(fontsize=8)"""),
    md("## 7. Generic 2D tools: monolayer from a layered bulk"),
    code("""st = build_mxene("Mo", "C", 1, None, vacuum=3.5); bulk = st.copy(); bulk.make_supercell([1, 1, 2])
print("bulk layers:", len(find_layers(bulk, gap=2.5)))
mono = monolayer_from_bulk(bulk, gap=2.5, vacuum=20)
print("monolayer:", mono.composition.formula, " vacuum", round(vacuum_thickness(mono), 1), "Å")"""),
    md("""## Command line
```bash
symphase mxene                                   # asks for M, X, n, passivation, site
symphase mxene --M Cr --X C --T F --hull --vasp  # Cr2CF2: POSCAR, mxene.json, hull.json, phase_diagram.png, vasp/
symphase mxene --M Ni --X N --T F --hull
symphase mxene --M Mo --X C --T none --hull      # bare Mo2C
symphase mxene --M Mo,Ti --X C --n 2 --T O       # ordered double-M Mo2TiC2O2
symphase mxene ... --hull --energies my_vasp.csv # replace model energies by your DFT (formula/name, eV/atom)
```
Continue with **29_quiz_mxenes_2d.ipynb**."""),
], "28_tutorial_mxenes_2d.ipynb")

XQ = [
    dict(type="mc", question="What is the composition of the MXene built from M = Cr, X = C, n = 1, passivation F?",
         options={"a": "CrCF", "b": "Cr2CF2", "c": "Cr3C2F2", "d": "Cr2C2F"}, answer="b", explanation="M(n+1) X(n) T2."),
    dict(type="mc", question="Hands-on: mxene_formula('Mo,Ti', 'C', 2, 'O') = ?",
         options={"a": "MoTi2C2O2", "b": "Mo2TiC2O2", "c": "Mo2Ti2C2O", "d": "MoTiCO"}, answer="b",
         explanation="Ordered double-M: outer Mo layers, inner Ti."),
    dict(type="mc", question="What is the 'fcc' termination site?",
         options={"a": "on top of the outer metal", "b": "the hollow above the metal of the opposite side (not above X)", "c": "above X", "d": "in the vacuum"},
         answer="b", explanation="'Model I'; hcp = above the X atom ('model II')."),
    dict(type="num", question="Hands-on: Cr2CF2 — hcp minus fcc termination energy (meV/atom, integer)?",
         answer=105, tol=4, explanation="≈ +105 meV/atom: F prefers the fcc hollow."),
    dict(type="num", question="Hands-on: relaxed lattice constant a of Cr2CF2 (Å, 2 decimals)?",
         answer=2.955, tol=0.02, explanation="2.955 Å (CHGNet)."),
    dict(type="num", question="Hands-on: E_hull of Cr2CF2 (meV/atom, integer)?", answer=265, tol=6,
         explanation="265 meV/atom above the Cr–C–F hull."),
    dict(type="mc", question="Hands-on: into which bulk phases does Cr2CF2 decompose?",
         options={"a": "Cr + C + F2", "b": "Cr7C3 + C + CrF2", "c": "Cr3C2 + CrF3", "d": "CrF3 + C"}, answer="b",
         explanation="0.29 Cr7C3 + 0.11 C + 0.60 CrF2 (per atom)."),
    dict(type="num", question="Hands-on: E_hull of Ni2NF2 (meV/atom, integer)?", answer=377, tol=6,
         explanation="377 meV/atom → Ni3N + N2 + NiF2."),
    dict(type="num", question="Hands-on: E_hull of bare 2D Mo2C (meV/atom, integer)?", answer=391, tol=6,
         explanation="391 meV/atom above bulk β-Mo2C — the cost of two bare surfaces."),
    dict(type="mc", question="Why are all three monolayers above the hull?",
         options={"a": "a bug", "b": "2D sheets have surface (unsaturated / terminated) bonds relative to the bulk competing phases — MXenes are metastable, made by etching",
                  "c": "temperature", "d": "magnetism"},
         answer="b", explanation="Typical MXene E_hull values are a few hundred meV/atom."),
    dict(type="num", question="Hands-on: Δμ*_F for Cr2C → Cr2CF2 (eV, 2 decimals)?", answer=-3.90, tol=0.03,
         explanation="F termination is favoured for Δμ_F > −3.90 eV."),
    dict(type="mc", question="What does Δμ_F > Δμ*_F mean?",
         options={"a": "the bare sheet is stable", "b": "fluorination of the bare sheet lowers the free energy (F-terminated surface wins)", "c": "F2 condenses", "d": "nothing"},
         answer="b", explanation="ΔG_F(Δμ) = [E(T) − E(bare) − 2μ⁰ − 2Δμ]/A < 0."),
    dict(type="num", question="Hands-on: CHGNet formation energy of CrF2 (eV/atom, 2 decimals)?", answer=-2.75, tol=0.02,
         explanation="−2.75 eV/atom vs experiment −2.69 eV/atom."),
    dict(type="mc", question="Which energy model does the module use by default, and why?",
         options={"a": "DFT on the fly", "b": "CHGNet universal potential — Materials Project scale, works offline; screening quality", "c": "Lennard-Jones", "d": "none"},
         answer="b", explanation="Materials Project / OQMD may be unreachable; replace by VASP for publication."),
    dict(type="mc", question="How do you replace the model energies by your own VASP results?",
         options={"a": "impossible", "b": "--energies CSV (formula or name, eV/atom), or write_vasp + symphase hull", "c": "--relax", "d": "--spiral"},
         answer="b", explanation="Both MXene and competing phases can be replaced."),
    dict(type="mc", question="How are the competing bulk phases obtained?",
         options={"a": "downloaded at run time", "b": "built-in crystallographic prototypes (AFLOW/ICSD parameters) relaxed with the same model, plus --extra files", "c": "guessed", "d": "only elements"},
         answer="b", explanation="e.g. Cr3C2 (D5_10), Cr7C3 (D10_1), Cr23C6 (D8_4), CrF3 (D0_12), NiF2 rutile, Ni3N, β-Mo2C."),
    dict(type="mc", question="Hands-on: which bulk phase has the largest CHGNet error vs experiment in the shipped examples?",
         options={"a": "CrF2", "b": "MoC (underbound by ≈ 0.15 eV/atom)", "c": "Cr23C6", "d": "NiF2"}, answer="b",
         explanation="It does not enter the Mo2C decomposition, but shows why DFT confirmation matters."),
    dict(type="mc", question="Which cell degrees of freedom relax for a 2D sheet?",
         options={"a": "all", "b": "in-plane only (vacuum kept) — FrechetCellFilter mask", "c": "none", "d": "c only"},
         answer="b", explanation="relax(..., two_d=True)."),
    dict(type="mc", question="What does monolayer_from_bulk do?",
         options={"a": "builds a MXene", "b": "finds the van der Waals gaps of a layered crystal, isolates one layer and adds vacuum", "c": "computes phonons", "d": "nothing"},
         answer="b", explanation="Use with exfoliation_energy for generic 2D materials."),
    dict(type="mc", question="Which command asks interactively for the elements and the passivation?",
         options={"a": "symphase hull", "b": "symphase mxene (without --M/--X/--T)", "c": "symphase run", "d": "symphase quiz"},
         answer="b", explanation="Arguments given on the command line skip the questions; --no-input never asks."),
]
build_bank(os.path.join(DATA, "quiz_mxene.json"), "Quiz 12 — MXenes and 2D materials", XQ)
quiz_nb("29_quiz_mxenes_2d.ipynb", "Quiz 12 — MXenes and 2D materials (20 questions)",
        XQ, "mxene", "Work through **28_tutorial_mxenes_2d.ipynb** first. Questions 2, 4–9, 11, 13 and 17 are hands-on.", MX_HELPER)
