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
