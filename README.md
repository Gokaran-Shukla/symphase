<p align="center">
  <img src="docs/figures/ternary.png" width="420" alt="Ti–Ni–Sn convex hull">
  <img src="docs/figures/binary.png" width="380" alt="Ni–Al convex hull">
</p>

# SymPhase

**Magnetic symmetry, symmetry-allowed charge / spin / orbital conductivity tensors and convex-hull
phase stability for VASP calculations — elements in, everything out, on all your cores.**

| command | input | output |
|---|---|---|
| `symphase tensors` | `POSCAR`, `OUTCAR`, `OSZICAR` of a VASP run | space group, **magnetic space group** (BNS/OG, type I–IV), magnetic point group, presence of **P, T, PT, T·τ**, mirrors, M·T, C<sub>n</sub>·T, all operations; allowed **σ<sub>ij</sub>**, **σ<sup>k</sup><sub>ij</sub>(spin)**, **σ<sup>k</sup><sub>ij</sub>(orbital)**, T-even / T-odd, anomalous Hall vector, unconventional spin polarisations |
| `symphase hull` | VASP runs, CSV, Materials Project | formation energy, **E<sub>hull</sub>**, decomposition, stability margin, figures for **any number of elements**, high-throughput screening; parallel (all cores or MPI) |
| `symphase run` | **just the elements**, e.g. `Ti Ni Sn` | all phases from OQMD (no key) or Materials Project, hull, figures, POSCARs, VASP inputs + **Ibex job array**, HTML report |

📊 **Presentation:** [`docs/SymPhase_presentation.pdf`](docs/SymPhase_presentation.pdf) (32 slides, Beamer source included)
📓 **Tutorials & quizzes:** [`notebooks/`](notebooks) — 3 tutorials, 2 quizzes × 20 self-checking questions

---

## Installation

```bash
git clone https://github.com/Gokaran-Shukla/symphase
cd symphase
pip install -e .                 # numpy, spglib>=2, pymatgen, matplotlib
pip install -e ".[mp,mpi,notebooks]"   # optional: Materials Project, MPI, Jupyter
```

On Ibex (conda):
```bash
source /home/shuklagn/anaconda3/etc/profile.d/conda.sh
conda create -n symphase python=3.11 -y && conda activate symphase
pip install -e .
```

## 1 · Elements in, everything out

```bash
symphase run Ti Ni Sn                 # or:  symphase run Ti-Ni-Sn
symphase run symphase.in              # file with one line:  elements = Ti Ni Sn
symphase run Ti Ni Sn write_vasp=yes ehull_window=0.05
```
Creates `symphase_Ni-Sn-Ti/` with `SUMMARY.html`, `hull_table.csv`, `figures/`, `structures/`, `vasp/`
(INCAR, KPOINTS, POSCAR, POTCAR.spec per phase + `submit_ibex.sh`) and a cache of the database entries.
Data source: Materials Project if `MP_API_KEY` is set, otherwise **OQMD** (no key needed).

After the Ibex jobs have finished, rebuild the hull purely from your own DFT energies:
```bash
symphase run Ti Ni Sn source=vasp vasp_dir=symphase_Ni-Sn-Ti/vasp outdir=TiNiSn_myDFT
```
All options: `symphase run -h` · example input: [`jobs/symphase.in`](jobs/symphase.in).

## 2 · Magnetic symmetry and response tensors from a VASP run

```bash
cd my_vasp_run/
symphase tensors                              # POSCAR + OUTCAR + OSZICAR
symphase tensors -p CONTCAR --mmin 0.1        # final structure, ignore |m| < 0.1 μB
symphase tensors --mdir 1 1 0                 # collinear moments along [110]
symphase tensors --magmom "0 0 3  0 0 -3  4*0"  # override moments (3N Cartesian)
symphase tensors --frame std --json res.json --out report.txt
```
```python
from symphase import analyze_tensors
r = analyze_tensors("examples/vasp/RuO2_Neel100_SOC")
r["magnetic"]["bns_symbol"], r["tensors"]["charge"]["hall_vector"]   # ("Pnn'm'", ['0', '-σ_xz', '0'])
```

Conventions: J<sub>i</sub> = σ<sub>ij</sub>E<sub>j</sub>, J<sup>S<sub>k</sub></sup><sub>i</sub> = σ<sup>k</sup><sub>ij</sub>E<sub>j</sub> (i flow, j field, k polarisation).
For g′ = gT the T-odd part picks a factor −1 (Seemann *et al.*, PRB 92, 155138 (2015)); Onsager is imposed on σ<sub>ij</sub>.
Moments are treated as axial vectors coupled to the lattice (relativistic analysis; without SOC the true symmetry is a larger spin group).

## 3 · Convex hull for any number of elements

```bash
symphase hull --scan calcs/ -n 40 --plot hull.png        # every VASP run below calcs/
symphase hull --csv energies.csv --bar --surface
symphase hull --csv refs.csv --screen candidates.csv     # E_hull of new candidates
symphase hull --csv data.csv --slice TiNiSn HfNiSn --subsystems 3
srun -n 160 python -m symphase hull --csv refs.csv --screen big.csv   # MPI
```
CSV format: `formula,energy[,energy_type,name,structure]` — energy of the formula as written, or `energy_type=per_atom`.

```python
from symphase.hull import HullAnalysis, load_csv
h = HullAnalysis(load_csv("examples/hull/TiNiSn_ternary.csv"), nproc=-1)
h.print_table(); h.plot(save="hull.pdf")
h.decomposition("Ti2NiSn"); h.hull_formation_energy("TiNi2Sn2")
```

| elements | figure |
|---|---|
| 2 | formation-energy hull curve |
| 3 | Gibbs triangle with shaded facets + 3D hull surface |
| 4 | tetrahedron with tie-lines |
| any | grid of all binary/ternary sub-hulls, E<sub>hull</sub> bar chart, pseudo-binary slices |

### Parallelism
Per-entry E<sub>hull</sub>/decompositions/stability margins, candidate screening, VASP-folder parsing, pseudo-binary hull
points and per-phase symmetry analyses run in parallel: **process** backend on one node (fork, copy-on-write,
`-n`/`SYMPHASE_NPROC`/`SLURM_CPUS_PER_TASK`), **MPI** backend across nodes (`mpi4py`, launched with `srun`/`mpirun`).
Results are identical to serial (tested). Job scripts: [`jobs/ibex_symphase.slurm`](jobs/ibex_symphase.slurm),
[`jobs/ibex_symphase_mpi.slurm`](jobs/ibex_symphase_mpi.slurm).

## Tutorials and quizzes

| notebook | content |
|---|---|
| `01_tutorial_symmetry_tensors.ipynb` | Pt, Fe, altermagnetic RuO₂, Néel-vector rotation, reading operations |
| `02_tutorial_convex_hull.ipynb` | 2-, 3-, 4-, 5-element hulls, slices, screening, parallel benchmark |
| `03_tutorial_automatic_workflow.ipynb` | `symphase run`, OQMD/MP, VASP + Ibex round trip |
| `04_quiz_symmetry_tensors.ipynb` | **20 questions** (conceptual + hands-on) |
| `05_quiz_convex_hull.ipynb` | **20 questions** (conceptual + hands-on) |

```python
from symphase.quiz import Quiz
q = Quiz("hull"); q.answer(10, 42.5); q.hint(11); q.score(); q.widget(3)
```

## Repository layout
```
symphase/        package: tensors.py, hull.py, hull_plots.py, auto.py, parallel.py, quiz.py, cli.py, data/
examples/vasp/   VASP test cases (Pt, Fe, RuO2 ×2, Mn3Sn)      examples/hull/  TOY energy data sets
notebooks/       tutorials + quizzes (+ build_notebooks.py)    docs/           presentation (PDF + .tex)
jobs/            Ibex SLURM scripts, example input              tests/          pytest suite (+ OQMD mock)
```

> ⚠️ `examples/hull/*` contain **toy energies** for teaching (see `make_toy_data.py`), not DFT results.

## Testing
```bash
pytest tests/
```

## How to cite / acknowledge
Please cite the underlying tools: spglib (Togo *et al.*, Sci. Technol. Adv. Mater. Meth. 4, 2384822 (2024)),
pymatgen (Ong *et al.*, Comput. Mater. Sci. 68, 314 (2013)), and the data source you use — OQMD
(Saal *et al.*, JOM 65, 1501 (2013); Kirklin *et al.*, npj Comput. Mater. 1, 15010 (2015)) or the Materials Project
(Jain *et al.*, APL Mater. 1, 011002 (2013)). Magnetic space-group labels: Stokes & Campbell (ISO-MAG) tables.

## License
MIT — © 2026 Gokaran Shukla
