# Magnetic supercells in SymPhase — step-by-step guide

This guide covers magnetic states that the rotating frame (`--spiral`) **cannot** describe:

| state | example | why the rotating frame fails |
|---|---|---|
| **multi-Q** (skyrmion crystal, triple-Q, vortex crystal) | `examples/vasp/Skyrmion_triangular` | several wave vectors at once; one rotation cannot generate the texture |
| **conical spiral** | `examples/vasp/Conical_J1J2_field` | uniform component along the spiral axis (field) – not a planar spiral |
| **spiral distorted by in-plane anisotropy** | `examples/vasp/Distorted_spiral_K` | anisotropy breaks rotation invariance about n → higher harmonics 3q, 5q, … |
| any exchange tensor not invariant about one axis | DMI vectors in several directions | condition Rₙᵀ J Rₙ = J violated |

For all of these SymPhase builds an explicit **magnetic supercell**, finds the classical ground
state inside it, characterises it and runs linear spin-wave theory on it. Every number quoted
below was produced with the shipped examples (`pytest tests/` checks them).

---

## Step 0 — Decide whether you need a supercell

```
Is the state a single planar spiral and is H invariant about one axis n
(Heisenberg J, D ∥ n, anisotropy only along n, no field)?
   ├─ yes → use   symphase magnons ... --spiral auto      (no supercell, any q, exact)
   └─ no  → use   symphase magnons ... --supercell auto   (this guide)
```

SymPhase enforces this itself:

* `--spiral` with a field, or with a non-invariant Hamiltonian, stops with
  `... use the supercell workflow (--supercell auto)`;
* `--relax` in a fixed cell prints
  `NOTE: the Luttinger–Tisza minimum is at q = ..., not q = 0 of this cell ... --supercell auto`
  when the cell is too small for the true ordering vector.

If you are unsure, run `--supercell auto`: it also finds single-Q spirals (only slower).

---

## Step 1 — Prepare the primitive-cell inputs

You need, in one folder:

| file | content | notes |
|---|---|---|
| `POSCAR` | **primitive** crystal cell | non-magnetic atoms are allowed and are carried into the supercell |
| `INCAR` and/or `OUTCAR` | magnetic moments (`MAGMOM`, or site moments from a finished run) | only identifies the magnetic atoms and \|m\| (S = \|m\|/2 unless `--spin`) – the starting directions do **not** matter |
| exchange | `--J J1 J2 ...` (one value per bond class) **or** `--exchange J.dat` | see Step 2 |

Optional terms:

* single-ion anisotropy: `--K 0.4 --K-axis 1 0 0` (easy axis, meV; `--K-local` for local axes);
* magnetic field: `--field Bx By Bz` in Tesla, Zeeman term −g μ_B B·S (`--g 2` by default);
* spin length: `--spin 1` (otherwise |m|/2).

## Step 2 — Get the exchange parameters

1. List the symmetry-distinct bonds of the primitive cell:
   ```bash
   symphase magnons --list-bonds --rmax 6.5
   ```
   Every line is one bond class `J1, J2, …`; bonds in one class are related by a crystal
   operation and must share one J.
2. Obtain J (and D, anisotropy) from DFT, e.g. with
   * the **four-state method** (four non-collinear VASP calculations per bond in a supercell,
     `LNONCOLLINEAR`, `LSORBIT` for DMI), or
   * Green-function codes such as TB2J on top of a Wannier/LCAO Hamiltonian, or
   * fits to total energies of several collinear orders.
3. Write them either as `--J` values in the listed order, or as an exchange file (1-based atoms,
   POSCAR order):
   ```
   # i j n1 n2 n3  J                      Heisenberg
   # i j n1 n2 n3  J  Dx Dy Dz            + DMI  D·(S_i × S_j)
   # i j n1 n2 n3  Jxx Jxy ... Jzz        full 3x3 tensor
   1 1  1  0  0  -1.000   0.000000  1.800000  0.000000
   ```
   Convention: H = Σ_bonds S_i·J_ij·S_j + …, **J > 0 is antiferromagnetic**, energies in meV.
   List each bond once (i→j, R); its reverse is implied.

## Step 3 — Find the ordering wave vector (Luttinger–Tisza)

`--supercell auto` starts with the Luttinger–Tisza (LT) scan: the lowest eigenvalue of the
3N × 3N Fourier-transformed coupling matrix J(q) over the Brillouin zone. Its minimum
`q_LT` is the natural ordering vector; the **star** of q_LT (all crystal rotations of it) lists
the wave vectors a multi-Q state can combine. Output (skyrmion example):

```
1. Luttinger–Tisza minimum: q_LT = [-0.09251 -0.09251  0.] r.l.u.  (λ = ..., λ(q=0) = -3.00000);  star: 3 member(s)
```

* `λ(q_LT) < λ(0)` → the ferromagnetic / q = 0 state is unstable towards modulation;
* the LT energy is a **lower bound** of the classical energy (exact for Bravais lattices without
  anisotropy or field).

## Step 4 — Choose the supercell

`auto` scans diagonal supercells n₁ × n₂ × n₃ along the directions spanned by q_LT and picks the
smallest one in which **every star member** has a commensurate approximant q_c = m/n whose LT
energy is within 1 % of the minimum (`penalty` column):

```
2. supercell search (penalty = LT energy cost of the best commensurate q, relative units):
     (5, 1, 1)        5 cells   penalty   0.14590
     (6, 1, 1)        6 cells   penalty   0.00000
   supercell M = [[6, 0, 0], [0, 1, 0], [0, 0, 1]]
```

Rules built in:

* if the star has several members (hexagonal, tetragonal, cubic lattices) the sizes along
  symmetry-related directions are kept **equal** (n × n), otherwise a triple-Q state cannot fit;
* limit the size with `--max-cells 400` (default);
* choose it yourself with `--supercell 7 1 1`, or with a full integer matrix
  `--supercell 2 1 0  -1 1 0  0 0 1` (rows = new lattice vectors in units of the old ones,
  e.g. √3 × √3 cells).

## Step 5 — Relax the classical spins from many starting states

In the supercell SymPhase minimises the classical energy (L-BFGS on unit vectors; exchange,
anisotropy and field mapped exactly) starting from:

| seed | purpose |
|---|---|
| your input state, tiled | keeps the DFT order as a candidate |
| ferromagnet ∥ field (or z) | field-polarised phase |
| single-Q spiral for each star member, using the **LT eigenvector** | correct polarisation automatically (helix vs cycloid) |
| conical seeds (single-Q + uniform part) | conical phases |
| multi-Q seeds (sum over the star, core-aligned and random phases, + uniform part) | skyrmion / vortex crystals |
| `--restarts N` random states (default 8) | anything unexpected |

The output lists the lowest energies per primitive cell and how many seeds reached the ground
state:

```
4. 58 seeds relaxed (L-BFGS); lowest energies per primitive cell:
          -5.056820 meV   multi-Q (3Q) + 0.0·n̂
          ...
   ground state reached from 16 seed(s);  input state: -4.620747 meV/cell
```

**Check:** the ground state should be reached from **several** independent seeds. If it is reached
only once, increase `--restarts` (e.g. 30) and rerun. All seed energies are written to
`supercell/seeds.dat` – competing states are visible there (skyrmion example at 14 T:
skyrmion crystal −5.0568, fan/conical −4.8671, ferromagnet −4.6207 meV/cell).

## Step 6 — Converge the supercell size

A supercell only allows commensurate q = m/n. For an incommensurate state scan the size and take
the one with the lowest energy **per primitive cell**:

```bash
symphase magnons --J -1.0 0.4 --rmax 6.1 --spin 1 --scan 4 15     # J2 = 0.4: q_exact = 0.14255
```
```
    size          cells   E (meV / primitive cell)   state
    (6, 1, 1)         6            -0.700000       single-Q spiral
    (7, 1, 1)         7            -0.712498       single-Q spiral  <- lowest
    (8, 1, 1)         8            -0.707107       single-Q spiral
    (14, 1, 1)       14            -0.712498       single-Q spiral  <- lowest
  best supercell: (7, 1, 1)  -> rerun with --supercell 7 1 1
```

* The energy is **not monotonic** in n: what matters is how close m/n gets to q_exact (1/7 =
  0.1429 vs 0.1425); multiples of the best n give the same energy.
* Compare with the rotating frame where it applies (`--spiral auto`: −0.712500 meV): the
  7-cell approximant is only 2 × 10⁻⁶ meV/cell above it.
* For 2D textures scan n × n (the scan uses the directions of q_LT; override with `--qdims 0 1`).
* Combine both: `--scan 4 13 --supercell auto` scans first, then runs the full analysis.
* A state that changes between neighbouring sizes near a phase boundary (e.g. 2 instead of 3
  skyrmions at 22 T in a 10 × 10 cell) is a finite-cell effect: compare E per cell in larger cells.

## Step 7 — Identify the state

From the relaxed spins SymPhase computes the spin structure factor on the commensurate grid
(weights sum to 1) and classifies:

| signature | class printed |
|---|---|
| only Q = 0 | `commensurate with the primitive cell (q = 0)` |
| one ±Q pair, no uniform part | `single-Q spiral` |
| one ±Q pair + w(0) > 0, non-coplanar | `conical spiral (uniform component + single Q)` |
| one ±Q pair + w(0) > 0, coplanar | `fan / canted spiral` |
| ±Q plus 3Q, 5Q, … (or 2Q) | `distorted single-Q spiral (harmonics 3q; weight …)` |
| ≥ 2 independent Q | `multi-Q state (n independent Q)` |
| multi-Q and topological charge ≠ 0 | `skyrmion crystal (3Q, N skyrmions per supercell)` |

Also printed: the uniform weight w(0), the net moment per site, the fundamental Q's and their
harmonics, and for quasi-2D lattices with one magnetic atom per cell the lattice
(Berg–Lüscher) **topological charge** per supercell. Plot: `supercell/state.png`
(texture coloured by s_z plus the strongest Fourier components).

Analytic checks you can reproduce:

* conical chain (B = 2 T): cos θ = gμ_B B / (2S(J(0) − J(Q))) = 0.46307, w(0) = cos²θ = 0.2144;
* skyrmion crystal 10 × 10 at 14 T: |Q| fixes three skyrmions per supercell → N_sk = −3.

## Step 8 — Spin waves in the supercell

The relaxed state is fed into LSWT with the supercell model (local frames, Colpa):

```
6. spin waves in the supercell: stable;  zero modes at Γ: 1;  100 folded bands
```

* **stable** (no imaginary modes) confirms that the state is a local minimum. If `UNSTABLE`,
  the relaxation stopped in a saddle point or the cell is wrong → more restarts / other size.
* Zero modes count the continuous symmetries broken by the state (e.g. rotation about B for the
  cone).
* Bands are **folded** into the supercell Brillouin zone (`supercell/bands.png`): a supercell of
  N cells has N × (magnetic atoms) bands.

## Step 9 — Check the state with DFT (optional but recommended)

`supercell/` contains everything for a VASP calculation of the predicted texture:

| file | use |
|---|---|
| `POSCAR` | supercell with all atoms (species grouped) |
| `INCAR` | `LNONCOLLINEAR = .TRUE.`, `ISYM = 0`, non-collinear `MAGMOM` of the ground state, hint for constrained moments |
| `J_super.dat` | the exchange model on the supercell (3 × 3 tensors) – rerun SymPhase on it directly |
| `spins.dat` | Cartesian positions and spin directions |
| `seeds.dat` | energies of all competing states |

Procedure:

1. Copy `POSCAR`, add your `POTCAR`, `KPOINTS` (scale the k-mesh down by the supercell size) and
   merge the `INCAR` tags with your usual settings (`ENCUT`, `EDIFF`, `LSORBIT = .TRUE.` if DMI
   or anisotropy matters).
2. Run with constrained moments first (`I_CONSTRAINED_M = 1`, `M_CONSTR` = MAGMOM, `LAMBDA`
   increased stepwise) to stay in the predicted texture, then compare total energies with the
   competing states from `seeds.dat` (ferromagnet, single-Q spiral) in the **same** supercell.
3. On Ibex use the same job script as for the hull calculations (template in `symphase/auto.py`,
   generated by `symphase run`), changing only the binary – non-collinear runs need `vasp_ncl`:
   ```bash
   #SBATCH --partition=batch --nodes=1 --ntasks=40 --time=24:00:00
   set +e; source /home/shuklagn/intel/oneapi/setvars.sh > /dev/null 2>&1; set -e
   srun /home/shuklagn/v_wan3.1/bin/vasp_ncl > vasp.out
   ```
   (adjust the paths if your `vasp_ncl` lives elsewhere).

## Step 10 — Symmetry and allowed responses of the new state

```bash
symphase tensors -p supercell/POSCAR -i supercell/INCAR
```

For the skyrmion crystal this gives the magnetic space group **P6m'm' (BNS 183.189)**, magnetic
point group 6m'm', no T and no PT symmetry – and an allowed T-odd antisymmetric conductivity
σ_xy, i.e. the (topological) Hall effect of the skyrmion crystal. Use `--analysis spin` for the
spin-group (non-relativistic) view.

---

## Complete examples

```bash
# conical spiral in a field (chain)
cd examples/vasp/Conical_J1J2_field
symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --field 0 0 2 --supercell auto

# spiral distorted by an in-plane easy axis, with a size scan
cd examples/vasp/Distorted_spiral_K
symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --K 0.4 --K-axis 1 0 0 --scan 4 13 --supercell auto

# skyrmion crystal (multi-Q) on a triangular monolayer
cd examples/vasp/Skyrmion_triangular
symphase magnons --exchange J_DMI.dat --rmax 3.1 --spin 1 --field 0 0 14 --supercell auto
symphase tensors -p supercell/POSCAR -i supercell/INCAR
```

Python API (used by tutorial 12):

```python
from symphase.supercell import run_supercell, scan_sizes, report, plot_state, write_outputs
res = run_supercell(lat, frac, types, symbols, mags, rotations, translations, size="auto",
                    exchange=rows, spin=1.0, rmax=3.1, field=[0, 0, 14])
report(res, symbols); plot_state(res); write_outputs(res, symbols, mags, "supercell")
```

## Troubleshooting

| symptom | cause / fix |
|---|---|
| ground state reached from one seed only | rugged landscape: `--restarts 30`, check `seeds.dat` |
| energy per cell changes with size | incommensurate q: `--scan`, use the lowest; quote the convergence |
| `UNSTABLE` spin waves | not a minimum (saddle) or wrong cell size → more restarts, other size |
| auto size too large (> `--max-cells`) | long period: give the size by hand, or a non-diagonal matrix |
| multi-Q expected but single-Q found | supercell not n × n, field window wrong, or single-Q really lower – check `seeds.dat` |
| defects (fewer skyrmions than expected) | finite-cell effect near a transition; compare larger cells |
| `no exchange given` | add `--J` or `--exchange` |

## Method summary

1. LT: λ(q) = min eig J(q), J(q)_{ab} = Σ_R S_a S_b J_ab(R) e^{2πi q·R} (+ anisotropy), grid + Nelder–Mead.
2. Star: q' = q W⁻¹ for all crystal rotations W, ±q identified.
3. Supercell: rows of M·A; atoms mapped (atom, cell t); each primitive bond (i → j, R) becomes one
   bond per cell, d unchanged (Cartesian) → exact mapping, checked by `test_supercell_model_mapping_is_exact`.
4. Energy E = Σ S_a S_b e_a·J·e_b + Σ S_a² e_a·A·e_a − gμ_B Σ S_a B·e_a; L-BFGS on v with e = v/|v|.
5. Structure factor w(Q) = ⟨|N⁻¹ Σ_t e(t) e^{−2πiQ·t}|²⟩_sublattices on Q = M⁻¹n.
6. Topological charge N_sk = (1/4π) Σ_triangles Ω, Ω = 2 atan2(e₁·(e₂×e₃), 1 + e₁·e₂ + e₂·e₃ + e₃·e₁).
7. LSWT in local frames with Zeeman term A_aa += gμ_B B·v_a; Colpa diagonalisation.
