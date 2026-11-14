Spiral distorted by an anisotropy in the spiral plane: J1–J2 chain (J1 = −1, J2 = +0.5 meV, S = 1)
with an easy axis K = 0.4 meV along x. The anisotropy breaks the rotation symmetry about the spiral
axis, so `--spiral` refuses; the spiral bunches towards ±x and acquires odd harmonics (3q).

    symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --K 0.4 --K-axis 1 0 0 --supercell auto --scan 4 13

The scan shows E per cell versus supercell length: 6 (and 12) cells are lowest (−0.974242 meV),
i.e. the period stays locked at 6a; the structure factor shows the 3q harmonic (weight ≈ 0.015).
