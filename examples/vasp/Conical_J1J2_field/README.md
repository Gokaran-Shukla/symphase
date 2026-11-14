Conical spiral: the frustrated J1–J2 chain (J1 = −1, J2 = +0.5 meV, S = 1) in a field B = 2 T along z.
The field adds a uniform component along z; the planar rotating-frame ansatz (`--spiral`) cannot
describe it, so SymPhase refuses `--spiral` with a field and the supercell workflow is used:

    symphase magnons --J -1.0 0.5 --rmax 6.1 --spin 1 --field 0 0 2 --supercell auto

Expected: 6-cell supercell, conical spiral q = 1/6, cone angle cos θ = gμ_B B / (2S(J(0) − J(Q)))
= 0.46307, E = −0.803609 meV per cell (both analytic), one Goldstone mode (rotation about B).
