Magnon–phonon coupling in a ferromagnetic chain (toy model, one Fe per cell, a = 3 Å).
J = −5 meV (FM), easy axis K = 0.3 meV along z, S = 1; Born–von Kármán springs kL = 600, kT = 300 meV/Å² (springs.dat);
magnetoelastic derivative ∂J/∂d_x with a symmetric xz element λ = 6 meV/Å (magnetoelastic.dat): a longitudinal
displacement tilts the easy axis and couples the magnon linearly to the longitudinal acoustic phonon.

    symphase magnons --J -5 --K 0.3 --K-axis 0 0 1 --spin 1 --rmax 3.1 --phonons springs.dat --magnetoelastic magnetoelastic.dat

Expected: anticrossing (magnon polaron) gap 1.11 meV at ω ≈ 7.7 meV, k ≈ 0.43 π/a; the gap is linear in λ.
Isotropic exchange striction (`--dJdr 3`) has NO effect here: it couples only non-collinear spins.
