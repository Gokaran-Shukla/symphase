Multi-Q state: Néel skyrmion crystal on a triangular monolayer (a = 3 Å, toy model).
J = −1 meV (FM nearest neighbour), interfacial DMI D_ij = D (ẑ × r̂_ij) with |D| = 1.8 meV (J_DMI.dat),
S = 1, field along z.

    symphase magnons --exchange J_DMI.dat --rmax 3.1 --spin 1 --field 0 0 14 --supercell auto

Expected: Luttinger–Tisza q ≈ (0.0925, 0.0925, 0) with a 3-member star -> 10 x 10 supercell;
ground state = skyrmion crystal (3Q + uniform component), topological charge −3 per supercell,
E = −5.0568 meV/cell (fan/conical −4.8671, ferromagnet −4.6207); stable spin waves.
Field scan: single-Q cycloid at 0 T, skyrmion crystal ≈ 8–20 T, field-polarised FM at 28 T.

Next: symmetry of the skyrmion crystal and its allowed (topological) Hall response:

    symphase tensors -p supercell/POSCAR -i supercell/INCAR     # -> P6m'm' (BNS 183.189), σ_xy allowed
