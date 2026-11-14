# VASP test cases for `symphase tensors`

Minimal POSCAR / OUTCAR / OSZICAR sets (OUTCARs contain only the blocks SymPhase reads); the `*_INCAR` folders
contain only POSCAR + INCAR (moments from `MAGMOM`).

| folder | system | magnetic space group | spin group (no SOC) |
|---|---|---|---|
| `fcc_Pt` | non-magnetic Pt | Fm-3m1' (225.117) | non-magnetic |
| `bcc_Fe_m001` | Fe, m ∥ z, SOC | I4/mm'm' (139.537) | ferromagnet, s-wave |
| `RuO2_Neel001_noSOC` | altermagnet, collinear, Néel ∥ z | P4_2'/mnm' (136.499) | altermagnet, d-wave |
| `RuO2_Neel100_SOC` | altermagnet, non-collinear, Néel ∥ x | Pnn'm' (58.398) | altermagnet, d-wave |
| `Mn3Sn_allin` | kagome AFM, all-in triangular order | P6_3/m'm'c' (194.271) | coplanar |
| `MnTe_altermagnet_INCAR` | NiAs-type MnTe, A-type AFM | — | altermagnet, g-wave |
| `Cr_bcc_AFM_INCAR` | bcc Cr, commensurate AFM | — | AFM [C₂‖τ] |
| `Mn2Au_PT_AFM_INCAR` | Mn₂Au | I4/m'm'm' | PT-AFM |
| `MoO2_monoclinic` | MoO₂, P2₁/c, non-magnetic (POSCAR + INCAR; AMCSD 0011769), unique axis b ∥ y | P2₁/c1' (14.76) | non-magnetic; 13 SHE/OHE components |
| `Triangular_120AFM_INCAR` | 120° triangular AFM (toy Cr layer), non-collinear MAGMOM | — | non-collinear AFM (coplanar), f-wave |
| `CantedChain_DMI_INCAR` | AFM chain + staggered DMI (`J_DM.dat`), canted | — | non-collinear weak FM |
| `Spiral_J1J2_chain` | frustrated J₁–J₂ chain, 1 atom/cell | — | spiral q = 1/6 (`--spiral auto`) |
| `Helix_DMI_chain` | FM chain + uniform DMI (`J_DMI.dat`) | — | incommensurate helix, 95.5 Å |
| `Conical_J1J2_field` | J₁–J₂ chain in B = 2 T | — | conical spiral via `--supercell auto` (analytic cone angle) |
| `Distorted_spiral_K` | J₁–J₂ chain + in-plane easy axis | — | distorted spiral (3q harmonic), `--scan` |
| `Skyrmion_triangular` | triangular FM + interfacial DMI + field | — | 3Q skyrmion crystal, N_sk = −3 per 10×10 cell |
| `MagnonPhonon_FM_chain` | FM chain + springs + magnetoelastic ∂J/∂d | — | magnon polarons, 1.11 meV anticrossing gap |
