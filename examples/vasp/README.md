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
