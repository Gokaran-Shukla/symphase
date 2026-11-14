# VASP test cases for `symphase tensors`

Minimal POSCAR / OUTCAR / OSZICAR sets (OUTCARs contain only the blocks SymPhase reads).

| folder | system | expected magnetic space group |
|---|---|---|
| `fcc_Pt` | non-magnetic Pt | Fm-3m1' (225.117) |
| `bcc_Fe_m001` | Fe, m ∥ z, SOC | I4/mm'm' (139.537) |
| `RuO2_Neel001_noSOC` | altermagnet, collinear, Néel ∥ z | P4_2'/mnm' (136.499) |
| `RuO2_Neel100_SOC` | altermagnet, non-collinear, Néel ∥ x | Pnn'm' (58.398) |
| `Mn3Sn_allin` | kagome AFM, all-in triangular order | P6_3/m'm'c' (194.271) |
