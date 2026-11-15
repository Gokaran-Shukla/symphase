# MXene examples (`symphase mxene --hull`)

| folder | command | result |
|---|---|---|
| `Cr2CF2/` | `symphase mxene --M Cr --X C --T F --hull --vasp` | a = 2.955 Å (fcc), E_hull = 265 meV/atom → Cr7C3 + C + CrF2 |
| `Ni2NF2/` | `symphase mxene --M Ni --X N --T F --hull --vasp` | a = 3.029 Å (fcc), E_hull = 377 meV/atom → Ni3N + N2 + NiF2 |
| `Mo2C/`   | `symphase mxene --M Mo --X C --T none --hull --vasp` | a = 2.971 Å, E_hull = 391 meV/atom → bulk β-Mo2C |

Each folder: relaxed `POSCAR`, `mxene.json` (energies of all termination sites), `library_chgnet.json` (relaxed competing
phases, reused as cache), `hull.json`, `phase_diagram.png`, `vasp/` (POSCAR + INCAR template for the DFT follow-up).
`termination_F.json`: bare vs F-terminated Cr2C, Ni2N, Mo2C (Δμ_F* where fluorination wins).
Energies: CHGNet universal potential (Materials Project PBE/PBE+U scale) — screening quality; confirm with VASP.
