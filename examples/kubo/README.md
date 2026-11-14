# Kubo magnitudes, constrained by symmetry (`symphase kubo`)

| folder | what | command |
|---|---|---|
| `Graphene_KaneMele` | Kane–Mele toy model as Wannier90 `kane_mele_hr.dat` + centres | `symphase kubo compute -d Graphene_KaneMele --kind spin --hr Graphene_KaneMele/kane_mele_hr.dat --centres Graphene_KaneMele/centres.xyz --spin-order blocks --dims 2 --grid 60 60` |
| `Pt_postw90` | fcc Pt + a postw90 SHC Fermi scan of σ^z_xy (**toy numbers**) | `symphase kubo import -d Pt_postw90 --kind spin --shc Pt_postw90/Pt-shc-fermiscan.dat --win Pt_postw90/Pt.win --ef 0` |
| `Fe_postw90` | bcc Fe (m ∥ z) + a postw90 AHC scan (**toy numbers** with small forbidden components) | `symphase kubo import -d Fe_postw90 --kind charge --part odd --ahc Fe_postw90/Fe-ahc-fermiscan.dat --ef 0` |

`symphase kubo plan -d <folder> --kind spin` lists the components a Kubo code must compute (with postw90 input blocks).
Expected: Kane–Mele σ^z_xy = 1 (ħ/e) e²/h (quantised); Pt: one number fixes all six non-zero SHC components;
Fe: the forbidden σ_yz, σ_zx are flagged (0.3 % symmetry violation) and the fitted Hall vector is (0, 0, 751.6) S/cm.
