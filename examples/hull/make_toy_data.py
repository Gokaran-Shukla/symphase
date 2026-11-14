"""
Generate the TOY example data sets shipped with SymPhase.

!!! The energies are NOT DFT results. They are hand-made, physically plausible
!!! numbers for teaching the workflow (signs and magnitudes resemble PBE data).
!!! For real work use `symphase run <elements>` (OQMD / Materials Project) or
!!! your own VASP runs.

Run:  python make_toy_data.py
"""
import os
import numpy as np
from pymatgen.core import Structure, Lattice, Composition

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "structures"), exist_ok=True)

# toy elemental reference energies (eV/atom), close to PBE magnitudes
MU = {"Ni": -5.57, "Al": -3.75, "Ti": -7.89, "Zr": -8.55, "Hf": -9.95, "Sn": -3.98}


def total_per_atom(formula, ef):
    c = Composition(formula)
    return ef + sum(c.get_atomic_fraction(el) * MU[str(el)] for el in c.elements)


# toy Debye temperatures (K) and sublattice mixing specs for the finite-T examples
THETA = {"Ni": 450, "Al": 428, "Ti": 420, "Zr": 291, "Hf": 252, "Sn": 200,
         "TiNiSn-HH": 405, "ZrNiSn-HH": 365, "HfNiSn-HH": 330, "TiNi2Sn-FH": 380, "ZrNi2Sn-FH": 350,
         "HfNi2Sn-FH": 330, "NiAl-B2": 470, "TiNiSn-other": 395,
         # alloys: θ_D interpolated between their end members
         "(Ti0.5Zr0.5)NiSn-SQS": 385, "(Ti0.75Zr0.25)NiSn-SQS": 395, "(Ti0.25Zr0.75)NiSn-SQS": 375,
         "(TiZr)Ni2Sn-SQS": 365, "(Zr0.5Hf0.5)NiSn-SQS": 348, "(Ti0.5Hf0.5)NiSn-SQS": 368,
         "(TiZrHf)NiSn-HE-SQS": 367, "(TiZrHf)Ni2Sn-HE": 353}
MIX = {"(Ti0.5Zr0.5)NiSn-SQS": "Ti0.5Zr0.5|Ni|Sn", "(Ti0.75Zr0.25)NiSn-SQS": "Ti0.75Zr0.25|Ni|Sn",
       "(Ti0.25Zr0.75)NiSn-SQS": "Ti0.25Zr0.75|Ni|Sn", "(TiZr)Ni2Sn-SQS": "Ti0.5Zr0.5|2*Ni|Sn",
       "(Zr0.5Hf0.5)NiSn-SQS": "Zr0.5Hf0.5|Ni|Sn", "(Ti0.5Hf0.5)NiSn-SQS": "Ti0.5Hf0.5|Ni|Sn",
       "(TiZrHf)NiSn-HE-SQS": "TiZrHf|Ni|Sn", "(TiZrHf)Ni2Sn-HE": "TiZrHf|2*Ni|Sn",
       "TiZr-bcc": "Ti0.5Zr0.5", "TiHf-bcc": "Ti0.5Hf0.5", "ZrHf-bcc": "Zr0.5Hf0.5"}


def write_csv(fn, rows, header):
    with open(os.path.join(HERE, fn), "w") as fh:
        fh.write("# TOY DATA for SymPhase tutorials - not DFT results\n")
        fh.write(header + "\n")
        fh.write("formula,energy,energy_type,name,structure,mixing,debye_T\n")
        for r in rows:
            f, ef, name = r[:3]
            st = r[3] if len(r) > 3 else ""
            th = THETA.get(name, THETA.get(f) if Composition(f).is_element else "")
            fh.write(f"{f},{total_per_atom(f, ef):.6f},per_atom,{name},{st},{MIX.get(name, '')},{th}\n")


# ------------------------------------------------------------------ structures
def save(st, name):
    p = os.path.join("structures", f"{name}.vasp")
    st.to(filename=os.path.join(HERE, p), fmt="poscar")
    return p


a = 5.95
hh = Structure.from_spacegroup("F-43m", Lattice.cubic(a), ["Ti", "Ni", "Sn"],
                               [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]).get_primitive_structure()
fh_ = Structure.from_spacegroup("Fm-3m", Lattice.cubic(6.05), ["Ti", "Ni", "Sn"],
                                [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]).get_primitive_structure()
zhh = hh.copy(); zhh.replace_species({"Ti": "Zr"}); zhh.scale_lattice(hh.volume * 1.07)
hhh = hh.copy(); hhh.replace_species({"Ti": "Hf"}); hhh.scale_lattice(hh.volume * 1.06)
b2 = Structure(Lattice.cubic(3.01), ["Ti", "Ni"], [[0, 0, 0], [.5, .5, .5]])
nial = Structure(Lattice.cubic(2.88), ["Ni", "Al"], [[0, 0, 0], [.5, .5, .5]])
l12 = Structure(Lattice.cubic(3.57), ["Al", "Ni", "Ni", "Ni"],
                [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
ni = Structure(Lattice.cubic(3.52), ["Ni"] * 4, [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]).get_primitive_structure()
al = Structure(Lattice.cubic(4.04), ["Al"] * 4, [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]).get_primitive_structure()
ti = Structure(Lattice.hexagonal(2.93, 4.64), ["Ti", "Ti"], [[1/3, 2/3, .25], [2/3, 1/3, .75]])
sn = Structure.from_spacegroup("Fd-3m", Lattice.cubic(6.65), ["Sn"], [[0, 0, 0]]).get_primitive_structure()

S = {k: save(v, k) for k, v in dict(TiNiSn_HH=hh, TiNi2Sn_FH=fh_, ZrNiSn_HH=zhh, HfNiSn_HH=hhh,
                                    TiNi_B2=b2, NiAl_B2=nial, Ni3Al_L12=l12, Ni_fcc=ni,
                                    Al_fcc=al, Ti_hcp=ti, Sn_diamond=sn).items()}

# ------------------------------------------------------------------ 1. binary Ni-Al
nial_rows = [
    ("Ni", 0.0, "Ni-fcc", S["Ni_fcc"]), ("Al", 0.0, "Al-fcc", S["Al_fcc"]),
    ("Ni", 0.025, "Ni-hcp"), ("Al", 0.03, "Al-hcp"),
    ("Ni3Al", -0.430, "Ni3Al-L12", S["Ni3Al_L12"]), ("Ni3Al", -0.405, "Ni3Al-D019"),
    ("Ni5Al3", -0.505, "Ni5Al3-Pt5Ga3"), ("Ni2Al", -0.470, "Ni2Al-hyp"),
    ("NiAl", -0.665, "NiAl-B2", S["NiAl_B2"]), ("NiAl", -0.540, "NiAl-L10"),
    ("Ni2Al3", -0.585, "Ni2Al3-D513"), ("NiAl2", -0.470, "NiAl2-hyp"),
    ("NiAl3", -0.420, "NiAl3-D011"), ("NiAl4", -0.300, "NiAl4-hyp"),
]
write_csv("NiAl_binary.csv", nial_rows, "# Ni-Al: 14 toy entries")

# ------------------------------------------------------------------ 2. ternary Ti-Ni-Sn
tins = [
    ("Ti", 0.0, "Ti-hcp", S["Ti_hcp"]), ("Ni", 0.0, "Ni-fcc", S["Ni_fcc"]),
    ("Sn", 0.0, "Sn-alpha", S["Sn_diamond"]), ("Ti", 0.06, "Ti-bcc"),
    ("TiNi", -0.380, "TiNi-B2", S["TiNi_B2"]), ("TiNi3", -0.460, "TiNi3-D024"),
    ("Ti2Ni", -0.300, "Ti2Ni-E93"), ("TiNi2", -0.330, "TiNi2-hyp"),
    ("Ni3Sn", -0.240, "Ni3Sn-D019"), ("Ni3Sn2", -0.300, "Ni3Sn2-B81"),
    ("Ni3Sn4", -0.270, "Ni3Sn4-mC14"), ("NiSn", -0.215, "NiSn-hyp"),
    ("Ti3Sn", -0.290, "Ti3Sn-D019"), ("Ti2Sn", -0.360, "Ti2Sn-B82"),
    ("Ti5Sn3", -0.430, "Ti5Sn3-D88"), ("Ti6Sn5", -0.400, "Ti6Sn5-hP22"),
    ("TiSn2", -0.250, "TiSn2-hyp"),
    ("TiNiSn", -0.610, "TiNiSn-HH", S["TiNiSn_HH"]), ("TiNi2Sn", -0.585, "TiNi2Sn-FH", S["TiNi2Sn_FH"]),
    ("TiNiSn2", -0.330, "TiNiSn2-hyp"), ("Ti2NiSn", -0.420, "Ti2NiSn-inverseHeusler"),
    ("TiNiSn", -0.560, "TiNiSn-other"), ("Ti2Ni2Sn", -0.460, "Ti2Ni2Sn-hyp"),
]
write_csv("TiNiSn_ternary.csv", tins, "# Ti-Ni-Sn half-Heusler system: 23 toy entries")

# ------------------------------------------------------------------ 3. quaternary Ti-Zr-Ni-Sn
zr = [
    ("Zr", 0.0, "Zr-hcp"), ("ZrNi", -0.490, "ZrNi-B33"), ("ZrNi3", -0.450, "ZrNi3-D019"),
    ("Zr2Ni", -0.390, "Zr2Ni-C16"), ("Zr5Sn3", -0.610, "Zr5Sn3-D88"), ("ZrSn2", -0.420, "ZrSn2-C54"),
    ("Zr4Sn", -0.300, "Zr4Sn-A15"), ("ZrNiSn", -0.700, "ZrNiSn-HH", S["ZrNiSn_HH"]),
    ("ZrNi2Sn", -0.640, "ZrNi2Sn-FH"), ("TiZr", 0.020, "TiZr-bcc"),
    ("TiZrNi2Sn2", -0.650, "(Ti0.5Zr0.5)NiSn-SQS"), ("Ti3ZrNi4Sn4", -0.628, "(Ti0.75Zr0.25)NiSn-SQS"),
    ("TiZr3Ni4Sn4", -0.672, "(Ti0.25Zr0.75)NiSn-SQS"), ("TiZrNi4Sn2", -0.600, "(TiZr)Ni2Sn-SQS"),
]
write_csv("TiZrNiSn_quaternary.csv", tins + zr, "# Ti-Zr-Ni-Sn: 37 toy entries")

# ------------------------------------------------------------------ 4. five components Ti-Zr-Hf-Ni-Sn
hf = [
    ("Hf", 0.0, "Hf-hcp"), ("HfNi", -0.510, "HfNi-B33"), ("HfNi3", -0.470, "HfNi3-D024"),
    ("Hf2Ni", -0.370, "Hf2Ni-C16"), ("Hf5Sn3", -0.630, "Hf5Sn3-D88"), ("HfSn2", -0.400, "HfSn2-C40"),
    ("HfNiSn", -0.690, "HfNiSn-HH", S["HfNiSn_HH"]), ("HfNi2Sn", -0.620, "HfNi2Sn-FH"),
    ("TiHf", 0.010, "TiHf-bcc"), ("ZrHf", 0.005, "ZrHf-bcc"),
    ("ZrHfNi2Sn2", -0.6965, "(Zr0.5Hf0.5)NiSn-SQS"), ("TiHfNi2Sn2", -0.648, "(Ti0.5Hf0.5)NiSn-SQS"),
    ("TiZrHfNi3Sn3", -0.665, "(TiZrHf)NiSn-HE-SQS"), ("TiZrHfNi6Sn3", -0.605, "(TiZrHf)Ni2Sn-HE"),
]
write_csv("TiZrHfNiSn_5component.csv", tins + zr + hf, "# Ti-Zr-Hf-Ni-Sn: 51 toy entries")

# ------------------------------------------------------------------ 5. toy VASP run folders (Ni-Al)
root = os.path.join(HERE, "vasp_runs_NiAl")
for f, ef, name, *st in nial_rows:
    if st:
        s = Structure.from_file(os.path.join(HERE, st[0]))
    else:  # placeholder cell with the right composition (composition is all the scanner needs)
        c = Composition(f)
        sp = [str(el) for el in c.elements for _ in range(int(c[el]))]
        n = len(sp)
        g = int(np.ceil(n ** (1 / 3)))
        coords = [[(i % g) / g, ((i // g) % g) / g, (i // g // g) / g] for i in range(n)]
        s = Structure(Lattice.cubic(2.6 * g), sp, coords)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    s.to(filename=os.path.join(d, "POSCAR"), fmt="poscar")
    etot = total_per_atom(f, ef) * s.composition.num_atoms
    with open(os.path.join(d, "OUTCAR"), "w") as fh:
        fh.write(" TOY OUTCAR written by make_toy_data.py (not a VASP output)\n")
        fh.write(f"  free  energy   TOTEN  =     {etot:16.8f} eV\n")
        fh.write(f"  energy  without entropy=     {etot:16.8f}  energy(sigma->0) =     {etot:16.8f}\n")
print("toy data written")
