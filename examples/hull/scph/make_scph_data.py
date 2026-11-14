"""TOY self-consistent-phonon data for SymPhase tutorials (NOT DFT/TDEP/SSCHA results).

Every phase is an anharmonic Einstein solid: 3 independent quartic oscillators per atom,
V = ½ k x² + λ x⁴, treated in the self-consistent harmonic approximation (SCHA).
  * Ni3Al-L12, Ni3Al-D019: TDEP-style folders  T####/outfile.free_energy (T, F_ph, S, Cv; eV/atom)
                            + T####/outfile.U0 (2nd value = static energy + <V − V_eff>, eV/atom)
  * all other entries     : SSCHA-style tables 'T  F_total' in meV/atom (static energy included)
"""
import csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from symphase.qha import quartic_oscillator

TEMPS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600]
SPECIAL = {"Ni3Al-L12": (2050.0, 500.0), "Ni3Al-D019": (1450.0, 1000.0)}
DEFAULT = (2000.0, 600.0)
MASS = 50.0

rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "..", "NiAl_binary.csv")) if not l.startswith("#"))]
out = []
for r in rows:
    name, e = r["name"], float(r["energy"])
    k, lam = SPECIAL.get(name, DEFAULT)
    q = quartic_oscillator(k, lam, MASS, TEMPS)
    Fph, dU = 3 * q["Fph"] / 1000.0, 3 * q["dU"] / 1000.0          # eV/atom (3 oscillators per atom)
    if name in SPECIAL:
        d = os.path.join(HERE, name)
        for T, f, du in zip(TEMPS, Fph, dU):
            sub = os.path.join(d, f"T{T:04d}")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, "outfile.free_energy"), "w") as fh:
                fh.write(f"{T:12.4f} {f:16.10f} {0.0:16.10f} {0.0:16.10f}\n")
            with open(os.path.join(sub, "outfile.U0"), "w") as fh:
                fh.write(f"{e + 3 * 0.0:16.10f} {e + du:16.10f}\n")
        out.append(dict(r, scph=f"scph/{name}", e_static=e, scph_unit="", scph_static=""))
    else:
        fn = os.path.join(HERE, f"{name}.dat")
        with open(fn, "w") as fh:
            fh.write("# TOY SCHA total free energy (static energy included)\n#  T(K)   F (meV/atom)\n")
            for T, f, du in zip(TEMPS, Fph, dU):
                fh.write(f"{T:8.1f} {1000 * (e + f + du):16.6f}\n")
        out.append(dict(r, scph=f"scph/{name}.dat", e_static=e, scph_unit="meV", scph_static="true"))
cols = ["formula", "energy", "energy_type", "name", "scph", "e_static", "scph_unit", "scph_static"]
with open(os.path.join(HERE, "..", "NiAl_scph.csv"), "w") as fh:
    fh.write("# TOY DATA for SymPhase tutorials - not DFT results\n"
             "# Ni-Al with self-consistent anharmonic phonon free energies (scph/: TDEP folders and SSCHA-style tables)\n")
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for o in out:
        w.writerow(o)
print("written", len(out), "entries")
