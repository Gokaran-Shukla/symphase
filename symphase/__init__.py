"""
SymPhase — symmetry & phase-stability toolkit for VASP calculations.

    symphase tensors   crystal + magnetic symmetry and symmetry-allowed charge,
                       spin and orbital conductivity tensors from POSCAR/OUTCAR/OSZICAR
    symphase hull      convex hull / energy above hull for any number of elements
                       (VASP runs, CSV, Materials Project), parallel over all cores / MPI
    symphase run       elements in, everything out: phases from OQMD / Materials Project,
                       hull, figures, structures, VASP + Ibex inputs, HTML report

Python:
    from symphase import analyze_tensors, HullAnalysis, run
"""
__version__ = "1.0.0"
__author__ = "Gokaran Shukla"

from .tensors import analyze_tensors  # noqa: E402,F401


def __getattr__(name):  # lazy: pymatgen is only needed for the hull part
    if name in ("HullAnalysis", "load_csv", "scan_vasp", "make_entry", "load_mp", "entry_from_vasp"):
        from . import hull
        return getattr(hull, name)
    if name in ("run", "read_input"):
        from . import auto
        return getattr(auto, name)
    raise AttributeError(name)
