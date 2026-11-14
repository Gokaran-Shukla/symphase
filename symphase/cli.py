"""symphase command-line entry point"""
import sys

USAGE = """SymPhase — symmetry & phase-stability toolkit for VASP

usage:  symphase <command> [options]

commands
  run       elements in, everything out:  symphase run Ti Ni Sn
  hull      convex hull from VASP runs / CSV / Materials Project
  tensors   magnetic + spin-group symmetry, conductivity tensors (POSCAR/INCAR/OUTCAR/OSZICAR)
  magnons   magnon chirality splitting (LSWT, symmetry-resolved exchange)
  quiz      list the built-in quizzes (use them in the notebooks)

  symphase <command> -h   for the options of each command
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print(f"symphase {__version__}")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "tensors":
        from .tensors import main as m
    elif cmd == "hull":
        from .hull import main as m
    elif cmd == "run":
        from .auto import main as m
    elif cmd == "magnons":
        from .magnons import main as m
    elif cmd == "quiz":
        from .quiz import list_quizzes
        list_quizzes()
        return
    else:
        print(f"unknown command '{cmd}'\n")
        print(USAGE)
        sys.exit(2)
    m(rest)


if __name__ == "__main__":
    main()
