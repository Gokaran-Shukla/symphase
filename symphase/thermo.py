# -*- coding: utf-8 -*-
"""
symphase.thermo — finite-temperature convex hulls
=================================================

    G_i(T) = E_i + F_vib,i(T) − T S_config,i            (per atom)

Configurational (mixing) entropy, ideal solution on sublattices
    S_config = −k_B Σ_s m_s Σ_a x_{s,a} ln x_{s,a}  /  Σ_s m_s
    given per entry as a sublattice spec  "Ti0.5Zr0.5 | Ni | Sn"  (optional multiplicity
    "2*Ni"), or directly as s_config (k_B per atom).  CSV columns: mixing, s_config.

Vibrational free energy (choose with vib=)
    "debye"   : Debye model with θ_D per entry (CSV column debye_T, K); compounds without θ_D
                use the composition-weighted mean of their elements (ΔF_vib ≈ 0 for them)
    "phonopy" : phonon DOS 'total_dos.dat' or 'thermal_properties.yaml' (CSV column phonons)
    "bartel"  : SISSO descriptor of Bartel et al., Nat. Commun. 9, 4168 (2018) via pymatgen's
                GibbsComputedStructureEntry (needs structures; 300–2000 K)
    "none"    : only configurational entropy

The hull is rebuilt at every temperature (in parallel over entries, as at T = 0);
`stabilisation_temperature` finds where a phase reaches the hull by bisection.
"""

import math
import os
import re

import numpy as np

from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

from .hull import HullAnalysis, entry_name

KB = 8.617333262e-5          # eV/K
THZ_TO_EV = 4.135667696e-3   # h * 1 THz in eV
KJMOL_TO_EV = 0.0103642688


# ----------------------------------------------------------------------------
#  configurational entropy
# ----------------------------------------------------------------------------
def parse_mixing(spec):
    """'Ti0.5Zr0.5 | 2*Ni | Sn' -> (S_config in k_B per atom, composition per formula)"""
    S, natoms = 0.0, 0.0
    total = Composition()
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        m = 1.0
        mm = re.match(r"^\s*([\d.]+)\s*\*\s*(.+)$", part)
        if mm:
            m, part = float(mm.group(1)), mm.group(2)
        c = Composition(part).fractional_composition
        S += -m * sum(x * math.log(x) for x in c.values() if x > 0)
        natoms += m
        total += c * m
    return S / natoms, total


def s_config_of(entry):
    d = entry.data
    if d.get("s_config") not in (None, ""):
        return float(d["s_config"])
    if d.get("mixing"):
        S, comp = parse_mixing(d["mixing"])
        if not comp.fractional_composition.almost_equals(entry.composition.fractional_composition,
                                                         rtol=0.0, atol=2e-3):
            raise ValueError(f"{entry_name(entry)}: mixing spec '{d['mixing']}' has composition "
                             f"{comp.reduced_formula}, entry is {entry.composition.reduced_formula}")
        return S
    return 0.0


# ----------------------------------------------------------------------------
#  vibrational free energy
# ----------------------------------------------------------------------------
def _debye_function(x, n=400):
    if x <= 0:
        return 1.0
    t = np.linspace(1e-8, x, n)
    return 3.0 / x ** 3 * np.trapezoid(t ** 3 / np.expm1(t), t)


def debye_free_energy(T, theta):
    """F_vib per atom (eV), including zero-point energy 9/8 k_B θ"""
    zp = 9.0 / 8.0 * KB * theta
    if T <= 0:
        return zp
    x = theta / T
    return zp + KB * T * (3.0 * math.log(-math.expm1(-x)) - _debye_function(x))


class PhononF:
    """F_vib(T) per atom from phonopy output"""

    def __init__(self, path):
        self.path = path
        if path.endswith((".yaml", ".yml")):
            self._from_yaml(path)
        else:
            d = np.loadtxt(path)
            nu, g = d[:, 0], d[:, 1]
            keep = nu > 1e-4
            self.nu, self.g = nu[keep], g[keep]
            nmodes = np.trapezoid(self.g, self.nu)
            self.natom = nmodes / 3.0
            self.mode = "dos"

    def _from_yaml(self, path):
        T, F, natom = [], [], None
        with open(path) as fh:
            for line in fh:
                m = re.match(r"\s*natom:\s*(\d+)", line)
                if m:
                    natom = int(m.group(1))
                m = re.match(r"\s*-\s*temperature:\s*([-\d.Ee+]+)", line)
                if m:
                    T.append(float(m.group(1)))
                m = re.match(r"\s*free_energy:\s*([-\d.Ee+]+)", line)
                if m:
                    F.append(float(m.group(1)))
        if not natom or not T:
            raise ValueError(f"{path}: not a phonopy thermal_properties.yaml")
        self.T, self.F = np.array(T), np.array(F) * KJMOL_TO_EV / natom
        self.natom, self.mode = natom, "table"

    def __call__(self, T):
        if self.mode == "table":
            return float(np.interp(T, self.T, self.F))
        e = self.nu * THZ_TO_EV
        f = 0.5 * e if T <= 0 else 0.5 * e + KB * T * np.log(-np.expm1(-e / (KB * T)))
        return float(np.trapezoid(self.g * f, self.nu) / self.natom)


# ----------------------------------------------------------------------------
#  finite-T hull
# ----------------------------------------------------------------------------
class FiniteTHull:
    def __init__(self, entries, vib="none", mixing=True, nproc=None, debye_default=None):
        self.entries = list(entries)
        self.vib = vib
        self.mixing = mixing
        self.nproc = nproc
        self.notes = []
        self.S = np.array([s_config_of(e) if mixing else 0.0 for e in self.entries])
        if vib == "debye":
            th = {}
            for e in self.entries:
                if e.composition.is_element and e.data.get("debye_T") not in (None, ""):
                    th.setdefault(str(e.composition.elements[0]), float(e.data["debye_T"]))
            self.theta = []
            n_fb = 0
            for e in self.entries:
                v = e.data.get("debye_T")
                if v not in (None, ""):
                    self.theta.append(float(v))
                    continue
                if all(str(el) in th for el in e.composition.elements):
                    self.theta.append(sum(e.composition.get_atomic_fraction(el) * th[str(el)]
                                          for el in e.composition.elements))
                    n_fb += 1
                elif debye_default:
                    self.theta.append(float(debye_default))
                    n_fb += 1
                else:
                    raise ValueError(f"{entry_name(e)}: no debye_T and no elemental Debye temperatures")
            if n_fb:
                self.notes.append(f"{n_fb} entries without debye_T use the mean of their elements' θ_D")
        elif vib == "phonopy":
            self.ph = []
            for e in self.entries:
                p = e.data.get("phonons")
                if not p:
                    raise ValueError(f"{entry_name(e)}: vib='phonopy' needs a 'phonons' file for every entry")
                p = p if os.path.isabs(p) else os.path.join(e.data.get("csv_dir", "."), p)
                self.ph.append(PhononF(p))
        elif vib == "bartel":
            if not all(isinstance(e, ComputedStructureEntry) for e in self.entries):
                raise ValueError("vib='bartel' needs structures for every entry (CSV 'structure' column or VASP runs)")
        elif vib != "none":
            raise ValueError(f"unknown vib model {vib}")

    def energies(self, T):
        """total G (eV per entry formula) at temperature T"""
        n = np.array([e.composition.num_atoms for e in self.entries])
        if self.vib == "bartel" and T > 0:
            from pymatgen.entries.computed_entries import GibbsComputedStructureEntry
            Tb = min(max(T, 300.0), 2000.0)
            if Tb != T:
                self.notes.append(f"Bartel model clipped to {Tb:.0f} K (valid 300-2000 K)")
            g = GibbsComputedStructureEntry.from_entries(self.entries, temp=Tb)
            gmap = {entry_name(x): x.energy_per_atom for x in g}
            Ga = np.array([gmap.get(entry_name(e), np.nan) for e in self.entries])
            if np.isnan(Ga).any():
                raise ValueError("Bartel model failed for some entries")
        else:
            Ga = np.array([e.energy_per_atom for e in self.entries])
            if self.vib == "debye":
                Ga = Ga + np.array([debye_free_energy(T, th) for th in self.theta])
            elif self.vib == "phonopy":
                Ga = Ga + np.array([p(T) for p in self.ph])
        Ga = Ga - T * KB * self.S
        return Ga * n

    def hull_at(self, T):
        E = self.energies(T)
        ents = []
        for e, g in zip(self.entries, E):
            ents.append(ComputedEntry(e.composition, float(g), entry_id=e.entry_id, data=dict(e.data)))
        return HullAnalysis(ents, nproc=self.nproc)

    def scan(self, temps):
        """E_hull(T) for every entry: dict(name -> array), and stable sets per T"""
        temps = list(temps)
        names = [entry_name(e) for e in self.entries]
        eh = {n: np.zeros(len(temps)) for n in names}
        stable = []
        for k, T in enumerate(temps):
            h = self.hull_at(T)
            rows = {r["name"]: r for r in h.table()}
            for n in names:
                eh[n][k] = rows[n]["e_hull"]
            stable.append(sorted(r["formula"] for r in rows.values() if r["stable"]))
        self.temps, self.ehull_T, self.stable_T = np.array(temps, float), eh, stable
        return eh

    def stabilisation_temperature(self, name, Tmax=None, tol=0.5):
        """lowest T at which the entry reaches the hull (bracketed by the scanned temperatures)"""
        def eh(T):
            r = {x["name"]: x for x in self.hull_at(T).table()}[name]
            return r["e_hull"]
        if eh(0.0) <= 1e-6:
            return 0.0
        lo, hi = 0.0, None
        if getattr(self, "temps", None) is not None and name in getattr(self, "ehull_T", {}):
            v = self.ehull_T[name]
            idx = np.where(v <= 1e-6)[0]
            if len(idx):
                hi = float(self.temps[idx[0]])
                lo = float(self.temps[idx[0] - 1]) if idx[0] > 0 else 0.0
        if hi is None:
            Tmax = Tmax or 3000.0
            if eh(Tmax) > 1e-6:
                return None
            hi = Tmax
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if eh(mid) > 1e-6:
                lo = mid
            else:
                hi = mid
        return hi

    def destabilisation_temperature(self, name, Tmax, tol=0.5):
        """highest T up to which an entry stable at 0 K stays on the hull"""
        def eh(T):
            return {x["name"]: x for x in self.hull_at(T).table()}[name]["e_hull"]
        if eh(Tmax) <= 1e-6:
            return None
        lo, hi = 0.0, Tmax
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if eh(mid) <= 1e-6:
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------ report / plots
    def report(self, file=None, max_rows=40, Tmax=3000.0):
        import sys
        out = file or sys.stdout
        print(f"\nFinite-temperature hull  (vibrations: {self.vib}, mixing entropy: "
              f"{'on' if self.mixing else 'off'})", file=out)
        hdr = "  " + f"{'name':<26s}" + "".join(f"{T:>9.0f}K" for T in self.temps) + "   S_conf(kB/at)   T_stab"
        print(hdr, file=out)
        print("  " + "-" * (len(hdr) - 2), file=out)
        order = sorted(self.ehull_T, key=lambda n: (self.ehull_T[n].max() > 1e-9, self.ehull_T[n].min()))
        always = [n for n in order if self.ehull_T[n].max() <= 1e-9]
        print(f"  stable at all temperatures ({len(always)}): " + ", ".join(always), file=out)
        shown = 0
        self.t_stab = {n: 0.0 for n in always}
        for n in order:
            v = self.ehull_T[n]
            if n in always or shown >= max_rows or (v.min() > 0.2):
                continue
            k = [entry_name(e) for e in self.entries].index(n)
            if v[0] <= 1e-6:                      # stable at 0 K, leaves the hull later
                td = self.destabilisation_temperature(n, Tmax=self.temps.max())
                ts = 0.0
                tss = f"stable ≤ {td:.0f} K" if td else "on hull"
            else:
                ts = self.stabilisation_temperature(n) if v.min() <= 1e-6 else None
                tss = "-" if ts is None else f"{ts:7.0f} K"
            self.t_stab[n] = ts
            print("  " + f"{n:<26s}" + "".join(f"{1000*x:10.1f}" for x in v) + f"   {self.S[k]:10.3f}     {tss}",
                  file=out)
            shown += 1
        print("  (E_hull in meV/atom;  T_stab = lowest temperature at which the phase reaches the hull)", file=out)
        for nt in sorted(set(self.notes)):
            print("  note: " + nt, file=out)

    def plot_ehull_vs_T(self, names=None, save=None, nmax=10, Tgrid=None, figsize=(9.6, 4.8), legend_out=True):
        import matplotlib.pyplot as plt
        from . import hull_plots  # noqa: F401  (style)
        Tg = np.linspace(self.temps.min(), self.temps.max(), 41) if Tgrid is None else Tgrid
        allnames = [entry_name(e) for e in self.entries]
        if names is None:
            cand = [n for n in allnames if self.ehull_T[n].min() < 0.08 and self.ehull_T[n].max() > 1e-6]
            names = sorted(cand, key=lambda n: self.ehull_T[n][0])[:nmax]
        curves = {n: [] for n in names}
        for T in Tg:
            rows = {r["name"]: r for r in self.hull_at(T).table()}
            for n in names:
                curves[n].append(1000 * rows[n]["e_hull"])
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("tab10")
        for k, n in enumerate(names):
            ax.plot(Tg, curves[n], lw=2.2, color=cmap(k % 10), label=n)
            ts = getattr(self, "t_stab", {}).get(n)
            if ts:
                ax.plot(ts, 0, "o", color=cmap(k % 10), ms=7, mec="white", zorder=5)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("temperature (K)")
        ax.set_ylabel("E$_{hull}$(T) (meV/atom)")
        ax.set_xlim(Tg[0], Tg[-1])
        ax.set_ylim(bottom=-1)
        if legend_out:
            ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        else:
            ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.set_title(f"energy above the finite-T hull  (vib: {self.vib}, mixing: {'on' if self.mixing else 'off'})",
                     fontsize=11.5)
        if save:
            fig.savefig(save)
        return fig

    def plot_hull_at(self, T, save=None, **kw):
        fig = self.hull_at(T).plot(**kw)
        ax = fig.axes[0]
        ax.set_title(ax.get_title() + f"  (T = {T:.0f} K)")
        if save:
            fig.savefig(save)
        return fig
