# -*- coding: utf-8 -*-
"""
symphase.qha — quasi-harmonic and anharmonic free energies
==========================================================

Quasi-harmonic approximation (QHA): the phonon free energy depends on volume, and

    G(T, P=0) = min_V  F(V, T),     F(V, T) = E_static(V) + F_vib(V, T) + F_anh(V, T)

(per atom, eV).  Thermal expansion, bulk modulus and heat capacity follow from V(T) and G(T):

    β(T) = d ln V / dT  (volumetric; linear α = β/3),   B_T = V ∂²F/∂V² |_{V(T)},   C_P = −T ∂²G/∂T²

Static energy  E_static(V): third-order Birch–Murnaghan equation of state, fitted to E–V data
               or given by (E0, V0, B0, B0').
Vibrational    F_vib(V, T), either
  * tabulated  : phonopy thermal_properties.yaml at several volumes (+ e-v.dat), interpolated in V
                 by a polynomial at every T (the phonopy-QHA workflow), or
  * Debye–Grüneisen model: F_vib = Debye(θ(V)), γ(V) = γ0 (V/V0)^q,
                 θ(V) = θ0 exp[γ0 (1 − (V/V0)^q) / q]        (θ0 (V0/V)^γ0 for q = 0).
Anharmonic     intrinsic anharmonicity beyond QHA (lowest order in T):
                 F_anh = −(3/2) k_B a(V) T²,   a(V) = a0 (V/V0)^m   (a > 0 lowers F;
                 S_anh = 3 k_B a T,  C_anh = 3 k_B a T),
               or a table F_anh(T) (e.g. from TDEP / thermodynamic integration, eV per atom).

Used by symphase.thermo (vib = "qha" / "qha-debye", column anh_a) for finite-T hulls.
"""

import glob
import math
import os
import re

import numpy as np

from .thermo import KB, PhononF, debye_free_energy

EV_A3_TO_GPA = 160.21766208


# ----------------------------------------------------------------------------
#  equation of state
# ----------------------------------------------------------------------------
def birch_murnaghan(V, E0, V0, B0, Bp):
    """3rd-order Birch–Murnaghan E(V); B0 in eV/Å³"""
    eta2 = (V0 / np.asarray(V, float)) ** (2.0 / 3.0)
    x = eta2 - 1.0
    return E0 + 9.0 * V0 * B0 / 16.0 * (x ** 3 * Bp + x ** 2 * (6.0 - 4.0 * eta2))


def fit_eos(V, E):
    """(E0, V0, B0 [eV/Å³], B0') of a Birch–Murnaghan fit"""
    from scipy.optimize import curve_fit
    V, E = np.asarray(V, float), np.asarray(E, float)
    c = np.polyfit(V, E, 2)
    V0 = -c[1] / (2 * c[0]) if c[0] > 0 else V[np.argmin(E)]
    B0 = max(2 * c[0] * V0, 1e-3)
    p, _ = curve_fit(birch_murnaghan, V, E, p0=[E.min(), V0, B0, 4.0], maxfev=20000)
    return tuple(float(x) for x in p)


# ----------------------------------------------------------------------------
#  Debye–Grüneisen
# ----------------------------------------------------------------------------
def debye_theta(V, V0, theta0, gamma0, q=1.0):
    r = np.asarray(V, float) / V0
    if abs(q) < 1e-12:
        return theta0 * r ** (-gamma0)
    return theta0 * np.exp(gamma0 * (1.0 - r ** q) / q)


# ----------------------------------------------------------------------------
#  QHA driver
# ----------------------------------------------------------------------------
class QHA:
    """Quasi-harmonic (+ anharmonic) free energy of one phase, per atom.

    volumes, energies : static E–V points (Å³/atom, eV/atom), or None with eos=(E0, V0, B0, B0')
    fvib              : list of callables F_vib(T) (eV/atom), one per volume (e.g. PhononF), or None
    debye             : dict(theta0=K, gamma0=..., q=1.0) for the Debye–Grüneisen model
    anh_a, anh_m      : intrinsic anharmonicity a0 (1/K) and its volume exponent m
    anh_table         : (T, F_anh) arrays, eV/atom, added at every volume
    """

    def __init__(self, volumes=None, energies=None, fvib=None, debye=None, eos=None, anh_a=0.0, anh_m=0.0,
                 anh_table=None, poly_degree=3):
        if eos is None:
            if volumes is None or energies is None:
                raise ValueError("give E–V data or eos=(E0, V0, B0, B0')")
            eos = fit_eos(volumes, energies)
        self.E0, self.V0, self.B0, self.Bp = eos
        self.volumes = None if volumes is None else np.asarray(volumes, float)
        self.fvib = fvib
        self.debye = debye
        if fvib is not None and (self.volumes is None or len(fvib) != len(self.volumes)):
            raise ValueError("fvib needs one F_vib(T) per volume")
        self.anh_a, self.anh_m = float(anh_a or 0.0), float(anh_m or 0.0)
        self.anh_table = None if anh_table is None else (np.asarray(anh_table[0], float),
                                                          np.asarray(anh_table[1], float))
        self.deg = poly_degree
        if fvib is not None:
            self.Vlo, self.Vhi = self.volumes.min(), self.volumes.max()
        else:
            self.Vlo, self.Vhi = 0.80 * self.V0, 1.25 * self.V0
        self._cache = {}
        self.warnings = []

    # ---------------------------------------------------------------- pieces
    def E_static(self, V):
        return birch_murnaghan(V, self.E0, self.V0, self.B0, self.Bp)

    def _fvib_poly(self, T):
        key = round(float(T), 8)
        if key not in self._cache:
            f = np.array([F(T) for F in self.fvib])
            deg = min(self.deg, len(self.volumes) - 1)
            self._cache[key] = np.polyfit(self.volumes - self.V0, f, deg)
        return self._cache[key]

    def F_vib(self, V, T):
        if self.fvib is not None:
            return np.polyval(self._fvib_poly(T), np.asarray(V, float) - self.V0)
        if self.debye is not None:
            th = debye_theta(V, self.V0, self.debye["theta0"], self.debye.get("gamma0", 0.0),
                             self.debye.get("q", 1.0))
            return np.vectorize(lambda t: debye_free_energy(T, t))(th)
        return np.zeros_like(np.asarray(V, float))

    def F_anh(self, V, T):
        out = -1.5 * KB * self.anh_a * (np.asarray(V, float) / self.V0) ** self.anh_m * T * T
        if self.anh_table is not None:
            out = out + np.interp(T, *self.anh_table)
        return out

    def F(self, V, T):
        return self.E_static(V) + self.F_vib(V, T) + self.F_anh(V, T)

    # ---------------------------------------------------------------- minimisation
    def equilibrium(self, T):
        """(V(T), G(T)) per atom at P = 0"""
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(lambda v: float(self.F(v, T)), bounds=(self.Vlo, self.Vhi), method="bounded",
                            options=dict(xatol=1e-9 * self.V0))
        if min(r.x - self.Vlo, self.Vhi - r.x) < 1e-4 * self.V0:
            msg = f"T = {T:.0f} K: F(V) minimum at the edge of the volume range (extrapolation)"
            if msg not in self.warnings:
                self.warnings.append(msg)
        return float(r.x), float(r.fun)

    def G(self, T):
        """G(T) per atom (eV) at P = 0"""
        return self.equilibrium(T)[1]

    def delta_G(self, T):
        """G(T) − E0: what is added to a static DFT energy taken at the EOS minimum"""
        return self.G(T) - self.E0

    def properties(self, temps, dT=None):
        """dict of arrays: T, V (Å³/atom), G (eV/atom), beta = d lnV/dT (volumetric, 1/K),
        alpha_lin = beta/3 (linear, 1/K), B_T (GPa), Cp (k_B/atom)"""
        T = np.asarray(temps, float)
        dT = dT or 2.0
        V, G, a, B, Cp = [], [], [], [], []
        for t in T:
            v0, g0 = self.equilibrium(t)
            vp, gp = self.equilibrium(t + dT)
            if t - dT >= 0:
                vm, gm = self.equilibrium(t - dT)
                alpha = (math.log(vp) - math.log(vm)) / (2 * dT)
                cp = -t * (gp - 2 * g0 + gm) / dT ** 2 / KB
            else:
                alpha = (math.log(vp) - math.log(v0)) / dT
                cp = 0.0
            h = 1e-3 * v0
            d2 = (self.F(v0 + h, t) - 2 * self.F(v0, t) + self.F(v0 - h, t)) / h ** 2
            V.append(v0), G.append(g0), a.append(alpha), B.append(v0 * float(d2) * EV_A3_TO_GPA), Cp.append(cp)
        a = np.array(a)
        return dict(T=T, V=np.array(V), G=np.array(G), beta=a, alpha_lin=a / 3.0, B_T=np.array(B), Cp=np.array(Cp))

    def summary(self):
        return (f"EOS: E0 = {self.E0:.5f} eV/atom, V0 = {self.V0:.4f} Å³/atom, B0 = {self.B0 * EV_A3_TO_GPA:.1f} GPa, "
                f"B0' = {self.Bp:.2f}; F_vib: " + ("phonopy tables at %d volumes" % len(self.fvib) if self.fvib
                                                  else ("Debye–Grüneisen θ0 = %.0f K, γ0 = %.2f, q = %.2f" % (
                                                      self.debye["theta0"], self.debye.get("gamma0", 0),
                                                      self.debye.get("q", 1.0)) if self.debye else "none"))
                + (f"; anharmonic a0 = {self.anh_a:.2e} /K (m = {self.anh_m})" if self.anh_a else "")
                + ("; anharmonic table" if self.anh_table is not None else ""))

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_phonopy_dir(cls, path, anh_a=0.0, anh_m=0.0, anh_table=None, ev_file="e-v.dat"):
        """folder with e-v.dat (V, E per cell, as for phonopy-qha) and thermal_properties*.yaml files
        (one per volume, natural sort order = the order of e-v.dat)"""
        d = np.loadtxt(os.path.join(path, ev_file), comments="#", ndmin=2)
        files = sorted(glob.glob(os.path.join(path, "thermal_properties*.y*ml*")),
                       key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
        if len(files) != len(d):
            raise ValueError(f"{path}: {len(d)} volumes in {ev_file} but {len(files)} thermal_properties files")
        fv = [PhononF(f) for f in files]
        nat = fv[0].natom
        return cls(volumes=d[:, 0] / nat, energies=d[:, 1] / nat, fvib=fv, anh_a=anh_a, anh_m=anh_m,
                   anh_table=anh_table)

    @classmethod
    def debye_gruneisen(cls, E0, V0, B0_GPa, Bp, theta0, gamma0, q=1.0, anh_a=0.0, anh_m=0.0, anh_table=None):
        return cls(eos=(E0, V0, B0_GPa / EV_A3_TO_GPA, Bp), debye=dict(theta0=theta0, gamma0=gamma0, q=q),
                   anh_a=anh_a, anh_m=anh_m, anh_table=anh_table)


def read_anh_table(path):
    """two columns: T (K), F_anh (meV/atom)  ->  (T, F in eV/atom)"""
    d = np.loadtxt(path, comments="#", ndmin=2)
    return d[:, 0], d[:, 1] * 1e-3


def plot_qha(qha, temps, save=None, title=None, figsize=(10.5, 3.4)):
    import matplotlib.pyplot as plt
    from . import hull_plots  # noqa: F401
    p = qha.properties(temps)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].plot(p["T"], p["V"], color="#16325c", lw=2)
    ax[0].set_ylabel("V (Å³/atom)")
    ax[1].plot(p["T"], p["alpha_lin"] * 1e6, color="#c0392b", lw=2)
    ax[1].set_ylabel("linear α = β/3 (10⁻⁶ /K)")
    ax[2].plot(p["T"], p["B_T"], color="#1e8449", lw=2)
    ax[2].set_ylabel("B$_T$ (GPa)")
    for a in ax:
        a.set_xlabel("T (K)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig
