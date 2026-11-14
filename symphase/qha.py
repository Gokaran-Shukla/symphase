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


# ----------------------------------------------------------------------------
#  self-consistent / effective-harmonic anharmonic phonons (TDEP, SSCHA, ...)
# ----------------------------------------------------------------------------
UNIT_TO_EV = {"ev": 1.0, "mev": 1e-3, "ry": 13.605693122994, "mry": 13.605693122994e-3, "ha": 27.211386245988,
              "kj/mol": 0.0103642688}


def read_tdep_free_energy(path):
    """TDEP outfile.free_energy: rows  T (K), F_ph (eV/atom), S (eV/K/atom), C_V (eV/K/atom)"""
    rows = [r for r in (_numeric(line) for line in open(path)) if r and len(r) >= 2]
    a = np.array([r[:4] + [np.nan] * (4 - len(r[:4])) for r in rows])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def read_tdep_U0(path):
    """TDEP outfile.U0: the SECOND value is the second-order baseline correction to add to F_ph (eV/atom)
    (the further values are higher-order corrections, not to be used with a harmonic reference)"""
    vals = []
    for line in open(path):
        vals += _numeric(line) or []
    if len(vals) < 2:
        raise ValueError(f"{path}: expected at least two numbers")
    return vals[1]


def _numeric(line):
    line = line.split("#")[0].strip()
    if not line:
        return None
    try:
        return [float(x) for x in line.split()]
    except ValueError:
        return None


class AnharmonicF:
    """F(T) per atom (eV) at fixed volume from self-consistent / effective-harmonic methods.

    points (T_i, F_i), F_i = F_ph[ω_eff(T_i)](T_i) + ΔF_i  (TDEP: ΔF = U0 baseline; SSCHA: total free energy).
    A natural cubic spline in T gives F(T) and S(T) = −dF/dT inside [T_min, T_max] (warning outside).
    includes_static: F contains the static lattice energy (TDEP with U0, SSCHA totals); then delta(T)
    = F(T) − e_static is the vibrational + anharmonic part to add to a static DFT energy."""

    def __init__(self, T, F, includes_static=False, e_static=None, label=""):
        from scipy.interpolate import CubicSpline
        o = np.argsort(T)
        self.T, self.F = np.asarray(T, float)[o], np.asarray(F, float)[o]
        if len(self.T) < 2:
            raise ValueError("need F at ≥ 2 temperatures")
        self.spl = CubicSpline(self.T, self.F, bc_type="natural")
        self.includes_static, self.e_static, self.label = includes_static, e_static, label
        self.warned = set()

    def __call__(self, T):
        T = float(T)
        if (T < self.T[0] - 1e-9 or T > self.T[-1] + 1e-9) and round(T) not in self.warned:
            self.warned.add(round(T))
        return float(self.spl(T))

    def entropy(self, T):
        return -float(self.spl(T, 1))

    def delta(self, T):
        if self.includes_static:
            if self.e_static is None:
                raise ValueError(f"{self.label}: free energy includes the static energy -> give e_static (eV/atom)")
            return self(T) - self.e_static
        return self(T)

    @classmethod
    def from_tdep(cls, path, use_U0=True, e_static=None):
        """folder with outfile.free_energy (+ outfile.U0) or sub-folders (one per temperature)"""
        files = sorted(glob.glob(os.path.join(path, "**", "outfile.free_energy"), recursive=True))
        if not files:
            raise ValueError(f"{path}: no outfile.free_energy found")
        Ts, Fs, has_u0 = [], [], []
        for f in files:
            T, F, _, _ = read_tdep_free_energy(f)
            u0f = os.path.join(os.path.dirname(f), "outfile.U0")
            u0 = read_tdep_U0(u0f) if (use_U0 and os.path.isfile(u0f)) else None
            has_u0.append(u0 is not None)
            for t, fv in zip(T, F):
                Ts.append(t)
                Fs.append(fv + (u0 or 0.0))
        if any(has_u0) and not all(has_u0):
            raise ValueError(f"{path}: outfile.U0 present for some temperatures only")
        T, idx = np.unique(np.round(Ts, 6), return_index=True)
        return cls(T, np.array(Fs)[idx], includes_static=all(has_u0), e_static=e_static, label=f"TDEP {path}")

    @classmethod
    def from_table(cls, path, unit="eV", per="atom", natom=1, includes_static=False, e_static=None):
        """two columns T, F (e.g. SSCHA free energies); unit eV|meV|Ry|mRy|Ha|kJ/mol, per 'atom' or 'cell'"""
        d = np.array([r for r in (_numeric(line) for line in open(path)) if r and len(r) >= 2])
        f = UNIT_TO_EV[unit.lower()] / (natom if per == "cell" else 1)
        return cls(d[:, 0], d[:, 1] * f, includes_static=includes_static, e_static=e_static, label=path)


class FreeEnergySurface(QHA):
    """anharmonic quasi-harmonic minimisation: total free energies F_i(T) (including the static energy,
    e.g. TDEP U0 + F_ph or SSCHA) at several volumes -> V(T), G(T), β, B_T, C_P"""

    def __init__(self, volumes, F_of_T, poly_degree=3):
        self.Ftot = list(F_of_T)
        V = np.asarray(volumes, float)
        super().__init__(volumes=V, energies=None, fvib=self.Ftot, eos=(0.0, float(np.mean(V)), 1.0, 4.0),
                         poly_degree=poly_degree)

    def E_static(self, V):
        return np.zeros_like(np.asarray(V, float))

    def summary(self):
        return f"free-energy surface from {len(self.Ftot)} volumes ({self.Vlo:.3f}–{self.Vhi:.3f} Å³/atom)"


# ----------------------------------------------------------------------------
#  1D anharmonic oscillator: exact, SCHA, harmonic (validation / tutorial)
# ----------------------------------------------------------------------------
def quartic_oscillator(k, lam, mass, temps, nbasis=120):
    """V = ½ k x² + λ x⁴ (meV, Å), mass in amu.  Returns dict of exact, SCHA and harmonic F(T) (meV),
    SCHA ω_eff and ⟨V − V_eff⟩ (the TDEP-like 'U0' correction)."""
    kB = 0.08617333262          # meV/K
    kap = HB2 / mass            # ħ²/(M Å²) meV
    w0 = math.sqrt(kap * k)     # meV
    # exact: harmonic-oscillator basis of frequency w0
    n = np.arange(nbasis)
    xs = np.sqrt(kap / (2 * w0))                       # √(ħ/2Mω) in Å
    X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
    X = xs * X
    H = np.diag(w0 * (n + 0.5)) + lam * np.linalg.matrix_power(X, 4)
    E = np.linalg.eigvalsh(H)[: nbasis // 2]
    out = dict(T=np.asarray(temps, float), exact=[], scha=[], harmonic=[], w_eff=[], dU=[], Fph=[])
    for T in out["T"]:
        b = 1.0 / (kB * max(T, 1e-6))
        out["exact"].append(E[0] - np.log(np.sum(np.exp(-b * (E - E[0])))) / b)
        fh = lambda w: 0.5 * w + (np.log(-np.expm1(-b * w)) / b if T > 0 else 0.0)   # noqa: E731
        out["harmonic"].append(fh(w0))
        w = w0
        for _ in range(500):
            x2 = kap / (2 * w) / math.tanh(b * w / 2) if T > 0 else kap / (2 * w)
            wn = math.sqrt(kap * (k + 12 * lam * x2))
            if abs(wn - w) < 1e-12:
                break
            w = 0.5 * (w + wn)
        x2 = kap / (2 * w) / math.tanh(b * w / 2) if T > 0 else kap / (2 * w)
        meff = w * w / kap                               # effective spring constant
        dU = 0.5 * k * x2 + 3 * lam * x2 * x2 - 0.5 * meff * x2      # ⟨V − ½ k_eff x²⟩
        out["Fph"].append(fh(w))
        out["dU"].append(dU)
        out["w_eff"].append(w)
        out["scha"].append(fh(w) + dU)
    for key in ("exact", "scha", "harmonic", "w_eff", "dU", "Fph"):
        out[key] = np.array(out[key])
    return out


HB2 = 4.18016   # ħ²/(amu Å²) in meV
