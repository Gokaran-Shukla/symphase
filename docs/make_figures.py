"""Regenerate the figures used in docs/SymPhase_presentation.tex (toy example data)."""
import os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from symphase.hull import HullAnalysis, load_csv
from symphase import hull_plots as hp
warnings.simplefilter("ignore")
H = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(H, "..", "examples", "hull"); F = os.path.join(H, "figures")
os.makedirs(F, exist_ok=True)
h2 = HullAnalysis(load_csv(f"{E}/NiAl_binary.csv")); h3 = HullAnalysis(load_csv(f"{E}/TiNiSn_ternary.csv"))
h4 = HullAnalysis(load_csv(f"{E}/TiZrNiSn_quaternary.csv")); h5 = HullAnalysis(load_csv(f"{E}/TiZrHfNiSn_5component.csv"))
jobs = {"binary": lambda s: h2.plot(save=s), "ternary": lambda s: h3.plot(save=s),
        "surface": lambda s: hp.plot_ternary_surface(h3, save=s), "quaternary": lambda s: h4.plot(save=s),
        "sub5": lambda s: hp.plot_subsystems(h5, order=2, save=s),
        "ehull_bar": lambda s: hp.plot_ehull_bar(h5, max_rows=18, save=s),
        "slice": lambda s: hp.plot_pseudo_binary(h5, "TiNiSn", "HfNiSn", save=s)}
for k, f in jobs.items():
    fig = f(os.path.join(F, f"{k}.pdf")); fig.savefig(os.path.join(F, f"{k}.png"), dpi=160); plt.close(fig)
    print("figure", k)

# ---------------------------------------------------------------- magnons and finite-T (toy parameters)
import numpy as np, spglib
from symphase.tensors import read_poscar
from symphase.magnons import analyse, plot_bands, plot_split_map
from symphase.thermo import FiniteTHull
V = os.path.join(H, "..", "examples", "vasp")
p = read_poscar(f"{V}/RuO2_Neel001_noSOC/POSCAR")
types = [0, 0, 1, 1, 1, 1]
mags = np.array([[0, 0, 1.2], [0, 0, -1.2]] + [[0, 0, 0]] * 4)
ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], types))
res = analyse(p["lattice"], p["frac"], types, ["Ru"] * 2 + ["O"] * 4, mags, ds.rotations, ds.translations,
              J=[-0.5, 3.0, -0.2, 0, 0.2, 0, -0.6, -0.2], K=0.05, rmax=6.5, n_per=50, nmap=61)
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11.5, "axes.titlesize": 12, "legend.fontsize": 9.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})
for k, f in {"magnon_bands": lambda s: plot_bands(res, save=s, title="RuO$_2$ (toy J): bands by chirality", figsize=(6.0, 4.2)),
             "magnon_map": lambda s: plot_split_map(res, save=s, figsize=(4.9, 4.2))}.items():
    fig = f(os.path.join(F, f"{k}.pdf")); fig.savefig(os.path.join(F, f"{k}.png"), dpi=160); plt.close(fig)
    print("figure", k)
ft = FiniteTHull(load_csv(f"{E}/TiZrNiSn_quaternary.csv"), vib="debye")
ft.scan([0, 200, 300, 400, 600, 900, 1200]); ft.report(file=open(os.devnull, "w"))
fig = ft.plot_ehull_vs_T(nmax=5, figsize=(6.4, 4.2), legend_out=False)
fig.axes[0].set_title("E$_{hull}$(T): Debye + mixing entropy (toy)", fontsize=12)
fig.savefig(os.path.join(F, "ehull_vs_T.pdf")); fig.savefig(os.path.join(F, "ehull_vs_T.png"), dpi=160); plt.close(fig)
print("figure ehull_vs_T")

# ---------------------------------------------------------------- non-collinear: triangular 120° AFM vs analytic
from symphase.tensors import read_incar
d = f"{V}/Triangular_120AFM_INCAR"
p = read_poscar(f"{d}/POSCAR"); mags = np.array(read_incar(f"{d}/INCAR")["magmom"]).reshape(-1, 3)
ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], [0, 0, 0]))
r = analyse(p["lattice"], p["frac"], [0, 0, 0], ["Cr"] * 3, mags, ds.rotations, ds.translations,
            J=[1.0], spin=1.0, rmax=3.1, n_per=50, nmap=3)
a = 3.0; a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * np.sqrt(3) / 2, 0]); K = np.array([4 * np.pi / (3 * a), 0, 0])
def wtri(k):
    g = (np.cos(k @ a1) + np.cos(k @ a2) + np.cos(k @ (a2 - a1))) / 3
    return 3 * np.sqrt(max((1 - g) * (1 + 2 * g), 0))
ana = np.array([np.sort([wtri(q), wtri(q + K), wtri(q - K)]) for q in r["q"]])
fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.plot(r["x"], r["omega"], color="#16325c", lw=2.2)
ax.plot(r["x"][::5], ana[::5], "o", color="#e67e22", ms=4.5, mfc="none", mew=1.3)
ax.plot([], [], color="#16325c", lw=2.2, label="SymPhase LSWT"); ax.plot([], [], "o", color="#e67e22", mfc="none", label="analytic")
ax.set_xticks([t for t, _ in r["ticks"]]); ax.set_xticklabels([l for _, l in r["ticks"]], fontsize=9); ax.minorticks_off()
for t, _ in r["ticks"]: ax.axvline(t, color="0.85", lw=0.7, zorder=0)
ax.set_xlim(r["x"][0], r["x"][-1]); ax.set_ylim(0, None); ax.set_ylabel("ω (JS)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10, frameon=False)
fig.savefig(os.path.join(F, "triangular.pdf")); fig.savefig(os.path.join(F, "triangular.png"), dpi=160); plt.close(fig)
print("figure triangular")

# ---------------------------------------------------------------- incommensurate spirals (rotating frame)
import math
a = 3.0
latc, frc = np.diag([a, 12, 12]), np.zeros((1, 3))
dsc = spglib.get_symmetry_dataset((latc, frc, [0])); m0 = np.array([[0, 0, 2.0]])
J2s = np.linspace(0.0, 1.2, 25)
qs = [abs(analyse(latc, frc, [0], ["Fe"], m0, dsc.rotations, dsc.translations, J=[-1.0, j2], spin=1.0, rmax=6.1,
                  spiral="auto", spiral_axis=[0, 0, 1], n_per=3, nmap=3)["spiral"]["q_rlu"][0]) for j2 in J2s]
ex = [math.acos(min(1, 1 / (4 * j))) / (2 * np.pi) if j > 0.25 else 0 for j in np.linspace(0, 1.2, 300)]
r = analyse(latc, frc, [0], ["Fe"], m0, dsc.rotations, dsc.translations, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
            spiral="auto", spiral_axis=[0, 0, 1], n_per=80, nmap=3)
Jk = lambda k: -2 * np.cos(k * a) + np.cos(2 * k * a)
Q = 2 * np.pi * abs(r["spiral"]["q_rlu"][0]) / a
seg = r["x"] <= r["ticks"][1][0] + 1e-9
kk = r["q"][seg, 0]
ana = [np.sqrt(max((Jk(k) - Jk(Q)) * ((Jk(k + Q) + Jk(k - Q)) / 2 - Jk(Q)), 0)) for k in kk]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
ax1.plot(np.linspace(0, 1.2, 300), ex, color="0.65", lw=3, label="arccos(|J$_1$|/4J$_2$)/2π")
ax1.plot(J2s, qs, "o", color="#16325c", ms=5, label="SymPhase")
ax1.axvline(0.25, ls=":", color="k"); ax1.set_xlabel("J$_2$/|J$_1$|"); ax1.set_ylabel("q (r.l.u.)")
ax1.legend(fontsize=8.5, loc="lower right"); ax1.set_title("J$_1$–J$_2$ chain: spiral q", fontsize=11)
ax2.plot(kk * a / (2 * np.pi), r["omega"][seg, 0], color="#16325c", lw=2.2, label="rotating-frame LSWT")
ax2.plot(kk[::5] * a / (2 * np.pi), ana[::5], "o", color="#e67e22", mfc="none", ms=4.5, label="analytic")
ax2.axvline(1 / 6, ls=":", color="k"); ax2.set_xlabel("k (r.l.u.)"); ax2.set_ylabel("ω (meV)")
ax2.legend(fontsize=8.5); ax2.set_title("magnons, q = 1/6 (1-atom cell)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(F, "spiral.pdf")); fig.savefig(os.path.join(F, "spiral.png"), dpi=160); plt.close(fig)
print("figure spiral")
