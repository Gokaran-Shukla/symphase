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
