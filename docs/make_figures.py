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

# ---------------------------------------------------------------- supercell workflow: cone, distorted spiral, skyrmion crystal
from symphase.supercell import run_supercell, scan_sizes
from symphase.magnons import read_exchange_file
from symphase.tensors import read_poscar
MUB = 0.05788381806
def chain_sc(size="auto", **kw):
    base = dict(J=[-1.0, 0.5], spin=1.0, rmax=6.1, restarts=3, spectrum=False); base.update(kw)
    return run_supercell(latc, frc, [0], ["Fe"], m0, dsc.rotations, dsc.translations, size=size, **base)
sk = os.path.join(H, "..", "examples", "vasp", "Skyrmion_triangular")
ps = read_poscar(f"{sk}/POSCAR"); dss = spglib.get_symmetry_dataset((ps["lattice"], ps["frac"], [0]))
exs = read_exchange_file(f"{sk}/J_DMI.dat")
def skx(B):
    return run_supercell(ps["lattice"], ps["frac"], [0], ["Mn"], np.array([[0, 0, 2.0]]), dss.rotations, dss.translations,
                         size=[10, 10, 1], restarts=2, spectrum=False, exchange=exs, spin=1.0, rmax=3.1, field=[0, 0, B])
r14 = skx(14)
Bs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
scan = [skx(B) for B in Bs]
Bc = [0.5, 1, 2, 3, 4]
cs = [chain_sc(field=[0, 0, B])["best"]["dirs"][:, 2].mean() for B in Bc]
fig = plt.figure(figsize=(12.4, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.32)
ax = fig.add_subplot(gs[0])
sc_, slat, sfrac = r14["model"], r14["slat"], r14["sfrac"]
e = r14["best"]["dirs"]; r0 = np.array([slat.T @ sfrac[I] for I in sc_["sites"]])
P = np.concatenate([r0 + u * slat[0] + v * slat[1] for u in range(2) for v in range(2)]); EE = np.tile(e, (4, 1))
sct = ax.scatter(P[:, 0], P[:, 1], c=EE[:, 2], cmap="coolwarm", vmin=-1, vmax=1, s=9, marker="h", lw=0)
ax.quiver(P[:, 0], P[:, 1], EE[:, 0], EE[:, 1], color="k", pivot="mid", scale=1.25 / 3.0, scale_units="xy",
          angles="xy", width=0.0028, headwidth=3.5)
cell = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ slat[:2, :2]
ax.plot(cell[:, 0], cell[:, 1], "--", color="0.2", lw=1)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("skyrmion crystal, 14 T (10×10 cell, N$_{sk}$ = −3)", fontsize=10)
cb = fig.colorbar(sct, ax=ax, fraction=0.035, pad=0.01, location="left"); cb.ax.set_title("s$_z$", fontsize=9)
ax = fig.add_subplot(gs[1])
ax.plot(Bs, [abs(x["Q_top"]) for x in scan], "o-", color="#c0392b", label="|N$_{sk}$| / supercell")
ax.plot(Bs, [3 * x["character"]["w0"] for x in scan], "s-", color="#16325c", ms=4, label="3 × uniform weight w(0)")
ax.axvspan(7, 22, color="#c0392b", alpha=0.08)
ax.set_xlabel("B (T)"); ax.set_ylim(-0.2, 4.4); ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=1)
ax.set_title("field scan: cycloid → SkX → FM", fontsize=10)
ax = fig.add_subplot(gs[2])
ax.plot(Bc, [2 * MUB * B / 0.5 for B in Bc], color="0.65", lw=3, label="analytic")
ax.plot(Bc, cs, "o", color="#16325c", label="supercell (6 cells)")
ax.set_xlabel("B (T)"); ax.set_ylabel("cos θ"); ax.legend(fontsize=8.5)
ax.set_title("conical spiral, J$_1$–J$_2$ chain", fontsize=10)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.15, top=0.9)
fig.savefig(os.path.join(F, "supercell.pdf")); fig.savefig(os.path.join(F, "supercell.png"), dpi=160); plt.close(fig)
print("figure supercell")

# ---------------------------------------------------------------- outlook 1 & 3: magnon–phonon coupling, QHA + anharmonic hulls
import copy
from symphase.magphon import analyse_mp, HB2_AMU
from symphase.qha import QHA
from symphase.thermo import FiniteTHull
from symphase.hull import load_csv
def fm_chain(lam=6.0, n_per=160):
    G = np.zeros((3, 3)); G[0, 2] = G[2, 0] = lam
    return analyse_mp(latc, frc, [0], ["Fe"], m0, dsc.rotations, dsc.translations, springs=[(0, 0, [1, 0, 0], 600.0, 300.0)],
                      magnetoelastic={(0, 0, (1, 0, 0)): [(0, G)]} if lam else None, J=[-5.0], K=0.3, K_axis=[0, 0, 1],
                      spin=1.0, rmax=3.1, n_per=n_per)
import matplotlib
rmp = fm_chain()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
cmap = matplotlib.colormaps["coolwarm"]
from matplotlib.collections import LineCollection
x = rmp["x"] * a / np.pi
for n in range(rmp["omega0"].shape[1]):
    a1.plot(x, rmp["omega0"][:, n], "--", color="0.6", lw=0.9)
for n in range(rmp["omega"].shape[1]):
    pts = np.array([x, rmp["omega"][:, n]]).T.reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=matplotlib.colors.Normalize(0, 1), lw=2.4)
    lc.set_array(0.5 * (rmp["w_magnon"][:-1, n] + rmp["w_magnon"][1:, n])); a1.add_collection(lc)
a1.set_xlim(0, 1); a1.set_ylim(0, 21); a1.set_xlabel("k (π/a)"); a1.set_ylabel("ω (meV)")
a1.set_title("FM chain: magnon polaron (colour = magnon weight)", fontsize=10)
gmax = max(rmp["gaps"], key=lambda g_: g_["gap"])
a1.annotate(f"gap {gmax['gap']:.2f} meV", xy=(gmax["x"] * a / np.pi, gmax["omega"] + 0.5 * gmax["gap"]),
            xytext=(0.55, 4.0), fontsize=9, arrowprops=dict(arrowstyle="->"))
ents = load_csv(os.path.join(H, "..", "examples", "hull", "NiAl_qha.csv")); noanh = copy.deepcopy(ents)
for e in noanh: e.data.pop("anh_a", None)
Tg = np.arange(0, 1601, 50)
for (lab, E, vib), c in zip([("harmonic", noanh, "debye"), ("QHA", noanh, "qha-debye"), ("QHA + anh.", ents, "qha-debye")],
                            ["0.5", "#16325c", "#c0392b"]):
    ft = FiniteTHull(E, vib=vib, mixing=False)
    eh = [{r_["name"]: r_["e_hull"] for r_ in ft.hull_at(T).table()}["Ni3Al-D019"] * 1e3 for T in Tg]
    a2.plot(Tg, eh, lw=2.2, color=c, label=f"{lab}: T* = {ft.stabilisation_temperature('Ni3Al-D019', Tmax=3000):.0f} K")
a2.set_xlabel("T (K)"); a2.set_ylabel("E$_{hull}$ (meV/atom)"); a2.set_ylim(-0.3, 11)
a2.legend(fontsize=8, frameon=False); a2.set_title("Ni$_3$Al-D0$_{19}$ (toy data)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(F, "magphon_qha.pdf")); fig.savefig(os.path.join(F, "magphon_qha.png"), dpi=160); plt.close(fig)
print("figure magphon_qha")
