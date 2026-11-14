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
