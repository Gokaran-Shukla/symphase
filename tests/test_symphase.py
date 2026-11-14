"""pytest suite for SymPhase  (run:  pytest tests/)"""
import os
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
EX = os.path.join(ROOT, "examples")

from symphase import analyze_tensors  # noqa: E402
from symphase.hull import HullAnalysis, load_csv, scan_vasp, make_entry  # noqa: E402


# ---------------------------------------------------------------- tensors
def test_pt_grey_group_one_she_component():
    r = analyze_tensors(f"{EX}/vasp/fcc_Pt")
    assert r["magnetic"]["bns_number"] == "225.117"
    assert r["magnetic"]["time_reversal"]
    assert r["tensors"]["spin"]["n_even"] == 1 and r["tensors"]["spin"]["n_odd"] == 0
    assert not r["tensors"]["charge"]["ahe_allowed"]


def test_fe_ferromagnet_ahe_along_m():
    r = analyze_tensors(f"{EX}/vasp/bcc_Fe_m001")
    assert r["magnetic"]["bns_symbol"] == "I4/mm'm'"
    assert r["tensors"]["charge"]["hall_vector"][2] != "0"
    assert r["tensors"]["charge"]["hall_vector"][:2] == ["0", "0"]


@pytest.mark.parametrize("d,kw,bns,ahe", [
    ("RuO2_Neel001_noSOC", {}, "136.499", False),
    ("RuO2_Neel100_SOC", {}, "58.398", True),
    ("RuO2_Neel001_noSOC", {"mdir": (1, 1, 0)}, "65.486", True)])
def test_ruo2_neel_orientations(d, kw, bns, ahe):
    r = analyze_tensors(f"{EX}/vasp/{d}", **kw)
    assert r["magnetic"]["bns_number"] == bns
    assert r["tensors"]["charge"]["ahe_allowed"] == ahe


def test_ruo2_spin_splitter_symmetric():
    odd = analyze_tensors(f"{EX}/vasp/RuO2_Neel001_noSOC", tensors="spin")["tensors"]["spin"]["odd"]
    assert odd["σ^z_xy"] == odd["σ^z_yx"] != "0"


def test_orbital_equals_spin_shape():
    t = analyze_tensors(f"{EX}/vasp/RuO2_Neel100_SOC")["tensors"]
    assert (t["spin"]["n_even"], t["spin"]["n_odd"]) == (t["orbital"]["n_even"], t["orbital"]["n_odd"])


# ---------------------------------------------------------------- hull
def test_binary_hull():
    h = HullAnalysis(load_csv(f"{EX}/hull/NiAl_binary.csv"))
    assert len(h.pd.stable_entries) == 6
    r = {x["name"]: x for x in h.table()}
    assert abs(1000 * r["Ni5Al3-Pt5Ga3"]["e_hull"] - 42.5) < 0.1
    assert abs(1000 * r["NiAl-L10"]["e_hull"] - 125.0) < 0.1


def test_scan_vasp_matches_csv():
    a = HullAnalysis(scan_vasp(f"{EX}/hull/vasp_runs_NiAl", verbose=False))
    b = HullAnalysis(load_csv(f"{EX}/hull/NiAl_binary.csv"))
    assert sorted(a.stable_formulas()) == sorted(b.stable_formulas())


def test_five_component_and_parallel_equals_serial():
    ents = load_csv(f"{EX}/hull/TiZrHfNiSn_5component.csv")
    rng = np.random.default_rng(3)
    els = ["Ti", "Zr", "Hf", "Ni", "Sn"]
    for k in range(300):
        n = rng.integers(2, 5)
        sub = rng.choice(els, n, replace=False)
        f = "".join(f"{e}{c}" for e, c in zip(sub, rng.integers(1, 4, n)))
        ents.append(make_entry(f, -6.0 * sum(rng.integers(1, 4, 1)), name=f"r{k}"))
    s = HullAnalysis(ents, nproc=1, backend="serial").table()
    p = HullAnalysis(ents, nproc=2, backend="process").table()
    assert [(r["name"], round(r["e_hull"], 9)) for r in s] == [(r["name"], round(r["e_hull"], 9)) for r in p]


def test_pseudo_binary_endpoints():
    h = HullAnalysis(load_csv(f"{EX}/hull/TiZrNiSn_quaternary.csv"))
    d = h.pseudo_binary("TiNiSn", "ZrNiSn", n=11)
    assert abs(d["e_hull_line"][0] + 0.61) < 1e-6 and abs(d["e_hull_line"][-1] + 0.70) < 1e-6


# ---------------------------------------------------------------- automatic pipeline (mock OQMD)
def test_run_pipeline_with_mock_oqmd(tmp_path):
    sys.path.insert(0, HERE)
    import mock_oqmd
    import symphase.auto as A
    A._http_json = mock_oqmd.fake_http_json
    h, info = A.run(["Ti", "Ni", "Sn"], outdir=str(tmp_path / "out"), write_vasp=True, plots=False)
    assert "TiNiSn" in h.stable_formulas()
    v = tmp_path / "out" / "vasp"
    assert (v / "submit_ibex.sh").exists() and (v / "joblist.txt").exists()
    d = sorted(p for p in v.iterdir() if p.is_dir() and p.name[0].isdigit())[0]
    species = open(d / "POSCAR").read().splitlines()[5].split()
    spec = open(d / "POTCAR.spec").read().split()
    assert [x.split("_")[0] for x in spec] == species


# ---------------------------------------------------------------- quizzes
@pytest.mark.parametrize("name", ["symmetry", "hull"])
def test_quiz_banks(name):
    from symphase.quiz import Quiz, _dec
    q = Quiz(name)
    assert len(q) == 20
    for i, qq in q.questions.items():
        k = _dec(qq["key"])
        assert q._correct(i, k[0] if isinstance(k, list) else k)
