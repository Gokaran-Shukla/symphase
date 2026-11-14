"""pytest suite for SymPhase  (run:  pytest tests/)"""
import os
import sys
import math
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
@pytest.mark.parametrize("name", ["symmetry", "hull", "magnons", "spirals", "supercells"])
def test_quiz_banks(name):
    from symphase.quiz import Quiz, _dec
    q = Quiz(name)
    assert len(q) == 20
    for i, qq in q.questions.items():
        k = _dec(qq["key"])
        assert q._correct(i, k[0] if isinstance(k, list) else k)


# ---------------------------------------------------------------- spin groups (non-relativistic)
@pytest.mark.parametrize("d,cls,wave,laue", [
    ("RuO2_Neel001_noSOC", "ALTERMAGNET", "d-wave", "²4/¹m²m¹m"),
    ("MnTe_altermagnet_INCAR", "ALTERMAGNET", "g-wave", "²6/²m¹m²m"),
    ("Cr_bcc_AFM_INCAR", "antiferromagnet (conventional)", "no non-relativistic", None),
    ("Mn2Au_PT_AFM_INCAR", "antiferromagnet (PT-symmetric)", "no non-relativistic", None),
    ("bcc_Fe_m001", "ferromagnet / ferrimagnet", "s-wave", None),
    ("fcc_Pt", "non-magnetic", "no non-relativistic", None)])
def test_spin_group_classes(d, cls, wave, laue):
    sg = analyze_tensors(f"{EX}/vasp/{d}", tensors="charge,spin")["spin_group"]
    assert sg["magnet_class"] == cls
    assert sg["spin_splitting"].startswith(wave)
    if laue:
        assert sg["spin_laue_group"] == laue


def test_ruo2_nr_spin_splitter_and_soc_only_she():
    t = analyze_tensors(f"{EX}/vasp/RuO2_Neel001_noSOC", tensors="spin")["spin_group"]["tensors"]["spin"]
    assert t["n_even"] == 0                                   # no SHE without SOC
    assert t["odd"]["σ^z_xy"] == t["odd"]["σ^z_yx"] != "0"    # NR spin splitter
    assert "σ^z_xy" in t["soc_only_even"]


def test_incar_saxis_rotation():
    import shutil, tempfile
    tmp = tempfile.mkdtemp()
    shutil.copy(f"{EX}/vasp/RuO2_Neel001_noSOC/POSCAR", tmp)
    with open(os.path.join(tmp, "INCAR"), "w") as fh:
        fh.write("LSORBIT = .TRUE.\nSAXIS = 1 0 0\nMAGMOM = 0 0 1.2  0 0 -1.2  12*0\n")
    r = analyze_tensors(tmp)
    assert r["magnetic"]["bns_number"] == "58.398"            # Néel vector rotated to x


def test_moo2_monoclinic_odd_y_rule():
    r = analyze_tensors(f"{EX}/vasp/MoO2_monoclinic", tensors="charge,spin,orbital")
    assert r["magnetic"]["bns_number"] == "14.76" and r["magnetic"]["magnetic_point_group"] == "2/m1'"
    assert r["tensors"]["charge"]["n_even"] == 4 and r["tensors"]["charge"]["even"]["σ_xz"] != "0"
    for kind in ("spin", "orbital"):
        t = r["tensors"][kind]
        assert t["n_even"] == 13 and t["n_odd"] == 0
        for name, v in t["even"].items():
            idx = name.split("^")[1].replace("_", "")      # e.g. 'xxy'
            assert (v != "0") == (idx.count("y") % 2 == 1), name
    sg = r["spin_group"]["tensors"]
    assert sg["spin"]["n_even"] == 0 and sg["orbital"]["n_even"] == 13


# ---------------------------------------------------------------- spin-space-group numbering
@pytest.mark.parametrize("d,bns", [("RuO2_Neel001_noSOC", "136.498"), ("MnTe_altermagnet_INCAR", "194.269"),
                                   ("Cr_bcc_AFM_INCAR", "221.97"), ("Mn2Au_PT_AFM_INCAR", "139.533"),
                                   ("bcc_Fe_m001", "229.140")])
def test_collinear_ssg_numbers(d, bns):
    ssg = analyze_tensors(f"{EX}/vasp/{d}", tensors="charge")["spin_group"]["ssg"]
    assert ssg["number"] == bns


def test_spin_group_ops_match_spinspg():
    spinspg = pytest.importorskip("spinspg")
    import spglib
    from symphase.tensors import read_poscar
    from symphase import spingroup as sgm
    r = analyze_tensors(f"{EX}/vasp/MnTe_altermagnet_INCAR", tensors="charge")
    p = read_poscar(f"{EX}/vasp/MnTe_altermagnet_INCAR/POSCAR")
    types = [0, 0, 1, 1]
    mags = np.array([[0, 0, 4.5], [0, 0, -4.5], [0, 0, 0], [0, 0, 0]])
    _, rots, _, _ = spinspg.get_spin_symmetry(p["lattice"], p["frac"], np.array(types), mags, symprec=1e-3)
    assert len(rots) >= r["spin_group"]["n_ops"]   # spinspg may list each lattice op with several spin parts


# ---------------------------------------------------------------- magnons
def _magnon_setup(lat, frac, types, mags):
    import spglib
    from symphase.magnons import analyse
    ds = spglib.get_symmetry_dataset((lat, frac, types))
    return lambda **kw: analyse(lat, frac, types, ["X"] * len(frac), mags, ds.rotations, ds.translations,
                                n_per=8, nmap=5, **kw)


def test_magnon_fm_analytic():
    a = 3.0
    run = _magnon_setup(np.eye(3) * a, np.zeros((1, 3)), [0], np.array([[0, 0, 2.0]]))
    r = run(J=[-1.0], rmax=3.1)
    ex = 2 * np.sum(1 - np.cos(r["q"] * a), axis=1)
    assert np.allclose(r["omega"][:, 0], ex, atol=1e-6) and np.allclose(r["dSz"], -1)


def test_magnon_afm_chain_degenerate():
    b = 3.0
    run = _magnon_setup(np.diag([2 * b, 20, 20]), np.array([[0, 0, 0], [0.5, 0, 0]]), [0, 0],
                        np.array([[0, 0, 2.], [0, 0, -2.]]))
    r = run(J=[1.0], rmax=3.1)
    assert np.allclose(r["omega"][:, 0], 2 * np.abs(np.sin(r["q"][:, 0] * b)), atol=1e-3)
    assert np.nanmax(np.abs(r["split"])) < 1e-8


def test_magnon_ruo2_dwave_split():
    from symphase.tensors import read_poscar
    import spglib
    from symphase.magnons import analyse
    p = read_poscar(f"{EX}/vasp/RuO2_Neel001_noSOC/POSCAR")
    types = [0, 0, 1, 1, 1, 1]
    mags = np.array([[0, 0, 1.2], [0, 0, -1.2]] + [[0, 0, 0]] * 4)
    ds = spglib.get_symmetry_dataset((p["lattice"], p["frac"], types))
    r = analyse(p["lattice"], p["frac"], types, ["Ru"] * 2 + ["O"] * 4, mags, ds.rotations, ds.translations,
                J=[-0.5, 3.0, -0.2, 0, 0.2, 0, -0.6, -0.2], K=0.05, rmax=6.5, n_per=8, nmap=9)
    assert len(r["classes"]) == 8 and abs(r["classes"][6]["L"] - r["classes"][7]["L"]) < 1e-6
    Z, Q = r["map_split"], r["map_q"]
    # d-wave: Δω odd under qx -> -qx and under qy -> -qy, zero on the axes
    assert np.allclose(Z, -Z[::-1, :], atol=1e-6) and np.allclose(Z, -Z[:, ::-1], atol=1e-6)
    assert np.nanmax(np.abs(Z)) > 0.1


# ---------------------------------------------------------------- finite-T hull
def test_mixing_entropy_stabilisation_temperature():
    from symphase.thermo import FiniteTHull, parse_mixing
    import math
    S, _ = parse_mixing("Ti0.5Zr0.5|Ni|Sn")
    assert abs(S - math.log(2) / 3) < 1e-12
    ft = FiniteTHull(load_csv(f"{EX}/hull/TiZrNiSn_quaternary.csv"), vib="none")
    ft.scan([0, 200, 300])
    T = ft.stabilisation_temperature("(Ti0.5Zr0.5)NiSn-SQS")
    assert abs(T - 0.005 * 3 / (8.617333262e-5 * math.log(2))) < 1.0     # 251 K


def test_debye_free_energy_limits():
    from symphase.thermo import debye_free_energy, KB
    th = 300.0
    assert abs(debye_free_energy(0, th) - 9 / 8 * KB * th) < 1e-12        # zero-point energy
    T = 5000.0                                                               # classical limit
    assert abs(debye_free_energy(T, th) - 3 * KB * T * math.log(th / T) - (-KB * T)) < 2e-3


# ---------------------------------------------------------------- non-collinear magnons / spin groups
def _sym(lat, frac, types):
    import spglib
    d = spglib.get_symmetry_dataset((lat, frac, types))
    return d.rotations, d.translations


def test_magnon_fm_direction_independent():
    from symphase.magnons import analyse
    lat, fr = np.eye(3) * 3.0, np.zeros((1, 3))
    R, T = _sym(lat, fr, [0])
    rz = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-1.0], rmax=3.1, n_per=6, nmap=3)
    rd = analyse(lat, fr, [0], ["Fe"], np.array([[1, 1, 1.]]) * 2 / np.sqrt(3), R, T, J=[-1.0], rmax=3.1,
                 n_per=6, nmap=3)
    assert np.allclose(rz["omega"], rd["omega"], atol=1e-10)
    assert np.allclose(rd["magnon_spin"][3, 0], -np.ones(3) / np.sqrt(3), atol=1e-8)


def test_magnon_triangular_120_analytic():
    from symphase.magnons import analyse
    from symphase.tensors import read_poscar, read_incar
    d = f"{EX}/vasp/Triangular_120AFM_INCAR"
    p = read_poscar(f"{d}/POSCAR")
    mags = np.array(read_incar(f"{d}/INCAR")["magmom"]).reshape(-1, 3)
    R, T = _sym(p["lattice"], p["frac"], [0, 0, 0])
    r = analyse(p["lattice"], p["frac"], [0, 0, 0], ["Cr"] * 3, mags, R, T, J=[1.0], spin=1.0, rmax=3.1,
                n_per=8, nmap=3)
    a = 3.0
    a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * np.sqrt(3) / 2, 0])
    K = np.array([4 * np.pi / (3 * a), 0, 0])

    def w(k):
        g = (np.cos(k @ a1) + np.cos(k @ a2) + np.cos(k @ (a2 - a1))) / 3
        return 3 * np.sqrt(max((1 - g) * (1 + 2 * g), 0))
    for q, om in zip(r["q"], r["omega"]):
        assert np.allclose(np.sort(om), np.sort([w(q), w(q + K), w(q - K)]), atol=1e-3)
    assert r["kind"] == "coplanar" and r["n_goldstone"] == 3 and not r["unstable"]


def test_canted_chain_relaxation():
    from symphase.magnons import analyse, read_exchange_file
    from symphase.tensors import read_poscar
    d = f"{EX}/vasp/CantedChain_DMI_INCAR"
    p = read_poscar(f"{d}/POSCAR")
    R, T = _sym(p["lattice"], p["frac"], [0, 0])
    mags = np.array([[0.0, 0, 2.0], [0.0, 0, -2.0]]) + np.array([[0.3, 0, 0], [0, 0, 0]])
    r = analyse(p["lattice"], p["frac"], [0, 0], ["Fe", "Fe"], mags, R, T,
                exchange=read_exchange_file(f"{d}/J_DM.dat"), spin=1.0, rmax=3.1, relax=True, n_per=6, nmap=3)
    e = np.array(r["directions"])
    ang = math.degrees(math.acos(np.clip(e[0] @ e[1], -1, 1)))
    assert abs(ang - (180 - math.degrees(math.atan(0.2)))) < 1e-3
    assert abs(r["E_classical"] + 2 * math.hypot(1.0, 0.2)) < 1e-6 and not r["unstable"]


@pytest.mark.parametrize("d,cls,wave", [
    ("CantedChain_DMI_INCAR", "non-collinear ferro/ferrimagnet", "s-wave"),
    ("Triangular_120AFM_INCAR", "non-collinear antiferromagnet (coplanar)", "f-wave")])
def test_noncollinear_spin_group_classes(d, cls, wave):
    sg = analyze_tensors(f"{EX}/vasp/{d}", tensors="charge")["spin_group"]
    assert sg["magnet_class"].startswith(cls) and sg["spin_splitting"].startswith(wave)
    assert sg["n_H"] >= 1


# ---------------------------------------------------------------- incommensurate spirals (rotating frame)
def _chain(a=3.0):
    lat, fr = np.diag([a, 12, 12]), np.zeros((1, 3))
    R, T = _sym(lat, fr, [0])
    return lat, fr, R, T


def test_spiral_j1j2_wavevector_and_dispersion():
    from symphase.magnons import analyse
    a = 3.0
    lat, fr, R, T = _chain(a)
    r = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
                spiral="auto", spiral_axis=[0, 0, 1], n_per=12, nmap=3)
    q = abs(r["spiral"]["q_rlu"][0])
    assert abs(q - 1 / 6) < 1e-5 and abs(r["spiral"]["E_spiral"] + 0.75) < 1e-8
    Jk = lambda k: -2 * np.cos(k * a) + np.cos(2 * k * a)
    Q = 2 * np.pi * q / a
    for qq, om in zip(r["q"], r["omega"]):
        k = qq[0]
        ex = np.sqrt(max((Jk(k) - Jk(Q)) * ((Jk(k + Q) + Jk(k - Q)) / 2 - Jk(Q)), 0))
        assert abs(om[0] - ex) < 1e-3


def test_spiral_dmi_helix_period():
    from symphase.magnons import analyse, read_exchange_file
    lat, fr, R, T = _chain()
    r = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T,
                exchange=read_exchange_file(f"{EX}/vasp/Helix_DMI_chain/J_DMI.dat"), spin=1.0, rmax=3.1,
                spiral="auto", n_per=6, nmap=3)
    assert abs(abs(r["spiral"]["q_rlu"][0]) - math.atan(0.2) / (2 * np.pi)) < 1e-6
    assert np.allclose(np.abs(r["spiral"]["axis"]), [1, 0, 0]) and not r["unstable"]


def test_spiral_equals_commensurate_supercell():
    from symphase.magnons import analyse
    a = 3.0
    lat, fr, R, T = _chain(a)
    rs = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
                 spiral=[1 / 6, 0, 0], spiral_axis=[0, 0, 1], n_per=6, nmap=3)
    L6, fr6 = np.diag([6 * a, 12, 12]), np.array([[k / 6, 0, 0] for k in range(6)])
    R6, T6 = _sym(L6, fr6, [0] * 6)
    m6 = np.array([[2 * np.cos(2 * np.pi * k / 6), 2 * np.sin(2 * np.pi * k / 6), 0] for k in range(6)])
    r6 = analyse(L6, fr6, [0] * 6, ["Fe"] * 6, m6, R6, T6, J=[-1.0, 0.5], spin=1.0, rmax=6.1, n_per=6, nmap=3)
    assert abs(r6["E_classical"] / 6 - rs["spiral"]["E_spiral"]) < 1e-9
    assert np.isclose(r6["omega"].max(), np.nanmax(rs["omega"]), atol=1e-3)


def test_relax_warns_about_incommensurate_spiral():
    from symphase.magnons import analyse
    lat, fr, R, T = _chain()
    r = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
                relax=True, n_per=4, nmap=3)
    assert r["spiral_check"]["lower"] and abs(abs(r["spiral_check"]["q_rlu"][0]) - 1 / 6) < 1e-4


def test_spiral_rejects_non_invariant_hamiltonian():
    from symphase.magnons import analyse, dm_matrix
    lat, fr, R, T = _chain()
    ex = [(1, 1, [1, 0, 0], -1.0 * np.eye(3) + dm_matrix([0.2, 0, 0]))]
    with pytest.raises(ValueError):
        analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, exchange=ex, spin=1.0, rmax=3.1,
                spiral="auto", spiral_axis=[0, 0, 1], n_per=4, nmap=3)


# ---------------------------------------------------------------- supercell workflow (multi-Q / conical / distorted)
MUB = 0.05788381806


def _sc(size="auto", **kw):
    from symphase.supercell import run_supercell
    lat, fr, R, T = _chain()
    base = dict(J=[-1.0, 0.5], spin=1.0, rmax=6.1, restarts=3, spectrum=False)
    base.update(kw)
    return run_supercell(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, size=size, **base)


def test_supercell_auto_size_j1j2():
    r = _sc()
    assert np.diag(r["M"]).tolist() == [6, 1, 1]
    assert abs(r["E_cell"] + 0.75) < 1e-7 and r["character"]["cls"] == "single-Q spiral"


def test_supercell_model_mapping_is_exact():
    from symphase.supercell import ClassicalModel
    r = _sc(size=[5, 1, 1])
    m = r["model"]
    cm = ClassicalModel(m["sites"], m["pairs"], m["Asite"], m["S"], m["hfield"])
    assert abs(cm.energy(np.tile([0, 0, 1.0], (5, 1))) / 5 - (-1.0 + 0.5)) < 1e-12


def test_conical_spiral_in_field_analytic():
    B = 2.0
    r = _sc(field=[0, 0, B])
    c = 2 * MUB * B / (2 * 1.0 * (-0.5 + 0.75))                # gμB B / (2S(J(0) − J(Q)))
    ez = r["best"]["dirs"][:, 2]
    assert np.allclose(ez, c, atol=1e-5)
    assert abs(r["E_cell"] - (c * c * -0.5 + (1 - c * c) * -0.75 - 2 * MUB * B * c)) < 1e-8
    assert r["character"]["cls"].startswith("conical") and abs(r["character"]["w0"] - c * c) < 1e-5


def test_ferromagnet_zeeman_gap():
    from symphase.magnons import analyse
    lat, fr, R, T = _chain()
    r = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-1.0], spin=1.0, rmax=3.1,
                field=[0, 0, 1.5], n_per=4, nmap=3)
    assert abs(np.nanmin(r["omega"]) - 2 * MUB * 1.5) < 1e-6


def test_spiral_mode_refuses_field():
    from symphase.magnons import analyse
    lat, fr, R, T = _chain()
    with pytest.raises(ValueError, match="supercell"):
        analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-1.0, 0.5], spin=1.0, rmax=6.1,
                spiral="auto", field=[0, 0, 1], n_per=4, nmap=3)


def test_distorted_spiral_has_harmonics():
    r = _sc(size=[6, 1, 1], K=0.4, K_axis=[1, 0, 0])
    assert r["E_cell"] < -0.75 - 0.4 / 2 - 1e-3               # below the undistorted planar spiral
    assert "distorted" in r["character"]["cls"]
    assert any(abs(n) == 3 and w > 1e-3 for _, w, n in r["character"]["harmonics"])


def test_skyrmion_crystal_triangular(tmp_path):
    from symphase.magnons import read_exchange_file
    from symphase.supercell import run_supercell, write_outputs
    from symphase.tensors import read_poscar
    p = read_poscar(f"{EX}/vasp/Skyrmion_triangular/POSCAR")
    R, T = _sym(p["lattice"], p["frac"], [0])
    ex = read_exchange_file(f"{EX}/vasp/Skyrmion_triangular/J_DMI.dat")
    r = run_supercell(p["lattice"], p["frac"], [0], ["Mn"], np.array([[0, 0, 2.]]), R, T, size="auto",
                      restarts=2, spectrum=False, exchange=ex, spin=1.0, rmax=3.1, field=[0, 0, 14])
    assert len(r["star"]) == 3 and np.diag(r["M"]).tolist() == [10, 10, 1]
    assert abs(r["Q_top"] + 3) < 1e-3 and r["character"]["cls"].startswith("skyrmion crystal")
    write_outputs(r, ["Mn"], np.array([[0, 0, 2.]]), str(tmp_path))
    assert read_poscar(str(tmp_path / "POSCAR"))["frac"].shape == (100, 3)
    assert len(read_exchange_file(str(tmp_path / "J_super.dat"))) == 300
