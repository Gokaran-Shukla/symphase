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
@pytest.mark.parametrize("name", ["symmetry", "hull", "magnons", "spirals", "supercells", "magphon_qha", "kubo_scph", "damping"])
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


# ---------------------------------------------------------------- OSSG numbers (non-collinear) via FindSpinGroup
@pytest.mark.parametrize("d,index", [("RuO2_Neel001_noSOC", "136.65.1.1.L"), ("MnTe_altermagnet_INCAR", "194.164.1.1.L"),
                                     ("Mn3Sn_allin", "194.6.1.1.P"), ("Triangular_120AFM_INCAR", "191.187.3.1.P")])
def test_ossg_numbers_and_msg_agreement(d, index):
    pytest.importorskip("findspingroup")
    r = analyze_tensors(f"{EX}/vasp/{d}", tensors="charge")
    o = r["spin_group"]["ossg"]
    assert o["available"] and o["index"] == index
    assert o["msg_agrees"] and o["msg_bns_number"] == r["magnetic"]["bns_number"]


# ---------------------------------------------------------------- magnon–phonon coupling
def _mp_chain(lam=6.0, dJdr=None, J=-2.0, K=0.3, path=(("Γ", (0, 0, 0)), ("X", (0.5, 0, 0)))):
    from symphase.magphon import analyse_mp
    lat, fr, R, T = _chain()
    me = None
    if lam:
        G = np.zeros((3, 3)); G[0, 2] = G[2, 0] = lam
        me = {(0, 0, (1, 0, 0)): [(0, G)]}
    return analyse_mp(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, springs=[(0, 0, [1, 0, 0], 600.0, 300.0)],
                      magnetoelastic=me, dJdr=dJdr, J=[J], K=K, K_axis=[0, 0, 1], spin=1.0, rmax=3.1, n_per=40,
                      path=list(path), masses=[55.845])


def test_magnon_phonon_two_oscillator_formula():
    from symphase.magphon import HB2_AMU
    r = _mp_chain()
    a, S, J, K, lam = 3.0, 1.0, -2.0, 0.3, 6.0
    k = r["q"][:, 0]
    wm = 2 * S * abs(J) * (1 - np.cos(k * a)) + 2 * K * S
    kap = HB2_AMU / 55.845
    wL, wT = np.sqrt(kap * 1200 * (1 - np.cos(k * a))), np.sqrt(kap * 600 * (1 - np.cos(k * a)))
    g = 2 * lam * S ** 1.5 * np.abs(np.sin(k * a))
    B, C = wm ** 2 + wL ** 2, wm ** 2 * wL ** 2 - wm * kap * g ** 2
    ref = np.sort(np.c_[np.sqrt((B + np.sqrt(B * B - 4 * C)) / 2), np.sqrt(np.maximum((B - np.sqrt(B * B - 4 * C)) / 2, 0)),
                        wT, wT], axis=1)
    assert np.abs(r["omega"][1:] - ref[1:]).max() < 1e-6          # (k = 0: zero modes regularised)
    assert r["max_gap"] > 0.1


def test_isotropic_striction_needs_noncollinear_order():
    r = _mp_chain(lam=0.0, dJdr=[3.0])
    assert r["coupled_bonds"] == 1 and np.nanmax(np.abs(r["omega"] - r["omega0"])) < 1e-12


def test_magnon_branch_equals_lswt():
    from symphase.magnons import analyse
    lat, fr, R, T = _chain()
    r = _mp_chain(lam=0.0)
    rl = analyse(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, J=[-2.0], K=0.3, K_axis=[0, 0, 1], spin=1.0,
                 rmax=3.1, n_per=6, nmap=3)
    for q, w in zip(rl["q"][1:], rl["omega"][1:]):
        assert np.min(np.abs(r["model"].modes(q, coupling=False)[0] - w[0])) < 1e-6


def test_phonopy_force_constants_equal_springs(tmp_path):
    phonopy = pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms
    from symphase.magphon import PhonopyFC, spring_fc, bond_springs, MagnonPhonon
    from symphase.tensors import read_poscar
    p = read_poscar(f"{EX}/vasp/Triangular_120AFM_INCAR/POSCAR")
    lat, frac = p["lattice"], p["frac"]
    ph = phonopy.Phonopy(PhonopyAtoms(symbols=["Fe"] * 3, cell=lat, scaled_positions=frac),
                         supercell_matrix=np.diag([3, 3, 1]), primitive_matrix=np.eye(3))
    sc = ph.supercell
    pos, L, n = sc.scaled_positions, np.array(sc.cell), len(sc.scaled_positions)
    fc = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            df = pos[j] - pos[i]; df -= np.round(df); d = L.T @ df
            if i != j and abs(np.linalg.norm(d) - 3.0) < 1e-3:
                r_ = d / np.linalg.norm(d)
                K = (400 * np.outer(r_, r_) + 150 * (np.eye(3) - np.outer(r_, r_))) * 1e-3
                fc[i, j] -= K; fc[i, i] += K
    ph.force_constants = fc
    ph.save(str(tmp_path / "ph.yaml"), settings={"force_constants": True})
    P = PhonopyFC(str(tmp_path / "ph.yaml"), lat, frac)
    mp = MagnonPhonon(lat, frac, ["Fe"] * 3, [], np.zeros((0, 3)), [], [], masses=P.masses,
                      springs=spring_fc(lat, frac, bond_springs(lat, frac, [0] * 3, 3.01, 400., 150., rmin=2.99)))
    for kf in ([0.1, 0.05, 0], [0.31, -0.2, 0], [1 / 3, 1 / 3, 0]):
        k = 2 * np.pi * np.linalg.inv(lat) @ np.array(kf)
        assert np.abs(mp.hessian(k, coupling=False, magnons=False)[:9, :9] - P.phi_k(k)).max() < 1e-8


# ---------------------------------------------------------------- quasi-harmonic and anharmonic free energies
def test_qha_debye_gruneisen_limits():
    from symphase.qha import QHA, EV_A3_TO_GPA
    from symphase.thermo import KB
    q0 = QHA.debye_gruneisen(0.0, 11.0, 180.0, 4.5, 450.0, 0.0)
    assert np.allclose(q0.properties([10, 500, 1000])["V"], 11.0, atol=1e-6)        # γ = 0: no expansion
    qa = QHA.debye_gruneisen(0.0, 11.0, 180.0, 4.5, 450.0, 0.0, anh_a=2e-5)
    assert abs((qa.G(1000) - q0.G(1000)) + 1.5 * KB * 2e-5 * 1e6) < 1e-9          # F_anh = −3/2 k_B a T²
    q = QHA.debye_gruneisen(0.0, 11.0, 180.0, 4.5, 450.0, 2.0)
    p = q.properties([600.0])
    gam = 2.0 * p["V"][0] / 11.0
    ratio = p["beta"][0] * p["B_T"][0] / EV_A3_TO_GPA * p["V"][0] / (gam * 3 * KB)
    assert 0.95 < ratio < 1.0                                                       # Grüneisen relation (C_V < 3k_B)


def test_qha_matches_phonopy_qha(tmp_path):
    pytest.importorskip("phonopy")
    from phonopy import PhonopyQHA
    from symphase.qha import QHA, birch_murnaghan, debye_theta, EV_A3_TO_GPA
    from symphase.thermo import debye_free_energy
    E0, V0, B0, Bp, th0, g0, nat, EVK = -5.0, 11.0, 180 / EV_A3_TO_GPA, 4.5, 450.0, 2.0, 4, 96.48533212
    Vs, Ts = V0 * np.linspace(0.94, 1.08, 11), np.arange(0, 1001, 10.0)
    FE = np.array([[debye_free_energy(T, debye_theta(V, V0, th0, g0)) for V in Vs] for T in Ts]) * nat * EVK
    ph = PhonopyQHA(volumes=Vs * nat, electronic_energies=birch_murnaghan(Vs, E0, V0, B0, Bp) * nat, temperatures=Ts,
                    free_energy=FE, cv=np.zeros_like(FE), entropy=np.zeros_like(FE), eos="birch_murnaghan", t_max=900)
    Vph, Gph = np.array(ph.volume_temperature) / nat, np.array(ph.gibbs_temperature) / nat
    with open(tmp_path / "e-v.dat", "w") as fh:
        for V in Vs:
            fh.write(f"{V * nat:.8f} {birch_murnaghan(V, E0, V0, B0, Bp) * nat:.10f}\n")
    for k, V in enumerate(Vs):
        with open(tmp_path / f"thermal_properties-{k + 1}.yaml", "w") as fh:
            fh.write(f"natom: {nat}\nthermal_properties:\n")
            for T, f in zip(Ts, FE[:, k]):
                fh.write(f"- temperature: {T:.4f}\n  free_energy: {f:.10f}\n")
    q = QHA.from_phonopy_dir(str(tmp_path))
    for T in (0, 300, 600, 900):
        V, G = q.equilibrium(T)
        assert abs(V - Vph[int(T / 10)]) < 2e-4 and abs(G - Gph[int(T / 10)]) < 2e-6


def test_qha_hull_shifts_stabilisation_temperature():
    import copy
    from symphase.thermo import FiniteTHull
    ents = load_csv(f"{EX}/hull/NiAl_qha.csv")
    noanh = copy.deepcopy(ents)
    for e in noanh:
        e.data.pop("anh_a", None)
    Th = FiniteTHull(noanh, vib="debye", mixing=False).stabilisation_temperature("Ni3Al-D019", Tmax=3000)
    Tq = FiniteTHull(noanh, vib="qha-debye", mixing=False).stabilisation_temperature("Ni3Al-D019", Tmax=3000)
    Ta = FiniteTHull(ents, vib="qha-debye", mixing=False).stabilisation_temperature("Ni3Al-D019", Tmax=3000)
    assert Ta < Tq < Th and abs(Th - 1393) < 5 and abs(Tq - 742) < 5 and abs(Ta - 667) < 5


# ---------------------------------------------------------------- Kubo magnitudes, symmetry-constrained
def test_kubo_haldane_and_kane_mele_quantised():
    from symphase.kubo import KuboTB, haldane_hr, kane_mele_hr, honeycomb_lattice
    lat, frac = honeycomb_lattice()
    c = frac @ lat
    R, H = haldane_hr()
    assert abs(abs(KuboTB(lat, R, H, centres=c, dims=2).conductivity([("charge", 0, 0, 1)], [0.0], (48, 48), 1e-6)[0, 0]) - 1) < 1e-6
    R, H = haldane_hr(M=1.0)
    assert abs(KuboTB(lat, R, H, centres=c, dims=2).conductivity([("charge", 0, 0, 1)], [0.0], (48, 48), 1e-6)[0, 0]) < 1e-6
    R, H = kane_mele_hr()
    v = KuboTB(lat, R, H, centres=np.r_[c, c], dims=2, spin_order="blocks").conductivity(
        [("spin", 2, 0, 1), ("charge", 0, 0, 1)], [0.0], (48, 48), 1e-6)[0]
    assert abs(abs(v[0]) - 1) < 1e-6 and abs(v[1]) < 1e-9


def test_kubo_3d_units_s_per_cm():
    from symphase.kubo import KuboTB, haldane_hr, honeycomb_lattice
    lat, frac = honeycomb_lattice(c=5.0)
    R, H = haldane_hr()
    s = KuboTB(lat, R, H, centres=frac @ lat, dims=3).conductivity([("charge", 0, 0, 1)], [0.0], (48, 48, 1), 1e-6)[0, 0]
    assert abs(abs(s) - 3.874045865e-5 / 5e-8) < 1e-3                  # (e²/h)/c in S/cm


def test_kubo_symmetry_fill_and_violation(tmp_path):
    from symphase.kubo import SymTensor, read_postw90_shc, read_postw90_ahc
    st = SymTensor.from_analysis(analyze_tensors(f"{EX}/vasp/fcc_Pt", tensors="spin", ossg="no"), "spin")
    assert st.n_par == 1
    sc = read_postw90_shc(f"{EX}/kubo/Pt_postw90/Pt-shc-fermiscan.dat", win=f"{EX}/kubo/Pt_postw90/Pt.win")
    fit = st.fit(sc[0.0])
    T = fit["full"].reshape(3, 3, 3)
    assert np.isclose(T[0, 1, 2], 1985.3) and np.isclose(T[1, 2, 0], 1985.3) and np.isclose(T[0, 2, 1], -1985.3)
    stc = SymTensor.from_analysis(analyze_tensors(f"{EX}/vasp/bcc_Fe_m001", tensors="charge", ossg="no"), "charge",
                                  parts=("odd",))
    f2 = stc.fit(read_postw90_ahc(f"{EX}/kubo/Fe_postw90/Fe-ahc-fermiscan.dat")[0.0])
    assert np.isclose(f2["full"][1], 751.6) and f2["forbidden_nonzero"] and 0 < f2["violation"] < 0.01


def test_kubo_compute_symmetric_check_component():
    from symphase.kubo import SymTensor, KuboTB, read_hr, read_centres, compute_symmetric
    d = f"{EX}/kubo/Graphene_KaneMele"
    st = SymTensor.from_analysis(analyze_tensors(d, tensors="spin", ossg="no"), "spin", parts=("even",))
    R, H = read_hr(f"{d}/kane_mele_hr.dat")
    from symphase.tensors import read_poscar
    m = KuboTB(read_poscar(f"{d}/POSCAR")["lattice"], R, H, centres=read_centres(f"{d}/centres.xyz"), dims=2,
               spin_order="blocks")
    fit = compute_symmetric(m, st, [0.0, 0.5], n=(40, 40), eta=1e-6)
    assert abs(abs(fit[0]["full"][2 * 9 + 0 * 3 + 1]) - 1) < 1e-4                  # 40×40 k grid
    assert fit[1]["check"]["rel_error"] < 1e-8 and abs(fit[1]["check"]["computed"]) > 0.1


# ---------------------------------------------------------------- self-consistent anharmonic phonons
def test_scha_quartic_oscillator_bound_and_accuracy():
    from symphase.qha import quartic_oscillator
    r = quartic_oscillator(2000.0, 1200.0, 55.845, [0, 300, 1000])
    assert np.all(r["scha"] >= r["exact"] - 1e-9)                      # Gibbs–Bogoliubov upper bound
    assert np.all(np.abs(r["scha"] - r["exact"]) < 0.25) and abs(r["harmonic"][2] - r["exact"][2]) > 4.0


def test_tdep_folder_reader_and_entropy(tmp_path):
    from symphase.qha import quartic_oscillator, AnharmonicF
    T = list(range(0, 1301, 100))
    q = quartic_oscillator(1450.0, 1000.0, 50.0, T)
    for t, f, du in zip(T, q["Fph"], q["dU"]):
        d = tmp_path / f"T{t:04d}"
        d.mkdir()
        (d / "outfile.free_energy").write_text(f"{t} {f / 1000:.12f} 0 0\n")
        (d / "outfile.U0").write_text(f"-1.0 {-5.0 + du / 1000:.12f} 0.0\n")
    F = AnharmonicF.from_tdep(str(tmp_path), e_static=-5.0)
    assert F.includes_static
    q2 = quartic_oscillator(1450.0, 1000.0, 50.0, [500.0, 501.0])
    assert abs(F.delta(500.0) * 1000 - q2["scha"][0]) < 0.05
    S_ref = -(q2["scha"][1] - q2["scha"][0]) / 1.0 / 1000
    assert abs(F.entropy(500.0) - S_ref) < 0.03 * abs(S_ref)


def test_free_energy_surface_matches_qha():
    from symphase.qha import QHA, FreeEnergySurface
    q = QHA.debye_gruneisen(-5.0, 11.0, 180.0, 4.5, 450.0, 2.0)
    Vs = 11.0 * np.linspace(0.95, 1.07, 9)
    surf = FreeEnergySurface(Vs, [(lambda T, V=V: float(q.F(V, T))) for V in Vs])
    for T in (300.0, 900.0):
        assert abs(surf.equilibrium(T)[0] - q.equilibrium(T)[0]) < 2e-3
        assert abs(surf.G(T) - q.G(T)) < 5e-5


def test_scph_hull_tdep_folders():
    from symphase.thermo import FiniteTHull
    ft = FiniteTHull(load_csv(f"{EX}/hull/NiAl_scph.csv"), vib="scph", mixing=False)
    assert abs(ft.stabilisation_temperature("Ni3Al-D019", Tmax=1600) - 790) < 10


# ---------------------------------------------------------------- two-magnon striction and damping
def _chain_mp(ncell, nsub, J, dJ, lamxz, dirs_sub, K, deform=None, spin=1.0):
    from symphase.magphon import MagnonPhonon, _G_of_bond
    a, nat = 3.0, ncell * nsub
    lat = np.diag([a * nat, 12.0, 12.0])
    frac = np.array([[t / nat, 0, 0] for t in range(nat)])
    dirs = np.array([dirs_sub[t % nsub] for t in range(nat)], float)
    pairs, me = [], {}
    for t in range(nat):
        j, R, d = (t + 1) % nat, (1 if t == nat - 1 else 0, 0, 0), np.array([a, 0, 0])
        spec = [("iso", dJ)] + ([(0, np.array([[0, 0, lamxz], [0, 0, 0], [lamxz, 0, 0]]))] if lamxz else [])
        Jb = J * np.eye(3)
        if deform is not None:
            G = _G_of_bond(spec, d)
            Jb = Jb + sum(G[g] * (deform(j) - deform(t))[g] for g in range(3))
        pairs.append((t, j, d, Jb))
        me[(t, j, R)] = spec
    As = [(-K * np.outer(e, e) if K else None) for e in dirs]
    return MagnonPhonon(lat, frac, ["Fe"] * nat, list(range(nat)), dirs, [spin] * nat, pairs, As, None, me,
                        springs=[], masses=[55.845] * nat)


def test_two_magnon_vertex_equals_frozen_phonon_supercell():
    from symphase.magdamp import MagnonPhononVertex
    from symphase.magnons import local_frames, hamiltonian
    a, nsub, ncell = 3.0, 2, 5
    dirs, J, dJ, lam, K = [[0, 0, 1], [0, 0, -1]], 2.0, 1.3, 0.7, 0.2
    vx = MagnonPhononVertex(_chain_mp(1, nsub, J, dJ, lam, dirs, K))
    f = np.array([[0.3 + 0.1j, 0.2, -0.1j], [-0.25, 0.15j, 0.05]])
    q = np.array([2 * np.pi * (2 / ncell) / (a * nsub), 0, 0])

    def deform(l):
        return lambda t: l * 2 * np.real(f[t % nsub] * np.exp(1j * q[0] * t * a))
    Hs = []
    for s in (1, -1):
        m = _chain_mp(ncell, nsub, J, dJ, lam, dirs, K, deform=deform(s * 1e-5))
        U, V = local_frames(m.v)
        Hs.append(hamiltonian(np.zeros(3), m.sites, U, V, m.S, m.pairs, m.Asite, None))
    dH = (Hs[0] - Hs[1]) / 2e-5
    nat = nsub * ncell
    pos = np.arange(nat) * a
    for kin in [2 * np.pi * m / (a * nat) for m in range(ncell)]:
        kout = kin + q[0]
        M, M2 = vx.M(np.array([kin, 0, 0]), q, f), vx.M(np.array([-kout, 0, 0]), q, f)
        for sa in range(nsub):
            for sb in range(nsub):
                ia, ib = range(sa, nat, nsub), range(sb, nat, nsub)
                ph = lambda i, j: np.exp(-1j * kout * pos[i] + 1j * kin * pos[j]) / ncell   # noqa: E731
                assert abs(sum(dH[i, j] * ph(i, j) for i in ia for j in ib) - M[sa, sb]) < 1e-8
                assert abs(sum(dH[i, nat + j] * ph(i, j) for i in ia for j in ib)
                           - (M[sa, nsub + sb] + M2[sb, nsub + sa])) < 1e-8
                assert abs(sum(dH[nat + i, j] * ph(i, j) for i in ia for j in ib)
                           - (M[nsub + sa, sb] + M2[nsub + sb, sa])) < 1e-8


def test_two_magnon_phonon_linewidth_semianalytic_1d():
    import math
    from symphase.magphon import analyse_mp
    from symphase.magdamp import MagnonPhononVertex, bose
    lat, fr, R, T = _chain()
    r = analyse_mp(lat, fr, [0], ["Fe"], np.array([[0, 0, 2.]]), R, T, springs=[(0, 0, [1, 0, 0], 600., 300.)],
                   dJdr=[3.0], J=[-5.0], K=0.3, K_axis=[0, 0, 1], spin=1.0, rmax=3.1, n_per=4,
                   path=[("G", (0, 0, 0)), ("X", (0.5, 0, 0))], masses=[55.845])
    assert np.nanmax(np.abs(r["omega"] - r["omega0"])) < 1e-12           # no linear hybridisation (collinear)
    vx = MagnonPhononVertex(r["model"])
    q = np.array([0.9, 0, 0])
    om, fs = vx.phonons(q)
    nu = int(np.argmax([abs(x[0, 0]) if x is not None else 0 for x in fs]))
    g = vx.phonon_linewidth(q, T=300.0, grid=(4000, 1, 1), sigma=0.02, modes=[nu])
    a, S, J, K = 3.0, 1.0, 5.0, 0.3
    eps = lambda k: 2 * S * J * (1 - np.cos(k * a)) + 2 * K * S                     # noqa: E731
    s = om[nu] / (4 * S * J * np.sin(q[0] * a / 2))
    ref = 0.0
    for base in (math.asin(s), math.pi - math.asin(s)):
        k = (base - q[0] * a / 2) / a
        Sm = vx.scattering(np.array([k, 0, 0]), q, fs[nu])[0, 0]
        ref += a * abs(Sm) ** 2 * (bose(eps(k), 300.0) - bose(eps(k + q[0]), 300.0)) / (
            4 * S * J * a * np.sin(q[0] * a / 2) * abs(math.cos(k * a + q[0] * a / 2)))
    assert ref > 0.05 and abs(g["gamma"][nu] - ref) < 1e-3 * ref and g["gamma_pair"][nu] == 0.0


def test_gilbert_damping_fm_exact_and_afm_exchange_enhanced():
    from symphase.magdamp import MagnonPhononVertex, gilbert_first_order
    from symphase.magphon import MagnonPhonon
    vx = MagnonPhononVertex(_chain_mp(1, 1, -5.0, 0.0, 0.0, [[0, 0, 1]], 0.3))
    al = 0.05
    o, g, wm = vx.damped_modes(np.array([0.7, 0, 0]), alpha=al, coupling=False)
    eps = 2 * 5.0 * (1 - np.cos(2.1)) + 0.6
    assert abs(o[0] - eps / (1 + al ** 2)) < 1e-9 and abs(g[0] - 2 * al * eps / (1 + al ** 2)) < 1e-9
    a = 3.0
    K3 = np.array([[500., 0, 0], [0, 200, 0], [0, 0, 200]])
    mp = MagnonPhonon(np.diag([2 * a, 12, 12]), np.array([[0, 0, 0], [0.5, 0, 0]]), ["Fe"] * 2, [0, 1],
                      np.array([[0, 0, 1.], [0, 0, -1.]]), [1.5, 1.5],
                      [(0, 1, np.array([a, 0, 0]), 4.0 * np.eye(3)), (1, 0, np.array([a, 0, 0]), 4.0 * np.eye(3))],
                      [-0.15 * np.diag([0, 0, 1.])] * 2, None, {},
                      springs=[(0, 1, np.array([a, 0, 0]), K3), (1, 0, np.array([a, 0, 0]), K3)], masses=[55.845] * 2)
    vx = MagnonPhononVertex(mp)
    for kx in (0.0, 0.4):
        o, g, wm = vx.damped_modes(np.array([kx, 0, 0]), alpha=1e-3, coupling=False)
        ex = sorted(gg for oo, gg, w in zip(o, g, wm) if w > 0.5)
        fo = sorted(gg for oo, gg in gilbert_first_order(mp, np.array([kx, 0, 0]), 1e-3))
        assert np.allclose(ex, fo, rtol=1e-3)
        assert abs(ex[0] - 2e-3 * (2 * 4.0 + 2 * 0.15) * 1.5) < 1e-5            # exchange-enhanced, k-independent
    o, g, wm = vx.damped_modes(np.array([0.5, 0, 0]), alpha=0.0, eta=0.3, coupling=False)
    assert np.allclose([gg for gg, w in zip(g, wm) if w < 0.5], 0.3)
