"""Offline stand-in for the OQMD REST API (used by the tests): serves the toy
Ti-Ni-Sn data in the same JSON format as http://oqmd.org/oqmdapi/formationenergy"""
import os, re, urllib.parse
import numpy as np
from pymatgen.core import Composition, Lattice, Structure
from symphase.hull import load_csv

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "examples", "hull", "TiZrHfNiSn_5component.csv")


def _records():
    recs = []
    ents = load_csv(CSV)
    refs = {}
    for e in ents:
        if e.composition.is_element:
            el = str(e.composition.elements[0])
            refs[el] = min(refs.get(el, 1e9), e.energy_per_atom)
    for i, e in enumerate(ents):
        st = getattr(e, "structure", None)
        if st is None:
            c = e.composition
            sp = [str(el) for el in c.elements for _ in range(int(round(c[el])))]
            g = int(np.ceil(len(sp) ** (1 / 3)))
            st = Structure(Lattice.cubic(2.7 * g), sp,
                           [[(k % g) / g, ((k // g) % g) / g, (k // g // g) / g] for k in range(len(sp))])
        ef = e.energy_per_atom - sum(e.composition.get_atomic_fraction(el) * refs[str(el)]
                                     for el in e.composition.elements)
        recs.append(dict(name=e.composition.reduced_formula, entry_id=1000 + i, delta_e=ef,
                         stability=None, spacegroup="P1", natoms=len(st), prototype=None, icsd_id=None,
                         unit_cell=st.lattice.matrix.tolist(),
                         sites=[f"{s.specie} @ {s.frac_coords[0]:.6f} {s.frac_coords[1]:.6f} {s.frac_coords[2]:.6f}"
                                for s in st],
                         _els=sorted(map(str, e.composition.elements))))
    return recs


_RECS = None


def fake_http_json(url, timeout=120, retries=3):
    global _RECS
    if _RECS is None:
        _RECS = _records()
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    m = re.match(r"element_set=([\w,]+) AND ntypes=(\d+)", q["filter"][0])
    els = sorted(m.group(1).split(","))
    data = [{k: v for k, v in r.items() if k != "_els"} for r in _RECS if r["_els"] == els]
    return {"data": data, "links": {"next": None}}
