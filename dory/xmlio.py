# dory/xmlio.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
import numpy as np
import logging

logger = logging.getLogger("dory.xmlio")


# ---------------- dataclasses ----------------

@dataclass
class Atom:
    label: str
    pos: np.ndarray  # 3-vector (Cartesian)


@dataclass
class Bond:
    frm: str
    to: str
    delta: np.ndarray  # 3-vector (Cartesian)


@dataclass
class OnsiteParams:
    A: Dict[str, float]  # keys: 's','p','d_t2g','d_eg'
    B: Dict[str, float]


@dataclass
class SKParamsBlock:
    ss_sigma: float = 0.0
    sp_sigma: float = 0.0
    ps_sigma: float = 0.0
    pp_sigma: float = 0.0
    pp_pi: float = 0.0
    sd_sigma: float = 0.0
    ds_sigma: float = 0.0
    pd_sigma: float = 0.0
    pd_pi: float = 0.0
    dp_sigma: float = 0.0
    dp_pi: float = 0.0
    dd_sigma: float = 0.0
    dd_pi: float = 0.0
    dd_delta: float = 0.0


@dataclass
class DoryConfig:
    basis: List[str]
    atomA: Atom
    atomB: Atom
    bonds: List[Bond]
    onsite: OnsiteParams
    sk_AB: SKParamsBlock
    sk_BA: SKParamsBlock
    use_builtin_sk: bool = True
    lattice_constant: Optional[float] = None  # <--- now included


# ---------------- helpers ----------------

def _parse_vec3(text: str) -> np.ndarray:
    vals = [float(x) for x in text.strip().replace(",", " ").split()]
    if len(vals) != 3:
        logger.error("Invalid vector: expected 3 numbers, got '%s'", text)
        raise ValueError(f"Expected 3 numbers, got {text}")
    return np.array(vals, dtype=float)


def _get_attr_float(elem: ET.Element, name: str, default: float = 0.0) -> float:
    v = elem.attrib.get(name, None)
    val = float(v) if v is not None else default
    if v is None:
        logger.debug("Attribute '%s' not found, using default %.3f", name, default)
    return val


def _parse_sk_block(elem: ET.Element) -> SKParamsBlock:
    block = SKParamsBlock(
        ss_sigma=_get_attr_float(elem, "ss_sigma"),
        sp_sigma=_get_attr_float(elem, "sp_sigma"),
        ps_sigma=_get_attr_float(elem, "ps_sigma"),
        pp_sigma=_get_attr_float(elem, "pp_sigma"),
        pp_pi=_get_attr_float(elem, "pp_pi"),
        sd_sigma=_get_attr_float(elem, "sd_sigma"),
        ds_sigma=_get_attr_float(elem, "ds_sigma"),
        pd_sigma=_get_attr_float(elem, "pd_sigma"),
        pd_pi=_get_attr_float(elem, "pd_pi"),
        dp_sigma=_get_attr_float(elem, "dp_sigma"),
        dp_pi=_get_attr_float(elem, "dp_pi"),
        dd_sigma=_get_attr_float(elem, "dd_sigma"),
        dd_pi=_get_attr_float(elem, "dd_pi"),
        dd_delta=_get_attr_float(elem, "dd_delta"),
    )
    logger.debug("Parsed SK block: %s", block)
    return block


# ---------------- main ----------------

def load_config(xml_path: str) -> DoryConfig:
    logger.info("Loading DORY config from '%s'", xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Basis
    basis_text = root.findtext("./basis")
    if basis_text is None:
        logger.error("Missing <basis> element in XML.")
        raise ValueError("Missing <basis> (e.g. 's', 'sp3', or explicit list).")
    basis_items = basis_text.strip().split()
    if basis_items == ["s"]:
        basis = ["s"]
    elif basis_items == ["sp3"]:
        basis = ["s", "px", "py", "pz"]
    elif basis_items == ["sp3d5"]:
        basis = ["s", "px", "py", "pz", "dxy", "dyz", "dxz", "dx2-y2", "dz2"]
    else:
        basis = basis_items
    logger.info("Basis orbitals: %s", basis)

    # Atoms
    eA = root.find("./atoms/atom[@id='A']")
    eB = root.find("./atoms/atom[@id='B']")
    if eA is None or eB is None:
        logger.error("Missing atoms A or B in XML.")
        raise ValueError("Need <atoms><atom id='A' .../><atom id='B' .../></atoms>")
    atomA = Atom("A", _parse_vec3(eA.attrib["pos"]))
    atomB = Atom("B", _parse_vec3(eB.attrib["pos"]))
    logger.info("Atom A at %s, Atom B at %s", atomA.pos, atomB.pos)

    # Bonds
    bonds: List[Bond] = []
    for eb in root.findall("./neighbors/bond"):
        frm = eb.attrib["from"]
        to = eb.attrib["to"]
        delta = _parse_vec3(eb.attrib["delta"])
        bonds.append(Bond(frm=frm, to=to, delta=delta))
        logger.debug("Bond parsed: %s→%s, δ=%s", frm, to, delta)
    logger.info("Parsed %d bonds.", len(bonds))

    # Onsite
    eOnA = root.find("./onsite/A")
    eOnB = root.find("./onsite/B")
    if eOnA is None or eOnB is None:
        logger.error("Missing onsite blocks <onsite><A .../><B .../></onsite>")
        raise ValueError("Need <onsite><A .../><B .../></onsite> with attributes.")
    onsite_A = {}
    onsite_B = {}
    for key in ("s", "p", "d_t2g", "d_eg"):
        if key in eOnA.attrib:
            onsite_A[key] = float(eOnA.attrib[key])
        if key in eOnB.attrib:
            onsite_B[key] = float(eOnB.attrib[key])
    logger.info("Onsite A=%s, Onsite B=%s", onsite_A, onsite_B)

    # SK (heteropolar)
    eAB = root.find("./sk/AB")
    eBA = root.find("./sk/BA")
    if eAB is None or eBA is None:
        logger.error("Missing <sk><AB .../><BA .../></sk> blocks.")
        raise ValueError("Need <sk><AB .../><BA .../></sk> blocks with SK parameters.")
    sk_AB = _parse_sk_block(eAB)
    sk_BA = _parse_sk_block(eBA)

    # builtin flag
    use_builtin = True
    eUse = root.find("./sk")
    if eUse is not None and "builtin" in eUse.attrib:
        use_builtin = eUse.attrib["builtin"].lower() != "false"
    logger.info("Use builtin SK builder: %s", use_builtin)

    # lattice constant (optional)
    lattice_constant = None
    eLat = root.find("./lattice_constant")
    if eLat is not None and eLat.text is not None:
        try:
            lattice_constant = float(eLat.text.strip())
            logger.info("Lattice constant a = %s %s",
                        lattice_constant, eLat.attrib.get("units", ""))
        except Exception:
            logger.warning("Failed to parse lattice_constant text: %s", eLat.text)

    config = DoryConfig(
        basis=basis,
        atomA=atomA,
        atomB=atomB,
        bonds=bonds,
        onsite=OnsiteParams(A=onsite_A, B=onsite_B),
        sk_AB=sk_AB,
        sk_BA=sk_BA,
        use_builtin_sk=use_builtin,
        lattice_constant=lattice_constant,
    )
    logger.info("DORY config loaded successfully.")
    return config