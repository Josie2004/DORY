    # dory/orbitals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import logging

logger = logging.getLogger("dory.orbitals")

ALL_D_ORBS = ["dxy", "dyz", "dxz", "dx2-y2", "dz2"]
ALL_S_ORBS = ["s", "s*"]   

@dataclass
class OrbitalOrder:
    """Holds the per-site orbital list and index maps."""
    basis: List[str]                   # e.g. ["s","px","py","pz","dxy","dyz","dxz","dx2-y2","dz2"]
    idxA: Dict[str, int]
    idxB: Dict[str, int]
    norb_site: int
    norb_total: int

def make_orbital_order(basis: List[str]) -> OrbitalOrder:
    """Construct index maps for a two-sublattice orbital ordering."""
    basis = list(basis)
    norb_site = len(basis)
    idxA = {orb: i for i, orb in enumerate(basis)}
    idxB = {orb: i + norb_site for i, orb in enumerate(basis)}

    logger.info("Creating orbital order with %d orbitals per site (%d total).", 
                norb_site, 2 * norb_site)
    logger.debug("Basis list: %s", basis)
    logger.debug("Index map A: %s", idxA)
    logger.debug("Index map B: %s", idxB)

    return OrbitalOrder(
        basis=basis,
        idxA=idxA,
        idxB=idxB,
        norb_site=norb_site,
        norb_total=2 * norb_site,
    )

def group_name(orb: str) -> str:
    """
    Return onsite group name for an orbital:
      's', 'p', 'd_t2g', 'd_eg'.
    """
    if orb == "s":
        return "s"
    if orb in ("px", "py", "pz"):
        return "p"
    if orb in ("dxy", "dyz", "dxz"):
        return "d_t2g"
    if orb in ("dx2-y2", "dz2"):
        return "d_eg"
    if orb == "s*":
        return "sstar"   


    logger.error("Unknown orbital encountered: %s", orb)
    raise ValueError(f"Unknown orbital {orb}")