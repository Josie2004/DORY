# dory/model3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np
import logging
from .orbitals import OrbitalOrder, make_orbital_order, group_name
from .sk import SKParams, sk_block_sp3d5

logger = logging.getLogger("dory.model3d")

@dataclass
class Neighbor:
    frm: str            # "A" or "B"
    to: str             # "A" or "B"
    delta: np.ndarray   # 3-vector

class Hamiltonian3D:
    """
    Build H(k) for a two-sublattice 3D TB model:
      H(k) = [ H_AA     H_AB(k) ]
             [ H_AB(k)† H_BB     ]
    """
    def __init__(
        self,
        basis: List[str],
        atomA_pos: np.ndarray,
        atomB_pos: np.ndarray,
        neighbors: List[Neighbor],
        onsite_A: Dict[str, float],
        onsite_B: Dict[str, float],
        sk_params: SKParams,
        mbuilder: Callable[[List[str], np.ndarray, Dict[str,float], str], np.ndarray] = sk_block_sp3d5,
    ):
        self.order: OrbitalOrder = make_orbital_order(basis)
        self.rA = np.asarray(atomA_pos, dtype=float)
        self.rB = np.asarray(atomB_pos, dtype=float)
        self.neighbors = neighbors
        self.onsite_A = onsite_A
        self.onsite_B = onsite_B
        self.sk_params = sk_params
        self.mbuilder = mbuilder

        logger.info("Hamiltonian3D initialised with %d orbitals per site (%d total).",
                    self.order.norb_site, self.order.norb_total)
        logger.debug("Atom A at %s, Atom B at %s", self.rA, self.rB)
        logger.debug("Onsite energies: A=%s | B=%s", self.onsite_A, self.onsite_B)
        logger.debug("Number of neighbors = %d", len(self.neighbors))

    def _onsite_matrix(self) -> np.ndarray:
        """Diagonal onsite H_AA ⊕ H_BB."""
        n = self.order.norb_total
        H = np.zeros((n, n), dtype=complex)

        # A
        for orb in self.order.basis:
            g = group_name(orb)  # 's','p','d_t2g','d_eg'
            if g in self.onsite_A:
                H[self.order.idxA[orb], self.order.idxA[orb]] = self.onsite_A[g]
                logger.debug("Onsite A: %s = %g", orb, self.onsite_A[g])
        # B
        for orb in self.order.basis:
            g = group_name(orb)
            if g in self.onsite_B:
                H[self.order.idxB[orb], self.order.idxB[orb]] = self.onsite_B[g]
                logger.debug("Onsite B: %s = %g", orb, self.onsite_B[g])
        return H

    def _hab_block(self, k: np.ndarray) -> np.ndarray:
        """Sum e^{ik·δ} M(δ) for all AB bonds; subtract phase for BA bonds via explicit list."""
        nb = self.order.norb_site
        H = np.zeros((nb, nb), dtype=complex)

        for b in self.neighbors:
            phase = np.exp(1j * float(np.dot(k, b.delta)))
            if b.frm == "A" and b.to == "B":
                logger.debug("Neighbor A→B, δ=%s, phase=%.4f+%.4fj", b.delta, phase.real, phase.imag)
                M = self.mbuilder(self.order.basis, b.delta, self.sk_params.AB, "AB")
                H += phase * M
            elif b.frm == "B" and b.to == "A":
                logger.debug("Neighbor B→A, δ=%s, phase=%.4f+%.4fj", b.delta, phase.real, phase.imag)
                M = self.mbuilder(self.order.basis, b.delta, self.sk_params.BA, "BA")
                H += phase * M
            else:
                logger.error("Invalid neighbor: from=%s to=%s", b.frm, b.to)
                raise ValueError("Neighbor must be between A and B.")
        return H

    def Hk(self, k: np.ndarray) -> np.ndarray:
        """Full H(k) as an (2*norb_site) x (2*norb_site) matrix."""
        logger.debug("Building H(k) for k=%s", k)
        oa = self._onsite_matrix()
        nb = self.order.norb_site
        H = np.array(oa, copy=True)
        HAB = self._hab_block(np.asarray(k, dtype=float))
        H[:nb, nb:] += HAB
        H[nb:, :nb] += HAB.conj().T
        logger.debug("H(k) matrix built, shape=%s", H.shape)
        return H

    def solve_k(self, k: np.ndarray, sort: bool = True):
        """Diagonalise H(k). Return eigenvalues (and optionally eigenvectors)."""
        logger.info("Solving H(k) at k=%s", k)
        H = self.Hk(k)
        w, v = np.linalg.eigh(H)
        if sort:
            idx = np.argsort(w.real)
            w = w[idx]
            v = v[:, idx]
        logger.debug("Eigenvalues (sorted): %s", w)
        return w, v