# dory/model3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import numpy as np
import logging
from .orbitals import OrbitalOrder, make_orbital_order, group_name
from .sk import SKParams, sk_block_sp3d5
from .spin import soc_block_d, soc_blocks_p, insert_block_on_spinful

logger = logging.getLogger("dory.model3d")

import numpy as np

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

    The site-local basis is defined in orbitals.py and may include:
      s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, (optional) s*
    """
    def __init__(
        self,
        basis: List[str],
        atomA_pos: np.ndarray,
        atomB_pos: np.ndarray,
        neighbors: List["Neighbor"],
        onsite_A: Dict[str, float],
        onsite_B: Dict[str, float],
        sk_params: SKParams,
        mbuilder: Callable[[List[str], np.ndarray, Dict[str, float], str], np.ndarray] = sk_block_sp3d5,
        # Optional args for spin orbit coupling
        enable_spin: bool = False,
        Delta_a_over_3: Optional[float] = None,   # p-SOC on A site  (λ = Δ/3)
        Delta_c_over_3: Optional[float] = None,   # p-SOC on B site  (λ = Δ/3)
        Delta_d_a: Optional[float] = None,        # d-SOC on A site (often 0 for Si)
        Delta_d_c: Optional[float] = None,        # d-SOC on B site
    ):
        self.order: OrbitalOrder = make_orbital_order(basis)
        self.rA = np.asarray(atomA_pos, dtype=float)
        self.rB = np.asarray(atomB_pos, dtype=float)
        self.neighbors = neighbors
        self.onsite_A = onsite_A
        self.onsite_B = onsite_B
        self.sk_params = sk_params
        self.mbuilder = mbuilder

        # Spin orbit coupling parameters
        self.enable_spin = bool(enable_spin)

        # p-manifold SOC strengths
        self.lam_p_A = float(Delta_a_over_3) if Delta_a_over_3 is not None else 0.0
        self.lam_p_B = float(Delta_c_over_3) if Delta_c_over_3 is not None else 0.0

        # d-manifold SOC (optional, default 0.0 for Si)
        self.lam_d_A = float(Delta_d_a) if Delta_d_a is not None else 0.0
        self.lam_d_B = float(Delta_d_c) if Delta_d_c is not None else 0.0

        logger.info("Hamiltonian3D initialised with %d orbitals per site (%d total).",
                    self.order.norb_site, self.order.norb_total)
        logger.debug("Atom A at %s, Atom B at %s", self.rA, self.rB)
        logger.debug("Onsite energies: A=%s | B=%s", self.onsite_A, self.onsite_B)
        logger.debug("Number of neighbors = %d", len(self.neighbors))

        logger.debug("delta over 3B is= %f", self.lam_p_B)
        logger.debug("delta over 3A is= %f", self.lam_p_A)

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

    def Hk_spinless(self, k: np.ndarray) -> np.ndarray:
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

    def Hk_spin(self, kvec: np.ndarray) -> np.ndarray:
        # construct hamiltonian and check if we need to include spin
        H0 = self.Hk_spinless(kvec)      
        if not self.enable_spin:
            return H0
        
        I2 = np.eye(2, dtype=complex)
        H = np.kron(I2, H0)     
             
        # add SOC on A/B p (and d) subspaces
        basis = self.order.basis
        have = set(basis).__contains__
        pnames = [o for o in ("px","py","pz") if have(o)]
        dnames = [o for o in ("dxy","dyz","dxz","dx2-y2","dz2") if have(o)]

        P_A = [self.order.idxA[o] for o in pnames]
        P_B = [self.order.idxB[o] for o in pnames]
        P_all = P_A + P_B
        D_A = [self.order.idxA[o] for o in dnames]
        D_B = [self.order.idxB[o] for o in dnames]

        # p-SOC (6×6 per spin block)
        if len(P_all) == 6 and (self.lam_p_A != 0.0 or self.lam_p_B != 0.0):
            H11, H12, H21, H22 = soc_blocks_p(self.lam_p_A, self.lam_p_B)
            H = insert_block_on_spinful(H, P_all, H11, H12, H21, H22)
        if len(D_A) == 5 and self.lam_d_A != 0.0:
            H = insert_block_on_spinful(H, D_A, soc_block_d(self.lam_d_A))
        if len(D_B) == 5 and self.lam_d_B != 0.0:
            H = insert_block_on_spinful(H, D_B, soc_block_d(self.lam_d_B))

        logger.debug("FINAL HAMILTONIAN= %d", H)
        np.savez_compressed("hamiltonian.npz", H=H)

        return H

    def solve_k(self, k: np.ndarray, sort: bool = True):
        """Diagonalise H(k). Return eigenvalues (and optionally eigenvectors)."""
        logger.info("Solving H(k) at k=%s", k)
        H = self.Hk_spin(k)

        gamma = np.asarray([0,0,0])
        if np.array_equal(k, np.zeros(3)):
            print("Saving at Gamma point")
            print(k)
            #np.savetxt("hamiltonian_real.csv", H.real, delimiter=",", fmt="%.10g")
            #np.savetxt("hamiltonian_imag.csv", H.imag, delimiter=",", fmt="%.10g")
            #np.savetxt("hamiltonian_mag.csv", np.abs(H), delimiter=",", fmt="%.10g")

            
        w, v = np.linalg.eigh(H)
        if sort:
            idx = np.argsort(w.real)
            w = w[idx]
            v = v[:, idx]
        logger.debug("Eigenvalues (sorted): %s", w)
        return w, v
    