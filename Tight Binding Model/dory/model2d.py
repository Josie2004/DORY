# dory/model3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import numpy as np
import logging
from .orbitals import OrbitalOrder, make_orbital_order, group_name
from .sk import SKParams, sk_block_sp3d5

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

class Hamiltonian3D:
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
        Lz: float = 1.0,      # slab thickness (same units as your delta vectors)
        Nsub: int = 1,
    ):
        self.order: OrbitalOrder = make_orbital_order(basis)
        self.rA = np.asarray(atomA_pos, dtype=float)
        self.rB = np.asarray(atomB_pos, dtype=float)
        self.neighbors = neighbors
        self.onsite_A = onsite_A

        self.onsite_B = onsite_B
        self.sk_params = sk_params
        self.mbuilder = mbuilder
        self.Lz = float(Lz)     # slab thickness (e.g. nm)
        self.Nsub = int(Nsub)     # number of subbands

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

    def kz_of_p(self, p: int) -> float:
        return (p + 1) * np.pi / self.Lz

    def _normalize_k(self, *, k=None, kxy=None, kz=None):
        """
        Return (kx, ky, kz) as floats.
        Accept either:
        - k: array-like length 3, OR
        - (kxy: array-like length 2, kz: float)
        """
        if k is not None:
            k = np.asarray(k, dtype=float).reshape(-1)
            if k.size != 3:
                raise ValueError("k must be length 3 when provided.")
            return float(k[0]), float(k[1]), float(k[2])
        if kxy is not None and kz is not None:
            kxy = np.asarray(kxy, dtype=float).reshape(-1)
            if kxy.size != 2:
                raise ValueError("kxy must be length 2 when provided.")
            return float(kxy[0]), float(kxy[1]), float(kz)
        raise ValueError("Provide either k (len 3) or (kxy, kz).")

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
    
    def _hab_block(self, kxy: np.ndarray = None, *, kz: float | None = None, k=None) -> np.ndarray:
        """
        Sum e^{i k·δ} M(δ) for AB bonds.
        Bloch is applied in-plane; out-of-plane uses a scalar e^{i kz δz}.
        Compatible with either k=[kx,ky,kz] or (kxy, kz).
        """
        kx, ky, kz = self._normalize_k(k=k, kxy=kxy, kz=kz)
        nb = self.order.norb_site
        H = np.zeros((nb, nb), dtype=complex)

        for b in self.neighbors:
            dx, dy, dz = float(b.delta[0]), float(b.delta[1]), float(b.delta[2])

            # in-plane Bloch phase
            phase_xy = np.exp(1j * (kx * dx + ky * dy))
            # out-of-plane factor
            phase_z  = 1.0 if (abs(dz) < 1e-12) else np.exp(1j * kz * dz)
            phase    = phase_xy * phase_z

            if b.frm == "A" and b.to == "B":
                M = self.mbuilder(self.order.basis, b.delta, self.sk_params.AB, "AB")
                H += phase * M
            elif b.frm == "B" and b.to == "A":
                M = self.mbuilder(self.order.basis, b.delta, self.sk_params.BA, "BA")
                H += phase * M
            else:
                raise ValueError("Neighbor must be between A and B.")
        return H


    def Hk_spinless(self, k: np.ndarray = None, *, kxy: np.ndarray = None, kz: float | None = None) -> np.ndarray:
        """Full H(k) as an (2*norb_site)×(2*norb_site) matrix. Accepts k or (kxy,kz)."""
        kx, ky, kz = self._normalize_k(k=k, kxy=kxy, kz=kz)

        oa = self._onsite_matrix()
        nb = self.order.norb_site
        H  = np.array(oa, copy=True)

        HAB = self._hab_block(kxy=np.array([kx, ky]), kz=kz)  # uses unified phasing
        H[:nb, nb:] += HAB
        H[nb:, :nb] += HAB.conj().T
        return H


    def solve_k(self, k: np.ndarray, sort: bool = True):
        """3D/bulk-compatible: accepts k=[kx,ky,kz]."""
        logger.info("Solving H(k) at k=%s", k)
        H = self.Hk_spinless(k=k)

        if np.array_equal(np.asarray(k, float), np.zeros(3)):
            print("Saving at Gamma point")
            # np.savetxt(...)

        w, v = np.linalg.eigh(H)
        if sort:
            idx = np.argsort(w.real)
            w, v = w[idx], v[:, idx]
        return w, v

    def solve_kxy_kz(self, kxy: np.ndarray, kz: float, sort: bool = True):
        """Slab path: accepts (kxy, kz)."""
        logger.info("Solving H(kxy,kz) at kxy=%s kz=%.6g", kxy, kz)
        H = self.Hk_spinless(kxy=kxy, kz=kz)
        w, v = np.linalg.eigh(H)
        if sort:
            idx = np.argsort(w.real)
            w, v = w[idx], v[:, idx]
        return w, v
