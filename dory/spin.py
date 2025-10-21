from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import logging

logger = logging.getLogger("dory.spin")

import numpy as np 

# 2×2 Pauli’s
_SX = np.array([[0, 1],
                [1, 0]], dtype=complex)
_SY = np.array([[0, -1j],
                [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0],
                [0,-1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

# 3×3 L-matrices in Cartesian p-basis (px,py,pz), ħ=1
_LX_p = np.array([[0, 0, 0],
                  [0, 0,-1j],
                  [0, 1j, 0]], dtype=complex)
_LY_p = np.array([[0, 0, 1j],
                  [0, 0, 0],
                  [-1j,0, 0]], dtype=complex)
_LZ_p = np.array([[0,-1j,0],
                  [1j,0, 0],
                  [0, 0, 0]], dtype=complex)

# 5×5 L-matrices in real cubic-harmonic d-basis (dxy, dyz, dxz, dx2-y2, dz2)
_LX_d = np.array([[0, 0, 0, 0, 0],
                  [0, 0,-1j,0, 0],
                  [0, 1j,0, 0, 0],
                  [0, 0, 0, 0, 2j],
                  [0, 0, 0,-2j,0 ]], dtype=complex)

_LY_d = np.array([[0, 0, 1j,0, 0],
                  [0, 0, 0, 0,-2j],
                  [-1j,0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 2j,0, 0, 0]], dtype=complex)

_LZ_d = np.array([[0,-1j,0, 0, 0],
                  [1j,0, 0, 0, 0],
                  [0, 0, 0,-2j,0],
                  [0, 0, 2j,0, 0],
                  [0, 0, 0, 0, 0]], dtype=complex)


def soc_block_d(lam_a: float) -> np.ndarray:
    # create 10 by 10 d orbital matrix 
    if lam_a == 0.0:
        return np.zeros((10,10), dtype=complex)
    H_d = lam_a * (np.kron(_LX_d, _SX) + np.kron(_LY_d, _SY) + np.kron(_LZ_d, _SZ))

    logger.debug("p block=\n%s", H_d)

    return H_d


def soc_blocks_p(lam_a: float, lam_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (H11, H12, H21, H22) for p-orbitals, each 3x3, in the
    spin-major basis order [px, py, pz] for each spin block.
    """
    if lam_a == 0.0:
        Z = np.zeros((3, 3), dtype=complex)
        return Z, Z, Z, Z

    # Build SOC in orbital-major (px↑,px↓, py↑,py↓, pz↑,pz↓)
    H_OM = lam_a * (np.kron(_LX_p, _SX) + np.kron(_LY_p, _SY) + np.kron(_LZ_p, _SZ))

    # Permute to spin-major so it looks like [[H11,H12],[H21,H22]]
    perm = [0, 2, 4, 1, 3, 5]  # orbital-major → spin-major
    H_SM = H_OM[np.ix_(perm, perm)]

    H11 = np.kron(np.eye(2),H_SM[:3, :3]) 
    H12 = np.kron(np.eye(2),H_SM[:3, 3:])
    H21 = np.kron(np.eye(2),H_SM[3:, :3])
    H22 = np.kron(np.eye(2),H_SM[3:, 3:]) 

    logger.debug("p block H11: %s", H11)
    logger.debug("p block H12: %s", H12)
    logger.debug("p block H21: %s", H21)
    logger.debug("p block H22: %s", H22)

    return H11, H12, H21, H22


def insert_block_on_spinful(
                        H: np.ndarray,
                        idx_spinless: list[int],
                        H11: np.ndarray, H12: np.ndarray,
                        H21: np.ndarray, H22: np.ndarray) -> np.ndarray:
    """
    Insert four (n×n) spin blocks into H at the subspace selected by idx_spinless (length n).
    Quadrants:
        [ H11  H12 ]
        [ H21  H22 ]
    """
    up   = [1, 2, 3, 11, 12, 13]
    dn = [21, 22, 23, 31, 32, 33]
    n = len(idx_spinless)
    # (Optional sanity checks)
    # assert H11.shape == (n,n) and H12.shape == (n,n) and H21.shape == (n,n) and H22.shape == (n,n)

    H[np.ix_(up, up)]   += H11
    H[np.ix_(up, dn)]   += H12
    H[np.ix_(dn, up)]   += H21
    H[np.ix_(dn, dn)]   += H22

    return H 