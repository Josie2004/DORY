
# valley.py
# Build the 6x6 (or 12x12 with spin) valley-orbit Hamiltonian.
import numpy as np

DEFAULT_ORDER = ['+x','-x','+y','-y','+z','-z']

def build_vo_block(delta_c: float, delta: float, ordering=DEFAULT_ORDER) -> np.ndarray:
    """Return the 6x6 spinless valley-orbit coupling matrix.
    Off-diagonals are c = Δc, and opposite-valley couplings are c'=(1+δ)Δc.
    Diagonal entries are 0 (central-cell mixing only).
    """
    H = np.zeros((6, 6), dtype=float)
    c  = float(delta_c)
    cp = (1.0 + float(delta)) * c
    # opposite indices in DEFAULT_ORDER
    opp = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            H[i, j] = cp if j == opp[i] else c
    return H

def build_valley_diag(shifts_meV: dict, ordering=DEFAULT_ORDER) -> np.ndarray:
    """Return diag(E_μ) from a dict of meV shifts per valley.
    Expected keys: Ex, Ex_bar, Ey, Ey_bar, Ez, Ez_bar.
    Missing keys default to 0.0.
    """
    key_map = ['Ex', 'Ex_bar', 'Ey', 'Ey_bar', 'Ez', 'Ez_bar']
    vals = [float(shifts_meV.get(k, 0.0)) for k in key_map]
    return np.diag(vals)

def with_spin(H6: np.ndarray, spin_enabled: bool) -> np.ndarray:
    """Promote 6x6 to 12x12 with ⊗ I2 if spin is requested."""
    return np.kron(H6, np.eye(2)) if spin_enabled else H6

def solve_valley(H: np.ndarray):
    """Diagonalize H; return eigenvalues and eigenvectors (columns)."""
    E, V = np.linalg.eigh(H)
    idx = np.argsort(E.real)
    return E[idx], V[:, idx]

def fit_vo_from_gaps(E12_meV: float, E23_meV: float):
    """Given singlet–doublet (E12) and doublet–triplet (E23) gaps, return (Δc, δ)."""
    delta_c = float(E12_meV)/6.0
    delta   = 0.0 if delta_c == 0 else float(E23_meV)/(2.0*delta_c)
    return delta_c, delta

def valley_weights(vec: np.ndarray, ordering=DEFAULT_ORDER):
    """Return dict of |amplitudes|^2 per valley for a 6-component eigenvector."""
    w = np.abs(vec)**2
    return {ordering[i]: float(w[i]) for i in range(6)}
