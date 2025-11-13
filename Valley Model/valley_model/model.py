
# model.py
import numpy as np
from .valley import build_vo_block, build_valley_diag, with_spin, solve_valley, valley_weights, DEFAULT_ORDER

def build_valley_model(params: dict, report: bool = True):
    """Construct H_valley = diag(E_μ) + H_VO and (optionally) ⊗ I_2 for spin.
    params: {
        'ordering': [...],
        'delta_c': float (meV),
        'delta': float,
        'spin': bool,
        'valley_shifts': {'Ex':..,'Ex_bar':..,'Ey':..,'Ey_bar':..,'Ez':..,'Ez_bar':..}
    }
    Returns: (E, V, H)
    """
    order   = params.get('ordering', DEFAULT_ORDER)
    delta_c = float(params['delta_c'])
    delta   = float(params.get('delta', 0.0))
    spin    = bool(params.get('spin', True))
    shifts  = params.get('valley_shifts', {})

    H_vo   = build_vo_block(delta_c, delta, ordering=order)
    H_diag = build_valley_diag(shifts, ordering=order)
    H6     = H_diag - H_vo

    E0 = float(params.get('baseline_meV', 0.0))
    if E0 != 0.0:
        H6 += E0 * np.eye(6)

    H      = with_spin(H6, spin_enabled=spin)
    E, V = solve_valley(H)

    if report:
        print("Valley model ordering:", order)
        print("Δc (meV) =", delta_c, "  δ =", delta)
        print("Valley shifts (meV):", {k:shifts.get(k,0.0) for k in ['Ex','Ex_bar','Ey','Ey_bar','Ez','Ez_bar']})
        print("\nEigenvalues (meV):", np.round(E.real, 6))
        # If spin is enabled, collapse weights by summing spin partners for the lowest state
        if spin:
            v0 = V[:, 0]
            # reshape (6,2) assuming valley-major ⊗ spin-minor
            v0_reshaped = v0.reshape(6, 2, order='C')
            w6 = (np.abs(v0_reshaped)**2).sum(axis=1)  # sum over spin
            weights = {order[i]: float(w6[i]) for i in range(6)}
        else:
            weights = valley_weights(V[:,0], ordering=order)
        print("Ground-state (A1-like) valley weights:", weights)
        print("baseline_meV =", E0)
    return E, V, H
