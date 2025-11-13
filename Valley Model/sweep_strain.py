
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#  import parsers and builders from the valley module
try:
    from valley_model.xmlio_valley import parse_valley_xml
    HAVE_XML = True
except Exception:
    HAVE_XML = False

from valley_model.valley import build_vo_block, build_valley_diag, solve_valley, DEFAULT_ORDER


# uniaxial z-strain range (dimensionless)
S_MIN = 0
S_MAX = -1e-2
S_STEP = -1e-3

# deformation potentials in eV... 
XI_D = 1.10   # hydrostatic
XI_U = 9.16   # uniaxial/shear

# poisson contraction model 
USE_POISSON = True
NU = 0.28

# read Δc, δ, baseline from an XML:
PARAMS_XML = "params_valley.xml"

# fallback defaults (used if XML isn't found or missing fields)
DELTA_C_FALLBACK = 2.0      # meV
DELTA_FALLBACK   = 0.25     # unitless
BASELINE_FALLBACK = 0 # meV

# output paths
OUT_DIR = Path("outputs_strain")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "energies_vs_strain.csv"
PNG_PATH = OUT_DIR / "energies_vs_strain.png"

def get_params():
    delta_c = DELTA_C_FALLBACK
    delta   = DELTA_FALLBACK
    baseline = BASELINE_FALLBACK
    if HAVE_XML:
        try:
            params = parse_valley_xml(PARAMS_XML)
            delta_c = float(params.get("delta_c", delta_c))
            delta   = float(params.get("delta", delta))
            baseline = float(params.get("baseline_meV", baseline))
            print(f"[sweep] Using XML params: Δc={delta_c} meV, δ={delta}, baseline={baseline} meV")
        except Exception as e:
            print("[sweep] XML parse failed, using fallbacks:", e)
    else:
        print(f"[sweep] xmlio_valley not found; using fallbacks Δc={delta_c}, δ={delta}, baseline={baseline}")
    return delta_c, delta, baseline

def valley_shifts_meV_from_strain(s: float, xi_d: float, xi_u: float, use_poisson: bool, nu: float):
    "Compute valley-diagonal energy shifts (meV) for uniaxial z-strain s."
    "ΔE_α = Ξ_d Tr(ε) + Ξ_u (ε_αα - Tr(ε)/3)"
    if use_poisson:
        exx = -nu*s
        eyy = -nu*s
        ezz = s
    else:
        exx = 0.0
        eyy = 0.0
        ezz = s
    tr = exx + eyy + ezz
    Ex = (xi_d*tr + xi_u*(exx - tr/3.0)) * 1000.0
    Ey = (xi_d*tr + xi_u*(eyy - tr/3.0)) * 1000.0
    Ez = (xi_d*tr + xi_u*(ezz - tr/3.0)) * 1000.0
    return {'Ex':Ex, 'Ex_bar':Ex, 'Ey':Ey, 'Ey_bar':Ey, 'Ez':Ez, 'Ez_bar':Ez}

ALPHA_PER_1e3 = 0.01   

def deviatoric_component_z(s, use_poisson=True, nu=0.28):
    """Return s_dev = εzz - Tr(ε)/3 (dimensionless)."""
    if use_poisson:
        exx, eyy, ezz = -nu*s, -nu*s, s
    else:
        exx, eyy, ezz = 0.0, 0.0, s
    tr = exx + eyy + ezz
    return ezz - tr/3.0

def delta_c_of_s(s, delta_c0, alpha_per_1e3, use_poisson=True, nu=0.28):
    """Δc(s) = Δc0 * (1 + alpha_per_1e-3 * |s_dev| / 1e-3). Keeps units in meV."""
    s_dev = abs(deviatoric_component_z(s, use_poisson, nu))
    scale = alpha_per_1e3 * (s_dev / 1e-3)
    return delta_c0 * (1.0 + scale)


def energies_for_s(s: float, delta_c0_meV: float, delta: float, baseline_meV: float) -> np.ndarray:
    shifts = valley_shifts_meV_from_strain(s, XI_D, XI_U, USE_POISSON, NU)
    H_diag = build_valley_diag(shifts, ordering=DEFAULT_ORDER)

    delta_c_meV = delta_c_of_s(s, delta_c0_meV, ALPHA_PER_1e3, USE_POISSON, NU)

    H_vo   = build_vo_block(delta_c_meV, delta, ordering=DEFAULT_ORDER)
    H6     = H_diag - H_vo + baseline_meV * np.eye(6)   

    E_abs, _ = solve_valley(H6)

    # binding wrt current CB minimum
    E_cb_min = baseline_meV + min(shifts['Ex'], shifts['Ey'], shifts['Ez'])
    return (E_abs - E_cb_min).real


def main():
    delta_c_meV, delta, baseline_meV = get_params()
    s_vals = np.arange(S_MIN, S_MAX + 0.5*S_STEP, S_STEP)
    all_E = np.array([energies_for_s(s, delta_c_meV, delta, baseline_meV) for s in s_vals])

    # Save CSV
    header = "strain,E0_meV,E1_meV,E2_meV,E3_meV,E4_meV,E5_meV"
    data = np.column_stack([s_vals, all_E])
    np.savetxt(CSV_PATH, data, delimiter=",", header=header, comments="", fmt="%.8g")
    print(f"[sweep] Wrote {CSV_PATH}")

    # Plot
    plt.figure(figsize=(6,4))
    for i in range(6):
        plt.scatter(s_vals, all_E[:, i], label=f"Level {i}")
    plt.xlabel("Uniaxial strain s (dimensionless)")
    plt.ylabel("Energy (meV)")
    plt.title("Valley–orbit levels vs uniaxial z-strain")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    print(f"[sweep] Wrote {PNG_PATH}")
    plt.show()

if __name__ == "__main__":
    main()
