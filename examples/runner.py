#!/usr/bin/env python3
"""
runner.py -- General TB bandstructure runner for XML configs
Usage:
    python runner.py examples/rocksalt_sp3d5_PbTe.xml --show --save
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dory.xmlio import load_config
from dory.model3d import Hamiltonian3D
from dory.sk import SKParams


# ---------- High-symmetry path (FCC BZ as default) ----------
HS_frac = {
    "L": np.array([0.5, 0.5, 0.5], float),
    "Γ": np.array([0.0, 0.0, 0.0], float),
    "X": np.array([1.0, 0.0, 0.0], float),
    "W": np.array([1.0, 0.5, 0.0], float),
    "K": np.array([0.75, 0.75, 0.0], float),
}
default_path = ["L", "Γ", "X", "W", "K", "L", "W", "X", "K", "Γ"]


def frac_to_kcart(frac: np.ndarray, a_nm: float) -> np.ndarray:
    """Convert fractional k (units 2π/a) to cartesian (nm⁻¹)."""
    return (2 * np.pi / a_nm) * frac


def interpolate_path(points_frac, labels, steps, a_nm):
    """Interpolate k-path through high-symmetry points."""
    fracs = []
    ticks, acc = [0], 0
    for s in range(len(labels) - 1):
        A = points_frac[labels[s]]
        B = points_frac[labels[s+1]]
        seg = np.linspace(A, B, steps, endpoint=True)
        if s > 0:
            seg = seg[1:]  # drop duplicate knot
        fracs.append(seg)
        acc += len(seg)
        ticks.append(acc - 1)
    kf = np.vstack(fracs)
    kc = frac_to_kcart(kf, a_nm)
    kd = np.zeros(kc.shape[0])
    if kc.shape[0] > 1:
        kd[1:] = np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))
    return kf, kc, kd, ticks


def main():
    parser = argparse.ArgumentParser(description="General TB bandstructure runner")
    parser.add_argument("xml", help="Input XML config file")
    parser.add_argument("--steps", type=int, default=350,
                        help="Steps per path segment")
    parser.add_argument("--save", action="store_true",
                        help="Save data and plot to out_<NAME>")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive matplotlib window")
    parser.add_argument("--labels", nargs="+", default=default_path,
                        help="Path labels sequence (default FCC loop)")
    args = parser.parse_args()

    # ==== Load config ====
    cfg = load_config(args.xml)
    name = Path(args.xml).stem
    a_nm = cfg.lattice_constant  # lattice constant (nm)
    sk = SKParams(AB=cfg.sk_AB.__dict__, BA=cfg.sk_BA.__dict__)
    ham = Hamiltonian3D(
        basis=cfg.basis,
        atomA_pos=cfg.atomA.pos,
        atomB_pos=cfg.atomB.pos,
        neighbors=cfg.bonds,
        onsite_A=cfg.onsite.A,
        onsite_B=cfg.onsite.B,
        sk_params=sk,
    )

    # ==== Build path ====
    kf, kc, kd, ticks = interpolate_path(HS_frac, args.labels, args.steps, a_nm)
    nb = len(cfg.basis) * 2
    Evals = np.zeros((kc.shape[0], nb))
    for i, k in enumerate(kc):
        w, _ = ham.solve_k(k)
        Evals[i, :] = w.real

    # ==== Save outputs ====
    outdir = Path("out_" + name)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)
        np.savetxt(outdir / f"{name}_k_points.dat", kf, fmt="%.8f")
        np.savetxt(outdir / f"{name}_k_distance.dat", kd, fmt="%.8f")
        np.savetxt(outdir / f"{name}_energies.dat", Evals, fmt="%.8f")

    # ==== Plot ====
    plt.figure(figsize=(9, 5.2))
    for b in range(nb):
        plt.plot(kd, Evals[:, b], "b-", lw=1)
    for idx in ticks:
        plt.axvline(kd[idx], color="k", ls="--", lw=0.5)
    plt.xticks([kd[idx] for idx in ticks], args.labels)
    plt.xlabel("k-path"); plt.ylabel("Energy (eV)")
    plt.title(f"{name} bandstructure (NN sp³d⁵ TB, a={a_nm:.3f} nm)")
    plt.tight_layout()

    if args.save:
        plt.savefig(outdir / f"{name}_bands.png", dpi=180)
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()