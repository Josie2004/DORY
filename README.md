# DORY – Tight-Binding Bandstructure Runner

This project provides a minimal framework for building and plotting **tight-binding (TB) Hamiltonians** for 3D crystals with a **two-atom basis** (e.g. diamond, zincblende, rocksalt).  

The first version is focused on 3D **sp³d⁵** and **sp³** models defined in simple XML files.

---

## Features (v1)

- Parse material parameters from XML:
  - Orbital basis (`s`,`sp3`, `sp3d5`) or any limited combinations.
  - Atomic positions (two atoms per cell)
  - Nearest-neighbour bonds (δ vectors)
  - Onsite energies
  - Slater–Koster (SK) integrals
  - Lattice constant
- Build the Hamiltonian for a 2-atom primitive cell:

  ![Hamiltonian](https://latex.codecogs.com/svg.latex?H%28%5Cmathbf%7Bk%7D%29%3D%5Cbegin%7Bbmatrix%7DH_%7BAA%7D%26H_%7BAB%7D%28%5Cmathbf%7Bk%7D%29%5C%5CH_%7BAB%7D%5E%7B%5Cdagger%7D%28%5Cmathbf%7Bk%7D%29%26H_%7BBB%7D%5Cend%7Bbmatrix%7D)

- Solve eigenvalues along high-symmetry paths in the **FCC Brillouin zone**.
- Save results (`.dat`) and plot bandstructures (`.png`).
- Option to **show plots interactively**.

---

## Example XML: Zincblende Si (sp³d⁵)

```xml
<dory>
  <basis> sp3d5 </basis>
  <lattice_constant units="nm">0.543</lattice_constant>

  <atoms>
    <atom id="A" pos="0.0 0.0 0.0"/>
    <atom id="B" pos="0.13575 0.13575 0.13575"/>
  </atoms>

  <neighbors>
    <bond from="A" to="B" delta=" 0.13575  0.13575  0.13575"/>
    <bond from="A" to="B" delta=" 0.13575 -0.13575 -0.13575"/>
    <bond from="A" to="B" delta="-0.13575  0.13575 -0.13575"/>
    <bond from="A" to="B" delta="-0.13575 -0.13575  0.13575"/>
  </neighbors>

  <onsite>
    <A s="-2.803316" p="4.096984" d_t2g="12.568228" d_eg="12.568228"/>
    <B s="-2.803316" p="4.096984" d_t2g="12.568228" d_eg="12.568228"/>
  </onsite>

  <sk builtin="true">
    <AB ss_sigma="-2.066560" sp_sigma="3.144266" ps_sigma="3.144266"
        pp_sigma="4.122363" pp_pi="-1.522175"
        sd_sigma="-2.131451" ds_sigma="-2.131451"
        pd_sigma="-1.127068" pd_pi="2.383978"
        dp_sigma="-1.127068" dp_pi="-2.383978"
        dd_sigma="-1.408578" dd_pi="2.284472" dd_delta="-1.541821"/>
    <BA ss_sigma="-2.066560" sp_sigma="3.144266" ps_sigma="-3.144266"
        pp_sigma="4.122363" pp_pi="-1.522175"
        sd_sigma="-2.131451" ds_sigma="-2.131451"
        pd_sigma="-1.127068" pd_pi="2.383978"
        dp_sigma="1.127068"  dp_pi="2.383978"
        dd_sigma="-1.408578" dd_pi="2.284472" dd_delta="-1.541821"/>
  </sk>
</dory>
```

---

## Usage

Run bandstructure:

```bash
python examples/runner.py examples/zincblende_sp3d5_Si.xml --show --save
```

Options:

- `--save` → writes results to `out_<NAME>/`
- `--show` → shows matplotlib plot interactively
- `--steps N` → number of interpolation steps per path segment
- `--labels L Γ X ...` → custom path (default = FCC loop)

---

## Output Files

In `out_<NAME>/`:

- `<NAME>_k_points.dat` – fractional k-points
- `<NAME>_k_distance.dat` – cumulative k-path distances
- `<NAME>_energies.dat` – eigenvalues (bands)
- `<NAME>_bands.png` – bandstructure plot

---

## Default High-Symmetry Path (FCC)

The default path is:

**L → Γ → X → W → K → L → W → X → K → Γ**

Fractional coordinates (units of (2π/a)):

- L = (0.5, 0.5, 0.5)
- Γ = (0.0, 0.0, 0.0)
- X = (1.0, 0.0, 0.0)
- W = (1.0, 0.5, 0.0)
- K = (0.75, 0.75, 0.0)

---

## Current Limitations

- Only 2-site primitive cells (A, B) are supported.
- Only 3D systems (`Hamiltonian3D`) implemented.
- Only nearest-neighbour TB couplings.
- Spin–orbit coupling not yet included.

---

## Roadmap

- Add `Hamiltonian1D` (chains) and `Hamiltonian2D` (square/honeycomb).
- Support N-atom supercells (beyond 2).
- Include spin–orbit coupling.
- Add strain-dependent couplings.
- Extend to interfaces & heterostructures.

---

## License

MIT.

---

