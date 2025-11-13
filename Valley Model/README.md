
# Valley-Orbit (VO) Model Scaffold

This code provides a clean, self-contained implementation of a valley basis state effective-mass
Hamiltonian for donor valley–orbit coupling in silicon (or other zincblend structures).

## Contents
- `main.py` — simple CLI entry point
- `valley_model/valley.py` — builds the 6x6 VO block, optional ⊗ I2 for spin
- `valley_model/model.py` — assembles H = diag(Eμ) + H_VO, solves, and reports
- `valley_model/xmlio_valley.py` — parses `<valley_model>` parameters from XML
- `valley_model/runner_valley.py` — convenience script to run the model
- `params_valley.xml` — example parameters

## Quick start
```bash
cd valley_orbit_project
python main.py valley params_valley.xml
# or
python valley_model/runner_valley.py params_valley.xml
```

## Parameters
- `delta_c` (meV): base off-diagonal VO coupling (sets the singlet–doublet gap E12 = 6Δc)
- `delta` (dimensionless): small enhancement for opposite-valley couplings (sets E23 = 2δΔc)
- `valley_shifts` (meV): per-valley diagonal shifts (e.g., from strain) in the ordering
  `+x, -x, +y, -y, +z, -z`

## Output
Prints eigenvalues (meV) and the ground-state valley composition. With spin enabled,
the Hamiltonian is promoted to 12×12 via ⊗ I₂ (energies are unchanged, but degeneracies double). This feature hasn't actually been implemented yet. 
