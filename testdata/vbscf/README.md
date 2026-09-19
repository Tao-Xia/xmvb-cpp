# VBSCF regression inputs

This directory contains version-controlled inputs used to validate the VBSCF
orbital optimizer across sparse HAO and full-AO OEO coordinate models.

| Input | Orbital model | Purpose |
| --- | --- | --- |
| `F2.xmi` | sparse HAO | Small finite-difference and optimizer smoke test |
| `F2_OEO.xmi` | full-AO OEO | Small derivative and coordinate regression |
| `C6H6.xmi` | sparse HAO | Covalent-structure contraction regression |
| `C6H6.full.xmi` | sparse HAO | Full-structure contraction regression |
| `241_VBSCF.xmi` | sparse HAO | Closed-shell benzene performance regression |
| `MnF2.xmi` | sparse HAO | Difficult open-shell transition-metal regression |
| `FeCl2.xmi` | full-AO OEO | Production-size open-shell full-AO regression |
| `10698_VBSCF.xmi` | sparse HAO | Larger closed-shell scaling regression |
| `7975_vb.xmi` | sparse HAO | Six-orbital XMVB numerical-reference case |
| `7963_vb.xmi` | sparse HAO | Six-orbital closed-shell numerical-reference case |
| `csd_YAMSAI_vb.xmi` | sparse HAO | Seven-orbital difficult convergence case |
| `csd_LOFLEA_vb.xmi` | sparse HAO | Large 12e/12o unique-string scaling case |

The decks include their orbital guesses, so they are self-contained. The four
`*_vb.xmi` decks above were recovered from the `Input File` sections of their
archived XMVB/XMVB-CPP outputs; insignificant trailing whitespace was
normalized. Molden files and other program outputs are generated artifacts and
do not belong in this directory.

For reproducible optimizer comparisons, use the
`nonredundant_truncated_newton` backend with its analytic HVP and pin
the BLAS/OpenMP thread counts explicitly. Historical numerical and performance
results are recorded under `article/` and `docs/`; this directory is the
canonical location for the corresponding inputs.
