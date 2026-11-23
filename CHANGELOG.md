# xmvb-cpp 0.1.5

## Changes

- Improve HAO/CIAH numerical stability when inactive orbitals become nearly linearly dependent.
- Reuse inactive-orbital factors and remove metric solves used only for reporting.
- Correct the iteration count reported after optimizer errors.
- Add optional numerical-AO and local-RI infrastructure for future development.

## Installation

Download the Linux x86-64 CPU program archive and its SHA-256 file from [GitHub Releases](https://github.com/Tao-Xia/xmvb-cpp/releases/tag/xmvb-cpp-0.1.5). See `INSTALL.md` for setup instructions. Reuse `runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif` from GHCR; no runtime update is required.

The optional numerical-AO/local-RI components do not yet provide periodic VBSCF calculations.
