# 0.1.5 — HAO/CIAH stability fixes

## Changes

- Stabilize the inactive-orbital frame, density, occupied-space projector and dual columns when localized orbitals become nearly linearly dependent. Reuse corrected factors rather than repeating inverse-Gram products.
- Remove physical-metric solves used only for CIAH reporting. Physical responses, acceptance criteria and convergence thresholds are unchanged.
- Report the number of committed iterations when an optimizer backend raises an exception, instead of incorrectly reporting zero iterations.
- Add frame/density/dual consistency and exception-reporting regression tests.
- Include optional numerical-AO and local-RI infrastructure previously committed to master. This is preparatory infrastructure, not periodic VBSCF support; optional dependencies remain disabled in the CPU release.

## Validation and limits

Hanhai25 native qualification: ten selected unit tests passed; O2/water-cation HAO/FULL, F2 HAO/OEO, FeCl2 and 241 converged. The original O2 case reaches -225.380240176481 Eh in 29 accepted steps and 187 HVPs. This fixes the observed failure; it does not resolve slow CIAH convergence for every ill-conditioned HAO case. Unsuccessful experimental preconditioners and factor-reuse variants are excluded.

## Installation

The program archive is `xmvb-cpp-0.1.5-linux-x86_64-cpu.tar.gz` (Linux x86-64, CPU). Reuse `runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif`; no runtime update is required. GPU executables and Arm binaries are not included in this archive.

Download the program archive and its SHA-256 file from this release, or use the public GHCR program artifact:

```sh
oras pull ghcr.io/tao-xia/xmvb-cpp-release:xmvb-cpp-0.1.5-linux-x86_64-cpu
sha256sum -c xmvb-cpp-0.1.5-linux-x86_64-cpu.tar.gz.sha256
```

Extract into a fresh program directory and set `XMVB_PROGRAM_DIR` to it. Keep `XMVB_RUNTIME_IMAGE` pointing to the existing runtime. Source builds are documented in BUILDING.md.
