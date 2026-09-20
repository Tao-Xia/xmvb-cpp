# XMVB source reference on Hanhai25

## Scope and provenance

The user-supplied `xmvb4.0.zip` is an independent numerical/code reference,
not a source dependency of XMVB-CPP. Vendor sources, binaries, Git metadata,
and generated molecular outputs are not committed to this repository.

- Archive SHA-256:
  `0189022430dccc51f44a5568a4db878d00943e9fff8388d975143b44533362fd`.
- Archived Git HEAD: `refs/heads/DeepVBH`,
  `2aaff6dd7ac06efee95047a8bbe6ef4c99a26f35`. This identifies the archived
  HEAD, not a verification that its working tree matches that commit.
- The package contains an XMVB 4.0 banner, XEDA build documentation, and
  TBVBSCF code. It is the supplied snapshot, not an independently verified
  pristine release. No top-level blanket source license was found.
- Local reference root:
  `/home/txia/project/xmvb-cpp/build/reference/xmvb4.0-20260920/`.
  The extracted original is under `source/xmvb4.0/`; `source-fabs/` is the
  explicitly patched comparison copy.
- Remote reference root:
  `/home/guqqgroup/taoxia/work/xmvb4.0-reference-20260920/`.
  The original uses `source/`, `build/`, and `install/`. The patched copy uses
  `source-fabs/`, `build-fabs/`, and `install-fabs/`.
- Remote scripts and logs:
  `/home/guqqgroup/taoxia/xmvb-runs/xmvb4.0-reference-20260920/`.
  Downloaded logs are under the local reference root's `runs/`.

Stale object/module files, archived build directories, Git metadata, and
historical training matrices are not transferred into the build source mirror.
The transferred source/resource files were checked with `rsync -anc`.

## Build and runtime contract

Build through Slurm, not on the login node. The tested setup uses 32 CPU cores,
64 GiB, `CPU-256C768GB`, `qos_cpu-256c768gb`, module `anaconda3/25.06`, and
the existing environment:

```text
/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
```

The environment provides GCC/GFortran 15.3.0, CMake 4.4.3, Ninja, OpenBLAS,
Libcint, and Libxc 7.1.2 including its Fortran interface. No sudo or changes
to this shared environment were needed. The build is Release, native CPU
specialization is off, and the vendor's `-ffast-math` is retained and recorded.
C compilation uses ccache.

The initial unmodified build (job `247289`) failed on a missing C declaration
for `cint1e_rinv_cart` and a value-less return from a non-void function.
The Libcint symbol was verified present. Job `247292` compiled and installed
the original source using only these additional compiler diagnostic settings:

```text
-Wno-error=implicit-function-declaration -Wno-error=return-mismatch
```

These preserve warnings; they do not repair the underlying vendor declarations.
The vendor already enables GFortran's `-fallow-argument-mismatch`.
`LAPACK_ROOT_DIR`, `CINT_ROOT_DIR`, and `LIBXC_ROOT_DIR` all select the environment
above; `LAPACKE_LIB` selects its `lib/libopenblas.so`. An explicit library
search path and rpath select the same environment instead of relying on the
vendor's hard-coded `/share/apps/libxc/6.0.0/lib64` link directory.

`build.sbatch original` and `build.sbatch fabs` in the remote log directory
record the complete CMake/build/install commands. Submit them with `sbatch`.

Always invoke the installed executable by an absolute path ending in
`/bin/xeda.exe` (or the vendor-supported `/bin/xmvb.exe`). The parser derives
resource paths from that executable name, not from `VBDIR`. The prefix must
contain both `basis/` and `data/`, including the auxiliary basis and exchange
grids even for the tested `INT=LIBCINT` calculation.

For example, inside an allocated Slurm job and an isolated writable run directory:

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
/home/guqqgroup/taoxia/work/xmvb4.0-reference-20260920/install/bin/xeda.exe \
  -n 8 F2.xmi > run.out 2> run.err
```

The Slurm scripts set the environment library path explicitly. Do not run the
build-tree binary directly: its path does not satisfy the resource lookup.

## Narrow L-BFGS stopping repair

`src/vbscf/vblbfgs.c` calls the C integer function `abs(de)` although `de` is
a `double`. For finite energy changes with magnitude below 1 Eh, integer
conversion yields zero, so the nominal energy-change check is ineffective.
The comparison copy changes only this expression and adds its proper header:

```diff
 #include <stdio.h>
 #include <stdlib.h>
+#include <math.h>
@@
-    if ((abs(de) < vb_str->epg && gxn < vb_str->gpg) || iflag == 0) {
+    if ((fabs(de) < vb_str->epg && gxn < vb_str->gpg) || iflag == 0) {
```

The separate `iflag == 0` termination remains unchanged. Therefore this is a
repair of the explicit energy test, not a redesign of all stopping conditions.
The original source and installed binary remain available for historical replay.
Job `247298` builds/installs the patched copy and job `247299` reruns F2.
Both complete successfully. The only source-file difference between the
original and patched copies is the two-line change shown above.

XMVB-CPP's L-BFGS and TNHVP termination checks use `std::abs` on doubles with
`<cmath>` available; those checks do not have this integer-conversion defect.

## Numerical qualification

The bundled `test/F2.xmi` uses 1.4 angstrom F2, Cartesian cc-pVDZ, a 2e/2o
active space, three structures, sparse HAO orbitals, exact Libcint integrals,
and `ISCF=5`. Its basis file is byte-identical to XMVB-CPP's cc-pVDZ file.
The canonical CPP deck differs only in optimizer/eigensolver controls.

Job `247293` runs the bundled deck unchanged in both executables on eight
threads. XMVB-CPP uses dense structure diagonalization, its default block-LBFGS
for `ISCF=5`, projected-gradient tolerance `1e-3`, and energy tolerance `1e-7 Eh`.
Vendor `ISCF=5` defaults to `gpg=2e-3` and `epg=1e-7`; these gradient norms and
termination conditions are not equivalent to the CPP settings.

| Calculation | Reported iterations | Total energy / Eh |
|---|---:|---:|
| Bundled historical `F2.out` | 26 | -198.751155644167 |
| Rebuilt unmodified snapshot, job 247293 | 25 | -198.751155635768 |
| Snapshot with `fabs` repair, job 247299 | 27 | -198.751155753415 |
| XMVB-CPP, same input, job 247293 | 8 | -198.751155821808 |

The rebuilt vendor result differs from its archived energy by `8.399e-9 Eh`.
The unmodified vendor/CPP final-energy difference is `1.86040e-7 Eh`.
This is a build/small-molecule smoke comparison, not an equal-accuracy optimizer
benchmark or proof of equality at fixed orbitals. Both programs generated their
own initial orbitals from the same deck; no identical-guess claim is made.

The unmodified snapshot stops at step 25 with printed energy change
`-9.595e-7 Eh`, exceeding its nominal `1e-7 Eh` threshold. The patched run
continues to step 27, with printed energy change `-7.22e-8 Eh` and GNORM
`0.0011057809`, satisfying its explicit energy/gradient checks. The final
vendor/CPP energy difference is then `6.8393e-8 Eh`; CPP reproduces the same
eight-step result in job `247299`. This isolates a real premature-stopping
effect on F2. It does not establish the cause of discrepancies on other systems.

Original installed executable SHA-256:
`d6c7c375e18ca263b5e9a36228199c7f91ce397b1a65ea8a68eab5a17616aa52`.
Patched installed executable SHA-256:
`5ad143a885a27b5c4fc2f36f38442129278a2c322f3ca1389fc52244875a9be2`.
The CPP binary is the production code qualified at `a5ff975`, with SHA-256
`99474663e121912118f74b60762aa55d4e660c3dba8b6db7438eaa7b91e40186`.

## Source map for subsequent audits

Paths below are relative to the vendor source root, not to XMVB-CPP `src/`.

| Purpose | Entry point |
|---|---|
| Input controls | `src/inpout/readinp.c`: `getcom` |
| Imported orbitals | `src/vbgus/vb_readguess.c`: `vb_readguess`, `$GUS` |
| Energy, H/S assembly, gradient | `src/vbscf/gradient_rdm.F90`: `Gradient_rdm`, `Hamhd`, `HamOv` |
| Structure eigenproblem | `src/math/eigencalc.c`: `eigencalc`, `LAPACKE_dsygv` |
| Hessian initialization | `src/vbscf/hes_rdm_vbscf.F90`: `iopt6_init` |
| Hessian/Newton implementation | `src/vbscf/Iopt6.F90`: `Iopt6`, `NewtonOpt`, `VBMatrix` |
| L-BFGS driver | `src/vbscf/vblbfgs.c`: `vblbfgs` |
| L-BFGS kernel | `src/vbscf/lbfgs.F90`: `lbfgs_driver`, `LBFGS_F` |

The source supports `VBSCF ISCF=5 GUESS=READ ITMAX=0` with complete `$GUS`
coefficients for an orbital-fixed evaluation with structure diagonalization.
It evaluates before taking an L-BFGS step, then normalizes and reevaluates for
reporting. It prints `Failed to converge in 0 iterations` in this deliberately
non-optimizing mode, so ordinary convergence-marker checks are inappropriate.
This mode has been source-reviewed but not numerically qualified here.

Next comparisons should freeze identical orbital coefficients/supports and
check energy, structure H/S elements, and derivatives separately. Do not assign
every final-energy difference to stopping thresholds, and do not infer general
correctness from this F2 smoke test.
