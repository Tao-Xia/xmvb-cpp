# XMVB source reference on Hanhai25

## Scope and provenance

The user-supplied `xmvb4.0.zip` is an independent numerical/code reference,
not a source dependency of XMVB-CPP. Vendor sources, binaries, Git metadata,
and generated molecular outputs are not committed to this repository.

- Archive SHA-256:
  `0189022430dccc51f44a5568a4db878d00943e9fff8388d975143b44533362fd`.
- The numerical baseline is the repository's `xmvb4.0` branch at
  `a3613ed32ebffcaf2d98c1c9124555708bea995b` (699 reachable commits). The
  local and `origin/xmvb4.0` refs in the archive agree at that commit.
- Archived Git HEAD is instead the user-modified `refs/heads/DeepVBH` at
  `2aaff6dd7ac06efee95047a8bbe6ef4c99a26f35`. Relative to their merge base,
  `xmvb4.0` has 30 unique commits and `DeepVBH` has one. The tips have identical
  `src/vbscf/` trees, but `xmvb4.0` includes later fixes elsewhere, including
  determinant-density handling when one spin sector is empty.
- The archived `DeepVBH` worktree has four additional uncommitted changes:
  `.gitignore`, `CMakeLists.txt`, `cmmakecmd`, and `src/vbscf/vblbfgs.c`.
  These add `-ffast-math`, local build settings, and iteration timing output.
- The package contains an XMVB 4.0 banner, XEDA build documentation, and
  TBVBSCF code. It is the supplied snapshot, not an independently verified
  pristine release. No top-level blanket source license was found.
- Primary local reference root:
  `/home/txia/project/xmvb-cpp/build/reference/xmvb4.0-a3613ed/`.
  `source/` is an exact `git archive` export; `source-fabs/` is its explicitly
  patched comparison copy.
- Primary remote reference root:
  `/home/guqqgroup/taoxia/work/xmvb4.0-a3613ed/`, with corresponding logs in
  `/home/guqqgroup/taoxia/xmvb-runs/xmvb4.0-a3613ed/`.
- The earlier `DeepVBH` worktree study remains under the secondary local root:
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

## Branch comparison

The `xmvb4.0` and `DeepVBH` tips share merge base
`f410e5b9b6e5c0003b29fafd9362c992bdf08d5c`. The entire committed
`src/vbscf/` tree is identical at the two tips, including `gradient_rdm.F90`,
`Iopt6.F90`, `lbfgs.F90`, and the integer-`abs` stopping defect in
`vblbfgs.c`. Thus the main VBSCF equations in the supplied `DeepVBH` commit do
not constitute a separate implementation.

The clean `xmvb4.0` tip is nevertheless the safer baseline. Its 30 later
commits include fixes outside `src/vbscf/`; one relevant example handles an
empty alpha or beta sector in determinant-density construction. Other tip
differences mainly concern the Python interface, point charges, print-level
control, DFVB/VBPT2 fixes, and executable packaging. `DeepVBH` has one unique
commit named `backup`, changing one driver build line. The uncommitted worktree
changes described above must be treated separately from either branch.

## Build and runtime contract

Build through Slurm, not on the login node. The tested setup uses 32 CPU cores,
64 GiB, `CPU-256C768GB`, `qos_cpu-256c768gb`, module `anaconda3/25.06`, and
the existing environment:

```text
/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
```

The environment provides GCC/GFortran 15.3.0, CMake 4.4.3, Ninja, OpenBLAS,
Libcint, and Libxc 7.1.2 including its Fortran interface. No sudo or changes
to this shared environment were needed. The primary build is Release, native
CPU specialization is off, and C compilation uses ccache. The clean `xmvb4.0`
branch compiles with `-O3` and does **not** enable `-ffast-math`; that option
exists only in the archived worktree's uncommitted changes.

The initial archived-worktree build (job `247289`) failed on a missing C declaration
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
search path and rpath select the same environment. The hard-coded
`/share/apps/libxc/6.0.0/lib64` link directory belongs only to the archived
worktree's uncommitted `CMakeLists.txt`; it is absent from the clean branch.

Clean-branch jobs `247304` (original) and `247309` (`fabs` copy) both compile
and install successfully. Their scripts in the primary remote log directory
record the complete CMake/build/install commands.

Always invoke the installed executable by an absolute path ending in
`/bin/xmvb.exe` (or the vendor-supported `/bin/xeda.exe`). The parser derives
resource paths from that executable name, not from `VBDIR`. The prefix must
contain both `basis/` and `data/`, including the auxiliary basis and exchange
grids even for the tested `INT=LIBCINT` calculation.

For example, inside an allocated Slurm job and an isolated writable run directory:

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
/home/guqqgroup/taoxia/work/xmvb4.0-a3613ed/install/bin/xmvb.exe \
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
The clean source and installed binary remain available for historical replay.
The only source-file difference in `source-fabs/` is the two-line change shown
above. The earlier jobs `247298`/`247299` applied the same repair to the archived
worktree; they are secondary evidence rather than the official branch baseline.

XMVB-CPP's L-BFGS and TNHVP termination checks use `std::abs` on doubles with
`<cmath>` available; those checks do not have this integer-conversion defect.

## Numerical qualification

The bundled `test/F2.xmi` uses 1.4 angstrom F2, Cartesian cc-pVDZ, a 2e/2o
active space, three structures, sparse HAO orbitals, exact Libcint integrals,
and `ISCF=5`. Its basis file is byte-identical to XMVB-CPP's cc-pVDZ file.
The canonical CPP deck differs only in optimizer/eigensolver controls.

The clean branch contains no bundled F2 deck, so jobs `247306` and `247310` use
the archive's unchanged `test/F2.xmi`, identified by SHA-256
`0fc25c69084521e04d5d8bfd1ef0aeeb5acc31c45cfcfec0321ba78a433a2bbc`.
XMVB-CPP uses dense structure diagonalization, its default block-LBFGS
for `ISCF=5`, projected-gradient tolerance `1e-3`, and energy tolerance `1e-7 Eh`.
Vendor `ISCF=5` defaults to `gpg=2e-3` and `epg=1e-7`; these gradient norms and
termination conditions are not equivalent to the CPP settings.

| Calculation | Reported iterations | Total energy / Eh |
|---|---:|---:|
| Bundled historical `F2.out` | 26 | -198.751155644167 |
| Clean `xmvb4.0@a3613ed`, job 247306 | 25 | -198.751155607048 |
| Clean `xmvb4.0` with `fabs`, job 247310 | 28 | -198.751155808524 |
| Archived `DeepVBH` worktree, job 247293 | 25 | -198.751155635768 |
| Archived worktree with `fabs`, job 247299 | 27 | -198.751155753415 |
| XMVB-CPP, same input, job 247293 | 8 | -198.751155821808 |

The clean and archived-worktree results differ by `2.8720e-8 Eh`; the latter
was built with its uncommitted `-ffast-math`. This association is direct build
evidence, not a proof that every bit of the difference comes from that option.
This is a build/small-molecule smoke comparison, not an equal-accuracy optimizer
benchmark or proof of equality at fixed orbitals. Both programs generated their
own initial orbitals from the same deck; no identical-guess claim is made.

The clean unmodified branch stops at step 25 with printed energy change
`-9.042e-7 Eh`, exceeding its nominal `1e-7 Eh` threshold. Its `fabs` copy
continues to step 28, with printed energy change `-7.51e-8 Eh` and GNORM
`0.0006574118`, satisfying its explicit energy/gradient checks. The final clean
vendor/CPP energy difference is then `1.3284e-8 Eh`. This isolates a real
premature-stopping effect on F2. It does not establish the cause of discrepancies
on other systems. The archived-worktree `fabs` results above remain useful for
reproducing that worktree, not as the primary XMVB 4.0 reference.

Clean-branch installed executable SHA-256:
`826fe0e5e918cb82a71f30cd425eb9e5ba34605568edbc7a3717a2afb4a76525`.
Clean-branch `fabs` executable SHA-256:
`3de7628492fb64c26a0300ef7e567fee193900b5b6d0c45117e865015021c2dd`.
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
