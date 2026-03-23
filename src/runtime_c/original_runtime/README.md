# Original Runtime Staging

This directory stages the original C runtime sources that will be absorbed into
the standalone `xmvb-cpp` runtime.

The current goal is to replace the legacy extractor path with a managed mixed
C/C++ runtime:

- `cpp/runtime_c/`
  C-side input, structure, integral, and guess preparation
- `cpp/runtime/`
  C++ loader that builds `CppVbInput`
- `cpp/vb/`
  C++ VBSCF matrix, analytic gradient, and optimizer kernels

## Imported First Batch

These files have been copied in as the first directly managed C runtime entry
points because they sit on the runtime boundary and have relatively small local
closures:

- `guess/vbguess.c`
- `vbutil/vbprep.c`
- `vbutil/readinp_vb.c`
- `vbint/int_data_trans.c`

## Imported Helper Closure

The standalone runtime now also compiles a growing set of helper closures
directly into `xmvb_cpp_runtime` through local wrapper translation units under
`cpp/runtime_c/local_runtime/`. This batch covers the VB/input side of the
runtime boundary so `readinp`, `readinp_vb`, `vbprep`, and `vbguess` can stop
pulling those symbols from the legacy static libraries:

- `src/vbutil/char_util.c`
- `src/vbutil/split_string.c`
- `src/vbutil/fvbsec.c`
- `src/vbutil/readstr.c`
- `src/vbutil/genstr.c`
- `src/vbutil/getstr.c`
- `src/vbutil/readorb.c`
- `src/vbutil/getorb.c`
- `src/vbutil/detect_guess.c`
- `src/vbutil/getpoints.c`
- `src/vbutil/getfrg.c`
- `src/vbutil/init_vb_param.c`
- `src/vbutil/readint_head.c`
- `src/vbutil/readint.c`
- `src/vbutil/detect_blocks.c`
- `src/vbutil/getvars.c`
- `src/vbutil/init_bovb.c`
- `src/vbutil/check_dir2e.c`
- `src/vbutil/get_total_mem.c`
- `src/vbutil/print_crd.c`
- `src/vbutil/print_init_guess.c`
- `src/vbutil/print_vb_comp_info.c`
- `src/vbutil/del_vb_str.c`
- `src/vbutil/del_dfvb_str.c`
- `src/vbint/build_g1d.c`
- `src/vbint/nb3idx.c`
- `src/math/lab.c`
- `src/math/cvitra.c`
- `src/math/normalize.c`
- selected small `mol` helpers: `xint_gtolen`, `charge2ele`, `get_ang_tag`

## Current Status

The standalone runtime has moved past the original HF-boundary plan:

- `run_cpp_vbscf` now links against `xmvb_cpp_runtime`, `xmvb_cpp_vb`, and
  external numerical libraries only
- the executable no longer links the legacy start-group of `symm`, `gopt`,
  `vbdrv`, `vbpt2`, `dfvb`, `vbscf`, `vbci`, `vbgus`, `inpout`, `vbint`,
  `vbmath`, `vbutil`, `eda`, `scf`, `solver`, `fock`, and `mol`
- the `cpp/runtime_c/local_runtime/` sources used by the standalone build are
  now locally owned source files instead of one-line wrappers that include
  `src/.../*.c` or `original_runtime/.../*.c`
- the current standalone path supports `INT=CINT`
- nuclear repulsion, input loading, VB preparation, one-electron integrals,
  two-electron integrals, and initial guess generation are all handled from the
  managed `cpp/runtime_c/` side

The remaining work is no longer “make the executable link without legacy
libraries”. That milestone is complete.

## Next Batch

The next phase is source-level and quality-level decoupling:

- continue replacing `#include "../../../src/.../*.c"` wrappers with locally
  owned sources where practical; for the active standalone build this is mostly
  complete, with the legacy `hf.c` staging wrapper left as the main exception
- shrink header coupling on `include/vb/vb.h`, `include/mol/mol.h`, and
  `include/scf/hf.h`
- improve standalone initial guess quality so convergence is closer to the
  former HF/SAD-backed path
- decide which legacy-only features stay out of scope for the standalone path,
  especially non-`INT=CINT` integral modes and unsupported guess modes such as
  `GUS_NBO`

## Current Staging Coverage

The copied sources are also compiled into a dedicated non-linked staging
library through CMake:

- `xmvb_cpp_original_runtime_staging`

This keeps the imported C runtime sources under the `cpp/` build without yet
introducing duplicate symbol conflicts into the executable link.
