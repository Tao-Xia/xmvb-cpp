#pragma once

#include <limits.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "inpout/input.h"
#include "mol/mol.h"
#include "scf/hf.h"
#include "runtime_c/local_runtime_api.h"
#include "vb/vb.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*XmvbCppDestroyHfFn)(hf_info hf);

struct XmvbCppRuntimeHandle {
  para_info parallel_info;
  inp_info input_info;
  mol_info molecule;
  hf_info hf_wavefunction;
  vb_info vb_wavefunction;
  XmvbCppDestroyHfFn destroy_hf_fn;
  int xscf_world_initialized;
  char runtime_xdat_path[PATH_MAX];
};

#ifdef __cplusplus
}
#endif
