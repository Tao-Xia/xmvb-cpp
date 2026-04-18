#include "runtime_c/local_runtime_grid_api.h"

#include <stdio.h>

#include "mol/xgrids.h"
#include "runtime_c/local_runtime_api_internal.h"

static void set_error_message(
    char* error_message,
    size_t error_message_capacity,
    const char* message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  snprintf(error_message, error_message_capacity, "%s", message);
}

int xmvb_cpp_runtime_compute_medium_grid_ao_overlap(
    const XmvbCppRuntimeHandle* runtime_handle,
    double* ao_overlap_matrix,
    size_t ao_overlap_matrix_size,
    char* error_message,
    size_t error_message_capacity) {
  Matrix legacy_overlap_matrix = NULL;
  grids_info legacy_grids = NULL;
  int n_basis_functions = 0;

  if (runtime_handle == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime handle is null");
    return 1;
  }
  if (runtime_handle->molecule == NULL || runtime_handle->molecule->bas == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime molecule basis is not initialized");
    return 1;
  }
  if (ao_overlap_matrix == NULL) {
    set_error_message(error_message, error_message_capacity, "AO overlap output buffer is null");
    return 1;
  }

  n_basis_functions = runtime_handle->molecule->bas->msize;
  if ((size_t)n_basis_functions * (size_t)n_basis_functions != ao_overlap_matrix_size) {
    set_error_message(error_message, error_message_capacity, "AO overlap output buffer has the wrong size");
    return 1;
  }

  // The standalone diagnostic does not necessarily link the legacy HF runtime
  // that usually initializes the static Lebedev tables through
  // `init_xscf_world()`.  Initialize the grid tables explicitly here before
  // calling `init_grid()`.
  get_grids();
  legacy_grids = init_grid(MEDIUM_GRIDS, runtime_handle->molecule);
  if (legacy_grids == NULL) {
    set_error_message(error_message, error_message_capacity, "legacy init_grid returned null");
    return 1;
  }

  // `cal_overlap_numerical` returns the legacy row-major dense matrix.  Copy
  // it into the caller buffer in column-major order so C++ code can map it
  // directly as `Eigen::MatrixXd` without carrying another row-major path.
  legacy_overlap_matrix = cal_overlap_numerical(legacy_grids);
  if (legacy_overlap_matrix == NULL) {
    del_grid(legacy_grids);
    set_error_message(error_message, error_message_capacity, "legacy cal_overlap_numerical returned null");
    return 1;
  }

  for (int row = 0; row < n_basis_functions; ++row) {
    for (int column = 0; column < n_basis_functions; ++column) {
      ao_overlap_matrix[(size_t)column * (size_t)n_basis_functions + (size_t)row] =
          legacy_overlap_matrix[row][column];
    }
  }

  free_matrix(legacy_overlap_matrix);
  del_grid(legacy_grids);
  return 0;
}
