#pragma once

#include <stddef.h>

#include "runtime_c/local_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Builds the legacy medium-grid numerical AO overlap matrix.
 *
 * The legacy `xgrids.c` path evaluates AO values on the medium Becke grid and
 * contracts
 *
 *   `S_num(mu,nu) = sum_g w_g chi_mu(r_g) chi_nu(r_g)`.
 *
 * This helper exposes that reference matrix through the managed runtime API.
 * The output buffer must store `n_basis_functions * n_basis_functions`
 * elements in column-major order so the caller can wrap it directly with
 * `Eigen::Map<const Eigen::MatrixXd>`.
 */
int xmvb_cpp_runtime_compute_medium_grid_ao_overlap(
    const XmvbCppRuntimeHandle* runtime_handle,
    double* ao_overlap_matrix,
    size_t ao_overlap_matrix_size,
    char* error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif
