#pragma once

#include <vector>

#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb {

/**
 * @brief Validates the raw libcint atom/basis/environment tables.
 *
 * The standalone runtime snapshots `atm`, `bas`, `basidx`, and `env` directly
 * from the mixed C/C++ initialization layer. Downstream C++ kernels rely on
 * these buffers having the exact libcint slot counts and at least one atom and
 * one shell.
 */
void validate_libcint_input_shape(const LibcintInput& input);

/**
 * @brief Returns the global AO start offset of one shell.
 *
 * `basidx` stores `(ao_offset, ao_count)` pairs in the historical runtime
 * layout, so this helper centralizes the indexing convention instead of
 * open-coding `2 * shell_index` throughout the codebase.
 */
int shell_ao_offset(const LibcintInput& input, int shell_index);

/**
 * @brief Returns the number of Cartesian AOs carried by one shell.
 */
int shell_ao_count(const LibcintInput& input, int shell_index);

/**
 * @brief Infers the total number of basis functions from the final shell.
 */
int infer_n_basis_functions(const LibcintInput& input);

/**
 * @brief Builds per-AO normalization factors from raw Cartesian overlaps.
 *
 * The libcint shell evaluators and integral kernels operate in the raw
 * Cartesian AO convention.  The rest of the modern C++ codebase uses the
 * normalized AO convention stored in `active_orbital_overlap_matrix`, so
 * callers multiply raw AO values or raw shell blocks by this vector at the
 * boundary.
 */
std::vector<double> build_cartesian_ao_normalization(const LibcintInput& input);

}  // namespace xmvb::vb
