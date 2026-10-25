#pragma once

#include <span>
#include "integrals/ao/input.hpp"
#include "integrals/active/two_electron_result.hpp"

namespace xmvb::integrals {

/** @brief Transform exact ERIs from borrowed, AO-row sparse orbital coefficients.
 * The CSR columns index active orbitals. Coefficients need not be orthogonal.
 * Outputs and optional accepted-point products are owned by the caller;
 * the AO graph and sparse input are never copied into worker-local storage.
 */
ActiveSpaceTwoElectronResult transform_exact_active_integrals(const AoIntegralInput& ao,
                                                              std::span<const int> row_offsets,
                                                              std::span<const int> orbital_indices,
                                                              std::span<const double> coefficients,
                                                              int n_active);

} // namespace xmvb::integrals
