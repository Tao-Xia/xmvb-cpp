#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/response/types.hpp"
#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb::detail {

/** @brief Applies the symmetric sparse AO-pair ERI operator to a vector block. */
void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao_integral_input,
    const Eigen::Ref<const ExactCtxPairMatrix>& pair_coefficients,
    int n_bf,
    std::size_t n_active_pairs,
    ExactCtxPairMatrix* pair_products);

/**
 * @brief Applies the AO-pair ERI operator to generated accepted and mixed rows.
 *
 * Only target rows `[row_begin, row_begin + row_count)` are materialized.
 * Source pair coefficients are evaluated directly from the accepted orbitals,
 * so this operation does not require a resident AO-pair coefficient matrix.
 */
void apply_generated_pair_rows(
    const AoIntegralInput& ao_integral_input,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::MatrixXd* dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::Index row_begin,
    Eigen::Index row_count,
    ExactCtxPairMatrix* pair_products,
    ExactCtxPairMatrix* directional_pair_products);

}  // namespace xmvb::vb::detail
