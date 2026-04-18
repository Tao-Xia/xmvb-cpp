#pragma once

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Evaluates the spin-resolved overlap derivative with respect to the
 * spatial overlap metric.
 */
ScalarBuffer evaluate_spin_resolved_overlap_metric_gradient(
    const ConstMatrixRef& left_pairing_matrix,
    const ConstMatrixRef& right_pairing_matrix,
    const ConstMatrixRef& spatial_overlap_matrix,
    int n_alpha_electrons,
    int n_beta_electrons,
    int n_pairs);

}  // namespace xmvb::pfaffian_vbscf
