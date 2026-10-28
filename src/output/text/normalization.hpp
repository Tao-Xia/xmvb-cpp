#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

/** @brief Exact Coulson-Chirgwin weights from raw C and its cached M*C.
 * No structure normalization, overlap matrix or extra physical action is needed.
 */
Eigen::VectorXd coulson_chirgwin_weights(const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                         const Eigen::Ref<const Eigen::VectorXd>& overlap_product);

/** @brief Exact norms shared by all states in one structure report. */
struct StructureNorms {
    Eigen::VectorXd values;
    double mean_overlap = 0.0;
};

/** @brief Prepares norms only for normalized coefficients/renormalized weights.
 * Reuses exact result diagonals or already available matrices. Otherwise uses
 * the accepted active overlap and the existing overlap-only contraction;
 * never builds Hamiltonian data, a complete overlap matrix or a CI solution.
 */
StructureNorms structure_norms(const vb::VbScfOptimizerResult& result);

} // namespace xmvb::output
