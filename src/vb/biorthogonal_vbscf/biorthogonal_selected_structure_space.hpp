#pragma once

#include <Eigen/Core>

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Fixed selected-structure metric data for the first prototype.
 *
 * `structure_to_determinant` has dimensions `(n_determinants, n_structures)`.
 * The fixed metric
 *
 * `M^(0) = T^T T`
 *
 * is stored together with one Cholesky factor `L` and its inverse. These
 * matrices are reusable across all macro iterations as long as the selected
 * structure list is frozen.
 */
struct BiorthogonalSelectedStructureSpace {
  Eigen::MatrixXd structure_to_determinant;
  Eigen::MatrixXd fixed_metric;
  Eigen::MatrixXd fixed_metric_cholesky_factor;
  Eigen::MatrixXd fixed_metric_factor_inverse;
  double fixed_metric_factorization_residual_frobenius_norm = 0.0;
  double fixed_metric_min_diagonal = 0.0;
};

/**
 * @brief Builds the fixed selected-structure metric for one frozen structure set.
 *
 * This is the algebraic realization of the first conservative biorthogonal
 * prototype with identical bra/ket structure coefficients, which yields the
 * orbitally fixed metric `M^(0) = T^T T`.
 */
BiorthogonalSelectedStructureSpace build_biorthogonal_selected_structure_space(
    const Eigen::Ref<const Eigen::MatrixXd>& structure_to_determinant);

/**
 * @brief Validates the fixed metric factorization and a minimum diagonal scale.
 *
 * `min_diagonal_tolerance` is compared against the smallest diagonal entry of
 * the stored Cholesky factor. This is a cheap conditioning surrogate for the
 * selected-structure metric.
 */
void validate_biorthogonal_selected_structure_space(
    const BiorthogonalSelectedStructureSpace& structure_space,
    double min_diagonal_tolerance);

}  // namespace xmvb::vb::biorthogonal_vbscf
