#include "vb/biorthogonal_vbscf/biorthogonal_selected_structure_space.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

}  // namespace

BiorthogonalSelectedStructureSpace build_biorthogonal_selected_structure_space(
    const Eigen::Ref<const Eigen::MatrixXd>& structure_to_determinant) {
  if (structure_to_determinant.rows() <= 0 || structure_to_determinant.cols() <= 0) {
    throw std::invalid_argument("structure_to_determinant must be non-empty");
  }
  throw_if_nonfinite(structure_to_determinant, "structure_to_determinant");

  BiorthogonalSelectedStructureSpace structure_space;
  structure_space.structure_to_determinant = structure_to_determinant;
  structure_space.fixed_metric =
      structure_to_determinant.transpose() * structure_to_determinant;
  throw_if_nonfinite(structure_space.fixed_metric, "fixed_metric");

  // The first biorthogonal prototype keeps the selected structure list frozen,
  // so the structure metric can be factored once and reused in every later
  // projected solve. This factorization is therefore a persistent module-level
  // asset, not a disposable local scratch result.
  Eigen::LLT<Eigen::MatrixXd> metric_factor(structure_space.fixed_metric);
  if (metric_factor.info() != Eigen::Success) {
    throw std::runtime_error(
        "fixed selected-structure metric is not positive definite");
  }

  structure_space.fixed_metric_cholesky_factor = metric_factor.matrixL();
  throw_if_nonfinite(
      structure_space.fixed_metric_cholesky_factor,
      "fixed_metric_cholesky_factor");

  const Eigen::VectorXd factor_diagonal =
      structure_space.fixed_metric_cholesky_factor.diagonal();
  if (factor_diagonal.size() != structure_space.fixed_metric_cholesky_factor.rows()) {
    throw std::runtime_error("fixed metric factor diagonal size mismatch");
  }
  structure_space.fixed_metric_min_diagonal = factor_diagonal.minCoeff();
  if (!(structure_space.fixed_metric_min_diagonal > 0.0)) {
    throw std::runtime_error("fixed metric factor has non-positive diagonal");
  }

  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
      structure_space.fixed_metric.rows(),
      structure_space.fixed_metric.cols());
  structure_space.fixed_metric_factor_inverse =
      structure_space.fixed_metric_cholesky_factor
          .template triangularView<Eigen::Lower>()
          .solve(identity);
  throw_if_nonfinite(
      structure_space.fixed_metric_factor_inverse,
      "fixed_metric_factor_inverse");

  const Eigen::MatrixXd reconstruction =
      structure_space.fixed_metric_cholesky_factor *
      structure_space.fixed_metric_cholesky_factor.transpose();
  structure_space.fixed_metric_factorization_residual_frobenius_norm =
      (reconstruction - structure_space.fixed_metric).norm();
  if (!std::isfinite(
          structure_space.fixed_metric_factorization_residual_frobenius_norm)) {
    throw std::runtime_error("fixed metric factorization residual is not finite");
  }

  return structure_space;
}

void validate_biorthogonal_selected_structure_space(
    const BiorthogonalSelectedStructureSpace& structure_space,
    double min_diagonal_tolerance) {
  if (min_diagonal_tolerance < 0.0) {
    throw std::invalid_argument("min_diagonal_tolerance must be non-negative");
  }
  if (structure_space.structure_to_determinant.rows() <= 0 ||
      structure_space.structure_to_determinant.cols() <= 0) {
    throw std::invalid_argument("structure_to_determinant must be non-empty");
  }
  const int n_structures = structure_space.structure_to_determinant.cols();
  if (structure_space.fixed_metric.rows() != n_structures ||
      structure_space.fixed_metric.cols() != n_structures) {
    throw std::invalid_argument("fixed_metric dimensions are inconsistent");
  }
  if (structure_space.fixed_metric_cholesky_factor.rows() != n_structures ||
      structure_space.fixed_metric_cholesky_factor.cols() != n_structures) {
    throw std::invalid_argument(
        "fixed_metric_cholesky_factor dimensions are inconsistent");
  }
  if (structure_space.fixed_metric_factor_inverse.rows() != n_structures ||
      structure_space.fixed_metric_factor_inverse.cols() != n_structures) {
    throw std::invalid_argument(
        "fixed_metric_factor_inverse dimensions are inconsistent");
  }
  throw_if_nonfinite(structure_space.structure_to_determinant, "structure_to_determinant");
  throw_if_nonfinite(structure_space.fixed_metric, "fixed_metric");
  throw_if_nonfinite(
      structure_space.fixed_metric_cholesky_factor,
      "fixed_metric_cholesky_factor");
  throw_if_nonfinite(
      structure_space.fixed_metric_factor_inverse,
      "fixed_metric_factor_inverse");
  if (!std::isfinite(
          structure_space.fixed_metric_factorization_residual_frobenius_norm)) {
    throw std::invalid_argument("fixed metric factorization residual must be finite");
  }
  if (!(structure_space.fixed_metric_min_diagonal > min_diagonal_tolerance)) {
    throw std::runtime_error("fixed selected-structure metric is too ill-conditioned");
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
