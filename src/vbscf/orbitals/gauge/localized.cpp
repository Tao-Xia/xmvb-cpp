#include "vbscf/orbitals/gauge/localized.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace xmvb::vb {

namespace {


double max_abs_entry(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  double max_value = 0.0;
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      max_value = std::max(max_value, std::abs(matrix(row, column)));
    }
  }
  return max_value;
}

}  // namespace

LocalizedRepresentativeSelector build_localized_representative_selector(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals, // C_i
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orthonormal_orbitals, // Q_i
    const Eigen::Ref<const Eigen::MatrixXd>& active_physical_orbitals,  // C_a
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_orbitals, // T_a
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix) {    // S

  LocalizedRepresentativeSelector selector;
  const Eigen::Index n_inactive = inactive_physical_orbitals.cols();
  const Eigen::Index n_active = active_physical_orbitals.cols();
  if (n_inactive == 0) {
    selector.inactive_right_transform = Eigen::MatrixXd::Zero(0, 0);
    selector.inactive_inverse_transpose_right_transform = Eigen::MatrixXd::Zero(0, 0);
    selector.active_inactive_coefficients = Eigen::MatrixXd::Zero(0, n_active);
    return selector;
  }

  // The localized physical representative satisfies C_i = Q_i U_i and
  // C_a - T_a = Q_i K_a. Recover these coordinates through the finite-
  // precision metric Gram matrix instead of assuming Q_i^T S Q_i is exactly
  // the identity. The latter assumption loses several digits for strongly
  // nonorthogonal localized inactive orbitals even though their span is valid.
  const Eigen::MatrixXd basis_overlap_times_inactive =
      basis_overlap_matrix * inactive_physical_orbitals;
  const Eigen::MatrixXd inactive_metric =
      inactive_orthonormal_orbitals.transpose() * basis_overlap_matrix *
      inactive_orthonormal_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> inactive_metric_ldlt(inactive_metric);
  if (inactive_metric_ldlt.info() != Eigen::Success ||
      (inactive_metric_ldlt.vectorD().array() <= 0.0).any()) {
    throw std::runtime_error(
        "localized representative selector inactive metric is not positive definite");
  }

  selector.inactive_right_transform = inactive_metric_ldlt.solve(
      inactive_orthonormal_orbitals.transpose() *
      basis_overlap_times_inactive);
  selector.active_inactive_coefficients = inactive_metric_ldlt.solve(
      inactive_orthonormal_orbitals.transpose() * basis_overlap_matrix *
      (active_physical_orbitals - active_auxiliary_orbitals));
  if (inactive_metric_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "localized representative selector coordinate solve failed");
  }

  Eigen::FullPivLU<Eigen::MatrixXd> transform_lu(selector.inactive_right_transform);
  if (!transform_lu.isInvertible()) {
    throw std::runtime_error(
        "localized representative selector inactive transform is singular");
  }
  selector.inactive_inverse_transpose_right_transform =
      transform_lu.inverse().transpose();

  constexpr double kSelectorReconstructionTolerance = 1.0e-10;
  const Eigen::MatrixXd reconstructed_inactive =
      inactive_orthonormal_orbitals * selector.inactive_right_transform;
  const Eigen::MatrixXd reconstructed_active =
      active_auxiliary_orbitals +
      inactive_orthonormal_orbitals * selector.active_inactive_coefficients;

  const double inactive_error =
      max_abs_entry(reconstructed_inactive - inactive_physical_orbitals);
  const double active_error =
      max_abs_entry(reconstructed_active - active_physical_orbitals);
  if (inactive_error > kSelectorReconstructionTolerance ||
      active_error > kSelectorReconstructionTolerance) {
    std::ostringstream message;
    message << "localized representative selector reconstruction is inconsistent "
               "with the accepted-point orbital blocks: inactive max error="
            << inactive_error << ", active max error=" << active_error
            << ", inactive transform reciprocal condition="
            << transform_lu.rcond()
            << ", inactive orthonormality max error="
            << max_abs_entry(
                   inactive_orthonormal_orbitals.transpose() *
                       basis_overlap_matrix * inactive_orthonormal_orbitals -
                   Eigen::MatrixXd::Identity(n_inactive, n_inactive));
    throw std::runtime_error(
        message.str());
  }

  return selector;
}

}  // namespace xmvb::vb
