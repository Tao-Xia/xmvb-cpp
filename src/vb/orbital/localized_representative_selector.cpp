#include "vb/orbital/localized_representative_selector.hpp"

#include <cmath>
#include <stdexcept>
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

  // In the accepted-point orthonormal frame the localized physical occupied
  // representative satisfies `C_i = Q_i U_i` and `C_a = T_a + Q_i K_a`.  The
  // AO-metric projections below recover those small matrices directly from the
  // cached dense orbital blocks without re-entering the older mixed-gauge
  // derivation.
  const Eigen::MatrixXd basis_overlap_times_inactive =
      basis_overlap_matrix * inactive_physical_orbitals;
  const Eigen::MatrixXd basis_overlap_times_active =
      basis_overlap_matrix * active_physical_orbitals;

  selector.inactive_right_transform =
      inactive_orthonormal_orbitals.transpose() * basis_overlap_times_inactive;
  selector.active_inactive_coefficients =
      inactive_orthonormal_orbitals.transpose() * basis_overlap_times_active;

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
    throw std::runtime_error(
        "localized representative selector reconstruction is inconsistent with the "
        "accepted-point orbital blocks");
  }

  return selector;
}

}  // namespace xmvb::vb
