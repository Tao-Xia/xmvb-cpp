#include "vb/orbital/active_space_matrix_backpropagator.hpp"

#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

}  // namespace

ActiveSpaceMatrixBackpropagationResult ActiveSpaceMatrixBackpropagator::backpropagate(
    const std::vector<double>& active_orbital_overlap_gradient,
    const std::vector<double>& active_one_electron_gradient,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space backprop dimensions must be positive");
  }

  const std::size_t ao_matrix_size =
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
  const std::size_t active_matrix_size =
      static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals;
  if (active_orbital_overlap_matrix.size() != ao_matrix_size ||
      ao_effective_h1e.size() != ao_matrix_size ||
      auxiliary_orbital_matrix.size() != ao_matrix_size) {
    throw std::invalid_argument("AO-sized matrix input mismatch");
  }
  if (active_orbital_overlap_gradient.size() != active_matrix_size ||
      active_one_electron_gradient.size() != active_matrix_size) {
    throw std::invalid_argument("active-space gradient size mismatch");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const Eigen::Map<const Matrix> basis_overlap(
      active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> ao_f11(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> sso_gradient(
      active_orbital_overlap_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::Map<const Matrix> hho_gradient(
      active_one_electron_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);

  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);

  // SSO = T^T S T
  const Matrix sso_gradient_symmetric =
      sso_gradient + sso_gradient.transpose();
  const Matrix sso_auxiliary_gradient =
      basis_overlap * active_auxiliary_orbitals * sso_gradient_symmetric;

  // HHO = T^T F T
  const Matrix hho_auxiliary_gradient =
      ao_f11 * active_auxiliary_orbitals * hho_gradient +
      ao_f11.transpose() * active_auxiliary_orbitals * hho_gradient.transpose();
  const Matrix ao_f11_gradient =
      active_auxiliary_orbitals * hho_gradient.transpose() *
      active_auxiliary_orbitals.transpose();

  Matrix auxiliary_gradient =
      Matrix::Zero(n_basis_functions, n_basis_functions);
  auxiliary_gradient.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals) = sso_auxiliary_gradient + hho_auxiliary_gradient;

  ActiveSpaceMatrixBackpropagationResult result;
  result.auxiliary_orbital_gradient.assign(
      auxiliary_gradient.data(),
      auxiliary_gradient.data() + auxiliary_gradient.size());
  result.ao_effective_one_electron_gradient.assign(
      ao_f11_gradient.data(),
      ao_f11_gradient.data() + ao_f11_gradient.size());
  return result;
}

}  // namespace xmvb::vb
