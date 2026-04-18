#include "vb/orbital/active_space_matrix_backpropagator.hpp"

#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

ActiveSpaceMatrixBackpropagationResult backpropagate_active_space_matrix_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& active_orbital_overlap_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_gradient,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (auxiliary_matrix.rows() != n_basis_functions ||
      auxiliary_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument("auxiliary orbital matrix shape mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_f11(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);

  ActiveSpaceMatrixBackpropagationResult result;
  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  result.active_auxiliary_orbital_gradient =
      Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
  result.ao_effective_one_electron_gradient.assign(ao_matrix_size, 0.0);

  Eigen::Map<Eigen::MatrixXd> ao_f11_gradient(
      result.ao_effective_one_electron_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  Eigen::MatrixXd& auxiliary_active_gradient =
      result.active_auxiliary_orbital_gradient;

  // `SSO = T_active^T S T_active` contributes symmetrically in the active
  // space, while `HHO = T_active^T F11 T_active` produces adjoints for both
  // the auxiliary orbitals and the AO effective one-electron matrix.  The
  // implementation accumulates directly into the output buffers so the exact
  // HVP hot path avoids intermediate dense matrix/vector copies.
  auxiliary_active_gradient.noalias() =
      basis_overlap * active_auxiliary_orbitals * active_orbital_overlap_gradient;
  auxiliary_active_gradient.noalias() +=
      basis_overlap * active_auxiliary_orbitals *
      active_orbital_overlap_gradient.transpose();
  auxiliary_active_gradient.noalias() +=
      ao_f11 * active_auxiliary_orbitals * active_one_electron_gradient;
  auxiliary_active_gradient.noalias() +=
      ao_f11.transpose() * active_auxiliary_orbitals *
      active_one_electron_gradient.transpose();
  ao_f11_gradient.noalias() =
      active_auxiliary_orbitals * active_one_electron_gradient.transpose() *
      active_auxiliary_orbitals.transpose();
  return result;
}

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
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  const std::size_t active_matrix_size =
      xmvb::to_size(n_active_orbitals) * n_active_orbitals;
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

  const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> active_orbital_overlap_gradient_matrix(
      active_orbital_overlap_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::Map<const Eigen::MatrixXd> active_one_electron_gradient_matrix(
      active_one_electron_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  return backpropagate_active_space_matrix_impl(
      active_orbital_overlap_gradient_matrix,
      active_one_electron_gradient_matrix,
      active_orbital_overlap_matrix,
      ao_effective_h1e,
      auxiliary_matrix,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

ActiveSpaceMatrixBackpropagationResult ActiveSpaceMatrixBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& active_orbital_overlap_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_gradient,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space backprop dimensions must be positive");
  }

  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (active_orbital_overlap_matrix.size() != ao_matrix_size ||
      ao_effective_h1e.size() != ao_matrix_size) {
    throw std::invalid_argument("AO-sized matrix input mismatch");
  }
  if (active_orbital_overlap_gradient.rows() != n_active_orbitals ||
      active_orbital_overlap_gradient.cols() != n_active_orbitals ||
      active_one_electron_gradient.rows() != n_active_orbitals ||
      active_one_electron_gradient.cols() != n_active_orbitals) {
    throw std::invalid_argument("active-space gradient size mismatch");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_f11(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);

  ActiveSpaceMatrixBackpropagationResult result;
  result.active_auxiliary_orbital_gradient =
      Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
  result.ao_effective_one_electron_gradient.assign(ao_matrix_size, 0.0);
  Eigen::Map<Eigen::MatrixXd> ao_f11_gradient(
      result.ao_effective_one_electron_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  Eigen::MatrixXd& auxiliary_active_gradient =
      result.active_auxiliary_orbital_gradient;

  // The matrix overload receives the raw determinant-level active-space
  // gradients.  Symmetry is imposed analytically here so callers do not need
  // to allocate `0.5 * (G + G^T)` temporaries before entering the exact HVP
  // pullback.
  auxiliary_active_gradient.noalias() =
      basis_overlap * active_auxiliary_orbitals * active_orbital_overlap_gradient;
  auxiliary_active_gradient.noalias() +=
      basis_overlap * active_auxiliary_orbitals *
      active_orbital_overlap_gradient.transpose();
  auxiliary_active_gradient *= 0.5;
  auxiliary_active_gradient.noalias() +=
      0.5 * ao_f11 * active_auxiliary_orbitals * active_one_electron_gradient;
  auxiliary_active_gradient.noalias() +=
      0.5 * ao_f11 * active_auxiliary_orbitals *
      active_one_electron_gradient.transpose();
  auxiliary_active_gradient.noalias() +=
      0.5 * ao_f11.transpose() * active_auxiliary_orbitals *
      active_one_electron_gradient;
  auxiliary_active_gradient.noalias() +=
      0.5 * ao_f11.transpose() * active_auxiliary_orbitals *
      active_one_electron_gradient.transpose();
  ao_f11_gradient.noalias() =
      0.5 * active_auxiliary_orbitals *
      active_one_electron_gradient.transpose() *
      active_auxiliary_orbitals.transpose();
  ao_f11_gradient.noalias() +=
      0.5 * active_auxiliary_orbitals *
      active_one_electron_gradient *
      active_auxiliary_orbitals.transpose();
  return result;
}

ActiveSpaceMatrixBackpropagationResult ActiveSpaceMatrixBackpropagator::backpropagate(
    const std::vector<double>& active_orbital_overlap_gradient,
    const std::vector<double>& active_one_electron_gradient,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space backprop dimensions must be positive");
  }

  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  const std::size_t active_matrix_size =
      xmvb::to_size(n_active_orbitals) * n_active_orbitals;
  if (active_orbital_overlap_matrix.size() != ao_matrix_size ||
      ao_effective_h1e.size() != ao_matrix_size) {
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

  const Eigen::Map<const Eigen::MatrixXd> active_orbital_overlap_gradient_matrix(
      active_orbital_overlap_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::Map<const Eigen::MatrixXd> active_one_electron_gradient_matrix(
      active_one_electron_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  return backpropagate_active_space_matrix_impl(
      active_orbital_overlap_gradient_matrix,
      active_one_electron_gradient_matrix,
      active_orbital_overlap_matrix,
      ao_effective_h1e,
      auxiliary_orbital_matrix,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

}  // namespace xmvb::vb
