#include "vb/orbital/active_space_one_electron_builder.hpp"

#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

ActiveSpaceOneElectronResult build_active_space_one_electron_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_f11_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (ao_f11_matrix.rows() != n_basis_functions ||
      ao_f11_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
  }
  if (auxiliary_matrix.rows() != n_basis_functions ||
      auxiliary_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument("auxiliary orbital matrix shape mismatch");
  }

  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  // Project the AO effective Hamiltonian into the active orbital block
  // `HHO = T_active^T * F11 * T_active`.
  const Eigen::MatrixXd temp =
      active_auxiliary_orbitals.transpose() * ao_f11_matrix;
  const Eigen::MatrixXd hho_matrix =
      temp * active_auxiliary_orbitals;

  ActiveSpaceOneElectronResult result;
  result.h1e_act.assign(
      hho_matrix.data(),
      hho_matrix.data() + hho_matrix.size());
  return result;
}

}  // namespace

ActiveSpaceOneElectronResult ActiveSpaceOneElectronBuilder::build(
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space one-electron dimensions must be positive");
  }

  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (ao_effective_h1e.size() != ao_matrix_size) {
    throw std::invalid_argument("AO effective one-electron matrix size mismatch");
  }
  if (auxiliary_orbital_matrix.size() != ao_matrix_size) {
    throw std::invalid_argument("auxiliary orbital matrix size mismatch");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const Eigen::Map<const Eigen::MatrixXd> ao_f11_matrix(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return build_active_space_one_electron_impl(
      ao_f11_matrix,
      auxiliary_matrix,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

ActiveSpaceOneElectronResult ActiveSpaceOneElectronBuilder::build(
    const std::vector<double>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space one-electron dimensions must be positive");
  }
  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (ao_effective_h1e.size() != ao_matrix_size) {
    throw std::invalid_argument("AO effective one-electron matrix size mismatch");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const Eigen::Map<const Eigen::MatrixXd> ao_f11_matrix(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  return build_active_space_one_electron_impl(
      ao_f11_matrix,
      auxiliary_orbital_matrix,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

}  // namespace xmvb::vb
