#include "integrals/active/one_electron.hpp"

#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::integrals {

namespace {

ActiveSpaceOneElectronResult
build_active_space_one_electron_impl(const Eigen::Ref<const Eigen::MatrixXd>& ao_f11_matrix,
                                     const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_matrix,
                                     int n_basis_functions,
                                     int n_inactive_doubly_occupied_orbitals,
                                     int n_active_orbitals) {
    if (ao_f11_matrix.rows() != n_basis_functions || ao_f11_matrix.cols() != n_basis_functions) {
        throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
    }
    if (auxiliary_matrix.rows() != n_basis_functions ||
        auxiliary_matrix.cols() != n_basis_functions) {
        throw std::invalid_argument("auxiliary orbital matrix shape mismatch");
    }

    const auto active_auxiliary_orbitals =
        auxiliary_matrix.middleCols(n_inactive_doubly_occupied_orbitals, n_active_orbitals);
    // Project the AO effective Hamiltonian into the active orbital block
    // `HHO = T_active^T * F11 * T_active`.
    return ActiveSpaceOneElectronBuilder{}.build(ao_f11_matrix, active_auxiliary_orbitals);
}

} // namespace

ActiveSpaceOneElectronResult ActiveSpaceOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& active_coefficients) const {
    if (active_coefficients.rows() <= 0 || active_coefficients.cols() <= 0 ||
        ao_effective_h1e.rows() != active_coefficients.rows() ||
        ao_effective_h1e.cols() != active_coefficients.rows()) {
        throw std::invalid_argument("active one-electron projection size mismatch");
    }
    ActiveSpaceOneElectronResult result;
    result.ao_image.noalias() = ao_effective_h1e * active_coefficients;
    result.h1e_act.noalias() = active_coefficients.transpose() * result.ao_image;
    return result;
}

ActiveSpaceOneElectronResult
ActiveSpaceOneElectronBuilder::build(const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
                                     const std::vector<double>& auxiliary_orbital_matrix,
                                     int n_basis_functions,
                                     int n_inactive_doubly_occupied_orbitals,
                                     int n_active_orbitals) const {
    if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
        throw std::invalid_argument("active-space one-electron dimensions must be positive");
    }

    if (ao_effective_h1e.rows() != n_basis_functions ||
        ao_effective_h1e.cols() != n_basis_functions) {
        throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
    }
    const std::size_t n_basis = n_basis_functions;
    const std::size_t ao_matrix_size = n_basis * n_basis;
    if (auxiliary_orbital_matrix.size() != ao_matrix_size) {
        throw std::invalid_argument("auxiliary orbital matrix size mismatch");
    }
    if (n_inactive_doubly_occupied_orbitals < 0 ||
        n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
        throw std::invalid_argument("invalid active-orbital column range");
    }

    const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
        auxiliary_orbital_matrix.data(), n_basis_functions, n_basis_functions);
    return build_active_space_one_electron_impl(ao_effective_h1e,
                                                auxiliary_matrix,
                                                n_basis_functions,
                                                n_inactive_doubly_occupied_orbitals,
                                                n_active_orbitals);
}

ActiveSpaceOneElectronResult ActiveSpaceOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
    if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
        throw std::invalid_argument("active-space one-electron dimensions must be positive");
    }
    if (ao_effective_h1e.rows() != n_basis_functions ||
        ao_effective_h1e.cols() != n_basis_functions) {
        throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
    }
    if (n_inactive_doubly_occupied_orbitals < 0 ||
        n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
        throw std::invalid_argument("invalid active-orbital column range");
    }
    return build_active_space_one_electron_impl(ao_effective_h1e,
                                                auxiliary_orbital_matrix,
                                                n_basis_functions,
                                                n_inactive_doubly_occupied_orbitals,
                                                n_active_orbitals);
}

} // namespace xmvb::integrals
