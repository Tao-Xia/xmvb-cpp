#pragma once

#include <Eigen/Core>

#include <vector>

#include "integrals/active/matrix_result.hpp"
#include "integrals/active/one_electron_result.hpp"

namespace xmvb::integrals {

/**
 * @brief Pull back raw active-matrix cotangents using the shared F Cbar image.
 *
 * F and the AO overlap are symmetric. The overlap cotangent remains in raw C
 * coordinates; h and its accepted AO image share the fixed integral frame R.
 * No F C or F^T C is regenerated. RI consumers can omit the unused dense AO
 * adjoint; exact consumers receive C W^T C^T for their AO integral pullback.
 */
ActiveSpaceMatrixBackpropagationResult
pullback_active_matrices(const std::vector<double>& overlap_gradient,
                         const std::vector<double>& one_electron_gradient,
                         const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap,
                         const Eigen::Ref<const Eigen::MatrixXd>& raw_active_coefficients,
                         const Eigen::Ref<const Eigen::MatrixXd>& integral_coefficients,
                         const ActiveSpaceOneElectronResult& one_electron,
                         const Eigen::MatrixXd& frame,
                         bool ao_adjoint_required);

/**
 * @brief Reverse-mode backpropagator for `SSO = T^T S T` and `HHO = T^T F11 T`.
 *
 * The input adjoints are assumed to be gradients of the objective with respect
 * to the active-space overlap matrix `SSO` and active-space one-electron matrix
 * `HHO`. The backpropagator returns the induced gradients with respect to:
 * - the active auxiliary block `T_active`
 * - the AO effective one-electron matrix `F11`
 */
class ActiveSpaceMatrixBackpropagator {
  public:
    /**
   * @brief Backpropagates active-space matrix adjoints.
   *
   * @param active_orbital_overlap_gradient Column-major gradient with respect to `SSO`.
   * @param active_one_electron_gradient Column-major gradient with respect to `HHO`.
   * @param ao_overlap_matrix Column-major AO overlap matrix.
   * @param ao_effective_h1e Column-major AO `F11`.
   * @param auxiliary_orbital_matrix Column-major full auxiliary orbital matrix.
   * @param n_basis_functions Number of AO basis functions.
   * @param n_inactive_doubly_occupied_orbitals Number of inactive doubly occupied orbitals.
   * @param n_active_orbitals Number of active orbitals.
   * @return ActiveSpaceMatrixBackpropagationResult Reverse-mode derivatives.
   */
    ActiveSpaceMatrixBackpropagationResult
    backpropagate(const std::vector<double>& active_orbital_overlap_gradient,
                  const std::vector<double>& active_one_electron_gradient,
                  const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
                  const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
                  const std::vector<double>& auxiliary_orbital_matrix,
                  int n_basis_functions,
                  int n_inactive_doubly_occupied_orbitals,
                  int n_active_orbitals) const;

    ActiveSpaceMatrixBackpropagationResult
    backpropagate(const Eigen::Ref<const Eigen::MatrixXd>& active_orbital_overlap_gradient,
                  const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_gradient,
                  const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
                  const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
                  const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_orbital_matrix,
                  int n_basis_functions,
                  int n_inactive_doubly_occupied_orbitals,
                  int n_active_orbitals) const;

    ActiveSpaceMatrixBackpropagationResult
    backpropagate(const std::vector<double>& active_orbital_overlap_gradient,
                  const std::vector<double>& active_one_electron_gradient,
                  const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
                  const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
                  const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_orbital_matrix,
                  int n_basis_functions,
                  int n_inactive_doubly_occupied_orbitals,
                  int n_active_orbitals) const;
};

} // namespace xmvb::integrals
