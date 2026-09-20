#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/result.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"

namespace xmvb::vb {

/**
 * @brief Borrowed accepted-point data for factor-native RI 2e responses.
 *
 * The cache retains only the molecule-static whitened AO-pair factors, the
 * accepted AO-by-active coefficient matrix, and the accepted active-pair
 * factors `B = L T(C)`. The factorization and active-space result supplied to
 * the builder must outlive every use of this cache.
 */
struct RiActiveTwoElectronResponseCache {
  int n_basis_functions = 0;
  int n_active_orbitals = 0;
  int n_auxiliary_functions = 0;
  std::vector<int> ao_pair_first_indices;
  std::vector<int> ao_pair_second_indices;
  std::vector<int> active_pair_first_indices;
  std::vector<int> active_pair_second_indices;
  const Eigen::MatrixXd* metric_whitened_ao_pair_factors = nullptr;
  const Eigen::MatrixXd* accepted_active_coefficients = nullptr;
  const Eigen::MatrixXd* accepted_active_pair_factors = nullptr;
};

/**
 * @brief Builds and validates a borrowed factor-native RI response cache.
 *
 * @param ao_ri_factorization Molecule-static whitened AO-pair factors.
 * @param accepted_active_space_result Accepted RI active-space result carrying
 *        both `C` and `B`.
 * @param n_active_orbitals Number of active orbitals in `C`.
 */
RiActiveTwoElectronResponseCache build_ri_active_two_electron_response_cache(
    const RiAoFactorization& ao_ri_factorization,
    const ActiveSpaceTwoElectronResult& accepted_active_space_result,
    int n_active_orbitals);

/**
 * @brief Forms the active-pair factor direction `delta B = L delta T(C)[D]`.
 *
 * The returned matrix has auxiliary-function rows and packed active-pair
 * columns. No AO four-index integral tensor is formed or consulted.
 */
Eigen::MatrixXd compute_ri_active_pair_factor_directional_derivative(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction);

/**
 * @brief Forms several active-pair factor directions in one RI transform.
 *
 * The packed-map path evaluates `L [delta T_1 ... delta T_b]` as one wide
 * matrix product. The direct-transform path shares the accepted `L_A C`
 * transform across all directions.
 */
std::vector<Eigen::MatrixXd>
compute_ri_active_pair_factor_directional_derivative_batch(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions);

/**
 * @brief Forms packed `delta G = B^T delta B + delta B^T B`.
 *
 * @param accepted_cache Accepted-point cache containing `B`.
 * @param directional_active_pair_factors Factor direction `delta B` with the
 *        same shape as accepted `B`.
 */
std::vector<double>
compute_ri_packed_active_two_electron_integral_directional_derivative(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>&
        directional_active_pair_factors);

/**
 * @brief Forms packed `delta G` columns for a block of factor directions.
 */
Eigen::MatrixXd
compute_ri_packed_active_two_electron_integral_directional_derivative_batch(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& directional_active_pair_factors);

/**
 * @brief Pulls an arbitrary packed active-2e adjoint back to active orbitals.
 *
 * This general accepted-point pullback is suitable both for the ordinary
 * active-space adjoint and for an outer-response `delta` adjoint. It acts
 * directly through `B = L T(C)` and never reconstructs AO four-index ERIs.
 */
Eigen::MatrixXd backpropagate_ri_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache);

/**
 * @brief Applies the fixed-packed-adjoint RI orbital Hessian to one direction.
 *
 * The result includes both the factor-response term driven by `delta B W`
 * and the derivative of the orbital pair-map pullback. A precomputed
 * `directional_active_pair_factors` may be supplied to share the factor action
 * with the packed-integral directional derivative.
 */
Eigen::MatrixXd
apply_ri_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const Eigen::MatrixXd* directional_active_pair_factors = nullptr);

}  // namespace xmvb::vb
