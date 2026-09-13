#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

struct AoIntegralInput;

using ExactCtxDenseMatrix = Eigen::MatrixXd;
using ExactCtxPairMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Accepted-point cache for exact fixed-adjoint active-space 2e HVP work.
 *
 * The exact matrix-free second-order path repeatedly applies the same accepted
 * packed 2e adjoint to different active-orbital tangents. Only accepted-point
 * state is retained here; pair coefficients and pair gradients are derived
 * and must not become persistent `N_pair x A_pair` buffers. The accepted pair
 * products and active coefficients are borrowed from the accepted active-space
 * result, whose lifetime must enclose every use of this cache.
 */
struct ExactPackedActiveTwoElectronAdjointCache {
  int n_basis_functions = 0;
  int n_active_orbitals = 0;
  std::vector<int> ao_pair_first_indices;
  std::vector<int> ao_pair_second_indices;
  std::vector<int> active_pair_first_indices;
  std::vector<int> active_pair_second_indices;
  const AoIntegralInput* ao_integral_input = nullptr;
  const ExactCtxPairMatrix* accepted_pair_products = nullptr;
  const ExactCtxDenseMatrix* accepted_active_coefficients = nullptr;
  ExactCtxPairMatrix active_pair_gradient_matrix;
};

/**
 * @brief Reusable work buffers for accepted-point exact 2e HVP applications.
 *
 * These buffers only depend on AO-pair and active-pair dimensions at one
 * accepted point. Reusing them across repeated matrix-free `H v` calls avoids
 * reallocation of the largest exact-2e intermediates on every Krylov matvec.
 */
struct ExactPackedActiveTwoElectronApplyWorkspace {
  ExactCtxDenseMatrix dense_active_direction;
  ExactCtxDenseMatrix dense_active_gradient_direction;
  ExactCtxPairMatrix mixed_pair_coefficients;
  ExactCtxPairMatrix transformed_pair_coefficients;
  ExactCtxPairMatrix pair_gradients;
  ExactCtxPairMatrix pair_products_tile;
  ExactCtxPairMatrix directional_pair_products_tile;
  ExactCtxPairMatrix pair_gradients_tile;
};

/**
 * @brief Reusable buffers for exact packed `\delta GGO` directional builds.
 *
 * The outer-response exact_ctx path repeatedly forms `\delta GGO` at one
 * accepted point.  Reusing the AO-pair and active-pair work buffers here
 * avoids several large allocations on every `H v` application.
 */
struct ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace {
  ExactCtxDenseMatrix dense_active_direction;
  ExactCtxPairMatrix directional_pair_coefficients;
  ExactCtxPairMatrix directional_pair_products;
  ExactCtxDenseMatrix active_pair_contraction;
  ExactCtxPairMatrix accepted_pair_products_tile;
  ExactCtxPairMatrix pair_gradients_tile;
  ExactCtxDenseMatrix dense_fixed_adjoint_direction;
};

}  // namespace xmvb::vb
