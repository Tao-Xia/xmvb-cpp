#pragma once

#include <Eigen/Core>

#include <vector>

#include "runtime/libcint_ri_integral_provider.hpp"

namespace xmvb::vb {

struct AoEffectiveOneElectronRiOperatorOptions {
  /**
   * @brief Tries a spectral factorization `X = U S U^T` before the RI sweep.
   *
   * When the symmetric input matrix is numerically low-rank, the exchange
   * contribution can be evaluated from `L_A U` with `O(n_ao^2 r)` work per
   * auxiliary instead of `O(n_ao^3)`.
   */
  bool attempt_spectral_factorization = false;

  /**
   * @brief Absolute cutoff for keeping spectral components in `X`.
   */
  double spectral_eigenvalue_cutoff = 1.0e-10;

  /**
   * @brief Enables the low-rank path only when `rank(X)` stays below this fraction.
   */
  double low_rank_fraction_cutoff = 0.75;
};

/**
 * @brief Signed low-rank factorization of a symmetric AO matrix.
 *
 * The columns are stored in column-major order with dimensions
 * `n_rows x (n_positive_components + n_negative_components)`.  Positive
 * components must come first, followed by negative components, and each column
 * is already scaled by `sqrt(|lambda|)`.  The represented symmetric matrix is
 *
 * `X = U_{+} U_{+}^T - U_{-} U_{-}^T`.
 */
struct AoEffectiveOneElectronRiLowRankFactors {
  Eigen::MatrixXd scaled_factor_matrix;
  int n_positive_components = 0;
  int n_negative_components = 0;
};

/**
 * @brief Applies the AO-side RI Coulomb-exchange operator to an AO matrix.
 *
 * The RI cache stores metric-whitened AO-pair factors `L_{A,mu,nu}` on packed
 * symmetric AO pairs. This helper reconstructs each auxiliary row as a
 * symmetric AO matrix and evaluates the linear map
 *
 * `G[X] = 2 * sum_A <L_A, X> * L_A - sum_A L_A * X * L_A`
 *
 * using the repository-wide column-major AO matrix convention. The forward
 * inactive-density path and the reverse-mode adjoint path both reuse this same
 * self-adjoint operator, so the RI implementation no longer needs to
 * materialize the much larger AO-pair Gram matrix `L^T L`.
 */
std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const std::vector<double>& input_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions,
    const AoEffectiveOneElectronRiOperatorOptions& options = {});

/**
 * @brief Applies the AO-side RI Coulomb-exchange operator to explicit factors.
 *
 * This overload skips the AO-level spectral factorization step and feeds the
 * supplied signed low-rank factors directly into the same low-rank RI sweep
 * used after the matrix-based path detects a low-rank input.
 */
std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const AoEffectiveOneElectronRiLowRankFactors& low_rank_factors,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions);

}  // namespace xmvb::vb
