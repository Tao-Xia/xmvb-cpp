#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/integrals/ao/ri/factorization.hpp"

namespace xmvb::vb {

struct AoEffectiveOneElectronRiOperatorOptions {
  /**
   * @brief Tries a spectral factorization `X = U S U^T` before the RI sweep.
   *
   * When the symmetric input matrix is numerically low-rank, the exchange
   * contribution can be evaluated from `L_A U` with `O(n_bf^2 r)` work per
   * auxiliary instead of `O(n_bf^3)`.
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
 * @brief Caller-owned buffers for repeated fused RI AO-H1E actions.
 *
 * The storage is resized on demand and retained between calls.  A workspace
 * must not be used concurrently by more than one call.
 */
struct AoEffectiveOneElectronRiFusedWorkspace {
  std::vector<Eigen::MatrixXd> partial_forward;
  std::vector<Eigen::MatrixXd> partial_adjoint;
  std::vector<Eigen::MatrixXd> factor_matrices;
  std::vector<Eigen::MatrixXd> left_products;
  std::vector<Eigen::MatrixXd> exchange_products;
  std::vector<double> weighted_packed_forward;
  std::vector<double> weighted_packed_adjoint;
};

/** Runtime-selected implementation for repeated RI AO-H1E Hessian actions. */
enum class AoEffectiveOneElectronRiStrategy {
  Uncalibrated,
  DenseFused,
  Spectral
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
    const RiAoFactorization& ri_factorization,
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
    const RiAoFactorization& ri_factorization,
    int n_basis_functions);

/**
 * @brief Applies the RI AO-H1E operator to a source and an adjoint together.
 *
 * Both inputs are symmetric AO matrices.  The metric-whitened RI factors are
 * swept once, producing the complete symmetric matrices `G_RI[source]` and
 * `G_RI[adjoint]`.  Since `G_RI` is a constant linear self-adjoint map, the
 * second result is also its transpose action.  Consequently the derivative
 * of that transpose action at a fixed adjoint is identically zero; Hessian
 * code only needs to pass the directional adjoint to this routine.
 *
 * The outputs may alias neither the inputs nor each other.  Caller-owned
 * workspace retains all thread-local AO-sized buffers across repeated HVPs.
 */
void apply_ao_effective_one_electron_ri_operator_fused(
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
    const RiAoFactorization& ri_factorization,
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose);

/**
 * @brief Applies and calibrates the repeated RI AO-H1E Hessian action.
 *
 * On the first call, dense-fused and signed-spectral contractions are timed on
 * the same inputs and checked for numerical agreement. The faster strategy is
 * retained in `strategy` for subsequent calls at the accepted orbital point.
 */
void apply_ao_effective_one_electron_ri_operator_adaptive(
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
    const RiAoFactorization& ri_factorization,
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    AoEffectiveOneElectronRiStrategy* strategy,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose);

}  // namespace xmvb::vb
