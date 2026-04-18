#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

enum class ActiveSpaceTwoElectronRepresentation {
  PackedExact,
  ResolutionOfIdentity,
};

/**
 * @brief Active-space two-electron integrals derived from AO ERIs.
 */
struct ActiveSpaceTwoElectronResult {
  /**
   * @brief Active-space two-electron representation carried by this result.
   *
   * `PackedExact` stores the legacy materialized `GGO` tensor. The production
   * RI path now carries `ResolutionOfIdentity` directly so downstream
   * determinant kernels can consume `L_{A,P}` without reconstructing packed
   * active ERIs first.
   */
  ActiveSpaceTwoElectronRepresentation representation =
      ActiveSpaceTwoElectronRepresentation::PackedExact;

  /**
   * @brief Packed active-space two-electron integrals `GGO`.
   *
   * The exact path always populates this buffer. The RI path may also fill it
   * as an auxiliary dense lookup cache for determinant-pair kernels while
   * still keeping `representation == ResolutionOfIdentity` so reverse-mode can
   * backpropagate through the RI factors.
   */
  std::vector<double> packed_active_two_electron_integrals;

  /**
   * @brief Number of auxiliary functions in the RI representation.
   *
   * This is zero on the current exact path.
   */
  int n_auxiliary_functions = 0;

  /**
   * @brief Metric-whitened RI active-pair factors `L_{A,P}`.
   *
   * Rows are auxiliary functions and columns are packed active-pair indices.
   * The matrix uses the project-standard column-major Eigen storage.
   */
  Eigen::MatrixXd ri_active_pair_factors;

  /**
   * @brief Optional dense active-orbital coefficient table on AO rows.
   *
   * Rows are AO basis functions and columns are active orbitals. This cached
   * dense matrix is reused on the forward/backward exact 2e path so callers do
   * not have to rebuild it from sparse orbital-preparation data.
   *
   * This matrix is cached on the dense forward path and
   * reused by reverse-mode backpropagation to avoid rebuilding the active
   * orbital coefficient table from sparse orbital-preparation data.
   * Persistent outputs should ignore this transient cache.
   */
  Eigen::MatrixXd dense_active_coefficients;

  /**
   * @brief Optional dense forward cache of `G * C` on AO-pair rows.
   *
   * Rows are packed AO-pair indices and columns are packed active-pair
   * indices. This dense matrix is populated on the dense forward path and
   * reused by reverse-mode backpropagation to skip a second sparse AO
   * two-electron sweep. Persistent outputs should ignore this transient cache.
   */
  Eigen::MatrixXd dense_ao_pair_products;
};

}  // namespace xmvb::vb
