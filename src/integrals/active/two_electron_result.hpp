#pragma once

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace xmvb::integrals {

enum class ActiveSpaceTwoElectronRepresentation {
    PackedExact,
    ResolutionOfIdentity,
};

/**
 * @brief Active-space two-electron integrals derived from AO ERIs.
 */
struct ActiveSpaceTwoElectronResult {
    /**
   * @brief Integral origin selecting the AO derivative backend.
   * Both values carry a packed four-index active kernel. ResolutionOfIdentity
   * additionally retains factors for AO transformations and derivative pullbacks,
   * not for repeated active-space Hamiltonian contractions.
   */
    ActiveSpaceTwoElectronRepresentation representation =
        ActiveSpaceTwoElectronRepresentation::PackedExact;

    /** Upper `R` with `S_active = R^T R` when the two-electron data and
   * `dense_active_coefficients` use `Cbar = C R^{-1}`. The associated active
   * one-electron integrals use this same frame. Empty means raw frame C.
   */
    Eigen::MatrixXd orbital_transform;

    /**
   * @brief Packed active-space two-electron integrals `GGO`.
   *
   * Always populated by forward preparation, for both exact and RI AO integrals.
   * The origin tag remains ResolutionOfIdentity for RI derivative pullbacks.
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

    /** @brief Budgeted accepted L_A C, shared by gradient and HVP consumers.
   * Rows (active orbital, auxiliary), columns AO. An absent resident image
   * uses the same exact auxiliary-tile transform, not a different operator.
   */
    std::shared_ptr<const Eigen::MatrixXd> ri_one_index_factors;

    /**
   * @brief Dense active-orbital coefficient table on AO rows.
   *
   * Rows are AO basis functions and columns are active orbitals. This cached
   * dense matrix is reused on the forward/backward exact 2e path so callers do
   * not have to rebuild it from sparse orbital-preparation data.
   *
   * Every forward representation caches this matrix because reverse-mode
   * backpropagation requires it. Persistent outputs should ignore this
   * transient cache.
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dense_ao_pair_products;
};

} // namespace xmvb::integrals
