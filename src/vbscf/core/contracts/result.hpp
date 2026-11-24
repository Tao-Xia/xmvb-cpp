#pragma once

#include <optional>
#include <vector>

#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Diagnostics from one converged matrix-free Davidson solve.
 */
struct DavidsonDiagnostics {
    /** Number of projected Davidson iterations. */
    int iterations = 0;

    /** Number of joint block evaluations of the Hamiltonian and overlap. */
    int block_actions = 0;

    /** Largest retained Davidson subspace dimension. */
    int peak_subspace_dimension = 0;

    /** Largest converged relative residual among the requested roots. */
    double maximum_relative_residual = 0.0;

    /** Largest CI stationarity residual, in hartree; not an orbital error bound. */
    double maximum_stationarity_residual = 0.0;
};

/** First-order TNVB diagnostics; absent for the determinant backend. */
struct TnVbDiagnostics {
    int structure_bond = 0;
    int total_bond = 0;
    bool gradient_evaluated = false;
    double fit_energy_error = 0.0;
    double rdm_identity_error = 0.0;
    /** Includes individual MPS construction; excludes Python startup. */
    double state_and_hs_wall_seconds = 0.0;
    double rdm_wall_seconds = 0.0;
};

/**
 * @brief Result of a single VBSCF matrix evaluation step.
 *
 * This object captures the complete state needed by downstream RDM and
 * gradient code:
 * - optional full structure Hamiltonian and overlap matrices
 * - consecutive lowest generalized eigenpairs required by the caller
 * - selected-state averaging metadata
 */
struct VbScfResult {
    /**
   * @brief Number of VB structures.
   */
    Eigen::Index n_structures = 0;

    /**
   * @brief Nuclear repulsion energy supplied by the caller.
   */
    double nuclear_repulsion_energy = 0.0;

    /**
   * @brief Inactive-space reference energy `E11`.
   */
    double one_electron_reference_energy = 0.0;

    /**
   * @brief State-averaged electronic energy from the selected states.
   */
    double electronic_energy = 0.0;

    /**
   * @brief State-averaged total energy including nuclear repulsion.
   */
    double total_energy = 0.0;

    /**
   * @brief Mean of the available structure-overlap diagonal estimate.
   * Exact only when structure_overlap_diagonal_exact is true.
   */
    double average_structure_overlap = 0.0;

    /** Final-solve diagnostics, present only for the Davidson eigensolver. */
    std::optional<DavidsonDiagnostics> davidson_diagnostics;

    std::optional<TnVbDiagnostics> tnvb_diagnostics;

    /**
   * @brief Available consecutive lowest generalized eigenvalues.
   *
   * Davidson results retain only the requested low roots. Dense reference
   * results contain the complete spectrum.
   */
    std::vector<double> electronic_state_energies;

    /**
   * @brief Selected-state total energies including nuclear repulsion.
   */
    std::vector<double> selected_state_total_energies;

    /**
   * @brief Selected state indices used for state averaging.
   */
    std::vector<int> selected_state_indices;

    /**
   * @brief Normalized weights used for state averaging.
   */
    std::vector<double> state_average_weights;

    /**
   * @brief Column-major generalized eigenvectors aligned with available roots.
   */
    std::vector<double> eigenvector_matrix;

    /**
   * @brief Available structure-overlap diagonal, possibly a Jacobi estimate.
   *
   * The exactness flag distinguishes physical norms from preconditioner data.
   * Output computes missing exact norms only when normalized coefficients are
   * requested; optimization does not complete this array for reporting.
   */
    std::vector<double> structure_overlap_diagonal;

    /** Whether `structure_overlap_diagonal` contains exact projected norms. */
    bool structure_overlap_diagonal_exact = false;

    /**
   * @brief Column-major products `S C` aligned with available Davidson roots.
   *
   * These products are sufficient for exact Coulson--Chirgwin weights. Dense
   * results may leave this empty because the explicit overlap matrix exists.
   */
    std::vector<double> overlap_eigenvector_matrix;

    /**
   * @brief Optional explicit structure Hamiltonian and overlap matrices.
   *
   * These remain empty throughout matrix-free Davidson optimization and
   * reporting. Dense reference runs materialize them explicitly.
   */
    StructureAccumulationResult structure_matrices;
};

} // namespace xmvb::vb
