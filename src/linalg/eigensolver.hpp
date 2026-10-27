#pragma once

#include "core/index.hpp"

#include <span>

#include <functional>
#include <vector>

#include <Eigen/Core>

namespace xmvb::linalg {

/**
 * @brief Result of a symmetric generalized eigenvalue problem.
 *
 * All matrices use column-major storage:
 * `matrix_data[column * dimension + row]`.
 */
struct GeneralizedEigenResult {
    /**
   * @brief Eigenvalues in ascending order.
   */
    std::vector<double> eigenvalues;

    /**
   * @brief Column-major eigenvector matrix.
   *
   * Column `i` contains the eigenvector associated with `eigenvalues[i]`.
   */
    std::vector<double> eigenvector_matrix;
};

/**
 * @brief Images returned by one block generalized-eigenvalue operator call.
 */
struct GeneralizedEigenActionResult {
    Eigen::MatrixXd hamiltonian;
    Eigen::MatrixXd overlap;
};

using GeneralizedEigenAction =
    std::function<GeneralizedEigenActionResult(const Eigen::Ref<const Eigen::MatrixXd>&)>;

/**
 * @brief Explicit numerical and memory budgets for block Davidson.
 */
struct DavidsonOptions {
    int n_roots;
    Eigen::Index max_iterations;
    int max_subspace_dimension;
    /** Maximum change of consecutive Ritz energies requested by the outer calculation. */
    double energy_tolerance;
    /** Maximum normalized eigen-equation backward error. */
    double residual_tolerance;
    /** Maximum norm of `(H - E S)c / ||c||`, in Hamiltonian units.
   * Unlike the relative backward error, this budget cannot be weakened by
   * adding a constant times S to H. It bounds CI stationarity, not the
   * orbital-gradient error (which also depends on gaps and coupling).
   */
    double stationarity_tolerance;
    /** @brief Optional invariant residual norm for a normalized Ritz state.
   * Called only after the algebraic and energy checks pass. This allows an
   * operator to certify the residual in its physical representation without
   * constructing or inverting a dense overlap matrix.
   */
    std::function<double(const Eigen::VectorXd&, double)> physical_residual_norm = {};
    /** @brief Optional inverse action on an algebraic residual covector.
   * Given `(H - E S)c` and the Ritz energy E, return the positive inverse
   * image in coefficient coordinates. The solver negates that image to form
   * a Davidson correction. When unset, the diagonal `(H_ii - E S_ii)`
   * preconditioner is used.
   */
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> residual_preconditioner = {};
    /** @brief Optional per-iteration observation after all root residuals are
   * evaluated, including the final converged iteration. The arguments are
   * iteration number, active subspace size, lowest Ritz energy, and largest
   * algebraic relative residual, and largest stationarity residual among roots.
   */
    std::function<void(Eigen::Index, int, double, double, double)> observe_iteration = {};
    /** @brief Exact identity metric: the action returns an empty overlap image.
   * Shares the generalized solver but omits the redundant S-basis storage,
   * products and reorthogonalization updates. Subspace/accuracy budgets stay
   * unchanged. The caller must establish an isometric coordinate model.
   */
    bool identity_overlap = false;
};

/**
 * @brief Builds dimension-scaled Davidson budgets for consecutive low roots.
 *
 * The basis and projected matrix are bounded by a fixed workspace budget.
 * Accuracy is mandatory:
 * this core routine does not invent a machine-precision stopping target.
 */
DavidsonOptions make_davidson_options(Eigen::Index dimension,
                                      int n_roots,
                                      double energy_tolerance,
                                      double relative_residual_tolerance,
                                      double stationarity_tolerance);

/** @brief Deterministic diagonal/random initial block, before any coordinate map. */
Eigen::MatrixXd
make_davidson_initial_vectors(const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                              const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                              int n_vectors);

/**
 * @brief Converged Davidson eigenpairs and solver diagnostics.
 */
struct DavidsonResult {
    GeneralizedEigenResult eigenpairs;
    /** @brief Exact retained Ritz image `H C`, including the finite CI residual. */
    Eigen::MatrixXd hamiltonian_eigenvectors;
    /** @brief Matrix `S C` for the converged roots, retained without forming S. */
    Eigen::MatrixXd overlap_eigenvectors;
    std::vector<double> relative_residual_norms;
    /** @brief CI stationarity residuals in Hamiltonian units. */
    std::vector<double> stationarity_residual_norms;
    Eigen::Index iterations = 0;
    int block_actions = 0;
    int peak_subspace_dimension = 0;
};

/**
 * @brief Solves the symmetric-definite problem `H C = S C E`.
 *
 * Both input matrices must use column-major storage and `overlap_matrix`
 * must be positive definite.
 */
class GeneralizedEigensolver {
  public:
    /**
   * @brief Solves the generalized eigenvalue problem.
   *
   * Input spans borrow storage; LAPACK receives private writable copies.
   *
   * @param hamiltonian_matrix Column-major symmetric Hamiltonian matrix.
   * @param overlap_matrix Column-major symmetric positive-definite overlap matrix.
   * @param dimension Matrix dimension.
   * @return GeneralizedEigenResult Eigenvalues and eigenvectors.
   */
    GeneralizedEigenResult solve_dense(std::span<const double> hamiltonian_matrix,
                                       std::span<const double> overlap_matrix,
                                       Eigen::Index dimension) const;

    /**
   * @brief Solves only the generalized eigenvalues.
   *
   * This skips the full eigenvector matrix and is intended for energy-only
   * trial-point screening where only selected-state energies are needed.
   */
    std::vector<double> solve_dense_eigenvalues(std::span<const double> hamiltonian_matrix,
                                                std::span<const double> overlap_matrix,
                                                Eigen::Index dimension) const;

    /**
   * @brief Solves the lowest generalized eigenpairs from block H/S actions.
   *
   * The basis, its Hamiltonian image, and its overlap image are updated
   * together, so metric orthogonalization never triggers redundant operator
   * calls. The projected Hamiltonian is extended incrementally, and only the
   * requested lowest Ritz pairs are solved unless a restart needs the full
   * subspace spectrum. Failure to meet the requested residual tolerance is
   * reported as an exception rather than returning unconverged Ritz pairs.
   */
    DavidsonResult solve_davidson(const GeneralizedEigenAction& action,
                                  const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                                  const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                                  const DavidsonOptions& options) const;

    /**
   * @brief Solves from caller-supplied vectors, such as previous-step roots.
   *
   * The initial block must contain at least `n_roots` linearly independent
   * vectors and no more than `max_subspace_dimension` columns.
   */
    DavidsonResult solve_davidson(const GeneralizedEigenAction& action,
                                  const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                                  const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                                  const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
                                  const DavidsonOptions& options) const;
};

} // namespace xmvb::linalg
