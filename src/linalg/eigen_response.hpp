#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "linalg/eigensolver.hpp"

namespace xmvb::linalg {

/**
 * @brief Accuracy and work limit for matrix-free generalized-eigen response.
 */
struct EigenResponseOptions {
    int max_iterations;
    double relative_residual_tolerance;
};

/** @brief Fixed symmetric inverse preconditioner on each response complement.
 * The action must be linear, finite and positive definite on the complement
 * of the corresponding overlap image. It replaces Jacobi on the unresolved
 * complement, but retains resolved recycle spectral modes and the recycled
 * Galerkin initial solution. The lower bound applies on that complement; it
 * only schedules true residual checks and never replaces the original
 * bordered certificate.
 */
struct EigenResponsePreconditioner {
    std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>&, Eigen::Index)> apply;
    double minimum_eigenvalue = 0.0;
};

/** @brief Unconditional Galerkin action of one frozen recycle space. */
struct EigenResponseRecycleApplication {
    Eigen::VectorXd solution;
    Eigen::VectorXd operator_image;
    Eigen::VectorXd hamiltonian_image;
    Eigen::VectorXd overlap_image;
    bool available = false;
    bool has_generalized_images = false;
};

/**
 * @brief Same-operator Galerkin space recycled across response right-hand sides.
 *
 * The stored columns are Euclidean-orthonormal projected response vectors,
 * their exact projected operator images, and—when supplied—the corresponding
 * full H/S images. The latter permit action-free reconstruction and bordered
 * residual certification after the Galerkin space is enriched. The space is
 * owned by one accepted orbital point and one selected root.
 */
class EigenResponseRecycleSpace {
  public:
    int dimension() const noexcept;
    int size() const noexcept;
    std::uint64_t revision() const noexcept;
    void clear();

    /** @brief Orthonormal columns spanning the frozen response model. */
    const Eigen::MatrixXd& basis() const noexcept;

    /** @brief Pseudoinverse of the symmetric projected response operator. */
    const Eigen::MatrixXd& projected_inverse() const;

    /** @brief Symmetric projected response operator @f$W^T C W@f$. */
    const Eigen::MatrixXd& projected_operator() const;

    /**
   * @brief Applies an SPD spectral/Jacobi preconditioner.
   *
   * Numerically resolved recycle Ritz modes use the inverse absolute
   * projected eigenvalues. The Euclidean complement uses the supplied
   * positive Jacobi inverse, projected on both sides to preserve symmetry.
   */
    Eigen::VectorXd apply_absolute_spectral_preconditioner(
        const Eigen::Ref<const Eigen::VectorXd>& right_hand_side,
        const Eigen::Ref<const Eigen::VectorXd>& jacobi_inverse,
        const EigenResponsePreconditioner* preconditioner = nullptr,
        Eigen::Index state = 0) const;

    /** @brief SPD bound after replacing resolved recycle modes by their spectra. */
    double preconditioner_lower_bound(double complement_lower_bound) const;

    /**
   * @brief Applies the frozen symmetric Galerkin inverse to any finite RHS.
   *
   * This operation is linear in the right-hand side and never rejects a
   * result based on residual improvement. Small projected spectral
   * modes are removed with one RHS-independent numerical-rank cutoff.
   */
    EigenResponseRecycleApplication
    galerkin_apply(const Eigen::Ref<const Eigen::VectorXd>& right_hand_side) const;

    bool append(const Eigen::Ref<const Eigen::VectorXd>& solution,
                const Eigen::Ref<const Eigen::VectorXd>& operator_image);

    bool append_generalized(const Eigen::Ref<const Eigen::VectorXd>& solution,
                            const Eigen::Ref<const Eigen::VectorXd>& operator_image,
                            const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_image,
                            const Eigen::Ref<const Eigen::VectorXd>& overlap_image);

  private:
    bool append_impl(const Eigen::Ref<const Eigen::VectorXd>& solution,
                     const Eigen::Ref<const Eigen::VectorXd>& operator_image,
                     const Eigen::VectorXd* hamiltonian_image,
                     const Eigen::VectorXd* overlap_image);
    void prepare_projected_inverse() const;

    Eigen::MatrixXd basis_;
    Eigen::MatrixXd operator_images_;
    Eigen::MatrixXd hamiltonian_images_;
    Eigen::MatrixXd overlap_images_;
    std::uint64_t revision_ = 0;
    mutable Eigen::MatrixXd projected_inverse_;
    mutable Eigen::MatrixXd projected_operator_;
    mutable Eigen::MatrixXd spectral_preconditioner_basis_;
    mutable Eigen::VectorXd spectral_preconditioner_inverse_eigenvalues_;
    mutable std::uint64_t projected_inverse_revision_ = static_cast<std::uint64_t>(-1);
};

/**
 * @brief Selected-root response and numerical diagnostics.
 */
struct EigenResponseResult {
    Eigen::MatrixXd eigenvector_response;
    Eigen::VectorXd eigenvalue_response;
    /** @brief Residual of the complete bordered response equations. */
    Eigen::MatrixXd bordered_residuals;
    Eigen::VectorXd relative_residual_norms;
    std::vector<int> iterations;
    int block_actions = 0;
};

/**
 * @brief Response of an equally weighted generalized-eigen subspace.
 *
 * `selected_matrix_response` is the selected-space effective-Hamiltonian
 * response in
 *
 * `H dC - S dC E - S C dE = -(dH C - dS C E)`.
 *
 * Unlike isolated-root response, `dE` is generally not diagonal.  Its
 * off-diagonal entries absorb the chosen rotation within the selected
 * subspace.  The returned vectors use the symmetric metric gauge
 *
 * `C^T S dC = -0.5 C^T dS C`.
 */
struct EigenSubspaceResponseResult {
    Eigen::MatrixXd eigenvector_response;
    Eigen::MatrixXd selected_matrix_response;
    /** @brief Top-block residual of the bordered response equation. */
    Eigen::MatrixXd equation_residual;
    Eigen::VectorXd relative_residual_norms;
    std::vector<int> iterations;
    int block_actions = 0;
};

/**
 * @brief Solve-free coupling data for an equally weighted selected subspace.
 *
 * `forcing` is @f$\delta H C-\delta S C E@f$. The gauge response satisfies
 * @f$C^T S\,\delta C_g=-\tfrac12 C^T\delta S C@f$. `horizontal_forcing`
 * is the Euclidean projection of the forcing, including the accepted Ritz-
 * residual image of that gauge response, away from the span of @f$SC@f$.
 * `gauge_selected_matrix_response` is the multiplier @f$M_g@f$ in
 * @f$F+(H\delta C_g-S\delta C_gE)-SCM_g@f$.
 */
struct EqualWeightEigenCoupling {
    Eigen::MatrixXd metric;
    Eigen::MatrixXd forcing;
    Eigen::MatrixXd gauge_target;
    /** Internal selected-state frame A_g, with gauge response C A_g. */
    Eigen::MatrixXd gauge_frame;
    Eigen::MatrixXd gauge_selected_matrix_response;
    Eigen::MatrixXd horizontal_forcing;
};

/**
 * @brief Prepares equal-weight orbital--structure coupling without a solve.
 *
 * This function performs no generalized-eigen response iteration and mutates no
 * recycle space. `selected_residuals` is the accepted Ritz residual
 * @f$HC-SCE@f$ and may be zero for an exact accepted eigensystem.
 * If horizontal_forcing is false, only the normalization gauge is prepared;
 * forcing and horizontal_forcing in the result remain empty.
 */
EqualWeightEigenCoupling prepare_equal_weight_generalized_eigen_coupling(
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_residuals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    bool horizontal_forcing = true);

/**
 * @brief Solves selected generalized-eigenpair response without a full spectrum.
 *
 * For each accepted pair `H c = E S c`, the solver applies preconditioned
 * MINRES to `H - E S` on the Euclidean complement of each constraint image
 * `S c`. Removing this bordered constraint direction keeps the metric gauge
 * out of the Krylov recurrence. The response along `c` is fixed by
 * `c^T S dc = -0.5 c^T dS c`, while the Ritz residual `(H - E S)c` is retained
 * in both the projected forcing and the eigenvalue response. The original
 * symmetric bordered equation then certifies the true residual even for a
 * finite-accuracy Davidson root. All selected roots advance together through
 * one block H/S action per iteration. The caller supplies accepted-point
 * `S C`; an optional accepted-point Ritz-residual cache avoids another action.
 *
 * The returned residual measures this linear equation at the supplied Ritz
 * root, not the error of that Ritz root relative to an exact eigenpair.
 */
EigenResponseResult solve_generalized_eigen_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces = {},
    const Eigen::MatrixXd* selected_residuals = nullptr,
    const EigenResponsePreconditioner* preconditioner = nullptr);

/**
 * @brief Evaluates isolated-root response in fixed common Galerkin spaces.
 *
 * The spaces are not enriched and no iterative refinement is performed. The
 * returned bordered residual is an accuracy diagnostic, not an acceptance
 * test; any finite residual is returned to the caller for an explicit
 * enrich-and-retry decision.
 */
EigenResponseResult evaluate_frozen_generalized_eigen_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const std::vector<const EigenResponseRecycleSpace*>& recycle_spaces,
    const Eigen::MatrixXd* selected_residuals = nullptr);

/**
 * @brief Applies the complete generalized eigenspectrum to selected responses.
 *
 * Dense solves and memory-affordable Davidson bases retain the full spectrum.
 * This route evaluates its first-order response and certifies the bordered true
 * residual with the independent H/S action; it never forms an orbital Hessian.
 */
EigenResponseResult solve_generalized_eigen_response_from_full_spectrum(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& full_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& full_eigenvectors,
    const std::vector<int>& selected_root_indices,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double relative_residual_tolerance);

/**
 * @brief Solves the response of one equally weighted selected-state cluster.
 *
 * All selected-selected rotations are gauge degrees of freedom for an equal
 * state average.  This routine therefore projects the linear systems against
 * the complete selected subspace and solves only selected-external response.
 * It remains well-defined when selected roots are degenerate; a degeneracy
 * with an omitted external root must instead be included in the cluster.
 *
 * The selected columns must be S-orthonormal generalized eigenvectors and
 * `delta_*_selected` must contain `dH C` and `dS C`, respectively.
 */
EigenSubspaceResponseResult solve_equal_weight_generalized_eigen_subspace_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces = {},
    const Eigen::MatrixXd* selected_residuals = nullptr);

/**
 * @brief Evaluates equal-weight subspace response in fixed Galerkin spaces.
 *
 * The complete selected metric image is projected out. The symmetric metric
 * gauge, its finite-Ritz forcing, and the full selected-space matrix response
 * are restored before the true residual is evaluated. Poor finite residuals
 * are reported without mutating or refining the supplied spaces.
 */
EigenSubspaceResponseResult evaluate_frozen_equal_weight_generalized_eigen_subspace_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const std::vector<const EigenResponseRecycleSpace*>& recycle_spaces,
    const Eigen::MatrixXd* selected_residuals = nullptr);

/**
 * @brief Full-spectrum reference for equal-weight invariant-subspace response.
 */
EigenSubspaceResponseResult
solve_equal_weight_generalized_eigen_subspace_response_from_full_spectrum(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& full_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& full_eigenvectors,
    const std::vector<int>& selected_root_indices,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double relative_residual_tolerance);

} // namespace xmvb::linalg
