#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "linalg/eigensolver.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/derivatives/gradient/active_space/result.hpp"

namespace xmvb::vb {

struct AcceptedPointContext;
struct SelectedMetric;

/**
 * @brief Move-only forward evaluation awaiting the active-space adjoint.
 *
 * The stored context contains the orbital-dependent integrals, structure
 * operator or matrices, and selected eigensystem. A rejected optimization
 * trial can discard it without evaluating derivatives; an accepted trial can
 * consume it to finish the exact gradient without repeating forward work.
 */
class ActiveSpaceForwardEvaluation {
  public:
    ActiveSpaceForwardEvaluation();
    ~ActiveSpaceForwardEvaluation();
    ActiveSpaceForwardEvaluation(ActiveSpaceForwardEvaluation&&) noexcept;
    ActiveSpaceForwardEvaluation& operator=(ActiveSpaceForwardEvaluation&&) noexcept;

    ActiveSpaceForwardEvaluation(const ActiveSpaceForwardEvaluation&) = delete;
    ActiveSpaceForwardEvaluation& operator=(const ActiveSpaceForwardEvaluation&) = delete;

    double total_energy() const noexcept;
    double wall_time_seconds() const noexcept;
    bool valid() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;

    friend class ActiveSpaceGradientEvaluator;
};

/**
 * @brief Gradient at prescribed, variationally nonstationary structure states.
 *
 * The supplied equal-weight state subspace is orthonormalized in its small
 * Gram metric and diagonalized within that subspace, without a full CI solve.
 * `structure_residuals` stores
 * @f$(H-E_sS)c_s@f$ for the normalized columns.  No structure eigensystem is
 * solved. Only `gradient.scf_result.total_energy` is populated in the SCF
 * result: fixed-state derivatives do not construct eigensolver output.
 */
struct ActiveSpaceFixedStructureGradient {
    ActiveSpaceGradientResult gradient;
    Eigen::MatrixXd normalized_coefficients;
    Eigen::MatrixXd structure_residuals;
    /** Derivative with respect to the original, unnormalized coefficient block. */
    Eigen::MatrixXd coefficient_gradient;
};

/**
 * @brief Analytic gradient evaluator for the active-space integral layer.
 *
 * This evaluator differentiates the VBSCF energy with respect
 * to:
 * - the active-space one-electron matrix `HHO`
 * - the packed active-space two-electron integrals `GGO`
 *
 * The generalized-eigen response and determinant/structure assembly are both
 * handled analytically. This provides the core gradient needed by a future
 * fully analytic orbital-response implementation.
 */
class ActiveSpaceGradientEvaluator {
  public:
    ActiveSpaceGradientEvaluator() = default;

    /**
   * @brief Evaluates the ground-state active-space analytic gradient.
   */
    ActiveSpaceGradientResult evaluate(const VbScfInput& input,
                                       double nuclear_repulsion_energy = 0.0) const;

    /**
   * @brief Evaluates a state-averaged active-space analytic gradient.
   */
    ActiveSpaceGradientResult evaluate(const VbScfInput& input,
                                       const std::vector<int>& selected_state_indices,
                                       const std::vector<double>& state_average_weights,
                                       double nuclear_repulsion_energy) const;

    /**
   * @brief Evaluates selected roots with the requested structure eigensolver.
   *
   * `initial_eigenvectors` contains consecutive lowest roots from the current
   * accepted point and is used only to recycle a Davidson subspace.
   */
    ActiveSpaceGradientResult
    evaluate(const VbScfInput& input,
             const std::vector<int>& selected_state_indices,
             const std::vector<double>& state_average_weights,
             double nuclear_repulsion_energy,
             StructureEigensolver structure_eigensolver,
             StructureSolveAccuracy structure_solve_accuracy,
             const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const;

    /** @brief Evaluates energy and retains the forward context for lazy differentiation. */
    ActiveSpaceForwardEvaluation
    evaluate_forward(const VbScfInput& input,
                     const std::vector<int>& selected_state_indices,
                     const std::vector<double>& state_average_weights,
                     double nuclear_repulsion_energy,
                     StructureEigensolver structure_eigensolver,
                     StructureSolveAccuracy structure_solve_accuracy,
                     const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const;

    /** @brief Completes the exact gradient from a retained forward evaluation. */
    ActiveSpaceGradientResult
    complete_gradient(const VbScfInput& input,
                      ActiveSpaceForwardEvaluation forward_evaluation) const;

    /** @brief Fixed-column energy/gradient; optional Gram belongs to input's orbitals. */
    ActiveSpaceFixedStructureGradient
    evaluate_fixed_structure(const VbScfInput& input,
                             const std::vector<int>& selected_state_indices,
                             const std::vector<double>& state_average_weights,
                             double nuclear_repulsion_energy,
                             const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
                             const SelectedMetric* metric = nullptr,
                             bool retain_context = false) const;

    /**
   * @brief Evaluates a state-averaged gradient reusing a prebuilt active-space context.
   */
    ActiveSpaceGradientResult
    evaluate(const VbScfInput& input,
             TimedPreparedActiveSpaceContext timed_prepared_active_space_context,
             const std::vector<int>& selected_state_indices,
             const std::vector<double>& state_average_weights,
             double nuclear_repulsion_energy) const;

    /**
   * @brief Rebuilds only the orbital-dependent integral layer under a frozen active-space adjoint.
   *
   * The returned result contains freshly rebuilt orbital preparation, AO-H1E,
   * active-space HHO/GGO inputs at `input`, while the active-space adjoint
   * buffers are copied from the accepted-point context. This provides a cheap
   * accepted-point core-curvature probe that avoids rebuilding the determinant
   * and generalized-eigen response on every HVP application.
   */
    ActiveSpaceGradientResult
    evaluate_with_fixed_active_space_adjoint(const VbScfInput& input,
                                             const AcceptedPointContext& accepted_point_context,
                                             double nuclear_repulsion_energy = 0.0) const;

  private:
    FullDeterminantStructureHamiltonianOverlapBuilder structure_builder_;
    xmvb::linalg::GeneralizedEigensolver generalized_eigensolver_;
};

} // namespace xmvb::vb
