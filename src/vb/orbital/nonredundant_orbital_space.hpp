#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

/**
 * @brief Block-local nonredundant orbital-replacement space in packed parameters.
 *
 * The directions are generated from the current occupied orbitals and block-local
 * virtual orbitals. For partially overlapping sparse blocks, the block-local AO
 * support is the union of every member orbital support rather than the support
 * of one representative row. Each reduced coordinate corresponds to a
 * first-order orbital replacement direction represented directly in the packed
 * sparse coefficient vector used by the optimizer and HVP machinery.
 */
class NonredundantOrbitalSpace {
public:
  struct ProjectionResult {
    Eigen::VectorXd reduced_gradient;
    Eigen::VectorXd packed_projected_gradient;
  };

  /**
   * @brief Explicit block-local nonredundant amplitudes for one reduced step.
   *
   * Each block stores the occupied and block-local virtual orbitals in the AO
   * basis restricted to that block support, together with the reduced step
   * expressed on the raw inactive/active, active/active, and occupied/virtual
   * candidate amplitudes used internally by the nonredundant basis
   * construction. In the mixed-chart path the `active_active_coefficients`
   * field carries the active-shape increment `ΔL_a` rather than an
   * antisymmetric active-active rotation because the physical VB active
   * orbitals remain genuinely nonorthogonal variational objects.
   */
  struct BlockRotationDirection {
    int n_inactive = 0;
    int n_occupied = 0;
    int n_virtual = 0;
    std::vector<int> basis_function_indices;
    std::vector<int> occupied_orbital_indices;
    Eigen::MatrixXd occupied_orbitals;
    Eigen::MatrixXd virtual_orbitals;
    Eigen::MatrixXd inactive_active_coefficients;
    Eigen::MatrixXd active_active_coefficients;
    Eigen::MatrixXd occupied_virtual_coefficients;
  };

  NonredundantOrbitalSpace(
      const OrbitalPreparationInput& orbital_preparation_input,
      const SparseOrbitalParameterView& parameter_view,
      const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbital_basis_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& physical_orbital_matrix,
      const Eigen::MatrixXd* ao_effective_h1e = nullptr);

  int reduced_size() const noexcept {
    return reduced_size_;
  }

  Eigen::VectorXd project_reduced_gradient(
      const Eigen::VectorXd& packed_gradient) const;

  ProjectionResult project_gradient(
      const Eigen::VectorXd& packed_gradient) const;

  ProjectionResult project_vector(
      const Eigen::VectorXd& packed_vector) const;

  bool has_reduced_curvature_diagonal() const noexcept {
    return has_reduced_curvature_diagonal_;
  }

  bool use_block_preconditioner_by_default() const noexcept {
    return use_block_preconditioner_by_default_;
  }

  Eigen::VectorXd apply_inverse_reduced_curvature(
      const Eigen::VectorXd& reduced_vector) const;

  /**
   * @brief Applies the block-local reduced curvature diagonal itself.
   *
   * When the block curvature model is unavailable, this falls back to the
   * identity map so higher-level matrix-free operators still have a cheap
   * positive baseline model.
   */
  Eigen::VectorXd apply_reduced_curvature(
      const Eigen::VectorXd& reduced_vector) const;

  /**
   * @brief Applies the reduced-space block preconditioner used by TNHVP.
   *
   * The reduced TN chart is block-separable. Blocks whitened by the spectral
   * factorization of `D^T D` already have identity metric in reduced
   * coordinates, so their baseline model is just the positive reduced
   * curvature diagonal. If a dense full-support block cannot be factorized, it
   * remains in the raw candidate chart, where the local Gram matrix
   * `G = D^T D` is not the identity. For that fallback we apply the symmetric
   * positive model `M_block = C_block^{1/2} G_block C_block^{1/2}` and return
   * `M_block^{-1} v`. Sparse blocks that are not factorized fall back to the
   * diagonal `C_block^{-1}` model because nesting an iterative `G^{-1}` solve
   * inside every outer PCG preconditioner application is usually not worth the
   * cost.
   */
  Eigen::VectorXd apply_inverse_reduced_block_preconditioner(
      const Eigen::VectorXd& reduced_vector) const;

  Eigen::VectorXd expand_step(
      const Eigen::VectorXd& reduced_step) const;

  /**
   * @brief Returns the first-order stored-coefficient direction of `retract_step()`.
   *
   * Sparse blocks are retracted back to each orbital's local
   * `x^T S x = 1` manifold, so the finite-step input path is not the raw
   * additive packed direction returned by `expand_step()`. This linearization
   * provides the actual derivative of `orbital_value_table` with respect to
   * the reduced step at the accepted point.
   */
  Eigen::VectorXd expand_retract_input_tangent(
      const OrbitalPreparationInput& orbital_preparation_input,
      const Eigen::VectorXd& reduced_step) const;

  /**
   * @brief Lifts one reduced step to a finite sparse-orbital trial point.
   *
   * All blocks now generate the finite step from the same internal mixed chart
   * `(Q_i, Q_a, Q_v, L_a)`. Dense full-support blocks can write the rotated
   * occupied columns back directly. Sparse blocks restrict that internal trial
   * point to each orbital's fixed AO support and then retract the support-local
   * target on the local `x^T S x = 1` manifold.
   */
  OrbitalPreparationInput retract_step(
      const OrbitalPreparationInput& orbital_preparation_input,
      const Eigen::VectorXd& reduced_step,
      double step_scale = 1.0) const;

  std::vector<BlockRotationDirection> expand_block_rotation_directions(
      const Eigen::VectorXd& reduced_step) const;

private:
  struct OrbitalProjector {
    std::vector<int> flat_indices;
    std::vector<int> packed_indices;
    std::vector<int> block_rows;
    std::vector<int> flat_index_by_block_row;
    std::vector<int> packed_index_by_block_row;
    Eigen::MatrixXd occupied_masked;
    Eigen::MatrixXd inactive_working_masked;
    Eigen::MatrixXd active_working_masked;
    Eigen::MatrixXd virtual_masked;
    Eigen::MatrixXd internal_virtual_masked;
  };

  struct BlockBasis {
    int n_inactive = 0;
    int n_occupied = 0;
    int n_virtual = 0;
    std::vector<int> basis_function_indices;
    std::vector<int> occupied_orbital_indices;
    Eigen::MatrixXd block_overlap_matrix;
    Eigen::MatrixXd occupied_orbitals;
    Eigen::MatrixXd reference_occupied_orbitals;
    Eigen::MatrixXd inactive_working_orbitals;
    Eigen::MatrixXd active_working_orbitals;
    Eigen::MatrixXd inactive_right_transform;
    Eigen::MatrixXd active_shape_matrix;
    Eigen::MatrixXd active_inactive_gauge_coefficients;
    Eigen::MatrixXd internal_virtual_orbitals;
    Eigen::MatrixXd inactive_metric_matrix;
    Eigen::MatrixXd inactive_metric_inverse;
    Eigen::MatrixXd inactive_metric_eigenvectors;
    Eigen::VectorXd inactive_metric_eigenvalues;
    Eigen::MatrixXd active_shape_metric_matrix;
    Eigen::MatrixXd active_shape_metric_inverse;
    Eigen::MatrixXd active_shape_metric_eigenvectors;
    Eigen::VectorXd active_shape_metric_eigenvalues;
    Eigen::MatrixXd active_shape_inverse;
    Eigen::MatrixXd inactive_plus_gauge_metric_matrix;
    Eigen::MatrixXd virtual_orbitals;
    std::vector<OrbitalProjector> orbitals;
    bool uses_dense_full_support_projector = false;
    // Raw candidate-chart metric diagonal in the uncompressed direction basis.
    Eigen::VectorXd candidate_metric_diagonal;
    bool has_exact_metric_factorization = false;
    // Raw candidate coefficients `a = E z` from reduced coordinates `z`.
    // For spectral-whitened blocks, `E^T D^T D E = I` and columns with
    // numerically zero metric norm are omitted from the reduced chart.
    Eigen::MatrixXd candidate_from_reduced;
    Eigen::VectorXd reduced_curvature_diagonal;
    int reduced_offset = 0;
  };

  Eigen::MatrixXd gather_dense_full_support_block_matrix(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& packed_vector) const;

  void accumulate_dense_full_support_block_matrix(
      const BlockBasis& block_basis,
      const Eigen::MatrixXd& block_matrix,
      Eigen::VectorXd* packed_vector) const;

  void write_dense_full_support_block_matrix(
      const BlockBasis& block_basis,
      const Eigen::MatrixXd& block_matrix,
      std::vector<double>* orbital_value_table) const;

  Eigen::MatrixXd build_dense_full_support_block_step(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& candidate_coefficients) const;

  // Build the finite-step occupied block from the internal mixed chart
  // `(Q_i, Q_a, Q_v, L_a)`.  The returned columns are physical occupied
  // orbitals on the block union support, ready for either dense write-back or
  // sparse support-local restriction.
  Eigen::MatrixXd build_mixed_chart_trial_occupied_orbitals(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& candidate_coefficients) const;

  Eigen::VectorXd project_dense_full_support_candidate_overlap(
      const BlockBasis& block_basis,
      const Eigen::Ref<const Eigen::MatrixXd>& block_columns) const;

  Eigen::VectorXd build_dense_full_support_metric_diagonal(
      const BlockBasis& block_basis) const;

  Eigen::VectorXd build_sparse_mixed_chart_metric_diagonal(
      const BlockBasis& block_basis) const;

  Eigen::VectorXd apply_dense_full_support_candidate_metric(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& candidate_coefficients) const;

  Eigen::VectorXd solve_dense_full_support_candidate_metric(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& right_hand_side) const;

  void initialize_dense_full_support_metric_cache(
      BlockBasis* block_basis) const;

  void initialize_block_mixed_chart_cache(
      BlockBasis* block_basis,
      const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_occupied_orbitals) const;

  Eigen::VectorXd project_block_candidate_overlap(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& packed_vector) const;

  void accumulate_block_candidate_combination(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& candidate_coefficients,
      Eigen::VectorXd* packed_vector) const;

  Eigen::VectorXd block_candidate_coefficients_from_reduced_step(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& reduced_step) const;

  Eigen::VectorXd apply_block_candidate_metric(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& candidate_coefficients) const;

  Eigen::VectorXd solve_block_candidate_metric(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& right_hand_side) const;

  static int block_reduced_size(const BlockBasis& block_basis) noexcept;

  void maybe_factorize_block_candidate_metric(
      BlockBasis* block_basis);

  ProjectionResult project_impl(
      const Eigen::VectorXd& packed_vector,
      bool recover_tangent_coordinates,
      bool build_packed_projection) const;

  std::vector<BlockBasis> block_bases_;
  int packed_parameter_size_ = 0;
  int reduced_size_ = 0;
  bool has_reduced_curvature_diagonal_ = false;
  bool use_block_preconditioner_by_default_ = true;
};

}  // namespace xmvb::vb
