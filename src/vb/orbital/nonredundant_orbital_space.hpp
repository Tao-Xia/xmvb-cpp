#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

/**
 * @brief Block-local nonredundant orbital rotation space in packed raw parameters.
 *
 * The directions are generated from the current occupied orbitals and block-local
 * virtual orbitals. For partially overlapping sparse blocks, the block-local AO
 * support is the union of every member orbital support rather than the support
 * of one representative row. Each reduced coordinate corresponds to a
 * first-order orbital rotation direction represented directly in the packed
 * sparse coefficient vector used by the optimizer.
 */
class NonredundantOrbitalSpace {
public:
  struct ProjectionResult {
    Eigen::VectorXd reduced_gradient;
    Eigen::VectorXd packed_projected_gradient;
  };

  /**
   * @brief Explicit block-local orbital-rotation amplitudes for one reduced step.
   *
   * Each block stores the occupied and block-local virtual orbitals in the AO
   * basis restricted to that block support, together with the reduced step
   * expressed on the raw inactive/active, active/active, and occupied/virtual
   * candidate amplitudes used internally by the nonredundant basis
   * construction. The active-active block is retained explicitly because the
   * physical VB active orbitals are nonorthogonal variational objects rather
   * than an internal gauge.
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
      const std::vector<double>* ao_effective_h1e = nullptr);

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

  Eigen::VectorXd expand_step(
      const Eigen::VectorXd& reduced_step) const;

  /**
   * @brief Retracts one reduced step back to a finite sparse-orbital trial point.
   *
   * The nonredundant basis is built from block-local occupied/virtual orbital
   * rotations, so finite trial points should be generated in that same block
   * orbital chart rather than by adding the tangent vector directly in the raw
   * sparse-coefficient coordinates. This retraction applies a Cayley-style
   * finite mixing inside each block and then scatters the resulting occupied
   * orbital coefficients back into the legacy sparse layout.
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
    std::vector<int> flat_index_by_block_row;
    std::vector<int> packed_index_by_block_row;
    Eigen::MatrixXd occupied_masked;
    Eigen::MatrixXd reference_occupied_masked;
    Eigen::MatrixXd virtual_masked;
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
    Eigen::VectorXd candidate_metric_diagonal;
    bool has_exact_metric_factorization = false;
    Eigen::MatrixXd candidate_metric_cholesky_factor;
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

  Eigen::VectorXd project_dense_full_support_candidate_overlap(
      const BlockBasis& block_basis,
      const Eigen::Ref<const Eigen::MatrixXd>& block_columns) const;

  Eigen::VectorXd build_dense_full_support_metric_diagonal(
      const BlockBasis& block_basis) const;

  Eigen::VectorXd apply_dense_full_support_candidate_metric(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& candidate_coefficients) const;

  Eigen::VectorXd solve_dense_full_support_candidate_metric(
      const BlockBasis& block_basis,
      const Eigen::VectorXd& right_hand_side) const;

  void initialize_dense_full_support_metric_cache(
      BlockBasis* block_basis) const;

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

  void maybe_factorize_small_block_candidate_metric(
      BlockBasis* block_basis);

  ProjectionResult project_impl(
      const Eigen::VectorXd& packed_vector,
      bool recover_tangent_coordinates,
      bool build_packed_projection) const;

  std::vector<BlockBasis> block_bases_;
  int packed_parameter_size_ = 0;
  int reduced_size_ = 0;
  bool has_reduced_curvature_diagonal_ = false;
};

}  // namespace xmvb::vb
