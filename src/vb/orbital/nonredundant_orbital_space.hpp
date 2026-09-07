#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

class NonredundantOrbitalSpace {
public:
  struct StructuralDiagnostics {
    int packed_parameter_size = 0;
    int reduced_size = 0;
    int orbital_count = 0;
    int full_local_rank_orbital_count = 0;
    int codimension_one_orbital_count = 0;
    int incomplete_local_span_orbital_count = 0;
    int gauge_intersection_orbital_count = 0;
    int quotient_dimension_mismatch_orbital_count = 0;
    int total_gauge_rank = 0;
    int total_expected_quotient_dimension = 0;
    double minimum_relative_scaling_residual = 1.0;
    double maximum_relative_scaling_residual = 0.0;
  };

  struct ProjectionResult {
    Eigen::VectorXd reduced_gradient;
    Eigen::VectorXd packed_projected_gradient;
  };

  NonredundantOrbitalSpace(
      const OrbitalPreparationInput& orbital_preparation_input,
      const SparseOrbitalParameterView& parameter_view,
      const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbital_basis_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& physical_orbital_matrix,
      const Eigen::MatrixXd* ao_effective_h1e = nullptr,
      bool collect_structural_diagnostics = false);

  int reduced_size() const noexcept {
    return reduced_size_;
  }

  int n_blocks() const noexcept {
    return static_cast<int>(block_bases_.size());
  }

  StructuralDiagnostics structural_diagnostics() const noexcept;

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

  std::uint64_t rank_signature() const noexcept {
    return rank_signature_;
  }

  // Reduced-space algebra expects vectors in the accepted-point NROS chart:
  // packed vectors have `packed_parameter_size_` entries and reduced vectors
  // have `reduced_size_` entries. These routines validate dimensions and
  // finite values before applying the local tangent projectors.
  Eigen::VectorXd apply_inverse_reduced_curvature(
      const Eigen::VectorXd& reduced_vector) const;

  Eigen::VectorXd apply_reduced_curvature(
      const Eigen::VectorXd& reduced_vector) const;

  Eigen::VectorXd apply_inverse_reduced_block_preconditioner(
      const Eigen::VectorXd& reduced_vector) const;

  Eigen::VectorXd expand_step(
      const Eigen::VectorXd& reduced_step) const;

  // `expand_retract_input_tangent` returns a full sparse-orbital tangent table,
  // while `retract_step` applies that tangent and renormalizes each sparse
  // orbital in the AO-overlap metric used by the accepted point.
  Eigen::VectorXd expand_retract_input_tangent(
      const OrbitalPreparationInput& orbital_preparation_input,
      const Eigen::VectorXd& reduced_step) const;

  OrbitalPreparationInput retract_step(
      const OrbitalPreparationInput& orbital_preparation_input,
      const Eigen::VectorXd& reduced_step,
      double step_scale = 1.0) const;

private:
  struct OrbitalProjector {
    std::vector<int> flat_indices;
    std::vector<int> packed_indices;
    std::vector<int> block_rows;
    // Physical tangent space basis U_p: (local_size × local_reduced_dim).
    // Columns are whitened tangent directions on the local `x^T S x = rho^2`
    // manifold.  `U_p^T U_p = I` by construction.
    Eigen::MatrixXd tangent_basis;
    // Diagonal of U_p^T (F_p - eps_p S_p) U_p for preconditioning.
    Eigen::VectorXd curvature_diagonal;
    // Positive-definite local block approximation to
    // U_p^T (F_p - eps_p S_p) U_p used by the block preconditioner.
    Eigen::MatrixXd curvature_block;
    // Cached inverse action of curvature_block. The accepted-point block is
    // immutable, so factoring it again in every Krylov iteration is wasted
    // cubic work.
    Eigen::MatrixXd inverse_curvature_block;
    int local_parameter_size = 0;
    int local_gauge_rank = 0;
    int local_combined_rank = 0;
    int expected_quotient_dimension = 0;
    int gauge_intersection_dimension = 0;
    double relative_scaling_residual = 0.0;
    int local_reduced_offset = 0;
    int local_reduced_size = 0;
  };

  struct BlockBasis {
    int n_inactive = 0;
    int n_occupied = 0;
    int n_virtual = 0;
    std::vector<int> basis_function_indices;
    Eigen::MatrixXd block_overlap_matrix;
    std::vector<OrbitalProjector> orbitals;
  };

  ProjectionResult project_impl(
      const Eigen::VectorXd& packed_vector,
      bool recover_tangent_coordinates,
      bool build_packed_projection) const;

  std::vector<BlockBasis> block_bases_;
  int packed_parameter_size_ = 0;
  int reduced_size_ = 0;
  std::uint64_t rank_signature_ = 1469598103934665603ull;
  bool has_reduced_curvature_diagonal_ = false;
  bool use_block_preconditioner_by_default_ = false;
};

}  // namespace xmvb::vb
