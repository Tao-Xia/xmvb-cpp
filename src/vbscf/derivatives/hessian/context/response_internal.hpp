#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"

namespace xmvb::vb {

struct AcceptedStructureResponseFactors;

/**
 * @brief Directional structure-matrix images of the selected states.
 *
 * The two matrices are `delta H C_sel` and `delta S C_sel` in the structure
 * basis. They depend only on selected accepted states, not on the complete
 * accepted eigensystem.
 */
struct SelectedStateDirectionalStructureImages {
  Eigen::MatrixXd delta_hamiltonian_selected;
  Eigen::MatrixXd delta_overlap_selected;
  std::optional<StructureIntegralDirection> direct_ci_direction;
};

struct SelectedStateGeneralizedEigenDirectionalResponse {
  Eigen::MatrixXd delta_selected_eigenvector_matrix;
  /** Full selected-space response for each orbital direction. */
  std::vector<Eigen::MatrixXd> selected_matrix_responses;
  std::vector<int> linear_iterations;
  int block_actions = 0;
  double max_relative_residual = 0.0;
};

struct AcceptedSelectedStateGeneralizedEigenResponseOperator {
  const StructureAction* structure_action = nullptr;
  Eigen::VectorXd selected_eigenvalues;
  Eigen::MatrixXd selected_eigenvectors;
  Eigen::MatrixXd overlap_selected;
  const std::vector<double>* full_eigenvalues = nullptr;
  const Eigen::MatrixXd* full_eigenvectors = nullptr;
  std::vector<int> selected_root_indices;
  /** Whether all selected states form one equal-weight invariant cluster. */
  bool use_equal_weight_subspace_response = false;
  double relative_residual_tolerance = 0.0;

  SelectedStateGeneralizedEigenDirectionalResponse apply(
      const SelectedStateDirectionalStructureImages& directional_images,
      double requested_relative_residual_tolerance = 0.0) const;

  /**
   * @brief Solves several directional responses as one state/direction block.
   *
   * Columns are ordered by direction, with all selected states contiguous
   * inside each direction.
   */
  SelectedStateGeneralizedEigenDirectionalResponse apply_direction_block(
      const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
      const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
      double requested_relative_residual_tolerance = 0.0) const;
};

/**
 * @brief Frozen accepted-point outer-response context reused across HVP applies.
 *
 * This object is the authoritative source of all base-point quantities used by
 * the outer response. Directional data are supplied separately for each HVP,
 * so a response cannot accidentally combine objects from different accepted
 * points.
 */
struct AcceptedOuterResponseContext {
  const VbScfInput* input = nullptr;
  const AcceptedPointContext* accepted_point_context = nullptr;
  AcceptedSelectedStateGeneralizedEigenResponseOperator
      selected_state_eigen_response_operator;
  std::shared_ptr<const AcceptedStructureResponseFactors> structure_factors;
};

AcceptedOuterResponseContext build_accepted_outer_response_context(
    const VbScfInput* input,
    const AcceptedPointContext* accepted_point_context);

}  // namespace xmvb::vb
