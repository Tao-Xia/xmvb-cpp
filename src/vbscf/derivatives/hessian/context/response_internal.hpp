#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "core/eigen_response.hpp"
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
  /** Top-block bordered residual for each direction. */
  std::vector<Eigen::MatrixXd> equation_residuals;
  std::vector<int> linear_iterations;
  int block_actions = 0;
  double max_relative_residual = 0.0;
};

struct AcceptedSelectedStateGeneralizedEigenResponseOperator {
  const StructureAction* structure_action = nullptr;
  Eigen::VectorXd selected_eigenvalues;
  Eigen::MatrixXd selected_eigenvectors;
  Eigen::MatrixXd overlap_selected;
  /** @brief Accepted Ritz residuals H C - S C E, retained without extra actions. */
  Eigen::MatrixXd selected_residuals;
  const std::vector<double>* full_eigenvalues = nullptr;
  const Eigen::MatrixXd* full_eigenvectors = nullptr;
  std::vector<int> selected_root_indices;
  /** Whether all selected states form one equal-weight invariant cluster. */
  bool use_equal_weight_subspace_response = false;
  double relative_residual_tolerance = 0.0;
  /** Same accepted-point response spaces, one for each selected root. */
  mutable std::vector<xmvb::core::EigenResponseRecycleSpace>
      response_recycle_spaces;

  /** @brief Revision of the common accepted-point response spaces. */
  std::uint64_t revision() const noexcept;

  SelectedStateGeneralizedEigenDirectionalResponse apply(
      const SelectedStateDirectionalStructureImages& directional_images,
      double requested_relative_residual_tolerance = 0.0,
      bool frozen = false) const;

  /**
   * @brief Solves several directional responses as one state/direction block.
   *
   * Enrich all directional right-hand sides before evaluating their common
   * Galerkin model. Frozen applications never modify that model; their true
   * residual is reported, not used to silently change the operator.
   * Columns are ordered by direction, with selected states contiguous.
   */
  SelectedStateGeneralizedEigenDirectionalResponse apply_direction_block(
      const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
      const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
      double requested_relative_residual_tolerance = 0.0,
      bool frozen = false) const;
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
