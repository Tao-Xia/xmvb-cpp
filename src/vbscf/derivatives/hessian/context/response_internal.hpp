#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/structures/assembly/coefficient_blocks.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"

namespace xmvb::vb {

/**
 * @brief Directional selected-column structure matrices in the accepted basis.
 *
 * These are the accepted-eigenvector transformed directional Hamiltonian and
 * overlap columns restricted to the selected states used by the outer
 * response. They are the only directional inputs required by the frozen
 * accepted-point generalized-eigen response operator.
 */
struct SelectedStateProjectedDirectionalMatrices {
  Eigen::MatrixXd transformed_delta_hamiltonian_selected;
  Eigen::MatrixXd transformed_delta_overlap_selected;
};

struct SelectedStateGeneralizedEigenDirectionalResponse {
  Eigen::MatrixXd delta_selected_eigenvector_matrix;
  std::vector<double> delta_selected_eigenvalues;
};

/**
 * @brief Accepted-point metadata for one selected generalized-eigen column.
 *
 * The accepted-point eigensystem fixes the energy gaps and gauge policy used
 * by the first-order generalized-eigen formulas. Only the directional
 * projected structure columns change between HVP applications.
 */
struct AcceptedSelectedStateEigenResponseColumnCache {
  int selected_state_index = -1;
  double selected_state_energy = 0.0;
  Eigen::VectorXd energy_gaps;
  Eigen::VectorXd gap_tolerances;
  Eigen::ArrayXi uses_equal_weight_gauge;
};

struct AcceptedSelectedStateGeneralizedEigenResponseOperator {
  const std::vector<double>* accepted_eigenvector_matrix_storage = nullptr;
  int n_structures = 0;
  Eigen::VectorXd accepted_eigenvalues;
  std::vector<AcceptedSelectedStateEigenResponseColumnCache> selected_columns;

  Eigen::Map<const Eigen::MatrixXd> accepted_eigenvector_matrix_view() const {
    if (accepted_eigenvector_matrix_storage == nullptr || n_structures <= 0) {
      throw std::runtime_error(
          "accepted selected-state eigen-response operator is missing its eigensystem storage");
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(n_structures) *
        static_cast<std::size_t>(n_structures);
    if (accepted_eigenvector_matrix_storage->size() != expected_size) {
      throw std::runtime_error(
          "accepted selected-state eigensystem storage dimensions are inconsistent");
    }
    return Eigen::Map<const Eigen::MatrixXd>(
        accepted_eigenvector_matrix_storage->data(),
        n_structures,
        n_structures);
  }

  SelectedStateGeneralizedEigenDirectionalResponse apply(
      const SelectedStateProjectedDirectionalMatrices&
          projected_directional_structure_matrices) const;
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
  const std::vector<StructureCoefficientBlock>* structure_coefficient_blocks =
      nullptr;
  AcceptedSelectedStateGeneralizedEigenResponseOperator
      selected_state_eigen_response_operator;
};

AcceptedOuterResponseContext build_accepted_outer_response_context(
    const VbScfInput* input,
    const AcceptedPointContext* accepted_point_context,
    const std::vector<StructureCoefficientBlock>* structure_coefficient_blocks);

}  // namespace xmvb::vb
