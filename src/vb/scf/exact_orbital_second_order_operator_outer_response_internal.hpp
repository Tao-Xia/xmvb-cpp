#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"

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
 * @brief Frozen accepted-point outer-response data reused across HVP applies.
 *
 * The current implementation still rebuilds directional projected structure
 * columns for each direction, but all accepted-point eigensystem metadata and
 * selected-state bookkeeping are cached once here and reused.
 */
struct AcceptedOuterResponseLinearResponseCache {
  const CppVbInput* input = nullptr;
  const CppActiveSpaceSecondOrderContext* accepted_point_context = nullptr;
  const std::vector<StructureCoefficientBlock>* structure_coefficient_blocks =
      nullptr;
  std::vector<int> selected_state_indices;
  Eigen::MatrixXd accepted_selected_eigenvector_columns;
  AcceptedSelectedStateGeneralizedEigenResponseOperator
      selected_state_eigen_response_operator;

  SelectedStateProjectedDirectionalMatrices
  build_projected_directional_structure_matrices(
      const std::vector<double>& delta_active_orbital_overlap_matrix,
      const std::vector<double>& delta_active_one_electron_matrix,
      const std::vector<double>& delta_packed_active_two_electron_integrals)
      const;
};

AcceptedOuterResponseLinearResponseCache
build_accepted_outer_response_linear_response_cache(
    const CppVbInput* input,
    const CppActiveSpaceSecondOrderContext* accepted_point_context,
    const std::vector<StructureCoefficientBlock>* structure_coefficient_blocks);

}  // namespace xmvb::vb
