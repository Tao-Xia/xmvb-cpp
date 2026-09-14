#include "vbscf/orbitals/charts/canonicalization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

void overwrite_sparse_orbitals_from_dense_physical_frame(
    const Eigen::MatrixXd& dense_orbitals,
    OrbitalPreparationInput* orbital_preparation_input) {
  std::vector<double> updated_orbital_values =
      orbital_preparation_input->orbital_value_table;
  const int n_basis_functions = orbital_preparation_input->n_basis_functions;
  for (int orbital_index = 0;
       orbital_index < orbital_preparation_input->n_orbitals;
       ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            *orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input->orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
              1;
      if (basis_function_index < 0 ||
          basis_function_index >= orbital_preparation_input->n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while overwriting the final physical frame");
      }
      updated_orbital_values[orbital_index * n_basis_functions +
                             coefficient_index] =
          dense_orbitals(basis_function_index, orbital_index);
    }
  }
  orbital_preparation_input->orbital_value_table = std::move(updated_orbital_values);
}

Eigen::MatrixXd build_self_adjoint_matrix_power(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double exponent,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  if (matrix.rows() == 0) {
    return Eigen::MatrixXd::Zero(0, 0);
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(matrix);
  if (eigen_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        std::string("failed eigendecomposition for ") + label);
  }

  Eigen::VectorXd powered_eigenvalues(matrix.rows());
  for (Eigen::Index index = 0; index < matrix.rows(); ++index) {
    const double eigenvalue = eigen_solver.eigenvalues()[index];
    if (!std::isfinite(eigenvalue) ||
        eigenvalue <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          std::string(label) + " is not numerically positive definite");
    }
    powered_eigenvalues[index] = std::pow(eigenvalue, exponent);
  }

  return eigen_solver.eigenvectors() *
      powered_eigenvalues.asDiagonal() *
      eigen_solver.eigenvectors().transpose();
}

Eigen::MatrixXd build_inactive_metric_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix) {
  if (inactive_physical_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(0, 0);
  }

  // Dimensions:
  // `inactive_physical_orbitals` is `(n_basis, n_inactive)` and
  // `basis_overlap_matrix` is `(n_basis, n_basis)`. The resulting metric is
  // the small occupied-space overlap `(n_inactive, n_inactive)`.
  const Eigen::MatrixXd inactive_metric =
      inactive_physical_orbitals.transpose() *
      basis_overlap_matrix *
      inactive_physical_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> inactive_metric_ldlt(inactive_metric);
  if (inactive_metric_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor inactive occupied overlap while repairing OEO representative");
  }

  const Eigen::MatrixXd inactive_metric_inverse =
      inactive_metric_ldlt.solve(
          Eigen::MatrixXd::Identity(
              inactive_metric.rows(),
              inactive_metric.cols()));
  if (inactive_metric_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to invert inactive occupied overlap while repairing OEO representative");
  }
  return inactive_metric_inverse;
}

Eigen::MatrixXd build_metric_preserving_inactive_repaired_active_physical_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix) {
  if (current_active_auxiliary.cols() != current_active_physical_orbitals.cols() ||
      current_active_auxiliary.cols() != reference_active_physical_orbitals.cols() ||
      current_active_auxiliary.rows() != inactive_physical_orbitals.rows() ||
      current_active_physical_orbitals.rows() != inactive_physical_orbitals.rows() ||
      reference_active_physical_orbitals.rows() != inactive_physical_orbitals.rows() ||
      basis_overlap_matrix.rows() != inactive_physical_orbitals.rows() ||
      basis_overlap_matrix.cols() != inactive_physical_orbitals.rows()) {
    throw std::invalid_argument(
        "metric-preserving inactive representative repair has inconsistent dimensions");
  }
  if (current_active_auxiliary.cols() == 0) {
    return Eigen::MatrixXd::Zero(current_active_auxiliary.rows(), 0);
  }
  if (inactive_physical_orbitals.cols() == 0) {
    return current_active_physical_orbitals;
  }

  const Eigen::MatrixXd inactive_metric_inverse =
      build_inactive_metric_inverse(
          inactive_physical_orbitals,
          basis_overlap_matrix);
  const Eigen::MatrixXd inactive_dual_orbitals =
      inactive_physical_orbitals * inactive_metric_inverse;
  Eigen::MatrixXd repaired_active_physical_orbitals =
      current_active_physical_orbitals;

  // `orbtyp=oeo` stores the physical active orbitals `C_a`, while the energy
  // depends on the projected auxiliaries `T_a = (I - P_i S) C_a`.  Adding an
  // inactive component `C_i K` leaves `T_a` unchanged because
  // `(I - P_i S) C_i = 0`.  The representative reset below therefore keeps the
  // current projected active orbitals and their metric norm exactly fixed, and
  // only refreshes the inactive coefficients so the physical occupied chart
  // stays close to the initial localized reference instead of drifting along
  // the inactive-null gauge.
  for (int active_index = 0;
       active_index < current_active_auxiliary.cols();
       ++active_index) {
    const Eigen::VectorXd current_auxiliary =
        current_active_auxiliary.col(active_index);
    const Eigen::VectorXd current_physical =
        current_active_physical_orbitals.col(active_index);
    const Eigen::VectorXd reference_physical =
        reference_active_physical_orbitals.col(active_index);

    const double auxiliary_norm_squared =
        current_auxiliary.transpose() *
        basis_overlap_matrix *
        current_auxiliary;
    if (!std::isfinite(auxiliary_norm_squared) ||
        auxiliary_norm_squared <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "encountered non-positive active auxiliary norm while resetting the OEO chart");
    }

    const double target_inactive_metric_norm_squared =
        std::max(0.0, 1.0 - auxiliary_norm_squared);
    if (target_inactive_metric_norm_squared <=
        64.0 * std::numeric_limits<double>::epsilon()) {
      repaired_active_physical_orbitals.col(active_index) = current_auxiliary;
      continue;
    }

    const Eigen::VectorXd current_inactive_coefficients =
        inactive_dual_orbitals.transpose() *
        basis_overlap_matrix *
        current_physical;
    const Eigen::VectorXd reference_direction =
        inactive_dual_orbitals.transpose() *
        basis_overlap_matrix *
        (reference_physical - current_auxiliary);
    const double reference_direction_metric_squared =
        reference_direction.dot(inactive_metric_inverse * reference_direction);

    Eigen::VectorXd repaired_inactive_coefficients =
        current_inactive_coefficients;
    if (std::isfinite(reference_direction_metric_squared) &&
        reference_direction_metric_squared >
            64.0 * std::numeric_limits<double>::epsilon()) {
      repaired_inactive_coefficients =
          std::sqrt(
              target_inactive_metric_norm_squared /
              reference_direction_metric_squared) *
          (inactive_metric_inverse * reference_direction);
    }

    Eigen::VectorXd repaired_orbital =
        current_auxiliary +
        inactive_physical_orbitals * repaired_inactive_coefficients;
    const double repaired_norm_squared =
        repaired_orbital.transpose() *
        basis_overlap_matrix *
        repaired_orbital;
    if (!std::isfinite(repaired_norm_squared) ||
        std::abs(repaired_norm_squared - 1.0) > 1.0e-8) {
      throw std::runtime_error(
          "metric-preserving OEO active representative reset changed the physical orbital norm");
    }
    repaired_active_physical_orbitals.col(active_index) = repaired_orbital;
  }

  return repaired_active_physical_orbitals;
}

Eigen::MatrixXd build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalPreparationResult& orbital_result,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_normalized_orbital_matrix) {
  const auto& physical_orbital_frame =
      orbital_result.physical_orbital_frame;
  if (reference_normalized_orbital_matrix.size() == 0 ||
      orbital_preparation_input.orbital_type != kOrbitalTypeOeo) {
    return physical_orbital_frame.normalized_orbital_matrix;
  }

  const int n_basis_functions =
      orbital_preparation_input.n_basis_functions;
  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      orbital_preparation_input.n_active_orbitals;
  if (n_inactive_doubly_occupied_orbitals <= 0 || n_active_orbitals <= 0) {
    return physical_orbital_frame.normalized_orbital_matrix;
  }

  if (reference_normalized_orbital_matrix.rows() != n_basis_functions ||
      reference_normalized_orbital_matrix.cols() !=
          orbital_preparation_input.n_orbitals ||
      physical_orbital_frame.normalized_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.normalized_orbital_matrix.cols() !=
          orbital_preparation_input.n_orbitals ||
      physical_orbital_frame.inactive_physical_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.inactive_physical_orbital_matrix.cols() !=
          n_inactive_doubly_occupied_orbitals ||
      physical_orbital_frame.active_physical_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.active_physical_orbital_matrix.cols() !=
          n_active_orbitals ||
      orbital_result.auxiliary_orbital_matrix.rows() != n_basis_functions ||
      orbital_result.auxiliary_orbital_matrix.cols() <
          n_inactive_doubly_occupied_orbitals + n_active_orbitals) {
    throw std::runtime_error(
        "cached OEO orbital preparation result is incomplete while repairing the output representative");
  }

  // Dimensions:
  // `reference_normalized_orbital_matrix` and the cached physical frame are
  // `(n_basis, n_orbitals)`, while the repaired active block is
  // `(n_basis, n_active)`. Only the occupied active columns change; inactive
  // and virtual orbitals are copied through unchanged for final export.
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
      orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::MatrixXd repaired_active_physical_orbitals =
      build_metric_preserving_inactive_repaired_active_physical_orbitals(
          physical_orbital_frame.inactive_physical_orbital_matrix,
          orbital_result.auxiliary_orbital_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals),
          physical_orbital_frame.active_physical_orbital_matrix,
          reference_normalized_orbital_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals),
          basis_overlap_matrix);

  Eigen::MatrixXd repaired_normalized_orbital_matrix =
      physical_orbital_frame.normalized_orbital_matrix;
  repaired_normalized_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals) = repaired_active_physical_orbitals;
  return repaired_normalized_orbital_matrix;
}

}  // namespace xmvb::vb
