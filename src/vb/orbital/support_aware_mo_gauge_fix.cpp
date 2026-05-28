#include "vb/orbital/support_aware_mo_gauge_fix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

namespace {


Eigen::MatrixXd build_dense_sparse_orbital_columns(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_count) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  Eigen::MatrixXd dense_orbitals =
      Eigen::MatrixXd::Zero(n_basis_functions, std::max(0, orbital_count));
  for (int orbital_index = 0; orbital_index < orbital_count; ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
          1;
      if (basis_function_index < 0 ||
          basis_function_index >= n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while building dense MO gauge columns");
      }
      dense_orbitals(basis_function_index, orbital_index) =
          orbital_preparation_input.orbital_value_table
              [orbital_index * n_basis_functions +
               coefficient_index];
    }
  }
  return dense_orbitals;
}

Eigen::MatrixXd build_dense_sparse_orbital_columns_from_full_vector(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& full_vector,
    int orbital_count) {
  if (full_vector.size() != orbital_preparation_input.orbital_value_table.size()) {
    throw std::invalid_argument(
        "full sparse vector size does not match orbital_value_table");
  }
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  Eigen::MatrixXd dense_orbitals =
      Eigen::MatrixXd::Zero(n_basis_functions, std::max(0, orbital_count));
  for (int orbital_index = 0; orbital_index < orbital_count; ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
          1;
      if (basis_function_index < 0 ||
          basis_function_index >= n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while building dense MO gauge gradient columns");
      }
      dense_orbitals(basis_function_index, orbital_index) =
          full_vector[orbital_index * n_basis_functions +
                      coefficient_index];
    }
  }
  return dense_orbitals;
}

void scatter_dense_sparse_orbital_columns(
    const Eigen::MatrixXd& dense_orbitals,
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_count,
    std::vector<double>* full_vector) {
  if (full_vector == nullptr) {
    throw std::invalid_argument("full_vector must not be null");
  }
  if (full_vector->size() != orbital_preparation_input.orbital_value_table.size()) {
    throw std::invalid_argument(
        "full sparse vector size does not match orbital_value_table");
  }
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  for (int orbital_index = 0; orbital_index < orbital_count; ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
          1;
      (*full_vector)[orbital_index * n_basis_functions +
                     coefficient_index] =
          dense_orbitals(basis_function_index, orbital_index);
    }
  }
}

std::vector<int> collect_support_indices_from_layout(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int coefficient_count =
      stored_sparse_orbital_coefficient_count(
          orbital_preparation_input,
          orbital_index);
  std::vector<int> support_indices;
  support_indices.reserve(coefficient_count);
  for (int coefficient_index = 0;
       coefficient_index < coefficient_count;
       ++coefficient_index) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [orbital_index *
                 orbital_preparation_input.n_basis_functions +
             coefficient_index] -
        1;
    if (basis_function_index < 0 ||
        basis_function_index >= orbital_preparation_input.n_basis_functions) {
      throw std::runtime_error(
          "invalid sparse orbital basis index while collecting MO gauge supports");
    }
    support_indices.push_back(basis_function_index);
  }
  return support_indices;
}

bool support_sets_match(
    const std::vector<int>& left_support,
    const std::vector<int>& right_support) {
  if (left_support.size() != right_support.size()) {
    return false;
  }
  std::vector<int> left_sorted = left_support;
  std::vector<int> right_sorted = right_support;
  std::sort(left_sorted.begin(), left_sorted.end());
  std::sort(right_sorted.begin(), right_sorted.end());
  return left_sorted == right_sorted;
}

bool inactive_support_layout_matches_reference(
    const OrbitalPreparationInput& reference_layout,
    const OrbitalPreparationInput& current_layout,
    int n_inactive_orbitals) {
  for (int orbital_index = 0;
       orbital_index < n_inactive_orbitals;
       ++orbital_index) {
    if (!support_sets_match(
            collect_support_indices_from_layout(reference_layout, orbital_index),
            collect_support_indices_from_layout(current_layout, orbital_index))) {
      return false;
    }
  }
  return true;
}

Eigen::MatrixXd build_inverse_square_root_metric(
    const Eigen::MatrixXd& metric,
    const char* label) {
  if (metric.rows() != metric.cols()) {
    throw std::invalid_argument(
        std::string(label) + " must be square for inverse square root");
  }
  if (metric.rows() == 0) {
    return Eigen::MatrixXd(0, 0);
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metric);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed to diagonalize ") + label);
  }

  Eigen::MatrixXd inverse_square_root =
      Eigen::MatrixXd::Zero(metric.rows(), metric.cols());
  constexpr double kMinimumEigenvalue = 1.0e-10;
  for (int eigen_index = 0; eigen_index < solver.eigenvalues().size(); ++eigen_index) {
    const double eigenvalue = solver.eigenvalues()[eigen_index];
    if (!(eigenvalue > kMinimumEigenvalue) || !std::isfinite(eigenvalue)) {
      throw std::runtime_error(
          std::string(label) +
          " is singular while building support-aware MO gauge");
    }
    inverse_square_root(eigen_index, eigen_index) =
        1.0 / std::sqrt(eigenvalue);
  }
  return solver.eigenvectors() *
      inverse_square_root *
      solver.eigenvectors().transpose();
}

Eigen::MatrixXd build_support_overlap_submatrix(
    const Eigen::MatrixXd& basis_overlap,
    const std::vector<int>& support_indices) {
  Eigen::MatrixXd support_overlap(
      static_cast<Eigen::Index>(support_indices.size()),
      static_cast<Eigen::Index>(support_indices.size()));
  for (Eigen::Index row = 0;
       row < static_cast<Eigen::Index>(support_indices.size());
       ++row) {
    const int basis_row = support_indices[row];
    for (Eigen::Index column = 0;
         column < static_cast<Eigen::Index>(support_indices.size());
         ++column) {
      support_overlap(row, column) =
          basis_overlap(
              basis_row,
              support_indices[column]);
    }
  }
  return support_overlap;
}

bool dense_matrix_is_effectively_identity(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols()) {
    return false;
  }
  constexpr double kIdentityTolerance = 1.0e-10;
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      const double target = row == column ? 1.0 : 0.0;
      if (std::abs(matrix(row, column) - target) > kIdentityTolerance) {
        return false;
      }
    }
  }
  return true;
}

SupportAwareInactiveMoGaugeTransform finalize_transform(
    const Eigen::MatrixXd& right_transform) {
  SupportAwareInactiveMoGaugeTransform result;
  result.n_inactive_orbitals = static_cast<int>(right_transform.cols());
  result.chart_changed = !dense_matrix_is_effectively_identity(right_transform);
  result.right_transform.assign(
      right_transform.data(),
      right_transform.data() + right_transform.size());

  const Eigen::FullPivLU<Eigen::MatrixXd> transform_lu(right_transform);
  if (!transform_lu.isInvertible()) {
    throw std::runtime_error(
        "support-aware inactive MO gauge transform is singular");
  }
  const Eigen::MatrixXd inverse_transpose =
      transform_lu.inverse().transpose();
  result.inverse_transpose_right_transform.assign(
      inverse_transpose.data(),
      inverse_transpose.data() + inverse_transpose.size());
  return result;
}

}  // namespace

bool orbital_input_has_support_aware_mo_gauge_reference(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return orbital_preparation_input.mo_gauge_reference_orbital_basis_counts.size() ==
          orbital_preparation_input.n_orbitals &&
      orbital_preparation_input.mo_gauge_reference_orbital_basis_index_table.size() ==
          orbital_preparation_input.n_orbitals *
              orbital_preparation_input.n_basis_functions;
}

SupportAwareInactiveMoGaugeTransform apply_support_aware_inactive_mo_gauge_fix(
    const OrbitalPreparationInput& reference_layout,
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument(
        "orbital_preparation_input must not be null for MO gauge fix");
  }

  const int n_inactive_orbitals =
      (orbital_preparation_input->n_total_electrons -
       orbital_preparation_input->n_active_electrons) / 2;
  if (n_inactive_orbitals <= 0) {
    return {};
  }
  if (reference_layout.n_basis_functions !=
          orbital_preparation_input->n_basis_functions ||
      reference_layout.n_orbitals != orbital_preparation_input->n_orbitals) {
    throw std::invalid_argument(
        "reference layout is incompatible with the MO gauge-fix target");
  }
  if (n_inactive_orbitals == 1) {
    return {};
  }
  // A support-aware re-gauging is only an exact chart change when the current
  // sparse layout already differs from the recorded legacy support chart, e.g.
  // after a temporary support expansion used only for guess construction. If
  // the current chart already equals the reference sparse layout, the dense
  // right rotation computed below would be truncated back onto the same sparse
  // rows and the stored `T^{-T}` gradient transform would no longer match the
  // actual coefficient update. In that case the safe behavior is to leave the
  // legacy chart untouched.
  if (inactive_support_layout_matches_reference(
          reference_layout,
          *orbital_preparation_input,
          n_inactive_orbitals)) {
    return {};
  }

  const int n_basis_functions = orbital_preparation_input->n_basis_functions;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      orbital_preparation_input->ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  // The inactive occupied span is physically meaningful, but the individual
  // orbitals inside that span are gauge degrees of freedom. Convert the
  // current sparse inactive rows into dense AO columns, S-orthonormalize that
  // span, then greedily pick one vector at a time that maximizes weight on the
  // original sparse support of the corresponding inactive orbital.
  const Eigen::MatrixXd current_inactive_orbitals =
      build_dense_sparse_orbital_columns(
          *orbital_preparation_input,
          n_inactive_orbitals);
  const Eigen::MatrixXd inactive_metric =
      current_inactive_orbitals.transpose() *
      basis_overlap *
      current_inactive_orbitals;
  Eigen::MatrixXd remaining_basis =
      current_inactive_orbitals *
      build_inverse_square_root_metric(
          inactive_metric,
          "inactive occupied overlap");
  Eigen::MatrixXd remaining_transform =
      build_inverse_square_root_metric(
          inactive_metric,
          "inactive occupied overlap");
  Eigen::MatrixXd localized_inactive_orbitals =
      Eigen::MatrixXd::Zero(n_basis_functions, n_inactive_orbitals);
  Eigen::MatrixXd inactive_right_transform =
      Eigen::MatrixXd::Zero(n_inactive_orbitals, n_inactive_orbitals);

  for (int target_orbital = 0;
       target_orbital < n_inactive_orbitals;
       ++target_orbital) {
    const std::vector<int> target_support =
        collect_support_indices_from_layout(
            reference_layout,
            target_orbital);
    if (target_support.empty()) {
      throw std::runtime_error(
          "encountered an empty inactive support while fixing the MO gauge");
    }

    const int remaining_dimension =
        static_cast<int>(remaining_basis.cols());
    if (remaining_dimension <= 0) {
      throw std::runtime_error(
          "inactive occupied subspace exhausted during MO gauge fix");
    }
    if (remaining_dimension == 1) {
      localized_inactive_orbitals.col(target_orbital) =
          remaining_basis.col(0);
      // At the final deflation step the remaining one-dimensional subspace is
      // already the exact last gauge vector. Its coefficient in the original
      // inactive span is the surviving column of `remaining_transform`, so the
      // right transform must keep that last column as well. Dropping it makes
      // the accumulated `C_new = C_old T` chart transform singular.
      inactive_right_transform.col(target_orbital) =
          remaining_transform.col(0);
      remaining_basis = Eigen::MatrixXd(n_basis_functions, 0);
      remaining_transform = Eigen::MatrixXd(n_inactive_orbitals, 0);
      continue;
    }

    Eigen::MatrixXd support_restricted_basis(
        static_cast<Eigen::Index>(target_support.size()),
        remaining_basis.cols());
    for (Eigen::Index row = 0;
         row < static_cast<Eigen::Index>(target_support.size());
         ++row) {
      support_restricted_basis.row(row) =
          remaining_basis.row(target_support[row]);
    }
    const Eigen::MatrixXd support_weight =
        support_restricted_basis.transpose() *
        build_support_overlap_submatrix(basis_overlap, target_support) *
        support_restricted_basis;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> support_solver(
        support_weight);
    if (support_solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed to diagonalize the support-weight matrix for MO gauge fix");
    }

    localized_inactive_orbitals.col(target_orbital) =
        remaining_basis *
        support_solver.eigenvectors().col(remaining_dimension - 1);
    inactive_right_transform.col(target_orbital) =
        remaining_transform *
        support_solver.eigenvectors().col(remaining_dimension - 1);
    remaining_basis =
        remaining_basis *
        support_solver.eigenvectors().leftCols(remaining_dimension - 1);
    remaining_transform =
        remaining_transform *
        support_solver.eigenvectors().leftCols(remaining_dimension - 1);
  }

  std::vector<double> updated_orbital_values =
      orbital_preparation_input->orbital_value_table;
  for (int orbital_index = 0;
       orbital_index < n_inactive_orbitals;
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
      updated_orbital_values[orbital_index *
                                 n_basis_functions +
                             coefficient_index] =
          localized_inactive_orbitals(
              basis_function_index,
              orbital_index);
    }
  }
  orbital_preparation_input->orbital_value_table =
      std::move(updated_orbital_values);
  enforce_strict_sparse_orbital_support(orbital_preparation_input);
  return finalize_transform(inactive_right_transform);
}

SupportAwareInactiveMoGaugeTransform apply_support_aware_inactive_mo_gauge_fix(
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr ||
      !orbital_input_has_support_aware_mo_gauge_reference(
          *orbital_preparation_input)) {
    return {};
  }

  OrbitalPreparationInput reference_layout;
  reference_layout.n_basis_functions =
      orbital_preparation_input->n_basis_functions;
  reference_layout.n_orbitals =
      orbital_preparation_input->n_orbitals;
  reference_layout.n_active_orbitals =
      orbital_preparation_input->n_active_orbitals;
  reference_layout.n_total_electrons =
      orbital_preparation_input->n_total_electrons;
  reference_layout.n_active_electrons =
      orbital_preparation_input->n_active_electrons;
  reference_layout.spin_multiplicity =
      orbital_preparation_input->spin_multiplicity;
  reference_layout.orbital_basis_counts =
      orbital_preparation_input->mo_gauge_reference_orbital_basis_counts;
  reference_layout.orbital_basis_index_table =
      orbital_preparation_input->mo_gauge_reference_orbital_basis_index_table;
  return apply_support_aware_inactive_mo_gauge_fix(
      reference_layout,
      orbital_preparation_input);
}

void transform_sparse_inactive_orbital_gradient(
    const SupportAwareInactiveMoGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_gradient) {
  if (sparse_orbital_gradient == nullptr) {
    throw std::invalid_argument("sparse_orbital_gradient must not be null");
  }
  if (!transform.chart_changed || transform.n_inactive_orbitals <= 1) {
    return;
  }
  const int n_inactive_orbitals = transform.n_inactive_orbitals;
  const Eigen::Map<const Eigen::MatrixXd> inverse_transpose_transform(
      transform.inverse_transpose_right_transform.data(),
      n_inactive_orbitals,
      n_inactive_orbitals);
  const Eigen::MatrixXd inactive_gradient =
      build_dense_sparse_orbital_columns_from_full_vector(
          orbital_preparation_input,
          *sparse_orbital_gradient,
          n_inactive_orbitals);
  const Eigen::MatrixXd transformed_gradient =
      inactive_gradient * inverse_transpose_transform;
  scatter_dense_sparse_orbital_columns(
      transformed_gradient,
      orbital_preparation_input,
      n_inactive_orbitals,
      sparse_orbital_gradient);
}

}  // namespace xmvb::vb
