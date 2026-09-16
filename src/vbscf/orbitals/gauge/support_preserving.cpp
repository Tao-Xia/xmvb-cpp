#include "vbscf/orbitals/gauge/support_preserving.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

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
    const std::vector<int>& left,
    const std::vector<int>& right) {
  if (left.size() != right.size()) return false;
  std::vector<int> sorted_left = left;
  std::vector<int> sorted_right = right;
  std::sort(sorted_left.begin(), sorted_left.end());
  std::sort(sorted_right.begin(), sorted_right.end());
  return sorted_left == sorted_right;
}

bool inactive_supports_match(
    const OrbitalPreparationInput& reference_layout,
    const OrbitalPreparationInput& current_layout,
    int n_inactive) {
  for (int orbital = 0; orbital < n_inactive; ++orbital) {
    if (!support_sets_match(
            collect_support_indices_from_layout(reference_layout, orbital),
            collect_support_indices_from_layout(current_layout, orbital))) {
      return false;
    }
  }
  return true;
}

bool inactive_metric_loses_half_precision(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap) {
  const Eigen::MatrixXd metric =
      inactive_orbitals.transpose() * ao_overlap * inactive_orbitals;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metric);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !(solver.eigenvalues().minCoeff() > 0.0)) {
    throw std::runtime_error(
        "inactive occupied overlap is not positive definite");
  }
  const double relative_roundoff_bound =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(metric.rows()) *
      solver.eigenvalues().maxCoeff() /
      solver.eigenvalues().minCoeff();
  return relative_roundoff_bound >
      std::sqrt(std::numeric_limits<double>::epsilon());
}

Eigen::MatrixXd build_admissible_gauge_basis(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const std::vector<int>& target_support) {
  const int n_bf = static_cast<int>(inactive_orbitals.rows());
  const int n_inactive = static_cast<int>(inactive_orbitals.cols());
  std::vector<unsigned char> allowed(n_bf, 0);
  for (const int basis : target_support) {
    if (basis < 0 || basis >= n_bf || allowed[basis] != 0) {
      throw std::runtime_error(
          "invalid target support while balancing the inactive gauge");
    }
    allowed[basis] = 1;
  }

  const int forbidden_count =
      n_bf - static_cast<int>(target_support.size());
  if (forbidden_count == 0) {
    return Eigen::MatrixXd::Identity(n_inactive, n_inactive);
  }
  Eigen::MatrixXd constraints(forbidden_count, n_inactive);
  int row = 0;
  for (int basis = 0; basis < n_bf; ++basis) {
    if (allowed[basis] == 0) {
      constraints.row(row++) = inactive_orbitals.row(basis);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      constraints, Eigen::ComputeFullV);
  if (svd.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor support constraints for inactive gauge balancing");
  }
  const double sigma_max = svd.singularValues().size() == 0
      ? 0.0
      : svd.singularValues()[0];
  const double rank_tolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(constraints.rows(), constraints.cols())) *
      sigma_max;
  int rank = 0;
  while (rank < svd.singularValues().size() &&
         svd.singularValues()[rank] > rank_tolerance) {
    ++rank;
  }
  Eigen::MatrixXd basis =
      svd.matrixV().rightCols(n_inactive - rank);
  return basis;
}

Eigen::MatrixXd build_balanced_inactive_right_transform(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap,
    const OrbitalPreparationInput& target_layout) {
  const int n_inactive = static_cast<int>(inactive_orbitals.cols());
  const Eigen::MatrixXd metric =
      inactive_orbitals.transpose() * ao_overlap * inactive_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> metric_factor(metric);
  if (metric_factor.info() != Eigen::Success ||
      !metric_factor.isPositive()) {
    throw std::runtime_error(
        "inactive occupied overlap is not positive definite");
  }

  std::vector<Eigen::MatrixXd> admissible_bases;
  admissible_bases.reserve(n_inactive);
  for (int orbital = 0; orbital < n_inactive; ++orbital) {
    admissible_bases.push_back(build_admissible_gauge_basis(
        inactive_orbitals,
        collect_support_indices_from_layout(target_layout, orbital)));
    if (admissible_bases.back().cols() == 0) {
      throw std::runtime_error(
          "inactive orbital has no support-preserving gauge representative");
    }
  }

  Eigen::MatrixXd transform =
      Eigen::MatrixXd::Identity(n_inactive, n_inactive);
  for (int orbital = 0; orbital < n_inactive; ++orbital) {
    const double norm_squared = metric(orbital, orbital);
    if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
      throw std::runtime_error(
          "inactive gauge contains a zero-norm orbital");
    }
    transform.col(orbital) /= std::sqrt(norm_squared);
  }

  const int maximum_sweeps = std::max(1, 4 * n_inactive);
  const double convergence_tolerance =
      8.0 * std::sqrt(std::numeric_limits<double>::epsilon()) *
      std::sqrt(static_cast<double>(n_inactive));
  for (int sweep = 0; sweep < maximum_sweeps; ++sweep) {
    double maximum_change = 0.0;
    for (int target = 0; target < n_inactive; ++target) {
      Eigen::MatrixXd other_transform(n_inactive, n_inactive - 1);
      if (target > 0) {
        other_transform.leftCols(target) = transform.leftCols(target);
      }
      if (target + 1 < n_inactive) {
        other_transform.rightCols(n_inactive - target - 1) =
            transform.rightCols(n_inactive - target - 1);
      }

      Eigen::MatrixXd residual_metric = metric;
      if (other_transform.cols() > 0) {
        const Eigen::MatrixXd cross = metric * other_transform;
        const Eigen::MatrixXd other_metric =
            other_transform.transpose() * cross;
        Eigen::LDLT<Eigen::MatrixXd> other_factor(other_metric);
        if (other_factor.info() != Eigen::Success ||
            !other_factor.isPositive()) {
          throw std::runtime_error(
              "inactive gauge transform lost rank while balancing supports");
        }
        residual_metric.noalias() -=
            cross * other_factor.solve(cross.transpose());
      }
      residual_metric =
          0.5 * (residual_metric + residual_metric.transpose()).eval();

      const Eigen::MatrixXd& admissible = admissible_bases[target];
      const Eigen::MatrixXd restricted_metric =
          admissible.transpose() * metric * admissible;
      const Eigen::MatrixXd restricted_residual =
          admissible.transpose() * residual_metric * admissible;
      Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
          restricted_residual,
          restricted_metric);
      if (solver.info() != Eigen::Success ||
          !solver.eigenvalues().allFinite() ||
          !solver.eigenvectors().allFinite()) {
        throw std::runtime_error(
            "failed to solve the support-preserving inactive gauge update");
      }

      Eigen::VectorXd updated =
          admissible * solver.eigenvectors().rightCols(1);
      const double updated_norm_squared = updated.dot(metric * updated);
      if (!(updated_norm_squared > 0.0) ||
          !std::isfinite(updated_norm_squared)) {
        throw std::runtime_error(
            "support-preserving inactive gauge update has zero norm");
      }
      updated /= std::sqrt(updated_norm_squared);
      const Eigen::VectorXd previous = transform.col(target);
      if (updated.dot(metric * previous) < 0.0) {
        updated = -updated;
      }
      maximum_change = std::max(
          maximum_change,
          std::sqrt((updated - previous).dot(metric * (updated - previous))));
      transform.col(target) = updated;
    }
    if (maximum_change <= convergence_tolerance) {
      break;
    }
  }
  return transform;
}

bool dense_matrix_is_effectively_identity(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols()) {
    return false;
  }
  const double identity_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(matrix.rows(), matrix.cols()));
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      const double target = row == column ? 1.0 : 0.0;
      if (std::abs(matrix(row, column) - target) > identity_tolerance) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool orbital_input_has_support_preserving_gauge_reference(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return orbital_preparation_input.mo_gauge_reference_orbital_basis_counts.size() ==
          orbital_preparation_input.n_orbitals &&
      orbital_preparation_input.mo_gauge_reference_orbital_basis_index_table.size() ==
          orbital_preparation_input.n_orbitals *
              orbital_preparation_input.n_basis_functions;
}

bool apply_support_preserving_inactive_gauge(
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
    return false;
  }
  if (reference_layout.n_basis_functions !=
          orbital_preparation_input->n_basis_functions ||
      reference_layout.n_orbitals != orbital_preparation_input->n_orbitals) {
    throw std::invalid_argument(
        "reference layout is incompatible with the MO gauge-fix target");
  }
  if (n_inactive_orbitals == 1) {
    return false;
  }
  const int n_basis_functions = orbital_preparation_input->n_basis_functions;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      orbital_preparation_input->ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  // The inactive occupied span is physical, whereas its individual columns
  // are gauge representatives. For each target support, construct the exact
  // null space of all forbidden AO rows. Cyclic determinant maximization then
  // chooses, within that admissible space, the normalized column with the
  // largest component outside the span of all other representatives. Each
  // update preserves strict support and increases the normalized Gram
  // determinant, directly removing avoidable near-linear dependence without
  // changing the occupied subspace.
  const Eigen::MatrixXd current_inactive_orbitals =
      build_dense_sparse_orbital_columns(
          *orbital_preparation_input,
          n_inactive_orbitals);
  if (!orbital_preparation_input->maintain_inactive_gauge &&
      inactive_supports_match(
          reference_layout,
          *orbital_preparation_input,
          n_inactive_orbitals) &&
      !inactive_metric_loses_half_precision(
          current_inactive_orbitals,
          basis_overlap)) {
    return false;
  }
  const Eigen::MatrixXd inactive_right_transform =
      build_balanced_inactive_right_transform(
          current_inactive_orbitals,
          basis_overlap,
          reference_layout);
  const Eigen::MatrixXd localized_inactive_orbitals =
      current_inactive_orbitals * inactive_right_transform;

  for (int orbital = 0; orbital < n_inactive_orbitals; ++orbital) {
    std::vector<unsigned char> allowed(n_basis_functions, 0);
    for (const int basis :
         collect_support_indices_from_layout(reference_layout, orbital)) {
      allowed[basis] = 1;
    }
    const double backward_error =
        128.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(n_basis_functions * n_inactive_orbitals) *
        current_inactive_orbitals.norm() *
        inactive_right_transform.col(orbital).norm();
    for (int basis = 0; basis < n_basis_functions; ++basis) {
      if (allowed[basis] == 0 &&
          std::abs(localized_inactive_orbitals(basis, orbital)) >
              backward_error) {
        throw std::runtime_error(
            "inactive gauge balancing violated strict orbital support");
      }
    }
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
  const bool chart_changed =
      !dense_matrix_is_effectively_identity(inactive_right_transform);
  orbital_preparation_input->maintain_inactive_gauge = true;
  return chart_changed;
}

bool apply_support_preserving_inactive_gauge(
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    return false;
  }

  if (!orbital_input_has_support_preserving_gauge_reference(
          *orbital_preparation_input)) {
    return apply_support_preserving_inactive_gauge(
        *orbital_preparation_input,
        orbital_preparation_input);
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
  return apply_support_preserving_inactive_gauge(
      reference_layout,
      orbital_preparation_input);
}

bool balance_active_gauge(OrbitalPreparationInput* input) {
  if (input == nullptr) {
    throw std::invalid_argument("active gauge target must not be null");
  }
  // Full-AO OEO with complete structures admits active GL gauge, so this
  // per-orbital section is not its complete quotient. The generic chart is
  // likewise outside the validated HAO/BDO sparse-ray case.
  if (input->orbital_type != kOrbitalTypeHao &&
      input->orbital_type != kOrbitalTypeBdo) {
    return false;
  }

  const int n_active = static_cast<int>(input->n_active_orbitals);
  const int n_inactive = static_cast<int>(
      (input->n_total_electrons - input->n_active_electrons) / 2);
  if (n_active == 0 || n_inactive == 0) return false;
  const int n_bf = static_cast<int>(input->n_basis_functions);
  const Eigen::MatrixXd occupied =
      build_dense_sparse_orbital_columns(*input, n_inactive + n_active);
  const auto inactive = occupied.leftCols(n_inactive);
  const Eigen::MatrixXd& S = input->ao_overlap_matrix;
  SparseParameterLayout layout(*input);
  bool changed = false;
  for (int orbital = n_inactive;
       orbital < n_inactive + n_active;
       ++orbital) {
    const int count = layout.orbital_coefficient_count(orbital);
    std::vector<int> support;
    support.reserve(count);
    for (int slot = 0; slot < count; ++slot) {
      support.push_back(
          input->orbital_basis_index_table[orbital * n_bf + slot] - 1);
    }
    const Eigen::MatrixXd N =
        build_admissible_gauge_basis(inactive, support);
    if (N.cols() == 0) continue;
    const Eigen::MatrixXd Z = inactive * N;
    const Eigen::MatrixXd gram = Z.transpose() * S * Z;
    Eigen::LDLT<Eigen::MatrixXd> factor(gram);
    if (factor.info() != Eigen::Success || !factor.isPositive()) {
      throw std::runtime_error(
          "support-admissible active gauge has singular AO metric");
    }
    const Eigen::VectorXd raw = occupied.col(orbital);
    const Eigen::VectorXd correction =
        Z * factor.solve(Z.transpose() * S * raw);
    Eigen::VectorXd balanced = raw - correction;
    const double norm_squared = balanced.dot(S * balanced);
    if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
      throw std::runtime_error(
          "balanced active representative has zero AO norm");
    }
    const int stored_count =
        stored_sparse_orbital_coefficient_count(*input, orbital);
    bool scaling_admissible = true;
    for (int slot = count; slot < stored_count; ++slot) {
      if (input->orbital_value_table[orbital * n_bf + slot] != 0.0) {
        scaling_admissible = false;
        break;
      }
    }
    const double scale = scaling_admissible
        ? 1.0 / std::sqrt(norm_squared)
        : 1.0;
    balanced *= scale;

    // Off-support cancellation is exact algebraically; permit only its
    // dimension-scaled floating-point backward error after normalization.
    const double backward_error =
        128.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(n_bf * n_inactive) *
        (raw.norm() + correction.norm()) * scale;
    std::vector<unsigned char> allowed(n_bf, 0);
    for (int ao : support) allowed[ao] = 1;
    for (int ao = 0; ao < n_bf; ++ao) {
      if (allowed[ao] == 0 &&
          std::abs(balanced[ao] - raw[ao]) > backward_error) {
        throw std::runtime_error(
            "active gauge balancing violated strict support");
      }
    }
    for (int slot = 0; slot < count; ++slot) {
      const int flat = orbital * n_bf + slot;
      const double value = balanced[support[slot]];
      changed = changed ||
          std::abs(input->orbital_value_table[flat] - value) >
              backward_error;
      input->orbital_value_table[flat] = value;
    }
  }
  enforce_strict_sparse_orbital_support(input);
  return changed;
}

}  // namespace xmvb::vb
