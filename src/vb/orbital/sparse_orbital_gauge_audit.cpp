#include "vb/orbital/sparse_orbital_gauge_audit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace xmvb::vb {
namespace {

struct RankRevealingSvd {
  Eigen::JacobiSVD<Eigen::MatrixXd> decomposition;
  int rank = 0;
  double tolerance = 0.0;
};

RankRevealingSvd factorize_with_backward_error_tolerance(
    const Eigen::MatrixXd& matrix,
    unsigned int computation_options) {
  RankRevealingSvd result{
      Eigen::JacobiSVD<Eigen::MatrixXd>(matrix, computation_options), 0, 0.0};
  if (matrix.rows() == 0 || matrix.cols() == 0) return result;
  const auto& singular_values = result.decomposition.singularValues();
  if (singular_values.size() == 0) return result;
  const double sigma_max = singular_values[0];
  result.tolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(matrix.rows(), matrix.cols())) *
      sigma_max;
  for (Eigen::Index index = 0; index < singular_values.size(); ++index) {
    if (singular_values[index] > result.tolerance) ++result.rank;
  }
  return result;
}

Eigen::MatrixXd build_raw_occupied_orbitals(
    const OrbitalPreparationInput& input,
    int n_occupied) {
  Eigen::MatrixXd occupied =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_occupied);
  for (int orbital = 0; orbital < n_occupied; ++orbital) {
    const int count = stored_sparse_orbital_coefficient_count(input, orbital);
    for (int coefficient = 0; coefficient < count; ++coefficient) {
      const int basis = input.orbital_basis_index_table[
          orbital * input.n_basis_functions + coefficient] - 1;
      if (basis < 0 || basis >= input.n_basis_functions) {
        throw std::runtime_error(
            "invalid AO index in strict-sparse gauge audit");
      }
      occupied(basis, orbital) = input.orbital_value_table[
          orbital * input.n_basis_functions + coefficient];
    }
  }
  return occupied;
}

struct GramEigensystem {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition;
  int rank = 0;
  double tolerance = 0.0;
};

GramEigensystem factorize_right_gram(const Eigen::MatrixXd& matrix) {
  const Eigen::MatrixXd gram = matrix.transpose() * matrix;
  GramEigensystem result{
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(gram), 0, 0.0};
  if (result.decomposition.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize audit Gram matrix");
  }
  if (gram.rows() == 0) return result;
  const double lambda_max =
      std::max(0.0, result.decomposition.eigenvalues().tail(1)[0]);
  // Forming J^T J loses relative accuracy near its null space.  A
  // backward-error threshold for the Gram matrix is therefore proportional
  // to epsilon * ||J||^2, not to epsilon^2 * ||J||^2.
  result.tolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(matrix.rows(), matrix.cols())) *
      lambda_max;
  for (Eigen::Index index = 0;
       index < result.decomposition.eigenvalues().size();
       ++index) {
    if (result.decomposition.eigenvalues()[index] > result.tolerance) {
      ++result.rank;
    }
  }
  return result;
}

int matrix_numerical_rank_from_gram(const Eigen::MatrixXd& matrix) {
  return factorize_right_gram(matrix).rank;
}

}  // namespace

SparseOrbitalGaugeAudit audit_sparse_orbital_gauge(
    const OrbitalPreparationInput& input,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::MatrixXd* current_packed_reduced_basis) {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 ||
      input.n_active_orbitals < 0) {
    throw std::invalid_argument("invalid orbital dimensions in gauge audit");
  }
  const int n_inactive =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const int n_active = input.n_active_orbitals;
  const int n_occupied = n_inactive + n_active;
  const int n_ao = input.n_basis_functions;
  const int packed_dimension = parameter_view.size();
  if (n_inactive < 0 || n_occupied > input.n_orbitals) {
    throw std::invalid_argument("invalid occupied partition in gauge audit");
  }

  SparseOrbitalGaugeAudit audit;
  audit.packed_dimension = packed_dimension;
  audit.gauge_parameter_dimension =
      n_inactive * n_inactive + n_inactive * n_active + n_active;

  const Eigen::MatrixXd occupied =
      build_raw_occupied_orbitals(input, n_occupied);
  const auto inactive = occupied.leftCols(n_inactive);
  const auto active = occupied.middleCols(n_inactive, n_active);

  // Map every AO/occupied entry to its packed differentiable coordinate.
  // Stored but nondifferentiable coefficients and off-support AO entries are
  // constraints on an admissible infinitesimal gauge transformation.
  std::vector<int> ao_occupied_to_packed(n_ao * n_occupied, -1);
  std::vector<char> packed_is_occupied(packed_dimension, 0);
  for (int orbital = 0; orbital < n_occupied; ++orbital) {
    const int differentiable_count =
        parameter_view.orbital_coefficient_count(orbital);
    const int stored_count =
        stored_sparse_orbital_coefficient_count(input, orbital);
    if (differentiable_count > stored_count) {
      throw std::runtime_error(
          "differentiable support exceeds stored support in gauge audit");
    }
    for (int coefficient = 0; coefficient < differentiable_count;
         ++coefficient) {
      const int basis = input.orbital_basis_index_table[
          orbital * n_ao + coefficient] - 1;
      const int packed = parameter_view.packed_index(orbital, coefficient);
      if (packed < 0 || packed >= packed_dimension) {
        throw std::runtime_error("invalid packed index in gauge audit");
      }
      ao_occupied_to_packed[orbital * n_ao + basis] = packed;
      packed_is_occupied[packed] = 1;
    }
  }
  for (const char is_occupied : packed_is_occupied) {
    if (!is_occupied) ++audit.unmapped_parameter_count;
  }

  const int gauge_parameter_dimension = audit.gauge_parameter_dimension;
  Eigen::MatrixXd gauge_action = Eigen::MatrixXd::Zero(
      n_ao * n_occupied, gauge_parameter_dimension);
  int parameter_column = 0;
  // dC_I = C_I K; K(source, target) follows column-major enumeration.
  for (int target = 0; target < n_inactive; ++target) {
    for (int source = 0; source < n_inactive; ++source, ++parameter_column) {
      gauge_action.block(target * n_ao, parameter_column, n_ao, 1) =
          inactive.col(source);
    }
  }
  // dC_A = C_I L.
  for (int target = 0; target < n_active; ++target) {
    for (int source = 0; source < n_inactive; ++source, ++parameter_column) {
      gauge_action.block(
          (n_inactive + target) * n_ao, parameter_column, n_ao, 1) =
          inactive.col(source);
    }
  }
  // Independent active-orbital radial scalings.
  for (int target = 0; target < n_active; ++target, ++parameter_column) {
    gauge_action.block(
        (n_inactive + target) * n_ao, parameter_column, n_ao, 1) =
        active.col(target);
  }
  if (parameter_column != gauge_parameter_dimension) {
    throw std::runtime_error("internal gauge parameter count mismatch");
  }

  int forbidden_count = 0;
  for (int row = 0; row < n_ao * n_occupied; ++row) {
    if (ao_occupied_to_packed[row] < 0) ++forbidden_count;
  }
  Eigen::MatrixXd forbidden_action =
      Eigen::MatrixXd::Zero(forbidden_count, gauge_parameter_dimension);
  Eigen::MatrixXd packed_action =
      Eigen::MatrixXd::Zero(packed_dimension, gauge_parameter_dimension);
  int forbidden_row = 0;
  for (int row = 0; row < n_ao * n_occupied; ++row) {
    const int packed = ao_occupied_to_packed[row];
    if (packed >= 0) {
      packed_action.row(packed) = gauge_action.row(row);
    } else {
      forbidden_action.row(forbidden_row++) = gauge_action.row(row);
    }
  }

  Eigen::MatrixXd admissible_parameter_basis;
  if (gauge_parameter_dimension == 0) {
    admissible_parameter_basis = Eigen::MatrixXd::Zero(0, 0);
  } else if (forbidden_count == 0) {
    admissible_parameter_basis =
        Eigen::MatrixXd::Identity(
            gauge_parameter_dimension, gauge_parameter_dimension);
  } else {
    const RankRevealingSvd constraint_svd =
        factorize_with_backward_error_tolerance(
            forbidden_action, Eigen::ComputeFullV);
    audit.support_constraint_rank = constraint_svd.rank;
    admissible_parameter_basis =
        constraint_svd.decomposition.matrixV().rightCols(
            gauge_parameter_dimension - constraint_svd.rank);
  }
  audit.admissible_gauge_parameter_dimension =
      static_cast<int>(admissible_parameter_basis.cols());

  const Eigen::MatrixXd packed_gauge_generators =
      packed_action * admissible_parameter_basis;
  Eigen::MatrixXd packed_gauge_basis =
      Eigen::MatrixXd::Zero(packed_dimension, 0);
  if (packed_gauge_generators.cols() > 0) {
    const RankRevealingSvd gauge_svd =
        factorize_with_backward_error_tolerance(
            packed_gauge_generators,
            Eigen::ComputeThinU | Eigen::ComputeThinV);
    audit.gauge_rank = gauge_svd.rank;
    packed_gauge_basis =
        gauge_svd.decomposition.matrixU().leftCols(gauge_svd.rank);
  }
  audit.quotient_dimension = packed_dimension - audit.gauge_rank;

  // Independently differentiate the physical map
  //   x -> (P_I, horizontal variations of [O_I c_A]).
  // Extra differentiable orbitals outside the occupied VB block are appended
  // as identity features so they cannot be mislabeled as gauge variables.
  const Eigen::Map<const Eigen::MatrixXd> overlap(
      input.ao_overlap_matrix.data(), n_ao, n_ao);
  Eigen::MatrixXd inactive_metric_inverse = Eigen::MatrixXd::Zero(
      n_inactive, n_inactive);
  Eigen::MatrixXd inactive_density = Eigen::MatrixXd::Zero(n_ao, n_ao);
  if (n_inactive > 0) {
    const Eigen::MatrixXd inactive_metric =
        inactive.transpose() * overlap * inactive;
    Eigen::LDLT<Eigen::MatrixXd> metric_ldlt(inactive_metric);
    if (metric_ldlt.info() != Eigen::Success) {
      throw std::runtime_error("failed to factor inactive metric in gauge audit");
    }
    inactive_metric_inverse = metric_ldlt.solve(
        Eigen::MatrixXd::Identity(n_inactive, n_inactive));
    if (!inactive_metric_inverse.allFinite()) {
      throw std::runtime_error("non-finite inactive inverse in gauge audit");
    }
    inactive_density =
        inactive * inactive_metric_inverse * inactive.transpose();
  }
  const Eigen::MatrixXd complement =
      Eigen::MatrixXd::Identity(n_ao, n_ao) - inactive_density * overlap;
  const Eigen::MatrixXd projected_active = complement * active;

  const int physical_feature_count =
      n_ao * n_ao + n_ao * n_active + audit.unmapped_parameter_count;
  Eigen::MatrixXd physical_jacobian = Eigen::MatrixXd::Zero(
      physical_feature_count, packed_dimension);
  int extra_feature_offset = n_ao * n_ao + n_ao * n_active;
  for (int packed = 0; packed < packed_dimension; ++packed) {
    if (!packed_is_occupied[packed]) {
      physical_jacobian(extra_feature_offset++, packed) = 1.0;
      continue;
    }

    const int flat = parameter_view.differentiable_parameter_indices()[packed];
    const int orbital = flat / n_ao;
    const int coefficient = flat % n_ao;
    const int basis = input.orbital_basis_index_table[
        orbital * n_ao + coefficient] - 1;
    Eigen::MatrixXd delta_inactive =
        Eigen::MatrixXd::Zero(n_ao, n_inactive);
    Eigen::MatrixXd delta_active =
        Eigen::MatrixXd::Zero(n_ao, n_active);
    if (orbital < n_inactive) {
      delta_inactive(basis, orbital) = 1.0;
    } else {
      delta_active(basis, orbital - n_inactive) = 1.0;
    }

    Eigen::MatrixXd delta_density = Eigen::MatrixXd::Zero(n_ao, n_ao);
    if (n_inactive > 0) {
      const Eigen::MatrixXd delta_metric =
          delta_inactive.transpose() * overlap * inactive +
          inactive.transpose() * overlap * delta_inactive;
      const Eigen::MatrixXd delta_inverse =
          -inactive_metric_inverse * delta_metric * inactive_metric_inverse;
      delta_density =
          delta_inactive * inactive_metric_inverse * inactive.transpose() +
          inactive * delta_inverse * inactive.transpose() +
          inactive * inactive_metric_inverse * delta_inactive.transpose();
    }
    Eigen::Map<Eigen::VectorXd>(
        physical_jacobian.col(packed).data(), n_ao * n_ao) =
        Eigen::Map<const Eigen::VectorXd>(
            delta_density.data(), n_ao * n_ao);

    const Eigen::MatrixXd delta_projected_active =
        complement * delta_active - delta_density * overlap * active;
    for (int active_orbital = 0; active_orbital < n_active;
         ++active_orbital) {
      const Eigen::VectorXd projected = projected_active.col(active_orbital);
      Eigen::VectorXd horizontal =
          delta_projected_active.col(active_orbital);
      const double norm_squared = projected.dot(overlap * projected);
      if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
        throw std::runtime_error(
            "degenerate projected active orbital in gauge audit");
      }
      horizontal.noalias() -=
          projected * (projected.dot(overlap * horizontal) / norm_squared);
      physical_jacobian.block(
          n_ao * n_ao + active_orbital * n_ao,
          packed,
          n_ao,
          1) = horizontal;
    }
  }

  const GramEigensystem physical_gram =
      factorize_right_gram(physical_jacobian);
  audit.physical_jacobian_rank = physical_gram.rank;
  audit.physical_jacobian_nullity =
      packed_dimension - physical_gram.rank;

  const double jacobian_norm = physical_jacobian.norm();
  audit.relative_gauge_annihilation_residual =
      packed_gauge_basis.cols() == 0 || !(jacobian_norm > 0.0)
          ? 0.0
          : (physical_jacobian * packed_gauge_basis).norm() / jacobian_norm;

  if (audit.gauge_rank == audit.physical_jacobian_nullity &&
      audit.gauge_rank > 0) {
    const Eigen::MatrixXd physical_kernel =
        physical_gram.decomposition.eigenvectors().leftCols(
            audit.physical_jacobian_nullity);
    const Eigen::MatrixXd gauge_kernel_overlap =
        packed_gauge_basis.transpose() * physical_kernel;
    const Eigen::JacobiSVD<Eigen::MatrixXd> angle_svd(
        gauge_kernel_overlap,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double minimum_cosine =
        angle_svd.singularValues().tail(1)[0];
    audit.maximum_gauge_kernel_principal_angle_sine =
        std::sqrt(std::max(0.0, 1.0 - minimum_cosine * minimum_cosine));
  } else if (audit.gauge_rank != audit.physical_jacobian_nullity) {
    audit.maximum_gauge_kernel_principal_angle_sine = 1.0;
  }

  if (current_packed_reduced_basis != nullptr) {
    if (current_packed_reduced_basis->rows() != packed_dimension ||
        !current_packed_reduced_basis->allFinite()) {
      throw std::invalid_argument(
          "current reduced basis has invalid shape in gauge audit");
    }
    audit.current_reduced_dimension =
        static_cast<int>(current_packed_reduced_basis->cols());
    audit.current_physical_image_rank = matrix_numerical_rank_from_gram(
        physical_jacobian * *current_packed_reduced_basis);
    audit.current_retained_gauge_dimension =
        audit.current_reduced_dimension - audit.current_physical_image_rank;
    audit.current_missing_physical_dimension =
        audit.physical_jacobian_rank - audit.current_physical_image_rank;
  }

  return audit;
}

}  // namespace xmvb::vb
