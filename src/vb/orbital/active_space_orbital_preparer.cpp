#include "vb/orbital/active_space_orbital_preparer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vb/orbital/legacy_jacobi_diagonalizer.hpp"
#include "vb/orbital/localized_representative_selector.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

void require_finite_matrix(const Eigen::MatrixXd& matrix, const char* label) {
  for (int column = 0; column < matrix.cols(); ++column) {
    for (int row = 0; row < matrix.rows(); ++row) {
      if (!std::isfinite(matrix(row, column))) {
        throw std::runtime_error(
            std::string(label) + " contains non-finite values at row=" +
            std::to_string(row) + " col=" + std::to_string(column));
      }
    }
  }
}

void require_finite_vector(const std::vector<double>& values, const char* label) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error(
          std::string(label) + " contains non-finite values at index=" +
          std::to_string(index));
    }
  }
}

int get_sparse_coefficient_count(
    const std::vector<int>& orbital_basis_counts,
    const std::vector<int>& orbital_basis_index_table,
    int n_basis_functions,
    int orbital_index) {
  const int explicit_count = orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_basis_index_table[xmvb::to_size(orbital_index) * n_basis_functions +
                                  coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<double> normalize_sparse_orbitals(
    const OrbitalPreparationInput& input,
    const Eigen::Map<const Eigen::MatrixXd>& active_orbital_overlap_matrix) {
  std::vector<double> normalized_values = input.orbital_value_table;

#pragma omp parallel for schedule(static)
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count = get_sparse_coefficient_count(
        input.orbital_basis_counts,
        input.orbital_basis_index_table,
        input.n_basis_functions,
        orbital_index);
    double squared_norm = 0.0;
    for (int left_index = 0; left_index < coefficient_count; ++left_index) {
      const int left_basis_function =
          input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                              input.n_basis_functions +
                                          left_index] -
          1;
      if (left_basis_function < 0 || left_basis_function >= input.n_basis_functions) {
        throw std::runtime_error("invalid sparse orbital basis index");
      }
      const double left_value =
          normalized_values[xmvb::to_size(orbital_index) * input.n_basis_functions +
                            left_index];
      for (int right_index = 0; right_index < coefficient_count; ++right_index) {
        const int right_basis_function =
            input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                                input.n_basis_functions +
                                            right_index] -
            1;
        const double right_value =
            normalized_values[xmvb::to_size(orbital_index) * input.n_basis_functions +
                              right_index];
        squared_norm +=
            left_value * right_value * active_orbital_overlap_matrix(left_basis_function, right_basis_function);
      }
    }

    if (!std::isfinite(squared_norm) ||
        squared_norm <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "orbital normalization failed for orbital " + std::to_string(orbital_index) +
          " with squared_norm=" + std::to_string(squared_norm));
    }

    const double normalization_factor = std::sqrt(1.0 / squared_norm);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      normalized_values[xmvb::to_size(orbital_index) * input.n_basis_functions +
                        coefficient_index] *= normalization_factor;
    }
  }

  return normalized_values;
}

Eigen::MatrixXd expand_sparse_orbitals(
    const OrbitalPreparationInput& input,
    const std::vector<double>& normalized_orbital_values) {
  Eigen::MatrixXd orbital_matrix =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);

#pragma omp parallel for schedule(static)
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count = get_sparse_coefficient_count(
        input.orbital_basis_counts,
        input.orbital_basis_index_table,
        input.n_basis_functions,
        orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      orbital_matrix(
          basis_function_index,
          orbital_index) =
          normalized_orbital_values[xmvb::to_size(orbital_index) *
                                        input.n_basis_functions +
                                    coefficient_index];
    }
  }

  return orbital_matrix;
}

Eigen::MatrixXd invert_self_adjoint_positive_definite(
    const Eigen::MatrixXd& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(matrix);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed LDLT factorization for ") + label);
  }
  Eigen::MatrixXd inverse =
      ldlt.solve(Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()));
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed SPD inverse solve for ") + label);
  }
  return inverse;
}

Eigen::MatrixXd build_inactive_density_low_rank_factors(
    const Eigen::MatrixXd& inactive_orbitals,
    const Eigen::MatrixXd& inactive_overlap_inverse,
    const char* label) {
  if (inactive_orbitals.cols() != inactive_overlap_inverse.rows() ||
      inactive_overlap_inverse.rows() != inactive_overlap_inverse.cols()) {
    throw std::invalid_argument(
        std::string(label) + " dimensions do not match inactive orbitals");
  }
  if (inactive_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(inactive_orbitals.rows(), 0);
  }

  // `P11 = C (C^T S C)^{-1} C^T` is symmetric positive semidefinite.  A Cholesky
  // factor of the small occupied-space inverse therefore gives an explicit AO
  // low-rank factor `F = C L` with `P11 = F F^T`.  The RI AO-H1E path can then
  // apply its low-rank kernels directly instead of diagonalizing the full AO
  // density matrix again on every objective call.
  Eigen::LLT<Eigen::MatrixXd> inverse_llt(inactive_overlap_inverse);
  if (inverse_llt.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed LLT factorization for ") + label);
  }

  Eigen::MatrixXd inactive_density_factors =
      inactive_orbitals * inverse_llt.matrixL();
  require_finite_matrix(inactive_density_factors, "inactive_density_low_rank_factors");
  return inactive_density_factors;
}

Eigen::MatrixXd build_self_adjoint_inverse_square_root(
    const Eigen::MatrixXd& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  if (matrix.rows() == 0) {
    return Eigen::MatrixXd::Zero(0, 0);
  }

  // The orthonormal-inactive gauge uses `Q_i = C_i R_i` with
  // `R_i^T (C_i^T S C_i) R_i = I`. A symmetric inverse square root is the
  // cleanest accepted-point gauge representative because it does not depend on
  // an arbitrary occupied-space column ordering beyond the eigensystem itself.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(matrix);
  if (eigen_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        std::string("failed eigendecomposition for ") + label);
  }

  const Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
  Eigen::VectorXd inverse_sqrt_eigenvalues(eigenvalues.size());
  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    const double eigenvalue = eigenvalues[index];
    if (!std::isfinite(eigenvalue) ||
        eigenvalue <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          std::string(label) + " is not numerically positive definite");
    }
    inverse_sqrt_eigenvalues[index] = 1.0 / std::sqrt(eigenvalue);
  }

  Eigen::MatrixXd inverse_square_root =
      eigen_solver.eigenvectors() *
      inverse_sqrt_eigenvalues.asDiagonal() *
      eigen_solver.eigenvectors().transpose();
  require_finite_matrix(inverse_square_root, "inactive_orthonormal_gauge_transform");
  return inverse_square_root;
}

Eigen::MatrixXd build_s_orthonormal_virtual_auxiliary_orbitals(
    const Eigen::MatrixXd& occupied_auxiliary_orbitals,
    const Eigen::MatrixXd& occupied_overlap_inverse,
    const Eigen::Map<const Eigen::MatrixXd>& basis_overlap_matrix,
    int n_virtual_orbitals) {
  const int n_basis_functions = static_cast<int>(basis_overlap_matrix.rows());
  if (n_virtual_orbitals <= 0) {
    return Eigen::MatrixXd::Zero(n_basis_functions, 0);
  }
  if (occupied_auxiliary_orbitals.rows() != n_basis_functions ||
      occupied_auxiliary_orbitals.cols() != occupied_overlap_inverse.rows() ||
      occupied_overlap_inverse.rows() != occupied_overlap_inverse.cols()) {
    throw std::invalid_argument(
        "occupied auxiliary dimensions do not match occupied-overlap inverse");
  }

  // Match legacy `OrbPrep6`: form the occupied-space projector in the AO
  // metric, diagonalize the projected metric `(I - P_occ S)^T S (I - P_occ S)`,
  // and keep the positive-eigenvalue directions as the `S`-orthonormal virtual
  // auxiliary block. This fixes the virtual gauge to the same projector-based
  // construction used by XMVB instead of an arbitrary QR complement.
  const Eigen::MatrixXd occupied_metric_action =
      occupied_overlap_inverse *
      occupied_auxiliary_orbitals.transpose() *
      basis_overlap_matrix;
  Eigen::MatrixXd complementary_projector =
      Eigen::MatrixXd::Identity(n_basis_functions, n_basis_functions) -
      occupied_auxiliary_orbitals * occupied_metric_action;
  const Eigen::MatrixXd virtual_overlap =
      complementary_projector.transpose() *
      basis_overlap_matrix *
      complementary_projector;
  require_finite_matrix(virtual_overlap, "virtual_overlap_projector_metric");

  const LegacyJacobiDiagonalizationResult eigenpairs =
      diagonalize_self_adjoint_legacy_jacobi(virtual_overlap);

  constexpr double kVirtualEigenvalueTolerance = 1.0e-10;
  Eigen::MatrixXd virtual_orbitals =
      Eigen::MatrixXd::Zero(n_basis_functions, n_virtual_orbitals);
  int virtual_column = 0;
  for (int eigen_index = 0;
       eigen_index < eigenpairs.eigenvalues.size();
       ++eigen_index) {
    const double eigenvalue = eigenpairs.eigenvalues[eigen_index];
    if (!(eigenvalue > kVirtualEigenvalueTolerance)) {
      continue;
    }
    if (virtual_column >= n_virtual_orbitals) {
      break;
    }

    Eigen::VectorXd virtual_orbital =
        complementary_projector *
        eigenpairs.eigenvectors.col(eigen_index);
    const double virtual_norm =
        virtual_orbital.transpose() *
        basis_overlap_matrix *
        virtual_orbital;
    if (!(virtual_norm > std::numeric_limits<double>::epsilon()) ||
        !std::isfinite(virtual_norm)) {
      continue;
    }

    virtual_orbitals.col(virtual_column) =
        virtual_orbital / std::sqrt(virtual_norm);
    ++virtual_column;
  }

  if (virtual_column != n_virtual_orbitals) {
    throw std::runtime_error(
        "failed to construct the full virtual auxiliary orbital space");
  }

  require_finite_matrix(virtual_orbitals, "virtual_auxiliary_orbitals");
  return virtual_orbitals;
}

Eigen::MatrixXd build_auxiliary_orbital_inverse(
    const Eigen::MatrixXd& auxiliary_orbital_matrix,
    const Eigen::MatrixXd& occupied_overlap_inverse,
    const Eigen::Map<const Eigen::MatrixXd>& basis_overlap_matrix,
    int n_occupied_orbitals) {
  if (auxiliary_orbital_matrix.rows() != basis_overlap_matrix.rows() ||
      auxiliary_orbital_matrix.cols() != basis_overlap_matrix.cols()) {
    throw std::invalid_argument(
        "auxiliary orbital matrix dimensions do not match AO overlap");
  }
  if (n_occupied_orbitals < 0 ||
      n_occupied_orbitals > auxiliary_orbital_matrix.cols()) {
    throw std::invalid_argument("invalid occupied auxiliary block size");
  }
  if (occupied_overlap_inverse.rows() != occupied_overlap_inverse.cols() ||
      occupied_overlap_inverse.rows() != n_occupied_orbitals) {
    throw std::invalid_argument(
        "occupied auxiliary overlap inverse dimensions do not match");
  }

  const int n_basis_functions = static_cast<int>(auxiliary_orbital_matrix.rows());
  const int n_virtual_orbitals = n_basis_functions - n_occupied_orbitals;
  const Eigen::MatrixXd s_times_auxiliary =
      basis_overlap_matrix * auxiliary_orbital_matrix;
  Eigen::MatrixXd auxiliary_orbital_inverse =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);

  if (n_occupied_orbitals > 0) {
    // The occupied auxiliary block uses the mixed gauge
    // `T_occ = [C_i, (I - P_i S) C_a]`, so its metric is block diagonal:
    // inactive-inactive uses `(C_i^T S C_i)`, inactive-active vanishes, and the
    // active-active block remains nonorthogonal.  For the full auxiliary basis
    // `T = [T_occ, T_vir]` with `T_occ^T S T_vir = 0` and `T_vir^T S T_vir = I`,
    // the inverse is available analytically as
    // `[S_occ^{-1} T_occ^T S; T_vir^T S]`.
    auxiliary_orbital_inverse.topRows(n_occupied_orbitals).noalias() =
        occupied_overlap_inverse *
        s_times_auxiliary.leftCols(n_occupied_orbitals).transpose();
  }
  if (n_virtual_orbitals > 0) {
    auxiliary_orbital_inverse.bottomRows(n_virtual_orbitals) =
        s_times_auxiliary.rightCols(n_virtual_orbitals).transpose();
  }

  require_finite_matrix(auxiliary_orbital_inverse, "auxiliary_orbital_inverse");
  return auxiliary_orbital_inverse;
}

}  // namespace

OrbitalPreparationResult ActiveSpaceOrbitalPreparer::prepare(
    const OrbitalPreparationInput& input) const {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 || input.n_active_orbitals <= 0) {
    throw std::invalid_argument("orbital preparation input dimensions must be positive");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const int n_occupied_orbitals = n_inactive_doubly_occupied_orbitals + input.n_active_orbitals;
  const int n_virtual_orbitals = input.n_basis_functions - n_occupied_orbitals;
  if (n_inactive_doubly_occupied_orbitals < 0 || n_virtual_orbitals < 0) {
    throw std::invalid_argument("invalid occupied/virtual partition");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const std::vector<double> normalized_orbital_values =
      normalize_sparse_orbitals(input, basis_overlap_matrix);
  require_finite_vector(normalized_orbital_values, "normalized_orbital_values");
  Eigen::MatrixXd original_orbital_matrix =
      expand_sparse_orbitals(input, normalized_orbital_values);
  require_finite_matrix(original_orbital_matrix, "original_orbital_matrix");

  Eigen::MatrixXd auxiliary_orbital_matrix =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_basis_functions);
  Eigen::MatrixXd inactive_auxiliary_transform =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_basis_functions);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    auxiliary_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals) =
        original_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
  }

  Eigen::MatrixXd inactive_density_matrix =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_basis_functions);
  Eigen::MatrixXd inactive_orthonormal_projector_matrix =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_basis_functions);
  Eigen::MatrixXd inactive_density_low_rank_factors =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_inactive_doubly_occupied_orbitals);
  Eigen::MatrixXd inactive_overlap_inverse =
      Eigen::MatrixXd::Zero(
          n_inactive_doubly_occupied_orbitals,
          n_inactive_doubly_occupied_orbitals);
  Eigen::MatrixXd inactive_orthonormal_orbitals =
      Eigen::MatrixXd::Zero(
          input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals);
  Eigen::MatrixXd inactive_orthonormal_gauge_transform =
      Eigen::MatrixXd::Zero(
          n_inactive_doubly_occupied_orbitals,
          n_inactive_doubly_occupied_orbitals);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const auto inactive_orbitals =
        auxiliary_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
    const Eigen::MatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap_matrix * inactive_orbitals;
    require_finite_matrix(inactive_overlap, "inactive_overlap");
    inactive_overlap_inverse =
        invert_self_adjoint_positive_definite(
            inactive_overlap,
            "inactive_overlap");
    require_finite_matrix(inactive_overlap_inverse, "inactive_overlap_inverse");
    const Eigen::MatrixXd auxiliary_inactive_orbitals =
        inactive_orbitals * inactive_overlap_inverse;
    inactive_auxiliary_transform.leftCols(n_inactive_doubly_occupied_orbitals) =
        auxiliary_inactive_orbitals;
    inactive_density_low_rank_factors =
        build_inactive_density_low_rank_factors(
            inactive_orbitals,
            inactive_overlap_inverse,
            "inactive_overlap_inverse");
    inactive_density_matrix = auxiliary_inactive_orbitals * inactive_orbitals.transpose();
    require_finite_matrix(inactive_density_matrix, "inactive_density_matrix");
    inactive_orthonormal_gauge_transform =
        build_self_adjoint_inverse_square_root(
            inactive_overlap,
            "inactive_overlap");
    inactive_orthonormal_orbitals =
        inactive_orbitals * inactive_orthonormal_gauge_transform;
    require_finite_matrix(
        inactive_orthonormal_orbitals,
        "inactive_orthonormal_orbitals");
    inactive_orthonormal_projector_matrix =
        inactive_orthonormal_orbitals *
        inactive_orthonormal_orbitals.transpose();
    require_finite_matrix(
        inactive_orthonormal_projector_matrix,
        "inactive_orthonormal_projector_matrix");
  }

  const auto active_orbitals = original_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals);
  Eigen::MatrixXd occupied_space_projector =
      Eigen::MatrixXd::Identity(input.n_basis_functions, input.n_basis_functions);
  occupied_space_projector -= inactive_density_matrix * basis_overlap_matrix;
  auxiliary_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals) = occupied_space_projector * active_orbitals;
  require_finite_matrix(auxiliary_orbital_matrix, "auxiliary_orbital_matrix");

  const Eigen::MatrixXd active_auxiliary_orbitals = auxiliary_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals);
  const Eigen::MatrixXd active_overlap_matrix =
      active_auxiliary_orbitals.transpose() * basis_overlap_matrix * active_auxiliary_orbitals;
  require_finite_matrix(active_overlap_matrix, "active_orbital_overlap_matrix");

  std::vector<int> active_sparse_row_offsets(
      xmvb::to_size(input.n_basis_functions) + 1,
      0);
  std::vector<int> active_sparse_orbital_indices;
  std::vector<double> active_sparse_values;
  active_sparse_orbital_indices.reserve(
      xmvb::to_size(input.n_basis_functions) * input.n_active_orbitals);
  active_sparse_values.reserve(
      xmvb::to_size(input.n_basis_functions) * input.n_active_orbitals);
  for (int basis_function_index = 0;
       basis_function_index < input.n_basis_functions;
       ++basis_function_index) {
    active_sparse_row_offsets[xmvb::to_size(basis_function_index)] =
        static_cast<int>(active_sparse_values.size());
    for (int active_orbital_index = 0;
         active_orbital_index < input.n_active_orbitals;
         ++active_orbital_index) {
      const double coefficient =
          active_auxiliary_orbitals(basis_function_index, active_orbital_index);
      if (coefficient == 0.0) {
        continue;
      }
      active_sparse_orbital_indices.push_back(active_orbital_index);
      active_sparse_values.push_back(coefficient);
    }
  }
  active_sparse_row_offsets[xmvb::to_size(input.n_basis_functions)] =
      static_cast<int>(active_sparse_values.size());
  const Eigen::MatrixXd active_orbital_overlap_inverse =
      invert_self_adjoint_positive_definite(
          active_overlap_matrix,
          "active_orbital_overlap_matrix");
  require_finite_matrix(active_orbital_overlap_inverse, "active_orbital_overlap_inverse");

  Eigen::MatrixXd occupied_overlap_inverse =
      Eigen::MatrixXd::Zero(n_occupied_orbitals, n_occupied_orbitals);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    occupied_overlap_inverse.topLeftCorner(
        n_inactive_doubly_occupied_orbitals,
        n_inactive_doubly_occupied_orbitals) = inactive_overlap_inverse;
  }
  occupied_overlap_inverse.bottomRightCorner(
      input.n_active_orbitals,
      input.n_active_orbitals) = active_orbital_overlap_inverse;

  if (n_virtual_orbitals > 0) {
    auxiliary_orbital_matrix.rightCols(n_virtual_orbitals) =
        build_s_orthonormal_virtual_auxiliary_orbitals(
            auxiliary_orbital_matrix.leftCols(n_occupied_orbitals),
            occupied_overlap_inverse,
            basis_overlap_matrix,
            n_virtual_orbitals);
  }

  const Eigen::MatrixXd active_overlap_source =
      basis_overlap_matrix * active_orbitals;
  const Eigen::MatrixXd inactive_active_overlap_matrix_full =
      inactive_auxiliary_transform.transpose() * active_overlap_source;
  const Eigen::MatrixXd inactive_active_overlap_matrix =
      inactive_active_overlap_matrix_full.topRows(n_inactive_doubly_occupied_orbitals);
  const Eigen::MatrixXd projected_active_overlap_matrix =
      occupied_space_projector.transpose() * active_overlap_source;
  const Eigen::MatrixXd auxiliary_orbital_inverse =
      build_auxiliary_orbital_inverse(
          auxiliary_orbital_matrix,
          occupied_overlap_inverse,
          basis_overlap_matrix,
          n_occupied_orbitals);
  require_finite_matrix(inactive_active_overlap_matrix, "inactive_active_overlap_matrix");
  require_finite_matrix(projected_active_overlap_matrix, "projected_active_overlap_matrix");

  OrbitalPreparationResult result;
  result.auxiliary_orbital_matrix = auxiliary_orbital_matrix;
  result.active_sparse_row_offsets = std::move(active_sparse_row_offsets);
  result.active_sparse_orbital_indices = std::move(active_sparse_orbital_indices);
  result.active_sparse_values = std::move(active_sparse_values);
  result.active_orbital_overlap_matrix.assign(
      active_overlap_matrix.data(),
      active_overlap_matrix.data() + active_overlap_matrix.size());
  result.inactive_density_matrix = inactive_density_matrix;
  result.inactive_orthonormal_projector_matrix =
      inactive_orthonormal_projector_matrix;
  result.inactive_density_low_rank_factors = inactive_density_low_rank_factors;
  result.occupied_space_projector.assign(
      occupied_space_projector.data(),
      occupied_space_projector.data() + occupied_space_projector.size());
  result.inactive_auxiliary_transform.assign(
      inactive_auxiliary_transform.data(),
      inactive_auxiliary_transform.data() + inactive_auxiliary_transform.size());
  result.inactive_active_overlap_matrix.assign(
      inactive_active_overlap_matrix.data(),
      inactive_active_overlap_matrix.data() + inactive_active_overlap_matrix.size());
  result.projected_active_overlap_matrix.assign(
      projected_active_overlap_matrix.data(),
      projected_active_overlap_matrix.data() + projected_active_overlap_matrix.size());
  result.auxiliary_orbital_inverse_matrix.assign(
      auxiliary_orbital_inverse.data(),
      auxiliary_orbital_inverse.data() + auxiliary_orbital_inverse.size());
  result.physical_orbital_frame.normalized_orbital_matrix =
      original_orbital_matrix;
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const auto inactive_physical_orbitals =
        original_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
    result.physical_orbital_frame.inactive_physical_orbital_matrix =
        inactive_physical_orbitals;
    result.physical_orbital_frame.inactive_orthonormal_orbital_matrix =
        inactive_orthonormal_orbitals;
    result.physical_orbital_frame.inactive_orthonormal_gauge_transform =
        inactive_orthonormal_gauge_transform;
  } else {
    result.physical_orbital_frame.inactive_physical_orbital_matrix =
        Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
    result.physical_orbital_frame.inactive_orthonormal_orbital_matrix =
        Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
    result.physical_orbital_frame.inactive_orthonormal_gauge_transform =
        Eigen::MatrixXd::Zero(0, 0);
  }
  if (input.n_active_orbitals > 0) {
    const auto active_physical_orbitals =
        original_orbital_matrix.middleCols(
            n_inactive_doubly_occupied_orbitals,
            input.n_active_orbitals);
    result.physical_orbital_frame.active_physical_orbital_matrix =
        active_physical_orbitals;
  } else {
    result.physical_orbital_frame.active_physical_orbital_matrix =
        Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
  }
  result.physical_orbital_frame.localized_representative_selector =
      build_localized_representative_selector(
          result.physical_orbital_frame.inactive_physical_orbital_matrix,
          result.physical_orbital_frame.inactive_orthonormal_orbital_matrix,
          result.physical_orbital_frame.active_physical_orbital_matrix,
          active_auxiliary_orbitals,
          basis_overlap_matrix);
  return result;
}

}  // namespace xmvb::vb
