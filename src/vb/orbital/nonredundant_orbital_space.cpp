#include "vb/orbital/nonredundant_orbital_space.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include "runtime/cpp_block_guess_builder.hpp"
#include "vb/orbital/legacy_jacobi_diagonalizer.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
namespace xmvb::vb {

namespace {

constexpr int kLegacyOrbitalTypeOeo = 3;

constexpr int kMaxExactMetricFactorizationDirections = 256;
constexpr double kMinimumDenseMetricEigenvalue = 1.0e-12;

double parse_env_double_with_default(
    const char* name,
    double default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      !std::isfinite(parsed)) {
    return default_value;
  }
  return parsed;
}

double active_shape_curvature_shift_fraction() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_NONREDUNDANT_ACTIVE_SHAPE_CURVATURE_SHIFT_FRACTION",
          2.0));
}

double nonredundant_preconditioner_min_curvature() {
  return std::max(
      1.0e-12,
      parse_env_double_with_default(
          "XMVB_CPP_NONREDUNDANT_PRECONDITIONER_MIN_CURVATURE",
          1.0e-3));
}

double nonredundant_preconditioner_max_curvature() {
  return std::max(
      nonredundant_preconditioner_min_curvature(),
      parse_env_double_with_default(
          "XMVB_CPP_NONREDUNDANT_PRECONDITIONER_MAX_CURVATURE",
          1.0e2));
}


struct BlockOrbitalData {
  int orbital_index = 0;
  int coefficient_count = 0;
  std::vector<int> block_row_for_slot;
};

struct BlockLocalSpace {
  int block_basis_count = 0;
  int n_inactive = 0;
  int n_occupied = 0;
  std::vector<int> basis_function_indices;
  std::vector<BlockOrbitalData> orbitals;
  Eigen::MatrixXd virtual_orbitals;
};

struct BlockDirectionLayout {
  int n_inactive = 0;
  int n_occupied = 0;
  int n_active = 0;
  int n_virtual = 0;
  int inactive_active_count = 0;
  int active_active_count = 0;
  int occupied_virtual_offset = 0;
  int direction_count = 0;
};

// Extract the AO metric seen by one sparse orbital on its own explicit support.
Eigen::MatrixXd build_local_sparse_overlap_metric(
    const Eigen::MatrixXd& block_overlap_matrix,
    const std::vector<int>& block_rows) {
  const Eigen::Index local_size =
      static_cast<Eigen::Index>(block_rows.size());
  Eigen::MatrixXd local_overlap =
      Eigen::MatrixXd::Zero(local_size, local_size);
  for (Eigen::Index row = 0; row < local_size; ++row) {
    const int block_row = block_rows[static_cast<std::size_t>(row)];
    for (Eigen::Index column = 0; column < local_size; ++column) {
      local_overlap(row, column) =
          block_overlap_matrix(
              block_row,
              block_rows[static_cast<std::size_t>(column)]);
    }
  }
  return local_overlap;
}

// Project one raw sparse increment to the local normalization tangent and
// retract it back to the same fixed-support chart while preserving the
// accepted-point raw norm.  This keeps `retract_step(0)` equal to the current
// stored coefficients instead of silently recentering the chart at the unit-
// norm representative.
Eigen::VectorXd retract_local_sparse_normalized_step(
    const Eigen::MatrixXd& local_overlap,
    const Eigen::VectorXd& local_coefficients,
    const Eigen::VectorXd& local_step) {
  const Eigen::VectorXd metric_times_coefficients =
      local_overlap * local_coefficients;
  const double squared_norm =
      local_coefficients.dot(metric_times_coefficients);
  if (!std::isfinite(squared_norm) ||
      squared_norm <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
        "sparse-support retraction encountered a non-positive local orbital norm");
  }

  const double rho = std::sqrt(squared_norm);
  const Eigen::VectorXd normalized_orbital =
      local_coefficients / rho;
  const Eigen::VectorXd metric_times_step =
      local_overlap * local_step;
  const double tangent_overlap =
      normalized_orbital.dot(metric_times_step);
  const Eigen::VectorXd tangent_step =
      local_step - normalized_orbital * tangent_overlap;
  Eigen::VectorXd trial_orbital =
      local_coefficients + tangent_step;
  const double trial_squared_norm =
      trial_orbital.dot(local_overlap * trial_orbital);
  if (!std::isfinite(trial_squared_norm) ||
      trial_squared_norm <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
        "sparse-support retraction produced a non-positive trial orbital norm");
  }

  trial_orbital *= rho / std::sqrt(trial_squared_norm);
  if (!trial_orbital.allFinite()) {
    throw std::runtime_error(
        "sparse-support retraction produced non-finite local coefficients");
  }
  return trial_orbital;
}

// Linearize `retract_local_sparse_normalized_step()` at zero step.  Because
// the finite retraction preserves the accepted-point raw norm, the first-order
// stored-coefficient tangent is just the support-local normalization-tangent
// projection of the raw increment.
Eigen::VectorXd linearize_local_sparse_normalized_step(
    const Eigen::MatrixXd& local_overlap,
    const Eigen::VectorXd& local_coefficients,
    const Eigen::VectorXd& local_step) {
  const Eigen::VectorXd metric_times_coefficients =
      local_overlap * local_coefficients;
  const double squared_norm =
      local_coefficients.dot(metric_times_coefficients);
  if (!std::isfinite(squared_norm) ||
      squared_norm <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
        "sparse-support tangent linearization encountered a non-positive local orbital norm");
  }

  const double rho = std::sqrt(squared_norm);
  const Eigen::VectorXd normalized_orbital =
      local_coefficients / rho;
  const double tangent_overlap =
      normalized_orbital.dot(local_overlap * local_step);
  return local_step - normalized_orbital * tangent_overlap;
}

std::vector<int> build_block_basis_function_indices(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<int>& block_orbitals) {
  if (block_orbitals.empty()) {
    return {};
  }

  std::vector<int> basis_function_indices;
  basis_function_indices.reserve(
      orbital_preparation_input.n_basis_functions);
  std::vector<char> seen_basis_functions(
      orbital_preparation_input.n_basis_functions,
      0);

  // Partial-overlap blocks do not admit a single representative sparse row.
  // Build the block-local AO support directly as the union of every orbital's
  // explicit basis functions, preserving first appearance order so the reduced
  // space stays close to the legacy sparse layout when possible.
  for (const int orbital_index : block_orbitals) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
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
        throw std::runtime_error("invalid block basis-function index");
      }
      if (seen_basis_functions[basis_function_index] != 0) {
        continue;
      }
      seen_basis_functions[basis_function_index] = 1;
      basis_function_indices.push_back(basis_function_index);
    }
  }
  return basis_function_indices;
}

BlockOrbitalData build_block_orbital_data(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index,
    const std::unordered_map<int, int>& basis_to_block_row) {
  BlockOrbitalData result;
  result.orbital_index = orbital_index;
  result.coefficient_count =
      stored_sparse_orbital_coefficient_count(
          orbital_preparation_input,
          orbital_index);
  result.block_row_for_slot.resize(result.coefficient_count, -1);

  for (int coefficient_index = 0;
       coefficient_index < result.coefficient_count;
       ++coefficient_index) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [orbital_index *
                 orbital_preparation_input.n_basis_functions +
             coefficient_index] -
        1;
    const auto iterator = basis_to_block_row.find(basis_function_index);
    if (iterator == basis_to_block_row.end()) {
      throw std::runtime_error(
          "orbital support does not match representative block support");
    }
    const int block_row = iterator->second;
    result.block_row_for_slot[coefficient_index] = block_row;
  }

  return result;
}

Eigen::VectorXd gather_block_orbital_from_dense_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_matrix,
    int orbital_index,
    const std::vector<int>& block_basis_function_indices) {
  if (orbital_index < 0 || orbital_index >= orbital_matrix.cols()) {
    throw std::invalid_argument("orbital index is out of range for dense block gather");
  }

  Eigen::VectorXd block_orbital =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(block_basis_function_indices.size()));
  for (int block_row = 0;
       block_row < static_cast<int>(block_basis_function_indices.size());
       ++block_row) {
    const int basis_function_index =
        block_basis_function_indices[block_row];
    if (basis_function_index < 0 || basis_function_index >= orbital_matrix.rows()) {
      throw std::runtime_error(
          "invalid block basis-function index while gathering dense orbital");
    }
    block_orbital(block_row) = orbital_matrix(basis_function_index, orbital_index);
  }
  return block_orbital;
}

Eigen::MatrixXd build_block_virtual_orbitals(
    const Eigen::MatrixXd& block_occupied_orbitals,
    const Eigen::MatrixXd& block_overlap_matrix) 
{    
  const int block_basis_count = static_cast<int>(block_overlap_matrix.rows());
  const int n_occupied = static_cast<int>(block_occupied_orbitals.cols());
  const int n_virtual = block_basis_count - n_occupied;
  if (n_virtual <= 0) {
    return Eigen::MatrixXd::Zero(block_basis_count, 0);
  }

  // Match legacy `genVirtualOrb` exactly: build the occupied-space projector
  // `P_occ = C_occ (C_occ^T S C_occ)^{-1} C_occ^T S`, diagonalize the metric
  // of its orthogonal complement, and normalize the positive-eigenvalue
  // directions.
  //
  // This is not a cosmetic gauge choice.  In the dense full-support / OEO
  // nonredundant path the virtual columns define the accepted-point chart used
  // later by `putDeltaOrb`-style finite updates.  "Same virtual subspace" is
  // therefore not sufficient: replacing this projector/eigensystem basis by an
  // arbitrary QR `S`-orthogonal complement changes the concrete reduced
  // coordinates seen by the optimizer and can steer FeCl/TiCl-class open-shell
  // cases toward a different localized active-orbital representative even when
  // the subspace itself is unchanged.
  const Eigen::MatrixXd occupied_overlap =
      block_occupied_orbitals.transpose() *
      block_overlap_matrix *
      block_occupied_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> occupied_overlap_factorization(occupied_overlap);
  if (occupied_overlap_factorization.info() != Eigen::Success) {
    throw std::runtime_error("failed to factorize occupied block overlap");
  }
  const Eigen::MatrixXd occupied_overlap_inverse =
      occupied_overlap_factorization.solve(
          Eigen::MatrixXd::Identity(n_occupied, n_occupied));
  if (occupied_overlap_factorization.info() != Eigen::Success) {
    throw std::runtime_error("failed to invert occupied block overlap");
  }
  const Eigen::MatrixXd occupied_metric_action =
      occupied_overlap_inverse *
      block_occupied_orbitals.transpose() *
      block_overlap_matrix;

  Eigen::MatrixXd complementary_projector =
      Eigen::MatrixXd::Identity(block_basis_count, block_basis_count) -
      block_occupied_orbitals * occupied_metric_action;

  const Eigen::MatrixXd virtual_overlap =
      complementary_projector.transpose() *
      block_overlap_matrix *
      complementary_projector;

  const LegacyJacobiDiagonalizationResult eigenpairs =
      diagonalize_self_adjoint_legacy_jacobi(virtual_overlap);

  constexpr double kVirtualEigenvalueTolerance = 1.0e-10;
  Eigen::MatrixXd virtual_orbitals =
      Eigen::MatrixXd::Zero(block_basis_count, n_virtual);
  int virtual_column = 0;
  for (int eigen_index = 0;
       eigen_index < eigenpairs.eigenvalues.size();
       ++eigen_index) {
    const double eigenvalue = eigenpairs.eigenvalues[eigen_index];
    if (!(eigenvalue > kVirtualEigenvalueTolerance)) {
      continue;
    }
    if (virtual_column >= n_virtual) {
      break;
    }

    Eigen::VectorXd virtual_orbital =
        complementary_projector * eigenpairs.eigenvectors.col(eigen_index);
    const double virtual_norm =
        virtual_orbital.dot(block_overlap_matrix * virtual_orbital);
    if (!(virtual_norm > std::numeric_limits<double>::epsilon()) ||
        !std::isfinite(virtual_norm)) {
      continue;
    }

    virtual_orbitals.col(virtual_column) =
        virtual_orbital / std::sqrt(virtual_norm);
    ++virtual_column;
  }
  if (virtual_column != n_virtual) {
    throw std::runtime_error("failed to build the full block-local virtual space");
  }
  if (!virtual_orbitals.allFinite()) {
    throw std::runtime_error("failed to build a finite block-local virtual space");
  }
  return virtual_orbitals;
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
    powered_eigenvalues(index) = std::pow(eigenvalue, exponent);
  }

  return eigen_solver.eigenvectors() *
      powered_eigenvalues.asDiagonal() *
      eigen_solver.eigenvectors().transpose();
}

int candidate_direction_count(const BlockLocalSpace& block_space) {
  const int n_active = block_space.n_occupied - block_space.n_inactive;
  const int n_virtual = static_cast<int>(block_space.virtual_orbitals.cols());
  const int inactive_active_count = block_space.n_inactive * n_active;
  const int active_active_count = n_active * n_active;
  return inactive_active_count +
      active_active_count +
      block_space.n_occupied * n_virtual;
}

BlockDirectionLayout build_block_direction_layout(
    int n_inactive,
    int n_occupied,
    int n_virtual) {
  BlockDirectionLayout layout;
  layout.n_inactive = n_inactive;
  layout.n_occupied = n_occupied;
  layout.n_active = n_occupied - n_inactive;
  layout.n_virtual = n_virtual;
  layout.inactive_active_count = n_inactive * layout.n_active;
  layout.active_active_count = layout.n_active * layout.n_active;
  layout.occupied_virtual_offset =
      layout.inactive_active_count + layout.active_active_count;
  layout.direction_count =
      layout.occupied_virtual_offset + n_occupied * n_virtual;
  return layout;
}

int inactive_active_direction_index(
    int inactive_index,
    int active_index,
    int n_inactive,
    int n_occupied) {
  const int n_active = n_occupied - n_inactive;
  return inactive_index * n_active + (active_index - n_inactive);
}

int active_active_direction_index(
    int source_active_index,
    int target_active_index,
    int n_inactive,
    int n_occupied) {
  const int n_active = n_occupied - n_inactive;
  const int inactive_active_count = n_inactive * n_active;
  const int source_active_offset = source_active_index - n_inactive;
  const int target_active_offset = target_active_index - n_inactive;
  return inactive_active_count +
      source_active_offset +
      target_active_offset * n_active;
}

int occupied_virtual_direction_index(
    int occupied_index,
    int virtual_index,
    int n_inactive,
    int n_occupied,
    int n_virtual) {
  const BlockDirectionLayout layout =
      build_block_direction_layout(n_inactive, n_occupied, n_virtual);
  return layout.occupied_virtual_offset +
      occupied_index * n_virtual +
      virtual_index;
}

Eigen::VectorXd build_candidate_metric_diagonal(
    const std::vector<Eigen::MatrixXd>& occupied_occupied_metrics,
    const std::vector<Eigen::MatrixXd>& virtual_virtual_metrics,
    int n_inactive,
    int n_occupied,
    int n_virtual) {
  const BlockDirectionLayout layout =
      build_block_direction_layout(n_inactive, n_occupied, n_virtual);
  const int direction_count = layout.direction_count;
  Eigen::VectorXd metric_diagonal =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(direction_count));

  for (int inactive_index = 0;
       inactive_index < n_inactive;
       ++inactive_index) {
    for (int active_index = n_inactive;
         active_index < n_occupied;
         ++active_index) {
      const int direction_index =
          inactive_active_direction_index(
              inactive_index,
              active_index,
              n_inactive,
              n_occupied);
      metric_diagonal(direction_index) =
          occupied_occupied_metrics[inactive_index](
              active_index,
              active_index) +
          occupied_occupied_metrics[active_index](
              inactive_index,
              inactive_index);
    }
  }

  for (int target_active_index = n_inactive;
       target_active_index < n_occupied;
       ++target_active_index) {
    for (int source_active_index = n_inactive;
         source_active_index < n_occupied;
         ++source_active_index) {
      const int direction_index =
          active_active_direction_index(
              source_active_index,
              target_active_index,
              n_inactive,
              n_occupied);
      metric_diagonal(direction_index) =
          occupied_occupied_metrics[target_active_index](
              source_active_index,
              source_active_index);
    }
  }

  for (int occupied_index = 0;
       occupied_index < n_occupied;
       ++occupied_index) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      const int direction_index =
          occupied_virtual_direction_index(
              occupied_index,
              virtual_index,
              n_inactive,
              n_occupied,
              n_virtual);
      metric_diagonal(direction_index) =
          virtual_virtual_metrics[occupied_index](
              virtual_index,
              virtual_index);
    }
  }

  return metric_diagonal;
}

Eigen::VectorXd build_candidate_metric_diagonal_from_common_grams(
    const Eigen::MatrixXd& occupied_occupied_metric,
    const Eigen::MatrixXd& virtual_virtual_metric,
    int n_inactive,
    int n_occupied,
    int n_virtual) {
  const BlockDirectionLayout layout =
      build_block_direction_layout(n_inactive, n_occupied, n_virtual);
  const int direction_count = layout.direction_count;
  Eigen::VectorXd metric_diagonal =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(direction_count));

  for (int inactive_index = 0;
       inactive_index < n_inactive;
       ++inactive_index) {
    for (int active_index = n_inactive;
         active_index < n_occupied;
         ++active_index) {
      const int direction_index =
          inactive_active_direction_index(
              inactive_index,
              active_index,
              n_inactive,
              n_occupied);
      metric_diagonal(direction_index) =
          occupied_occupied_metric(active_index, active_index) +
          occupied_occupied_metric(inactive_index, inactive_index);
    }
  }

  for (int target_active_index = n_inactive;
       target_active_index < n_occupied;
       ++target_active_index) {
    for (int source_active_index = n_inactive;
         source_active_index < n_occupied;
         ++source_active_index) {
      const int direction_index =
          active_active_direction_index(
              source_active_index,
              target_active_index,
              n_inactive,
              n_occupied);
      metric_diagonal(direction_index) =
          occupied_occupied_metric(
              source_active_index,
              source_active_index);
    }
  }

  for (int occupied_index = 0;
       occupied_index < n_occupied;
       ++occupied_index) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      const int direction_index =
          occupied_virtual_direction_index(
              occupied_index,
              virtual_index,
              n_inactive,
              n_occupied,
              n_virtual);
      metric_diagonal(direction_index) =
          virtual_virtual_metric(virtual_index, virtual_index);
    }
  }

  return metric_diagonal;
}

Eigen::MatrixXd build_block_effective_one_electron_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    int n_basis_functions,
    const std::vector<int>& basis_function_indices) {

  const int block_basis_count = static_cast<int>(basis_function_indices.size());

  Eigen::MatrixXd block_effective_one_electron =
      Eigen::MatrixXd::Zero(block_basis_count, block_basis_count);

  // `ao_effective_h1e` is the full AO `F11`. Extracting
  // the block-local submatrix lets us estimate orbital-energy denominators using
  // the same local support that defines the nonredundant directions.
  for (int row = 0; row < block_basis_count; ++row) {
    const int ao_row = basis_function_indices[row];
    for (int column = 0; column < block_basis_count; ++column) {
      const int ao_column = basis_function_indices[column];
      block_effective_one_electron(row, column) =
          ao_effective_h1e(ao_row, ao_column);
    }
  }
  return block_effective_one_electron;
}

Eigen::VectorXd build_candidate_curvature_diagonal(
    const BlockLocalSpace& block_space,
    const Eigen::MatrixXd& block_occupied_orbitals,
    const Eigen::MatrixXd& block_effective_one_electron) {
  const int n_inactive = block_space.n_inactive;
  const int n_occupied = block_space.n_occupied;
  const int n_virtual = static_cast<int>(block_space.virtual_orbitals.cols());
  const int direction_count = candidate_direction_count(block_space);
  Eigen::VectorXd candidate_curvature =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(direction_count));
  if (direction_count == 0) {
    return candidate_curvature;
  }

  Eigen::VectorXd occupied_energies =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_occupied));

  for (int occupied_index = 0;
       occupied_index < n_occupied;
       ++occupied_index) {
    const Eigen::VectorXd orbital = block_occupied_orbitals.col(occupied_index);
    occupied_energies(occupied_index) =
        orbital.dot(block_effective_one_electron * orbital);
  }

  Eigen::VectorXd virtual_energies =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_virtual));
  for (int virtual_index = 0;
       virtual_index < n_virtual;
       ++virtual_index) {
    const Eigen::VectorXd orbital = block_space.virtual_orbitals.col(virtual_index);
    virtual_energies(virtual_index) =
        orbital.dot(block_effective_one_electron * orbital);
  }

  constexpr double kMinimumCandidateCurvature = 1.0e-8;
  auto positive_gap_average =
      [](double gap_sum, int gap_count, double fallback_value) {
        return gap_count > 0
            ? gap_sum / static_cast<double>(gap_count)
            : fallback_value;
      };
  double active_shape_reference_gap_sum = 0.0;
  int active_shape_reference_gap_count = 0;
  for (int active_index = n_inactive;
       active_index < n_occupied;
       ++active_index) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      active_shape_reference_gap_sum +=
          std::abs(
              virtual_energies(virtual_index) -
              occupied_energies(active_index));
      ++active_shape_reference_gap_count;
    }
  }
  if (active_shape_reference_gap_count == 0) {
    for (int inactive_index = 0;
         inactive_index < n_inactive;
         ++inactive_index) {
      for (int active_index = n_inactive;
           active_index < n_occupied;
           ++active_index) {
        active_shape_reference_gap_sum +=
            std::abs(
                occupied_energies(active_index) -
                occupied_energies(inactive_index));
        ++active_shape_reference_gap_count;
      }
    }
  }
  const double active_shape_reference_gap =
      positive_gap_average(
          active_shape_reference_gap_sum,
          active_shape_reference_gap_count,
          1.0);
  const double active_shape_curvature_shift =
      std::max(
          kMinimumCandidateCurvature,
          active_shape_curvature_shift_fraction() *
              active_shape_reference_gap);

  for (int inactive_index = 0;
       inactive_index < n_inactive;
       ++inactive_index) {
    for (int active_index = n_inactive;
         active_index < n_occupied;
         ++active_index) {
      const int direction_index =
          inactive_active_direction_index(
              inactive_index,
              active_index,
              n_inactive,
              n_occupied);
      candidate_curvature(direction_index) =
          std::max(
              kMinimumCandidateCurvature,
              std::abs(
                  occupied_energies(active_index) -
                  occupied_energies(inactive_index)));
    }
  }

  for (int target_active_index = n_inactive;
       target_active_index < n_occupied;
       ++target_active_index) {
    for (int source_active_index = n_inactive;
         source_active_index < n_occupied;
         ++source_active_index) {
      const int direction_index =
          active_active_direction_index(
              source_active_index,
              target_active_index,
              n_inactive,
              n_occupied);
      // The explicit active-active block in the nonredundant VB chart is the
      // physical active-shape variable, not a pure orthogonal orbital
      // rotation. Near-degenerate active orbital energies therefore make the
      // usual denominator `|eps_p - eps_q|` artificially collapse exactly
      // where the local geometry is most ill-conditioned. Use a positive
      // Levenberg-Marquardt-style shift tied to the surrounding active gaps
      // instead of the nearly zero raw energy difference.
      candidate_curvature(direction_index) =
          active_shape_curvature_shift;
    }
  }

  for (int occupied_index = 0;
       occupied_index < n_occupied;
       ++occupied_index) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      const int direction_index =
          occupied_virtual_direction_index(
              occupied_index,
              virtual_index,
              n_inactive,
              n_occupied,
              n_virtual);
      candidate_curvature(direction_index) =
          std::max(
              kMinimumCandidateCurvature,
              std::abs(
                  virtual_energies(virtual_index) -
                  occupied_energies(occupied_index)));
    }
  }

  return candidate_curvature;
}

Eigen::VectorXd normalize_curvature_diagonal(
    const Eigen::VectorXd& curvature_diagonal) {
  if (curvature_diagonal.size() == 0) {
    return Eigen::VectorXd::Zero(0);
  }

  Eigen::VectorXd normalized = curvature_diagonal;
  const double minimum_curvature =
      nonredundant_preconditioner_min_curvature();
  const double maximum_curvature =
      nonredundant_preconditioner_max_curvature();
  // The TN/L-BFGS preconditioner should approximate the local Hessian
  // diagonal itself, not a mean-rescaled surrogate. Mean normalization flattens
  // precisely the gap information that PCG needs on stiff VB valleys. Keep the
  // physical block-local scale and only apply a positive LM-style clamp.
  for (Eigen::Index index = 0; index < normalized.size(); ++index) {
    double value = normalized(index);
    if (!std::isfinite(value) || value <= 0.0) {
      value = 1.0;
    }
    normalized(index) =
        std::clamp(
            value,
            minimum_curvature,
            maximum_curvature);
  }
  return normalized;
}

}  // namespace

NonredundantOrbitalSpace::NonredundantOrbitalSpace(
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbital_basis_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& physical_orbital_matrix,
    const Eigen::MatrixXd* ao_effective_h1e)
    : packed_parameter_size_(parameter_view.size()) {
  if (ao_effective_h1e != nullptr &&
      (ao_effective_h1e->rows() != orbital_preparation_input.n_basis_functions ||
       ao_effective_h1e->cols() != orbital_preparation_input.n_basis_functions)) {
    throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_occupied_orbitals =
      n_inactive_doubly_occupied_orbitals +
      orbital_preparation_input.n_active_orbitals;
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_occupied_orbitals > orbital_preparation_input.n_orbitals) {
    throw std::invalid_argument("invalid occupied-space partition for nonredundant optimizer");
  }
  if (occupied_orbital_basis_matrix.rows() != orbital_preparation_input.n_basis_functions ||
      occupied_orbital_basis_matrix.cols() != n_occupied_orbitals) {
    throw std::invalid_argument("occupied orbital basis matrix shape mismatch");
  }
  if (physical_orbital_matrix.rows() != orbital_preparation_input.n_basis_functions ||
      physical_orbital_matrix.cols() != orbital_preparation_input.n_orbitals) {
    throw std::invalid_argument("physical orbital matrix shape mismatch");
  }
  use_block_preconditioner_by_default_ =
      !(orbital_preparation_input.spin_multiplicity > 1 &&
        orbital_preparation_input.orbital_type == kLegacyOrbitalTypeOeo &&
        orbital_preparation_input.n_active_orbitals > 0 &&
        orbital_preparation_input.n_active_orbitals <= 7);

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
      orbital_preparation_input.active_orbital_overlap_matrix.data(),
      orbital_preparation_input.n_basis_functions,
      orbital_preparation_input.n_basis_functions);
  const auto blocks = detect_orbital_blocks(orbital_preparation_input);

  // Each block is treated independently: build the occupied block span in the
  // AO metric, construct its virtual complement, then translate the allowed
  // rotations back into the packed sparse-coefficient coordinates seen by the
  // optimizer.
  for (const auto& block : blocks) {
    std::vector<int> occupied_block_orbitals;
    occupied_block_orbitals.reserve(block.size());
    for (const int orbital_index : block) {
      if (orbital_index < n_occupied_orbitals) {
        occupied_block_orbitals.push_back(orbital_index);
      }
    }
    if (occupied_block_orbitals.empty()) {
      continue;
    }

    const std::vector<int> block_basis_function_indices =
        build_block_basis_function_indices(
            orbital_preparation_input,
            block);
    const int block_basis_count =
        static_cast<int>(block_basis_function_indices.size());
    if (block_basis_count < static_cast<int>(occupied_block_orbitals.size())) {
      throw std::runtime_error("block basis count is smaller than occupied block size");
    }

    BlockLocalSpace block_space;
    block_space.block_basis_count = block_basis_count;
    block_space.n_occupied = static_cast<int>(occupied_block_orbitals.size());
    block_space.n_inactive = 0;
    block_space.basis_function_indices = block_basis_function_indices;
    std::unordered_map<int, int> basis_to_block_row;
    basis_to_block_row.reserve(block_space.basis_function_indices.size());
    for (int block_row = 0; block_row < block_basis_count; ++block_row) {
      basis_to_block_row.emplace(
          block_space.basis_function_indices[block_row],
          block_row);
    }

    Eigen::MatrixXd block_raw_occupied_orbitals =
        Eigen::MatrixXd::Zero(block_basis_count, block_space.n_occupied);
    Eigen::MatrixXd block_occupied_orbitals =
        Eigen::MatrixXd::Zero(block_basis_count, block_space.n_occupied);

    Eigen::MatrixXd block_reference_occupied_orbitals =
        Eigen::MatrixXd::Zero(block_basis_count, block_space.n_occupied);

    for (int local_orbital_index = 0;
         local_orbital_index < block_space.n_occupied;
         ++local_orbital_index) {
      const int orbital_index =
          occupied_block_orbitals[local_orbital_index];
      const int coefficient_count =
          stored_sparse_orbital_coefficient_count(
              orbital_preparation_input,
              orbital_index);
      for (int coefficient_index = 0;
           coefficient_index < coefficient_count;
           ++coefficient_index) {
        const int block_row =
            basis_to_block_row.at(
                orbital_preparation_input.orbital_basis_index_table
                    [orbital_index *
                         orbital_preparation_input.n_basis_functions +
                     coefficient_index] -
                1);
        block_raw_occupied_orbitals(
            block_row,
            local_orbital_index) =
            orbital_preparation_input.orbital_value_table
                [orbital_index *
                     orbital_preparation_input.n_basis_functions +
                 coefficient_index];
      }
      block_occupied_orbitals.col(local_orbital_index) =
          gather_block_orbital_from_dense_matrix(
              occupied_orbital_basis_matrix,
              orbital_index,
              block_basis_function_indices);
      block_reference_occupied_orbitals.col(local_orbital_index) =
          gather_block_orbital_from_dense_matrix(
              physical_orbital_matrix,
              orbital_index,
              block_basis_function_indices);
      auto orbital_data = build_block_orbital_data(
          orbital_preparation_input,
          orbital_index,
          basis_to_block_row);
      if (orbital_index < n_inactive_doubly_occupied_orbitals) {
        ++block_space.n_inactive;
      }
      block_space.orbitals.push_back(std::move(orbital_data));
    }

    Eigen::MatrixXd block_overlap =
        Eigen::MatrixXd::Zero(block_basis_count, block_basis_count);
    for (int row = 0; row < block_basis_count; ++row) {
      for (int column = 0; column < block_basis_count; ++column) {
        block_overlap(row, column) =
            basis_overlap_matrix(
                block_space.basis_function_indices[row],
                block_space.basis_function_indices[column]);
      }
    }
    // Keep the block virtual basis on the same accepted-point physical chart
    // as the occupied columns used by the sparse-support finite-step
    // retraction.  Mixing a virtual complement built from the auxiliary
    // occupied block with occupied updates built from the physical occupied
    // block makes the reduced chart internally inconsistent on HAO systems:
    // `project_gradient()` and `retract_step()` then see different
    // occupied/virtual splittings, which is exactly the MnF2 failure mode.
    block_space.virtual_orbitals =
        build_block_virtual_orbitals(
            block_raw_occupied_orbitals,
            block_overlap);

    const int n_virtual = static_cast<int>(block_space.virtual_orbitals.cols());
    const int direction_count = candidate_direction_count(block_space);
    if (direction_count == 0) {
      continue;
    }

    BlockBasis block_basis;
    block_basis.n_inactive = block_space.n_inactive;
    block_basis.n_occupied = block_space.n_occupied;
    block_basis.n_virtual = n_virtual;
    block_basis.basis_function_indices = block_space.basis_function_indices;
    block_basis.occupied_orbital_indices = occupied_block_orbitals;
    block_basis.block_overlap_matrix = block_overlap;
    block_basis.occupied_orbitals = block_raw_occupied_orbitals;
    block_basis.reference_occupied_orbitals =
        block_reference_occupied_orbitals;
    block_basis.virtual_orbitals = block_space.virtual_orbitals;
    // The dense full-support mixed chart is an OEO-specific construction.  It
    // rewrites the accepted point into `(Q_i, Q_a, L_a)` with an auxiliary-
    // active shape matrix and is only valid when the block orbitals really
    // follow that OEO semantics.  HAO blocks may also become full-support after
    // nonredundant support adaptation, but forcing them through the OEO chart
    // breaks the reduced finite-step manifold, which is exactly what happened
    // for MnF2.  Keep non-OEO blocks on the generic sparse-support chart even
    // when every orbital spans the whole block support.
    block_basis.uses_dense_full_support_projector =
        orbital_preparation_input.orbital_type == kLegacyOrbitalTypeOeo;
    block_basis.orbitals.reserve(block_space.orbitals.size());
    for (const auto& orbital : block_space.orbitals) {
      if (orbital.coefficient_count != block_basis_count) {
        block_basis.uses_dense_full_support_projector = false;
        break;
      }
    }

    // `occupied_masked` / `virtual_masked` keep the legacy raw sparse chart
    // used by the current sparse finite-step path.  The mixed-chart migration
    // also needs per-orbital masked copies of `(Q_i, Q_a, Q_v)` so sparse
    // `project_vector()` / `expand_step()` can use the same block-local chart
    // as the dense full-support formulas without touching the fixed-support
    // storage layout yet.
    for (int local_orbital_index = 0;
         local_orbital_index < block_space.n_occupied;
         ++local_orbital_index) {
      const auto& orbital = block_space.orbitals[local_orbital_index];
      OrbitalProjector projector;
      projector.packed_indices.reserve(orbital.coefficient_count);
      projector.block_rows.reserve(orbital.coefficient_count);
      if (block_basis.uses_dense_full_support_projector) {
        projector.flat_index_by_block_row.assign(block_basis_count, -1);
        projector.packed_index_by_block_row.assign(block_basis_count, -1);
      } else {
        projector.occupied_masked =
            Eigen::MatrixXd::Zero(orbital.coefficient_count, block_space.n_occupied);
        projector.inactive_working_masked =
            Eigen::MatrixXd::Zero(orbital.coefficient_count, block_space.n_inactive);
        projector.active_working_masked =
            Eigen::MatrixXd::Zero(
                orbital.coefficient_count,
                block_space.n_occupied - block_space.n_inactive);
        projector.virtual_masked =
            Eigen::MatrixXd::Zero(orbital.coefficient_count, n_virtual);
        projector.internal_virtual_masked =
            Eigen::MatrixXd::Zero(orbital.coefficient_count, n_virtual);
      }
      for (int coefficient_index = 0;
           coefficient_index < orbital.coefficient_count;
           ++coefficient_index) {
        const int flat_index =
            orbital.orbital_index * orbital_preparation_input.n_basis_functions +
            coefficient_index;
        const int packed_index =
            parameter_view.packed_index(orbital.orbital_index, coefficient_index);
        if (packed_index < 0) {
          throw std::runtime_error("failed to locate packed sparse-orbital parameter");
        }
        projector.flat_indices.push_back(flat_index);
        projector.packed_indices.push_back(packed_index);
        const int block_row =
            orbital.block_row_for_slot[coefficient_index];
        projector.block_rows.push_back(block_row);
        if (block_basis.uses_dense_full_support_projector) {
          projector.flat_index_by_block_row[block_row] = flat_index;
          projector.packed_index_by_block_row[block_row] = packed_index;
        } else {
          projector.occupied_masked.row(coefficient_index) =
              block_raw_occupied_orbitals.row(block_row);
          if (n_virtual > 0) {
            projector.virtual_masked.row(coefficient_index) =
                block_space.virtual_orbitals.row(block_row);
          }
        }
      }
      if (block_basis.uses_dense_full_support_projector) {
        for (int block_row = 0; block_row < block_basis_count; ++block_row) {
          if (projector.flat_index_by_block_row[block_row] < 0 ||
              projector.packed_index_by_block_row[block_row] < 0) {
            throw std::runtime_error(
                "dense full-support projector is missing a block row");
          }
        }
      }
      block_basis.orbitals.push_back(std::move(projector));
    }

    initialize_block_mixed_chart_cache(
        &block_basis,
        block_occupied_orbitals);

    if (!block_basis.uses_dense_full_support_projector) {
      const int n_active = block_basis.n_occupied - block_basis.n_inactive;
      for (auto& projector : block_basis.orbitals) {
        for (Eigen::Index local_index = 0;
             local_index < static_cast<Eigen::Index>(projector.block_rows.size());
             ++local_index) {
          const int block_row = projector.block_rows[local_index];
          if (block_basis.n_inactive > 0) {
            projector.inactive_working_masked.row(local_index) =
                block_basis.inactive_working_orbitals.row(block_row);
          }
          if (n_active > 0) {
            projector.active_working_masked.row(local_index) =
                block_basis.active_working_orbitals.row(block_row);
          }
          if (block_basis.n_virtual > 0) {
            projector.internal_virtual_masked.row(local_index) =
                block_basis.internal_virtual_orbitals.row(block_row);
          }
        }
      }
    }

    if (block_basis.uses_dense_full_support_projector) {
      const int n_active = block_basis.n_occupied - block_basis.n_inactive;
      Eigen::MatrixXd internal_occupied_orbitals =
          Eigen::MatrixXd::Zero(block_basis_count, block_basis.n_occupied);
      if (block_basis.n_inactive > 0) {
        internal_occupied_orbitals.leftCols(block_basis.n_inactive) =
            block_basis.inactive_working_orbitals;
      }
      if (n_active > 0) {
        internal_occupied_orbitals.middleCols(
            block_basis.n_inactive,
            n_active) = block_basis.active_working_orbitals;
      }
      block_basis.occupied_orbitals = internal_occupied_orbitals;
      block_basis.virtual_orbitals = block_basis.internal_virtual_orbitals;
      initialize_dense_full_support_metric_cache(&block_basis);
    }

    // Keep the reduced chart in the raw block-local candidate amplitudes `a`
    // defined by the implicit direction matrix `D`. Rebuilding the accepted
    // point should not allocate the dense Gram matrix `D^T D` or run an
    // `O(N^3)` eigendecomposition on it. All metric actions for this chart stay
    // implicit through `project_block_candidate_overlap` and
    // `accumulate_block_candidate_combination`.
    if (block_basis.uses_dense_full_support_projector) {
      block_basis.candidate_metric_diagonal =
          build_dense_full_support_metric_diagonal(block_basis);
    } else {
      block_basis.candidate_metric_diagonal =
          build_sparse_mixed_chart_metric_diagonal(block_basis);
    }
    block_basis.reduced_offset = reduced_size_;
    reduced_size_ += direction_count;

    if (ao_effective_h1e != nullptr) {
      const Eigen::MatrixXd block_effective_one_electron =
          build_block_effective_one_electron_matrix(
              *ao_effective_h1e,
              orbital_preparation_input.n_basis_functions,
              block_space.basis_function_indices);
      const Eigen::VectorXd candidate_curvature =
          build_candidate_curvature_diagonal(
              block_space,
              block_occupied_orbitals,
              block_effective_one_electron);
      block_basis.reduced_curvature_diagonal =
          normalize_curvature_diagonal(candidate_curvature);
      has_reduced_curvature_diagonal_ =
          has_reduced_curvature_diagonal_ ||
          (block_basis.reduced_curvature_diagonal.size() == direction_count);
    }

    block_bases_.push_back(std::move(block_basis));
    maybe_factorize_small_block_candidate_metric(&block_bases_.back());
  }
}

Eigen::MatrixXd NonredundantOrbitalSpace::gather_dense_full_support_block_matrix(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& packed_vector) const {
  const int block_basis_count = static_cast<int>(block_basis.basis_function_indices.size());
  Eigen::MatrixXd block_matrix =
      Eigen::MatrixXd::Zero(block_basis_count, block_basis.n_occupied);
  for (int occupied_index = 0;
       occupied_index < block_basis.n_occupied;
       ++occupied_index) {
    const auto& projector = block_basis.orbitals[occupied_index];
    for (int block_row = 0; block_row < block_basis_count; ++block_row) {
      const int packed_index =
          projector.packed_index_by_block_row[block_row];
      if (packed_index < 0 || packed_index >= packed_parameter_size_) {
        throw std::runtime_error(
            "dense full-support projector is missing a packed block row");
      }
      block_matrix(block_row, occupied_index) = packed_vector[packed_index];
    }
  }
  return block_matrix;
}

void NonredundantOrbitalSpace::accumulate_dense_full_support_block_matrix(
    const BlockBasis& block_basis,
    const Eigen::MatrixXd& block_matrix,
    Eigen::VectorXd* packed_vector) const {
  const int block_basis_count = static_cast<int>(block_basis.basis_function_indices.size());

  for (int occupied_index = 0;
       occupied_index < block_basis.n_occupied;
       ++occupied_index) {
    const auto& projector = block_basis.orbitals[occupied_index];
    for (int block_row = 0; block_row < block_basis_count; ++block_row) {
      const int packed_index =
          projector.packed_index_by_block_row[block_row];
      if (packed_index < 0 || packed_index >= packed_parameter_size_) {
        throw std::runtime_error(
            "dense full-support projector is missing a packed block row");
      }
      (*packed_vector)[packed_index] += block_matrix(block_row, occupied_index);
    }
  }
}

void NonredundantOrbitalSpace::write_dense_full_support_block_matrix(
    const BlockBasis& block_basis,
    const Eigen::MatrixXd& block_matrix,
    std::vector<double>* orbital_value_table) const {
  const int block_basis_count = static_cast<int>(block_basis.basis_function_indices.size());

  for (int occupied_index = 0;
       occupied_index < block_basis.n_occupied;
       ++occupied_index) {
    const auto& projector = block_basis.orbitals[occupied_index];
    for (int block_row = 0; block_row < block_basis_count; ++block_row) {
      const int flat_index =
          projector.flat_index_by_block_row[block_row];
      if (flat_index < 0 ||
          flat_index >= static_cast<int>(orbital_value_table->size())) {
        throw std::runtime_error(
            "dense full-support projector is missing a flat block row");
      }
      (*orbital_value_table)[flat_index] =
          block_matrix(block_row, occupied_index);
    }
  }
}

void NonredundantOrbitalSpace::initialize_dense_full_support_metric_cache(
    BlockBasis* block_basis) const {
  if (!block_basis->uses_dense_full_support_projector) {
    return;
  }

  // The dense full-support / OEO mixed chart repeatedly applies and solves the
  // small reduced Gram operator built from the gauge matrices
  // `U_i`, `L_a`, and `K_a`. Cache the resulting SPD metrics and their
  // eigendecompositions once at the accepted point so the hot path does not
  // refactor the same `n_i x n_i` / `n_a x n_a` matrices on every PCG step.
  block_basis->inactive_metric_matrix =
      block_basis->inactive_right_transform *
      block_basis->inactive_right_transform.transpose();
  block_basis->active_shape_metric_matrix =
      block_basis->active_shape_matrix *
      block_basis->active_shape_matrix.transpose();
  block_basis->inactive_plus_gauge_metric_matrix =
      block_basis->inactive_metric_matrix +
      block_basis->active_inactive_gauge_coefficients *
          block_basis->active_inactive_gauge_coefficients.transpose();

  if (block_basis->inactive_metric_matrix.rows() > 0) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> inactive_solver(
        block_basis->inactive_metric_matrix);
    if (inactive_solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed eigendecomposition for dense full-support inactive metric");
    }
    block_basis->inactive_metric_eigenvectors =
        inactive_solver.eigenvectors();
    block_basis->inactive_metric_eigenvalues =
        inactive_solver.eigenvalues();
    Eigen::VectorXd inverse_eigenvalues =
        block_basis->inactive_metric_eigenvalues;
    for (Eigen::Index index = 0;
         index < inverse_eigenvalues.size();
         ++index) {
      const double eigenvalue = inverse_eigenvalues(index);
      if (!std::isfinite(eigenvalue) ||
          eigenvalue <= kMinimumDenseMetricEigenvalue) {
        throw std::runtime_error(
            "dense full-support inactive metric is not numerically positive definite");
      }
      inverse_eigenvalues(index) = 1.0 / eigenvalue;
    }
    block_basis->inactive_metric_inverse =
        block_basis->inactive_metric_eigenvectors *
        inverse_eigenvalues.asDiagonal() *
        block_basis->inactive_metric_eigenvectors.transpose();
  } else {
    block_basis->inactive_metric_inverse = Eigen::MatrixXd::Zero(0, 0);
    block_basis->inactive_metric_eigenvectors = Eigen::MatrixXd::Zero(0, 0);
    block_basis->inactive_metric_eigenvalues = Eigen::VectorXd::Zero(0);
  }

  if (block_basis->active_shape_metric_matrix.rows() > 0) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> active_solver(
        block_basis->active_shape_metric_matrix);
    if (active_solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed eigendecomposition for dense full-support active metric");
    }
    block_basis->active_shape_metric_eigenvectors =
        active_solver.eigenvectors();
    block_basis->active_shape_metric_eigenvalues =
        active_solver.eigenvalues();
    Eigen::VectorXd inverse_metric_eigenvalues =
        block_basis->active_shape_metric_eigenvalues;
    for (Eigen::Index index = 0;
         index < inverse_metric_eigenvalues.size();
         ++index) {
      const double eigenvalue = inverse_metric_eigenvalues(index);
      if (!std::isfinite(eigenvalue) ||
          eigenvalue <= kMinimumDenseMetricEigenvalue) {
        throw std::runtime_error(
            "dense full-support active metric is not numerically positive definite");
      }
      inverse_metric_eigenvalues(index) = 1.0 / eigenvalue;
    }
    block_basis->active_shape_metric_inverse =
        block_basis->active_shape_metric_eigenvectors *
        inverse_metric_eigenvalues.asDiagonal() *
        block_basis->active_shape_metric_eigenvectors.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> active_shape_solver(
        block_basis->active_shape_matrix);
    if (active_shape_solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed eigendecomposition for dense full-support active shape");
    }
    Eigen::VectorXd inverse_shape_eigenvalues =
        active_shape_solver.eigenvalues();
    for (Eigen::Index index = 0;
         index < inverse_shape_eigenvalues.size();
         ++index) {
      const double eigenvalue = inverse_shape_eigenvalues(index);
      if (!std::isfinite(eigenvalue) ||
          eigenvalue <= kMinimumDenseMetricEigenvalue) {
        throw std::runtime_error(
            "dense full-support active shape is not numerically positive definite");
      }
      inverse_shape_eigenvalues(index) = 1.0 / eigenvalue;
    }
    block_basis->active_shape_inverse =
        active_shape_solver.eigenvectors() *
        inverse_shape_eigenvalues.asDiagonal() *
        active_shape_solver.eigenvectors().transpose();
  } else {
    block_basis->active_shape_metric_inverse = Eigen::MatrixXd::Zero(0, 0);
    block_basis->active_shape_metric_eigenvectors = Eigen::MatrixXd::Zero(0, 0);
    block_basis->active_shape_metric_eigenvalues = Eigen::VectorXd::Zero(0);
    block_basis->active_shape_inverse = Eigen::MatrixXd::Zero(0, 0);
  }
}

void NonredundantOrbitalSpace::initialize_block_mixed_chart_cache(
    BlockBasis* block_basis,
    const Eigen::Ref<const Eigen::MatrixXd>& auxiliary_occupied_orbitals) const {
  const int block_basis_count =
      static_cast<int>(block_basis->basis_function_indices.size());
  const int n_active =
      block_basis->n_occupied - block_basis->n_inactive;

  if (block_basis->n_inactive > 0) {
    const Eigen::MatrixXd inactive_metric =
        block_basis->reference_occupied_orbitals
            .leftCols(block_basis->n_inactive)
            .transpose() *
        block_basis->block_overlap_matrix *
        block_basis->reference_occupied_orbitals
            .leftCols(block_basis->n_inactive);
    const Eigen::MatrixXd inactive_inverse_square_root =
        build_self_adjoint_matrix_power(
            inactive_metric,
            -0.5,
            "block mixed-chart inactive metric");
    block_basis->inactive_working_orbitals =
        block_basis->reference_occupied_orbitals
            .leftCols(block_basis->n_inactive) *
        inactive_inverse_square_root;
    block_basis->inactive_right_transform =
        build_self_adjoint_matrix_power(
            inactive_metric,
            0.5,
            "block mixed-chart inactive metric");
  } else {
    block_basis->inactive_working_orbitals =
        Eigen::MatrixXd::Zero(block_basis_count, 0);
    block_basis->inactive_right_transform =
        Eigen::MatrixXd::Zero(0, 0);
  }

  if (n_active > 0) {
    const Eigen::MatrixXd active_auxiliary_orbitals =
        auxiliary_occupied_orbitals.middleCols(
            block_basis->n_inactive,
            n_active);
    const Eigen::MatrixXd active_metric =
        active_auxiliary_orbitals.transpose() *
        block_basis->block_overlap_matrix *
        active_auxiliary_orbitals;
    const Eigen::MatrixXd active_inverse_square_root =
        build_self_adjoint_matrix_power(
            active_metric,
            -0.5,
            "block mixed-chart active auxiliary metric");
    block_basis->active_working_orbitals =
        active_auxiliary_orbitals * active_inverse_square_root;
    block_basis->active_shape_matrix =
        build_self_adjoint_matrix_power(
            active_metric,
            0.5,
            "block mixed-chart active auxiliary metric");
    if (block_basis->n_inactive > 0) {
      block_basis->active_inactive_gauge_coefficients =
          block_basis->inactive_working_orbitals.transpose() *
          block_basis->block_overlap_matrix *
          block_basis->reference_occupied_orbitals.middleCols(
              block_basis->n_inactive,
              n_active);
    } else {
      block_basis->active_inactive_gauge_coefficients =
          Eigen::MatrixXd::Zero(0, n_active);
    }
  } else {
    block_basis->active_working_orbitals =
        Eigen::MatrixXd::Zero(block_basis_count, 0);
    block_basis->active_shape_matrix =
        Eigen::MatrixXd::Zero(0, 0);
    block_basis->active_inactive_gauge_coefficients =
        Eigen::MatrixXd::Zero(block_basis->n_inactive, 0);
  }

  Eigen::MatrixXd internal_occupied_orbitals =
      Eigen::MatrixXd::Zero(block_basis_count, block_basis->n_occupied);
  if (block_basis->n_inactive > 0) {
    internal_occupied_orbitals.leftCols(block_basis->n_inactive) =
        block_basis->inactive_working_orbitals;
  }
  if (n_active > 0) {
    internal_occupied_orbitals.middleCols(
        block_basis->n_inactive,
        n_active) = block_basis->active_working_orbitals;
  }
  block_basis->internal_virtual_orbitals =
      build_block_virtual_orbitals(
          internal_occupied_orbitals,
          block_basis->block_overlap_matrix);
}

Eigen::MatrixXd NonredundantOrbitalSpace::build_dense_full_support_block_step(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& candidate_coefficients) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  Eigen::Map<const Eigen::MatrixXd> inactive_active_coefficients(
      candidate_coefficients.data(),
      n_active,
      n_inactive);
  Eigen::Map<const Eigen::MatrixXd> active_shape_step(
      candidate_coefficients.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<const Eigen::MatrixXd> occupied_virtual_coefficients(
      candidate_coefficients.data() + layout.occupied_virtual_offset,
      n_virtual,
      block_basis.n_occupied);

  Eigen::MatrixXd block_step =
      Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(block_basis.basis_function_indices.size()),
          block_basis.n_occupied);

  // Dense full-support `orbtyp=oeo` blocks now follow the mixed chart
  // `(Q_i, Q_a, L_a)`. The reduced variables are:
  // 1. inactive-active rotations `K_ia`,
  // 2. active-shape increments `ΔL_a`,
  // 3. occupied-virtual rotations `(K_iv, K_av)`.
  //
  // The physical occupied step is reconstructed from the current gauge
  // `C_i = Q_i U_i` and `C_a = Q_a L_a + Q_i K_a`:
  // `δC_i = Q_a K_ia U_i + Q_v K_iv U_i`
  // `δC_a = -Q_i K_ia^T L_a + Q_a (ΔL_a + K_ia K_a)
  //         + Q_v (K_av L_a + K_iv K_a)`.
  if (n_inactive > 0) {
    if (n_active > 0) {
      block_step.leftCols(n_inactive).noalias() +=
          block_basis.active_working_orbitals *
          inactive_active_coefficients *
          block_basis.inactive_right_transform;
    }
    if (n_virtual > 0) {
      block_step.leftCols(n_inactive).noalias() +=
          block_basis.virtual_orbitals *
          occupied_virtual_coefficients.leftCols(n_inactive) *
          block_basis.inactive_right_transform;
    }
  }
  if (n_active > 0) {
    block_step.middleCols(n_inactive, n_active).noalias() +=
        block_basis.active_working_orbitals * active_shape_step;
    if (n_inactive > 0) {
      block_step.middleCols(n_inactive, n_active).noalias() +=
          block_basis.active_working_orbitals *
          (inactive_active_coefficients *
           block_basis.active_inactive_gauge_coefficients);
      block_step.middleCols(n_inactive, n_active).noalias() -=
          block_basis.inactive_working_orbitals *
          (inactive_active_coefficients.transpose() *
           block_basis.active_shape_matrix);
    }
    if (n_virtual > 0) {
      block_step.middleCols(n_inactive, n_active).noalias() +=
          block_basis.virtual_orbitals *
          (occupied_virtual_coefficients.rightCols(n_active) *
           block_basis.active_shape_matrix);
      if (n_inactive > 0) {
        block_step.middleCols(n_inactive, n_active).noalias() +=
            block_basis.virtual_orbitals *
            (occupied_virtual_coefficients.leftCols(n_inactive) *
             block_basis.active_inactive_gauge_coefficients);
      }
    }
  }
  return block_step;
}

// Lift one reduced block step to a finite occupied-block trial point.  The
// internal orthogonal frame `(Q_i, Q_a, Q_v)` is advanced by a Cayley update,
// while the active nonorthogonal shape still follows `L_a + Delta L_a`.
Eigen::MatrixXd NonredundantOrbitalSpace::build_mixed_chart_trial_occupied_orbitals(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& candidate_coefficients) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  const int n_occupied = layout.n_occupied;
  const int block_basis_count =
      static_cast<int>(block_basis.basis_function_indices.size());
  if (layout.direction_count == 0) {
    return block_basis.reference_occupied_orbitals;
  }

  Eigen::Map<const Eigen::MatrixXd> inactive_active_coefficients(
      candidate_coefficients.data(),
      n_active,
      n_inactive);
  Eigen::Map<const Eigen::MatrixXd> active_shape_step(
      candidate_coefficients.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<const Eigen::MatrixXd> occupied_virtual_coefficients(
      candidate_coefficients.data() + layout.occupied_virtual_offset,
      n_virtual,
      n_occupied);

  const int frame_size = n_occupied + n_virtual;
  Eigen::MatrixXd frame =
      Eigen::MatrixXd::Zero(block_basis_count, frame_size);
  if (n_inactive > 0) {
    frame.leftCols(n_inactive) =
        block_basis.inactive_working_orbitals;
  }
  if (n_active > 0) {
    frame.middleCols(n_inactive, n_active) =
        block_basis.active_working_orbitals;
  }
  if (n_virtual > 0) {
    frame.rightCols(n_virtual) =
        block_basis.internal_virtual_orbitals;
  }

  Eigen::MatrixXd omega =
      Eigen::MatrixXd::Zero(frame_size, frame_size);
  if (n_inactive > 0 && n_active > 0) {
    omega.block(n_inactive, 0, n_active, n_inactive) =
        inactive_active_coefficients;
    omega.block(0, n_inactive, n_inactive, n_active) =
        -inactive_active_coefficients.transpose();
  }
  if (n_virtual > 0 && n_inactive > 0) {
    const auto inactive_virtual_coefficients =
        occupied_virtual_coefficients.leftCols(n_inactive);
    omega.block(n_occupied, 0, n_virtual, n_inactive) =
        inactive_virtual_coefficients;
    omega.block(0, n_occupied, n_inactive, n_virtual) =
        -inactive_virtual_coefficients.transpose();
  }
  if (n_virtual > 0 && n_active > 0) {
    const auto active_virtual_coefficients =
        occupied_virtual_coefficients.rightCols(n_active);
    omega.block(n_occupied, n_inactive, n_virtual, n_active) =
        active_virtual_coefficients;
    omega.block(n_inactive, n_occupied, n_active, n_virtual) =
        -active_virtual_coefficients.transpose();
  }

  const Eigen::MatrixXd cayley_left =
      Eigen::MatrixXd::Identity(frame_size, frame_size) - 0.5 * omega;
  const Eigen::MatrixXd cayley_right =
      Eigen::MatrixXd::Identity(frame_size, frame_size) + 0.5 * omega;
  Eigen::FullPivLU<Eigen::MatrixXd> cayley_solver(cayley_left);
  if (!cayley_solver.isInvertible()) {
    throw std::runtime_error(
        "mixed-chart Cayley retraction encountered a singular frame update");
  }
  const Eigen::MatrixXd rotated_frame =
      frame * cayley_solver.solve(cayley_right);

  Eigen::MatrixXd trial_occupied_orbitals =
      Eigen::MatrixXd::Zero(block_basis_count, n_occupied);
  if (n_inactive > 0) {
    trial_occupied_orbitals.leftCols(n_inactive).noalias() =
        rotated_frame.leftCols(n_inactive) *
        block_basis.inactive_right_transform;
  }
  if (n_active > 0) {
    trial_occupied_orbitals.middleCols(n_inactive, n_active).noalias() =
        rotated_frame.middleCols(n_inactive, n_active) *
        (block_basis.active_shape_matrix + active_shape_step);
    if (n_inactive > 0) {
      trial_occupied_orbitals.middleCols(n_inactive, n_active).noalias() +=
          rotated_frame.leftCols(n_inactive) *
          block_basis.active_inactive_gauge_coefficients;
    }
  }
  return trial_occupied_orbitals;
}

Eigen::VectorXd NonredundantOrbitalSpace::project_dense_full_support_candidate_overlap(
    const BlockBasis& block_basis,
    const Eigen::Ref<const Eigen::MatrixXd>& block_columns) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  Eigen::VectorXd candidate_overlap =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(layout.direction_count));
  if (layout.direction_count == 0) {
    return candidate_overlap;
  }

  const Eigen::MatrixXd inactive_block =
      block_columns.leftCols(n_inactive);
  const Eigen::MatrixXd active_block =
      block_columns.middleCols(n_inactive, n_active);

  if (n_inactive > 0 && n_active > 0) {
    Eigen::Map<Eigen::MatrixXd> inactive_active_overlap(
        candidate_overlap.data(),
        n_active,
        n_inactive);
    inactive_active_overlap.noalias() =
        block_basis.active_working_orbitals.transpose() *
        inactive_block *
        block_basis.inactive_right_transform.transpose();
    inactive_active_overlap.noalias() +=
        block_basis.active_working_orbitals.transpose() *
        active_block *
        block_basis.active_inactive_gauge_coefficients.transpose();
    inactive_active_overlap.noalias() -=
        block_basis.active_shape_matrix *
        active_block.transpose() *
        block_basis.inactive_working_orbitals;
  }

  if (n_active > 0) {
    Eigen::Map<Eigen::MatrixXd> active_shape_overlap(
        candidate_overlap.data() + layout.inactive_active_count,
        n_active,
        n_active);
    active_shape_overlap.noalias() =
        block_basis.active_working_orbitals.transpose() * active_block;
  }

  if (n_virtual > 0) {
    Eigen::Map<Eigen::MatrixXd> occupied_virtual_overlap(
        candidate_overlap.data() + layout.occupied_virtual_offset,
        n_virtual,
        block_basis.n_occupied);
    if (n_inactive > 0) {
      occupied_virtual_overlap.leftCols(n_inactive).noalias() =
          block_basis.virtual_orbitals.transpose() *
          inactive_block *
          block_basis.inactive_right_transform.transpose();
      if (n_active > 0) {
        occupied_virtual_overlap.leftCols(n_inactive).noalias() +=
            block_basis.virtual_orbitals.transpose() *
            active_block *
            block_basis.active_inactive_gauge_coefficients.transpose();
      }
    }
    if (n_active > 0) {
      occupied_virtual_overlap.rightCols(n_active).noalias() =
          block_basis.virtual_orbitals.transpose() *
          active_block *
          block_basis.active_shape_matrix.transpose();
    }
  }

  return candidate_overlap;
}

Eigen::VectorXd NonredundantOrbitalSpace::build_dense_full_support_metric_diagonal(
    const BlockBasis& block_basis) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  Eigen::VectorXd metric_diagonal =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(layout.direction_count));
  if (layout.direction_count == 0) {
    return metric_diagonal;
  }

  // The dense full-support / OEO mixed chart admits an exact analytic metric
  // diagonal.  With
  //   `delta C_i = Q_a A U_i + Q_v X U_i`
  //   `delta C_a = -Q_i A^T L_a + Q_a (B + A K_a) + Q_v (Y L_a + X K_a)`
  // and `Q_i, Q_a, Q_v` mutually `S`-orthonormal, the block Gram action
  // `D^T D` on one direction `(A, B, X, Y)` is
  //   `G_A = A M_i + N_a A + B K_a^T`
  //   `G_B = B + A K_a`
  //   `G_X = X M_i + Y L_a K_a^T`
  //   `G_Y = X K_a L_a^T + Y N_a`
  // where `M_i = U_i U_i^T + K_a K_a^T` and `N_a = L_a L_a^T`.
  //
  // Therefore the diagonal entries are
  //   `d_ia(r,s) = M_i(s,s) + N_a(r,r)`
  //   `d_aa(r,t) = 1`
  //   `d_iv(u,s) = M_i(s,s)`
  //   `d_av(u,r) = N_a(r,r)`.
  Eigen::VectorXd inactive_metric_diagonal =
      Eigen::VectorXd::Zero(block_basis.inactive_metric_matrix.rows());
  if (block_basis.inactive_metric_matrix.rows() > 0) {
    inactive_metric_diagonal =
        block_basis.inactive_metric_matrix.diagonal();
  }
  Eigen::VectorXd inactive_plus_gauge_diagonal =
      inactive_metric_diagonal;
  if (block_basis.active_inactive_gauge_coefficients.rows() > 0) {
    inactive_plus_gauge_diagonal.noalias() +=
        block_basis.active_inactive_gauge_coefficients
            .rowwise()
            .squaredNorm();
  }
  Eigen::VectorXd active_shape_metric_diagonal =
      Eigen::VectorXd::Zero(block_basis.active_shape_metric_matrix.rows());
  if (block_basis.active_shape_metric_matrix.rows() > 0) {
    active_shape_metric_diagonal =
        block_basis.active_shape_metric_matrix.diagonal();
  }

  int direction_index = 0;
  for (int inactive_index = 0;
       inactive_index < n_inactive;
       ++inactive_index) {
    for (int active_index = 0;
         active_index < n_active;
         ++active_index) {
      metric_diagonal(direction_index) =
          inactive_plus_gauge_diagonal(inactive_index) +
          active_shape_metric_diagonal(active_index);
      ++direction_index;
    }
  }

  for (int target_active_index = 0;
       target_active_index < n_active;
       ++target_active_index) {
    for (int source_active_index = 0;
         source_active_index < n_active;
         ++source_active_index) {
      (void) source_active_index;
      metric_diagonal(direction_index) = 1.0;
      ++direction_index;
    }
  }

  for (int occupied_index = 0;
       occupied_index < n_inactive;
       ++occupied_index) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      (void) virtual_index;
      metric_diagonal(direction_index) =
          inactive_plus_gauge_diagonal(occupied_index);
      ++direction_index;
    }
  }

  for (int occupied_active_index = 0;
       occupied_active_index < n_active;
       ++occupied_active_index) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      (void) virtual_index;
      metric_diagonal(direction_index) =
          active_shape_metric_diagonal(occupied_active_index);
      ++direction_index;
    }
  }

  return metric_diagonal;
}

Eigen::VectorXd NonredundantOrbitalSpace::build_sparse_mixed_chart_metric_diagonal(
    const BlockBasis& block_basis) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  Eigen::VectorXd metric_diagonal =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(layout.direction_count));
  if (layout.direction_count == 0) {
    return metric_diagonal;
  }

  std::vector<Eigen::VectorXd> inactive_working_norms;
  std::vector<Eigen::VectorXd> active_working_norms;
  std::vector<Eigen::VectorXd> internal_virtual_norms;
  std::vector<Eigen::MatrixXd> inactive_active_cross_metrics;
  inactive_working_norms.reserve(block_basis.orbitals.size());
  active_working_norms.reserve(block_basis.orbitals.size());
  internal_virtual_norms.reserve(block_basis.orbitals.size());
  inactive_active_cross_metrics.reserve(block_basis.orbitals.size());
  for (const auto& projector : block_basis.orbitals) {
    Eigen::VectorXd inactive_norms = Eigen::VectorXd::Zero(n_inactive);
    Eigen::VectorXd active_norms = Eigen::VectorXd::Zero(n_active);
    Eigen::VectorXd virtual_norms = Eigen::VectorXd::Zero(n_virtual);
    Eigen::MatrixXd inactive_active_cross =
        Eigen::MatrixXd::Zero(n_inactive, n_active);
    if (n_inactive > 0) {
      inactive_norms =
          projector.inactive_working_masked.colwise().squaredNorm().transpose();
    }
    if (n_active > 0) {
      active_norms =
          projector.active_working_masked.colwise().squaredNorm().transpose();
    }
    if (n_virtual > 0) {
      virtual_norms =
          projector.internal_virtual_masked.colwise().squaredNorm().transpose();
    }
    if (n_inactive > 0 && n_active > 0) {
      inactive_active_cross.noalias() =
          projector.inactive_working_masked.transpose() *
          projector.active_working_masked;
    }
    inactive_working_norms.push_back(std::move(inactive_norms));
    active_working_norms.push_back(std::move(active_norms));
    internal_virtual_norms.push_back(std::move(virtual_norms));
    inactive_active_cross_metrics.push_back(std::move(inactive_active_cross));
  }

  constexpr double kMinimumMetricDiagonal = 1.0e-10;
  int direction_index = 0;
  for (int inactive_source = 0;
       inactive_source < n_inactive;
       ++inactive_source) {
    for (int active_source = 0;
         active_source < n_active;
         ++active_source) {
      double diagonal_value = 0.0;
      for (int inactive_target = 0;
           inactive_target < n_inactive;
           ++inactive_target) {
        const double coefficient =
            block_basis.inactive_right_transform(
                inactive_source,
                inactive_target);
        diagonal_value +=
            active_working_norms[inactive_target](active_source) *
            coefficient * coefficient;
      }
      for (int active_target = 0;
           active_target < n_active;
           ++active_target) {
        const int occupied_index = n_inactive + active_target;
        const double shape_coefficient =
            block_basis.active_shape_matrix(
                active_source,
                active_target);
        const double gauge_coefficient =
            block_basis.active_inactive_gauge_coefficients(
                inactive_source,
                active_target);
        diagonal_value +=
            inactive_working_norms[occupied_index](inactive_source) *
            shape_coefficient * shape_coefficient;
        diagonal_value +=
            active_working_norms[occupied_index](active_source) *
            gauge_coefficient * gauge_coefficient;
        diagonal_value -=
            2.0 *
            inactive_active_cross_metrics[occupied_index](
                inactive_source,
                active_source) *
            shape_coefficient * gauge_coefficient;
      }
      metric_diagonal(direction_index) =
          std::max(kMinimumMetricDiagonal, diagonal_value);
      ++direction_index;
    }
  }

  for (int target_active = 0;
       target_active < n_active;
       ++target_active) {
    const int occupied_index = n_inactive + target_active;
    for (int source_active = 0;
         source_active < n_active;
         ++source_active) {
      metric_diagonal(direction_index) =
          std::max(
              kMinimumMetricDiagonal,
              active_working_norms[occupied_index](source_active));
      ++direction_index;
    }
  }

  for (int inactive_source = 0;
       inactive_source < n_inactive;
       ++inactive_source) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      double diagonal_value = 0.0;
      for (int inactive_target = 0;
           inactive_target < n_inactive;
           ++inactive_target) {
        const double coefficient =
            block_basis.inactive_right_transform(
                inactive_source,
                inactive_target);
        diagonal_value +=
            internal_virtual_norms[inactive_target](virtual_index) *
            coefficient * coefficient;
      }
      for (int active_target = 0;
           active_target < n_active;
           ++active_target) {
        const double coefficient =
            block_basis.active_inactive_gauge_coefficients(
                inactive_source,
                active_target);
        diagonal_value +=
            internal_virtual_norms[n_inactive + active_target](virtual_index) *
            coefficient * coefficient;
      }
      metric_diagonal(direction_index) =
          std::max(kMinimumMetricDiagonal, diagonal_value);
      ++direction_index;
    }
  }

  for (int active_source = 0;
       active_source < n_active;
       ++active_source) {
    for (int virtual_index = 0;
         virtual_index < n_virtual;
         ++virtual_index) {
      double diagonal_value = 0.0;
      for (int active_target = 0;
           active_target < n_active;
           ++active_target) {
        const double coefficient =
            block_basis.active_shape_matrix(
                active_source,
                active_target);
        diagonal_value +=
            internal_virtual_norms[n_inactive + active_target](virtual_index) *
            coefficient * coefficient;
      }
      metric_diagonal(direction_index) =
          std::max(kMinimumMetricDiagonal, diagonal_value);
      ++direction_index;
    }
  }

  return metric_diagonal;
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_dense_full_support_candidate_metric(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& candidate_coefficients) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  Eigen::VectorXd metric_action =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(layout.direction_count));
  if (layout.direction_count == 0) {
    return metric_action;
  }

  Eigen::Map<const Eigen::MatrixXd> inactive_active_coefficients(
      candidate_coefficients.data(),
      n_active,
      n_inactive);
  Eigen::Map<const Eigen::MatrixXd> active_shape_coefficients(
      candidate_coefficients.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<const Eigen::MatrixXd> occupied_virtual_coefficients(
      candidate_coefficients.data() + layout.occupied_virtual_offset,
      n_virtual,
      block_basis.n_occupied);

  Eigen::Map<Eigen::MatrixXd> inactive_active_metric_action(
      metric_action.data(),
      n_active,
      n_inactive);
  Eigen::Map<Eigen::MatrixXd> active_shape_metric_action(
      metric_action.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<Eigen::MatrixXd> occupied_virtual_metric_action(
      metric_action.data() + layout.occupied_virtual_offset,
      n_virtual,
      block_basis.n_occupied);

  const Eigen::MatrixXd& active_inactive_gauge_coefficients =
      block_basis.active_inactive_gauge_coefficients;
  const Eigen::MatrixXd& active_shape_matrix =
      block_basis.active_shape_matrix;
  const Eigen::MatrixXd& active_shape_metric =
      block_basis.active_shape_metric_matrix;
  const Eigen::MatrixXd& inactive_plus_gauge_metric =
      block_basis.inactive_plus_gauge_metric_matrix;

  if (n_inactive > 0 && n_active > 0) {
    inactive_active_metric_action.noalias() =
        inactive_active_coefficients * inactive_plus_gauge_metric;
    inactive_active_metric_action.noalias() +=
        active_shape_metric * inactive_active_coefficients;
    inactive_active_metric_action.noalias() +=
        active_shape_coefficients *
        active_inactive_gauge_coefficients.transpose();
  }

  if (n_active > 0) {
    active_shape_metric_action = active_shape_coefficients;
    if (n_inactive > 0) {
      active_shape_metric_action.noalias() +=
          inactive_active_coefficients *
          active_inactive_gauge_coefficients;
    }
  }

  if (n_virtual > 0) {
    if (n_inactive > 0) {
      occupied_virtual_metric_action.leftCols(n_inactive).noalias() =
          occupied_virtual_coefficients.leftCols(n_inactive) *
          inactive_plus_gauge_metric;
      if (n_active > 0) {
        occupied_virtual_metric_action.leftCols(n_inactive).noalias() +=
            occupied_virtual_coefficients.rightCols(n_active) *
            active_shape_matrix *
            active_inactive_gauge_coefficients.transpose();
      }
    }
    if (n_active > 0) {
      occupied_virtual_metric_action.rightCols(n_active).noalias() =
          occupied_virtual_coefficients.rightCols(n_active) *
          active_shape_metric;
      if (n_inactive > 0) {
        occupied_virtual_metric_action.rightCols(n_active).noalias() +=
            occupied_virtual_coefficients.leftCols(n_inactive) *
            active_inactive_gauge_coefficients *
            active_shape_matrix.transpose();
      }
    }
  }

  return metric_action;
}

Eigen::VectorXd NonredundantOrbitalSpace::solve_dense_full_support_candidate_metric(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& right_hand_side) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  Eigen::VectorXd solution =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(layout.direction_count));
  if (layout.direction_count == 0) {
    return solution;
  }

  Eigen::Map<const Eigen::MatrixXd> inactive_active_rhs(
      right_hand_side.data(),
      n_active,
      n_inactive);
  Eigen::Map<const Eigen::MatrixXd> active_shape_rhs(
      right_hand_side.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<const Eigen::MatrixXd> occupied_virtual_rhs(
      right_hand_side.data() + layout.occupied_virtual_offset,
      n_virtual,
      block_basis.n_occupied);

  Eigen::Map<Eigen::MatrixXd> inactive_active_solution(
      solution.data(),
      n_active,
      n_inactive);
  Eigen::Map<Eigen::MatrixXd> active_shape_solution(
      solution.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<Eigen::MatrixXd> occupied_virtual_solution(
      solution.data() + layout.occupied_virtual_offset,
      n_virtual,
      block_basis.n_occupied);

  const Eigen::MatrixXd& active_inactive_gauge_coefficients =
      block_basis.active_inactive_gauge_coefficients;
  const Eigen::MatrixXd& inactive_metric =
      block_basis.inactive_metric_matrix;
  const Eigen::MatrixXd& active_shape_metric =
      block_basis.active_shape_metric_matrix;

  if (n_active > 0 && n_inactive > 0) {
    // Eliminating `B` from
    //   `R_A = A (U_i U_i^T + K_a K_a^T) + N_a A + B K_a^T`
    //   `R_B = B + A K_a`
    // gives the Sylvester equation
    //   `N_a A + A U_i U_i^T = R_A - R_B K_a^T`.
    const Eigen::MatrixXd reduced_inactive_active_rhs =
        inactive_active_rhs -
        active_shape_rhs *
            active_inactive_gauge_coefficients.transpose();

    const Eigen::MatrixXd transformed_rhs =
        block_basis.active_shape_metric_eigenvectors.transpose() *
        reduced_inactive_active_rhs *
        block_basis.inactive_metric_eigenvectors;
    Eigen::MatrixXd transformed_solution =
        Eigen::MatrixXd::Zero(n_active, n_inactive);
    for (int active_index = 0;
         active_index < n_active;
         ++active_index) {
      for (int inactive_index = 0;
           inactive_index < n_inactive;
           ++inactive_index) {
        const double denominator =
            block_basis.active_shape_metric_eigenvalues(active_index) +
            block_basis.inactive_metric_eigenvalues(inactive_index);
        if (!std::isfinite(denominator) ||
            denominator <= kMinimumDenseMetricEigenvalue) {
          throw std::runtime_error(
              "dense full-support inactive-active Sylvester solve is singular");
        }
        transformed_solution(active_index, inactive_index) =
            transformed_rhs(active_index, inactive_index) /
            denominator;
      }
    }
    inactive_active_solution.noalias() =
        block_basis.active_shape_metric_eigenvectors *
        transformed_solution *
        block_basis.inactive_metric_eigenvectors.transpose();
    active_shape_solution.noalias() =
        active_shape_rhs -
        inactive_active_solution *
            active_inactive_gauge_coefficients;
  } else if (n_active > 0) {
    active_shape_solution = active_shape_rhs;
  }

  if (n_virtual > 0) {
    if (n_inactive > 0) {
      // Eliminating `Y` from
      //   `R_X = X (U_i U_i^T + K_a K_a^T) + Y L_a K_a^T`
      //   `R_Y = X K_a L_a^T + Y N_a`
      // gives
      //   `X U_i U_i^T = R_X - R_Y L_a^{-T} K_a^T`.
      Eigen::MatrixXd reduced_virtual_inactive_rhs =
          occupied_virtual_rhs.leftCols(n_inactive);
      if (n_active > 0) {
        reduced_virtual_inactive_rhs.noalias() -=
            occupied_virtual_rhs.rightCols(n_active) *
            block_basis.active_shape_inverse *
            active_inactive_gauge_coefficients.transpose();
      }

      occupied_virtual_solution.leftCols(n_inactive).noalias() =
          reduced_virtual_inactive_rhs *
          block_basis.inactive_metric_inverse;
    }

    if (n_active > 0) {
      Eigen::MatrixXd active_virtual_rhs =
          occupied_virtual_rhs.rightCols(n_active);
      if (n_inactive > 0) {
        active_virtual_rhs.noalias() -=
            occupied_virtual_solution.leftCols(n_inactive) *
            active_inactive_gauge_coefficients *
            block_basis.active_shape_matrix.transpose();
      }
      occupied_virtual_solution.rightCols(n_active).noalias() =
          active_virtual_rhs *
          block_basis.active_shape_metric_inverse;
    }
  }

  return solution;
}

Eigen::VectorXd NonredundantOrbitalSpace::project_block_candidate_overlap(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& packed_vector) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_occupied = layout.n_occupied;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  const int direction_count = layout.direction_count;
  Eigen::VectorXd candidate_overlap =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(direction_count));
  if (direction_count == 0) {
    return candidate_overlap;
  }

  if (block_basis.uses_dense_full_support_projector) {
    // Dense full-support/OEO blocks gather the physical occupied columns once
    // and then project them against the mixed-chart tangent map.
    const Eigen::MatrixXd block_columns =
        gather_dense_full_support_block_matrix(
            block_basis,
            packed_vector);
    return project_dense_full_support_candidate_overlap(
        block_basis,
        block_columns);
  }

  Eigen::MatrixXd inactive_active_overlap =
      Eigen::MatrixXd::Zero(n_active, n_inactive);
  Eigen::MatrixXd active_inactive_overlap =
      Eigen::MatrixXd::Zero(n_inactive, n_active);
  Eigen::MatrixXd active_shape_overlap =
      Eigen::MatrixXd::Zero(n_active, n_active);
  Eigen::MatrixXd virtual_inactive_overlap =
      Eigen::MatrixXd::Zero(n_virtual, n_inactive);
  Eigen::MatrixXd virtual_active_overlap =
      Eigen::MatrixXd::Zero(n_virtual, n_active);

  for (int occupied_index = 0;
       occupied_index < n_occupied;
       ++occupied_index) {
    const auto& projector = block_basis.orbitals[occupied_index];
    Eigen::VectorXd local_vector(
        static_cast<Eigen::Index>(projector.packed_indices.size()));
    for (Eigen::Index local_index = 0;
         local_index < local_vector.size();
         ++local_index) {
      local_vector[local_index] =
          packed_vector[projector.packed_indices[local_index]];
    }

    if (occupied_index < n_inactive) {
      if (n_active > 0) {
        inactive_active_overlap.col(occupied_index).noalias() =
            projector.active_working_masked.transpose() * local_vector;
      }
      if (n_virtual > 0) {
        virtual_inactive_overlap.col(occupied_index).noalias() =
            projector.internal_virtual_masked.transpose() * local_vector;
      }
      continue;
    }

    const int active_index = occupied_index - n_inactive;
    if (n_inactive > 0) {
      active_inactive_overlap.col(active_index).noalias() =
          projector.inactive_working_masked.transpose() * local_vector;
    }
    if (n_active > 0) {
      active_shape_overlap.col(active_index).noalias() =
          projector.active_working_masked.transpose() * local_vector;
    }
    if (n_virtual > 0) {
      virtual_active_overlap.col(active_index).noalias() =
          projector.internal_virtual_masked.transpose() * local_vector;
    }
  }

  if (n_inactive > 0 && n_active > 0) {
    Eigen::Map<Eigen::MatrixXd> inactive_active_candidate_overlap(
        candidate_overlap.data(),
        n_active,
        n_inactive);
    inactive_active_candidate_overlap.noalias() =
        inactive_active_overlap *
        block_basis.inactive_right_transform.transpose();
    inactive_active_candidate_overlap.noalias() +=
        active_shape_overlap *
        block_basis.active_inactive_gauge_coefficients.transpose();
    inactive_active_candidate_overlap.noalias() -=
        block_basis.active_shape_matrix *
        active_inactive_overlap.transpose();
  }

  if (n_active > 0) {
    Eigen::Map<Eigen::MatrixXd> active_shape_candidate_overlap(
        candidate_overlap.data() + layout.inactive_active_count,
        n_active,
        n_active);
    active_shape_candidate_overlap = active_shape_overlap;
  }

  if (n_virtual > 0) {
    Eigen::Map<Eigen::MatrixXd> occupied_virtual_overlap(
        candidate_overlap.data() + layout.occupied_virtual_offset,
        n_virtual,
        n_occupied);
    if (n_inactive > 0) {
      occupied_virtual_overlap.leftCols(n_inactive).noalias() =
          virtual_inactive_overlap *
          block_basis.inactive_right_transform.transpose();
      if (n_active > 0) {
        occupied_virtual_overlap.leftCols(n_inactive).noalias() +=
            virtual_active_overlap *
            block_basis.active_inactive_gauge_coefficients.transpose();
      }
    }
    if (n_active > 0) {
      occupied_virtual_overlap.rightCols(n_active).noalias() =
          virtual_active_overlap *
          block_basis.active_shape_matrix.transpose();
    }
  }

  return candidate_overlap;
}

void NonredundantOrbitalSpace::accumulate_block_candidate_combination(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& candidate_coefficients,
    Eigen::VectorXd* packed_vector) const {
  const BlockDirectionLayout layout =
      build_block_direction_layout(
          block_basis.n_inactive,
          block_basis.n_occupied,
          block_basis.n_virtual);
  const int n_inactive = layout.n_inactive;
  const int n_occupied = layout.n_occupied;
  const int n_active = layout.n_active;
  const int n_virtual = layout.n_virtual;
  const int direction_count = layout.direction_count;
  if (direction_count == 0) {
    return;
  }

  Eigen::Map<const Eigen::MatrixXd> inactive_active_coefficients(
      candidate_coefficients.data(),
      n_active,
      n_inactive);
  Eigen::Map<const Eigen::MatrixXd> active_shape_coefficients(
      candidate_coefficients.data() + layout.inactive_active_count,
      n_active,
      n_active);
  Eigen::Map<const Eigen::MatrixXd> occupied_virtual_coefficients(
      candidate_coefficients.data() + layout.occupied_virtual_offset,
      n_virtual,
      n_occupied);

  if (block_basis.uses_dense_full_support_projector) {
    const Eigen::MatrixXd block_step =
        build_dense_full_support_block_step(
            block_basis,
            candidate_coefficients);
    accumulate_dense_full_support_block_matrix(
        block_basis,
        block_step,
        packed_vector);
    return;
  }

  // Sparse blocks use the same internal `(Q_i, Q_a, L_a, Q_v)` chart as the
  // dense formulas, then sample each physical occupied column on its fixed
  // support before scattering back to packed sparse storage.
  Eigen::MatrixXd inactive_active_step =
      Eigen::MatrixXd::Zero(n_active, n_inactive);
  Eigen::MatrixXd inactive_virtual_step =
      Eigen::MatrixXd::Zero(n_virtual, n_inactive);
  Eigen::MatrixXd active_inactive_step =
      Eigen::MatrixXd::Zero(n_inactive, n_active);
  Eigen::MatrixXd active_working_step =
      active_shape_coefficients;
  Eigen::MatrixXd active_virtual_step =
      Eigen::MatrixXd::Zero(n_virtual, n_active);

  if (n_inactive > 0 && n_active > 0) {
    inactive_active_step.noalias() =
        inactive_active_coefficients *
        block_basis.inactive_right_transform;
    active_inactive_step.noalias() =
        -inactive_active_coefficients.transpose() *
        block_basis.active_shape_matrix;
    active_working_step.noalias() +=
        inactive_active_coefficients *
        block_basis.active_inactive_gauge_coefficients;
  }
  if (n_virtual > 0) {
    if (n_inactive > 0) {
      inactive_virtual_step.noalias() =
          occupied_virtual_coefficients.leftCols(n_inactive) *
          block_basis.inactive_right_transform;
      if (n_active > 0) {
        active_virtual_step.noalias() +=
            occupied_virtual_coefficients.leftCols(n_inactive) *
            block_basis.active_inactive_gauge_coefficients;
      }
    }
    if (n_active > 0) {
      active_virtual_step.noalias() +=
          occupied_virtual_coefficients.rightCols(n_active) *
          block_basis.active_shape_matrix;
    }
  }

  for (int occupied_index = 0;
       occupied_index < n_occupied;
       ++occupied_index) {
    const auto& projector = block_basis.orbitals[occupied_index];
    Eigen::VectorXd local_step =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(projector.packed_indices.size()));
    if (occupied_index < n_inactive) {
      if (n_active > 0) {
        local_step.noalias() +=
            projector.active_working_masked *
            inactive_active_step.col(occupied_index);
      }
      if (n_virtual > 0) {
        local_step.noalias() +=
            projector.internal_virtual_masked *
            inactive_virtual_step.col(occupied_index);
      }
    } else {
      const int active_index = occupied_index - n_inactive;
      if (n_inactive > 0) {
        local_step.noalias() +=
            projector.inactive_working_masked *
            active_inactive_step.col(active_index);
      }
      if (n_active > 0) {
        local_step.noalias() +=
            projector.active_working_masked *
            active_working_step.col(active_index);
      }
      if (n_virtual > 0) {
        local_step.noalias() +=
            projector.internal_virtual_masked *
            active_virtual_step.col(active_index);
      }
    }
    for (Eigen::Index local_index = 0;
         local_index < local_step.size();
         ++local_index) {
      (*packed_vector)[projector.packed_indices[local_index]] +=
          local_step[local_index];
    }
  }
}

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_impl(
    const Eigen::VectorXd& packed_vector,
    bool recover_tangent_coordinates,
    bool build_packed_projection) const {
  ProjectionResult result;
  result.reduced_gradient =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(reduced_size_));
  result.packed_projected_gradient =
      build_packed_projection
          ? Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_))
          : Eigen::VectorXd();

  for (const auto& block_basis : block_bases_) {
    const Eigen::VectorXd candidate_overlap =
        project_block_candidate_overlap(block_basis, packed_vector);
    const Eigen::VectorXd local_reduced =
        block_basis.has_exact_metric_factorization
            ? block_basis.candidate_metric_cholesky_factor
                  .triangularView<Eigen::Lower>()
                  .solve(candidate_overlap)
            : (recover_tangent_coordinates
                   ? solve_block_candidate_metric(
                         block_basis,
                         candidate_overlap)
                   : candidate_overlap);
    result.reduced_gradient.segment(
        block_basis.reduced_offset,
        local_reduced.size()) = local_reduced;
    if (build_packed_projection) {
      // The packed tangent projection must use the actual candidate
      // coefficients `a` in `D a`, not the reduced covector `D^T g` itself.
      // For exact-factorized blocks `local_reduced` is the whitened reduced
      // coordinate `z = L^{-1} D^T g`, so recover `a = L^{-T} z`.  For large
      // blocks without the exact factorization we solve the implicit metric
      // system `(D^T D) a = D^T v` whenever the caller asks for the packed
      // tangent projection.
      const Eigen::VectorXd packed_projection_coefficients =
          block_basis.has_exact_metric_factorization
              ? block_basis.candidate_metric_cholesky_factor
                    .transpose()
                    .triangularView<Eigen::Upper>()
                    .solve(local_reduced)
              : (recover_tangent_coordinates
                     ? local_reduced
                     : solve_block_candidate_metric(
                           block_basis,
                           candidate_overlap));
      accumulate_block_candidate_combination(
          block_basis,
          packed_projection_coefficients,
          &result.packed_projected_gradient);
    }
  }

  return result;
}

Eigen::VectorXd NonredundantOrbitalSpace::project_reduced_gradient(
    const Eigen::VectorXd& packed_gradient) const {
  // This is the hot-path pullback used by exact-context HVP evaluations.
  // Small blocks may first whiten their candidate chart through an exact
  // `D^T D` factorization, but we still skip reconstructing `D a` when the
  // caller only needs reduced coordinates.
  return project_impl(
             packed_gradient,
             false,
             false)
      .reduced_gradient;
}

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_gradient(
    const Eigen::VectorXd& packed_gradient) const {
  // `project_gradient` returns the reduced covector seen by the optimizer and
  // also the packed tangent projection used by line-search fallbacks and
  // secant updates.
  return project_impl(
      packed_gradient,
      false,
      true);
}

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_vector(
    const Eigen::VectorXd& packed_vector) const {
  // `project_vector` recovers reduced coordinates for a packed tangent vector.
  // Large blocks still solve the implicit block metric equation
  // `D^T D a = D^T v`; small blocks instead reuse their exact whitening
  // factorization so `expand_step(a)` stays in an orthonormal tangent chart.
  return project_impl(
      packed_vector,
      true,
      true);
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_inverse_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  if (!has_reduced_curvature_diagonal_) {
    return reduced_vector;
  }

  Eigen::VectorXd preconditioned = reduced_vector;
  constexpr double kMinimumCurvature = 1.0e-12;
  // The diagonal is block-local and already lives in reduced coordinates, so
  // applying the inverse preconditioner is just a segmented elementwise divide.
  for (const auto& block_basis : block_bases_) {
    if (block_basis.reduced_curvature_diagonal.size() == 0) {
      continue;
    }
    preconditioned.segment(
        block_basis.reduced_offset,
        block_basis.reduced_curvature_diagonal.size())
        .array() /=
        block_basis.reduced_curvature_diagonal
            .array()
            .max(kMinimumCurvature);
  }
  return preconditioned;
}

Eigen::VectorXd
NonredundantOrbitalSpace::apply_inverse_reduced_block_preconditioner(
    const Eigen::VectorXd& reduced_vector) const {
  constexpr double kMinimumCurvature = 1.0e-12;
  Eigen::VectorXd preconditioned = reduced_vector;
  for (const auto& block_basis : block_bases_) {
    const Eigen::Index local_size =
        static_cast<Eigen::Index>(block_basis.candidate_metric_diagonal.size());
    if (local_size == 0) {
      continue;
    }

    Eigen::VectorXd local_vector =
        reduced_vector.segment(block_basis.reduced_offset, local_size);
    const bool has_block_curvature =
        block_basis.reduced_curvature_diagonal.size() == local_size;
    if (has_block_curvature) {
      // The reduced block model is
      //   `M = C^{1/2} G C^{1/2}`
      // where `G = D^T D` is the accepted-point block Gram matrix in the raw
      // candidate chart and `C` is the positive reduced curvature diagonal.
      // Whiten the right-hand side by `C^{-1/2}`, solve the block metric, then
      // apply the second `C^{-1/2}` so the overall action is symmetric
      // positive and compatible with PCG.
      local_vector.array() /=
          block_basis.reduced_curvature_diagonal
              .array()
              .max(kMinimumCurvature)
              .sqrt();
    }

    if (block_basis.uses_dense_full_support_projector) {
      local_vector =
          solve_dense_full_support_candidate_metric(
              block_basis,
              local_vector);
    } else if (block_basis.has_exact_metric_factorization) {
      // Exact-factorized blocks already use the whitened chart `z = L^{-1} D^T v`
      // with identity reduced metric, so only the curvature scaling above is
      // required here.
    } else {
      // Sparse large blocks keep the cheaper diagonal-only model. Re-entering
      // the iterative block metric solve here would nest one Krylov iteration
      // inside another and tends to lose wall time on localized charts.
      if (has_block_curvature) {
        local_vector =
            local_vector.array() /
            block_basis.reduced_curvature_diagonal
                .array()
                .max(kMinimumCurvature)
                .sqrt();
      }
      preconditioned.segment(block_basis.reduced_offset, local_size) =
          local_vector;
      continue;
    }

    if (has_block_curvature) {
      local_vector.array() /=
          block_basis.reduced_curvature_diagonal
              .array()
              .max(kMinimumCurvature)
              .sqrt();
    }
    preconditioned.segment(block_basis.reduced_offset, local_size) =
        local_vector;
  }
  return preconditioned;
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  if (!has_reduced_curvature_diagonal_) {
    return reduced_vector;
  }

  Eigen::VectorXd curved = reduced_vector;
  constexpr double kMinimumCurvature = 1.0e-12;
  // The diagonal already lives in reduced coordinates, so the direct action of
  // the block-local curvature model is just a segmented elementwise multiply.
  for (const auto& block_basis : block_bases_) {
    if (block_basis.reduced_curvature_diagonal.size() == 0) {
      continue;
    }
    curved.segment(
        block_basis.reduced_offset,
        block_basis.reduced_curvature_diagonal.size())
        .array() *=
        block_basis.reduced_curvature_diagonal
            .array()
            .max(kMinimumCurvature);
  }
  return curved;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_step(
    const Eigen::VectorXd& reduced_step) const {
  Eigen::VectorXd packed_step =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
  // Expand a reduced-coordinate step back to the packed sparse-parameter
  // vector expected by the main optimizer objective.
  for (const auto& block_basis : block_bases_) {
    const Eigen::VectorXd candidate_coefficients =
        block_candidate_coefficients_from_reduced_step(
            block_basis,
            reduced_step);
    accumulate_block_candidate_combination(
        block_basis,
        candidate_coefficients,
        &packed_step);
  }
  return packed_step;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_retract_input_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step) const {
  Eigen::VectorXd input_tangent =
      Eigen::VectorXd::Zero(
          static_cast<Eigen::Index>(
              orbital_preparation_input.orbital_value_table.size()));

  for (const auto& block_basis : block_bases_) {
    if (block_basis.n_occupied <= 0) {
      continue;
    }

    const Eigen::VectorXd candidate_coefficients =
        block_candidate_coefficients_from_reduced_step(
            block_basis,
            reduced_step);
    if (block_basis.uses_dense_full_support_projector) {
      const Eigen::MatrixXd block_step =
          build_dense_full_support_block_step(
              block_basis,
              candidate_coefficients);
      for (int occupied_index = 0;
           occupied_index < block_basis.n_occupied;
           ++occupied_index) {
        const auto& projector = block_basis.orbitals[occupied_index];
        for (int block_row = 0;
             block_row < static_cast<int>(block_basis.basis_function_indices.size());
             ++block_row) {
          const int flat_index =
              projector.flat_index_by_block_row[block_row];
          if (flat_index < 0 ||
              flat_index >= input_tangent.size()) {
            throw std::runtime_error(
                "dense full-support tangent expansion is missing a flat block row");
          }
          input_tangent[flat_index] +=
              block_step(block_row, occupied_index);
        }
      }
      continue;
    }

    const BlockDirectionLayout layout =
        build_block_direction_layout(
            block_basis.n_inactive,
            block_basis.n_occupied,
            block_basis.n_virtual);
    Eigen::Map<const Eigen::MatrixXd> inactive_active_coefficients(
        candidate_coefficients.data(),
        layout.n_active,
        layout.n_inactive);
    Eigen::Map<const Eigen::MatrixXd> active_shape_coefficients(
        candidate_coefficients.data() + layout.inactive_active_count,
        layout.n_active,
        layout.n_active);
    Eigen::Map<const Eigen::MatrixXd> occupied_virtual_coefficients(
        candidate_coefficients.data() + layout.occupied_virtual_offset,
        layout.n_virtual,
        layout.n_occupied);

    Eigen::MatrixXd inactive_active_step =
        Eigen::MatrixXd::Zero(layout.n_active, layout.n_inactive);
    Eigen::MatrixXd inactive_virtual_step =
        Eigen::MatrixXd::Zero(layout.n_virtual, layout.n_inactive);
    Eigen::MatrixXd active_inactive_step =
        Eigen::MatrixXd::Zero(layout.n_inactive, layout.n_active);
    Eigen::MatrixXd active_working_step =
        active_shape_coefficients;
    Eigen::MatrixXd active_virtual_step =
        Eigen::MatrixXd::Zero(layout.n_virtual, layout.n_active);

    if (layout.n_inactive > 0 && layout.n_active > 0) {
      inactive_active_step.noalias() =
          inactive_active_coefficients *
          block_basis.inactive_right_transform;
      active_inactive_step.noalias() =
          -inactive_active_coefficients.transpose() *
          block_basis.active_shape_matrix;
      active_working_step.noalias() +=
          inactive_active_coefficients *
          block_basis.active_inactive_gauge_coefficients;
    }
    if (layout.n_virtual > 0) {
      if (layout.n_inactive > 0) {
        inactive_virtual_step.noalias() =
            occupied_virtual_coefficients.leftCols(layout.n_inactive) *
            block_basis.inactive_right_transform;
        if (layout.n_active > 0) {
          active_virtual_step.noalias() +=
              occupied_virtual_coefficients.leftCols(layout.n_inactive) *
              block_basis.active_inactive_gauge_coefficients;
        }
      }
      if (layout.n_active > 0) {
        active_virtual_step.noalias() +=
            occupied_virtual_coefficients.rightCols(layout.n_active) *
            block_basis.active_shape_matrix;
      }
    }

    for (int occupied_index = 0;
         occupied_index < block_basis.n_occupied;
         ++occupied_index) {
      const auto& projector = block_basis.orbitals[occupied_index];
      Eigen::VectorXd local_step =
          Eigen::VectorXd::Zero(
              static_cast<Eigen::Index>(projector.flat_indices.size()));
      if (occupied_index < block_basis.n_inactive) {
        if (layout.n_active > 0) {
          local_step.noalias() +=
              projector.active_working_masked *
              inactive_active_step.col(occupied_index);
        }
        if (layout.n_virtual > 0) {
          local_step.noalias() +=
              projector.internal_virtual_masked *
              inactive_virtual_step.col(occupied_index);
        }
      } else {
        const int active_index = occupied_index - block_basis.n_inactive;
        if (layout.n_inactive > 0) {
          local_step.noalias() +=
              projector.inactive_working_masked *
              active_inactive_step.col(active_index);
        }
        if (layout.n_active > 0) {
          local_step.noalias() +=
              projector.active_working_masked *
              active_working_step.col(active_index);
        }
        if (layout.n_virtual > 0) {
          local_step.noalias() +=
              projector.internal_virtual_masked *
              active_virtual_step.col(active_index);
        }
      }

      Eigen::VectorXd local_coefficients =
          Eigen::VectorXd::Zero(
              static_cast<Eigen::Index>(projector.flat_indices.size()));
      for (Eigen::Index local_index = 0;
           local_index < local_coefficients.size();
           ++local_index) {
        local_coefficients[local_index] =
            orbital_preparation_input.orbital_value_table
                [projector.flat_indices[local_index]];
      }
      const Eigen::MatrixXd local_overlap =
          build_local_sparse_overlap_metric(
              block_basis.block_overlap_matrix,
              projector.block_rows);
      const Eigen::VectorXd local_tangent =
          linearize_local_sparse_normalized_step(
              local_overlap,
              local_coefficients,
              local_step);

      for (Eigen::Index local_index = 0;
           local_index < local_tangent.size();
           ++local_index) {
        input_tangent[projector.flat_indices[local_index]] +=
            local_tangent[local_index];
      }
    }
  }

  return input_tangent;
}

OrbitalPreparationInput NonredundantOrbitalSpace::retract_step(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step,
    double step_scale) const {
  OrbitalPreparationInput trial_input = orbital_preparation_input;
  std::vector<double> updated_orbital_values =
      orbital_preparation_input.orbital_value_table;

  // Both dense and sparse blocks now generate the finite step from the same
  // internal mixed chart `(Q_i, Q_a, Q_v, L_a)`.  Dense full-support blocks
  // can write the rotated block columns back directly; sparse blocks first
  // restrict the internal finite step to each orbital's fixed support and then
  // retract that support-local target on the local `x^T S x = 1` manifold.
  for (const auto& block_basis : block_bases_) {
    if (block_basis.n_occupied <= 0) {
      continue;
    }

    Eigen::VectorXd candidate_coefficients =
        block_candidate_coefficients_from_reduced_step(
            block_basis,
            reduced_step);
    candidate_coefficients *= step_scale;
    const Eigen::MatrixXd trial_occupied_orbitals =
        build_mixed_chart_trial_occupied_orbitals(
            block_basis,
            candidate_coefficients);

    if (block_basis.uses_dense_full_support_projector) {
      if (!trial_occupied_orbitals.allFinite()) {
        throw std::runtime_error(
            "nonredundant dense full-support retraction produced non-finite coefficients");
      }
      write_dense_full_support_block_matrix(
          block_basis,
          trial_occupied_orbitals,
          &updated_orbital_values);
      continue;
    }

    for (int occupied_index = 0;
         occupied_index < block_basis.n_occupied;
         ++occupied_index) {
      const auto& projector = block_basis.orbitals[occupied_index];
      const Eigen::Index local_size =
          static_cast<Eigen::Index>(projector.flat_indices.size());
      Eigen::VectorXd local_coefficients =
          Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index local_index = 0;
           local_index < local_coefficients.size();
           ++local_index) {
        local_coefficients[local_index] =
            orbital_preparation_input.orbital_value_table
                [projector.flat_indices[local_index]];
      }
      const Eigen::MatrixXd local_overlap =
          build_local_sparse_overlap_metric(
              block_basis.block_overlap_matrix,
              projector.block_rows);
      Eigen::VectorXd local_target =
          Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index local_index = 0;
           local_index < local_size;
           ++local_index) {
        local_target[local_index] =
            trial_occupied_orbitals(
                projector.block_rows[static_cast<std::size_t>(local_index)],
                occupied_index);
      }
      const Eigen::VectorXd local_retracted =
          retract_local_sparse_normalized_step(
              local_overlap,
              local_coefficients,
              local_target - local_coefficients);

      for (Eigen::Index local_index = 0;
           local_index < local_retracted.size();
           ++local_index) {
        updated_orbital_values
            [projector.flat_indices[local_index]] =
                local_retracted[local_index];
      }
    }
  }

  trial_input.orbital_value_table = std::move(updated_orbital_values);
  enforce_strict_sparse_orbital_support(&trial_input);
  return trial_input;
}

Eigen::VectorXd NonredundantOrbitalSpace::block_candidate_coefficients_from_reduced_step(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& reduced_step) const {
  const Eigen::VectorXd local_reduced =
      reduced_step.segment(
          block_basis.reduced_offset,
          block_basis.candidate_metric_diagonal.size());
  if (!block_basis.has_exact_metric_factorization) {
    return local_reduced;
  }
  return block_basis.candidate_metric_cholesky_factor
      .transpose()
      .triangularView<Eigen::Upper>()
      .solve(local_reduced);
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_block_candidate_metric(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& candidate_coefficients) const {
  if (block_basis.uses_dense_full_support_projector) {
    return apply_dense_full_support_candidate_metric(
        block_basis,
        candidate_coefficients);
  }

  Eigen::VectorXd packed_direction =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
  accumulate_block_candidate_combination(
      block_basis,
      candidate_coefficients,
      &packed_direction);
  return project_block_candidate_overlap(block_basis, packed_direction);
}

Eigen::VectorXd NonredundantOrbitalSpace::solve_block_candidate_metric(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& right_hand_side) const {
  if (right_hand_side.size() == 0) {
    return Eigen::VectorXd::Zero(0);
  }
  if (block_basis.uses_dense_full_support_projector) {
    return solve_dense_full_support_candidate_metric(
        block_basis,
        right_hand_side);
  }

  constexpr double kMinimumMetricDiagonal = 1.0e-10;
  const Eigen::ArrayXd inverse_diagonal =
      block_basis.candidate_metric_diagonal
          .array()
          .max(kMinimumMetricDiagonal)
          .inverse();
  Eigen::VectorXd solution =
      right_hand_side.array() * inverse_diagonal;
  Eigen::VectorXd residual =
      right_hand_side -
      apply_block_candidate_metric(block_basis, solution);
  const double rhs_norm = right_hand_side.norm();
  if (!(rhs_norm > 0.0) || !std::isfinite(rhs_norm)) {
    return Eigen::VectorXd::Zero(right_hand_side.size());
  }

  Eigen::VectorXd preconditioned_residual =
      residual.array() * inverse_diagonal;
  Eigen::VectorXd search_direction = preconditioned_residual;
  double residual_dot_preconditioned =
      residual.dot(preconditioned_residual);
  if (!std::isfinite(residual_dot_preconditioned) ||
      residual_dot_preconditioned <= 0.0) {
    return solution;
  }

  const double tolerance =
      1.0e-10 * std::max(1.0, rhs_norm);
  const int max_iterations =
      std::min(
          512,
          std::max(
              32,
              static_cast<int>(2 * std::sqrt(
                  static_cast<double>(right_hand_side.size())))));
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (residual.norm() <= tolerance) {
      break;
    }
    const Eigen::VectorXd metric_times_direction =
        apply_block_candidate_metric(block_basis, search_direction);
    const double denominator =
        search_direction.dot(metric_times_direction);
    if (!std::isfinite(denominator) ||
        denominator <= kMinimumMetricDiagonal) {
      break;
    }

    const double alpha =
        residual_dot_preconditioned / denominator;
    solution.noalias() += alpha * search_direction;
    residual.noalias() -= alpha * metric_times_direction;
    preconditioned_residual =
        residual.array() * inverse_diagonal;
    const double next_residual_dot_preconditioned =
        residual.dot(preconditioned_residual);
    if (!std::isfinite(next_residual_dot_preconditioned) ||
        next_residual_dot_preconditioned <= 0.0) {
      break;
    }
    const double beta =
        next_residual_dot_preconditioned / residual_dot_preconditioned;
    search_direction =
        preconditioned_residual + beta * search_direction;
    residual_dot_preconditioned =
        next_residual_dot_preconditioned;
  }

  return solution;
}

void NonredundantOrbitalSpace::maybe_factorize_small_block_candidate_metric(
    BlockBasis* block_basis) {
  const int direction_count =
      static_cast<int>(block_basis->candidate_metric_diagonal.size());
  if (direction_count <= 0 ||
      direction_count > kMaxExactMetricFactorizationDirections) {
    return;
  }

  // Small blocks can afford an exact factorization of `D^T D`. Whitening the
  // raw candidate amplitudes removes the remaining active-space
  // nonorthogonality from the reduced TN chart, so trust-region radii and
  // Krylov residuals are measured in an actual orthonormal tangent basis.
  Eigen::MatrixXd candidate_metric =
      Eigen::MatrixXd::Zero(direction_count, direction_count);
  Eigen::VectorXd unit_direction =
      Eigen::VectorXd::Zero(direction_count);
  for (int column = 0; column < direction_count; ++column) {
    unit_direction.setZero();
    unit_direction(column) = 1.0;
    candidate_metric.col(column) =
        apply_block_candidate_metric(*block_basis, unit_direction);
  }
  candidate_metric =
      0.5 * (candidate_metric + candidate_metric.transpose());

  Eigen::LLT<Eigen::MatrixXd> candidate_metric_llt(candidate_metric);
  if (candidate_metric_llt.info() != Eigen::Success) {
    return;
  }

  block_basis->has_exact_metric_factorization = true;
  block_basis->candidate_metric_cholesky_factor =
      candidate_metric_llt.matrixL();

  if (block_basis->reduced_curvature_diagonal.size() != direction_count) {
    return;
  }

  const Eigen::MatrixXd candidate_from_reduced =
      block_basis->candidate_metric_cholesky_factor
          .transpose()
          .triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(direction_count, direction_count));
  Eigen::VectorXd transformed_reduced_curvature =
      Eigen::VectorXd::Zero(direction_count);
  for (int reduced_index = 0; reduced_index < direction_count; ++reduced_index) {
    transformed_reduced_curvature(reduced_index) =
        (block_basis->reduced_curvature_diagonal.array() *
         candidate_from_reduced.col(reduced_index).array().square())
            .sum();
  }
  block_basis->reduced_curvature_diagonal =
      normalize_curvature_diagonal(transformed_reduced_curvature);
}

std::vector<NonredundantOrbitalSpace::BlockRotationDirection>
NonredundantOrbitalSpace::expand_block_rotation_directions(
    const Eigen::VectorXd& reduced_step) const {
  std::vector<BlockRotationDirection> directions;
  directions.reserve(block_bases_.size());
  for (const auto& block_basis : block_bases_) {
    const BlockDirectionLayout layout =
        build_block_direction_layout(
            block_basis.n_inactive,
            block_basis.n_occupied,
            block_basis.n_virtual);
    const int n_inactive = layout.n_inactive;
    const int n_occupied = layout.n_occupied;
    const int n_active = layout.n_active;
    const int n_virtual = layout.n_virtual;
    
    const Eigen::VectorXd candidate_coefficients =
        block_candidate_coefficients_from_reduced_step(
            block_basis,
            reduced_step);

    BlockRotationDirection direction;
    direction.n_inactive = n_inactive;
    direction.n_occupied = n_occupied;
    direction.n_virtual = n_virtual;
    direction.basis_function_indices = block_basis.basis_function_indices;
    direction.occupied_orbital_indices = block_basis.occupied_orbital_indices;
    if (block_basis.uses_dense_full_support_projector) {
      direction.occupied_orbitals = block_basis.occupied_orbitals;
      direction.virtual_orbitals = block_basis.virtual_orbitals;
    } else {
      direction.occupied_orbitals =
          Eigen::MatrixXd::Zero(
              static_cast<Eigen::Index>(block_basis.basis_function_indices.size()),
              n_occupied);
      if (n_inactive > 0) {
        direction.occupied_orbitals.leftCols(n_inactive) =
            block_basis.inactive_working_orbitals;
      }
      if (n_active > 0) {
        direction.occupied_orbitals.middleCols(
            n_inactive,
            n_active) = block_basis.active_working_orbitals;
      }
      direction.virtual_orbitals = block_basis.internal_virtual_orbitals;
    }
    direction.inactive_active_coefficients =
        Eigen::MatrixXd::Zero(
            n_active,
            n_inactive);
    direction.active_active_coefficients =
        Eigen::MatrixXd::Zero(
            n_active,
            n_active);
    direction.occupied_virtual_coefficients =
        Eigen::MatrixXd::Zero(
            n_virtual,
            n_occupied);

    if (layout.inactive_active_count > 0) {
      direction.inactive_active_coefficients =
          Eigen::Map<const Eigen::MatrixXd>(
              candidate_coefficients.data(),
              n_active,
              n_inactive);
    }
    if (layout.active_active_count > 0) {
      direction.active_active_coefficients =
          Eigen::Map<const Eigen::MatrixXd>(
              candidate_coefficients.data() + layout.inactive_active_count,
              n_active,
              n_active);
    }
    if (n_virtual > 0 && n_occupied > 0) {
      direction.occupied_virtual_coefficients =
          Eigen::Map<const Eigen::MatrixXd>(
              candidate_coefficients.data() + layout.occupied_virtual_offset,
              n_virtual,
              n_occupied);
    }

    directions.push_back(std::move(direction));
  }
  return directions;
}

}  // namespace xmvb::vb
