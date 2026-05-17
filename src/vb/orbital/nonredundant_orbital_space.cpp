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


}  // namespace

Eigen::MatrixXd build_block_effective_one_electron_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    int n_basis_functions,
    const std::vector<int>& basis_function_indices) {
  const int block_basis_count = static_cast<int>(basis_function_indices.size());
  Eigen::MatrixXd block_effective_one_electron =
      Eigen::MatrixXd::Zero(block_basis_count, block_basis_count);
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

Eigen::MatrixXd build_block_effective_one_electron_matrix_on_rows(
    const Eigen::MatrixXd& block_effective_h1e,
    const std::vector<int>& block_rows) {
  const Eigen::Index local_size =
      static_cast<Eigen::Index>(block_rows.size());
  Eigen::MatrixXd local_F =
      Eigen::MatrixXd::Zero(local_size, local_size);
  for (Eigen::Index row = 0; row < local_size; ++row) {
    const int block_row = block_rows[static_cast<std::size_t>(row)];
    for (Eigen::Index column = 0; column < local_size; ++column) {
      local_F(row, column) =
          block_effective_h1e(
              block_row,
              block_rows[static_cast<std::size_t>(column)]);
    }
  }
  return local_F;
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
    block_basis.has_full_ao_packed_support =
        orbital_preparation_input.orbital_type == kLegacyOrbitalTypeOeo;
    block_basis.orbitals.reserve(block_space.orbitals.size());
    for (const auto& orbital : block_space.orbitals) {
      if (orbital.coefficient_count != block_basis_count) {
        block_basis.has_full_ao_packed_support = false;
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
      if (block_basis.has_full_ao_packed_support) {
        projector.flat_index_by_block_row.assign(block_basis_count, -1);
        projector.packed_index_by_block_row.assign(block_basis_count, -1);
      }
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
        if (block_basis.has_full_ao_packed_support) {
          projector.flat_index_by_block_row[block_row] = flat_index;
          projector.packed_index_by_block_row[block_row] = packed_index;
        }
        projector.occupied_masked.row(coefficient_index) =
            block_raw_occupied_orbitals.row(block_row);
        if (n_virtual > 0) {
          projector.virtual_masked.row(coefficient_index) =
              block_space.virtual_orbitals.row(block_row);
        }
      }
      if (block_basis.has_full_ao_packed_support) {
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

    {
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

    if (block_basis.has_full_ao_packed_support) {
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
    }

    // Construct per-orbital physical tangent space basis U_p.
    // For each orbital p:
    //   1. Build raw basis B_p from allowed physical directions
    //   2. Project to tangent space: T_p = (I - x x^T S / rho^2) B_p
    //   3. Whiten: G_p = T_p^T T_p, Cholesky, U_p = T_p L^{-T}
    //   4. Store U_p with local_reduced_offset/size
    //   5. If Fock matrix available, compute curvature diagonal
    {
      // Build block-local Fock submatrix once per block for curvature.
      Eigen::MatrixXd block_effective_h1e;
      if (ao_effective_h1e != nullptr) {
        block_effective_h1e = build_block_effective_one_electron_matrix(
            *ao_effective_h1e,
            orbital_preparation_input.n_basis_functions,
            block_basis.basis_function_indices);
      }

      const int n_active = block_basis.n_occupied - block_basis.n_inactive;
      for (int occupied_index = 0;
           occupied_index < block_basis.n_occupied;
           ++occupied_index) {
        auto& projector = block_basis.orbitals[occupied_index];
        const Eigen::Index local_size =
            static_cast<Eigen::Index>(projector.block_rows.size());

        // Gather local coefficients x_p from orbital_value_table
        Eigen::VectorXd local_coefficients =
            Eigen::VectorXd::Zero(local_size);
        for (Eigen::Index i = 0; i < local_size; ++i) {
          local_coefficients[i] =
              orbital_preparation_input.orbital_value_table
                  [projector.flat_indices[i]];
        }
        const Eigen::MatrixXd local_overlap =
            build_local_sparse_overlap_metric(
                block_basis.block_overlap_matrix,
                projector.block_rows);

        // Compute rho^2 = x_p^T S_p x_p
        const Eigen::VectorXd Sx = local_overlap * local_coefficients;
        const double rho_squared = local_coefficients.dot(Sx);

        // Build raw basis B_p
        int raw_dim = 0;
        if (occupied_index < block_basis.n_inactive) {
          // inactive: [active_working, virtual]
          raw_dim = n_active + block_basis.n_virtual;
        } else {
          // active: [inactive_working, active_working (excluding self), virtual]
          raw_dim = block_basis.n_inactive + (n_active > 0 ? n_active - 1 : 0) +
                    block_basis.n_virtual;
        }

        if (raw_dim == 0) {
          projector.local_reduced_offset = reduced_size_;
          projector.local_reduced_size = 0;
          projector.tangent_basis =
              Eigen::MatrixXd::Zero(local_size, 0);
          continue;
        }

        Eigen::MatrixXd B_p =
            Eigen::MatrixXd::Zero(local_size, raw_dim);
        int col_offset = 0;

        if (occupied_index < block_basis.n_inactive) {
          // inactive orbital: allowed directions = active + virtual
          if (n_active > 0) {
            B_p.middleCols(col_offset, n_active) =
                projector.active_working_masked;
            col_offset += n_active;
          }
          if (block_basis.n_virtual > 0) {
            B_p.middleCols(col_offset, block_basis.n_virtual) =
                projector.internal_virtual_masked;
            col_offset += block_basis.n_virtual;
          }
        } else {
          const int active_index = occupied_index - block_basis.n_inactive;
          // active orbital: allowed directions = inactive + active(excl self) + virtual
          if (block_basis.n_inactive > 0) {
            B_p.middleCols(col_offset, block_basis.n_inactive) =
                projector.inactive_working_masked;
            col_offset += block_basis.n_inactive;
          }
          if (n_active > 0) {
            // active_working_masked has n_active columns (one per active orbital).
            // Exclude the self column (active_index).
            if (n_active > 1) {
              int dest_col = 0;
              for (int j = 0; j < n_active; ++j) {
                if (j == active_index) continue;
                B_p.col(col_offset + dest_col) =
                    projector.active_working_masked.col(j);
                ++dest_col;
              }
            }
            col_offset += n_active - 1;
          }
          if (block_basis.n_virtual > 0) {
            B_p.middleCols(col_offset, block_basis.n_virtual) =
                projector.internal_virtual_masked;
            col_offset += block_basis.n_virtual;
          }
        }

        // Tangent projection: T_p = (I - x_hat x_hat^T S_p) B_p
        // where x_hat = x_p / rho, so x_hat x_hat^T S_p = x_p S_p / rho^2
        Eigen::MatrixXd T_p = B_p;
        if (rho_squared > std::numeric_limits<double>::epsilon()) {
          const double inv_rho2 = 1.0 / rho_squared;
          const Eigen::VectorXd normalized_Sx = Sx * inv_rho2;
          // T_p = B_p - x_p * (x_p^T S_p B_p / rho^2)
          // = B_p - x_p * (normalized_Sx^T B_p)
          const Eigen::VectorXd projection = normalized_Sx.transpose() * B_p;
          T_p.noalias() -= local_coefficients * projection.transpose();
        }

        // Whiten: G_p = T_p^T T_p, Cholesky, U_p = T_p L^{-T}
        Eigen::MatrixXd G_p = T_p.transpose() * T_p;
        Eigen::LLT<Eigen::MatrixXd> llt(G_p);
        if (llt.info() != Eigen::Success) {
          // Degenerate — skip this orbital
          projector.local_reduced_offset = reduced_size_;
          projector.local_reduced_size = 0;
          projector.tangent_basis =
              Eigen::MatrixXd::Zero(local_size, 0);
          continue;
        }
        // U_p = T_p * L^{-T}: solve L^T X = T_p^T, then U_p = X^T
        const Eigen::MatrixXd L_invT_TpT =
            llt.matrixU().solve(T_p.transpose());
        Eigen::MatrixXd U_p = L_invT_TpT.transpose();

        projector.tangent_basis = std::move(U_p);
        projector.local_reduced_offset = reduced_size_;
        projector.local_reduced_size =
            static_cast<int>(projector.tangent_basis.cols());
        reduced_size_ += projector.local_reduced_size;

        // Compute curvature diagonal: diag(U_p^T (F_p - eps_p S_p) U_p)
        // where eps_p = x_p^T F_p x_p / rho_p^2 is the orbital energy.
        if (ao_effective_h1e != nullptr &&
            projector.local_reduced_size > 0) {
          // Build orbital-local Fock and overlap on block_rows.
          const Eigen::MatrixXd local_F =
              build_block_effective_one_electron_matrix_on_rows(
                  block_effective_h1e, projector.block_rows);
          // eps_p = x_p^T F x_p / rho_p^2
          const double eps_p =
              rho_squared > std::numeric_limits<double>::epsilon()
                  ? local_coefficients.dot(local_F * local_coefficients) /
                        rho_squared
                  : 0.0;
          // H_p = U_p^T (F_p - eps_p S_p) U_p, take diagonal
          const Eigen::MatrixXd F_U = local_F * projector.tangent_basis;
          const Eigen::MatrixXd S_U = local_overlap * projector.tangent_basis;
          Eigen::VectorXd curv(projector.local_reduced_size);
          for (int k = 0; k < projector.local_reduced_size; ++k) {
            curv[k] = projector.tangent_basis.col(k).dot(F_U.col(k)) -
                      eps_p * projector.tangent_basis.col(k).dot(S_U.col(k));
          }
          projector.curvature_diagonal =
              normalize_curvature_diagonal(curv);
          has_reduced_curvature_diagonal_ = true;
        }
      }
    }

    block_basis.reduced_offset =
        block_basis.orbitals.empty()
            ? reduced_size_
            : block_basis.orbitals.front().local_reduced_offset;

    block_bases_.push_back(std::move(block_basis));
  }
}

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_impl(
    const Eigen::VectorXd& packed_vector,
    bool recover_tangent_coordinates,
    bool build_packed_projection) const {
  (void)recover_tangent_coordinates;
  ProjectionResult result;
  result.reduced_gradient =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(reduced_size_));
  result.packed_projected_gradient =
      build_packed_projection
          ? Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_))
          : Eigen::VectorXd();

  // Physical tangent projection: z_p = U_p^T g_p for each orbital
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.local_reduced_size <= 0) continue;
      const Eigen::Index local_size =
          static_cast<Eigen::Index>(projector.packed_indices.size());
      Eigen::VectorXd g_p = Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index i = 0; i < local_size; ++i) {
        g_p[i] = packed_vector[projector.packed_indices[i]];
      }
      const Eigen::VectorXd z_p =
          projector.tangent_basis.transpose() * g_p;
      result.reduced_gradient.segment(
          projector.local_reduced_offset,
          projector.local_reduced_size) = z_p;
      if (build_packed_projection) {
        const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
        for (Eigen::Index i = 0; i < dx_p.size(); ++i) {
          result.packed_projected_gradient[projector.packed_indices[i]] +=
              dx_p[i];
        }
      }
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
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.curvature_diagonal.size() == 0) continue;
      preconditioned.segment(
          projector.local_reduced_offset,
          projector.local_reduced_size)
          .array() /=
          projector.curvature_diagonal
              .array()
              .max(kMinimumCurvature);
    }
  }
  return preconditioned;
}

Eigen::VectorXd
NonredundantOrbitalSpace::apply_inverse_reduced_block_preconditioner(
    const Eigen::VectorXd& reduced_vector) const {
  // With U_p^T U_p = I, the reduced metric is identity.  Use the
  // per-orbital curvature diagonal as the preconditioner model.
  return apply_inverse_reduced_curvature(reduced_vector);
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  if (!has_reduced_curvature_diagonal_) {
    return reduced_vector;
  }

  Eigen::VectorXd curved = reduced_vector;
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.curvature_diagonal.size() == 0) continue;
      curved.segment(
          projector.local_reduced_offset,
          projector.local_reduced_size)
          .array() *= projector.curvature_diagonal.array();
    }
  }
  return curved;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_step(
    const Eigen::VectorXd& reduced_step) const {
  Eigen::VectorXd packed_step =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.local_reduced_size <= 0) continue;
      const Eigen::VectorXd z_p =
          reduced_step.segment(
              projector.local_reduced_offset,
              projector.local_reduced_size);
      const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
      for (Eigen::Index i = 0; i < dx_p.size(); ++i) {
        packed_step[projector.packed_indices[i]] += dx_p[i];
      }
    }
  }
  return packed_step;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_retract_input_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step) const {
  // Linear-add: retract tangent = identity = same as expand_step in flat space.
  Eigen::VectorXd input_tangent =
      Eigen::VectorXd::Zero(
          static_cast<Eigen::Index>(
              orbital_preparation_input.orbital_value_table.size()));
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.local_reduced_size <= 0) continue;
      const Eigen::VectorXd z_p =
          reduced_step.segment(
              projector.local_reduced_offset,
              projector.local_reduced_size);
      const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
      for (Eigen::Index i = 0; i < dx_p.size(); ++i) {
        input_tangent[projector.flat_indices[i]] += dx_p[i];
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

  // Linear-add retraction: x_p^+ = x_p + alpha * U_p z_p
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.local_reduced_size <= 0) continue;
      const Eigen::VectorXd z_p =
          reduced_step.segment(
              projector.local_reduced_offset,
              projector.local_reduced_size);
      const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
      for (Eigen::Index i = 0; i < dx_p.size(); ++i) {
        updated_orbital_values[projector.flat_indices[i]] +=
            step_scale * dx_p[i];
      }
    }
  }

  trial_input.orbital_value_table = std::move(updated_orbital_values);
  enforce_strict_sparse_orbital_support(&trial_input);
  return trial_input;
}

}  // namespace xmvb::vb
