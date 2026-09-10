#include "vbscf/derivatives/hessian/responses/structure_directional_response.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <cblas.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "vbscf/determinants/pair_storage.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"
#include "vbscf/structures/assembly/block_kernels.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin_channels.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin_backward.hpp"

namespace xmvb::vb {
namespace {

int current_openmp_max_threads() {
  int max_threads = 1;
#ifdef _OPENMP
  max_threads = omp_get_max_threads();
#endif
  return std::max(1, max_threads);
}

int choose_directional_structure_threads(int n_structures) {
  return std::min(current_openmp_max_threads(), std::max(1, n_structures));
}

void throw_if_nonfinite(
    const std::vector<double>& values,
    const char* label) {
  const auto nonfinite_it = std::find_if(
      values.begin(),
      values.end(),
      [](double value) { return !std::isfinite(value); });
  if (nonfinite_it == values.end()) {
    return;
  }
  throw std::runtime_error(std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::MatrixXd& values,
    const char* label) {
  if (!values.allFinite()) {
    throw std::runtime_error(std::string(label) + " contains non-finite values");
  }
}

double lookup_directional_active_two_electron_kernel_value(
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    int row_packed_pair_index,
    int column_packed_pair_index) {
  if (delta_packed_active_two_electron_integrals.empty()) {
    return 0.0;
  }
  const int packed_pair_of_pairs_index =
      TwoElectronIndexer::packed_pair_of_pairs_index(
          row_packed_pair_index,
          column_packed_pair_index);
  return delta_packed_active_two_electron_integrals[
      packed_pair_of_pairs_index];
}

constexpr double kDirectionalStructureContributionTolerance = 1.0e-15;
struct ProjectionBlock {
  struct ProjectionSlot {
    const OppositeSpinPackedPairProjection* borrowed_projection = nullptr;
    OppositeSpinPackedPairProjection owned_projection;

    const OppositeSpinPackedPairProjection& projection() const {
      return (borrowed_projection != nullptr)
          ? *borrowed_projection
          : owned_projection;
    }
  };

  int n_rows = 0;
  int n_cols = 0;
  std::vector<ProjectionSlot> projections;

  const OppositeSpinPackedPairProjection& projection(
      int row_local,
      int column_local) const {
    return projections[(column_local) * (n_rows) + (row_local)]
        .projection();
  }
};

struct OppositeSpinChannelFamily {
  std::vector<int> packed_pair_indices;
  std::vector<Eigen::MatrixXd> alpha_channel_matrices;
};

int count_projection_pair_union(
    const ProjectionBlock& first_projection_block,
    const ProjectionBlock& second_projection_block,
    int n_packed_active_pairs) {
  if (first_projection_block.n_rows != second_projection_block.n_rows ||
      first_projection_block.n_cols != second_projection_block.n_cols) {
    throw std::invalid_argument(
        "projection blocks must share the same local shape");
  }
  if (first_projection_block.n_rows <= 0 ||
      first_projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return 0;
  }

  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0u);
  int distinct_pair_count = 0;
  for (int column_local = 0;
       column_local < first_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < first_projection_block.n_rows;
         ++row_local) {
      const auto& first_projection =
          first_projection_block.projection(row_local, column_local);
      for (const int packed_pair_index : first_projection.packed_pair_indices) {
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        if (touched_mask[packed_pair_index] == 0u) {
          touched_mask[packed_pair_index] = 1u;
          ++distinct_pair_count;
        }
      }
      const auto& second_projection =
          second_projection_block.projection(row_local, column_local);
      for (const int packed_pair_index : second_projection.packed_pair_indices) {
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        if (touched_mask[packed_pair_index] == 0u) {
          touched_mask[packed_pair_index] = 1u;
          ++distinct_pair_count;
        }
      }
    }
  }
  return distinct_pair_count;
}

OppositeSpinChannelFamily build_channel_family(
    const ProjectionBlock& projection_block,
    int n_packed_active_pairs) {
  OppositeSpinChannelFamily channel_family;
  if (projection_block.n_rows <= 0 ||
      projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return channel_family;
  }

  std::vector<int> channel_index_by_packed_pair(
      n_packed_active_pairs,
      -1);
  for (int column_local = 0;
       column_local < projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          projection_block.projection(row_local, column_local);
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <=
            kDirectionalStructureContributionTolerance) {
          continue;
        }
        int& channel_index =
            channel_index_by_packed_pair[packed_pair_index];
        if (channel_index < 0) {
          channel_index =
              static_cast<int>(channel_family.packed_pair_indices.size());
          channel_family.packed_pair_indices.push_back(packed_pair_index);
          channel_family.alpha_channel_matrices.emplace_back(
              Eigen::MatrixXd::Zero(
                  projection_block.n_rows,
                  projection_block.n_cols));
        }
        channel_family.alpha_channel_matrices[channel_index](row_local, column_local) =
            packed_pair_value;
      }
    }
  }
  return channel_family;
}

/**
 * @brief Reuses the packed-pair to local-channel map across many block gathers.
 *
 * Each gathered structure block only touches a small subset of packed pairs.
 * Reinitializing an `n_packed_active_pairs` sized lookup table for every block
 * would add avoidable O(n_pairs) work. This builder tracks only the touched
 * packed pairs and resets those entries between block gathers.
 */
struct OppositeSpinChannelBuilder {
  explicit OppositeSpinChannelBuilder(
      int n_packed_active_pairs)
      : channel_index_by_packed_pair(
            n_packed_active_pairs,
            -1) {
    touched_packed_pair_indices.reserve(
        n_packed_active_pairs);
  }

  void reset(OppositeSpinChannelFamily* channel_family) {
    for (const int packed_pair_index : touched_packed_pair_indices) {
      channel_index_by_packed_pair[packed_pair_index] = -1;
    }
    touched_packed_pair_indices.clear();
    channel_family->packed_pair_indices.clear();
  }

  void accumulate(
      const OppositeSpinPackedPairProjection& projection,
      int n_rows,
      int n_cols,
      int row_local,
      int column_local,
      OppositeSpinChannelFamily* channel_family) {
    for (std::size_t entry_index = 0;
         entry_index < projection.packed_pair_indices.size();
         ++entry_index) {
      const int packed_pair_index = projection.packed_pair_indices[entry_index];
      if (packed_pair_index < 0 ||
          packed_pair_index >=
              static_cast<int>(channel_index_by_packed_pair.size())) {
        throw std::out_of_range(
            "packed_pair_index outside local channel range");
      }
      const double packed_pair_value =
          projection.packed_pair_values[entry_index];
      if (std::abs(packed_pair_value) <=
          kDirectionalStructureContributionTolerance) {
        continue;
      }
      int& channel_index =
          channel_index_by_packed_pair[packed_pair_index];
      if (channel_index < 0) {
        channel_index =
            static_cast<int>(channel_family->packed_pair_indices.size());
        touched_packed_pair_indices.push_back(packed_pair_index);
        channel_family->packed_pair_indices.push_back(packed_pair_index);
        if (channel_index <
            static_cast<int>(channel_family->alpha_channel_matrices.size())) {
          Eigen::MatrixXd& reused_channel_matrix =
              channel_family->alpha_channel_matrices[channel_index];
          reused_channel_matrix.resize(n_rows, n_cols);
          reused_channel_matrix.setZero();
        } else {
          channel_family->alpha_channel_matrices.emplace_back(
              Eigen::MatrixXd::Zero(n_rows, n_cols));
        }
      }
      channel_family->alpha_channel_matrices[channel_index](row_local, column_local) = packed_pair_value;
    }
  }

private:
  std::vector<int> channel_index_by_packed_pair;
  std::vector<int> touched_packed_pair_indices;
};

struct DirectionalSpinPairEntry {
  double overlap_determinant = 0.0;
  double total_hamiltonian = 0.0;
  double delta_overlap_determinant = 0.0;
  double delta_total_hamiltonian = 0.0;
  OppositeSpinPackedPairProjection delta_first_order_projection;
};

std::size_t square_storage_size(int dimension) {
  return dimension * dimension;
}

void reset_projection_block(
    int n_rows,
    int n_cols,
    ProjectionBlock* local_projection_block) {
  if (local_projection_block == nullptr) {
    return;
  }
  local_projection_block->n_rows = n_rows;
  local_projection_block->n_cols = n_cols;
  // Every slot is overwritten before it is consumed, so avoid an extra full
  // memory pass that default-initializes the whole block on every gather.
  local_projection_block->projections.resize(
      n_rows * n_cols);
}

double frobenius_inner_product(
    const Eigen::MatrixXd& left_matrix,
    const Eigen::MatrixXd& right_matrix) {
  if (left_matrix.size() == 0) {
    return 0.0;
  }
  return cblas_ddot(
      static_cast<int>(left_matrix.size()),
      left_matrix.data(),
      1,
      right_matrix.data(),
      1);
}

double contract_structure_pair_kernel(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_kernel,
    const Eigen::MatrixXd& beta_kernel,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  if (left_coefficients.size() == 0 || right_coefficients.size() == 0) {
    return 0.0;
  }

  beta_push->resize(right_coefficients.rows(), beta_kernel.rows());
  beta_push->noalias() = right_coefficients * beta_kernel.transpose();
  image->resize(alpha_kernel.rows(), beta_kernel.rows());
  image->noalias() = alpha_kernel * (*beta_push);
  return frobenius_inner_product(left_coefficients, *image);
}

struct SameSpinContractionScratch {
  Eigen::MatrixXd alpha_overlap_push;
  Eigen::MatrixXd alpha_total_push;
  Eigen::MatrixXd alpha_delta_overlap_push;
  Eigen::MatrixXd alpha_delta_total_push;
  Eigen::MatrixXd beta_overlap_push;
  Eigen::MatrixXd beta_total_push;
  Eigen::MatrixXd beta_delta_overlap_push;
  Eigen::MatrixXd beta_delta_total_push;
};

struct DirectionalSameSpinContribution {
  double overlap = 0.0;
  double hamiltonian = 0.0;
};

DirectionalSameSpinContribution
contract_close_shell_same_spin_direction(
    const std::vector<double>& left_diagonal_coefficients,
    const std::vector<double>& right_diagonal_coefficients,
    const Eigen::MatrixXd& overlap_subblock,
    const Eigen::MatrixXd& total_subblock,
    const Eigen::MatrixXd& delta_overlap_subblock,
    const Eigen::MatrixXd& delta_total_subblock) {
  // Close-shell diagonal structure blocks satisfy
  //   C_L = diag(l), C_R = diag(r), alpha == beta,
  // so the directional same-spin contraction reduces to one gathered spin
  // channel and a factor of two for the duplicated alpha/beta contribution.
  DirectionalSameSpinContribution result;
  result.overlap =
      2.0 *
      contract_diagonal_structure_pair_kernel(
          left_diagonal_coefficients,
          right_diagonal_coefficients,
          delta_overlap_subblock,
          overlap_subblock);
  result.hamiltonian =
      2.0 *
      (contract_diagonal_structure_pair_kernel(
           left_diagonal_coefficients,
           right_diagonal_coefficients,
           delta_total_subblock,
           overlap_subblock) +
       contract_diagonal_structure_pair_kernel(
           left_diagonal_coefficients,
           right_diagonal_coefficients,
           total_subblock,
           delta_overlap_subblock));
  return result;
}

void build_left_alpha_push(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_kernel,
    Eigen::MatrixXd* alpha_push) {
  alpha_push->resize(alpha_kernel.cols(), left_coefficients.cols());
  alpha_push->noalias() = alpha_kernel.transpose() * left_coefficients;
}

void build_right_beta_push(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& beta_kernel,
    Eigen::MatrixXd* beta_push) {
  beta_push->resize(right_coefficients.rows(), beta_kernel.rows());
  beta_push->noalias() = right_coefficients * beta_kernel.transpose();
}

DirectionalSameSpinContribution
contract_same_spin_direction(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_overlap_subblock,
    const Eigen::MatrixXd& alpha_total_subblock,
    const Eigen::MatrixXd& alpha_delta_overlap_subblock,
    const Eigen::MatrixXd& alpha_delta_total_subblock,
    const Eigen::MatrixXd& beta_overlap_subblock,
    const Eigen::MatrixXd& beta_total_subblock,
    const Eigen::MatrixXd& beta_delta_overlap_subblock,
    const Eigen::MatrixXd& beta_delta_total_subblock,
    SameSpinContractionScratch* scratch) {
  if (left_coefficients.size() == 0 || right_coefficients.size() == 0) {
    return {};
  }

  // Each same-spin term has the form
  //   <L, A * (R * B^T)> = <A^T * L, R * B^T>.
  // The directional overlap/Hamiltonian need six such pairings, with repeated
  // alpha and beta kernels. Building the four unique left pushes and four
  // unique right pushes once per structure pair avoids repeated dense GEMMs
  // while preserving the mathematical contraction and matrix dimensions:
  //   L: n_alpha_left x n_beta_left
  //   R: n_alpha_right x n_beta_right
  //   A^T L and R B^T: n_alpha_right x n_beta_left.
  build_left_alpha_push(
      left_coefficients,
      right_coefficients,
      alpha_overlap_subblock,
      &scratch->alpha_overlap_push);
  build_left_alpha_push(
      left_coefficients,
      right_coefficients,
      alpha_total_subblock,
      &scratch->alpha_total_push);
  build_left_alpha_push(
      left_coefficients,
      right_coefficients,
      alpha_delta_overlap_subblock,
      &scratch->alpha_delta_overlap_push);
  build_left_alpha_push(
      left_coefficients,
      right_coefficients,
      alpha_delta_total_subblock,
      &scratch->alpha_delta_total_push);
  build_right_beta_push(
      left_coefficients,
      right_coefficients,
      beta_overlap_subblock,
      &scratch->beta_overlap_push);
  build_right_beta_push(
      left_coefficients,
      right_coefficients,
      beta_total_subblock,
      &scratch->beta_total_push);
  build_right_beta_push(
      left_coefficients,
      right_coefficients,
      beta_delta_overlap_subblock,
      &scratch->beta_delta_overlap_push);
  build_right_beta_push(
      left_coefficients,
      right_coefficients,
      beta_delta_total_subblock,
      &scratch->beta_delta_total_push);

  DirectionalSameSpinContribution result;
  result.overlap =
      frobenius_inner_product(
          scratch->alpha_delta_overlap_push,
          scratch->beta_overlap_push) +
      frobenius_inner_product(
          scratch->alpha_overlap_push,
          scratch->beta_delta_overlap_push);
  result.hamiltonian =
      frobenius_inner_product(
          scratch->alpha_delta_total_push,
          scratch->beta_overlap_push) +
      frobenius_inner_product(
          scratch->alpha_total_push,
          scratch->beta_delta_overlap_push) +
      frobenius_inner_product(
          scratch->alpha_delta_overlap_push,
          scratch->beta_total_push) +
      frobenius_inner_product(
          scratch->alpha_overlap_push,
          scratch->beta_delta_total_push);
  return result;
}

std::vector<double> apply_directional_two_electron_kernel(
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<double>& delta_packed_active_two_electron_integrals);

void materialize_directional_projected_pair_values(
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_orbitals,
    const OppositeSpinPackedPairProjection& accepted_projection,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    OppositeSpinPackedPairProjection* directional_projection) {
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  directional_projection->projected_pair_values =
      apply_active_space_two_electron_kernel_to_sparse_projection(
          two_electron_view,
          n_orbitals,
          directional_projection->packed_pair_indices,
          directional_projection->packed_pair_values);
  const std::vector<double> delta_kernel_times_accepted =
      apply_directional_two_electron_kernel(
          n_orbitals,
          accepted_projection.packed_pair_indices,
          accepted_projection.packed_pair_values,
          delta_packed_active_two_electron_integrals);
  if (directional_projection->projected_pair_values.size() <
      delta_kernel_times_accepted.size()) {
    directional_projection->projected_pair_values.resize(
        delta_kernel_times_accepted.size(),
        0.0);
  }
  for (std::size_t pair_index = 0;
       pair_index < delta_kernel_times_accepted.size();
       ++pair_index) {
    directional_projection->projected_pair_values[pair_index] +=
        delta_kernel_times_accepted[pair_index];
  }
}

OppositeSpinPackedPairProjection
build_sparse_packed_pair_projection(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& coefficient_matrix,
    bool coefficient_matrix_is_right_by_left,
    int n_orbitals) {
  OppositeSpinPackedPairProjection projection;
  if (occ_L.empty()) {
    return projection;
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  std::vector<double> dense_pair_values(
      n_packed_active_pairs,
      0.0);
  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0u);
  std::vector<int> touched_indices;
  touched_indices.reserve(occ_L.size() * occ_R.size());

  for (int left_column = 0;
       left_column < static_cast<int>(occ_L.size());
       ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0;
         right_row < static_cast<int>(occ_R.size());
         ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      if (touched_mask[packed_pair_index] == 0u) {
        touched_mask[packed_pair_index] = 1u;
        touched_indices.push_back(packed_pair_index);
      }
      const double coefficient =
          coefficient_matrix_is_right_by_left
              ? coefficient_matrix(right_row, left_column)
              : coefficient_matrix(left_column, right_row);
      dense_pair_values[packed_pair_index] += coefficient;
    }
  }

  projection.packed_pair_indices.reserve(touched_indices.size());
  projection.packed_pair_values.reserve(touched_indices.size());
  for (const int packed_pair_index : touched_indices) {
    const double packed_pair_value =
        dense_pair_values[packed_pair_index];
    if (std::abs(packed_pair_value) <=
        kDirectionalStructureContributionTolerance) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(packed_pair_value);
  }
  return projection;
}

std::vector<double> apply_directional_two_electron_kernel(
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> projected_pair_values(
      n_packed_active_pairs,
      0.0);
  if (delta_packed_active_two_electron_integrals.empty() ||
      packed_pair_indices.empty()) {
    return projected_pair_values;
  }

  for (std::size_t entry_index = 0;
       entry_index < packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index = packed_pair_indices[entry_index];
    const double packed_pair_value = packed_pair_values[entry_index];
    for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_pair,
              packed_pair_index);
      projected_pair_values[row_pair] +=
          delta_packed_active_two_electron_integrals[
              packed_pair_of_pairs_index] *
          packed_pair_value;
    }
  }
  return projected_pair_values;
}

double project_sparse_pair(
    const OppositeSpinPackedPairProjection& sparse_projection,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals) {
  if (target_packed_pair_index >= 0 &&
      target_packed_pair_index <
          static_cast<int>(sparse_projection.projected_pair_values.size())) {
    return sparse_projection.projected_pair_values[
        target_packed_pair_index];
  }

  double projected_value = 0.0;
  for (std::size_t entry_index = 0;
       entry_index < sparse_projection.packed_pair_indices.size();
       ++entry_index) {
    projected_value +=
        lookup_active_space_two_electron_kernel_value(
            two_electron_view,
            target_packed_pair_index,
            sparse_projection.packed_pair_indices[entry_index],
            n_orbitals) *
        sparse_projection.packed_pair_values[entry_index];
  }
  return projected_value;
}

double project_directional_sparse_pair(
    const OppositeSpinPackedPairProjection& accepted_projection,
    const OppositeSpinPackedPairProjection& directional_projection,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  if (target_packed_pair_index >= 0 &&
      target_packed_pair_index <
          static_cast<int>(directional_projection.projected_pair_values.size())) {
    return directional_projection.projected_pair_values[
        target_packed_pair_index];
  }
  double projected_value =
      project_sparse_pair(
          directional_projection,
          target_packed_pair_index,
          two_electron_view,
          n_orbitals);
  for (std::size_t entry_index = 0;
       entry_index < accepted_projection.packed_pair_indices.size();
       ++entry_index) {
    projected_value +=
        lookup_directional_active_two_electron_kernel_value(
            delta_packed_active_two_electron_integrals,
            target_packed_pair_index,
            accepted_projection.packed_pair_indices[entry_index]) *
        accepted_projection.packed_pair_values[entry_index];
  }
  return projected_value;
}

void build_projected_channel_block(
    const ProjectionBlock& local_projection_block,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* projected_channel_block) {
  projected_channel_block->resize(
      local_projection_block.n_rows,
      local_projection_block.n_cols);
  for (int column_local = 0;
       column_local < local_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < local_projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          local_projection_block.projection(
              row_local,
              column_local);
      (*projected_channel_block)(row_local, column_local) =
          project_sparse_pair(
              projection,
              target_packed_pair_index,
              two_electron_view,
              n_orbitals);
    }
  }
}

void build_directional_projected_channel_block(
    const ProjectionBlock& accepted_projection_block,
    const ProjectionBlock& directional_projection_block,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* projected_channel_block) {
  projected_channel_block->resize(
      accepted_projection_block.n_rows,
      accepted_projection_block.n_cols);
  for (int column_local = 0;
       column_local < accepted_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < accepted_projection_block.n_rows;
         ++row_local) {
      const auto& accepted_projection =
          accepted_projection_block.projection(
              row_local,
              column_local);
      const auto& directional_projection =
          directional_projection_block.projection(
              row_local,
              column_local);
      (*projected_channel_block)(row_local, column_local) =
          project_directional_sparse_pair(
              accepted_projection,
              directional_projection,
              target_packed_pair_index,
              two_electron_view,
              n_orbitals,
              delta_packed_active_two_electron_integrals);
    }
  }
}

double contract_opposite_spin_direction(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const ProjectionBlock& accepted_alpha_projection_block,
    const ProjectionBlock& directional_alpha_projection_block,
    const OppositeSpinChannelFamily& accepted_alpha_channels,
    const OppositeSpinChannelFamily& directional_alpha_channels,
    const ProjectionBlock& accepted_beta_projection_block,
    const ProjectionBlock& directional_beta_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* beta_projected_channel_block,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  const int alpha_channel_count =
      count_projection_pair_union(
          accepted_alpha_projection_block,
          directional_alpha_projection_block,
          n_packed_active_pairs);
  const int beta_channel_count =
      count_projection_pair_union(
          accepted_beta_projection_block,
          directional_beta_projection_block,
          n_packed_active_pairs);
  double contraction = 0.0;
  if (alpha_channel_count <= beta_channel_count) {
    for (std::size_t channel_index = 0;
         channel_index < accepted_alpha_channels.packed_pair_indices.size();
         ++channel_index) {
      build_directional_projected_channel_block(
          accepted_beta_projection_block,
          directional_beta_projection_block,
          accepted_alpha_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          delta_packed_active_two_electron_integrals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel(
              left_coefficients,
              right_coefficients,
              accepted_alpha_channels.alpha_channel_matrices[channel_index],
              *beta_projected_channel_block,
              beta_push,
              image);
    }
    for (std::size_t channel_index = 0;
         channel_index < directional_alpha_channels.packed_pair_indices.size();
         ++channel_index) {
      build_projected_channel_block(
          accepted_beta_projection_block,
          directional_alpha_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel(
              left_coefficients,
              right_coefficients,
              directional_alpha_channels.alpha_channel_matrices[channel_index],
              *beta_projected_channel_block,
              beta_push,
              image);
    }
  } else {
    const OppositeSpinChannelFamily accepted_beta_channels =
        build_channel_family(
            accepted_beta_projection_block,
            n_packed_active_pairs);
    const OppositeSpinChannelFamily directional_beta_channels =
        build_channel_family(
            directional_beta_projection_block,
            n_packed_active_pairs);
    for (std::size_t channel_index = 0;
         channel_index < accepted_beta_channels.packed_pair_indices.size();
         ++channel_index) {
      build_directional_projected_channel_block(
          accepted_alpha_projection_block,
          directional_alpha_projection_block,
          accepted_beta_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          delta_packed_active_two_electron_integrals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel(
              left_coefficients,
              right_coefficients,
              *beta_projected_channel_block,
              accepted_beta_channels.alpha_channel_matrices[channel_index],
              beta_push,
              image);
    }
    for (std::size_t channel_index = 0;
         channel_index < directional_beta_channels.packed_pair_indices.size();
         ++channel_index) {
      build_projected_channel_block(
          accepted_alpha_projection_block,
          directional_beta_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel(
              left_coefficients,
              right_coefficients,
              *beta_projected_channel_block,
              directional_beta_channels.alpha_channel_matrices[channel_index],
              beta_push,
              image);
    }
  }
  return contraction;
}

double contract_close_shell_opposite_spin_direction(
    const std::vector<double>& left_diagonal_coefficients,
    const std::vector<double>& right_diagonal_coefficients,
    const OppositeSpinChannelFamily& accepted_alpha_channels,
    const OppositeSpinChannelFamily& directional_alpha_channels,
    const ProjectionBlock& accepted_projection_block,
    const ProjectionBlock& directional_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* projected_channel_block) {
  double contraction = 0.0;
  for (std::size_t channel_index = 0;
       channel_index < accepted_alpha_channels.packed_pair_indices.size();
       ++channel_index) {
    build_directional_projected_channel_block(
        accepted_projection_block,
        directional_projection_block,
        accepted_alpha_channels.packed_pair_indices[channel_index],
        two_electron_view,
        n_orbitals,
        delta_packed_active_two_electron_integrals,
        projected_channel_block);
    contraction +=
        contract_diagonal_structure_pair_kernel(
            left_diagonal_coefficients,
            right_diagonal_coefficients,
            accepted_alpha_channels.alpha_channel_matrices[channel_index],
            *projected_channel_block);
  }
  for (std::size_t channel_index = 0;
       channel_index < directional_alpha_channels.packed_pair_indices.size();
       ++channel_index) {
    build_projected_channel_block(
        accepted_projection_block,
        directional_alpha_channels.packed_pair_indices[channel_index],
        two_electron_view,
        n_orbitals,
        projected_channel_block);
    contraction +=
        contract_diagonal_structure_pair_kernel(
            left_diagonal_coefficients,
            right_diagonal_coefficients,
            directional_alpha_channels.alpha_channel_matrices[channel_index],
            *projected_channel_block);
  }
  return contraction;
}

DirectionalSpinPairEntry build_directional_spin_pair_entry(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinPolynomialDirectionalPairData& directional_pair_data) {
  DirectionalSpinPairEntry result;
  result.overlap_determinant = pair_evaluation.overlap_result.overlap_determinant;
  result.total_hamiltonian = pair_evaluation.total_hamiltonian;
  if (occ_L.empty()) {
    return result;
  }
  result.delta_overlap_determinant =
      directional_pair_data.delta_overlap_determinant;
  result.delta_total_hamiltonian =
      directional_pair_data.delta_total_hamiltonian;
  result.delta_first_order_projection =
      build_sparse_packed_pair_projection(
          occ_L,
          occ_R,
          directional_pair_data.delta_cofactor_1st,
          true,
          n_active_orbitals);
  materialize_directional_projected_pair_values(
      active_space_two_electron_result,
      n_active_orbitals,
      pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection,
      delta_packed_active_two_electron_integrals,
      &result.delta_first_order_projection);
  return result;
}

// One directional structure build visits a sparse, support-induced subset of
// the accepted determinant-pair table. Memoize exactly that subset for the
// lifetime of the build: no fixed tile shape, eviction budget, or recomputation
// policy is needed, and untouched determinant pairs consume no directional
// storage.
class DirectionalSpinPairMemo {
public:
  DirectionalSpinPairMemo(
      const std::vector<std::vector<int>>& unique_spin_determinants,
      const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_active_orbitals,
      const std::vector<double>& delta_packed_active_two_electron_integrals,
      const std::vector<SameSpinPolynomialDirectionalPairData>&
          directional_pair_data)
      : unique_spin_determinants_(unique_spin_determinants),
        ordered_pair_cache_(ordered_pair_cache),
        active_space_two_electron_result_(active_space_two_electron_result),
        n_active_orbitals_(n_active_orbitals),
        delta_packed_active_two_electron_integrals_(
            delta_packed_active_two_electron_integrals),
        directional_pair_data_(directional_pair_data),
        cached_entries_(square_storage_size(
            static_cast<int>(unique_spin_determinants.size()))),
        entry_flags_(square_storage_size(
            static_cast<int>(unique_spin_determinants.size()))) {
    const std::size_t expected_cache_entries =
        square_storage_size(static_cast<int>(unique_spin_determinants_.size()));
    if (ordered_pair_cache_.size() != expected_cache_entries) {
      throw std::invalid_argument(
          "ordered same-spin pair cache size does not match unique determinant count");
    }
    if (directional_pair_data_.size() != expected_cache_entries) {
      throw std::invalid_argument(
          "directional same-spin pair data does not match unique determinant count");
    }
  }

  const SpinDeterminantPairEvaluation& pair_evaluation(
      int left_unique_index,
      int right_unique_index) const {
    if (left_unique_index < 0 ||
        left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
        right_unique_index < 0 ||
        right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
      throw std::out_of_range("directional spin pair lookup index out of range");
    }
    return ordered_pair_cache_[ordered_spin_pair_storage_index(
        left_unique_index,
        right_unique_index,
        static_cast<int>(unique_spin_determinants_.size()))];
  }

  const DirectionalSpinPairEntry& entry(
      int left_unique_index,
      int right_unique_index) {
    if (left_unique_index < 0 ||
        left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
        right_unique_index < 0 ||
        right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
      throw std::out_of_range("directional spin pair lookup index out of range");
    }

    const std::size_t pair_index = ordered_spin_pair_storage_index(
        left_unique_index,
        right_unique_index,
        static_cast<int>(unique_spin_determinants_.size()));
    std::call_once(entry_flags_[pair_index], [&]() {
      cached_entries_[pair_index] = build_directional_spin_pair_entry(
            unique_spin_determinants_[left_unique_index],
            unique_spin_determinants_[right_unique_index],
            active_space_two_electron_result_,
            ordered_pair_cache_[pair_index],
            n_active_orbitals_,
            delta_packed_active_two_electron_integrals_,
            directional_pair_data_[pair_index]);
    });
    return cached_entries_[pair_index];
  }

private:
  const std::vector<std::vector<int>>& unique_spin_determinants_;
  const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache_;
  const ActiveSpaceTwoElectronResult& active_space_two_electron_result_;
  int n_active_orbitals_ = 0;
  const std::vector<double>& delta_packed_active_two_electron_integrals_;
  const std::vector<SameSpinPolynomialDirectionalPairData>&
      directional_pair_data_;
  std::vector<DirectionalSpinPairEntry> cached_entries_;
  std::vector<std::once_flag> entry_flags_;
};

void gather_directional_spin_block(
    DirectionalSpinPairMemo* pair_memo,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* total_block,
    Eigen::MatrixXd* delta_overlap_block,
    Eigen::MatrixXd* delta_total_block,
    ProjectionBlock* accepted_projection_block,
    ProjectionBlock* directional_projection_block,
    OppositeSpinChannelBuilder* accepted_channel_builder,
    OppositeSpinChannelFamily* accepted_channel_family,
    OppositeSpinChannelBuilder* directional_channel_builder,
    OppositeSpinChannelFamily* directional_channel_family) {
  const bool gather_accepted_channels =
      accepted_channel_builder != nullptr ||
      accepted_channel_family != nullptr;
  const bool gather_directional_channels =
      directional_channel_builder != nullptr ||
      directional_channel_family != nullptr;

  overlap_block->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  total_block->resize(overlap_block->rows(), overlap_block->cols());
  delta_overlap_block->resize(overlap_block->rows(), overlap_block->cols());
  delta_total_block->resize(overlap_block->rows(), overlap_block->cols());
  reset_projection_block(
      overlap_block->rows(),
      overlap_block->cols(),
      accepted_projection_block);
  reset_projection_block(
      overlap_block->rows(),
      overlap_block->cols(),
      directional_projection_block);
  if (gather_accepted_channels) {
    accepted_channel_builder->reset(accepted_channel_family);
  }
  if (gather_directional_channels) {
    directional_channel_builder->reset(directional_channel_family);
  }
  // The fixed-size memo owns stable entries for the entire directional build,
  // so local projection blocks can borrow sparse packed-pair payloads without
  // copying them.
  const bool can_borrow_directional_projections =
      directional_projection_block != nullptr;

  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[column_local];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[row_local];
      const auto& pair_evaluation =
          pair_memo->pair_evaluation(row_global, column_global);
      const auto& directional_entry =
          pair_memo->entry(row_global, column_global);
      (*overlap_block)(row_local, column_local) =
          pair_evaluation.overlap_result.overlap_determinant;
      (*total_block)(row_local, column_local) =
          pair_evaluation.total_hamiltonian;
      (*delta_overlap_block)(row_local, column_local) =
          directional_entry.delta_overlap_determinant;
      (*delta_total_block)(row_local, column_local) =
          directional_entry.delta_total_hamiltonian;
      const auto& accepted_projection =
          pair_evaluation
              .opposite_spin_pair_cache
              .first_order_cofactor_projection;

      if (accepted_projection_block != nullptr) {
        auto& accepted_projection_slot = accepted_projection_block->projections[(column_local) * (overlap_block->rows()) + (row_local)];
        accepted_projection_slot.borrowed_projection = &accepted_projection;
        accepted_projection_slot.owned_projection =
            OppositeSpinPackedPairProjection{};
      }
      if (gather_accepted_channels) {
        accepted_channel_builder->accumulate(
            accepted_projection,
            overlap_block->rows(),
            overlap_block->cols(),
            row_local,
            column_local,
            accepted_channel_family);
      }

      if (directional_projection_block != nullptr) {
        auto& directional_projection_slot = directional_projection_block->projections[(column_local) * (overlap_block->rows()) + (row_local)];
        if (can_borrow_directional_projections) {
          directional_projection_slot.borrowed_projection =
              &directional_entry.delta_first_order_projection;
          directional_projection_slot.owned_projection =
              OppositeSpinPackedPairProjection{};
        } else {
          directional_projection_slot.borrowed_projection = nullptr;
          directional_projection_slot.owned_projection =
              directional_entry.delta_first_order_projection;
        }
      }
      if (gather_directional_channels) {
        directional_channel_builder->accumulate(
            directional_entry.delta_first_order_projection,
            overlap_block->rows(),
            overlap_block->cols(),
            row_local,
            column_local,
            directional_channel_family);
      }
    }
  }
}

}  // namespace

SelectedStateProjectedDirectionalMatrices
build_projected_structure_direction(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairCache& directional_pair_cache) {
  if (accepted.input == nullptr || accepted.accepted_point_context == nullptr ||
      accepted.structure_coefficient_blocks == nullptr) {
    throw std::invalid_argument(
        "projected structure direction requires a complete accepted context");
  }
  const auto& input = *accepted.input;
  const auto& accepted_point_context = *accepted.accepted_point_context;
  const auto& coefficient_blocks = *accepted.structure_coefficient_blocks;
  const auto& delta_ao_overlap_matrix = direction.overlap;
  const auto& delta_active_one_electron_matrix = direction.one_electron;
  const auto& delta_packed_active_two_electron_integrals =
      direction.packed_two_electron;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  const int n_structures = input.structure_data.n_structures;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_selected_states =
      static_cast<int>(accepted_point_context.selected_state_indices.size());
  if (n_determinants <= 0 || n_structures <= 0 || n_active_orbitals <= 0 ||
      n_selected_states <= 0) {
    throw std::invalid_argument(
        "projected directional structure response requires positive dimensions");
  }

  const auto& same_spin_pair_cache = accepted_point_context.same_spin_pair_cache;
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "projected directional structure response requires same-spin cache");
  }
  const std::size_t expected_eigenvector_size =
      n_structures * n_structures;
  if (accepted_point_context.eigen_result.eigenvector_matrix.size() !=
      expected_eigenvector_size) {
    throw std::invalid_argument(
        "accepted-point generalized eigensystem dimensions are inconsistent");
  }

  const Eigen::Map<const Eigen::MatrixXd> eigenvector_matrix(
      accepted_point_context.eigen_result.eigenvector_matrix.data(),
      n_structures,
      n_structures);
  const auto& selected_state_indices =
      accepted_point_context.selected_state_indices;
  const bool single_selected_state = n_selected_states == 1;
  const int selected_state_index =
      single_selected_state ? selected_state_indices.front() : -1;
  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const int state_index =
        selected_state_indices[selected_state_offset];
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
  }

  if (static_cast<int>(coefficient_blocks.size()) != n_structures) {
    throw std::invalid_argument(
        "projected directional structure coefficient blocks do not match n_structures");
  }
  const bool shared_same_spin_pair_cache =
      same_spin_pair_cache.shares_same_spin_pair_cache_between_spins();
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int n_packed_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(
          accepted_prepared_active_space.active_space_two_electron_result);

  SelectedStateProjectedDirectionalMatrices result;
  result.transformed_delta_hamiltonian_selected =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  result.transformed_delta_overlap_selected =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  std::exception_ptr parallel_exception;
  std::atomic<bool> parallel_failed(false);
  const int n_parallel_threads =
      choose_directional_structure_threads(n_structures);
  DirectionalSpinPairMemo shared_projected_alpha_provider(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      accepted_prepared_active_space.active_space_two_electron_result,
      n_active_orbitals,
      delta_packed_active_two_electron_integrals,
      directional_pair_cache.alpha.ordered_pair_data);
  const auto& beta_directional_pair_data =
      directional_pair_cache.close_shell_same_spin
          ? directional_pair_cache.alpha.ordered_pair_data
          : directional_pair_cache.beta.ordered_pair_data;
  DirectionalSpinPairMemo shared_projected_beta_provider(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      same_spin_pair_cache.beta_pair_cache_ref(),
      accepted_prepared_active_space.active_space_two_electron_result,
      n_active_orbitals,
      delta_packed_active_two_electron_integrals,
      beta_directional_pair_data);
  std::vector<Eigen::MatrixXd> thread_transformed_delta_hamiltonian_selected(
      std::max(1, n_parallel_threads),
      Eigen::MatrixXd::Zero(n_structures, n_selected_states));
  std::vector<Eigen::MatrixXd> thread_transformed_delta_overlap_selected(
      std::max(1, n_parallel_threads),
      Eigen::MatrixXd::Zero(n_structures, n_selected_states));

// The projected outer-response builder fuses the old
//   structure-pair scalar accumulation + U^T (delta M) U_sel
// pipeline into one block contraction. The tile sweeps over determinant
// supports stay unchanged, but the H/S outputs now live directly in the
// minimal `(all_states, selected_states)` transformed basis required by the
// selected-state directional response.
#pragma omp parallel if(n_parallel_threads > 1 && n_structures > 2) num_threads(n_parallel_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::MatrixXd alpha_overlap_subblock;
    Eigen::MatrixXd alpha_total_subblock;
    Eigen::MatrixXd alpha_delta_overlap_subblock;
    Eigen::MatrixXd alpha_delta_total_subblock;
    Eigen::MatrixXd beta_overlap_subblock;
    Eigen::MatrixXd beta_total_subblock;
    Eigen::MatrixXd beta_delta_overlap_subblock;
    Eigen::MatrixXd beta_delta_total_subblock;
    Eigen::MatrixXd beta_projected_channel_block;
    Eigen::MatrixXd beta_push;
    Eigen::MatrixXd image;
    SameSpinContractionScratch same_spin_scratch;
    ProjectionBlock accepted_alpha_projection_block;
    ProjectionBlock directional_alpha_projection_block;
    ProjectionBlock accepted_beta_projection_block;
    ProjectionBlock directional_beta_projection_block;
    OppositeSpinChannelBuilder accepted_alpha_channel_builder(
        n_packed_active_pairs);
    OppositeSpinChannelBuilder directional_alpha_channel_builder(
        n_packed_active_pairs);
    OppositeSpinChannelFamily accepted_alpha_channels;
    OppositeSpinChannelFamily directional_alpha_channels;
    Eigen::VectorXd directional_overlap_column =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::VectorXd directional_hamiltonian_column =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::VectorXd transformed_left_hamiltonian =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::VectorXd transformed_left_overlap =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::MatrixXd& local_transformed_delta_hamiltonian_selected =
        thread_transformed_delta_hamiltonian_selected[thread_index];
    Eigen::MatrixXd& local_transformed_delta_overlap_selected =
        thread_transformed_delta_overlap_selected[thread_index];

// The projected outer-response reduction used to add each thread-local matrix
// through one OpenMP critical section. That made the exact_ctx HVP depend on
// thread arrival order and reproduced the user's 5-step / 16-step / 32-step
// trajectory drift on the same 32-thread input. Keep the per-thread work local
// here, but assign `right_structure` deterministically and reduce the thread
// buffers in a fixed order after the parallel region.
#pragma omp for schedule(static, 1)
    for (int right_structure = 0;
         right_structure < n_structures;
         ++right_structure) {
      if (parallel_failed.load(std::memory_order_relaxed)) {
        continue;
      }
      const auto& right_block =
          coefficient_blocks[right_structure];
      try {
        directional_overlap_column.head(right_structure + 1).setZero();
        directional_hamiltonian_column.head(right_structure + 1).setZero();
        for (int left_structure = 0;
             left_structure <= right_structure;
             ++left_structure) {
          if (parallel_failed.load(std::memory_order_relaxed)) {
            continue;
          }
          const auto& left_block =
              coefficient_blocks[left_structure];
          if (left_block.local_coefficients.size() == 0 ||
              right_block.local_coefficients.size() == 0) {
            directional_overlap_column[left_structure] = 0.0;
            directional_hamiltonian_column[left_structure] = 0.0;
            continue;
          }

          const bool structure_pair_close_shell_diagonal =
              close_shell_same_spin &&
              left_block.close_shell_diagonal &&
              right_block.close_shell_diagonal;
          if (structure_pair_close_shell_diagonal) {
            gather_directional_spin_block(
                &shared_projected_alpha_provider,
                left_block.alpha_support,
                right_block.alpha_support,
                &alpha_overlap_subblock,
                &alpha_total_subblock,
                &alpha_delta_overlap_subblock,
                &alpha_delta_total_subblock,
                &accepted_alpha_projection_block,
                &directional_alpha_projection_block,
                &accepted_alpha_channel_builder,
                &accepted_alpha_channels,
                &directional_alpha_channel_builder,
                &directional_alpha_channels);
          } else {
            gather_directional_spin_block(
                &shared_projected_alpha_provider,
                left_block.alpha_support,
                right_block.alpha_support,
                &alpha_overlap_subblock,
                &alpha_total_subblock,
                &alpha_delta_overlap_subblock,
                &alpha_delta_total_subblock,
                &accepted_alpha_projection_block,
                &directional_alpha_projection_block,
                &accepted_alpha_channel_builder,
                &accepted_alpha_channels,
                &directional_alpha_channel_builder,
                &directional_alpha_channels);
            gather_directional_spin_block(
                shared_same_spin_pair_cache
                    ? &shared_projected_alpha_provider
                    : &shared_projected_beta_provider,
                left_block.beta_support,
                right_block.beta_support,
                &beta_overlap_subblock,
                &beta_total_subblock,
                &beta_delta_overlap_subblock,
                &beta_delta_total_subblock,
                &accepted_beta_projection_block,
                &directional_beta_projection_block,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
          }

          const auto same_spin_contraction =
              structure_pair_close_shell_diagonal
                  ? contract_close_shell_same_spin_direction(
                        left_block.local_diagonal_coefficients,
                        right_block.local_diagonal_coefficients,
                        alpha_overlap_subblock,
                        alpha_total_subblock,
                        alpha_delta_overlap_subblock,
                        alpha_delta_total_subblock)
                  : contract_same_spin_direction(
                        left_block.local_coefficients,
                        right_block.local_coefficients,
                        alpha_overlap_subblock,
                        alpha_total_subblock,
                        alpha_delta_overlap_subblock,
                        alpha_delta_total_subblock,
                        beta_overlap_subblock,
                        beta_total_subblock,
                        beta_delta_overlap_subblock,
                        beta_delta_total_subblock,
                        &same_spin_scratch);
          const double directional_overlap =
              same_spin_contraction.overlap;
          double directional_hamiltonian =
              same_spin_contraction.hamiltonian;
          directional_hamiltonian +=
              structure_pair_close_shell_diagonal
                  ? contract_close_shell_opposite_spin_direction(
                        left_block.local_diagonal_coefficients,
                        right_block.local_diagonal_coefficients,
                        accepted_alpha_channels,
                        directional_alpha_channels,
                        accepted_alpha_projection_block,
                        directional_alpha_projection_block,
                        two_electron_view,
                        n_active_orbitals,
                        delta_packed_active_two_electron_integrals,
                        &beta_projected_channel_block)
                  : contract_opposite_spin_direction(
                        left_block.local_coefficients,
                        right_block.local_coefficients,
                        accepted_alpha_projection_block,
                        directional_alpha_projection_block,
                        accepted_alpha_channels,
                        directional_alpha_channels,
                        accepted_beta_projection_block,
                        directional_beta_projection_block,
                        two_electron_view,
                        n_active_orbitals,
                        delta_packed_active_two_electron_integrals,
                        &beta_projected_channel_block,
                        &beta_push,
                        &image);

          directional_overlap_column[left_structure] = directional_overlap;
          directional_hamiltonian_column[left_structure] =
              directional_hamiltonian;
        }

        const auto directional_hamiltonian_column_head =
            directional_hamiltonian_column.head(right_structure + 1);
        const auto directional_overlap_column_head =
            directional_overlap_column.head(right_structure + 1);
        transformed_left_hamiltonian.noalias() =
            eigenvector_matrix.topRows(right_structure + 1).transpose() *
            directional_hamiltonian_column_head;
        transformed_left_overlap.noalias() =
            eigenvector_matrix.topRows(right_structure + 1).transpose() *
            directional_overlap_column_head;
        const auto eigenvector_row = eigenvector_matrix.row(right_structure);
        if (single_selected_state) {
          const double selected_right_value =
              eigenvector_row[selected_state_index];
          local_transformed_delta_hamiltonian_selected.col(0).noalias() +=
              selected_right_value * transformed_left_hamiltonian;
          local_transformed_delta_overlap_selected.col(0).noalias() +=
              selected_right_value * transformed_left_overlap;
        } else {
          for (int selected_state_offset = 0;
               selected_state_offset < n_selected_states;
               ++selected_state_offset) {
            const int state_index =
                selected_state_indices[selected_state_offset];
            const double selected_right_value =
                eigenvector_row[state_index];
            local_transformed_delta_hamiltonian_selected
                .col(selected_state_offset)
                .noalias() +=
                selected_right_value * transformed_left_hamiltonian;
            local_transformed_delta_overlap_selected
                .col(selected_state_offset)
                .noalias() +=
                selected_right_value * transformed_left_overlap;
          }
        }

        if (right_structure > 0) {
          const double directional_hamiltonian_diagonal =
              directional_hamiltonian_column[right_structure];
          const double directional_overlap_diagonal =
              directional_overlap_column[right_structure];
          const auto eigenvector_right_column =
              eigenvector_matrix.row(right_structure).transpose();
          if (single_selected_state) {
            const double transformed_selected_hamiltonian_head =
                transformed_left_hamiltonian[selected_state_index] -
                directional_hamiltonian_diagonal *
                    eigenvector_row[selected_state_index];
            const double transformed_selected_overlap_head =
                transformed_left_overlap[selected_state_index] -
                directional_overlap_diagonal *
                    eigenvector_row[selected_state_index];
            local_transformed_delta_hamiltonian_selected.col(0).noalias() +=
                transformed_selected_hamiltonian_head *
                eigenvector_right_column;
            local_transformed_delta_overlap_selected.col(0).noalias() +=
                transformed_selected_overlap_head *
                eigenvector_right_column;
          } else {
            for (int selected_state_offset = 0;
                 selected_state_offset < n_selected_states;
                 ++selected_state_offset) {
              const int state_index =
                  selected_state_indices[selected_state_offset];
              const double transformed_selected_hamiltonian_head =
                  transformed_left_hamiltonian[state_index] -
                  directional_hamiltonian_diagonal *
                      eigenvector_row[state_index];
              const double transformed_selected_overlap_head =
                  transformed_left_overlap[state_index] -
                  directional_overlap_diagonal *
                      eigenvector_row[state_index];
              local_transformed_delta_hamiltonian_selected
                  .col(selected_state_offset)
                  .noalias() +=
                  transformed_selected_hamiltonian_head *
                  eigenvector_right_column;
              local_transformed_delta_overlap_selected
                  .col(selected_state_offset)
                  .noalias() +=
                  transformed_selected_overlap_head *
                  eigenvector_right_column;
            }
          }
        }
      } catch (...) {
        parallel_failed.store(true, std::memory_order_relaxed);
#pragma omp critical(exact_ctx_projected_directional_structure_exception)
        {
          if (!parallel_exception) {
            parallel_exception = std::current_exception();
          }
        }
      }
    }
  }

  for (int reduction_thread = 0;
       reduction_thread < static_cast<int>(
           thread_transformed_delta_hamiltonian_selected.size());
       ++reduction_thread) {
    result.transformed_delta_hamiltonian_selected +=
        thread_transformed_delta_hamiltonian_selected[reduction_thread];
    result.transformed_delta_overlap_selected +=
        thread_transformed_delta_overlap_selected[reduction_thread];
  }

  if (parallel_exception) {
    try {
      std::rethrow_exception(parallel_exception);
    } catch (const std::exception& error) {
      throw std::runtime_error(
          std::string(
              "exact outer-response projected directional structure build failed: ") +
          error.what());
    } catch (...) {
      throw std::runtime_error(
          "exact outer-response projected directional structure build failed "
          "with an unknown exception");
    }
  }

  throw_if_nonfinite(
      result.transformed_delta_hamiltonian_selected,
      "exact outer-response projected directional Hamiltonian");
  throw_if_nonfinite(
      result.transformed_delta_overlap_selected,
      "exact outer-response projected directional overlap");
  return result;
}

}  // namespace xmvb::vb
