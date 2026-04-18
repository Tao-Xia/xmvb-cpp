#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <stdexcept>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"
#include "vb/matrices/determinant_pair_storage_utils.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

constexpr double kNormalizedWeightTolerance = 1.0e-10;

struct UniquePairContribution {
  int unique_alpha_id = 0;
  int unique_beta_id = 0;
  double right_value = 0.0;
  double left_value = 0.0;
  double residual_value = 0.0;
};

struct AggregatedUniquePairContribution {
  int unique_alpha_id = 0;
  int unique_beta_id = 0;
  double right_value = 0.0;
  double left_value = 0.0;
  double residual_value = 0.0;
};

void validate_selected_state_indices(
    const std::vector<int>& selected_state_indices,
    int n_states);

void validate_nonnegative_weights(
    const std::vector<double>& state_average_weights);

void validate_local_matrix_shape(
    const Eigen::MatrixXd& matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    const char* label);

int parallel_task_thread_count(
    int n_tasks);

void capture_parallel_exception(
    std::atomic<bool>* failed,
    std::exception_ptr* first_exception);

struct WeightedStructureEntry {
  int unique_alpha_id = 0;
  int unique_beta_id = 0;
  double coefficient = 0.0;
};

struct DeterminantUniquePairLookup {
  int n_unique_alpha = 0;
  int n_unique_beta = 0;
  std::vector<int> determinant_index_by_unique_pair;
};

std::vector<int> build_selected_structure_lookup(
    int n_structures,
    const std::vector<int>& selected_structure_indices) {
  std::vector<int> structure_to_selected_column(
      xmvb::to_size(n_structures),
      -1);
  for (int selected_column = 0;
       selected_column < static_cast<int>(selected_structure_indices.size());
       ++selected_column) {
    const int structure_index =
        selected_structure_indices[xmvb::to_size(selected_column)];
    if (structure_index < 0 || structure_index >= n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    if (structure_to_selected_column[xmvb::to_size(structure_index)] >= 0) {
      throw std::invalid_argument("selected_structure_indices must be unique");
    }
    structure_to_selected_column[xmvb::to_size(structure_index)] =
        selected_column;
  }
  return structure_to_selected_column;
}

DeterminantUniquePairLookup build_determinant_unique_pair_lookup(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache) {
  DeterminantUniquePairLookup lookup;
  lookup.n_unique_alpha = static_cast<int>(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants.size());
  lookup.n_unique_beta = static_cast<int>(
      same_spin_pair_cache.beta_reuse_table.unique_determinants.size());
  lookup.determinant_index_by_unique_pair.assign(
      xmvb::product_size(lookup.n_unique_alpha, lookup.n_unique_beta),
      -1);

  const int n_determinants = static_cast<int>(
      same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id.size());
  if (same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id.size() !=
      xmvb::to_size(n_determinants)) {
    throw std::invalid_argument(
        "same-spin determinant-to-unique maps must have the same length");
  }
  for (int determinant_index = 0;
       determinant_index < n_determinants;
       ++determinant_index) {
    const int unique_alpha_id =
        same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id[xmvb::to_size(
            determinant_index)];
    const int unique_beta_id =
        same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id[xmvb::to_size(
            determinant_index)];
    if (unique_alpha_id < 0 || unique_alpha_id >= lookup.n_unique_alpha ||
        unique_beta_id < 0 || unique_beta_id >= lookup.n_unique_beta) {
      throw std::out_of_range("determinant-to-unique spin id is out of range");
    }
    const std::size_t linear_index =
        xmvb::col_major_index(
            unique_alpha_id,
            unique_beta_id,
            lookup.n_unique_alpha);
    int& stored_determinant_index =
        xmvb::index_at(lookup.determinant_index_by_unique_pair, linear_index);
    if (stored_determinant_index >= 0) {
      throw std::logic_error(
          "full determinant topology contains duplicate unique spin-string pairs");
    }
    stored_determinant_index = determinant_index;
  }
  return lookup;
}

void validate_structure_problem_shapes(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    const Eigen::MatrixXd& structure_coefficient_matrix,
    const std::vector<double>& eigenvalues,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  if (full_structure_data.n_structures <= 0) {
    throw std::invalid_argument("full_structure_data.n_structures must be positive");
  }
  if (full_structure_data.n_active_orbitals <= 0) {
    throw std::invalid_argument("full_structure_data.n_active_orbitals must be positive");
  }
  if (full_structure_data.alpha_det.size() != full_structure_data.beta_det.size() ||
      full_structure_data.alpha_det.size() !=
          full_structure_data.determinant_to_structure_terms.size()) {
    throw std::invalid_argument("full determinant structure arrays are inconsistent");
  }
  if (selected_structure_indices.empty()) {
    throw std::invalid_argument("selected_structure_indices must not be empty");
  }
  if (structure_coefficient_matrix.rows() != static_cast<int>(
                                            selected_structure_indices.size()) ||
      structure_coefficient_matrix.cols() != static_cast<int>(
                                            selected_structure_indices.size())) {
    throw std::invalid_argument(
        "structure_coefficient_matrix dimensions are inconsistent");
  }
  if (static_cast<int>(eigenvalues.size()) !=
      static_cast<int>(selected_structure_indices.size())) {
    throw std::invalid_argument("eigenvalues size does not match selected structures");
  }
  validate_selected_state_indices(
      selected_state_indices,
      static_cast<int>(selected_structure_indices.size()));
  validate_nonnegative_weights(state_average_weights);
  if (same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id.size() !=
          full_structure_data.alpha_det.size() ||
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id.size() !=
          full_structure_data.beta_det.size()) {
    throw std::invalid_argument(
        "same-spin cache determinant maps do not match full determinant topology");
  }
}

std::vector<StructureCoefficientBlock> build_selected_structure_blocks(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache) {
  const std::vector<int> structure_to_selected_column =
      build_selected_structure_lookup(
          full_structure_data.n_structures,
          selected_structure_indices);
  return build_structure_coefficient_blocks(
      full_structure_data.determinant_to_structure_terms,
      structure_to_selected_column,
      static_cast<int>(selected_structure_indices.size()),
      same_spin_pair_cache.alpha_reuse_table,
      same_spin_pair_cache.beta_reuse_table,
      true);
}

StructureCoefficientBlock build_selected_state_right_block(
    const std::vector<StructureCoefficientBlock>& selected_structure_blocks,
    const Eigen::Ref<const Eigen::VectorXd>& structure_state_coefficients) {
  if (structure_state_coefficients.size() !=
      static_cast<int>(selected_structure_blocks.size())) {
    throw std::invalid_argument(
        "structure_state_coefficients size does not match selected_structure_blocks");
  }

  std::vector<WeightedStructureEntry> entries;
  for (int selected_structure = 0;
       selected_structure < structure_state_coefficients.size();
       ++selected_structure) {
    const double state_coefficient =
        structure_state_coefficients(selected_structure);
    if (state_coefficient == 0.0) {
      continue;
    }
    const StructureCoefficientBlock& structure_block =
        selected_structure_blocks[xmvb::to_size(selected_structure)];
    for (int beta_local = 0;
         beta_local < static_cast<int>(structure_block.beta_support.size());
         ++beta_local) {
      const int unique_beta_id =
          structure_block.beta_support[xmvb::to_size(beta_local)];
      for (int alpha_local = 0;
           alpha_local < static_cast<int>(structure_block.alpha_support.size());
           ++alpha_local) {
        const double coefficient =
            state_coefficient *
            structure_block.local_coefficients(alpha_local, beta_local);
        if (coefficient == 0.0) {
          continue;
        }
        entries.push_back(WeightedStructureEntry{
            structure_block.alpha_support[xmvb::to_size(alpha_local)],
            unique_beta_id,
            coefficient});
      }
    }
  }

  std::sort(
      entries.begin(),
      entries.end(),
      [](const WeightedStructureEntry& left,
         const WeightedStructureEntry& right) {
        if (left.unique_alpha_id != right.unique_alpha_id) {
          return left.unique_alpha_id < right.unique_alpha_id;
        }
        return left.unique_beta_id < right.unique_beta_id;
      });

  StructureCoefficientBlock right_block;
  std::vector<WeightedStructureEntry> aggregated_entries;
  aggregated_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    if (!aggregated_entries.empty() &&
        aggregated_entries.back().unique_alpha_id == entry.unique_alpha_id &&
        aggregated_entries.back().unique_beta_id == entry.unique_beta_id) {
      aggregated_entries.back().coefficient += entry.coefficient;
      continue;
    }
    aggregated_entries.push_back(entry);
  }
  for (const auto& entry : aggregated_entries) {
    if (entry.coefficient == 0.0) {
      continue;
    }
    right_block.alpha_support.push_back(entry.unique_alpha_id);
    right_block.beta_support.push_back(entry.unique_beta_id);
  }
  xmvb::vb::detail::sort_and_deduplicate_support(&right_block.alpha_support);
  xmvb::vb::detail::sort_and_deduplicate_support(&right_block.beta_support);
  right_block.local_coefficients =
      Eigen::MatrixXd::Zero(
          static_cast<int>(right_block.alpha_support.size()),
          static_cast<int>(right_block.beta_support.size()));
  for (const auto& entry : aggregated_entries) {
    if (entry.coefficient == 0.0) {
      continue;
    }
    const auto alpha_iterator =
        std::lower_bound(
            right_block.alpha_support.begin(),
            right_block.alpha_support.end(),
            entry.unique_alpha_id);
    const auto beta_iterator =
        std::lower_bound(
            right_block.beta_support.begin(),
            right_block.beta_support.end(),
            entry.unique_beta_id);
    if (alpha_iterator == right_block.alpha_support.end() ||
        *alpha_iterator != entry.unique_alpha_id ||
        beta_iterator == right_block.beta_support.end() ||
        *beta_iterator != entry.unique_beta_id) {
      throw std::logic_error("failed to locate state right-block support entry");
    }
    right_block.local_coefficients(
        static_cast<int>(alpha_iterator - right_block.alpha_support.begin()),
        static_cast<int>(beta_iterator - right_block.beta_support.begin())) =
        entry.coefficient;
  }
  validate_local_matrix_shape(
      right_block.local_coefficients,
      right_block.alpha_support,
      right_block.beta_support,
      "state_right_block.local_coefficients");
  return right_block;
}

Eigen::VectorXd build_full_determinant_coefficients_from_block(
    const StructureCoefficientBlock& coefficient_block,
    const DeterminantUniquePairLookup& determinant_lookup,
    int n_determinants) {
  Eigen::VectorXd determinant_coefficients =
      Eigen::VectorXd::Zero(n_determinants);
  for (int beta_local = 0;
       beta_local < static_cast<int>(coefficient_block.beta_support.size());
       ++beta_local) {
    const int unique_beta_id =
        coefficient_block.beta_support[xmvb::to_size(beta_local)];
    for (int alpha_local = 0;
         alpha_local < static_cast<int>(coefficient_block.alpha_support.size());
         ++alpha_local) {
      const double coefficient =
          coefficient_block.local_coefficients(alpha_local, beta_local);
      if (coefficient == 0.0) {
        continue;
      }
      const int unique_alpha_id =
          coefficient_block.alpha_support[xmvb::to_size(alpha_local)];
      const std::size_t linear_index =
          xmvb::col_major_index(
              unique_alpha_id,
              unique_beta_id,
              determinant_lookup.n_unique_alpha);
      const int determinant_index =
          determinant_lookup.determinant_index_by_unique_pair[linear_index];
      if (determinant_index < 0 || determinant_index >= n_determinants) {
        throw std::logic_error(
            "state coefficient block references a missing determinant row");
      }
      determinant_coefficients(determinant_index) = coefficient;
    }
  }
  return determinant_coefficients;
}

void gather_nonorthogonal_same_spin_overlap_row(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int row_index,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_row) {
  if (overlap_row == nullptr) {
    throw std::invalid_argument("overlap_row must not be null");
  }
  overlap_row->resize(1, static_cast<int>(column_indices.size()));
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_index = column_indices[xmvb::to_size(column_local)];
    if (row_index < 0 || row_index >= n_unique_determinants ||
        column_index < 0 || column_index >= n_unique_determinants) {
      throw std::out_of_range("same-spin overlap row index is out of range");
    }
    const auto& pair_evaluation =
        ordered_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
            row_index,
            column_index,
            n_unique_determinants)];
    (*overlap_row)(0, column_local) =
        pair_evaluation.overlap_result.overlap_determinant;
  }
}

void gather_biorthogonal_forward_spin_row(
    const BiorthogonalForwardSpinPairTileProvider& tile_provider,
    int row_index,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_row,
    Eigen::MatrixXd* total_row,
    LocalSpinProjectionBlock* projection_row) {
  if (overlap_row == nullptr || total_row == nullptr || projection_row == nullptr) {
    throw std::invalid_argument("biorthogonal forward spin row outputs must not be null");
  }
  overlap_row->resize(1, static_cast<int>(column_indices.size()));
  total_row->resize(1, static_cast<int>(column_indices.size()));
  projection_row->n_rows = 1;
  projection_row->n_cols = static_cast<int>(column_indices.size());
  projection_row->first_order_projections.assign(
      xmvb::to_size(projection_row->n_cols),
      LocalSpinProjectionBlock::ProjectionSlot{});
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_index = column_indices[xmvb::to_size(column_local)];
    const auto& entry = tile_provider.entry(row_index, column_index);
    (*overlap_row)(0, column_local) = entry.overlap;
    (*total_row)(0, column_local) = entry.total_hamiltonian;
    auto& projection_slot =
        projection_row->first_order_projections[xmvb::to_size(column_local)];
    projection_slot.borrowed_projection = nullptr;
    projection_slot.owned_projection = entry.first_order_projection;
  }
}

void build_action_values_for_state_row(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const StructureCoefficientBlock& right_block,
    const BiorthogonalForwardSpinPairTileProvider& alpha_biorthogonal_provider,
    const BiorthogonalForwardSpinPairTileProvider& beta_biorthogonal_provider,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view,
    int n_active_orbitals,
    int row_unique_alpha_id,
    int row_unique_beta_id,
    double* overlap_value,
    double* hamiltonian_value) {
  if (overlap_value == nullptr || hamiltonian_value == nullptr) {
    throw std::invalid_argument("state-row action outputs must not be null");
  }
  *overlap_value = 0.0;
  *hamiltonian_value = 0.0;
  if (right_block.local_coefficients.size() == 0) {
    return;
  }

  Eigen::MatrixXd alpha_overlap_row;
  Eigen::MatrixXd beta_overlap_row;
  Eigen::MatrixXd alpha_total_row;
  Eigen::MatrixXd beta_total_row;
  LocalSpinProjectionBlock alpha_projection_row;
  LocalSpinProjectionBlock beta_projection_row;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd image;
  Eigen::MatrixXd beta_projected_channel_block;

  gather_nonorthogonal_same_spin_overlap_row(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      same_spin_pair_cache.n_unique_alpha,
      row_unique_alpha_id,
      right_block.alpha_support,
      &alpha_overlap_row);
  gather_nonorthogonal_same_spin_overlap_row(
      same_spin_pair_cache.beta_pair_cache_ref(),
      same_spin_pair_cache.n_unique_beta,
      row_unique_beta_id,
      right_block.beta_support,
      &beta_overlap_row);
  build_structure_action_image(
      right_block.local_coefficients,
      alpha_overlap_row,
      beta_overlap_row,
      &beta_push,
      &image);
  *overlap_value = image(0, 0);

  gather_biorthogonal_forward_spin_row(
      alpha_biorthogonal_provider,
      row_unique_alpha_id,
      right_block.alpha_support,
      &alpha_overlap_row,
      &alpha_total_row,
      &alpha_projection_row);
  gather_biorthogonal_forward_spin_row(
      beta_biorthogonal_provider,
      row_unique_beta_id,
      right_block.beta_support,
      &beta_overlap_row,
      &beta_total_row,
      &beta_projection_row);
  build_structure_action_image(
      right_block.local_coefficients,
      alpha_total_row,
      beta_overlap_row,
      &beta_push,
      &image);
  *hamiltonian_value = image(0, 0);
  build_structure_action_image(
      right_block.local_coefficients,
      alpha_overlap_row,
      beta_total_row,
      &beta_push,
      &image);
  *hamiltonian_value += image(0, 0);
  Eigen::MatrixXd opposite_spin_image = Eigen::MatrixXd::Zero(1, 1);
  add_local_opposite_spin_action_image(
      right_block.local_coefficients,
      alpha_projection_row,
      beta_projection_row,
      right_right_two_electron_view,
      n_active_orbitals,
      &beta_projected_channel_block,
      &beta_push,
      &image,
      &opposite_spin_image);
  *hamiltonian_value += opposite_spin_image(0, 0);
}

void accumulate_state_action_vectors(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const StructureCoefficientBlock& right_block,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::VectorXd* overlap_action,
    Eigen::VectorXd* hamiltonian_action) {
  if (overlap_action == nullptr || hamiltonian_action == nullptr) {
    throw std::invalid_argument("state action outputs must not be null");
  }
  const int n_determinants = static_cast<int>(full_structure_data.alpha_det.size());
  *overlap_action = Eigen::VectorXd::Zero(n_determinants);
  *hamiltonian_action = Eigen::VectorXd::Zero(n_determinants);
  const int n_threads = parallel_task_thread_count(n_determinants);
  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;

#pragma omp parallel num_threads(n_threads) if(n_determinants > 1)
  {
    const BiorthogonalForwardSpinPairTileProvider alpha_biorthogonal_provider(
        same_spin_pair_cache.alpha_reuse_table.unique_determinants,
        true,
        orbital_integrals,
        right_right_two_electron_view);
    const BiorthogonalForwardSpinPairTileProvider beta_biorthogonal_provider(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        false,
        orbital_integrals,
        right_right_two_electron_view);

#pragma omp for schedule(dynamic)
    for (int determinant_index = 0;
         determinant_index < n_determinants;
         ++determinant_index) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        const int row_unique_alpha_id =
            same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id[xmvb::to_size(
                determinant_index)];
        const int row_unique_beta_id =
            same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id[xmvb::to_size(
                determinant_index)];
        double overlap_value = 0.0;
        double hamiltonian_value = 0.0;
        build_action_values_for_state_row(
            same_spin_pair_cache,
            right_block,
            alpha_biorthogonal_provider,
            beta_biorthogonal_provider,
            right_right_two_electron_view,
            full_structure_data.n_active_orbitals,
            row_unique_alpha_id,
            row_unique_beta_id,
            &overlap_value,
            &hamiltonian_value);
        (*overlap_action)(determinant_index) = overlap_value;
        (*hamiltonian_action)(determinant_index) = hamiltonian_value;
      } catch (...) {
        capture_parallel_exception(&failed, &first_exception);
      }
    }
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

BiorthogonalSelectedStateDeterminantCoefficients
build_state_coefficients_from_full_vectors(
    int state_index,
    double eigenvalue,
    double normalized_state_weight,
    const std::vector<int>& determinant_to_unique_alpha_id,
    const std::vector<int>& determinant_to_unique_beta_id,
    int n_unique_alpha,
    int n_unique_beta,
    const Eigen::Ref<const Eigen::VectorXd>& right_determinant_coefficients,
    const Eigen::Ref<const Eigen::VectorXd>& left_determinant_coefficients,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_action) {
  if (right_determinant_coefficients.size() != left_determinant_coefficients.size() ||
      right_determinant_coefficients.size() != hamiltonian_action.size()) {
    throw std::invalid_argument("determinant vectors must have the same length");
  }

  BiorthogonalSelectedStateDeterminantCoefficients state_coefficients;
  state_coefficients.state_index = state_index;
  state_coefficients.eigenvalue = eigenvalue;
  state_coefficients.normalized_state_weight = normalized_state_weight;
  state_coefficients.right_determinant_coefficients.assign(
      xmvb::to_size(right_determinant_coefficients.size()),
      0.0);
  state_coefficients.left_determinant_coefficients.assign(
      xmvb::to_size(left_determinant_coefficients.size()),
      0.0);
  state_coefficients.residual_determinant_coefficients.assign(
      xmvb::to_size(hamiltonian_action.size()),
      0.0);

  std::vector<UniquePairContribution> pair_contributions;
  pair_contributions.reserve(xmvb::to_size(right_determinant_coefficients.size()));
  for (int determinant_index = 0;
       determinant_index < right_determinant_coefficients.size();
       ++determinant_index) {
    const double right_value = right_determinant_coefficients(determinant_index);
    const double left_value = left_determinant_coefficients(determinant_index);
    const double residual_value =
        hamiltonian_action(determinant_index) - eigenvalue * right_value;

    state_coefficients.right_determinant_coefficients[xmvb::to_size(
        determinant_index)] = right_value;
    state_coefficients.left_determinant_coefficients[xmvb::to_size(
        determinant_index)] = left_value;
    state_coefficients.residual_determinant_coefficients[xmvb::to_size(
        determinant_index)] = residual_value;

    const int unique_alpha_id =
        determinant_to_unique_alpha_id[xmvb::to_size(determinant_index)];
    const int unique_beta_id =
        determinant_to_unique_beta_id[xmvb::to_size(determinant_index)];
    if (unique_alpha_id < 0 || unique_alpha_id >= n_unique_alpha ||
        unique_beta_id < 0 || unique_beta_id >= n_unique_beta) {
      throw std::out_of_range("determinant-to-unique spin id is out of range");
    }
    if (right_value == 0.0 && left_value == 0.0 && residual_value == 0.0) {
      continue;
    }
    pair_contributions.push_back(UniquePairContribution{
        unique_alpha_id,
        unique_beta_id,
        right_value,
        left_value,
        residual_value});
  }

  std::sort(
      pair_contributions.begin(),
      pair_contributions.end(),
      [](const UniquePairContribution& left,
         const UniquePairContribution& right) {
        if (left.unique_alpha_id != right.unique_alpha_id) {
          return left.unique_alpha_id < right.unique_alpha_id;
        }
        return left.unique_beta_id < right.unique_beta_id;
      });

  std::vector<AggregatedUniquePairContribution> aggregated_pairs;
  aggregated_pairs.reserve(pair_contributions.size());
  for (const auto& pair_contribution : pair_contributions) {
    if (!aggregated_pairs.empty() &&
        aggregated_pairs.back().unique_alpha_id == pair_contribution.unique_alpha_id &&
        aggregated_pairs.back().unique_beta_id == pair_contribution.unique_beta_id) {
      aggregated_pairs.back().right_value += pair_contribution.right_value;
      aggregated_pairs.back().left_value += pair_contribution.left_value;
      aggregated_pairs.back().residual_value += pair_contribution.residual_value;
      continue;
    }
    aggregated_pairs.push_back(AggregatedUniquePairContribution{
        pair_contribution.unique_alpha_id,
        pair_contribution.unique_beta_id,
        pair_contribution.right_value,
        pair_contribution.left_value,
        pair_contribution.residual_value});
  }

  for (const auto& aggregated_pair : aggregated_pairs) {
    if (aggregated_pair.right_value == 0.0 &&
        aggregated_pair.left_value == 0.0 &&
        aggregated_pair.residual_value == 0.0) {
      continue;
    }
    state_coefficients.alpha_support.push_back(aggregated_pair.unique_alpha_id);
    state_coefficients.beta_support.push_back(aggregated_pair.unique_beta_id);
  }
  xmvb::vb::detail::sort_and_deduplicate_support(&state_coefficients.alpha_support);
  xmvb::vb::detail::sort_and_deduplicate_support(&state_coefficients.beta_support);
  state_coefficients.local_right_coefficient_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<int>(state_coefficients.alpha_support.size()),
          static_cast<int>(state_coefficients.beta_support.size()));
  state_coefficients.local_left_coefficient_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<int>(state_coefficients.alpha_support.size()),
          static_cast<int>(state_coefficients.beta_support.size()));
  state_coefficients.local_residual_coefficient_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<int>(state_coefficients.alpha_support.size()),
          static_cast<int>(state_coefficients.beta_support.size()));

  for (const auto& aggregated_pair : aggregated_pairs) {
    if (aggregated_pair.right_value == 0.0 &&
        aggregated_pair.left_value == 0.0 &&
        aggregated_pair.residual_value == 0.0) {
      continue;
    }
    const auto alpha_iterator =
        std::lower_bound(
            state_coefficients.alpha_support.begin(),
            state_coefficients.alpha_support.end(),
            aggregated_pair.unique_alpha_id);
    const auto beta_iterator =
        std::lower_bound(
            state_coefficients.beta_support.begin(),
            state_coefficients.beta_support.end(),
            aggregated_pair.unique_beta_id);
    if (alpha_iterator == state_coefficients.alpha_support.end() ||
        *alpha_iterator != aggregated_pair.unique_alpha_id ||
        beta_iterator == state_coefficients.beta_support.end() ||
        *beta_iterator != aggregated_pair.unique_beta_id) {
      throw std::logic_error("failed to locate aggregated unique-pair support entry");
    }

    const int alpha_local =
        static_cast<int>(alpha_iterator - state_coefficients.alpha_support.begin());
    const int beta_local =
        static_cast<int>(beta_iterator - state_coefficients.beta_support.begin());
    state_coefficients.local_right_coefficient_matrix(alpha_local, beta_local) =
        aggregated_pair.right_value;
    state_coefficients.local_left_coefficient_matrix(alpha_local, beta_local) =
        aggregated_pair.left_value;
    state_coefficients.local_residual_coefficient_matrix(alpha_local, beta_local) =
        aggregated_pair.residual_value;
    if (aggregated_pair.right_value != 0.0) {
      ++state_coefficients.nonzero_right_coefficient_count;
    }
    if (aggregated_pair.left_value != 0.0) {
      ++state_coefficients.nonzero_left_coefficient_count;
    }
    if (aggregated_pair.residual_value != 0.0) {
      ++state_coefficients.nonzero_residual_coefficient_count;
    }
  }

  validate_local_matrix_shape(
      state_coefficients.local_right_coefficient_matrix,
      state_coefficients.alpha_support,
      state_coefficients.beta_support,
      "local_right_coefficient_matrix");
  validate_local_matrix_shape(
      state_coefficients.local_left_coefficient_matrix,
      state_coefficients.alpha_support,
      state_coefficients.beta_support,
      "local_left_coefficient_matrix");
  validate_local_matrix_shape(
      state_coefficients.local_residual_coefficient_matrix,
      state_coefficients.alpha_support,
      state_coefficients.beta_support,
      "local_residual_coefficient_matrix");
  return state_coefficients;
}

void validate_selected_state_indices(
    const std::vector<int>& selected_state_indices,
    int n_states) {
  if (selected_state_indices.empty()) {
    throw std::invalid_argument("selected_state_indices must not be empty");
  }
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 || state_index >= n_states) {
      throw std::out_of_range("selected state index is out of range");
    }
  }
}

void validate_nonnegative_weights(
    const std::vector<double>& state_average_weights) {
  if (state_average_weights.empty()) {
    throw std::invalid_argument("state_average_weights must not be empty");
  }
  for (const double state_weight : state_average_weights) {
    if (state_weight < 0.0) {
      throw std::invalid_argument("state_average_weights must be non-negative");
    }
  }
}

void validate_normalized_weights(
    const std::vector<double>& normalized_state_weights) {
  validate_nonnegative_weights(normalized_state_weights);
  double weight_sum = 0.0;
  for (const double state_weight : normalized_state_weights) {
    weight_sum += state_weight;
  }
  if (weight_sum <= 0.0) {
    throw std::invalid_argument("normalized_state_weights must sum to a positive value");
  }
  if (std::fabs(weight_sum - 1.0) > kNormalizedWeightTolerance) {
    throw std::invalid_argument(
        "normalized_state_weights must sum to 1 within tolerance");
  }
}

void validate_input_shapes(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  if (evaluation_result.n_selected_structures <= 0) {
    throw std::invalid_argument("evaluation_result.n_selected_structures must be positive");
  }
  if (evaluation_result.n_determinants <= 0) {
    throw std::invalid_argument("evaluation_result.n_determinants must be positive");
  }
  if (evaluation_result.left_determinant_coefficient_matrix.rows() !=
          0 ||
      evaluation_result.left_determinant_coefficient_matrix.cols() !=
          0) {
    if (evaluation_result.left_determinant_coefficient_matrix.rows() !=
            evaluation_result.n_determinants ||
        evaluation_result.left_determinant_coefficient_matrix.cols() !=
            evaluation_result.n_selected_structures) {
      throw std::invalid_argument(
          "left_determinant_coefficient_matrix dimensions are inconsistent");
    }
  }
  if (evaluation_result.right_determinant_coefficient_matrix.rows() !=
          0 ||
      evaluation_result.right_determinant_coefficient_matrix.cols() !=
          0) {
    if (evaluation_result.right_determinant_coefficient_matrix.rows() !=
            evaluation_result.n_determinants ||
        evaluation_result.right_determinant_coefficient_matrix.cols() !=
            evaluation_result.n_selected_structures) {
      throw std::invalid_argument(
          "right_determinant_coefficient_matrix dimensions are inconsistent");
    }
  }
  if (evaluation_result.biorthogonal_hamiltonian_action_on_selected_columns.rows() !=
          evaluation_result.n_determinants ||
      evaluation_result.biorthogonal_hamiltonian_action_on_selected_columns.cols() !=
          evaluation_result.n_selected_structures) {
    throw std::invalid_argument(
        "biorthogonal_hamiltonian_action_on_selected_columns dimensions are inconsistent");
  }
  if (evaluation_result.structure_coefficient_matrix.rows() !=
          evaluation_result.n_selected_structures ||
      evaluation_result.structure_coefficient_matrix.cols() !=
          evaluation_result.n_selected_structures) {
    throw std::invalid_argument(
        "structure_coefficient_matrix dimensions are inconsistent");
  }
  if (static_cast<int>(evaluation_result.eigenvalues.size()) !=
      evaluation_result.n_selected_structures) {
    throw std::invalid_argument("eigenvalues size does not match n_selected_structures");
  }

  if (selected_state_indices.size() != state_average_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and state_average_weights must have the same length");
  }
  validate_selected_state_indices(
      selected_state_indices,
      evaluation_result.n_selected_structures);
  validate_nonnegative_weights(state_average_weights);

  if (same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id.size() !=
          xmvb::to_size(evaluation_result.n_determinants) ||
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id.size() !=
          xmvb::to_size(evaluation_result.n_determinants)) {
    throw std::invalid_argument(
        "same_spin_pair_cache determinant-to-unique maps do not match n_determinants");
  }
  if (same_spin_pair_cache.alpha_reuse_table.unique_determinants.empty() ||
      same_spin_pair_cache.beta_reuse_table.unique_determinants.empty()) {
    throw std::invalid_argument(
        "same_spin_pair_cache unique determinant sets must not be empty");
  }
}

void validate_local_matrix_shape(
    const Eigen::MatrixXd& matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    const char* label) {
  if (matrix.rows() != static_cast<int>(alpha_support.size()) ||
      matrix.cols() != static_cast<int>(beta_support.size())) {
        throw std::invalid_argument(std::string(label) + " shape does not match support dimensions");
  }
}

void sort_and_deduplicate_support(
    std::vector<int>* support) {
  if (support == nullptr) {
    throw std::invalid_argument("support must not be null");
  }
  std::sort(support->begin(), support->end());
  support->erase(
      std::unique(support->begin(), support->end()),
      support->end());
}

int parallel_task_thread_count(
    int n_tasks) {
  if (n_tasks <= 1) {
    return 1;
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads > n_tasks) {
    n_threads = n_tasks;
  }
  if (n_threads < 1) {
    n_threads = 1;
  }
  return n_threads;
}

void capture_parallel_exception(
    std::atomic<bool>* failed,
    std::exception_ptr* first_exception) {
  if (failed == nullptr || first_exception == nullptr) {
    throw std::invalid_argument("parallel failure outputs must not be null");
  }
#pragma omp critical
  {
    if (!failed->load(std::memory_order_relaxed)) {
      *first_exception = std::current_exception();
      failed->store(true, std::memory_order_relaxed);
    }
  }
}

Eigen::VectorXd build_selected_hamiltonian_action_column(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    int state_index) {
  if (state_index < 0 || state_index >= evaluation_result.n_selected_structures) {
    throw std::out_of_range("state_index is out of range for selected-space eigensystem");
  }

  // The overlap residual of one selected state only needs one determinant-side
  // Hamiltonian image `Y_sel * c_n`. Evaluating this per requested state avoids
  // materializing the full dense `Y_sel * C` product when the SCF objective
  // keeps only a small number of states.
  return evaluation_result.biorthogonal_hamiltonian_action_on_selected_columns *
      evaluation_result.structure_coefficient_matrix.col(state_index);
}

Eigen::VectorXd build_selected_right_determinant_coefficients(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    int state_index) {
  if (state_index < 0 || state_index >= evaluation_result.n_selected_structures) {
    throw std::out_of_range("state_index is out of range for selected-space eigensystem");
  }

  // Building `r^(n) = T_sel c_n` only for the requested state avoids the
  // previous full dense `T_sel C` contraction, which was asymptotically much
  // more expensive than the selected-state backward actually needs.
  return evaluation_result.selected_structure_to_determinant *
      evaluation_result.structure_coefficient_matrix.col(state_index);
}

Eigen::VectorXd build_selected_left_determinant_coefficients(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    int state_index) {
  if (state_index < 0 || state_index >= evaluation_result.n_selected_structures) {
    throw std::out_of_range("state_index is out of range for selected-space eigensystem");
  }

  // Likewise `l^(n) = U_sel c_n` should be formed per selected state rather
  // than materializing the full dense `U_sel C` matrix at every forward point.
  return evaluation_result.overlap_action_on_selected_columns *
      evaluation_result.structure_coefficient_matrix.col(state_index);
}

BiorthogonalSelectedStateMatrices build_biorthogonal_selected_state_matrices_impl(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights) {
  const int n_determinants = evaluation_result.n_determinants;
  const int n_selected_structures = evaluation_result.n_selected_structures;
  const int n_unique_alpha = static_cast<int>(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants.size());
  const int n_unique_beta = static_cast<int>(
      same_spin_pair_cache.beta_reuse_table.unique_determinants.size());

  BiorthogonalSelectedStateMatrices result;
  result.n_structures = n_selected_structures;
  result.n_determinants = n_determinants;
  result.n_unique_alpha = n_unique_alpha;
  result.n_unique_beta = n_unique_beta;
  result.selected_state_indices = selected_state_indices;
  result.normalized_state_weights = normalized_state_weights;
  result.determinant_to_unique_alpha_id =
      same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id;
  result.determinant_to_unique_beta_id =
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id;
  result.states.resize(selected_state_indices.size());
  const int n_selected_states = static_cast<int>(selected_state_indices.size());
  const int n_threads = parallel_task_thread_count(n_selected_states);
  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;

  // Each selected state builds an independent `R/L/Q` determinant bundle, so a
  // thread can construct the full local support and coefficient matrices
  // without synchronizing with the other states.
#pragma omp parallel for schedule(dynamic) num_threads(n_threads) if(n_selected_states > 1)
  for (int state_offset = 0; state_offset < n_selected_states; ++state_offset) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    try {
      const int state_index = selected_state_indices[xmvb::to_size(state_offset)];
      const Eigen::VectorXd selected_hamiltonian_action =
          build_selected_hamiltonian_action_column(
              evaluation_result,
              state_index);
      const Eigen::VectorXd right_determinant_coefficients =
          build_selected_right_determinant_coefficients(
              evaluation_result,
              state_index);
      const Eigen::VectorXd left_determinant_coefficients =
          build_selected_left_determinant_coefficients(
              evaluation_result,
              state_index);
      BiorthogonalSelectedStateDeterminantCoefficients state_coefficients;
      state_coefficients.state_index = state_index;
      state_coefficients.eigenvalue =
          evaluation_result.eigenvalues[xmvb::to_size(state_index)];
      state_coefficients.normalized_state_weight =
          normalized_state_weights[xmvb::to_size(state_offset)];
      state_coefficients.right_determinant_coefficients.assign(
          xmvb::to_size(n_determinants),
          0.0);
      state_coefficients.left_determinant_coefficients.assign(
          xmvb::to_size(n_determinants),
          0.0);
      state_coefficients.residual_determinant_coefficients.assign(
          xmvb::to_size(n_determinants),
          0.0);
      std::vector<UniquePairContribution> pair_contributions;
      pair_contributions.reserve(xmvb::to_size(n_determinants));

      for (int determinant_index = 0;
           determinant_index < n_determinants;
           ++determinant_index) {
        const double right_value =
            right_determinant_coefficients(determinant_index);
        const double left_value =
            left_determinant_coefficients(determinant_index);
        const double residual_value =
            selected_hamiltonian_action(determinant_index) -
            state_coefficients.eigenvalue * right_value;

        state_coefficients.right_determinant_coefficients[xmvb::to_size(
            determinant_index)] = right_value;
        state_coefficients.left_determinant_coefficients[xmvb::to_size(
            determinant_index)] = left_value;
        state_coefficients.residual_determinant_coefficients[xmvb::to_size(
            determinant_index)] = residual_value;

        const int unique_alpha_id =
            result.determinant_to_unique_alpha_id[xmvb::to_size(determinant_index)];
        const int unique_beta_id =
            result.determinant_to_unique_beta_id[xmvb::to_size(determinant_index)];
        if (unique_alpha_id < 0 || unique_alpha_id >= n_unique_alpha ||
            unique_beta_id < 0 || unique_beta_id >= n_unique_beta) {
          throw std::out_of_range("determinant-to-unique spin id is out of range");
        }

        if (right_value == 0.0 && left_value == 0.0 && residual_value == 0.0) {
          continue;
        }
        pair_contributions.push_back(UniquePairContribution{
            unique_alpha_id,
            unique_beta_id,
            right_value,
            left_value,
            residual_value});
      }

      std::sort(
          pair_contributions.begin(),
          pair_contributions.end(),
          [](const UniquePairContribution& left,
             const UniquePairContribution& right) {
            if (left.unique_alpha_id != right.unique_alpha_id) {
              return left.unique_alpha_id < right.unique_alpha_id;
            }
            return left.unique_beta_id < right.unique_beta_id;
          });

      std::vector<AggregatedUniquePairContribution> aggregated_pairs;
      aggregated_pairs.reserve(pair_contributions.size());
      for (const auto& pair_contribution : pair_contributions) {
        if (!aggregated_pairs.empty() &&
            aggregated_pairs.back().unique_alpha_id == pair_contribution.unique_alpha_id &&
            aggregated_pairs.back().unique_beta_id == pair_contribution.unique_beta_id) {
          aggregated_pairs.back().right_value += pair_contribution.right_value;
          aggregated_pairs.back().left_value += pair_contribution.left_value;
          aggregated_pairs.back().residual_value += pair_contribution.residual_value;
          continue;
        }
        aggregated_pairs.push_back(AggregatedUniquePairContribution{
            pair_contribution.unique_alpha_id,
            pair_contribution.unique_beta_id,
            pair_contribution.right_value,
            pair_contribution.left_value,
            pair_contribution.residual_value});
      }

      for (const auto& aggregated_pair : aggregated_pairs) {
        if (aggregated_pair.right_value == 0.0 &&
            aggregated_pair.left_value == 0.0 &&
            aggregated_pair.residual_value == 0.0) {
          continue;
        }
        state_coefficients.alpha_support.push_back(aggregated_pair.unique_alpha_id);
        state_coefficients.beta_support.push_back(aggregated_pair.unique_beta_id);
      }
      sort_and_deduplicate_support(&state_coefficients.alpha_support);
      sort_and_deduplicate_support(&state_coefficients.beta_support);

      state_coefficients.local_right_coefficient_matrix =
          Eigen::MatrixXd::Zero(
              static_cast<int>(state_coefficients.alpha_support.size()),
              static_cast<int>(state_coefficients.beta_support.size()));
      state_coefficients.local_left_coefficient_matrix =
          Eigen::MatrixXd::Zero(
              static_cast<int>(state_coefficients.alpha_support.size()),
              static_cast<int>(state_coefficients.beta_support.size()));
      state_coefficients.local_residual_coefficient_matrix =
          Eigen::MatrixXd::Zero(
              static_cast<int>(state_coefficients.alpha_support.size()),
              static_cast<int>(state_coefficients.beta_support.size()));

      for (const auto& aggregated_pair : aggregated_pairs) {
        if (aggregated_pair.right_value == 0.0 &&
            aggregated_pair.left_value == 0.0 &&
            aggregated_pair.residual_value == 0.0) {
          continue;
        }
        const auto alpha_iterator =
            std::lower_bound(
                state_coefficients.alpha_support.begin(),
                state_coefficients.alpha_support.end(),
                aggregated_pair.unique_alpha_id);
        const auto beta_iterator =
            std::lower_bound(
                state_coefficients.beta_support.begin(),
                state_coefficients.beta_support.end(),
                aggregated_pair.unique_beta_id);
        if (alpha_iterator == state_coefficients.alpha_support.end() ||
            *alpha_iterator != aggregated_pair.unique_alpha_id ||
            beta_iterator == state_coefficients.beta_support.end() ||
            *beta_iterator != aggregated_pair.unique_beta_id) {
          throw std::logic_error("failed to locate aggregated unique-pair support entry");
        }

        const int alpha_local =
            static_cast<int>(alpha_iterator - state_coefficients.alpha_support.begin());
        const int beta_local =
            static_cast<int>(beta_iterator - state_coefficients.beta_support.begin());
        state_coefficients.local_right_coefficient_matrix(alpha_local, beta_local) =
            aggregated_pair.right_value;
        state_coefficients.local_left_coefficient_matrix(alpha_local, beta_local) =
            aggregated_pair.left_value;
        state_coefficients.local_residual_coefficient_matrix(alpha_local, beta_local) =
            aggregated_pair.residual_value;
        if (aggregated_pair.right_value != 0.0) {
          ++state_coefficients.nonzero_right_coefficient_count;
        }
        if (aggregated_pair.left_value != 0.0) {
          ++state_coefficients.nonzero_left_coefficient_count;
        }
        if (aggregated_pair.residual_value != 0.0) {
          ++state_coefficients.nonzero_residual_coefficient_count;
        }
      }

      validate_local_matrix_shape(
          state_coefficients.local_right_coefficient_matrix,
          state_coefficients.alpha_support,
          state_coefficients.beta_support,
          "local_right_coefficient_matrix");
      validate_local_matrix_shape(
          state_coefficients.local_left_coefficient_matrix,
          state_coefficients.alpha_support,
          state_coefficients.beta_support,
          "local_left_coefficient_matrix");
      validate_local_matrix_shape(
          state_coefficients.local_residual_coefficient_matrix,
          state_coefficients.alpha_support,
          state_coefficients.beta_support,
          "local_residual_coefficient_matrix");

      result.states[xmvb::to_size(state_offset)] = std::move(state_coefficients);
    } catch (...) {
      capture_parallel_exception(&failed, &first_exception);
    }
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  return result;
}

}  // namespace

std::vector<double> normalize_biorthogonal_state_average_weights(
    const std::vector<double>& state_average_weights) {
  return xmvb::vb::normalize_state_average_weights(state_average_weights);
}

BiorthogonalSelectedStateMatrices
build_biorthogonal_selected_state_matrices_from_structure_problem(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    const Eigen::MatrixXd& structure_coefficient_matrix,
    const std::vector<double>& eigenvalues,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  validate_structure_problem_shapes(
      full_structure_data,
      selected_structure_indices,
      structure_coefficient_matrix,
      eigenvalues,
      same_spin_pair_cache,
      selected_state_indices,
      state_average_weights);
  const std::vector<double> normalized_state_weights =
      normalize_biorthogonal_state_average_weights(state_average_weights);
  const std::vector<StructureCoefficientBlock> selected_structure_blocks =
      build_selected_structure_blocks(
          full_structure_data,
          selected_structure_indices,
          same_spin_pair_cache);
  const DeterminantUniquePairLookup determinant_lookup =
      build_determinant_unique_pair_lookup(
          same_spin_pair_cache);

  BiorthogonalSelectedStateMatrices result;
  result.n_structures = static_cast<int>(selected_structure_indices.size());
  result.n_determinants = static_cast<int>(full_structure_data.alpha_det.size());
  result.n_unique_alpha = static_cast<int>(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants.size());
  result.n_unique_beta = static_cast<int>(
      same_spin_pair_cache.beta_reuse_table.unique_determinants.size());
  result.selected_state_indices = selected_state_indices;
  result.normalized_state_weights = normalized_state_weights;
  result.determinant_to_unique_alpha_id =
      same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id;
  result.determinant_to_unique_beta_id =
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id;
  result.states.resize(selected_state_indices.size());

  const int n_selected_states = static_cast<int>(selected_state_indices.size());
  const int n_threads = parallel_task_thread_count(n_selected_states);
  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;

#pragma omp parallel for schedule(dynamic) num_threads(n_threads) if(n_selected_states > 1)
  for (int state_offset = 0; state_offset < n_selected_states; ++state_offset) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    try {
      const int state_index = selected_state_indices[xmvb::to_size(state_offset)];
      const StructureCoefficientBlock right_block =
          build_selected_state_right_block(
              selected_structure_blocks,
              structure_coefficient_matrix.col(state_index));
      const Eigen::VectorXd right_determinant_coefficients =
          build_full_determinant_coefficients_from_block(
              right_block,
              determinant_lookup,
              result.n_determinants);
      Eigen::VectorXd left_determinant_coefficients;
      Eigen::VectorXd hamiltonian_action;
      accumulate_state_action_vectors(
          full_structure_data,
          same_spin_pair_cache,
          right_block,
          orbital_integrals,
          right_right_two_electron_view,
          &left_determinant_coefficients,
          &hamiltonian_action);
      result.states[xmvb::to_size(state_offset)] =
          build_state_coefficients_from_full_vectors(
              state_index,
              eigenvalues[xmvb::to_size(state_index)],
              normalized_state_weights[xmvb::to_size(state_offset)],
              result.determinant_to_unique_alpha_id,
              result.determinant_to_unique_beta_id,
              result.n_unique_alpha,
              result.n_unique_beta,
              right_determinant_coefficients,
              left_determinant_coefficients,
              hamiltonian_action);
    } catch (...) {
      capture_parallel_exception(&failed, &first_exception);
    }
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  return result;
}

BiorthogonalSelectedStateMatrices build_biorthogonal_selected_state_matrices(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  validate_input_shapes(
      evaluation_result,
      same_spin_pair_cache,
      selected_state_indices,
      state_average_weights);
  const std::vector<double> normalized_state_weights =
      normalize_biorthogonal_state_average_weights(state_average_weights);
  return build_biorthogonal_selected_state_matrices_impl(
      evaluation_result,
      same_spin_pair_cache,
      selected_state_indices,
      normalized_state_weights);
}

BiorthogonalSelectedStateMatrices
build_biorthogonal_selected_state_matrices_from_normalized_weights(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights) {
  validate_input_shapes(
      evaluation_result,
      same_spin_pair_cache,
      selected_state_indices,
      normalized_state_weights);
  validate_normalized_weights(normalized_state_weights);
  return build_biorthogonal_selected_state_matrices_impl(
      evaluation_result,
      same_spin_pair_cache,
      selected_state_indices,
      normalized_state_weights);
}

BiorthogonalDeterminantPairWeightTablesFromCoefficients
build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
    const BiorthogonalSelectedStateMatrices& selected_state_matrices) {
  const int n_determinants = selected_state_matrices.n_determinants;
  if (n_determinants <= 0) {
    throw std::invalid_argument("selected_state_matrices.n_determinants must be positive");
  }
  if (selected_state_matrices.states.size() !=
      selected_state_matrices.selected_state_indices.size()) {
    throw std::invalid_argument(
        "selected_state_matrices states and selected_state_indices size mismatch");
  }
  if (selected_state_matrices.states.size() !=
      selected_state_matrices.normalized_state_weights.size()) {
    throw std::invalid_argument(
        "selected_state_matrices states and normalized_state_weights size mismatch");
  }

  BiorthogonalDeterminantPairWeightTablesFromCoefficients pair_weights;
  pair_weights.n_determinants = n_determinants;
  pair_weights.ordered_hamiltonian_weights.assign(
      xmvb::product_size(n_determinants, n_determinants),
      0.0);
  pair_weights.ordered_overlap_weights.assign(
      xmvb::product_size(n_determinants, n_determinants),
      0.0);
  for (std::size_t state_offset = 0;
       state_offset < selected_state_matrices.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_state_matrices.states[state_offset];
    if (state_coefficients.right_determinant_coefficients.size() !=
            xmvb::to_size(n_determinants) ||
        state_coefficients.left_determinant_coefficients.size() !=
            xmvb::to_size(n_determinants) ||
        state_coefficients.residual_determinant_coefficients.size() !=
            xmvb::to_size(n_determinants)) {
      throw std::invalid_argument(
          "state determinant coefficient sizes do not match n_determinants");
    }
  }

  const int n_threads = parallel_task_thread_count(n_determinants);
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_determinants > 1)
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    const std::size_t row_offset =
        xmvb::to_size(determinant_index_left) * n_determinants;
    for (std::size_t state_offset = 0;
         state_offset < selected_state_matrices.states.size();
         ++state_offset) {
      const auto& state_coefficients = selected_state_matrices.states[state_offset];
      const double state_weight =
          selected_state_matrices.normalized_state_weights[state_offset];
      const double hamiltonian_left =
          state_coefficients.left_determinant_coefficients[xmvb::to_size(
              determinant_index_left)];
      const double overlap_left =
          state_coefficients.residual_determinant_coefficients[xmvb::to_size(
              determinant_index_left)];
      if (hamiltonian_left == 0.0 && overlap_left == 0.0) {
        continue;
      }
      for (int determinant_index_right = 0;
           determinant_index_right < n_determinants;
           ++determinant_index_right) {
        const double right_value =
            state_coefficients.right_determinant_coefficients[xmvb::to_size(
                determinant_index_right)];
        const std::size_t ordered_index =
            row_offset + xmvb::to_size(determinant_index_right);
        pair_weights.ordered_hamiltonian_weights[ordered_index] +=
            state_weight * hamiltonian_left * right_value;
        pair_weights.ordered_overlap_weights[ordered_index] +=
            state_weight * overlap_left * right_value;
      }
    }
  }

  const std::size_t n_unordered_pairs =
      xmvb::vb::unordered_determinant_pair_count(n_determinants);
  pair_weights.unordered_combined_hamiltonian_weights.assign(n_unordered_pairs, 0.0);
  pair_weights.unordered_combined_overlap_weights.assign(n_unordered_pairs, 0.0);
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_determinants > 1)
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    const std::size_t left_row_offset =
        xmvb::to_size(determinant_index_left) * n_determinants;
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          xmvb::vb::canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const std::size_t direct_ordered_index =
          left_row_offset + xmvb::to_size(determinant_index_right);
      const double hamiltonian_direct =
          pair_weights.ordered_hamiltonian_weights[direct_ordered_index];
      const double overlap_direct =
          pair_weights.ordered_overlap_weights[direct_ordered_index];
      if (determinant_index_left == determinant_index_right) {
        pair_weights.unordered_combined_hamiltonian_weights[unordered_index] =
            hamiltonian_direct;
        pair_weights.unordered_combined_overlap_weights[unordered_index] =
            overlap_direct;
      } else {
        const std::size_t swapped_ordered_index =
            xmvb::to_size(determinant_index_right) * n_determinants +
            xmvb::to_size(determinant_index_left);
        pair_weights.unordered_combined_hamiltonian_weights[unordered_index] =
            hamiltonian_direct +
            pair_weights.ordered_hamiltonian_weights[swapped_ordered_index];
        pair_weights.unordered_combined_overlap_weights[unordered_index] =
            overlap_direct +
            pair_weights.ordered_overlap_weights[swapped_ordered_index];
      }
    }
  }

  return pair_weights;
}

}  // namespace xmvb::vb::biorthogonal_vbscf
