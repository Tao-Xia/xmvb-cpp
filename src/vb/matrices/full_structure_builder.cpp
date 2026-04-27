#include "vb/matrices/full_structure_builder.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp_utils.hpp"
#include "vb/matrices/determinant_pair_storage_utils.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/structure_block_kernels.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {


struct ForwardSpinPairEntry {
  double overlap_determinant = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  OppositeSpinPackedPairProjection first_order_projection;
};

struct LocalSpinProjectionBlock {
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
  std::vector<ProjectionSlot> first_order_projections;

  const OppositeSpinPackedPairProjection& first_order_projection(
      int row_local,
      int column_local) const {
    return first_order_projections[(column_local) * (n_rows) + (row_local)]
        .projection();
  }
};

struct LocalOppositeSpinChannelFamily {
  std::vector<int> packed_pair_indices;
  std::vector<Eigen::MatrixXd> alpha_channel_matrices;
};

struct ForwardSpinPairTile {
  int row_tile = 0;
  int column_tile = 0;
  int row_begin = 0;
  int row_end = 0;
  int column_begin = 0;
  int column_end = 0;
  std::uint64_t last_access_stamp = 0;
  std::vector<ForwardSpinPairEntry> entries;

  const ForwardSpinPairEntry& entry(
      int global_row,
      int global_column) const {
    const int local_row = global_row - row_begin;
    const int local_column = global_column - column_begin;
    return entries[(local_column) * (row_end - row_begin) + (local_row)];
  }
};

std::size_t square_storage_size(int dimension) {
  return (dimension) * (dimension);
}

std::size_t structure_matrix_index(
    int row,
    int column,
    int n_structures) {
  return (column) * (n_structures) + (row);
}

int structure_matrix_tile_size() {
  const char* env_value = std::getenv("XMVB_CPP_STRUCTURE_TILE_SIZE");
  if (env_value == nullptr || env_value[0] == '\0') {
    return 256;
  }
  const int tile_size = std::atoi(env_value);
  if (tile_size <= 0) {
    throw std::invalid_argument(
        "XMVB_CPP_STRUCTURE_TILE_SIZE must be a positive integer");
  }
  return tile_size;
}

int structure_matrix_tile_cache_tiles() {
  const char* env_value = std::getenv("XMVB_CPP_STRUCTURE_TILE_CACHE_TILES");
  if (env_value == nullptr || env_value[0] == '\0') {
    return 4;
  }
  const int cache_tiles = std::atoi(env_value);
  if (cache_tiles <= 0) {
    throw std::invalid_argument(
        "XMVB_CPP_STRUCTURE_TILE_CACHE_TILES must be a positive integer");
  }
  return cache_tiles;
}

int forward_structure_matrix_thread_count(int n_structures) {
  // The tiled forward builder allocates one pair of spin-tile providers per
  // worker. Cap the team by the outer structure-row count so small structure
  // spaces do not pay for idle tile caches or empty dynamic-schedule workers.
  if (n_structures <= 2) {
    return 1;
  }
  return std::max(
      1,
      std::min(
          xmvb::effective_openmp_thread_count(),
          n_structures));
}

OppositeSpinPackedPairProjection build_sparse_packed_pair_projection(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& coefficient_matrix,
    bool coefficient_matrix_is_right_by_left,
    int n_orbitals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  OppositeSpinPackedPairProjection projection;
  if (occ_L.empty()) {
    return projection;
  }

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
      const int packed_pair_index =
          TwoElectronIndexer::packed_pair_index(
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
    if (packed_pair_value == 0.0) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(packed_pair_value);
  }
  return projection;
}

template <typename TwoElectronInput>
ForwardSpinPairEntry evaluate_forward_spin_pair_entry(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const TwoElectronInput& two_electron_input) {
  const SpinDeterminantPairEvaluation pair_evaluation =
      pair_evaluator.evaluate_same_spin_pair(
          occ_L,
          occ_R,
          ovlp_act,
          h1e_act,
          n_orbitals,
          two_electron_input);

  ForwardSpinPairEntry entry;
  entry.overlap_determinant =
      pair_evaluation.overlap_result.overlap_determinant;
  entry.one_electron_hamiltonian = pair_evaluation.one_electron_hamiltonian;
  entry.total_hamiltonian = pair_evaluation.total_hamiltonian;
  if (!occ_L.empty() && pair_evaluation.overlap_result.nullity < 2) {
    entry.first_order_projection =
        build_sparse_packed_pair_projection(
            occ_L,
            occ_R,
            calc_cofactor_1st(pair_evaluation.overlap_result),
            true,
            n_orbitals);
  }
  return entry;
}

template <typename TwoElectronInput>
class ForwardSpinPairTileProvider {
public:
  ForwardSpinPairTileProvider(
      const std::vector<std::vector<int>>& unique_spin_determinants,
      const std::vector<SpinDeterminantPairEvaluation>* ordered_spin_pair_cache,
      FullDeterminantPairEvaluator pair_evaluator,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const TwoElectronInput& two_electron_input,
      int tile_size,
      int max_cached_tiles)
      : unique_spin_determinants_(unique_spin_determinants),
        ordered_spin_pair_cache_(ordered_spin_pair_cache),
        pair_evaluator_(std::move(pair_evaluator)),
        ovlp_act_(ovlp_act),
        h1e_act_(h1e_act),
        n_orbitals_(n_orbitals),
        two_electron_input_(two_electron_input),
        tile_size_(tile_size),
        max_cached_tiles_(max_cached_tiles) {
    if (tile_size_ <= 0) {
      throw std::invalid_argument("tile_size must be positive");
    }
    if (max_cached_tiles_ <= 0) {
      throw std::invalid_argument("max_cached_tiles must be positive");
    }
    if (ordered_spin_pair_cache_ != nullptr) {
      const std::size_t expected_cache_entries =
          square_storage_size(static_cast<int>(unique_spin_determinants_.size()));
      if (ordered_spin_pair_cache_->size() != expected_cache_entries) {
        throw std::invalid_argument(
            "ordered same-spin pair cache size does not match unique determinant count");
      }
    }
    cached_tiles_.reserve(max_cached_tiles_);
  }

  const ForwardSpinPairEntry& entry(
      int left_unique_index,
      int right_unique_index) {
    if (left_unique_index < 0 ||
        left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
        right_unique_index < 0 ||
        right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
      throw std::out_of_range("unique spin tile lookup index out of range");
    }

    const int row_tile = left_unique_index / tile_size_;
    const int column_tile = right_unique_index / tile_size_;
    ForwardSpinPairTile* tile = find_or_build_tile(row_tile, column_tile);
    tile->last_access_stamp = ++access_stamp_;
    return tile->entry(left_unique_index, right_unique_index);
  }

  bool has_ordered_spin_pair_cache() const {
    return ordered_spin_pair_cache_ != nullptr;
  }

  const SpinDeterminantPairEvaluation& pair_evaluation(
      int left_unique_index,
      int right_unique_index) const {
    if (ordered_spin_pair_cache_ == nullptr) {
      throw std::logic_error(
          "ordered same-spin pair cache is not available for direct lookup");
    }
    if (left_unique_index < 0 ||
        left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
        right_unique_index < 0 ||
        right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
      throw std::out_of_range("unique spin cache lookup index out of range");
    }
    return (*ordered_spin_pair_cache_)[ordered_spin_pair_storage_index(
            left_unique_index,
            right_unique_index,
            static_cast<int>(unique_spin_determinants_.size()))];
  }

private:
  ForwardSpinPairTile* find_or_build_tile(
      int row_tile,
      int column_tile) {
    for (auto& tile : cached_tiles_) {
      if (tile.row_tile == row_tile && tile.column_tile == column_tile) {
        return &tile;
      }
    }

    const int row_begin = row_tile * tile_size_;
    const int column_begin = column_tile * tile_size_;
    const int row_end = std::min(
        row_begin + tile_size_,
        static_cast<int>(unique_spin_determinants_.size()));
    const int column_end = std::min(
        column_begin + tile_size_,
        static_cast<int>(unique_spin_determinants_.size()));

    ForwardSpinPairTile built_tile;
    built_tile.row_tile = row_tile;
    built_tile.column_tile = column_tile;
    built_tile.row_begin = row_begin;
    built_tile.row_end = row_end;
    built_tile.column_begin = column_begin;
    built_tile.column_end = column_end;
    built_tile.last_access_stamp = ++access_stamp_;
    built_tile.entries.resize(
        (row_end - row_begin) * (column_end - column_begin));

    // Each tile stores only the forward payload needed by the matrix-form
    // structure assembly: same-spin scalar channels and the sparse first-order
    // opposite-spin regrouping. The tile can then be reused across many
    // structure-pair contractions without materializing the global `N^2`
    // dense channel families.
    for (int column_index = column_begin;
         column_index < column_end;
         ++column_index) {
      for (int row_index = row_begin;
           row_index < row_end;
           ++row_index) {
        built_tile.entries[(column_index - column_begin) * (row_end - row_begin) + (row_index - row_begin)] =
            evaluate_forward_spin_pair_entry(
                unique_spin_determinants_[row_index],
                unique_spin_determinants_[column_index],
                pair_evaluator_,
                ovlp_act_,
                h1e_act_,
                n_orbitals_,
                two_electron_input_);
      }
    }

    if (static_cast<int>(cached_tiles_.size()) == max_cached_tiles_) {
      auto victim_iterator = cached_tiles_.begin();
      for (auto iterator = cached_tiles_.begin();
           iterator != cached_tiles_.end();
           ++iterator) {
        if (iterator->last_access_stamp < victim_iterator->last_access_stamp) {
          victim_iterator = iterator;
        }
      }
      *victim_iterator = std::move(built_tile);
      return &(*victim_iterator);
    }

    cached_tiles_.push_back(std::move(built_tile));
    return &cached_tiles_.back();
  }

  const std::vector<std::vector<int>>& unique_spin_determinants_;
  const std::vector<SpinDeterminantPairEvaluation>* ordered_spin_pair_cache_ =
      nullptr;
  FullDeterminantPairEvaluator pair_evaluator_;
  const std::vector<double>& ovlp_act_;
  const Eigen::Ref<const Eigen::MatrixXd> h1e_act_;
  int n_orbitals_ = 0;
  const TwoElectronInput& two_electron_input_;
  int tile_size_ = 0;
  int max_cached_tiles_ = 0;
  std::uint64_t access_stamp_ = 0;
  std::vector<ForwardSpinPairTile> cached_tiles_;
};

void validate_full_determinant_input(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    int n_orbitals,
    int n_structures) {
  const int n_determinants = static_cast<int>(alpha_det.size());
  if (n_determinants <= 0) {
    throw std::invalid_argument("at least one determinant is required");
  }
  if (static_cast<int>(beta_det.size()) != n_determinants ||
      static_cast<int>(determinant_to_structure_terms.size()) != n_determinants) {
    throw std::invalid_argument("all determinant-indexed inputs must have the same size");
  }
  if (n_orbitals <= 0 || n_structures <= 0) {
    throw std::invalid_argument("n_orbitals and n_structures must be positive");
  }
}

void symmetrize_structure_matrices(
    StructureAccumulationResult& accumulation_result) {
  const int n_structures = accumulation_result.n_structures;
  for (int row = 0; row < n_structures; ++row) {
    for (int column = 0; column < row; ++column) {
      const std::size_t lower_index =
          structure_matrix_index(row, column, n_structures);
      const std::size_t upper_index =
          structure_matrix_index(column, row, n_structures);
      accumulation_result.overlap_matrix[lower_index] =
          accumulation_result.overlap_matrix[upper_index];
      accumulation_result.hamiltonian_matrix[lower_index] =
          accumulation_result.hamiltonian_matrix[upper_index];
    }
  }
}

template <typename TwoElectronInput>
void collect_pair_evaluations(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const TwoElectronInput& two_electron_input,
    std::vector<FullDeterminantPairEvaluation>* pair_evaluations) {
  if (pair_evaluations == nullptr) {
    throw std::invalid_argument("pair_evaluations must not be null");
  }

  const int n_determinants = static_cast<int>(alpha_det.size());
  const std::size_t n_determinant_pairs =
      unordered_determinant_pair_count(n_determinants);
  if (pair_evaluations->size() != n_determinant_pairs) {
    throw std::invalid_argument(
        "pair_evaluations size does not match determinant pair count");
  }

  int n_threads = 1;
  n_threads = xmvb::effective_openmp_thread_count();
  if (n_threads > n_determinants) {
    n_threads = n_determinants;
  }
  if (n_threads < 1) {
    n_threads = 1;
  }

  if (n_threads == 1) {
    const FullDeterminantPairEvaluator thread_pair_evaluator = pair_evaluator;
    auto determinant_pair = determinant_pair_from_storage_index(0);
    for (std::size_t pair_storage_index = 0;
         pair_storage_index < n_determinant_pairs;
         ++pair_storage_index) {
      (*pair_evaluations)[pair_storage_index] =
          evaluate_full_determinant_pair_with_optional_same_spin_cache(
              same_spin_pair_cache,
              thread_pair_evaluator,
              alpha_det,
              beta_det,
              determinant_pair.left,
              determinant_pair.right,
              ovlp_act,
              h1e_act,
              n_orbitals,
              two_electron_input,
              true);
      advance_unordered_determinant_pair(&determinant_pair);
    }
    return;
  }

  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;
  const std::size_t pair_chunk_size =
      unordered_determinant_pair_parallel_chunk_size(
          n_determinant_pairs,
          n_threads);
  const std::size_t pair_chunk_stride =
      pair_chunk_size * static_cast<std::size_t>(n_threads);

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    const FullDeterminantPairEvaluator thread_pair_evaluator = pair_evaluator;
    // Pair evaluation is mathematically independent for each triangular
    // storage slot.  Chunk-striping keeps the pair-to-thread assignment
    // deterministic while spreading expensive determinant rows across workers.
    for (std::size_t pair_begin =
             static_cast<std::size_t>(thread_index) * pair_chunk_size;
         pair_begin < n_determinant_pairs;
         pair_begin += pair_chunk_stride) {
      const std::size_t pair_end =
          std::min(n_determinant_pairs, pair_begin + pair_chunk_size);
      auto determinant_pair = determinant_pair_from_storage_index(pair_begin);
      for (std::size_t pair_storage_index = pair_begin;
           pair_storage_index < pair_end;
           ++pair_storage_index) {
        if (failed.load(std::memory_order_relaxed)) {
          break;
        }

        try {
          (*pair_evaluations)[pair_storage_index] =
              evaluate_full_determinant_pair_with_optional_same_spin_cache(
                  same_spin_pair_cache,
                  thread_pair_evaluator,
                  alpha_det,
                  beta_det,
                  determinant_pair.left,
                  determinant_pair.right,
                  ovlp_act,
                  h1e_act,
                  n_orbitals,
                  two_electron_input,
                  true);
        } catch (...) {
#pragma omp critical
          {
            if (!failed.load(std::memory_order_relaxed)) {
              first_exception = std::current_exception();
              failed.store(true, std::memory_order_relaxed);
            }
          }
        }
        advance_unordered_determinant_pair(&determinant_pair);
      }
    }
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

template <typename TileProvider>
void gather_forward_spin_block(
    TileProvider* tile_provider,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* total_block,
    LocalSpinProjectionBlock* local_projection_block) {
  if (tile_provider == nullptr ||
      overlap_block == nullptr ||
      total_block == nullptr ||
      local_projection_block == nullptr) {
    throw std::invalid_argument(
        "forward spin block gather received a null output/input pointer");
  }

  overlap_block->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  total_block->resize(overlap_block->rows(), overlap_block->cols());
  local_projection_block->n_rows = overlap_block->rows();
  local_projection_block->n_cols = overlap_block->cols();
  const std::size_t block_row_count = overlap_block->rows();
  const std::size_t block_col_count = overlap_block->cols();
  local_projection_block->first_order_projections.assign(
      block_row_count * block_col_count,
      LocalSpinProjectionBlock::ProjectionSlot{});

  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[column_local];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[row_local];
      auto& projection_slot = local_projection_block->first_order_projections[(column_local) * (overlap_block->rows()) + (row_local)];
      if (tile_provider->has_ordered_spin_pair_cache()) {
        const auto& pair_evaluation =
            tile_provider->pair_evaluation(row_global, column_global);
        (*overlap_block)(row_local, column_local) =
            pair_evaluation.overlap_result.overlap_determinant;
        (*total_block)(row_local, column_local) =
            pair_evaluation.total_hamiltonian;
        projection_slot.borrowed_projection =
            &pair_evaluation
                 .opposite_spin_pair_cache
                 .first_order_cofactor_projection;
        projection_slot.owned_projection = OppositeSpinPackedPairProjection{};
      } else {
        const ForwardSpinPairEntry& entry =
            tile_provider->entry(row_global, column_global);
        (*overlap_block)(row_local, column_local) =
            entry.overlap_determinant;
        (*total_block)(row_local, column_local) = entry.total_hamiltonian;
        projection_slot.borrowed_projection = nullptr;
        projection_slot.owned_projection = entry.first_order_projection;
      }
    }
  }
}

int count_local_projection_block_distinct_pairs(
    const LocalSpinProjectionBlock& local_projection_block,
    int n_packed_active_pairs) {
  if (local_projection_block.n_rows <= 0 ||
      local_projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return 0;
  }

  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0u);
  int distinct_pair_count = 0;
  for (int column_local = 0;
       column_local < local_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < local_projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          local_projection_block.first_order_projection(
              row_local,
              column_local);
      for (const int packed_pair_index : projection.packed_pair_indices) {
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local opposite-spin channel range");
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

LocalOppositeSpinChannelFamily build_local_channel_family(
    const LocalSpinProjectionBlock& local_projection_block,
    int n_packed_active_pairs) {
  LocalOppositeSpinChannelFamily channel_family;
  if (local_projection_block.n_rows <= 0 ||
      local_projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return channel_family;
  }

  std::vector<int> local_channel_index_by_packed_pair(
      n_packed_active_pairs,
      -1);

  for (int column_local = 0;
       column_local < local_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < local_projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          local_projection_block.first_order_projection(
              row_local,
              column_local);
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local opposite-spin channel range");
        }
        const double packed_pair_value = projection.packed_pair_values[entry_index];
        int& channel_index =
            local_channel_index_by_packed_pair[packed_pair_index];
        if (channel_index < 0) {
          channel_index = static_cast<int>(channel_family.packed_pair_indices.size());
          channel_family.packed_pair_indices.push_back(packed_pair_index);
          channel_family.alpha_channel_matrices.emplace_back(
              Eigen::MatrixXd::Zero(
                  local_projection_block.n_rows,
                  local_projection_block.n_cols));
        }
        channel_family.alpha_channel_matrices[channel_index](row_local, column_local) =
            packed_pair_value;
      }
    }
  }

  return channel_family;
}

void build_local_projected_channel(
    const LocalSpinProjectionBlock& local_projection_block,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* projected_channel_block) {
  if (projected_channel_block == nullptr) {
    throw std::invalid_argument("projected_channel_block must not be null");
  }

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
          local_projection_block.first_order_projection(
              row_local,
              column_local);
      (*projected_channel_block)(row_local, column_local) =
          xmvb::vb::project_sparse_projection_onto_packed_pair(
              projection,
              target_packed_pair_index,
              two_electron_view,
              n_orbitals);
    }
  }
}

double contract_local_opposite_spin_block(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const LocalSpinProjectionBlock& alpha_projection_block,
    const LocalSpinProjectionBlock& beta_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* beta_projected_channel_block,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  if (beta_projected_channel_block == nullptr ||
      beta_push == nullptr ||
      image == nullptr) {
    throw std::invalid_argument(
        "opposite-spin local contraction outputs must not be null");
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  const int alpha_channel_count =
      count_local_projection_block_distinct_pairs(
          alpha_projection_block,
          n_packed_active_pairs);
  const int beta_channel_count =
      count_local_projection_block_distinct_pairs(
          beta_projection_block,
          n_packed_active_pairs);
  double contraction = 0.0;
  if (alpha_channel_count <= beta_channel_count) {
    const LocalOppositeSpinChannelFamily alpha_channels =
        build_local_channel_family(
            alpha_projection_block,
            n_packed_active_pairs);
    for (std::size_t channel_index = 0;
         channel_index < alpha_channels.packed_pair_indices.size();
         ++channel_index) {
      build_local_projected_channel(
          beta_projection_block,
          alpha_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          beta_projected_channel_block);
      contraction +=
          contract_dense_structure_pair_kernel(
              left_coefficients,
              right_coefficients,
              alpha_channels.alpha_channel_matrices[channel_index],
              *beta_projected_channel_block,
              beta_push,
              image);
    }
  } else {
    const LocalOppositeSpinChannelFamily beta_channels =
        build_local_channel_family(
            beta_projection_block,
            n_packed_active_pairs);
    for (std::size_t channel_index = 0;
         channel_index < beta_channels.packed_pair_indices.size();
         ++channel_index) {
      build_local_projected_channel(
          alpha_projection_block,
          beta_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          beta_projected_channel_block);
      contraction +=
          contract_dense_structure_pair_kernel(
              left_coefficients,
              right_coefficients,
              *beta_projected_channel_block,
              beta_channels.alpha_channel_matrices[channel_index],
              beta_push,
              image);
    }
  }
  return contraction;
}

template <typename TwoElectronInput>
StructureAccumulationResult build_tiled_matrix_form_structure_matrices(
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    int n_structures,
    const SpinDeterminantReuseTable& alpha_reuse_table,
    const SpinDeterminantReuseTable& beta_reuse_table,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const TwoElectronInput& two_electron_input,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<SpinDeterminantPairEvaluation>* alpha_pair_cache = nullptr,
    const std::vector<SpinDeterminantPairEvaluation>* beta_pair_cache = nullptr)
{
  const int n_determinants = static_cast<int>(determinant_to_structure_terms.size());
  const bool close_shell_same_spin =
      alpha_reuse_table.unique_determinants ==
          beta_reuse_table.unique_determinants &&
      alpha_reuse_table.determinant_to_unique_id ==
          beta_reuse_table.determinant_to_unique_id;
  const bool shared_same_spin_pair_kernels =
      alpha_reuse_table.unique_determinants ==
      beta_reuse_table.unique_determinants;
  const int tile_size = structure_matrix_tile_size();
  const int max_cached_tiles = structure_matrix_tile_cache_tiles();
  const auto coefficient_blocks =
      build_structure_coefficient_blocks(
          determinant_to_structure_terms,
          n_structures,
          alpha_reuse_table,
          beta_reuse_table,
          true);
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(two_electron_input);
  const int n_threads =
      forward_structure_matrix_thread_count(n_structures);

  StructureAccumulationResult result;
  result.n_structures = n_structures;
  const std::size_t matrix_size = square_storage_size(n_structures);
  result.overlap_matrix.assign(matrix_size, 0.0);
  result.hamiltonian_matrix.assign(matrix_size, 0.0);
  result.determinant_overlap_cache.assign(
      n_determinants,
      0.0);

  ForwardSpinPairTileProvider<TwoElectronInput> alpha_provider(
      alpha_reuse_table.unique_determinants,
      alpha_pair_cache,
      pair_evaluator,
      ovlp_act,
      h1e_act,
      n_orbitals,
      two_electron_input,
      tile_size,
      max_cached_tiles);
  ForwardSpinPairTileProvider<TwoElectronInput> beta_provider(
      beta_reuse_table.unique_determinants,
      beta_pair_cache,
      pair_evaluator,
      ovlp_act,
      h1e_act,
      n_orbitals,
      two_electron_input,
      tile_size,
      max_cached_tiles);
  for (int determinant_index = 0;
       determinant_index < n_determinants;
       ++determinant_index) {
    const int unique_alpha_id =
        alpha_reuse_table.determinant_to_unique_id[determinant_index];
    const int unique_beta_id =
        beta_reuse_table.determinant_to_unique_id[determinant_index];
    const double alpha_overlap =
        alpha_provider.has_ordered_spin_pair_cache()
            ? alpha_provider
                  .pair_evaluation(unique_alpha_id, unique_alpha_id)
                  .overlap_result
                  .overlap_determinant
            : alpha_provider.entry(unique_alpha_id, unique_alpha_id)
                  .overlap_determinant;
    const double beta_overlap =
        close_shell_same_spin
            ? alpha_overlap
            : (beta_provider.has_ordered_spin_pair_cache()
                   ? beta_provider
                         .pair_evaluation(unique_beta_id, unique_beta_id)
                         .overlap_result
                         .overlap_determinant
                   : beta_provider.entry(unique_beta_id, unique_beta_id)
                         .overlap_determinant);
    result.determinant_overlap_cache[determinant_index] =
        alpha_overlap * beta_overlap;
  }

  // The tiled forward path computes only the unique-spin blocks touched by the
  // current structure pair `(I,J)`. If a whole support block fits inside one
  // tile, the local contraction is algebraically identical to the dense
  // matrix-form kernel, but without building the global `N_alpha^2` /
  // `N_beta^2` channel families upfront.
#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
    const FullDeterminantPairEvaluator thread_pair_evaluator = pair_evaluator;
    ForwardSpinPairTileProvider<TwoElectronInput> thread_alpha_provider(
        alpha_reuse_table.unique_determinants,
        alpha_pair_cache,
        thread_pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        two_electron_input,
        tile_size,
        max_cached_tiles);
    ForwardSpinPairTileProvider<TwoElectronInput> thread_beta_provider(
        beta_reuse_table.unique_determinants,
        beta_pair_cache,
        thread_pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        two_electron_input,
        tile_size,
        max_cached_tiles);
    Eigen::MatrixXd alpha_overlap_subblock;
    Eigen::MatrixXd alpha_total_subblock;
    Eigen::MatrixXd beta_overlap_subblock;
    Eigen::MatrixXd beta_total_subblock;
    Eigen::MatrixXd beta_projected_channel_block;
    Eigen::MatrixXd beta_push;
    Eigen::MatrixXd image;
    LocalSpinProjectionBlock alpha_projection_block;
    LocalSpinProjectionBlock beta_projection_block;

#pragma omp for schedule(dynamic, 1)
    for (int right_structure = 0;
         right_structure < n_structures;
         ++right_structure) {
      const auto& right_block =
          coefficient_blocks[right_structure];
      for (int left_structure = 0;
           left_structure <= right_structure;
           ++left_structure) {
        const auto& left_block =
            coefficient_blocks[left_structure];
        const std::size_t linear_index =
            structure_matrix_index(
                left_structure,
                right_structure,
                n_structures);

        if (left_block.local_coefficients.size() == 0 ||
            right_block.local_coefficients.size() == 0) {
          result.overlap_matrix[linear_index] = 0.0;
          result.hamiltonian_matrix[linear_index] = 0.0;
          continue;
        }

        gather_forward_spin_block(
            &thread_alpha_provider,
            left_block.alpha_support,
            right_block.alpha_support,
            &alpha_overlap_subblock,
            &alpha_total_subblock,
            &alpha_projection_block);

        const bool structure_pair_close_shell_diagonal =
            close_shell_same_spin &&
            left_block.close_shell_diagonal &&
            right_block.close_shell_diagonal;
        if (structure_pair_close_shell_diagonal) {
          beta_overlap_subblock = alpha_overlap_subblock;
          beta_total_subblock = alpha_total_subblock;
          beta_projection_block = alpha_projection_block;
        } else {
          gather_forward_spin_block(
              shared_same_spin_pair_kernels
                  ? &thread_alpha_provider
                  : &thread_beta_provider,
              left_block.beta_support,
              right_block.beta_support,
              &beta_overlap_subblock,
              &beta_total_subblock,
              &beta_projection_block);
        }

        double overlap_value = 0.0;
        double total_hamiltonian_value = 0.0;
        if (structure_pair_close_shell_diagonal) {
          overlap_value =
              contract_diagonal_structure_pair_kernel(
                  left_block.local_diagonal_coefficients,
                  right_block.local_diagonal_coefficients,
                  alpha_overlap_subblock,
                  alpha_overlap_subblock);
          total_hamiltonian_value =
              2.0 *
              contract_diagonal_structure_pair_kernel(
                  left_block.local_diagonal_coefficients,
                  right_block.local_diagonal_coefficients,
                  alpha_total_subblock,
                  alpha_overlap_subblock);
        } else {
          overlap_value =
              contract_dense_structure_pair_kernel(
                  left_block.local_coefficients,
                  right_block.local_coefficients,
                  alpha_overlap_subblock,
                  beta_overlap_subblock,
                  &beta_push,
                  &image);
          total_hamiltonian_value =
              contract_dense_structure_pair_kernel(
                  left_block.local_coefficients,
                  right_block.local_coefficients,
                  alpha_total_subblock,
                  beta_overlap_subblock,
                  &beta_push,
                  &image) +
              contract_dense_structure_pair_kernel(
                  left_block.local_coefficients,
                  right_block.local_coefficients,
                  alpha_overlap_subblock,
                  beta_total_subblock,
                  &beta_push,
                  &image);
        }
        total_hamiltonian_value +=
            structure_pair_close_shell_diagonal
                ? [&]() {
                    const int n_packed_active_pairs =
                        packed_active_pair_count(n_orbitals);
                    const LocalOppositeSpinChannelFamily alpha_channels =
                        build_local_channel_family(
                            alpha_projection_block,
                            n_packed_active_pairs);
                    double contraction = 0.0;
                    for (std::size_t channel_index = 0;
                         channel_index < alpha_channels.packed_pair_indices.size();
                         ++channel_index) {
                      build_local_projected_channel(
                          alpha_projection_block,
                          alpha_channels.packed_pair_indices[channel_index],
                          two_electron_view,
                          n_orbitals,
                          &beta_projected_channel_block);
                      contraction +=
                          contract_diagonal_structure_pair_kernel(
                              left_block.local_diagonal_coefficients,
                              right_block.local_diagonal_coefficients,
                              alpha_channels.alpha_channel_matrices[channel_index],
                              beta_projected_channel_block);
                    }
                    return contraction;
                  }()
                : contract_local_opposite_spin_block(
                      left_block.local_coefficients,
                      right_block.local_coefficients,
                      alpha_projection_block,
                      beta_projection_block,
                      two_electron_view,
                      n_orbitals,
                      &beta_projected_channel_block,
                      &beta_push,
                      &image);

        result.overlap_matrix[linear_index] = overlap_value;
        result.hamiltonian_matrix[linear_index] =
            total_hamiltonian_value;
      }
    }
  }

  symmetrize_structure_matrices(result);
  return result;
}

}  // namespace

const FullDeterminantPairEvaluation& FullDeterminantStructureBuildResult::pair_evaluation(
    int determinant_index_left,
    int determinant_index_right) const {
  if (pair_evaluations.empty()) {
    throw std::logic_error(
        "determinant pair evaluations were not retained by this build result");
  }
  if (determinant_index_left < 0 || determinant_index_right < 0 ||
      determinant_index_left >= n_determinants || determinant_index_right >= n_determinants) {
    throw std::out_of_range("determinant pair index out of range");
  }
  return pair_evaluations[canonical_determinant_pair_storage_index(
      determinant_index_left,
      determinant_index_right)];
}

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    VBSCFAlgorithm algorithm)
    : determinant_overlap_resolver_(),
      determinant_hamiltonian_resolver_(algorithm) {}

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    DeterminantOverlapResolver determinant_overlap_resolver,
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
    VBSCFAlgorithm algorithm)
    : determinant_overlap_resolver_(std::move(determinant_overlap_resolver)),
      determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)) {
  (void)algorithm;
}


StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures) const {
  return build_impl(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      n_structures,
      nullptr,
      false)
      .structure_matrices;
}

StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_structures) const {
  return build_impl(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      ovlp_act,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result,
      n_structures,
      nullptr,
      false)
      .structure_matrices;
}

StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures,
    const SameSpinPairCacheContext& same_spin_pair_cache) const {
  return build_impl(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      n_structures,
      &same_spin_pair_cache,
      false)
      .structure_matrices;
}

StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_structures,
    const SameSpinPairCacheContext& same_spin_pair_cache) const {
  return build_impl(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      ovlp_act,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result,
      n_structures,
      &same_spin_pair_cache,
      false)
      .structure_matrices;
}

FullDeterminantPairEvaluator
FullDeterminantStructureHamiltonianOverlapBuilder::make_pair_evaluator() const {
  return FullDeterminantPairEvaluator(
      determinant_overlap_resolver_,
      determinant_hamiltonian_resolver_);
}

FullDeterminantStructureBuildResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_impl(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures,
    const SameSpinPairCacheContext* same_spin_pair_cache,
    bool store_pair_evaluations) const {
  validate_full_determinant_input(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      n_orbitals,
      n_structures);

  const int n_determinants = static_cast<int>(alpha_det.size());
  FullDeterminantStructureBuildResult build_result;
  build_result.n_determinants = n_determinants;
  const std::size_t n_determinant_pairs =
      unordered_determinant_pair_count(n_determinants);
  if (store_pair_evaluations) {
    build_result.pair_evaluations.resize(n_determinant_pairs);
  }
  if (same_spin_pair_cache != nullptr) {
    if (same_spin_pair_cache->alpha_reuse_table.determinant_to_unique_id.size() !=
            alpha_det.size() ||
        same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id.size() !=
            beta_det.size()) {
      throw std::invalid_argument(
          "same-spin cache context does not match the determinant expansion");
    }
  }

  const SpinDeterminantReuseTable alpha_reuse_table =
      (same_spin_pair_cache != nullptr)
          ? same_spin_pair_cache->alpha_reuse_table
          : build_spin_determinant_reuse_table(alpha_det);
  const SpinDeterminantReuseTable beta_reuse_table =
      (same_spin_pair_cache != nullptr)
          ? same_spin_pair_cache->beta_reuse_table
          : build_spin_determinant_reuse_table(beta_det);
  const FullDeterminantPairEvaluator pair_evaluator = make_pair_evaluator();
  build_result.structure_matrices =
      build_tiled_matrix_form_structure_matrices(
          determinant_to_structure_terms,
          n_structures,
          alpha_reuse_table,
          beta_reuse_table,
          ovlp_act,
          h1e_act,
          n_orbitals,
          eri_act,
          pair_evaluator,
          (same_spin_pair_cache != nullptr)
              ? &same_spin_pair_cache->alpha_pair_cache_ref()
              : nullptr,
          (same_spin_pair_cache != nullptr)
              ? &same_spin_pair_cache->beta_pair_cache_ref()
              : nullptr);

  if (!store_pair_evaluations) {
    return build_result;
  }

  // Diagnostic pair retention should reuse the same tiled/cache inputs rather
  // than routing structure assembly back through the deleted pairwise path.
  SameSpinPairCacheContext local_same_spin_pair_cache;
  const SameSpinPairCacheContext* active_same_spin_pair_cache =
      same_spin_pair_cache;
  if (active_same_spin_pair_cache == nullptr) {
    local_same_spin_pair_cache = build_same_spin_pair_cache_context(
        alpha_det,
        beta_det,
        pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        eri_act,
        SameSpinPairCacheBuildOptions{});
    active_same_spin_pair_cache = &local_same_spin_pair_cache;
  }
  collect_pair_evaluations(
      active_same_spin_pair_cache,
      pair_evaluator,
      alpha_det,
      beta_det,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      &build_result.pair_evaluations);
  return build_result;
}

FullDeterminantStructureBuildResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_impl(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_structures,
    const SameSpinPairCacheContext* same_spin_pair_cache,
    bool store_pair_evaluations) const {
  validate_full_determinant_input(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      n_orbitals,
      n_structures);

  const int n_determinants = static_cast<int>(alpha_det.size());
  FullDeterminantStructureBuildResult build_result;
  build_result.n_determinants = n_determinants;
  const std::size_t n_determinant_pairs =
      unordered_determinant_pair_count(n_determinants);
  if (store_pair_evaluations) {
    build_result.pair_evaluations.resize(n_determinant_pairs);
  }
  if (same_spin_pair_cache != nullptr) {
    if (same_spin_pair_cache->alpha_reuse_table.determinant_to_unique_id.size() !=
            alpha_det.size() ||
        same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id.size() !=
            beta_det.size()) {
      throw std::invalid_argument(
          "same-spin cache context does not match the determinant expansion");
    }
  }

  const SpinDeterminantReuseTable alpha_reuse_table =
      (same_spin_pair_cache != nullptr)
          ? same_spin_pair_cache->alpha_reuse_table
          : build_spin_determinant_reuse_table(alpha_det);
  const SpinDeterminantReuseTable beta_reuse_table =
      (same_spin_pair_cache != nullptr)
          ? same_spin_pair_cache->beta_reuse_table
          : build_spin_determinant_reuse_table(beta_det);
  const FullDeterminantPairEvaluator pair_evaluator = make_pair_evaluator();
  build_result.structure_matrices =
      build_tiled_matrix_form_structure_matrices(
          determinant_to_structure_terms,
          n_structures,
          alpha_reuse_table,
          beta_reuse_table,
          ovlp_act,
          h1e_act,
          n_orbitals,
          active_space_two_electron_result,
          pair_evaluator,
          (same_spin_pair_cache != nullptr)
              ? &same_spin_pair_cache->alpha_pair_cache_ref()
              : nullptr,
          (same_spin_pair_cache != nullptr)
              ? &same_spin_pair_cache->beta_pair_cache_ref()
              : nullptr);

  if (!store_pair_evaluations) {
    return build_result;
  }

  SameSpinPairCacheContext local_same_spin_pair_cache;
  const SameSpinPairCacheContext* active_same_spin_pair_cache =
      same_spin_pair_cache;
  if (active_same_spin_pair_cache == nullptr) {
    local_same_spin_pair_cache = build_same_spin_pair_cache_context(
        alpha_det,
        beta_det,
        pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        SameSpinPairCacheBuildOptions{});
    active_same_spin_pair_cache = &local_same_spin_pair_cache;
  }
  collect_pair_evaluations(
      active_same_spin_pair_cache,
      pair_evaluator,
      alpha_det,
      beta_det,
      ovlp_act,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result,
      &build_result.pair_evaluations);
  return build_result;
}

FullDeterminantStructureBuildResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_with_pair_evaluations(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures) const {
  return build_impl(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      n_structures,
      nullptr,
      true);
}


StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const FullDeterminantStructureData& input) const 
{
  const Eigen::Map<const Eigen::MatrixXd> h1e_act(
      input.h1e_act.data(),
      input.n_active_orbitals,
      input.n_active_orbitals);
  return build(
      input.alpha_det,
      input.beta_det,
      input.determinant_to_structure_terms,
      input.ovlp_act,
      h1e_act,
      input.n_active_orbitals,
      input.eri_act,
      input.n_structures);
}

FullDeterminantStructureBuildResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_with_pair_evaluations(
    const FullDeterminantStructureData& input) const {
  const Eigen::Map<const Eigen::MatrixXd> h1e_act(
      input.h1e_act.data(),
      input.n_active_orbitals,
      input.n_active_orbitals);
  return build_with_pair_evaluations(
      input.alpha_det,
      input.beta_det,
      input.determinant_to_structure_terms,
      input.ovlp_act,
      h1e_act,
      input.n_active_orbitals,
      input.eri_act,
      input.n_structures);
}


}  // namespace xmvb::vb
