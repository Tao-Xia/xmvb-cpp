#include "vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

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

namespace {

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

struct SpinExcitation {
  std::vector<int> removed_orbitals;
  std::vector<int> added_orbitals;
  int rank = 0;
};

double apply_annihilation(
    int orbital,
    std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  const auto iterator =
      std::lower_bound(occupied_orbitals->begin(), occupied_orbitals->end(), orbital);
  if (iterator == occupied_orbitals->end() || *iterator != orbital) {
    return 0.0;
  }
  const int parity = static_cast<int>(std::distance(occupied_orbitals->begin(), iterator));
  occupied_orbitals->erase(iterator);
  return parity_sign(parity);
}

double apply_creation(
    int orbital,
    std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  const auto iterator =
      std::lower_bound(occupied_orbitals->begin(), occupied_orbitals->end(), orbital);
  if (iterator != occupied_orbitals->end() && *iterator == orbital) {
    return 0.0;
  }
  const int parity = static_cast<int>(std::distance(occupied_orbitals->begin(), iterator));
  occupied_orbitals->insert(iterator, orbital);
  return parity_sign(parity);
}

SpinExcitation build_spin_excitation(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals) {
  SpinExcitation excitation;
  std::set_difference(
      right_occupied_orbitals.begin(),
      right_occupied_orbitals.end(),
      left_occupied_orbitals.begin(),
      left_occupied_orbitals.end(),
      std::back_inserter(excitation.removed_orbitals));
  std::set_difference(
      left_occupied_orbitals.begin(),
      left_occupied_orbitals.end(),
      right_occupied_orbitals.begin(),
      right_occupied_orbitals.end(),
      std::back_inserter(excitation.added_orbitals));
  if (excitation.removed_orbitals.size() != excitation.added_orbitals.size()) {
    throw std::runtime_error("left/right determinants carry inconsistent electron counts");
  }
  excitation.rank = static_cast<int>(excitation.removed_orbitals.size());
  return excitation;
}

double compute_excitation_phase(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals,
    const SpinExcitation& excitation) {
  if (excitation.rank == 0) {
    return left_occupied_orbitals == right_occupied_orbitals ? 1.0 : 0.0;
  }

  std::vector<int> occupied_orbitals = right_occupied_orbitals;
  double phase = 1.0;
  for (int removed_orbital : excitation.removed_orbitals) {
    phase *= apply_annihilation(removed_orbital, &occupied_orbitals);
    if (phase == 0.0) {
      return 0.0;
    }
  }
  for (auto added_iterator = excitation.added_orbitals.rbegin();
       added_iterator != excitation.added_orbitals.rend();
       ++added_iterator) {
    phase *= apply_creation(*added_iterator, &occupied_orbitals);
    if (phase == 0.0) {
      return 0.0;
    }
  }
  return occupied_orbitals == left_occupied_orbitals ? phase : 0.0;
}

OppositeSpinPackedPairProjection build_biorthogonal_first_order_projection(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const BiorthogonalOrbitalIntegrals& orbital_integrals) {
  OppositeSpinPackedPairProjection projection;
  if (occ_L.empty()) {
    return projection;
  }

  const SpinExcitation excitation = build_spin_excitation(occ_L, occ_R);
  if (excitation.rank >= 2) {
    return projection;
  }

  const double phase = compute_excitation_phase(occ_L, occ_R, excitation);
  if (phase == 0.0) {
    return projection;
  }

  const int n_orbitals = orbital_integrals.n_orbitals;
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  std::vector<double> dense_pair_values(
      xmvb::to_size(n_packed_active_pairs),
      0.0);
  std::vector<unsigned char> touched_mask(
      xmvb::to_size(n_packed_active_pairs),
      0u);
  std::vector<int> touched_indices;

  auto add_transformed_pair = [&](int right_orbital, int left_orbital, double coefficient) {
    for (int right_basis_left = 0; right_basis_left < n_orbitals; ++right_basis_left) {
      const double transformed_coefficient =
          coefficient * orbital_integrals.left_dual_from_right_transform(
                            right_basis_left,
                            left_orbital);
      if (std::abs(transformed_coefficient) <= 1.0e-15) {
        continue;
      }
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          right_orbital,
          right_basis_left);
      if (touched_mask[xmvb::to_size(packed_pair_index)] == 0u) {
        touched_mask[xmvb::to_size(packed_pair_index)] = 1u;
        touched_indices.push_back(packed_pair_index);
      }
      dense_pair_values[xmvb::to_size(packed_pair_index)] += transformed_coefficient;
    }
  };

  if (excitation.rank == 0) {
    for (std::size_t electron_index = 0; electron_index < occ_R.size(); ++electron_index) {
      add_transformed_pair(
          occ_R[electron_index],
          occ_L[electron_index],
          1.0);
    }
  } else {
    add_transformed_pair(
        excitation.removed_orbitals.front(),
        excitation.added_orbitals.front(),
        phase);
  }

  projection.packed_pair_indices.reserve(touched_indices.size());
  projection.packed_pair_values.reserve(touched_indices.size());
  for (const int packed_pair_index : touched_indices) {
    const double packed_pair_value =
        dense_pair_values[xmvb::to_size(packed_pair_index)];
    if (packed_pair_value == 0.0) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(packed_pair_value);
  }
  return projection;
}

BiorthogonalForwardSpinPairEntry evaluate_forward_spin_pair_entry(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    bool is_alpha_spin,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  const BiorthogonalDeterminant left_determinant =
      is_alpha_spin
          ? BiorthogonalDeterminant{occ_L, {}}
          : BiorthogonalDeterminant{{}, occ_L};
  const BiorthogonalDeterminant right_determinant =
      is_alpha_spin
          ? BiorthogonalDeterminant{occ_R, {}}
          : BiorthogonalDeterminant{{}, occ_R};
  const BiorthogonalDeterminantHamiltonianEntry pair_entry =
      evaluate_biorthogonal_determinant_hamiltonian(
          left_determinant,
          right_determinant,
          orbital_integrals,
          right_right_two_electron_view);

  BiorthogonalForwardSpinPairEntry forward_entry;
  forward_entry.overlap = pair_entry.overlap;
  forward_entry.one_electron_hamiltonian = pair_entry.one_electron_hamiltonian;
  forward_entry.total_hamiltonian = pair_entry.total_hamiltonian;
  forward_entry.first_order_projection =
      build_biorthogonal_first_order_projection(
          occ_L,
          occ_R,
          orbital_integrals);
  return forward_entry;
}

double evaluate_nonorthogonal_overlap_spin_pair(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& determinant_overlap_resolver) {
  if (occ_L.empty()) {
    return 1.0;
  }
  const std::vector<double> overlap_submatrix =
      build_overlap_submatrix(
          occ_L,
          occ_R,
          ovlp_act,
          n_orbitals);
  return determinant_overlap_resolver.resolve(
      overlap_submatrix,
      static_cast<int>(occ_L.size()))
      .overlap_determinant;
}

}  // namespace

const OppositeSpinPackedPairProjection&
LocalSpinProjectionBlock::ProjectionSlot::projection() const {
  return (borrowed_projection != nullptr)
      ? *borrowed_projection
      : owned_projection;
}

const OppositeSpinPackedPairProjection&
LocalSpinProjectionBlock::first_order_projection(
    int row_local,
    int column_local) const {
  return xmvb::index_at(
             first_order_projections,
             xmvb::col_major_index(row_local, column_local, n_rows))
      .projection();
}

const BiorthogonalForwardSpinPairEntry&
BiorthogonalForwardSpinPairTileProvider::CachedTile::entry(
    int global_row,
    int global_column) const {
  const int local_row = global_row - row_begin;
  const int local_column = global_column - column_begin;
  return xmvb::index_at(
      entries,
      xmvb::col_major_index(
          local_row,
          local_column,
          row_end - row_begin));
}

double NonorthogonalOverlapSpinPairTileProvider::CachedTile::entry(
    int global_row,
    int global_column) const {
  const int local_row = global_row - row_begin;
  const int local_column = global_column - column_begin;
  return overlap_block(local_row, local_column);
}

BiorthogonalForwardSpinPairTileProvider::BiorthogonalForwardSpinPairTileProvider(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    bool is_alpha_spin,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view,
    int tile_size,
    int max_cached_tiles)
    : unique_spin_determinants_(unique_spin_determinants),
      is_alpha_spin_(is_alpha_spin),
      orbital_integrals_(orbital_integrals),
      right_right_two_electron_view_(right_right_two_electron_view),
      tile_size_(tile_size > 0 ? tile_size : structure_matrix_tile_size()),
      max_cached_tiles_(
          max_cached_tiles > 0
              ? max_cached_tiles
              : structure_matrix_tile_cache_tiles()) {
  if (tile_size_ <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  if (max_cached_tiles_ <= 0) {
    throw std::invalid_argument("max_cached_tiles must be positive");
  }
  cached_tiles_.reserve(xmvb::to_size(max_cached_tiles_));
}

const BiorthogonalForwardSpinPairEntry&
BiorthogonalForwardSpinPairTileProvider::entry(
    int left_unique_index,
    int right_unique_index) const {
  if (left_unique_index < 0 ||
      left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
      right_unique_index < 0 ||
      right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
    throw std::out_of_range("unique spin tile lookup index out of range");
  }

  const int row_tile = left_unique_index / tile_size_;
  const int column_tile = right_unique_index / tile_size_;
  CachedTile* tile = find_or_build_tile(row_tile, column_tile);
  tile->last_access_stamp = ++access_stamp_;
  return tile->entry(left_unique_index, right_unique_index);
}

BiorthogonalForwardSpinPairTileProvider::CachedTile*
BiorthogonalForwardSpinPairTileProvider::find_or_build_tile(
    int row_tile,
    int column_tile) const {
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

  CachedTile built_tile;
  built_tile.row_tile = row_tile;
  built_tile.column_tile = column_tile;
  built_tile.row_begin = row_begin;
  built_tile.row_end = row_end;
  built_tile.column_begin = column_begin;
  built_tile.column_end = column_end;
  built_tile.last_access_stamp = ++access_stamp_;
  built_tile.entries.resize(
      xmvb::to_size(row_end - row_begin) * xmvb::to_size(column_end - column_begin));

  // Each cached tile stores only the same-spin channels and sparse first-order
  // opposite-spin payloads needed by later matrix-form block contraction. This
  // keeps the forward path aligned with the nonorthogonal tiled builder while
  // avoiding a dense global biorthogonal cache.
  for (int column_index = column_begin;
       column_index < column_end;
       ++column_index) {
    for (int row_index = row_begin;
         row_index < row_end;
         ++row_index) {
      xmvb::index_at(
          built_tile.entries,
          xmvb::col_major_index(
              row_index - row_begin,
              column_index - column_begin,
              row_end - row_begin)) =
          evaluate_forward_spin_pair_entry(
              unique_spin_determinants_[xmvb::to_size(row_index)],
              unique_spin_determinants_[xmvb::to_size(column_index)],
              is_alpha_spin_,
              orbital_integrals_,
              right_right_two_electron_view_);
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

NonorthogonalOverlapSpinPairTileProvider::NonorthogonalOverlapSpinPairTileProvider(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    const std::vector<double>& ovlp_act,
    int n_orbitals,
    int tile_size,
    int max_cached_tiles)
    : unique_spin_determinants_(unique_spin_determinants),
      ovlp_act_(ovlp_act),
      n_orbitals_(n_orbitals),
      tile_size_(tile_size > 0 ? tile_size : structure_matrix_tile_size()),
      max_cached_tiles_(
          max_cached_tiles > 0
              ? max_cached_tiles
              : structure_matrix_tile_cache_tiles()) {
  if (n_orbitals_ <= 0) {
    throw std::invalid_argument("n_orbitals must be positive");
  }
  if (tile_size_ <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  if (max_cached_tiles_ <= 0) {
    throw std::invalid_argument("max_cached_tiles must be positive");
  }
  cached_tiles_.reserve(xmvb::to_size(max_cached_tiles_));
}

double NonorthogonalOverlapSpinPairTileProvider::entry(
    int left_unique_index,
    int right_unique_index) const {
  if (left_unique_index < 0 ||
      left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
      right_unique_index < 0 ||
      right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
    throw std::out_of_range("nonorthogonal overlap tile lookup index out of range");
  }

  const int row_tile = left_unique_index / tile_size_;
  const int column_tile = right_unique_index / tile_size_;
  CachedTile* tile = find_or_build_tile(row_tile, column_tile);
  tile->last_access_stamp = ++access_stamp_;
  return tile->entry(left_unique_index, right_unique_index);
}

NonorthogonalOverlapSpinPairTileProvider::CachedTile*
NonorthogonalOverlapSpinPairTileProvider::find_or_build_tile(
    int row_tile,
    int column_tile) const {
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

  CachedTile built_tile;
  built_tile.row_tile = row_tile;
  built_tile.column_tile = column_tile;
  built_tile.row_begin = row_begin;
  built_tile.row_end = row_end;
  built_tile.column_begin = column_begin;
  built_tile.column_end = column_end;
  built_tile.last_access_stamp = ++access_stamp_;
  built_tile.overlap_block.resize(
      row_end - row_begin,
      column_end - column_begin);

  const xmvb::vb::DeterminantOverlapResolver determinant_overlap_resolver;
  for (int column_index = column_begin;
       column_index < column_end;
       ++column_index) {
    for (int row_index = row_begin;
         row_index < row_end;
         ++row_index) {
      built_tile.overlap_block(
          row_index - row_begin,
          column_index - column_begin) =
          evaluate_nonorthogonal_overlap_spin_pair(
              unique_spin_determinants_[xmvb::to_size(row_index)],
              unique_spin_determinants_[xmvb::to_size(column_index)],
              ovlp_act_,
              n_orbitals_,
              determinant_overlap_resolver);
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

void gather_biorthogonal_forward_spin_block(
    const BiorthogonalForwardSpinPairTileProvider& tile_provider,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* one_electron_block,
    Eigen::MatrixXd* total_block,
    LocalSpinProjectionBlock* local_projection_block) {
  if (overlap_block == nullptr &&
      one_electron_block == nullptr &&
      total_block == nullptr &&
      local_projection_block == nullptr) {
    throw std::invalid_argument("at least one local biorthogonal block output is required");
  }

  const int n_rows = static_cast<int>(row_indices.size());
  const int n_cols = static_cast<int>(column_indices.size());
  if (overlap_block != nullptr) {
    overlap_block->resize(n_rows, n_cols);
  }
  if (one_electron_block != nullptr) {
    one_electron_block->resize(n_rows, n_cols);
  }
  if (total_block != nullptr) {
    total_block->resize(n_rows, n_cols);
  }
  if (local_projection_block != nullptr) {
    local_projection_block->n_rows = n_rows;
    local_projection_block->n_cols = n_cols;
    local_projection_block->first_order_projections.assign(
        xmvb::to_size(n_rows) * xmvb::to_size(n_cols),
        LocalSpinProjectionBlock::ProjectionSlot{});
  }

  for (int column_local = 0; column_local < n_cols; ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    for (int row_local = 0; row_local < n_rows; ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      const auto& entry = tile_provider.entry(row_global, column_global);
      if (overlap_block != nullptr) {
        (*overlap_block)(row_local, column_local) = entry.overlap;
      }
      if (one_electron_block != nullptr) {
        (*one_electron_block)(row_local, column_local) =
            entry.one_electron_hamiltonian;
      }
      if (total_block != nullptr) {
        (*total_block)(row_local, column_local) = entry.total_hamiltonian;
      }
      if (local_projection_block != nullptr) {
        auto& slot = xmvb::index_at(
            local_projection_block->first_order_projections,
            xmvb::col_major_index(
                row_local,
                column_local,
                n_rows));
        slot.borrowed_projection = nullptr;
        slot.owned_projection = entry.first_order_projection;
      }
    }
  }
}

void gather_nonorthogonal_same_spin_overlap_block(
    const NonorthogonalOverlapSpinPairTileProvider& tile_provider,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block) {
  if (overlap_block == nullptr) {
    throw std::invalid_argument("overlap_block must not be null");
  }
  overlap_block->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      (*overlap_block)(row_local, column_local) =
          tile_provider.entry(row_global, column_global);
    }
  }
}

void gather_nonorthogonal_same_spin_overlap_block(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique_determinants,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block) {
  if (overlap_block == nullptr) {
    throw std::invalid_argument("overlap_block must not be null");
  }
  overlap_block->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    if (column_global < 0 || column_global >= n_unique_determinants) {
      throw std::out_of_range("column index is out of range");
    }
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      if (row_global < 0 || row_global >= n_unique_determinants) {
        throw std::out_of_range("row index is out of range");
      }
      (*overlap_block)(row_local, column_local) =
          pair_cache[ordered_spin_pair_storage_index(
              row_global,
              column_global,
              n_unique_determinants)]
              .overlap_result
              .overlap_determinant;
    }
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
