#pragma once

#include <cstdint>
#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Canonical unique-spin tile size for matrix-form biorthogonal kernels.
 *
 * Exact selected-space forward and same-spin tile providers must agree on the
 * unique-string tiling so streamed determinant-column assembly can reuse the
 * same cache-friendly block size as the underlying pair evaluators.
 */
int structure_matrix_tile_size();

/**
 * @brief Maximum number of cached unique-spin tiles per provider.
 */
int structure_matrix_tile_cache_tiles();

/**
 * @brief Local borrowed-or-owned first-order opposite-spin projections.
 *
 * `n_rows` and `n_cols` describe one local unique-spin-string block used by a
 * matrix-form structure contraction. Tile-backed callers copy sparse
 * projections into `owned_projection` so the block remains valid even if the
 * source same-spin tile is evicted from the cache later.
 */
struct LocalSpinProjectionBlock {
  struct ProjectionSlot {
    const OppositeSpinPackedPairProjection* borrowed_projection = nullptr;
    OppositeSpinPackedPairProjection owned_projection;

    const OppositeSpinPackedPairProjection& projection() const;
  };

  int n_rows = 0;
  int n_cols = 0;
  std::vector<ProjectionSlot> first_order_projections;

  const OppositeSpinPackedPairProjection& first_order_projection(
      int row_local,
      int column_local) const;
};

/**
 * @brief Ordered same-spin biorthogonal channels for one unique-string pair.
 *
 * The scalar payload stores the exact overlap, one-electron Hamiltonian, and
 * total Hamiltonian channels needed by matrix-form block contraction. The
 * sparse first-order projection is the same opposite-spin regrouping payload
 * used by the nonorthogonal tiled forward path.
 */
struct BiorthogonalForwardSpinPairEntry {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  OppositeSpinPackedPairProjection first_order_projection;
};

/**
 * @brief On-demand tile cache for ordered nonorthogonal same-spin overlaps.
 *
 * Exact selected-space biorthogonal actions need only the nonorthogonal
 * same-spin overlap determinants `S_alpha` / `S_beta`. Building the full
 * ordered same-spin cache every SCF step is much more expensive than the
 * nonorthogonal tiled forward path, so this provider mirrors the tile cache
 * strategy and materializes only the overlap tiles touched by the current
 * contraction block.
 */
class NonorthogonalOverlapSpinPairTileProvider {
public:
  NonorthogonalOverlapSpinPairTileProvider(
      const std::vector<std::vector<int>>& unique_spin_determinants,
      const std::vector<double>& ovlp_act,
      int n_orbitals,
      int tile_size = -1,
      int max_cached_tiles = -1);

  double entry(
      int left_unique_index,
      int right_unique_index) const;

  int n_unique_determinants() const {
    return static_cast<int>(unique_spin_determinants_.size());
  }

private:
  struct CachedTile {
    int row_tile = 0;
    int column_tile = 0;
    int row_begin = 0;
    int row_end = 0;
    int column_begin = 0;
    int column_end = 0;
    std::uint64_t last_access_stamp = 0;
    Eigen::MatrixXd overlap_block;

    double entry(
        int global_row,
        int global_column) const;
  };
  CachedTile* find_or_build_tile(
      int row_tile,
      int column_tile) const;

  const std::vector<std::vector<int>>& unique_spin_determinants_;
  const std::vector<double>& ovlp_act_;
  int n_orbitals_ = 0;
  int tile_size_ = 0;
  int max_cached_tiles_ = 0;
  mutable std::uint64_t access_stamp_ = 0;
  mutable std::vector<CachedTile> cached_tiles_;
};

/**
 * @brief On-demand tile cache for ordered same-spin biorthogonal pair data.
 *
 * This provider evaluates only the unique-spin-string tiles touched by the
 * current contraction block, so callers no longer need dense
 * `N_unique x N_unique` same-spin channel matrices.
 */
class BiorthogonalForwardSpinPairTileProvider {
public:
  BiorthogonalForwardSpinPairTileProvider(
      const std::vector<std::vector<int>>& unique_spin_determinants,
      bool is_alpha_spin,
      const BiorthogonalOrbitalIntegrals& orbital_integrals,
      const ActiveSpaceTwoElectronView& right_right_two_electron_view,
      int tile_size = -1,
      int max_cached_tiles = -1);

  const BiorthogonalForwardSpinPairEntry& entry(
      int left_unique_index,
      int right_unique_index) const;

  int n_unique_determinants() const {
    return static_cast<int>(unique_spin_determinants_.size());
  }

private:
  struct CachedTile {
    int row_tile = 0;
    int column_tile = 0;
    int row_begin = 0;
    int row_end = 0;
    int column_begin = 0;
    int column_end = 0;
    std::uint64_t last_access_stamp = 0;
    std::vector<BiorthogonalForwardSpinPairEntry> entries;

    const BiorthogonalForwardSpinPairEntry& entry(
        int global_row,
        int global_column) const;
  };
  CachedTile* find_or_build_tile(
      int row_tile,
      int column_tile) const;

  const std::vector<std::vector<int>>& unique_spin_determinants_;
  bool is_alpha_spin_ = false;
  const BiorthogonalOrbitalIntegrals& orbital_integrals_;
  const ActiveSpaceTwoElectronView& right_right_two_electron_view_;
  int tile_size_ = 0;
  int max_cached_tiles_ = 0;
  mutable std::uint64_t access_stamp_ = 0;
  mutable std::vector<CachedTile> cached_tiles_;
};

/**
 * @brief Gathers one local biorthogonal same-spin block from the tile cache.
 *
 * `row_indices` and `column_indices` index left and right unique spin strings,
 * respectively. Any requested scalar block is rebuilt densely for the local
 * support only, while the opposite-spin projections are copied into
 * `local_projection_block`.
 */
void gather_biorthogonal_forward_spin_block(
    const BiorthogonalForwardSpinPairTileProvider& tile_provider,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* one_electron_block,
    Eigen::MatrixXd* total_block,
    LocalSpinProjectionBlock* local_projection_block);

/**
 * @brief Gathers one local nonorthogonal same-spin overlap block from cache.
 *
 * The ordered cache already stores all same-spin overlap determinants, so this
 * helper only extracts the requested rows and columns without building the full
 * dense overlap matrix explicitly.
 */
void gather_nonorthogonal_same_spin_overlap_block(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique_determinants,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block);

/**
 * @brief Gathers one local nonorthogonal same-spin overlap block from tiles.
 *
 * This overload keeps the exact selected-space forward path aligned with the
 * nonorthogonal tiled builder: callers request only the local rows/columns
 * needed by the current contraction block and the provider reuses overlap
 * tiles across repeated block visits.
 */
void gather_nonorthogonal_same_spin_overlap_block(
    const NonorthogonalOverlapSpinPairTileProvider& tile_provider,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block);

}  // namespace xmvb::vb::biorthogonal_vbscf
