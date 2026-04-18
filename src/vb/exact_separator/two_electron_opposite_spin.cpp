#include "vb/exact_separator/two_electron.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "vb/exact_separator/component_terms.hpp"
#include "vb/exact_separator/leaf_boundary_message.hpp"
#include "vb/exact_separator/leaf_coefficient_operator.hpp"
#include "vb/exact_separator/spin_state_aggregate.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::exact_separator {

namespace {


struct SpinMaskCofactorAggregate {
  // Exact one-spin one-leaf separator aggregate in support-orbital
  // coordinates. `signed_overlap` is the exact overlap rebuilt from the mask
  // sum in block order, `signed_first_cofactor` is the corresponding exact
  // first-cofactor matrix, and `signed_second_cofactor_entries` is the exact
  // sparse same-spin `(2,2)` payload accumulated over separator masks. Eigen::MatrixXd
  // rows are ket/right orbitals and columns are bra/left orbitals.
  struct SecondCofactorEntry {
    int row_first_orbital = -1;
    int row_second_orbital = -1;
    int col_first_orbital = -1;
    int col_second_orbital = -1;
    double value = 0.0;
  };

  double signed_overlap = 0.0;
  Eigen::MatrixXd signed_first_cofactor;
  double signed_same_spin_two_electron = 0.0;
  std::vector<SecondCofactorEntry> signed_second_cofactor_entries;
  std::uint64_t mask_state_count = 0;
  std::uint64_t skipped_zero_mask_state_count = 0;
};

struct SpinDeletionKey {
  std::vector<int> row_labels;
  std::vector<int> col_labels;
};

bool operator<(const SpinDeletionKey& left, const SpinDeletionKey& right) {
  return std::tie(left.row_labels, left.col_labels) <
      std::tie(right.row_labels, right.col_labels);
}

struct SpinDeletionPayload {
  int row_count = 0;
  int col_count = 0;
  std::map<SpinDeletionKey, double> sectors;
};

enum class PartialSide {
  LeftFrontier,
  RightComplement,
};

struct MinorEntry {
  int local_index = -1;
  int orbital_label = -1;
  double value = 0.0;
};

struct CofactorEntry {
  int row_local_index = -1;
  int col_local_index = -1;
  int row_orbital_label = -1;
  int col_orbital_label = -1;
  double value = 0.0;
};

struct RowPairEntry {
  int first_row_local_index = -1;
  int second_row_local_index = -1;
  int first_row_orbital_label = -1;
  int second_row_orbital_label = -1;
  double value = 0.0;
};

struct ColPairEntry {
  int first_col_local_index = -1;
  int second_col_local_index = -1;
  int first_col_orbital_label = -1;
  int second_col_orbital_label = -1;
  double value = 0.0;
};

struct RowPairColEntry {
  int first_row_local_index = -1;
  int second_row_local_index = -1;
  int col_local_index = -1;
  int first_row_orbital_label = -1;
  int second_row_orbital_label = -1;
  int col_orbital_label = -1;
  double value = 0.0;
};

struct RowColPairEntry {
  int row_local_index = -1;
  int first_col_local_index = -1;
  int second_col_local_index = -1;
  int row_orbital_label = -1;
  int first_col_orbital_label = -1;
  int second_col_orbital_label = -1;
  double value = 0.0;
};

struct SecondCofactorLocalEntry {
  int first_row_local_index = -1;
  int second_row_local_index = -1;
  int first_col_local_index = -1;
  int second_col_local_index = -1;
  int first_row_orbital_label = -1;
  int second_row_orbital_label = -1;
  int first_col_orbital_label = -1;
  int second_col_orbital_label = -1;
  double value = 0.0;
};

struct PartialSpinPayload {
  // Exact partial deleted-minor payload for one one-spin mask block.
  //
  // The one-leaf recurrence only needs the local sectors that can contribute
  // to the full square spin block after merging the frontier and root
  // complement:
  // - closed overlap `(0,0)` when the local block is square;
  // - row-open `(1,0)` or col-open `(0,1)` deleted minors when the local block
  //   is rectangular by one;
  // - first cofactors `(1,1)` when the local block is square.
  int n_rows = 0;
  int n_cols = 0;
  double closed_overlap = 0.0;
  std::vector<MinorEntry> row_open_entries;
  std::vector<MinorEntry> col_open_entries;
  std::vector<CofactorEntry> cofactor_entries;
};

struct Degree2SpinPayload {
  // Exact local deleted-minor payload up to degree 2 for one spin block.
  //
  // The entry values already include the validated local deleted-minor sign
  // convention for the chosen `PartialSide`. The one-mask same-spin merge only
  // needs to apply the additional block-shift phase derived from the frontier /
  // root split and then canonicalize the final support-orbital row/column
  // pairs.
  int n_rows = 0;
  int n_cols = 0;
  double overlap = 0.0;
  std::vector<MinorEntry> row_open_entries;
  std::vector<MinorEntry> col_open_entries;
  std::vector<CofactorEntry> cofactor_entries;
  std::vector<RowPairEntry> row_pair_entries;
  std::vector<ColPairEntry> col_pair_entries;
  std::vector<RowPairColEntry> row_pair_col_entries;
  std::vector<RowColPairEntry> row_col_pair_entries;
  std::vector<SecondCofactorLocalEntry> second_cofactor_entries;
};

void fill_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    Eigen::MatrixXd* overlap_block);

Eigen::MatrixXd build_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size);

double determinant_of_dense_matrix(
    const Eigen::MatrixXd& matrix,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

double contract_same_spin_second_cofactors(
    const std::vector<SpinMaskCofactorAggregate::SecondCofactorEntry>& second_cofactor_entries,
    const std::vector<double>& packed_active_two_electron_integrals);

void fill_signed_open_minor_vector(
    const Eigen::MatrixXd& overlap_block,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    Eigen::VectorXd* signed_open_minor_vector);

Eigen::VectorXd build_signed_open_minor_vector(
    const Eigen::MatrixXd& overlap_block,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

bool is_square_payload(const PartialSpinPayload& payload);

PartialSpinPayload build_partial_spin_payload(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int selected_root_rows,
    int selected_root_cols,
    bool zero_selected_root_block,
    PartialSide side,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinDeletionPayload spin_payload_from_partial(const PartialSpinPayload& payload);

template <typename Callback>
void for_each_index_subset(
    int begin,
    int end,
    int rank,
    const Callback& callback) {
  if (rank < 0 || end < begin) {
    throw std::invalid_argument("invalid subset enumeration request");
  }
  if (rank == 0) {
    callback({});
    return;
  }
  for (int first = begin; first < end; ++first) {
    if (rank == 1) {
      callback({first});
      continue;
    }
    for (int second = first + 1; second < end; ++second) {
      callback({first, second});
    }
  }
}

int deleted_index_sum(const std::vector<int>& deleted_indices) {
  return std::accumulate(deleted_indices.begin(), deleted_indices.end(), 0);
}

void fill_deleted_minor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    Eigen::MatrixXd* minor);

Eigen::MatrixXd build_deleted_minor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols) {
  Eigen::MatrixXd minor;
  fill_deleted_minor_matrix(
      overlap_block,
      deleted_rows,
      deleted_cols,
      &minor);
  return minor;
}

void fill_deleted_minor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    Eigen::MatrixXd* minor) {
  if (minor == nullptr) {
    throw std::invalid_argument("minor must not be null");
  }
  if (overlap_block.rows() < static_cast<int>(deleted_rows.size()) ||
      overlap_block.cols() < static_cast<int>(deleted_cols.size())) {
    throw std::invalid_argument("deleted-minor request is larger than the source matrix");
  }

  minor->resize(
      overlap_block.rows() - static_cast<int>(deleted_rows.size()),
      overlap_block.cols() - static_cast<int>(deleted_cols.size()));
  int minor_row = 0;
  std::size_t deleted_row_cursor = 0U;
  for (int row = 0; row < overlap_block.rows(); ++row) {
    if (deleted_row_cursor < deleted_rows.size() &&
        row == deleted_rows[deleted_row_cursor]) {
      ++deleted_row_cursor;
      continue;
    }
    int minor_col = 0;
    std::size_t deleted_col_cursor = 0U;
    for (int col = 0; col < overlap_block.cols(); ++col) {
      if (deleted_col_cursor < deleted_cols.size() &&
          col == deleted_cols[deleted_col_cursor]) {
        ++deleted_col_cursor;
        continue;
      }
      (*minor)(minor_row, minor_col) = overlap_block(row, col);
      ++minor_col;
    }
    ++minor_row;
  }
}

double deleted_minor_sign(
    int n_rows,
    int n_cols,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    bool left_frontier) {
  int parity =
      deleted_index_sum(deleted_rows) +
      deleted_index_sum(deleted_cols);
  if (left_frontier) {
    parity += std::min(n_rows, n_cols) * (n_rows - n_cols);
    if ((deleted_rows.size() == 2U && deleted_cols.size() == 1U) ||
        (deleted_rows.size() == 1U && deleted_cols.size() == 2U)) {
      parity += 1;
    }
  }
  return parity_sign(parity);
}

bool deleted_indices_in_suffix(
    const std::vector<int>& deleted_indices,
    int internal_begin) {
  for (const int deleted_index : deleted_indices) {
    if (deleted_index < internal_begin) {
      return false;
    }
  }
  return true;
}

void cleanup_spin_payload(SpinDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  for (auto iterator = payload->sectors.begin(); iterator != payload->sectors.end();) {
    if (std::abs(iterator->second) <= 1.0e-15) {
      iterator = payload->sectors.erase(iterator);
    } else {
      ++iterator;
    }
  }
}

SpinDeletionPayload build_spin_deleted_minor_payload(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int zeroed_prefix_rows,
    int zeroed_prefix_cols,
    bool left_frontier,
    const DeterminantOverlapResolver& overlap_resolver,
    bool restrict_deleted_to_internal_suffix,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const int n_rows = static_cast<int>(right_occ.size());
  const int n_cols = static_cast<int>(left_occ.size());
  SpinDeletionPayload payload;
  payload.row_count = n_rows;
  payload.col_count = n_cols;

  Eigen::MatrixXd overlap_block =
      build_overlap_block(left_occ, right_occ, overlap_storage, n_orbitals);
  if (zeroed_prefix_rows > 0 && zeroed_prefix_cols > 0) {
    overlap_block.topLeftCorner(zeroed_prefix_rows, zeroed_prefix_cols).setZero();
  }

  for (int deleted_row_rank = 0; deleted_row_rank <= 2; ++deleted_row_rank) {
    if (deleted_row_rank > n_rows) {
      continue;
    }
    for_each_index_subset(
        0,
        n_rows,
        deleted_row_rank,
        [&](const std::vector<int>& deleted_rows) {
          if (restrict_deleted_to_internal_suffix &&
              !deleted_indices_in_suffix(deleted_rows, zeroed_prefix_rows)) {
            return;
          }
          for (int deleted_col_rank = 0; deleted_col_rank <= 2; ++deleted_col_rank) {
            if (deleted_col_rank > n_cols) {
              continue;
            }
            if (n_rows - deleted_row_rank != n_cols - deleted_col_rank) {
              continue;
            }
            for_each_index_subset(
                0,
                n_cols,
                deleted_col_rank,
                [&](const std::vector<int>& deleted_cols) {
                  if (restrict_deleted_to_internal_suffix &&
                      !deleted_indices_in_suffix(deleted_cols, zeroed_prefix_cols)) {
                    return;
                  }
                  const Eigen::MatrixXd minor = build_deleted_minor_matrix(
                      overlap_block,
                      deleted_rows,
                      deleted_cols);
                  const double determinant = determinant_of_dense_matrix(
                      minor,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  if (std::abs(determinant) <= 1.0e-15) {
                    return;
                  }

                  std::vector<int> deleted_row_labels;
                  std::vector<int> deleted_col_labels;
                  deleted_row_labels.reserve(deleted_rows.size());
                  deleted_col_labels.reserve(deleted_cols.size());
                  for (const int row : deleted_rows) {
                    deleted_row_labels.push_back(
                        right_occ[xmvb::to_size(row)]);
                  }
                  for (const int col : deleted_cols) {
                    deleted_col_labels.push_back(
                        left_occ[xmvb::to_size(col)]);
                  }
                  payload.sectors[SpinDeletionKey{
                      std::move(deleted_row_labels),
                      std::move(deleted_col_labels)}] +=
                      deleted_minor_sign(
                          n_rows,
                          n_cols,
                          deleted_rows,
                          deleted_cols,
                          left_frontier) *
                      determinant;
                });
          }
        });
  }

  cleanup_spin_payload(&payload);
  return payload;
}

SpinDeletionKey make_spin_key(
    std::vector<int> row_labels = {},
    std::vector<int> col_labels = {}) {
  return SpinDeletionKey{std::move(row_labels), std::move(col_labels)};
}

int sector_degree(const SpinDeletionKey& key) {
  return std::max(
      static_cast<int>(key.row_labels.size()),
      static_cast<int>(key.col_labels.size()));
}

int spin_sector_merge_parity(
    int left_row_count,
    int left_col_count,
    const SpinDeletionKey& left_key,
    int right_row_count,
    int right_col_count,
    const SpinDeletionKey& right_key) {
  const int left_row_rank = static_cast<int>(left_key.row_labels.size());
  const int left_col_rank = static_cast<int>(left_key.col_labels.size());
  const int left_row_excess = std::max(0, left_row_rank - left_col_rank);
  const int left_col_excess = std::max(0, left_col_rank - left_row_rank);
  const int right_degree = sector_degree(right_key);
  return
      left_row_excess * right_col_count +
      left_col_excess * right_row_count +
      right_degree * (left_row_count + left_col_count);
}

SpinDeletionPayload merge_spin_deletion_payloads(
    const SpinDeletionPayload& left,
    const SpinDeletionPayload& right) {
  SpinDeletionPayload merged;
  merged.row_count = left.row_count + right.row_count;
  merged.col_count = left.col_count + right.col_count;

  for (const auto& [left_key, left_value] : left.sectors) {
    if (std::abs(left_value) <= 1.0e-15) {
      continue;
    }
    for (const auto& [right_key, right_value] : right.sectors) {
      if (std::abs(right_value) <= 1.0e-15) {
        continue;
      }
      const std::size_t combined_row_rank =
          left_key.row_labels.size() + right_key.row_labels.size();
      const std::size_t combined_col_rank =
          left_key.col_labels.size() + right_key.col_labels.size();
      if (combined_row_rank > 2U || combined_col_rank > 2U) {
        continue;
      }

      SpinDeletionKey combined_key;
      combined_key.row_labels.reserve(combined_row_rank);
      combined_key.row_labels.insert(
          combined_key.row_labels.end(),
          left_key.row_labels.begin(),
          left_key.row_labels.end());
      combined_key.row_labels.insert(
          combined_key.row_labels.end(),
          right_key.row_labels.begin(),
          right_key.row_labels.end());
      combined_key.col_labels.reserve(combined_col_rank);
      combined_key.col_labels.insert(
          combined_key.col_labels.end(),
          left_key.col_labels.begin(),
          left_key.col_labels.end());
      combined_key.col_labels.insert(
          combined_key.col_labels.end(),
          right_key.col_labels.begin(),
          right_key.col_labels.end());

      const int parity = spin_sector_merge_parity(
          left.row_count,
          left.col_count,
          left_key,
          right.row_count,
          right.col_count,
          right_key);
      merged.sectors[std::move(combined_key)] +=
          parity_sign(parity) * left_value * right_value;
    }
  }

  cleanup_spin_payload(&merged);
  return merged;
}

void add_scaled_spin_payload(
    const SpinDeletionPayload& source,
    double scale,
    SpinDeletionPayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  if (destination->sectors.empty()) {
    destination->row_count = source.row_count;
    destination->col_count = source.col_count;
  }
  for (const auto& [key, value] : source.sectors) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    destination->sectors[key] += scale * value;
  }
}

std::uint32_t open_state_mask_limit(int n_bits) {
  if (n_bits < 0 || n_bits >= 31) {
    throw std::invalid_argument("open-state mask dimension is out of range");
  }
  return static_cast<std::uint32_t>(1U) << static_cast<std::uint32_t>(n_bits);
}

int block_order_parity(
    int n_root_occ,
    std::uint32_t selected_mask,
    int leaf_occ_size) {
  // The one-leaf recurrence reorders the original block order
  //   [root..., leaf...]
  // into
  //   [selected_root..., leaf..., root_remainder...].
  // Each mask contribution therefore carries the determinant sign of this
  // permutation on the row side and on the column side.
  std::vector<int> block_order;
  block_order.reserve(xmvb::to_size(n_root_occ + leaf_occ_size));
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (static_cast<std::uint32_t>(1U) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  for (int leaf_position = 0; leaf_position < leaf_occ_size; ++leaf_position) {
    block_order.push_back(n_root_occ + leaf_position);
  }
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (static_cast<std::uint32_t>(1U) << root_position)) == 0U) {
      block_order.push_back(root_position);
    }
  }

  int parity = 0;
  for (std::size_t left_index = 0; left_index + 1 < block_order.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < block_order.size();
         ++right_index) {
      if (block_order[left_index] > block_order[right_index]) {
        parity ^= 1;
      }
    }
  }
  return parity;
}

std::vector<int> select_occ_by_mask(
    const std::vector<int>& occ,
    std::uint32_t mask) {
  std::vector<int> selected;
  selected.reserve(occ.size());
  for (int index = 0; index < static_cast<int>(occ.size()); ++index) {
    if ((mask & (static_cast<std::uint32_t>(1U) << index)) != 0U) {
      selected.push_back(occ[xmvb::to_size(index)]);
    }
  }
  return selected;
}

std::vector<std::vector<int>> build_mask_occ_table(const std::vector<int>& occ) {
  const std::uint32_t limit = open_state_mask_limit(static_cast<int>(occ.size()));
  std::vector<std::vector<int>> occ_by_mask(xmvb::to_size(limit));
  for (std::uint32_t mask = 0; mask < limit; ++mask) {
    occ_by_mask[xmvb::to_size(mask)] = select_occ_by_mask(occ, mask);
  }
  return occ_by_mask;
}

std::vector<std::vector<int>> build_mask_position_table(int count) {
  const std::uint32_t limit = open_state_mask_limit(count);
  std::vector<std::vector<int>> positions_by_mask(xmvb::to_size(limit));
  for (std::uint32_t mask = 0; mask < limit; ++mask) {
    auto& positions = positions_by_mask[xmvb::to_size(mask)];
    positions.reserve(xmvb::to_size(__builtin_popcount(mask)));
    for (int index = 0; index < count; ++index) {
      if ((mask & (static_cast<std::uint32_t>(1U) << index)) != 0U) {
        positions.push_back(index);
      }
    }
  }
  return positions_by_mask;
}

int popcount(std::uint32_t mask) {
  int count = 0;
  while (mask != 0U) {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

std::vector<std::vector<std::uint32_t>> build_masks_by_popcount(int n_bits) {
  const std::uint32_t limit = open_state_mask_limit(n_bits);
  std::vector<std::vector<std::uint32_t>> masks_by_popcount(
      xmvb::to_size(n_bits + 1));
  for (std::uint32_t mask = 0; mask < limit; ++mask) {
    masks_by_popcount[xmvb::to_size(popcount(mask))].push_back(mask);
  }
  return masks_by_popcount;
}

void append_occ_block(
    const std::vector<int>& block,
    std::vector<int>* ordered_occ) {
  if (ordered_occ == nullptr) {
    throw std::invalid_argument("ordered occupied-orbital output must not be null");
  }
  ordered_occ->insert(ordered_occ->end(), block.begin(), block.end());
}

void fill_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    Eigen::MatrixXd* overlap_block) {
  // Writes the dense occupied-overlap block into reusable storage. This keeps
  // hot one-leaf mask builders from allocating a fresh Eigen matrix for every
  // frontier/root payload.
  if (overlap_block == nullptr) {
    throw std::invalid_argument("overlap_block must not be null");
  }
  overlap_block->resize(
      static_cast<int>(right_occ.size()),
      static_cast<int>(left_occ.size()));
  for (int col = 0; col < static_cast<int>(left_occ.size()); ++col) {
    const int left_orbital = left_occ[xmvb::to_size(col)];
    for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
      const int right_orbital = right_occ[xmvb::to_size(row)];
      (*overlap_block)(row, col) =
          support_overlap_storage[xmvb::to_size(left_orbital) *
                                      xmvb::to_size(support_size) +
                                  xmvb::to_size(right_orbital)];
    }
  }
}

Eigen::MatrixXd build_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size) {
  // Dense block in the repository-wide convention:
  //   row    = ket/right occupied orbital
  //   column = bra/left occupied orbital
  Eigen::MatrixXd overlap_block;
  fill_overlap_block(
      left_occ,
      right_occ,
      support_overlap_storage,
      support_size,
      &overlap_block);
  return overlap_block;
}

double determinant_of_dense_matrix(
    const Eigen::MatrixXd& matrix,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  ++(*subdeterminant_evaluations);
  switch (matrix.rows()) {
    case 0:
      return 1.0;
    case 1:
      return matrix(0, 0);
    default:
      break;
  }
  return overlap_resolver.resolve_matrix(matrix).overlap_determinant;
}

void fill_signed_open_minor_vector(
    const Eigen::MatrixXd& overlap_block,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    Eigen::VectorXd* signed_open_minor_vector) {
  // Reuses one dynamic vector for the one-off rectangular null-space minors
  // that generate the exact `(1,0)` or `(0,1)` sectors. The algebra matches
  // the existing validated path exactly; only the temporary lifetime changes.
  if (signed_open_minor_vector == nullptr) {
    throw std::invalid_argument("signed_open_minor_vector must not be null");
  }
  signed_open_minor_vector->resize(0);
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  const int n_rows = overlap_block.rows();
  const int n_cols = overlap_block.cols();
  if (!((n_rows == n_cols + 1) || (n_cols == n_rows + 1))) {
    throw std::invalid_argument(
        "build_signed_open_minor_vector requires a one-off rectangular matrix");
  }

  const int null_vector_size = std::max(n_rows, n_cols);
  if (null_vector_size == 0) {
    return;
  }
  const int rank_dimension = std::min(n_rows, n_cols);
  ++(*subdeterminant_evaluations);
  const Eigen::MatrixXd kernel_operator =
      (n_rows == n_cols + 1) ? overlap_block.transpose() : overlap_block;
  Eigen::FullPivLU<Eigen::MatrixXd> rectangular_lu(kernel_operator);
  rectangular_lu.setThreshold(overlap_resolver.linear_dependence_threshold());
  if (rectangular_lu.rank() != rank_dimension ||
      rectangular_lu.dimensionOfKernel() != 1) {
    *signed_open_minor_vector = Eigen::VectorXd::Zero(null_vector_size);
    return;
  }

  const Eigen::VectorXd null_vector = rectangular_lu.kernel().col(0);
  Eigen::Index anchor_index_eigen = 0;
  null_vector.cwiseAbs().maxCoeff(&anchor_index_eigen);
  const int anchor_index = static_cast<int>(anchor_index_eigen);
  if (std::abs(null_vector(anchor_index)) <= 1.0e-15) {
    *signed_open_minor_vector = Eigen::VectorXd::Zero(null_vector_size);
    return;
  }

  Eigen::MatrixXd anchor_minor;
  if (n_rows == n_cols + 1) {
    anchor_minor.resize(n_cols, n_cols);
    if (anchor_index > 0) {
      anchor_minor.topRows(anchor_index) = overlap_block.topRows(anchor_index);
    }
    const int trailing_row_count = n_cols - anchor_index;
    if (trailing_row_count > 0) {
      anchor_minor.bottomRows(trailing_row_count) =
          overlap_block.bottomRows(trailing_row_count);
    }
  } else {
    anchor_minor.resize(n_rows, n_rows);
    if (anchor_index > 0) {
      anchor_minor.leftCols(anchor_index) = overlap_block.leftCols(anchor_index);
    }
    const int trailing_col_count = n_rows - anchor_index;
    if (trailing_col_count > 0) {
      anchor_minor.rightCols(trailing_col_count) =
          overlap_block.rightCols(trailing_col_count);
    }
  }
  const double anchor_minor_value =
      parity_sign(anchor_index) *
      determinant_of_dense_matrix(
          anchor_minor,
          overlap_resolver,
          subdeterminant_evaluations);
  *signed_open_minor_vector =
      (anchor_minor_value / null_vector(anchor_index)) * null_vector;
}

Eigen::VectorXd build_signed_open_minor_vector(
    const Eigen::MatrixXd& overlap_block,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  Eigen::VectorXd signed_open_minor_vector;
  fill_signed_open_minor_vector(
      overlap_block,
      overlap_resolver,
      subdeterminant_evaluations,
      &signed_open_minor_vector);
  return signed_open_minor_vector;
}

bool is_square_payload(const PartialSpinPayload& payload) {
  return payload.n_rows == payload.n_cols;
}

PartialSpinPayload build_partial_spin_payload(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int selected_root_rows,
    int selected_root_cols,
    bool zero_selected_root_block,
    PartialSide side,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  PartialSpinPayload payload;
  payload.n_rows = static_cast<int>(right_occ.size());
  payload.n_cols = static_cast<int>(left_occ.size());

  Eigen::MatrixXd overlap_block =
      build_overlap_block(left_occ, right_occ, overlap_storage, n_orbitals);
  if (zero_selected_root_block && selected_root_rows > 0 && selected_root_cols > 0) {
    overlap_block.topLeftCorner(selected_root_rows, selected_root_cols).setZero();
  }

  const int internal_row_begin = zero_selected_root_block ? selected_root_rows : 0;
  const int internal_col_begin = zero_selected_root_block ? selected_root_cols : 0;

  if (payload.n_rows == payload.n_cols) {
    if (payload.n_rows == 0) {
      payload.closed_overlap = 1.0;
      return payload;
    }
    ++(*subdeterminant_evaluations);
    const auto overlap_result = overlap_resolver.resolve_matrix(overlap_block);
    payload.closed_overlap = overlap_result.overlap_determinant;
    const Eigen::MatrixXd cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);
    for (int col = 0; col < payload.n_cols; ++col) {
      if (col < internal_col_begin) {
        continue;
      }
      for (int row = 0; row < payload.n_rows; ++row) {
        if (row < internal_row_begin) {
          continue;
        }
        const double value = cofactor_1st(row, col);
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload.cofactor_entries.push_back({
            .row_local_index = row,
            .col_local_index = col,
            .row_orbital_label = right_occ[xmvb::to_size(row)],
            .col_orbital_label = left_occ[xmvb::to_size(col)],
            .value = value,
        });
      }
    }
    return payload;
  }

  if (payload.n_rows == payload.n_cols + 1) {
    const Eigen::VectorXd signed_deleted_row_minors =
        build_signed_open_minor_vector(
            overlap_block,
            overlap_resolver,
            subdeterminant_evaluations);
    for (int row = 0; row < payload.n_rows; ++row) {
      if (row < internal_row_begin) {
        continue;
      }
      const double determinant = parity_sign(row) * signed_deleted_row_minors(row);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      const double sign =
          (side == PartialSide::LeftFrontier)
              ? parity_sign(row + payload.n_cols)
              : parity_sign(row);
      payload.row_open_entries.push_back({
          .local_index = row,
          .orbital_label = right_occ[xmvb::to_size(row)],
          .value = sign * determinant,
      });
    }
    return payload;
  }

  if (payload.n_cols == payload.n_rows + 1) {
    const Eigen::VectorXd signed_deleted_col_minors =
        build_signed_open_minor_vector(
            overlap_block,
            overlap_resolver,
            subdeterminant_evaluations);
    for (int col = 0; col < payload.n_cols; ++col) {
      if (col < internal_col_begin) {
        continue;
      }
      const double determinant = parity_sign(col) * signed_deleted_col_minors(col);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      const double sign =
          (side == PartialSide::LeftFrontier)
              ? parity_sign(payload.n_rows + col)
              : parity_sign(col);
      payload.col_open_entries.push_back({
          .local_index = col,
          .orbital_label = left_occ[xmvb::to_size(col)],
          .value = sign * determinant,
      });
    }
  }

  return payload;
}

std::size_t pair_count(int count) {
  if (count <= 1) {
    return 0U;
  }
  return xmvb::to_size(count * (count - 1) / 2);
}

void reset_degree2_spin_payload(
    int n_rows,
    int n_cols,
    Degree2SpinPayload* payload) {
  // Resets one reusable local degree-2 payload while retaining already
  // allocated vector capacity between mask evaluations.
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  payload->n_rows = n_rows;
  payload->n_cols = n_cols;
  payload->overlap = 0.0;
  payload->row_open_entries.clear();
  payload->col_open_entries.clear();
  payload->cofactor_entries.clear();
  payload->row_pair_entries.clear();
  payload->col_pair_entries.clear();
  payload->row_pair_col_entries.clear();
  payload->row_col_pair_entries.clear();
  payload->second_cofactor_entries.clear();

  payload->row_open_entries.reserve(xmvb::to_size(n_rows));
  payload->col_open_entries.reserve(xmvb::to_size(n_cols));
  payload->cofactor_entries.reserve(
      xmvb::to_size(n_rows) *
      xmvb::to_size(n_cols));
  payload->row_pair_entries.reserve(pair_count(n_rows));
  payload->col_pair_entries.reserve(pair_count(n_cols));
  payload->row_pair_col_entries.reserve(
      pair_count(n_rows) *
      xmvb::to_size(n_cols));
  payload->row_col_pair_entries.reserve(
      xmvb::to_size(n_rows) *
      pair_count(n_cols));
  payload->second_cofactor_entries.reserve(
      pair_count(n_rows) * pair_count(n_cols));
}

class Degree2SpinPayloadBuilder {
 public:
  Degree2SpinPayloadBuilder(
      const std::vector<double>& overlap_storage,
      int n_orbitals,
      const DeterminantOverlapResolver& overlap_resolver,
      std::uint64_t* subdeterminant_evaluations)
      : overlap_storage_(overlap_storage),
        n_orbitals_(n_orbitals),
        overlap_resolver_(overlap_resolver),
        subdeterminant_evaluations_(subdeterminant_evaluations) {
    if (subdeterminant_evaluations_ == nullptr) {
      throw std::invalid_argument("subdeterminant_evaluations must not be null");
    }
  }

  const Degree2SpinPayload& build(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int selected_root_rows,
      int selected_root_cols,
      bool zero_selected_root_block,
      PartialSide side,
      bool restrict_deleted_to_internal_suffix = true,
      int max_deleted_rank = 2) {
    // Builds one exact local degree-2 payload while reusing all intermediate
    // matrices, vectors, and sparse entry buffers across masks. This is the
    // production hot-path version used by the one-leaf same-spin aggregator.
    if (max_deleted_rank < 0 || max_deleted_rank > 2) {
      throw std::invalid_argument("max_deleted_rank must be in [0, 2]");
    }
    reset_degree2_spin_payload(
        static_cast<int>(right_occ.size()),
        static_cast<int>(left_occ.size()),
        &payload_);

    fill_overlap_block(
        left_occ,
        right_occ,
        overlap_storage_,
        n_orbitals_,
        &overlap_block_);
    if (zero_selected_root_block && selected_root_rows > 0 && selected_root_cols > 0) {
      overlap_block_.topLeftCorner(selected_root_rows, selected_root_cols).setZero();
    }

    const int internal_row_begin =
        (zero_selected_root_block && restrict_deleted_to_internal_suffix)
            ? selected_root_rows
            : 0;
    const int internal_col_begin =
        (zero_selected_root_block && restrict_deleted_to_internal_suffix)
            ? selected_root_cols
            : 0;
    const bool left_frontier = side == PartialSide::LeftFrontier;

    if (payload_.n_rows == payload_.n_cols) {
      if (payload_.n_rows == 0) {
        payload_.overlap = 1.0;
        return payload_;
      }

      ++(*subdeterminant_evaluations_);
      const auto overlap_result = overlap_resolver_.resolve_matrix(overlap_block_);
      payload_.overlap = overlap_result.overlap_determinant;
      first_cofactor_ = xmvb::vb::calc_cofactor_1st(overlap_result);
      append_square_first_order_entries(
          left_occ,
          right_occ,
          internal_row_begin,
          internal_col_begin);
      if (max_deleted_rank <= 1) {
        return payload_;
      }
      if (overlap_result.nullity == 0) {
        append_square_second_order_entries(
            left_occ,
            right_occ,
            overlap_result,
            internal_row_begin,
            internal_col_begin);
        return payload_;
      }
    } else if (payload_.n_rows == payload_.n_cols + 1) {
      fill_signed_open_minor_vector(
          overlap_block_,
          overlap_resolver_,
          subdeterminant_evaluations_,
          &signed_open_minor_vector_);
      for (int row = internal_row_begin; row < payload_.n_rows; ++row) {
        const double determinant =
            parity_sign(row) * signed_open_minor_vector_(row);
        if (std::abs(determinant) <= 1.0e-15) {
          continue;
        }
        const double sign =
            (side == PartialSide::LeftFrontier)
                ? parity_sign(row + payload_.n_cols)
                : parity_sign(row);
        payload_.row_open_entries.push_back({
            .local_index = row,
            .orbital_label = right_occ[xmvb::to_size(row)],
            .value = sign * determinant,
        });
      }
      if (max_deleted_rank <= 1) {
        return payload_;
      }
      if (try_build_row_open_degree2(
              left_occ,
              right_occ,
              internal_row_begin,
              internal_col_begin,
              left_frontier)) {
        return payload_;
      }
    } else if (payload_.n_cols == payload_.n_rows + 1) {
      fill_signed_open_minor_vector(
          overlap_block_,
          overlap_resolver_,
          subdeterminant_evaluations_,
          &signed_open_minor_vector_);
      for (int col = internal_col_begin; col < payload_.n_cols; ++col) {
        const double determinant =
            parity_sign(col) * signed_open_minor_vector_(col);
        if (std::abs(determinant) <= 1.0e-15) {
          continue;
        }
        const double sign =
            (side == PartialSide::LeftFrontier)
                ? parity_sign(payload_.n_rows + col)
                : parity_sign(col);
        payload_.col_open_entries.push_back({
            .local_index = col,
            .orbital_label = left_occ[xmvb::to_size(col)],
            .value = sign * determinant,
        });
      }
      if (max_deleted_rank <= 1) {
        return payload_;
      }
      if (try_build_col_open_degree2(
              left_occ,
              right_occ,
              internal_row_begin,
              internal_col_begin,
              left_frontier)) {
        return payload_;
      }
    } else if (payload_.n_rows == payload_.n_cols + 2) {
      if (max_deleted_rank <= 1) {
        return payload_;
      }
      if (try_build_row_excess_two_degree2(
              left_occ,
              right_occ,
              internal_row_begin,
              left_frontier)) {
        return payload_;
      }
    } else if (payload_.n_cols == payload_.n_rows + 2) {
      if (max_deleted_rank <= 1) {
        return payload_;
      }
      if (try_build_col_excess_two_degree2(
              left_occ,
              right_occ,
              internal_col_begin,
              left_frontier)) {
        return payload_;
      }
    }

    build_degree2_sectors(
        left_occ,
        right_occ,
        internal_row_begin,
        internal_col_begin,
        zero_selected_root_block && restrict_deleted_to_internal_suffix,
        left_frontier);
    return payload_;
  }

 private:
  static int position_without_index(int original_index, int removed_index) {
    return (original_index < removed_index) ? original_index : (original_index - 1);
  }

  static int inverse_move_index(int destination_index, int from, int to) {
    if (from == to) {
      return destination_index;
    }
    if (from < to) {
      if (destination_index < from || destination_index > to) {
        return destination_index;
      }
      if (destination_index == to) {
        return from;
      }
      return destination_index + 1;
    }

    if (destination_index < to || destination_index > from) {
      return destination_index;
    }
    if (destination_index == to) {
      return from;
    }
    return destination_index - 1;
  }

  void append_square_first_order_entries(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int internal_row_begin,
      int internal_col_begin) {
    // Emits the exact square `(1,1)` entries from the already resolved local
    // cofactor matrix. The factorized degree-2 path reuses these first-order
    // values rather than rebuilding them per deleted minor.
    for (int col = internal_col_begin; col < payload_.n_cols; ++col) {
      for (int row = internal_row_begin; row < payload_.n_rows; ++row) {
        const double value = first_cofactor_(row, col);
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload_.cofactor_entries.push_back({
            .row_local_index = row,
            .col_local_index = col,
            .row_orbital_label = right_occ[xmvb::to_size(row)],
            .col_orbital_label = left_occ[xmvb::to_size(col)],
            .value = value,
        });
      }
    }
  }

  void append_square_second_order_entries(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      const xmvb::vb::DeterminantOverlapResult& overlap_result,
      int internal_row_begin,
      int internal_col_begin) {
    // For a nonsingular square block, every exact `(2,2)` deleted minor comes
    // from one inverse-based second-cofactor formula. This removes the old
    // per-entry deleted-minor determinant loop from the dominant square path.
    inverse_overlap_ =
        xmvb::vb::build_inverse_overlap_submatrix_from_result(overlap_result);
    for (int first_col = internal_col_begin; first_col + 1 < payload_.n_cols; ++first_col) {
      for (int second_col = first_col + 1; second_col < payload_.n_cols; ++second_col) {
        for (int first_row = internal_row_begin; first_row + 1 < payload_.n_rows; ++first_row) {
          for (int second_row = first_row + 1; second_row < payload_.n_rows; ++second_row) {
            const double value =
                payload_.overlap *
                (inverse_overlap_(first_col, first_row) *
                     inverse_overlap_(second_col, second_row) -
                 inverse_overlap_(second_col, first_row) *
                     inverse_overlap_(first_col, second_row));
            if (std::abs(value) <= 1.0e-15) {
              continue;
            }
            payload_.second_cofactor_entries.push_back({
                .first_row_local_index = first_row,
                .second_row_local_index = second_row,
                .first_col_local_index = first_col,
                .second_col_local_index = second_col,
                .first_row_orbital_label =
                    right_occ[xmvb::to_size(first_row)],
                .second_row_orbital_label =
                    right_occ[xmvb::to_size(second_row)],
                .first_col_orbital_label =
                    left_occ[xmvb::to_size(first_col)],
                .second_col_orbital_label =
                    left_occ[xmvb::to_size(second_col)],
                .value = value,
            });
          }
        }
      }
    }
  }

  bool try_build_row_open_degree2(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int internal_row_begin,
      int internal_col_begin,
      bool left_frontier) {
    // Full-row-rank `(k+1) x k` blocks can recover every `(2,1)` sector from
    // one anchor square factorization plus rank-1 row updates. This makes the
    // row-open degree-2 builder output-size dominated.
    if (signed_open_minor_vector_.size() != payload_.n_rows) {
      return false;
    }
    if (payload_.n_cols == 0) {
      return true;
    }
    Eigen::Index anchor_row_eigen = 0;
    signed_open_minor_vector_.cwiseAbs().maxCoeff(&anchor_row_eigen);
    const int anchor_row = static_cast<int>(anchor_row_eigen);
    if (std::abs(signed_open_minor_vector_(anchor_row)) <= 1.0e-15) {
      return false;
    }

    fill_deleted_minor_matrix(overlap_block_, {anchor_row}, {}, &square_minor_);
    ++(*subdeterminant_evaluations_);
    const auto anchor_result = overlap_resolver_.resolve_matrix(square_minor_);
    if (anchor_result.nullity != 0) {
      return false;
    }
    inverse_overlap_ =
        xmvb::vb::build_inverse_overlap_submatrix_from_result(anchor_result);
    first_cofactor_ =
        anchor_result.overlap_determinant * inverse_overlap_.transpose();

    const double threshold = overlap_resolver_.linear_dependence_threshold();
    for (int deleted_first_row = internal_row_begin;
         deleted_first_row < payload_.n_rows;
         ++deleted_first_row) {
      int from = 0;
      int to = 0;
      double permutation_sign = 1.0;
      if (deleted_first_row == anchor_row) {
        first_cofactor_ =
            anchor_result.overlap_determinant * inverse_overlap_.transpose();
      } else {
        from = position_without_index(deleted_first_row, anchor_row);
        to = position_without_index(anchor_row, deleted_first_row);
        const Eigen::RowVectorXd row_delta =
            overlap_block_.row(anchor_row) - overlap_block_.row(deleted_first_row);
        row_update_workspace_ = row_delta * inverse_overlap_;
        const double denominator = 1.0 + row_update_workspace_(from);
        if (std::abs(denominator) <= threshold) {
          return false;
        }
        updated_inverse_ =
            inverse_overlap_ -
            (inverse_overlap_.col(from) * row_update_workspace_) / denominator;
        const double updated_determinant =
            anchor_result.overlap_determinant * denominator;
        first_cofactor_ = updated_determinant * updated_inverse_.transpose();
        permutation_sign = parity_sign(std::abs(to - from));
      }

      for (int deleted_second_row = deleted_first_row + 1;
           deleted_second_row < payload_.n_rows;
           ++deleted_second_row) {
        const int row_index_in_deleted_square = deleted_second_row - 1;
        const int source_row =
            (deleted_first_row == anchor_row)
                ? row_index_in_deleted_square
                : inverse_move_index(row_index_in_deleted_square, from, to);
        for (int deleted_col = internal_col_begin;
             deleted_col < payload_.n_cols;
             ++deleted_col) {
          const double cofactor_value =
              permutation_sign * first_cofactor_(source_row, deleted_col);
          if (std::abs(cofactor_value) <= 1.0e-15) {
            continue;
          }
          const double determinant =
              parity_sign(row_index_in_deleted_square + deleted_col) *
              cofactor_value;
          const double value =
              deleted_minor_sign(
                  payload_.n_rows,
                  payload_.n_cols,
                  {deleted_first_row, deleted_second_row},
                  {deleted_col},
                  left_frontier) *
              determinant;
          if (std::abs(value) <= 1.0e-15) {
            continue;
          }
          payload_.row_pair_col_entries.push_back({
              .first_row_local_index = deleted_first_row,
              .second_row_local_index = deleted_second_row,
              .col_local_index = deleted_col,
              .first_row_orbital_label =
                  right_occ[xmvb::to_size(deleted_first_row)],
              .second_row_orbital_label =
                  right_occ[xmvb::to_size(deleted_second_row)],
              .col_orbital_label =
                  left_occ[xmvb::to_size(deleted_col)],
              .value = value,
          });
        }
      }
    }
    return true;
  }

  bool try_build_col_open_degree2(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int internal_row_begin,
      int internal_col_begin,
      bool left_frontier) {
    // Full-column-rank `k x (k+1)` blocks are handled by the column analogue
    // of the row-open update path above. One anchor square factorization feeds
    // all exact `(1,2)` sectors.
    if (signed_open_minor_vector_.size() != payload_.n_cols) {
      return false;
    }
    if (payload_.n_rows == 0) {
      return true;
    }
    Eigen::Index anchor_col_eigen = 0;
    signed_open_minor_vector_.cwiseAbs().maxCoeff(&anchor_col_eigen);
    const int anchor_col = static_cast<int>(anchor_col_eigen);
    if (std::abs(signed_open_minor_vector_(anchor_col)) <= 1.0e-15) {
      return false;
    }

    fill_deleted_minor_matrix(overlap_block_, {}, {anchor_col}, &square_minor_);
    ++(*subdeterminant_evaluations_);
    const auto anchor_result = overlap_resolver_.resolve_matrix(square_minor_);
    if (anchor_result.nullity != 0) {
      return false;
    }
    inverse_overlap_ =
        xmvb::vb::build_inverse_overlap_submatrix_from_result(anchor_result);
    first_cofactor_ =
        anchor_result.overlap_determinant * inverse_overlap_.transpose();

    const double threshold = overlap_resolver_.linear_dependence_threshold();
    for (int deleted_first_col = internal_col_begin;
         deleted_first_col < payload_.n_cols;
         ++deleted_first_col) {
      int from = 0;
      int to = 0;
      double permutation_sign = 1.0;
      if (deleted_first_col == anchor_col) {
        first_cofactor_ =
            anchor_result.overlap_determinant * inverse_overlap_.transpose();
      } else {
        from = position_without_index(deleted_first_col, anchor_col);
        to = position_without_index(anchor_col, deleted_first_col);
        const Eigen::VectorXd column_delta =
            overlap_block_.col(anchor_col) - overlap_block_.col(deleted_first_col);
        column_update_workspace_ = inverse_overlap_ * column_delta;
        const double denominator = 1.0 + column_update_workspace_(from);
        if (std::abs(denominator) <= threshold) {
          return false;
        }
        updated_inverse_ =
            inverse_overlap_ -
            (column_update_workspace_ * inverse_overlap_.row(from)) / denominator;
        const double updated_determinant =
            anchor_result.overlap_determinant * denominator;
        first_cofactor_ = updated_determinant * updated_inverse_.transpose();
        permutation_sign = parity_sign(std::abs(to - from));
      }

      for (int deleted_second_col = deleted_first_col + 1;
           deleted_second_col < payload_.n_cols;
           ++deleted_second_col) {
        const int col_index_in_deleted_square = deleted_second_col - 1;
        const int source_col =
            (deleted_first_col == anchor_col)
                ? col_index_in_deleted_square
                : inverse_move_index(col_index_in_deleted_square, from, to);
        for (int deleted_row = internal_row_begin;
             deleted_row < payload_.n_rows;
             ++deleted_row) {
          const double cofactor_value =
              permutation_sign * first_cofactor_(deleted_row, source_col);
          if (std::abs(cofactor_value) <= 1.0e-15) {
            continue;
          }
          const double determinant =
              parity_sign(deleted_row + col_index_in_deleted_square) *
              cofactor_value;
          const double value =
              deleted_minor_sign(
                  payload_.n_rows,
                  payload_.n_cols,
                  {deleted_row},
                  {deleted_first_col, deleted_second_col},
                  left_frontier) *
              determinant;
          if (std::abs(value) <= 1.0e-15) {
            continue;
          }
          payload_.row_col_pair_entries.push_back({
              .row_local_index = deleted_row,
              .first_col_local_index = deleted_first_col,
              .second_col_local_index = deleted_second_col,
              .row_orbital_label =
                  right_occ[xmvb::to_size(deleted_row)],
              .first_col_orbital_label =
                  left_occ[xmvb::to_size(deleted_first_col)],
              .second_col_orbital_label =
                  left_occ[xmvb::to_size(deleted_second_col)],
              .value = value,
          });
        }
      }
    }
    return true;
  }

  bool try_build_row_excess_two_degree2(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int internal_row_begin,
      bool left_frontier) {
    // A full-rank `(k+2) x k` block carries all exact `(2,0)` deleted minors
    // in the two-dimensional left null space. One anchor minor fixes the
    // global normalization, then every payload entry is a `2 x 2` kernel
    // minor.
    Eigen::FullPivLU<Eigen::MatrixXd> rectangular_lu(overlap_block_.transpose());
    rectangular_lu.setThreshold(overlap_resolver_.linear_dependence_threshold());
    if (rectangular_lu.rank() != payload_.n_cols ||
        rectangular_lu.dimensionOfKernel() != 2) {
      return false;
    }
    kernel_basis_ = rectangular_lu.kernel();
    const auto anchor = find_anchor_kernel_pair();
    if (anchor.first < 0) {
      return false;
    }

    fill_deleted_minor_matrix(overlap_block_, {anchor.first, anchor.second}, {}, &square_minor_);
    const double anchor_value =
        deleted_minor_sign(
            payload_.n_rows,
            payload_.n_cols,
            {anchor.first, anchor.second},
            {},
            left_frontier) *
        determinant_of_dense_matrix(
            square_minor_,
            overlap_resolver_,
            subdeterminant_evaluations_);
    if (std::abs(anchor_value) <= 1.0e-15) {
      return false;
    }
    const double anchor_kernel_minor =
        kernel_basis_(anchor.first, 0) * kernel_basis_(anchor.second, 1) -
        kernel_basis_(anchor.first, 1) * kernel_basis_(anchor.second, 0);
    if (std::abs(anchor_kernel_minor) <= 1.0e-15) {
      return false;
    }

    for (int first_row = internal_row_begin; first_row + 1 < payload_.n_rows; ++first_row) {
      for (int second_row = first_row + 1; second_row < payload_.n_rows; ++second_row) {
        const double kernel_minor =
            kernel_basis_(first_row, 0) * kernel_basis_(second_row, 1) -
            kernel_basis_(first_row, 1) * kernel_basis_(second_row, 0);
        const double value = anchor_value * kernel_minor / anchor_kernel_minor;
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload_.row_pair_entries.push_back({
            .first_row_local_index = first_row,
            .second_row_local_index = second_row,
            .first_row_orbital_label =
                right_occ[xmvb::to_size(first_row)],
            .second_row_orbital_label =
                right_occ[xmvb::to_size(second_row)],
            .value = value,
        });
      }
    }
    return true;
  }

  bool try_build_col_excess_two_degree2(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int internal_col_begin,
      bool left_frontier) {
    // The `(0,2)` sector of a full-rank `k x (k+2)` block is the right-null
    // analogue of the row-excess-two case above.
    Eigen::FullPivLU<Eigen::MatrixXd> rectangular_lu(overlap_block_);
    rectangular_lu.setThreshold(overlap_resolver_.linear_dependence_threshold());
    if (rectangular_lu.rank() != payload_.n_rows ||
        rectangular_lu.dimensionOfKernel() != 2) {
      return false;
    }
    kernel_basis_ = rectangular_lu.kernel();
    const auto anchor = find_anchor_kernel_pair();
    if (anchor.first < 0) {
      return false;
    }

    fill_deleted_minor_matrix(overlap_block_, {}, {anchor.first, anchor.second}, &square_minor_);
    const double anchor_value =
        deleted_minor_sign(
            payload_.n_rows,
            payload_.n_cols,
            {},
            {anchor.first, anchor.second},
            left_frontier) *
        determinant_of_dense_matrix(
            square_minor_,
            overlap_resolver_,
            subdeterminant_evaluations_);
    if (std::abs(anchor_value) <= 1.0e-15) {
      return false;
    }
    const double anchor_kernel_minor =
        kernel_basis_(anchor.first, 0) * kernel_basis_(anchor.second, 1) -
        kernel_basis_(anchor.first, 1) * kernel_basis_(anchor.second, 0);
    if (std::abs(anchor_kernel_minor) <= 1.0e-15) {
      return false;
    }

    for (int first_col = internal_col_begin; first_col + 1 < payload_.n_cols; ++first_col) {
      for (int second_col = first_col + 1; second_col < payload_.n_cols; ++second_col) {
        const double kernel_minor =
            kernel_basis_(first_col, 0) * kernel_basis_(second_col, 1) -
            kernel_basis_(first_col, 1) * kernel_basis_(second_col, 0);
        const double value = anchor_value * kernel_minor / anchor_kernel_minor;
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload_.col_pair_entries.push_back({
            .first_col_local_index = first_col,
            .second_col_local_index = second_col,
            .first_col_orbital_label =
                left_occ[xmvb::to_size(first_col)],
            .second_col_orbital_label =
                left_occ[xmvb::to_size(second_col)],
            .value = value,
        });
      }
    }
    return true;
  }

  std::pair<int, int> find_anchor_kernel_pair() const {
    std::pair<int, int> anchor{-1, -1};
    double best_abs_minor = 0.0;
    for (int first = 0; first + 1 < kernel_basis_.rows(); ++first) {
      for (int second = first + 1; second < kernel_basis_.rows(); ++second) {
        const double minor =
            kernel_basis_(first, 0) * kernel_basis_(second, 1) -
            kernel_basis_(first, 1) * kernel_basis_(second, 0);
        if (std::abs(minor) <= best_abs_minor) {
          continue;
        }
        best_abs_minor = std::abs(minor);
        anchor = {first, second};
      }
    }
    return anchor;
  }

  void build_degree2_sectors(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int internal_row_begin,
      int internal_col_begin,
      bool zero_selected_root_block,
      bool left_frontier) {
    // Enumerates the exact degree-2 deleted-minor sectors that feed the
    // explicit one-mask same-spin merge. The reusable `deleted_minor_`
    // workspace avoids allocating a fresh dense minor for every local sector.
    for (int deleted_row_rank = 0; deleted_row_rank <= 2; ++deleted_row_rank) {
      if (deleted_row_rank > payload_.n_rows) {
        continue;
      }
      for_each_index_subset(
          0,
          payload_.n_rows,
          deleted_row_rank,
          [&](const std::vector<int>& deleted_rows) {
            if (zero_selected_root_block &&
                !deleted_indices_in_suffix(deleted_rows, internal_row_begin)) {
              return;
            }
            for (int deleted_col_rank = 0; deleted_col_rank <= 2; ++deleted_col_rank) {
              if (deleted_col_rank > payload_.n_cols) {
                continue;
              }
              if (payload_.n_rows - deleted_row_rank !=
                  payload_.n_cols - deleted_col_rank) {
                continue;
              }
              const bool is_degree2_sector =
                  (deleted_row_rank == 2) || (deleted_col_rank == 2);
              if (!is_degree2_sector) {
                continue;
              }
              for_each_index_subset(
                  0,
                  payload_.n_cols,
                  deleted_col_rank,
                  [&](const std::vector<int>& deleted_cols) {
                    if (zero_selected_root_block &&
                        !deleted_indices_in_suffix(deleted_cols, internal_col_begin)) {
                      return;
                    }

                    fill_deleted_minor_matrix(
                        overlap_block_,
                        deleted_rows,
                        deleted_cols,
                        &deleted_minor_);
                    const double determinant = determinant_of_dense_matrix(
                        deleted_minor_,
                        overlap_resolver_,
                        subdeterminant_evaluations_);
                    if (std::abs(determinant) <= 1.0e-15) {
                      return;
                    }

                    const double value =
                        deleted_minor_sign(
                            payload_.n_rows,
                            payload_.n_cols,
                            deleted_rows,
                            deleted_cols,
                            left_frontier) *
                        determinant;
                    if (std::abs(value) <= 1.0e-15) {
                      return;
                    }

                    append_sector_entry(
                        left_occ,
                        right_occ,
                        deleted_row_rank,
                        deleted_rows,
                        deleted_col_rank,
                        deleted_cols,
                        value);
                  });
            }
          });
    }
  }

  void append_sector_entry(
      const std::vector<int>& left_occ,
      const std::vector<int>& right_occ,
      int deleted_row_rank,
      const std::vector<int>& deleted_rows,
      int deleted_col_rank,
      const std::vector<int>& deleted_cols,
      double value) {
    // Converts one local deleted-minor into the typed sparse representation
    // expected by the explicit same-spin case split.
    if (deleted_row_rank == 2 && deleted_col_rank == 0) {
      payload_.row_pair_entries.push_back({
          .first_row_local_index = deleted_rows[0],
          .second_row_local_index = deleted_rows[1],
          .first_row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[0])],
          .second_row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[1])],
          .value = value,
      });
      return;
    }
    if (deleted_row_rank == 0 && deleted_col_rank == 2) {
      payload_.col_pair_entries.push_back({
          .first_col_local_index = deleted_cols[0],
          .second_col_local_index = deleted_cols[1],
          .first_col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[0])],
          .second_col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[1])],
          .value = value,
      });
      return;
    }
    if (deleted_row_rank == 2 && deleted_col_rank == 1) {
      payload_.row_pair_col_entries.push_back({
          .first_row_local_index = deleted_rows[0],
          .second_row_local_index = deleted_rows[1],
          .col_local_index = deleted_cols[0],
          .first_row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[0])],
          .second_row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[1])],
          .col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[0])],
          .value = value,
      });
      return;
    }
    if (deleted_row_rank == 1 && deleted_col_rank == 2) {
      payload_.row_col_pair_entries.push_back({
          .row_local_index = deleted_rows[0],
          .first_col_local_index = deleted_cols[0],
          .second_col_local_index = deleted_cols[1],
          .row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[0])],
          .first_col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[0])],
          .second_col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[1])],
          .value = value,
      });
      return;
    }
    if (deleted_row_rank == 2 && deleted_col_rank == 2) {
      payload_.second_cofactor_entries.push_back({
          .first_row_local_index = deleted_rows[0],
          .second_row_local_index = deleted_rows[1],
          .first_col_local_index = deleted_cols[0],
          .second_col_local_index = deleted_cols[1],
          .first_row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[0])],
          .second_row_orbital_label =
              right_occ[xmvb::to_size(deleted_rows[1])],
          .first_col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[0])],
          .second_col_orbital_label =
              left_occ[xmvb::to_size(deleted_cols[1])],
          .value = value,
      });
    }
  }

  const std::vector<double>& overlap_storage_;
  int n_orbitals_ = 0;
  const DeterminantOverlapResolver& overlap_resolver_;
  std::uint64_t* subdeterminant_evaluations_ = nullptr;
  Eigen::MatrixXd overlap_block_;
  Eigen::MatrixXd first_cofactor_;
  Eigen::MatrixXd inverse_overlap_;
  Eigen::MatrixXd updated_inverse_;
  Eigen::MatrixXd square_minor_;
  Eigen::MatrixXd deleted_minor_;
  Eigen::MatrixXd kernel_basis_;
  Eigen::RowVectorXd row_update_workspace_;
  Eigen::VectorXd column_update_workspace_;
  Eigen::VectorXd signed_open_minor_vector_;
  Degree2SpinPayload payload_;
};

Degree2SpinPayload build_degree2_spin_payload(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int selected_root_rows,
    int selected_root_cols,
    bool zero_selected_root_block,
    PartialSide side,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // Compatibility wrapper for non-hot callers. The one-leaf production path
  // uses `Degree2SpinPayloadBuilder` directly so the reusable state survives
  // across mask iterations.
  Degree2SpinPayloadBuilder builder(
      overlap_storage,
      n_orbitals,
      overlap_resolver,
      subdeterminant_evaluations);
  return builder.build(
      left_occ,
      right_occ,
      selected_root_rows,
      selected_root_cols,
      zero_selected_root_block,
      side);
}

SpinDeletionPayload spin_payload_from_partial(const PartialSpinPayload& payload) {
  SpinDeletionPayload spin_payload;
  spin_payload.row_count = payload.n_rows;
  spin_payload.col_count = payload.n_cols;
  if (is_square_payload(payload) && std::abs(payload.closed_overlap) > 1.0e-15) {
    spin_payload.sectors[make_spin_key()] += payload.closed_overlap;
  }
  for (const auto& entry : payload.row_open_entries) {
    spin_payload.sectors[make_spin_key({entry.orbital_label}, {})] += entry.value;
  }
  for (const auto& entry : payload.col_open_entries) {
    spin_payload.sectors[make_spin_key({}, {entry.orbital_label})] += entry.value;
  }
  for (const auto& entry : payload.cofactor_entries) {
    spin_payload.sectors[make_spin_key(
        {entry.row_orbital_label},
        {entry.col_orbital_label})] += entry.value;
  }
  cleanup_spin_payload(&spin_payload);
  return spin_payload;
}

void append_canonical_second_cofactor_entry(
    int row_first_orbital,
    int row_second_orbital,
    int col_first_orbital,
    int col_second_orbital,
    double value,
    std::vector<SpinMaskCofactorAggregate::SecondCofactorEntry>* entries) {
  if (entries == nullptr) {
    throw std::invalid_argument("entries must not be null");
  }
  if (std::abs(value) <= 1.0e-15) {
    return;
  }
  double canonical_value = value;
  if (row_first_orbital > row_second_orbital) {
    std::swap(row_first_orbital, row_second_orbital);
    canonical_value = -canonical_value;
  }
  if (col_first_orbital > col_second_orbital) {
    std::swap(col_first_orbital, col_second_orbital);
    canonical_value = -canonical_value;
  }
  if (std::abs(canonical_value) <= 1.0e-15) {
    return;
  }

  entries->push_back({
      .row_first_orbital = row_first_orbital,
      .row_second_orbital = row_second_orbital,
      .col_first_orbital = col_first_orbital,
      .col_second_orbital = col_second_orbital,
      .value = canonical_value,
  });
}

void compress_second_cofactor_entries(
    std::vector<SpinMaskCofactorAggregate::SecondCofactorEntry>* entries) {
  // Canonical `(2,2)` entries are accumulated mask-by-mask. Sorting and
  // summing identical orbital-label quadruples once at the end avoids repeated
  // ordered-map updates inside the hot mask loop while still producing the
  // exact sparse global second-cofactor payload needed by the final
  // same-spin contraction.
  if (entries == nullptr) {
    throw std::invalid_argument("entries must not be null");
  }
  auto& values = *entries;
  std::sort(
      values.begin(),
      values.end(),
      [](const auto& left, const auto& right) {
        return std::tie(
                   left.row_first_orbital,
                   left.row_second_orbital,
                   left.col_first_orbital,
                   left.col_second_orbital) <
            std::tie(
                   right.row_first_orbital,
                   right.row_second_orbital,
                   right.col_first_orbital,
                   right.col_second_orbital);
      });

  std::size_t write_index = 0U;
  for (const auto& entry : values) {
    if (std::abs(entry.value) <= 1.0e-15) {
      continue;
    }
    if (write_index == 0U ||
        values[write_index - 1].row_first_orbital != entry.row_first_orbital ||
        values[write_index - 1].row_second_orbital != entry.row_second_orbital ||
        values[write_index - 1].col_first_orbital != entry.col_first_orbital ||
        values[write_index - 1].col_second_orbital != entry.col_second_orbital) {
      values[write_index++] = entry;
      continue;
    }
    values[write_index - 1].value += entry.value;
  }
  values.resize(write_index);
  values.erase(
      std::remove_if(
          values.begin(),
          values.end(),
          [](const auto& entry) { return std::abs(entry.value) <= 1.0e-15; }),
      values.end());
}

struct RawDeletedMinorTerm {
  int deleted_row_rank = 0;
  int deleted_col_rank = 0;
  std::array<int, 2> row_local_indices{{-1, -1}};
  std::array<int, 2> col_local_indices{{-1, -1}};
  std::array<int, 2> row_block_indices{{-1, -1}};
  std::array<int, 2> col_block_indices{{-1, -1}};
  std::array<int, 2> row_orbital_labels{{-1, -1}};
  std::array<int, 2> col_orbital_labels{{-1, -1}};
  double determinant = 0.0;
};

using RawDeletedMinorBuckets =
    std::array<std::array<std::vector<RawDeletedMinorTerm>, 3>, 3>;

int deleted_index_sum(
    const std::array<int, 2>& deleted_indices,
    int deleted_rank) {
  int sum = 0;
  for (int index = 0; index < deleted_rank; ++index) {
    sum += deleted_indices[xmvb::to_size(index)];
  }
  return sum;
}

double deleted_minor_sign(
    int n_rows,
    int n_cols,
    const std::array<int, 2>& deleted_rows,
    int deleted_row_rank,
    const std::array<int, 2>& deleted_cols,
    int deleted_col_rank,
    bool left_frontier) {
  int parity =
      deleted_index_sum(deleted_rows, deleted_row_rank) +
      deleted_index_sum(deleted_cols, deleted_col_rank);
  if (left_frontier) {
    parity += std::min(n_rows, n_cols) * (n_rows - n_cols);
    if ((deleted_row_rank == 2 && deleted_col_rank == 1) ||
        (deleted_row_rank == 1 && deleted_col_rank == 2)) {
      parity += 1;
    }
  }
  return parity_sign(parity);
}

void append_raw_deleted_minor_term(
    int deleted_row_rank,
    int deleted_col_rank,
    const std::array<int, 2>& row_local_indices,
    const std::array<int, 2>& col_local_indices,
    const std::array<int, 2>& row_block_indices,
    const std::array<int, 2>& col_block_indices,
    const std::array<int, 2>& row_orbital_labels,
    const std::array<int, 2>& col_orbital_labels,
    double determinant,
    RawDeletedMinorBuckets* buckets) {
  if (buckets == nullptr) {
    throw std::invalid_argument("raw deleted-minor buckets must not be null");
  }
  if (deleted_row_rank < 0 || deleted_row_rank > 2 ||
      deleted_col_rank < 0 || deleted_col_rank > 2) {
    throw std::invalid_argument("deleted-minor rank is out of range");
  }
  if (std::abs(determinant) <= 1.0e-15) {
    return;
  }
  (*buckets)[xmvb::to_size(deleted_row_rank)]
            [xmvb::to_size(deleted_col_rank)]
      .push_back(RawDeletedMinorTerm{
          .deleted_row_rank = deleted_row_rank,
          .deleted_col_rank = deleted_col_rank,
          .row_local_indices = row_local_indices,
          .col_local_indices = col_local_indices,
          .row_block_indices = row_block_indices,
          .col_block_indices = col_block_indices,
          .row_orbital_labels = row_orbital_labels,
          .col_orbital_labels = col_orbital_labels,
          .determinant = determinant,
      });
}

RawDeletedMinorBuckets build_raw_deleted_minor_buckets(
    const Degree2SpinPayload& payload,
    bool left_frontier,
    int root_row_count,
    int root_col_count,
    int selected_root_row_count,
    int selected_root_col_count,
    const std::vector<int>& selected_root_row_positions,
    const std::vector<int>& selected_root_col_positions,
    const std::vector<int>& root_remainder_row_positions,
    const std::vector<int>& root_remainder_col_positions,
    bool frontier_side) {
  // Reinterprets the exported local degree-0/1/2 payload in the cleanest
  // possible exact form: raw deleted-minor determinants indexed by deleted
  // local row/column sets. The global one-leaf merge can then rebuild the
  // full cofactor / second-cofactor objects directly from the determinant
  // definition, instead of relying on a hand-written case-sign algebra.
  RawDeletedMinorBuckets buckets;
  const auto recover_raw_minor =
      [&](const std::array<int, 2>& deleted_rows,
          int deleted_row_rank,
          const std::array<int, 2>& deleted_cols,
          int deleted_col_rank,
          double signed_value) {
        return deleted_minor_sign(
                   payload.n_rows,
                   payload.n_cols,
                   deleted_rows,
                   deleted_row_rank,
                   deleted_cols,
                   deleted_col_rank,
                   left_frontier) *
            signed_value;
      };
  const auto row_block_index =
      [&](int local_index) {
        if (local_index < 0) {
          return -1;
        }
        if (frontier_side) {
          if (local_index < selected_root_row_count) {
            return selected_root_row_positions[xmvb::to_size(local_index)];
          }
          return root_row_count + (local_index - selected_root_row_count);
        }
        return root_remainder_row_positions[xmvb::to_size(local_index)];
      };
  const auto col_block_index =
      [&](int local_index) {
        if (local_index < 0) {
          return -1;
        }
        if (frontier_side) {
          if (local_index < selected_root_col_count) {
            return selected_root_col_positions[xmvb::to_size(local_index)];
          }
          return root_col_count + (local_index - selected_root_col_count);
        }
        return root_remainder_col_positions[xmvb::to_size(local_index)];
      };

  if (payload.n_rows == payload.n_cols && std::abs(payload.overlap) > 1.0e-15) {
    append_raw_deleted_minor_term(
        0,
        0,
        {{-1, -1}},
        {{-1, -1}},
        {{-1, -1}},
        {{-1, -1}},
        {{-1, -1}},
        {{-1, -1}},
        payload.overlap,
        &buckets);
  }

  for (const auto& entry : payload.row_open_entries) {
    append_raw_deleted_minor_term(
        1,
        0,
        {{entry.local_index, -1}},
        {{-1, -1}},
        {{row_block_index(entry.local_index), -1}},
        {{-1, -1}},
        {{entry.orbital_label, -1}},
        {{-1, -1}},
        recover_raw_minor(
            {{entry.local_index, -1}},
            1,
            {{-1, -1}},
            0,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.col_open_entries) {
    append_raw_deleted_minor_term(
        0,
        1,
        {{-1, -1}},
        {{entry.local_index, -1}},
        {{-1, -1}},
        {{col_block_index(entry.local_index), -1}},
        {{-1, -1}},
        {{entry.orbital_label, -1}},
        recover_raw_minor(
            {{-1, -1}},
            0,
            {{entry.local_index, -1}},
            1,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.cofactor_entries) {
    append_raw_deleted_minor_term(
        1,
        1,
        {{entry.row_local_index, -1}},
        {{entry.col_local_index, -1}},
        {{row_block_index(entry.row_local_index), -1}},
        {{col_block_index(entry.col_local_index), -1}},
        {{entry.row_orbital_label, -1}},
        {{entry.col_orbital_label, -1}},
        recover_raw_minor(
            {{entry.row_local_index, -1}},
            1,
            {{entry.col_local_index, -1}},
            1,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.row_pair_entries) {
    append_raw_deleted_minor_term(
        2,
        0,
        {{entry.first_row_local_index, entry.second_row_local_index}},
        {{-1, -1}},
        {{row_block_index(entry.first_row_local_index),
          row_block_index(entry.second_row_local_index)}},
        {{-1, -1}},
        {{entry.first_row_orbital_label, entry.second_row_orbital_label}},
        {{-1, -1}},
        recover_raw_minor(
            {{entry.first_row_local_index, entry.second_row_local_index}},
            2,
            {{-1, -1}},
            0,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.col_pair_entries) {
    append_raw_deleted_minor_term(
        0,
        2,
        {{-1, -1}},
        {{entry.first_col_local_index, entry.second_col_local_index}},
        {{-1, -1}},
        {{col_block_index(entry.first_col_local_index),
          col_block_index(entry.second_col_local_index)}},
        {{-1, -1}},
        {{entry.first_col_orbital_label, entry.second_col_orbital_label}},
        recover_raw_minor(
            {{-1, -1}},
            0,
            {{entry.first_col_local_index, entry.second_col_local_index}},
            2,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.row_pair_col_entries) {
    append_raw_deleted_minor_term(
        2,
        1,
        {{entry.first_row_local_index, entry.second_row_local_index}},
        {{entry.col_local_index, -1}},
        {{row_block_index(entry.first_row_local_index),
          row_block_index(entry.second_row_local_index)}},
        {{col_block_index(entry.col_local_index), -1}},
        {{entry.first_row_orbital_label, entry.second_row_orbital_label}},
        {{entry.col_orbital_label, -1}},
        recover_raw_minor(
            {{entry.first_row_local_index, entry.second_row_local_index}},
            2,
            {{entry.col_local_index, -1}},
            1,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.row_col_pair_entries) {
    append_raw_deleted_minor_term(
        1,
        2,
        {{entry.row_local_index, -1}},
        {{entry.first_col_local_index, entry.second_col_local_index}},
        {{row_block_index(entry.row_local_index), -1}},
        {{col_block_index(entry.first_col_local_index),
          col_block_index(entry.second_col_local_index)}},
        {{entry.row_orbital_label, -1}},
        {{entry.first_col_orbital_label, entry.second_col_orbital_label}},
        recover_raw_minor(
            {{entry.row_local_index, -1}},
            1,
            {{entry.first_col_local_index, entry.second_col_local_index}},
            2,
            entry.value),
        &buckets);
  }
  for (const auto& entry : payload.second_cofactor_entries) {
    append_raw_deleted_minor_term(
        2,
        2,
        {{entry.first_row_local_index, entry.second_row_local_index}},
        {{entry.first_col_local_index, entry.second_col_local_index}},
        {{row_block_index(entry.first_row_local_index),
          row_block_index(entry.second_row_local_index)}},
        {{col_block_index(entry.first_col_local_index),
          col_block_index(entry.second_col_local_index)}},
        {{entry.first_row_orbital_label, entry.second_row_orbital_label}},
        {{entry.first_col_orbital_label, entry.second_col_orbital_label}},
        recover_raw_minor(
            {{entry.first_row_local_index, entry.second_row_local_index}},
            2,
            {{entry.first_col_local_index, entry.second_col_local_index}},
            2,
            entry.value),
        &buckets);
  }
  return buckets;
}

bool block_index_is_deleted(
    int block_index,
    const std::array<int, 2>& deleted_block_indices,
    int deleted_rank) {
  for (int deleted_index = 0; deleted_index < deleted_rank; ++deleted_index) {
    if (deleted_block_indices[xmvb::to_size(deleted_index)] == block_index) {
      return true;
    }
  }
  return false;
}

int block_order_parity_with_deleted(
    int n_root_occ,
    std::uint32_t selected_mask,
    int leaf_occ_size,
    const std::array<int, 2>& deleted_block_indices,
    int deleted_rank) {
  std::vector<int> block_order;
  block_order.reserve(
      xmvb::to_size(n_root_occ + leaf_occ_size - deleted_rank));
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (static_cast<std::uint32_t>(1U) << root_position)) == 0U) {
      continue;
    }
    if (block_index_is_deleted(root_position, deleted_block_indices, deleted_rank)) {
      continue;
    }
    block_order.push_back(root_position);
  }
  for (int leaf_position = 0; leaf_position < leaf_occ_size; ++leaf_position) {
    const int block_index = n_root_occ + leaf_position;
    if (block_index_is_deleted(block_index, deleted_block_indices, deleted_rank)) {
      continue;
    }
    block_order.push_back(block_index);
  }
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (static_cast<std::uint32_t>(1U) << root_position)) != 0U) {
      continue;
    }
    if (block_index_is_deleted(root_position, deleted_block_indices, deleted_rank)) {
      continue;
    }
    block_order.push_back(root_position);
  }

  int parity = 0;
  for (std::size_t left_index = 0; left_index + 1 < block_order.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < block_order.size();
         ++right_index) {
      if (block_order[left_index] > block_order[right_index]) {
        parity ^= 1;
      }
    }
  }
  return parity;
}

std::array<int, 2> merge_deleted_indices_in_original_order(
    const std::array<int, 2>& frontier_block_indices,
    int frontier_rank,
    const std::array<int, 2>& root_block_indices,
    int root_rank) {
  std::array<int, 2> merged{{-1, -1}};
  int write_index = 0;
  int frontier_index = 0;
  int root_index = 0;
  while (frontier_index < frontier_rank || root_index < root_rank) {
    if (root_index >= root_rank ||
        (frontier_index < frontier_rank &&
         frontier_block_indices[xmvb::to_size(frontier_index)] <
             root_block_indices[xmvb::to_size(root_index)])) {
      merged[xmvb::to_size(write_index++)] =
          frontier_block_indices[xmvb::to_size(frontier_index++)];
      continue;
    }
    merged[xmvb::to_size(write_index++)] =
        root_block_indices[xmvb::to_size(root_index++)];
  }
  return merged;
}

template <typename LabelArray>
std::array<int, 2> merge_deleted_labels_in_original_order(
    const std::array<int, 2>& frontier_block_indices,
    const LabelArray& frontier_labels,
    int frontier_rank,
    const std::array<int, 2>& root_block_indices,
    const LabelArray& root_labels,
    int root_rank) {
  std::array<int, 2> merged{{-1, -1}};
  int write_index = 0;
  int frontier_index = 0;
  int root_index = 0;
  while (frontier_index < frontier_rank || root_index < root_rank) {
    if (root_index >= root_rank ||
        (frontier_index < frontier_rank &&
         frontier_block_indices[xmvb::to_size(frontier_index)] <
             root_block_indices[xmvb::to_size(root_index)])) {
      merged[xmvb::to_size(write_index++)] =
          frontier_labels[xmvb::to_size(frontier_index++)];
      continue;
    }
    merged[xmvb::to_size(write_index++)] =
        root_labels[xmvb::to_size(root_index++)];
  }
  return merged;
}

double combine_exact_one_leaf_deleted_minor_terms(
    int right_root_count,
    int right_leaf_count,
    std::uint32_t row_mask,
    int left_root_count,
    int left_leaf_count,
    std::uint32_t col_mask,
    const RawDeletedMinorTerm& frontier_term,
    const RawDeletedMinorTerm& root_term) {
  // Exact deleted-minor algebra for one sector uses the permutation parity of
  // the surviving rows/columns after those deletions are applied. This is the
  // key difference from the overlap-only `mask_sign`: degree-1/2 channels need
  // the reduced permutation sign, not the undeleted one.
  const std::array<int, 2> deleted_row_block_indices =
      merge_deleted_indices_in_original_order(
          frontier_term.row_block_indices,
          frontier_term.deleted_row_rank,
          root_term.row_block_indices,
          root_term.deleted_row_rank);
  const std::array<int, 2> deleted_col_block_indices =
      merge_deleted_indices_in_original_order(
          frontier_term.col_block_indices,
          frontier_term.deleted_col_rank,
          root_term.col_block_indices,
          root_term.deleted_col_rank);
  const int deleted_row_rank =
      frontier_term.deleted_row_rank + root_term.deleted_row_rank;
  const int deleted_col_rank =
      frontier_term.deleted_col_rank + root_term.deleted_col_rank;

  const int parity =
      deleted_index_sum(deleted_row_block_indices, deleted_row_rank) +
      deleted_index_sum(deleted_col_block_indices, deleted_col_rank) +
      block_order_parity_with_deleted(
          right_root_count,
          row_mask,
          right_leaf_count,
          deleted_row_block_indices,
          deleted_row_rank) +
      block_order_parity_with_deleted(
          left_root_count,
          col_mask,
          left_leaf_count,
          deleted_col_block_indices,
          deleted_col_rank);
  return
      parity_sign(parity) *
      frontier_term.determinant *
      root_term.determinant;
}

bool accumulate_exact_one_leaf_deleted_minor_pair(
    int right_root_count,
    int right_leaf_count,
    std::uint32_t row_mask,
    int left_root_count,
    int left_leaf_count,
    std::uint32_t col_mask,
    const RawDeletedMinorTerm& frontier_term,
    const RawDeletedMinorTerm& root_term,
    SpinMaskCofactorAggregate* aggregate) {
  if (aggregate == nullptr) {
    throw std::invalid_argument("aggregate must not be null");
  }
  const int deleted_row_rank =
      frontier_term.deleted_row_rank + root_term.deleted_row_rank;
  const int deleted_col_rank =
      frontier_term.deleted_col_rank + root_term.deleted_col_rank;
  if (deleted_row_rank != deleted_col_rank || deleted_row_rank > 2) {
    return false;
  }

  const double value = combine_exact_one_leaf_deleted_minor_terms(
      right_root_count,
      right_leaf_count,
      row_mask,
      left_root_count,
      left_leaf_count,
      col_mask,
      frontier_term,
      root_term);
  if (std::abs(value) <= 1.0e-15) {
    return false;
  }

  if (deleted_row_rank == 0) {
    aggregate->signed_overlap += value;
    return true;
  }

  if (deleted_row_rank == 1) {
    const int row_orbital =
        (frontier_term.deleted_row_rank == 1)
            ? frontier_term.row_orbital_labels[0]
            : root_term.row_orbital_labels[0];
    const int col_orbital =
        (frontier_term.deleted_col_rank == 1)
            ? frontier_term.col_orbital_labels[0]
            : root_term.col_orbital_labels[0];
    aggregate->signed_first_cofactor(row_orbital, col_orbital) += value;
    return true;
  }

  const std::array<int, 2> row_orbitals =
      merge_deleted_labels_in_original_order(
          frontier_term.row_block_indices,
          frontier_term.row_orbital_labels,
          frontier_term.deleted_row_rank,
          root_term.row_block_indices,
          root_term.row_orbital_labels,
          root_term.deleted_row_rank);
  const std::array<int, 2> col_orbitals =
      merge_deleted_labels_in_original_order(
          frontier_term.col_block_indices,
          frontier_term.col_orbital_labels,
          frontier_term.deleted_col_rank,
          root_term.col_block_indices,
          root_term.col_orbital_labels,
          root_term.deleted_col_rank);

  const std::size_t previous_size =
      aggregate->signed_second_cofactor_entries.size();
  append_canonical_second_cofactor_entry(
      row_orbitals[0],
      row_orbitals[1],
      col_orbitals[0],
      col_orbitals[1],
      value,
      &aggregate->signed_second_cofactor_entries);
  return aggregate->signed_second_cofactor_entries.size() != previous_size;
}

bool accumulate_exact_one_leaf_payload_merge(
    int right_root_count,
    int right_leaf_count,
    std::uint32_t row_mask,
    int left_root_count,
    int left_leaf_count,
    std::uint32_t col_mask,
    const std::vector<int>& selected_root_row_positions,
    const std::vector<int>& selected_root_col_positions,
    const std::vector<int>& root_remainder_row_positions,
    const std::vector<int>& root_remainder_col_positions,
    const Degree2SpinPayload& frontier,
    const Degree2SpinPayload& root,
    int max_deleted_rank,
    SpinMaskCofactorAggregate* aggregate) {
  // Exact one-leaf contraction in deleted-minor space. Every local payload
  // entry is first converted back to its raw deleted determinant; the global
  // one-spin overlap / cofactor / second-cofactor tensors are then rebuilt by
  // the literal block-order deleted-minor definition.
  if (max_deleted_rank < 0 || max_deleted_rank > 2) {
    throw std::invalid_argument("max_deleted_rank must be in [0, 2]");
  }
  const RawDeletedMinorBuckets frontier_buckets =
      build_raw_deleted_minor_buckets(
          frontier,
          true,
          right_root_count,
          left_root_count,
          static_cast<int>(selected_root_row_positions.size()),
          static_cast<int>(selected_root_col_positions.size()),
          selected_root_row_positions,
          selected_root_col_positions,
          root_remainder_row_positions,
          root_remainder_col_positions,
          true);
  const RawDeletedMinorBuckets root_buckets =
      build_raw_deleted_minor_buckets(
          root,
          false,
          right_root_count,
          left_root_count,
          static_cast<int>(selected_root_row_positions.size()),
          static_cast<int>(selected_root_col_positions.size()),
          selected_root_row_positions,
          selected_root_col_positions,
          root_remainder_row_positions,
          root_remainder_col_positions,
          false);

  bool contributed = false;
  for (int frontier_deleted_rows = 0;
       frontier_deleted_rows <= max_deleted_rank;
       ++frontier_deleted_rows) {
    for (int frontier_deleted_cols = 0;
         frontier_deleted_cols <= max_deleted_rank;
         ++frontier_deleted_cols) {
      const auto& frontier_terms =
          frontier_buckets[xmvb::to_size(frontier_deleted_rows)]
                         [xmvb::to_size(frontier_deleted_cols)];
      if (frontier_terms.empty()) {
        continue;
      }
      for (int root_deleted_rows = 0;
           root_deleted_rows + frontier_deleted_rows <= max_deleted_rank;
           ++root_deleted_rows) {
        for (int root_deleted_cols = 0;
             root_deleted_cols + frontier_deleted_cols <= max_deleted_rank;
             ++root_deleted_cols) {
          const int deleted_row_rank = frontier_deleted_rows + root_deleted_rows;
          const int deleted_col_rank = frontier_deleted_cols + root_deleted_cols;
          if (deleted_row_rank != deleted_col_rank) {
            continue;
          }
          const auto& root_terms =
              root_buckets[xmvb::to_size(root_deleted_rows)]
                          [xmvb::to_size(root_deleted_cols)];
          if (root_terms.empty()) {
            continue;
          }
            for (const auto& frontier_term : frontier_terms) {
            for (const auto& root_term : root_terms) {
              contributed |= accumulate_exact_one_leaf_deleted_minor_pair(
                  right_root_count,
                  right_leaf_count,
                  row_mask,
                  left_root_count,
                  left_leaf_count,
                  col_mask,
                  frontier_term,
                  root_term,
                  aggregate);
            }
          }
        }
      }
    }
  }
  return contributed;
}

void reset_spin_mask_cofactor_aggregate(
    int support_size,
    SpinMaskCofactorAggregate* aggregate) {
  // Reuses one exact support-space aggregate across sector evaluations while
  // keeping already allocated sparse-entry capacity.
  if (aggregate == nullptr) {
    throw std::invalid_argument("aggregate must not be null");
  }
  aggregate->signed_overlap = 0.0;
  aggregate->signed_same_spin_two_electron = 0.0;
  if (aggregate->signed_first_cofactor.rows() != support_size ||
      aggregate->signed_first_cofactor.cols() != support_size) {
    aggregate->signed_first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  } else {
    aggregate->signed_first_cofactor.setZero();
  }
  aggregate->signed_second_cofactor_entries.clear();
  aggregate->mask_state_count = 0;
  aggregate->skipped_zero_mask_state_count = 0;
}

OneLeafBoundarySpinSectorMessage public_boundary_sector_from_internal(
    const BoundarySector& sector,
    int sector_index,
    const SpinMaskCofactorAggregate& aggregate) {
  // Converts one exact support-space sector aggregate into the exported
  // degree-0/1/2 boundary-message representation. The entries already carry
  // the fully merged one-leaf block-order sign convention.
  OneLeafBoundarySpinSectorMessage result;
  result.sector_index = sector_index;
  result.row_mask = sector.row_mask;
  result.col_mask = sector.col_mask;
  result.overlap = aggregate.signed_overlap;

  for (int col_orbital = 0; col_orbital < aggregate.signed_first_cofactor.cols(); ++col_orbital) {
    for (int row_orbital = 0; row_orbital < aggregate.signed_first_cofactor.rows(); ++row_orbital) {
      const double value = aggregate.signed_first_cofactor(row_orbital, col_orbital);
      if (std::abs(value) <= 1.0e-15) {
        continue;
      }
      result.first_cofactor_entries.push_back({
          .row_orbital = row_orbital,
          .col_orbital = col_orbital,
          .value = value,
      });
    }
  }

  result.second_cofactor_entries.reserve(
      aggregate.signed_second_cofactor_entries.size());
  for (const auto& entry : aggregate.signed_second_cofactor_entries) {
    result.second_cofactor_entries.push_back({
        .row_first_orbital = entry.row_first_orbital,
        .row_second_orbital = entry.row_second_orbital,
        .col_first_orbital = entry.col_first_orbital,
        .col_second_orbital = entry.col_second_orbital,
        .value = entry.value,
    });
  }
  return result;
}

OneLeafBoundarySpinAggregate public_boundary_aggregate_from_internal(
    const SpinMaskCofactorAggregate& aggregate) {
  OneLeafBoundarySpinAggregate result;
  result.overlap = aggregate.signed_overlap;
  result.first_cofactor = aggregate.signed_first_cofactor;
  result.same_spin_two_electron = aggregate.signed_same_spin_two_electron;
  result.second_cofactor_entries.reserve(
      aggregate.signed_second_cofactor_entries.size());
  for (const auto& entry : aggregate.signed_second_cofactor_entries) {
    result.second_cofactor_entries.push_back({
        .row_first_orbital = entry.row_first_orbital,
        .row_second_orbital = entry.row_second_orbital,
        .col_first_orbital = entry.col_first_orbital,
        .col_second_orbital = entry.col_second_orbital,
        .value = entry.value,
    });
  }
  return result;
}

void add_boundary_sector_to_aggregate(
    const OneLeafBoundarySpinSectorMessage& sector,
    SpinMaskCofactorAggregate* aggregate) {
  // Accumulates one exported boundary sector back into the dense support-space
  // aggregate used by the exact checker and final same-spin contraction.
  if (aggregate == nullptr) {
    throw std::invalid_argument("aggregate must not be null");
  }
  aggregate->signed_overlap += sector.overlap;
  for (const auto& entry : sector.first_cofactor_entries) {
    aggregate->signed_first_cofactor(entry.row_orbital, entry.col_orbital) +=
        entry.value;
  }
  for (const auto& entry : sector.second_cofactor_entries) {
    append_canonical_second_cofactor_entry(
        entry.row_first_orbital,
        entry.row_second_orbital,
        entry.col_first_orbital,
        entry.col_second_orbital,
        entry.value,
        &aggregate->signed_second_cofactor_entries);
  }
}

OneLeafBoundarySpinMessage build_one_leaf_spin_boundary_message_impl(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // Exposes the exact one-leaf one-spin boundary basis explicitly.
  //
  // For each admissible selected root boundary sector `(row_mask, col_mask)`,
  // the returned message stores the fully merged exact degree-0/1/2 sector
  // amplitudes in support-space coordinates. This is the concrete one-leaf
  // realization of `M^(0)(s)`, `M^(1)(mu; s)`, and `M^(2)(nu; s)`.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }

  OneLeafBoundarySpinMessage message;
  message.support_size = support_size;
  message.base_selected_row_minus_col =
      static_cast<int>(left_leaf_occ.size()) -
      static_cast<int>(right_leaf_occ.size());
  const BoundarySectorFamilyIndexerRange family_range(
      static_cast<int>(right_root_occ.size()),
      static_cast<int>(left_root_occ.size()),
      message.base_selected_row_minus_col,
      -2,
      2);
  message.families.reserve(xmvb::to_size(family_range.family_count()));

  const std::uint32_t row_limit =
      open_state_mask_limit(static_cast<int>(right_root_occ.size()));
  const std::uint32_t col_limit =
      open_state_mask_limit(static_cast<int>(left_root_occ.size()));
  const std::uint32_t full_row_mask = row_limit - 1U;
  const std::uint32_t full_col_mask = col_limit - 1U;
  const auto right_root_occ_by_mask = build_mask_occ_table(right_root_occ);
  const auto left_root_occ_by_mask = build_mask_occ_table(left_root_occ);
  const auto right_root_positions_by_mask =
      build_mask_position_table(static_cast<int>(right_root_occ.size()));
  const auto left_root_positions_by_mask =
      build_mask_position_table(static_cast<int>(left_root_occ.size()));

  std::vector<int> frontier_right_occ;
  std::vector<int> frontier_left_occ;
  frontier_right_occ.reserve(right_root_occ.size() + right_leaf_occ.size());
  frontier_left_occ.reserve(left_root_occ.size() + left_leaf_occ.size());

  Degree2SpinPayloadBuilder frontier_payload_builder(
      support_overlap_storage,
      support_size,
      overlap_resolver,
      subdeterminant_evaluations);
  Degree2SpinPayloadBuilder root_payload_builder(
      support_overlap_storage,
      support_size,
      overlap_resolver,
      subdeterminant_evaluations);
  SpinMaskCofactorAggregate sector_aggregate;
  reset_spin_mask_cofactor_aggregate(support_size, &sector_aggregate);

  for (int family_index = 0;
       family_index < family_range.family_count();
       ++family_index) {
    const BoundarySectorFamilyIndexer& family_indexer =
        family_range.family(family_index);
    OneLeafBoundarySpinSectorFamilyMessage family;
    family.family_indexer = family_indexer;
    family.sectors.reserve(
        xmvb::to_size(family.family_indexer.sector_indexer.sector_count()));

    for (int sector_index = 0;
         sector_index < family.family_indexer.sector_indexer.sector_count();
         ++sector_index) {
      const BoundarySector& sector =
          family.family_indexer.sector_indexer.sector(sector_index);
      const auto& selected_root_rows =
          right_root_occ_by_mask[xmvb::to_size(sector.row_mask)];
      const auto& selected_root_row_positions =
          right_root_positions_by_mask[xmvb::to_size(sector.row_mask)];
      const auto& selected_root_cols =
          left_root_occ_by_mask[xmvb::to_size(sector.col_mask)];
      const auto& selected_root_col_positions =
          left_root_positions_by_mask[xmvb::to_size(sector.col_mask)];
      const auto& root_rows =
          right_root_occ_by_mask[xmvb::to_size(full_row_mask ^ sector.row_mask)];
      const auto& root_row_positions =
          right_root_positions_by_mask[xmvb::to_size(full_row_mask ^ sector.row_mask)];
      const auto& root_cols =
          left_root_occ_by_mask[xmvb::to_size(full_col_mask ^ sector.col_mask)];
      const auto& root_col_positions =
          left_root_positions_by_mask[xmvb::to_size(full_col_mask ^ sector.col_mask)];

      frontier_right_occ.clear();
      append_occ_block(selected_root_rows, &frontier_right_occ);
      append_occ_block(right_leaf_occ, &frontier_right_occ);
      frontier_left_occ.clear();
      append_occ_block(selected_root_cols, &frontier_left_occ);
      append_occ_block(left_leaf_occ, &frontier_left_occ);

      const Degree2SpinPayload& frontier_payload = frontier_payload_builder.build(
          frontier_left_occ,
          frontier_right_occ,
          static_cast<int>(selected_root_rows.size()),
          static_cast<int>(selected_root_cols.size()),
          true,
          PartialSide::LeftFrontier,
          true,
          2);
      const Degree2SpinPayload& root_payload = root_payload_builder.build(
          root_cols,
          root_rows,
          0,
          0,
          false,
          PartialSide::RightComplement,
          true,
          2);

      reset_spin_mask_cofactor_aggregate(support_size, &sector_aggregate);
      accumulate_exact_one_leaf_payload_merge(
          static_cast<int>(right_root_occ.size()),
          static_cast<int>(right_leaf_occ.size()),
          sector.row_mask,
          static_cast<int>(left_root_occ.size()),
          static_cast<int>(left_leaf_occ.size()),
          sector.col_mask,
          selected_root_row_positions,
          selected_root_col_positions,
          root_row_positions,
          root_col_positions,
          frontier_payload,
          root_payload,
          2,
          &sector_aggregate);
      compress_second_cofactor_entries(&sector_aggregate.signed_second_cofactor_entries);
      family.sectors.push_back(public_boundary_sector_from_internal(
          sector,
          sector_index,
          sector_aggregate));
    }

    message.families.push_back(std::move(family));
  }
  return message;
}

OneLeafBoundarySpinAggregate contract_one_leaf_spin_boundary_message_impl(
    const OneLeafBoundarySpinMessage& message,
    const std::vector<double>& packed_active_two_electron_integrals) {
  // Exact one-leaf contraction from the exported boundary message.
  //
  // The public message now already stores the exact degree-0/1/2 boundary
  // amplitudes sector-by-sector, so contraction is only a direct sector sum in
  // support-space coordinates. No reduced-state overlap closure remains here.
  if (message.support_size <= 0) {
    throw std::invalid_argument("boundary message support_size must be positive");
  }

  SpinMaskCofactorAggregate aggregate;
  reset_spin_mask_cofactor_aggregate(message.support_size, &aggregate);
  for (const auto& family : message.families) {
    for (const auto& sector : family.sectors) {
      add_boundary_sector_to_aggregate(sector, &aggregate);
    }
  }

  compress_second_cofactor_entries(&aggregate.signed_second_cofactor_entries);
  if (!packed_active_two_electron_integrals.empty()) {
    aggregate.signed_same_spin_two_electron =
        contract_same_spin_second_cofactors(
            aggregate.signed_second_cofactor_entries,
            packed_active_two_electron_integrals);
  }
  return public_boundary_aggregate_from_internal(aggregate);
}

double contract_same_spin_second_cofactors(
    const std::vector<SpinMaskCofactorAggregate::SecondCofactorEntry>& second_cofactor_entries,
    const std::vector<double>& packed_active_two_electron_integrals) {
  // Exact same-spin contraction:
  //   sum_{r1<r2,c1<c2} [(r1 c1|r2 c2) - (r1 c2|r2 c1)] C^{(2)}_{r1 r2,c1 c2}
  // where the second-cofactor entries are already accumulated in
  // support-orbital labels with the correct determinant sign convention.
  double total = 0.0;
  for (const auto& entry : second_cofactor_entries) {
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            entry.row_first_orbital,
            entry.col_first_orbital,
            entry.row_second_orbital,
            entry.col_second_orbital);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            entry.row_first_orbital,
            entry.col_second_orbital,
            entry.row_second_orbital,
            entry.col_first_orbital);
    total +=
        (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
         packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]) *
        entry.value;
  }
  return total;
}

struct OneLeafCollapsedAggregateBundle {
  ComponentSpinCoefficientOperator root_operator;
  ComponentSpinCoefficientOperator leaf_operator;
  IndexedOneLeafDirectSpinAggregateTable alpha_table;
  IndexedOneLeafDirectSpinAggregateTable beta_table;
  std::uint64_t processed_term_quadruple_count = 0;
  std::uint64_t alpha_mask_state_count = 0;
  std::uint64_t beta_mask_state_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
};

OneLeafCollapsedAggregateBundle build_one_leaf_collapsed_aggregate_bundle(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver,
    bool compute_one_electron) {
  // Builds the exact one-leaf alpha/beta direct aggregate tables once and
  // materializes all scalar/cofactor contractions needed by the production
  // one-leaf Hamiltonian channels. This is the shared hot-path state that
  // lets overlap, one-electron, same-spin, and opposite-spin reuse the same
  // exact state-pair aggregates instead of rebuilding them per channel.
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.size() != 2U) {
    throw std::invalid_argument(
        "collapsed one-leaf bundle requires exactly two ordered components");
  }

  OneLeafCollapsedAggregateBundle bundle;
  bundle.root_operator =
      build_component_spin_coefficient_operator(ordered_components.front());
  bundle.leaf_operator =
      build_component_spin_coefficient_operator(ordered_components.back());
  bundle.processed_term_quadruple_count =
      bundle.root_operator.raw_nonzero_pair_count *
      bundle.leaf_operator.raw_nonzero_pair_count;

  const std::vector<double> zero_one_electron_storage(
      xmvb::to_size(support_size) *
          xmvb::to_size(support_size),
      0.0);
  const std::vector<double>& one_electron_storage =
      compute_one_electron ? support_one_electron_storage : zero_one_electron_storage;

  bundle.alpha_table =
      build_indexed_one_leaf_direct_spin_aggregate_table(
          bundle.root_operator.alpha_states,
          bundle.leaf_operator.alpha_states,
          support_overlap_storage,
          one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          true,
          false,
          false,
          overlap_resolver,
          &bundle.subdeterminant_evaluations);
  bundle.beta_table =
      build_indexed_one_leaf_direct_spin_aggregate_table(
          bundle.root_operator.beta_states,
          bundle.leaf_operator.beta_states,
          support_overlap_storage,
          one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          true,
          false,
          false,
          overlap_resolver,
          &bundle.subdeterminant_evaluations);
  bundle.alpha_mask_state_count = bundle.alpha_table.total_state_count;
  bundle.beta_mask_state_count = bundle.beta_table.total_state_count;
  return bundle;
}

TwoElectronStarPairResult
convert_one_leaf_two_electron_result(
    const CollapsedTwoElectronOneLeafStarPairStats& stats) {
  TwoElectronStarPairResult result;
  result.overlap = stats.collapsed_overlap;
  result.same_spin_alpha_two_electron =
      stats.collapsed_same_spin_alpha_two_electron;
  result.same_spin_beta_two_electron =
      stats.collapsed_same_spin_beta_two_electron;
  result.opposite_spin_two_electron =
      stats.collapsed_opposite_spin_two_electron;
  result.two_electron = stats.collapsed_two_electron;
  result.collapsed_leaf_state_count =
      stats.alpha_mask_state_count + stats.beta_mask_state_count;
  result.hypercube_assignment_count = stats.processed_term_quadruple_count;
  result.subdeterminant_evaluations = stats.subdeterminant_evaluations;
  result.dp_transition_count = 0;
  return result;
}

CollapsedTwoElectronOneLeafStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_impl(
    bool compute_exact_reference,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.size() != 2U) {
    throw std::invalid_argument(
        "collapsed one-leaf two-electron path requires exactly two ordered components");
  }

  CollapsedTwoElectronOneLeafStarPairStats stats;
  if (compute_exact_reference) {
    const ExactTwoElectronStarPairStats exact_stats =
        evaluate_component_ordered_open_state_star_pair_two_electron_exact(
            support_overlap_storage,
            packed_active_two_electron_integrals,
            support_size,
            ordered_components,
            overlap_resolver);
    stats.exact_overlap = exact_stats.exact_overlap;
    stats.exact_same_spin_alpha_two_electron =
        exact_stats.exact_same_spin_alpha_two_electron;
    stats.exact_same_spin_beta_two_electron =
        exact_stats.exact_same_spin_beta_two_electron;
    stats.exact_opposite_spin_two_electron =
        exact_stats.exact_opposite_spin_two_electron;
    stats.exact_two_electron = exact_stats.exact_two_electron;
  } else {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    stats.exact_overlap = nan;
    stats.exact_same_spin_alpha_two_electron = nan;
    stats.exact_same_spin_beta_two_electron = nan;
    stats.exact_opposite_spin_two_electron = nan;
    stats.exact_two_electron = nan;
  }

  const OneLeafCollapsedAggregateBundle bundle =
      build_one_leaf_collapsed_aggregate_bundle(
          support_overlap_storage,
          {},
          packed_active_two_electron_integrals,
          support_size,
          ordered_components,
          overlap_resolver,
          false);
  stats.processed_term_quadruple_count = bundle.processed_term_quadruple_count;
  stats.alpha_mask_state_count = bundle.alpha_mask_state_count;
  stats.beta_mask_state_count = bundle.beta_mask_state_count;
  stats.skipped_zero_alpha_mask_state_count = 0;
  stats.skipped_zero_beta_mask_state_count = 0;
  stats.subdeterminant_evaluations = bundle.subdeterminant_evaluations;

  const OneLeafDirectAggregateContractionResult contraction =
      contract_one_leaf_direct_aggregate_channels(
          bundle.root_operator,
          bundle.leaf_operator,
          bundle.alpha_table,
          bundle.beta_table,
          &packed_active_two_electron_integrals);
  stats.collapsed_overlap = contraction.overlap;
  stats.collapsed_same_spin_alpha_two_electron =
      contraction.same_spin_alpha_two_electron;
  stats.collapsed_same_spin_beta_two_electron =
      contraction.same_spin_beta_two_electron;
  stats.collapsed_opposite_spin_two_electron =
      contraction.opposite_spin_two_electron;

  stats.collapsed_two_electron =
      stats.collapsed_same_spin_alpha_two_electron +
      stats.collapsed_same_spin_beta_two_electron +
      stats.collapsed_opposite_spin_two_electron;
  if (compute_exact_reference) {
    stats.overlap_absolute_error =
        std::abs(stats.collapsed_overlap - stats.exact_overlap);
    stats.same_spin_alpha_absolute_error =
        std::abs(
            stats.collapsed_same_spin_alpha_two_electron -
            stats.exact_same_spin_alpha_two_electron);
    stats.same_spin_beta_absolute_error =
        std::abs(
            stats.collapsed_same_spin_beta_two_electron -
            stats.exact_same_spin_beta_two_electron);
    stats.opposite_spin_absolute_error =
        std::abs(
            stats.collapsed_opposite_spin_two_electron -
            stats.exact_opposite_spin_two_electron);
    stats.total_two_electron_absolute_error =
        std::abs(stats.collapsed_two_electron - stats.exact_two_electron);
  }
  return stats;
}

}  // namespace

OneLeafBoundarySpinMessage build_one_leaf_spin_boundary_message(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  return build_one_leaf_spin_boundary_message_impl(
      left_root_occ,
      left_leaf_occ,
      right_root_occ,
      right_leaf_occ,
      support_overlap_storage,
      support_size,
      overlap_resolver,
      subdeterminant_evaluations);
}

OneLeafBoundarySpinAggregate contract_one_leaf_spin_boundary_message(
    const OneLeafBoundarySpinMessage& message,
    const std::vector<double>& packed_active_two_electron_integrals) {
  return contract_one_leaf_spin_boundary_message_impl(
      message,
      packed_active_two_electron_integrals);
}

CollapsedTwoElectronOneLeafStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  return evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_impl(
      true,
      support_overlap_storage,
      packed_active_two_electron_integrals,
      support_size,
      ordered_components,
      overlap_resolver);
}

TwoElectronStarPairResult
evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  return convert_one_leaf_two_electron_result(
      evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_impl(
          false,
          support_overlap_storage,
          packed_active_two_electron_integrals,
          support_size,
          ordered_components,
          overlap_resolver));
}

OneLeafHamiltonianStarPairResult
evaluate_component_ordered_open_state_star_pair_hamiltonian_one_leaf_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  const OneLeafCollapsedAggregateBundle bundle =
      build_one_leaf_collapsed_aggregate_bundle(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      ordered_components,
      overlap_resolver,
      true);
  const OneLeafDirectAggregateContractionResult contraction =
      contract_one_leaf_direct_aggregate_channels(
          bundle.root_operator,
          bundle.leaf_operator,
          bundle.alpha_table,
          bundle.beta_table,
          &packed_active_two_electron_integrals);

  OneLeafHamiltonianStarPairResult result;
  result.overlap = contraction.overlap;
  result.one_electron = contraction.one_electron;
  result.same_spin_alpha_two_electron =
      contraction.same_spin_alpha_two_electron;
  result.same_spin_beta_two_electron =
      contraction.same_spin_beta_two_electron;
  result.opposite_spin_two_electron =
      contraction.opposite_spin_two_electron;
  result.two_electron =
      result.same_spin_alpha_two_electron +
      result.same_spin_beta_two_electron +
      result.opposite_spin_two_electron;
  result.collapsed_leaf_state_count =
      bundle.alpha_mask_state_count + bundle.beta_mask_state_count;
  result.hypercube_assignment_count = bundle.processed_term_quadruple_count;
  result.subdeterminant_evaluations = bundle.subdeterminant_evaluations;
  result.dp_transition_count = 0;
  return result;
}

CollapsedOppositeSpinOneLeafStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_opposite_spin_one_leaf(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  const CollapsedTwoElectronOneLeafStarPairStats full_stats =
      evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
          support_overlap_storage,
          packed_active_two_electron_integrals,
          support_size,
          ordered_components,
          overlap_resolver);

  CollapsedOppositeSpinOneLeafStarPairStats stats;
  stats.exact_overlap = full_stats.exact_overlap;
  stats.collapsed_overlap = full_stats.collapsed_overlap;
  stats.overlap_absolute_error = full_stats.overlap_absolute_error;
  stats.exact_opposite_spin_two_electron =
      full_stats.exact_opposite_spin_two_electron;
  stats.collapsed_opposite_spin_two_electron =
      full_stats.collapsed_opposite_spin_two_electron;
  stats.opposite_spin_absolute_error =
      full_stats.opposite_spin_absolute_error;
  stats.processed_term_quadruple_count =
      full_stats.processed_term_quadruple_count;
  stats.alpha_mask_state_count = full_stats.alpha_mask_state_count;
  stats.beta_mask_state_count = full_stats.beta_mask_state_count;
  stats.skipped_zero_alpha_mask_state_count =
      full_stats.skipped_zero_alpha_mask_state_count;
  stats.skipped_zero_beta_mask_state_count =
      full_stats.skipped_zero_beta_mask_state_count;
  stats.subdeterminant_evaluations = full_stats.subdeterminant_evaluations;
  return stats;
}

}  // namespace xmvb::vb::exact_separator
