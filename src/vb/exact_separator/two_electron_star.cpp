#include "vb/exact_separator/two_electron.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::exact_separator {

namespace {


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

struct PartialSpinPayload {
  // Exact one-spin partial payload for one open-state mask.
  //
  // `n_rows`/`n_cols` are the row/column dimensions of the current overlap
  // block. The payload carries the exact deleted-minor sectors needed by the
  // star recurrence:
  // - `(0,0)` closed overlap when the block is square;
  // - `(1,0)` row-open minors when the block is row-open;
  // - `(0,1)` col-open minors when the block is col-open;
  // - `(1,1)` first cofactors when the block is square.
  int n_rows = 0;
  int n_cols = 0;
  double closed_overlap = 0.0;
  std::vector<MinorEntry> row_open_entries;
  std::vector<MinorEntry> col_open_entries;
  std::vector<CofactorEntry> cofactor_entries;
};

struct PartialSpinPayloadCacheKey {
  std::vector<int> left_occ;
  std::vector<int> right_occ;
  int selected_root_rows = 0;
  int selected_root_cols = 0;
  bool zero_selected_root_block = false;
  PartialSide side = PartialSide::LeftFrontier;
};

bool operator==(
    const PartialSpinPayloadCacheKey& left,
    const PartialSpinPayloadCacheKey& right) {
  return left.left_occ == right.left_occ &&
      left.right_occ == right.right_occ &&
      left.selected_root_rows == right.selected_root_rows &&
      left.selected_root_cols == right.selected_root_cols &&
      left.zero_selected_root_block == right.zero_selected_root_block &&
      left.side == right.side;
}

struct PartialSpinPayloadCacheKeyHasher {
  std::size_t operator()(const PartialSpinPayloadCacheKey& key) const;
};

struct SpinMaskPayload {
  std::uint32_t row_mask = 0;
  std::uint32_t col_mask = 0;
  PartialSpinPayload payload;
};

struct SpinDeletionKey {
  // Block-ordered deleted row/column labels of one spin sector.
  //
  // The vectors store support-orbital labels in the exact component/block
  // order used by the recurrence, not in sorted numerical order. This is the
  // order whose antisymmetry is already absorbed into the stored value.
  std::vector<int> row_labels;
  std::vector<int> col_labels;
};

bool operator<(const SpinDeletionKey& left, const SpinDeletionKey& right) {
  return std::tie(left.row_labels, left.col_labels) <
      std::tie(right.row_labels, right.col_labels);
}

struct SpinDeletionPayload {
  // Exact one-spin deleted-minor sectors up to second order.
  //
  // `sectors[{rows, cols}]` stores the signed deleted minor obtained after
  // deleting `rows.size()` block-ordered rows and `cols.size()` block-ordered
  // columns from the current spin overlap block. Only sectors with at most two
  // deleted rows and at most two deleted columns are carried because the
  // pure two-electron Hamiltonian needs overlap `(0,0)`, first cofactors
  // `(1,1)`, and second cofactors `(2,2)`.
  int row_count = 0;
  int col_count = 0;
  std::map<SpinDeletionKey, double> sectors;
};

struct JointDeletionKey {
  // Coupled alpha/beta deleted-minor key for one exact separator state.
  //
  // The exact multi-leaf recurrence must preserve the correlation between the
  // alpha and beta spin sectors contributed by the same leaf determinant-term
  // combination. A joint key keeps those two spin sectors together so overlap
  // and two-electron channels are accumulated from one coupled coefficient map
  // instead of from a product of independently aggregated spin sums.
  SpinDeletionKey alpha_key;
  SpinDeletionKey beta_key;
};

bool operator<(const JointDeletionKey& left, const JointDeletionKey& right) {
  return std::tie(left.alpha_key, left.beta_key) <
      std::tie(right.alpha_key, right.beta_key);
}

struct JointDeletionPayload {
  // Coupled alpha/beta deleted-minor sectors up to second order per spin.
  //
  // The stored coefficient is the exact weight of the product
  //   alpha_sector * beta_sector
  // after summing all leaf determinant-term combinations belonging to one
  // separator mask state. This keeps the spin coupling exact throughout the
  // frontier recursion and avoids the incorrect "product of separately summed
  // alpha/beta payloads" factorization.
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  std::map<JointDeletionKey, double> sectors;
};

struct TwoElectronLeafMessage {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  JointDeletionPayload payload;
};

void hash_combine(std::size_t* seed, std::size_t value) {
  if (seed == nullptr) {
    throw std::invalid_argument("hash seed must not be null");
  }
  *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

std::size_t hash_occ_list(const std::vector<int>& occ) {
  std::size_t seed = occ.size();
  for (const int orbital : occ) {
    hash_combine(&seed, xmvb::to_size(orbital + 0x10000));
  }
  return seed;
}

std::size_t PartialSpinPayloadCacheKeyHasher::operator()(
    const PartialSpinPayloadCacheKey& key) const {
  std::size_t seed = 0;
  hash_combine(&seed, hash_occ_list(key.left_occ));
  hash_combine(&seed, hash_occ_list(key.right_occ));
  hash_combine(&seed, xmvb::to_size(key.selected_root_rows));
  hash_combine(&seed, xmvb::to_size(key.selected_root_cols));
  hash_combine(&seed, key.zero_selected_root_block ? 1U : 0U);
  hash_combine(&seed, xmvb::to_size(key.side));
  return seed;
}

std::uint32_t open_state_mask_limit(int n_bits) {
  if (n_bits < 0 || n_bits >= 31) {
    throw std::invalid_argument("open-state mask width is out of range");
  }
  return (n_bits == 0) ? 1U : (static_cast<std::uint32_t>(1U) << n_bits);
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

void assign_concatenated_occ(
    const std::vector<int>& first,
    const std::vector<int>& second,
    std::vector<int>* result) {
  if (result == nullptr) {
    throw std::invalid_argument("result must not be null");
  }
  result->clear();
  result->reserve(first.size() + second.size());
  result->insert(result->end(), first.begin(), first.end());
  result->insert(result->end(), second.begin(), second.end());
}

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

double parity_sign_from_count(int count) {
  return parity_sign(count & 1);
}

int canonicalization_parity(const std::vector<int>& occupied_orbitals) {
  int parity = 0;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals.size();
         ++right_index) {
      if (occupied_orbitals[left_index] > occupied_orbitals[right_index]) {
        parity ^= 1;
      }
    }
  }
  return parity;
}

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes) {
  if (selected_masks.size() != leaf_occ_sizes.size()) {
    throw std::invalid_argument("selected_masks and leaf_occ_sizes must have the same length");
  }
  std::vector<int> block_order;
  block_order.reserve(
      xmvb::to_size(n_root_occ) +
      std::accumulate(leaf_occ_sizes.begin(), leaf_occ_sizes.end(), 0));

  std::uint32_t used_mask = 0U;
  int next_leaf_label = n_root_occ;
  for (std::size_t leaf_index = 0; leaf_index < selected_masks.size(); ++leaf_index) {
    const std::uint32_t mask = selected_masks[leaf_index];
    used_mask |= mask;
    for (int root_position = 0; root_position < n_root_occ; ++root_position) {
      if ((mask & (static_cast<std::uint32_t>(1U) << root_position)) != 0U) {
        block_order.push_back(root_position);
      }
    }
    for (int leaf_position = 0;
         leaf_position < leaf_occ_sizes[xmvb::to_size(leaf_index)];
         ++leaf_position) {
      block_order.push_back(next_leaf_label++);
    }
  }

  const std::uint32_t full_mask =
      (n_root_occ == 0) ? 0U
                        : ((static_cast<std::uint32_t>(1U) << n_root_occ) - 1U);
  const std::uint32_t remainder_mask = full_mask ^ used_mask;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((remainder_mask & (static_cast<std::uint32_t>(1U) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

Eigen::MatrixXd build_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals) {
  Eigen::MatrixXd overlap_block(
      static_cast<int>(right_occ.size()),
      static_cast<int>(left_occ.size()));
  for (int column = 0; column < static_cast<int>(left_occ.size()); ++column) {
    const int left_orbital = left_occ[xmvb::to_size(column)];
    for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
      const int right_orbital = right_occ[xmvb::to_size(row)];
      overlap_block(row, column) =
          overlap_storage[xmvb::to_size(left_orbital) *
                              xmvb::to_size(n_orbitals) +
                          xmvb::to_size(right_orbital)];
    }
  }
  return overlap_block;
}

double determinant_of_dense_matrix(
    const Eigen::MatrixXd& matrix,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
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

Eigen::VectorXd build_signed_open_minor_vector(
    const Eigen::MatrixXd& overlap_block,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
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
    return Eigen::VectorXd(0);
  }
  const int rank_dimension = std::min(n_rows, n_cols);
  ++(*subdeterminant_evaluations);
  const Eigen::MatrixXd kernel_operator =
      (n_rows == n_cols + 1) ? overlap_block.transpose() : overlap_block;
  Eigen::FullPivLU<Eigen::MatrixXd> rectangular_lu(kernel_operator);
  rectangular_lu.setThreshold(overlap_resolver.linear_dependence_threshold());
  if (rectangular_lu.rank() != rank_dimension ||
      rectangular_lu.dimensionOfKernel() != 1) {
    return Eigen::VectorXd::Zero(null_vector_size);
  }

  const Eigen::VectorXd null_vector = rectangular_lu.kernel().col(0);
  Eigen::Index anchor_index_eigen = 0;
  null_vector.cwiseAbs().maxCoeff(&anchor_index_eigen);
  const int anchor_index = static_cast<int>(anchor_index_eigen);
  if (std::abs(null_vector(anchor_index)) <= 1.0e-15) {
    return Eigen::VectorXd::Zero(null_vector_size);
  }

  const Eigen::MatrixXd anchor_minor =
      (n_rows == n_cols + 1)
          ? [&, anchor_index]() {
              Eigen::MatrixXd minor(n_cols, n_cols);
              if (anchor_index > 0) {
                minor.topRows(anchor_index) = overlap_block.topRows(anchor_index);
              }
              const int trailing_row_count = n_cols - anchor_index;
              if (trailing_row_count > 0) {
                minor.bottomRows(trailing_row_count) =
                    overlap_block.bottomRows(trailing_row_count);
              }
              return minor;
            }()
          : [&, anchor_index]() {
              Eigen::MatrixXd minor(n_rows, n_rows);
              if (anchor_index > 0) {
                minor.leftCols(anchor_index) = overlap_block.leftCols(anchor_index);
              }
              const int trailing_col_count = n_rows - anchor_index;
              if (trailing_col_count > 0) {
                minor.rightCols(trailing_col_count) =
                    overlap_block.rightCols(trailing_col_count);
              }
              return minor;
            }();
  const double anchor_minor_value =
      parity_sign(anchor_index) *
      determinant_of_dense_matrix(anchor_minor, overlap_resolver, subdeterminant_evaluations);
  return (anchor_minor_value / null_vector(anchor_index)) * null_vector;
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
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
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
    for (int column = 0; column < payload.n_cols; ++column) {
      if (column < internal_col_begin) {
        continue;
      }
      for (int row = 0; row < payload.n_rows; ++row) {
        if (row < internal_row_begin) {
          continue;
        }
        const double value = cofactor_1st(row, column);
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload.cofactor_entries.push_back({
            .row_local_index = row,
            .col_local_index = column,
            .row_orbital_label = right_occ[xmvb::to_size(row)],
            .col_orbital_label = left_occ[xmvb::to_size(column)],
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
    for (int column = 0; column < payload.n_cols; ++column) {
      if (column < internal_col_begin) {
        continue;
      }
      const double determinant = parity_sign(column) * signed_deleted_col_minors(column);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      const double sign =
          (side == PartialSide::LeftFrontier)
              ? parity_sign(payload.n_rows + column)
              : parity_sign(column);
      payload.col_open_entries.push_back({
          .local_index = column,
          .orbital_label = left_occ[xmvb::to_size(column)],
          .value = sign * determinant,
      });
    }
  }

  return payload;
}

const PartialSpinPayload& build_partial_spin_payload_cached(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int selected_root_rows,
    int selected_root_cols,
    bool zero_selected_root_block,
    PartialSide side,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache) {
  if (payload_cache == nullptr) {
    throw std::invalid_argument("payload_cache must not be null");
  }
  const PartialSpinPayloadCacheKey cache_key{
      left_occ,
      right_occ,
      selected_root_rows,
      selected_root_cols,
      zero_selected_root_block,
      side,
  };
  auto iterator = payload_cache->find(cache_key);
  if (iterator == payload_cache->end()) {
    iterator = payload_cache
                   ->emplace(
                       cache_key,
                       build_partial_spin_payload(
                           left_occ,
                           right_occ,
                           overlap_storage,
                           n_orbitals,
                           selected_root_rows,
                           selected_root_cols,
                           zero_selected_root_block,
                           side,
                           overlap_resolver,
                           subdeterminant_evaluations))
                   .first;
  }
  return iterator->second;
}

std::vector<SpinMaskPayload> build_frontier_spin_mask_payloads(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache) {
  std::vector<SpinMaskPayload> entries;
  const std::uint32_t row_limit =
      open_state_mask_limit(static_cast<int>(right_root_occ.size()));
  const std::uint32_t col_limit =
      open_state_mask_limit(static_cast<int>(left_root_occ.size()));
  const auto right_root_occ_by_mask = build_mask_occ_table(right_root_occ);
  const auto left_root_occ_by_mask = build_mask_occ_table(left_root_occ);
  std::vector<int> frontier_left_occ;
  std::vector<int> frontier_right_occ;
  frontier_left_occ.reserve(left_root_occ.size() + left_leaf_occ.size());
  frontier_right_occ.reserve(right_root_occ.size() + right_leaf_occ.size());

  for (std::uint32_t row_mask = 0; row_mask < row_limit; ++row_mask) {
    const auto& selected_root_rows =
        right_root_occ_by_mask[xmvb::to_size(row_mask)];
    for (std::uint32_t col_mask = 0; col_mask < col_limit; ++col_mask) {
      const auto& selected_root_cols =
          left_root_occ_by_mask[xmvb::to_size(col_mask)];
      assign_concatenated_occ(selected_root_cols, left_leaf_occ, &frontier_left_occ);
      assign_concatenated_occ(selected_root_rows, right_leaf_occ, &frontier_right_occ);
      const PartialSpinPayload& payload = build_partial_spin_payload_cached(
          frontier_left_occ,
          frontier_right_occ,
          overlap_storage,
          n_orbitals,
          static_cast<int>(selected_root_rows.size()),
          static_cast<int>(selected_root_cols.size()),
          true,
          PartialSide::LeftFrontier,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
      const bool nonzero =
          std::abs(payload.closed_overlap) > 1.0e-15 ||
          !payload.row_open_entries.empty() ||
          !payload.col_open_entries.empty() ||
          !payload.cofactor_entries.empty();
      if (!nonzero) {
        continue;
      }
      entries.push_back({row_mask, col_mask, payload});
    }
  }
  return entries;
}

SpinDeletionKey make_spin_key(
    std::vector<int> row_labels = {},
    std::vector<int> col_labels = {}) {
  return SpinDeletionKey{std::move(row_labels), std::move(col_labels)};
}

bool is_spin_payload_nonzero(const SpinDeletionPayload& payload) {
  for (const auto& [key, value] : payload.sectors) {
    static_cast<void>(key);
    if (std::abs(value) > 1.0e-15) {
      return true;
    }
  }
  return false;
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
  // Exact block-order sign used by the generic sector convolution.
  //
  // The one-electron merge identities show that only the unpaired open legs of
  // the left payload cross the unresolved block sizes of the right payload:
  // - a left `(1,0)` row-open sector contributes one factor of `right.n_cols`;
  // - a left `(0,1)` col-open sector contributes one factor of `right.n_rows`;
  // - a balanced left cofactor sector `(1,1)` contributes no such factor.
  //
  // The entire right sector, viewed as a degree-`max(r,c)` object in the
  // separator exterior algebra, must still cross the already-built left block.
  // This reproduces the validated one-electron signs while also removing the
  // spurious extra `right.n_rows + right.n_cols` phase that would otherwise
  // appear in balanced cases such as `cofactor x cofactor -> second cofactor`.
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

      merged.sectors[combined_key] +=
          parity_sign(spin_sector_merge_parity(
              left.row_count,
              left.col_count,
              left_key,
              right.row_count,
              right.col_count,
              right_key)) *
          left_value * right_value;
    }
  }
  cleanup_spin_payload(&merged);
  return merged;
}

JointDeletionKey make_joint_key(
    SpinDeletionKey alpha_key = {},
    SpinDeletionKey beta_key = {}) {
  return JointDeletionKey{std::move(alpha_key), std::move(beta_key)};
}

bool is_empty_spin_key(const SpinDeletionKey& key) {
  return key.row_labels.empty() && key.col_labels.empty();
}

bool combine_spin_keys(
    const SpinDeletionKey& left_key,
    const SpinDeletionKey& right_key,
    SpinDeletionKey* combined_key) {
  if (combined_key == nullptr) {
    throw std::invalid_argument("combined_key must not be null");
  }
  const std::size_t combined_row_rank =
      left_key.row_labels.size() + right_key.row_labels.size();
  const std::size_t combined_col_rank =
      left_key.col_labels.size() + right_key.col_labels.size();
  if (combined_row_rank > 2U || combined_col_rank > 2U) {
    return false;
  }
  combined_key->row_labels.clear();
  combined_key->col_labels.clear();
  combined_key->row_labels.reserve(combined_row_rank);
  combined_key->col_labels.reserve(combined_col_rank);
  combined_key->row_labels.insert(
      combined_key->row_labels.end(),
      left_key.row_labels.begin(),
      left_key.row_labels.end());
  combined_key->row_labels.insert(
      combined_key->row_labels.end(),
      right_key.row_labels.begin(),
      right_key.row_labels.end());
  combined_key->col_labels.insert(
      combined_key->col_labels.end(),
      left_key.col_labels.begin(),
      left_key.col_labels.end());
  combined_key->col_labels.insert(
      combined_key->col_labels.end(),
      right_key.col_labels.begin(),
      right_key.col_labels.end());
  return true;
}

JointDeletionPayload make_identity_joint_payload() {
  JointDeletionPayload payload;
  payload.sectors[make_joint_key(make_spin_key(), make_spin_key())] = 1.0;
  return payload;
}

bool is_joint_payload_nonzero(const JointDeletionPayload& payload) {
  for (const auto& [key, value] : payload.sectors) {
    static_cast<void>(key);
    if (std::abs(value) > 1.0e-15) {
      return true;
    }
  }
  return false;
}

void cleanup_joint_payload(JointDeletionPayload* payload) {
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

JointDeletionPayload build_joint_payload(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload) {
  JointDeletionPayload joint_payload;
  joint_payload.alpha_row_count = alpha_payload.row_count;
  joint_payload.alpha_col_count = alpha_payload.col_count;
  joint_payload.beta_row_count = beta_payload.row_count;
  joint_payload.beta_col_count = beta_payload.col_count;
  for (const auto& [alpha_key, alpha_value] : alpha_payload.sectors) {
    if (std::abs(alpha_value) <= 1.0e-15) {
      continue;
    }
    for (const auto& [beta_key, beta_value] : beta_payload.sectors) {
      if (std::abs(beta_value) <= 1.0e-15) {
        continue;
      }
      joint_payload.sectors[make_joint_key(alpha_key, beta_key)] +=
          alpha_value * beta_value;
    }
  }
  cleanup_joint_payload(&joint_payload);
  return joint_payload;
}

void add_scaled_joint_payload(
    const JointDeletionPayload& source,
    double scale,
    JointDeletionPayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  if (destination->sectors.empty()) {
    destination->alpha_row_count = source.alpha_row_count;
    destination->alpha_col_count = source.alpha_col_count;
    destination->beta_row_count = source.beta_row_count;
    destination->beta_col_count = source.beta_col_count;
  }
  for (const auto& [key, value] : source.sectors) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    destination->sectors[key] += scale * value;
  }
}

JointDeletionPayload merge_joint_deletion_payloads(
    const JointDeletionPayload& left,
    const JointDeletionPayload& right) {
  JointDeletionPayload merged;
  merged.alpha_row_count = left.alpha_row_count + right.alpha_row_count;
  merged.alpha_col_count = left.alpha_col_count + right.alpha_col_count;
  merged.beta_row_count = left.beta_row_count + right.beta_row_count;
  merged.beta_col_count = left.beta_col_count + right.beta_col_count;

  for (const auto& [left_key, left_value] : left.sectors) {
    if (std::abs(left_value) <= 1.0e-15) {
      continue;
    }
    for (const auto& [right_key, right_value] : right.sectors) {
      if (std::abs(right_value) <= 1.0e-15) {
        continue;
      }

      JointDeletionKey combined_key;
      if (!combine_spin_keys(
              left_key.alpha_key,
              right_key.alpha_key,
              &combined_key.alpha_key) ||
          !combine_spin_keys(
              left_key.beta_key,
              right_key.beta_key,
              &combined_key.beta_key)) {
        continue;
      }

      const int parity =
          spin_sector_merge_parity(
              left.alpha_row_count,
              left.alpha_col_count,
              left_key.alpha_key,
              right.alpha_row_count,
              right.alpha_col_count,
              right_key.alpha_key) ^
          spin_sector_merge_parity(
              left.beta_row_count,
              left.beta_col_count,
              left_key.beta_key,
              right.beta_row_count,
              right.beta_col_count,
              right_key.beta_key);
      merged.sectors[combined_key] +=
          parity_sign(parity) * left_value * right_value;
    }
  }
  cleanup_joint_payload(&merged);
  return merged;
}

double overlap_sector_value(const JointDeletionPayload& payload) {
  const auto iterator = payload.sectors.find(
      make_joint_key(make_spin_key(), make_spin_key()));
  return (iterator == payload.sectors.end()) ? 0.0 : iterator->second;
}

double contract_opposite_spin_first_sectors(
    const JointDeletionPayload& payload,
    const std::vector<double>& packed_active_two_electron_integrals) {
  double total = 0.0;
  for (const auto& [key, value] : payload.sectors) {
    if (key.alpha_key.row_labels.size() != 1U ||
        key.alpha_key.col_labels.size() != 1U ||
        key.beta_key.row_labels.size() != 1U ||
        key.beta_key.col_labels.size() != 1U ||
        std::abs(value) <= 1.0e-15) {
      continue;
    }
    const int eri_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            key.beta_key.row_labels[0],
            key.beta_key.col_labels[0],
            key.alpha_key.row_labels[0],
            key.alpha_key.col_labels[0]);
    total +=
        packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
        value;
  }
  return total;
}

double contract_same_spin_second_sectors(
    const JointDeletionPayload& payload,
    bool alpha_channel,
    const std::vector<double>& packed_active_two_electron_integrals) {
  double total = 0.0;
  for (const auto& [key, value] : payload.sectors) {
    const SpinDeletionKey& active_key =
        alpha_channel ? key.alpha_key : key.beta_key;
    const SpinDeletionKey& spectator_key =
        alpha_channel ? key.beta_key : key.alpha_key;
    if (active_key.row_labels.size() != 2U ||
        active_key.col_labels.size() != 2U ||
        !is_empty_spin_key(spectator_key) ||
        std::abs(value) <= 1.0e-15) {
      continue;
    }
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            active_key.row_labels[0],
            active_key.col_labels[0],
            active_key.row_labels[1],
            active_key.col_labels[1]);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            active_key.row_labels[0],
            active_key.col_labels[1],
            active_key.row_labels[1],
            active_key.col_labels[0]);
    total +=
        (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
         packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]) *
        value;
  }
  return total;
}

std::vector<TwoElectronLeafMessage> build_two_electron_leaf_messages(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* hypercube_assignment_count,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache) {
  if (subdeterminant_evaluations == nullptr || hypercube_assignment_count == nullptr) {
    throw std::invalid_argument("leaf-message counters must not be null");
  }

  std::map<
      std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
      JointDeletionPayload>
      aggregated_messages;

  for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
    for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
      ++(*hypercube_assignment_count);
      const auto alpha_payloads = build_frontier_spin_mask_payloads(
          left_root_term.alpha_occ,
          left_leaf_term.alpha_occ,
          right_root_term.alpha_occ,
          right_leaf_term.alpha_occ,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
      const auto beta_payloads = build_frontier_spin_mask_payloads(
          left_root_term.beta_occ,
          left_leaf_term.beta_occ,
          right_root_term.beta_occ,
          right_leaf_term.beta_occ,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
      const double leaf_coefficient =
          left_leaf_term.coefficient * right_leaf_term.coefficient;
      if (std::abs(leaf_coefficient) <= 1.0e-15) {
        continue;
      }

      for (const auto& alpha_entry : alpha_payloads) {
        for (const auto& beta_entry : beta_payloads) {
          const auto key = std::make_tuple(
              alpha_entry.row_mask,
              alpha_entry.col_mask,
              beta_entry.row_mask,
              beta_entry.col_mask);
          auto& payload = aggregated_messages[key];
          const SpinDeletionPayload alpha_payload =
              spin_payload_from_partial(alpha_entry.payload);
          const SpinDeletionPayload beta_payload =
              spin_payload_from_partial(beta_entry.payload);
          const JointDeletionPayload joint_payload =
              build_joint_payload(alpha_payload, beta_payload);
          add_scaled_joint_payload(joint_payload, leaf_coefficient, &payload);
        }
      }
    }
  }

  std::vector<TwoElectronLeafMessage> messages;
  messages.reserve(aggregated_messages.size());
  for (auto& [key, payload] : aggregated_messages) {
    cleanup_joint_payload(&payload);
    if (!is_joint_payload_nonzero(payload)) {
      continue;
    }
    messages.push_back(TwoElectronLeafMessage{
        .alpha_row_mask = std::get<0>(key),
        .alpha_col_mask = std::get<1>(key),
        .beta_row_mask = std::get<2>(key),
        .beta_col_mask = std::get<3>(key),
        .payload = std::move(payload),
    });
  }
  return messages;
}

SpinDeletionPayload build_root_spin_payload(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& right_root_occ,
    std::uint32_t used_row_mask,
    std::uint32_t used_col_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache) {
  const std::uint32_t row_full_mask =
      right_root_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(right_root_occ.size())) - 1U);
  const std::uint32_t col_full_mask =
      left_root_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(left_root_occ.size())) - 1U);
  const auto root_rows = select_occ_by_mask(right_root_occ, row_full_mask ^ used_row_mask);
  const auto root_cols = select_occ_by_mask(left_root_occ, col_full_mask ^ used_col_mask);
  const PartialSpinPayload& payload = build_partial_spin_payload_cached(
      root_cols,
      root_rows,
      overlap_storage,
      n_orbitals,
      0,
      0,
      false,
      PartialSide::RightComplement,
      overlap_resolver,
      subdeterminant_evaluations,
      payload_cache);
  return spin_payload_from_partial(payload);
}

JointDeletionPayload build_root_joint_payload(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache) {
  const SpinDeletionPayload alpha_payload =
      build_root_spin_payload(
          left_root_term.alpha_occ,
          right_root_term.alpha_occ,
          used_alpha_row_mask,
          used_alpha_col_mask,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
  const SpinDeletionPayload beta_payload =
      build_root_spin_payload(
          left_root_term.beta_occ,
          right_root_term.beta_occ,
          used_beta_row_mask,
          used_beta_col_mask,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
  return build_joint_payload(alpha_payload, beta_payload);
}

TwoElectronStarPairResult
convert_star_two_electron_result(
    const CollapsedTwoElectronStarPairStats& stats) {
  TwoElectronStarPairResult result;
  result.overlap = stats.collapsed_overlap;
  result.same_spin_alpha_two_electron =
      stats.collapsed_same_spin_alpha_two_electron;
  result.same_spin_beta_two_electron =
      stats.collapsed_same_spin_beta_two_electron;
  result.opposite_spin_two_electron =
      stats.collapsed_opposite_spin_two_electron;
  result.two_electron = stats.collapsed_two_electron;
  result.collapsed_leaf_state_count = stats.collapsed_leaf_state_count;
  result.hypercube_assignment_count = stats.hypercube_assignment_count;
  result.subdeterminant_evaluations = stats.subdeterminant_evaluations;
  result.dp_transition_count = stats.dp_transition_count;
  return result;
}

CollapsedTwoElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_impl(
    bool compute_exact_reference,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }
  if (ordered_components.size() == 1U) {
    const ExactTwoElectronStarPairStats exact =
        evaluate_component_ordered_open_state_star_pair_two_electron_exact(
            support_overlap_storage,
            packed_active_two_electron_integrals,
            support_size,
            ordered_components,
            overlap_resolver);
    CollapsedTwoElectronStarPairStats stats;
    stats.exact_overlap = exact.exact_overlap;
    stats.collapsed_overlap = exact.exact_overlap;
    stats.exact_same_spin_alpha_two_electron =
        exact.exact_same_spin_alpha_two_electron;
    stats.collapsed_same_spin_alpha_two_electron =
        exact.exact_same_spin_alpha_two_electron;
    stats.exact_same_spin_beta_two_electron =
        exact.exact_same_spin_beta_two_electron;
    stats.collapsed_same_spin_beta_two_electron =
        exact.exact_same_spin_beta_two_electron;
    stats.exact_opposite_spin_two_electron =
        exact.exact_opposite_spin_two_electron;
    stats.collapsed_opposite_spin_two_electron =
        exact.exact_opposite_spin_two_electron;
    stats.exact_two_electron = exact.exact_two_electron;
    stats.collapsed_two_electron = exact.exact_two_electron;
    return stats;
  }
  if (ordered_components.size() == 2U) {
    CollapsedTwoElectronStarPairStats stats;
    if (compute_exact_reference) {
      const CollapsedTwoElectronOneLeafStarPairStats one_leaf_stats =
          evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
              support_overlap_storage,
              packed_active_two_electron_integrals,
              support_size,
              ordered_components,
              overlap_resolver);
      stats.exact_overlap = one_leaf_stats.exact_overlap;
      stats.collapsed_overlap = one_leaf_stats.collapsed_overlap;
      stats.overlap_absolute_error = one_leaf_stats.overlap_absolute_error;
      stats.exact_same_spin_alpha_two_electron =
          one_leaf_stats.exact_same_spin_alpha_two_electron;
      stats.collapsed_same_spin_alpha_two_electron =
          one_leaf_stats.collapsed_same_spin_alpha_two_electron;
      stats.same_spin_alpha_absolute_error =
          one_leaf_stats.same_spin_alpha_absolute_error;
      stats.exact_same_spin_beta_two_electron =
          one_leaf_stats.exact_same_spin_beta_two_electron;
      stats.collapsed_same_spin_beta_two_electron =
          one_leaf_stats.collapsed_same_spin_beta_two_electron;
      stats.same_spin_beta_absolute_error =
          one_leaf_stats.same_spin_beta_absolute_error;
      stats.exact_opposite_spin_two_electron =
          one_leaf_stats.exact_opposite_spin_two_electron;
      stats.collapsed_opposite_spin_two_electron =
          one_leaf_stats.collapsed_opposite_spin_two_electron;
      stats.opposite_spin_absolute_error =
          one_leaf_stats.opposite_spin_absolute_error;
      stats.exact_two_electron = one_leaf_stats.exact_two_electron;
      stats.collapsed_two_electron = one_leaf_stats.collapsed_two_electron;
      stats.total_two_electron_absolute_error =
          one_leaf_stats.total_two_electron_absolute_error;
      stats.collapsed_leaf_state_count =
          one_leaf_stats.alpha_mask_state_count +
          one_leaf_stats.beta_mask_state_count;
      stats.hypercube_assignment_count =
          one_leaf_stats.processed_term_quadruple_count;
      stats.subdeterminant_evaluations =
          one_leaf_stats.subdeterminant_evaluations;
      stats.dp_transition_count = 0;
      return stats;
    }

    const TwoElectronStarPairResult one_leaf_result =
        evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_collapsed(
            support_overlap_storage,
            packed_active_two_electron_integrals,
            support_size,
            ordered_components,
            overlap_resolver);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    stats.exact_overlap = nan;
    stats.exact_same_spin_alpha_two_electron = nan;
    stats.exact_same_spin_beta_two_electron = nan;
    stats.exact_opposite_spin_two_electron = nan;
    stats.exact_two_electron = nan;
    stats.collapsed_overlap = one_leaf_result.overlap;
    stats.collapsed_same_spin_alpha_two_electron =
        one_leaf_result.same_spin_alpha_two_electron;
    stats.collapsed_same_spin_beta_two_electron =
        one_leaf_result.same_spin_beta_two_electron;
    stats.collapsed_opposite_spin_two_electron =
        one_leaf_result.opposite_spin_two_electron;
    stats.collapsed_two_electron = one_leaf_result.two_electron;
    stats.collapsed_leaf_state_count = one_leaf_result.collapsed_leaf_state_count;
    stats.hypercube_assignment_count = one_leaf_result.hypercube_assignment_count;
    stats.subdeterminant_evaluations = one_leaf_result.subdeterminant_evaluations;
    stats.dp_transition_count = one_leaf_result.dp_transition_count;
    return stats;
  }
  CollapsedTwoElectronStarPairStats stats;
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

  const auto& root_component = ordered_components.front();
  const int n_leaves = static_cast<int>(ordered_components.size()) - 1;
  std::vector<int> left_leaf_alpha_sizes;
  std::vector<int> left_leaf_beta_sizes;
  std::vector<int> right_leaf_alpha_sizes;
  std::vector<int> right_leaf_beta_sizes;
  left_leaf_alpha_sizes.reserve(xmvb::to_size(n_leaves));
  left_leaf_beta_sizes.reserve(xmvb::to_size(n_leaves));
  right_leaf_alpha_sizes.reserve(xmvb::to_size(n_leaves));
  right_leaf_beta_sizes.reserve(xmvb::to_size(n_leaves));
  for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
    const auto& leaf_component = ordered_components[xmvb::to_size(leaf_index + 1)];
    left_leaf_alpha_sizes.push_back(
        leaf_component.left_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.left_orientation_terms.front().alpha_occ.size()));
    left_leaf_beta_sizes.push_back(
        leaf_component.left_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.left_orientation_terms.front().beta_occ.size()));
    right_leaf_alpha_sizes.push_back(
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().alpha_occ.size()));
    right_leaf_beta_sizes.push_back(
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().beta_occ.size()));
  }

  std::unordered_map<PartialSpinPayloadCacheKey,
                     PartialSpinPayload,
                     PartialSpinPayloadCacheKeyHasher> partial_payload_cache;

  for (const auto& left_root_term : root_component.left_orientation_terms) {
    for (const auto& right_root_term : root_component.right_orientation_terms) {
      const double root_coefficient =
          left_root_term.coefficient * right_root_term.coefficient;
      if (std::abs(root_coefficient) <= 1.0e-15) {
        continue;
      }

      std::vector<std::vector<TwoElectronLeafMessage>> leaf_messages(
          xmvb::to_size(n_leaves));
      for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
        leaf_messages[xmvb::to_size(leaf_index)] =
            build_two_electron_leaf_messages(
                left_root_term,
                right_root_term,
                ordered_components[xmvb::to_size(leaf_index + 1)],
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &stats.subdeterminant_evaluations,
                &stats.hypercube_assignment_count,
                &partial_payload_cache);
        stats.collapsed_leaf_state_count += static_cast<std::uint64_t>(
            leaf_messages[xmvb::to_size(leaf_index)].size());
      }

      std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
               JointDeletionPayload>
          root_payload_cache;
      std::vector<std::uint32_t> selected_alpha_row_masks(
          xmvb::to_size(n_leaves),
          0U);
      std::vector<std::uint32_t> selected_alpha_col_masks(
          xmvb::to_size(n_leaves),
          0U);
      std::vector<std::uint32_t> selected_beta_row_masks(
          xmvb::to_size(n_leaves),
          0U);
      std::vector<std::uint32_t> selected_beta_col_masks(
          xmvb::to_size(n_leaves),
          0U);

      const auto accumulate_frontier =
          [&](const auto& self,
              int leaf_index,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              const JointDeletionPayload& frontier_payload) -> void {
            if (!is_joint_payload_nonzero(frontier_payload)) {
              return;
            }

            if (leaf_index == n_leaves) {
              const auto cache_key = std::make_tuple(
                  used_alpha_row_mask,
                  used_alpha_col_mask,
                  used_beta_row_mask,
                  used_beta_col_mask);
              auto cache_iterator = root_payload_cache.find(cache_key);
              if (cache_iterator == root_payload_cache.end()) {
                cache_iterator = root_payload_cache.emplace(
                    cache_key,
                    build_root_joint_payload(
                        left_root_term,
                        right_root_term,
                        used_alpha_row_mask,
                        used_alpha_col_mask,
                        used_beta_row_mask,
                        used_beta_col_mask,
                        support_overlap_storage,
                        support_size,
                        overlap_resolver,
                        &stats.subdeterminant_evaluations,
                        &partial_payload_cache))
                                     .first;
              }
              const auto& root_payload = cache_iterator->second;
              if (!is_joint_payload_nonzero(root_payload)) {
                return;
              }

              int parity = 0;
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.alpha_occ.size()),
                  selected_alpha_row_masks,
                  right_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.alpha_occ.size()),
                  selected_alpha_col_masks,
                  left_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.beta_occ.size()),
                  selected_beta_row_masks,
                  right_leaf_beta_sizes);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.beta_occ.size()),
                  selected_beta_col_masks,
                  left_leaf_beta_sizes);
              const double signed_coefficient =
                  root_coefficient * parity_sign(parity);

              const JointDeletionPayload full_payload =
                  merge_joint_deletion_payloads(frontier_payload, root_payload);
              const double overlap = overlap_sector_value(full_payload);
              const double same_spin_alpha =
                  contract_same_spin_second_sectors(
                      full_payload,
                      true,
                      packed_active_two_electron_integrals) *
                  1.0;
              const double same_spin_beta =
                  contract_same_spin_second_sectors(
                      full_payload,
                      false,
                      packed_active_two_electron_integrals);
              const double opposite_spin =
                  contract_opposite_spin_first_sectors(
                      full_payload,
                      packed_active_two_electron_integrals);

              stats.collapsed_overlap += signed_coefficient * overlap;
              stats.collapsed_same_spin_alpha_two_electron +=
                  signed_coefficient * same_spin_alpha;
              stats.collapsed_same_spin_beta_two_electron +=
                  signed_coefficient * same_spin_beta;
              stats.collapsed_opposite_spin_two_electron +=
                  signed_coefficient * opposite_spin;
              return;
            }

            for (const auto& message :
                 leaf_messages[xmvb::to_size(leaf_index)]) {
              if ((used_alpha_row_mask & message.alpha_row_mask) != 0U ||
                  (used_alpha_col_mask & message.alpha_col_mask) != 0U ||
                  (used_beta_row_mask & message.beta_row_mask) != 0U ||
                  (used_beta_col_mask & message.beta_col_mask) != 0U) {
                continue;
              }

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] =
                  message.alpha_row_mask;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] =
                  message.alpha_col_mask;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] =
                  message.beta_row_mask;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] =
                  message.beta_col_mask;

              ++stats.dp_transition_count;
              const JointDeletionPayload merged_payload =
                  merge_joint_deletion_payloads(frontier_payload, message.payload);
              self(
                  self,
                  leaf_index + 1,
                  used_alpha_row_mask | message.alpha_row_mask,
                  used_alpha_col_mask | message.alpha_col_mask,
                  used_beta_row_mask | message.beta_row_mask,
                  used_beta_col_mask | message.beta_col_mask,
                  merged_payload);

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] = 0U;
            }
          };

      const JointDeletionPayload identity_payload = make_identity_joint_payload();
      accumulate_frontier(
          accumulate_frontier,
          0,
          0U,
          0U,
          0U,
          0U,
          identity_payload);
    }
  }

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

CollapsedTwoElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  return evaluate_component_ordered_open_state_star_pair_two_electron_impl(
      true,
      support_overlap_storage,
      packed_active_two_electron_integrals,
      support_size,
      ordered_components,
      overlap_resolver);
}

TwoElectronStarPairResult
evaluate_component_ordered_open_state_star_pair_two_electron_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  return convert_star_two_electron_result(
      evaluate_component_ordered_open_state_star_pair_two_electron_impl(
          false,
          support_overlap_storage,
          packed_active_two_electron_integrals,
          support_size,
          ordered_components,
          overlap_resolver));
}

}  // namespace xmvb::vb::exact_separator
