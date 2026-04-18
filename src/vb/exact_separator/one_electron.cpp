#include "vb/exact_separator/one_electron.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "vb/exact_separator/leaf_coefficient_operator.hpp"
#include "vb/exact_separator/spin_state_aggregate.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

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

struct DenseMaskKey {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
};

bool operator==(const DenseMaskKey& left, const DenseMaskKey& right) {
  return left.alpha_row_mask == right.alpha_row_mask &&
      left.alpha_col_mask == right.alpha_col_mask &&
      left.beta_row_mask == right.beta_row_mask &&
      left.beta_col_mask == right.beta_col_mask;
}

struct DenseMaskKeyHasher {
  std::size_t operator()(const DenseMaskKey& key) const;
};

struct DenseStructurePayload {
  // Exact open-state payload in dense support-orbital coordinates.
  // Dimensions:
  // - open vectors: `(support_size)`
  // - cofactor matrices: `(support_size, support_size)` with
  //   rows = right/support rows and cols = left/support cols.
  double overlap = 0.0;
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  Eigen::VectorXd alpha_row_open;
  Eigen::VectorXd alpha_col_open;
  Eigen::VectorXd beta_row_open;
  Eigen::VectorXd beta_col_open;
  Eigen::MatrixXd alpha_cofactor;
  Eigen::MatrixXd beta_cofactor;
};

struct DenseStructureLeafMessage {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  DenseStructurePayload payload;
};

struct SpinRootStateKey {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
};

bool operator<(const SpinRootStateKey& left, const SpinRootStateKey& right) {
  return std::tie(left.alpha_occ, left.beta_occ) <
      std::tie(right.alpha_occ, right.beta_occ);
}

struct SpinCoupledRootChannelLayout {
  // Width-1 and width-2 root-channel layouts are the only supported reduced
  // channel tables for the current experimental one-leaf fast path.
  std::vector<SpinRootStateKey> left_states;
  std::vector<SpinRootStateKey> right_states;
  std::map<SpinRootStateKey, int> left_state_index;
  std::map<SpinRootStateKey, int> right_state_index;
  int channel_count = 0;
  bool supported = false;
};

struct SpinCoupledRootChannelScalarBasis {
  // Scalar analogue of the matrix-valued root-channel basis used by the
  // explicit one-leaf diagnostics in the analyzer.
  double b00 = 0.0;
  double b_left = 0.0;
  double b_right = 0.0;
  double b_left_right = 0.0;
};

struct MaskTupleIndexer {
  // Direct-address indexer for the four separator masks
  // `(alpha_row, alpha_col, beta_row, beta_col)`.
  //
  // The recurrence visits these states as a small dense hyper-rectangle when
  // the root component is narrow. In that regime a packed array avoids the
  // tree and tuple overhead of `std::map`.
  std::uint32_t alpha_row_limit = 1U;
  std::uint32_t alpha_col_limit = 1U;
  std::uint32_t beta_row_limit = 1U;
  std::uint32_t beta_col_limit = 1U;

  std::size_t total_state_count() const {
    return xmvb::to_size(alpha_row_limit) *
        xmvb::to_size(alpha_col_limit) *
        xmvb::to_size(beta_row_limit) *
        xmvb::to_size(beta_col_limit);
  }

  std::size_t pack(
      std::uint32_t alpha_row_mask,
      std::uint32_t alpha_col_mask,
      std::uint32_t beta_row_mask,
      std::uint32_t beta_col_mask) const {
    return ((((xmvb::to_size(alpha_row_mask) *
               xmvb::to_size(alpha_col_limit)) +
              xmvb::to_size(alpha_col_mask)) *
                 xmvb::to_size(beta_row_limit)) +
            xmvb::to_size(beta_row_mask)) *
               xmvb::to_size(beta_col_limit) +
        xmvb::to_size(beta_col_mask);
  }

  void unpack(
      std::size_t packed_state,
      std::uint32_t* alpha_row_mask,
      std::uint32_t* alpha_col_mask,
      std::uint32_t* beta_row_mask,
      std::uint32_t* beta_col_mask) const {
    if (alpha_row_mask == nullptr ||
        alpha_col_mask == nullptr ||
        beta_row_mask == nullptr ||
        beta_col_mask == nullptr) {
      throw std::invalid_argument("unpack output pointers must not be null");
    }
    *beta_col_mask =
        static_cast<std::uint32_t>(packed_state % xmvb::to_size(beta_col_limit));
    packed_state /= xmvb::to_size(beta_col_limit);
    *beta_row_mask =
        static_cast<std::uint32_t>(packed_state % xmvb::to_size(beta_row_limit));
    packed_state /= xmvb::to_size(beta_row_limit);
    *alpha_col_mask =
        static_cast<std::uint32_t>(packed_state % xmvb::to_size(alpha_col_limit));
    packed_state /= xmvb::to_size(alpha_col_limit);
    *alpha_row_mask = static_cast<std::uint32_t>(packed_state);
  }
};

struct RootMaskOccTables {
  std::vector<std::vector<int>> alpha_row_occ_by_mask;
  std::vector<std::vector<int>> alpha_col_occ_by_mask;
  std::vector<std::vector<int>> beta_row_occ_by_mask;
  std::vector<std::vector<int>> beta_col_occ_by_mask;
};

std::uint32_t open_state_mask_limit(int n_bits);

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

std::size_t DenseMaskKeyHasher::operator()(const DenseMaskKey& key) const {
  std::size_t seed = 0;
  hash_combine(&seed, xmvb::to_size(key.alpha_row_mask));
  hash_combine(&seed, xmvb::to_size(key.alpha_col_mask));
  hash_combine(&seed, xmvb::to_size(key.beta_row_mask));
  hash_combine(&seed, xmvb::to_size(key.beta_col_mask));
  return seed;
}

std::vector<int> select_occ_by_mask(
    const std::vector<int>& occ,
    std::uint32_t mask) {
  std::vector<int> selected;
  selected.reserve(occ.size());
  for (int index = 0; index < static_cast<int>(occ.size()); ++index) {
    if ((mask & (static_cast<std::uint32_t>(1) << index)) != 0U) {
      selected.push_back(occ[xmvb::to_size(index)]);
    }
  }
  return selected;
}

std::vector<std::vector<int>> build_mask_occ_table(const std::vector<int>& occ) {
  // Precompute every masked occupied-orbital subsequence once per root state.
  // The recurrence then reuses these lists instead of allocating a fresh
  // `std::vector<int>` inside each `(row_mask, col_mask)` loop iteration.
  const std::uint32_t limit = open_state_mask_limit(static_cast<int>(occ.size()));
  std::vector<std::vector<int>> occ_by_mask(xmvb::to_size(limit));
  for (std::uint32_t mask = 0; mask < limit; ++mask) {
    occ_by_mask[mask] = select_occ_by_mask(occ, mask);
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

int popcount(std::uint32_t mask) {
  int count = 0;
  while (mask != 0U) {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

int canonicalization_parity(const std::vector<int>& occupied_orbitals) {
  // The recurrence works in block order. The exact determinant reference uses
  // canonical ascending occupied-orbital order, so we need the permutation
  // parity that sorts the block-ordered list back to canonical order.
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

void insert_spin_root_state(
    const SpinRootStateKey& state,
    std::vector<SpinRootStateKey>* states,
    std::map<SpinRootStateKey, int>* state_index) {
  if (states == nullptr || state_index == nullptr) {
    throw std::invalid_argument("spin root state containers must not be null");
  }
  if (state_index->find(state) != state_index->end()) {
    return;
  }
  state_index->emplace(state, static_cast<int>(states->size()));
  states->push_back(state);
}

void finalize_spin_coupled_root_channel_layout(
    SpinCoupledRootChannelLayout* layout) {
  if (layout == nullptr) {
    throw std::invalid_argument("layout must not be null");
  }
  layout->supported =
      ((layout->left_states.size() == 1 && layout->right_states.size() == 2) ||
       (layout->left_states.size() == 2 && layout->right_states.size() == 2));
  if (!layout->supported) {
    layout->channel_count = 0;
    return;
  }
  layout->channel_count = (layout->left_states.size() == 1) ? 2 : 4;
}

SpinCoupledRootChannelLayout build_spin_coupled_root_channel_layout(
    const ComponentData& root_component) {
  SpinCoupledRootChannelLayout layout;
  for (const auto& left_root_term : root_component.left_orientation_terms) {
    insert_spin_root_state(
        SpinRootStateKey{left_root_term.alpha_occ, left_root_term.beta_occ},
        &layout.left_states,
        &layout.left_state_index);
  }
  for (const auto& right_root_term : root_component.right_orientation_terms) {
    insert_spin_root_state(
        SpinRootStateKey{right_root_term.alpha_occ, right_root_term.beta_occ},
        &layout.right_states,
        &layout.right_state_index);
  }
  finalize_spin_coupled_root_channel_layout(&layout);
  return layout;
}

void accumulate_direct_spin_coupled_root_channel_scalar_contribution(
    const SpinCoupledRootChannelLayout& layout,
    int left_state_index,
    int right_state_index,
    double contribution,
    double scale,
    bool alpha_channel,
    SpinCoupledRootChannelScalarBasis* basis) {
  // Route one exact residual leaf scalar directly into the supported Walsh
  // basis over the root-state table, so the fast path does not need the full
  // explicit root-pair matrix.
  if (basis == nullptr) {
    throw std::invalid_argument("basis must not be null");
  }
  if (!layout.supported || std::abs(contribution) <= 1.0e-15 || std::abs(scale) <= 1.0e-15) {
    return;
  }

  const double value = contribution * scale;
  const double s_l = (left_state_index == 0) ? 1.0 : -1.0;
  const double s_r = (right_state_index == 0) ? 1.0 : -1.0;
  const double odd_channel_sign = alpha_channel ? 1.0 : -1.0;
  if (layout.left_states.size() == 1) {
    basis->b00 += 0.5 * value;
    basis->b_right += 0.5 * odd_channel_sign * s_r * value;
    return;
  }

  basis->b00 += 0.25 * value;
  basis->b_left += 0.25 * odd_channel_sign * s_l * value;
  basis->b_right += 0.25 * odd_channel_sign * s_r * value;
  basis->b_left_right += 0.25 * s_l * s_r * value;
}

double contract_total_spin_coupled_root_channel_scalar_basis(
    const SpinCoupledRootChannelLayout& layout,
    const SpinCoupledRootChannelScalarBasis& basis) {
  if (!layout.supported) {
    return 0.0;
  }
  return (layout.left_states.size() == 1) ? (2.0 * basis.b00) : (4.0 * basis.b00);
}

double reconstruct_spin_coupled_root_target_scalar(
    const SpinCoupledRootChannelLayout& layout,
    int left_state_index,
    int right_state_index,
    const SpinCoupledRootChannelScalarBasis& basis,
    bool alpha_channel) {
  if (!layout.supported) {
    return 0.0;
  }

  const double s_l = (left_state_index == 0) ? 1.0 : -1.0;
  const double s_r = (right_state_index == 0) ? 1.0 : -1.0;
  if (layout.left_states.size() == 1) {
    return alpha_channel ? (basis.b00 + s_r * basis.b_right)
                         : (basis.b00 - s_r * basis.b_right);
  }
  if (alpha_channel) {
    return basis.b00 + s_l * basis.b_left + s_r * basis.b_right +
        (s_l * s_r) * basis.b_left_right;
  }
  return basis.b00 - s_l * basis.b_left - s_r * basis.b_right +
      (s_l * s_r) * basis.b_left_right;
}

double determinant_of_dense_matrix(
    const Eigen::MatrixXd& matrix,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // The hot recurrence still evaluates many small square minors. We keep the
  // math on Eigen's LU/SVD path, but avoid the extra matrix->vector flatten
  // that the legacy resolver entry point required.
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
  // Exact open-state rectangular payload:
  // - if `overlap_block` is `(n+1) x n`, the signed deleted-row minors form
  //   the one-dimensional left null vector;
  // - if `overlap_block` is `n x (n+1)`, the signed deleted-column minors form
  //   the one-dimensional right null vector.
  //
  // A direct loop over all deleted-row / deleted-column determinants costs
  // `O(n)` square determinant resolves. Here we instead:
  // 1. compute the one-dimensional null vector with one rectangular SVD;
  // 2. evaluate one anchor square minor to fix the overall scale/sign.
  //
  // This keeps the result exactly aligned with the current determinant-based
  // convention while collapsing the open-minor path to one factorization plus
  // one anchor determinant.
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

  const Eigen::VectorXd null_vector =
      rectangular_lu.kernel().col(0);
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
                minor.topRows(anchor_index) =
                    overlap_block.topRows(anchor_index);
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
                minor.leftCols(anchor_index) =
                    overlap_block.leftCols(anchor_index);
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
      determinant_of_dense_matrix(
          anchor_minor,
          overlap_resolver,
          subdeterminant_evaluations);
  return (anchor_minor_value / null_vector(anchor_index)) * null_vector;
}

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes) {
  // After component ordering, the canonicalization sign depends only on how
  // many selected root orbitals are inserted ahead of each leaf block and how
  // many root orbitals remain in the final root-remainder block.
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
      if ((mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
        block_order.push_back(root_position);
      }
    }
    for (int leaf_position = 0;
         leaf_position < leaf_occ_sizes[leaf_index];
         ++leaf_position) {
      block_order.push_back(next_leaf_label++);
    }
  }

  const std::uint32_t full_mask =
      (n_root_occ == 0) ? 0U
                        : ((static_cast<std::uint32_t>(1) << n_root_occ) - 1U);
  const std::uint32_t remainder_mask = full_mask ^ used_mask;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((remainder_mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

int one_leaf_block_parity(
    int n_root_occ,
    std::uint32_t selected_mask,
    int leaf_occ_size) {
  std::vector<int> block_order;
  block_order.reserve(xmvb::to_size(n_root_occ + leaf_occ_size));
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  for (int leaf_position = 0; leaf_position < leaf_occ_size; ++leaf_position) {
    block_order.push_back(n_root_occ + leaf_position);
  }
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (static_cast<std::uint32_t>(1) << root_position)) == 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

std::uint32_t open_state_mask_limit(int n_bits) {
  if (n_bits < 0 || n_bits >= 31) {
    throw std::invalid_argument("open-state mask width is out of range");
  }
  return (n_bits == 0) ? 1U : (static_cast<std::uint32_t>(1) << n_bits);
}

RootMaskOccTables build_root_mask_occ_tables(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term) {
  RootMaskOccTables tables;
  tables.alpha_row_occ_by_mask = build_mask_occ_table(right_root_term.alpha_occ);
  tables.alpha_col_occ_by_mask = build_mask_occ_table(left_root_term.alpha_occ);
  tables.beta_row_occ_by_mask = build_mask_occ_table(right_root_term.beta_occ);
  tables.beta_col_occ_by_mask = build_mask_occ_table(left_root_term.beta_occ);
  return tables;
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

DenseStructurePayload make_zero_dense_structure_payload(int support_size) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  DenseStructurePayload payload;
  payload.alpha_row_open = Eigen::VectorXd::Zero(support_size);
  payload.alpha_col_open = Eigen::VectorXd::Zero(support_size);
  payload.beta_row_open = Eigen::VectorXd::Zero(support_size);
  payload.beta_col_open = Eigen::VectorXd::Zero(support_size);
  payload.alpha_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  payload.beta_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  return payload;
}

void ensure_dense_payload_storage(
    const DenseStructurePayload& source,
    DenseStructurePayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (destination->alpha_row_open.size() != 0) {
    return;
  }
  destination->alpha_row_open = Eigen::VectorXd::Zero(source.alpha_row_open.size());
  destination->alpha_col_open = Eigen::VectorXd::Zero(source.alpha_col_open.size());
  destination->beta_row_open = Eigen::VectorXd::Zero(source.beta_row_open.size());
  destination->beta_col_open = Eigen::VectorXd::Zero(source.beta_col_open.size());
  destination->alpha_cofactor =
      Eigen::MatrixXd::Zero(source.alpha_cofactor.rows(), source.alpha_cofactor.cols());
  destination->beta_cofactor =
      Eigen::MatrixXd::Zero(source.beta_cofactor.rows(), source.beta_cofactor.cols());
  destination->alpha_row_count = source.alpha_row_count;
  destination->alpha_col_count = source.alpha_col_count;
  destination->beta_row_count = source.beta_row_count;
  destination->beta_col_count = source.beta_col_count;
}

void reset_dense_payload(
    int support_size,
    DenseStructurePayload* payload) {
  // Reuse already-allocated Eigen buffers along one recursion branch. This
  // keeps the merge algebra unchanged, but avoids allocating six dense
  // objects for every frontier transition.
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  if (payload->alpha_row_open.size() == 0) {
    *payload = make_zero_dense_structure_payload(support_size);
  } else {
    payload->alpha_row_open.setZero();
    payload->alpha_col_open.setZero();
    payload->beta_row_open.setZero();
    payload->beta_col_open.setZero();
    payload->alpha_cofactor.setZero();
    payload->beta_cofactor.setZero();
    payload->overlap = 0.0;
    payload->alpha_row_count = 0;
    payload->alpha_col_count = 0;
    payload->beta_row_count = 0;
    payload->beta_col_count = 0;
  }
}

bool is_payload_nonzero(const PartialSpinPayload& payload) {
  return std::abs(payload.closed_overlap) > 1.0e-15 ||
      !payload.row_open_entries.empty() ||
      !payload.col_open_entries.empty() ||
      !payload.cofactor_entries.empty();
}

bool is_square_payload(const PartialSpinPayload& payload) {
  return payload.n_rows == payload.n_cols;
}

bool is_payload_nonzero(const DenseStructurePayload& payload) {
  return std::abs(payload.overlap) > 1.0e-15 ||
      (payload.alpha_row_open.size() > 0 &&
       (payload.alpha_row_open.squaredNorm() > 1.0e-30 ||
        payload.alpha_col_open.squaredNorm() > 1.0e-30 ||
        payload.beta_row_open.squaredNorm() > 1.0e-30 ||
        payload.beta_col_open.squaredNorm() > 1.0e-30 ||
        payload.alpha_cofactor.squaredNorm() > 1.0e-30 ||
        payload.beta_cofactor.squaredNorm() > 1.0e-30));
}

double parity_sign_from_count(int count) {
  return parity_sign(count & 1);
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
    payload.row_open_entries.reserve(xmvb::to_size(payload.n_rows));
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
    payload.col_open_entries.reserve(xmvb::to_size(payload.n_cols));
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

void add_scaled_dense_payload(
    const DenseStructurePayload& source,
    double scale,
    DenseStructurePayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  ensure_dense_payload_storage(source, destination);
  destination->overlap += scale * source.overlap;
  destination->alpha_row_open.noalias() += scale * source.alpha_row_open;
  destination->alpha_col_open.noalias() += scale * source.alpha_col_open;
  destination->beta_row_open.noalias() += scale * source.beta_row_open;
  destination->beta_col_open.noalias() += scale * source.beta_col_open;
  destination->alpha_cofactor.noalias() += scale * source.alpha_cofactor;
  destination->beta_cofactor.noalias() += scale * source.beta_cofactor;
}

DenseStructurePayload dense_structure_payload_from_spin_partials(
    const PartialSpinPayload& alpha,
    const PartialSpinPayload& beta,
    int support_size) {
  DenseStructurePayload payload = make_zero_dense_structure_payload(support_size);
  payload.alpha_row_count = alpha.n_rows;
  payload.alpha_col_count = alpha.n_cols;
  payload.beta_row_count = beta.n_rows;
  payload.beta_col_count = beta.n_cols;
  const bool alpha_square = is_square_payload(alpha);
  const bool beta_square = is_square_payload(beta);

  if (alpha_square && beta_square) {
    payload.overlap = alpha.closed_overlap * beta.closed_overlap;
  }
  if (beta_square) {
    for (const auto& entry : alpha.row_open_entries) {
      payload.alpha_row_open(entry.orbital_label) +=
          entry.value * beta.closed_overlap;
    }
    for (const auto& entry : alpha.col_open_entries) {
      payload.alpha_col_open(entry.orbital_label) +=
          entry.value * beta.closed_overlap;
    }
    for (const auto& entry : alpha.cofactor_entries) {
      payload.alpha_cofactor(entry.row_orbital_label, entry.col_orbital_label) +=
          entry.value * beta.closed_overlap;
    }
  }
  if (alpha_square) {
    for (const auto& entry : beta.row_open_entries) {
      payload.beta_row_open(entry.orbital_label) +=
          entry.value * alpha.closed_overlap;
    }
    for (const auto& entry : beta.col_open_entries) {
      payload.beta_col_open(entry.orbital_label) +=
          entry.value * alpha.closed_overlap;
    }
    for (const auto& entry : beta.cofactor_entries) {
      payload.beta_cofactor(entry.row_orbital_label, entry.col_orbital_label) +=
          entry.value * alpha.closed_overlap;
    }
  }
  return payload;
}

void merge_dense_structure_payloads(
    const DenseStructurePayload& left,
    const DenseStructurePayload& right,
    DenseStructurePayload* merged) {
  // This is the exact tensor merge for one separator state. The implementation
  // writes into a caller-owned payload so the recursion can reuse one buffer
  // per depth instead of allocating a fresh dense payload for every branch.
  if (merged == nullptr) {
    throw std::invalid_argument("merged must not be null");
  }
  reset_dense_payload(static_cast<int>(left.alpha_row_open.size()), merged);
  merged->overlap = left.overlap * right.overlap;
  merged->alpha_row_count = left.alpha_row_count + right.alpha_row_count;
  merged->alpha_col_count = left.alpha_col_count + right.alpha_col_count;
  merged->beta_row_count = left.beta_row_count + right.beta_row_count;
  merged->beta_col_count = left.beta_col_count + right.beta_col_count;

  merged->alpha_row_open.noalias() =
      parity_sign_from_count(right.alpha_col_count) * right.overlap * left.alpha_row_open +
      parity_sign_from_count(left.alpha_row_count + left.alpha_col_count) *
          left.overlap * right.alpha_row_open;
  merged->alpha_col_open.noalias() =
      parity_sign_from_count(right.alpha_row_count) * right.overlap * left.alpha_col_open +
      parity_sign_from_count(left.alpha_row_count + left.alpha_col_count) *
          left.overlap * right.alpha_col_open;
  merged->beta_row_open.noalias() =
      parity_sign_from_count(right.beta_col_count) * right.overlap * left.beta_row_open +
      parity_sign_from_count(left.beta_row_count + left.beta_col_count) *
          left.overlap * right.beta_row_open;
  merged->beta_col_open.noalias() =
      parity_sign_from_count(right.beta_row_count) * right.overlap * left.beta_col_open +
      parity_sign_from_count(left.beta_row_count + left.beta_col_count) *
          left.overlap * right.beta_col_open;
  merged->alpha_cofactor.noalias() =
      right.overlap * left.alpha_cofactor +
      parity_sign_from_count(left.alpha_row_count + left.alpha_col_count) *
          left.overlap * right.alpha_cofactor;
  merged->beta_cofactor.noalias() =
      right.overlap * left.beta_cofactor +
      parity_sign_from_count(left.beta_row_count + left.beta_col_count) *
          left.overlap * right.beta_cofactor;

  merged->alpha_cofactor.noalias() +=
      parity_sign_from_count(
          left.alpha_row_count + left.alpha_col_count + right.alpha_col_count) *
      (left.alpha_row_open * right.alpha_col_open.transpose());
  merged->alpha_cofactor.noalias() +=
      parity_sign_from_count(
          left.alpha_row_count + left.alpha_col_count + right.alpha_row_count) *
      (right.alpha_row_open * left.alpha_col_open.transpose());
  merged->beta_cofactor.noalias() +=
      parity_sign_from_count(
          left.beta_row_count + left.beta_col_count + right.beta_col_count) *
      (left.beta_row_open * right.beta_col_open.transpose());
  merged->beta_cofactor.noalias() +=
      parity_sign_from_count(
          left.beta_row_count + left.beta_col_count + right.beta_row_count) *
      (right.beta_row_open * left.beta_col_open.transpose());
}

double contract_dense_one_electron_merge(
    const DenseStructurePayload& frontier,
    const DenseStructurePayload& root,
    const Eigen::MatrixXd& one_electron_matrix) {
  double total = 0.0;
  total += frontier.alpha_cofactor.cwiseProduct(one_electron_matrix).sum() * root.overlap;
  total += frontier.overlap * root.alpha_cofactor.cwiseProduct(one_electron_matrix).sum();
  total += frontier.alpha_row_open.dot(one_electron_matrix * root.alpha_col_open);
  total += root.alpha_row_open.dot(one_electron_matrix * frontier.alpha_col_open);
  total += frontier.beta_cofactor.cwiseProduct(one_electron_matrix).sum() * root.overlap;
  total += frontier.overlap * root.beta_cofactor.cwiseProduct(one_electron_matrix).sum();
  total += frontier.beta_row_open.dot(one_electron_matrix * root.beta_col_open);
  total += root.beta_row_open.dot(one_electron_matrix * frontier.beta_col_open);
  return total;
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
                       PartialSpinPayloadCacheKeyHasher>* payload_cache = nullptr) {
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
      assign_concatenated_occ(
          selected_root_cols,
          left_leaf_occ,
          &frontier_left_occ);
      assign_concatenated_occ(
          selected_root_rows,
          right_leaf_occ,
          &frontier_right_occ);
      PartialSpinPayload payload =
          (payload_cache == nullptr)
              ? build_partial_spin_payload(
                    frontier_left_occ,
                    frontier_right_occ,
                    overlap_storage,
                    n_orbitals,
                    static_cast<int>(selected_root_rows.size()),
                    static_cast<int>(selected_root_cols.size()),
                    true,
                    PartialSide::LeftFrontier,
                    overlap_resolver,
                    subdeterminant_evaluations)
              : build_partial_spin_payload_cached(
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
      if (!is_payload_nonzero(payload)) {
        continue;
      }
      entries.push_back({row_mask, col_mask, std::move(payload)});
    }
  }
  return entries;
}

std::vector<DenseStructureLeafMessage> build_dense_leaf_messages(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* hypercube_assignment_count,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache = nullptr) {
  if (subdeterminant_evaluations == nullptr || hypercube_assignment_count == nullptr) {
    throw std::invalid_argument("leaf-message counters must not be null");
  }

  constexpr std::size_t k_max_dense_mask_states = 16384;
  const MaskTupleIndexer indexer{
      open_state_mask_limit(static_cast<int>(right_root_term.alpha_occ.size())),
      open_state_mask_limit(static_cast<int>(left_root_term.alpha_occ.size())),
      open_state_mask_limit(static_cast<int>(right_root_term.beta_occ.size())),
      open_state_mask_limit(static_cast<int>(left_root_term.beta_occ.size())),
  };
  const bool use_dense_mask_table =
      indexer.total_state_count() <= k_max_dense_mask_states;
  std::vector<DenseStructurePayload> dense_messages;
  std::vector<std::uint8_t> dense_message_touched;
  std::vector<std::size_t> dense_message_indices;
  std::unordered_map<DenseMaskKey, DenseStructurePayload, DenseMaskKeyHasher>
      sparse_messages;
  if (use_dense_mask_table) {
    dense_messages.resize(indexer.total_state_count());
    dense_message_touched.assign(indexer.total_state_count(), static_cast<std::uint8_t>(0));
  }

  for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
    for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
      ++(*hypercube_assignment_count);
      const double leaf_coefficient =
          left_leaf_term.coefficient * right_leaf_term.coefficient;
      if (std::abs(leaf_coefficient) <= 1.0e-15) {
        continue;
      }
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
      for (const auto& alpha_entry : alpha_payloads) {
        for (const auto& beta_entry : beta_payloads) {
          const DenseStructurePayload term_payload =
              dense_structure_payload_from_spin_partials(
                  alpha_entry.payload,
                  beta_entry.payload,
                  support_size);
          if (!is_payload_nonzero(term_payload)) {
            continue;
          }
          if (use_dense_mask_table) {
            const std::size_t packed_state = indexer.pack(
                alpha_entry.row_mask,
                alpha_entry.col_mask,
                beta_entry.row_mask,
                beta_entry.col_mask);
            if (dense_message_touched[packed_state] == 0U) {
              dense_message_touched[packed_state] = 1U;
              dense_message_indices.push_back(packed_state);
            }
            add_scaled_dense_payload(
                term_payload,
                leaf_coefficient,
                &dense_messages[packed_state]);
          } else {
            const DenseMaskKey key{
                alpha_entry.row_mask,
                alpha_entry.col_mask,
                beta_entry.row_mask,
                beta_entry.col_mask,
            };
            add_scaled_dense_payload(
                term_payload,
                leaf_coefficient,
                &sparse_messages[key]);
          }
        }
      }
    }
  }

  std::vector<DenseStructureLeafMessage> messages;
  if (use_dense_mask_table) {
    messages.reserve(dense_message_indices.size());
    for (const std::size_t packed_state : dense_message_indices) {
      DenseStructurePayload& payload = dense_messages[packed_state];
      if (!is_payload_nonzero(payload)) {
        continue;
      }
      DenseStructureLeafMessage message;
      indexer.unpack(
          packed_state,
          &message.alpha_row_mask,
          &message.alpha_col_mask,
          &message.beta_row_mask,
          &message.beta_col_mask);
      message.payload = std::move(payload);
      messages.push_back(std::move(message));
    }
  } else {
    messages.reserve(sparse_messages.size());
    for (auto& [key, payload] : sparse_messages) {
      if (!is_payload_nonzero(payload)) {
        continue;
      }
      DenseStructureLeafMessage message;
      message.alpha_row_mask = key.alpha_row_mask;
      message.alpha_col_mask = key.alpha_col_mask;
      message.beta_row_mask = key.beta_row_mask;
      message.beta_col_mask = key.beta_col_mask;
      message.payload = std::move(payload);
      messages.push_back(std::move(message));
    }
  }
  return messages;
}

DenseStructurePayload build_dense_root_payload(
    const RootMaskOccTables& root_mask_occ_tables,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<PartialSpinPayloadCacheKey,
                       PartialSpinPayload,
                       PartialSpinPayloadCacheKeyHasher>* payload_cache = nullptr) {
  const std::uint32_t alpha_row_full_mask = static_cast<std::uint32_t>(
      root_mask_occ_tables.alpha_row_occ_by_mask.size() - 1U);
  const std::uint32_t alpha_col_full_mask = static_cast<std::uint32_t>(
      root_mask_occ_tables.alpha_col_occ_by_mask.size() - 1U);
  const std::uint32_t beta_row_full_mask = static_cast<std::uint32_t>(
      root_mask_occ_tables.beta_row_occ_by_mask.size() - 1U);
  const std::uint32_t beta_col_full_mask = static_cast<std::uint32_t>(
      root_mask_occ_tables.beta_col_occ_by_mask.size() - 1U);

  const auto& alpha_root_rows = root_mask_occ_tables.alpha_row_occ_by_mask
      [xmvb::to_size(alpha_row_full_mask ^ used_alpha_row_mask)];
  const auto& alpha_root_cols = root_mask_occ_tables.alpha_col_occ_by_mask
      [xmvb::to_size(alpha_col_full_mask ^ used_alpha_col_mask)];
  const auto& beta_root_rows = root_mask_occ_tables.beta_row_occ_by_mask
      [xmvb::to_size(beta_row_full_mask ^ used_beta_row_mask)];
  const auto& beta_root_cols = root_mask_occ_tables.beta_col_occ_by_mask
      [xmvb::to_size(beta_col_full_mask ^ used_beta_col_mask)];

  const PartialSpinPayload alpha_payload =
      (payload_cache == nullptr)
          ? build_partial_spin_payload(
                alpha_root_cols,
                alpha_root_rows,
                overlap_storage,
                n_orbitals,
                0,
                0,
                false,
                PartialSide::RightComplement,
                overlap_resolver,
                subdeterminant_evaluations)
          : build_partial_spin_payload_cached(
                alpha_root_cols,
                alpha_root_rows,
                overlap_storage,
                n_orbitals,
                0,
                0,
                false,
                PartialSide::RightComplement,
                overlap_resolver,
                subdeterminant_evaluations,
                payload_cache);
  const PartialSpinPayload beta_payload =
      (payload_cache == nullptr)
          ? build_partial_spin_payload(
                beta_root_cols,
                beta_root_rows,
                overlap_storage,
                n_orbitals,
                0,
                0,
                false,
                PartialSide::RightComplement,
                overlap_resolver,
                subdeterminant_evaluations)
          : build_partial_spin_payload_cached(
                beta_root_cols,
                beta_root_rows,
                overlap_storage,
                n_orbitals,
                0,
                0,
                false,
                PartialSide::RightComplement,
                overlap_resolver,
                subdeterminant_evaluations,
                payload_cache);
  return dense_structure_payload_from_spin_partials(alpha_payload, beta_payload, support_size);
}

std::optional<CollapsedOneElectronStarPairStats>
evaluate_component_ordered_channel_state_star_pair_one_electron_covered_one_leaf(
    double exact_overlap,
    double exact_one_electron,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const ComponentData& root_component,
    const ComponentData& leaf_component,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  // Exact one-leaf degree-1 boundary path:
  // 1. compress root and leaf orientation-term products into sparse
  //    coefficient operators;
  // 2. build each unique one-spin root/leaf state pair through the exact
  //    degree-0/1 boundary aggregate exported from the one-leaf separator
  //    message algebra;
  // 3. contract overlap and one-electron scalars through the compressed root /
  //    leaf operators instead of traversing the raw term hypercube.
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }

  CollapsedOneElectronStarPairStats stats;
  stats.exact_overlap = exact_overlap;
  stats.exact_one_electron = exact_one_electron;
  const ComponentSpinCoefficientOperator root_operator =
      build_component_spin_coefficient_operator(root_component);
  const ComponentSpinCoefficientOperator leaf_operator =
      build_component_spin_coefficient_operator(leaf_component);
  stats.hypercube_assignment_count =
      root_operator.raw_nonzero_pair_count *
      leaf_operator.raw_nonzero_pair_count;
  const std::vector<double> empty_packed_active_two_electron_integrals;

  const IndexedOneLeafDirectSpinAggregateTable alpha_table =
      build_indexed_one_leaf_direct_spin_aggregate_table(
          root_operator.alpha_states,
          leaf_operator.alpha_states,
          support_overlap_storage,
          support_one_electron_storage,
          empty_packed_active_two_electron_integrals,
          support_size,
          false,
          false,
          false,
          overlap_resolver,
          &stats.subdeterminant_evaluations);
  const IndexedOneLeafDirectSpinAggregateTable beta_table =
      build_indexed_one_leaf_direct_spin_aggregate_table(
          root_operator.beta_states,
          leaf_operator.beta_states,
          support_overlap_storage,
          support_one_electron_storage,
          empty_packed_active_two_electron_integrals,
          support_size,
          false,
          false,
          false,
          overlap_resolver,
          &stats.subdeterminant_evaluations);

  stats.collapsed_leaf_state_count =
      alpha_table.total_state_count + beta_table.total_state_count;
  stats.dp_transition_count = 0;

  const OneLeafDirectAggregateContractionResult contraction =
      contract_one_leaf_direct_aggregate_channels(
          root_operator,
          leaf_operator,
          alpha_table,
          beta_table,
          nullptr);
  stats.collapsed_overlap = contraction.overlap;
  stats.collapsed_one_electron = contraction.one_electron;
  stats.constructed_one_electron = stats.collapsed_one_electron;
  if (std::isfinite(stats.exact_overlap)) {
    stats.overlap_absolute_error =
        std::abs(stats.exact_overlap - stats.collapsed_overlap);
  } else {
    stats.overlap_absolute_error = 0.0;
  }
  if (std::isfinite(stats.exact_one_electron)) {
    stats.one_electron_absolute_error =
        std::abs(stats.exact_one_electron - stats.collapsed_one_electron);
    stats.constructed_one_electron_absolute_error =
        std::abs(stats.exact_one_electron - stats.constructed_one_electron);
    if (stats.one_electron_absolute_error > 1.0e-10) {
      return std::nullopt;
    }
  } else {
    stats.one_electron_absolute_error = 0.0;
    stats.constructed_one_electron_absolute_error = 0.0;
  }
  return stats;
}

}  // namespace

CollapsedOneElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_one_electron(
    double exact_overlap,
    double exact_one_electron,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  // Exact multi-leaf open-state recurrence. The propagated leaf payload
  // carries unresolved row/column deletions and cofactors rather than the old
  // scalar `(overlap, H_alpha, H_beta)` hypothesis.
  CollapsedOneElectronStarPairStats stats;
  stats.exact_overlap = exact_overlap;
  stats.exact_one_electron = exact_one_electron;
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (support_overlap_storage.size() !=
      xmvb::to_size(support_size * support_size)) {
    throw std::invalid_argument(
        "support_overlap_storage must hold a square support-space matrix");
  }
  if (support_one_electron_storage.size() !=
      xmvb::to_size(support_size * support_size)) {
    throw std::invalid_argument(
        "support_one_electron_storage must hold a square support-space matrix");
  }
  const auto& root_component = ordered_components.front();
  const int n_leaves = static_cast<int>(ordered_components.size()) - 1;
  if (n_leaves == 0) {
    stats.collapsed_overlap = exact_overlap;
    stats.collapsed_one_electron = exact_one_electron;
    stats.constructed_one_electron = exact_one_electron;
    return stats;
  }
  if (n_leaves == 1) {
    const auto fast_path_stats =
        evaluate_component_ordered_channel_state_star_pair_one_electron_covered_one_leaf(
            exact_overlap,
            exact_one_electron,
            support_overlap_storage,
            support_one_electron_storage,
            support_size,
            root_component,
            ordered_components[1],
            overlap_resolver);
    if (fast_path_stats.has_value()) {
      return *fast_path_stats;
    }
  }

  const Eigen::MatrixXd support_one_electron_matrix =
      Eigen::Map<const Eigen::MatrixXd>(
          support_one_electron_storage.data(),
          support_size,
          support_size)
          .eval();

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
      const RootMaskOccTables root_mask_occ_tables =
          build_root_mask_occ_tables(left_root_term, right_root_term);
      std::vector<std::vector<DenseStructureLeafMessage>> leaf_messages(
          xmvb::to_size(n_leaves));
      for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
        leaf_messages[xmvb::to_size(leaf_index)] = build_dense_leaf_messages(
            left_root_term,
            right_root_term,
            ordered_components[xmvb::to_size(leaf_index + 1)],
            support_overlap_storage,
            support_size,
            support_size,
            overlap_resolver,
            &stats.subdeterminant_evaluations,
            &stats.hypercube_assignment_count,
            &partial_payload_cache);
        stats.collapsed_leaf_state_count += static_cast<std::uint64_t>(
            leaf_messages[xmvb::to_size(leaf_index)].size());
      }

      std::unordered_map<DenseMaskKey, DenseStructurePayload, DenseMaskKeyHasher>
          sparse_root_payload_cache;
      constexpr std::size_t k_max_dense_root_states = 16384;
      const MaskTupleIndexer root_indexer{
          open_state_mask_limit(static_cast<int>(right_root_term.alpha_occ.size())),
          open_state_mask_limit(static_cast<int>(left_root_term.alpha_occ.size())),
          open_state_mask_limit(static_cast<int>(right_root_term.beta_occ.size())),
          open_state_mask_limit(static_cast<int>(left_root_term.beta_occ.size())),
      };
      const bool use_dense_root_cache =
          root_indexer.total_state_count() <= k_max_dense_root_states;
      std::vector<DenseStructurePayload> dense_root_payload_cache;
      std::vector<std::uint8_t> dense_root_payload_ready;
      if (use_dense_root_cache) {
        dense_root_payload_cache.resize(root_indexer.total_state_count());
        dense_root_payload_ready.assign(
            root_indexer.total_state_count(),
            static_cast<std::uint8_t>(0));
      }
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
      std::vector<DenseStructurePayload> frontier_stack;
      frontier_stack.reserve(xmvb::to_size(n_leaves + 1));
      for (int depth = 0; depth <= n_leaves; ++depth) {
        frontier_stack.push_back(make_zero_dense_structure_payload(support_size));
      }
      frontier_stack.front().overlap = 1.0;

      const auto accumulate_frontier =
          [&](const auto& self,
              int leaf_index,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              const DenseStructurePayload& frontier_payload) -> void {
            if (!is_payload_nonzero(frontier_payload)) {
              return;
            }
            if (leaf_index == n_leaves) {
              const DenseStructurePayload* root_payload = nullptr;
              if (use_dense_root_cache) {
                const std::size_t packed_state = root_indexer.pack(
                    used_alpha_row_mask,
                    used_alpha_col_mask,
                    used_beta_row_mask,
                    used_beta_col_mask);
                if (dense_root_payload_ready[packed_state] == 0U) {
                  dense_root_payload_cache[packed_state] = build_dense_root_payload(
                      root_mask_occ_tables,
                      used_alpha_row_mask,
                      used_alpha_col_mask,
                      used_beta_row_mask,
                      used_beta_col_mask,
                      support_overlap_storage,
                      support_size,
                      support_size,
                      overlap_resolver,
                      &stats.subdeterminant_evaluations,
                      &partial_payload_cache);
                  dense_root_payload_ready[packed_state] = 1U;
                }
                root_payload = &dense_root_payload_cache[packed_state];
              } else {
                const DenseMaskKey cache_key{
                    used_alpha_row_mask,
                    used_alpha_col_mask,
                    used_beta_row_mask,
                    used_beta_col_mask,
                };
                auto cache_iterator = sparse_root_payload_cache.find(cache_key);
                if (cache_iterator == sparse_root_payload_cache.end()) {
                  cache_iterator = sparse_root_payload_cache.emplace(
                      cache_key,
                      build_dense_root_payload(
                          root_mask_occ_tables,
                          used_alpha_row_mask,
                          used_alpha_col_mask,
                          used_beta_row_mask,
                          used_beta_col_mask,
                          support_overlap_storage,
                          support_size,
                          support_size,
                          overlap_resolver,
                          &stats.subdeterminant_evaluations,
                          &partial_payload_cache))
                                       .first;
                }
                root_payload = &cache_iterator->second;
              }
              if (!is_payload_nonzero(*root_payload)) {
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

              stats.collapsed_overlap +=
                  signed_coefficient * frontier_payload.overlap * root_payload->overlap;
              stats.collapsed_one_electron +=
                  signed_coefficient *
                  contract_dense_one_electron_merge(
                      frontier_payload,
                      *root_payload,
                      support_one_electron_matrix);
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
              DenseStructurePayload& merged_payload =
                  frontier_stack[xmvb::to_size(leaf_index + 1)];
              merge_dense_structure_payloads(
                  frontier_payload,
                  message.payload,
                  &merged_payload);
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

      accumulate_frontier(
          accumulate_frontier,
          0,
          0U,
          0U,
          0U,
          0U,
          frontier_stack.front());
    }
  }

  stats.constructed_one_electron = stats.collapsed_one_electron;
  stats.overlap_absolute_error =
      std::abs(stats.exact_overlap - stats.collapsed_overlap);
  stats.one_electron_absolute_error =
      std::abs(stats.exact_one_electron - stats.collapsed_one_electron);
  stats.constructed_one_electron_absolute_error =
      std::abs(stats.exact_one_electron - stats.constructed_one_electron);
  return stats;
}

OneElectronStarPairResult
evaluate_component_ordered_open_state_star_pair_one_electron_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const CollapsedOneElectronStarPairStats stats =
      evaluate_component_ordered_open_state_star_pair_one_electron(
          nan,
          nan,
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          ordered_components,
          overlap_resolver);

  OneElectronStarPairResult result;
  result.overlap = stats.collapsed_overlap;
  result.one_electron = stats.collapsed_one_electron;
  result.collapsed_leaf_state_count = stats.collapsed_leaf_state_count;
  result.hypercube_assignment_count = stats.hypercube_assignment_count;
  result.subdeterminant_evaluations = stats.subdeterminant_evaluations;
  result.dp_transition_count = stats.dp_transition_count;
  return result;
}

}  // namespace xmvb::vb::exact_separator
