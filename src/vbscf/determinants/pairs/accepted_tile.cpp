#include "vbscf/determinants/pairs/accepted_tile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "vbscf/determinants/pairs/contractions.hpp"

namespace xmvb::vb {
namespace {

struct SpinStringHash {
  std::size_t operator()(const std::vector<int>& occupied) const noexcept {
    std::size_t value = occupied.size();
    for (const int orbital : occupied) {
      value = value * 1315423911u + static_cast<std::size_t>(orbital + 257);
    }
    return value;
  }
};

struct TileTraversal {
  std::vector<int> order;
  std::vector<int> parent;
};

struct OverlapTraversalState {
  std::vector<int> right_internal_order;
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd inverse;
  double determinant = 0.0;
  double determinant_sign = 0.0;
  double log_abs_determinant =
      -std::numeric_limits<double>::infinity();
  int updates_since_anchor = 0;
  bool regular = false;
};

std::vector<std::vector<int>> build_substitution_graph(
    const std::vector<std::vector<int>>& strings,
    int n_active_orbitals) {
  std::unordered_map<std::vector<int>, int, SpinStringHash> index;
  index.reserve(strings.size());
  for (int id = 0; id < static_cast<int>(strings.size()); ++id) {
    index.emplace(strings[id], id);
  }

  std::vector<std::vector<int>> graph(strings.size());
  for (int id = 0; id < static_cast<int>(strings.size()); ++id) {
    const auto& occupied = strings[id];
    std::vector<unsigned char> is_occupied(n_active_orbitals, 0u);
    for (const int orbital : occupied) {
      if (orbital < 0 || orbital >= n_active_orbitals) {
        throw std::out_of_range("spin string orbital is outside active space");
      }
      is_occupied[orbital] = 1u;
    }
    for (int slot = 0; slot < static_cast<int>(occupied.size()); ++slot) {
      for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
        if (is_occupied[orbital] != 0u) {
          continue;
        }
        std::vector<int> neighbor = occupied;
        neighbor[slot] = orbital;
        std::sort(neighbor.begin(), neighbor.end());
        const auto found = index.find(neighbor);
        if (found != index.end() && found->second != id) {
          graph[id].push_back(found->second);
        }
      }
    }
    std::sort(graph[id].begin(), graph[id].end());
    graph[id].erase(
        std::unique(graph[id].begin(), graph[id].end()), graph[id].end());
  }
  return graph;
}

TileTraversal build_tile_traversal(
    const std::vector<std::vector<int>>& graph,
    int begin,
    int end) {
  const int size = end - begin;
  TileTraversal result;
  result.parent.assign(size, -2);
  result.order.reserve(size);
  std::queue<int> pending;
  for (int root = 0; root < size; ++root) {
    if (result.parent[root] != -2) {
      continue;
    }
    result.parent[root] = -1;
    pending.push(root);
    while (!pending.empty()) {
      const int local = pending.front();
      pending.pop();
      result.order.push_back(local);
      const int global = begin + local;
      for (const int neighbor : graph[global]) {
        if (neighbor < begin || neighbor >= end) {
          continue;
        }
        const int neighbor_local = neighbor - begin;
        if (result.parent[neighbor_local] != -2) {
          continue;
        }
        result.parent[neighbor_local] = local;
        pending.push(neighbor_local);
      }
    }
  }
  return result;
}

bool find_single_replacement(
    const std::vector<int>& current_internal,
    const std::vector<int>& next_sorted,
    int* slot,
    int* old_orbital,
    int* new_orbital) {
  *slot = -1;
  *old_orbital = -1;
  *new_orbital = -1;
  for (int current_slot = 0;
       current_slot < static_cast<int>(current_internal.size());
       ++current_slot) {
    const int orbital = current_internal[current_slot];
    if (!std::binary_search(next_sorted.begin(), next_sorted.end(), orbital)) {
      if (*slot >= 0) {
        return false;
      }
      *slot = current_slot;
      *old_orbital = orbital;
    }
  }
  for (const int orbital : next_sorted) {
    if (std::find(current_internal.begin(), current_internal.end(), orbital) ==
        current_internal.end()) {
      if (*new_orbital >= 0) {
        return false;
      }
      *new_orbital = orbital;
    }
  }
  return *slot >= 0 && *new_orbital >= 0;
}

int permutation_sign_to_sorted(const std::vector<int>& internal_order) {
  int inversions = 0;
  for (int left = 0; left < static_cast<int>(internal_order.size()); ++left) {
    for (int right = left + 1;
         right < static_cast<int>(internal_order.size());
         ++right) {
      inversions += internal_order[left] > internal_order[right] ? 1 : 0;
    }
  }
  return (inversions % 2 == 0) ? 1 : -1;
}

OverlapTraversalState anchor_state(
    const std::vector<int>& occ_left,
    const std::vector<int>& occ_right,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    const DeterminantOverlapResolver& resolver) {
  OverlapTraversalState state;
  state.right_internal_order = occ_right;
  if (occ_left.empty()) {
    state.overlap.resize(0, 0);
    state.inverse.resize(0, 0);
    state.determinant = 1.0;
    state.determinant_sign = 1.0;
    state.log_abs_determinant = 0.0;
    state.regular = true;
    return state;
  }
  state.overlap = build_overlap_submatrix(
      occ_left, occ_right, active_overlap, n_active_orbitals);
  const DeterminantOverlapResult resolved =
      resolver.resolve_matrix(state.overlap);
  state.regular = resolved.nullity == 0;
  if (state.regular) {
    state.inverse = resolved.inverse_overlap_submatrix;
    state.determinant = resolved.overlap_determinant;
    state.determinant_sign = resolved.determinant_sign;
    state.log_abs_determinant = resolved.log_abs_determinant;
  }
  return state;
}

double inverse_backward_error(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse) {
  if (matrix.rows() == 0) {
    return 0.0;
  }
  const Eigen::MatrixXd residual =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()) -
      matrix * inverse;
  const auto infinity_norm = [](const Eigen::Ref<const Eigen::MatrixXd>& value) {
    return value.cwiseAbs().rowwise().sum().maxCoeff();
  };
  const double scale = std::max(
      1.0, infinity_norm(matrix) * infinity_norm(inverse));
  return infinity_norm(residual) / scale;
}

double rounding_bound(int dimension, int updates) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double operations = static_cast<double>(
      std::max(1, 8 * dimension * (updates + 1)));
  const double product = operations * epsilon;
  return product < 1.0 ? product / (1.0 - product) : 1.0;
}

bool woodbury_child_state(
    const OverlapTraversalState& parent,
    const std::vector<int>& occ_left,
    const std::vector<int>& child_sorted,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    OverlapTraversalState* child) {
  if (!parent.regular || child == nullptr) {
    return false;
  }
  int slot = -1;
  int old_orbital = -1;
  int new_orbital = -1;
  if (!find_single_replacement(
          parent.right_internal_order,
          child_sorted,
          &slot,
          &old_orbital,
          &new_orbital)) {
    return false;
  }

  *child = parent;
  child->right_internal_order[slot] = new_orbital;
  child->overlap = build_overlap_submatrix(
      occ_left,
      child->right_internal_order,
      active_overlap,
      n_active_orbitals);
  const Eigen::RowVectorXd row_delta =
      child->overlap.row(slot) - parent.overlap.row(slot);
  const Eigen::VectorXd inverse_column = parent.inverse.col(slot);
  const Eigen::RowVectorXd row_image = row_delta * parent.inverse;
  const double eta = 1.0 + row_image(slot);
  const double eta_scale = 1.0 +
      row_delta.lpNorm<Eigen::Infinity>() *
          inverse_column.lpNorm<Eigen::Infinity>();
  const double certificate = rounding_bound(
      static_cast<int>(parent.inverse.rows()),
      parent.updates_since_anchor + 1);
  if (!std::isfinite(eta) ||
      std::abs(eta) <= certificate * eta_scale) {
    return false;
  }

  child->inverse.noalias() = parent.inverse -
      (inverse_column * row_image) / eta;
  child->determinant = parent.determinant * eta;
  child->determinant_sign = parent.determinant_sign *
      (std::signbit(eta) ? -1.0 : 1.0);
  child->log_abs_determinant =
      parent.log_abs_determinant + std::log(std::abs(eta));
  child->updates_since_anchor = parent.updates_since_anchor + 1;
  child->regular = std::isfinite(child->determinant) &&
      std::isfinite(child->log_abs_determinant) &&
      child->inverse.allFinite() &&
      inverse_backward_error(child->overlap, child->inverse) <= certificate;
  return child->regular;
}

DeterminantOverlapResult canonical_overlap_result(
    const OverlapTraversalState& state,
    const std::vector<int>& canonical_right,
    const DeterminantOverlapResolver& resolver) {
  if (!state.regular) {
    return resolver.resolve_matrix(state.overlap);
  }
  const int dimension = static_cast<int>(canonical_right.size());
  DeterminantOverlapResult result;
  result.n_electrons = dimension;
  result.nullity = 0;
  if (dimension == 0) {
    result.overlap_submatrix.resize(0, 0);
    result.inverse_overlap_submatrix.resize(0, 0);
    result.first_order_cofactor_matrix.resize(0, 0);
    result.overlap_determinant = 1.0;
    result.determinant_sign = 1.0;
    result.log_abs_determinant = 0.0;
    return result;
  }
  result.overlap_submatrix.resize(dimension, dimension);
  result.inverse_overlap_submatrix.resize(dimension, dimension);
  for (int canonical_row = 0; canonical_row < dimension; ++canonical_row) {
    const auto found = std::find(
        state.right_internal_order.begin(),
        state.right_internal_order.end(),
        canonical_right[canonical_row]);
    if (found == state.right_internal_order.end()) {
      throw std::logic_error("internal spin-string ordering lost an orbital");
    }
    const int internal_row = static_cast<int>(
        std::distance(state.right_internal_order.begin(), found));
    result.overlap_submatrix.row(canonical_row) =
        state.overlap.row(internal_row);
    result.inverse_overlap_submatrix.col(canonical_row) =
        state.inverse.col(internal_row);
  }
  const int parity = permutation_sign_to_sorted(state.right_internal_order);
  result.overlap_determinant = parity * state.determinant;
  result.determinant_sign = parity * state.determinant_sign;
  result.log_abs_determinant = state.log_abs_determinant;
  cache_first_order_cofactor(&result);
  return result;
}

}  // namespace

const SpinDeterminantPairEvaluation& AcceptedSpinPairTile::pair(
    int left_local,
    int right_local) const {
  if (left_local < 0 || left_local >= left_size ||
      right_local < 0 || right_local >= right_size) {
    throw std::out_of_range("accepted pair tile index is out of range");
  }
  return pairs[static_cast<std::size_t>(left_local) * right_size +
      right_local];
}

AcceptedPairTileProvider::AcceptedPairTileProvider(
    std::vector<std::vector<int>> unique_spin_strings,
    int n_active_orbitals,
    double linear_dependence_threshold)
    : unique_spin_strings_(std::move(unique_spin_strings)),
      substitution_graph_(build_substitution_graph(
          unique_spin_strings_, n_active_orbitals)),
      n_active_orbitals_(n_active_orbitals),
      overlap_resolver_(linear_dependence_threshold),
      pair_evaluator_(overlap_resolver_,
                      DeterminantHamiltonianResolver(overlap_resolver_)) {}

AcceptedSpinPairTile AcceptedPairTileProvider::build(
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    AcceptedPairTileBuildOptions options) const {
  if (left_begin < 0 || left_end <= left_begin || left_end > size() ||
      right_begin < 0 || right_end <= right_begin || right_end > size()) {
    throw std::invalid_argument("accepted pair tile bounds are invalid");
  }
  if (active_one_electron.rows() != n_active_orbitals_ ||
      active_one_electron.cols() != n_active_orbitals_ ||
      active_overlap.size() != static_cast<std::size_t>(n_active_orbitals_) *
          n_active_orbitals_) {
    throw std::invalid_argument("accepted pair tile active-space dimensions differ");
  }

  AcceptedSpinPairTile tile;
  tile.left_begin = left_begin;
  tile.right_begin = right_begin;
  tile.left_size = left_end - left_begin;
  tile.right_size = right_end - right_begin;
  tile.pairs.resize(
      static_cast<std::size_t>(tile.left_size) * tile.right_size);
  const TileTraversal traversal = build_tile_traversal(
      substitution_graph_, right_begin, right_end);

  for (int left_local = 0; left_local < tile.left_size; ++left_local) {
    const auto& occ_left = unique_spin_strings_[left_begin + left_local];
    std::vector<OverlapTraversalState> states(tile.right_size);
    for (const int right_local : traversal.order) {
      const int right = right_begin + right_local;
      const auto& occ_right = unique_spin_strings_[right];
      const int parent = traversal.parent[right_local];
      bool updated = false;
      if (parent >= 0) {
        updated = woodbury_child_state(
            states[parent],
            occ_left,
            occ_right,
            active_overlap,
            n_active_orbitals_,
            &states[right_local]);
      }
      if (!updated) {
        states[right_local] = anchor_state(
            occ_left,
            occ_right,
            active_overlap,
            n_active_orbitals_,
            overlap_resolver_);
        ++tile.statistics.anchors;
        if (parent >= 0) {
          ++tile.statistics.certified_reanchors;
        }
      } else {
        ++tile.statistics.woodbury_updates;
      }

      DeterminantOverlapResult overlap_result;
      if (states[right_local].regular) {
        overlap_result = canonical_overlap_result(
            states[right_local], occ_right, overlap_resolver_);
      } else {
        const Eigen::MatrixXd canonical_overlap = build_overlap_submatrix(
            occ_left,
            occ_right,
            active_overlap,
            n_active_orbitals_);
        overlap_result = overlap_resolver_.resolve_matrix(canonical_overlap);
      }
      SpinDeterminantPairEvaluation evaluation =
          pair_evaluator_.evaluate_same_spin_pair(
              occ_left,
              occ_right,
              std::move(overlap_result),
              active_one_electron,
              n_active_orbitals_,
              active_two_electron,
              true);
      if (options.populate_opposite_spin_projection ||
          options.populate_response_payload) {
        complete_same_spin_pair_evaluation(
            occ_left,
            occ_right,
            active_one_electron,
            n_active_orbitals_,
            active_two_electron,
            options.materialize_projected_pair_values,
            options.populate_response_payload,
            &evaluation);
      }
      tile.pairs[static_cast<std::size_t>(left_local) * tile.right_size +
          right_local] = std::move(evaluation);
    }
  }
  return tile;
}

}  // namespace xmvb::vb
