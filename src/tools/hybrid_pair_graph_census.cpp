#include "tools/hybrid_pair_graph_census.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"

namespace xmvb::tools {
namespace {

struct StableCompletion {
  Eigen::MatrixXd base;
  Eigen::MatrixXd inverse_base;
  Eigen::MatrixXd core_left;
  Eigen::MatrixXd core_right;
  int core_rank = 0;
  bool overlap_is_certified_regular = false;
};

double infinity_norm(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return matrix.size() == 0
      ? 0.0
      : matrix.cwiseAbs().rowwise().sum().maxCoeff();
}

double relative_matrix_error(
    const Eigen::Ref<const Eigen::MatrixXd>& value,
    const Eigen::Ref<const Eigen::MatrixXd>& reference) {
  return infinity_norm(value - reference) /
      (1.0 + infinity_norm(reference));
}

double inverse_backward_error(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse) {
  const Eigen::MatrixXd residual =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()) -
      matrix * inverse;
  return infinity_norm(residual) /
      (1.0 + infinity_norm(matrix) * infinity_norm(inverse));
}

void increment_histogram(std::vector<long long>* histogram, int index) {
  if (index < 0) {
    throw std::invalid_argument("hybrid census rank must be non-negative");
  }
  if (index >= static_cast<int>(histogram->size())) {
    histogram->resize(index + 1, 0);
  }
  ++(*histogram)[index];
}

bool find_single_replacement(
    const std::vector<int>& current_internal_order,
    const std::vector<int>& next_sorted_order,
    int* slot,
    int* new_orbital) {
  std::vector<int> current_sorted = current_internal_order;
  std::sort(current_sorted.begin(), current_sorted.end());
  std::vector<int> removed;
  std::vector<int> inserted;
  std::set_difference(
      current_sorted.begin(), current_sorted.end(),
      next_sorted_order.begin(), next_sorted_order.end(),
      std::back_inserter(removed));
  std::set_difference(
      next_sorted_order.begin(), next_sorted_order.end(),
      current_sorted.begin(), current_sorted.end(),
      std::back_inserter(inserted));
  if (removed.size() != 1 || inserted.size() != 1) {
    return false;
  }
  const auto old = std::find(
      current_internal_order.begin(), current_internal_order.end(), removed[0]);
  if (old == current_internal_order.end()) {
    return false;
  }
  *slot = static_cast<int>(old - current_internal_order.begin());
  *new_orbital = inserted[0];
  return true;
}

StableCompletion build_stable_completion(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
  StableCompletion result;
  const int n = static_cast<int>(overlap.rows());
  if (n == 0) {
    return result;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      overlap, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (svd.info() != Eigen::Success) {
    throw std::runtime_error("hybrid census overlap SVD failed");
  }
  const Eigen::VectorXd singular_values = svd.singularValues();
  if (singular_values.size() != n || !singular_values.allFinite()) {
    throw std::runtime_error(
        "hybrid census overlap has no finite stable completion");
  }

  constexpr double highest_inverse_power = 4.0;
  const double condition_limit = std::pow(
      std::numeric_limits<double>::epsilon(),
      -1.0 / (2.0 * highest_inverse_power));
  if (singular_values(n - 1) > 0.0) {
    Eigen::MatrixXd direct_inverse;
    direct_inverse.noalias() =
        svd.matrixV() * singular_values.cwiseInverse().asDiagonal() *
        svd.matrixU().transpose();
    if (xmvb::vb::is_certified_regular_overlap(overlap, direct_inverse)) {
      result.base = overlap;
      result.inverse_base = std::move(direct_inverse);
      result.overlap_is_certified_regular = true;
      return result;
    }
  }

  // A completely zero occupied-overlap block has no intrinsic scale.  Orbital
  // overlaps are dimensionless and bounded by normalization, so unit scale is
  // the natural certified completion for this exceptional case.  Nonzero
  // blocks retain the same relative condition criterion as production.
  const double spectral_scale =
      singular_values(0) > 0.0 ? singular_values(0) : 1.0;
  const double tau =
      static_cast<double>(n) * spectral_scale / condition_limit;
  Eigen::VectorXd completed_singular_values = singular_values;
  for (int index = 0; index < n; ++index) {
    if (singular_values(index) < tau) {
      completed_singular_values(index) = tau;
      ++result.core_rank;
    }
  }

  result.base.noalias() =
      svd.matrixU() * completed_singular_values.asDiagonal() *
      svd.matrixV().transpose();
  result.inverse_base.noalias() =
      svd.matrixV() * completed_singular_values.cwiseInverse().asDiagonal() *
      svd.matrixU().transpose();
  result.overlap_is_certified_regular =
      result.core_rank == 0 &&
      xmvb::vb::is_certified_regular_overlap(
          overlap, result.inverse_base);

  if (result.core_rank > 0) {
    result.core_left.noalias() =
        svd.matrixU().rightCols(result.core_rank) *
        (singular_values.tail(result.core_rank).array() - tau)
            .matrix()
            .asDiagonal();
    result.core_right = svd.matrixV().rightCols(result.core_rank);
  }
  return result;
}

int numerical_rank(const Eigen::Ref<const Eigen::VectorXd>& singular_values,
                   int dimension) {
  if (singular_values.size() == 0 || singular_values(0) == 0.0) {
    return 0;
  }
  const double threshold =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(1, dimension)) * singular_values(0);
  return static_cast<int>((singular_values.array() > threshold).count());
}

struct LowRankFactors {
  Eigen::MatrixXd left;
  Eigen::MatrixXd right;
  int rank = 0;
};

LowRankFactors factor_low_rank(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  LowRankFactors result;
  if (matrix.size() == 0) {
    return result;
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success) {
    throw std::runtime_error("hybrid census low-rank SVD failed");
  }
  result.rank = numerical_rank(
      svd.singularValues(), static_cast<int>(matrix.rows()));
  if (result.rank == 0) {
    result.left.resize(matrix.rows(), 0);
    result.right.resize(matrix.cols(), 0);
    return result;
  }
  result.left.noalias() =
      svd.matrixU().leftCols(result.rank) *
      svd.singularValues().head(result.rank).asDiagonal();
  result.right = svd.matrixV().leftCols(result.rank);
  return result;
}

bool apply_block_woodbury(
    const Eigen::Ref<const Eigen::MatrixXd>& base,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse,
    const Eigen::Ref<const Eigen::MatrixXd>& update_left,
    const Eigen::Ref<const Eigen::MatrixXd>& update_right,
    Eigen::MatrixXd* updated_base,
    Eigen::MatrixXd* updated_inverse) {
  if (update_left.cols() == 0) {
    *updated_base = base;
    *updated_inverse = inverse;
    return true;
  }
  if (update_left.rows() != base.rows() ||
      update_right.rows() != base.cols() ||
      update_left.cols() != update_right.cols()) {
    return false;
  }

  const Eigen::MatrixXd inverse_left = inverse * update_left;
  const Eigen::MatrixXd right_inverse =
      update_right.transpose() * inverse;
  const int rank = static_cast<int>(update_left.cols());
  const Eigen::MatrixXd middle =
      Eigen::MatrixXd::Identity(rank, rank) +
      update_right.transpose() * inverse_left;
  Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
  if (!middle_lu.isInvertible()) {
    return false;
  }
  *updated_base = base + update_left * update_right.transpose();
  *updated_inverse = inverse -
      inverse_left * middle_lu.solve(right_inverse);
  return xmvb::vb::is_certified_regular_overlap(
      *updated_base, *updated_inverse);
}

struct PropagationState {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd base;
  Eigen::MatrixXd inverse_base;
  Eigen::MatrixXd core_left;
  Eigen::MatrixXd core_right;
};

void append_update(
    const Eigen::Ref<const Eigen::MatrixXd>& delta,
    PropagationState* state,
    int* appended_rank) {
  const LowRankFactors edge = factor_low_rank(delta);
  *appended_rank = edge.rank;
  const int old_rank = static_cast<int>(state->core_left.cols());
  Eigen::MatrixXd combined_left(
      delta.rows(), old_rank + edge.rank);
  Eigen::MatrixXd combined_right(
      delta.cols(), old_rank + edge.rank);
  if (old_rank > 0) {
    combined_left.leftCols(old_rank) = state->core_left;
    combined_right.leftCols(old_rank) = state->core_right;
  }
  if (edge.rank > 0) {
    combined_left.rightCols(edge.rank) = edge.left;
    combined_right.rightCols(edge.rank) = edge.right;
  }
  const LowRankFactors compressed = factor_low_rank(
      combined_left * combined_right.transpose());
  state->core_left = compressed.left;
  state->core_right = compressed.right;
}

int absorb_certified_core_directions(PropagationState* state) {
  const int rank = static_cast<int>(state->core_left.cols());
  if (rank == 0) {
    return 0;
  }
  const Eigen::MatrixXd core_middle =
      Eigen::MatrixXd::Identity(rank, rank) +
      state->core_right.transpose() * state->inverse_base * state->core_left;
  const Eigen::JacobiSVD<Eigen::MatrixXd> core_svd(
      core_middle, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (core_svd.info() != Eigen::Success) {
    throw std::runtime_error("hybrid census core SVD failed");
  }

  for (int retained_rank = 0; retained_rank <= rank; ++retained_rank) {
    const int absorbed_rank = rank - retained_rank;
    Eigen::MatrixXd safe_left(state->base.rows(), absorbed_rank);
    Eigen::MatrixXd safe_right(state->base.cols(), absorbed_rank);
    if (absorbed_rank > 0) {
      const Eigen::MatrixXd safe_basis =
          core_svd.matrixV().leftCols(absorbed_rank);
      safe_left.noalias() = state->core_left * safe_basis;
      safe_right.noalias() = state->core_right * safe_basis;
    }

    Eigen::MatrixXd updated_base;
    Eigen::MatrixXd updated_inverse;
    if (!apply_block_woodbury(
            state->base,
            state->inverse_base,
            safe_left,
            safe_right,
            &updated_base,
            &updated_inverse)) {
      continue;
    }

    if (retained_rank > 0) {
      const Eigen::MatrixXd dangerous_basis =
          core_svd.matrixV().rightCols(retained_rank);
      Eigen::MatrixXd retained_left = state->core_left * dangerous_basis;
      Eigen::MatrixXd retained_right = state->core_right * dangerous_basis;
      state->core_left = std::move(retained_left);
      state->core_right = std::move(retained_right);
    } else {
      state->core_left.resize(state->base.rows(), 0);
      state->core_right.resize(state->base.cols(), 0);
    }
    state->base = std::move(updated_base);
    state->inverse_base = std::move(updated_inverse);
    return absorbed_rank;
  }
  throw std::runtime_error(
      "hybrid census could not retain its already-certified base");
}

}  // namespace

HybridPairGraphCensus run_hybrid_pair_graph_census(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<int>& traversal_order,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    int max_pairs) {
  HybridPairGraphCensus census;
  if (unique_strings.empty()) {
    return census;
  }
  if (traversal_order.size() != unique_strings.size()) {
    throw std::invalid_argument(
        "hybrid census traversal does not cover every unique string");
  }

  const Eigen::Map<const Eigen::MatrixXd> overlap_map(
      active_overlap.data(), n_active_orbitals, n_active_orbitals);
  std::vector<int> current_left = unique_strings[traversal_order.front()];
  std::vector<int> current_right = unique_strings[traversal_order.front()];
  PropagationState state;
  bool have_previous = false;
  long long point_core_rank_sum = 0;
  long long propagated_core_rank_sum = 0;
  long long appended_update_rank_sum = 0;
  long long appended_update_count = 0;

  for (int left_position = 0;
       left_position < static_cast<int>(traversal_order.size());
       ++left_position) {
    const bool reverse = left_position % 2 != 0;
    for (int right_offset = 0;
         right_offset < static_cast<int>(traversal_order.size());
         ++right_offset) {
      if (max_pairs > 0 && census.visited_pairs >= max_pairs) {
        goto finish;
      }
      const int right_position = reverse
          ? static_cast<int>(traversal_order.size()) - 1 - right_offset
          : right_offset;
      const auto& target_left =
          unique_strings[traversal_order[left_position]];
      const auto& target_right =
          unique_strings[traversal_order[right_position]];

      if (have_previous) {
        const bool change_left = right_offset == 0;
        int slot = -1;
        int new_orbital = -1;
        const bool rank_one_edge = change_left
            ? find_single_replacement(
                  current_left, target_left, &slot, &new_orbital)
            : find_single_replacement(
                  current_right, target_right, &slot, &new_orbital);
        if (rank_one_edge) {
          if (change_left) {
            current_left[slot] = new_orbital;
          } else {
            current_right[slot] = new_orbital;
          }
        } else {
          current_left = target_left;
          current_right = target_right;
          ++census.topology_breaks;
        }
      }

      const Eigen::MatrixXd overlap = xmvb::vb::build_overlap_submatrix(
          current_left, current_right, overlap_map);
      StableCompletion completion = build_stable_completion(overlap);
      ++census.visited_pairs;
      point_core_rank_sum += completion.core_rank;
      increment_histogram(
          &census.point_core_rank_counts, completion.core_rank);
      if (completion.overlap_is_certified_regular) {
        ++census.certified_regular_pairs;
      }
      if (completion.core_rank <
          static_cast<int>(std::numeric_limits<long long>::digits - 1)) {
        census.interpolation_node_equivalents +=
            1LL << completion.core_rank;
      }

      if (!have_previous) {
        state.overlap = overlap;
        state.base = std::move(completion.base);
        state.inverse_base = std::move(completion.inverse_base);
        state.core_left = std::move(completion.core_left);
        state.core_right = std::move(completion.core_right);
        have_previous = true;
      } else {
        int appended_rank = 0;
        append_update(overlap - state.overlap, &state, &appended_rank);
        increment_histogram(
            &census.appended_update_rank_counts, appended_rank);
        appended_update_rank_sum += appended_rank;
        ++appended_update_count;
        const int rank_before_absorption =
            static_cast<int>(state.core_left.cols());
        const int absorbed_rank = absorb_certified_core_directions(&state);
        census.absorbed_update_directions += absorbed_rank;
        census.retained_update_directions +=
            rank_before_absorption - absorbed_rank;
        state.overlap = overlap;
      }

      const int propagated_core_rank =
          static_cast<int>(state.core_left.cols());
      propagated_core_rank_sum += propagated_core_rank;
      increment_histogram(
          &census.propagated_core_rank_counts, propagated_core_rank);
      const Eigen::MatrixXd reconstructed = state.base +
          state.core_left * state.core_right.transpose();
      census.max_reconstruction_relative_error = std::max(
          census.max_reconstruction_relative_error,
          relative_matrix_error(reconstructed, overlap));
      const double base_backward_error =
          inverse_backward_error(state.base, state.inverse_base);
      census.max_base_inverse_backward_error = std::max(
          census.max_base_inverse_backward_error,
          base_backward_error);
      if (!xmvb::vb::is_certified_regular_overlap(
              state.base, state.inverse_base)) {
        ++census.uncertified_stable_bases;
      }
      Eigen::FullPivLU<Eigen::MatrixXd> direct_base_lu(state.base);
      if (!direct_base_lu.isInvertible()) {
        ++census.numerical_reanchors;
        throw std::runtime_error(
            "hybrid census propagated base became singular");
      }
      census.max_propagated_inverse_relative_error = std::max(
          census.max_propagated_inverse_relative_error,
          relative_matrix_error(
              state.inverse_base, direct_base_lu.inverse()));
    }
  }

finish:
  if (census.visited_pairs > 0) {
    census.mean_point_core_rank = static_cast<double>(point_core_rank_sum) /
        static_cast<double>(census.visited_pairs);
    census.mean_propagated_core_rank =
        static_cast<double>(propagated_core_rank_sum) /
        static_cast<double>(census.visited_pairs);
  }
  if (appended_update_count > 0) {
    census.mean_appended_update_rank =
        static_cast<double>(appended_update_rank_sum) /
        static_cast<double>(appended_update_count);
  }
  return census;
}

void print_hybrid_pair_graph_census(
    const char* label,
    const HybridPairGraphCensus& census) {
  std::cout << label << "_hybrid_visited_pairs = "
            << census.visited_pairs << '\n';
  std::cout << label << "_hybrid_certified_regular_pairs = "
            << census.certified_regular_pairs << '\n';
  std::cout << label << "_hybrid_uncertified_stable_bases = "
            << census.uncertified_stable_bases << '\n';
  long long cumulative = 0;
  for (int rank = 0;
       rank < static_cast<int>(census.point_core_rank_counts.size());
       ++rank) {
    cumulative += census.point_core_rank_counts[rank];
    std::cout << label << "_hybrid_point_core_rank_" << rank
              << "_pairs = " << census.point_core_rank_counts[rank] << '\n';
    std::cout << label << "_hybrid_point_core_rank_le_" << rank
              << "_fraction = "
              << (census.visited_pairs > 0
                      ? static_cast<double>(cumulative) /
                            static_cast<double>(census.visited_pairs)
                      : 0.0)
              << '\n';
  }
  cumulative = 0;
  for (int rank = 0;
       rank < static_cast<int>(census.propagated_core_rank_counts.size());
       ++rank) {
    cumulative += census.propagated_core_rank_counts[rank];
    std::cout << label << "_hybrid_propagated_core_rank_" << rank
              << "_pairs = "
              << census.propagated_core_rank_counts[rank] << '\n';
    std::cout << label << "_hybrid_propagated_core_rank_le_" << rank
              << "_fraction = "
              << (census.visited_pairs > 0
                      ? static_cast<double>(cumulative) /
                            static_cast<double>(census.visited_pairs)
                      : 0.0)
              << '\n';
  }
  for (int rank = 0;
       rank < static_cast<int>(census.appended_update_rank_counts.size());
       ++rank) {
    std::cout << label << "_hybrid_appended_update_rank_" << rank
              << "_count = "
              << census.appended_update_rank_counts[rank] << '\n';
  }
  std::cout << label << "_hybrid_mean_point_core_rank = "
            << census.mean_point_core_rank << '\n';
  std::cout << label << "_hybrid_mean_propagated_core_rank = "
            << census.mean_propagated_core_rank << '\n';
  std::cout << label << "_hybrid_mean_appended_update_rank = "
            << census.mean_appended_update_rank << '\n';
  std::cout << label << "_hybrid_absorbed_update_directions = "
            << census.absorbed_update_directions << '\n';
  std::cout << label << "_hybrid_retained_update_directions = "
            << census.retained_update_directions << '\n';
  std::cout << label << "_hybrid_topology_breaks = "
            << census.topology_breaks << '\n';
  std::cout << label << "_hybrid_numerical_reanchors = "
            << census.numerical_reanchors << '\n';
  std::cout << label << "_hybrid_interpolation_node_equivalents = "
            << census.interpolation_node_equivalents << '\n';
  std::cout << label << "_hybrid_max_reconstruction_relative_error = "
            << census.max_reconstruction_relative_error << '\n';
  std::cout << label << "_hybrid_max_base_inverse_backward_error = "
            << census.max_base_inverse_backward_error << '\n';
  std::cout << label << "_hybrid_max_propagated_inverse_relative_error = "
            << census.max_propagated_inverse_relative_error << '\n';
}

}  // namespace xmvb::tools
