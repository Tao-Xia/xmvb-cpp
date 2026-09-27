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
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd base;
  Eigen::MatrixXd inverse_base;
  int core_rank = 0;
  bool overlap_is_certified_regular = false;
  double reconstruction_relative_error = 0.0;
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
  result.overlap = overlap;
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
  if (singular_values.size() != n || singular_values(0) <= 0.0 ||
      !singular_values.allFinite()) {
    throw std::runtime_error(
        "hybrid census overlap has no finite stable completion");
  }

  constexpr double highest_inverse_power = 4.0;
  const double condition_limit = std::pow(
      std::numeric_limits<double>::epsilon(),
      -1.0 / (2.0 * highest_inverse_power));
  const double tau =
      static_cast<double>(n) * singular_values(0) / condition_limit;
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

  Eigen::MatrixXd reconstructed = result.base;
  if (result.core_rank > 0) {
    reconstructed.noalias() +=
        svd.matrixU().rightCols(result.core_rank) *
        (singular_values.tail(result.core_rank).array() - tau)
            .matrix()
            .asDiagonal() *
        svd.matrixV().rightCols(result.core_rank).transpose();
  }
  result.reconstruction_relative_error =
      relative_matrix_error(reconstructed, overlap);
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

bool update_inverse_by_block_woodbury(
    const Eigen::Ref<const Eigen::MatrixXd>& old_base,
    const Eigen::Ref<const Eigen::MatrixXd>& old_inverse,
    const Eigen::Ref<const Eigen::MatrixXd>& new_base,
    int* update_rank,
    Eigen::MatrixXd* new_inverse) {
  const Eigen::MatrixXd delta = new_base - old_base;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      delta, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success) {
    return false;
  }
  *update_rank = numerical_rank(
      svd.singularValues(), static_cast<int>(delta.rows()));
  if (*update_rank == 0) {
    *new_inverse = old_inverse;
    return true;
  }
  if (*update_rank >= delta.rows()) {
    return false;
  }

  const Eigen::MatrixXd update_left =
      svd.matrixU().leftCols(*update_rank) *
      svd.singularValues().head(*update_rank).asDiagonal();
  const Eigen::MatrixXd update_right =
      svd.matrixV().leftCols(*update_rank);
  const Eigen::MatrixXd inverse_left = old_inverse * update_left;
  const Eigen::MatrixXd right_inverse =
      update_right.transpose() * old_inverse;
  const Eigen::MatrixXd middle =
      Eigen::MatrixXd::Identity(*update_rank, *update_rank) +
      update_right.transpose() * inverse_left;
  Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
  if (!middle_lu.isInvertible()) {
    return false;
  }
  *new_inverse = old_inverse -
      inverse_left * middle_lu.solve(right_inverse);
  return xmvb::vb::is_certified_regular_overlap(new_base, *new_inverse);
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
  StableCompletion previous_completion;
  Eigen::MatrixXd propagated_inverse;
  bool have_previous = false;
  long long core_rank_sum = 0;
  long long base_update_rank_sum = 0;
  long long base_update_count = 0;

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
      core_rank_sum += completion.core_rank;
      increment_histogram(&census.core_rank_counts, completion.core_rank);
      census.max_reconstruction_relative_error = std::max(
          census.max_reconstruction_relative_error,
          completion.reconstruction_relative_error);
      census.max_base_inverse_backward_error = std::max(
          census.max_base_inverse_backward_error,
          inverse_backward_error(completion.base, completion.inverse_base));
      if (!xmvb::vb::is_certified_regular_overlap(
              completion.base, completion.inverse_base)) {
        ++census.uncertified_stable_bases;
      }
      if (completion.overlap_is_certified_regular) {
        ++census.certified_regular_pairs;
      }
      if (completion.core_rank <
          static_cast<int>(std::numeric_limits<long long>::digits - 1)) {
        census.interpolation_node_equivalents +=
            1LL << completion.core_rank;
      }

      if (!have_previous) {
        propagated_inverse = completion.inverse_base;
        previous_completion = std::move(completion);
        have_previous = true;
        continue;
      }

      int update_rank = 0;
      Eigen::MatrixXd updated_inverse;
      const bool updated = update_inverse_by_block_woodbury(
          previous_completion.base,
          propagated_inverse,
          completion.base,
          &update_rank,
          &updated_inverse);
      increment_histogram(&census.base_update_rank_counts, update_rank);
      base_update_rank_sum += update_rank;
      ++base_update_count;
      if (update_rank >= completion.base.rows()) {
        ++census.full_rank_base_updates;
      } else if (updated) {
        ++census.low_rank_base_updates;
        propagated_inverse = std::move(updated_inverse);
      } else {
        ++census.failed_base_updates;
      }
      if (!updated) {
        propagated_inverse = completion.inverse_base;
        ++census.numerical_reanchors;
      }
      census.max_propagated_inverse_relative_error = std::max(
          census.max_propagated_inverse_relative_error,
          relative_matrix_error(
              propagated_inverse, completion.inverse_base));
      previous_completion = std::move(completion);
    }
  }

finish:
  if (census.visited_pairs > 0) {
    census.mean_core_rank = static_cast<double>(core_rank_sum) /
        static_cast<double>(census.visited_pairs);
  }
  if (base_update_count > 0) {
    census.mean_base_update_rank =
        static_cast<double>(base_update_rank_sum) /
        static_cast<double>(base_update_count);
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
       rank < static_cast<int>(census.core_rank_counts.size());
       ++rank) {
    cumulative += census.core_rank_counts[rank];
    std::cout << label << "_hybrid_core_rank_" << rank
              << "_pairs = " << census.core_rank_counts[rank] << '\n';
    std::cout << label << "_hybrid_core_rank_le_" << rank
              << "_fraction = "
              << (census.visited_pairs > 0
                      ? static_cast<double>(cumulative) /
                            static_cast<double>(census.visited_pairs)
                      : 0.0)
              << '\n';
  }
  for (int rank = 0;
       rank < static_cast<int>(census.base_update_rank_counts.size());
       ++rank) {
    std::cout << label << "_hybrid_base_update_rank_" << rank
              << "_count = " << census.base_update_rank_counts[rank] << '\n';
  }
  std::cout << label << "_hybrid_mean_core_rank = "
            << census.mean_core_rank << '\n';
  std::cout << label << "_hybrid_mean_base_update_rank = "
            << census.mean_base_update_rank << '\n';
  std::cout << label << "_hybrid_low_rank_base_updates = "
            << census.low_rank_base_updates << '\n';
  std::cout << label << "_hybrid_full_rank_base_updates = "
            << census.full_rank_base_updates << '\n';
  std::cout << label << "_hybrid_failed_base_updates = "
            << census.failed_base_updates << '\n';
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
