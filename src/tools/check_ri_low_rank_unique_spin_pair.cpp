#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
  int max_pair_checks = 64;
  int max_rank_update_checks = 32;
  int max_feature_benchmark_pairs = 20000;
};

struct LocalityStats {
  int n_strings = 0;
  int n_electrons = 0;
  std::vector<long long> replacement_histogram;
  std::vector<long long> positional_histogram;
  double mean_replacement_distance = 0.0;
  double mean_positional_distance = 0.0;
  double replacement_d1_fraction = 0.0;
  double replacement_d2_fraction = 0.0;
  double positional_d1_fraction = 0.0;
  double positional_d2_fraction = 0.0;
};

struct TraversalStats {
  int n_edges = 0;
  double mean_replacement_distance = 0.0;
  double mean_positional_distance = 0.0;
  int max_replacement_distance = 0;
  int max_positional_distance = 0;
  double replacement_d1_fraction = 0.0;
  double replacement_d2_fraction = 0.0;
  double positional_d1_fraction = 0.0;
  double positional_d2_fraction = 0.0;
};

struct SameSpinFormulaStats {
  int checked = 0;
  int skipped_singular = 0;
  double max_abs_phi2_diff = 0.0;
  double max_abs_h2_diff = 0.0;
};

struct OppositeSpinFormulaStats {
  int checked = 0;
  int skipped_singular = 0;
  double max_abs_phi_diff = 0.0;
  double max_abs_h_diff = 0.0;
};

struct RankUpdateStats {
  int checked = 0;
  int skipped_singular = 0;
  double direct_overlap_seconds = 0.0;
  double rank_update_seconds = 0.0;
  double mean_replacement_distance = 0.0;
  double mean_update_rank = 0.0;
  int max_update_rank = 0;
  double max_abs_inverse_diff = 0.0;
  double max_abs_det_diff = 0.0;
  double max_abs_phi2_diff = 0.0;
};

struct RankOneOverlapStats {
  int checked = 0;
  int resets = 0;
  int skipped_singular = 0;
  double direct_overlap_seconds = 0.0;
  double rank_one_seconds = 0.0;
  double max_abs_inverse_diff = 0.0;
  double max_abs_det_diff = 0.0;
};

struct RiQUpdateStats {
  int checked = 0;
  int resets = 0;
  int selected_column_rebuilds = 0;
  double direct_q_seconds = 0.0;
  double rank_update_q_seconds = 0.0;
  double max_abs_q_diff = 0.0;
};

struct RiKUpdateStats {
  int checked = 0;
  int resets = 0;
  int skipped_singular = 0;
  double direct_seconds = 0.0;
  double update_seconds = 0.0;
  double max_abs_phi2_diff = 0.0;
};

struct OppositeSpinFeatureBenchmarkStats {
  int direct_pairs = 0;
  int low_rank_pairs = 0;
  int low_rank_initializations = 0;
  int low_rank_updates = 0;
  int low_rank_resets = 0;
  int scalar_pairs = 0;
  int scalar_initializations = 0;
  int scalar_updates = 0;
  int scalar_resets = 0;
  int scalar_selected_column_rebuilds = 0;
  double direct_seconds = 0.0;
  double low_rank_seconds = 0.0;
  double scalar_seconds = 0.0;
  double direct_checksum = 0.0;
  double low_rank_checksum = 0.0;
  double scalar_checksum = 0.0;
  double direct_linear_checksum = 0.0;
  double scalar_linear_checksum = 0.0;
  double checksum_abs_diff = 0.0;
  double scalar_checksum_abs_diff = 0.0;
  double scalar_linear_checksum_abs_diff = 0.0;
};

struct RightStringUpdateData {
  std::vector<int> rows;
  Eigen::MatrixXd update_right_transpose;
  Eigen::MatrixXd middle_inverse;
  Eigen::MatrixXd c_matrix;
  Eigen::MatrixXd d_matrix;
  Eigen::MatrixXd inverse_updated;
  double determinant_updated = 0.0;
  bool ok = false;
};

struct RiFeatureTraversalState {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd inverse_overlap;
  double determinant = 0.0;
  int right_index = -1;
  std::vector<Eigen::MatrixXd> channel_matrices;
  bool valid = false;
};

struct ScalarRiFeatureTraversalState {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd inverse_overlap;
  Eigen::VectorXd x_features;
  Eigen::MatrixXd selected_channel_column;
  std::vector<int> right_internal_order;
  double determinant = 0.0;
  int right_index = -1;
  int selected_slot = -1;
  int canonical_sign = 1;
  bool valid = false;
};

void print_usage() {
  std::cerr
      << "usage: check_ri_low_rank_unique_spin_pair <input.xmi>"
      << " [--standard-two-electron-mode ri|auto]"
      << " [--max-pair-checks N]"
      << " [--max-rank-update-checks N]"
      << " [--max-feature-benchmark-pairs N]\n";
}

int parse_positive_int(const std::string& text, const char* option_name) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(option_name) + " must be positive");
  }
  return value;
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("missing input path");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; ++argument_index) {
    const std::string name = argv[argument_index];
    if (name == "--standard-two-electron-mode") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--standard-two-electron-mode requires a value");
      }
      const std::string value = argv[++argument_index];
      if (value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else if (value == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else {
        throw std::invalid_argument(
            "invalid --standard-two-electron-mode value: " + value);
      }
      continue;
    }
    if (name == "--max-pair-checks") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--max-pair-checks requires a value");
      }
      options.max_pair_checks =
          parse_positive_int(argv[++argument_index], "--max-pair-checks");
      continue;
    }
    if (name == "--max-rank-update-checks") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--max-rank-update-checks requires a value");
      }
      options.max_rank_update_checks =
          parse_positive_int(argv[++argument_index], "--max-rank-update-checks");
      continue;
    }
    if (name == "--max-feature-benchmark-pairs") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument(
            "--max-feature-benchmark-pairs requires a value");
      }
      options.max_feature_benchmark_pairs = parse_positive_int(
          argv[++argument_index],
          "--max-feature-benchmark-pairs");
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  return options;
}

int replacement_distance(
    const std::vector<int>& left,
    const std::vector<int>& right) {
  int common_count = 0;
  int left_index = 0;
  int right_index = 0;
  while (left_index < static_cast<int>(left.size()) &&
         right_index < static_cast<int>(right.size())) {
    if (left[left_index] == right[right_index]) {
      ++common_count;
      ++left_index;
      ++right_index;
    } else if (left[left_index] < right[right_index]) {
      ++left_index;
    } else {
      ++right_index;
    }
  }
  return static_cast<int>(left.size()) - common_count;
}

int positional_distance(
    const std::vector<int>& left,
    const std::vector<int>& right) {
  int distance = 0;
  for (int index = 0; index < static_cast<int>(left.size()); ++index) {
    if (left[index] != right[index]) {
      ++distance;
    }
  }
  return distance;
}

void add_to_histogram(std::vector<long long>* histogram, int value) {
  if (value >= static_cast<int>(histogram->size())) {
    histogram->resize(value + 1, 0);
  }
  ++(*histogram)[value];
}

double histogram_fraction_leq(
    const std::vector<long long>& histogram,
    int threshold,
    long long total_count) {
  if (total_count == 0) {
    return 0.0;
  }
  long long selected_count = 0;
  for (int index = 0;
       index <= threshold && index < static_cast<int>(histogram.size());
       ++index) {
    selected_count += histogram[index];
  }
  return static_cast<double>(selected_count) / static_cast<double>(total_count);
}

LocalityStats compute_locality_stats(
    const std::vector<std::vector<int>>& unique_strings) {
  LocalityStats stats;
  stats.n_strings = static_cast<int>(unique_strings.size());
  stats.n_electrons = unique_strings.empty()
                          ? 0
                          : static_cast<int>(unique_strings.front().size());
  stats.replacement_histogram.assign(stats.n_electrons + 1, 0);
  stats.positional_histogram.assign(stats.n_electrons + 1, 0);

  long long pair_count = 0;
  long long replacement_sum = 0;
  long long positional_sum = 0;
  for (int left = 0; left < stats.n_strings; ++left) {
    for (int right = left + 1; right < stats.n_strings; ++right) {
      const int replacement = replacement_distance(
          unique_strings[left],
          unique_strings[right]);
      const int positional = positional_distance(
          unique_strings[left],
          unique_strings[right]);
      add_to_histogram(&stats.replacement_histogram, replacement);
      add_to_histogram(&stats.positional_histogram, positional);
      replacement_sum += replacement;
      positional_sum += positional;
      ++pair_count;
    }
  }

  if (pair_count > 0) {
    stats.mean_replacement_distance =
        static_cast<double>(replacement_sum) / static_cast<double>(pair_count);
    stats.mean_positional_distance =
        static_cast<double>(positional_sum) / static_cast<double>(pair_count);
    stats.replacement_d1_fraction =
        histogram_fraction_leq(stats.replacement_histogram, 1, pair_count);
    stats.replacement_d2_fraction =
        histogram_fraction_leq(stats.replacement_histogram, 2, pair_count);
    stats.positional_d1_fraction =
        histogram_fraction_leq(stats.positional_histogram, 1, pair_count);
    stats.positional_d2_fraction =
        histogram_fraction_leq(stats.positional_histogram, 2, pair_count);
  }
  return stats;
}

TraversalStats compute_greedy_traversal_stats(
    const std::vector<std::vector<int>>& unique_strings) {
  TraversalStats stats;
  const int n_strings = static_cast<int>(unique_strings.size());
  if (n_strings <= 1) {
    return stats;
  }

  std::vector<char> visited(n_strings, 0);
  int current = 0;
  visited[current] = 1;
  long long replacement_sum = 0;
  long long positional_sum = 0;
  int replacement_d1_count = 0;
  int replacement_d2_count = 0;
  int positional_d1_count = 0;
  int positional_d2_count = 0;

  for (int edge = 0; edge < n_strings - 1; ++edge) {
    int best_index = -1;
    int best_replacement = std::numeric_limits<int>::max();
    int best_positional = std::numeric_limits<int>::max();
    for (int candidate = 0; candidate < n_strings; ++candidate) {
      if (visited[candidate]) {
        continue;
      }
      const int replacement = replacement_distance(
          unique_strings[current],
          unique_strings[candidate]);
      const int positional = positional_distance(
          unique_strings[current],
          unique_strings[candidate]);
      if (replacement < best_replacement ||
          (replacement == best_replacement && positional < best_positional)) {
        best_index = candidate;
        best_replacement = replacement;
        best_positional = positional;
      }
    }

    current = best_index;
    visited[current] = 1;
    ++stats.n_edges;
    replacement_sum += best_replacement;
    positional_sum += best_positional;
    stats.max_replacement_distance =
        std::max(stats.max_replacement_distance, best_replacement);
    stats.max_positional_distance =
        std::max(stats.max_positional_distance, best_positional);
    if (best_replacement <= 1) {
      ++replacement_d1_count;
    }
    if (best_replacement <= 2) {
      ++replacement_d2_count;
    }
    if (best_positional <= 1) {
      ++positional_d1_count;
    }
    if (best_positional <= 2) {
      ++positional_d2_count;
    }
  }

  stats.mean_replacement_distance =
      static_cast<double>(replacement_sum) / static_cast<double>(stats.n_edges);
  stats.mean_positional_distance =
      static_cast<double>(positional_sum) / static_cast<double>(stats.n_edges);
  stats.replacement_d1_fraction =
      static_cast<double>(replacement_d1_count) / static_cast<double>(stats.n_edges);
  stats.replacement_d2_fraction =
      static_cast<double>(replacement_d2_count) / static_cast<double>(stats.n_edges);
  stats.positional_d1_fraction =
      static_cast<double>(positional_d1_count) / static_cast<double>(stats.n_edges);
  stats.positional_d2_fraction =
      static_cast<double>(positional_d2_count) / static_cast<double>(stats.n_edges);
  return stats;
}

bool find_single_replacement_slot(
    const std::vector<int>& current_internal_order,
    const std::vector<int>& next_sorted_order,
    int* slot,
    int* old_orbital,
    int* new_orbital);

bool initialize_scalar_ri_feature_traversal_state(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R_internal,
    const std::vector<int>& occ_R_sorted,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int right_index,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ScalarRiFeatureTraversalState* state);

bool initialize_ri_feature_traversal_state(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int right_index,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    RiFeatureTraversalState* state);

double compute_phi2_from_channel_matrices(
    const std::vector<Eigen::MatrixXd>& channel_matrices);

std::vector<int> build_greedy_traversal_order(
    const std::vector<std::vector<int>>& unique_strings) {
  const int n_strings = static_cast<int>(unique_strings.size());
  std::vector<int> order;
  if (n_strings == 0) {
    return order;
  }
  order.reserve(n_strings);

  std::vector<char> visited(n_strings, 0);
  int current = 0;
  visited[current] = 1;
  order.push_back(current);
  for (int edge = 0; edge < n_strings - 1; ++edge) {
    int best_index = -1;
    int best_replacement = std::numeric_limits<int>::max();
    int best_positional = std::numeric_limits<int>::max();
    for (int candidate = 0; candidate < n_strings; ++candidate) {
      if (visited[candidate]) {
        continue;
      }
      const int replacement = replacement_distance(
          unique_strings[current],
          unique_strings[candidate]);
      const int positional = positional_distance(
          unique_strings[current],
          unique_strings[candidate]);
      if (replacement < best_replacement ||
          (replacement == best_replacement && positional < best_positional)) {
        best_index = candidate;
        best_replacement = replacement;
        best_positional = positional;
      }
    }
    current = best_index;
    visited[current] = 1;
    order.push_back(current);
  }
  return order;
}

std::vector<int> build_slot_stable_traversal_order(
    const std::vector<std::vector<int>>& unique_strings) {
  const int n_strings = static_cast<int>(unique_strings.size());
  std::vector<int> order;
  if (n_strings == 0) {
    return order;
  }
  order.reserve(n_strings);

  std::vector<char> visited(n_strings, 0);
  std::vector<int> current_internal_order = unique_strings.front();
  int selected_slot = -1;
  int current = 0;
  visited[current] = 1;
  order.push_back(current);

  for (int edge = 0; edge < n_strings - 1; ++edge) {
    std::vector<int> current_sorted = current_internal_order;
    std::sort(current_sorted.begin(), current_sorted.end());
    int best_index = -1;
    int best_replacement = std::numeric_limits<int>::max();
    int best_slot_penalty = std::numeric_limits<int>::max();
    int best_positional = std::numeric_limits<int>::max();
    int best_slot = -1;
    int best_old_orbital = -1;
    int best_new_orbital = -1;

    for (int candidate = 0; candidate < n_strings; ++candidate) {
      if (visited[candidate]) {
        continue;
      }
      const int replacement =
          replacement_distance(current_sorted, unique_strings[candidate]);
      int slot = -1;
      int old_orbital = -1;
      int new_orbital = -1;
      int slot_penalty = 2;
      if (replacement == 1 &&
          find_single_replacement_slot(
              current_internal_order,
              unique_strings[candidate],
              &slot,
              &old_orbital,
              &new_orbital)) {
        slot_penalty =
            (selected_slot >= 0 && slot == selected_slot) ? 0 : 1;
      }
      const int positional = positional_distance(
          current_internal_order,
          unique_strings[candidate]);
      if (replacement < best_replacement ||
          (replacement == best_replacement && slot_penalty < best_slot_penalty) ||
          (replacement == best_replacement &&
           slot_penalty == best_slot_penalty &&
           positional < best_positional)) {
        best_index = candidate;
        best_replacement = replacement;
        best_slot_penalty = slot_penalty;
        best_positional = positional;
        best_slot = slot;
        best_old_orbital = old_orbital;
        best_new_orbital = new_orbital;
      }
    }

    visited[best_index] = 1;
    order.push_back(best_index);
    if (best_replacement == 1 && best_slot >= 0) {
      (void)best_old_orbital;
      current_internal_order[best_slot] = best_new_orbital;
      selected_slot = best_slot;
    } else {
      current_internal_order = unique_strings[best_index];
      selected_slot = -1;
    }
    current = best_index;
    (void)current;
  }
  return order;
}

Eigen::MatrixXd build_ri_occupied_block(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int auxiliary_index) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Eigen::MatrixXd block(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int left_orbital = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int right_orbital = occ_R[right_row];
      const int packed_pair_index =
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              right_orbital,
              left_orbital);
      block(right_row, left_column) =
          ri_active_pair_factors(auxiliary_index, packed_pair_index);
    }
  }
  return block;
}

Eigen::VectorXd compute_ri_trace_features(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& inverse_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors) {
  const int n_auxiliary_functions =
      static_cast<int>(ri_active_pair_factors.rows());
  Eigen::VectorXd features(n_auxiliary_functions);
  for (int auxiliary_index = 0;
       auxiliary_index < n_auxiliary_functions;
       ++auxiliary_index) {
    const Eigen::MatrixXd occupied_block = build_ri_occupied_block(
        occ_L,
        occ_R,
        ri_active_pair_factors,
        auxiliary_index);
    features(auxiliary_index) =
        (occupied_block * inverse_overlap).trace();
  }
  return features;
}

double compute_ri_same_spin_two_electron_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& inverse_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors) {
  // RI transforms the active two-electron tensor into auxiliary-channel
  // occupied blocks A_Lambda. For a regular same-spin pair the two-electron
  // phi is 1/2 sum_Lambda [(Tr A X)^2 - Tr(A X A X)].
  double phi = 0.0;
  for (int auxiliary_index = 0;
       auxiliary_index < ri_active_pair_factors.rows();
       ++auxiliary_index) {
    const Eigen::MatrixXd occupied_block = build_ri_occupied_block(
        occ_L,
        occ_R,
        ri_active_pair_factors,
        auxiliary_index);
    const Eigen::MatrixXd channel_matrix = occupied_block * inverse_overlap;
    const double trace_value = channel_matrix.trace();
    phi += trace_value * trace_value -
           (channel_matrix * channel_matrix).trace();
  }
  return 0.5 * phi;
}

void fill_direct_ri_first_order_feature(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& inverse_overlap,
    double determinant,
    const Eigen::MatrixXd& ri_active_pair_factors,
    Eigen::VectorXd* feature) {
  // Regular first-order cofactor features are
  // y_Lambda = det(S) * Tr[R^T B_Lambda L * S^{-1}].
  // Building them as a sequence of RI-factor column axpy operations is the
  // direct recomputation baseline for opposite-spin feature generation.
  feature->setZero(ri_active_pair_factors.rows());
  for (int left_column = 0;
       left_column < static_cast<int>(occ_L.size());
       ++left_column) {
    const int left_orbital = occ_L[left_column];
    for (int right_row = 0;
         right_row < static_cast<int>(occ_R.size());
         ++right_row) {
      const int right_orbital = occ_R[right_row];
      const int packed_pair_index =
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              right_orbital,
              left_orbital);
      const double coefficient =
          determinant * inverse_overlap(left_column, right_row);
      feature->noalias() +=
          coefficient * ri_active_pair_factors.col(packed_pair_index);
    }
  }
}

double feature_checksum(const Eigen::VectorXd& feature) {
  return feature.squaredNorm();
}

double feature_linear_checksum(const Eigen::VectorXd& feature) {
  double checksum = 0.0;
  for (int index = 0; index < feature.size(); ++index) {
    const double weight = static_cast<double>((index % 17) + 1);
    checksum += weight * feature(index);
  }
  return checksum;
}

double compute_ri_same_spin_two_electron_phi_from_blocks(
    const std::vector<Eigen::MatrixXd>& occupied_blocks,
    const Eigen::MatrixXd& inverse_overlap) {
  double phi = 0.0;
  for (const auto& occupied_block : occupied_blocks) {
    const Eigen::MatrixXd channel_matrix = occupied_block * inverse_overlap;
    const double trace_value = channel_matrix.trace();
    phi += trace_value * trace_value -
           (channel_matrix * channel_matrix).trace();
  }
  return 0.5 * phi;
}

std::vector<Eigen::MatrixXd> build_all_ri_occupied_blocks(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& ri_active_pair_factors) {
  std::vector<Eigen::MatrixXd> blocks;
  blocks.reserve(static_cast<std::size_t>(ri_active_pair_factors.rows()));
  for (int auxiliary_index = 0;
       auxiliary_index < ri_active_pair_factors.rows();
       ++auxiliary_index) {
    blocks.push_back(build_ri_occupied_block(
        occ_L,
        occ_R,
        ri_active_pair_factors,
        auxiliary_index));
  }
  return blocks;
}

std::vector<Eigen::MatrixXd> build_all_ri_channel_matrices(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& inverse_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors) {
  std::vector<Eigen::MatrixXd> channel_matrices;
  channel_matrices.reserve(
      static_cast<std::size_t>(ri_active_pair_factors.rows()));
  for (int auxiliary_index = 0;
       auxiliary_index < ri_active_pair_factors.rows();
       ++auxiliary_index) {
    const Eigen::MatrixXd occupied_block = build_ri_occupied_block(
        occ_L,
        occ_R,
        ri_active_pair_factors,
        auxiliary_index);
    channel_matrices.push_back(occupied_block * inverse_overlap);
  }
  return channel_matrices;
}

Eigen::MatrixXd build_selected_channel_column(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R_internal,
    const Eigen::MatrixXd& inverse_overlap,
    int selected_slot,
    const Eigen::MatrixXd& ri_active_pair_factors) {
  Eigen::MatrixXd selected_column =
      Eigen::MatrixXd::Zero(
          ri_active_pair_factors.rows(),
          static_cast<int>(occ_R_internal.size()));
  for (int right_row = 0;
       right_row < static_cast<int>(occ_R_internal.size());
       ++right_row) {
    const int right_orbital = occ_R_internal[right_row];
    for (int left_column = 0;
         left_column < static_cast<int>(occ_L.size());
         ++left_column) {
      const int packed_pair_index =
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              right_orbital,
              occ_L[left_column]);
      selected_column.col(right_row).noalias() +=
          inverse_overlap(left_column, selected_slot) *
          ri_active_pair_factors.col(packed_pair_index);
    }
  }
  return selected_column;
}

Eigen::VectorXd build_right_string_delta_qx(
    const std::vector<int>& occ_L,
    int old_right_orbital,
    int new_right_orbital,
    const Eigen::MatrixXd& inverse_overlap,
    int selected_slot,
    const Eigen::MatrixXd& ri_active_pair_factors) {
  Eigen::VectorXd qx =
      Eigen::VectorXd::Zero(ri_active_pair_factors.rows());
  for (int left_column = 0;
       left_column < static_cast<int>(occ_L.size());
       ++left_column) {
    const int old_pair =
        xmvb::vb::TwoElectronIndexer::packed_pair_index(
            old_right_orbital,
            occ_L[left_column]);
    const int new_pair =
        xmvb::vb::TwoElectronIndexer::packed_pair_index(
            new_right_orbital,
            occ_L[left_column]);
    qx.noalias() +=
        inverse_overlap(left_column, selected_slot) *
        (ri_active_pair_factors.col(new_pair) -
         ri_active_pair_factors.col(old_pair));
  }
  return qx;
}

xmvb::vb::DeterminantOverlapResult resolve_overlap_result(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  const Eigen::MatrixXd overlap_block =
      xmvb::vb::build_overlap_submatrix(
          occ_L,
          occ_R,
          active_overlap,
          n_active_orbitals);
  return overlap_resolver.resolve_matrix(overlap_block);
}

SameSpinFormulaStats check_same_spin_formula(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& h1e_act,
    const xmvb::vb::ActiveSpaceTwoElectronResult& active_two_electron,
    int n_active_orbitals,
    int max_checks) {
  SameSpinFormulaStats stats;
  const auto& ri_factors = active_two_electron.ri_active_pair_factors;
  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;

  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) && stats.checked < max_checks;
       ++left) {
    for (int right = 0;
         right < static_cast<int>(unique_strings.size()) && stats.checked < max_checks;
         ++right) {
      const auto overlap_result = resolve_overlap_result(
          unique_strings[left],
          unique_strings[right],
          active_overlap,
          n_active_orbitals,
          overlap_resolver);
      if (overlap_result.nullity != 0 ||
          overlap_result.overlap_determinant == 0.0) {
        ++stats.skipped_singular;
        continue;
      }

      const Eigen::MatrixXd inverse_overlap =
          xmvb::vb::build_inverse_overlap_submatrix_from_result(overlap_result);
      const double ri_phi2 = compute_ri_same_spin_two_electron_phi(
          unique_strings[left],
          unique_strings[right],
          inverse_overlap,
          ri_factors);
      const auto reference_phi = xmvb::vb::compute_same_spin_original_phi(
          unique_strings[left],
          unique_strings[right],
          h1e_act,
          n_active_orbitals,
          active_two_electron,
          overlap_result,
          nullptr);
      const double reference_phi2 =
          reference_phi.total_phi - reference_phi.one_electron_phi;
      const double phi_diff = std::abs(ri_phi2 - reference_phi2);
      const double h_diff =
          std::abs(overlap_result.overlap_determinant * (ri_phi2 - reference_phi2));
      stats.max_abs_phi2_diff = std::max(stats.max_abs_phi2_diff, phi_diff);
      stats.max_abs_h2_diff = std::max(stats.max_abs_h2_diff, h_diff);
      ++stats.checked;
    }
  }

  return stats;
}

OppositeSpinFormulaStats check_opposite_spin_formula(
    const std::vector<std::vector<int>>& alpha_strings,
    const std::vector<std::vector<int>>& beta_strings,
    const std::vector<double>& active_overlap,
    const xmvb::vb::ActiveSpaceTwoElectronResult& active_two_electron,
    int n_active_orbitals,
    int max_checks) {
  OppositeSpinFormulaStats stats;
  const auto& ri_factors = active_two_electron.ri_active_pair_factors;
  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;

  for (int alpha_left = 0;
       alpha_left < static_cast<int>(alpha_strings.size()) &&
           stats.checked < max_checks;
       ++alpha_left) {
    for (int alpha_right = 0;
         alpha_right < static_cast<int>(alpha_strings.size()) &&
             stats.checked < max_checks;
         ++alpha_right) {
      const auto alpha_overlap_result = resolve_overlap_result(
          alpha_strings[alpha_left],
          alpha_strings[alpha_right],
          active_overlap,
          n_active_orbitals,
          overlap_resolver);
      if (alpha_overlap_result.nullity != 0 ||
          alpha_overlap_result.overlap_determinant == 0.0) {
        ++stats.skipped_singular;
        continue;
      }
      const Eigen::MatrixXd alpha_inverse =
          xmvb::vb::build_inverse_overlap_submatrix_from_result(
              alpha_overlap_result);
      const Eigen::VectorXd alpha_features =
          compute_ri_trace_features(
              alpha_strings[alpha_left],
              alpha_strings[alpha_right],
              alpha_inverse,
              ri_factors);

      for (int beta_left = 0;
           beta_left < static_cast<int>(beta_strings.size()) &&
               stats.checked < max_checks;
           ++beta_left) {
        for (int beta_right = 0;
             beta_right < static_cast<int>(beta_strings.size()) &&
                 stats.checked < max_checks;
             ++beta_right) {
          const auto beta_overlap_result = resolve_overlap_result(
              beta_strings[beta_left],
              beta_strings[beta_right],
              active_overlap,
              n_active_orbitals,
              overlap_resolver);
          if (beta_overlap_result.nullity != 0 ||
              beta_overlap_result.overlap_determinant == 0.0) {
            ++stats.skipped_singular;
            continue;
          }
          const Eigen::MatrixXd beta_inverse =
              xmvb::vb::build_inverse_overlap_submatrix_from_result(
                  beta_overlap_result);
          const Eigen::VectorXd beta_features =
              compute_ri_trace_features(
                  beta_strings[beta_left],
                  beta_strings[beta_right],
                  beta_inverse,
                  ri_factors);
          const double ri_phi = alpha_features.dot(beta_features);
          const double reference_phi = xmvb::vb::compute_opposite_spin_original_phi(
              alpha_strings[alpha_left],
              alpha_strings[alpha_right],
              alpha_overlap_result,
              beta_strings[beta_left],
              beta_strings[beta_right],
              beta_overlap_result,
              active_two_electron,
              nullptr,
              nullptr);
          const double determinant_product =
              alpha_overlap_result.overlap_determinant *
              beta_overlap_result.overlap_determinant;
          const double phi_diff = std::abs(ri_phi - reference_phi);
          const double h_diff =
              std::abs(determinant_product * (ri_phi - reference_phi));
          stats.max_abs_phi_diff = std::max(stats.max_abs_phi_diff, phi_diff);
          stats.max_abs_h_diff = std::max(stats.max_abs_h_diff, h_diff);
          ++stats.checked;
        }
      }
    }
  }
  return stats;
}

std::vector<int> changed_positions(
    const std::vector<int>& left,
    const std::vector<int>& right) {
  std::vector<int> positions;
  for (int index = 0; index < static_cast<int>(left.size()); ++index) {
    if (left[index] != right[index]) {
      positions.push_back(index);
    }
  }
  return positions;
}

bool apply_right_string_woodbury_update(
    const Eigen::MatrixXd& overlap_old,
    const Eigen::MatrixXd& overlap_new,
    const Eigen::MatrixXd& inverse_old,
    double determinant_old,
    const std::vector<int>& right_old,
    const std::vector<int>& right_new,
    Eigen::MatrixXd* inverse_updated,
    double* determinant_updated,
    int* update_rank) {
  const std::vector<int> rows = changed_positions(right_old, right_new);
  if (rows.empty()) {
    *inverse_updated = inverse_old;
    *determinant_updated = determinant_old;
    *update_rank = 0;
    return true;
  }

  const int n_electrons = static_cast<int>(right_old.size());
  const int rank = static_cast<int>(rows.size());
  Eigen::MatrixXd update_left = Eigen::MatrixXd::Zero(n_electrons, rank);
  Eigen::MatrixXd update_right_transpose(rank, n_electrons);
  const Eigen::MatrixXd delta_overlap = overlap_new - overlap_old;
  for (int local = 0; local < rank; ++local) {
    update_left(rows[local], local) = 1.0;
    update_right_transpose.row(local) = delta_overlap.row(rows[local]);
  }

  const Eigen::MatrixXd middle =
      Eigen::MatrixXd::Identity(rank, rank) +
      update_right_transpose * inverse_old * update_left;
  const Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
  if (!middle_lu.isInvertible()) {
    return false;
  }

  *inverse_updated =
      inverse_old -
      inverse_old * update_left * middle_lu.inverse() *
          update_right_transpose * inverse_old;
  *determinant_updated = determinant_old * middle.determinant();
  *update_rank = rank;
  return true;
}

RightStringUpdateData build_right_string_update_data(
    const Eigen::MatrixXd& overlap_old,
    const Eigen::MatrixXd& overlap_new,
    const Eigen::MatrixXd& inverse_old,
    double determinant_old,
    const std::vector<int>& right_old,
    const std::vector<int>& right_new) {
  RightStringUpdateData data;
  data.rows = changed_positions(right_old, right_new);
  if (data.rows.empty()) {
    data.inverse_updated = inverse_old;
    data.determinant_updated = determinant_old;
    data.ok = true;
    return data;
  }

  const int n_electrons = static_cast<int>(right_old.size());
  const int rank = static_cast<int>(data.rows.size());
  const Eigen::MatrixXd delta_overlap = overlap_new - overlap_old;
  data.update_right_transpose.resize(rank, n_electrons);
  Eigen::MatrixXd update_left = Eigen::MatrixXd::Zero(n_electrons, rank);
  for (int local = 0; local < rank; ++local) {
    update_left(data.rows[local], local) = 1.0;
    data.update_right_transpose.row(local) =
        delta_overlap.row(data.rows[local]);
  }

  const Eigen::MatrixXd middle =
      Eigen::MatrixXd::Identity(rank, rank) +
      data.update_right_transpose * inverse_old * update_left;
  const Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
  if (!middle_lu.isInvertible()) {
    return data;
  }

  data.middle_inverse = middle_lu.inverse();
  data.c_matrix = inverse_old * update_left * data.middle_inverse;
  data.d_matrix = data.update_right_transpose * inverse_old;
  data.inverse_updated =
      inverse_old - data.c_matrix * data.d_matrix;
  data.determinant_updated = determinant_old * middle.determinant();
  data.ok = true;
  return data;
}

Eigen::MatrixXd build_right_string_ri_delta_rows(
    const std::vector<int>& occ_L,
    const std::vector<int>& right_old,
    const std::vector<int>& right_new,
    const std::vector<int>& changed_rows,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int auxiliary_index) {
  Eigen::MatrixXd delta_rows(
      static_cast<int>(changed_rows.size()),
      static_cast<int>(occ_L.size()));
  for (int local = 0; local < static_cast<int>(changed_rows.size()); ++local) {
    const int row = changed_rows[local];
    const int old_right_orbital = right_old[row];
    const int new_right_orbital = right_new[row];
    for (int left_column = 0;
         left_column < static_cast<int>(occ_L.size());
         ++left_column) {
      const int left_orbital = occ_L[left_column];
      const int old_pair =
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              old_right_orbital,
              left_orbital);
      const int new_pair =
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              new_right_orbital,
              left_orbital);
      delta_rows(local, left_column) =
          ri_active_pair_factors(auxiliary_index, new_pair) -
          ri_active_pair_factors(auxiliary_index, old_pair);
    }
  }
  return delta_rows;
}

void update_right_string_channel_matrices(
    const std::vector<int>& occ_L,
    const std::vector<int>& right_old,
    const std::vector<int>& right_new,
    const Eigen::MatrixXd& inverse_old,
    const RightStringUpdateData& update_data,
    const Eigen::MatrixXd& ri_active_pair_factors,
    std::vector<Eigen::MatrixXd>* channel_matrices) {
  if (update_data.rows.empty()) {
    return;
  }

  for (int auxiliary_index = 0;
       auxiliary_index < static_cast<int>(channel_matrices->size());
       ++auxiliary_index) {
    Eigen::MatrixXd& channel_matrix = (*channel_matrices)[auxiliary_index];
    Eigen::MatrixXd selected_columns(
        channel_matrix.rows(),
        static_cast<int>(update_data.rows.size()));
    for (int local = 0;
         local < static_cast<int>(update_data.rows.size());
         ++local) {
      selected_columns.col(local) =
          channel_matrix.col(update_data.rows[local]);
    }

    const Eigen::MatrixXd delta_rows =
        build_right_string_ri_delta_rows(
            occ_L,
            right_old,
            right_new,
            update_data.rows,
            ri_active_pair_factors,
            auxiliary_index);
    channel_matrix.noalias() -=
        selected_columns * update_data.middle_inverse * update_data.d_matrix;
    const Eigen::MatrixXd row_update =
        delta_rows * inverse_old -
        (delta_rows * update_data.c_matrix) * update_data.d_matrix;
    for (int local = 0;
         local < static_cast<int>(update_data.rows.size());
         ++local) {
      channel_matrix.row(update_data.rows[local]).noalias() +=
          row_update.row(local);
    }
  }
}

int permutation_sign_to_sorted(
    const std::vector<int>& internal_order,
    const std::vector<int>& sorted_order) {
  std::vector<int> positions;
  positions.reserve(internal_order.size());
  for (const int orbital : internal_order) {
    const auto iterator = std::find(
        sorted_order.begin(),
        sorted_order.end(),
        orbital);
    if (iterator == sorted_order.end()) {
      throw std::invalid_argument(
          "internal occupied order is not a permutation of sorted order");
    }
    positions.push_back(
        static_cast<int>(std::distance(sorted_order.begin(), iterator)));
  }

  int inversions = 0;
  for (int left = 0; left < static_cast<int>(positions.size()); ++left) {
    for (int right = left + 1; right < static_cast<int>(positions.size()); ++right) {
      if (positions[left] > positions[right]) {
        ++inversions;
      }
    }
  }
  return (inversions % 2 == 0) ? 1 : -1;
}

bool find_single_replacement_slot(
    const std::vector<int>& current_internal_order,
    const std::vector<int>& next_sorted_order,
    int* slot,
    int* old_orbital,
    int* new_orbital) {
  int removed = -1;
  int added = -1;
  for (const int orbital : current_internal_order) {
    if (std::find(next_sorted_order.begin(), next_sorted_order.end(), orbital) ==
        next_sorted_order.end()) {
      if (removed >= 0) {
        return false;
      }
      removed = orbital;
    }
  }
  for (const int orbital : next_sorted_order) {
    if (std::find(
            current_internal_order.begin(),
            current_internal_order.end(),
            orbital) == current_internal_order.end()) {
      if (added >= 0) {
        return false;
      }
      added = orbital;
    }
  }
  if (removed < 0 || added < 0) {
    return false;
  }

  const auto removed_iterator = std::find(
      current_internal_order.begin(),
      current_internal_order.end(),
      removed);
  *slot = static_cast<int>(
      std::distance(current_internal_order.begin(), removed_iterator));
  *old_orbital = removed;
  *new_orbital = added;
  return true;
}

void apply_right_string_ri_block_update(
    const std::vector<int>& occ_L,
    const std::vector<int>& right_old,
    const std::vector<int>& right_new,
    const Eigen::MatrixXd& ri_active_pair_factors,
    std::vector<Eigen::MatrixXd>* occupied_blocks) {
  const std::vector<int> rows = changed_positions(right_old, right_new);
  for (int auxiliary_index = 0;
       auxiliary_index < static_cast<int>(occupied_blocks->size());
       ++auxiliary_index) {
    Eigen::MatrixXd& occupied_block = (*occupied_blocks)[auxiliary_index];
    for (const int row : rows) {
      const int right_orbital = right_new[row];
      for (int left_column = 0;
           left_column < static_cast<int>(occ_L.size());
           ++left_column) {
        const int packed_pair_index =
            xmvb::vb::TwoElectronIndexer::packed_pair_index(
                right_orbital,
                occ_L[left_column]);
        occupied_block(row, left_column) =
            ri_active_pair_factors(auxiliary_index, packed_pair_index);
      }
    }
  }
}

RankUpdateStats check_right_string_rank_update(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int max_checks) {
  RankUpdateStats stats;
  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;

  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) && stats.checked < max_checks;
       ++left) {
    for (int right_old = 0;
         right_old < static_cast<int>(unique_strings.size()) &&
             stats.checked < max_checks;
         ++right_old) {
      for (int right_new = 0;
           right_new < static_cast<int>(unique_strings.size()) &&
               stats.checked < max_checks;
           ++right_new) {
        if (right_old == right_new ||
            replacement_distance(
                unique_strings[right_old],
                unique_strings[right_new]) != 1) {
          continue;
        }

        const Eigen::MatrixXd overlap_old =
            xmvb::vb::build_overlap_submatrix(
                unique_strings[left],
                unique_strings[right_old],
                active_overlap,
                n_active_orbitals);
        const Eigen::MatrixXd overlap_new =
            xmvb::vb::build_overlap_submatrix(
                unique_strings[left],
                unique_strings[right_new],
                active_overlap,
                n_active_orbitals);
        const auto overlap_result_old =
            overlap_resolver.resolve_matrix(overlap_old);
        const auto direct_overlap_start =
            std::chrono::high_resolution_clock::now();
        const auto overlap_result_new =
            overlap_resolver.resolve_matrix(overlap_new);
        const Eigen::MatrixXd inverse_new =
            xmvb::vb::build_inverse_overlap_submatrix_from_result(
                overlap_result_new);
        const auto direct_overlap_end =
            std::chrono::high_resolution_clock::now();
        stats.direct_overlap_seconds +=
            std::chrono::duration<double>(
                direct_overlap_end - direct_overlap_start)
                .count();
        if (overlap_result_old.nullity != 0 ||
            overlap_result_new.nullity != 0 ||
            overlap_result_old.overlap_determinant == 0.0 ||
            overlap_result_new.overlap_determinant == 0.0) {
          ++stats.skipped_singular;
          continue;
        }

        const Eigen::MatrixXd inverse_old =
            xmvb::vb::build_inverse_overlap_submatrix_from_result(
                overlap_result_old);
        Eigen::MatrixXd inverse_updated;
        double determinant_updated = 0.0;
        int update_rank = 0;
        const auto rank_update_start =
            std::chrono::high_resolution_clock::now();
        const bool rank_update_ok = apply_right_string_woodbury_update(
                overlap_old,
                overlap_new,
                inverse_old,
                overlap_result_old.overlap_determinant,
                unique_strings[right_old],
                unique_strings[right_new],
                &inverse_updated,
                &determinant_updated,
                &update_rank);
        const auto rank_update_end =
            std::chrono::high_resolution_clock::now();
        stats.rank_update_seconds +=
            std::chrono::duration<double>(rank_update_end - rank_update_start)
                .count();
        if (!rank_update_ok) {
          continue;
        }

        std::vector<Eigen::MatrixXd> updated_blocks = build_all_ri_occupied_blocks(
            unique_strings[left],
            unique_strings[right_old],
            ri_active_pair_factors);
        apply_right_string_ri_block_update(
            unique_strings[left],
            unique_strings[right_old],
            unique_strings[right_new],
            ri_active_pair_factors,
            &updated_blocks);

        const double phi_updated =
            compute_ri_same_spin_two_electron_phi_from_blocks(
                updated_blocks,
                inverse_updated);
        const double phi_direct =
            compute_ri_same_spin_two_electron_phi(
                unique_strings[left],
                unique_strings[right_new],
                inverse_new,
                ri_active_pair_factors);

        const double inverse_diff =
            (inverse_updated - inverse_new).cwiseAbs().maxCoeff();
        const double determinant_diff =
            std::abs(determinant_updated - overlap_result_new.overlap_determinant);
        const double phi_diff = std::abs(phi_updated - phi_direct);
        stats.max_abs_inverse_diff =
            std::max(stats.max_abs_inverse_diff, inverse_diff);
        stats.max_abs_det_diff =
            std::max(stats.max_abs_det_diff, determinant_diff);
        stats.max_abs_phi2_diff =
            std::max(stats.max_abs_phi2_diff, phi_diff);
        stats.mean_replacement_distance += 1.0;
        stats.mean_update_rank += static_cast<double>(update_rank);
        stats.max_update_rank = std::max(stats.max_update_rank, update_rank);
        ++stats.checked;
      }
    }
  }

  if (stats.checked > 0) {
    stats.mean_replacement_distance /= static_cast<double>(stats.checked);
    stats.mean_update_rank /= static_cast<double>(stats.checked);
  }
  return stats;
}

RankOneOverlapStats benchmark_slot_stable_rank_one_overlap_update(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    int max_checks) {
  RankOneOverlapStats stats;
  if (unique_strings.empty()) {
    return stats;
  }

  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  const std::vector<int> right_order =
      build_slot_stable_traversal_order(unique_strings);
  const int n_electrons =
      static_cast<int>(unique_strings.front().size());

  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) &&
           stats.checked < max_checks;
       ++left) {
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd inverse_overlap;
    double determinant = 0.0;
    std::vector<int> current_internal_order;
    bool state_valid = false;

    for (const int right : right_order) {
      if (stats.checked >= max_checks) {
        break;
      }

      if (!state_valid) {
        current_internal_order = unique_strings[right];
        overlap = xmvb::vb::build_overlap_submatrix(
            unique_strings[left],
            current_internal_order,
            active_overlap,
            n_active_orbitals);
        const auto overlap_result = overlap_resolver.resolve_matrix(overlap);
        if (overlap_result.nullity != 0 ||
            overlap_result.overlap_determinant == 0.0) {
          ++stats.skipped_singular;
          continue;
        }
        inverse_overlap =
            xmvb::vb::build_inverse_overlap_submatrix_from_result(
                overlap_result);
        determinant = overlap_result.overlap_determinant;
        state_valid = true;
        continue;
      }

      int slot = -1;
      int old_orbital = -1;
      int new_orbital = -1;
      if (!find_single_replacement_slot(
              current_internal_order,
              unique_strings[right],
              &slot,
              &old_orbital,
              &new_orbital)) {
        state_valid = false;
        ++stats.resets;
        continue;
      }

      std::vector<int> next_internal_order = current_internal_order;
      next_internal_order[slot] = new_orbital;
      const Eigen::MatrixXd overlap_new =
          xmvb::vb::build_overlap_submatrix(
              unique_strings[left],
              next_internal_order,
              active_overlap,
              n_active_orbitals);

      const auto direct_start = std::chrono::high_resolution_clock::now();
      const auto overlap_result_new =
          overlap_resolver.resolve_matrix(overlap_new);
      const Eigen::MatrixXd inverse_direct =
          xmvb::vb::build_inverse_overlap_submatrix_from_result(
              overlap_result_new);
      const auto direct_end = std::chrono::high_resolution_clock::now();
      stats.direct_overlap_seconds +=
          std::chrono::duration<double>(direct_end - direct_start).count();

      if (overlap_result_new.nullity != 0 ||
          overlap_result_new.overlap_determinant == 0.0) {
        state_valid = false;
        ++stats.skipped_singular;
        continue;
      }

      const auto rank_one_start = std::chrono::high_resolution_clock::now();
      Eigen::VectorXd row_delta(n_electrons);
      for (int left_column = 0; left_column < n_electrons; ++left_column) {
        const int left_orbital = unique_strings[left][left_column];
        row_delta(left_column) =
            active_overlap[left_orbital * n_active_orbitals + new_orbital] -
            active_overlap[left_orbital * n_active_orbitals + old_orbital];
      }
      const Eigen::VectorXd selected_inverse_column =
          inverse_overlap.col(slot);
      const Eigen::RowVectorXd row_delta_times_inverse =
          row_delta.transpose() * inverse_overlap;
      const double eta =
          1.0 + row_delta_times_inverse(slot);
      Eigen::MatrixXd inverse_updated;
      double determinant_updated = 0.0;
      if (std::abs(eta) > std::numeric_limits<double>::epsilon()) {
        inverse_updated.noalias() =
            inverse_overlap -
            (selected_inverse_column * row_delta_times_inverse) / eta;
        determinant_updated = determinant * eta;
      }
      const auto rank_one_end = std::chrono::high_resolution_clock::now();
      stats.rank_one_seconds +=
          std::chrono::duration<double>(rank_one_end - rank_one_start).count();

      if (std::abs(eta) <= std::numeric_limits<double>::epsilon() ||
          determinant_updated == 0.0) {
        state_valid = false;
        ++stats.resets;
        continue;
      }

      stats.max_abs_inverse_diff = std::max(
          stats.max_abs_inverse_diff,
          (inverse_updated - inverse_direct).cwiseAbs().maxCoeff());
      stats.max_abs_det_diff = std::max(
          stats.max_abs_det_diff,
          std::abs(
              determinant_updated - overlap_result_new.overlap_determinant));

      overlap = overlap_new;
      inverse_overlap = std::move(inverse_updated);
      determinant = determinant_updated;
      current_internal_order = std::move(next_internal_order);
      ++stats.checked;
    }
  }

  return stats;
}

RiQUpdateStats benchmark_slot_stable_ri_q_update(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int max_checks) {
  RiQUpdateStats stats;
  if (unique_strings.empty()) {
    return stats;
  }

  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  const std::vector<int> right_order =
      build_slot_stable_traversal_order(unique_strings);
  Eigen::VectorXd direct_q(ri_active_pair_factors.rows());

  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) &&
           stats.checked < max_checks;
       ++left) {
    ScalarRiFeatureTraversalState state;

    for (const int right : right_order) {
      if (stats.checked >= max_checks) {
        break;
      }

      if (!state.valid) {
        initialize_scalar_ri_feature_traversal_state(
            unique_strings[left],
            unique_strings[right],
            unique_strings[right],
            active_overlap,
            ri_active_pair_factors,
            n_active_orbitals,
            right,
            overlap_resolver,
            &state);
        continue;
      }

      int slot = -1;
      int old_orbital = -1;
      int new_orbital = -1;
      if (!find_single_replacement_slot(
              state.right_internal_order,
              unique_strings[right],
              &slot,
              &old_orbital,
              &new_orbital)) {
        state.valid = false;
        ++stats.resets;
        continue;
      }

      std::vector<int> next_internal_order = state.right_internal_order;
      next_internal_order[slot] = new_orbital;
      const Eigen::MatrixXd overlap_new =
          xmvb::vb::build_overlap_submatrix(
              unique_strings[left],
              next_internal_order,
              active_overlap,
              n_active_orbitals);
      const RightStringUpdateData update_data =
          build_right_string_update_data(
              state.overlap,
              overlap_new,
              state.inverse_overlap,
              state.determinant,
              state.right_internal_order,
              next_internal_order);
      if (!update_data.ok ||
          update_data.rows.size() != 1 ||
          update_data.rows.front() != slot ||
          update_data.determinant_updated == 0.0) {
        state.valid = false;
        ++stats.resets;
        continue;
      }

      const auto direct_q_start = std::chrono::high_resolution_clock::now();
      fill_direct_ri_first_order_feature(
          unique_strings[left],
          next_internal_order,
          update_data.inverse_updated,
          1.0,
          ri_active_pair_factors,
          &direct_q);
      const auto direct_q_end = std::chrono::high_resolution_clock::now();
      stats.direct_q_seconds +=
          std::chrono::duration<double>(direct_q_end - direct_q_start).count();

      const auto rank_q_start = std::chrono::high_resolution_clock::now();
      if (state.selected_slot != slot ||
          state.selected_channel_column.rows() != ri_active_pair_factors.rows()) {
        state.selected_channel_column =
            build_selected_channel_column(
                unique_strings[left],
                state.right_internal_order,
                state.inverse_overlap,
                slot,
                ri_active_pair_factors);
        state.selected_slot = slot;
        ++stats.selected_column_rebuilds;
      }

      const double inverse_middle = update_data.middle_inverse(0, 0);
      const Eigen::VectorXd qx =
          build_right_string_delta_qx(
              unique_strings[left],
              old_orbital,
              new_orbital,
              state.inverse_overlap,
              slot,
              ri_active_pair_factors);
      const Eigen::VectorXd d_m =
          state.selected_channel_column * update_data.d_matrix.transpose();
      state.x_features.noalias() +=
          inverse_middle * (qx - d_m);
      state.selected_channel_column *= inverse_middle;
      state.selected_channel_column.col(slot).noalias() +=
          inverse_middle * qx;
      const auto rank_q_end = std::chrono::high_resolution_clock::now();
      stats.rank_update_q_seconds +=
          std::chrono::duration<double>(rank_q_end - rank_q_start).count();

      stats.max_abs_q_diff = std::max(
          stats.max_abs_q_diff,
          (state.x_features - direct_q).cwiseAbs().maxCoeff());

      state.overlap = overlap_new;
      state.inverse_overlap = update_data.inverse_updated;
      state.determinant = update_data.determinant_updated;
      state.right_index = right;
      state.right_internal_order = std::move(next_internal_order);
      state.canonical_sign = permutation_sign_to_sorted(
          state.right_internal_order,
          unique_strings[right]);
      ++stats.checked;
    }
  }

  return stats;
}

RiKUpdateStats benchmark_ri_k_channel_matrix_update(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int max_checks) {
  RiKUpdateStats stats;
  if (unique_strings.empty()) {
    return stats;
  }

  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  const std::vector<int> right_order =
      build_slot_stable_traversal_order(unique_strings);

  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) &&
           stats.checked < max_checks;
       ++left) {
    RiFeatureTraversalState state;

    for (const int right : right_order) {
      if (stats.checked >= max_checks) {
        break;
      }

      if (!state.valid) {
        if (!initialize_ri_feature_traversal_state(
                unique_strings[left],
                unique_strings[right],
                active_overlap,
                ri_active_pair_factors,
                n_active_orbitals,
                right,
                overlap_resolver,
                &state)) {
          ++stats.skipped_singular;
        }
        continue;
      }

      const Eigen::MatrixXd overlap_new =
          xmvb::vb::build_overlap_submatrix(
              unique_strings[left],
              unique_strings[right],
              active_overlap,
              n_active_orbitals);
      const auto direct_start = std::chrono::high_resolution_clock::now();
      const auto overlap_result_new =
          overlap_resolver.resolve_matrix(overlap_new);
      if (overlap_result_new.nullity != 0 ||
          overlap_result_new.overlap_determinant == 0.0) {
        state.valid = false;
        ++stats.skipped_singular;
        continue;
      }
      const Eigen::MatrixXd inverse_direct =
          xmvb::vb::build_inverse_overlap_submatrix_from_result(
              overlap_result_new);
      const double direct_phi2 =
          compute_ri_same_spin_two_electron_phi(
              unique_strings[left],
              unique_strings[right],
              inverse_direct,
              ri_active_pair_factors);
      const auto direct_end = std::chrono::high_resolution_clock::now();
      stats.direct_seconds +=
          std::chrono::duration<double>(direct_end - direct_start).count();

      const auto update_start = std::chrono::high_resolution_clock::now();
      const RightStringUpdateData update_data =
          build_right_string_update_data(
              state.overlap,
              overlap_new,
              state.inverse_overlap,
              state.determinant,
              unique_strings[state.right_index],
              unique_strings[right]);
      if (!update_data.ok || update_data.determinant_updated == 0.0) {
        state.valid = false;
        ++stats.resets;
        continue;
      }
      update_right_string_channel_matrices(
          unique_strings[left],
          unique_strings[state.right_index],
          unique_strings[right],
          state.inverse_overlap,
          update_data,
          ri_active_pair_factors,
          &state.channel_matrices);
      const double updated_phi2 =
          compute_phi2_from_channel_matrices(state.channel_matrices);
      const auto update_end = std::chrono::high_resolution_clock::now();
      stats.update_seconds +=
          std::chrono::duration<double>(update_end - update_start).count();

      stats.max_abs_phi2_diff =
          std::max(stats.max_abs_phi2_diff, std::abs(updated_phi2 - direct_phi2));
      state.overlap = overlap_new;
      state.inverse_overlap = update_data.inverse_updated;
      state.determinant = update_data.determinant_updated;
      state.right_index = right;
      ++stats.checked;
    }
  }

  return stats;
}

bool initialize_ri_feature_traversal_state(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int right_index,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    RiFeatureTraversalState* state) {
  const Eigen::MatrixXd overlap =
      xmvb::vb::build_overlap_submatrix(
          occ_L,
          occ_R,
          active_overlap,
          n_active_orbitals);
  const auto overlap_result = overlap_resolver.resolve_matrix(overlap);
  if (overlap_result.nullity != 0 ||
      overlap_result.overlap_determinant == 0.0) {
    state->valid = false;
    return false;
  }

  state->overlap = overlap;
  state->inverse_overlap =
      xmvb::vb::build_inverse_overlap_submatrix_from_result(overlap_result);
  state->determinant = overlap_result.overlap_determinant;
  state->right_index = right_index;
  state->channel_matrices = build_all_ri_channel_matrices(
      occ_L,
      occ_R,
      state->inverse_overlap,
      ri_active_pair_factors);
  state->valid = true;
  return true;
}

bool initialize_scalar_ri_feature_traversal_state(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R_internal,
    const std::vector<int>& occ_R_sorted,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int right_index,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ScalarRiFeatureTraversalState* state) {
  const Eigen::MatrixXd overlap =
      xmvb::vb::build_overlap_submatrix(
          occ_L,
          occ_R_internal,
          active_overlap,
          n_active_orbitals);
  const auto overlap_result = overlap_resolver.resolve_matrix(overlap);
  if (overlap_result.nullity != 0 ||
      overlap_result.overlap_determinant == 0.0) {
    state->valid = false;
    return false;
  }

  state->overlap = overlap;
  state->inverse_overlap =
      xmvb::vb::build_inverse_overlap_submatrix_from_result(overlap_result);
  state->determinant = overlap_result.overlap_determinant;
  state->right_index = right_index;
  state->right_internal_order = occ_R_internal;
  state->x_features.resize(ri_active_pair_factors.rows());
  fill_direct_ri_first_order_feature(
      occ_L,
      occ_R_internal,
      state->inverse_overlap,
      1.0,
      ri_active_pair_factors,
      &state->x_features);
  state->selected_channel_column.resize(0, 0);
  state->selected_slot = -1;
  state->canonical_sign =
      permutation_sign_to_sorted(occ_R_internal, occ_R_sorted);
  state->valid = true;
  return true;
}

double traversal_state_feature_checksum(
    const RiFeatureTraversalState& state) {
  double checksum = 0.0;
  for (const auto& channel_matrix : state.channel_matrices) {
    const double feature = state.determinant * channel_matrix.trace();
    checksum += feature * feature;
  }
  return checksum;
}

double compute_phi2_from_channel_matrices(
    const std::vector<Eigen::MatrixXd>& channel_matrices) {
  double phi2 = 0.0;
  for (const auto& channel_matrix : channel_matrices) {
    const double trace_value = channel_matrix.trace();
    phi2 += trace_value * trace_value -
            (channel_matrix * channel_matrix).trace();
  }
  return 0.5 * phi2;
}

void accumulate_scalar_state_checksums(
    const ScalarRiFeatureTraversalState& state,
    OppositeSpinFeatureBenchmarkStats* stats) {
  const Eigen::VectorXd canonical_feature =
      static_cast<double>(state.canonical_sign) *
      state.determinant *
      state.x_features;
  stats->scalar_checksum += feature_checksum(canonical_feature);
  stats->scalar_linear_checksum += feature_linear_checksum(canonical_feature);
}

bool update_scalar_ri_feature_traversal_state(
    const std::vector<int>& occ_L,
    const std::vector<int>& next_right_sorted,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int next_right_index,
    ScalarRiFeatureTraversalState* state,
    int* selected_column_rebuilt) {
  int slot = -1;
  int old_orbital = -1;
  int new_orbital = -1;
  if (!find_single_replacement_slot(
          state->right_internal_order,
          next_right_sorted,
          &slot,
          &old_orbital,
          &new_orbital)) {
    return false;
  }

  std::vector<int> next_internal_order = state->right_internal_order;
  next_internal_order[slot] = new_orbital;
  const Eigen::MatrixXd overlap_new =
      xmvb::vb::build_overlap_submatrix(
          occ_L,
          next_internal_order,
          active_overlap,
          n_active_orbitals);
  const RightStringUpdateData update_data =
      build_right_string_update_data(
          state->overlap,
          overlap_new,
          state->inverse_overlap,
          state->determinant,
          state->right_internal_order,
          next_internal_order);
  if (!update_data.ok ||
      update_data.rows.size() != 1 ||
      update_data.rows.front() != slot ||
      update_data.determinant_updated == 0.0) {
    return false;
  }

  if (state->selected_slot != slot ||
      state->selected_channel_column.rows() != ri_active_pair_factors.rows()) {
    state->selected_channel_column =
        build_selected_channel_column(
            occ_L,
            state->right_internal_order,
            state->inverse_overlap,
            slot,
            ri_active_pair_factors);
    state->selected_slot = slot;
    ++(*selected_column_rebuilt);
  }

  const double inverse_middle = update_data.middle_inverse(0, 0);
  const Eigen::VectorXd qx =
      build_right_string_delta_qx(
          occ_L,
          old_orbital,
          new_orbital,
          state->inverse_overlap,
          slot,
          ri_active_pair_factors);
  const Eigen::VectorXd d_m =
      state->selected_channel_column * update_data.d_matrix.transpose();
  state->x_features.noalias() +=
      inverse_middle * (qx - d_m);

  state->selected_channel_column *= inverse_middle;
  state->selected_channel_column.col(slot).noalias() +=
      inverse_middle * qx;
  state->overlap = overlap_new;
  state->inverse_overlap = update_data.inverse_updated;
  state->determinant = update_data.determinant_updated;
  state->right_index = next_right_index;
  state->right_internal_order = std::move(next_internal_order);
  state->canonical_sign =
      permutation_sign_to_sorted(state->right_internal_order, next_right_sorted);
  return true;
}

OppositeSpinFeatureBenchmarkStats benchmark_opposite_spin_feature_generation(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& ri_active_pair_factors,
    int n_active_orbitals,
    int max_feature_pairs) {
  OppositeSpinFeatureBenchmarkStats stats;
  if (unique_strings.empty()) {
    return stats;
  }

  const xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  Eigen::VectorXd feature(ri_active_pair_factors.rows());

  const auto direct_start = std::chrono::high_resolution_clock::now();
  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) &&
           stats.direct_pairs < max_feature_pairs;
       ++left) {
    for (int right = 0;
         right < static_cast<int>(unique_strings.size()) &&
             stats.direct_pairs < max_feature_pairs;
         ++right) {
      const auto overlap_result = resolve_overlap_result(
          unique_strings[left],
          unique_strings[right],
          active_overlap,
          n_active_orbitals,
          overlap_resolver);
      if (overlap_result.nullity != 0 ||
          overlap_result.overlap_determinant == 0.0) {
        continue;
      }
      const Eigen::MatrixXd inverse_overlap =
          xmvb::vb::build_inverse_overlap_submatrix_from_result(overlap_result);
      fill_direct_ri_first_order_feature(
          unique_strings[left],
          unique_strings[right],
          inverse_overlap,
          overlap_result.overlap_determinant,
          ri_active_pair_factors,
          &feature);
      stats.direct_checksum += feature_checksum(feature);
      stats.direct_linear_checksum += feature_linear_checksum(feature);
      ++stats.direct_pairs;
    }
  }
  const auto direct_end = std::chrono::high_resolution_clock::now();
  stats.direct_seconds =
      std::chrono::duration<double>(direct_end - direct_start).count();

  const std::vector<int> right_order =
      build_greedy_traversal_order(unique_strings);
  const std::vector<int> scalar_right_order =
      build_slot_stable_traversal_order(unique_strings);
  const auto low_rank_start = std::chrono::high_resolution_clock::now();
  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) &&
           stats.low_rank_pairs < max_feature_pairs;
       ++left) {
    RiFeatureTraversalState state;
    for (const int right : right_order) {
      if (stats.low_rank_pairs >= max_feature_pairs) {
        break;
      }
      if (!state.valid) {
        if (initialize_ri_feature_traversal_state(
                unique_strings[left],
                unique_strings[right],
                active_overlap,
                ri_active_pair_factors,
                n_active_orbitals,
                right,
                overlap_resolver,
                &state)) {
          ++stats.low_rank_initializations;
          stats.low_rank_checksum += traversal_state_feature_checksum(state);
          ++stats.low_rank_pairs;
        }
        continue;
      }

      const Eigen::MatrixXd overlap_new =
          xmvb::vb::build_overlap_submatrix(
              unique_strings[left],
              unique_strings[right],
              active_overlap,
              n_active_orbitals);
      const RightStringUpdateData update_data =
          build_right_string_update_data(
              state.overlap,
              overlap_new,
              state.inverse_overlap,
              state.determinant,
              unique_strings[state.right_index],
              unique_strings[right]);
      if (!update_data.ok || update_data.determinant_updated == 0.0) {
        ++stats.low_rank_resets;
        if (initialize_ri_feature_traversal_state(
                unique_strings[left],
                unique_strings[right],
                active_overlap,
                ri_active_pair_factors,
                n_active_orbitals,
                right,
                overlap_resolver,
                &state)) {
          ++stats.low_rank_initializations;
          stats.low_rank_checksum += traversal_state_feature_checksum(state);
          ++stats.low_rank_pairs;
        }
        continue;
      }

      update_right_string_channel_matrices(
          unique_strings[left],
          unique_strings[state.right_index],
          unique_strings[right],
          state.inverse_overlap,
          update_data,
          ri_active_pair_factors,
          &state.channel_matrices);
      state.overlap = overlap_new;
      state.inverse_overlap = update_data.inverse_updated;
      state.determinant = update_data.determinant_updated;
      state.right_index = right;
      ++stats.low_rank_updates;
      stats.low_rank_checksum += traversal_state_feature_checksum(state);
      ++stats.low_rank_pairs;
    }
  }
  const auto low_rank_end = std::chrono::high_resolution_clock::now();
  stats.low_rank_seconds =
      std::chrono::duration<double>(low_rank_end - low_rank_start).count();
  stats.checksum_abs_diff =
      std::abs(stats.direct_checksum - stats.low_rank_checksum);

  const auto scalar_start = std::chrono::high_resolution_clock::now();
  for (int left = 0;
       left < static_cast<int>(unique_strings.size()) &&
           stats.scalar_pairs < max_feature_pairs;
       ++left) {
    ScalarRiFeatureTraversalState state;
    for (const int right : scalar_right_order) {
      if (stats.scalar_pairs >= max_feature_pairs) {
        break;
      }
      if (!state.valid) {
        if (initialize_scalar_ri_feature_traversal_state(
                unique_strings[left],
                unique_strings[right],
                unique_strings[right],
                active_overlap,
                ri_active_pair_factors,
                n_active_orbitals,
                right,
                overlap_resolver,
                &state)) {
          ++stats.scalar_initializations;
          accumulate_scalar_state_checksums(state, &stats);
          ++stats.scalar_pairs;
        }
        continue;
      }

      int selected_column_rebuilt = 0;
      if (!update_scalar_ri_feature_traversal_state(
              unique_strings[left],
              unique_strings[right],
              active_overlap,
              ri_active_pair_factors,
              n_active_orbitals,
              right,
              &state,
              &selected_column_rebuilt)) {
        ++stats.scalar_resets;
        if (initialize_scalar_ri_feature_traversal_state(
                unique_strings[left],
                unique_strings[right],
                unique_strings[right],
                active_overlap,
                ri_active_pair_factors,
                n_active_orbitals,
                right,
                overlap_resolver,
                &state)) {
          ++stats.scalar_initializations;
          accumulate_scalar_state_checksums(state, &stats);
          ++stats.scalar_pairs;
        }
        continue;
      }

      ++stats.scalar_updates;
      stats.scalar_selected_column_rebuilds += selected_column_rebuilt;
      accumulate_scalar_state_checksums(state, &stats);
      ++stats.scalar_pairs;
    }
  }
  const auto scalar_end = std::chrono::high_resolution_clock::now();
  stats.scalar_seconds =
      std::chrono::duration<double>(scalar_end - scalar_start).count();
  stats.scalar_checksum_abs_diff =
      std::abs(stats.direct_checksum - stats.scalar_checksum);
  stats.scalar_linear_checksum_abs_diff =
      std::abs(stats.direct_linear_checksum - stats.scalar_linear_checksum);
  return stats;
}

void print_histogram(
    const std::string& prefix,
    const std::vector<long long>& histogram) {
  std::cout << prefix << "_histogram =";
  for (int index = 0; index < static_cast<int>(histogram.size()); ++index) {
    if (histogram[index] == 0) {
      continue;
    }
    std::cout << ' ' << index << ':' << histogram[index];
  }
  std::cout << '\n';
}

void print_locality(
    const std::string& label,
    const LocalityStats& locality,
    const TraversalStats& traversal,
    int n_auxiliary_functions) {
  std::cout << label << "_n_unique = " << locality.n_strings << '\n';
  std::cout << label << "_n_electrons = " << locality.n_electrons << '\n';
  std::cout << label << "_mean_replacement_distance = "
            << locality.mean_replacement_distance << '\n';
  std::cout << label << "_mean_positional_distance = "
            << locality.mean_positional_distance << '\n';
  std::cout << label << "_replacement_d1_fraction = "
            << locality.replacement_d1_fraction << '\n';
  std::cout << label << "_replacement_d2_fraction = "
            << locality.replacement_d2_fraction << '\n';
  std::cout << label << "_positional_d1_fraction = "
            << locality.positional_d1_fraction << '\n';
  std::cout << label << "_positional_d2_fraction = "
            << locality.positional_d2_fraction << '\n';
  print_histogram(label + "_replacement", locality.replacement_histogram);
  print_histogram(label + "_positional", locality.positional_histogram);

  std::cout << label << "_greedy_edges = " << traversal.n_edges << '\n';
  std::cout << label << "_greedy_mean_replacement_distance = "
            << traversal.mean_replacement_distance << '\n';
  std::cout << label << "_greedy_mean_positional_distance = "
            << traversal.mean_positional_distance << '\n';
  std::cout << label << "_greedy_max_replacement_distance = "
            << traversal.max_replacement_distance << '\n';
  std::cout << label << "_greedy_max_positional_distance = "
            << traversal.max_positional_distance << '\n';
  std::cout << label << "_greedy_replacement_d1_fraction = "
            << traversal.replacement_d1_fraction << '\n';
  std::cout << label << "_greedy_replacement_d2_fraction = "
            << traversal.replacement_d2_fraction << '\n';
  std::cout << label << "_greedy_positional_d1_fraction = "
            << traversal.positional_d1_fraction << '\n';
  std::cout << label << "_greedy_positional_d2_fraction = "
            << traversal.positional_d2_fraction << '\n';

  const double rho_ss =
      locality.n_electrons > 0
          ? static_cast<double>(n_auxiliary_functions) *
                traversal.mean_positional_distance /
                static_cast<double>(locality.n_electrons * locality.n_electrons)
          : 0.0;
  std::cout << label << "_rho_ss_greedy_positional = " << rho_ss << '\n';
}

void print_same_spin_formula_stats(
    const std::string& label,
    const SameSpinFormulaStats& stats) {
  std::cout << label << "_same_spin_formula_checked = " << stats.checked << '\n';
  std::cout << label << "_same_spin_formula_skipped_singular = "
            << stats.skipped_singular << '\n';
  std::cout << label << "_same_spin_formula_max_abs_phi2_diff = "
            << stats.max_abs_phi2_diff << '\n';
  std::cout << label << "_same_spin_formula_max_abs_h2_diff = "
            << stats.max_abs_h2_diff << '\n';
}

void print_rank_update_stats(
    const std::string& label,
    const RankUpdateStats& stats) {
  std::cout << label << "_rank_update_checked = " << stats.checked << '\n';
  std::cout << label << "_rank_update_skipped_singular = "
            << stats.skipped_singular << '\n';
  std::cout << label << "_rank_update_direct_overlap_seconds = "
            << stats.direct_overlap_seconds << '\n';
  std::cout << label << "_rank_update_seconds = "
            << stats.rank_update_seconds << '\n';
  std::cout << label << "_rank_update_overlap_speedup = "
            << (stats.rank_update_seconds > 0.0
                    ? stats.direct_overlap_seconds / stats.rank_update_seconds
                    : 0.0)
            << '\n';
  std::cout << label << "_rank_update_mean_replacement_distance = "
            << stats.mean_replacement_distance << '\n';
  std::cout << label << "_rank_update_mean_update_rank = "
            << stats.mean_update_rank << '\n';
  std::cout << label << "_rank_update_max_update_rank = "
            << stats.max_update_rank << '\n';
  std::cout << label << "_rank_update_max_abs_inverse_diff = "
            << stats.max_abs_inverse_diff << '\n';
  std::cout << label << "_rank_update_max_abs_det_diff = "
            << stats.max_abs_det_diff << '\n';
  std::cout << label << "_rank_update_max_abs_phi2_diff = "
            << stats.max_abs_phi2_diff << '\n';
}

void print_rank_one_overlap_stats(
    const std::string& label,
    const RankOneOverlapStats& stats) {
  std::cout << label << "_rank1_overlap_checked = " << stats.checked << '\n';
  std::cout << label << "_rank1_overlap_resets = " << stats.resets << '\n';
  std::cout << label << "_rank1_overlap_skipped_singular = "
            << stats.skipped_singular << '\n';
  std::cout << label << "_rank1_overlap_direct_seconds = "
            << stats.direct_overlap_seconds << '\n';
  std::cout << label << "_rank1_overlap_update_seconds = "
            << stats.rank_one_seconds << '\n';
  std::cout << label << "_rank1_overlap_speedup = "
            << (stats.rank_one_seconds > 0.0
                    ? stats.direct_overlap_seconds / stats.rank_one_seconds
                    : 0.0)
            << '\n';
  std::cout << label << "_rank1_overlap_max_abs_inverse_diff = "
            << stats.max_abs_inverse_diff << '\n';
  std::cout << label << "_rank1_overlap_max_abs_det_diff = "
            << stats.max_abs_det_diff << '\n';
}

void print_ri_q_update_stats(
    const std::string& label,
    const RiQUpdateStats& stats) {
  std::cout << label << "_ri_q_update_checked = " << stats.checked << '\n';
  std::cout << label << "_ri_q_update_resets = " << stats.resets << '\n';
  std::cout << label << "_ri_q_update_selected_column_rebuilds = "
            << stats.selected_column_rebuilds << '\n';
  std::cout << label << "_ri_q_update_direct_seconds = "
            << stats.direct_q_seconds << '\n';
  std::cout << label << "_ri_q_update_seconds = "
            << stats.rank_update_q_seconds << '\n';
  std::cout << label << "_ri_q_update_speedup = "
            << (stats.rank_update_q_seconds > 0.0
                    ? stats.direct_q_seconds / stats.rank_update_q_seconds
                    : 0.0)
            << '\n';
  std::cout << label << "_ri_q_update_max_abs_q_diff = "
            << stats.max_abs_q_diff << '\n';
}

void print_ri_k_update_stats(
    const std::string& label,
    const RiKUpdateStats& stats) {
  std::cout << label << "_ri_k_update_checked = " << stats.checked << '\n';
  std::cout << label << "_ri_k_update_resets = " << stats.resets << '\n';
  std::cout << label << "_ri_k_update_skipped_singular = "
            << stats.skipped_singular << '\n';
  std::cout << label << "_ri_k_update_direct_seconds = "
            << stats.direct_seconds << '\n';
  std::cout << label << "_ri_k_update_seconds = "
            << stats.update_seconds << '\n';
  std::cout << label << "_ri_k_update_speedup = "
            << (stats.update_seconds > 0.0
                    ? stats.direct_seconds / stats.update_seconds
                    : 0.0)
            << '\n';
  std::cout << label << "_ri_k_update_max_abs_phi2_diff = "
            << stats.max_abs_phi2_diff << '\n';
}

void print_feature_benchmark_stats(
    const std::string& label,
    const OppositeSpinFeatureBenchmarkStats& stats) {
  std::cout << label << "_os_feature_direct_pairs = "
            << stats.direct_pairs << '\n';
  std::cout << label << "_os_feature_low_rank_pairs = "
            << stats.low_rank_pairs << '\n';
  std::cout << label << "_os_feature_low_rank_initializations = "
            << stats.low_rank_initializations << '\n';
  std::cout << label << "_os_feature_low_rank_updates = "
            << stats.low_rank_updates << '\n';
  std::cout << label << "_os_feature_low_rank_resets = "
            << stats.low_rank_resets << '\n';
  std::cout << label << "_os_feature_scalar_pairs = "
            << stats.scalar_pairs << '\n';
  std::cout << label << "_os_feature_scalar_initializations = "
            << stats.scalar_initializations << '\n';
  std::cout << label << "_os_feature_scalar_updates = "
            << stats.scalar_updates << '\n';
  std::cout << label << "_os_feature_scalar_resets = "
            << stats.scalar_resets << '\n';
  std::cout << label << "_os_feature_scalar_selected_column_rebuilds = "
            << stats.scalar_selected_column_rebuilds << '\n';
  std::cout << label << "_os_feature_direct_seconds = "
            << stats.direct_seconds << '\n';
  std::cout << label << "_os_feature_low_rank_seconds = "
            << stats.low_rank_seconds << '\n';
  std::cout << label << "_os_feature_scalar_seconds = "
            << stats.scalar_seconds << '\n';
  std::cout << label << "_os_feature_speedup = "
            << (stats.low_rank_seconds > 0.0
                    ? stats.direct_seconds / stats.low_rank_seconds
                    : 0.0)
            << '\n';
  std::cout << label << "_os_feature_scalar_speedup = "
            << (stats.scalar_seconds > 0.0
                    ? stats.direct_seconds / stats.scalar_seconds
                    : 0.0)
            << '\n';
  std::cout << label << "_os_feature_checksum_abs_diff = "
            << stats.checksum_abs_diff << '\n';
  std::cout << label << "_os_feature_scalar_checksum_abs_diff = "
            << stats.scalar_checksum_abs_diff << '\n';
  std::cout << label << "_os_feature_scalar_linear_checksum_abs_diff = "
            << stats.scalar_linear_checksum_abs_diff << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_start = std::chrono::high_resolution_clock::now();
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto prepared_active_space =
        xmvb::vb::prepare_timed_active_space_context(
            load_result.input,
            orbital_preparer,
            ao_effective_one_electron_builder,
            active_space_one_electron_builder,
            active_space_two_electron_builder)
            .prepared_active_space;
    const auto load_end = std::chrono::high_resolution_clock::now();

    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    const auto& active_two_electron =
        prepared_active_space.active_space_two_electron_result;
    if (active_two_electron.representation !=
            xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity ||
        active_two_electron.ri_active_pair_factors.rows() == 0) {
      throw std::runtime_error(
          "diagnostic requires RI active-space two-electron factors");
    }

    const auto alpha_reuse_table = xmvb::vb::build_spin_determinant_reuse_table(
        load_result.input.structure_data.alpha_det);
    const auto beta_reuse_table = xmvb::vb::build_spin_determinant_reuse_table(
        load_result.input.structure_data.beta_det);
    const auto& alpha_strings = alpha_reuse_table.unique_determinants;
    const auto& beta_strings = beta_reuse_table.unique_determinants;

    const LocalityStats alpha_locality =
        compute_locality_stats(alpha_strings);
    const LocalityStats beta_locality =
        compute_locality_stats(beta_strings);
    const TraversalStats alpha_traversal =
        compute_greedy_traversal_stats(alpha_strings);
    const TraversalStats beta_traversal =
        compute_greedy_traversal_stats(beta_strings);

    const SameSpinFormulaStats alpha_same_spin_formula =
        check_same_spin_formula(
            alpha_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_active_space.active_space_one_electron_result.h1e_act,
            active_two_electron,
            n_active_orbitals,
            options.max_pair_checks);
    const SameSpinFormulaStats beta_same_spin_formula =
        check_same_spin_formula(
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_active_space.active_space_one_electron_result.h1e_act,
            active_two_electron,
            n_active_orbitals,
            options.max_pair_checks);
    const OppositeSpinFormulaStats opposite_spin_formula =
        check_opposite_spin_formula(
            alpha_strings,
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron,
            n_active_orbitals,
            options.max_pair_checks);
    const RankUpdateStats alpha_rank_update =
        check_right_string_rank_update(
            alpha_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_rank_update_checks);
    const RankUpdateStats beta_rank_update =
        check_right_string_rank_update(
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_rank_update_checks);
    const RankOneOverlapStats alpha_rank_one_overlap =
        benchmark_slot_stable_rank_one_overlap_update(
            alpha_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const RankOneOverlapStats beta_rank_one_overlap =
        benchmark_slot_stable_rank_one_overlap_update(
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const RiQUpdateStats alpha_ri_q_update =
        benchmark_slot_stable_ri_q_update(
            alpha_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const RiQUpdateStats beta_ri_q_update =
        benchmark_slot_stable_ri_q_update(
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const RiKUpdateStats alpha_ri_k_update =
        benchmark_ri_k_channel_matrix_update(
            alpha_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const RiKUpdateStats beta_ri_k_update =
        benchmark_ri_k_channel_matrix_update(
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const OppositeSpinFeatureBenchmarkStats alpha_feature_benchmark =
        benchmark_opposite_spin_feature_generation(
            alpha_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);
    const OppositeSpinFeatureBenchmarkStats beta_feature_benchmark =
        benchmark_opposite_spin_feature_generation(
            beta_strings,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            active_two_electron.ri_active_pair_factors,
            n_active_orbitals,
            options.max_feature_benchmark_pairs);

    std::cout << std::setprecision(15);
    std::cout << "input_path = " << options.input_path << '\n';
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     options.standard_two_electron_mode)
              << '\n';
    std::cout << "n_determinants = "
              << load_result.input.structure_data.alpha_det.size() << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "n_auxiliary_functions = "
              << active_two_electron.n_auxiliary_functions << '\n';
    std::cout << "n_packed_active_pairs = "
              << active_two_electron.ri_active_pair_factors.cols() << '\n';
    std::cout << "load_and_prepare_seconds = "
              << std::chrono::duration<double>(load_end - load_start).count()
              << '\n';
    print_locality(
        "alpha",
        alpha_locality,
        alpha_traversal,
        active_two_electron.n_auxiliary_functions);
    print_locality(
        "beta",
        beta_locality,
        beta_traversal,
        active_two_electron.n_auxiliary_functions);
    print_same_spin_formula_stats("alpha", alpha_same_spin_formula);
    print_same_spin_formula_stats("beta", beta_same_spin_formula);
    std::cout << "opposite_spin_formula_checked = "
              << opposite_spin_formula.checked << '\n';
    std::cout << "opposite_spin_formula_skipped_singular = "
              << opposite_spin_formula.skipped_singular << '\n';
    std::cout << "opposite_spin_formula_max_abs_phi_diff = "
              << opposite_spin_formula.max_abs_phi_diff << '\n';
    std::cout << "opposite_spin_formula_max_abs_h_diff = "
              << opposite_spin_formula.max_abs_h_diff << '\n';
    print_rank_update_stats("alpha", alpha_rank_update);
    print_rank_update_stats("beta", beta_rank_update);
    print_rank_one_overlap_stats("alpha", alpha_rank_one_overlap);
    print_rank_one_overlap_stats("beta", beta_rank_one_overlap);
    print_ri_q_update_stats("alpha", alpha_ri_q_update);
    print_ri_q_update_stats("beta", beta_ri_q_update);
    print_ri_k_update_stats("alpha", alpha_ri_k_update);
    print_ri_k_update_stats("beta", beta_ri_k_update);
    print_feature_benchmark_stats("alpha", alpha_feature_benchmark);
    print_feature_benchmark_stats("beta", beta_feature_benchmark);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
