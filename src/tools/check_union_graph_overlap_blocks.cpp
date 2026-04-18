#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/union_graph_rank_predictor.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Pair = xmvb::vb::OrbitalPair;

struct Options {
  std::string input_path;
  int left_structure = 0;
  int right_structure = 1;
  double singular_value_threshold = 1.0e-8;
  int max_rank_cap = -1;
};

struct RankSweepEntry {
  int rank_cap = 0;
  double offblock_fro_error = 0.0;
  double offblock_rel_error = 0.0;
  double approximate_overlap = 0.0;
  double absolute_error = 0.0;
  double relative_error = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_union_graph_overlap_blocks <input.xmi> "
               "[--left-structure I] [--right-structure J] "
               "[--singular-value-threshold tol] "
               "[--max-rank-cap R]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--left-structure") {
      options.left_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right-structure") {
      options.right_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--singular-value-threshold") {
      options.singular_value_threshold = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--max-rank-cap") {
      options.max_rank_cap = std::stoi(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.left_structure < 0 || options.right_structure < 0) {
    throw std::invalid_argument("structure indices must be non-negative");
  }
  if (options.singular_value_threshold <= 0.0) {
    throw std::invalid_argument("--singular-value-threshold must be positive");
  }
  if (options.max_rank_cap < -1) {
    throw std::invalid_argument("--max-rank-cap must be >= -1");
  }
  return options;
}

std::string format_pairs(
    const std::vector<Pair>& pairs,
    bool one_based) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pair_index > 0) {
      stream << " ";
    }
    const int shift = one_based ? 1 : 0;
    stream << "(" << pairs[pair_index].first + shift << "," << pairs[pair_index].second + shift
           << ")";
  }
  stream << "]";
  return stream.str();
}

std::string format_indices(
    const std::vector<int>& indices,
    bool one_based) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index > 0) {
      stream << " ";
    }
    stream << indices[index] + (one_based ? 1 : 0);
  }
  stream << "]";
  return stream.str();
}

std::string format_double_list(
    const std::vector<double>& values,
    int max_count) {
  std::ostringstream stream;
  stream << "[";
  const int n_to_print = std::min(max_count, static_cast<int>(values.size()));
  for (int value_index = 0; value_index < n_to_print; ++value_index) {
    if (value_index > 0) {
      stream << " ";
    }
    stream << values[xmvb::to_size(value_index)];
  }
  if (static_cast<int>(values.size()) > n_to_print) {
    if (n_to_print > 0) {
      stream << " ";
    }
    stream << "...";
  }
  stream << "]";
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    const auto& active_overlap_storage =
        load_result.input.orbital_preparation_input.active_orbital_overlap_matrix;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "check_union_graph_overlap_blocks currently supports only singlet closed-shell structures");
    }
    if (options.left_structure >= raw_structure_data.n_structures ||
        options.right_structure >= raw_structure_data.n_structures) {
      throw std::out_of_range("structure index out of range for selected input");
    }

    const auto left_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.left_structure);
    const auto right_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.right_structure);
    const auto support_orbitals =
        xmvb::vb::build_support_orbitals(left_pairs, right_pairs);
    const auto support_index = xmvb::vb::build_support_index(support_orbitals);
    const auto left_pairs_local =
        xmvb::vb::remap_pairs_to_support(left_pairs, support_index);
    const auto right_pairs_local =
        xmvb::vb::remap_pairs_to_support(right_pairs, support_index);

    const auto left_terms = xmvb::vb::enumerate_legacy_determinant_terms(left_pairs_local);
    const auto right_terms = xmvb::vb::enumerate_legacy_determinant_terms(right_pairs_local);
    const auto support_overlap = xmvb::vb::build_support_overlap_matrix(
        support_orbitals,
        active_overlap_storage,
        n_active_orbitals);
    const auto components = xmvb::vb::build_union_graph_components(
        left_pairs_local,
        right_pairs_local,
        support_orbitals);
    const auto block_diagonal_overlap = xmvb::vb::build_block_diagonalized_support_overlap(
        support_overlap,
        components);
    const auto offblock_overlap = xmvb::vb::build_offblock_support_overlap(
        support_overlap,
        components);
    const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
        support_overlap,
        components,
        options.singular_value_threshold);
    const auto screening_summary = xmvb::vb::summarize_union_graph_screening(
        support_overlap,
        offblock_overlap,
        components,
        cross_blocks);
    const auto rank_prediction = xmvb::vb::predict_union_graph_rank_cap(
        screening_summary);

    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    const double exact_overlap = xmvb::vb::legacy_structure_overlap(
        left_terms,
        right_terms,
        support_overlap,
        overlap_resolver);
    const double block_diagonal_overlap_value = xmvb::vb::legacy_structure_overlap(
        left_terms,
        right_terms,
        block_diagonal_overlap,
        overlap_resolver);
    const double block_diagonal_absolute_error =
        std::abs(exact_overlap - block_diagonal_overlap_value);
    const double block_diagonal_relative_error =
        block_diagonal_absolute_error / std::max(1.0, std::abs(exact_overlap));

    const int max_available_rank = screening_summary.max_numerical_rank;
    const int max_rank_cap =
        options.max_rank_cap >= 0 ? std::min(options.max_rank_cap, max_available_rank)
                                  : max_available_rank;
    std::vector<RankSweepEntry> rank_sweep;
    rank_sweep.reserve(xmvb::to_size(max_rank_cap + 1));
    for (int rank_cap = 0; rank_cap <= max_rank_cap; ++rank_cap) {
      const auto truncated_offblock = xmvb::vb::build_blockwise_truncated_offblock(
          support_overlap,
          components,
          rank_cap);
      const auto approximate_support_overlap = block_diagonal_overlap + truncated_offblock;
      const double approximate_overlap = xmvb::vb::legacy_structure_overlap(
          left_terms,
          right_terms,
          approximate_support_overlap,
          overlap_resolver);
      const double offblock_fro_error = (offblock_overlap - truncated_offblock).norm();
      const double absolute_error = std::abs(exact_overlap - approximate_overlap);

      RankSweepEntry entry;
      entry.rank_cap = rank_cap;
      entry.offblock_fro_error = offblock_fro_error;
      entry.offblock_rel_error =
          offblock_fro_error / std::max(1.0, offblock_overlap.norm());
      entry.approximate_overlap = approximate_overlap;
      entry.absolute_error = absolute_error;
      entry.relative_error = absolute_error / std::max(1.0, std::abs(exact_overlap));
      rank_sweep.push_back(entry);
    }

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "left_structure = " << options.left_structure << '\n';
    std::cout << "right_structure = " << options.right_structure << '\n';
    std::cout << "n_active_pairs = " << left_pairs.size() << '\n';
    std::cout << "left_pairs = " << format_pairs(left_pairs, true) << '\n';
    std::cout << "right_pairs = " << format_pairs(right_pairs, true) << '\n';
    std::cout << "support_orbitals = " << format_indices(support_orbitals, true) << '\n';
    std::cout << "support_size = " << screening_summary.support_size << '\n';
    std::cout << "component_count = " << screening_summary.component_count << '\n';
    std::cout << "component_signature = " << screening_summary.component_signature << '\n';
    std::cout << "n_single_vertex_components = "
              << screening_summary.n_single_vertex_components << '\n';
    std::cout << "n_doubled_edge_components = "
              << screening_summary.n_doubled_edge_components << '\n';
    std::cout << "n_alternating_cycle_components = "
              << screening_summary.n_alternating_cycle_components << '\n';
    std::cout << "n_general_components = "
              << screening_summary.n_general_components << '\n';
    std::cout << "max_component_size = " << screening_summary.max_component_size << '\n';
    std::cout << "cross_block_count = " << screening_summary.cross_block_count << '\n';
    std::cout << "support_overlap_frobenius = "
              << screening_summary.support_overlap_frobenius << '\n';
    std::cout << "offblock_overlap_frobenius = "
              << screening_summary.offblock_overlap_frobenius << '\n';
    std::cout << "offblock_overlap_fraction = "
              << screening_summary.offblock_overlap_fraction << '\n';
    std::cout << "max_cross_block_frobenius = "
              << screening_summary.max_cross_block_frobenius << '\n';
    std::cout << "max_cross_block_spectral = "
              << screening_summary.max_cross_block_spectral << '\n';
    std::cout << "max_cross_block_max_abs = "
              << screening_summary.max_cross_block_max_abs << '\n';
    std::cout << "max_cross_block_second_singular = "
              << screening_summary.max_cross_block_second_singular << '\n';
    std::cout << "max_cross_block_third_singular = "
              << screening_summary.max_cross_block_third_singular << '\n';
    std::cout << "sum_cross_block_frobenius = "
              << screening_summary.sum_cross_block_frobenius << '\n';
    std::cout << "predicted_rank_cap = "
              << rank_prediction.predicted_rank_cap << '\n';
    std::cout << "prediction_reason = "
              << xmvb::vb::union_graph_rank_prediction_reason_name(
                     rank_prediction.reason)
              << '\n';
    std::cout << "exact_overlap = " << exact_overlap << '\n';
    std::cout << "block_diagonal_overlap = " << block_diagonal_overlap_value << '\n';
    std::cout << "block_diagonal_abs_error = " << block_diagonal_absolute_error << '\n';
    std::cout << "block_diagonal_rel_error = " << block_diagonal_relative_error << '\n';
    std::cout << "singular_value_threshold = " << options.singular_value_threshold << '\n';
    std::cout << "max_available_rank = " << max_available_rank << '\n';
    std::cout << "reported_max_rank_cap = " << max_rank_cap << '\n';

    for (const auto& component : components) {
      std::vector<Pair> left_pairs_original;
      std::vector<Pair> right_pairs_original;
      left_pairs_original.reserve(component.left_pairs.size());
      right_pairs_original.reserve(component.right_pairs.size());
      for (const auto& pair : component.left_pairs) {
        left_pairs_original.emplace_back(
            support_orbitals[xmvb::to_size(pair.first)],
            support_orbitals[xmvb::to_size(pair.second)]);
      }
      for (const auto& pair : component.right_pairs) {
        right_pairs_original.emplace_back(
            support_orbitals[xmvb::to_size(pair.first)],
            support_orbitals[xmvb::to_size(pair.second)]);
      }
      std::cout << "component[" << component.index << "]"
                << " type=" << component.type
                << " orbitals=" << format_indices(component.active_orbitals, true)
                << " left_pairs=" << format_pairs(left_pairs_original, true)
                << " right_pairs=" << format_pairs(right_pairs_original, true)
                << '\n';
    }

    for (const auto& cross_block : cross_blocks) {
      std::cout << "cross_block"
                << " left_component=" << cross_block.left_component
                << " right_component=" << cross_block.right_component
                << " shape=" << cross_block.n_rows << "x" << cross_block.n_columns
                << " frobenius=" << cross_block.frobenius_norm
                << " spectral=" << cross_block.spectral_norm
                << " max_abs=" << cross_block.max_abs
                << " numerical_rank=" << cross_block.numerical_rank
                << " singular_values=" << format_double_list(cross_block.singular_values, 6)
                << '\n';
    }

    for (const auto& entry : rank_sweep) {
      std::cout << "rank_sweep"
                << " rank_cap=" << entry.rank_cap
                << " offblock_fro_error=" << entry.offblock_fro_error
                << " offblock_rel_error=" << entry.offblock_rel_error
                << " approximate_overlap=" << entry.approximate_overlap
                << " abs_error=" << entry.absolute_error
                << " rel_error=" << entry.relative_error
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
