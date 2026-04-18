#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Pair = std::pair<int, int>;
using OrbitalPair = xmvb::vb::OrbitalPair;

struct Options {
  std::string input_path;
  int filter_left_structure = -1;
  int filter_right_structure = -1;
  int max_pairs = 0;
  int min_node_count = 3;
  int top_pairs = 24;
  int report_every = 0;
  double singular_value_threshold = 1.0e-8;
  double edge_max_abs_threshold = 0.0;
  std::string active_overlap_source = "input";
};

struct PerStructureCache {
  std::vector<OrbitalPair> active_pairs;
  int determinant_term_count = 0;
};

struct StarPairRecord {
  int left_structure = 0;
  int right_structure = 0;
  int support_size = 0;
  int node_count = 0;
  int root_node = -1;
  int width_upper_bound = 0;
  int articulation_count = 0;
  int left_determinant_term_count = 0;
  int right_determinant_term_count = 0;
  std::uint64_t determinant_pair_count = 0;
  std::vector<int> component_sizes;
  std::vector<int> component_covalent_labels;
};

void print_usage() {
  std::cerr << "usage: scan_star_separator_pairs <input.xmi>"
               " [--left-structure I --right-structure J]"
               " [--max-pairs N]"
               " [--min-node-count N]"
               " [--top-pairs N]"
               " [--report-every N]"
               " [--active-overlap-source input|prepared_active_space]"
               " [--singular-value-threshold F]"
               " [--edge-max-abs-threshold F]\n";
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
      options.filter_left_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right-structure") {
      options.filter_right_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--max-pairs") {
      options.max_pairs = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--min-node-count") {
      options.min_node_count = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-pairs") {
      options.top_pairs = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--report-every") {
      options.report_every = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--active-overlap-source") {
      options.active_overlap_source = argument_value;
      continue;
    }
    if (argument_name == "--singular-value-threshold") {
      options.singular_value_threshold = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--edge-max-abs-threshold") {
      options.edge_max_abs_threshold = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if ((options.filter_left_structure < 0) != (options.filter_right_structure < 0)) {
    throw std::invalid_argument(
        "--left-structure and --right-structure must be provided together");
  }
  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if (options.min_node_count <= 0) {
    throw std::invalid_argument("--min-node-count must be positive");
  }
  if (options.top_pairs <= 0) {
    throw std::invalid_argument("--top-pairs must be positive");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
  }
  if (options.active_overlap_source != "input" &&
      options.active_overlap_source != "prepared_active_space") {
    throw std::invalid_argument(
        "--active-overlap-source must be input or prepared_active_space");
  }
  if (options.singular_value_threshold <= 0.0) {
    throw std::invalid_argument("--singular-value-threshold must be positive");
  }
  if (options.edge_max_abs_threshold < 0.0) {
    throw std::invalid_argument("--edge-max-abs-threshold must be non-negative");
  }
  return options;
}

std::vector<Pair> build_pair_list(
    int structure_count,
    int filter_left_structure,
    int filter_right_structure,
    int max_pairs) {
  if (filter_left_structure >= 0 && filter_right_structure >= 0) {
    if (filter_left_structure >= structure_count ||
        filter_right_structure >= structure_count ||
        filter_left_structure <= filter_right_structure) {
      throw std::invalid_argument("requested structure filter is out of range");
    }
    return {{filter_left_structure, filter_right_structure}};
  }

  std::vector<Pair> pairs;
  for (int left_structure = 0; left_structure < structure_count; ++left_structure) {
    for (int right_structure = 0; right_structure < left_structure; ++right_structure) {
      pairs.emplace_back(left_structure, right_structure);
    }
  }
  if (max_pairs > 0 && static_cast<int>(pairs.size()) > max_pairs) {
    pairs.resize(xmvb::to_size(max_pairs));
  }
  return pairs;
}

bool is_connected_star_graph(
    const xmvb::vb::MetricAwareComponentGraph& graph,
    int* root_node) {
  if (root_node == nullptr) {
    throw std::invalid_argument("root_node must not be null");
  }
  *root_node = -1;
  if (graph.node_count <= 0) {
    return false;
  }
  if (graph.node_count == 1) {
    *root_node = 0;
    return true;
  }

  for (int candidate_root = 0; candidate_root < graph.node_count; ++candidate_root) {
    bool is_star = true;
    for (int node = 0; node < graph.node_count; ++node) {
      const int degree = static_cast<int>(graph.adjacency[xmvb::to_size(node)].size());
      if (node == candidate_root) {
        if (degree != graph.node_count - 1) {
          is_star = false;
          break;
        }
      } else if (degree != 1) {
        is_star = false;
        break;
      }
    }
    if (is_star) {
      *root_node = candidate_root;
      return true;
    }
  }

  if (graph.node_count == 2) {
    *root_node = 0;
    return static_cast<int>(graph.adjacency[0].size()) == 1 &&
        static_cast<int>(graph.adjacency[1].size()) == 1;
  }
  return false;
}

int count_covalent_labels(const xmvb::vb::UnionGraphComponent& component) {
  int count = 0;
  for (const auto& pair : component.left_pairs) {
    if (pair.first != pair.second) {
      ++count;
    }
  }
  for (const auto& pair : component.right_pairs) {
    if (pair.first != pair.second) {
      ++count;
    }
  }
  return count;
}

std::string format_int_list(const std::vector<int>& values) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ", ";
    }
    stream << values[index];
  }
  stream << "]";
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto started_at = std::chrono::steady_clock::now();
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    std::vector<double> active_overlap_storage;
    if (options.active_overlap_source == "input") {
      active_overlap_storage =
          load_result.input.orbital_preparation_input.active_orbital_overlap_matrix;
    } else {
      xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
      xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
      xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
      xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
      const auto timed_active_space = xmvb::vb::prepare_timed_active_space_context(
          load_result.input,
          orbital_preparer,
          ao_effective_one_electron_builder,
          active_space_one_electron_builder,
          active_space_two_electron_builder);
      active_overlap_storage =
          timed_active_space.prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    }

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "scan_star_separator_pairs currently supports only singlet closed-shell structures");
    }

    std::vector<PerStructureCache> structure_cache(
        xmvb::to_size(raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[xmvb::to_size(structure_index)];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      cache.determinant_term_count = static_cast<int>(
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs).size());
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.filter_left_structure,
        options.filter_right_structure,
        options.max_pairs);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    xmvb::vb::MetricAwareComponentGraphOptions graph_options;
    graph_options.edge_max_abs_threshold = options.edge_max_abs_threshold;

    std::vector<StarPairRecord> star_pairs;
    std::map<int, int> star_node_count_histogram;
    std::map<int, int> star_width_histogram;

    for (std::size_t pair_index = 0; pair_index < pair_list.size(); ++pair_index) {
      const auto [left_structure, right_structure] = pair_list[pair_index];
      const auto& left_cache = structure_cache[xmvb::to_size(left_structure)];
      const auto& right_cache = structure_cache[xmvb::to_size(right_structure)];

      const auto support_orbitals =
          xmvb::vb::build_support_orbitals(left_cache.active_pairs, right_cache.active_pairs);
      const auto support_index =
          xmvb::vb::build_support_index(support_orbitals);
      const auto left_pairs_local =
          xmvb::vb::remap_pairs_to_support(left_cache.active_pairs, support_index);
      const auto right_pairs_local =
          xmvb::vb::remap_pairs_to_support(right_cache.active_pairs, support_index);
      const auto support_overlap = xmvb::vb::build_support_overlap_matrix(
          support_orbitals,
          active_overlap_storage,
          n_active_orbitals);
      const auto union_components = xmvb::vb::build_union_graph_components(
          left_pairs_local,
          right_pairs_local,
          support_orbitals);
      const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
          support_overlap,
          union_components,
          options.singular_value_threshold);
      const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
          union_components,
          cross_blocks,
          graph_options);
      const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
          metric_graph,
          union_components);
      if (metric_summary.connected_component_count != 1) {
        continue;
      }

      int root_node = -1;
      if (!is_connected_star_graph(metric_graph, &root_node)) {
        continue;
      }
      if (metric_graph.node_count < options.min_node_count) {
        continue;
      }

      StarPairRecord record;
      record.left_structure = left_structure;
      record.right_structure = right_structure;
      record.support_size = static_cast<int>(support_orbitals.size());
      record.node_count = metric_graph.node_count;
      record.root_node = root_node;
      record.width_upper_bound = metric_summary.weighted_min_degree_width_upper_bound;
      record.articulation_count = metric_summary.articulation_count;
      record.left_determinant_term_count = left_cache.determinant_term_count;
      record.right_determinant_term_count = right_cache.determinant_term_count;
      record.determinant_pair_count =
          static_cast<std::uint64_t>(left_cache.determinant_term_count) *
          static_cast<std::uint64_t>(right_cache.determinant_term_count);
      record.component_sizes.reserve(union_components.size());
      record.component_covalent_labels.reserve(union_components.size());
      for (const auto& component : union_components) {
        record.component_sizes.push_back(
            static_cast<int>(component.active_orbitals.size()));
        record.component_covalent_labels.push_back(count_covalent_labels(component));
      }
      star_pairs.push_back(std::move(record));
      ++star_node_count_histogram[metric_graph.node_count];
      ++star_width_histogram[metric_summary.weighted_min_degree_width_upper_bound];

      if (options.report_every > 0 &&
          (pair_index + 1) % xmvb::to_size(options.report_every) == 0) {
        const auto elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at).count();
        std::cerr << "progress = " << (pair_index + 1) << "/" << pair_list.size()
                  << " multi_leaf_stars = " << star_pairs.size()
                  << " elapsed_s = " << std::fixed << std::setprecision(2)
                  << elapsed_seconds << '\n';
      }
    }

    std::sort(
        star_pairs.begin(),
        star_pairs.end(),
        [](const StarPairRecord& left, const StarPairRecord& right) {
          if (left.node_count != right.node_count) {
            return left.node_count > right.node_count;
          }
          if (left.width_upper_bound != right.width_upper_bound) {
            return left.width_upper_bound > right.width_upper_bound;
          }
          if (left.determinant_pair_count != right.determinant_pair_count) {
            return left.determinant_pair_count > right.determinant_pair_count;
          }
          if (left.left_structure != right.left_structure) {
            return left.left_structure < right.left_structure;
          }
          return left.right_structure < right.right_structure;
        });

    const auto elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_at).count();

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "structure_count = " << raw_structure_data.n_structures << '\n';
    std::cout << "selected_pairs = " << pair_list.size() << '\n';
    std::cout << "min_node_count = " << options.min_node_count << '\n';
    std::cout << "active_overlap_source = " << options.active_overlap_source << '\n';
    std::cout << "singular_value_threshold = " << options.singular_value_threshold << '\n';
    std::cout << "edge_max_abs_threshold = " << options.edge_max_abs_threshold << '\n';
    std::cout << "multi_leaf_star_pair_count = " << star_pairs.size() << '\n';
    std::cout << "star_node_count_histogram = {";
    bool first = true;
    for (const auto& [node_count, count] : star_node_count_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << node_count << ": " << count;
    }
    std::cout << "}\n";
    std::cout << "star_width_histogram = {";
    first = true;
    for (const auto& [width, count] : star_width_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << width << ": " << count;
    }
    std::cout << "}\n";
    if (!star_pairs.empty()) {
      const auto& best = star_pairs.front();
      std::cout << "best_pair"
                << " left_structure=" << best.left_structure
                << " right_structure=" << best.right_structure
                << " node_count=" << best.node_count
                << " root_node=" << best.root_node
                << " width_upper_bound=" << best.width_upper_bound
                << " support_size=" << best.support_size
                << " determinant_pair_count=" << best.determinant_pair_count
                << '\n';
    }
    std::cout << "elapsed_wall_time_seconds = " << elapsed_seconds << '\n';
    std::cout << "top_star_pairs\n";
    for (int pair_index = 0;
         pair_index < options.top_pairs &&
         pair_index < static_cast<int>(star_pairs.size());
         ++pair_index) {
      const auto& record = star_pairs[xmvb::to_size(pair_index)];
      std::cout << "pair[" << pair_index << "]"
                << " left_structure=" << record.left_structure
                << " right_structure=" << record.right_structure
                << " node_count=" << record.node_count
                << " root_node=" << record.root_node
                << " width_upper_bound=" << record.width_upper_bound
                << " articulation_count=" << record.articulation_count
                << " support_size=" << record.support_size
                << " left_determinant_term_count=" << record.left_determinant_term_count
                << " right_determinant_term_count=" << record.right_determinant_term_count
                << " determinant_pair_count=" << record.determinant_pair_count
                << " component_sizes=" << format_int_list(record.component_sizes)
                << " component_covalent_labels="
                << format_int_list(record.component_covalent_labels)
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
