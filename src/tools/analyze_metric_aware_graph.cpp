#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Pair = xmvb::vb::OrbitalPair;

struct Options {
  std::string input_path;
  int left_structure = 0;
  int right_structure = 1;
  double singular_value_threshold = 1.0e-8;
  double edge_max_abs_threshold = 0.0;
  int top_edges = 16;
  int top_separators = 16;
};

void print_usage() {
  std::cerr << "usage: analyze_metric_aware_graph <input.xmi> "
               "[--left-structure I] [--right-structure J] "
               "[--singular-value-threshold tol] "
               "[--edge-max-abs-threshold tol] "
               "[--top-edges N] "
               "[--top-separators N]\n";
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
    if (argument_name == "--edge-max-abs-threshold") {
      options.edge_max_abs_threshold = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--top-edges") {
      options.top_edges = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-separators") {
      options.top_separators = std::stoi(argument_value);
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
  if (options.edge_max_abs_threshold < 0.0) {
    throw std::invalid_argument("--edge-max-abs-threshold must be non-negative");
  }
  if (options.top_edges <= 0 || options.top_separators <= 0) {
    throw std::invalid_argument("--top-edges and --top-separators must be positive");
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

int count_covalent_labels(const xmvb::vb::UnionGraphComponent& component) {
  int covalent_labels = 0;
  for (const auto& pair : component.left_pairs) {
    if (pair.first != pair.second) {
      ++covalent_labels;
    }
  }
  for (const auto& pair : component.right_pairs) {
    if (pair.first != pair.second) {
      ++covalent_labels;
    }
  }
  return covalent_labels;
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
          "analyze_metric_aware_graph currently supports only singlet closed-shell structures");
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
    xmvb::vb::MetricAwareComponentGraphOptions graph_options;
    graph_options.edge_max_abs_threshold = options.edge_max_abs_threshold;
    const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
        union_components,
        cross_blocks,
        graph_options);
    const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
        metric_graph,
        union_components);

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "left_structure = " << options.left_structure << '\n';
    std::cout << "right_structure = " << options.right_structure << '\n';
    std::cout << "left_pairs = " << format_pairs(left_pairs, true) << '\n';
    std::cout << "right_pairs = " << format_pairs(right_pairs, true) << '\n';
    std::cout << "support_orbitals = " << format_indices(support_orbitals, true) << '\n';
    std::cout << "support_size = " << support_orbitals.size() << '\n';
    std::cout << "singular_value_threshold = " << options.singular_value_threshold << '\n';
    std::cout << "edge_max_abs_threshold = " << options.edge_max_abs_threshold << '\n';
    std::cout << "union_component_count = " << union_components.size() << '\n';
    for (const auto& component : union_components) {
      std::cout << "union_component[" << component.index << "]"
                << " type=" << component.type
                << " active_orbitals=" << format_indices(component.active_orbitals, true)
                << " left_pairs=" << format_pairs(component.left_pairs, true)
                << " right_pairs=" << format_pairs(component.right_pairs, true)
                << " covalent_labels=" << count_covalent_labels(component)
                << '\n';
    }

    std::cout << "metric_graph_node_count = " << metric_summary.node_count << '\n';
    std::cout << "metric_graph_edge_count = " << metric_summary.edge_count << '\n';
    std::cout << "metric_graph_connected_component_count = "
              << metric_summary.connected_component_count << '\n';
    std::cout << "metric_graph_articulation_count = "
              << metric_summary.articulation_count << '\n';
    std::cout << "metric_graph_max_degree = " << metric_summary.max_degree << '\n';
    std::cout << "metric_graph_max_component_node_count = "
              << metric_summary.max_component_node_count << '\n';
    std::cout << "metric_graph_max_component_covalent_labels = "
              << metric_summary.max_component_covalent_labels << '\n';
    std::cout << "metric_graph_weighted_min_degree_width_upper_bound = "
              << metric_summary.weighted_min_degree_width_upper_bound << '\n';
    std::cout << "metric_graph_weighted_min_degree_order = "
              << format_indices(metric_summary.weighted_min_degree_order, false) << '\n';

    for (const auto& connected_component : metric_summary.connected_components) {
      std::cout << "metric_connected_component[" << connected_component.index << "]"
                << " graph_nodes=" << format_indices(connected_component.graph_nodes, false)
                << " total_active_orbitals=" << connected_component.total_active_orbitals
                << " total_left_pairs=" << connected_component.total_left_pairs
                << " total_right_pairs=" << connected_component.total_right_pairs
                << " total_covalent_labels=" << connected_component.total_covalent_labels
                << '\n';
    }

    const int n_edges_to_print = std::min(
        options.top_edges,
        static_cast<int>(metric_graph.edges.size()));
    for (int edge_index = 0; edge_index < n_edges_to_print; ++edge_index) {
      const auto& edge = metric_graph.edges[xmvb::to_size(edge_index)];
      std::cout << "metric_edge[" << edge_index << "]"
                << " left_component=" << edge.left_component
                << " right_component=" << edge.right_component
                << " max_abs=" << edge.max_abs
                << " spectral_norm=" << edge.spectral_norm
                << " frobenius_norm=" << edge.frobenius_norm
                << " numerical_rank=" << edge.numerical_rank
                << '\n';
    }

    const int n_separators_to_print = std::min(
        options.top_separators,
        static_cast<int>(metric_summary.articulation_candidates.size()));
    for (int separator_index = 0; separator_index < n_separators_to_print; ++separator_index) {
      const auto& candidate =
          metric_summary.articulation_candidates[xmvb::to_size(separator_index)];
      std::cout << "articulation_candidate[" << separator_index << "]"
                << " component=" << candidate.articulation_component
                << " resulting_component_count=" << candidate.resulting_component_count
                << " largest_piece_node_count=" << candidate.largest_piece_node_count
                << " largest_piece_covalent_labels=" << candidate.largest_piece_covalent_labels
                << '\n';
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
