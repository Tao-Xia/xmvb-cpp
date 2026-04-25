#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Pair = xmvb::vb::OrbitalPair;

struct Options {
  std::string input_path;
  int left_structure = 0;
  int right_structure = 1;
};

void print_usage() {
  std::cerr << "usage: debug_exact_raw_vb_overlap_pair <input.xmi>"
               " [--left-structure I] [--right-structure J]\n";
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
    throw std::invalid_argument("unknown argument: " + argument_name);
  }
  return options;
}

std::string format_pairs(const std::vector<Pair>& pairs) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pair_index > 0) {
      stream << " ";
    }
    stream << "(" << pairs[pair_index].first + 1 << "," << pairs[pair_index].second + 1
           << ")";
  }
  stream << "]";
  return stream.str();
}

std::string format_indices(const std::vector<int>& indices, bool one_based) {
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

std::string format_term(const xmvb::vb::LegacyStructureDeterminantTerm& term) {
  std::ostringstream stream;
  stream << "coef=" << term.coefficient
         << " alpha=" << format_indices(term.alpha_occ, true)
         << " beta=" << format_indices(term.beta_occ, true);
  return stream.str();
}

std::vector<Pair> build_original_pairs_for_metric_component(
    const std::vector<int>& graph_nodes,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    const std::vector<int>& support_orbitals,
    bool use_left_pairs) {
  std::vector<Pair> pairs;
  for (const int graph_node : graph_nodes) {
    const auto& component = union_components[graph_node];
    const auto& source_pairs = use_left_pairs ? component.left_pairs : component.right_pairs;
    for (const auto& pair : source_pairs) {
      pairs.emplace_back(
          support_orbitals[pair.first],
          support_orbitals[pair.second]);
    }
  }
  return pairs;
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
          "debug_exact_raw_vb_overlap_pair currently supports only singlet closed-shell inputs");
    }

    const auto left_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.left_structure);
    const auto right_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.right_structure);
    const auto left_terms =
        xmvb::vb::enumerate_legacy_determinant_terms(left_pairs);
    const auto right_terms =
        xmvb::vb::enumerate_legacy_determinant_terms(right_pairs);

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
        1.0e-8);
    const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
        union_components,
        cross_blocks,
        {});
    const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
        metric_graph,
        union_components);

    xmvb::vb::Matrix full_active_overlap(n_active_orbitals, n_active_orbitals);
    for (int column = 0; column < n_active_orbitals; ++column) {
      for (int row = 0; row < n_active_orbitals; ++row) {
        full_active_overlap(row, column) =
            active_overlap_storage[column *
                                       n_active_orbitals +
                                   row];
      }
    }

    const auto block_diagonal_support_overlap =
        xmvb::vb::build_block_diagonalized_support_overlap(
            support_overlap,
            union_components);
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;

    // `exact_full_overlap` is the determinant-expanded raw-VB overlap on the
    // full active metric. `exact_support_overlap` is the same computation on
    // the reduced support metric, so any difference would indicate a bug in the
    // support remapping rather than in the factorization logic.
    const double exact_full_overlap = xmvb::vb::legacy_structure_overlap(
        left_terms,
        right_terms,
        full_active_overlap,
        overlap_resolver);
    const auto left_terms_local =
        xmvb::vb::enumerate_legacy_determinant_terms(left_pairs_local);
    const auto right_terms_local =
        xmvb::vb::enumerate_legacy_determinant_terms(right_pairs_local);
    const double exact_support_overlap = xmvb::vb::legacy_structure_overlap(
        left_terms_local,
        right_terms_local,
        support_overlap,
        overlap_resolver);
    const double exact_block_diagonal_overlap = xmvb::vb::legacy_structure_overlap(
        left_terms_local,
        right_terms_local,
        block_diagonal_support_overlap,
        overlap_resolver);

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "left_structure = " << options.left_structure << '\n';
    std::cout << "right_structure = " << options.right_structure << '\n';
    std::cout << "left_pairs = " << format_pairs(left_pairs) << '\n';
    std::cout << "right_pairs = " << format_pairs(right_pairs) << '\n';
    std::cout << "left_determinant_term_count = " << left_terms.size() << '\n';
    std::cout << "right_determinant_term_count = " << right_terms.size() << '\n';
    std::cout << "support_orbitals = " << format_indices(support_orbitals, true) << '\n';
    std::cout << "metric_connected_component_count = "
              << metric_summary.connected_component_count << '\n';
    std::cout << "width_upper_bound = "
              << metric_summary.weighted_min_degree_width_upper_bound << '\n';
    std::cout << "exact_full_overlap = " << exact_full_overlap << '\n';
    std::cout << "exact_support_overlap = " << exact_support_overlap << '\n';
    std::cout << "exact_block_diagonal_support_overlap = "
              << exact_block_diagonal_overlap << '\n';
    std::cout << "support_overlap_matrix\n";
    for (int row = 0; row < support_overlap.rows(); ++row) {
      std::cout << "  row[" << row << "]";
      for (int column = 0; column < support_overlap.cols(); ++column) {
        std::cout << " " << support_overlap(row, column);
      }
      std::cout << '\n';
    }
    std::cout << "left_terms\n";
    for (std::size_t term_index = 0; term_index < left_terms.size(); ++term_index) {
      std::cout << "  left_term[" << term_index << "] "
                << format_term(left_terms[term_index]) << '\n';
    }
    std::cout << "right_terms\n";
    for (std::size_t term_index = 0; term_index < right_terms.size(); ++term_index) {
      std::cout << "  right_term[" << term_index << "] "
                << format_term(right_terms[term_index]) << '\n';
    }
    std::cout << "union_components\n";
    for (std::size_t component_index = 0;
         component_index < union_components.size();
         ++component_index) {
      const auto& union_component = union_components[component_index];
      std::vector<int> active_orbitals_one_based;
      active_orbitals_one_based.reserve(union_component.local_vertices.size());
      for (const int local_vertex : union_component.local_vertices) {
        active_orbitals_one_based.push_back(
            support_orbitals[local_vertex]);
      }
      std::cout << "  union_component[" << component_index << "]"
                << " type=" << union_component.type
                << " local_vertices=" << format_indices(union_component.local_vertices, true)
                << " active_orbitals="
                << format_indices(active_orbitals_one_based, true)
                << " left_pairs=" << format_pairs(union_component.left_pairs)
                << " right_pairs=" << format_pairs(union_component.right_pairs)
                << '\n';
    }
    std::cout << "cross_blocks\n";
    for (const auto& cross_block : cross_blocks) {
      std::cout << "  cross_block(" << cross_block.left_component
                << "," << cross_block.right_component << ")"
                << " max_abs=" << cross_block.max_abs
                << " fro=" << cross_block.frobenius_norm
                << " rank=" << cross_block.numerical_rank
                << '\n';
    }
    std::cout << "component_breakdown\n";

    // Each metric connected component is printed in the orbital basis of the
    // original active space. The local exact overlap is evaluated on the full
    // active metric, so a nonzero value in a supposedly disconnected block is
    // easy to spot directly from the output.
    for (std::size_t component_index = 0;
         component_index < metric_summary.connected_components.size();
         ++component_index) {
      const auto& connected_component =
          metric_summary.connected_components[component_index];
      const auto left_component_pairs = build_original_pairs_for_metric_component(
          connected_component.graph_nodes,
          union_components,
          support_orbitals,
          true);
      const auto right_component_pairs = build_original_pairs_for_metric_component(
          connected_component.graph_nodes,
          union_components,
          support_orbitals,
          false);
      const auto left_component_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(left_component_pairs);
      const auto right_component_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(right_component_pairs);
      const double local_overlap = xmvb::vb::legacy_structure_overlap(
          left_component_terms,
          right_component_terms,
          full_active_overlap,
          overlap_resolver);

      std::cout << "component[" << component_index << "]"
                << " graph_nodes=" << format_indices(connected_component.graph_nodes, false)
                << " left_pairs=" << format_pairs(left_component_pairs)
                << " right_pairs=" << format_pairs(right_component_pairs)
                << " left_term_count=" << left_component_terms.size()
                << " right_term_count=" << right_component_terms.size()
                << " local_overlap=" << local_overlap
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
