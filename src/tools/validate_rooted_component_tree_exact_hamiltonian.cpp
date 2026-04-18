#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/exact_separator/component_tree.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using OrbitalPair = xmvb::vb::OrbitalPair;
using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedHamiltonianComponentTreeStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ComponentTree;
using xmvb::vb::exact_separator::OrientationTerm;

struct Options {
  std::string input_path;
  int max_pairs = 0;
  int top_examples = 8;
  double tolerance = 1.0e-10;
  double edge_threshold = 0.0;
};

struct Example {
  int left_structure = 0;
  int right_structure = 0;
  int root_component = -1;
  int node_count = 0;
  int support_size = 0;
  double overlap_abs_error = 0.0;
  double one_electron_abs_error = 0.0;
  double same_spin_alpha_abs_error = 0.0;
  double same_spin_beta_abs_error = 0.0;
  double opposite_spin_abs_error = 0.0;
  double total_two_electron_abs_error = 0.0;
  double total_electronic_abs_error = 0.0;
};

struct RootedTreeLayout {
  int root_old_index = -1;
  std::vector<int> preorder_old_nodes;
  std::vector<int> old_to_preorder;
  std::vector<std::vector<int>> preorder_children;
  std::vector<int> ordered_support_orbitals;
};

void print_usage() {
  std::cerr << "usage: validate_rooted_component_tree_exact_hamiltonian <input.xmi>"
               " [--max-pairs N]"
               " [--top-examples N]"
               " [--tolerance F]"
               " [--edge-threshold F]\n";
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
    if (argument_name == "--max-pairs") {
      options.max_pairs = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-examples") {
      options.top_examples = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--edge-threshold") {
      options.edge_threshold = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if (options.top_examples <= 0) {
    throw std::invalid_argument("--top-examples must be positive");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  if (options.edge_threshold < 0.0) {
    throw std::invalid_argument("--edge-threshold must be non-negative");
  }
  return options;
}

std::vector<std::pair<int, int>> build_pair_list(int structure_count, int max_pairs) {
  std::vector<std::pair<int, int>> pairs;
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

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

Matrix build_support_submatrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& full_storage,
    int n_active_orbitals) {
  // Both the active-space overlap matrix and the active-space one-electron
  // matrix are stored in column-major form with the same orbital convention:
  // rows index bra/right orbitals and columns index ket/left orbitals. This
  // helper extracts the support-local block in that exact orientation.
  const int support_size = static_cast<int>(support_orbitals.size());
  Matrix support_submatrix(support_size, support_size);
  for (int column = 0; column < support_size; ++column) {
    const int full_column = support_orbitals[xmvb::to_size(column)];
    for (int row = 0; row < support_size; ++row) {
      const int full_row = support_orbitals[xmvb::to_size(row)];
      support_submatrix(row, column) =
          full_storage[xmvb::to_size(full_column) *
                           xmvb::to_size(n_active_orbitals) +
                       xmvb::to_size(full_row)];
    }
  }
  return support_submatrix;
}

std::vector<double> build_support_local_packed_two_electron_integrals(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals) {
  // The exact rooted-tree Hamiltonian routine expects the same packed active
  // two-electron convention as the determinant-space reference. This remaps
  // the global active-space ERIs onto the support-local orbital numbering
  // induced by the component-tree preorder.
  const int support_size = static_cast<int>(support_orbitals.size());
  const int packed_size =
      TwoElectronIndexer::two_electron_storage_index(
          support_size - 1,
          support_size - 1,
          support_size - 1,
          support_size - 1) +
      1;
  std::vector<double> support_local_packed(
      xmvb::to_size(packed_size),
      0.0);
  for (int p = 0; p < support_size; ++p) {
    const int full_p = support_orbitals[xmvb::to_size(p)];
    for (int q = 0; q < support_size; ++q) {
      const int full_q = support_orbitals[xmvb::to_size(q)];
      for (int r = 0; r < support_size; ++r) {
        const int full_r = support_orbitals[xmvb::to_size(r)];
        for (int s = 0; s < support_size; ++s) {
          const int full_s = support_orbitals[xmvb::to_size(s)];
          support_local_packed[xmvb::to_size(
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] =
              packed_active_two_electron_integrals[xmvb::to_size(
                  TwoElectronIndexer::two_electron_storage_index(
                      full_p,
                      full_q,
                      full_r,
                      full_s))];
        }
      }
    }
  }
  return support_local_packed;
}

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
  const auto legacy_terms = xmvb::vb::enumerate_legacy_determinant_terms(pairs);
  std::vector<OrientationTerm> terms;
  terms.reserve(legacy_terms.size());
  for (const auto& legacy_term : legacy_terms) {
    terms.push_back(OrientationTerm{
        .alpha_occ = legacy_term.alpha_occ,
        .beta_occ = legacy_term.beta_occ,
        .coefficient = legacy_term.coefficient,
    });
  }
  return terms;
}

ComponentData build_local_component_data(
    const xmvb::vb::UnionGraphComponent& union_component,
    int graph_node) {
  ComponentData component;
  component.graph_node = graph_node;
  component.left_pairs = union_component.left_pairs;
  component.right_pairs = union_component.right_pairs;
  component.left_orientation_terms =
      enumerate_orientation_terms(component.left_pairs);
  component.right_orientation_terms =
      enumerate_orientation_terms(component.right_pairs);
  return component;
}

int component_min_vertex(const xmvb::vb::UnionGraphComponent& component) {
  if (component.local_vertices.empty()) {
    throw std::invalid_argument("union component must not be empty");
  }
  return *std::min_element(component.local_vertices.begin(), component.local_vertices.end());
}

bool is_connected_tree_graph(const xmvb::vb::MetricAwareGraphSummary& summary) {
  if (summary.node_count <= 0) {
    return false;
  }
  return summary.connected_component_count == 1 &&
      summary.edge_count == summary.node_count - 1;
}

bool is_connected_star_graph(const xmvb::vb::MetricAwareComponentGraph& graph) {
  if (graph.node_count <= 2) {
    return true;
  }
  for (int candidate_root = 0; candidate_root < graph.node_count; ++candidate_root) {
    bool is_star = true;
    for (int node = 0; node < graph.node_count; ++node) {
      const int degree =
          static_cast<int>(graph.adjacency[xmvb::to_size(node)].size());
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
      return true;
    }
  }
  return false;
}

int choose_tree_root(
    const xmvb::vb::MetricAwareComponentGraph& graph,
    const std::vector<xmvb::vb::UnionGraphComponent>& components) {
  if (graph.node_count != static_cast<int>(components.size())) {
    throw std::invalid_argument("graph and component count must match");
  }

  int best_root = -1;
  int best_degree = -1;
  int best_covalent_weight = -1;
  int best_min_vertex = std::numeric_limits<int>::max();
  for (int node = 0; node < graph.node_count; ++node) {
    const int degree =
        static_cast<int>(graph.adjacency[xmvb::to_size(node)].size());
    const int covalent_weight =
        graph.component_covalent_label_counts[xmvb::to_size(node)];
    const int min_vertex =
        component_min_vertex(components[xmvb::to_size(node)]);
    if (degree > best_degree ||
        (degree == best_degree && covalent_weight > best_covalent_weight) ||
        (degree == best_degree && covalent_weight == best_covalent_weight &&
         min_vertex < best_min_vertex)) {
      best_root = node;
      best_degree = degree;
      best_covalent_weight = covalent_weight;
      best_min_vertex = min_vertex;
    }
  }
  if (best_root < 0) {
    throw std::runtime_error("failed to choose a rooted-tree center");
  }
  return best_root;
}

RootedTreeLayout build_rooted_tree_layout(
    const xmvb::vb::MetricAwareComponentGraph& graph,
    const std::vector<xmvb::vb::UnionGraphComponent>& components,
    const std::vector<int>& support_orbitals) {
  if (graph.node_count != static_cast<int>(components.size())) {
    throw std::invalid_argument("graph and component count must match");
  }

  RootedTreeLayout layout;
  layout.root_old_index = choose_tree_root(graph, components);
  layout.old_to_preorder.assign(xmvb::to_size(graph.node_count), -1);

  std::vector<std::vector<int>> old_children(xmvb::to_size(graph.node_count));
  const std::function<void(int, int)> dfs =
      [&](int node, int parent) {
        layout.old_to_preorder[xmvb::to_size(node)] =
            static_cast<int>(layout.preorder_old_nodes.size());
        layout.preorder_old_nodes.push_back(node);

        std::vector<int> neighbors = graph.adjacency[xmvb::to_size(node)];
        std::sort(
            neighbors.begin(),
            neighbors.end(),
            [&](int left, int right) {
              const int left_min =
                  component_min_vertex(components[xmvb::to_size(left)]);
              const int right_min =
                  component_min_vertex(components[xmvb::to_size(right)]);
              if (left_min != right_min) {
                return left_min < right_min;
              }
              return left < right;
            });
        for (const int neighbor : neighbors) {
          if (neighbor == parent) {
            continue;
          }
          old_children[xmvb::to_size(node)].push_back(neighbor);
          dfs(neighbor, node);
        }
      };
  dfs(layout.root_old_index, -1);

  if (static_cast<int>(layout.preorder_old_nodes.size()) != graph.node_count) {
    throw std::runtime_error("rooted-tree DFS did not visit all graph nodes");
  }

  layout.preorder_children.assign(
      xmvb::to_size(graph.node_count),
      {});
  for (const int old_node : layout.preorder_old_nodes) {
    const int preorder_node =
        layout.old_to_preorder[xmvb::to_size(old_node)];
    auto& children =
        layout.preorder_children[xmvb::to_size(preorder_node)];
    for (const int old_child : old_children[xmvb::to_size(old_node)]) {
      children.push_back(layout.old_to_preorder[xmvb::to_size(old_child)]);
    }

    // The support-local orbital numbering must follow the component preorder,
    // because each component stores local pair labels relative to that
    // reordered support. Within one component we keep the canonical ascending
    // local-vertex order so the determinant terms match the original support
    // ordering convention.
    std::vector<int> local_vertices =
        components[xmvb::to_size(old_node)].local_vertices;
    std::sort(local_vertices.begin(), local_vertices.end());
    for (const int local_vertex : local_vertices) {
      layout.ordered_support_orbitals.push_back(
          support_orbitals[xmvb::to_size(local_vertex)]);
    }
  }

  return layout;
}

ComponentTree build_component_tree(
    const RootedTreeLayout& layout,
    const std::vector<xmvb::vb::UnionGraphComponent>& ordered_union_components) {
  if (ordered_union_components.size() != layout.preorder_old_nodes.size()) {
    throw std::invalid_argument("ordered union components must match the tree node count");
  }

  ComponentTree tree;
  tree.root_index = 0;
  tree.children = layout.preorder_children;
  tree.components.reserve(ordered_union_components.size());
  for (std::size_t node = 0; node < ordered_union_components.size(); ++node) {
    const int original_graph_node =
        layout.preorder_old_nodes[xmvb::to_size(node)];
    tree.components.push_back(build_local_component_data(
        ordered_union_components[node],
        original_graph_node));
  }
  return tree;
}

void maybe_record_example(
    const Options& options,
    Example example,
    std::vector<Example>* examples) {
  if (examples == nullptr) {
    return;
  }
  examples->push_back(std::move(example));
  std::sort(
      examples->begin(),
      examples->end(),
      [](const Example& left, const Example& right) {
        if (left.total_electronic_abs_error != right.total_electronic_abs_error) {
          return left.total_electronic_abs_error > right.total_electronic_abs_error;
        }
        return left.total_two_electron_abs_error > right.total_two_electron_abs_error;
      });
  if (static_cast<int>(examples->size()) > options.top_examples) {
    examples->resize(xmvb::to_size(options.top_examples));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "validate_rooted_component_tree_exact_hamiltonian currently supports only singlet closed-shell structures");
    }

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
    const auto& prepared_active_space = timed_active_space.prepared_active_space;
    if (prepared_active_space.active_space_two_electron_result.representation !=
        xmvb::vb::ActiveSpaceTwoElectronRepresentation::PackedExact) {
      throw std::runtime_error(
          "validate_rooted_component_tree_exact_hamiltonian requires exact packed ERIs");
    }

    const auto& active_overlap_storage =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    const auto& active_one_electron_storage =
        prepared_active_space.active_space_one_electron_result.h1e_act;
    const auto& packed_active_two_electron_integrals =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    std::vector<std::vector<OrbitalPair>> structure_pairs(
        xmvb::to_size(raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      structure_pairs[xmvb::to_size(structure_index)] =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
    }

    const auto pair_list = build_pair_list(raw_structure_data.n_structures, options.max_pairs);
    const DeterminantOverlapResolver overlap_resolver;

    int total_pairs = 0;
    int connected_tree_pairs = 0;
    int nonstar_tree_pairs = 0;
    int exact_match_count = 0;
    int mismatch_count = 0;
    double max_overlap_abs_error = 0.0;
    double max_one_electron_abs_error = 0.0;
    double max_same_spin_alpha_abs_error = 0.0;
    double max_same_spin_beta_abs_error = 0.0;
    double max_opposite_spin_abs_error = 0.0;
    double max_total_two_electron_abs_error = 0.0;
    double max_total_electronic_abs_error = 0.0;
    std::uint64_t max_subtree_message_state_count = 0;
    std::uint64_t max_dp_transition_count = 0;
    std::vector<Example> examples;

    for (const auto& [left_structure, right_structure] : pair_list) {
      ++total_pairs;
      const auto& left_pairs = structure_pairs[xmvb::to_size(left_structure)];
      const auto& right_pairs = structure_pairs[xmvb::to_size(right_structure)];

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
          {.edge_max_abs_threshold = options.edge_threshold});
      const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
          metric_graph,
          union_components);
      if (!is_connected_tree_graph(metric_summary)) {
        continue;
      }

      ++connected_tree_pairs;
      if (!is_connected_star_graph(metric_graph)) {
        ++nonstar_tree_pairs;
      }

      const RootedTreeLayout layout = build_rooted_tree_layout(
          metric_graph,
          union_components,
          support_orbitals);
      const auto ordered_support_index =
          xmvb::vb::build_support_index(layout.ordered_support_orbitals);
      const auto ordered_left_pairs =
          xmvb::vb::remap_pairs_to_support(left_pairs, ordered_support_index);
      const auto ordered_right_pairs =
          xmvb::vb::remap_pairs_to_support(right_pairs, ordered_support_index);
      const auto ordered_union_components = xmvb::vb::build_union_graph_components(
          ordered_left_pairs,
          ordered_right_pairs,
          layout.ordered_support_orbitals);
      if (ordered_union_components.size() != union_components.size()) {
        throw std::runtime_error(
            "component-tree support reorder changed the number of union components");
      }

      const ComponentTree tree =
          build_component_tree(layout, ordered_union_components);
      const std::vector<double> ordered_overlap_storage = flatten_column_major_matrix(
          xmvb::vb::build_support_overlap_matrix(
              layout.ordered_support_orbitals,
              active_overlap_storage,
              n_active_orbitals));
      const std::vector<double> ordered_one_electron_storage =
          flatten_column_major_matrix(
              build_support_submatrix(
                  layout.ordered_support_orbitals,
                  active_one_electron_storage,
                  n_active_orbitals));
      const std::vector<double> ordered_packed_two_electron =
          build_support_local_packed_two_electron_integrals(
              layout.ordered_support_orbitals,
              packed_active_two_electron_integrals);

      const CollapsedHamiltonianComponentTreeStats stats =
          xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian(
              ordered_overlap_storage,
              ordered_one_electron_storage,
              ordered_packed_two_electron,
              static_cast<int>(layout.ordered_support_orbitals.size()),
              tree,
              overlap_resolver);

      max_overlap_abs_error =
          std::max(max_overlap_abs_error, stats.overlap_absolute_error);
      max_one_electron_abs_error =
          std::max(max_one_electron_abs_error, stats.one_electron_absolute_error);
      max_same_spin_alpha_abs_error = std::max(
          max_same_spin_alpha_abs_error,
          stats.same_spin_alpha_absolute_error);
      max_same_spin_beta_abs_error = std::max(
          max_same_spin_beta_abs_error,
          stats.same_spin_beta_absolute_error);
      max_opposite_spin_abs_error = std::max(
          max_opposite_spin_abs_error,
          stats.opposite_spin_absolute_error);
      max_total_two_electron_abs_error = std::max(
          max_total_two_electron_abs_error,
          stats.total_two_electron_absolute_error);
      max_total_electronic_abs_error = std::max(
          max_total_electronic_abs_error,
          stats.total_electronic_hamiltonian_absolute_error);
      max_subtree_message_state_count = std::max(
          max_subtree_message_state_count,
          stats.subtree_message_state_count);
      max_dp_transition_count = std::max(
          max_dp_transition_count,
          stats.dp_transition_count);

      const bool matches =
          stats.overlap_absolute_error <= options.tolerance &&
          stats.one_electron_absolute_error <= options.tolerance &&
          stats.same_spin_alpha_absolute_error <= options.tolerance &&
          stats.same_spin_beta_absolute_error <= options.tolerance &&
          stats.opposite_spin_absolute_error <= options.tolerance &&
          stats.total_two_electron_absolute_error <= options.tolerance &&
          stats.total_electronic_hamiltonian_absolute_error <= options.tolerance;
      if (matches) {
        ++exact_match_count;
      } else {
        ++mismatch_count;
        maybe_record_example(
            options,
            Example{
                .left_structure = left_structure,
                .right_structure = right_structure,
                .root_component = layout.root_old_index,
                .node_count = metric_graph.node_count,
                .support_size = static_cast<int>(layout.ordered_support_orbitals.size()),
                .overlap_abs_error = stats.overlap_absolute_error,
                .one_electron_abs_error = stats.one_electron_absolute_error,
                .same_spin_alpha_abs_error = stats.same_spin_alpha_absolute_error,
                .same_spin_beta_abs_error = stats.same_spin_beta_absolute_error,
                .opposite_spin_abs_error = stats.opposite_spin_absolute_error,
                .total_two_electron_abs_error =
                    stats.total_two_electron_absolute_error,
                .total_electronic_abs_error =
                    stats.total_electronic_hamiltonian_absolute_error,
            },
            &examples);
      }
    }

    if (connected_tree_pairs == 0) {
      throw std::runtime_error("no connected tree-like metric graph pairs were found");
    }

    std::cout << std::setprecision(15);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "edge_threshold = " << options.edge_threshold << '\n';
    std::cout << "total_pairs = " << total_pairs << '\n';
    std::cout << "connected_tree_pairs = " << connected_tree_pairs << '\n';
    std::cout << "nonstar_tree_pairs = " << nonstar_tree_pairs << '\n';
    std::cout << "exact_match_count = " << exact_match_count << '\n';
    std::cout << "mismatch_count = " << mismatch_count << '\n';
    std::cout << "max_overlap_abs_error = " << max_overlap_abs_error << '\n';
    std::cout << "max_one_electron_abs_error = " << max_one_electron_abs_error << '\n';
    std::cout << "max_same_spin_alpha_abs_error = "
              << max_same_spin_alpha_abs_error << '\n';
    std::cout << "max_same_spin_beta_abs_error = "
              << max_same_spin_beta_abs_error << '\n';
    std::cout << "max_opposite_spin_abs_error = "
              << max_opposite_spin_abs_error << '\n';
    std::cout << "max_total_two_electron_abs_error = "
              << max_total_two_electron_abs_error << '\n';
    std::cout << "max_total_electronic_abs_error = "
              << max_total_electronic_abs_error << '\n';
    std::cout << "max_subtree_message_state_count = "
              << max_subtree_message_state_count << '\n';
    std::cout << "max_dp_transition_count = "
              << max_dp_transition_count << '\n';
    if (!examples.empty()) {
      std::cout << "top_mismatches:\n";
      for (const auto& example : examples) {
        std::cout << "  left=" << example.left_structure
                  << " right=" << example.right_structure
                  << " root=" << example.root_component
                  << " nodes=" << example.node_count
                  << " support=" << example.support_size
                  << " overlap_err=" << example.overlap_abs_error
                  << " h1_err=" << example.one_electron_abs_error
                  << " aa_err=" << example.same_spin_alpha_abs_error
                  << " bb_err=" << example.same_spin_beta_abs_error
                  << " ab_err=" << example.opposite_spin_abs_error
                  << " h2_err=" << example.total_two_electron_abs_error
                  << " htot_err=" << example.total_electronic_abs_error
                  << '\n';
      }
    }

    return (mismatch_count == 0) ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
