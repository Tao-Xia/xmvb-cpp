#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/exact_separator/bundle_channels.hpp"
#include "vb/exact_separator/component_tree.hpp"
#include "vb/exact_separator/leaf_boundary_bundle_table.hpp"
#include "vb/exact_separator/leaf_coefficient_operator.hpp"
#include "vb/exact_separator/one_electron.hpp"
#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using OrbitalPair = xmvb::vb::OrbitalPair;
using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedOneElectronStarPairStats;
using xmvb::vb::exact_separator::CollapsedTwoElectronOneLeafStarPairStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ComponentSpinCoefficientOperator;
using xmvb::vb::exact_separator::ComponentTree;
using xmvb::vb::exact_separator::ComponentTreeHamiltonianResult;
using xmvb::vb::exact_separator::ComponentTreeOneElectronResult;
using xmvb::vb::exact_separator::ComponentTreeOppositeSpinDebugResult;
using xmvb::vb::exact_separator::ComponentTreeOppositeSpinResult;
using xmvb::vb::exact_separator::ComponentTreeOverlapResult;
using xmvb::vb::exact_separator::ComponentTreeSameSpinResult;
using xmvb::vb::exact_separator::BoundarySpinBundle;
using xmvb::vb::exact_separator::IndexedOneLeafBoundarySpinBundleTable;
using xmvb::vb::exact_separator::OrientationTerm;

struct Options {
  std::string input_path;
  int left_structure = -1;
  int right_structure = -1;
  double edge_threshold = 0.0;
};

struct RootedTreeLayout {
  int root_old_index = -1;
  std::vector<int> preorder_old_nodes;
  std::vector<int> old_to_preorder;
  std::vector<std::vector<int>> preorder_children;
  std::vector<int> ordered_support_orbitals;
};

void print_usage() {
  std::cerr << "usage: debug_rooted_component_tree_star_pair <input.xmi>"
               " --left-structure I --right-structure J"
               " [--edge-threshold F]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 6 || ((argc - 2) % 2 != 0)) {
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
    if (argument_name == "--edge-threshold") {
      options.edge_threshold = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.left_structure < 0 || options.right_structure < 0) {
    throw std::invalid_argument("both --left-structure and --right-structure are required");
  }
  if (options.left_structure <= options.right_structure) {
    throw std::invalid_argument("--left-structure must be greater than --right-structure");
  }
  if (options.edge_threshold < 0.0) {
    throw std::invalid_argument("--edge-threshold must be non-negative");
  }
  return options;
}

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

Matrix build_support_submatrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& full_storage,
    int n_active_orbitals) {
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
  int minimum = component.local_vertices.front();
  for (const int vertex : component.local_vertices) {
    minimum = std::min(minimum, vertex);
  }
  return minimum;
}

bool is_connected_tree_graph(const xmvb::vb::MetricAwareGraphSummary& summary) {
  return summary.node_count > 0 &&
      summary.connected_component_count == 1 &&
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
    throw std::runtime_error("failed to choose a tree root");
  }
  return best_root;
}

RootedTreeLayout build_rooted_tree_layout(
    const xmvb::vb::MetricAwareComponentGraph& graph,
    const std::vector<xmvb::vb::UnionGraphComponent>& components,
    const std::vector<int>& support_orbitals) {
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

  layout.preorder_children.assign(
      xmvb::to_size(graph.node_count),
      {});
  for (const int old_node : layout.preorder_old_nodes) {
    const int preorder_node = layout.old_to_preorder[xmvb::to_size(old_node)];
    for (const int old_child : old_children[xmvb::to_size(old_node)]) {
      layout.preorder_children[xmvb::to_size(preorder_node)].push_back(
          layout.old_to_preorder[xmvb::to_size(old_child)]);
    }

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
  ComponentTree tree;
  tree.root_index = 0;
  tree.children = layout.preorder_children;
  for (std::size_t node = 0; node < ordered_union_components.size(); ++node) {
    tree.components.push_back(build_local_component_data(
        ordered_union_components[node],
        layout.preorder_old_nodes[node]));
  }
  return tree;
}

std::string format_ints(const std::vector<int>& values) {
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

std::string format_pairs(const std::vector<OrbitalPair>& pairs) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    if (index > 0) {
      stream << ", ";
    }
    stream << "(" << pairs[index].first << "," << pairs[index].second << ")";
  }
  stream << "]";
  return stream.str();
}

void print_component_summary(const ComponentTree& tree) {
  std::cout << "ordered_components:\n";
  for (std::size_t node = 0; node < tree.components.size(); ++node) {
    const ComponentData& component = tree.components[node];
    std::cout << "  node=" << node
              << " old_graph_node=" << component.graph_node
              << " children=" << format_ints(tree.children[node]) << '\n';
    std::cout << "    left_pairs=" << format_pairs(component.left_pairs) << '\n';
    std::cout << "    right_pairs=" << format_pairs(component.right_pairs) << '\n';
    std::cout << "    left_term_count=" << component.left_orientation_terms.size()
              << " right_term_count=" << component.right_orientation_terms.size() << '\n';
  }
}

void print_hamiltonian_result(
    const std::string& label,
    const ComponentTreeHamiltonianResult& result,
    const ComponentTreeHamiltonianResult& exact) {
  std::cout << '[' << label << "]\n";
  std::cout << "  overlap = " << result.overlap
            << " err = " << std::abs(result.overlap - exact.overlap) << '\n';
  std::cout << "  one_electron = " << result.one_electron
            << " err = " << std::abs(result.one_electron - exact.one_electron) << '\n';
  std::cout << "  same_spin_alpha = " << result.same_spin_alpha_two_electron
            << " err = "
            << std::abs(
                   result.same_spin_alpha_two_electron -
                   exact.same_spin_alpha_two_electron)
            << '\n';
  std::cout << "  same_spin_beta = " << result.same_spin_beta_two_electron
            << " err = "
            << std::abs(
                   result.same_spin_beta_two_electron -
                   exact.same_spin_beta_two_electron)
            << '\n';
  std::cout << "  opposite_spin = " << result.opposite_spin_two_electron
            << " err = "
            << std::abs(
                   result.opposite_spin_two_electron -
                   exact.opposite_spin_two_electron)
            << '\n';
  std::cout << "  two_electron = " << result.two_electron
            << " err = " << std::abs(result.two_electron - exact.two_electron) << '\n';
  std::cout << "  total_electronic = " << result.total_electronic_hamiltonian
            << " err = "
            << std::abs(
                   result.total_electronic_hamiltonian -
                   exact.total_electronic_hamiltonian)
            << '\n';
  std::cout << "  subtree_message_state_count = " << result.subtree_message_state_count << '\n';
  std::cout << "  subtree_term_pair_count = " << result.subtree_term_pair_count << '\n';
  std::cout << "  subdeterminant_evaluations = " << result.subdeterminant_evaluations << '\n';
  std::cout << "  dp_transition_count = " << result.dp_transition_count << '\n';
}

double contract_first_cofactor_matrix(
    const Matrix& first_cofactor,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  double total = 0.0;
  for (int col = 0; col < first_cofactor.cols(); ++col) {
    for (int row = 0; row < first_cofactor.rows(); ++row) {
      total +=
          support_one_electron_storage[xmvb::to_size(col) *
                                           xmvb::to_size(support_size) +
                                       xmvb::to_size(row)] *
          first_cofactor(row, col);
    }
  }
  return total;
}

Matrix build_support_first_cofactor_from_bundle(
    const BoundarySpinBundle& bundle,
    int support_size) {
  Matrix first_cofactor = Matrix::Zero(support_size, support_size);
  for (int support_pair_index = 0;
       support_pair_index < bundle.layout.support_pair_count();
       ++support_pair_index) {
    int row_orbital = -1;
    int col_orbital = -1;
    xmvb::vb::exact_separator::decode_support_pair_index(
        support_pair_index,
        support_size,
        &row_orbital,
        &col_orbital);
    double value = 0.0;
    for (int flat_sector_index = 0;
         flat_sector_index < bundle.layout.total_sector_count();
         ++flat_sector_index) {
      value += bundle.degree1[xmvb::to_size(
          bundle.layout.flat_degree1_index(
              support_pair_index,
              flat_sector_index))];
    }
    first_cofactor(row_orbital, col_orbital) = value;
  }
  return first_cofactor;
}

Matrix accumulate_exact_one_leaf_first_cofactor(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const IndexedOneLeafBoundarySpinBundleTable& active_table,
    const IndexedOneLeafBoundarySpinBundleTable& spectator_table,
    bool alpha_active,
    int support_size) {
  Matrix total = Matrix::Zero(support_size, support_size);
  for (const auto& root_entry : root_operator.entries) {
    for (const auto& leaf_entry : leaf_operator.entries) {
      const double coefficient =
          root_entry.coefficient * leaf_entry.coefficient;
      if (std::abs(coefficient) <= 1.0e-15) {
        continue;
      }

      const int active_root_index =
          alpha_active ? root_entry.alpha_state_index : root_entry.beta_state_index;
      const int active_leaf_index =
          alpha_active ? leaf_entry.alpha_state_index : leaf_entry.beta_state_index;
      const int spectator_root_index =
          alpha_active ? root_entry.beta_state_index : root_entry.alpha_state_index;
      const int spectator_leaf_index =
          alpha_active ? leaf_entry.beta_state_index : leaf_entry.alpha_state_index;

      const BoundarySpinBundle& active_bundle =
          active_table.bundles[xmvb::to_size(
              active_table.index(active_root_index, active_leaf_index))];
      const BoundarySpinBundle& spectator_bundle =
          spectator_table.bundles[xmvb::to_size(
              spectator_table.index(spectator_root_index, spectator_leaf_index))];
      total +=
          coefficient *
          xmvb::vb::exact_separator::contract_boundary_spin_bundle_overlap(
              spectator_bundle) *
          build_support_first_cofactor_from_bundle(active_bundle, support_size);
    }
  }
  return total;
}

struct MatrixDiffSummary {
  double max_abs = 0.0;
  int row = -1;
  int col = -1;
  double actual = 0.0;
  double reference = 0.0;
};

struct MatrixDiffEntry {
  double abs_diff = 0.0;
  int row = -1;
  int col = -1;
  double actual = 0.0;
  double reference = 0.0;
};

MatrixDiffSummary summarize_matrix_difference(
    const Matrix& actual,
    const Matrix& reference) {
  if (actual.rows() != reference.rows() || actual.cols() != reference.cols()) {
    throw std::invalid_argument("matrix dimensions do not match");
  }
  MatrixDiffSummary summary;
  for (int col = 0; col < actual.cols(); ++col) {
    for (int row = 0; row < actual.rows(); ++row) {
      const double diff = std::abs(actual(row, col) - reference(row, col));
      if (diff <= summary.max_abs) {
        continue;
      }
      summary.max_abs = diff;
      summary.row = row;
      summary.col = col;
      summary.actual = actual(row, col);
      summary.reference = reference(row, col);
    }
  }
  return summary;
}

std::vector<MatrixDiffEntry> top_matrix_differences(
    const Matrix& actual,
    const Matrix& reference,
    int max_count) {
  if (actual.rows() != reference.rows() || actual.cols() != reference.cols()) {
    throw std::invalid_argument("matrix dimensions do not match");
  }
  std::vector<MatrixDiffEntry> diffs;
  for (int col = 0; col < actual.cols(); ++col) {
    for (int row = 0; row < actual.rows(); ++row) {
      const double abs_diff = std::abs(actual(row, col) - reference(row, col));
      if (abs_diff <= 1.0e-15) {
        continue;
      }
      diffs.push_back(MatrixDiffEntry{
          .abs_diff = abs_diff,
          .row = row,
          .col = col,
          .actual = actual(row, col),
          .reference = reference(row, col),
      });
    }
  }
  std::sort(
      diffs.begin(),
      diffs.end(),
      [](const MatrixDiffEntry& left, const MatrixDiffEntry& right) {
        if (left.abs_diff != right.abs_diff) {
          return left.abs_diff > right.abs_diff;
        }
        if (left.row != right.row) {
          return left.row < right.row;
        }
        return left.col < right.col;
      });
  if (static_cast<int>(diffs.size()) > max_count) {
    diffs.resize(xmvb::to_size(max_count));
  }
  return diffs;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error("only singlet closed-shell inputs are supported");
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
      throw std::runtime_error("debug tool requires exact packed ERIs");
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

    const std::vector<OrbitalPair> left_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.left_structure);
    const std::vector<OrbitalPair> right_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.right_structure);
    const std::vector<int> support_orbitals =
        xmvb::vb::build_support_orbitals(left_pairs, right_pairs);
    const auto support_index = xmvb::vb::build_support_index(support_orbitals);
    const std::vector<OrbitalPair> left_pairs_local =
        xmvb::vb::remap_pairs_to_support(left_pairs, support_index);
    const std::vector<OrbitalPair> right_pairs_local =
        xmvb::vb::remap_pairs_to_support(right_pairs, support_index);
    const Matrix support_overlap = xmvb::vb::build_support_overlap_matrix(
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
      throw std::runtime_error("requested pair is not a connected tree-like pair");
    }

    const RootedTreeLayout layout = build_rooted_tree_layout(
        metric_graph,
        union_components,
        support_orbitals);
    const auto ordered_support_index =
        xmvb::vb::build_support_index(layout.ordered_support_orbitals);
    const std::vector<OrbitalPair> ordered_left_pairs =
        xmvb::vb::remap_pairs_to_support(left_pairs, ordered_support_index);
    const std::vector<OrbitalPair> ordered_right_pairs =
        xmvb::vb::remap_pairs_to_support(right_pairs, ordered_support_index);
    const auto ordered_union_components = xmvb::vb::build_union_graph_components(
        ordered_left_pairs,
        ordered_right_pairs,
        layout.ordered_support_orbitals);
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
    const int support_size = static_cast<int>(layout.ordered_support_orbitals.size());
    const DeterminantOverlapResolver overlap_resolver;

    std::cout << std::setprecision(15);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "left_structure = " << options.left_structure << '\n';
    std::cout << "right_structure = " << options.right_structure << '\n';
    std::cout << "support_size = " << support_size << '\n';
    std::cout << "support_orbitals = " << format_ints(support_orbitals) << '\n';
    std::cout << "ordered_support_orbitals = "
              << format_ints(layout.ordered_support_orbitals) << '\n';
    std::cout << "graph_node_count = " << metric_graph.node_count << '\n';
    std::cout << "is_star = " << (is_connected_star_graph(metric_graph) ? "true" : "false")
              << '\n';
    std::cout << "root_old_index = " << layout.root_old_index << '\n';
    std::cout << "preorder_old_nodes = " << format_ints(layout.preorder_old_nodes) << '\n';
    print_component_summary(tree);

    const ComponentTreeHamiltonianResult exact =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian_exact(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            ordered_packed_two_electron,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeHamiltonianResult production =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian_collapsed(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            ordered_packed_two_electron,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeHamiltonianResult generic_bundle =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian_bundle_boundary_collapsed(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            ordered_packed_two_electron,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeOverlapResult boundary_overlap =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_overlap_boundary_collapsed(
            ordered_overlap_storage,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeOneElectronResult boundary_one_electron =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_one_electron_boundary_collapsed(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeOppositeSpinResult boundary_opposite_spin =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_opposite_spin_boundary_collapsed(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            ordered_packed_two_electron,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeOppositeSpinDebugResult boundary_opposite_spin_debug =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_opposite_spin_boundary_debug(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            ordered_packed_two_electron,
            support_size,
            tree,
            overlap_resolver);
    const ComponentTreeSameSpinResult boundary_same_spin =
        xmvb::vb::exact_separator::evaluate_rooted_component_tree_same_spin_boundary_collapsed(
            ordered_overlap_storage,
            ordered_one_electron_storage,
            ordered_packed_two_electron,
            support_size,
            tree,
            overlap_resolver);

    ComponentTreeHamiltonianResult boundary_composed;
    boundary_composed.overlap = boundary_one_electron.overlap;
    boundary_composed.one_electron = boundary_one_electron.one_electron;
    boundary_composed.same_spin_alpha_two_electron =
        boundary_same_spin.same_spin_alpha_two_electron;
    boundary_composed.same_spin_beta_two_electron =
        boundary_same_spin.same_spin_beta_two_electron;
    boundary_composed.opposite_spin_two_electron =
        boundary_opposite_spin.opposite_spin_two_electron;
    boundary_composed.two_electron =
        boundary_composed.same_spin_alpha_two_electron +
        boundary_composed.same_spin_beta_two_electron +
        boundary_composed.opposite_spin_two_electron;
    boundary_composed.total_electronic_hamiltonian =
        boundary_composed.one_electron + boundary_composed.two_electron;
    boundary_composed.subtree_message_state_count =
        boundary_one_electron.subtree_message_state_count +
        boundary_opposite_spin.subtree_message_state_count +
        boundary_same_spin.subtree_message_state_count;
    boundary_composed.subtree_term_pair_count =
        boundary_one_electron.subtree_term_pair_count +
        boundary_opposite_spin.subtree_term_pair_count +
        boundary_same_spin.subtree_term_pair_count;
    boundary_composed.subdeterminant_evaluations =
        boundary_one_electron.subdeterminant_evaluations +
        boundary_opposite_spin.subdeterminant_evaluations +
        boundary_same_spin.subdeterminant_evaluations;
    boundary_composed.dp_transition_count =
        boundary_one_electron.dp_transition_count +
        boundary_opposite_spin.dp_transition_count +
        boundary_same_spin.dp_transition_count;

    std::cout << "[exact]\n";
    std::cout << "  overlap = " << exact.overlap << '\n';
    std::cout << "  one_electron = " << exact.one_electron << '\n';
    std::cout << "  same_spin_alpha = " << exact.same_spin_alpha_two_electron << '\n';
    std::cout << "  same_spin_beta = " << exact.same_spin_beta_two_electron << '\n';
    std::cout << "  opposite_spin = " << exact.opposite_spin_two_electron << '\n';
    std::cout << "  two_electron = " << exact.two_electron << '\n';
    std::cout << "  total_electronic = " << exact.total_electronic_hamiltonian << '\n';

    print_hamiltonian_result("production_collapsed", production, exact);
    print_hamiltonian_result("generic_bundle", generic_bundle, exact);

    std::cout << "[boundary_overlap_only]\n";
    std::cout << "  overlap = " << boundary_overlap.overlap
              << " err = " << std::abs(boundary_overlap.overlap - exact.overlap) << '\n';
    std::cout << "  subtree_message_state_count = "
              << boundary_overlap.subtree_message_state_count << '\n';
    std::cout << "  subtree_term_pair_count = "
              << boundary_overlap.subtree_term_pair_count << '\n';
    std::cout << "  subdeterminant_evaluations = "
              << boundary_overlap.subdeterminant_evaluations << '\n';
    std::cout << "  dp_transition_count = "
              << boundary_overlap.dp_transition_count << '\n';

    std::cout << "[boundary_one_electron]\n";
    std::cout << "  overlap = " << boundary_one_electron.overlap
              << " err = " << std::abs(boundary_one_electron.overlap - exact.overlap) << '\n';
    std::cout << "  one_electron = " << boundary_one_electron.one_electron
              << " err = "
              << std::abs(boundary_one_electron.one_electron - exact.one_electron) << '\n';
    std::cout << "  subtree_message_state_count = "
              << boundary_one_electron.subtree_message_state_count << '\n';
    std::cout << "  subtree_term_pair_count = "
              << boundary_one_electron.subtree_term_pair_count << '\n';
    std::cout << "  subdeterminant_evaluations = "
              << boundary_one_electron.subdeterminant_evaluations << '\n';
    std::cout << "  dp_transition_count = "
              << boundary_one_electron.dp_transition_count << '\n';

    std::cout << "[boundary_opposite_spin]\n";
    std::cout << "  overlap = " << boundary_opposite_spin.overlap
              << " err = " << std::abs(boundary_opposite_spin.overlap - exact.overlap) << '\n';
    std::cout << "  one_electron = " << boundary_opposite_spin.one_electron
              << " err = "
              << std::abs(boundary_opposite_spin.one_electron - exact.one_electron) << '\n';
    std::cout << "  opposite_spin = " << boundary_opposite_spin.opposite_spin_two_electron
              << " err = "
              << std::abs(
                     boundary_opposite_spin.opposite_spin_two_electron -
                     exact.opposite_spin_two_electron)
              << '\n';
    std::cout << "  subtree_message_state_count = "
              << boundary_opposite_spin.subtree_message_state_count << '\n';
    std::cout << "  subtree_term_pair_count = "
              << boundary_opposite_spin.subtree_term_pair_count << '\n';
    std::cout << "  subdeterminant_evaluations = "
              << boundary_opposite_spin.subdeterminant_evaluations << '\n';
    std::cout << "  dp_transition_count = "
              << boundary_opposite_spin.dp_transition_count << '\n';

    std::cout << "[boundary_same_spin]\n";
    std::cout << "  overlap = " << boundary_same_spin.overlap
              << " err = " << std::abs(boundary_same_spin.overlap - exact.overlap) << '\n';
    std::cout << "  one_electron = " << boundary_same_spin.one_electron
              << " err = "
              << std::abs(boundary_same_spin.one_electron - exact.one_electron) << '\n';
    std::cout << "  same_spin_alpha = " << boundary_same_spin.same_spin_alpha_two_electron
              << " err = "
              << std::abs(
                     boundary_same_spin.same_spin_alpha_two_electron -
                     exact.same_spin_alpha_two_electron)
              << '\n';
    std::cout << "  same_spin_beta = " << boundary_same_spin.same_spin_beta_two_electron
              << " err = "
              << std::abs(
                     boundary_same_spin.same_spin_beta_two_electron -
                     exact.same_spin_beta_two_electron)
              << '\n';
    std::cout << "  subtree_message_state_count = "
              << boundary_same_spin.subtree_message_state_count << '\n';
    std::cout << "  subtree_term_pair_count = "
              << boundary_same_spin.subtree_term_pair_count << '\n';
    std::cout << "  subdeterminant_evaluations = "
              << boundary_same_spin.subdeterminant_evaluations << '\n';
    std::cout << "  dp_transition_count = "
              << boundary_same_spin.dp_transition_count << '\n';

    print_hamiltonian_result("boundary_composed_total", boundary_composed, exact);

    if (tree.components.size() == 2U) {
      const CollapsedOneElectronStarPairStats one_leaf_one =
          xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_one_electron(
              exact.overlap,
              exact.one_electron,
              ordered_overlap_storage,
              ordered_one_electron_storage,
              support_size,
              tree.components,
              overlap_resolver);
      const CollapsedTwoElectronOneLeafStarPairStats one_leaf_two =
          xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
              ordered_overlap_storage,
              ordered_packed_two_electron,
              support_size,
              tree.components,
              overlap_resolver);
      std::cout << "[one_leaf_specialized]\n";
      std::cout << "  overlap = " << one_leaf_one.collapsed_overlap
                << " err = " << one_leaf_one.overlap_absolute_error << '\n';
      std::cout << "  one_electron = " << one_leaf_one.collapsed_one_electron
                << " err = " << one_leaf_one.one_electron_absolute_error << '\n';
      std::cout << "  same_spin_alpha = "
                << one_leaf_two.collapsed_same_spin_alpha_two_electron
                << " err = " << one_leaf_two.same_spin_alpha_absolute_error << '\n';
      std::cout << "  same_spin_beta = "
                << one_leaf_two.collapsed_same_spin_beta_two_electron
                << " err = " << one_leaf_two.same_spin_beta_absolute_error << '\n';
      std::cout << "  opposite_spin = "
                << one_leaf_two.collapsed_opposite_spin_two_electron
                << " err = " << one_leaf_two.opposite_spin_absolute_error << '\n';
      std::cout << "  two_electron = " << one_leaf_two.collapsed_two_electron
                << " err = " << one_leaf_two.total_two_electron_absolute_error << '\n';
      std::cout << "  total_electronic = "
                << (one_leaf_one.collapsed_one_electron + one_leaf_two.collapsed_two_electron)
                << " err = "
                << std::abs(
                       one_leaf_one.collapsed_one_electron +
                           one_leaf_two.collapsed_two_electron -
                       exact.total_electronic_hamiltonian)
                << '\n';
      std::cout << "  total_boundary_sector_count = "
                << (one_leaf_one.collapsed_leaf_state_count +
                    one_leaf_two.alpha_mask_state_count +
                    one_leaf_two.beta_mask_state_count)
                << '\n';
      std::cout << "  subdeterminant_evaluations = "
                << (one_leaf_one.subdeterminant_evaluations +
                    one_leaf_two.subdeterminant_evaluations)
                << '\n';

      const ComponentSpinCoefficientOperator root_operator =
          xmvb::vb::exact_separator::build_component_spin_coefficient_operator(
              tree.components.front());
      const ComponentSpinCoefficientOperator leaf_operator =
          xmvb::vb::exact_separator::build_component_spin_coefficient_operator(
              tree.components.back());
      const auto alpha_state_pairs =
          xmvb::vb::exact_separator::collect_referenced_one_leaf_spin_state_pairs(
              root_operator,
              leaf_operator,
              true);
      const auto beta_state_pairs =
          xmvb::vb::exact_separator::collect_referenced_one_leaf_spin_state_pairs(
              root_operator,
              leaf_operator,
              false);
      std::uint64_t one_leaf_debug_subdeterminants = 0;
      const IndexedOneLeafBoundarySpinBundleTable alpha_table =
          xmvb::vb::exact_separator::build_indexed_one_leaf_boundary_spin_bundle_table(
              root_operator.alpha_states,
              leaf_operator.alpha_states,
              alpha_state_pairs,
              ordered_overlap_storage,
              support_size,
              overlap_resolver,
              &one_leaf_debug_subdeterminants);
      const IndexedOneLeafBoundarySpinBundleTable beta_table =
          xmvb::vb::exact_separator::build_indexed_one_leaf_boundary_spin_bundle_table(
              root_operator.beta_states,
              leaf_operator.beta_states,
              beta_state_pairs,
              ordered_overlap_storage,
              support_size,
              overlap_resolver,
              &one_leaf_debug_subdeterminants);
      const Matrix exact_alpha_first_cofactor =
          accumulate_exact_one_leaf_first_cofactor(
              root_operator,
              leaf_operator,
              alpha_table,
              beta_table,
              true,
              support_size);
      const Matrix exact_beta_first_cofactor =
          accumulate_exact_one_leaf_first_cofactor(
              root_operator,
              leaf_operator,
              beta_table,
              alpha_table,
              false,
              support_size);
      const MatrixDiffSummary alpha_summary = summarize_matrix_difference(
          boundary_opposite_spin_debug.alpha_first_cofactor,
          exact_alpha_first_cofactor);
      const MatrixDiffSummary beta_summary = summarize_matrix_difference(
          boundary_opposite_spin_debug.beta_first_cofactor,
          exact_beta_first_cofactor);
      const std::vector<MatrixDiffEntry> alpha_top_diffs =
          top_matrix_differences(
              boundary_opposite_spin_debug.alpha_first_cofactor,
              exact_alpha_first_cofactor,
              8);
      const std::vector<MatrixDiffEntry> beta_top_diffs =
          top_matrix_differences(
              boundary_opposite_spin_debug.beta_first_cofactor,
              exact_beta_first_cofactor,
              8);

      std::cout << "[degree1_first_cofactor_debug]\n";
      std::cout << "  alpha_h1e_from_generic = "
                << contract_first_cofactor_matrix(
                       boundary_opposite_spin_debug.alpha_first_cofactor,
                       ordered_one_electron_storage,
                       support_size)
                << '\n';
      std::cout << "  alpha_h1e_from_exact_one_leaf = "
                << contract_first_cofactor_matrix(
                       exact_alpha_first_cofactor,
                       ordered_one_electron_storage,
                       support_size)
                << '\n';
      std::cout << "  alpha_max_abs_diff = " << alpha_summary.max_abs
                << " at (row=" << alpha_summary.row
                << ", col=" << alpha_summary.col
                << ") generic=" << alpha_summary.actual
                << " exact=" << alpha_summary.reference << '\n';
      std::cout << "  beta_h1e_from_generic = "
                << contract_first_cofactor_matrix(
                       boundary_opposite_spin_debug.beta_first_cofactor,
                       ordered_one_electron_storage,
                       support_size)
                << '\n';
      std::cout << "  beta_h1e_from_exact_one_leaf = "
                << contract_first_cofactor_matrix(
                       exact_beta_first_cofactor,
                       ordered_one_electron_storage,
                       support_size)
                << '\n';
      std::cout << "  beta_max_abs_diff = " << beta_summary.max_abs
                << " at (row=" << beta_summary.row
                << ", col=" << beta_summary.col
                << ") generic=" << beta_summary.actual
                << " exact=" << beta_summary.reference << '\n';
      std::cout << "  alpha_top_diffs:\n";
      for (const auto& entry : alpha_top_diffs) {
        std::cout << "    row=" << entry.row
                  << " (orb=" << layout.ordered_support_orbitals[xmvb::to_size(entry.row)]
                  << "), col=" << entry.col
                  << " (orb=" << layout.ordered_support_orbitals[xmvb::to_size(entry.col)]
                  << "), abs_diff=" << entry.abs_diff
                  << ", generic=" << entry.actual
                  << ", exact=" << entry.reference << '\n';
      }
      std::cout << "  beta_top_diffs:\n";
      for (const auto& entry : beta_top_diffs) {
        std::cout << "    row=" << entry.row
                  << " (orb=" << layout.ordered_support_orbitals[xmvb::to_size(entry.row)]
                  << "), col=" << entry.col
                  << " (orb=" << layout.ordered_support_orbitals[xmvb::to_size(entry.col)]
                  << "), abs_diff=" << entry.abs_diff
                  << ", generic=" << entry.actual
                  << ", exact=" << entry.reference << '\n';
      }
      std::cout << "  one_leaf_debug_subdeterminants = "
                << one_leaf_debug_subdeterminants << '\n';
    }

    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
