#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/exact_separator/component_tree.hpp"
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
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ComponentTree;
using xmvb::vb::exact_separator::ComponentTreeHamiltonianResult;
using xmvb::vb::exact_separator::ComponentTreeOneElectronResult;
using xmvb::vb::exact_separator::ComponentTreeOverlapResult;
using xmvb::vb::exact_separator::ExactTwoElectronStarPairStats;
using xmvb::vb::exact_separator::OneElectronStarPairResult;
using xmvb::vb::exact_separator::OrientationTerm;
using xmvb::vb::exact_separator::TwoElectronStarPairResult;

struct Options {
  std::string input_path;
  int max_pairs = 0;
  int repeat = 10;
  int report_every = 0;
  double edge_threshold = 0.0;
};

struct RootedTreeLayout {
  int root_old_index = -1;
  std::vector<int> preorder_old_nodes;
  std::vector<int> old_to_preorder;
  std::vector<std::vector<int>> preorder_children;
  std::vector<int> ordered_support_orbitals;
};

enum class PairCategory {
  OneNode,
  OneLeafStar,
  MultiLeafStar,
  NonStarTree,
};

struct PreparedTreePairCase {
  int left_structure = 0;
  int right_structure = 0;
  int node_count = 0;
  int support_size = 0;
  bool is_star = false;
  PairCategory category = PairCategory::OneNode;
  std::vector<double> ordered_overlap_storage;
  std::vector<double> ordered_one_electron_storage;
  std::vector<double> ordered_packed_two_electron;
  std::vector<ComponentData> ordered_components;
  ComponentTree tree;
};

struct ErrorSummary {
  double max_overlap_abs_error = 0.0;
  double max_one_electron_abs_error = 0.0;
  double max_same_spin_alpha_abs_error = 0.0;
  double max_same_spin_beta_abs_error = 0.0;
  double max_opposite_spin_abs_error = 0.0;
  double max_total_two_electron_abs_error = 0.0;
  double max_total_electronic_abs_error = 0.0;
};

struct KernelBenchmarkSummary {
  double wall_time_seconds = 0.0;
  double checksum = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::vector<ComponentTreeHamiltonianResult> first_pass_results;
};

struct OverlapBenchmarkSummary {
  double wall_time_seconds = 0.0;
  double checksum = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::vector<ComponentTreeOverlapResult> first_pass_results;
};

struct RootedOneElectronBenchmarkSummary {
  double wall_time_seconds = 0.0;
  double checksum = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::vector<ComponentTreeOneElectronResult> first_pass_results;
};

struct OneElectronBenchmarkSummary {
  double wall_time_seconds = 0.0;
  double checksum = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

struct TwoElectronBenchmarkSummary {
  double wall_time_seconds = 0.0;
  double checksum = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

void print_usage() {
  std::cerr << "usage: benchmark_rooted_component_tree_pairs <input.xmi>"
               " [--max-pairs N]"
               " [--repeat N]"
               " [--report-every N]"
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
    if (argument_name == "--repeat") {
      options.repeat = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--report-every") {
      options.report_every = std::stoi(argument_value);
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
  if (options.repeat <= 0) {
    throw std::invalid_argument("--repeat must be positive");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
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
  // Both support matrices use the repository-wide column-major layout
  // `storage[column * n + row]`. Rows index bra/right orbitals, columns index
  // ket/left orbitals. The rooted-tree kernel consumes the same convention on
  // its support-local orbital numbering.
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
  // The packed ERI storage keeps the full active-space chemist index symmetry.
  // This helper remaps the global active orbital labels onto the preorder-local
  // support numbering used by the component-tree recurrence.
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

    // The preorder-local orbital numbering is part of the exact kernel
    // contract. Each component stores local pair labels against this support
    // ordering, while vertices inside one component remain in ascending local
    // order to preserve the determinant expansion convention.
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

PairCategory classify_pair_category(const ComponentTree& tree, bool is_star) {
  const int node_count = static_cast<int>(tree.components.size());
  if (node_count <= 1) {
    return PairCategory::OneNode;
  }
  if (is_star && node_count == 2) {
    return PairCategory::OneLeafStar;
  }
  if (is_star) {
    return PairCategory::MultiLeafStar;
  }
  return PairCategory::NonStarTree;
}

const char* pair_category_name(PairCategory category) {
  switch (category) {
    case PairCategory::OneNode:
      return "one_node";
    case PairCategory::OneLeafStar:
      return "one_leaf_star";
    case PairCategory::MultiLeafStar:
      return "multi_leaf_star";
    case PairCategory::NonStarTree:
      return "nonstar_tree";
  }
  return "unknown";
}

void update_error_summary(
    const ComponentTreeHamiltonianResult& exact,
    const ComponentTreeHamiltonianResult& collapsed,
    ErrorSummary* summary) {
  if (summary == nullptr) {
    throw std::invalid_argument("summary must not be null");
  }
  summary->max_overlap_abs_error = std::max(
      summary->max_overlap_abs_error,
      std::abs(collapsed.overlap - exact.overlap));
  summary->max_one_electron_abs_error = std::max(
      summary->max_one_electron_abs_error,
      std::abs(collapsed.one_electron - exact.one_electron));
  summary->max_same_spin_alpha_abs_error = std::max(
      summary->max_same_spin_alpha_abs_error,
      std::abs(
          collapsed.same_spin_alpha_two_electron -
          exact.same_spin_alpha_two_electron));
  summary->max_same_spin_beta_abs_error = std::max(
      summary->max_same_spin_beta_abs_error,
      std::abs(
          collapsed.same_spin_beta_two_electron -
          exact.same_spin_beta_two_electron));
  summary->max_opposite_spin_abs_error = std::max(
      summary->max_opposite_spin_abs_error,
      std::abs(
          collapsed.opposite_spin_two_electron -
          exact.opposite_spin_two_electron));
  summary->max_total_two_electron_abs_error = std::max(
      summary->max_total_two_electron_abs_error,
      std::abs(collapsed.two_electron - exact.two_electron));
  summary->max_total_electronic_abs_error = std::max(
      summary->max_total_electronic_abs_error,
      std::abs(
          collapsed.total_electronic_hamiltonian -
          exact.total_electronic_hamiltonian));
}

template <typename Evaluator>
KernelBenchmarkSummary benchmark_cases(
    const std::vector<PreparedTreePairCase>& cases,
    int repeat,
    const Evaluator& evaluator,
    bool record_first_pass,
    int report_every,
    const char* label) {
  KernelBenchmarkSummary summary;
  if (record_first_pass) {
    summary.first_pass_results.reserve(cases.size());
  }

  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const auto result = evaluator(cases[case_index]);
      summary.checksum +=
          result.total_electronic_hamiltonian +
          0.125 * result.one_electron +
          0.03125 * result.overlap;
      if (repeat_index == 0) {
        summary.subtree_message_state_count += result.subtree_message_state_count;
        summary.subtree_term_pair_count += result.subtree_term_pair_count;
        summary.subdeterminant_evaluations += result.subdeterminant_evaluations;
        summary.dp_transition_count += result.dp_transition_count;
        if (record_first_pass) {
          summary.first_pass_results.push_back(result);
        }
      }
      if (report_every > 0 &&
          repeat_index == 0 &&
          ((static_cast<int>(case_index) + 1) % report_every) == 0) {
        std::cout << label << "_progress_pairs = "
                  << (case_index + 1) << "/" << cases.size() << '\n';
      }
    }
  }
  summary.wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
  return summary;
}

template <typename Evaluator>
OverlapBenchmarkSummary benchmark_overlap_cases(
    const std::vector<PreparedTreePairCase>& cases,
    int repeat,
    const Evaluator& evaluator,
    bool record_first_pass,
    int report_every,
    const char* label) {
  OverlapBenchmarkSummary summary;
  if (record_first_pass) {
    summary.first_pass_results.reserve(cases.size());
  }

  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const ComponentTreeOverlapResult result = evaluator(cases[case_index]);
      summary.checksum += result.overlap;
      if (repeat_index == 0) {
        summary.subtree_message_state_count += result.subtree_message_state_count;
        summary.subtree_term_pair_count += result.subtree_term_pair_count;
        summary.subdeterminant_evaluations += result.subdeterminant_evaluations;
        summary.dp_transition_count += result.dp_transition_count;
        if (record_first_pass) {
          summary.first_pass_results.push_back(result);
        }
      }
      if (report_every > 0 &&
          repeat_index == 0 &&
          ((static_cast<int>(case_index) + 1) % report_every) == 0) {
        std::cout << label << "_progress_pairs = "
                  << (case_index + 1) << "/" << cases.size() << '\n';
      }
    }
  }
  summary.wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
  return summary;
}

template <typename Evaluator>
RootedOneElectronBenchmarkSummary benchmark_rooted_one_electron_cases(
    const std::vector<PreparedTreePairCase>& cases,
    int repeat,
    const Evaluator& evaluator,
    bool record_first_pass,
    int report_every,
    const char* label) {
  RootedOneElectronBenchmarkSummary summary;
  if (record_first_pass) {
    summary.first_pass_results.reserve(cases.size());
  }

  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const ComponentTreeOneElectronResult result = evaluator(cases[case_index]);
      summary.checksum += result.one_electron + 0.125 * result.overlap;
      if (repeat_index == 0) {
        summary.subtree_message_state_count += result.subtree_message_state_count;
        summary.subtree_term_pair_count += result.subtree_term_pair_count;
        summary.subdeterminant_evaluations += result.subdeterminant_evaluations;
        summary.dp_transition_count += result.dp_transition_count;
        if (record_first_pass) {
          summary.first_pass_results.push_back(result);
        }
      }
      if (report_every > 0 &&
          repeat_index == 0 &&
          ((static_cast<int>(case_index) + 1) % report_every) == 0) {
        std::cout << label << "_progress_pairs = "
                  << (case_index + 1) << "/" << cases.size() << '\n';
      }
    }
  }
  summary.wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
  return summary;
}

void print_category_summary(
    PairCategory category,
    const std::vector<PreparedTreePairCase>& cases,
    const KernelBenchmarkSummary& exact_summary,
    const KernelBenchmarkSummary& collapsed_summary,
    const ErrorSummary& error_summary,
    int repeat) {
  if (cases.empty()) {
    return;
  }

  double average_support_size = 0.0;
  double average_node_count = 0.0;
  for (const auto& pair_case : cases) {
    average_support_size += static_cast<double>(pair_case.support_size);
    average_node_count += static_cast<double>(pair_case.node_count);
  }
  average_support_size /= static_cast<double>(cases.size());
  average_node_count /= static_cast<double>(cases.size());

  const double exact_per_repeat_seconds =
      exact_summary.wall_time_seconds / static_cast<double>(repeat);
  const double collapsed_per_repeat_seconds =
      collapsed_summary.wall_time_seconds / static_cast<double>(repeat);
  const double speedup =
      exact_per_repeat_seconds / std::max(1.0e-15, collapsed_per_repeat_seconds);

  std::cout << '[' << pair_category_name(category) << "]\n";
  std::cout << "  pair_count = " << cases.size() << '\n';
  std::cout << "  avg_support_size = " << average_support_size << '\n';
  std::cout << "  avg_node_count = " << average_node_count << '\n';
  std::cout << "  exact_total_seconds = " << exact_summary.wall_time_seconds << '\n';
  std::cout << "  collapsed_total_seconds = "
            << collapsed_summary.wall_time_seconds << '\n';
  std::cout << "  exact_seconds_per_repeat = " << exact_per_repeat_seconds << '\n';
  std::cout << "  collapsed_seconds_per_repeat = "
            << collapsed_per_repeat_seconds << '\n';
  std::cout << "  collapsed_speedup_vs_exact = " << speedup << '\n';
  std::cout << "  collapsed_avg_subtree_message_state_count = "
            << (static_cast<double>(collapsed_summary.subtree_message_state_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  collapsed_avg_subtree_term_pair_count = "
            << (static_cast<double>(collapsed_summary.subtree_term_pair_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  collapsed_avg_subdeterminant_evaluations = "
            << (static_cast<double>(collapsed_summary.subdeterminant_evaluations) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  collapsed_avg_dp_transition_count = "
            << (static_cast<double>(collapsed_summary.dp_transition_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  max_overlap_abs_error = "
            << error_summary.max_overlap_abs_error << '\n';
  std::cout << "  max_one_electron_abs_error = "
            << error_summary.max_one_electron_abs_error << '\n';
  std::cout << "  max_same_spin_alpha_abs_error = "
            << error_summary.max_same_spin_alpha_abs_error << '\n';
  std::cout << "  max_same_spin_beta_abs_error = "
            << error_summary.max_same_spin_beta_abs_error << '\n';
  std::cout << "  max_opposite_spin_abs_error = "
            << error_summary.max_opposite_spin_abs_error << '\n';
  std::cout << "  max_total_two_electron_abs_error = "
            << error_summary.max_total_two_electron_abs_error << '\n';
  std::cout << "  max_total_electronic_abs_error = "
            << error_summary.max_total_electronic_abs_error << '\n';
  std::cout << "  exact_checksum = " << exact_summary.checksum << '\n';
  std::cout << "  collapsed_checksum = " << collapsed_summary.checksum << '\n';
}

void print_one_leaf_generic_bundle_summary(
    const std::vector<PreparedTreePairCase>& cases,
    const KernelBenchmarkSummary& exact_summary,
    const KernelBenchmarkSummary& boundary_bundle_summary,
    const ErrorSummary& error_summary,
    int repeat) {
  if (cases.empty()) {
    return;
  }

  const double exact_per_repeat_seconds =
      exact_summary.wall_time_seconds / static_cast<double>(repeat);
  const double boundary_per_repeat_seconds =
      boundary_bundle_summary.wall_time_seconds / static_cast<double>(repeat);

  std::cout << "[one_leaf_star_generic_bundle]\n";
  std::cout << "  pair_count = " << cases.size() << '\n';
  std::cout << "  exact_total_seconds = " << exact_summary.wall_time_seconds << '\n';
  std::cout << "  boundary_total_seconds = "
            << boundary_bundle_summary.wall_time_seconds << '\n';
  std::cout << "  exact_seconds_per_repeat = " << exact_per_repeat_seconds << '\n';
  std::cout << "  boundary_seconds_per_repeat = " << boundary_per_repeat_seconds << '\n';
  std::cout << "  boundary_speedup_vs_exact = "
            << (exact_per_repeat_seconds / std::max(1.0e-15, boundary_per_repeat_seconds))
            << '\n';
  std::cout << "  boundary_avg_subtree_message_state_count = "
            << (static_cast<double>(boundary_bundle_summary.subtree_message_state_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_subtree_term_pair_count = "
            << (static_cast<double>(boundary_bundle_summary.subtree_term_pair_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_subdeterminant_evaluations = "
            << (static_cast<double>(boundary_bundle_summary.subdeterminant_evaluations) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_dp_transition_count = "
            << (static_cast<double>(boundary_bundle_summary.dp_transition_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  max_overlap_abs_error = "
            << error_summary.max_overlap_abs_error << '\n';
  std::cout << "  max_one_electron_abs_error = "
            << error_summary.max_one_electron_abs_error << '\n';
  std::cout << "  max_same_spin_alpha_abs_error = "
            << error_summary.max_same_spin_alpha_abs_error << '\n';
  std::cout << "  max_same_spin_beta_abs_error = "
            << error_summary.max_same_spin_beta_abs_error << '\n';
  std::cout << "  max_opposite_spin_abs_error = "
            << error_summary.max_opposite_spin_abs_error << '\n';
  std::cout << "  max_total_two_electron_abs_error = "
            << error_summary.max_total_two_electron_abs_error << '\n';
  std::cout << "  max_total_electronic_abs_error = "
            << error_summary.max_total_electronic_abs_error << '\n';
  std::cout << "  exact_checksum = " << exact_summary.checksum << '\n';
  std::cout << "  boundary_checksum = " << boundary_bundle_summary.checksum << '\n';
}

void print_overlap_summary(
    const std::vector<PreparedTreePairCase>& cases,
    const OverlapBenchmarkSummary& exact_summary,
    const OverlapBenchmarkSummary& boundary_summary,
    int repeat) {
  if (cases.empty()) {
    return;
  }

  double max_overlap_abs_error = 0.0;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    max_overlap_abs_error = std::max(
        max_overlap_abs_error,
        std::abs(
            boundary_summary.first_pass_results[index].overlap -
            exact_summary.first_pass_results[index].overlap));
  }

  const double exact_per_repeat_seconds =
      exact_summary.wall_time_seconds / static_cast<double>(repeat);
  const double boundary_per_repeat_seconds =
      boundary_summary.wall_time_seconds / static_cast<double>(repeat);

  std::cout << "[nonstar_tree_overlap_only]\n";
  std::cout << "  pair_count = " << cases.size() << '\n';
  std::cout << "  exact_total_seconds = " << exact_summary.wall_time_seconds << '\n';
  std::cout << "  boundary_total_seconds = " << boundary_summary.wall_time_seconds << '\n';
  std::cout << "  exact_seconds_per_repeat = " << exact_per_repeat_seconds << '\n';
  std::cout << "  boundary_seconds_per_repeat = " << boundary_per_repeat_seconds << '\n';
  std::cout << "  boundary_speedup_vs_exact = "
            << (exact_per_repeat_seconds /
                std::max(1.0e-15, boundary_per_repeat_seconds))
            << '\n';
  std::cout << "  boundary_avg_message_state_count = "
            << (static_cast<double>(boundary_summary.subtree_message_state_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_term_pair_count = "
            << (static_cast<double>(boundary_summary.subtree_term_pair_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_subdeterminant_evaluations = "
            << (static_cast<double>(boundary_summary.subdeterminant_evaluations) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_dp_transition_count = "
            << (static_cast<double>(boundary_summary.dp_transition_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  max_overlap_abs_error = " << max_overlap_abs_error << '\n';
  std::cout << "  exact_checksum = " << exact_summary.checksum << '\n';
  std::cout << "  boundary_checksum = " << boundary_summary.checksum << '\n';
}

void print_rooted_one_electron_summary(
    const std::vector<PreparedTreePairCase>& cases,
    const RootedOneElectronBenchmarkSummary& exact_summary,
    const RootedOneElectronBenchmarkSummary& boundary_summary,
    int repeat) {
  if (cases.empty()) {
    return;
  }

  double max_overlap_abs_error = 0.0;
  double max_one_electron_abs_error = 0.0;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    max_overlap_abs_error = std::max(
        max_overlap_abs_error,
        std::abs(
            boundary_summary.first_pass_results[index].overlap -
            exact_summary.first_pass_results[index].overlap));
    max_one_electron_abs_error = std::max(
        max_one_electron_abs_error,
        std::abs(
            boundary_summary.first_pass_results[index].one_electron -
            exact_summary.first_pass_results[index].one_electron));
  }

  const double exact_per_repeat_seconds =
      exact_summary.wall_time_seconds / static_cast<double>(repeat);
  const double boundary_per_repeat_seconds =
      boundary_summary.wall_time_seconds / static_cast<double>(repeat);

  std::cout << "[nonstar_tree_one_electron_only]\n";
  std::cout << "  pair_count = " << cases.size() << '\n';
  std::cout << "  exact_total_seconds = " << exact_summary.wall_time_seconds << '\n';
  std::cout << "  boundary_total_seconds = " << boundary_summary.wall_time_seconds << '\n';
  std::cout << "  exact_seconds_per_repeat = " << exact_per_repeat_seconds << '\n';
  std::cout << "  boundary_seconds_per_repeat = " << boundary_per_repeat_seconds << '\n';
  std::cout << "  boundary_speedup_vs_exact = "
            << (exact_per_repeat_seconds /
                std::max(1.0e-15, boundary_per_repeat_seconds))
            << '\n';
  std::cout << "  boundary_avg_message_state_count = "
            << (static_cast<double>(boundary_summary.subtree_message_state_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_term_pair_count = "
            << (static_cast<double>(boundary_summary.subtree_term_pair_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_subdeterminant_evaluations = "
            << (static_cast<double>(boundary_summary.subdeterminant_evaluations) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  boundary_avg_dp_transition_count = "
            << (static_cast<double>(boundary_summary.dp_transition_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  max_overlap_abs_error = " << max_overlap_abs_error << '\n';
  std::cout << "  max_one_electron_abs_error = " << max_one_electron_abs_error << '\n';
  std::cout << "  exact_checksum = " << exact_summary.checksum << '\n';
  std::cout << "  boundary_checksum = " << boundary_summary.checksum << '\n';
}

OneElectronBenchmarkSummary benchmark_one_leaf_one_electron_collapsed(
    const std::vector<PreparedTreePairCase>& cases,
    int repeat,
    const DeterminantOverlapResolver& overlap_resolver,
    int report_every) {
  OneElectronBenchmarkSummary summary;
  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const auto result =
          xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_one_electron_collapsed(
              cases[case_index].ordered_overlap_storage,
              cases[case_index].ordered_one_electron_storage,
              cases[case_index].support_size,
              cases[case_index].ordered_components,
              overlap_resolver);
      summary.checksum += result.one_electron + 0.125 * result.overlap;
      if (repeat_index == 0) {
        summary.collapsed_leaf_state_count += result.collapsed_leaf_state_count;
        summary.hypercube_assignment_count += result.hypercube_assignment_count;
        summary.subdeterminant_evaluations += result.subdeterminant_evaluations;
        summary.dp_transition_count += result.dp_transition_count;
      }
      if (report_every > 0 &&
          repeat_index == 0 &&
          ((static_cast<int>(case_index) + 1) % report_every) == 0) {
        std::cout << "one_leaf_star_one_electron_collapsed_progress_pairs = "
                  << (case_index + 1) << "/" << cases.size() << '\n';
      }
    }
  }
  summary.wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
  return summary;
}

TwoElectronBenchmarkSummary benchmark_one_leaf_two_electron_collapsed(
    const std::vector<PreparedTreePairCase>& cases,
    int repeat,
    const DeterminantOverlapResolver& overlap_resolver,
    int report_every) {
  TwoElectronBenchmarkSummary summary;
  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const auto result =
          xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_collapsed(
              cases[case_index].ordered_overlap_storage,
              cases[case_index].ordered_packed_two_electron,
              cases[case_index].support_size,
              cases[case_index].ordered_components,
              overlap_resolver);
      summary.checksum += result.two_electron + 0.125 * result.overlap;
      if (repeat_index == 0) {
        summary.collapsed_leaf_state_count += result.collapsed_leaf_state_count;
        summary.hypercube_assignment_count += result.hypercube_assignment_count;
        summary.subdeterminant_evaluations += result.subdeterminant_evaluations;
        summary.dp_transition_count += result.dp_transition_count;
      }
      if (report_every > 0 &&
          repeat_index == 0 &&
          ((static_cast<int>(case_index) + 1) % report_every) == 0) {
        std::cout << "one_leaf_star_two_electron_collapsed_progress_pairs = "
                  << (case_index + 1) << "/" << cases.size() << '\n';
      }
    }
  }
  summary.wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
  return summary;
}

double benchmark_one_leaf_two_electron_exact(
    const std::vector<PreparedTreePairCase>& cases,
    int repeat,
    const DeterminantOverlapResolver& overlap_resolver,
    int report_every,
    double* checksum) {
  if (checksum == nullptr) {
    throw std::invalid_argument("checksum must not be null");
  }
  *checksum = 0.0;
  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const ExactTwoElectronStarPairStats result =
          xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_two_electron_exact(
              cases[case_index].ordered_overlap_storage,
              cases[case_index].ordered_packed_two_electron,
              cases[case_index].support_size,
              cases[case_index].ordered_components,
              overlap_resolver);
      *checksum += result.exact_two_electron + 0.125 * result.exact_overlap;
      if (report_every > 0 &&
          repeat_index == 0 &&
          ((static_cast<int>(case_index) + 1) % report_every) == 0) {
        std::cout << "one_leaf_star_two_electron_exact_progress_pairs = "
                  << (case_index + 1) << "/" << cases.size() << '\n';
      }
    }
  }
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
}

void print_one_leaf_split_summary(
    const std::vector<PreparedTreePairCase>& cases,
    const OneElectronBenchmarkSummary& one_electron_collapsed,
    const TwoElectronBenchmarkSummary& two_electron_collapsed,
    double two_electron_exact_seconds,
    double two_electron_exact_checksum,
    int repeat) {
  if (cases.empty()) {
    return;
  }

  std::cout << "[one_leaf_star_split]\n";
  std::cout << "  pair_count = " << cases.size() << '\n';
  std::cout << "  one_electron_collapsed_total_seconds = "
            << one_electron_collapsed.wall_time_seconds << '\n';
  std::cout << "  one_electron_collapsed_seconds_per_repeat = "
            << (one_electron_collapsed.wall_time_seconds /
                static_cast<double>(repeat))
            << '\n';
  std::cout << "  one_electron_collapsed_avg_leaf_state_count = "
            << (static_cast<double>(one_electron_collapsed.collapsed_leaf_state_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  one_electron_collapsed_avg_subdeterminant_evaluations = "
            << (static_cast<double>(one_electron_collapsed.subdeterminant_evaluations) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  one_electron_collapsed_avg_dp_transition_count = "
            << (static_cast<double>(one_electron_collapsed.dp_transition_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  one_electron_collapsed_checksum = "
            << one_electron_collapsed.checksum << '\n';

  std::cout << "  two_electron_exact_total_seconds = "
            << two_electron_exact_seconds << '\n';
  std::cout << "  two_electron_exact_seconds_per_repeat = "
            << (two_electron_exact_seconds / static_cast<double>(repeat)) << '\n';
  std::cout << "  two_electron_exact_checksum = "
            << two_electron_exact_checksum << '\n';

  std::cout << "  two_electron_collapsed_total_seconds = "
            << two_electron_collapsed.wall_time_seconds << '\n';
  std::cout << "  two_electron_collapsed_seconds_per_repeat = "
            << (two_electron_collapsed.wall_time_seconds /
                static_cast<double>(repeat))
            << '\n';
  std::cout << "  two_electron_collapsed_speedup_vs_exact = "
            << ((two_electron_exact_seconds / static_cast<double>(repeat)) /
                std::max(
                    1.0e-15,
                    two_electron_collapsed.wall_time_seconds /
                        static_cast<double>(repeat)))
            << '\n';
  std::cout << "  two_electron_collapsed_avg_leaf_state_count = "
            << (static_cast<double>(two_electron_collapsed.collapsed_leaf_state_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  two_electron_collapsed_avg_subdeterminant_evaluations = "
            << (static_cast<double>(two_electron_collapsed.subdeterminant_evaluations) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  two_electron_collapsed_avg_dp_transition_count = "
            << (static_cast<double>(two_electron_collapsed.dp_transition_count) /
                static_cast<double>(cases.size()))
            << '\n';
  std::cout << "  two_electron_collapsed_checksum = "
            << two_electron_collapsed.checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto benchmark_started_at = std::chrono::steady_clock::now();
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "benchmark_rooted_component_tree_pairs currently supports only singlet closed-shell structures");
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
          "benchmark_rooted_component_tree_pairs requires exact packed ERIs");
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

    int connected_tree_pairs = 0;
    int star_tree_pairs = 0;
    int nonstar_tree_pairs = 0;
    std::vector<PreparedTreePairCase> cases;
    cases.reserve(pair_list.size());

    const auto preparation_started_at = std::chrono::steady_clock::now();
    for (const auto& [left_structure, right_structure] : pair_list) {
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
      const bool is_star = is_connected_star_graph(metric_graph);
      if (is_star) {
        ++star_tree_pairs;
      } else {
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

      PreparedTreePairCase pair_case;
      pair_case.left_structure = left_structure;
      pair_case.right_structure = right_structure;
      pair_case.node_count = static_cast<int>(ordered_union_components.size());
      pair_case.support_size = static_cast<int>(layout.ordered_support_orbitals.size());
      pair_case.is_star = is_star;
      pair_case.tree = build_component_tree(layout, ordered_union_components);
      pair_case.ordered_components = pair_case.tree.components;
      pair_case.category = classify_pair_category(pair_case.tree, is_star);
      pair_case.ordered_overlap_storage = flatten_column_major_matrix(
          xmvb::vb::build_support_overlap_matrix(
              layout.ordered_support_orbitals,
              active_overlap_storage,
              n_active_orbitals));
      pair_case.ordered_one_electron_storage = flatten_column_major_matrix(
          build_support_submatrix(
              layout.ordered_support_orbitals,
              active_one_electron_storage,
              n_active_orbitals));
      pair_case.ordered_packed_two_electron = build_support_local_packed_two_electron_integrals(
          layout.ordered_support_orbitals,
          packed_active_two_electron_integrals);
      cases.push_back(std::move(pair_case));
    }
    const double preparation_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - preparation_started_at).count();

    if (cases.empty()) {
      throw std::runtime_error("no connected tree-like metric graph pairs were found");
    }

    std::vector<PreparedTreePairCase> one_node_cases;
    std::vector<PreparedTreePairCase> one_leaf_star_cases;
    std::vector<PreparedTreePairCase> multi_leaf_star_cases;
    std::vector<PreparedTreePairCase> nonstar_cases;
    for (const auto& pair_case : cases) {
      switch (pair_case.category) {
        case PairCategory::OneNode:
          one_node_cases.push_back(pair_case);
          break;
        case PairCategory::OneLeafStar:
          one_leaf_star_cases.push_back(pair_case);
          break;
        case PairCategory::MultiLeafStar:
          multi_leaf_star_cases.push_back(pair_case);
          break;
        case PairCategory::NonStarTree:
          nonstar_cases.push_back(pair_case);
          break;
      }
    }

    const auto exact_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian_exact(
              pair_case.ordered_overlap_storage,
              pair_case.ordered_one_electron_storage,
              pair_case.ordered_packed_two_electron,
              pair_case.support_size,
              pair_case.tree,
              overlap_resolver);
        };
    const auto collapsed_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian_collapsed(
              pair_case.ordered_overlap_storage,
              pair_case.ordered_one_electron_storage,
              pair_case.ordered_packed_two_electron,
              pair_case.support_size,
              pair_case.tree,
              overlap_resolver);
        };
    const auto generic_bundle_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::
              evaluate_rooted_component_tree_hamiltonian_bundle_boundary_collapsed(
                  pair_case.ordered_overlap_storage,
                  pair_case.ordered_one_electron_storage,
                  pair_case.ordered_packed_two_electron,
                  pair_case.support_size,
                  pair_case.tree,
                  overlap_resolver);
        };
    const auto exact_overlap_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::evaluate_rooted_component_tree_overlap_exact(
              pair_case.ordered_overlap_storage,
              pair_case.support_size,
              pair_case.tree,
              overlap_resolver);
        };
    const auto boundary_overlap_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::evaluate_rooted_component_tree_overlap_boundary_collapsed(
              pair_case.ordered_overlap_storage,
              pair_case.support_size,
              pair_case.tree,
              overlap_resolver);
        };
    const auto exact_one_electron_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::evaluate_rooted_component_tree_one_electron_exact(
              pair_case.ordered_overlap_storage,
              pair_case.ordered_one_electron_storage,
              pair_case.support_size,
              pair_case.tree,
              overlap_resolver);
        };
    const auto boundary_one_electron_evaluator =
        [&](const PreparedTreePairCase& pair_case) {
          return xmvb::vb::exact_separator::evaluate_rooted_component_tree_one_electron_boundary_collapsed(
              pair_case.ordered_overlap_storage,
              pair_case.ordered_one_electron_storage,
              pair_case.support_size,
              pair_case.tree,
              overlap_resolver);
        };

    std::cout << std::setprecision(15);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "total_pairs = " << pair_list.size() << '\n';
    std::cout << "connected_tree_pairs = " << connected_tree_pairs << '\n';
    std::cout << "star_tree_pairs = " << star_tree_pairs << '\n';
    std::cout << "nonstar_tree_pairs = " << nonstar_tree_pairs << '\n';
    std::cout << "one_node_pairs = " << one_node_cases.size() << '\n';
    std::cout << "one_leaf_star_pairs = " << one_leaf_star_cases.size() << '\n';
    std::cout << "multi_leaf_star_pairs = " << multi_leaf_star_cases.size() << '\n';
    std::cout << "repeat = " << options.repeat << '\n';
    std::cout << "edge_threshold = " << options.edge_threshold << '\n';
    std::cout << "pair_preparation_seconds = " << preparation_seconds << '\n';

    const auto run_category =
        [&](PairCategory category,
            const std::vector<PreparedTreePairCase>& category_cases) {
          if (category_cases.empty()) {
            return;
          }

          const std::string exact_label =
              std::string(pair_category_name(category)) + "_exact";
          const std::string collapsed_label =
              std::string(pair_category_name(category)) + "_collapsed";
          const KernelBenchmarkSummary exact_summary = benchmark_cases(
              category_cases,
              options.repeat,
              exact_evaluator,
              true,
              options.report_every,
              exact_label.c_str());
          const KernelBenchmarkSummary collapsed_summary = benchmark_cases(
              category_cases,
              options.repeat,
              collapsed_evaluator,
              true,
              options.report_every,
              collapsed_label.c_str());

          ErrorSummary error_summary;
          for (std::size_t index = 0; index < category_cases.size(); ++index) {
            update_error_summary(
                exact_summary.first_pass_results[index],
                collapsed_summary.first_pass_results[index],
                &error_summary);
          }
          print_category_summary(
              category,
              category_cases,
              exact_summary,
              collapsed_summary,
              error_summary,
              options.repeat);
        };

    run_category(PairCategory::OneNode, one_node_cases);
    run_category(PairCategory::OneLeafStar, one_leaf_star_cases);
    run_category(PairCategory::MultiLeafStar, multi_leaf_star_cases);
    run_category(PairCategory::NonStarTree, nonstar_cases);

    if (!one_leaf_star_cases.empty()) {
      const KernelBenchmarkSummary one_leaf_generic_bundle_summary =
          benchmark_cases(
              one_leaf_star_cases,
              options.repeat,
              generic_bundle_evaluator,
              true,
              options.report_every,
              "one_leaf_star_generic_bundle");
      ErrorSummary one_leaf_generic_bundle_error_summary;
      for (std::size_t index = 0; index < one_leaf_star_cases.size(); ++index) {
        update_error_summary(
            exact_evaluator(one_leaf_star_cases[index]),
            one_leaf_generic_bundle_summary.first_pass_results[index],
            &one_leaf_generic_bundle_error_summary);
      }
      const KernelBenchmarkSummary one_leaf_exact_summary = benchmark_cases(
          one_leaf_star_cases,
          options.repeat,
          exact_evaluator,
          false,
          options.report_every,
          "one_leaf_star_exact_replay");
      print_one_leaf_generic_bundle_summary(
          one_leaf_star_cases,
          one_leaf_exact_summary,
          one_leaf_generic_bundle_summary,
          one_leaf_generic_bundle_error_summary,
          options.repeat);
    }

    if (!nonstar_cases.empty()) {
      const OverlapBenchmarkSummary exact_overlap_summary =
          benchmark_overlap_cases(
              nonstar_cases,
              options.repeat,
              exact_overlap_evaluator,
              true,
              options.report_every,
              "nonstar_tree_overlap_exact");
      const OverlapBenchmarkSummary boundary_overlap_summary =
          benchmark_overlap_cases(
              nonstar_cases,
              options.repeat,
              boundary_overlap_evaluator,
              true,
              options.report_every,
              "nonstar_tree_overlap_boundary");
      print_overlap_summary(
          nonstar_cases,
          exact_overlap_summary,
          boundary_overlap_summary,
          options.repeat);

      const RootedOneElectronBenchmarkSummary exact_one_electron_summary =
          benchmark_rooted_one_electron_cases(
              nonstar_cases,
              options.repeat,
              exact_one_electron_evaluator,
              true,
              options.report_every,
              "nonstar_tree_one_electron_exact");
      const RootedOneElectronBenchmarkSummary boundary_one_electron_summary =
          benchmark_rooted_one_electron_cases(
              nonstar_cases,
              options.repeat,
              boundary_one_electron_evaluator,
              true,
              options.report_every,
              "nonstar_tree_one_electron_boundary");
      print_rooted_one_electron_summary(
          nonstar_cases,
          exact_one_electron_summary,
          boundary_one_electron_summary,
          options.repeat);
    }

    if (!one_leaf_star_cases.empty()) {
      const OneElectronBenchmarkSummary one_electron_collapsed =
          benchmark_one_leaf_one_electron_collapsed(
              one_leaf_star_cases,
              options.repeat,
              overlap_resolver,
              options.report_every);
      double two_electron_exact_checksum = 0.0;
      const double two_electron_exact_seconds = benchmark_one_leaf_two_electron_exact(
          one_leaf_star_cases,
          options.repeat,
          overlap_resolver,
          options.report_every,
          &two_electron_exact_checksum);
      const TwoElectronBenchmarkSummary two_electron_collapsed =
          benchmark_one_leaf_two_electron_collapsed(
              one_leaf_star_cases,
              options.repeat,
              overlap_resolver,
              options.report_every);
      print_one_leaf_split_summary(
          one_leaf_star_cases,
          one_electron_collapsed,
          two_electron_collapsed,
          two_electron_exact_seconds,
          two_electron_exact_checksum,
          options.repeat);
    }

    const double total_wall_time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - benchmark_started_at).count();
    std::cout << "total_wall_time_seconds = " << total_wall_time_seconds << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
