#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/union_graph_screening.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using Pair = std::pair<int, int>;
using OrbitalPair = xmvb::vb::OrbitalPair;
using CanonicalDeterminantKey = std::pair<std::vector<int>, std::vector<int>>;
using FullDeterminantPairKey =
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>;
using ZeroedBlockDeterminantKey = std::tuple<
    std::vector<int>,
    std::vector<int>,
    std::vector<int>,
    std::vector<int>>;

enum class PairOrder {
  Lexicographic,
  Random,
};

enum class ActiveOverlapSource {
  Input,
  OptimizedVbscf,
};

struct Options {
  std::string input_path;
  PairOrder pair_order = PairOrder::Lexicographic;
  ActiveOverlapSource active_overlap_source = ActiveOverlapSource::Input;
  int filter_left_structure = -1;
  int filter_right_structure = -1;
  int max_pairs = 0;
  int report_every = 0;
  int top_examples = 12;
  int optimizer_max_iterations = 25;
  std::uint32_t seed = 0;
  double optimizer_gradient_tolerance = 2.0e-3;
  double optimizer_energy_tolerance = 1.0e-7;
  double singular_value_threshold = 1.0e-8;
  double edge_max_abs_threshold = 0.0;
  double tolerance = 1.0e-12;
};

struct OptimizedOverlapSummary {
  bool converged = false;
  std::string termination_reason;
  int accepted_iterations = 0;
  int objective_evaluations = 0;
  double initial_total_energy = 0.0;
  double final_total_energy = 0.0;
  double final_gradient_inf_norm = 0.0;
  double final_gradient_l2_norm = 0.0;
  double total_wall_time_seconds = 0.0;
};

struct ActiveOverlapSelectionResult {
  std::vector<double> active_overlap_matrix;
  std::optional<OptimizedOverlapSummary> optimized_overlap_summary;
};

struct PerStructureCache {
  std::vector<OrbitalPair> active_pairs;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> legacy_terms;
  std::map<CanonicalDeterminantKey, double> coefficient_lookup;
};

struct OrientationTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

struct ComponentData {
  int graph_node = -1;
  std::vector<OrbitalPair> left_pairs;
  std::vector<OrbitalPair> right_pairs;
  std::vector<OrientationTerm> left_orientation_terms;
  std::vector<OrientationTerm> right_orientation_terms;
};

struct LeafMessageEntry {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  double value = 0.0;
  std::vector<int> alpha_block_rows;
  std::vector<int> alpha_block_cols;
  std::vector<int> beta_block_rows;
  std::vector<int> beta_block_cols;
  std::vector<int> left_alpha_occ;
  std::vector<int> left_beta_occ;
  std::vector<int> right_alpha_occ;
  std::vector<int> right_beta_occ;
  double left_term_coefficient = 0.0;
  double right_term_coefficient = 0.0;
};

struct CollapsedLeafMessageEntry {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  double value = 0.0;
};

struct ExactSeparatorLeafStateKey {
  std::vector<OrbitalPair> left_pairs;
  std::vector<OrbitalPair> right_pairs;
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
};

bool operator<(
    const ExactSeparatorLeafStateKey& left,
    const ExactSeparatorLeafStateKey& right) {
  return std::tie(
             left.left_pairs,
             left.right_pairs,
             left.alpha_row_mask,
             left.alpha_col_mask,
             left.beta_row_mask,
             left.beta_col_mask) <
      std::tie(
             right.left_pairs,
             right.right_pairs,
             right.alpha_row_mask,
             right.alpha_col_mask,
             right.beta_row_mask,
             right.beta_col_mask);
}

struct ExactSeparatorStateKey {
  std::vector<OrbitalPair> root_left_pairs;
  std::vector<OrbitalPair> root_right_pairs;
  std::vector<int> left_root_alpha_occ;
  std::vector<int> left_root_beta_occ;
  std::vector<int> right_root_alpha_occ;
  std::vector<int> right_root_beta_occ;
  std::vector<ExactSeparatorLeafStateKey> leaf_states;
};

bool operator<(
    const ExactSeparatorStateKey& left,
    const ExactSeparatorStateKey& right) {
  return std::tie(
             left.root_left_pairs,
             left.root_right_pairs,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.leaf_states) <
      std::tie(
             right.root_left_pairs,
             right.root_right_pairs,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.leaf_states);
}

struct ExactSeparatorStateCollector {
  std::set<ExactSeparatorStateKey> unique_exact_separator_states;
  struct ExactLeafMessageStateKey {
    std::vector<OrbitalPair> leaf_left_pairs;
    std::vector<OrbitalPair> leaf_right_pairs;
    std::vector<int> left_root_alpha_occ;
    std::vector<int> left_root_beta_occ;
    std::vector<int> right_root_alpha_occ;
    std::vector<int> right_root_beta_occ;
    std::uint32_t alpha_row_mask = 0;
    std::uint32_t alpha_col_mask = 0;
    std::uint32_t beta_row_mask = 0;
    std::uint32_t beta_col_mask = 0;
  };

  struct ExactLeafMessageBundleKey {
    std::vector<OrbitalPair> leaf_left_pairs;
    std::vector<OrbitalPair> leaf_right_pairs;
    std::vector<int> left_root_alpha_occ;
    std::vector<int> left_root_beta_occ;
    std::vector<int> right_root_alpha_occ;
    std::vector<int> right_root_beta_occ;
  };

  struct ExactMergeStateKey {
    std::vector<int> left_root_alpha_occ;
    std::vector<int> left_root_beta_occ;
    std::vector<int> right_root_alpha_occ;
    std::vector<int> right_root_beta_occ;
    int leaf_index = 0;
    std::uint32_t used_alpha_row_mask = 0;
    std::uint32_t used_alpha_col_mask = 0;
    std::uint32_t used_beta_row_mask = 0;
    std::uint32_t used_beta_col_mask = 0;
  };

  std::set<ExactLeafMessageStateKey> unique_leaf_message_states;
  std::set<ExactLeafMessageBundleKey> unique_leaf_message_bundles;
  std::set<ExactMergeStateKey> unique_merge_states;
};

bool operator<(
    const ExactSeparatorStateCollector::ExactLeafMessageStateKey& left,
    const ExactSeparatorStateCollector::ExactLeafMessageStateKey& right) {
  return std::tie(
             left.leaf_left_pairs,
             left.leaf_right_pairs,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.alpha_row_mask,
             left.alpha_col_mask,
             left.beta_row_mask,
             left.beta_col_mask) <
      std::tie(
             right.leaf_left_pairs,
             right.leaf_right_pairs,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.alpha_row_mask,
             right.alpha_col_mask,
             right.beta_row_mask,
             right.beta_col_mask);
}

bool operator<(
    const ExactSeparatorStateCollector::ExactMergeStateKey& left,
    const ExactSeparatorStateCollector::ExactMergeStateKey& right) {
  return std::tie(
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.leaf_index,
             left.used_alpha_row_mask,
             left.used_alpha_col_mask,
             left.used_beta_row_mask,
             left.used_beta_col_mask) <
      std::tie(
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.leaf_index,
             right.used_alpha_row_mask,
             right.used_alpha_col_mask,
             right.used_beta_row_mask,
             right.used_beta_col_mask);
}

bool operator<(
    const ExactSeparatorStateCollector::ExactLeafMessageBundleKey& left,
    const ExactSeparatorStateCollector::ExactLeafMessageBundleKey& right) {
  return std::tie(
             left.leaf_left_pairs,
             left.leaf_right_pairs,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ) <
      std::tie(
             right.leaf_left_pairs,
             right.leaf_right_pairs,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ);
}

struct SpinMaskDeterminantEntry {
  std::uint32_t row_mask = 0;
  std::uint32_t col_mask = 0;
  double value = 0.0;
};

struct StarPairStats {
  bool covered = false;
  double exact_overlap = 0.0;
  double star_overlap = 0.0;
  double absolute_error = 0.0;
  std::uint64_t reference_determinant_pair_count = 0;
  std::uint64_t local_term_pair_visits = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::uint64_t coefficient_factorization_checks = 0;
  std::uint64_t coefficient_factorization_mismatches = 0;
  double max_coefficient_factorization_abs_error = 0.0;
  int metric_width_upper_bound = 0;
  int node_count = 0;
  int root_node = -1;
};

struct CollapsedStarPairStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double absolute_error = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t schur_fast_spin_assignment_count = 0;
  std::uint64_t rectangular_fallback_spin_assignment_count = 0;
  std::uint64_t singular_fallback_spin_assignment_count = 0;
  std::uint64_t schur_fast_subdeterminant_evaluation_count = 0;
  std::uint64_t rectangular_fallback_subdeterminant_evaluation_count = 0;
  std::uint64_t singular_fallback_subdeterminant_evaluation_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

struct PairExample {
  int left_structure = 0;
  int right_structure = 0;
  int node_count = 0;
  int root_node = -1;
  int metric_width_upper_bound = 0;
  std::uint64_t reference_determinant_pair_count = 0;
  std::uint64_t local_term_pair_visits = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  double exact_overlap = 0.0;
  double star_overlap = 0.0;
  double absolute_error = 0.0;
};

struct ExactWorkCollector {
  // `reference_full_determinant_pairs` stores canonical raw-VB determinant-pair
  // contributions exactly as the legacy overlap code would visit them:
  //   (left alpha occ, left beta occ, right alpha occ, right beta occ).
  // `reference_spin_determinants` stores the alpha/beta determinant subproblems
  // that appear in those raw determinant pairs.
  // `separator_root_spin_determinants` stores ordinary root-remainder minors
  // visited by the separator recurrence, and
  // `separator_zeroed_block_spin_determinants` stores leaf/root minors where
  // the selected root-root overlap block has been zeroed.
  std::set<FullDeterminantPairKey> reference_full_determinant_pairs;
  std::set<CanonicalDeterminantKey> reference_spin_determinants;
  std::set<CanonicalDeterminantKey> separator_root_spin_determinants;
  std::set<ZeroedBlockDeterminantKey> separator_zeroed_block_spin_determinants;
};

std::vector<double> flatten_column_major_matrix(const Matrix& matrix);

std::vector<int> remap_occ_to_global_labels(
    const std::vector<int>& local_occ,
    const std::vector<int>& local_to_global_orbitals);

std::vector<OrbitalPair> remap_pairs_to_global_labels(
    const std::vector<OrbitalPair>& local_pairs,
    const std::vector<int>& local_to_global_orbitals);

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes);

std::vector<CollapsedLeafMessageEntry> build_collapsed_leaf_messages_component_ordered(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    CollapsedStarPairStats* stats,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* hypercube_assignment_count);

void print_usage() {
  std::cerr << "usage: analyze_star_separator_overlap_dataset <input.xmi>"
               " [--pair-order lexicographic|random]"
               " [--active-overlap-source input|optimized_vbscf]"
               " [--left-structure I --right-structure J]"
               " [--max-pairs N]"
               " [--seed S]"
               " [--report-every N]"
               " [--top-examples N]"
               " [--optimizer-max-iterations N]"
               " [--optimizer-gradient-tolerance F]"
               " [--optimizer-energy-tolerance F]"
               " [--singular-value-threshold F]"
               " [--edge-max-abs-threshold F]"
               " [--tolerance F]\n";
}

PairOrder parse_pair_order(const std::string& value) {
  if (value == "lexicographic") {
    return PairOrder::Lexicographic;
  }
  if (value == "random") {
    return PairOrder::Random;
  }
  throw std::invalid_argument("unsupported --pair-order value: " + value);
}

const char* pair_order_name(PairOrder order) {
  switch (order) {
    case PairOrder::Lexicographic:
      return "lexicographic";
    case PairOrder::Random:
      return "random";
  }
  return "unknown";
}

ActiveOverlapSource parse_active_overlap_source(const std::string& value) {
  if (value == "input") {
    return ActiveOverlapSource::Input;
  }
  if (value == "optimized_vbscf") {
    return ActiveOverlapSource::OptimizedVbscf;
  }
  throw std::invalid_argument("unsupported --active-overlap-source value: " + value);
}

const char* active_overlap_source_name(ActiveOverlapSource source) {
  switch (source) {
    case ActiveOverlapSource::Input:
      return "input";
    case ActiveOverlapSource::OptimizedVbscf:
      return "optimized_vbscf";
  }
  return "unknown";
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
    if (argument_name == "--pair-order") {
      options.pair_order = parse_pair_order(argument_value);
      continue;
    }
    if (argument_name == "--active-overlap-source") {
      options.active_overlap_source = parse_active_overlap_source(argument_value);
      continue;
    }
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
    if (argument_name == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(argument_value));
      continue;
    }
    if (argument_name == "--report-every") {
      options.report_every = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-examples") {
      options.top_examples = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--optimizer-max-iterations") {
      options.optimizer_max_iterations = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--optimizer-gradient-tolerance") {
      options.optimizer_gradient_tolerance = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--optimizer-energy-tolerance") {
      options.optimizer_energy_tolerance = std::stod(argument_value);
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
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if ((options.filter_left_structure < 0) != (options.filter_right_structure < 0)) {
    throw std::invalid_argument(
        "--left-structure and --right-structure must be provided together");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
  }
  if (options.top_examples <= 0) {
    throw std::invalid_argument("--top-examples must be positive");
  }
  if (options.optimizer_max_iterations <= 0) {
    throw std::invalid_argument("--optimizer-max-iterations must be positive");
  }
  if (options.optimizer_gradient_tolerance <= 0.0) {
    throw std::invalid_argument("--optimizer-gradient-tolerance must be positive");
  }
  if (options.optimizer_energy_tolerance <= 0.0) {
    throw std::invalid_argument("--optimizer-energy-tolerance must be positive");
  }
  if (options.singular_value_threshold <= 0.0) {
    throw std::invalid_argument("--singular-value-threshold must be positive");
  }
  if (options.edge_max_abs_threshold < 0.0) {
    throw std::invalid_argument("--edge-max-abs-threshold must be non-negative");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  return options;
}

std::vector<Pair> build_pair_list(
    int structure_count,
    PairOrder pair_order,
    std::uint32_t seed,
    int max_pairs,
    int filter_left_structure,
    int filter_right_structure) {
  if (filter_left_structure >= 0 && filter_right_structure >= 0) {
    if (filter_left_structure >= structure_count ||
        filter_right_structure >= structure_count ||
        filter_left_structure <= filter_right_structure ||
        filter_right_structure < 0) {
      throw std::invalid_argument("requested structure filter is out of range");
    }
    return {{filter_left_structure, filter_right_structure}};
  }
  if (structure_count < 2) {
    return {};
  }
  std::vector<Pair> pairs;
  pairs.reserve(xmvb::to_size(structure_count) *
                xmvb::to_size(structure_count - 1) / 2);
  for (int left_structure = 0; left_structure < structure_count; ++left_structure) {
    for (int right_structure = 0; right_structure < left_structure; ++right_structure) {
      pairs.emplace_back(left_structure, right_structure);
    }
  }
  if (pair_order == PairOrder::Random) {
    std::mt19937 rng(seed);
    std::shuffle(pairs.begin(), pairs.end(), rng);
  }
  if (max_pairs > 0 && static_cast<int>(pairs.size()) > max_pairs) {
    pairs.resize(xmvb::to_size(max_pairs));
  }
  return pairs;
}

ActiveOverlapSelectionResult select_active_overlap_matrix(
    const Options& options,
    const xmvb::vb::CppVbInputLoadResult& load_result) {
  ActiveOverlapSelectionResult result;
  if (options.active_overlap_source == ActiveOverlapSource::Input) {
    result.active_overlap_matrix =
        load_result.input.orbital_preparation_input.active_orbital_overlap_matrix;
    return result;
  }

  xmvb::vb::CppVbScfOptimizerOptions optimizer_options;
  optimizer_options.backend = xmvb::vb::CppVbScfOptimizerBackend::Lbfgspp;
  optimizer_options.max_iterations = options.optimizer_max_iterations;
  optimizer_options.gradient_tolerance = options.optimizer_gradient_tolerance;
  optimizer_options.energy_tolerance = options.optimizer_energy_tolerance;
  optimizer_options.verbose = false;
  optimizer_options.retain_accepted_iteration_trace = true;

  xmvb::vb::CppVbScfOptimizer optimizer(optimizer_options);
  const auto optimization_result = optimizer.optimize(
      load_result.input,
      load_result.nuclear_repulsion_energy);
  if (optimization_result.accepted_iteration_trace.empty()) {
    throw std::runtime_error(
        "optimized_vbscf overlap source requires at least one accepted optimizer snapshot");
  }

  const auto& final_snapshot = optimization_result.accepted_iteration_trace.back();
  result.active_overlap_matrix = final_snapshot.active_orbital_overlap_matrix;

  OptimizedOverlapSummary summary;
  summary.converged = optimization_result.converged;
  summary.termination_reason = optimization_result.termination_reason;
  summary.accepted_iterations = optimization_result.n_iterations;
  summary.objective_evaluations =
      static_cast<int>(optimization_result.total_energy_history.size());
  summary.initial_total_energy = optimization_result.initial_total_energy;
  summary.final_total_energy = optimization_result.final_total_energy;
  summary.final_gradient_inf_norm = optimization_result.final_gradient_inf_norm;
  summary.final_gradient_l2_norm = optimization_result.final_gradient_l2_norm;
  summary.total_wall_time_seconds = optimization_result.total_wall_time_seconds;
  result.optimized_overlap_summary = std::move(summary);
  return result;
}

ComponentData build_component_data(
    int graph_node,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    const std::vector<int>& support_orbitals) {
  ComponentData component;
  component.graph_node = graph_node;
  const auto& union_component = union_components[xmvb::to_size(graph_node)];
  for (const auto& pair : union_component.left_pairs) {
    component.left_pairs.emplace_back(
        support_orbitals[xmvb::to_size(pair.first)],
        support_orbitals[xmvb::to_size(pair.second)]);
  }
  for (const auto& pair : union_component.right_pairs) {
    component.right_pairs.emplace_back(
        support_orbitals[xmvb::to_size(pair.first)],
        support_orbitals[xmvb::to_size(pair.second)]);
  }
  return component;
}

ComponentData build_local_component_data(
    int graph_node,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components) {
  ComponentData component;
  component.graph_node = graph_node;
  const auto& union_component = union_components[xmvb::to_size(graph_node)];
  component.left_pairs = union_component.left_pairs;
  component.right_pairs = union_component.right_pairs;
  return component;
}

std::vector<int> build_component_ordered_support_orbitals(
    const std::vector<int>& support_orbitals,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    int root_node,
    const std::vector<int>& leaf_nodes) {
  // The collapsed prototype rewrites the support basis so that the root
  // component occupies the leading contiguous orbital block and the leaf
  // components follow in the DP merge order. Exact raw-VB overlap is invariant
  // under a consistent orbital relabeling, so this does not change the value
  // of the overlap. It does, however, remove inter-component orbital
  // interleaving, which is what makes the block-parity collapse below possible.
  std::vector<int> ordered_support_orbitals;
  ordered_support_orbitals.reserve(support_orbitals.size());

  const auto append_component = [&](int component_index) {
    std::vector<int> local_vertices =
        union_components[xmvb::to_size(component_index)].local_vertices;
    std::sort(local_vertices.begin(), local_vertices.end());
    for (const int local_vertex : local_vertices) {
      ordered_support_orbitals.push_back(
          support_orbitals[xmvb::to_size(local_vertex)]);
    }
  };

  append_component(root_node);
  for (const int leaf_node : leaf_nodes) {
    append_component(leaf_node);
  }
  return ordered_support_orbitals;
}

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
  // The separator recurrence must use the same local determinant expansion as
  // the reference raw-VB overlap. Enumerating only pair-orientation flips is
  // not enough, because the exact determinant-term coefficients also include
  // the permutation sign required to canonicalize the alpha and beta occupied
  // lists. Reusing the legacy exact enumerator keeps the prototype on the same
  // determinant basis as `legacy_structure_overlap`.
  const auto legacy_terms = xmvb::vb::enumerate_legacy_determinant_terms(pairs);
  std::vector<OrientationTerm> terms;
  terms.reserve(legacy_terms.size());
  for (const auto& legacy_term : legacy_terms) {
    OrientationTerm term;
    term.alpha_occ = legacy_term.alpha_occ;
    term.beta_occ = legacy_term.beta_occ;
    term.coefficient = legacy_term.coefficient;
    terms.push_back(std::move(term));
  }
  return terms;
}

std::vector<int> remap_occ_to_global_labels(
    const std::vector<int>& local_occ,
    const std::vector<int>& local_to_global_orbitals) {
  std::vector<int> global_occ;
  global_occ.reserve(local_occ.size());
  for (const int local_orbital : local_occ) {
    if (local_orbital < 0 ||
        local_orbital >= static_cast<int>(local_to_global_orbitals.size())) {
      throw std::out_of_range("local occupied orbital is out of range");
    }
    global_occ.push_back(local_to_global_orbitals[xmvb::to_size(local_orbital)]);
  }
  return global_occ;
}

std::vector<OrbitalPair> remap_pairs_to_global_labels(
    const std::vector<OrbitalPair>& local_pairs,
    const std::vector<int>& local_to_global_orbitals) {
  std::vector<OrbitalPair> global_pairs;
  global_pairs.reserve(local_pairs.size());
  for (const auto& [left_orbital, right_orbital] : local_pairs) {
    if (left_orbital < 0 ||
        left_orbital >= static_cast<int>(local_to_global_orbitals.size()) ||
        right_orbital < 0 ||
        right_orbital >= static_cast<int>(local_to_global_orbitals.size())) {
      throw std::out_of_range("local pair orbital is out of range");
    }
    global_pairs.emplace_back(
        local_to_global_orbitals[xmvb::to_size(left_orbital)],
        local_to_global_orbitals[xmvb::to_size(right_orbital)]);
  }
  return global_pairs;
}

std::map<CanonicalDeterminantKey, double> build_coefficient_lookup(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& terms) {
  std::map<CanonicalDeterminantKey, double> lookup;
  for (const auto& term : terms) {
    lookup[{term.alpha_occ, term.beta_occ}] = term.coefficient;
  }
  return lookup;
}

double determinant_for_occ_lists(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ExactWorkCollector* work_collector = nullptr) {
  if (left_occ.size() != right_occ.size()) {
    return 0.0;
  }
  if (work_collector != nullptr) {
    work_collector->separator_root_spin_determinants.insert({left_occ, right_occ});
  }
  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      left_occ,
      right_occ,
      active_overlap_storage,
      n_active_orbitals);
  return overlap_resolver
      .resolve(overlap_submatrix, static_cast<int>(left_occ.size()))
      .overlap_determinant;
}

double determinant_with_zeroed_selected_root_block(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::vector<int>* ordered_left_occ,
    std::vector<int>* ordered_right_occ,
    ExactWorkCollector* work_collector = nullptr) {
  if (ordered_left_occ == nullptr || ordered_right_occ == nullptr) {
    throw std::invalid_argument("ordered occupied-orbital outputs must not be null");
  }

  std::vector<std::pair<int, bool>> left_occ_with_root_flag;
  left_occ_with_root_flag.reserve(left_root_occ.size() + left_leaf_occ.size());
  for (const int orbital : left_root_occ) {
    left_occ_with_root_flag.emplace_back(orbital, true);
  }
  for (const int orbital : left_leaf_occ) {
    left_occ_with_root_flag.emplace_back(orbital, false);
  }
  std::sort(left_occ_with_root_flag.begin(), left_occ_with_root_flag.end());

  std::vector<std::pair<int, bool>> right_occ_with_root_flag;
  right_occ_with_root_flag.reserve(right_root_occ.size() + right_leaf_occ.size());
  for (const int orbital : right_root_occ) {
    right_occ_with_root_flag.emplace_back(orbital, true);
  }
  for (const int orbital : right_leaf_occ) {
    right_occ_with_root_flag.emplace_back(orbital, false);
  }
  std::sort(right_occ_with_root_flag.begin(), right_occ_with_root_flag.end());

  ordered_left_occ->clear();
  ordered_right_occ->clear();
  ordered_left_occ->reserve(left_occ_with_root_flag.size());
  ordered_right_occ->reserve(right_occ_with_root_flag.size());
  for (const auto& [orbital, is_root] : left_occ_with_root_flag) {
    static_cast<void>(is_root);
    ordered_left_occ->push_back(orbital);
  }
  for (const auto& [orbital, is_root] : right_occ_with_root_flag) {
    static_cast<void>(is_root);
    ordered_right_occ->push_back(orbital);
  }
  if (ordered_left_occ->size() != ordered_right_occ->size()) {
    return 0.0;
  }
  if (work_collector != nullptr) {
    std::vector<int> left_root_flags;
    std::vector<int> right_root_flags;
    left_root_flags.reserve(left_occ_with_root_flag.size());
    right_root_flags.reserve(right_occ_with_root_flag.size());
    for (const auto& [orbital, is_root] : left_occ_with_root_flag) {
      static_cast<void>(orbital);
      left_root_flags.push_back(is_root ? 1 : 0);
    }
    for (const auto& [orbital, is_root] : right_occ_with_root_flag) {
      static_cast<void>(orbital);
      right_root_flags.push_back(is_root ? 1 : 0);
    }
    work_collector->separator_zeroed_block_spin_determinants.insert(
        {*ordered_left_occ, *ordered_right_occ, left_root_flags, right_root_flags});
  }

  auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      *ordered_left_occ,
      *ordered_right_occ,
      active_overlap_storage,
      n_active_orbitals);
  const int dimension = static_cast<int>(ordered_left_occ->size());
  for (int column = 0; column < dimension; ++column) {
    if (!left_occ_with_root_flag[xmvb::to_size(column)].second) {
      continue;
    }
    for (int row = 0; row < dimension; ++row) {
      if (!right_occ_with_root_flag[xmvb::to_size(row)].second) {
        continue;
      }
      overlap_submatrix[xmvb::to_size(column) * dimension + row] = 0.0;
    }
  }
  return overlap_resolver
      .resolve(overlap_submatrix, dimension)
      .overlap_determinant;
}

std::vector<int> select_occ_by_mask(
    const std::vector<int>& occ,
    std::uint32_t mask) {
  std::vector<int> selected;
  selected.reserve(occ.size());
  for (int index = 0; index < static_cast<int>(occ.size()); ++index) {
    if ((mask & (static_cast<std::uint32_t>(1) << index)) != 0U) {
      selected.push_back(occ[xmvb::to_size(index)]);
    }
  }
  return selected;
}

double lookup_global_coefficient(
    const std::vector<int>& alpha_occ,
    const std::vector<int>& beta_occ,
    const std::map<CanonicalDeterminantKey, double>& coefficient_lookup) {
  std::vector<int> alpha_key = alpha_occ;
  std::vector<int> beta_key = beta_occ;
  std::sort(alpha_key.begin(), alpha_key.end());
  std::sort(beta_key.begin(), beta_key.end());
  const auto iterator = coefficient_lookup.find({alpha_key, beta_key});
  if (iterator == coefficient_lookup.end()) {
    return 0.0;
  }
  return iterator->second;
}

int popcount(std::uint32_t mask) {
  int count = 0;
  while (mask != 0U) {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

int canonicalization_parity(const std::vector<int>& occupied_orbitals) {
  // The leaf/root minors are evaluated in block order. The exact raw-VB
  // reference, however, builds determinant overlaps in canonical occupied
  // order. This parity counts the sign of sorting the block-ordered occupied
  // list into the canonical ascending order.
  int parity = 0;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals.size();
         ++right_index) {
      if (occupied_orbitals[left_index] > occupied_orbitals[right_index]) {
        parity ^= 1;
      }
    }
  }
  return parity;
}

struct CoefficientFactorPart {
  double coefficient = 0.0;
  const std::vector<int>* alpha_occ = nullptr;
  const std::vector<int>* beta_occ = nullptr;
};

std::vector<int> concatenate_occ_lists(const std::vector<std::vector<int>>& occ_lists) {
  std::size_t total_size = 0;
  for (const auto& occ : occ_lists) {
    total_size += occ.size();
  }
  std::vector<int> concatenated;
  concatenated.reserve(total_size);
  for (const auto& occ : occ_lists) {
    concatenated.insert(concatenated.end(), occ.begin(), occ.end());
  }
  return concatenated;
}

double factorized_global_coefficient(
    const std::vector<CoefficientFactorPart>& component_parts) {
  double coefficient = 1.0;
  std::vector<std::vector<int>> alpha_occ_lists;
  std::vector<std::vector<int>> beta_occ_lists;
  alpha_occ_lists.reserve(component_parts.size());
  beta_occ_lists.reserve(component_parts.size());
  for (const auto& part : component_parts) {
    if (part.alpha_occ == nullptr || part.beta_occ == nullptr) {
      throw std::invalid_argument("component occupied-orbital lists must not be null");
    }
    coefficient *= part.coefficient;
    alpha_occ_lists.push_back(*part.alpha_occ);
    beta_occ_lists.push_back(*part.beta_occ);
  }
  const auto alpha_concat = concatenate_occ_lists(alpha_occ_lists);
  const auto beta_concat = concatenate_occ_lists(beta_occ_lists);
  const int inter_component_parity =
      canonicalization_parity(alpha_concat) ^ canonicalization_parity(beta_concat);
  return coefficient * parity_sign(inter_component_parity);
}

void record_coefficient_factorization_check(
    double reference_coefficient,
    double factorized_coefficient,
    StarPairStats* stats) {
  if (stats == nullptr) {
    throw std::invalid_argument("stats must not be null");
  }
  ++stats->coefficient_factorization_checks;
  const double absolute_error =
      std::abs(reference_coefficient - factorized_coefficient);
  stats->max_coefficient_factorization_abs_error = std::max(
      stats->max_coefficient_factorization_abs_error,
      absolute_error);
  if (absolute_error > 1.0e-12) {
    ++stats->coefficient_factorization_mismatches;
  }
}

std::vector<LeafMessageEntry> build_leaf_message_entries(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    ExactWorkCollector* work_collector) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
  const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
  const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
  const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

  std::vector<LeafMessageEntry> entries;
  for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
    for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
      for (std::uint32_t alpha_row_mask = 0;
           alpha_row_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_rows);
           ++alpha_row_mask) {
        const int alpha_row_count = popcount(alpha_row_mask);
        for (std::uint32_t alpha_col_mask = 0;
             alpha_col_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_cols);
             ++alpha_col_mask) {
          const int alpha_col_count = popcount(alpha_col_mask);
          if (alpha_row_count + static_cast<int>(right_leaf_term.alpha_occ.size()) !=
              alpha_col_count + static_cast<int>(left_leaf_term.alpha_occ.size())) {
            continue;
          }

          const auto alpha_rows_root =
              select_occ_by_mask(right_root_term.alpha_occ, alpha_row_mask);
          const auto alpha_cols_root =
              select_occ_by_mask(left_root_term.alpha_occ, alpha_col_mask);
          std::vector<int> ordered_alpha_cols;
          std::vector<int> ordered_alpha_rows;
          const double alpha_determinant = determinant_with_zeroed_selected_root_block(
              alpha_cols_root,
              left_leaf_term.alpha_occ,
              alpha_rows_root,
              right_leaf_term.alpha_occ,
              active_overlap_storage,
              n_active_orbitals,
              overlap_resolver,
              &ordered_alpha_cols,
              &ordered_alpha_rows,
              work_collector);
          ++(*subdeterminant_evaluations);
          if (std::abs(alpha_determinant) <= 1.0e-15) {
            continue;
          }

          for (std::uint32_t beta_row_mask = 0;
               beta_row_mask < (static_cast<std::uint32_t>(1) << n_beta_root_rows);
               ++beta_row_mask) {
            const int beta_row_count = popcount(beta_row_mask);
            for (std::uint32_t beta_col_mask = 0;
                 beta_col_mask < (static_cast<std::uint32_t>(1) << n_beta_root_cols);
                 ++beta_col_mask) {
              const int beta_col_count = popcount(beta_col_mask);
              if (beta_row_count + static_cast<int>(right_leaf_term.beta_occ.size()) !=
                  beta_col_count + static_cast<int>(left_leaf_term.beta_occ.size())) {
                continue;
              }

              const auto beta_rows_root =
                  select_occ_by_mask(right_root_term.beta_occ, beta_row_mask);
              const auto beta_cols_root =
                  select_occ_by_mask(left_root_term.beta_occ, beta_col_mask);
              std::vector<int> ordered_beta_cols;
              std::vector<int> ordered_beta_rows;
              const double beta_determinant = determinant_with_zeroed_selected_root_block(
                  beta_cols_root,
                  left_leaf_term.beta_occ,
                  beta_rows_root,
                  right_leaf_term.beta_occ,
                  active_overlap_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  &ordered_beta_cols,
                  &ordered_beta_rows,
                  work_collector);
              ++(*subdeterminant_evaluations);
              if (std::abs(beta_determinant) <= 1.0e-15) {
                continue;
              }

              LeafMessageEntry entry;
              entry.alpha_row_mask = alpha_row_mask;
              entry.alpha_col_mask = alpha_col_mask;
              entry.beta_row_mask = beta_row_mask;
              entry.beta_col_mask = beta_col_mask;
              entry.value = alpha_determinant * beta_determinant;
              entry.alpha_block_rows = ordered_alpha_rows;
              entry.alpha_block_cols = ordered_alpha_cols;
              entry.beta_block_rows = std::move(ordered_beta_rows);
              entry.beta_block_cols = std::move(ordered_beta_cols);
              entry.left_alpha_occ = left_leaf_term.alpha_occ;
              entry.left_beta_occ = left_leaf_term.beta_occ;
              entry.right_alpha_occ = right_leaf_term.alpha_occ;
              entry.right_beta_occ = right_leaf_term.beta_occ;
              entry.left_term_coefficient = left_leaf_term.coefficient;
              entry.right_term_coefficient = right_leaf_term.coefficient;
              entries.push_back(std::move(entry));
            }
          }
        }
      }
    }
  }
  return entries;
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

StarPairStats evaluate_star_pair(
    double exact_overlap,
    std::uint64_t reference_determinant_pair_count,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::MetricAwareGraphSummary& metric_summary,
    const xmvb::vb::MetricAwareComponentGraph& metric_graph,
    const std::vector<ComponentData>& components,
    int root_node,
    const std::map<CanonicalDeterminantKey, double>& left_coefficient_lookup,
    const std::map<CanonicalDeterminantKey, double>& right_coefficient_lookup,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ExactWorkCollector* work_collector) {
  StarPairStats stats;
  stats.covered = true;
  stats.exact_overlap = exact_overlap;
  stats.metric_width_upper_bound = metric_summary.weighted_min_degree_width_upper_bound;
  stats.node_count = metric_graph.node_count;
  stats.root_node = root_node;
  stats.reference_determinant_pair_count = reference_determinant_pair_count;

  const auto& root_component = components[xmvb::to_size(root_node)];
  std::vector<int> leaf_nodes = metric_graph.adjacency[xmvb::to_size(root_node)];
  std::sort(leaf_nodes.begin(), leaf_nodes.end());
  int total_alpha_electrons = static_cast<int>(root_component.right_orientation_terms.empty()
                                                   ? 0
                                                   : root_component.right_orientation_terms.front()
                                                         .alpha_occ.size());
  int total_beta_electrons = static_cast<int>(root_component.right_orientation_terms.empty()
                                                  ? 0
                                                  : root_component.right_orientation_terms.front()
                                                        .beta_occ.size());
  for (const int leaf_node : leaf_nodes) {
    const auto& leaf_component = components[xmvb::to_size(leaf_node)];
    total_alpha_electrons +=
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().alpha_occ.size());
    total_beta_electrons +=
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().beta_occ.size());
  }

  for (const auto& left_root_term : root_component.left_orientation_terms) {
    for (const auto& right_root_term : root_component.right_orientation_terms) {
      ++stats.local_term_pair_visits;

      const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
      const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
      const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
      const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

      const std::uint32_t alpha_row_full_mask =
          (n_alpha_root_rows == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_rows) - 1U);
      const std::uint32_t alpha_col_full_mask =
          (n_alpha_root_cols == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_cols) - 1U);
      const std::uint32_t beta_row_full_mask =
          (n_beta_root_rows == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_rows) - 1U);
      const std::uint32_t beta_col_full_mask =
          (n_beta_root_cols == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_cols) - 1U);

      if (leaf_nodes.size() == 1) {
        const auto& leaf_component = components[xmvb::to_size(leaf_nodes[0])];
        stats.local_term_pair_visits +=
            static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
            static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
        for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
          for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
            std::vector<int> left_alpha_occ = left_root_term.alpha_occ;
            std::vector<int> left_beta_occ = left_root_term.beta_occ;
            std::vector<int> right_alpha_occ = right_root_term.alpha_occ;
            std::vector<int> right_beta_occ = right_root_term.beta_occ;
            left_alpha_occ.insert(
                left_alpha_occ.end(),
                left_leaf_term.alpha_occ.begin(),
                left_leaf_term.alpha_occ.end());
            left_beta_occ.insert(
                left_beta_occ.end(),
                left_leaf_term.beta_occ.begin(),
                left_leaf_term.beta_occ.end());
            right_alpha_occ.insert(
                right_alpha_occ.end(),
                right_leaf_term.alpha_occ.begin(),
                right_leaf_term.alpha_occ.end());
            right_beta_occ.insert(
                right_beta_occ.end(),
                right_leaf_term.beta_occ.begin(),
                right_leaf_term.beta_occ.end());

            const double left_coefficient = lookup_global_coefficient(
                left_alpha_occ,
                left_beta_occ,
                left_coefficient_lookup);
            const double right_coefficient = lookup_global_coefficient(
                right_alpha_occ,
                right_beta_occ,
                right_coefficient_lookup);
            const double left_factorized_coefficient = factorized_global_coefficient(
                {{left_root_term.coefficient, &left_root_term.alpha_occ, &left_root_term.beta_occ},
                 {left_leaf_term.coefficient, &left_leaf_term.alpha_occ, &left_leaf_term.beta_occ}});
            const double right_factorized_coefficient = factorized_global_coefficient(
                {{right_root_term.coefficient,
                  &right_root_term.alpha_occ,
                  &right_root_term.beta_occ},
                 {right_leaf_term.coefficient,
                  &right_leaf_term.alpha_occ,
                  &right_leaf_term.beta_occ}});
            record_coefficient_factorization_check(
                left_coefficient,
                left_factorized_coefficient,
                &stats);
            record_coefficient_factorization_check(
                right_coefficient,
                right_factorized_coefficient,
                &stats);
            if (std::abs(left_factorized_coefficient) <= 1.0e-15 ||
                std::abs(right_factorized_coefficient) <= 1.0e-15) {
              continue;
            }

            for (std::uint32_t alpha_row_mask = 0;
                 alpha_row_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_rows);
                 ++alpha_row_mask) {
              const int alpha_row_count = popcount(alpha_row_mask);
              for (std::uint32_t alpha_col_mask = 0;
                   alpha_col_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_cols);
                   ++alpha_col_mask) {
                const int alpha_col_count = popcount(alpha_col_mask);
                if (alpha_row_count + static_cast<int>(right_leaf_term.alpha_occ.size()) !=
                    alpha_col_count + static_cast<int>(left_leaf_term.alpha_occ.size())) {
                  continue;
                }

                const auto alpha_rows_root =
                    select_occ_by_mask(right_root_term.alpha_occ, alpha_row_mask);
                const auto alpha_cols_root =
                    select_occ_by_mask(left_root_term.alpha_occ, alpha_col_mask);
                std::vector<int> ordered_alpha_cols;
                std::vector<int> ordered_alpha_rows;
                const double alpha_leaf_determinant =
                    determinant_with_zeroed_selected_root_block(
                        alpha_cols_root,
                        left_leaf_term.alpha_occ,
                        alpha_rows_root,
                        right_leaf_term.alpha_occ,
                        active_overlap_storage,
                        n_active_orbitals,
                        overlap_resolver,
                        &ordered_alpha_cols,
                        &ordered_alpha_rows,
                        work_collector);
                ++stats.subdeterminant_evaluations;
                if (std::abs(alpha_leaf_determinant) <= 1.0e-15) {
                  continue;
                }

                for (std::uint32_t beta_row_mask = 0;
                     beta_row_mask < (static_cast<std::uint32_t>(1) << n_beta_root_rows);
                     ++beta_row_mask) {
                  const int beta_row_count = popcount(beta_row_mask);
                  for (std::uint32_t beta_col_mask = 0;
                       beta_col_mask < (static_cast<std::uint32_t>(1) << n_beta_root_cols);
                       ++beta_col_mask) {
                    const int beta_col_count = popcount(beta_col_mask);
                    if (beta_row_count + static_cast<int>(right_leaf_term.beta_occ.size()) !=
                        beta_col_count + static_cast<int>(left_leaf_term.beta_occ.size())) {
                      continue;
                    }

                    const auto beta_rows_root =
                        select_occ_by_mask(right_root_term.beta_occ, beta_row_mask);
                    const auto beta_cols_root =
                        select_occ_by_mask(left_root_term.beta_occ, beta_col_mask);
                    std::vector<int> ordered_beta_cols;
                    std::vector<int> ordered_beta_rows;
                    const double beta_leaf_determinant =
                        determinant_with_zeroed_selected_root_block(
                            beta_cols_root,
                            left_leaf_term.beta_occ,
                            beta_rows_root,
                            right_leaf_term.beta_occ,
                            active_overlap_storage,
                            n_active_orbitals,
                            overlap_resolver,
                            &ordered_beta_cols,
                            &ordered_beta_rows,
                            work_collector);
                    ++stats.subdeterminant_evaluations;
                    if (std::abs(beta_leaf_determinant) <= 1.0e-15) {
                      continue;
                    }

                    const std::uint32_t alpha_row_remainder =
                        alpha_row_full_mask ^ alpha_row_mask;
                    const std::uint32_t alpha_col_remainder =
                        alpha_col_full_mask ^ alpha_col_mask;
                    const std::uint32_t beta_row_remainder =
                        beta_row_full_mask ^ beta_row_mask;
                    const std::uint32_t beta_col_remainder =
                        beta_col_full_mask ^ beta_col_mask;
                    const auto alpha_root_rows_remainder =
                        select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
                    const auto alpha_root_cols_remainder =
                        select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
                    const auto beta_root_rows_remainder =
                        select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
                    const auto beta_root_cols_remainder =
                        select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

                    const double alpha_root_determinant = determinant_for_occ_lists(
                        alpha_root_cols_remainder,
                        alpha_root_rows_remainder,
                        active_overlap_storage,
                        n_active_orbitals,
                        overlap_resolver,
                        work_collector);
                    ++stats.subdeterminant_evaluations;
                    if (std::abs(alpha_root_determinant) <= 1.0e-15) {
                      continue;
                    }

                    const double beta_root_determinant = determinant_for_occ_lists(
                        beta_root_cols_remainder,
                        beta_root_rows_remainder,
                        active_overlap_storage,
                        n_active_orbitals,
                        overlap_resolver,
                        work_collector);
                    ++stats.subdeterminant_evaluations;
                    if (std::abs(beta_root_determinant) <= 1.0e-15) {
                      continue;
                    }

                    std::vector<int> alpha_rows = ordered_alpha_rows;
                    std::vector<int> alpha_cols = ordered_alpha_cols;
                    std::vector<int> beta_rows = ordered_beta_rows;
                    std::vector<int> beta_cols = ordered_beta_cols;
                    alpha_rows.insert(
                        alpha_rows.end(),
                        alpha_root_rows_remainder.begin(),
                        alpha_root_rows_remainder.end());
                    alpha_cols.insert(
                        alpha_cols.end(),
                        alpha_root_cols_remainder.begin(),
                        alpha_root_cols_remainder.end());
                    beta_rows.insert(
                        beta_rows.end(),
                        beta_root_rows_remainder.begin(),
                        beta_root_rows_remainder.end());
                    beta_cols.insert(
                        beta_cols.end(),
                        beta_root_cols_remainder.begin(),
                        beta_root_cols_remainder.end());

                    int parity = 0;
                    parity ^= canonicalization_parity(alpha_rows);
                    parity ^= canonicalization_parity(alpha_cols);
                    parity ^= canonicalization_parity(beta_rows);
                    parity ^= canonicalization_parity(beta_cols);

                    ++stats.dp_transition_count;
                    stats.star_overlap +=
                        left_factorized_coefficient * right_factorized_coefficient *
                        parity_sign(parity) *
                        alpha_leaf_determinant * beta_leaf_determinant *
                        alpha_root_determinant * beta_root_determinant;
                  }
                }
              }
            }
          }
        }
        continue;
      }

      if (leaf_nodes.size() == 2) {
        std::array<std::vector<LeafMessageEntry>, 2> leaf_messages;
        for (int leaf_slot = 0; leaf_slot < 2; ++leaf_slot) {
          const auto& leaf_component =
              components[xmvb::to_size(leaf_nodes[xmvb::to_size(leaf_slot)])];
          stats.local_term_pair_visits +=
              static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
              static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
          leaf_messages[leaf_slot] = build_leaf_message_entries(
              left_root_term,
              right_root_term,
              leaf_component,
              active_overlap_storage,
              n_active_orbitals,
              overlap_resolver,
              &stats.subdeterminant_evaluations,
              work_collector);
        }

        for (const auto& first_entry : leaf_messages[0]) {
          for (const auto& second_entry : leaf_messages[1]) {
            if ((first_entry.alpha_row_mask & second_entry.alpha_row_mask) != 0U ||
                (first_entry.alpha_col_mask & second_entry.alpha_col_mask) != 0U ||
                (first_entry.beta_row_mask & second_entry.beta_row_mask) != 0U ||
                (first_entry.beta_col_mask & second_entry.beta_col_mask) != 0U) {
              continue;
            }

            const std::uint32_t used_alpha_row_mask =
                first_entry.alpha_row_mask | second_entry.alpha_row_mask;
            const std::uint32_t used_alpha_col_mask =
                first_entry.alpha_col_mask | second_entry.alpha_col_mask;
            const std::uint32_t used_beta_row_mask =
                first_entry.beta_row_mask | second_entry.beta_row_mask;
            const std::uint32_t used_beta_col_mask =
                first_entry.beta_col_mask | second_entry.beta_col_mask;

            const std::uint32_t alpha_row_remainder =
                alpha_row_full_mask ^ used_alpha_row_mask;
            const std::uint32_t alpha_col_remainder =
                alpha_col_full_mask ^ used_alpha_col_mask;
            const std::uint32_t beta_row_remainder =
                beta_row_full_mask ^ used_beta_row_mask;
            const std::uint32_t beta_col_remainder =
                beta_col_full_mask ^ used_beta_col_mask;

            const auto alpha_root_rows =
                select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
            const auto alpha_root_cols =
                select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
            const auto beta_root_rows =
                select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
            const auto beta_root_cols =
                select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

            const double alpha_root_determinant = determinant_for_occ_lists(
                alpha_root_cols,
                alpha_root_rows,
                active_overlap_storage,
                n_active_orbitals,
                overlap_resolver,
                work_collector);
            ++stats.subdeterminant_evaluations;
            if (std::abs(alpha_root_determinant) <= 1.0e-15) {
              continue;
            }

            const double beta_root_determinant = determinant_for_occ_lists(
                beta_root_cols,
                beta_root_rows,
                active_overlap_storage,
                n_active_orbitals,
                overlap_resolver,
                work_collector);
            ++stats.subdeterminant_evaluations;
            if (std::abs(beta_root_determinant) <= 1.0e-15) {
              continue;
            }

            std::vector<int> alpha_rows = first_entry.alpha_block_rows;
            std::vector<int> alpha_cols = first_entry.alpha_block_cols;
            std::vector<int> beta_rows = first_entry.beta_block_rows;
            std::vector<int> beta_cols = first_entry.beta_block_cols;
            alpha_rows.insert(
                alpha_rows.end(),
                second_entry.alpha_block_rows.begin(),
                second_entry.alpha_block_rows.end());
            alpha_cols.insert(
                alpha_cols.end(),
                second_entry.alpha_block_cols.begin(),
                second_entry.alpha_block_cols.end());
            beta_rows.insert(
                beta_rows.end(),
                second_entry.beta_block_rows.begin(),
                second_entry.beta_block_rows.end());
            beta_cols.insert(
                beta_cols.end(),
                second_entry.beta_block_cols.begin(),
                second_entry.beta_block_cols.end());
            alpha_rows.insert(alpha_rows.end(), alpha_root_rows.begin(), alpha_root_rows.end());
            alpha_cols.insert(alpha_cols.end(), alpha_root_cols.begin(), alpha_root_cols.end());
            beta_rows.insert(beta_rows.end(), beta_root_rows.begin(), beta_root_rows.end());
            beta_cols.insert(beta_cols.end(), beta_root_cols.begin(), beta_root_cols.end());

            int parity = 0;
            parity ^= canonicalization_parity(alpha_rows);
            parity ^= canonicalization_parity(alpha_cols);
            parity ^= canonicalization_parity(beta_rows);
            parity ^= canonicalization_parity(beta_cols);

            std::vector<int> left_alpha_occ = left_root_term.alpha_occ;
            std::vector<int> left_beta_occ = left_root_term.beta_occ;
            std::vector<int> right_alpha_occ = right_root_term.alpha_occ;
            std::vector<int> right_beta_occ = right_root_term.beta_occ;
            left_alpha_occ.insert(
                left_alpha_occ.end(),
                first_entry.left_alpha_occ.begin(),
                first_entry.left_alpha_occ.end());
            left_alpha_occ.insert(
                left_alpha_occ.end(),
                second_entry.left_alpha_occ.begin(),
                second_entry.left_alpha_occ.end());
            left_beta_occ.insert(
                left_beta_occ.end(),
                first_entry.left_beta_occ.begin(),
                first_entry.left_beta_occ.end());
            left_beta_occ.insert(
                left_beta_occ.end(),
                second_entry.left_beta_occ.begin(),
                second_entry.left_beta_occ.end());
            right_alpha_occ.insert(
                right_alpha_occ.end(),
                first_entry.right_alpha_occ.begin(),
                first_entry.right_alpha_occ.end());
            right_alpha_occ.insert(
                right_alpha_occ.end(),
                second_entry.right_alpha_occ.begin(),
                second_entry.right_alpha_occ.end());
            right_beta_occ.insert(
                right_beta_occ.end(),
                first_entry.right_beta_occ.begin(),
                first_entry.right_beta_occ.end());
            right_beta_occ.insert(
                right_beta_occ.end(),
                second_entry.right_beta_occ.begin(),
                second_entry.right_beta_occ.end());

            const double left_coefficient = lookup_global_coefficient(
                left_alpha_occ,
                left_beta_occ,
                left_coefficient_lookup);
            const double right_coefficient = lookup_global_coefficient(
                right_alpha_occ,
                right_beta_occ,
                right_coefficient_lookup);
            const double left_factorized_coefficient = factorized_global_coefficient(
                {{left_root_term.coefficient,
                  &left_root_term.alpha_occ,
                  &left_root_term.beta_occ},
                 {first_entry.left_term_coefficient,
                  &first_entry.left_alpha_occ,
                  &first_entry.left_beta_occ},
                 {second_entry.left_term_coefficient,
                  &second_entry.left_alpha_occ,
                  &second_entry.left_beta_occ}});
            const double right_factorized_coefficient = factorized_global_coefficient(
                {{right_root_term.coefficient,
                  &right_root_term.alpha_occ,
                  &right_root_term.beta_occ},
                 {first_entry.right_term_coefficient,
                  &first_entry.right_alpha_occ,
                  &first_entry.right_beta_occ},
                 {second_entry.right_term_coefficient,
                  &second_entry.right_alpha_occ,
                  &second_entry.right_beta_occ}});
            record_coefficient_factorization_check(
                left_coefficient,
                left_factorized_coefficient,
                &stats);
            record_coefficient_factorization_check(
                right_coefficient,
                right_factorized_coefficient,
                &stats);
            if (std::abs(left_factorized_coefficient) <= 1.0e-15 ||
                std::abs(right_factorized_coefficient) <= 1.0e-15) {
              continue;
            }

            ++stats.dp_transition_count;
            stats.star_overlap +=
                left_factorized_coefficient * right_factorized_coefficient *
                parity_sign(parity) *
                first_entry.value * second_entry.value *
                alpha_root_determinant * beta_root_determinant;
          }
        }
        continue;
      }

      std::vector<std::vector<LeafMessageEntry>> leaf_messages(leaf_nodes.size());
      for (std::size_t leaf_order = 0; leaf_order < leaf_nodes.size(); ++leaf_order) {
        const auto& leaf_component = components[xmvb::to_size(leaf_nodes[leaf_order])];
        stats.local_term_pair_visits +=
            static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
            static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
        leaf_messages[leaf_order] = build_leaf_message_entries(
            left_root_term,
            right_root_term,
            leaf_component,
            active_overlap_storage,
            n_active_orbitals,
            overlap_resolver,
            &stats.subdeterminant_evaluations,
            work_collector);
      }

      std::vector<int> alpha_block_rows;
      std::vector<int> alpha_block_cols;
      std::vector<int> beta_block_rows;
      std::vector<int> beta_block_cols;
      std::vector<int> left_alpha_occ;
      std::vector<int> left_beta_occ;
      std::vector<int> right_alpha_occ;
      std::vector<int> right_beta_occ;
      std::vector<CoefficientFactorPart> left_coefficient_parts;
      std::vector<CoefficientFactorPart> right_coefficient_parts;
      alpha_block_rows.reserve(xmvb::to_size(total_alpha_electrons));
      alpha_block_cols.reserve(xmvb::to_size(total_alpha_electrons));
      beta_block_rows.reserve(xmvb::to_size(total_beta_electrons));
      beta_block_cols.reserve(xmvb::to_size(total_beta_electrons));
      left_alpha_occ.reserve(xmvb::to_size(total_alpha_electrons));
      left_beta_occ.reserve(xmvb::to_size(total_beta_electrons));
      right_alpha_occ.reserve(xmvb::to_size(total_alpha_electrons));
      right_beta_occ.reserve(xmvb::to_size(total_beta_electrons));
      left_coefficient_parts.reserve(1 + leaf_messages.size());
      right_coefficient_parts.reserve(1 + leaf_messages.size());
      left_alpha_occ.insert(
          left_alpha_occ.end(),
          left_root_term.alpha_occ.begin(),
          left_root_term.alpha_occ.end());
      left_beta_occ.insert(
          left_beta_occ.end(),
          left_root_term.beta_occ.begin(),
          left_root_term.beta_occ.end());
      right_alpha_occ.insert(
          right_alpha_occ.end(),
          right_root_term.alpha_occ.begin(),
          right_root_term.alpha_occ.end());
      right_beta_occ.insert(
          right_beta_occ.end(),
          right_root_term.beta_occ.begin(),
          right_root_term.beta_occ.end());
      left_coefficient_parts.push_back(
          {left_root_term.coefficient, &left_root_term.alpha_occ, &left_root_term.beta_occ});
      right_coefficient_parts.push_back(
          {right_root_term.coefficient,
           &right_root_term.alpha_occ,
           &right_root_term.beta_occ});

      const auto accumulate_leaf_assignments =
          [&](const auto& self,
              std::size_t leaf_order,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              double state_value) -> void {
            if (std::abs(state_value) <= 1.0e-15) {
              return;
            }

            if (leaf_order == leaf_messages.size()) {
              const std::uint32_t alpha_row_remainder =
                  alpha_row_full_mask ^ used_alpha_row_mask;
              const std::uint32_t alpha_col_remainder =
                  alpha_col_full_mask ^ used_alpha_col_mask;
              const std::uint32_t beta_row_remainder =
                  beta_row_full_mask ^ used_beta_row_mask;
              const std::uint32_t beta_col_remainder =
                  beta_col_full_mask ^ used_beta_col_mask;

              const auto alpha_root_rows =
                  select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
              const auto alpha_root_cols =
                  select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
              const auto beta_root_rows =
                  select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
              const auto beta_root_cols =
                  select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

              const double alpha_root_determinant = determinant_for_occ_lists(
                  alpha_root_cols,
                  alpha_root_rows,
                  active_overlap_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  work_collector);
              ++stats.subdeterminant_evaluations;
              if (std::abs(alpha_root_determinant) <= 1.0e-15) {
                return;
              }

              const double beta_root_determinant = determinant_for_occ_lists(
                  beta_root_cols,
                  beta_root_rows,
                  active_overlap_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  work_collector);
              ++stats.subdeterminant_evaluations;
              if (std::abs(beta_root_determinant) <= 1.0e-15) {
                return;
              }

              std::vector<int> alpha_rows = alpha_block_rows;
              std::vector<int> alpha_cols = alpha_block_cols;
              std::vector<int> beta_rows = beta_block_rows;
              std::vector<int> beta_cols = beta_block_cols;
              alpha_rows.insert(alpha_rows.end(), alpha_root_rows.begin(), alpha_root_rows.end());
              alpha_cols.insert(alpha_cols.end(), alpha_root_cols.begin(), alpha_root_cols.end());
              beta_rows.insert(beta_rows.end(), beta_root_rows.begin(), beta_root_rows.end());
              beta_cols.insert(beta_cols.end(), beta_root_cols.begin(), beta_root_cols.end());

              int parity = 0;
              parity ^= canonicalization_parity(alpha_rows);
              parity ^= canonicalization_parity(alpha_cols);
              parity ^= canonicalization_parity(beta_rows);
              parity ^= canonicalization_parity(beta_cols);

              const double left_coefficient = lookup_global_coefficient(
                  left_alpha_occ,
                  left_beta_occ,
                  left_coefficient_lookup);
              const double right_coefficient = lookup_global_coefficient(
                  right_alpha_occ,
                  right_beta_occ,
                  right_coefficient_lookup);
              const double left_factorized_coefficient =
                  factorized_global_coefficient(left_coefficient_parts);
              const double right_factorized_coefficient =
                  factorized_global_coefficient(right_coefficient_parts);
              record_coefficient_factorization_check(
                  left_coefficient,
                  left_factorized_coefficient,
                  &stats);
              record_coefficient_factorization_check(
                  right_coefficient,
                  right_factorized_coefficient,
                  &stats);
              if (std::abs(left_factorized_coefficient) <= 1.0e-15 ||
                  std::abs(right_factorized_coefficient) <= 1.0e-15) {
                return;
              }

              stats.star_overlap +=
                  left_factorized_coefficient * right_factorized_coefficient *
                  parity_sign(parity) * state_value *
                  alpha_root_determinant * beta_root_determinant;
              return;
            }

            for (const auto& entry : leaf_messages[leaf_order]) {
              if ((used_alpha_row_mask & entry.alpha_row_mask) != 0U ||
                  (used_alpha_col_mask & entry.alpha_col_mask) != 0U ||
                  (used_beta_row_mask & entry.beta_row_mask) != 0U ||
                  (used_beta_col_mask & entry.beta_col_mask) != 0U) {
                continue;
              }

              const std::size_t alpha_row_size = alpha_block_rows.size();
              const std::size_t alpha_col_size = alpha_block_cols.size();
              const std::size_t beta_row_size = beta_block_rows.size();
              const std::size_t beta_col_size = beta_block_cols.size();
              const std::size_t left_alpha_size = left_alpha_occ.size();
              const std::size_t left_beta_size = left_beta_occ.size();
              const std::size_t right_alpha_size = right_alpha_occ.size();
              const std::size_t right_beta_size = right_beta_occ.size();
              const std::size_t left_coefficient_size = left_coefficient_parts.size();
              const std::size_t right_coefficient_size = right_coefficient_parts.size();
              alpha_block_rows.insert(
                  alpha_block_rows.end(),
                  entry.alpha_block_rows.begin(),
                  entry.alpha_block_rows.end());
              alpha_block_cols.insert(
                  alpha_block_cols.end(),
                  entry.alpha_block_cols.begin(),
                  entry.alpha_block_cols.end());
              beta_block_rows.insert(
                  beta_block_rows.end(),
                  entry.beta_block_rows.begin(),
                  entry.beta_block_rows.end());
              beta_block_cols.insert(
                  beta_block_cols.end(),
                  entry.beta_block_cols.begin(),
                  entry.beta_block_cols.end());
              left_alpha_occ.insert(
                  left_alpha_occ.end(),
                  entry.left_alpha_occ.begin(),
                  entry.left_alpha_occ.end());
              left_beta_occ.insert(
                  left_beta_occ.end(),
                  entry.left_beta_occ.begin(),
                  entry.left_beta_occ.end());
              right_alpha_occ.insert(
                  right_alpha_occ.end(),
                  entry.right_alpha_occ.begin(),
                  entry.right_alpha_occ.end());
              right_beta_occ.insert(
                  right_beta_occ.end(),
                  entry.right_beta_occ.begin(),
                  entry.right_beta_occ.end());
              left_coefficient_parts.push_back(
                  {entry.left_term_coefficient, &entry.left_alpha_occ, &entry.left_beta_occ});
              right_coefficient_parts.push_back(
                  {entry.right_term_coefficient,
                   &entry.right_alpha_occ,
                   &entry.right_beta_occ});

              ++stats.dp_transition_count;
              self(
                  self,
                  leaf_order + 1,
                  used_alpha_row_mask | entry.alpha_row_mask,
                  used_alpha_col_mask | entry.alpha_col_mask,
                  used_beta_row_mask | entry.beta_row_mask,
                  used_beta_col_mask | entry.beta_col_mask,
                  state_value * entry.value);

              alpha_block_rows.resize(alpha_row_size);
              alpha_block_cols.resize(alpha_col_size);
              beta_block_rows.resize(beta_row_size);
              beta_block_cols.resize(beta_col_size);
              left_alpha_occ.resize(left_alpha_size);
              left_beta_occ.resize(left_beta_size);
              right_alpha_occ.resize(right_alpha_size);
              right_beta_occ.resize(right_beta_size);
              left_coefficient_parts.resize(left_coefficient_size);
              right_coefficient_parts.resize(right_coefficient_size);
            }
          };
      accumulate_leaf_assignments(
          accumulate_leaf_assignments,
          0,
          0U,
          0U,
          0U,
          0U,
          1.0);
    }
  }
  stats.absolute_error = std::abs(stats.exact_overlap - stats.star_overlap);
  return stats;
}

CollapsedStarPairStats evaluate_component_ordered_collapsed_star_pair(
    double exact_overlap,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const std::vector<int>& ordered_support_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ExactSeparatorStateCollector* state_collector) {
  // This routine evaluates the same exact star-separated overlap as the
  // explicit local-term recurrence above, but in the component-ordered support
  // basis and with leaf messages already aggregated onto separator masks. The
  // remaining root merge therefore depends only on mask compatibility,
  // root-remainder minors, and a block-level canonicalization parity.
  CollapsedStarPairStats stats;
  stats.exact_overlap = exact_overlap;
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }
  if (ordered_support_orbitals.size() != xmvb::to_size(support_size)) {
    throw std::invalid_argument("ordered_support_orbitals size must match support_size");
  }

  const auto& root_component = ordered_components.front();
  const auto root_left_pairs_global =
      remap_pairs_to_global_labels(root_component.left_pairs, ordered_support_orbitals);
  const auto root_right_pairs_global =
      remap_pairs_to_global_labels(root_component.right_pairs, ordered_support_orbitals);
  const int n_leaves = static_cast<int>(ordered_components.size()) - 1;
  if (n_leaves == 0) {
    stats.collapsed_overlap = exact_overlap;
    if (state_collector != nullptr) {
      for (const auto& left_root_term : root_component.left_orientation_terms) {
        for (const auto& right_root_term : root_component.right_orientation_terms) {
          ExactSeparatorStateKey state_key;
          state_key.root_left_pairs = root_left_pairs_global;
          state_key.root_right_pairs = root_right_pairs_global;
          state_key.left_root_alpha_occ =
              remap_occ_to_global_labels(
                  left_root_term.alpha_occ,
                  ordered_support_orbitals);
          state_key.left_root_beta_occ =
              remap_occ_to_global_labels(
                  left_root_term.beta_occ,
                  ordered_support_orbitals);
          state_key.right_root_alpha_occ =
              remap_occ_to_global_labels(
                  right_root_term.alpha_occ,
                  ordered_support_orbitals);
          state_key.right_root_beta_occ =
              remap_occ_to_global_labels(
                  right_root_term.beta_occ,
                  ordered_support_orbitals);
          state_collector->unique_exact_separator_states.insert(std::move(state_key));
        }
      }
    }
    return stats;
  }

  std::vector<int> left_leaf_alpha_sizes;
  std::vector<int> left_leaf_beta_sizes;
  std::vector<int> right_leaf_alpha_sizes;
  std::vector<int> right_leaf_beta_sizes;
  left_leaf_alpha_sizes.reserve(xmvb::to_size(n_leaves));
  left_leaf_beta_sizes.reserve(xmvb::to_size(n_leaves));
  right_leaf_alpha_sizes.reserve(xmvb::to_size(n_leaves));
  right_leaf_beta_sizes.reserve(xmvb::to_size(n_leaves));
  for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
    const auto& leaf_component = ordered_components[xmvb::to_size(leaf_index + 1)];
    left_leaf_alpha_sizes.push_back(
        leaf_component.left_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.left_orientation_terms.front().alpha_occ.size()));
    left_leaf_beta_sizes.push_back(
        leaf_component.left_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.left_orientation_terms.front().beta_occ.size()));
    right_leaf_alpha_sizes.push_back(
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().alpha_occ.size()));
    right_leaf_beta_sizes.push_back(
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().beta_occ.size()));
  }

  for (const auto& left_root_term : root_component.left_orientation_terms) {
    for (const auto& right_root_term : root_component.right_orientation_terms) {
      const auto left_root_alpha_occ_global =
          remap_occ_to_global_labels(
              left_root_term.alpha_occ,
              ordered_support_orbitals);
      const auto left_root_beta_occ_global =
          remap_occ_to_global_labels(
              left_root_term.beta_occ,
              ordered_support_orbitals);
      const auto right_root_alpha_occ_global =
          remap_occ_to_global_labels(
              right_root_term.alpha_occ,
              ordered_support_orbitals);
      const auto right_root_beta_occ_global =
          remap_occ_to_global_labels(
              right_root_term.beta_occ,
              ordered_support_orbitals);
      std::map<
          std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
          std::pair<double, double>>
          root_remainder_cache;
      std::vector<std::vector<CollapsedLeafMessageEntry>> collapsed_leaf_messages(
          xmvb::to_size(n_leaves));
      for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
        const auto& leaf_component = ordered_components[xmvb::to_size(leaf_index + 1)];
        collapsed_leaf_messages[xmvb::to_size(leaf_index)] =
            build_collapsed_leaf_messages_component_ordered(
                left_root_term,
                right_root_term,
                leaf_component,
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &stats,
                &stats.subdeterminant_evaluations,
                &stats.hypercube_assignment_count);
        stats.collapsed_leaf_state_count +=
            static_cast<std::uint64_t>(
                collapsed_leaf_messages[xmvb::to_size(leaf_index)].size());
        if (state_collector != nullptr) {
          const auto leaf_left_pairs_global =
              remap_pairs_to_global_labels(
                  leaf_component.left_pairs,
                  ordered_support_orbitals);
          const auto leaf_right_pairs_global =
              remap_pairs_to_global_labels(
                  leaf_component.right_pairs,
                  ordered_support_orbitals);
          if (!collapsed_leaf_messages[xmvb::to_size(leaf_index)].empty()) {
            ExactSeparatorStateCollector::ExactLeafMessageBundleKey bundle_key;
            bundle_key.leaf_left_pairs = leaf_left_pairs_global;
            bundle_key.leaf_right_pairs = leaf_right_pairs_global;
            bundle_key.left_root_alpha_occ = left_root_alpha_occ_global;
            bundle_key.left_root_beta_occ = left_root_beta_occ_global;
            bundle_key.right_root_alpha_occ = right_root_alpha_occ_global;
            bundle_key.right_root_beta_occ = right_root_beta_occ_global;
            state_collector->unique_leaf_message_bundles.insert(std::move(bundle_key));
          }
          for (const auto& entry :
               collapsed_leaf_messages[xmvb::to_size(leaf_index)]) {
            ExactSeparatorStateCollector::ExactLeafMessageStateKey leaf_state_key;
            leaf_state_key.leaf_left_pairs = leaf_left_pairs_global;
            leaf_state_key.leaf_right_pairs = leaf_right_pairs_global;
            leaf_state_key.left_root_alpha_occ = left_root_alpha_occ_global;
            leaf_state_key.left_root_beta_occ = left_root_beta_occ_global;
            leaf_state_key.right_root_alpha_occ = right_root_alpha_occ_global;
            leaf_state_key.right_root_beta_occ = right_root_beta_occ_global;
            leaf_state_key.alpha_row_mask = entry.alpha_row_mask;
            leaf_state_key.alpha_col_mask = entry.alpha_col_mask;
            leaf_state_key.beta_row_mask = entry.beta_row_mask;
            leaf_state_key.beta_col_mask = entry.beta_col_mask;
            state_collector->unique_leaf_message_states.insert(std::move(leaf_state_key));
          }
        }
      }

      const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
      const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
      const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
      const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

      const std::uint32_t alpha_row_full_mask =
          (n_alpha_root_rows == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_rows) - 1U);
      const std::uint32_t alpha_col_full_mask =
          (n_alpha_root_cols == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_cols) - 1U);
      const std::uint32_t beta_row_full_mask =
          (n_beta_root_rows == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_rows) - 1U);
      const std::uint32_t beta_col_full_mask =
          (n_beta_root_cols == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_cols) - 1U);

      std::vector<std::uint32_t> selected_alpha_row_masks(xmvb::to_size(n_leaves), 0U);
      std::vector<std::uint32_t> selected_alpha_col_masks(xmvb::to_size(n_leaves), 0U);
      std::vector<std::uint32_t> selected_beta_row_masks(xmvb::to_size(n_leaves), 0U);
      std::vector<std::uint32_t> selected_beta_col_masks(xmvb::to_size(n_leaves), 0U);

      const auto accumulate_collapsed_messages =
          [&](const auto& self,
              int leaf_index,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              double state_value) -> void {
            if (std::abs(state_value) <= 1.0e-15) {
              return;
            }
            if (state_collector != nullptr) {
              ExactSeparatorStateCollector::ExactMergeStateKey merge_state_key;
              merge_state_key.left_root_alpha_occ = left_root_alpha_occ_global;
              merge_state_key.left_root_beta_occ = left_root_beta_occ_global;
              merge_state_key.right_root_alpha_occ = right_root_alpha_occ_global;
              merge_state_key.right_root_beta_occ = right_root_beta_occ_global;
              merge_state_key.leaf_index = leaf_index;
              merge_state_key.used_alpha_row_mask = used_alpha_row_mask;
              merge_state_key.used_alpha_col_mask = used_alpha_col_mask;
              merge_state_key.used_beta_row_mask = used_beta_row_mask;
              merge_state_key.used_beta_col_mask = used_beta_col_mask;
              state_collector->unique_merge_states.insert(std::move(merge_state_key));
            }
            if (leaf_index == n_leaves) {
              const std::uint32_t alpha_row_remainder =
                  alpha_row_full_mask ^ used_alpha_row_mask;
              const std::uint32_t alpha_col_remainder =
                  alpha_col_full_mask ^ used_alpha_col_mask;
              const std::uint32_t beta_row_remainder =
                  beta_row_full_mask ^ used_beta_row_mask;
              const std::uint32_t beta_col_remainder =
                  beta_col_full_mask ^ used_beta_col_mask;

              const auto alpha_root_rows =
                  select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
              const auto alpha_root_cols =
                  select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
              const auto beta_root_rows =
                  select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
              const auto beta_root_cols =
                  select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);
              const auto cache_key = std::make_tuple(
                  used_alpha_row_mask,
                  used_alpha_col_mask,
                  used_beta_row_mask,
                  used_beta_col_mask);
              auto cache_iterator = root_remainder_cache.find(cache_key);
              double alpha_root_determinant = 0.0;
              double beta_root_determinant = 0.0;
              if (cache_iterator == root_remainder_cache.end()) {
                alpha_root_determinant = determinant_for_occ_lists(
                    alpha_root_cols,
                    alpha_root_rows,
                    support_overlap_storage,
                    support_size,
                    overlap_resolver,
                    nullptr);
                ++stats.subdeterminant_evaluations;
                if (std::abs(alpha_root_determinant) > 1.0e-15) {
                  beta_root_determinant = determinant_for_occ_lists(
                      beta_root_cols,
                      beta_root_rows,
                      support_overlap_storage,
                      support_size,
                      overlap_resolver,
                      nullptr);
                  ++stats.subdeterminant_evaluations;
                }
                cache_iterator = root_remainder_cache.emplace(
                    cache_key,
                    std::make_pair(alpha_root_determinant, beta_root_determinant)).first;
              }
              alpha_root_determinant = cache_iterator->second.first;
              beta_root_determinant = cache_iterator->second.second;
              if (std::abs(alpha_root_determinant) <= 1.0e-15 ||
                  std::abs(beta_root_determinant) <= 1.0e-15) {
                return;
              }

              int parity = 0;
              parity ^= component_ordered_block_parity(
                  n_alpha_root_rows,
                  selected_alpha_row_masks,
                  right_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  n_alpha_root_cols,
                  selected_alpha_col_masks,
                  left_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  n_beta_root_rows,
                  selected_beta_row_masks,
                  right_leaf_beta_sizes);
              parity ^= component_ordered_block_parity(
                  n_beta_root_cols,
                  selected_beta_col_masks,
                  left_leaf_beta_sizes);

              stats.collapsed_overlap +=
                  left_root_term.coefficient * right_root_term.coefficient *
                  parity_sign(parity) * state_value *
                  alpha_root_determinant * beta_root_determinant;
              if (state_collector != nullptr) {
                // This key is the strongest exact terminal state that the
                // current collapsed star recurrence reaches without falling
                // back to individual determinant-pair identities: root
                // component identity, root determinant terms, and one
                // aggregated separator mask tuple per leaf component.
                ExactSeparatorStateKey state_key;
                state_key.root_left_pairs = root_left_pairs_global;
                state_key.root_right_pairs = root_right_pairs_global;
                state_key.left_root_alpha_occ = left_root_alpha_occ_global;
                state_key.left_root_beta_occ = left_root_beta_occ_global;
                state_key.right_root_alpha_occ = right_root_alpha_occ_global;
                state_key.right_root_beta_occ = right_root_beta_occ_global;
                state_key.leaf_states.reserve(xmvb::to_size(n_leaves));
                for (int selected_leaf_index = 0;
                     selected_leaf_index < n_leaves;
                     ++selected_leaf_index) {
                  const auto& leaf_component =
                      ordered_components[xmvb::to_size(selected_leaf_index + 1)];
                  ExactSeparatorLeafStateKey leaf_key;
                  leaf_key.left_pairs =
                      remap_pairs_to_global_labels(
                          leaf_component.left_pairs,
                          ordered_support_orbitals);
                  leaf_key.right_pairs =
                      remap_pairs_to_global_labels(
                          leaf_component.right_pairs,
                          ordered_support_orbitals);
                  leaf_key.alpha_row_mask =
                      selected_alpha_row_masks[xmvb::to_size(selected_leaf_index)];
                  leaf_key.alpha_col_mask =
                      selected_alpha_col_masks[xmvb::to_size(selected_leaf_index)];
                  leaf_key.beta_row_mask =
                      selected_beta_row_masks[xmvb::to_size(selected_leaf_index)];
                  leaf_key.beta_col_mask =
                      selected_beta_col_masks[xmvb::to_size(selected_leaf_index)];
                  state_key.leaf_states.push_back(std::move(leaf_key));
                }
                state_collector->unique_exact_separator_states.insert(std::move(state_key));
              }
              return;
            }

            for (const auto& entry :
                 collapsed_leaf_messages[xmvb::to_size(leaf_index)]) {
              if ((used_alpha_row_mask & entry.alpha_row_mask) != 0U ||
                  (used_alpha_col_mask & entry.alpha_col_mask) != 0U ||
                  (used_beta_row_mask & entry.beta_row_mask) != 0U ||
                  (used_beta_col_mask & entry.beta_col_mask) != 0U) {
                continue;
              }

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] =
                  entry.alpha_row_mask;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] =
                  entry.alpha_col_mask;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] =
                  entry.beta_row_mask;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] =
                  entry.beta_col_mask;

              ++stats.dp_transition_count;
              self(
                  self,
                  leaf_index + 1,
                  used_alpha_row_mask | entry.alpha_row_mask,
                  used_alpha_col_mask | entry.alpha_col_mask,
                  used_beta_row_mask | entry.beta_row_mask,
                  used_beta_col_mask | entry.beta_col_mask,
                  state_value * entry.value);

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] = 0U;
            }
          };
      accumulate_collapsed_messages(
          accumulate_collapsed_messages,
          0,
          0U,
          0U,
          0U,
          0U,
          1.0);
    }
  }

  stats.absolute_error = std::abs(stats.exact_overlap - stats.collapsed_overlap);
  return stats;
}

Matrix build_full_active_overlap_matrix(
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals) {
  Matrix overlap(n_active_orbitals, n_active_orbitals);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      overlap(row, column) =
          active_overlap_storage[xmvb::to_size(column) *
                                     xmvb::to_size(n_active_orbitals) +
                                 row];
    }
  }
  return overlap;
}

std::string format_ratio(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0) {
    return "0";
  }
  return std::to_string(numerator) + "/" + std::to_string(denominator);
}

std::uint64_t integer_power(std::uint64_t base, int exponent) {
  if (exponent < 0) {
    throw std::invalid_argument("exponent must be non-negative");
  }
  std::uint64_t result = 1;
  for (int power = 0; power < exponent; ++power) {
    result *= base;
  }
  return result;
}

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

int count_covalent_pairs(const std::vector<OrbitalPair>& pairs) {
  int count = 0;
  for (const auto& [left_orbital, right_orbital] : pairs) {
    if (left_orbital != right_orbital) {
      ++count;
    }
  }
  return count;
}

Matrix build_signed_selector_matrix(
    const std::vector<OrbitalPair>& pairs,
    std::uint64_t sign_mask,
    bool is_alpha,
    int support_size) {
  Matrix selector = Matrix::Zero(static_cast<int>(pairs.size()), support_size);
  int covalent_index = 0;
  for (int pair_index = 0; pair_index < static_cast<int>(pairs.size()); ++pair_index) {
    const auto& [left_orbital, right_orbital] = pairs[xmvb::to_size(pair_index)];
    if (left_orbital == right_orbital) {
      selector(pair_index, left_orbital) = 1.0;
      continue;
    }
    const double sign =
        ((sign_mask & (static_cast<std::uint64_t>(1) << covalent_index)) == 0U) ? -1.0 : 1.0;
    if (is_alpha) {
      selector(pair_index, left_orbital) = 1.0;
      selector(pair_index, right_orbital) = sign;
    } else {
      selector(pair_index, right_orbital) = 1.0;
      selector(pair_index, left_orbital) = sign;
    }
    ++covalent_index;
  }
  return selector;
}

Matrix build_basis_selector_matrix(
    const std::vector<int>& occupied_orbitals,
    int support_size) {
  Matrix selector = Matrix::Zero(static_cast<int>(occupied_orbitals.size()), support_size);
  for (int selector_row = 0;
       selector_row < static_cast<int>(occupied_orbitals.size());
       ++selector_row) {
    selector(selector_row, occupied_orbitals[xmvb::to_size(selector_row)]) = 1.0;
  }
  return selector;
}

Matrix select_selector_rows_by_mask(
    const Matrix& full_selector,
    std::uint32_t mask) {
  Matrix selected_selector(popcount(mask), full_selector.cols());
  int selected_row = 0;
  for (int row_index = 0; row_index < full_selector.rows(); ++row_index) {
    if ((mask & (static_cast<std::uint32_t>(1) << row_index)) == 0U) {
      continue;
    }
    selected_selector.row(selected_row) = full_selector.row(row_index);
    ++selected_row;
  }
  return selected_selector;
}

double determinant_with_zeroed_selected_root_block_from_selectors(
    const Matrix& left_root_selector,
    const Matrix& left_leaf_selector,
    const Matrix& right_root_selector,
    const Matrix& right_leaf_selector,
    const Matrix& support_overlap,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  const int n_root_rows = right_root_selector.rows();
  const int n_root_cols = left_root_selector.rows();
  const int n_leaf_rows = right_leaf_selector.rows();
  const int n_leaf_cols = left_leaf_selector.rows();
  const int dimension = n_root_rows + n_leaf_rows;
  if (dimension != n_root_cols + n_leaf_cols) {
    return 0.0;
  }

  Matrix right_selector(dimension, support_overlap.rows());
  Matrix left_selector(dimension, support_overlap.cols());
  right_selector.setZero();
  left_selector.setZero();
  if (n_root_rows > 0) {
    right_selector.topRows(n_root_rows) = right_root_selector;
  }
  if (n_leaf_rows > 0) {
    right_selector.bottomRows(n_leaf_rows) = right_leaf_selector;
  }
  if (n_root_cols > 0) {
    left_selector.topRows(n_root_cols) = left_root_selector;
  }
  if (n_leaf_cols > 0) {
    left_selector.bottomRows(n_leaf_cols) = left_leaf_selector;
  }

  Matrix overlap_submatrix = right_selector * support_overlap * left_selector.transpose();
  if (n_root_rows > 0 && n_root_cols > 0) {
    overlap_submatrix.topLeftCorner(n_root_rows, n_root_cols).setZero();
  }
  return overlap_resolver
      .resolve(flatten_column_major_matrix(overlap_submatrix), dimension)
      .overlap_determinant;
}

double determinant_of_dense_matrix(
    const Matrix& matrix,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // `matrix` is the explicit dense overlap/minor matrix whose determinant is
  // needed by the separator recurrence. In the collapsed Schur path these
  // minors are tiny root-mask blocks, so evaluating them by direct closed
  // forms avoids the overhead of dispatching through the generic determinant
  // resolver for 1x1, 2x2, and 3x3 cases.
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  ++(*subdeterminant_evaluations);
  switch (matrix.rows()) {
    case 0:
      return 1.0;
    case 1:
      return matrix(0, 0);
    case 2:
      return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
    case 3:
      return matrix(0, 0) *
                 (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
             matrix(0, 1) *
                 (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
             matrix(0, 2) *
                 (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
    default:
      break;
  }
  return overlap_resolver
      .resolve(flatten_column_major_matrix(matrix), matrix.rows())
      .overlap_determinant;
}

std::vector<SpinMaskDeterminantEntry> build_spin_mask_determinants_from_selectors(
    const Matrix& left_root_full_selector,
    const Matrix& left_leaf_selector,
    const Matrix& right_root_full_selector,
    const Matrix& right_leaf_selector,
    const Matrix& support_overlap,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* schur_fast_spin_assignment_count,
    std::uint64_t* rectangular_fallback_spin_assignment_count,
    std::uint64_t* singular_fallback_spin_assignment_count,
    std::uint64_t* schur_fast_subdeterminant_evaluation_count,
    std::uint64_t* rectangular_fallback_subdeterminant_evaluation_count,
    std::uint64_t* singular_fallback_subdeterminant_evaluation_count) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (schur_fast_spin_assignment_count == nullptr ||
      rectangular_fallback_spin_assignment_count == nullptr ||
      singular_fallback_spin_assignment_count == nullptr ||
      schur_fast_subdeterminant_evaluation_count == nullptr ||
      rectangular_fallback_subdeterminant_evaluation_count == nullptr ||
      singular_fallback_subdeterminant_evaluation_count == nullptr) {
    throw std::invalid_argument("collapsed spin-assignment counters must not be null");
  }

  const int n_root_rows = right_root_full_selector.rows();
  const int n_root_cols = left_root_full_selector.rows();
  const int n_leaf_rows = right_leaf_selector.rows();
  const int n_leaf_cols = left_leaf_selector.rows();

  if (n_leaf_rows == 0 && n_leaf_cols == 0) {
    std::vector<SpinMaskDeterminantEntry> entries;
    entries.push_back({0U, 0U, 1.0});
    return entries;
  }

  if (n_leaf_rows != n_leaf_cols) {
    // Rectangular leaf blocks cannot use the square leaf-leaf Schur collapse.
    // We therefore fall back to the exact zeroed-root-block determinant for
    // every compatible root mask pair.
    ++(*rectangular_fallback_spin_assignment_count);
    const std::uint64_t branch_start = *subdeterminant_evaluations;
    std::vector<SpinMaskDeterminantEntry> entries;
    for (std::uint32_t row_mask = 0;
         row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
         ++row_mask) {
      const int row_count = popcount(row_mask);
      for (std::uint32_t col_mask = 0;
           col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
           ++col_mask) {
        const int col_count = popcount(col_mask);
        if (row_count + n_leaf_rows != col_count + n_leaf_cols) {
          continue;
        }
        const double determinant =
            determinant_with_zeroed_selected_root_block_from_selectors(
                select_selector_rows_by_mask(left_root_full_selector, col_mask),
                left_leaf_selector,
                select_selector_rows_by_mask(right_root_full_selector, row_mask),
                right_leaf_selector,
                support_overlap,
                overlap_resolver);
        ++(*subdeterminant_evaluations);
        if (std::abs(determinant) <= 1.0e-15) {
          continue;
        }
        entries.push_back({row_mask, col_mask, determinant});
      }
    }
    *rectangular_fallback_subdeterminant_evaluation_count +=
        (*subdeterminant_evaluations - branch_start);
    return entries;
  }

  const int n_leaf = n_leaf_cols;
  const Matrix leaf_leaf_block =
      right_leaf_selector * support_overlap * left_leaf_selector.transpose();
  const auto leaf_leaf_result = overlap_resolver.resolve(
      flatten_column_major_matrix(leaf_leaf_block),
      n_leaf);
  ++(*subdeterminant_evaluations);

  if (std::abs(leaf_leaf_result.overlap_determinant) <= 1.0e-15 ||
      leaf_leaf_result.inverse_overlap_submatrix.rows() != n_leaf ||
      leaf_leaf_result.inverse_overlap_submatrix.cols() != n_leaf) {
    // Singular square leaf blocks also lose the Schur path. This branch keeps
    // the recurrence exact, but it pays the full masked zeroed-block
    // determinant cost for the current spin assignment.
    ++(*singular_fallback_spin_assignment_count);
    const std::uint64_t branch_start = *subdeterminant_evaluations - 1U;
    std::vector<SpinMaskDeterminantEntry> entries;
    for (std::uint32_t row_mask = 0;
         row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
         ++row_mask) {
      for (std::uint32_t col_mask = 0;
           col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
           ++col_mask) {
        if (popcount(row_mask) != popcount(col_mask)) {
          continue;
        }
        const double determinant =
            determinant_with_zeroed_selected_root_block_from_selectors(
                select_selector_rows_by_mask(left_root_full_selector, col_mask),
                left_leaf_selector,
                select_selector_rows_by_mask(right_root_full_selector, row_mask),
                right_leaf_selector,
                support_overlap,
                overlap_resolver);
        ++(*subdeterminant_evaluations);
        if (std::abs(determinant) <= 1.0e-15) {
          continue;
        }
        entries.push_back({row_mask, col_mask, determinant});
      }
    }
    *singular_fallback_subdeterminant_evaluation_count +=
        (*subdeterminant_evaluations - branch_start);
    return entries;
  }

  const Matrix root_to_leaf_block =
      right_root_full_selector * support_overlap * left_leaf_selector.transpose();
  const Matrix leaf_to_root_block =
      right_leaf_selector * support_overlap * left_root_full_selector.transpose();
  Matrix schur_mask_matrix =
      -(root_to_leaf_block * leaf_leaf_result.inverse_overlap_submatrix * leaf_to_root_block);
  ++(*schur_fast_spin_assignment_count);
  const std::uint64_t branch_start = *subdeterminant_evaluations - 1U;

  std::vector<SpinMaskDeterminantEntry> entries;
  for (std::uint32_t row_mask = 0;
       row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
       ++row_mask) {
    const int row_count = popcount(row_mask);
    for (std::uint32_t col_mask = 0;
         col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
         ++col_mask) {
      const int col_count = popcount(col_mask);
      if (row_count != col_count) {
        continue;
      }
      if (row_count == 0) {
        entries.push_back({row_mask, col_mask, leaf_leaf_result.overlap_determinant});
        continue;
      }

      Matrix minor(row_count, col_count);
      int minor_row = 0;
      for (int row_index = 0; row_index < n_root_rows; ++row_index) {
        if ((row_mask & (static_cast<std::uint32_t>(1) << row_index)) == 0U) {
          continue;
        }
        int minor_col = 0;
        for (int col_index = 0; col_index < n_root_cols; ++col_index) {
          if ((col_mask & (static_cast<std::uint32_t>(1) << col_index)) == 0U) {
            continue;
          }
          minor(minor_row, minor_col) = schur_mask_matrix(row_index, col_index);
          ++minor_col;
        }
        ++minor_row;
      }

      const double determinant =
          leaf_leaf_result.overlap_determinant *
          determinant_of_dense_matrix(minor, overlap_resolver, subdeterminant_evaluations);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      entries.push_back({row_mask, col_mask, determinant});
    }
  }
  *schur_fast_subdeterminant_evaluation_count +=
      (*subdeterminant_evaluations - branch_start);
  return entries;
}

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes) {
  // After component ordering, the canonicalization sign for one spin channel
  // depends only on how many selected root orbitals are inserted ahead of each
  // leaf block and how many root orbitals remain for the final remainder
  // block. The detailed leaf occupied-orbital identities no longer enter.
  if (selected_masks.size() != leaf_occ_sizes.size()) {
    throw std::invalid_argument("selected_masks and leaf_occ_sizes must have the same length");
  }
  std::vector<int> block_order;
  block_order.reserve(
      xmvb::to_size(n_root_occ) +
      std::accumulate(leaf_occ_sizes.begin(), leaf_occ_sizes.end(), 0));

  std::uint32_t used_mask = 0U;
  int next_leaf_label = n_root_occ;
  for (std::size_t leaf_index = 0; leaf_index < selected_masks.size(); ++leaf_index) {
    const std::uint32_t mask = selected_masks[leaf_index];
    used_mask |= mask;
    for (int root_position = 0; root_position < n_root_occ; ++root_position) {
      if ((mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
        block_order.push_back(root_position);
      }
    }
    for (int leaf_position = 0;
         leaf_position < leaf_occ_sizes[leaf_index];
         ++leaf_position) {
      block_order.push_back(next_leaf_label++);
    }
  }

  const std::uint32_t full_mask =
      (n_root_occ == 0) ? 0U
                        : ((static_cast<std::uint32_t>(1) << n_root_occ) - 1U);
  const std::uint32_t remainder_mask = full_mask ^ used_mask;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((remainder_mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

std::vector<CollapsedLeafMessageEntry> build_collapsed_leaf_messages_component_ordered(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    CollapsedStarPairStats* stats,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* hypercube_assignment_count) {
  // First-stage leaf-message collapse: for one fixed root determinant-term
  // pair, sum every exact local leaf contribution that lands on the same
  // separator mask tuple
  //   (alpha_row_mask, alpha_col_mask, beta_row_mask, beta_col_mask).
  // The returned message list therefore carries only boundary-state data and a
  // scalar value, not explicit leaf occupied-orbital lists.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (hypercube_assignment_count == nullptr) {
    throw std::invalid_argument("hypercube_assignment_count must not be null");
  }
  if (stats == nullptr) {
    throw std::invalid_argument("collapsed stats must not be null");
  }

  const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
  const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
  const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
  const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());
  const int n_left_covariant_labels = count_covalent_pairs(leaf_component.left_pairs);
  const int n_right_covariant_labels = count_covalent_pairs(leaf_component.right_pairs);
  const std::uint64_t n_left_sign_assignments =
      static_cast<std::uint64_t>(1) << n_left_covariant_labels;
  const std::uint64_t n_right_sign_assignments =
      static_cast<std::uint64_t>(1) << n_right_covariant_labels;
  const Eigen::Map<const Matrix> support_overlap(
      support_overlap_storage.data(),
      support_size,
      support_size);
  const Matrix left_alpha_root_full_selector =
      build_basis_selector_matrix(left_root_term.alpha_occ, support_size);
  const Matrix right_alpha_root_full_selector =
      build_basis_selector_matrix(right_root_term.alpha_occ, support_size);
  const Matrix left_beta_root_full_selector =
      build_basis_selector_matrix(left_root_term.beta_occ, support_size);
  const Matrix right_beta_root_full_selector =
      build_basis_selector_matrix(right_root_term.beta_occ, support_size);

  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>, double>
      aggregated_messages;
  for (std::uint64_t left_sign_mask = 0;
       left_sign_mask < n_left_sign_assignments;
       ++left_sign_mask) {
    const Matrix left_alpha_selector =
        build_signed_selector_matrix(leaf_component.left_pairs, left_sign_mask, true, support_size);
    const Matrix left_beta_selector =
        build_signed_selector_matrix(leaf_component.left_pairs, left_sign_mask, false, support_size);
    for (std::uint64_t right_sign_mask = 0;
         right_sign_mask < n_right_sign_assignments;
         ++right_sign_mask) {
      ++(*hypercube_assignment_count);
      const Matrix right_alpha_selector =
          build_signed_selector_matrix(
              leaf_component.right_pairs,
              right_sign_mask,
              true,
              support_size);
      const Matrix right_beta_selector =
          build_signed_selector_matrix(
              leaf_component.right_pairs,
              right_sign_mask,
              false,
              support_size);
      const auto alpha_entries = build_spin_mask_determinants_from_selectors(
          left_alpha_root_full_selector,
          left_alpha_selector,
          right_alpha_root_full_selector,
          right_alpha_selector,
          support_overlap,
          overlap_resolver,
          subdeterminant_evaluations,
          &stats->schur_fast_spin_assignment_count,
          &stats->rectangular_fallback_spin_assignment_count,
          &stats->singular_fallback_spin_assignment_count,
          &stats->schur_fast_subdeterminant_evaluation_count,
          &stats->rectangular_fallback_subdeterminant_evaluation_count,
          &stats->singular_fallback_subdeterminant_evaluation_count);
      const auto beta_entries = build_spin_mask_determinants_from_selectors(
          left_beta_root_full_selector,
          left_beta_selector,
          right_beta_root_full_selector,
          right_beta_selector,
          support_overlap,
          overlap_resolver,
          subdeterminant_evaluations,
          &stats->schur_fast_spin_assignment_count,
          &stats->rectangular_fallback_spin_assignment_count,
          &stats->singular_fallback_spin_assignment_count,
          &stats->schur_fast_subdeterminant_evaluation_count,
          &stats->rectangular_fallback_subdeterminant_evaluation_count,
          &stats->singular_fallback_subdeterminant_evaluation_count);

      for (const auto& alpha_entry : alpha_entries) {
        for (const auto& beta_entry : beta_entries) {
          aggregated_messages[{
              alpha_entry.row_mask,
              alpha_entry.col_mask,
              beta_entry.row_mask,
              beta_entry.col_mask}] +=
              alpha_entry.value * beta_entry.value;
        }
      }
    }
  }

  const double average_factor =
      1.0 / static_cast<double>(
                static_cast<std::uint64_t>(1)
                << (n_left_covariant_labels + n_right_covariant_labels));
  for (auto& [key, value] : aggregated_messages) {
    value *= average_factor;
  }

  std::vector<CollapsedLeafMessageEntry> entries;
  entries.reserve(aggregated_messages.size());
  for (const auto& [key, value] : aggregated_messages) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    CollapsedLeafMessageEntry entry;
    entry.alpha_row_mask = std::get<0>(key);
    entry.alpha_col_mask = std::get<1>(key);
    entry.beta_row_mask = std::get<2>(key);
    entry.beta_col_mask = std::get<3>(key);
    entry.value = value;
    entries.push_back(std::move(entry));
  }
  return entries;
}

void record_reference_determinant_work(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    ExactWorkCollector* work_collector) {
  if (work_collector == nullptr) {
    return;
  }
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      work_collector->reference_full_determinant_pairs.insert(
          {left_term.alpha_occ,
           left_term.beta_occ,
           right_term.alpha_occ,
           right_term.beta_occ});
      work_collector->reference_spin_determinants.insert(
          {left_term.alpha_occ, right_term.alpha_occ});
      work_collector->reference_spin_determinants.insert(
          {left_term.beta_occ, right_term.beta_occ});
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto started_at = std::chrono::steady_clock::now();
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    const auto overlap_selection = select_active_overlap_matrix(options, load_result);
    const auto& active_overlap_storage = overlap_selection.active_overlap_matrix;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "analyze_star_separator_overlap_dataset currently supports only singlet closed-shell structures");
    }

    std::vector<PerStructureCache> structure_cache(
        xmvb::to_size(raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[xmvb::to_size(structure_index)];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      cache.legacy_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs);
      cache.coefficient_lookup = build_coefficient_lookup(cache.legacy_terms);
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.pair_order,
        options.seed,
        options.max_pairs,
        options.filter_left_structure,
        options.filter_right_structure);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    const Matrix full_active_overlap =
        build_full_active_overlap_matrix(active_overlap_storage, n_active_orbitals);
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    xmvb::vb::MetricAwareComponentGraphOptions graph_options;
    graph_options.edge_max_abs_threshold = options.edge_max_abs_threshold;

    int covered_pair_count = 0;
    int exact_match_count = 0;
    int mismatch_count = 0;
    double max_abs_error = 0.0;
    int collapsed_exact_match_count = 0;
    int collapsed_mismatch_count = 0;
    double collapsed_max_abs_error = 0.0;
    std::uint64_t total_reference_determinant_pair_count = 0;
    std::uint64_t total_local_term_pair_visits = 0;
    std::uint64_t total_subdeterminant_evaluations = 0;
    std::uint64_t total_dp_transition_count = 0;
    std::uint64_t total_binary_separator_state_count = 0;
    std::uint64_t total_ternary_separator_state_count = 0;
    std::uint64_t total_collapsed_leaf_state_count = 0;
    std::uint64_t total_collapsed_hypercube_assignment_count = 0;
    std::uint64_t total_collapsed_schur_fast_spin_assignment_count = 0;
    std::uint64_t total_collapsed_rectangular_fallback_spin_assignment_count = 0;
    std::uint64_t total_collapsed_singular_fallback_spin_assignment_count = 0;
    std::uint64_t total_collapsed_schur_fast_subdeterminant_evaluation_count = 0;
    std::uint64_t total_collapsed_rectangular_fallback_subdeterminant_evaluation_count = 0;
    std::uint64_t total_collapsed_singular_fallback_subdeterminant_evaluation_count = 0;
    std::uint64_t total_collapsed_subdeterminant_evaluations = 0;
    std::uint64_t total_collapsed_dp_transition_count = 0;
    std::uint64_t total_coefficient_factorization_checks = 0;
    std::uint64_t total_coefficient_factorization_mismatches = 0;
    double max_coefficient_factorization_abs_error = 0.0;
    ExactWorkCollector work_collector;
    ExactSeparatorStateCollector exact_separator_state_collector;
    std::map<int, int> covered_width_histogram;
    std::map<int, int> covered_node_count_histogram;
    std::vector<PairExample> examples;

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
      if (metric_graph.node_count == 2) {
        const int node0_size = static_cast<int>(union_components[0].active_orbitals.size());
        const int node1_size = static_cast<int>(union_components[1].active_orbitals.size());
        if (node1_size < node0_size) {
          root_node = 1;
        }
      }

      std::vector<ComponentData> components(
          xmvb::to_size(metric_graph.node_count));
      for (int graph_node = 0; graph_node < metric_graph.node_count; ++graph_node) {
        auto& component = components[xmvb::to_size(graph_node)];
        component = build_component_data(graph_node, union_components, support_orbitals);
        component.left_orientation_terms =
            enumerate_orientation_terms(component.left_pairs);
        component.right_orientation_terms =
            enumerate_orientation_terms(component.right_pairs);
      }
      if (metric_graph.node_count == 2) {
        const auto estimate_local_term_visits = [&](int candidate_root) {
          const int candidate_leaf = 1 - candidate_root;
          const auto& root_component = components[xmvb::to_size(candidate_root)];
          const auto& leaf_component = components[xmvb::to_size(candidate_leaf)];
          const std::uint64_t root_term_pairs =
              static_cast<std::uint64_t>(root_component.left_orientation_terms.size()) *
              static_cast<std::uint64_t>(root_component.right_orientation_terms.size());
          const std::uint64_t leaf_term_pairs =
              static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
              static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
          return root_term_pairs * (1ULL + leaf_term_pairs);
        };
        const std::uint64_t root0_visits = estimate_local_term_visits(0);
        const std::uint64_t root1_visits = estimate_local_term_visits(1);
        if (root1_visits < root0_visits) {
          root_node = 1;
        } else if (root0_visits < root1_visits) {
          root_node = 0;
        }
      }

      const double exact_overlap = xmvb::vb::legacy_structure_overlap(
          left_cache.legacy_terms,
          right_cache.legacy_terms,
          full_active_overlap,
          overlap_resolver);
      const std::uint64_t reference_count =
          static_cast<std::uint64_t>(left_cache.legacy_terms.size()) *
          static_cast<std::uint64_t>(right_cache.legacy_terms.size());
      record_reference_determinant_work(
          left_cache.legacy_terms,
          right_cache.legacy_terms,
          &work_collector);
      const StarPairStats stats = evaluate_star_pair(
          exact_overlap,
          reference_count,
          active_overlap_storage,
          n_active_orbitals,
          metric_summary,
          metric_graph,
          components,
          root_node,
          left_cache.coefficient_lookup,
          right_cache.coefficient_lookup,
          overlap_resolver,
          &work_collector);

      const auto component_ordered_support_orbitals =
          build_component_ordered_support_orbitals(
              support_orbitals,
              union_components,
              root_node,
              metric_graph.adjacency[xmvb::to_size(root_node)]);
      const auto component_ordered_support_index =
          xmvb::vb::build_support_index(component_ordered_support_orbitals);
      const auto component_ordered_left_pairs_local =
          xmvb::vb::remap_pairs_to_support(
              left_cache.active_pairs,
              component_ordered_support_index);
      const auto component_ordered_right_pairs_local =
          xmvb::vb::remap_pairs_to_support(
              right_cache.active_pairs,
              component_ordered_support_index);
      const auto component_ordered_support_overlap =
          xmvb::vb::build_support_overlap_matrix(
              component_ordered_support_orbitals,
              active_overlap_storage,
              n_active_orbitals);
      const auto component_ordered_support_overlap_storage =
          flatten_column_major_matrix(component_ordered_support_overlap);
      const auto component_ordered_union_components =
          xmvb::vb::build_union_graph_components(
              component_ordered_left_pairs_local,
              component_ordered_right_pairs_local,
              component_ordered_support_orbitals);
      if (component_ordered_union_components.size() != union_components.size()) {
        throw std::runtime_error(
            "component-ordered support relabeling changed the union-component count");
      }

      std::vector<ComponentData> component_ordered_components;
      component_ordered_components.reserve(component_ordered_union_components.size());
      for (std::size_t component_index = 0;
           component_index < component_ordered_union_components.size();
           ++component_index) {
        auto component = build_local_component_data(
            static_cast<int>(component_index),
            component_ordered_union_components);
        component.left_orientation_terms =
            enumerate_orientation_terms(component.left_pairs);
        component.right_orientation_terms =
            enumerate_orientation_terms(component.right_pairs);
        component_ordered_components.push_back(std::move(component));
      }

      const auto component_ordered_left_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(
              component_ordered_left_pairs_local);
      const auto component_ordered_right_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(
              component_ordered_right_pairs_local);
      const double component_ordered_exact_overlap = xmvb::vb::legacy_structure_overlap(
          component_ordered_left_terms,
          component_ordered_right_terms,
          component_ordered_support_overlap,
          overlap_resolver);
      const auto collapsed_stats = evaluate_component_ordered_collapsed_star_pair(
          component_ordered_exact_overlap,
          component_ordered_support_overlap_storage,
          component_ordered_support_overlap.rows(),
          component_ordered_components,
          component_ordered_support_orbitals,
          overlap_resolver,
          &exact_separator_state_collector);

      ++covered_pair_count;
      ++covered_width_histogram[stats.metric_width_upper_bound];
      ++covered_node_count_histogram[stats.node_count];
      total_reference_determinant_pair_count += stats.reference_determinant_pair_count;
      total_local_term_pair_visits += stats.local_term_pair_visits;
      total_subdeterminant_evaluations += stats.subdeterminant_evaluations;
      total_dp_transition_count += stats.dp_transition_count;
      total_binary_separator_state_count +=
          integer_power(2, stats.metric_width_upper_bound);
      total_ternary_separator_state_count +=
          integer_power(3, stats.metric_width_upper_bound);
      total_collapsed_leaf_state_count +=
          collapsed_stats.collapsed_leaf_state_count;
      total_collapsed_hypercube_assignment_count +=
          collapsed_stats.hypercube_assignment_count;
      total_collapsed_schur_fast_spin_assignment_count +=
          collapsed_stats.schur_fast_spin_assignment_count;
      total_collapsed_rectangular_fallback_spin_assignment_count +=
          collapsed_stats.rectangular_fallback_spin_assignment_count;
      total_collapsed_singular_fallback_spin_assignment_count +=
          collapsed_stats.singular_fallback_spin_assignment_count;
      total_collapsed_schur_fast_subdeterminant_evaluation_count +=
          collapsed_stats.schur_fast_subdeterminant_evaluation_count;
      total_collapsed_rectangular_fallback_subdeterminant_evaluation_count +=
          collapsed_stats.rectangular_fallback_subdeterminant_evaluation_count;
      total_collapsed_singular_fallback_subdeterminant_evaluation_count +=
          collapsed_stats.singular_fallback_subdeterminant_evaluation_count;
      total_collapsed_subdeterminant_evaluations +=
          collapsed_stats.subdeterminant_evaluations;
      total_collapsed_dp_transition_count +=
          collapsed_stats.dp_transition_count;
      total_coefficient_factorization_checks +=
          stats.coefficient_factorization_checks;
      total_coefficient_factorization_mismatches +=
          stats.coefficient_factorization_mismatches;
      max_coefficient_factorization_abs_error = std::max(
          max_coefficient_factorization_abs_error,
          stats.max_coefficient_factorization_abs_error);
      max_abs_error = std::max(max_abs_error, stats.absolute_error);
      collapsed_max_abs_error = std::max(
          collapsed_max_abs_error,
          collapsed_stats.absolute_error);
      if (stats.absolute_error <= options.tolerance) {
        ++exact_match_count;
      } else {
        ++mismatch_count;
      }
      if (collapsed_stats.absolute_error <= options.tolerance) {
        ++collapsed_exact_match_count;
      } else {
        ++collapsed_mismatch_count;
      }

      PairExample example;
      example.left_structure = left_structure;
      example.right_structure = right_structure;
      example.node_count = stats.node_count;
      example.root_node = stats.root_node;
      example.metric_width_upper_bound = stats.metric_width_upper_bound;
      example.reference_determinant_pair_count = stats.reference_determinant_pair_count;
      example.local_term_pair_visits = stats.local_term_pair_visits;
      example.subdeterminant_evaluations = stats.subdeterminant_evaluations;
      example.dp_transition_count = stats.dp_transition_count;
      example.exact_overlap = stats.exact_overlap;
      example.star_overlap = stats.star_overlap;
      example.absolute_error = stats.absolute_error;
      examples.push_back(std::move(example));

      if (options.report_every > 0 &&
          (pair_index + 1) % xmvb::to_size(options.report_every) == 0) {
        const auto elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at).count();
        std::cerr << "progress = " << (pair_index + 1) << "/" << pair_list.size()
                  << " covered_pairs = " << covered_pair_count
                  << " elapsed_s = " << std::fixed << std::setprecision(2)
                  << elapsed_seconds << '\n';
      }
    }

    std::sort(
        examples.begin(),
        examples.end(),
        [](const PairExample& left, const PairExample& right) {
          if (left.absolute_error != right.absolute_error) {
            return left.absolute_error > right.absolute_error;
          }
          const auto left_saved =
              static_cast<std::int64_t>(left.reference_determinant_pair_count) -
              static_cast<std::int64_t>(left.local_term_pair_visits);
          const auto right_saved =
              static_cast<std::int64_t>(right.reference_determinant_pair_count) -
              static_cast<std::int64_t>(right.local_term_pair_visits);
          if (left_saved != right_saved) {
            return left_saved > right_saved;
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
    std::cout << "selected_pairs = " << pair_list.size() << '\n';
    std::cout << "pair_order = " << pair_order_name(options.pair_order) << '\n';
    std::cout << "active_overlap_source = "
              << active_overlap_source_name(options.active_overlap_source) << '\n';
    std::cout << "covered_star_pair_count = " << covered_pair_count << '\n';
    std::cout << "exact_match_count = " << exact_match_count << '\n';
    std::cout << "mismatch_count = " << mismatch_count << '\n';
    std::cout << "max_abs_error = " << max_abs_error << '\n';
    std::cout << "collapsed_exact_match_count = "
              << collapsed_exact_match_count << '\n';
    std::cout << "collapsed_mismatch_count = "
              << collapsed_mismatch_count << '\n';
    std::cout << "collapsed_max_abs_error = "
              << collapsed_max_abs_error << '\n';
    if (overlap_selection.optimized_overlap_summary.has_value()) {
      const auto& optimized_summary = overlap_selection.optimized_overlap_summary.value();
      std::cout << "optimizer_converged = " << std::boolalpha
                << optimized_summary.converged << '\n';
      std::cout << "optimizer_termination_reason = "
                << optimized_summary.termination_reason << '\n';
      std::cout << "optimizer_accepted_iterations = "
                << optimized_summary.accepted_iterations << '\n';
      std::cout << "optimizer_objective_evaluations = "
                << optimized_summary.objective_evaluations << '\n';
      std::cout << "optimizer_total_wall_time_seconds = "
                << optimized_summary.total_wall_time_seconds << '\n';
      std::cout << std::noboolalpha;
    }
    std::cout << "covered_width_histogram = {";
    bool first = true;
    for (const auto& [width, count] : covered_width_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << width << ": " << count;
    }
    std::cout << "}\n";
    std::cout << "covered_node_count_histogram = {";
    first = true;
    for (const auto& [node_count, count] : covered_node_count_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << node_count << ": " << count;
    }
    std::cout << "}\n";
    std::cout << "total_reference_determinant_pair_count = "
              << total_reference_determinant_pair_count << '\n';
    std::cout << "unique_reference_determinant_pair_count = "
              << work_collector.reference_full_determinant_pairs.size() << '\n';
    std::cout << "unique_exact_separator_state_count = "
              << exact_separator_state_collector.unique_exact_separator_states.size() << '\n';
    std::cout << "separator_state_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_exact_separator_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_exact_leaf_message_state_count = "
              << exact_separator_state_collector.unique_leaf_message_states.size() << '\n';
    std::cout << "leaf_message_state_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_exact_leaf_message_bundle_count = "
              << exact_separator_state_collector.unique_leaf_message_bundles.size() << '\n';
    std::cout << "leaf_message_bundle_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_bundles.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_exact_merge_state_count = "
              << exact_separator_state_collector.unique_merge_states.size() << '\n';
    std::cout << "merge_state_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_merge_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "layered_state_total_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_states.size() +
                            exact_separator_state_collector.unique_merge_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    const std::size_t unique_separator_spin_kernel_count =
        work_collector.separator_root_spin_determinants.size() +
        work_collector.separator_zeroed_block_spin_determinants.size();
    std::cout << "bundle_merge_spin_kernel_total_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_bundles.size() +
                            exact_separator_state_collector.unique_merge_states.size() +
                            unique_separator_spin_kernel_count) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_reference_spin_determinant_count = "
              << work_collector.reference_spin_determinants.size() << '\n';
    std::cout << "total_binary_separator_state_count = "
              << total_binary_separator_state_count << '\n';
    std::cout << "total_ternary_separator_state_count = "
              << total_ternary_separator_state_count << '\n';
    std::cout << "total_local_term_pair_visits = "
              << total_local_term_pair_visits << '\n';
    std::cout << "total_collapsed_leaf_state_count = "
              << total_collapsed_leaf_state_count << '\n';
    std::cout << "total_collapsed_hypercube_assignment_count = "
              << total_collapsed_hypercube_assignment_count << '\n';
    std::cout << "total_collapsed_schur_fast_spin_assignment_count = "
              << total_collapsed_schur_fast_spin_assignment_count << '\n';
    std::cout << "total_collapsed_rectangular_fallback_spin_assignment_count = "
              << total_collapsed_rectangular_fallback_spin_assignment_count << '\n';
    std::cout << "total_collapsed_singular_fallback_spin_assignment_count = "
              << total_collapsed_singular_fallback_spin_assignment_count << '\n';
    std::cout << "total_collapsed_schur_fast_subdeterminant_evaluation_count = "
              << total_collapsed_schur_fast_subdeterminant_evaluation_count << '\n';
    std::cout << "total_collapsed_rectangular_fallback_subdeterminant_evaluation_count = "
              << total_collapsed_rectangular_fallback_subdeterminant_evaluation_count << '\n';
    std::cout << "total_collapsed_singular_fallback_subdeterminant_evaluation_count = "
              << total_collapsed_singular_fallback_subdeterminant_evaluation_count << '\n';
    std::cout << "total_subdeterminant_evaluations = "
              << total_subdeterminant_evaluations << '\n';
    std::cout << "total_collapsed_subdeterminant_evaluations = "
              << total_collapsed_subdeterminant_evaluations << '\n';
    std::cout << "unique_separator_root_spin_determinant_count = "
              << work_collector.separator_root_spin_determinants.size() << '\n';
    std::cout << "unique_separator_zeroed_block_spin_determinant_count = "
              << work_collector.separator_zeroed_block_spin_determinants.size() << '\n';
    std::cout << "unique_separator_total_spin_determinant_count = "
              << (work_collector.separator_root_spin_determinants.size() +
                  work_collector.separator_zeroed_block_spin_determinants.size())
              << '\n';
    std::cout << "total_dp_transition_count = "
              << total_dp_transition_count << '\n';
    std::cout << "total_collapsed_dp_transition_count = "
              << total_collapsed_dp_transition_count << '\n';
    std::cout << "total_coefficient_factorization_checks = "
              << total_coefficient_factorization_checks << '\n';
    std::cout << "total_coefficient_factorization_mismatches = "
              << total_coefficient_factorization_mismatches << '\n';
    std::cout << "max_coefficient_factorization_abs_error = "
              << max_coefficient_factorization_abs_error << '\n';
    std::cout << "reference_over_binary_state_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "reference_over_ternary_state_ratio = "
              << (total_ternary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_ternary_separator_state_count))
              << '\n';
    std::cout << "reference_over_local_term_pair_ratio = "
              << (total_local_term_pair_visits == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_local_term_pair_visits))
              << '\n';
    std::cout << "reference_pairs_over_unique_reference_pairs_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "reference_spin_over_unique_reference_spin_ratio = "
              << (work_collector.reference_spin_determinants.empty()
                      ? 0.0
                      : (2.0 * static_cast<double>(total_reference_determinant_pair_count)) /
                            static_cast<double>(
                                work_collector.reference_spin_determinants.size()))
              << '\n';
    const std::size_t unique_separator_total_spin_determinant_count =
        work_collector.separator_root_spin_determinants.size() +
        work_collector.separator_zeroed_block_spin_determinants.size();
    std::cout << "separator_spin_over_unique_separator_spin_ratio = "
              << (unique_separator_total_spin_determinant_count == 0
                      ? 0.0
                      : static_cast<double>(total_subdeterminant_evaluations) /
                            static_cast<double>(unique_separator_total_spin_determinant_count))
              << '\n';
    std::cout << "reference_unique_spin_over_separator_unique_spin_ratio = "
              << (unique_separator_total_spin_determinant_count == 0
                      ? 0.0
                      : static_cast<double>(
                            work_collector.reference_spin_determinants.size()) /
                            static_cast<double>(unique_separator_total_spin_determinant_count))
              << '\n';
    std::cout << "local_term_over_binary_state_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_local_term_pair_visits) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "collapsed_leaf_state_over_binary_state_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_collapsed_leaf_state_count) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "reference_over_subdeterminant_ratio = "
              << (total_subdeterminant_evaluations == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_subdeterminant_evaluations))
              << '\n';
    std::cout << "reference_over_collapsed_subdeterminant_ratio = "
              << (total_collapsed_subdeterminant_evaluations == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(
                                total_collapsed_subdeterminant_evaluations))
              << '\n';
    const std::uint64_t total_collapsed_classified_spin_assignment_count =
        total_collapsed_schur_fast_spin_assignment_count +
        total_collapsed_rectangular_fallback_spin_assignment_count +
        total_collapsed_singular_fallback_spin_assignment_count;
    std::cout << "collapsed_schur_fast_spin_assignment_fraction = "
              << (total_collapsed_classified_spin_assignment_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_schur_fast_spin_assignment_count) /
                            static_cast<double>(
                                total_collapsed_classified_spin_assignment_count))
              << '\n';
    std::cout << "collapsed_rectangular_fallback_spin_assignment_fraction = "
              << (total_collapsed_classified_spin_assignment_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_rectangular_fallback_spin_assignment_count) /
                            static_cast<double>(
                                total_collapsed_classified_spin_assignment_count))
              << '\n';
    std::cout << "collapsed_singular_fallback_spin_assignment_fraction = "
              << (total_collapsed_classified_spin_assignment_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_singular_fallback_spin_assignment_count) /
                            static_cast<double>(
                                total_collapsed_classified_spin_assignment_count))
              << '\n';
    const std::uint64_t total_collapsed_classified_subdeterminant_evaluation_count =
        total_collapsed_schur_fast_subdeterminant_evaluation_count +
        total_collapsed_rectangular_fallback_subdeterminant_evaluation_count +
        total_collapsed_singular_fallback_subdeterminant_evaluation_count;
    std::cout << "collapsed_schur_fast_subdeterminant_fraction = "
              << (total_collapsed_classified_subdeterminant_evaluation_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_schur_fast_subdeterminant_evaluation_count) /
                            static_cast<double>(
                                total_collapsed_classified_subdeterminant_evaluation_count))
              << '\n';
    std::cout << "collapsed_rectangular_fallback_subdeterminant_fraction = "
              << (total_collapsed_classified_subdeterminant_evaluation_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_rectangular_fallback_subdeterminant_evaluation_count) /
                            static_cast<double>(
                                total_collapsed_classified_subdeterminant_evaluation_count))
              << '\n';
    std::cout << "collapsed_singular_fallback_subdeterminant_fraction = "
              << (total_collapsed_classified_subdeterminant_evaluation_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_singular_fallback_subdeterminant_evaluation_count) /
                            static_cast<double>(
                                total_collapsed_classified_subdeterminant_evaluation_count))
              << '\n';
    std::cout << "elapsed_wall_time_seconds = " << elapsed_seconds << '\n';

    const int n_examples =
        std::min(options.top_examples, static_cast<int>(examples.size()));
    std::cout << "top_examples\n";
    for (int example_index = 0; example_index < n_examples; ++example_index) {
      const auto& example = examples[xmvb::to_size(example_index)];
      const auto saved =
          static_cast<std::int64_t>(example.reference_determinant_pair_count) -
          static_cast<std::int64_t>(example.local_term_pair_visits);
      std::cout << "example[" << example_index << "]"
                << " left_structure=" << example.left_structure
                << " right_structure=" << example.right_structure
                << " node_count=" << example.node_count
                << " root_node=" << example.root_node
                << " width_upper_bound=" << example.metric_width_upper_bound
                << " reference_pair_count=" << example.reference_determinant_pair_count
                << " local_term_pair_visits=" << example.local_term_pair_visits
                << " saved=" << saved
                << " subdeterminant_evaluations=" << example.subdeterminant_evaluations
                << " dp_transitions=" << example.dp_transition_count
                << " exact_overlap=" << example.exact_overlap
                << " star_overlap=" << example.star_overlap
                << " abs_error=" << example.absolute_error
                << '\n';
    }
    return mismatch_count == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
