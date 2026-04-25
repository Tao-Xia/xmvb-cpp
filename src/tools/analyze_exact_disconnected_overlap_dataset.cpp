#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/union_graph_screening.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"

namespace {

using Pair = std::pair<int, int>;

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
  int max_pairs = 0;
  int report_every = 0;
  int top_examples = 12;
  int mismatch_limit = 0;
  int optimizer_max_iterations = 25;
  std::uint32_t seed = 0;
  double optimizer_gradient_tolerance = 2.0e-3;
  double optimizer_energy_tolerance = 1.0e-7;
  double tolerance = 1.0e-12;
  bool stop_on_first_mismatch = false;
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
  std::vector<xmvb::vb::OrbitalPair> active_pairs;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> determinant_terms;
};

struct PairExample {
  int left_structure = 0;
  int right_structure = 0;
  int metric_connected_component_count = 0;
  int width_upper_bound = 0;
  std::uint64_t full_pair_determinant_count = 0;
  std::uint64_t disconnected_factored_determinant_count = 0;
  double exact_overlap = 0.0;
  double factored_overlap = 0.0;
  double absolute_error = 0.0;
  double compression_ratio = 0.0;
  std::string component_signature;
};

void print_usage() {
  std::cerr << "usage: analyze_exact_disconnected_overlap_dataset <input.xmi>"
               " [--pair-order lexicographic|random]"
               " [--active-overlap-source input|optimized_vbscf]"
               " [--max-pairs N]"
               " [--seed S]"
               " [--report-every N]"
               " [--top-examples N]"
               " [--mismatch-limit N]"
               " [--optimizer-max-iterations N]"
               " [--optimizer-gradient-tolerance F]"
               " [--optimizer-energy-tolerance F]"
               " [--stop-on-first-mismatch true|false]"
               " [--tolerance F]\n";
}

bool parse_bool(const std::string& value) {
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument("unsupported boolean value: " + value);
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
    if (argument_name == "--mismatch-limit") {
      options.mismatch_limit = std::stoi(argument_value);
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
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--stop-on-first-mismatch") {
      options.stop_on_first_mismatch = parse_bool(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
  }
  if (options.top_examples <= 0) {
    throw std::invalid_argument("--top-examples must be positive");
  }
  if (options.mismatch_limit < 0) {
    throw std::invalid_argument("--mismatch-limit must be >= 0");
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
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  return options;
}

std::vector<Pair> build_pair_list(
    int structure_count,
    PairOrder pair_order,
    std::uint32_t seed,
    int max_pairs) {
  if (structure_count < 2) {
    return {};
  }
  std::vector<Pair> pairs;
  pairs.reserve(structure_count *
                structure_count - 1 / 2);
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
    pairs.resize(max_pairs);
  }
  return pairs;
}

std::string build_component_signature(
    const std::vector<xmvb::vb::UnionGraphComponent>& components) {
  std::vector<std::string> labels;
  labels.reserve(components.size());
  for (const auto& component : components) {
    labels.push_back(
        component.type + ":" +
        std::to_string(static_cast<int>(component.active_orbitals.size())));
  }
  std::sort(labels.begin(), labels.end());

  std::string signature;
  for (std::size_t label_index = 0; label_index < labels.size(); ++label_index) {
    if (label_index > 0) {
      signature += "+";
    }
    signature += labels[label_index];
  }
  return signature;
}

std::string format_pairs(
    const std::vector<xmvb::vb::OrbitalPair>& pairs) {
  std::string result = "[";
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pair_index > 0) {
      result += " ";
    }
    result += "(" + std::to_string(pairs[pair_index].first + 1) + "," +
              std::to_string(pairs[pair_index].second + 1) + ")";
  }
  result += "]";
  return result;
}

std::string format_indices(
    const std::vector<int>& indices) {
  std::string result = "[";
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index > 0) {
      result += " ";
    }
    result += std::to_string(indices[index]);
  }
  result += "]";
  return result;
}

void print_connected_component_details(
    const std::vector<xmvb::vb::MetricAwareConnectedComponent>& connected_components,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    const std::vector<int>& support_orbitals) {
  for (std::size_t component_index = 0;
       component_index < connected_components.size();
       ++component_index) {
    const auto& connected_component = connected_components[component_index];
    std::cerr << "  metric_component[" << component_index << "]"
              << " graph_nodes=" << format_indices(connected_component.graph_nodes)
              << " total_active_orbitals=" << connected_component.total_active_orbitals
              << " total_left_pairs=" << connected_component.total_left_pairs
              << " total_right_pairs=" << connected_component.total_right_pairs
              << " total_covalent_labels=" << connected_component.total_covalent_labels
              << '\n';
    for (const int graph_node : connected_component.graph_nodes) {
      const auto& union_component = union_components[graph_node];
      std::vector<int> active_orbitals_one_based;
      active_orbitals_one_based.reserve(union_component.local_vertices.size());
      for (const int local_vertex : union_component.local_vertices) {
        active_orbitals_one_based.push_back(
            support_orbitals[local_vertex] + 1);
      }
      std::cerr << "    union_component[" << graph_node << "]"
                << " type=" << union_component.type
                << " active_orbitals=" << format_indices(active_orbitals_one_based)
                << " left_pairs=" << format_pairs(union_component.left_pairs)
                << " right_pairs=" << format_pairs(union_component.right_pairs)
                << '\n';
    }
  }
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

  // For the optimized-orbital mode we intentionally reuse the determinant-VBSCF
  // optimizer, because the purpose of this tool is to measure whether the exact
  // graph-disconnected special case survives on the same orbitals used in a real
  // SCF trajectory.
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

xmvb::vb::Matrix build_full_active_overlap_matrix(
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals) {
  xmvb::vb::Matrix active_overlap(n_active_orbitals, n_active_orbitals);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      active_overlap(row, column) =
          active_overlap_storage[column *
                                     n_active_orbitals +
                                 row];
    }
  }
  return active_overlap;
}

std::vector<xmvb::vb::OrbitalPair> build_original_pairs_for_metric_component(
    const std::vector<int>& graph_nodes,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    const std::vector<int>& support_orbitals,
    bool use_left_pairs) {
  std::vector<xmvb::vb::OrbitalPair> pairs;
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
    const auto started_at = std::chrono::steady_clock::now();
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    const auto overlap_selection = select_active_overlap_matrix(options, load_result);
    const auto& active_overlap_storage = overlap_selection.active_overlap_matrix;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "analyze_exact_disconnected_overlap_dataset currently supports only singlet closed-shell structures");
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.pair_order,
        options.seed,
        options.max_pairs);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    // Cache exact determinant terms for each raw structure once. These are the
    // unique determinant terms entering the exact structure-pair overlap
    // expansion for that single raw structure.
    std::vector<PerStructureCache> structure_cache(
        raw_structure_data.n_structures);
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[structure_index];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      cache.determinant_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs);
    }

    const xmvb::vb::Matrix full_active_overlap =
        build_full_active_overlap_matrix(active_overlap_storage, n_active_orbitals);
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;

    std::uint64_t total_full_pair_determinant_count = 0;
    std::uint64_t total_hybrid_pair_determinant_count = 0;
    std::uint64_t disconnected_full_pair_determinant_count = 0;
    std::uint64_t disconnected_factored_pair_determinant_count = 0;
    int disconnected_pair_count = 0;
    int disconnected_pair_better_count = 0;
    int disconnected_pair_equal_count = 0;
    int disconnected_pair_worse_count = 0;
    int exact_match_count = 0;
    int mismatch_count = 0;
    int printed_mismatch_count = 0;
    bool stop_requested = false;
    double max_abs_error = 0.0;
    std::map<int, int> disconnected_component_count_histogram;
    std::map<int, int> disconnected_width_histogram;
    std::map<int, int> mismatch_width_histogram;
    std::vector<PairExample> examples;

    std::size_t processed_pair_count = 0;
    for (std::size_t pair_index = 0; pair_index < pair_list.size(); ++pair_index) {
      const auto [left_structure, right_structure] = pair_list[pair_index];
      processed_pair_count = pair_index + 1;
      const auto& left_cache = structure_cache[left_structure];
      const auto& right_cache = structure_cache[right_structure];
      const std::uint64_t full_pair_determinant_count =
          static_cast<std::uint64_t>(left_cache.determinant_terms.size()) *
          static_cast<std::uint64_t>(right_cache.determinant_terms.size());
      total_full_pair_determinant_count += full_pair_determinant_count;
      total_hybrid_pair_determinant_count += full_pair_determinant_count;

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
          1.0e-8);
      const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
          union_components,
          cross_blocks,
          {});
      const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
          metric_graph,
          union_components);
      if (metric_summary.connected_component_count <= 1) {
        if (options.report_every > 0 &&
            (pair_index + 1) % options.report_every == 0) {
          const auto elapsed_seconds = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - started_at).count();
          std::cerr << "progress = " << (pair_index + 1) << "/" << pair_list.size()
                    << " elapsed_s = " << std::fixed << std::setprecision(2)
                    << elapsed_seconds << '\n';
        }
        continue;
      }

      ++disconnected_pair_count;
      ++disconnected_component_count_histogram[metric_summary.connected_component_count];
      ++disconnected_width_histogram[metric_summary.weighted_min_degree_width_upper_bound];
      disconnected_full_pair_determinant_count += full_pair_determinant_count;

      // This is the exact reference for the current raw-structure pair:
      // determinant-term expansion over the full pair support.
      const double exact_overlap = xmvb::vb::legacy_structure_overlap(
          left_cache.determinant_terms,
          right_cache.determinant_terms,
          full_active_overlap,
          overlap_resolver);

      // In the metric-disconnected special case, the full exact overlap must
      // factor into a product of exact overlaps over the disconnected graph
      // components. The pair-local work estimate is the sum of the component
      // determinant-pair counts, not the product for the full structure pair.
      double factored_overlap = 1.0;
      std::uint64_t factored_pair_determinant_count = 0;
      for (const auto& connected_component : metric_summary.connected_components) {
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
        if (left_component_pairs.size() != right_component_pairs.size()) {
          // A disconnected block with different pair counts on bra and ket has
          // identically zero overlap, so the exact factorized answer becomes
          // zero without any determinant-pair expansion inside that block.
          factored_overlap = 0.0;
          continue;
        }
        const auto left_component_terms =
            xmvb::vb::enumerate_legacy_determinant_terms(left_component_pairs);
        const auto right_component_terms =
            xmvb::vb::enumerate_legacy_determinant_terms(right_component_pairs);
        factored_pair_determinant_count +=
            static_cast<std::uint64_t>(left_component_terms.size()) *
            static_cast<std::uint64_t>(right_component_terms.size());
        factored_overlap *= xmvb::vb::legacy_structure_overlap(
            left_component_terms,
            right_component_terms,
            full_active_overlap,
            overlap_resolver);
      }

      disconnected_factored_pair_determinant_count += factored_pair_determinant_count;
      total_hybrid_pair_determinant_count -= full_pair_determinant_count;
      total_hybrid_pair_determinant_count += factored_pair_determinant_count;
      if (factored_pair_determinant_count < full_pair_determinant_count) {
        ++disconnected_pair_better_count;
      } else if (factored_pair_determinant_count == full_pair_determinant_count) {
        ++disconnected_pair_equal_count;
      } else {
        ++disconnected_pair_worse_count;
      }

      const double absolute_error = std::abs(exact_overlap - factored_overlap);
      max_abs_error = std::max(max_abs_error, absolute_error);
      if (absolute_error <= options.tolerance) {
        ++exact_match_count;
      } else {
        ++mismatch_count;
        ++mismatch_width_histogram[metric_summary.weighted_min_degree_width_upper_bound];
        if (options.mismatch_limit > 0 &&
            printed_mismatch_count < options.mismatch_limit) {
          std::cerr << std::setprecision(16);
          std::cerr << "mismatch[" << printed_mismatch_count << "]"
                    << " pair_index=" << pair_index
                    << " left_structure=" << left_structure
                    << " right_structure=" << right_structure
                    << " metric_connected_components="
                    << metric_summary.connected_component_count
                    << " width_upper_bound="
                    << metric_summary.weighted_min_degree_width_upper_bound
                    << " full_pair_count=" << full_pair_determinant_count
                    << " factored_pair_count=" << factored_pair_determinant_count
                    << " exact_overlap=" << exact_overlap
                    << " factored_overlap=" << factored_overlap
                    << " abs_error=" << absolute_error
                    << " component_signature="
                    << build_component_signature(union_components)
                    << '\n';
          print_connected_component_details(
              metric_summary.connected_components,
              union_components,
              support_orbitals);
          ++printed_mismatch_count;
        }
        if (options.stop_on_first_mismatch) {
          stop_requested = true;
        }
      }

      PairExample example;
      example.left_structure = left_structure;
      example.right_structure = right_structure;
      example.metric_connected_component_count = metric_summary.connected_component_count;
      example.width_upper_bound = metric_summary.weighted_min_degree_width_upper_bound;
      example.full_pair_determinant_count = full_pair_determinant_count;
      example.disconnected_factored_determinant_count = factored_pair_determinant_count;
      example.exact_overlap = exact_overlap;
      example.factored_overlap = factored_overlap;
      example.absolute_error = absolute_error;
      example.compression_ratio =
          factored_pair_determinant_count == 0
              ? 0.0
              : static_cast<double>(full_pair_determinant_count) /
                    static_cast<double>(factored_pair_determinant_count);
      example.component_signature = build_component_signature(union_components);
      examples.push_back(std::move(example));

      if (options.report_every > 0 &&
          (pair_index + 1) % options.report_every == 0) {
        const auto elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at).count();
        std::cerr << "progress = " << (pair_index + 1) << "/" << pair_list.size()
                  << " elapsed_s = " << std::fixed << std::setprecision(2)
                  << elapsed_seconds << '\n';
      }
      if (stop_requested) {
        break;
      }
    }

    std::sort(
        examples.begin(),
        examples.end(),
        [](const PairExample& left, const PairExample& right) {
          const auto left_saved =
              static_cast<std::int64_t>(left.full_pair_determinant_count) -
              static_cast<std::int64_t>(left.disconnected_factored_determinant_count);
          const auto right_saved =
              static_cast<std::int64_t>(right.full_pair_determinant_count) -
              static_cast<std::int64_t>(right.disconnected_factored_determinant_count);
          if (left_saved != right_saved) {
            return left_saved > right_saved;
          }
          if (left.absolute_error != right.absolute_error) {
            return left.absolute_error > right.absolute_error;
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
    std::cout << "pair_order = " << pair_order_name(options.pair_order) << '\n';
    std::cout << "active_overlap_source = "
              << active_overlap_source_name(options.active_overlap_source) << '\n';
    std::cout << "selected_pairs = " << pair_list.size() << '\n';
    std::cout << "processed_pairs = " << processed_pair_count << '\n';
    std::cout << "disconnected_pair_count = " << disconnected_pair_count << '\n';
    std::cout << "disconnected_pair_better_count = " << disconnected_pair_better_count << '\n';
    std::cout << "disconnected_pair_equal_count = " << disconnected_pair_equal_count << '\n';
    std::cout << "disconnected_pair_worse_count = " << disconnected_pair_worse_count << '\n';
    std::cout << "exact_match_count = " << exact_match_count << '\n';
    std::cout << "mismatch_count = " << mismatch_count << '\n';
    std::cout << "max_abs_error = " << max_abs_error << '\n';
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
    std::cout << "disconnected_component_count_histogram = {";
    bool first = true;
    for (const auto& [component_count, pair_count] : disconnected_component_count_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << component_count << ": " << pair_count;
    }
    std::cout << "}\n";
    std::cout << "disconnected_width_histogram = {";
    first = true;
    for (const auto& [width, pair_count] : disconnected_width_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << width << ": " << pair_count;
    }
    std::cout << "}\n";
    std::cout << "mismatch_width_histogram = {";
    first = true;
    for (const auto& [width, pair_count] : mismatch_width_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << width << ": " << pair_count;
    }
    std::cout << "}\n";
    std::cout << "total_full_pair_determinant_count = "
              << total_full_pair_determinant_count << '\n';
    std::cout << "total_hybrid_pair_determinant_count = "
              << total_hybrid_pair_determinant_count << '\n';
    std::cout << "disconnected_full_pair_determinant_count = "
              << disconnected_full_pair_determinant_count << '\n';
    std::cout << "disconnected_factored_pair_determinant_count = "
              << disconnected_factored_pair_determinant_count << '\n';
    std::cout << "hybrid_savings_pair_determinant_count = "
              << (total_full_pair_determinant_count - total_hybrid_pair_determinant_count) << '\n';
    std::cout << "hybrid_compression_ratio = "
              << (total_hybrid_pair_determinant_count == 0
                      ? 0.0
                      : static_cast<double>(total_full_pair_determinant_count) /
                            static_cast<double>(total_hybrid_pair_determinant_count))
              << '\n';
    std::cout << "disconnected_only_compression_ratio = "
              << (disconnected_factored_pair_determinant_count == 0
                      ? 0.0
                      : static_cast<double>(disconnected_full_pair_determinant_count) /
                            static_cast<double>(disconnected_factored_pair_determinant_count))
              << '\n';
    std::cout << "elapsed_wall_time_seconds = " << elapsed_seconds << '\n';

    const int n_examples_to_print =
        std::min(options.top_examples, static_cast<int>(examples.size()));
    std::cout << "top_examples\n";
    for (int example_index = 0; example_index < n_examples_to_print; ++example_index) {
      const auto& example = examples[example_index];
      const auto saved =
          static_cast<std::int64_t>(example.full_pair_determinant_count) -
          static_cast<std::int64_t>(example.disconnected_factored_determinant_count);
      std::cout << "example[" << example_index << "]"
                << " left_structure=" << example.left_structure
                << " right_structure=" << example.right_structure
                << " metric_connected_components="
                << example.metric_connected_component_count
                << " width_upper_bound=" << example.width_upper_bound
                << " full_pair_count=" << example.full_pair_determinant_count
                << " factored_pair_count="
                << example.disconnected_factored_determinant_count
                << " saved_pair_count=" << saved
                << " compression_ratio=" << example.compression_ratio
                << " exact_overlap=" << example.exact_overlap
                << " factored_overlap=" << example.factored_overlap
                << " abs_error=" << example.absolute_error
                << " component_signature=" << example.component_signature
                << '\n';
    }
    return mismatch_count == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
