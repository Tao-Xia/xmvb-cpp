#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
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
  int top_signatures = 12;
  int optimizer_max_iterations = 25;
  std::uint32_t seed = 0;
  double optimizer_gradient_tolerance = 2.0e-3;
  double optimizer_energy_tolerance = 1.0e-7;
  double singular_value_threshold = 1.0e-8;
  double edge_max_abs_threshold = 0.0;
  std::optional<std::string> csv_path;
};

struct PairRecord {
  int left_structure = 0;
  int right_structure = 0;
  int left_determinant_term_count = 0;
  int right_determinant_term_count = 0;
  int union_component_count = 0;
  int metric_connected_component_count = 0;
  int max_component_covalent_labels = 0;
  int articulation_count = 0;
  int weighted_min_degree_width_upper_bound = 0;
  std::uint64_t determinant_pair_count = 0;
  std::string component_signature;
};

struct SignatureSummary {
  int pair_count = 0;
  std::map<int, int> max_component_covalent_labels_histogram;
  std::map<int, int> articulation_count_histogram;
  std::map<int, int> width_upper_bound_histogram;
};

struct PerStructureCache {
  std::vector<xmvb::vb::OrbitalPair> active_pairs;
  int determinant_term_count = 0;
};

struct WidthDeterminantSummary {
  std::uint64_t pair_count = 0;
  std::uint64_t total_determinant_pair_count = 0;
  std::uint64_t min_determinant_pair_count = 0;
  std::uint64_t max_determinant_pair_count = 0;
  double total_binary_separator_compression_ratio = 0.0;
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

void print_usage() {
  std::cerr << "usage: analyze_metric_aware_graph_dataset <input.xmi>"
               " [--pair-order lexicographic|random]"
               " [--active-overlap-source input|optimized_vbscf]"
               " [--max-pairs N]"
               " [--seed S]"
               " [--report-every N]"
               " [--top-signatures N]"
               " [--optimizer-max-iterations N]"
               " [--optimizer-gradient-tolerance F]"
               " [--optimizer-energy-tolerance F]"
               " [--singular-value-threshold F]"
               " [--edge-max-abs-threshold F]"
               " [--csv output.csv]\n";
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
    if (argument_name == "--top-signatures") {
      options.top_signatures = std::stoi(argument_value);
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
    if (argument_name == "--csv") {
      options.csv_path = argument_value;
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
  if (options.top_signatures <= 0) {
    throw std::invalid_argument("--top-signatures must be positive");
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

template <typename T>
double mean_or_zero(const std::vector<T>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = std::accumulate(
      values.begin(),
      values.end(),
      0.0,
      [](double accumulator, const T& value) {
        return accumulator + static_cast<double>(value);
      });
  return sum / static_cast<double>(values.size());
}

template <typename T>
double median_or_zero(std::vector<T> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t center = values.size() / 2;
  if (values.size() % 2 == 1) {
    return static_cast<double>(values[center]);
  }
  return 0.5 * (static_cast<double>(values[center - 1]) +
                static_cast<double>(values[center]));
}

std::uint64_t integer_power(std::uint64_t base, int exponent) {
  if (exponent < 0) {
    throw std::invalid_argument("integer_power exponent must be non-negative");
  }
  std::uint64_t result = 1;
  for (int power = 0; power < exponent; ++power) {
    result *= base;
  }
  return result;
}

template <typename Key>
std::string format_histogram(const std::map<Key, int>& histogram) {
  std::ostringstream stream;
  stream << "{";
  bool first = true;
  for (const auto& [key, count] : histogram) {
    if (!first) {
      stream << ", ";
    }
    first = false;
    stream << key << ": " << count;
  }
  stream << "}";
  return stream.str();
}

void write_csv(
    const std::string& path,
    const std::vector<PairRecord>& records) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open CSV output: " + path);
  }
  output << "left_structure,right_structure,union_component_count,"
            "metric_connected_component_count,max_component_covalent_labels,"
            "articulation_count,weighted_min_degree_width_upper_bound,"
            "left_determinant_term_count,right_determinant_term_count,"
            "determinant_pair_count,"
            "component_signature\n";
  for (const auto& record : records) {
    output << record.left_structure << ","
           << record.right_structure << ","
           << record.union_component_count << ","
           << record.metric_connected_component_count << ","
           << record.max_component_covalent_labels << ","
           << record.articulation_count << ","
           << record.weighted_min_degree_width_upper_bound << ","
           << record.left_determinant_term_count << ","
           << record.right_determinant_term_count << ","
           << record.determinant_pair_count << ","
           << "\"" << record.component_signature << "\"\n";
  }
}

ActiveOverlapSelectionResult select_active_overlap_matrix(
    const Options& options,
    const xmvb::vb::CppVbInputLoadResult& load_result) {
  ActiveOverlapSelectionResult result;
  if (options.active_overlap_source == ActiveOverlapSource::Input) {
    result.active_overlap_matrix =
        load_result.input.orbital_preparation_input.ao_overlap_matrix;
    return result;
  }

  // The structure-pair analysis needs the final active-space metric SSO in the
  // optimized orbital basis. `active_overlap_matrix` stores that M x M spatial
  // overlap in column-major order, where M is the number of active orbitals.
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
          "analyze_metric_aware_graph_dataset currently supports only singlet closed-shell structures");
    }

    std::vector<PerStructureCache> structure_cache(
        raw_structure_data.n_structures);
    std::map<int, int> determinant_term_count_histogram;
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[structure_index];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      // `determinant_term_count` is the number of unique determinant terms in the
      // exact raw-structure expansion for this single structure. This is the
      // explicit term count that the determinant-based overlap/Hamiltonian path
      // would enumerate before any pairwise combination across two structures.
      cache.determinant_term_count = static_cast<int>(
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs).size());
      ++determinant_term_count_histogram[cache.determinant_term_count];
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.pair_order,
        options.seed,
        options.max_pairs);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    xmvb::vb::MetricAwareComponentGraphOptions graph_options;
    graph_options.edge_max_abs_threshold = options.edge_max_abs_threshold;

    std::vector<PairRecord> records;
    records.reserve(pair_list.size());
    std::map<int, int> max_component_covalent_labels_histogram;
    std::map<int, int> articulation_count_histogram;
    std::map<int, int> width_upper_bound_histogram;
    std::map<int, int> union_component_count_histogram;
    std::map<int, int> metric_connected_component_count_histogram;
    std::map<int, WidthDeterminantSummary> width_determinant_summaries;
    std::map<std::string, SignatureSummary> signature_summaries;
    std::vector<int> max_component_covalent_labels_values;
    std::vector<int> articulation_count_values;
    std::vector<int> width_upper_bound_values;
    std::vector<double> binary_separator_compression_values;
    std::vector<std::uint64_t> determinant_pair_count_values;
    std::uint64_t total_determinant_pair_count = 0;
    std::uint64_t total_binary_separator_state_count = 0;
    std::uint64_t total_ternary_separator_state_count = 0;

    for (std::size_t pair_index = 0; pair_index < pair_list.size(); ++pair_index) {
      const auto [left_structure, right_structure] = pair_list[pair_index];
      const auto& left_cache = structure_cache[left_structure];
      const auto& right_cache = structure_cache[right_structure];

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
      const auto components = xmvb::vb::build_union_graph_components(
          left_pairs_local,
          right_pairs_local,
          support_orbitals);
      const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
          support_overlap,
          components,
          options.singular_value_threshold);
      const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
          components,
          cross_blocks,
          graph_options);
      const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
          metric_graph,
          components);
      const auto block_diagonal_overlap =
          xmvb::vb::build_block_diagonalized_support_overlap(
              support_overlap,
              components);
      const auto offblock_overlap =
          xmvb::vb::build_offblock_support_overlap(
              support_overlap,
              components);
      const auto screening_summary = xmvb::vb::summarize_union_graph_screening(
          support_overlap,
          offblock_overlap,
          components,
          cross_blocks);

      PairRecord record;
      record.left_structure = left_structure;
      record.right_structure = right_structure;
      record.left_determinant_term_count = left_cache.determinant_term_count;
      record.right_determinant_term_count = right_cache.determinant_term_count;
      record.union_component_count = screening_summary.component_count;
      record.metric_connected_component_count = metric_summary.connected_component_count;
      record.max_component_covalent_labels = metric_summary.max_component_covalent_labels;
      record.articulation_count = metric_summary.articulation_count;
      record.weighted_min_degree_width_upper_bound =
          metric_summary.weighted_min_degree_width_upper_bound;
      record.determinant_pair_count =
          static_cast<std::uint64_t>(record.left_determinant_term_count) *
          static_cast<std::uint64_t>(record.right_determinant_term_count);
      record.component_signature = screening_summary.component_signature;
      records.push_back(record);

      ++max_component_covalent_labels_histogram[record.max_component_covalent_labels];
      ++articulation_count_histogram[record.articulation_count];
      ++width_upper_bound_histogram[record.weighted_min_degree_width_upper_bound];
      ++union_component_count_histogram[record.union_component_count];
      ++metric_connected_component_count_histogram[record.metric_connected_component_count];
      max_component_covalent_labels_values.push_back(record.max_component_covalent_labels);
      articulation_count_values.push_back(record.articulation_count);
      width_upper_bound_values.push_back(record.weighted_min_degree_width_upper_bound);
      determinant_pair_count_values.push_back(record.determinant_pair_count);
      total_determinant_pair_count += record.determinant_pair_count;

      // `2^w` is the optimistic sign-only separator count. `3^w` is a more
      // conservative exact-message proxy when one separator state carries more
      // than a single binary degree of freedom.
      const std::uint64_t binary_separator_state_count =
          integer_power(2, record.weighted_min_degree_width_upper_bound);
      const std::uint64_t ternary_separator_state_count =
          integer_power(3, record.weighted_min_degree_width_upper_bound);
      total_binary_separator_state_count += binary_separator_state_count;
      total_ternary_separator_state_count += ternary_separator_state_count;
      const double binary_separator_compression_ratio =
          static_cast<double>(record.determinant_pair_count) /
          static_cast<double>(binary_separator_state_count);
      binary_separator_compression_values.push_back(binary_separator_compression_ratio);

      auto& width_summary =
          width_determinant_summaries[record.weighted_min_degree_width_upper_bound];
      ++width_summary.pair_count;
      width_summary.total_determinant_pair_count += record.determinant_pair_count;
      width_summary.total_binary_separator_compression_ratio +=
          binary_separator_compression_ratio;
      if (width_summary.pair_count == 1) {
        width_summary.min_determinant_pair_count = record.determinant_pair_count;
        width_summary.max_determinant_pair_count = record.determinant_pair_count;
      } else {
        width_summary.min_determinant_pair_count = std::min(
            width_summary.min_determinant_pair_count,
            record.determinant_pair_count);
        width_summary.max_determinant_pair_count = std::max(
            width_summary.max_determinant_pair_count,
            record.determinant_pair_count);
      }

      auto& signature_summary = signature_summaries[record.component_signature];
      ++signature_summary.pair_count;
      ++signature_summary.max_component_covalent_labels_histogram[record.max_component_covalent_labels];
      ++signature_summary.articulation_count_histogram[record.articulation_count];
      ++signature_summary.width_upper_bound_histogram[record.weighted_min_degree_width_upper_bound];

      if (options.report_every > 0 &&
          (pair_index + 1) % options.report_every == 0) {
        const auto elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at).count();
        std::cerr << "progress = " << (pair_index + 1) << "/" << pair_list.size()
                  << " elapsed_s = " << std::fixed << std::setprecision(2)
                  << elapsed_seconds << '\n';
      }
    }

    if (options.csv_path.has_value()) {
      write_csv(options.csv_path.value(), records);
    }

    std::vector<std::pair<std::string, SignatureSummary>> sorted_signatures(
        signature_summaries.begin(),
        signature_summaries.end());
    std::sort(
        sorted_signatures.begin(),
        sorted_signatures.end(),
        [](const auto& left, const auto& right) {
          if (left.second.pair_count != right.second.pair_count) {
            return left.second.pair_count > right.second.pair_count;
          }
          return left.first < right.first;
        });

    const auto elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_at).count();

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "structure_count = " << raw_structure_data.n_structures << '\n';
    std::cout << "selected_pairs = " << records.size() << '\n';
    std::cout << "pair_order = " << pair_order_name(options.pair_order) << '\n';
    std::cout << "active_overlap_source = "
              << active_overlap_source_name(options.active_overlap_source) << '\n';
    std::cout << "max_pairs = " << options.max_pairs << '\n';
    std::cout << "optimizer_max_iterations = "
              << options.optimizer_max_iterations << '\n';
    std::cout << "optimizer_gradient_tolerance = "
              << options.optimizer_gradient_tolerance << '\n';
    std::cout << "optimizer_energy_tolerance = "
              << options.optimizer_energy_tolerance << '\n';
    std::cout << "singular_value_threshold = " << options.singular_value_threshold << '\n';
    std::cout << "edge_max_abs_threshold = " << options.edge_max_abs_threshold << '\n';
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
      std::cout << "optimizer_initial_total_energy = "
                << optimized_summary.initial_total_energy << '\n';
      std::cout << "optimizer_final_total_energy = "
                << optimized_summary.final_total_energy << '\n';
      std::cout << "optimizer_final_gradient_inf_norm = "
                << optimized_summary.final_gradient_inf_norm << '\n';
      std::cout << "optimizer_final_gradient_l2_norm = "
                << optimized_summary.final_gradient_l2_norm << '\n';
      std::cout << "optimizer_total_wall_time_seconds = "
                << optimized_summary.total_wall_time_seconds << '\n';
      std::cout << std::noboolalpha;
    }
    std::cout << "union_component_count_histogram = "
              << format_histogram(union_component_count_histogram) << '\n';
    std::cout << "metric_connected_component_count_histogram = "
              << format_histogram(metric_connected_component_count_histogram) << '\n';
    std::cout << "max_component_covalent_labels_histogram = "
              << format_histogram(max_component_covalent_labels_histogram) << '\n';
    std::cout << "articulation_count_histogram = "
              << format_histogram(articulation_count_histogram) << '\n';
    std::cout << "weighted_min_degree_width_upper_bound_histogram = "
              << format_histogram(width_upper_bound_histogram) << '\n';
    std::cout << "max_component_covalent_labels_mean = "
              << mean_or_zero(max_component_covalent_labels_values) << '\n';
    std::cout << "max_component_covalent_labels_median = "
              << median_or_zero(max_component_covalent_labels_values) << '\n';
    std::cout << "articulation_count_mean = "
              << mean_or_zero(articulation_count_values) << '\n';
    std::cout << "articulation_count_median = "
              << median_or_zero(articulation_count_values) << '\n';
    std::cout << "weighted_min_degree_width_upper_bound_mean = "
              << mean_or_zero(width_upper_bound_values) << '\n';
    std::cout << "weighted_min_degree_width_upper_bound_median = "
              << median_or_zero(width_upper_bound_values) << '\n';
    std::cout << "determinant_term_count_histogram = "
              << format_histogram(determinant_term_count_histogram) << '\n';
    std::cout << "determinant_pair_count_mean = "
              << mean_or_zero(determinant_pair_count_values) << '\n';
    std::cout << "determinant_pair_count_median = "
              << median_or_zero(determinant_pair_count_values) << '\n';
    std::cout << "total_determinant_pair_count = "
              << total_determinant_pair_count << '\n';
    std::cout << "binary_separator_compression_ratio_mean = "
              << mean_or_zero(binary_separator_compression_values) << '\n';
    std::cout << "binary_separator_compression_ratio_median = "
              << median_or_zero(binary_separator_compression_values) << '\n';
    std::cout << "total_binary_separator_state_count = "
              << total_binary_separator_state_count << '\n';
    std::cout << "total_binary_separator_compression_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_determinant_pair_count) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "total_ternary_separator_state_count = "
              << total_ternary_separator_state_count << '\n';
    std::cout << "total_ternary_separator_compression_ratio = "
              << (total_ternary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_determinant_pair_count) /
                            static_cast<double>(total_ternary_separator_state_count))
              << '\n';
    std::cout << "elapsed_wall_time_seconds = " << elapsed_seconds << '\n';
    if (options.csv_path.has_value()) {
      std::cout << "csv = " << options.csv_path.value() << '\n';
    }
    std::cout << "width_determinant_summaries\n";
    for (const auto& [width_upper_bound, width_summary] : width_determinant_summaries) {
      const double mean_determinant_pair_count =
          width_summary.pair_count == 0
              ? 0.0
              : static_cast<double>(width_summary.total_determinant_pair_count) /
                    static_cast<double>(width_summary.pair_count);
      const double mean_binary_separator_compression_ratio =
          width_summary.pair_count == 0
              ? 0.0
              : width_summary.total_binary_separator_compression_ratio /
                    static_cast<double>(width_summary.pair_count);
      std::cout << "width = " << width_upper_bound
                << " | pair_count = " << width_summary.pair_count
                << " | mean_determinant_pair_count = "
                << mean_determinant_pair_count
                << " | min_determinant_pair_count = "
                << width_summary.min_determinant_pair_count
                << " | max_determinant_pair_count = "
                << width_summary.max_determinant_pair_count
                << " | mean_binary_separator_compression_ratio = "
                << mean_binary_separator_compression_ratio
                << '\n';
    }
    std::cout << "signature_summaries\n";
    for (int signature_index = 0;
         signature_index < options.top_signatures &&
         signature_index < static_cast<int>(sorted_signatures.size());
         ++signature_index) {
      const auto& [signature, summary] =
          sorted_signatures[signature_index];
      std::cout << "signature = " << signature
                << " | pair_count = " << summary.pair_count
                << " | max_component_covalent_labels_hist = "
                << format_histogram(summary.max_component_covalent_labels_histogram)
                << " | articulation_count_hist = "
                << format_histogram(summary.articulation_count_histogram)
                << " | width_upper_bound_hist = "
                << format_histogram(summary.width_upper_bound_histogram)
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
