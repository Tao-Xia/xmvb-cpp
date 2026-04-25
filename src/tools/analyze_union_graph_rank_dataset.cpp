#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
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
#include "vb/matrices/union_graph_rank_predictor.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Pair = std::pair<int, int>;

enum class PairOrder {
  Lexicographic,
  Random,
};

struct Options {
  std::string input_path;
  PairOrder pair_order = PairOrder::Lexicographic;
  int max_pairs = 0;
  int max_rank_cap = -1;
  int report_every = 0;
  int top_signatures = 12;
  std::uint32_t seed = 0;
  double normalized_error_floor = 1.0e-14;
  double target_normalized_error = 1.0e-2;
  xmvb::vb::UnionGraphRankPredictorOptions predictor_options;
  std::optional<std::string> csv_path;
};

struct PairRecord {
  int left_structure = 0;
  int right_structure = 0;
  int component_count = 0;
  int max_available_rank = 0;
  int required_rank = 0;
  int predicted_rank = 0;
  int n_general_components = 0;
  double exact_overlap = 0.0;
  double offblock_overlap_fraction = 0.0;
  double max_cross_block_second_singular = 0.0;
  double max_cross_block_third_singular = 0.0;
  std::string component_signature;
  std::string prediction_reason;
};

struct SignatureSummary {
  int pair_count = 0;
  std::map<int, int> max_rank_histogram;
  std::map<int, int> required_rank_histogram;
  std::map<int, int> predicted_rank_histogram;
};

struct PerStructureCache {
  std::vector<xmvb::vb::OrbitalPair> active_pairs;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> determinant_terms_global;
};

void print_usage() {
  std::cerr << "usage: analyze_union_graph_rank_dataset <input.xmi>"
               " [--pair-order lexicographic|random]"
               " [--max-pairs N]"
               " [--max-rank-cap R]"
               " [--seed S]"
               " [--normalized-error-floor F]"
               " [--target-normalized-error T]"
               " [--predictor-offblock-threshold F]"
               " [--predictor-second-singular-threshold F]"
               " [--predictor-third-singular-threshold F]"
               " [--predictor-max-rank R]"
               " [--report-every N]"
               " [--top-signatures N]"
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
    if (argument_name == "--max-pairs") {
      options.max_pairs = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--max-rank-cap") {
      options.max_rank_cap = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(argument_value));
      continue;
    }
    if (argument_name == "--normalized-error-floor") {
      options.normalized_error_floor = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--target-normalized-error") {
      options.target_normalized_error = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--predictor-offblock-threshold") {
      options.predictor_options.zero_rank_offblock_threshold = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--predictor-second-singular-threshold") {
      options.predictor_options.rank_one_second_singular_threshold =
          std::stod(argument_value);
      continue;
    }
    if (argument_name == "--predictor-third-singular-threshold") {
      options.predictor_options.rank_two_third_singular_threshold =
          std::stod(argument_value);
      continue;
    }
    if (argument_name == "--predictor-max-rank") {
      options.predictor_options.max_predicted_rank = std::stoi(argument_value);
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
    if (argument_name == "--csv") {
      options.csv_path = argument_value;
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if (options.max_rank_cap < -1) {
    throw std::invalid_argument("--max-rank-cap must be >= -1");
  }
  if (options.normalized_error_floor <= 0.0) {
    throw std::invalid_argument("--normalized-error-floor must be positive");
  }
  if (options.target_normalized_error <= 0.0) {
    throw std::invalid_argument("--target-normalized-error must be positive");
  }
  if (options.predictor_options.max_predicted_rank < -1) {
    throw std::invalid_argument("--predictor-max-rank must be >= -1");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
  }
  if (options.top_signatures <= 0) {
    throw std::invalid_argument("--top-signatures must be positive");
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
  output << "left_structure,right_structure,component_count,component_signature,"
            "n_general_components,max_available_rank,required_rank,predicted_rank,"
            "prediction_reason,"
            "offblock_overlap_fraction,max_cross_block_second_singular,"
            "max_cross_block_third_singular,exact_overlap\n";
  output << std::setprecision(17);
  for (const auto& record : records) {
    output << record.left_structure << ","
           << record.right_structure << ","
           << record.component_count << ","
           << "\"" << record.component_signature << "\"" << ","
           << record.n_general_components << ","
           << record.max_available_rank << ","
           << record.required_rank << ","
           << record.predicted_rank << ","
           << "\"" << record.prediction_reason << "\"" << ","
           << record.offblock_overlap_fraction << ","
           << record.max_cross_block_second_singular << ","
           << record.max_cross_block_third_singular << ","
           << record.exact_overlap << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    auto predictor_options = options.predictor_options;
    if (predictor_options.max_predicted_rank < 0) {
      predictor_options.max_predicted_rank = options.max_rank_cap;
    }
    const auto started_at = std::chrono::steady_clock::now();
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    const auto& active_overlap_storage =
        load_result.input.orbital_preparation_input.active_orbital_overlap_matrix;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "analyze_union_graph_rank_dataset currently supports only singlet closed-shell structures");
    }

    std::vector<PerStructureCache> structure_cache(
        raw_structure_data.n_structures);
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[structure_index];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      cache.determinant_terms_global =
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs);
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.pair_order,
        options.seed,
        options.max_pairs);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    std::vector<PairRecord> records;
    records.reserve(pair_list.size());
    std::map<int, int> component_count_histogram;
    std::map<int, int> max_available_rank_histogram;
    std::map<int, int> required_rank_histogram;
    std::map<int, int> predicted_rank_histogram;
    std::map<std::string, int> prediction_reason_histogram;
    std::map<std::string, SignatureSummary> signature_summaries;
    std::vector<double> offblock_overlap_fractions;
    std::vector<double> second_singular_values;
    std::vector<double> third_singular_values;
    std::vector<int> predicted_ranks;
    std::vector<int> overprediction_values;
    int underprediction_count = 0;
    int exact_prediction_count = 0;

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
      const auto left_terms_local =
          xmvb::vb::remap_legacy_determinant_terms(
              left_cache.determinant_terms_global,
              support_index);
      const auto right_terms_local =
          xmvb::vb::remap_legacy_determinant_terms(
              right_cache.determinant_terms_global,
              support_index);

      const auto support_overlap = xmvb::vb::build_support_overlap_matrix(
          support_orbitals,
          active_overlap_storage,
          n_active_orbitals);
      const auto components = xmvb::vb::build_union_graph_components(
          left_pairs_local,
          right_pairs_local,
          support_orbitals);
      const auto offblock_overlap = xmvb::vb::build_offblock_support_overlap(
          support_overlap,
          components);
      const auto block_diagonal_overlap =
          xmvb::vb::build_block_diagonalized_support_overlap(
              support_overlap,
              components);
      const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
          support_overlap,
          components,
          1.0e-8);
      const auto screening_summary = xmvb::vb::summarize_union_graph_screening(
          support_overlap,
          offblock_overlap,
          components,
          cross_blocks);
      const auto rank_prediction = xmvb::vb::predict_union_graph_rank_cap(
          screening_summary,
          predictor_options);

      const double exact_overlap = xmvb::vb::legacy_structure_overlap(
          left_terms_local,
          right_terms_local,
          support_overlap,
          overlap_resolver);
      const int max_available_rank = screening_summary.max_numerical_rank;
      const int effective_max_rank =
          options.max_rank_cap >= 0 ? std::min(options.max_rank_cap, max_available_rank)
                                    : max_available_rank;
      const double error_scale =
          std::max(std::abs(exact_overlap), options.normalized_error_floor);

      int required_rank = effective_max_rank;
      bool found_required_rank = false;
      for (int rank_cap = 0; rank_cap <= effective_max_rank; ++rank_cap) {
        const auto truncated_offblock = xmvb::vb::build_blockwise_truncated_offblock(
            support_overlap,
            components,
            rank_cap);
        const auto approximate_support_overlap =
            block_diagonal_overlap + truncated_offblock;
        const double approximate_overlap = xmvb::vb::legacy_structure_overlap(
            left_terms_local,
            right_terms_local,
            approximate_support_overlap,
            overlap_resolver);
        const double absolute_error = std::abs(exact_overlap - approximate_overlap);
        const double normalized_error = absolute_error / error_scale;
        if (normalized_error <= options.target_normalized_error) {
          required_rank = rank_cap;
          found_required_rank = true;
          break;
        }
      }
      if (!found_required_rank && effective_max_rank < max_available_rank) {
        required_rank = max_available_rank;
      }

      PairRecord record;
      record.left_structure = left_structure;
      record.right_structure = right_structure;
      record.component_count = screening_summary.component_count;
      record.max_available_rank = max_available_rank;
      record.required_rank = required_rank;
      record.predicted_rank = rank_prediction.predicted_rank_cap;
      record.n_general_components = screening_summary.n_general_components;
      record.exact_overlap = exact_overlap;
      record.offblock_overlap_fraction = screening_summary.offblock_overlap_fraction;
      record.max_cross_block_second_singular =
          screening_summary.max_cross_block_second_singular;
      record.max_cross_block_third_singular =
          screening_summary.max_cross_block_third_singular;
      record.component_signature = screening_summary.component_signature;
      record.prediction_reason =
          xmvb::vb::union_graph_rank_prediction_reason_name(
              rank_prediction.reason);
      records.push_back(record);

      ++component_count_histogram[record.component_count];
      ++max_available_rank_histogram[record.max_available_rank];
      ++required_rank_histogram[record.required_rank];
      ++predicted_rank_histogram[record.predicted_rank];
      ++prediction_reason_histogram[record.prediction_reason];
      offblock_overlap_fractions.push_back(record.offblock_overlap_fraction);
      second_singular_values.push_back(record.max_cross_block_second_singular);
      third_singular_values.push_back(record.max_cross_block_third_singular);
      predicted_ranks.push_back(record.predicted_rank);
      overprediction_values.push_back(
          std::max(record.predicted_rank - record.required_rank, 0));
      underprediction_count +=
          static_cast<int>(record.predicted_rank < record.required_rank);
      exact_prediction_count +=
          static_cast<int>(record.predicted_rank == record.required_rank);
      auto& signature_summary = signature_summaries[record.component_signature];
      ++signature_summary.pair_count;
      ++signature_summary.max_rank_histogram[record.max_available_rank];
      ++signature_summary.required_rank_histogram[record.required_rank];
      ++signature_summary.predicted_rank_histogram[record.predicted_rank];

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
    std::cout << "max_pairs = " << options.max_pairs << '\n';
    std::cout << "max_rank_cap = " << options.max_rank_cap << '\n';
    std::cout << "normalized_error_floor = " << options.normalized_error_floor << '\n';
    std::cout << "target_normalized_error = " << options.target_normalized_error << '\n';
    std::cout << "predictor_zero_rank_offblock_threshold = "
              << predictor_options.zero_rank_offblock_threshold << '\n';
    std::cout << "predictor_rank_one_second_singular_threshold = "
              << predictor_options.rank_one_second_singular_threshold << '\n';
    std::cout << "predictor_rank_two_third_singular_threshold = "
              << predictor_options.rank_two_third_singular_threshold << '\n';
    std::cout << "predictor_max_rank = "
              << predictor_options.max_predicted_rank << '\n';
    std::cout << "component_count_histogram = "
              << format_histogram(component_count_histogram) << '\n';
    std::cout << "max_available_rank_histogram = "
              << format_histogram(max_available_rank_histogram) << '\n';
    std::cout << "required_rank_histogram = "
              << format_histogram(required_rank_histogram) << '\n';
    std::cout << "predicted_rank_histogram = "
              << format_histogram(predicted_rank_histogram) << '\n';
    std::cout << "prediction_reason_histogram = "
              << format_histogram(prediction_reason_histogram) << '\n';
    std::cout << "predictor_underprediction_count = "
              << underprediction_count << '\n';
    std::cout << "predictor_exact_match_count = "
              << exact_prediction_count << '\n';
    std::cout << "predictor_mean_rank = "
              << mean_or_zero(predicted_ranks) << '\n';
    std::cout << "predictor_mean_overprediction = "
              << mean_or_zero(overprediction_values) << '\n';
    std::cout << "offblock_overlap_fraction_mean = "
              << mean_or_zero(offblock_overlap_fractions) << '\n';
    std::cout << "offblock_overlap_fraction_median = "
              << median_or_zero(offblock_overlap_fractions) << '\n';
    std::cout << "max_cross_block_second_singular_mean = "
              << mean_or_zero(second_singular_values) << '\n';
    std::cout << "max_cross_block_second_singular_median = "
              << median_or_zero(second_singular_values) << '\n';
    std::cout << "max_cross_block_third_singular_mean = "
              << mean_or_zero(third_singular_values) << '\n';
    std::cout << "max_cross_block_third_singular_median = "
              << median_or_zero(third_singular_values) << '\n';
    std::cout << "elapsed_wall_time_seconds = " << elapsed_seconds << '\n';
    if (options.csv_path.has_value()) {
      std::cout << "csv = " << options.csv_path.value() << '\n';
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
                << " | max_rank_hist = " << format_histogram(summary.max_rank_histogram)
                << " | required_rank_hist = " << format_histogram(summary.required_rank_histogram)
                << " | predicted_rank_hist = "
                << format_histogram(summary.predicted_rank_histogram)
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
