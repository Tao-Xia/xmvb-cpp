#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/full_structure_expander.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Pair = xmvb::vb::OrbitalPair;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int max_structures = 8;
  int report_count = 8;
  double tolerance = 1.0e-8;
  bool restrict_covalent = true;
};

struct StructureSummary {
  int raw_structure_index = 0;
  std::vector<Pair> pairs;
};

struct StructurePairExactValues {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

struct MatrixErrorEntry {
  int row_local = 0;
  int column_local = 0;
  int row_raw = 0;
  int column_raw = 0;
  double exact_value = 0.0;
  double candidate_value = 0.0;
  double absolute_error = 0.0;
  double relative_error = 0.0;
};

struct MatrixErrorSummary {
  std::string name;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  int mismatch_count = 0;
  std::vector<MatrixErrorEntry> worst_entries;
};

struct TwoPairKernelDiagnostics {
  double permanent_max_connected_overlap_abs = 0.0;
  double permanent_max_connected_one_electron_abs = 0.0;
  double determinant_max_connected_overlap_abs = 0.0;
  double determinant_max_connected_one_electron_abs = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_covalent_pair_cluster_hamiltonian <input.xmi> "
               "[--max-structures N] [--report-count N] [--tolerance tol] "
               "[--restrict-covalent 0|1]\n";
}

bool parse_bool_text(const std::string& text) {
  if (text == "1" || text == "true" || text == "TRUE") {
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + text);
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
    if (argument_name == "--max-structures") {
      options.max_structures = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--report-count") {
      options.report_count = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--restrict-covalent") {
      options.restrict_covalent = parse_bool_text(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_structures < 0) {
    throw std::invalid_argument("--max-structures must be non-negative");
  }
  if (options.report_count <= 0) {
    throw std::invalid_argument("--report-count must be positive");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  return options;
}

std::vector<Pair> normalize_covalent_pairs(const std::vector<Pair>& pairs) {
  std::vector<Pair> normalized_pairs;
  normalized_pairs.reserve(pairs.size());
  for (const auto& pair : pairs) {
    if (pair.first == pair.second) {
      throw std::invalid_argument("normalize_covalent_pairs expects non-diagonal pairs");
    }
    normalized_pairs.push_back(std::minmax(pair.first, pair.second));
  }
  std::sort(normalized_pairs.begin(), normalized_pairs.end());
  return normalized_pairs;
}

bool is_disjoint_covalent_structure(
    const std::vector<Pair>& pairs,
    int n_active_orbitals) {
  std::vector<int> orbital_counts(xmvb::to_size(n_active_orbitals), 0);
  for (const auto& pair : pairs) {
    if (pair.first == pair.second) {
      return false;
    }
    if (pair.first < 0 || pair.first >= n_active_orbitals ||
        pair.second < 0 || pair.second >= n_active_orbitals) {
      throw std::out_of_range("pair orbital index out of range");
    }
    ++orbital_counts[xmvb::to_size(pair.first)];
    ++orbital_counts[xmvb::to_size(pair.second)];
  }
  for (const int count : orbital_counts) {
    if (count != 1) {
      return false;
    }
  }
  return true;
}

std::string format_pairs(const std::vector<Pair>& pairs) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pair_index > 0) {
      stream << " ";
    }
    stream << "(" << pairs[pair_index].first + 1
           << "," << pairs[pair_index].second + 1 << ")";
  }
  stream << "]";
  return stream.str();
}

std::string pair_list_key(const std::vector<Pair>& pairs) {
  std::ostringstream stream;
  for (const auto& pair : pairs) {
    stream << pair.first << ":" << pair.second << ";";
  }
  return stream.str();
}

std::string pair_list_pair_key(
    const std::vector<Pair>& left_pairs,
    const std::vector<Pair>& right_pairs) {
  return pair_list_key(left_pairs) + "|" + pair_list_key(right_pairs);
}

double matrix_permanent_ryser(const Matrix& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("permanent requires a square matrix");
  }
  const int dimension = matrix.rows();
  if (dimension == 0) {
    return 1.0;
  }
  if (dimension >= 63) {
    throw std::invalid_argument("permanent evaluation only supports dimension < 63");
  }

  const std::uint64_t subset_count = std::uint64_t{1} << dimension;
  double permanent_value = 0.0;
  std::vector<double> row_sums(xmvb::to_size(dimension), 0.0);
  for (std::uint64_t subset = 1; subset < subset_count; ++subset) {
    std::fill(row_sums.begin(), row_sums.end(), 0.0);
    for (int column = 0; column < dimension; ++column) {
      if (((subset >> column) & std::uint64_t{1}) == 0) {
        continue;
      }
      for (int row = 0; row < dimension; ++row) {
        row_sums[xmvb::to_size(row)] += matrix(row, column);
      }
    }

    double contribution = 1.0;
    for (const double row_sum : row_sums) {
      contribution *= row_sum;
    }
    const int subset_size = __builtin_popcountll(subset);
    if (((dimension - subset_size) % 2) != 0) {
      contribution = -contribution;
    }
    permanent_value += contribution;
  }
  return permanent_value;
}

double matrix_determinant(const Matrix& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant requires a square matrix");
  }
  if (matrix.rows() == 0) {
    return 1.0;
  }
  return matrix.fullPivLu().determinant();
}

Matrix build_minor_matrix(
    const Matrix& matrix,
    const std::vector<int>& removed_rows,
    const std::vector<int>& removed_columns) {
  if (removed_rows.size() != removed_columns.size()) {
    throw std::invalid_argument("removed row/column counts must match");
  }
  std::vector<bool> row_removed(xmvb::to_size(matrix.rows()), false);
  std::vector<bool> column_removed(xmvb::to_size(matrix.cols()), false);
  for (const int row : removed_rows) {
    row_removed[xmvb::to_size(row)] = true;
  }
  for (const int column : removed_columns) {
    column_removed[xmvb::to_size(column)] = true;
  }

  Matrix minor_matrix(
      matrix.rows() - static_cast<int>(removed_rows.size()),
      matrix.cols() - static_cast<int>(removed_columns.size()));
  int minor_row = 0;
  for (int row = 0; row < matrix.rows(); ++row) {
    if (row_removed[xmvb::to_size(row)]) {
      continue;
    }
    int minor_column = 0;
    for (int column = 0; column < matrix.cols(); ++column) {
      if (column_removed[xmvb::to_size(column)]) {
        continue;
      }
      minor_matrix(minor_row, minor_column) = matrix(row, column);
      ++minor_column;
    }
    ++minor_row;
  }
  return minor_matrix;
}

double permanent_minor(
    const Matrix& matrix,
    const std::vector<int>& removed_rows,
    const std::vector<int>& removed_columns) {
  return matrix_permanent_ryser(build_minor_matrix(matrix, removed_rows, removed_columns));
}

double first_cofactor(
    const Matrix& matrix,
    int removed_row,
    int removed_column) {
  const double sign = ((removed_row + removed_column) % 2 == 0) ? 1.0 : -1.0;
  return sign * matrix_determinant(
                    build_minor_matrix(
                        matrix,
                        {removed_row},
                        {removed_column}));
}

double second_cofactor(
    const Matrix& matrix,
    int removed_row_first,
    int removed_row_second,
    int removed_column_first,
    int removed_column_second) {
  const double sign =
      ((removed_row_first + removed_row_second +
        removed_column_first + removed_column_second) %
       2 == 0)
      ? 1.0
      : -1.0;
  return sign * matrix_determinant(
                    build_minor_matrix(
                        matrix,
                        {removed_row_first, removed_row_second},
                        {removed_column_first, removed_column_second}));
}

void set_symmetric_matrix_entry(
    std::vector<double>* matrix,
    int dimension,
    int row,
    int column,
    double value) {
  if (matrix == nullptr) {
    throw std::invalid_argument("matrix must not be null");
  }
  const std::size_t upper_index =
      xmvb::to_size(column) * dimension + xmvb::to_size(row);
  const std::size_t lower_index =
      xmvb::to_size(row) * dimension + xmvb::to_size(column);
  (*matrix)[upper_index] = value;
  (*matrix)[lower_index] = value;
}

StructurePairExactValues compute_exact_structure_pair_values(
    const std::vector<Pair>& left_pairs,
    const std::vector<Pair>& right_pairs,
    const std::vector<double>& active_overlap_matrix,
    const std::vector<double>& active_one_electron_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    std::map<std::string, std::vector<xmvb::vb::LegacyStructureDeterminantTerm>>* term_cache,
    std::map<std::string, StructurePairExactValues>* value_cache) {
  if (term_cache == nullptr || value_cache == nullptr) {
    throw std::invalid_argument("caches must not be null");
  }
  const std::string key = pair_list_pair_key(left_pairs, right_pairs);
  const auto cached_value_iterator = value_cache->find(key);
  if (cached_value_iterator != value_cache->end()) {
    return cached_value_iterator->second;
  }

  const auto get_terms = [&](const std::vector<Pair>& pairs)
      -> const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& {
    const std::string terms_key = pair_list_key(pairs);
    const auto iterator = term_cache->find(terms_key);
    if (iterator != term_cache->end()) {
      return iterator->second;
    }
    auto [inserted_iterator, inserted] = term_cache->emplace(
        terms_key,
        xmvb::vb::enumerate_legacy_determinant_terms(pairs));
    (void)inserted;
    return inserted_iterator->second;
  };

  const auto& left_terms = get_terms(left_pairs);
  const auto& right_terms = get_terms(right_pairs);
  xmvb::vb::FullDeterminantPairEvaluator pair_evaluator;

  StructurePairExactValues values;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const auto pair_value = pair_evaluator.evaluate(
          left_term.alpha_occ,
          right_term.alpha_occ,
          left_term.beta_occ,
          right_term.beta_occ,
          active_overlap_matrix,
          active_one_electron_matrix,
          n_active_orbitals,
          packed_active_two_electron_integrals);
      const double coefficient =
          left_term.coefficient * right_term.coefficient;
      values.overlap += coefficient * pair_value.overlap_determinant;
      values.one_electron_hamiltonian +=
          coefficient * pair_value.one_electron_hamiltonian;
      values.total_hamiltonian +=
          coefficient * pair_value.total_hamiltonian;
    }
  }

  (*value_cache)[key] = values;
  return values;
}

template <typename MatrixGetter>
MatrixErrorSummary summarize_matrix_error(
    const std::string& name,
    const std::vector<double>& exact_matrix,
    const std::vector<double>& candidate_matrix,
    const std::vector<StructureSummary>& structures,
    int report_count,
    double tolerance,
    const MatrixGetter& matrix_getter) {
  if (exact_matrix.size() != candidate_matrix.size()) {
    throw std::invalid_argument("matrix sizes must match");
  }

  MatrixErrorSummary summary;
  summary.name = name;
  const int n_structures = static_cast<int>(structures.size());
  summary.worst_entries.reserve(xmvb::to_size(n_structures) *
                               xmvb::to_size(n_structures + 1) / 2);
  for (int row = 0; row < n_structures; ++row) {
    for (int column = 0; column <= row; ++column) {
      const double exact_value = matrix_getter(exact_matrix, n_structures, row, column);
      const double candidate_value = matrix_getter(candidate_matrix, n_structures, row, column);
      const double absolute_error = std::abs(exact_value - candidate_value);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(exact_value));
      if (absolute_error > tolerance) {
        ++summary.mismatch_count;
      }
      summary.max_abs_error = std::max(summary.max_abs_error, absolute_error);
      summary.max_rel_error = std::max(summary.max_rel_error, relative_error);
      summary.worst_entries.push_back({
          .row_local = row,
          .column_local = column,
          .row_raw = structures[xmvb::to_size(row)].raw_structure_index,
          .column_raw = structures[xmvb::to_size(column)].raw_structure_index,
          .exact_value = exact_value,
          .candidate_value = candidate_value,
          .absolute_error = absolute_error,
          .relative_error = relative_error,
      });
    }
  }

  std::sort(
      summary.worst_entries.begin(),
      summary.worst_entries.end(),
      [](const MatrixErrorEntry& left, const MatrixErrorEntry& right) {
        if (left.absolute_error != right.absolute_error) {
          return left.absolute_error > right.absolute_error;
        }
        if (left.row_local != right.row_local) {
          return left.row_local < right.row_local;
        }
        return left.column_local < right.column_local;
      });
  if (static_cast<int>(summary.worst_entries.size()) > report_count) {
    summary.worst_entries.resize(xmvb::to_size(report_count));
  }
  return summary;
}

void print_matrix_error_summary(
    const MatrixErrorSummary& summary,
    const std::vector<StructureSummary>& structures) {
  std::cout << "matrix = " << summary.name << '\n';
  std::cout << "max_abs_error = " << summary.max_abs_error << '\n';
  std::cout << "max_rel_error = " << summary.max_rel_error << '\n';
  std::cout << "mismatch_count = " << summary.mismatch_count << '\n';
  for (std::size_t entry_index = 0;
       entry_index < summary.worst_entries.size();
       ++entry_index) {
    const auto& entry = summary.worst_entries[entry_index];
    const auto& left_structure = structures[xmvb::to_size(entry.row_local)];
    const auto& right_structure = structures[xmvb::to_size(entry.column_local)];
    std::cout << "pair[" << entry_index << "]"
              << " row_local=" << entry.row_local
              << " column_local=" << entry.column_local
              << " row_raw=" << entry.row_raw
              << " column_raw=" << entry.column_raw
              << " exact=" << entry.exact_value
              << " candidate=" << entry.candidate_value
              << " abs_error=" << entry.absolute_error
              << " rel_error=" << entry.relative_error
              << " left_pairs=" << format_pairs(left_structure.pairs)
              << " right_pairs=" << format_pairs(right_structure.pairs)
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto overall_started_at = std::chrono::steady_clock::now();
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    const auto& raw_structure_data = load_result.raw_structure_data;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "check_covalent_pair_cluster_hamiltonian currently supports only singlet closed-shell structures");
    }
    if ((raw_structure_data.n_active_electrons % 2) != 0) {
      throw std::runtime_error("n_active_electrons must be even");
    }

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto timed_active_space = xmvb::vb::prepare_timed_active_space_context(
        input,
        orbital_preparer,
        ao_effective_one_electron_builder,
        active_space_one_electron_builder,
        active_space_two_electron_builder);
    const auto& prepared_active_space = timed_active_space.prepared_active_space;
    const auto& active_overlap_matrix =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    const auto& active_one_electron_matrix =
        prepared_active_space.active_space_one_electron_result.h1e_act;
    const auto& packed_active_two_electron_integrals =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;

    int available_disjoint_covalent_structures = 0;
    std::vector<StructureSummary> selected_structures;
    selected_structures.reserve(xmvb::to_size(
        options.max_structures > 0 ? options.max_structures : raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto pairs = xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      const bool is_disjoint_covalent =
          is_disjoint_covalent_structure(pairs, n_active_orbitals);
      if (is_disjoint_covalent) {
        ++available_disjoint_covalent_structures;
      }
      if (options.restrict_covalent && !is_disjoint_covalent) {
        continue;
      }
      if (is_disjoint_covalent) {
        pairs = normalize_covalent_pairs(pairs);
      }
      selected_structures.push_back({
          .raw_structure_index = structure_index,
          .pairs = std::move(pairs),
      });
      if (options.max_structures > 0 &&
          static_cast<int>(selected_structures.size()) >= options.max_structures) {
        break;
      }
    }

    if (selected_structures.empty()) {
      throw std::runtime_error("no structures selected for validation");
    }
    const int n_pairs =
        static_cast<int>(selected_structures.front().pairs.size());
    for (const auto& structure : selected_structures) {
      if (static_cast<int>(structure.pairs.size()) != n_pairs) {
        throw std::runtime_error("selected structures do not share a common pair count");
      }
    }

    std::vector<int> selected_structure_indices;
    selected_structure_indices.reserve(selected_structures.size());
    for (const auto& structure : selected_structures) {
      selected_structure_indices.push_back(structure.raw_structure_index);
    }

    xmvb::vb::FullDeterminantStructureExpander expander;
    const auto subset_structure_data =
        expander.expand_subset(raw_structure_data, selected_structure_indices);
    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
    const auto exact_build_started_at = std::chrono::steady_clock::now();
    const auto exact_structure_matrices = structure_builder.build(
        subset_structure_data.alpha_det,
        subset_structure_data.beta_det,
        subset_structure_data.determinant_to_structure_terms,
        active_overlap_matrix,
        active_one_electron_matrix,
        n_active_orbitals,
        packed_active_two_electron_integrals,
        subset_structure_data.n_structures);
    const int n_packed_active_pairs =
        n_active_orbitals * (n_active_orbitals + 1) / 2;
    const std::vector<double> zero_eri(
        xmvb::to_size(n_packed_active_pairs) *
            xmvb::to_size(n_packed_active_pairs + 1) / 2,
        0.0);
    const auto exact_one_electron_structure_matrices = structure_builder.build(
        subset_structure_data.alpha_det,
        subset_structure_data.beta_det,
        subset_structure_data.determinant_to_structure_terms,
        active_overlap_matrix,
        active_one_electron_matrix,
        n_active_orbitals,
        zero_eri,
        subset_structure_data.n_structures);
    const double exact_build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - exact_build_started_at).count();

    const int n_selected_structures = static_cast<int>(selected_structures.size());
    const std::size_t matrix_size =
        xmvb::to_size(n_selected_structures) *
        xmvb::to_size(n_selected_structures);
    std::vector<double> candidate_overlap_matrix_permanent(matrix_size, 0.0);
    std::vector<double> candidate_one_electron_matrix_permanent(matrix_size, 0.0);
    std::vector<double> candidate_total_matrix_permanent(matrix_size, 0.0);
    std::vector<double> candidate_overlap_matrix_determinant(matrix_size, 0.0);
    std::vector<double> candidate_one_electron_matrix_determinant(matrix_size, 0.0);
    std::vector<double> candidate_total_matrix_determinant(matrix_size, 0.0);

    std::map<std::string, std::vector<xmvb::vb::LegacyStructureDeterminantTerm>> term_cache;
    std::map<std::string, StructurePairExactValues> exact_value_cache;
    TwoPairKernelDiagnostics diagnostics;

    const auto candidate_started_at = std::chrono::steady_clock::now();
    for (int row = 0; row < n_selected_structures; ++row) {
      const auto& left_structure = selected_structures[xmvb::to_size(row)];
      for (int column = 0; column <= row; ++column) {
        const auto& right_structure =
            selected_structures[xmvb::to_size(column)];

        Matrix overlap_kernel(n_pairs, n_pairs);
        Matrix one_electron_kernel(n_pairs, n_pairs);
        Matrix total_kernel(n_pairs, n_pairs);
        for (int left_pair_index = 0; left_pair_index < n_pairs; ++left_pair_index) {
          for (int right_pair_index = 0; right_pair_index < n_pairs; ++right_pair_index) {
            const auto values = compute_exact_structure_pair_values(
                {left_structure.pairs[xmvb::to_size(left_pair_index)]},
                {right_structure.pairs[xmvb::to_size(right_pair_index)]},
                active_overlap_matrix,
                active_one_electron_matrix,
                n_active_orbitals,
                packed_active_two_electron_integrals,
                &term_cache,
                &exact_value_cache);
            overlap_kernel(left_pair_index, right_pair_index) = values.overlap;
            one_electron_kernel(left_pair_index, right_pair_index) =
                values.one_electron_hamiltonian;
            total_kernel(left_pair_index, right_pair_index) =
                values.total_hamiltonian;
          }
        }

        const double candidate_overlap_permanent =
            matrix_permanent_ryser(overlap_kernel);
        const double candidate_overlap_determinant =
            matrix_determinant(overlap_kernel);
        double candidate_one_electron_permanent = 0.0;
        double candidate_total_permanent = 0.0;
        double candidate_one_electron_determinant = 0.0;
        double candidate_total_determinant = 0.0;
        for (int left_pair_index = 0; left_pair_index < n_pairs; ++left_pair_index) {
          for (int right_pair_index = 0; right_pair_index < n_pairs; ++right_pair_index) {
            const double overlap_minor_permanent = permanent_minor(
                overlap_kernel,
                {left_pair_index},
                {right_pair_index});
            const double overlap_minor_determinant =
                first_cofactor(
                    overlap_kernel,
                    left_pair_index,
                    right_pair_index);
            candidate_one_electron_permanent +=
                one_electron_kernel(left_pair_index, right_pair_index) *
                overlap_minor_permanent;
            candidate_total_permanent +=
                total_kernel(left_pair_index, right_pair_index) *
                overlap_minor_permanent;
            candidate_one_electron_determinant +=
                one_electron_kernel(left_pair_index, right_pair_index) *
                overlap_minor_determinant;
            candidate_total_determinant +=
                total_kernel(left_pair_index, right_pair_index) *
                overlap_minor_determinant;
          }
        }

        for (int left_first = 0; left_first < n_pairs - 1; ++left_first) {
          for (int left_second = left_first + 1; left_second < n_pairs; ++left_second) {
            for (int right_first = 0; right_first < n_pairs - 1; ++right_first) {
              for (int right_second = right_first + 1; right_second < n_pairs; ++right_second) {
                const auto two_pair_values = compute_exact_structure_pair_values(
                    {
                        left_structure.pairs[xmvb::to_size(left_first)],
                        left_structure.pairs[xmvb::to_size(left_second)],
                    },
                    {
                        right_structure.pairs[xmvb::to_size(right_first)],
                        right_structure.pairs[xmvb::to_size(right_second)],
                    },
                    active_overlap_matrix,
                    active_one_electron_matrix,
                    n_active_orbitals,
                    packed_active_two_electron_integrals,
                    &term_cache,
                    &exact_value_cache);

                const double disconnected_overlap_permanent =
                    overlap_kernel(left_first, right_first) *
                        overlap_kernel(left_second, right_second) +
                    overlap_kernel(left_first, right_second) *
                        overlap_kernel(left_second, right_first);
                const double disconnected_one_electron_permanent =
                    one_electron_kernel(left_first, right_first) *
                        overlap_kernel(left_second, right_second) +
                    one_electron_kernel(left_first, right_second) *
                        overlap_kernel(left_second, right_first) +
                    one_electron_kernel(left_second, right_first) *
                        overlap_kernel(left_first, right_second) +
                    one_electron_kernel(left_second, right_second) *
                        overlap_kernel(left_first, right_first);
                const double disconnected_total_permanent =
                    total_kernel(left_first, right_first) *
                        overlap_kernel(left_second, right_second) +
                    total_kernel(left_first, right_second) *
                        overlap_kernel(left_second, right_first) +
                    total_kernel(left_second, right_first) *
                        overlap_kernel(left_first, right_second) +
                    total_kernel(left_second, right_second) *
                        overlap_kernel(left_first, right_first);
                diagnostics.permanent_max_connected_overlap_abs =
                    std::max(
                        diagnostics.permanent_max_connected_overlap_abs,
                        std::abs(
                            two_pair_values.overlap -
                            disconnected_overlap_permanent));
                diagnostics.permanent_max_connected_one_electron_abs =
                    std::max(
                        diagnostics.permanent_max_connected_one_electron_abs,
                        std::abs(
                            two_pair_values.one_electron_hamiltonian -
                            disconnected_one_electron_permanent));
                const double connected_total_permanent =
                    two_pair_values.total_hamiltonian -
                    disconnected_total_permanent;
                candidate_total_permanent +=
                    connected_total_permanent *
                    permanent_minor(
                        overlap_kernel,
                        {left_first, left_second},
                        {right_first, right_second});

                const double disconnected_overlap_determinant =
                    overlap_kernel(left_first, right_first) *
                        overlap_kernel(left_second, right_second) -
                    overlap_kernel(left_first, right_second) *
                        overlap_kernel(left_second, right_first);
                const double disconnected_one_electron_determinant =
                    one_electron_kernel(left_first, right_first) *
                        overlap_kernel(left_second, right_second) -
                    one_electron_kernel(left_first, right_second) *
                        overlap_kernel(left_second, right_first) -
                    one_electron_kernel(left_second, right_first) *
                        overlap_kernel(left_first, right_second) +
                    one_electron_kernel(left_second, right_second) *
                        overlap_kernel(left_first, right_first);
                const double disconnected_total_determinant =
                    total_kernel(left_first, right_first) *
                        overlap_kernel(left_second, right_second) -
                    total_kernel(left_first, right_second) *
                        overlap_kernel(left_second, right_first) -
                    total_kernel(left_second, right_first) *
                        overlap_kernel(left_first, right_second) +
                    total_kernel(left_second, right_second) *
                        overlap_kernel(left_first, right_first);
                diagnostics.determinant_max_connected_overlap_abs =
                    std::max(
                        diagnostics.determinant_max_connected_overlap_abs,
                        std::abs(
                            two_pair_values.overlap -
                            disconnected_overlap_determinant));
                diagnostics.determinant_max_connected_one_electron_abs =
                    std::max(
                        diagnostics.determinant_max_connected_one_electron_abs,
                        std::abs(
                            two_pair_values.one_electron_hamiltonian -
                            disconnected_one_electron_determinant));
                const double connected_total_determinant =
                    two_pair_values.total_hamiltonian -
                    disconnected_total_determinant;
                candidate_total_determinant +=
                    connected_total_determinant *
                    second_cofactor(
                        overlap_kernel,
                        left_first,
                        left_second,
                        right_first,
                        right_second);
              }
            }
          }
        }

        set_symmetric_matrix_entry(
            &candidate_overlap_matrix_permanent,
            n_selected_structures,
            column,
            row,
            candidate_overlap_permanent);
        set_symmetric_matrix_entry(
            &candidate_one_electron_matrix_permanent,
            n_selected_structures,
            column,
            row,
            candidate_one_electron_permanent);
        set_symmetric_matrix_entry(
            &candidate_total_matrix_permanent,
            n_selected_structures,
            column,
            row,
            candidate_total_permanent);
        set_symmetric_matrix_entry(
            &candidate_overlap_matrix_determinant,
            n_selected_structures,
            column,
            row,
            candidate_overlap_determinant);
        set_symmetric_matrix_entry(
            &candidate_one_electron_matrix_determinant,
            n_selected_structures,
            column,
            row,
            candidate_one_electron_determinant);
        set_symmetric_matrix_entry(
            &candidate_total_matrix_determinant,
            n_selected_structures,
            column,
            row,
            candidate_total_determinant);
      }
    }
    const double candidate_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - candidate_started_at).count();

    const auto matrix_entry = [](
        const std::vector<double>& matrix,
        int dimension,
        int row,
        int column) {
      return matrix[xmvb::to_size(column) * dimension + row];
    };

    const auto overlap_summary_permanent = summarize_matrix_error(
        "permanent_overlap",
        exact_structure_matrices.overlap_matrix,
        candidate_overlap_matrix_permanent,
        selected_structures,
        options.report_count,
        options.tolerance,
        matrix_entry);
    const auto one_electron_summary_permanent = summarize_matrix_error(
        "permanent_one_electron",
        exact_one_electron_structure_matrices.hamiltonian_matrix,
        candidate_one_electron_matrix_permanent,
        selected_structures,
        options.report_count,
        options.tolerance,
        matrix_entry);
    const auto total_summary_permanent = summarize_matrix_error(
        "permanent_total_hamiltonian",
        exact_structure_matrices.hamiltonian_matrix,
        candidate_total_matrix_permanent,
        selected_structures,
        options.report_count,
        options.tolerance,
        matrix_entry);
    const auto overlap_summary_determinant = summarize_matrix_error(
        "determinant_overlap",
        exact_structure_matrices.overlap_matrix,
        candidate_overlap_matrix_determinant,
        selected_structures,
        options.report_count,
        options.tolerance,
        matrix_entry);
    const auto one_electron_summary_determinant = summarize_matrix_error(
        "determinant_one_electron",
        exact_one_electron_structure_matrices.hamiltonian_matrix,
        candidate_one_electron_matrix_determinant,
        selected_structures,
        options.report_count,
        options.tolerance,
        matrix_entry);
    const auto total_summary_determinant = summarize_matrix_error(
        "determinant_total_hamiltonian",
        exact_structure_matrices.hamiltonian_matrix,
        candidate_total_matrix_determinant,
        selected_structures,
        options.report_count,
        options.tolerance,
        matrix_entry);

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "candidate_model = covalent_pair_cluster\n";
    std::cout << "available_structures = " << raw_structure_data.n_structures << '\n';
    std::cout << "validated_structures = " << n_selected_structures << '\n';
    std::cout << "selected_structure_indices = [";
    for (int index = 0; index < n_selected_structures; ++index) {
      if (index > 0) {
        std::cout << " ";
      }
      std::cout << selected_structures[xmvb::to_size(index)].raw_structure_index;
    }
    std::cout << "]\n";
    std::cout << "n_active_electrons = " << raw_structure_data.n_active_electrons << '\n';
    std::cout << "n_active_pairs = " << n_pairs << '\n';
    std::cout << "restrict_covalent = " << options.restrict_covalent << '\n';
    std::cout << "available_disjoint_covalent_structures = "
              << available_disjoint_covalent_structures << '\n';
    std::cout << "tolerance = " << options.tolerance << '\n';
    std::cout << "load_input_wall_time_seconds = " << load_result.total_seconds << '\n';
    std::cout << "prepare_active_space_wall_time_seconds = "
              << (timed_active_space.timings.orbital_preparation_wall_time_seconds +
                  timed_active_space.timings.ao_effective_one_electron_wall_time_seconds +
                  timed_active_space.timings.active_one_electron_wall_time_seconds +
                  timed_active_space.timings.active_two_electron_wall_time_seconds)
              << '\n';
    std::cout << "exact_subset_build_wall_time_seconds = " << exact_build_seconds << '\n';
    std::cout << "candidate_build_wall_time_seconds = " << candidate_seconds << '\n';
    std::cout << "permanent_two_pair_connected_overlap_max_abs = "
              << diagnostics.permanent_max_connected_overlap_abs << '\n';
    std::cout << "permanent_two_pair_connected_one_electron_max_abs = "
              << diagnostics.permanent_max_connected_one_electron_abs << '\n';
    std::cout << "determinant_two_pair_connected_overlap_max_abs = "
              << diagnostics.determinant_max_connected_overlap_abs << '\n';
    std::cout << "determinant_two_pair_connected_one_electron_max_abs = "
              << diagnostics.determinant_max_connected_one_electron_abs << '\n';
    print_matrix_error_summary(overlap_summary_permanent, selected_structures);
    std::cout << '\n';
    print_matrix_error_summary(one_electron_summary_permanent, selected_structures);
    std::cout << '\n';
    print_matrix_error_summary(total_summary_permanent, selected_structures);
    std::cout << '\n';
    print_matrix_error_summary(overlap_summary_determinant, selected_structures);
    std::cout << '\n';
    print_matrix_error_summary(one_electron_summary_determinant, selected_structures);
    std::cout << '\n';
    print_matrix_error_summary(total_summary_determinant, selected_structures);
    std::cout << "total_elapsed_wall_time_seconds = "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - overall_started_at)
                     .count()
              << '\n';

    const bool success =
        (overlap_summary_permanent.mismatch_count == 0 &&
         one_electron_summary_permanent.mismatch_count == 0 &&
         total_summary_permanent.mismatch_count == 0) ||
        (overlap_summary_determinant.mismatch_count == 0 &&
         one_electron_summary_determinant.mismatch_count == 0 &&
         total_summary_determinant.mismatch_count == 0);
    return success ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "check_covalent_pair_cluster_hamiltonian failed: "
              << error.what() << '\n';
    return 1;
  }
}
