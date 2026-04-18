#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/full_structure_expander.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Pair = std::pair<int, int>;
using PairOverlapKey = std::pair<Pair, Pair>;

enum class CandidateKind {
  ExactTerms,
  PairDet,
  PairPerm,
  PfaffianPlus,
  PfaffianMinus,
};

struct Options {
  std::string input_path;
  int max_structures = 8;
  int report_count = 8;
  double tolerance = 1.0e-6;
  bool restrict_covalent = false;
  std::string candidate = "all";
  std::string pair_phase_mode = "legacy";
};

struct DeterminantTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

struct StructureGeminalExpansion {
  int structure_index = 0;
  std::vector<Pair> pairs;
  std::vector<DeterminantTerm> determinant_terms;
  bool is_disjoint_covalent = false;
};

struct PairError {
  int row_local = 0;
  int column_local = 0;
  int row_original = 0;
  int column_original = 0;
  double exact_overlap = 0.0;
  double candidate_overlap = 0.0;
  double absolute_error = 0.0;
  double relative_error = 0.0;
};

struct CandidateReport {
  std::string name;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  int mismatch_count = 0;
  std::vector<PairError> pair_errors;
};

void print_usage() {
  std::cerr << "usage: check_geminal_structure_overlap <input.xmi> "
               "[--max-structures N] [--report-count N] [--tolerance tol] "
               "[--restrict-covalent 0|1] "
               "[--pair-phase-mode legacy|antisymmetric] "
               "[--candidate all|exact_terms|pair_det|pair_perm|pfaffian_plus|pfaffian_minus]\n";
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

std::string candidate_name(CandidateKind candidate_kind) {
  switch (candidate_kind) {
    case CandidateKind::ExactTerms:
      return "exact_terms";
    case CandidateKind::PairDet:
      return "pair_det";
    case CandidateKind::PairPerm:
      return "pair_perm";
    case CandidateKind::PfaffianPlus:
      return "pfaffian_plus";
    case CandidateKind::PfaffianMinus:
      return "pfaffian_minus";
  }
  throw std::invalid_argument("unhandled candidate kind");
}

std::vector<CandidateKind> parse_candidate_selection(const std::string& candidate_text) {
  if (candidate_text == "all") {
    return {
        CandidateKind::ExactTerms,
        CandidateKind::PairDet,
        CandidateKind::PairPerm,
        CandidateKind::PfaffianPlus,
        CandidateKind::PfaffianMinus,
    };
  }

  for (const CandidateKind candidate_kind : {
           CandidateKind::ExactTerms,
           CandidateKind::PairDet,
           CandidateKind::PairPerm,
           CandidateKind::PfaffianPlus,
           CandidateKind::PfaffianMinus,
       }) {
    if (candidate_text == candidate_name(candidate_kind)) {
      return {candidate_kind};
    }
  }

  throw std::invalid_argument("unknown candidate: " + candidate_text);
}

int parse_pair_swapped_term_phase(const std::string& pair_phase_mode) {
  if (pair_phase_mode == "legacy") {
    return 1;
  }
  if (pair_phase_mode == "antisymmetric") {
    return -1;
  }
  throw std::invalid_argument("unknown --pair-phase-mode: " + pair_phase_mode);
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
    if (argument_name == "--candidate") {
      options.candidate = argument_value;
      continue;
    }
    if (argument_name == "--pair-phase-mode") {
      options.pair_phase_mode = argument_value;
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
  parse_candidate_selection(options.candidate);
  parse_pair_swapped_term_phase(options.pair_phase_mode);
  return options;
}

int packed_active_two_electron_size(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  return xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
             n_active_orbitals - 1,
             n_active_orbitals - 1,
             n_active_orbitals - 1,
             n_active_orbitals - 1) +
      1;
}

int canonicalize_spin_string(std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  int permutation_sign = 1;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals->size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals->size();
         ++right_index) {
      if ((*occupied_orbitals)[left_index] > (*occupied_orbitals)[right_index]) {
        std::swap((*occupied_orbitals)[left_index], (*occupied_orbitals)[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

std::vector<Pair> extract_active_pairs(
    const xmvb::vb::RawStructureData& raw_structure_data,
    int structure_index) {
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  const int active_start = 2 * n_inactive_doubly_occupied_orbitals;
  const int active_stop = active_start + raw_structure_data.n_active_electrons;
  if (structure_index < 0 || structure_index >= raw_structure_data.n_structures) {
    throw std::out_of_range("structure index out of range");
  }
  if (active_start < 0 || active_stop > raw_structure_data.n_total_electrons) {
    throw std::runtime_error("active-electron window is out of range for raw structures");
  }

  const int* structure_orbitals = raw_structure_data.structure_orbitals_data(structure_index);
  std::vector<Pair> pairs;
  pairs.reserve(xmvb::to_size(n_active_beta_electrons));
  for (int pair_index = 0; pair_index < n_active_beta_electrons; ++pair_index) {
    const int left_orbital =
        structure_orbitals[active_start + 2 * pair_index] -
        n_inactive_doubly_occupied_orbitals - 1;
    const int right_orbital =
        structure_orbitals[active_start + 2 * pair_index + 1] -
        n_inactive_doubly_occupied_orbitals - 1;
    pairs.emplace_back(left_orbital, right_orbital);
  }
  return pairs;
}

std::vector<DeterminantTerm> enumerate_geminal_determinant_terms(
    const std::vector<Pair>& pairs,
    int swapped_term_phase) {
  std::vector<DeterminantTerm> partial_terms;
  DeterminantTerm initial_term;
  initial_term.coefficient = 1.0;
  partial_terms.push_back(std::move(initial_term));

  for (const auto& pair : pairs) {
    const int left_orbital = pair.first;
    const int right_orbital = pair.second;
    const bool diagonal_pair = left_orbital == right_orbital;
    std::vector<DeterminantTerm> next_terms;
    next_terms.reserve(
        partial_terms.size() * xmvb::to_size(diagonal_pair ? 1 : 2));
    for (const auto& partial_term : partial_terms) {
      DeterminantTerm direct_term = partial_term;
      direct_term.alpha_occ.push_back(left_orbital);
      direct_term.beta_occ.push_back(right_orbital);
      next_terms.push_back(std::move(direct_term));
      if (!diagonal_pair) {
        DeterminantTerm swapped_term = partial_term;
        swapped_term.alpha_occ.push_back(right_orbital);
        swapped_term.beta_occ.push_back(left_orbital);
        swapped_term.coefficient *= static_cast<double>(swapped_term_phase);
        next_terms.push_back(std::move(swapped_term));
      }
    }
    partial_terms = std::move(next_terms);
  }

  std::map<std::pair<std::vector<int>, std::vector<int>>, double> accumulated_terms;
  for (auto& term : partial_terms) {
    const int alpha_sign = canonicalize_spin_string(&term.alpha_occ);
    const int beta_sign = canonicalize_spin_string(&term.beta_occ);
    accumulated_terms[{term.alpha_occ, term.beta_occ}] +=
        term.coefficient * static_cast<double>(alpha_sign * beta_sign);
  }

  std::vector<DeterminantTerm> result;
  result.reserve(accumulated_terms.size());
  for (const auto& [key, coefficient] : accumulated_terms) {
    if (std::abs(coefficient) <= 1.0e-12) {
      continue;
    }
    DeterminantTerm term;
    term.alpha_occ = key.first;
    term.beta_occ = key.second;
    term.coefficient = coefficient;
    result.push_back(std::move(term));
  }
  return result;
}

bool is_disjoint_covalent_structure(
    const std::vector<Pair>& pairs,
    int n_active_orbitals) {
  std::vector<int> counts(xmvb::to_size(n_active_orbitals), 0);
  for (const auto& pair : pairs) {
    if (pair.first == pair.second) {
      return false;
    }
    if (pair.first < 0 || pair.first >= n_active_orbitals ||
        pair.second < 0 || pair.second >= n_active_orbitals) {
      throw std::runtime_error("pair orbital is out of active-space range");
    }
    ++counts[xmvb::to_size(pair.first)];
    ++counts[xmvb::to_size(pair.second)];
    if (counts[xmvb::to_size(pair.first)] > 1 ||
        counts[xmvb::to_size(pair.second)] > 1) {
      return false;
    }
  }
  return true;
}

StructureGeminalExpansion build_structure_geminal_expansion(
    const xmvb::vb::RawStructureData& raw_structure_data,
    int structure_index,
    int n_active_orbitals,
    int swapped_term_phase) {
  StructureGeminalExpansion expansion;
  expansion.structure_index = structure_index;
  expansion.pairs = extract_active_pairs(raw_structure_data, structure_index);
  expansion.determinant_terms =
      enumerate_geminal_determinant_terms(expansion.pairs, swapped_term_phase);
  expansion.is_disjoint_covalent =
      is_disjoint_covalent_structure(expansion.pairs, n_active_orbitals);
  return expansion;
}

double active_overlap_element(
    int row,
    int column,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals) {
  if (row < 0 || row >= n_active_orbitals || column < 0 || column >= n_active_orbitals) {
    throw std::out_of_range("active overlap index is out of range");
  }
  return orbital_overlap_matrix[xmvb::to_size(column) *
                                    xmvb::to_size(n_active_orbitals) +
                                row];
}

double determinant_overlap(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      left_occ,
      right_occ,
      orbital_overlap_matrix,
      n_active_orbitals);
  return overlap_resolver
      .resolve(overlap_submatrix, static_cast<int>(left_occ.size()))
      .overlap_determinant;
}

double geminal_structure_overlap(
    const StructureGeminalExpansion& left_structure,
    const StructureGeminalExpansion& right_structure,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  double overlap_value = 0.0;
  for (const auto& left_term : left_structure.determinant_terms) {
    for (const auto& right_term : right_structure.determinant_terms) {
      const double alpha_overlap = determinant_overlap(
          left_term.alpha_occ,
          right_term.alpha_occ,
          orbital_overlap_matrix,
          n_active_orbitals,
          overlap_resolver);
      const double beta_overlap = determinant_overlap(
          left_term.beta_occ,
          right_term.beta_occ,
          orbital_overlap_matrix,
          n_active_orbitals,
          overlap_resolver);
      overlap_value +=
          left_term.coefficient * right_term.coefficient * alpha_overlap * beta_overlap;
    }
  }
  return overlap_value;
}

const std::vector<DeterminantTerm>& cached_single_pair_terms(
    const Pair& pair,
    int swapped_term_phase,
    std::map<Pair, std::vector<DeterminantTerm>>* pair_term_cache) {
  if (pair_term_cache == nullptr) {
    throw std::invalid_argument("pair_term_cache must not be null");
  }
  const auto iterator = pair_term_cache->find(pair);
  if (iterator != pair_term_cache->end()) {
    return iterator->second;
  }
  auto [inserted_iterator, inserted] = pair_term_cache->emplace(
      pair,
      enumerate_geminal_determinant_terms({pair}, swapped_term_phase));
  (void)inserted;
  return inserted_iterator->second;
}

double single_pair_overlap(
    const Pair& left_pair,
    const Pair& right_pair,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    int swapped_term_phase,
    std::map<Pair, std::vector<DeterminantTerm>>* pair_term_cache,
    std::map<PairOverlapKey, double>* pair_overlap_cache) {
  if (pair_overlap_cache == nullptr) {
    throw std::invalid_argument("pair_overlap_cache must not be null");
  }
  const PairOverlapKey key{left_pair, right_pair};
  const auto iterator = pair_overlap_cache->find(key);
  if (iterator != pair_overlap_cache->end()) {
    return iterator->second;
  }

  const auto& left_terms =
      cached_single_pair_terms(left_pair, swapped_term_phase, pair_term_cache);
  const auto& right_terms =
      cached_single_pair_terms(right_pair, swapped_term_phase, pair_term_cache);
  double overlap_value = 0.0;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const double alpha_overlap = determinant_overlap(
          left_term.alpha_occ,
          right_term.alpha_occ,
          orbital_overlap_matrix,
          n_active_orbitals,
          overlap_resolver);
      const double beta_overlap = determinant_overlap(
          left_term.beta_occ,
          right_term.beta_occ,
          orbital_overlap_matrix,
          n_active_orbitals,
          overlap_resolver);
      overlap_value +=
          left_term.coefficient * right_term.coefficient * alpha_overlap * beta_overlap;
    }
  }

  (*pair_overlap_cache)[key] = overlap_value;
  return overlap_value;
}

xmvb::vb::Matrix build_pair_overlap_kernel(
    const StructureGeminalExpansion& left_structure,
    const StructureGeminalExpansion& right_structure,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    int swapped_term_phase,
    std::map<Pair, std::vector<DeterminantTerm>>* pair_term_cache,
    std::map<PairOverlapKey, double>* pair_overlap_cache) {
  const int n_left_pairs = static_cast<int>(left_structure.pairs.size());
  const int n_right_pairs = static_cast<int>(right_structure.pairs.size());
  if (n_left_pairs != n_right_pairs) {
    throw std::runtime_error("pair overlap kernel expects equal pair counts");
  }

  xmvb::vb::Matrix pair_overlap_matrix(n_left_pairs, n_right_pairs);
  for (int row = 0; row < n_left_pairs; ++row) {
    for (int column = 0; column < n_right_pairs; ++column) {
      pair_overlap_matrix(row, column) = single_pair_overlap(
          left_structure.pairs[xmvb::to_size(row)],
          right_structure.pairs[xmvb::to_size(column)],
          orbital_overlap_matrix,
          n_active_orbitals,
          overlap_resolver,
          swapped_term_phase,
          pair_term_cache,
          pair_overlap_cache);
    }
  }
  return pair_overlap_matrix;
}

double matrix_permanent_ryser(const xmvb::vb::Matrix& matrix) {
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

double pair_determinant_candidate(
    const StructureGeminalExpansion& left_structure,
    const StructureGeminalExpansion& right_structure,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    int swapped_term_phase,
    std::map<Pair, std::vector<DeterminantTerm>>* pair_term_cache,
    std::map<PairOverlapKey, double>* pair_overlap_cache) {
  const auto pair_overlap_matrix = build_pair_overlap_kernel(
      left_structure,
      right_structure,
      orbital_overlap_matrix,
      n_active_orbitals,
      overlap_resolver,
      swapped_term_phase,
      pair_term_cache,
      pair_overlap_cache);
  return pair_overlap_matrix.fullPivLu().determinant();
}

double pair_permanent_candidate(
    const StructureGeminalExpansion& left_structure,
    const StructureGeminalExpansion& right_structure,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    int swapped_term_phase,
    std::map<Pair, std::vector<DeterminantTerm>>* pair_term_cache,
    std::map<PairOverlapKey, double>* pair_overlap_cache) {
  const auto pair_overlap_matrix = build_pair_overlap_kernel(
      left_structure,
      right_structure,
      orbital_overlap_matrix,
      n_active_orbitals,
      overlap_resolver,
      swapped_term_phase,
      pair_term_cache,
      pair_overlap_cache);
  return matrix_permanent_ryser(pair_overlap_matrix);
}

std::vector<int> build_structure_support(
    const StructureGeminalExpansion& left_structure,
    const StructureGeminalExpansion& right_structure) {
  std::vector<int> support_orbitals;
  for (const auto& pair : left_structure.pairs) {
    support_orbitals.push_back(pair.first);
    support_orbitals.push_back(pair.second);
  }
  for (const auto& pair : right_structure.pairs) {
    support_orbitals.push_back(pair.first);
    support_orbitals.push_back(pair.second);
  }
  std::sort(support_orbitals.begin(), support_orbitals.end());
  support_orbitals.erase(
      std::unique(support_orbitals.begin(), support_orbitals.end()),
      support_orbitals.end());
  return support_orbitals;
}

xmvb::vb::Matrix build_spatial_overlap_support(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals) {
  const int support_size = static_cast<int>(support_orbitals.size());
  xmvb::vb::Matrix support_overlap(support_size, support_size);
  for (int row = 0; row < support_size; ++row) {
    for (int column = 0; column < support_size; ++column) {
      support_overlap(row, column) = active_overlap_element(
          support_orbitals[xmvb::to_size(row)],
          support_orbitals[xmvb::to_size(column)],
          orbital_overlap_matrix,
          n_active_orbitals);
    }
  }
  return support_overlap;
}

xmvb::vb::Matrix build_spin_orbital_overlap_support(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals) {
  const auto spatial_overlap = build_spatial_overlap_support(
      support_orbitals,
      orbital_overlap_matrix,
      n_active_orbitals);
  const int support_size = spatial_overlap.rows();
  xmvb::vb::Matrix spin_overlap =
      xmvb::vb::Matrix::Zero(2 * support_size, 2 * support_size);
  spin_overlap.topLeftCorner(support_size, support_size) = spatial_overlap;
  spin_overlap.bottomRightCorner(support_size, support_size) = spatial_overlap;
  return spin_overlap;
}

xmvb::vb::Matrix build_pairing_matrix(
    const std::vector<Pair>& pairs,
    const std::vector<int>& support_orbitals) {
  const int support_size = static_cast<int>(support_orbitals.size());
  std::map<int, int> support_index;
  for (int support_position = 0; support_position < support_size; ++support_position) {
    support_index.emplace(
        support_orbitals[xmvb::to_size(support_position)],
        support_position);
  }

  xmvb::vb::Matrix pairing_matrix =
      xmvb::vb::Matrix::Zero(2 * support_size, 2 * support_size);
  for (const auto& pair : pairs) {
    const auto left_iterator = support_index.find(pair.first);
    const auto right_iterator = support_index.find(pair.second);
    if (left_iterator == support_index.end() || right_iterator == support_index.end()) {
      throw std::runtime_error("pairing support orbital not found");
    }
    const int left_index = left_iterator->second;
    const int right_index = right_iterator->second;
    if (left_index == right_index) {
      pairing_matrix(left_index, support_size + right_index) += 1.0;
      pairing_matrix(support_size + right_index, left_index) -= 1.0;
      continue;
    }
    pairing_matrix(left_index, support_size + right_index) += 1.0;
    pairing_matrix(support_size + right_index, left_index) -= 1.0;
    pairing_matrix(right_index, support_size + left_index) -= 1.0;
    pairing_matrix(support_size + left_index, right_index) += 1.0;
  }
  return pairing_matrix;
}

double skew_symmetric_pfaffian(xmvb::vb::Matrix matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("Pfaffian requires a square matrix");
  }
  const int dimension = matrix.rows();
  if ((dimension % 2) != 0) {
    return 0.0;
  }
  if (dimension == 0) {
    return 1.0;
  }

  double pfaffian_value = 1.0;
  for (int pivot_row = 0; pivot_row < dimension - 1; pivot_row += 2) {
    int pivot_column = pivot_row + 1;
    double pivot_magnitude = std::abs(matrix(pivot_row, pivot_column));
    for (int column = pivot_row + 2; column < dimension; ++column) {
      const double candidate_magnitude = std::abs(matrix(pivot_row, column));
      if (candidate_magnitude > pivot_magnitude) {
        pivot_magnitude = candidate_magnitude;
        pivot_column = column;
      }
    }

    if (pivot_magnitude <= 1.0e-14) {
      return 0.0;
    }

    if (pivot_column != pivot_row + 1) {
      matrix.row(pivot_row + 1).swap(matrix.row(pivot_column));
      matrix.col(pivot_row + 1).swap(matrix.col(pivot_column));
      pfaffian_value = -pfaffian_value;
    }

    const double pivot_value = matrix(pivot_row, pivot_row + 1);
    pfaffian_value *= pivot_value;
    for (int row = pivot_row + 2; row < dimension; ++row) {
      for (int column = row + 1; column < dimension; ++column) {
        const double updated_value =
            matrix(row, column) -
            (matrix(pivot_row, row) * matrix(pivot_row + 1, column) -
             matrix(pivot_row, column) * matrix(pivot_row + 1, row)) /
                pivot_value;
        matrix(row, column) = updated_value;
        matrix(column, row) = -updated_value;
      }
    }
  }
  return pfaffian_value;
}

double pfaffian_metric_candidate(
    const StructureGeminalExpansion& left_structure,
    const StructureGeminalExpansion& right_structure,
    const std::vector<double>& orbital_overlap_matrix,
    int n_active_orbitals,
    double right_pairing_sign) {
  const auto support_orbitals = build_structure_support(left_structure, right_structure);
  const auto spin_orbital_overlap = build_spin_orbital_overlap_support(
      support_orbitals,
      orbital_overlap_matrix,
      n_active_orbitals);
  const auto left_pairing_matrix =
      build_pairing_matrix(left_structure.pairs, support_orbitals);
  auto right_pairing_matrix =
      build_pairing_matrix(right_structure.pairs, support_orbitals);
  right_pairing_matrix *= right_pairing_sign;

  const int spin_dimension = spin_orbital_overlap.rows();
  xmvb::vb::Matrix pfaffian_matrix =
      xmvb::vb::Matrix::Zero(2 * spin_dimension, 2 * spin_dimension);
  pfaffian_matrix.topLeftCorner(spin_dimension, spin_dimension) =
      left_pairing_matrix;
  pfaffian_matrix.topRightCorner(spin_dimension, spin_dimension) =
      spin_orbital_overlap;
  pfaffian_matrix.bottomLeftCorner(spin_dimension, spin_dimension) =
      -spin_orbital_overlap.transpose();
  pfaffian_matrix.bottomRightCorner(spin_dimension, spin_dimension) =
      right_pairing_matrix;
  return skew_symmetric_pfaffian(std::move(pfaffian_matrix));
}

std::string format_indices(const std::vector<int>& indices) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index > 0) {
      stream << " ";
    }
    stream << indices[index];
  }
  stream << "]";
  return stream.str();
}

std::string format_pairs(const std::vector<Pair>& pairs) {
  std::string text = "[";
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pair_index > 0) {
      text += " ";
    }
    text += "(" + std::to_string(pairs[pair_index].first + 1) + "," +
        std::to_string(pairs[pair_index].second + 1) + ")";
  }
  text += "]";
  return text;
}

template <typename Evaluator>
CandidateReport evaluate_candidate(
    const std::string& name,
    const std::vector<StructureGeminalExpansion>& geminal_expansions,
    const std::vector<int>& selected_structure_indices,
    const std::vector<double>& exact_overlap_matrix,
    int n_selected_structures,
    double tolerance,
    const Evaluator& evaluator) {
  CandidateReport report;
  report.name = name;
  report.pair_errors.reserve(xmvb::to_size(n_selected_structures) *
                             xmvb::to_size(n_selected_structures + 1) / 2);

  for (int row = 0; row < n_selected_structures; ++row) {
    for (int column = 0; column <= row; ++column) {
      const double candidate_overlap_value = evaluator(
          geminal_expansions[xmvb::to_size(row)],
          geminal_expansions[xmvb::to_size(column)]);
      const double exact_overlap_value =
          exact_overlap_matrix[xmvb::to_size(column) *
                                   xmvb::to_size(n_selected_structures) +
                               row];
      const double absolute_error = std::abs(exact_overlap_value - candidate_overlap_value);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(exact_overlap_value));
      if (absolute_error > tolerance) {
        ++report.mismatch_count;
      }
      report.max_abs_error = std::max(report.max_abs_error, absolute_error);
      report.max_rel_error = std::max(report.max_rel_error, relative_error);

      PairError error;
      error.row_local = row;
      error.column_local = column;
      error.row_original = selected_structure_indices[xmvb::to_size(row)];
      error.column_original = selected_structure_indices[xmvb::to_size(column)];
      error.exact_overlap = exact_overlap_value;
      error.candidate_overlap = candidate_overlap_value;
      error.absolute_error = absolute_error;
      error.relative_error = relative_error;
      report.pair_errors.push_back(error);
    }
  }

  std::sort(
      report.pair_errors.begin(),
      report.pair_errors.end(),
      [](const PairError& left, const PairError& right) {
        if (left.absolute_error != right.absolute_error) {
          return left.absolute_error > right.absolute_error;
        }
        if (left.row_local != right.row_local) {
          return left.row_local < right.row_local;
        }
        return left.column_local < right.column_local;
      });
  return report;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto selected_candidates = parse_candidate_selection(options.candidate);
    const int swapped_term_phase = parse_pair_swapped_term_phase(options.pair_phase_mode);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;
    const auto& active_overlap_matrix =
        load_result.input.orbital_preparation_input.active_orbital_overlap_matrix;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "check_geminal_structure_overlap currently supports only singlet closed-shell structures");
    }
    if (raw_structure_data.n_active_electrons % 2 != 0) {
      throw std::runtime_error("n_active_electrons must be even for pair-geminal validation");
    }

    int total_disjoint_covalent_structures = 0;
    std::vector<int> selected_structure_indices;
    std::vector<StructureGeminalExpansion> geminal_expansions;
    selected_structure_indices.reserve(xmvb::to_size(
        options.max_structures > 0 ? options.max_structures : raw_structure_data.n_structures));
    geminal_expansions.reserve(selected_structure_indices.capacity());

    for (int structure_index = 0; structure_index < raw_structure_data.n_structures; ++structure_index) {
      auto expansion = build_structure_geminal_expansion(
          raw_structure_data,
          structure_index,
          n_active_orbitals,
          swapped_term_phase);
      if (expansion.is_disjoint_covalent) {
        ++total_disjoint_covalent_structures;
      }
      if (options.restrict_covalent && !expansion.is_disjoint_covalent) {
        continue;
      }
      selected_structure_indices.push_back(structure_index);
      geminal_expansions.push_back(std::move(expansion));
      if (options.max_structures > 0 &&
          static_cast<int>(selected_structure_indices.size()) >= options.max_structures) {
        break;
      }
    }

    const int n_selected_structures = static_cast<int>(selected_structure_indices.size());
    if (n_selected_structures <= 0) {
      throw std::runtime_error("no structures selected for validation");
    }

    const std::vector<double> zero_h1e(
        xmvb::to_size(n_active_orbitals) *
            xmvb::to_size(n_active_orbitals),
        0.0);
    const std::vector<double> zero_eri(
        xmvb::to_size(packed_active_two_electron_size(n_active_orbitals)),
        0.0);

    xmvb::vb::FullDeterminantStructureExpander expander;
    const auto subset_structure_data =
        expander.expand_subset(raw_structure_data, selected_structure_indices);
    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
    const auto exact_overlap_result = structure_builder.build(
        subset_structure_data.alpha_det,
        subset_structure_data.beta_det,
        subset_structure_data.determinant_to_structure_terms,
        active_overlap_matrix,
        zero_h1e,
        n_active_orbitals,
        zero_eri,
        subset_structure_data.n_structures);

    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    std::map<Pair, std::vector<DeterminantTerm>> pair_term_cache;
    std::map<PairOverlapKey, double> pair_overlap_cache;
    std::vector<CandidateReport> reports;
    reports.reserve(selected_candidates.size());

    for (const CandidateKind candidate_kind : selected_candidates) {
      switch (candidate_kind) {
        case CandidateKind::ExactTerms:
          reports.push_back(evaluate_candidate(
              candidate_name(candidate_kind),
              geminal_expansions,
              selected_structure_indices,
              exact_overlap_result.overlap_matrix,
              n_selected_structures,
              options.tolerance,
              [&](const StructureGeminalExpansion& left_structure,
                  const StructureGeminalExpansion& right_structure) {
                return geminal_structure_overlap(
                    left_structure,
                    right_structure,
                    active_overlap_matrix,
                    n_active_orbitals,
                    overlap_resolver);
              }));
          break;

        case CandidateKind::PairDet:
          reports.push_back(evaluate_candidate(
              candidate_name(candidate_kind),
              geminal_expansions,
              selected_structure_indices,
              exact_overlap_result.overlap_matrix,
              n_selected_structures,
              options.tolerance,
              [&](const StructureGeminalExpansion& left_structure,
                  const StructureGeminalExpansion& right_structure) {
                return pair_determinant_candidate(
                    left_structure,
                    right_structure,
                    active_overlap_matrix,
                    n_active_orbitals,
                    overlap_resolver,
                    swapped_term_phase,
                    &pair_term_cache,
                    &pair_overlap_cache);
              }));
          break;

        case CandidateKind::PairPerm:
          reports.push_back(evaluate_candidate(
              candidate_name(candidate_kind),
              geminal_expansions,
              selected_structure_indices,
              exact_overlap_result.overlap_matrix,
              n_selected_structures,
              options.tolerance,
              [&](const StructureGeminalExpansion& left_structure,
                  const StructureGeminalExpansion& right_structure) {
                return pair_permanent_candidate(
                    left_structure,
                    right_structure,
                    active_overlap_matrix,
                    n_active_orbitals,
                    overlap_resolver,
                    swapped_term_phase,
                    &pair_term_cache,
                    &pair_overlap_cache);
              }));
          break;

        case CandidateKind::PfaffianPlus:
          reports.push_back(evaluate_candidate(
              candidate_name(candidate_kind),
              geminal_expansions,
              selected_structure_indices,
              exact_overlap_result.overlap_matrix,
              n_selected_structures,
              options.tolerance,
              [&](const StructureGeminalExpansion& left_structure,
                  const StructureGeminalExpansion& right_structure) {
                return pfaffian_metric_candidate(
                    left_structure,
                    right_structure,
                    active_overlap_matrix,
                    n_active_orbitals,
                    1.0);
              }));
          break;

        case CandidateKind::PfaffianMinus:
          reports.push_back(evaluate_candidate(
              candidate_name(candidate_kind),
              geminal_expansions,
              selected_structure_indices,
              exact_overlap_result.overlap_matrix,
              n_selected_structures,
              options.tolerance,
              [&](const StructureGeminalExpansion& left_structure,
                  const StructureGeminalExpansion& right_structure) {
                return pfaffian_metric_candidate(
                    left_structure,
                    right_structure,
                    active_overlap_matrix,
                    n_active_orbitals,
                    -1.0);
              }));
          break;
      }
    }

    bool exact_terms_ok = true;
    bool requested_candidate_ok = true;
    for (const auto& report : reports) {
      if (report.name == "exact_terms" && report.mismatch_count != 0) {
        exact_terms_ok = false;
      }
      if (options.candidate != "all" && report.mismatch_count != 0) {
        requested_candidate_ok = false;
      }
    }

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "available_structures = " << raw_structure_data.n_structures << '\n';
    std::cout << "validated_structures = " << n_selected_structures << '\n';
    std::cout << "selected_structure_indices = " << format_indices(selected_structure_indices)
              << '\n';
    std::cout << "n_active_electrons = " << raw_structure_data.n_active_electrons << '\n';
    std::cout << "n_active_pairs = " << raw_structure_data.n_active_electrons / 2 << '\n';
    std::cout << "restrict_covalent = " << (options.restrict_covalent ? 1 : 0) << '\n';
    std::cout << "available_disjoint_covalent_structures = "
              << total_disjoint_covalent_structures << '\n';
    std::cout << "candidate_selection = " << options.candidate << '\n';
    std::cout << "pair_phase_mode = " << options.pair_phase_mode << '\n';
    std::cout << "tolerance = " << options.tolerance << '\n';
    std::cout << "baseline_exact_terms_ok = " << (exact_terms_ok ? 1 : 0) << '\n';

    for (const auto& report : reports) {
      std::cout << '\n';
      std::cout << "candidate = " << report.name << '\n';
      std::cout << "max_abs_error = " << report.max_abs_error << '\n';
      std::cout << "max_rel_error = " << report.max_rel_error << '\n';
      std::cout << "mismatch_count = " << report.mismatch_count << '\n';

      const int n_to_report =
          std::min(options.report_count, static_cast<int>(report.pair_errors.size()));
      for (int report_index = 0; report_index < n_to_report; ++report_index) {
        const auto& error = report.pair_errors[xmvb::to_size(report_index)];
        const auto& left_structure =
            geminal_expansions[xmvb::to_size(error.row_local)];
        const auto& right_structure =
            geminal_expansions[xmvb::to_size(error.column_local)];
        std::cout << "pair[" << report_index << "]"
                  << " row_local=" << error.row_local
                  << " column_local=" << error.column_local
                  << " row_raw=" << error.row_original
                  << " column_raw=" << error.column_original
                  << " exact=" << error.exact_overlap
                  << " candidate=" << error.candidate_overlap
                  << " abs_error=" << error.absolute_error
                  << " rel_error=" << error.relative_error
                  << " left_terms=" << left_structure.determinant_terms.size()
                  << " right_terms=" << right_structure.determinant_terms.size()
                  << " left_disjoint_covalent="
                  << (left_structure.is_disjoint_covalent ? 1 : 0)
                  << " right_disjoint_covalent="
                  << (right_structure.is_disjoint_covalent ? 1 : 0)
                  << " left_pairs=" << format_pairs(left_structure.pairs)
                  << " right_pairs=" << format_pairs(right_structure.pairs)
                  << '\n';
      }
    }

    return (exact_terms_ok && requested_candidate_ok) ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
