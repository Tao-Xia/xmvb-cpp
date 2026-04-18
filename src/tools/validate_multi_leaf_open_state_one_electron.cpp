#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using OrbitalPair = xmvb::vb::OrbitalPair;
using LegacyTerm = xmvb::vb::LegacyStructureDeterminantTerm;
using CanonicalDeterminantKey = std::pair<std::vector<int>, std::vector<int>>;
using FullDeterminantPairKey =
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>;
using CofactorKey = std::pair<int, int>;

enum class PartialSide {
  LeftFrontier,
  RightComplement,
};

struct Options {
  std::string input_path;
  std::string synthetic_case;
  int left_structure = -1;
  int right_structure = -1;
  int dump_pair_mismatches = 0;
  double tolerance = 1.0e-10;
};

struct ExactSpinValue {
  double overlap = 0.0;
  double one_electron = 0.0;
  std::map<CofactorKey, double> one_electron_pair_weights;
};

struct MinorEntry {
  int local_index = -1;
  int orbital_label = -1;
  double value = 0.0;
};

struct CofactorEntry {
  int row_local_index = -1;
  int col_local_index = -1;
  int row_orbital_label = -1;
  int col_orbital_label = -1;
  double value = 0.0;
};

struct PartialSpinPayload {
  int n_rows = 0;
  int n_cols = 0;
  double closed_overlap = 0.0;
  std::vector<MinorEntry> row_open_entries;
  std::vector<MinorEntry> col_open_entries;
  std::vector<CofactorEntry> cofactor_entries;
};

struct SpinMaskPayload {
  std::uint64_t row_mask = 0;
  std::uint64_t col_mask = 0;
  PartialSpinPayload payload;
};

struct StructurePayload {
  double overlap = 0.0;
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  std::map<int, double> alpha_row_open;
  std::map<int, double> alpha_col_open;
  std::map<int, double> beta_row_open;
  std::map<int, double> beta_col_open;
  std::map<CofactorKey, double> alpha_cofactor;
  std::map<CofactorKey, double> beta_cofactor;
};

struct StructureLeafMessage {
  std::uint64_t alpha_row_mask = 0;
  std::uint64_t alpha_col_mask = 0;
  std::uint64_t beta_row_mask = 0;
  std::uint64_t beta_col_mask = 0;
  StructurePayload payload;
};

struct OrientationTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

struct ComponentData {
  std::vector<OrbitalPair> left_pairs;
  std::vector<OrbitalPair> right_pairs;
  std::vector<OrientationTerm> left_orientation_terms;
  std::vector<OrientationTerm> right_orientation_terms;
};

struct StructureExactValue {
  double overlap = 0.0;
  double one_electron = 0.0;
  std::map<CofactorKey, double> one_electron_pair_weights;
};

struct SpinMinorKey {
  std::vector<int> left_occ;
  std::vector<int> right_occ;
  std::vector<int> left_root_flags;
  std::vector<int> right_root_flags;
};

bool operator<(
    const SpinMinorKey& left,
    const SpinMinorKey& right) {
  return std::tie(
             left.left_occ,
             left.right_occ,
             left.left_root_flags,
             left.right_root_flags) <
      std::tie(
             right.left_occ,
             right.right_occ,
             right.left_root_flags,
             right.right_root_flags);
}

struct WorkCollector {
  std::set<FullDeterminantPairKey> reference_determinant_pairs;
  std::set<CanonicalDeterminantKey> reference_spin_determinants;
  std::set<SpinMinorKey> open_state_spin_work;
};

struct LeafMessageStateKey {
  int leaf_index = 0;
  std::vector<int> left_root_alpha_occ;
  std::vector<int> left_root_beta_occ;
  std::vector<int> right_root_alpha_occ;
  std::vector<int> right_root_beta_occ;
  std::uint64_t alpha_row_mask = 0;
  std::uint64_t alpha_col_mask = 0;
  std::uint64_t beta_row_mask = 0;
  std::uint64_t beta_col_mask = 0;
};

bool operator<(
    const LeafMessageStateKey& left,
    const LeafMessageStateKey& right) {
  return std::tie(
             left.leaf_index,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.alpha_row_mask,
             left.alpha_col_mask,
             left.beta_row_mask,
             left.beta_col_mask) <
      std::tie(
             right.leaf_index,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.alpha_row_mask,
             right.alpha_col_mask,
             right.beta_row_mask,
             right.beta_col_mask);
}

struct MergeStateKey {
  int leaf_index = 0;
  std::vector<int> left_root_alpha_occ;
  std::vector<int> left_root_beta_occ;
  std::vector<int> right_root_alpha_occ;
  std::vector<int> right_root_beta_occ;
  std::uint64_t used_alpha_row_mask = 0;
  std::uint64_t used_alpha_col_mask = 0;
  std::uint64_t used_beta_row_mask = 0;
  std::uint64_t used_beta_col_mask = 0;
};

bool operator<(
    const MergeStateKey& left,
    const MergeStateKey& right) {
  return std::tie(
             left.leaf_index,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.used_alpha_row_mask,
             left.used_alpha_col_mask,
             left.used_beta_row_mask,
             left.used_beta_col_mask) <
      std::tie(
             right.leaf_index,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.used_alpha_row_mask,
             right.used_alpha_col_mask,
             right.used_beta_row_mask,
             right.used_beta_col_mask);
}

struct StateCollector {
  std::set<LeafMessageStateKey> leaf_message_states;
  std::set<MergeStateKey> merge_states;
};

struct ValidationResult {
  int node_count = 0;
  int root_node = -1;
  int n_leaves = 0;
  int support_size = 0;
  double exact_overlap = 0.0;
  double reconstructed_overlap = 0.0;
  double overlap_abs_error = 0.0;
  double exact_one_electron = 0.0;
  double reconstructed_one_electron = 0.0;
  double one_electron_abs_error = 0.0;
  std::size_t unique_reference_determinant_pair_count = 0;
  std::size_t unique_reference_spin_determinant_count = 0;
  std::size_t unique_open_state_spin_work_count = 0;
  std::size_t unique_leaf_message_state_count = 0;
  std::size_t unique_merge_state_count = 0;
  std::size_t combined_open_state_work_count = 0;
  std::map<CofactorKey, double> exact_one_electron_pair_weights;
  std::map<CofactorKey, double> reconstructed_one_electron_pair_weights;
};

struct PairMismatchRecord {
  int left_orbital = -1;
  int right_orbital = -1;
  int left_component = -1;
  int right_component = -1;
  double one_electron_element = 0.0;
  double exact_weight = 0.0;
  double reconstructed_weight = 0.0;
  double exact_contribution = 0.0;
  double reconstructed_contribution = 0.0;
  double contribution_error = 0.0;
};

struct SyntheticCase {
  std::string name;
  int root_node = 0;
  int support_size = 0;
  std::vector<ComponentData> ordered_components;
  std::vector<LegacyTerm> exact_left_terms;
  std::vector<LegacyTerm> exact_right_terms;
  std::vector<double> overlap_storage;
  std::vector<double> one_electron_storage;
};

void print_usage() {
  std::cerr << "usage: validate_multi_leaf_open_state_one_electron <input.xmi>"
               " --left-structure I --right-structure J"
               " [--dump-pair-mismatches N]"
               " [--tolerance F]\n"
               "   or: validate_multi_leaf_open_state_one_electron"
               " --synthetic-case three_component_demo"
               " [--dump-pair-mismatches N]"
               " [--tolerance F]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  int argument_index = 1;
  if (std::string(argv[1]).rfind("--", 0) != 0) {
    options.input_path = argv[1];
    argument_index = 2;
  }
  if (((argc - argument_index) % 2) != 0) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  for (; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--synthetic-case") {
      options.synthetic_case = argument_value;
      continue;
    }
    if (argument_name == "--left-structure") {
      options.left_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right-structure") {
      options.right_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--dump-pair-mismatches") {
      options.dump_pair_mismatches = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (!options.synthetic_case.empty() && !options.input_path.empty()) {
    throw std::invalid_argument("input file and --synthetic-case are mutually exclusive");
  }
  if (options.synthetic_case.empty()) {
    if (options.input_path.empty()) {
      throw std::invalid_argument("an input file or --synthetic-case is required");
    }
    if (options.left_structure < 0 || options.right_structure < 0) {
      throw std::invalid_argument(
          "--left-structure and --right-structure are required");
    }
    if (options.left_structure <= options.right_structure) {
      throw std::invalid_argument(
          "--left-structure must be strictly greater than --right-structure");
    }
  } else if (options.synthetic_case != "three_component_demo" &&
             options.synthetic_case != "three_component_block_local_demo") {
    throw std::invalid_argument("unsupported --synthetic-case value");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  if (options.dump_pair_mismatches < 0) {
    throw std::invalid_argument("--dump-pair-mismatches must be >= 0");
  }
  return options;
}

std::uint64_t mask_limit(int n_bits) {
  if (n_bits < 0 || n_bits >= 63) {
    throw std::invalid_argument("mask width is out of range for this validator");
  }
  return (n_bits == 0) ? 1ULL : (1ULL << n_bits);
}

int popcount(std::uint64_t mask) {
  int count = 0;
  while (mask != 0U) {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

double parity_sign(int parity) {
  return ((parity & 1) == 0) ? 1.0 : -1.0;
}

int canonicalization_parity(const std::vector<int>& occupied_orbitals) {
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

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint64_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes) {
  if (selected_masks.size() != leaf_occ_sizes.size()) {
    throw std::invalid_argument("selected_masks and leaf_occ_sizes must have the same length");
  }

  std::vector<int> block_order;
  block_order.reserve(
      xmvb::to_size(n_root_occ) +
      xmvb::to_size(
          std::accumulate(leaf_occ_sizes.begin(), leaf_occ_sizes.end(), 0)));

  std::uint64_t used_mask = 0U;
  int next_leaf_label = n_root_occ;
  for (std::size_t leaf_index = 0; leaf_index < selected_masks.size(); ++leaf_index) {
    const std::uint64_t mask = selected_masks[leaf_index];
    used_mask |= mask;
    for (int root_position = 0; root_position < n_root_occ; ++root_position) {
      if ((mask & (1ULL << root_position)) != 0U) {
        block_order.push_back(root_position);
      }
    }
    for (int leaf_position = 0; leaf_position < leaf_occ_sizes[leaf_index]; ++leaf_position) {
      block_order.push_back(next_leaf_label++);
    }
  }

  const std::uint64_t full_mask =
      (n_root_occ == 0) ? 0ULL : (mask_limit(n_root_occ) - 1ULL);
  const std::uint64_t remainder_mask = full_mask ^ used_mask;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((remainder_mask & (1ULL << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

std::vector<int> concatenate_occ(
    const std::vector<int>& first,
    const std::vector<int>& second) {
  std::vector<int> result;
  result.reserve(first.size() + second.size());
  result.insert(result.end(), first.begin(), first.end());
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

std::vector<int> select_occ_by_mask(
    const std::vector<int>& occupied_orbitals,
    std::uint64_t mask) {
  std::vector<int> selected;
  selected.reserve(xmvb::to_size(popcount(mask)));
  for (int index = 0; index < static_cast<int>(occupied_orbitals.size()); ++index) {
    if ((mask & (1ULL << index)) != 0U) {
      selected.push_back(occupied_orbitals[xmvb::to_size(index)]);
    }
  }
  return selected;
}

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

double determinant_of_dense_matrix(
    const Matrix& matrix,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
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
  return overlap_resolver.resolve(flatten_column_major_matrix(matrix), matrix.rows())
      .overlap_determinant;
}

Matrix build_support_submatrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& active_matrix_storage,
    int n_active_orbitals) {
  const int support_size = static_cast<int>(support_orbitals.size());
  Matrix support_matrix(support_size, support_size);
  for (int row = 0; row < support_size; ++row) {
    for (int column = 0; column < support_size; ++column) {
      support_matrix(row, column) =
          active_matrix_storage[xmvb::to_size(
                                    support_orbitals[xmvb::to_size(column)]) *
                                    xmvb::to_size(n_active_orbitals) +
                                support_orbitals[xmvb::to_size(row)]];
    }
  }
  return support_matrix;
}

double one_electron_element(
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    int left_orbital,
    int right_orbital) {
  return one_electron_storage[xmvb::to_size(left_orbital) *
                                  xmvb::to_size(n_orbitals) +
                              xmvb::to_size(right_orbital)];
}

Matrix build_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals) {
  Matrix overlap_block(
      static_cast<int>(right_occ.size()),
      static_cast<int>(left_occ.size()));
  for (int column = 0; column < static_cast<int>(left_occ.size()); ++column) {
    const int left_orbital = left_occ[xmvb::to_size(column)];
    for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
      const int right_orbital = right_occ[xmvb::to_size(row)];
      overlap_block(row, column) =
          overlap_storage[xmvb::to_size(left_orbital) *
                              xmvb::to_size(n_orbitals) +
                          xmvb::to_size(right_orbital)];
    }
  }
  return overlap_block;
}

ExactSpinValue evaluate_exact_spin_value(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector) {
  if (left_occ.size() != right_occ.size()) {
    throw std::invalid_argument("exact spin value requires balanced occupied lists");
  }
  if (work_collector != nullptr) {
    work_collector->reference_spin_determinants.insert({left_occ, right_occ});
  }

  ExactSpinValue result;
  if (left_occ.empty()) {
    result.overlap = 1.0;
    return result;
  }

  const Matrix overlap_block =
      build_overlap_block(left_occ, right_occ, overlap_storage, n_orbitals);
  const auto overlap_result =
      overlap_resolver.resolve(flatten_column_major_matrix(overlap_block), overlap_block.rows());
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);

  result.overlap = overlap_result.overlap_determinant;
  for (int column = 0; column < static_cast<int>(left_occ.size()); ++column) {
    const int left_orbital = left_occ[xmvb::to_size(column)];
    for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
      const int right_orbital = right_occ[xmvb::to_size(row)];
      const double cofactor_value = cofactor_1st(row, column);
      if (std::abs(cofactor_value) <= 1.0e-15) {
        continue;
      }
      result.one_electron +=
          one_electron_element(one_electron_storage, n_orbitals, left_orbital, right_orbital) *
          cofactor_value;
      result.one_electron_pair_weights[{right_orbital, left_orbital}] +=
          cofactor_value;
    }
  }
  return result;
}

PartialSpinPayload build_partial_spin_payload(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int selected_root_rows,
    int selected_root_cols,
    bool zero_selected_root_block,
    PartialSide side,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector) {
  PartialSpinPayload payload;
  payload.n_rows = static_cast<int>(right_occ.size());
  payload.n_cols = static_cast<int>(left_occ.size());

  Matrix overlap_block = build_overlap_block(left_occ, right_occ, overlap_storage, n_orbitals);
  if (zero_selected_root_block && selected_root_rows > 0 && selected_root_cols > 0) {
    overlap_block.topLeftCorner(selected_root_rows, selected_root_cols).setZero();
  }
  const int internal_row_begin = zero_selected_root_block ? selected_root_rows : 0;
  const int internal_col_begin = zero_selected_root_block ? selected_root_cols : 0;
  std::vector<int> left_root_flags(xmvb::to_size(payload.n_cols), 0);
  std::vector<int> right_root_flags(xmvb::to_size(payload.n_rows), 0);
  if (zero_selected_root_block) {
    for (int column = 0; column < selected_root_cols; ++column) {
      left_root_flags[xmvb::to_size(column)] = 1;
    }
    for (int row = 0; row < selected_root_rows; ++row) {
      right_root_flags[xmvb::to_size(row)] = 1;
    }
  }

  if (payload.n_rows == payload.n_cols) {
    if (payload.n_rows == 0) {
      payload.closed_overlap = 1.0;
      return payload;
    }
    if (work_collector != nullptr) {
      work_collector->open_state_spin_work.insert({
          left_occ,
          right_occ,
          left_root_flags,
          right_root_flags,
      });
    }
    const auto overlap_result =
        overlap_resolver.resolve(flatten_column_major_matrix(overlap_block), overlap_block.rows());
    payload.closed_overlap = overlap_result.overlap_determinant;
    const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);
    for (int column = 0; column < payload.n_cols; ++column) {
      if (column < internal_col_begin) {
        continue;
      }
      for (int row = 0; row < payload.n_rows; ++row) {
        if (row < internal_row_begin) {
          continue;
        }
        const double value = cofactor_1st(row, column);
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload.cofactor_entries.push_back({
            .row_local_index = row,
            .col_local_index = column,
            .row_orbital_label = right_occ[xmvb::to_size(row)],
            .col_orbital_label = left_occ[xmvb::to_size(column)],
            .value = value,
        });
      }
    }
    return payload;
  }

  if (payload.n_rows == payload.n_cols + 1) {
    payload.row_open_entries.reserve(xmvb::to_size(payload.n_rows));
    for (int row = 0; row < payload.n_rows; ++row) {
      if (row < internal_row_begin) {
        continue;
      }
      Matrix minor(payload.n_cols, payload.n_cols);
      for (int minor_column = 0; minor_column < payload.n_cols; ++minor_column) {
        int source_row = 0;
        for (int source_index = 0; source_index < payload.n_rows; ++source_index) {
          if (source_index == row) {
            continue;
          }
          minor(source_row, minor_column) = overlap_block(source_index, minor_column);
          ++source_row;
        }
      }
      if (work_collector != nullptr) {
        std::vector<int> minor_right_occ;
        std::vector<int> minor_right_root_flags;
        minor_right_occ.reserve(xmvb::to_size(payload.n_cols));
        minor_right_root_flags.reserve(xmvb::to_size(payload.n_cols));
        for (int source_index = 0; source_index < payload.n_rows; ++source_index) {
          if (source_index == row) {
            continue;
          }
          minor_right_occ.push_back(right_occ[xmvb::to_size(source_index)]);
          minor_right_root_flags.push_back(
              right_root_flags[xmvb::to_size(source_index)]);
        }
        work_collector->open_state_spin_work.insert({
            left_occ,
            std::move(minor_right_occ),
            left_root_flags,
            std::move(minor_right_root_flags),
        });
      }
      const double determinant = determinant_of_dense_matrix(minor, overlap_resolver);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      const double sign =
          (side == PartialSide::LeftFrontier)
              ? parity_sign(row + payload.n_cols)
              : parity_sign(row);
      payload.row_open_entries.push_back({
          .local_index = row,
          .orbital_label = right_occ[xmvb::to_size(row)],
          .value = sign * determinant,
      });
    }
    return payload;
  }

  if (payload.n_cols == payload.n_rows + 1) {
    payload.col_open_entries.reserve(xmvb::to_size(payload.n_cols));
    for (int column = 0; column < payload.n_cols; ++column) {
      if (column < internal_col_begin) {
        continue;
      }
      Matrix minor(payload.n_rows, payload.n_rows);
      int minor_column = 0;
      for (int source_column = 0; source_column < payload.n_cols; ++source_column) {
        if (source_column == column) {
          continue;
        }
        for (int row = 0; row < payload.n_rows; ++row) {
          minor(row, minor_column) = overlap_block(row, source_column);
        }
        ++minor_column;
      }
      if (work_collector != nullptr) {
        std::vector<int> minor_left_occ;
        std::vector<int> minor_left_root_flags;
        minor_left_occ.reserve(xmvb::to_size(payload.n_rows));
        minor_left_root_flags.reserve(xmvb::to_size(payload.n_rows));
        for (int source_column = 0; source_column < payload.n_cols; ++source_column) {
          if (source_column == column) {
            continue;
          }
          minor_left_occ.push_back(left_occ[xmvb::to_size(source_column)]);
          minor_left_root_flags.push_back(
              left_root_flags[xmvb::to_size(source_column)]);
        }
        work_collector->open_state_spin_work.insert({
            std::move(minor_left_occ),
            right_occ,
            std::move(minor_left_root_flags),
            right_root_flags,
        });
      }
      const double determinant = determinant_of_dense_matrix(minor, overlap_resolver);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      const double sign =
          (side == PartialSide::LeftFrontier)
              ? parity_sign(payload.n_rows + column)
              : parity_sign(column);
      payload.col_open_entries.push_back({
          .local_index = column,
          .orbital_label = left_occ[xmvb::to_size(column)],
          .value = sign * determinant,
      });
    }
  }

  return payload;
}

bool is_payload_nonzero(const PartialSpinPayload& payload) {
  return std::abs(payload.closed_overlap) > 1.0e-15 ||
      !payload.row_open_entries.empty() ||
      !payload.col_open_entries.empty() ||
      !payload.cofactor_entries.empty();
}

bool is_square_payload(const PartialSpinPayload& payload) {
  return payload.n_rows == payload.n_cols;
}

std::vector<SpinMaskPayload> build_frontier_spin_mask_payloads(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector) {
  std::vector<SpinMaskPayload> entries;
  const std::uint64_t row_limit = mask_limit(static_cast<int>(right_root_occ.size()));
  const std::uint64_t col_limit = mask_limit(static_cast<int>(left_root_occ.size()));
  for (std::uint64_t row_mask = 0; row_mask < row_limit; ++row_mask) {
    const auto selected_root_rows = select_occ_by_mask(right_root_occ, row_mask);
    for (std::uint64_t col_mask = 0; col_mask < col_limit; ++col_mask) {
      const auto selected_root_cols = select_occ_by_mask(left_root_occ, col_mask);
      const auto frontier_left_occ =
          concatenate_occ(selected_root_cols, left_leaf_occ);
      const auto frontier_right_occ =
          concatenate_occ(selected_root_rows, right_leaf_occ);
      PartialSpinPayload payload = build_partial_spin_payload(
          frontier_left_occ,
          frontier_right_occ,
          overlap_storage,
          n_orbitals,
          static_cast<int>(selected_root_rows.size()),
          static_cast<int>(selected_root_cols.size()),
          true,
          PartialSide::LeftFrontier,
          overlap_resolver,
          work_collector);
      if (!is_payload_nonzero(payload)) {
        continue;
      }
      entries.push_back({row_mask, col_mask, std::move(payload)});
    }
  }
  return entries;
}

void cleanup_scalar(double* value) {
  if (std::abs(*value) <= 1.0e-15) {
    *value = 0.0;
  }
}

template <typename Key>
void cleanup_map(std::map<Key, double>* values) {
  for (auto iterator = values->begin(); iterator != values->end();) {
    if (std::abs(iterator->second) <= 1.0e-15) {
      iterator = values->erase(iterator);
    } else {
      ++iterator;
    }
  }
}

void cleanup_payload(StructurePayload* payload) {
  cleanup_scalar(&payload->overlap);
  cleanup_map(&payload->alpha_row_open);
  cleanup_map(&payload->alpha_col_open);
  cleanup_map(&payload->beta_row_open);
  cleanup_map(&payload->beta_col_open);
  cleanup_map(&payload->alpha_cofactor);
  cleanup_map(&payload->beta_cofactor);
}

bool is_payload_nonzero(const StructurePayload& payload) {
  return std::abs(payload.overlap) > 1.0e-15 ||
      !payload.alpha_row_open.empty() ||
      !payload.alpha_col_open.empty() ||
      !payload.beta_row_open.empty() ||
      !payload.beta_col_open.empty() ||
      !payload.alpha_cofactor.empty() ||
      !payload.beta_cofactor.empty();
}

double parity_sign_from_count(int count) {
  return parity_sign(count & 1);
}

template <typename Key>
void add_scaled_map(
    const std::map<Key, double>& source,
    double scale,
    std::map<Key, double>* destination) {
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  for (const auto& [key, value] : source) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    (*destination)[key] += scale * value;
  }
}

void add_scaled_payload(
    const StructurePayload& source,
    double scale,
    StructurePayload* destination) {
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  if (!is_payload_nonzero(*destination)) {
    destination->alpha_row_count = source.alpha_row_count;
    destination->alpha_col_count = source.alpha_col_count;
    destination->beta_row_count = source.beta_row_count;
    destination->beta_col_count = source.beta_col_count;
  }
  destination->overlap += scale * source.overlap;
  add_scaled_map(source.alpha_row_open, scale, &destination->alpha_row_open);
  add_scaled_map(source.alpha_col_open, scale, &destination->alpha_col_open);
  add_scaled_map(source.beta_row_open, scale, &destination->beta_row_open);
  add_scaled_map(source.beta_col_open, scale, &destination->beta_col_open);
  add_scaled_map(source.alpha_cofactor, scale, &destination->alpha_cofactor);
  add_scaled_map(source.beta_cofactor, scale, &destination->beta_cofactor);
}

void add_row_col_pair_weights(
    const std::map<int, double>& row_map,
    const std::map<int, double>& col_map,
    double scale,
    std::map<CofactorKey, double>* destination) {
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  for (const auto& [row_orbital, row_value] : row_map) {
    if (std::abs(row_value) <= 1.0e-15) {
      continue;
    }
    for (const auto& [col_orbital, col_value] : col_map) {
      if (std::abs(col_value) <= 1.0e-15) {
        continue;
      }
      (*destination)[{row_orbital, col_orbital}] +=
          scale * row_value * col_value;
    }
  }
}

void add_one_electron_merge_pair_weights(
    const StructurePayload& frontier,
    const StructurePayload& root,
    double scale,
    std::map<CofactorKey, double>* destination) {
  add_scaled_map(frontier.alpha_cofactor, scale * root.overlap, destination);
  add_scaled_map(root.alpha_cofactor, scale * frontier.overlap, destination);
  add_row_col_pair_weights(
      frontier.alpha_row_open,
      root.alpha_col_open,
      scale,
      destination);
  add_row_col_pair_weights(
      root.alpha_row_open,
      frontier.alpha_col_open,
      scale,
      destination);

  add_scaled_map(frontier.beta_cofactor, scale * root.overlap, destination);
  add_scaled_map(root.beta_cofactor, scale * frontier.overlap, destination);
  add_row_col_pair_weights(
      frontier.beta_row_open,
      root.beta_col_open,
      scale,
      destination);
  add_row_col_pair_weights(
      root.beta_row_open,
      frontier.beta_col_open,
      scale,
      destination);
}

StructurePayload structure_payload_from_spin_partials(
    const PartialSpinPayload& alpha,
    const PartialSpinPayload& beta) {
  StructurePayload payload;
  payload.alpha_row_count = alpha.n_rows;
  payload.alpha_col_count = alpha.n_cols;
  payload.beta_row_count = beta.n_rows;
  payload.beta_col_count = beta.n_cols;
  const bool alpha_square = is_square_payload(alpha);
  const bool beta_square = is_square_payload(beta);

  if (alpha_square && beta_square) {
    payload.overlap = alpha.closed_overlap * beta.closed_overlap;
  }
  if (beta_square) {
    for (const auto& entry : alpha.row_open_entries) {
      payload.alpha_row_open[entry.orbital_label] +=
          entry.value * beta.closed_overlap;
    }
    for (const auto& entry : alpha.col_open_entries) {
      payload.alpha_col_open[entry.orbital_label] +=
          entry.value * beta.closed_overlap;
    }
    for (const auto& entry : alpha.cofactor_entries) {
      payload.alpha_cofactor[{entry.row_orbital_label, entry.col_orbital_label}] +=
          entry.value * beta.closed_overlap;
    }
  }
  if (alpha_square) {
    for (const auto& entry : beta.row_open_entries) {
      payload.beta_row_open[entry.orbital_label] +=
          entry.value * alpha.closed_overlap;
    }
    for (const auto& entry : beta.col_open_entries) {
      payload.beta_col_open[entry.orbital_label] +=
          entry.value * alpha.closed_overlap;
    }
    for (const auto& entry : beta.cofactor_entries) {
      payload.beta_cofactor[{entry.row_orbital_label, entry.col_orbital_label}] +=
          entry.value * alpha.closed_overlap;
    }
  }
  cleanup_payload(&payload);
  return payload;
}

StructurePayload merge_structure_payloads(
    const StructurePayload& left,
    const StructurePayload& right) {
  StructurePayload merged;
  merged.overlap = left.overlap * right.overlap;
  merged.alpha_row_count = left.alpha_row_count + right.alpha_row_count;
  merged.alpha_col_count = left.alpha_col_count + right.alpha_col_count;
  merged.beta_row_count = left.beta_row_count + right.beta_row_count;
  merged.beta_col_count = left.beta_col_count + right.beta_col_count;

  add_scaled_map(
      left.alpha_row_open,
      parity_sign_from_count(right.alpha_col_count) * right.overlap,
      &merged.alpha_row_open);
  add_scaled_map(
      right.alpha_row_open,
      parity_sign_from_count(left.alpha_row_count + left.alpha_col_count) * left.overlap,
      &merged.alpha_row_open);
  add_scaled_map(
      left.alpha_col_open,
      parity_sign_from_count(right.alpha_row_count) * right.overlap,
      &merged.alpha_col_open);
  add_scaled_map(
      right.alpha_col_open,
      parity_sign_from_count(left.alpha_row_count + left.alpha_col_count) * left.overlap,
      &merged.alpha_col_open);
  add_scaled_map(
      left.beta_row_open,
      parity_sign_from_count(right.beta_col_count) * right.overlap,
      &merged.beta_row_open);
  add_scaled_map(
      right.beta_row_open,
      parity_sign_from_count(left.beta_row_count + left.beta_col_count) * left.overlap,
      &merged.beta_row_open);
  add_scaled_map(
      left.beta_col_open,
      parity_sign_from_count(right.beta_row_count) * right.overlap,
      &merged.beta_col_open);
  add_scaled_map(
      right.beta_col_open,
      parity_sign_from_count(left.beta_row_count + left.beta_col_count) * left.overlap,
      &merged.beta_col_open);
  add_scaled_map(left.alpha_cofactor, right.overlap, &merged.alpha_cofactor);
  add_scaled_map(
      right.alpha_cofactor,
      parity_sign_from_count(left.alpha_row_count + left.alpha_col_count) * left.overlap,
      &merged.alpha_cofactor);
  add_scaled_map(left.beta_cofactor, right.overlap, &merged.beta_cofactor);
  add_scaled_map(
      right.beta_cofactor,
      parity_sign_from_count(left.beta_row_count + left.beta_col_count) * left.overlap,
      &merged.beta_cofactor);

  // Once two frontier blocks are merged into a larger frontier, a row-open
  // state on one side and a column-open state on the other side become a
  // resolved cofactor contribution of the combined block. This is the
  // multi-leaf `Q <- QZ + ZQ + RC + CR` update that is absent in the one-leaf
  // recurrence.
  for (const auto& [row_orbital, row_value] : left.alpha_row_open) {
    for (const auto& [col_orbital, col_value] : right.alpha_col_open) {
      merged.alpha_cofactor[{row_orbital, col_orbital}] +=
          parity_sign_from_count(
              left.alpha_row_count + left.alpha_col_count + right.alpha_col_count) *
          row_value * col_value;
    }
  }
  for (const auto& [row_orbital, row_value] : right.alpha_row_open) {
    for (const auto& [col_orbital, col_value] : left.alpha_col_open) {
      merged.alpha_cofactor[{row_orbital, col_orbital}] +=
          parity_sign_from_count(
              left.alpha_row_count + left.alpha_col_count + right.alpha_row_count) *
          row_value * col_value;
    }
  }
  for (const auto& [row_orbital, row_value] : left.beta_row_open) {
    for (const auto& [col_orbital, col_value] : right.beta_col_open) {
      merged.beta_cofactor[{row_orbital, col_orbital}] +=
          parity_sign_from_count(
              left.beta_row_count + left.beta_col_count + right.beta_col_count) *
          row_value * col_value;
    }
  }
  for (const auto& [row_orbital, row_value] : right.beta_row_open) {
    for (const auto& [col_orbital, col_value] : left.beta_col_open) {
      merged.beta_cofactor[{row_orbital, col_orbital}] +=
          parity_sign_from_count(
              left.beta_row_count + left.beta_col_count + right.beta_row_count) *
          row_value * col_value;
    }
  }

  cleanup_payload(&merged);
  return merged;
}

double contract_cofactor_map(
    const std::map<CofactorKey, double>& cofactor_map,
    const std::vector<double>& one_electron_storage,
    int n_orbitals) {
  double total = 0.0;
  for (const auto& [key, value] : cofactor_map) {
    total +=
        one_electron_element(one_electron_storage, n_orbitals, key.second, key.first) * value;
  }
  return total;
}

double contract_row_col_maps(
    const std::map<int, double>& row_map,
    const std::map<int, double>& col_map,
    const std::vector<double>& one_electron_storage,
    int n_orbitals) {
  double total = 0.0;
  for (const auto& [row_orbital, row_value] : row_map) {
    for (const auto& [col_orbital, col_value] : col_map) {
      total +=
          one_electron_element(one_electron_storage, n_orbitals, col_orbital, row_orbital) *
          row_value * col_value;
    }
  }
  return total;
}

double contract_one_electron_merge(
    const StructurePayload& frontier,
    const StructurePayload& root,
    const std::vector<double>& one_electron_storage,
    int n_orbitals) {
  double total = 0.0;
  total += contract_cofactor_map(frontier.alpha_cofactor, one_electron_storage, n_orbitals) *
      root.overlap;
  total += frontier.overlap *
      contract_cofactor_map(root.alpha_cofactor, one_electron_storage, n_orbitals);
  total += contract_row_col_maps(
      frontier.alpha_row_open,
      root.alpha_col_open,
      one_electron_storage,
      n_orbitals);
  total += contract_row_col_maps(
      root.alpha_row_open,
      frontier.alpha_col_open,
      one_electron_storage,
      n_orbitals);

  total += contract_cofactor_map(frontier.beta_cofactor, one_electron_storage, n_orbitals) *
      root.overlap;
  total += frontier.overlap *
      contract_cofactor_map(root.beta_cofactor, one_electron_storage, n_orbitals);
  total += contract_row_col_maps(
      frontier.beta_row_open,
      root.beta_col_open,
      one_electron_storage,
      n_orbitals);
  total += contract_row_col_maps(
      root.beta_row_open,
      frontier.beta_col_open,
      one_electron_storage,
      n_orbitals);
  return total;
}

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
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

std::vector<xmvb::vb::UnionGraphComponent> sort_components_by_min_vertex(
    std::vector<xmvb::vb::UnionGraphComponent> components) {
  std::sort(
      components.begin(),
      components.end(),
      [](const auto& left, const auto& right) {
        const int left_min =
            *std::min_element(left.local_vertices.begin(), left.local_vertices.end());
        const int right_min =
            *std::min_element(right.local_vertices.begin(), right.local_vertices.end());
        return left_min < right_min;
      });
  return components;
}

ComponentData build_local_component_data(
    const xmvb::vb::UnionGraphComponent& union_component) {
  ComponentData component;
  component.left_pairs = union_component.left_pairs;
  component.right_pairs = union_component.right_pairs;
  component.left_orientation_terms =
      enumerate_orientation_terms(component.left_pairs);
  component.right_orientation_terms =
      enumerate_orientation_terms(component.right_pairs);
  return component;
}

std::vector<int> build_component_ordered_support_orbitals(
    const std::vector<int>& support_orbitals,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    int root_node,
    const std::vector<int>& leaf_nodes) {
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

void record_reference_determinant_pair(
    const LegacyTerm& left_term,
    const LegacyTerm& right_term,
    WorkCollector* work_collector) {
  if (work_collector == nullptr) {
    return;
  }
  work_collector->reference_determinant_pairs.insert(
      {left_term.alpha_occ,
       left_term.beta_occ,
       right_term.alpha_occ,
       right_term.beta_occ});
}

StructureExactValue compute_exact_structure_value(
    const std::vector<LegacyTerm>& left_terms,
    const std::vector<LegacyTerm>& right_terms,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector) {
  StructureExactValue result;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      record_reference_determinant_pair(left_term, right_term, work_collector);
      const ExactSpinValue alpha = evaluate_exact_spin_value(
          left_term.alpha_occ,
          right_term.alpha_occ,
          overlap_storage,
          one_electron_storage,
          n_orbitals,
          overlap_resolver,
          work_collector);
      const ExactSpinValue beta = evaluate_exact_spin_value(
          left_term.beta_occ,
          right_term.beta_occ,
          overlap_storage,
          one_electron_storage,
          n_orbitals,
          overlap_resolver,
          work_collector);
      const double coefficient = left_term.coefficient * right_term.coefficient;
      result.overlap += coefficient * alpha.overlap * beta.overlap;
      result.one_electron += coefficient *
          (alpha.one_electron * beta.overlap +
           beta.one_electron * alpha.overlap);
      add_scaled_map(
          alpha.one_electron_pair_weights,
          coefficient * beta.overlap,
          &result.one_electron_pair_weights);
      add_scaled_map(
          beta.one_electron_pair_weights,
          coefficient * alpha.overlap,
          &result.one_electron_pair_weights);
    }
  }
  cleanup_map(&result.one_electron_pair_weights);
  return result;
}

std::vector<StructureLeafMessage> build_leaf_messages(
    int leaf_index,
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector,
    StateCollector* state_collector) {
  std::map<
      std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>,
      StructurePayload>
      aggregated_messages;

  for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
    for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
      const auto alpha_payloads = build_frontier_spin_mask_payloads(
          left_root_term.alpha_occ,
          left_leaf_term.alpha_occ,
          right_root_term.alpha_occ,
          right_leaf_term.alpha_occ,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          work_collector);
      const auto beta_payloads = build_frontier_spin_mask_payloads(
          left_root_term.beta_occ,
          left_leaf_term.beta_occ,
          right_root_term.beta_occ,
          right_leaf_term.beta_occ,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          work_collector);
      const double leaf_coefficient =
          left_leaf_term.coefficient * right_leaf_term.coefficient;
      for (const auto& alpha_entry : alpha_payloads) {
        for (const auto& beta_entry : beta_payloads) {
          const auto key = std::make_tuple(
              alpha_entry.row_mask,
              alpha_entry.col_mask,
              beta_entry.row_mask,
              beta_entry.col_mask);
          const StructurePayload term_payload =
              structure_payload_from_spin_partials(
                  alpha_entry.payload,
                  beta_entry.payload);
          if (!is_payload_nonzero(term_payload)) {
            continue;
          }
          add_scaled_payload(
              term_payload,
              leaf_coefficient,
              &aggregated_messages[key]);
        }
      }
    }
  }

  std::vector<StructureLeafMessage> messages;
  messages.reserve(aggregated_messages.size());
  for (auto& [key, payload] : aggregated_messages) {
    cleanup_payload(&payload);
    if (!is_payload_nonzero(payload)) {
      continue;
    }
    StructureLeafMessage message;
    message.alpha_row_mask = std::get<0>(key);
    message.alpha_col_mask = std::get<1>(key);
    message.beta_row_mask = std::get<2>(key);
    message.beta_col_mask = std::get<3>(key);
    message.payload = std::move(payload);
    messages.push_back(std::move(message));
    if (state_collector != nullptr) {
      state_collector->leaf_message_states.insert({
          .leaf_index = leaf_index,
          .left_root_alpha_occ = left_root_term.alpha_occ,
          .left_root_beta_occ = left_root_term.beta_occ,
          .right_root_alpha_occ = right_root_term.alpha_occ,
          .right_root_beta_occ = right_root_term.beta_occ,
          .alpha_row_mask = std::get<0>(key),
          .alpha_col_mask = std::get<1>(key),
          .beta_row_mask = std::get<2>(key),
          .beta_col_mask = std::get<3>(key),
      });
    }
  }
  return messages;
}

StructurePayload build_root_payload(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint64_t used_alpha_row_mask,
    std::uint64_t used_alpha_col_mask,
    std::uint64_t used_beta_row_mask,
    std::uint64_t used_beta_col_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector) {
  const std::uint64_t alpha_row_full_mask =
      (left_root_term.alpha_occ.empty() && right_root_term.alpha_occ.empty())
          ? 0ULL
          : (mask_limit(static_cast<int>(right_root_term.alpha_occ.size())) - 1ULL);
  const std::uint64_t alpha_col_full_mask =
      left_root_term.alpha_occ.empty()
          ? 0ULL
          : (mask_limit(static_cast<int>(left_root_term.alpha_occ.size())) - 1ULL);
  const std::uint64_t beta_row_full_mask =
      right_root_term.beta_occ.empty()
          ? 0ULL
          : (mask_limit(static_cast<int>(right_root_term.beta_occ.size())) - 1ULL);
  const std::uint64_t beta_col_full_mask =
      left_root_term.beta_occ.empty()
          ? 0ULL
          : (mask_limit(static_cast<int>(left_root_term.beta_occ.size())) - 1ULL);

  const auto alpha_root_rows =
      select_occ_by_mask(right_root_term.alpha_occ, alpha_row_full_mask ^ used_alpha_row_mask);
  const auto alpha_root_cols =
      select_occ_by_mask(left_root_term.alpha_occ, alpha_col_full_mask ^ used_alpha_col_mask);
  const auto beta_root_rows =
      select_occ_by_mask(right_root_term.beta_occ, beta_row_full_mask ^ used_beta_row_mask);
  const auto beta_root_cols =
      select_occ_by_mask(left_root_term.beta_occ, beta_col_full_mask ^ used_beta_col_mask);

  const PartialSpinPayload alpha_payload = build_partial_spin_payload(
      alpha_root_cols,
      alpha_root_rows,
      overlap_storage,
      n_orbitals,
      0,
      0,
      false,
      PartialSide::RightComplement,
      overlap_resolver,
      work_collector);
  const PartialSpinPayload beta_payload = build_partial_spin_payload(
      beta_root_cols,
      beta_root_rows,
      overlap_storage,
      n_orbitals,
      0,
      0,
      false,
      PartialSide::RightComplement,
      overlap_resolver,
      work_collector);
  return structure_payload_from_spin_partials(alpha_payload, beta_payload);
}

ValidationResult validate_multi_leaf_pair(
    int root_node,
    const std::vector<ComponentData>& ordered_components,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const std::vector<LegacyTerm>& exact_left_terms,
    const std::vector<LegacyTerm>& exact_right_terms,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector,
    StateCollector* state_collector) {
  ValidationResult result;
  result.node_count = static_cast<int>(ordered_components.size());
  result.root_node = root_node;
  result.n_leaves = static_cast<int>(ordered_components.size()) - 1;
  result.support_size = n_orbitals;

  const StructureExactValue exact = compute_exact_structure_value(
      exact_left_terms,
      exact_right_terms,
      overlap_storage,
      one_electron_storage,
      n_orbitals,
      overlap_resolver,
      work_collector);
  result.exact_overlap = exact.overlap;
  result.exact_one_electron = exact.one_electron;
  result.exact_one_electron_pair_weights = exact.one_electron_pair_weights;

  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }
  if (ordered_components.size() < 3) {
    throw std::invalid_argument("validate_multi_leaf_pair requires a multi-leaf star");
  }

  const auto& root_component = ordered_components.front();
  std::vector<int> left_leaf_alpha_sizes;
  std::vector<int> left_leaf_beta_sizes;
  std::vector<int> right_leaf_alpha_sizes;
  std::vector<int> right_leaf_beta_sizes;
  left_leaf_alpha_sizes.reserve(xmvb::to_size(result.n_leaves));
  left_leaf_beta_sizes.reserve(xmvb::to_size(result.n_leaves));
  right_leaf_alpha_sizes.reserve(xmvb::to_size(result.n_leaves));
  right_leaf_beta_sizes.reserve(xmvb::to_size(result.n_leaves));
  for (int leaf_index = 0; leaf_index < result.n_leaves; ++leaf_index) {
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
      const double root_coefficient =
          left_root_term.coefficient * right_root_term.coefficient;
      std::vector<std::vector<StructureLeafMessage>> leaf_messages(
          xmvb::to_size(result.n_leaves));
      for (int leaf_index = 0; leaf_index < result.n_leaves; ++leaf_index) {
        leaf_messages[xmvb::to_size(leaf_index)] = build_leaf_messages(
            leaf_index,
            left_root_term,
            right_root_term,
            ordered_components[xmvb::to_size(leaf_index + 1)],
            overlap_storage,
            n_orbitals,
            overlap_resolver,
            work_collector,
            state_collector);
      }

      std::map<
          std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>,
          StructurePayload>
          root_payload_cache;
      std::vector<std::uint64_t> selected_alpha_row_masks(
          xmvb::to_size(result.n_leaves),
          0ULL);
      std::vector<std::uint64_t> selected_alpha_col_masks(
          xmvb::to_size(result.n_leaves),
          0ULL);
      std::vector<std::uint64_t> selected_beta_row_masks(
          xmvb::to_size(result.n_leaves),
          0ULL);
      std::vector<std::uint64_t> selected_beta_col_masks(
          xmvb::to_size(result.n_leaves),
          0ULL);

      const auto accumulate_frontier =
          [&](const auto& self,
              int leaf_index,
              std::uint64_t used_alpha_row_mask,
              std::uint64_t used_alpha_col_mask,
              std::uint64_t used_beta_row_mask,
              std::uint64_t used_beta_col_mask,
              const StructurePayload& frontier_payload) -> void {
            if (!is_payload_nonzero(frontier_payload)) {
              return;
            }
            if (state_collector != nullptr) {
              state_collector->merge_states.insert({
                  .leaf_index = leaf_index,
                  .left_root_alpha_occ = left_root_term.alpha_occ,
                  .left_root_beta_occ = left_root_term.beta_occ,
                  .right_root_alpha_occ = right_root_term.alpha_occ,
                  .right_root_beta_occ = right_root_term.beta_occ,
                  .used_alpha_row_mask = used_alpha_row_mask,
                  .used_alpha_col_mask = used_alpha_col_mask,
                  .used_beta_row_mask = used_beta_row_mask,
                  .used_beta_col_mask = used_beta_col_mask,
              });
            }

            if (leaf_index == result.n_leaves) {
              const auto cache_key = std::make_tuple(
                  used_alpha_row_mask,
                  used_alpha_col_mask,
                  used_beta_row_mask,
                  used_beta_col_mask);
              auto cache_iterator = root_payload_cache.find(cache_key);
              if (cache_iterator == root_payload_cache.end()) {
                cache_iterator = root_payload_cache.emplace(
                    cache_key,
                    build_root_payload(
                        left_root_term,
                        right_root_term,
                        used_alpha_row_mask,
                        used_alpha_col_mask,
                        used_beta_row_mask,
                        used_beta_col_mask,
                        overlap_storage,
                        n_orbitals,
                        overlap_resolver,
                        work_collector))
                                     .first;
              }
              const StructurePayload& root_payload = cache_iterator->second;
              if (!is_payload_nonzero(root_payload)) {
                return;
              }

              int parity = 0;
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.alpha_occ.size()),
                  selected_alpha_row_masks,
                  right_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.alpha_occ.size()),
                  selected_alpha_col_masks,
                  left_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.beta_occ.size()),
                  selected_beta_row_masks,
                  right_leaf_beta_sizes);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.beta_occ.size()),
                  selected_beta_col_masks,
                  left_leaf_beta_sizes);
              const double signed_coefficient =
                  root_coefficient * parity_sign(parity);

              result.reconstructed_overlap +=
                  signed_coefficient * frontier_payload.overlap * root_payload.overlap;
              result.reconstructed_one_electron +=
                  signed_coefficient *
                  contract_one_electron_merge(
                      frontier_payload,
                      root_payload,
                      one_electron_storage,
                      n_orbitals);
              add_one_electron_merge_pair_weights(
                  frontier_payload,
                  root_payload,
                  signed_coefficient,
                  &result.reconstructed_one_electron_pair_weights);
              return;
            }

            for (const auto& message :
                 leaf_messages[xmvb::to_size(leaf_index)]) {
              if ((used_alpha_row_mask & message.alpha_row_mask) != 0U ||
                  (used_alpha_col_mask & message.alpha_col_mask) != 0U ||
                  (used_beta_row_mask & message.beta_row_mask) != 0U ||
                  (used_beta_col_mask & message.beta_col_mask) != 0U) {
                continue;
              }

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] =
                  message.alpha_row_mask;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] =
                  message.alpha_col_mask;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] =
                  message.beta_row_mask;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] =
                  message.beta_col_mask;

              const StructurePayload merged_payload =
                  merge_structure_payloads(frontier_payload, message.payload);
              self(
                  self,
                  leaf_index + 1,
                  used_alpha_row_mask | message.alpha_row_mask,
                  used_alpha_col_mask | message.alpha_col_mask,
                  used_beta_row_mask | message.beta_row_mask,
                  used_beta_col_mask | message.beta_col_mask,
                  merged_payload);

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] = 0ULL;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] = 0ULL;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] = 0ULL;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] = 0ULL;
            }
          };

      StructurePayload identity_payload;
      identity_payload.overlap = 1.0;
      accumulate_frontier(
          accumulate_frontier,
          0,
          0ULL,
          0ULL,
          0ULL,
          0ULL,
          identity_payload);
    }
  }

  result.overlap_abs_error =
      std::abs(result.exact_overlap - result.reconstructed_overlap);
  result.one_electron_abs_error =
      std::abs(result.exact_one_electron - result.reconstructed_one_electron);
  cleanup_map(&result.exact_one_electron_pair_weights);
  cleanup_map(&result.reconstructed_one_electron_pair_weights);
  result.unique_reference_determinant_pair_count =
      work_collector->reference_determinant_pairs.size();
  result.unique_reference_spin_determinant_count =
      work_collector->reference_spin_determinants.size();
  result.unique_open_state_spin_work_count =
      work_collector->open_state_spin_work.size();
  result.unique_leaf_message_state_count =
      (state_collector == nullptr) ? 0 : state_collector->leaf_message_states.size();
  result.unique_merge_state_count =
      (state_collector == nullptr) ? 0 : state_collector->merge_states.size();
  result.combined_open_state_work_count =
      result.unique_open_state_spin_work_count +
      result.unique_leaf_message_state_count +
      result.unique_merge_state_count;
  return result;
}

SyntheticCase build_three_component_base_case(const std::string& name) {
  SyntheticCase synthetic;
  synthetic.name = name;
  synthetic.root_node = 0;
  synthetic.support_size = 6;

  const std::vector<OrbitalPair> root_pairs = {{0, 1}};
  const std::vector<OrbitalPair> leaf1_pairs = {{2, 3}};
  const std::vector<OrbitalPair> leaf2_pairs = {{4, 5}};

  ComponentData root_component;
  root_component.left_pairs = root_pairs;
  root_component.right_pairs = root_pairs;
  root_component.left_orientation_terms = enumerate_orientation_terms(root_pairs);
  root_component.right_orientation_terms = enumerate_orientation_terms(root_pairs);

  ComponentData leaf1_component;
  leaf1_component.left_pairs = leaf1_pairs;
  leaf1_component.right_pairs = leaf1_pairs;
  leaf1_component.left_orientation_terms = enumerate_orientation_terms(leaf1_pairs);
  leaf1_component.right_orientation_terms = enumerate_orientation_terms(leaf1_pairs);

  ComponentData leaf2_component;
  leaf2_component.left_pairs = leaf2_pairs;
  leaf2_component.right_pairs = leaf2_pairs;
  leaf2_component.left_orientation_terms = enumerate_orientation_terms(leaf2_pairs);
  leaf2_component.right_orientation_terms = enumerate_orientation_terms(leaf2_pairs);

  synthetic.ordered_components = {
      root_component,
      leaf1_component,
      leaf2_component,
  };

  std::vector<OrbitalPair> full_pairs;
  full_pairs.insert(full_pairs.end(), root_pairs.begin(), root_pairs.end());
  full_pairs.insert(full_pairs.end(), leaf1_pairs.begin(), leaf1_pairs.end());
  full_pairs.insert(full_pairs.end(), leaf2_pairs.begin(), leaf2_pairs.end());
  synthetic.exact_left_terms = xmvb::vb::enumerate_legacy_determinant_terms(full_pairs);
  synthetic.exact_right_terms = xmvb::vb::enumerate_legacy_determinant_terms(full_pairs);

  Matrix overlap(6, 6);
  overlap <<
      1.00, 0.15, 0.30, 0.05, 0.25, 0.04,
      0.15, 1.00, 0.02, 0.28, 0.03, 0.26,
      0.30, 0.02, 1.00, 0.12, 0.00, 0.00,
      0.05, 0.28, 0.12, 1.00, 0.00, 0.00,
      0.25, 0.03, 0.00, 0.00, 1.00, 0.11,
      0.04, 0.26, 0.00, 0.00, 0.11, 1.00;
  synthetic.overlap_storage = flatten_column_major_matrix(overlap);
  return synthetic;
}

SyntheticCase build_three_component_demo_case() {
  SyntheticCase synthetic = build_three_component_base_case("three_component_demo");
  Matrix one_electron(6, 6);
  one_electron <<
      -1.10, -0.12, -0.21, -0.05, -0.18, -0.04,
      -0.12, -0.95, -0.03, -0.24, -0.02, -0.20,
      -0.21, -0.03, -0.72, -0.08,  0.00,  0.00,
      -0.05, -0.24, -0.08, -0.68,  0.00,  0.00,
      -0.18, -0.02,  0.00,  0.00, -0.70, -0.07,
      -0.04, -0.20,  0.00,  0.00, -0.07, -0.66;
  synthetic.one_electron_storage = flatten_column_major_matrix(one_electron);
  return synthetic;
}

SyntheticCase build_three_component_block_local_demo_case() {
  SyntheticCase synthetic =
      build_three_component_base_case("three_component_block_local_demo");
  Matrix one_electron(6, 6);
  one_electron <<
      -1.10, -0.12,  0.00,  0.00,  0.00,  0.00,
      -0.12, -0.95,  0.00,  0.00,  0.00,  0.00,
       0.00,  0.00, -0.72, -0.08,  0.00,  0.00,
       0.00,  0.00, -0.08, -0.68,  0.00,  0.00,
       0.00,  0.00,  0.00,  0.00, -0.70, -0.07,
       0.00,  0.00,  0.00,  0.00, -0.07, -0.66;
  synthetic.one_electron_storage = flatten_column_major_matrix(one_electron);
  return synthetic;
}

std::vector<int> build_orbital_component_index(
    const std::vector<ComponentData>& ordered_components,
    int n_orbitals) {
  std::vector<int> component_index_by_orbital(
      xmvb::to_size(n_orbitals),
      -1);
  for (int component_index = 0;
       component_index < static_cast<int>(ordered_components.size());
       ++component_index) {
    const auto& component =
        ordered_components[xmvb::to_size(component_index)];
    for (const auto& pair : component.left_pairs) {
      component_index_by_orbital[xmvb::to_size(pair.first)] =
          component_index;
      component_index_by_orbital[xmvb::to_size(pair.second)] =
          component_index;
    }
    for (const auto& pair : component.right_pairs) {
      component_index_by_orbital[xmvb::to_size(pair.first)] =
          component_index;
      component_index_by_orbital[xmvb::to_size(pair.second)] =
          component_index;
    }
  }
  return component_index_by_orbital;
}

std::vector<PairMismatchRecord> build_pair_mismatch_records(
    const ValidationResult& result,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const std::vector<int>& component_index_by_orbital) {
  std::set<CofactorKey> all_keys;
  for (const auto& [key, _] : result.exact_one_electron_pair_weights) {
    all_keys.insert(key);
  }
  for (const auto& [key, _] : result.reconstructed_one_electron_pair_weights) {
    all_keys.insert(key);
  }

  std::vector<PairMismatchRecord> records;
  records.reserve(all_keys.size());
  for (const auto& key : all_keys) {
    const int right_orbital = key.first;
    const int left_orbital = key.second;
    const auto exact_iterator = result.exact_one_electron_pair_weights.find(key);
    const auto reconstructed_iterator =
        result.reconstructed_one_electron_pair_weights.find(key);
    const double exact_weight =
        (exact_iterator == result.exact_one_electron_pair_weights.end())
            ? 0.0
            : exact_iterator->second;
    const double reconstructed_weight =
        (reconstructed_iterator == result.reconstructed_one_electron_pair_weights.end())
            ? 0.0
            : reconstructed_iterator->second;
    const double h_element =
        one_electron_element(one_electron_storage, n_orbitals, left_orbital, right_orbital);
    records.push_back({
        .left_orbital = left_orbital,
        .right_orbital = right_orbital,
        .left_component = component_index_by_orbital[xmvb::to_size(left_orbital)],
        .right_component =
            component_index_by_orbital[xmvb::to_size(right_orbital)],
        .one_electron_element = h_element,
        .exact_weight = exact_weight,
        .reconstructed_weight = reconstructed_weight,
        .exact_contribution = h_element * exact_weight,
        .reconstructed_contribution = h_element * reconstructed_weight,
        .contribution_error = h_element * (exact_weight - reconstructed_weight),
    });
  }

  std::sort(
      records.begin(),
      records.end(),
      [](const PairMismatchRecord& left, const PairMismatchRecord& right) {
        return std::abs(left.contribution_error) > std::abs(right.contribution_error);
      });
  return records;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    WorkCollector work_collector;
    StateCollector state_collector;
    ValidationResult result;
    double prepare_active_space_seconds = 0.0;
    std::string case_label;
    std::vector<double> debug_one_electron_storage;
    int debug_n_orbitals = 0;
    std::vector<ComponentData> debug_components;

    if (!options.synthetic_case.empty()) {
      const SyntheticCase synthetic =
          (options.synthetic_case == "three_component_demo")
              ? build_three_component_demo_case()
              : build_three_component_block_local_demo_case();
      case_label = synthetic.name;
      debug_one_electron_storage = synthetic.one_electron_storage;
      debug_n_orbitals = synthetic.support_size;
      debug_components = synthetic.ordered_components;
      result = validate_multi_leaf_pair(
          synthetic.root_node,
          synthetic.ordered_components,
          synthetic.overlap_storage,
          synthetic.one_electron_storage,
          synthetic.support_size,
          synthetic.exact_left_terms,
          synthetic.exact_right_terms,
          overlap_resolver,
          &work_collector,
          &state_collector);
    } else {
      const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
      const auto& raw_structure_data = load_result.raw_structure_data;

      if (raw_structure_data.spin_multiplicity != 1) {
        throw std::runtime_error(
            "validate_multi_leaf_open_state_one_electron currently supports only singlet closed-shell structures");
      }
      if (options.left_structure >= raw_structure_data.n_structures ||
          options.right_structure >= raw_structure_data.n_structures) {
        throw std::out_of_range("structure index is out of range for the selected input");
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
      const auto& active_overlap_storage =
          prepared_active_space.orbital_result.active_orbital_overlap_matrix;
      const auto& active_one_electron_storage =
          prepared_active_space.active_space_one_electron_result.h1e_act;
      const int n_active_orbitals =
          load_result.input.orbital_preparation_input.n_active_orbitals;
      prepare_active_space_seconds =
          timed_active_space.timings.orbital_preparation_wall_time_seconds +
          timed_active_space.timings.ao_effective_one_electron_wall_time_seconds +
          timed_active_space.timings.active_one_electron_wall_time_seconds +
          timed_active_space.timings.active_two_electron_wall_time_seconds;

      const auto left_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, options.left_structure);
      const auto right_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, options.right_structure);
      const auto support_orbitals =
          xmvb::vb::build_support_orbitals(left_pairs, right_pairs);
      const auto support_index =
          xmvb::vb::build_support_index(support_orbitals);
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
          cross_blocks);
      const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
          metric_graph,
          union_components);
      if (metric_summary.connected_component_count != 1) {
        throw std::runtime_error("selected pair does not form one connected metric component");
      }

      int root_node = -1;
      if (!is_connected_star_graph(metric_graph, &root_node)) {
        throw std::runtime_error("selected pair is not a connected star graph");
      }
      if (metric_graph.node_count < 3) {
        throw std::runtime_error("selected pair is not a multi-leaf star");
      }

      const auto ordered_support_orbitals = build_component_ordered_support_orbitals(
          support_orbitals,
          union_components,
          root_node,
          metric_graph.adjacency[xmvb::to_size(root_node)]);
      const auto ordered_support_index =
          xmvb::vb::build_support_index(ordered_support_orbitals);
      const auto ordered_left_pairs =
          xmvb::vb::remap_pairs_to_support(left_pairs, ordered_support_index);
      const auto ordered_right_pairs =
          xmvb::vb::remap_pairs_to_support(right_pairs, ordered_support_index);
      const auto ordered_overlap_storage = flatten_column_major_matrix(
          xmvb::vb::build_support_overlap_matrix(
              ordered_support_orbitals,
              active_overlap_storage,
              n_active_orbitals));
      const auto ordered_one_electron_storage = flatten_column_major_matrix(
          build_support_submatrix(
              ordered_support_orbitals,
              active_one_electron_storage,
              n_active_orbitals));
      const auto ordered_union_components = sort_components_by_min_vertex(
          xmvb::vb::build_union_graph_components(
              ordered_left_pairs,
              ordered_right_pairs,
              ordered_support_orbitals));
      if (ordered_union_components.size() != union_components.size()) {
        throw std::runtime_error(
            "component-ordered support changed the number of union components");
      }

      std::vector<ComponentData> ordered_components;
      ordered_components.reserve(ordered_union_components.size());
      for (const auto& component : ordered_union_components) {
        ordered_components.push_back(build_local_component_data(component));
      }
      debug_one_electron_storage = ordered_one_electron_storage;
      debug_n_orbitals = static_cast<int>(ordered_support_orbitals.size());
      debug_components = ordered_components;
      const auto exact_left_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(ordered_left_pairs);
      const auto exact_right_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(ordered_right_pairs);
      case_label = options.input_path;
      result = validate_multi_leaf_pair(
          root_node,
          ordered_components,
          ordered_overlap_storage,
          ordered_one_electron_storage,
          static_cast<int>(ordered_support_orbitals.size()),
          exact_left_terms,
          exact_right_terms,
          overlap_resolver,
          &work_collector,
          &state_collector);
    }

    const std::int64_t spin_kernel_absolute_reduction =
        static_cast<std::int64_t>(result.unique_reference_spin_determinant_count) -
        static_cast<std::int64_t>(result.unique_open_state_spin_work_count);
    const std::int64_t combined_work_absolute_reduction =
        static_cast<std::int64_t>(result.unique_reference_determinant_pair_count) -
        static_cast<std::int64_t>(result.combined_open_state_work_count);

    std::cout << std::setprecision(16);
    std::cout << "case = " << case_label << '\n';
    if (options.synthetic_case.empty()) {
      std::cout << "input_file = " << options.input_path << '\n';
      std::cout << "left_structure = " << options.left_structure << '\n';
      std::cout << "right_structure = " << options.right_structure << '\n';
    } else {
      std::cout << "synthetic_case = " << options.synthetic_case << '\n';
    }
    std::cout << "node_count = " << result.node_count << '\n';
    std::cout << "root_node = " << result.root_node << '\n';
    std::cout << "n_leaves = " << result.n_leaves << '\n';
    std::cout << "support_size = " << result.support_size << '\n';
    std::cout << "exact_overlap = " << result.exact_overlap << '\n';
    std::cout << "reconstructed_overlap = " << result.reconstructed_overlap << '\n';
    std::cout << "overlap_abs_error = " << result.overlap_abs_error << '\n';
    std::cout << "exact_one_electron = " << result.exact_one_electron << '\n';
    std::cout << "reconstructed_one_electron = " << result.reconstructed_one_electron << '\n';
    std::cout << "one_electron_abs_error = " << result.one_electron_abs_error << '\n';
    std::cout << "exact_match = " << std::boolalpha
              << (result.overlap_abs_error <= options.tolerance &&
                  result.one_electron_abs_error <= options.tolerance)
              << std::noboolalpha << '\n';
    std::cout << "unique_reference_determinant_pair_count = "
              << result.unique_reference_determinant_pair_count << '\n';
    std::cout << "unique_reference_spin_determinant_count = "
              << result.unique_reference_spin_determinant_count << '\n';
    std::cout << "unique_open_state_spin_work_count = "
              << result.unique_open_state_spin_work_count << '\n';
    std::cout << "unique_leaf_message_state_count = "
              << result.unique_leaf_message_state_count << '\n';
    std::cout << "unique_merge_state_count = "
              << result.unique_merge_state_count << '\n';
    std::cout << "combined_open_state_work_count = "
              << result.combined_open_state_work_count << '\n';
    std::cout << "spin_kernel_absolute_reduction = "
              << spin_kernel_absolute_reduction << '\n';
    std::cout << "spin_kernel_compression_ratio = "
              << (result.unique_open_state_spin_work_count == 0
                      ? 0.0
                      : static_cast<double>(result.unique_reference_spin_determinant_count) /
                            static_cast<double>(result.unique_open_state_spin_work_count))
              << '\n';
    std::cout << "spin_kernel_reduction_ratio = "
              << (result.unique_reference_spin_determinant_count == 0
                      ? 0.0
                      : 1.0 -
                            static_cast<double>(result.unique_open_state_spin_work_count) /
                                static_cast<double>(
                                    result.unique_reference_spin_determinant_count))
              << '\n';
    std::cout << "combined_work_absolute_reduction_vs_reference_pair = "
              << combined_work_absolute_reduction << '\n';
    std::cout << "combined_work_over_reference_pair_ratio = "
              << (result.unique_reference_determinant_pair_count == 0
                      ? 0.0
                      : static_cast<double>(result.combined_open_state_work_count) /
                            static_cast<double>(
                                result.unique_reference_determinant_pair_count))
              << '\n';
    std::cout << "combined_work_reduction_ratio_vs_reference_pair = "
              << (result.unique_reference_determinant_pair_count == 0
                      ? 0.0
                      : 1.0 -
                            static_cast<double>(result.combined_open_state_work_count) /
                                static_cast<double>(
                                    result.unique_reference_determinant_pair_count))
              << '\n';
    std::cout << "prepare_active_space_wall_time_seconds = "
              << prepare_active_space_seconds
              << '\n';
    if (options.dump_pair_mismatches > 0) {
      const auto component_index_by_orbital =
          build_orbital_component_index(debug_components, debug_n_orbitals);
      const auto mismatch_records = build_pair_mismatch_records(
          result,
          debug_one_electron_storage,
          debug_n_orbitals,
          component_index_by_orbital);
      std::cout << "pair_mismatch_count = " << mismatch_records.size() << '\n';
      for (int record_index = 0;
           record_index < options.dump_pair_mismatches &&
           record_index < static_cast<int>(mismatch_records.size());
           ++record_index) {
        const auto& record = mismatch_records[xmvb::to_size(record_index)];
        std::cout << "pair_mismatch[" << record_index << "]"
                  << " left_orbital=" << record.left_orbital
                  << " right_orbital=" << record.right_orbital
                  << " left_component=" << record.left_component
                  << " right_component=" << record.right_component
                  << " h=" << record.one_electron_element
                  << " exact_weight=" << record.exact_weight
                  << " reconstructed_weight=" << record.reconstructed_weight
                  << " exact_contribution=" << record.exact_contribution
                  << " reconstructed_contribution=" << record.reconstructed_contribution
                  << " contribution_error=" << record.contribution_error
                  << '\n';
      }
    }

    return (result.overlap_abs_error <= options.tolerance &&
            result.one_electron_abs_error <= options.tolerance)
        ? 0
        : 1;
  } catch (const std::exception& exception) {
    std::cerr << "Error! " << exception.what() << '\n';
    return 1;
  }
}
