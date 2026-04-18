#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using OrbitalPair = xmvb::vb::OrbitalPair;
using LegacyTerm = xmvb::vb::LegacyStructureDeterminantTerm;
using CanonicalDeterminantKey = std::pair<std::vector<int>, std::vector<int>>;

enum class SpinChannel {
  Alpha,
  Beta,
};

enum class PartialSide {
  LeftFrontier,
  RightComplement,
};

struct Options {
  std::string input_path;
  int filter_left_structure = -1;
  int filter_right_structure = -1;
  int max_pairs = 0;
  int top_examples = 8;
  double tolerance = 1.0e-10;
};

struct ExactSpinValue {
  double overlap = 0.0;
  double one_electron = 0.0;
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

struct OpenStateSpinValue {
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct PairCache {
  std::vector<OrbitalPair> active_pairs;
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
  std::set<CanonicalDeterminantKey> reference_spin_determinants;
  std::set<SpinMinorKey> open_state_spin_work;
};

struct Example {
  int left_structure = 0;
  int right_structure = 0;
  int root_choice = 0;
  SpinChannel spin = SpinChannel::Alpha;
  int left_root_term = 0;
  int right_root_term = 0;
  int left_leaf_term = 0;
  int right_leaf_term = 0;
  double exact_overlap = 0.0;
  double reconstructed_overlap = 0.0;
  double overlap_abs_error = 0.0;
  double exact_one_electron = 0.0;
  double reconstructed_one_electron = 0.0;
  double one_electron_abs_error = 0.0;
};

void print_usage() {
  std::cerr << "usage: validate_one_leaf_open_state_one_electron <input.xmi>"
               " [--left-structure I --right-structure J]"
               " [--max-pairs N]"
               " [--top-examples N]"
               " [--tolerance F]\n";
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
    if (argument_name == "--top-examples") {
      options.top_examples = std::stoi(argument_value);
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
  if (options.top_examples <= 0) {
    throw std::invalid_argument("--top-examples must be positive");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  return options;
}

std::vector<std::pair<int, int>> build_pair_list(
    int structure_count,
    int filter_left_structure,
    int filter_right_structure,
    int max_pairs) {
  if (filter_left_structure >= 0 && filter_right_structure >= 0) {
    if (filter_left_structure >= structure_count ||
        filter_right_structure >= structure_count ||
        filter_left_structure <= filter_right_structure) {
      throw std::invalid_argument("requested structure filter is out of range");
    }
    return {{filter_left_structure, filter_right_structure}};
  }

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

int one_leaf_block_parity(
    int n_root_occ,
    std::uint64_t selected_mask,
    int leaf_occ_size) {
  std::vector<int> block_order;
  block_order.reserve(xmvb::to_size(n_root_occ + leaf_occ_size));
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (1ULL << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  for (int leaf_position = 0; leaf_position < leaf_occ_size; ++leaf_position) {
    block_order.push_back(n_root_occ + leaf_position);
  }
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((selected_mask & (1ULL << root_position)) == 0U) {
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
      result.one_electron +=
          one_electron_element(one_electron_storage, n_orbitals, left_orbital, right_orbital) *
          cofactor_1st(row, column);
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

  // A balanced partial state carries the exact closed overlap plus the full
  // unresolved row/column cofactor kernel on the current block ordering.
  //
  // For the zeroed frontier block, however, deleted row/column labels that
  // sit on the selected-root interface are not yet resolved "inside" the
  // partial region. Those interface labels must stay in the transfer algebra
  // and be represented through complementary off-by-one masks instead of being
  // absorbed into the local `Q` payload. Otherwise the same global cofactor
  // contribution is counted once through `Q * Z` and again through `R * C`
  // or `C * R`.
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

  // A row-open partial state keeps one unresolved deleted row label. Each
  // payload entry is the exact square minor obtained after deleting that row
  // from the zeroed local overlap block.
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

  // A column-open partial state is the transpose-analogue: one unresolved
  // deleted column label, stored through the corresponding square minors.
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

OpenStateSpinValue evaluate_one_leaf_open_state_spin(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    WorkCollector* work_collector) {
  const int n_root_rows = static_cast<int>(right_root_occ.size());
  const int n_root_cols = static_cast<int>(left_root_occ.size());
  const int n_leaf_rows = static_cast<int>(right_leaf_occ.size());
  const int n_leaf_cols = static_cast<int>(left_leaf_occ.size());
  const std::uint64_t row_mask_limit = mask_limit(n_root_rows);
  const std::uint64_t col_mask_limit = mask_limit(n_root_cols);
  const std::uint64_t row_full_mask =
      (n_root_rows == 0) ? 0ULL : (row_mask_limit - 1ULL);
  const std::uint64_t col_full_mask =
      (n_root_cols == 0) ? 0ULL : (col_mask_limit - 1ULL);

  OpenStateSpinValue result;
  for (std::uint64_t row_mask = 0; row_mask < row_mask_limit; ++row_mask) {
    const auto selected_root_rows = select_occ_by_mask(right_root_occ, row_mask);
    const auto root_remainder_rows =
        select_occ_by_mask(right_root_occ, row_full_mask ^ row_mask);
    for (std::uint64_t col_mask = 0; col_mask < col_mask_limit; ++col_mask) {
      const auto selected_root_cols = select_occ_by_mask(left_root_occ, col_mask);
      const auto root_remainder_cols =
          select_occ_by_mask(left_root_occ, col_full_mask ^ col_mask);

      const auto frontier_left_occ =
          concatenate_occ(selected_root_cols, left_leaf_occ);
      const auto frontier_right_occ =
          concatenate_occ(selected_root_rows, right_leaf_occ);
      const PartialSpinPayload frontier = build_partial_spin_payload(
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
      const PartialSpinPayload root = build_partial_spin_payload(
          root_remainder_cols,
          root_remainder_rows,
          overlap_storage,
          n_orbitals,
          0,
          0,
          false,
          PartialSide::RightComplement,
          overlap_resolver,
          work_collector);

      const int row_parity =
          one_leaf_block_parity(n_root_rows, row_mask, n_leaf_rows);
      const int col_parity =
          one_leaf_block_parity(n_root_cols, col_mask, n_leaf_cols);
      const double mask_sign = parity_sign(row_parity ^ col_parity);

      // Closed/balanced masks reproduce the overlap recurrence and the
      // within-frontier/within-root cofactor sectors.
      if (frontier.n_rows == frontier.n_cols && root.n_rows == root.n_cols) {
        result.overlap += mask_sign * frontier.closed_overlap * root.closed_overlap;
        for (const auto& entry : frontier.cofactor_entries) {
          result.one_electron +=
              mask_sign *
              one_electron_element(
                  one_electron_storage,
                  n_orbitals,
                  entry.col_orbital_label,
                  entry.row_orbital_label) *
              entry.value * root.closed_overlap;
        }
        for (const auto& entry : root.cofactor_entries) {
          result.one_electron +=
              mask_sign *
              one_electron_element(
                  one_electron_storage,
                  n_orbitals,
                  entry.col_orbital_label,
                  entry.row_orbital_label) *
              frontier.closed_overlap * entry.value;
        }
      }

      // Cross-region one-electron terms require one unresolved deleted row on
      // the frontier side and one unresolved deleted column on the root side.
      if (frontier.n_rows == frontier.n_cols + 1 &&
          root.n_cols == root.n_rows + 1) {
        for (const auto& row_entry : frontier.row_open_entries) {
          for (const auto& col_entry : root.col_open_entries) {
            result.one_electron +=
                mask_sign *
                one_electron_element(
                    one_electron_storage,
                    n_orbitals,
                    col_entry.orbital_label,
                    row_entry.orbital_label) *
                row_entry.value * col_entry.value;
          }
        }
      }

      // The transpose sector carries the opposite cross-region pattern.
      if (frontier.n_cols == frontier.n_rows + 1 &&
          root.n_rows == root.n_cols + 1) {
        for (const auto& col_entry : frontier.col_open_entries) {
          for (const auto& row_entry : root.row_open_entries) {
            result.one_electron +=
                mask_sign *
                one_electron_element(
                    one_electron_storage,
                    n_orbitals,
                    col_entry.orbital_label,
                    row_entry.orbital_label) *
                col_entry.value * row_entry.value;
          }
        }
      }
    }
  }

  return result;
}

std::vector<int> build_component_ordered_support_orbitals(
    const std::vector<int>& support_orbitals,
    const std::vector<xmvb::vb::UnionGraphComponent>& components,
    int root_component_index) {
  if (components.size() != 2) {
    throw std::invalid_argument(
        "build_component_ordered_support_orbitals expects exactly two components");
  }
  const int leaf_component_index = 1 - root_component_index;
  std::vector<int> ordered_support_orbitals;
  ordered_support_orbitals.reserve(support_orbitals.size());
  const auto append_component = [&](int component_index) {
    std::vector<int> local_vertices =
        components[xmvb::to_size(component_index)].local_vertices;
    std::sort(local_vertices.begin(), local_vertices.end());
    for (const int local_vertex : local_vertices) {
      ordered_support_orbitals.push_back(
          support_orbitals[xmvb::to_size(local_vertex)]);
    }
  };
  append_component(root_component_index);
  append_component(leaf_component_index);
  return ordered_support_orbitals;
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

const std::vector<int>& term_occ(
    const LegacyTerm& term,
    SpinChannel spin) {
  return (spin == SpinChannel::Alpha) ? term.alpha_occ : term.beta_occ;
}

const char* spin_name(SpinChannel spin) {
  return (spin == SpinChannel::Alpha) ? "alpha" : "beta";
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
        if (left.one_electron_abs_error != right.one_electron_abs_error) {
          return left.one_electron_abs_error > right.one_electron_abs_error;
        }
        return left.overlap_abs_error > right.overlap_abs_error;
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
          "validate_one_leaf_open_state_one_electron currently supports only singlet closed-shell structures");
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

    std::vector<PairCache> structure_cache(
        xmvb::to_size(raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      structure_cache[xmvb::to_size(structure_index)].active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.filter_left_structure,
        options.filter_right_structure,
        options.max_pairs);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    WorkCollector work_collector;

    int total_pairs = 0;
    int one_leaf_pairs = 0;
    int tested_root_choices = 0;
    int tested_spin_cases = 0;
    int exact_match_count = 0;
    int mismatch_count = 0;
    double max_overlap_abs_error = 0.0;
    double max_one_electron_abs_error = 0.0;
    std::vector<Example> examples;

    for (const auto& [left_structure, right_structure] : pair_list) {
      ++total_pairs;
      const auto& left_pairs =
          structure_cache[xmvb::to_size(left_structure)].active_pairs;
      const auto& right_pairs =
          structure_cache[xmvb::to_size(right_structure)].active_pairs;

      const auto support_orbitals =
          xmvb::vb::build_support_orbitals(left_pairs, right_pairs);
      const auto support_index = xmvb::vb::build_support_index(support_orbitals);
      const auto left_pairs_local =
          xmvb::vb::remap_pairs_to_support(left_pairs, support_index);
      const auto right_pairs_local =
          xmvb::vb::remap_pairs_to_support(right_pairs, support_index);
      const auto union_components = xmvb::vb::build_union_graph_components(
          left_pairs_local,
          right_pairs_local,
          support_orbitals);
      if (union_components.size() != 2) {
        continue;
      }
      ++one_leaf_pairs;

      for (int root_choice = 0; root_choice < 2; ++root_choice) {
        ++tested_root_choices;
        const auto ordered_support_orbitals = build_component_ordered_support_orbitals(
            support_orbitals,
            union_components,
            root_choice);
        const auto ordered_support_index =
            xmvb::vb::build_support_index(ordered_support_orbitals);
        const auto ordered_left_pairs =
            xmvb::vb::remap_pairs_to_support(left_pairs, ordered_support_index);
        const auto ordered_right_pairs =
            xmvb::vb::remap_pairs_to_support(right_pairs, ordered_support_index);
        auto ordered_components = sort_components_by_min_vertex(
            xmvb::vb::build_union_graph_components(
                ordered_left_pairs,
                ordered_right_pairs,
                ordered_support_orbitals));
        if (ordered_components.size() != 2) {
          throw std::runtime_error(
              "component ordering changed the number of union components");
        }

        const auto support_overlap = flatten_column_major_matrix(
            xmvb::vb::build_support_overlap_matrix(
                ordered_support_orbitals,
                active_overlap_storage,
                n_active_orbitals));
        const auto support_one_electron = flatten_column_major_matrix(
            build_support_submatrix(
                ordered_support_orbitals,
                active_one_electron_storage,
                n_active_orbitals));
        const int support_size = static_cast<int>(ordered_support_orbitals.size());

        const auto left_root_terms =
            xmvb::vb::enumerate_legacy_determinant_terms(ordered_components[0].left_pairs);
        const auto right_root_terms =
            xmvb::vb::enumerate_legacy_determinant_terms(ordered_components[0].right_pairs);
        const auto left_leaf_terms =
            xmvb::vb::enumerate_legacy_determinant_terms(ordered_components[1].left_pairs);
        const auto right_leaf_terms =
            xmvb::vb::enumerate_legacy_determinant_terms(ordered_components[1].right_pairs);

        for (int left_root_term_index = 0;
             left_root_term_index < static_cast<int>(left_root_terms.size());
             ++left_root_term_index) {
          const auto& left_root_term =
              left_root_terms[xmvb::to_size(left_root_term_index)];
          for (int right_root_term_index = 0;
               right_root_term_index < static_cast<int>(right_root_terms.size());
               ++right_root_term_index) {
            const auto& right_root_term =
                right_root_terms[xmvb::to_size(right_root_term_index)];
            for (int left_leaf_term_index = 0;
                 left_leaf_term_index < static_cast<int>(left_leaf_terms.size());
                 ++left_leaf_term_index) {
              const auto& left_leaf_term =
                  left_leaf_terms[xmvb::to_size(left_leaf_term_index)];
              for (int right_leaf_term_index = 0;
                   right_leaf_term_index < static_cast<int>(right_leaf_terms.size());
                   ++right_leaf_term_index) {
                const auto& right_leaf_term =
                    right_leaf_terms[xmvb::to_size(right_leaf_term_index)];

                for (const SpinChannel spin : {SpinChannel::Alpha, SpinChannel::Beta}) {
                  ++tested_spin_cases;
                  const auto ordered_left_occ = concatenate_occ(
                      term_occ(left_root_term, spin),
                      term_occ(left_leaf_term, spin));
                  const auto ordered_right_occ = concatenate_occ(
                      term_occ(right_root_term, spin),
                      term_occ(right_leaf_term, spin));
                  const ExactSpinValue exact = evaluate_exact_spin_value(
                      ordered_left_occ,
                      ordered_right_occ,
                      support_overlap,
                      support_one_electron,
                      support_size,
                      overlap_resolver,
                      &work_collector);
                  const OpenStateSpinValue reconstructed =
                      evaluate_one_leaf_open_state_spin(
                          term_occ(left_root_term, spin),
                          term_occ(left_leaf_term, spin),
                          term_occ(right_root_term, spin),
                          term_occ(right_leaf_term, spin),
                          support_overlap,
                          support_one_electron,
                          support_size,
                          overlap_resolver,
                          &work_collector);

                  const double overlap_abs_error =
                      std::abs(exact.overlap - reconstructed.overlap);
                  const double one_electron_abs_error =
                      std::abs(exact.one_electron - reconstructed.one_electron);
                  max_overlap_abs_error =
                      std::max(max_overlap_abs_error, overlap_abs_error);
                  max_one_electron_abs_error =
                      std::max(max_one_electron_abs_error, one_electron_abs_error);

                  if (overlap_abs_error <= options.tolerance &&
                      one_electron_abs_error <= options.tolerance) {
                    ++exact_match_count;
                  } else {
                    ++mismatch_count;
                    maybe_record_example(
                        options,
                        Example{
                            .left_structure = left_structure,
                            .right_structure = right_structure,
                            .root_choice = root_choice,
                            .spin = spin,
                            .left_root_term = left_root_term_index,
                            .right_root_term = right_root_term_index,
                            .left_leaf_term = left_leaf_term_index,
                            .right_leaf_term = right_leaf_term_index,
                            .exact_overlap = exact.overlap,
                            .reconstructed_overlap = reconstructed.overlap,
                            .overlap_abs_error = overlap_abs_error,
                            .exact_one_electron = exact.one_electron,
                            .reconstructed_one_electron = reconstructed.one_electron,
                            .one_electron_abs_error = one_electron_abs_error,
                        },
                        &examples);
                  }
                }
              }
            }
          }
        }
      }
    }

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "total_structure_pairs = " << total_pairs << '\n';
    std::cout << "one_leaf_structure_pairs = " << one_leaf_pairs << '\n';
    std::cout << "tested_root_choices = " << tested_root_choices << '\n';
    std::cout << "tested_spin_cases = " << tested_spin_cases << '\n';
    std::cout << "exact_match_count = " << exact_match_count << '\n';
    std::cout << "mismatch_count = " << mismatch_count << '\n';
    std::cout << "max_overlap_abs_error = " << max_overlap_abs_error << '\n';
    std::cout << "max_one_electron_abs_error = " << max_one_electron_abs_error << '\n';
    std::cout << "unique_reference_spin_determinant_count = "
              << work_collector.reference_spin_determinants.size() << '\n';
    std::cout << "unique_open_state_spin_work_count = "
              << work_collector.open_state_spin_work.size() << '\n';
    const std::int64_t absolute_reduction =
        static_cast<std::int64_t>(work_collector.reference_spin_determinants.size()) -
        static_cast<std::int64_t>(work_collector.open_state_spin_work.size());
    std::cout << "absolute_reduction = " << absolute_reduction << '\n';
    std::cout << "compression_ratio = "
              << (work_collector.open_state_spin_work.empty()
                      ? 0.0
                      : static_cast<double>(
                            work_collector.reference_spin_determinants.size()) /
                            static_cast<double>(work_collector.open_state_spin_work.size()))
              << '\n';
    std::cout << "reduction_ratio = "
              << (work_collector.reference_spin_determinants.empty()
                      ? 0.0
                      : 1.0 -
                            static_cast<double>(work_collector.open_state_spin_work.size()) /
                                static_cast<double>(
                                    work_collector.reference_spin_determinants.size()))
              << '\n';
    std::cout << "prepare_active_space_wall_time_seconds = "
              << timed_active_space.timings.orbital_preparation_wall_time_seconds +
                     timed_active_space.timings.ao_effective_one_electron_wall_time_seconds +
                     timed_active_space.timings.active_one_electron_wall_time_seconds +
                     timed_active_space.timings.active_two_electron_wall_time_seconds
              << '\n';

    if (!examples.empty()) {
      std::cout << "top_examples\n";
      for (std::size_t example_index = 0;
           example_index < examples.size();
           ++example_index) {
        const auto& example = examples[example_index];
        std::cout << "example[" << example_index << "]"
                  << " left_structure=" << example.left_structure
                  << " right_structure=" << example.right_structure
                  << " root_choice=" << example.root_choice
                  << " spin=" << spin_name(example.spin)
                  << " left_root_term=" << example.left_root_term
                  << " right_root_term=" << example.right_root_term
                  << " left_leaf_term=" << example.left_leaf_term
                  << " right_leaf_term=" << example.right_leaf_term
                  << " exact_overlap=" << example.exact_overlap
                  << " reconstructed_overlap=" << example.reconstructed_overlap
                  << " overlap_abs_error=" << example.overlap_abs_error
                  << " exact_one_electron=" << example.exact_one_electron
                  << " reconstructed_one_electron=" << example.reconstructed_one_electron
                  << " one_electron_abs_error=" << example.one_electron_abs_error
                  << '\n';
      }
    }

    return (mismatch_count == 0) ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "Error! " << exception.what() << '\n';
    return 1;
  }
}
