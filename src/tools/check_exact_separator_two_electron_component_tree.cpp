#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/exact_separator/component_tree.hpp"
#include "vb/exact_separator/component_terms.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::Matrix;
using xmvb::vb::OrbitalPair;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedHamiltonianComponentTreeStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ComponentTreeOneElectronResult;
using xmvb::vb::exact_separator::ComponentTreeOppositeSpinResult;
using xmvb::vb::exact_separator::ComponentTreeSameSpinResult;
using xmvb::vb::exact_separator::ComponentTree;
using xmvb::vb::exact_separator::ComponentTreeOverlapResult;
using xmvb::vb::exact_separator::GlobalOrientationTerm;
using xmvb::vb::exact_separator::OrientationTerm;

struct SameSpinDeletionPatternBreakdown {
  double root00_child22 = 0.0;
  double root11_child11 = 0.0;
  double root01_child21 = 0.0;
  double root10_child12 = 0.0;
};

struct OverlapCheckResult {
  ComponentTreeOverlapResult exact;
  ComponentTreeOverlapResult boundary;
  double absolute_error = 0.0;
};

struct OneElectronCheckResult {
  ComponentTreeOneElectronResult exact;
  ComponentTreeOneElectronResult boundary;
  double overlap_absolute_error = 0.0;
  double one_electron_absolute_error = 0.0;
};

struct OppositeSpinCheckResult {
  ComponentTreeOppositeSpinResult exact;
  ComponentTreeOppositeSpinResult boundary;
  double overlap_absolute_error = 0.0;
  double one_electron_absolute_error = 0.0;
  double opposite_spin_absolute_error = 0.0;
};

struct SameSpinCheckResult {
  ComponentTreeSameSpinResult exact;
  ComponentTreeSameSpinResult boundary;
  double overlap_absolute_error = 0.0;
  double one_electron_absolute_error = 0.0;
  double same_spin_alpha_absolute_error = 0.0;
  double same_spin_beta_absolute_error = 0.0;
};

Matrix build_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size) {
  Matrix overlap_block(
      static_cast<int>(right_occ.size()),
      static_cast<int>(left_occ.size()));
  for (int col = 0; col < static_cast<int>(left_occ.size()); ++col) {
    const int left_orbital = left_occ[xmvb::to_size(col)];
    for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
      const int right_orbital = right_occ[xmvb::to_size(row)];
      overlap_block(row, col) =
          support_overlap_storage[xmvb::to_size(left_orbital) *
                                      xmvb::to_size(support_size) +
                                  xmvb::to_size(right_orbital)];
    }
  }
  return overlap_block;
}

double determinant_of_dense_matrix(
    const Matrix& matrix,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
  switch (matrix.rows()) {
    case 0:
      return 1.0;
    case 1:
      return matrix(0, 0);
    default:
      break;
  }
  return overlap_resolver.resolve_matrix(matrix).overlap_determinant;
}

SameSpinDeletionPatternBreakdown compute_chain_same_spin_alpha_breakdown(
    const ComponentTree& chain_tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  // This diagnostic decomposes the exact alpha-alpha matrix element of the
  // smallest non-star chain into four deletion-distribution patterns:
  // - root(0 rows, 0 cols) + child-subtree(2 rows, 2 cols),
  // - root(1 row, 1 col) + child-subtree(1 row, 1 col),
  // - root(0 rows, 1 col) + child-subtree(2 rows, 1 col),
  // - root(1 row, 0 col) + child-subtree(1 row, 2 cols).
  //
  // The current rooted-tree implementation carries only the first two through
  // its scalar deleted-sector payload. If the exact decomposition places the
  // missing weight into the mixed `(0,1)` / `(1,0)` root patterns, then the
  // remaining same-spin mismatch is unambiguously caused by the absence of
  // higher-open child payloads such as `(2,1)` and `(1,2)`.
  const std::vector<ComponentData> ordered_components = chain_tree.components;
  const std::vector<GlobalOrientationTerm> left_terms =
      xmvb::vb::exact_separator::build_global_orientation_terms(
          ordered_components,
          true);
  const std::vector<GlobalOrientationTerm> right_terms =
      xmvb::vb::exact_separator::build_global_orientation_terms(
          ordered_components,
          false);

  SameSpinDeletionPatternBreakdown breakdown;
  const DeterminantOverlapResolver overlap_resolver;
  std::vector<int> root_left_orbitals;
  std::vector<int> root_right_orbitals;
  for (const auto& pair : chain_tree.components.front().left_pairs) {
    root_left_orbitals.push_back(pair.first);
    root_left_orbitals.push_back(pair.second);
  }
  for (const auto& pair : chain_tree.components.front().right_pairs) {
    root_right_orbitals.push_back(pair.first);
    root_right_orbitals.push_back(pair.second);
  }

  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const double structure_coefficient =
          left_term.coefficient * right_term.coefficient;
      if (std::abs(structure_coefficient) <= 1.0e-15) {
        continue;
      }

      const Matrix alpha_overlap = build_overlap_block(
          left_term.alpha_occ,
          right_term.alpha_occ,
          support_overlap_storage,
          support_size);
      const Matrix beta_overlap = build_overlap_block(
          left_term.beta_occ,
          right_term.beta_occ,
          support_overlap_storage,
          support_size);
      const double beta_determinant =
          determinant_of_dense_matrix(beta_overlap, overlap_resolver);
      if (std::abs(beta_determinant) <= 1.0e-15) {
        continue;
      }

      const auto contains_left_root_alpha =
          [&](int left_orbital) {
            return std::find(
                       root_left_orbitals.begin(),
                       root_left_orbitals.end(),
                       left_orbital) !=
                root_left_orbitals.end();
          };
      const auto contains_right_root_alpha =
          [&](int right_orbital) {
            return std::find(
                       root_right_orbitals.begin(),
                       root_right_orbitals.end(),
                       right_orbital) !=
                root_right_orbitals.end();
          };

      const int alpha_dimension = alpha_overlap.rows();
      for (int row_first = 0; row_first + 1 < alpha_dimension; ++row_first) {
        for (int row_second = row_first + 1; row_second < alpha_dimension; ++row_second) {
          for (int col_first = 0; col_first + 1 < alpha_dimension; ++col_first) {
            for (int col_second = col_first + 1; col_second < alpha_dimension; ++col_second) {
              Matrix minor(alpha_dimension - 2, alpha_dimension - 2);
              int minor_row = 0;
              for (int row = 0; row < alpha_dimension; ++row) {
                if (row == row_first || row == row_second) {
                  continue;
                }
                int minor_col = 0;
                for (int col = 0; col < alpha_dimension; ++col) {
                  if (col == col_first || col == col_second) {
                    continue;
                  }
                  minor(minor_row, minor_col) = alpha_overlap(row, col);
                  ++minor_col;
                }
                ++minor_row;
              }
              const double second_cofactor =
                  xmvb::vb::exact_separator::parity_sign(
                      row_first + row_second + col_first + col_second) *
                  determinant_of_dense_matrix(minor, overlap_resolver);
              if (std::abs(second_cofactor) <= 1.0e-15) {
                continue;
              }

              const int row_root_count =
                  static_cast<int>(contains_right_root_alpha(
                      right_term.alpha_occ[xmvb::to_size(row_first)])) +
                  static_cast<int>(contains_right_root_alpha(
                      right_term.alpha_occ[xmvb::to_size(row_second)]));
              const int col_root_count =
                  static_cast<int>(contains_left_root_alpha(
                      left_term.alpha_occ[xmvb::to_size(col_first)])) +
                  static_cast<int>(contains_left_root_alpha(
                      left_term.alpha_occ[xmvb::to_size(col_second)]));
              const int direct_index =
                  TwoElectronIndexer::two_electron_storage_index(
                      right_term.alpha_occ[xmvb::to_size(row_first)],
                      left_term.alpha_occ[xmvb::to_size(col_first)],
                      right_term.alpha_occ[xmvb::to_size(row_second)],
                      left_term.alpha_occ[xmvb::to_size(col_second)]);
              const int exchange_index =
                  TwoElectronIndexer::two_electron_storage_index(
                      right_term.alpha_occ[xmvb::to_size(row_first)],
                      left_term.alpha_occ[xmvb::to_size(col_second)],
                      right_term.alpha_occ[xmvb::to_size(row_second)],
                      left_term.alpha_occ[xmvb::to_size(col_first)]);
              const double weighted_contribution =
                  structure_coefficient *
                  beta_determinant *
                  second_cofactor *
                  (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
                   packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]);

              if (row_root_count == 0 && col_root_count == 0) {
                breakdown.root00_child22 += weighted_contribution;
              } else if (row_root_count == 1 && col_root_count == 1) {
                breakdown.root11_child11 += weighted_contribution;
              } else if (row_root_count == 0 && col_root_count == 1) {
                breakdown.root01_child21 += weighted_contribution;
              } else if (row_root_count == 1 && col_root_count == 0) {
                breakdown.root10_child12 += weighted_contribution;
              } else {
                throw std::runtime_error("unexpected root deletion pattern in chain breakdown");
              }
            }
          }
        }
      }
    }
  }
  return breakdown;
}

CollapsedHamiltonianComponentTreeStats run_case(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  const CollapsedHamiltonianComponentTreeStats stats =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_hamiltonian(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          DeterminantOverlapResolver());

  std::cout << "case = " << label << '\n';
  std::cout << "exact_overlap = " << stats.exact_overlap << '\n';
  std::cout << "collapsed_overlap = " << stats.collapsed_overlap << '\n';
  std::cout << "exact_one_electron = " << stats.exact_one_electron << '\n';
  std::cout << "collapsed_one_electron = " << stats.collapsed_one_electron << '\n';
  std::cout << "exact_same_spin_alpha = "
            << stats.exact_same_spin_alpha_two_electron << '\n';
  std::cout << "collapsed_same_spin_alpha = "
            << stats.collapsed_same_spin_alpha_two_electron << '\n';
  std::cout << "exact_same_spin_beta = "
            << stats.exact_same_spin_beta_two_electron << '\n';
  std::cout << "collapsed_same_spin_beta = "
            << stats.collapsed_same_spin_beta_two_electron << '\n';
  std::cout << "exact_opposite_spin = "
            << stats.exact_opposite_spin_two_electron << '\n';
  std::cout << "collapsed_opposite_spin = "
            << stats.collapsed_opposite_spin_two_electron << '\n';
  std::cout << "exact_total = " << stats.exact_two_electron << '\n';
  std::cout << "collapsed_total = " << stats.collapsed_two_electron << '\n';
  std::cout << "exact_total_electronic = "
            << stats.exact_total_electronic_hamiltonian << '\n';
  std::cout << "collapsed_total_electronic = "
            << stats.collapsed_total_electronic_hamiltonian << '\n';
  std::cout << "overlap_absolute_error = " << stats.overlap_absolute_error << '\n';
  std::cout << "one_electron_absolute_error = "
            << stats.one_electron_absolute_error << '\n';
  std::cout << "same_spin_alpha_absolute_error = "
            << stats.same_spin_alpha_absolute_error << '\n';
  std::cout << "same_spin_beta_absolute_error = "
            << stats.same_spin_beta_absolute_error << '\n';
  std::cout << "opposite_spin_absolute_error = "
            << stats.opposite_spin_absolute_error << '\n';
  std::cout << "total_two_electron_absolute_error = "
            << stats.total_two_electron_absolute_error << '\n';
  std::cout << "total_electronic_absolute_error = "
            << stats.total_electronic_hamiltonian_absolute_error << '\n';
  std::cout << "subtree_message_state_count = "
            << stats.subtree_message_state_count << '\n';
  std::cout << "subtree_term_pair_count = "
            << stats.subtree_term_pair_count << '\n';
  std::cout << "subdeterminant_evaluations = "
            << stats.subdeterminant_evaluations << '\n';
  std::cout << "dp_transition_count = " << stats.dp_transition_count << '\n';
  std::cout << '\n';
  return stats;
}

OverlapCheckResult run_overlap_case(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    int support_size) {
  const DeterminantOverlapResolver overlap_resolver;
  OverlapCheckResult result;
  result.exact =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_overlap_exact(
          support_overlap_storage,
          support_size,
          tree,
          overlap_resolver);
  result.boundary =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_overlap_boundary_collapsed(
          support_overlap_storage,
          support_size,
          tree,
          overlap_resolver);
  result.absolute_error = std::abs(result.boundary.overlap - result.exact.overlap);

  std::cout << "overlap_case = " << label << '\n';
  std::cout << "overlap_exact = " << result.exact.overlap << '\n';
  std::cout << "overlap_boundary = " << result.boundary.overlap << '\n';
  std::cout << "overlap_abs_error = " << result.absolute_error << '\n';
  std::cout << "boundary_message_state_count = "
            << result.boundary.subtree_message_state_count << '\n';
  std::cout << "boundary_term_pair_count = "
            << result.boundary.subtree_term_pair_count << '\n';
  std::cout << "boundary_subdeterminant_evaluations = "
            << result.boundary.subdeterminant_evaluations << '\n';
  std::cout << "boundary_dp_transition_count = "
            << result.boundary.dp_transition_count << '\n';
  std::cout << '\n';
  return result;
}

OneElectronCheckResult run_one_electron_case(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  const DeterminantOverlapResolver overlap_resolver;
  OneElectronCheckResult result;
  result.exact =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_one_electron_exact(
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          tree,
          overlap_resolver);
  result.boundary =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_one_electron_boundary_collapsed(
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          tree,
          overlap_resolver);
  result.overlap_absolute_error =
      std::abs(result.boundary.overlap - result.exact.overlap);
  result.one_electron_absolute_error =
      std::abs(result.boundary.one_electron - result.exact.one_electron);

  std::cout << "one_electron_case = " << label << '\n';
  std::cout << "one_electron_overlap_exact = " << result.exact.overlap << '\n';
  std::cout << "one_electron_overlap_boundary = " << result.boundary.overlap << '\n';
  std::cout << "one_electron_overlap_abs_error = "
            << result.overlap_absolute_error << '\n';
  std::cout << "one_electron_exact = " << result.exact.one_electron << '\n';
  std::cout << "one_electron_boundary = " << result.boundary.one_electron << '\n';
  std::cout << "one_electron_abs_error = "
            << result.one_electron_absolute_error << '\n';
  std::cout << "boundary_message_state_count = "
            << result.boundary.subtree_message_state_count << '\n';
  std::cout << "boundary_term_pair_count = "
            << result.boundary.subtree_term_pair_count << '\n';
  std::cout << "boundary_subdeterminant_evaluations = "
            << result.boundary.subdeterminant_evaluations << '\n';
  std::cout << "boundary_dp_transition_count = "
            << result.boundary.dp_transition_count << '\n';
  std::cout << '\n';
  return result;
}

OppositeSpinCheckResult run_opposite_spin_case(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  const DeterminantOverlapResolver overlap_resolver;
  OppositeSpinCheckResult result;
  result.exact =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_opposite_spin_exact(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  result.boundary =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_opposite_spin_boundary_collapsed(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  result.overlap_absolute_error =
      std::abs(result.boundary.overlap - result.exact.overlap);
  result.one_electron_absolute_error =
      std::abs(result.boundary.one_electron - result.exact.one_electron);
  result.opposite_spin_absolute_error = std::abs(
      result.boundary.opposite_spin_two_electron -
      result.exact.opposite_spin_two_electron);

  std::cout << "opposite_spin_case = " << label << '\n';
  std::cout << "opposite_spin_overlap_exact = " << result.exact.overlap << '\n';
  std::cout << "opposite_spin_overlap_boundary = " << result.boundary.overlap << '\n';
  std::cout << "opposite_spin_overlap_abs_error = "
            << result.overlap_absolute_error << '\n';
  std::cout << "opposite_spin_one_electron_exact = "
            << result.exact.one_electron << '\n';
  std::cout << "opposite_spin_one_electron_boundary = "
            << result.boundary.one_electron << '\n';
  std::cout << "opposite_spin_one_electron_abs_error = "
            << result.one_electron_absolute_error << '\n';
  std::cout << "opposite_spin_exact = "
            << result.exact.opposite_spin_two_electron << '\n';
  std::cout << "opposite_spin_boundary = "
            << result.boundary.opposite_spin_two_electron << '\n';
  std::cout << "opposite_spin_abs_error = "
            << result.opposite_spin_absolute_error << '\n';
  std::cout << "boundary_message_state_count = "
            << result.boundary.subtree_message_state_count << '\n';
  std::cout << "boundary_term_pair_count = "
            << result.boundary.subtree_term_pair_count << '\n';
  std::cout << "boundary_subdeterminant_evaluations = "
            << result.boundary.subdeterminant_evaluations << '\n';
  std::cout << "boundary_dp_transition_count = "
            << result.boundary.dp_transition_count << '\n';
  std::cout << '\n';
  return result;
}

SameSpinCheckResult run_same_spin_case(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  const DeterminantOverlapResolver overlap_resolver;
  SameSpinCheckResult result;
  result.exact =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_same_spin_exact(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  result.boundary =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_same_spin_boundary_collapsed(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  result.overlap_absolute_error =
      std::abs(result.boundary.overlap - result.exact.overlap);
  result.one_electron_absolute_error =
      std::abs(result.boundary.one_electron - result.exact.one_electron);
  result.same_spin_alpha_absolute_error = std::abs(
      result.boundary.same_spin_alpha_two_electron -
      result.exact.same_spin_alpha_two_electron);
  result.same_spin_beta_absolute_error = std::abs(
      result.boundary.same_spin_beta_two_electron -
      result.exact.same_spin_beta_two_electron);

  std::cout << "same_spin_case = " << label << '\n';
  std::cout << "same_spin_overlap_exact = " << result.exact.overlap << '\n';
  std::cout << "same_spin_overlap_boundary = " << result.boundary.overlap << '\n';
  std::cout << "same_spin_overlap_abs_error = "
            << result.overlap_absolute_error << '\n';
  std::cout << "same_spin_one_electron_exact = "
            << result.exact.one_electron << '\n';
  std::cout << "same_spin_one_electron_boundary = "
            << result.boundary.one_electron << '\n';
  std::cout << "same_spin_one_electron_abs_error = "
            << result.one_electron_absolute_error << '\n';
  std::cout << "same_spin_alpha_exact = "
            << result.exact.same_spin_alpha_two_electron << '\n';
  std::cout << "same_spin_alpha_boundary = "
            << result.boundary.same_spin_alpha_two_electron << '\n';
  std::cout << "same_spin_alpha_abs_error = "
            << result.same_spin_alpha_absolute_error << '\n';
  std::cout << "same_spin_beta_exact = "
            << result.exact.same_spin_beta_two_electron << '\n';
  std::cout << "same_spin_beta_boundary = "
            << result.boundary.same_spin_beta_two_electron << '\n';
  std::cout << "same_spin_beta_abs_error = "
            << result.same_spin_beta_absolute_error << '\n';
  std::cout << "boundary_message_state_count = "
            << result.boundary.subtree_message_state_count << '\n';
  std::cout << "boundary_term_pair_count = "
            << result.boundary.subtree_term_pair_count << '\n';
  std::cout << "boundary_subdeterminant_evaluations = "
            << result.boundary.subdeterminant_evaluations << '\n';
  std::cout << "boundary_dp_transition_count = "
            << result.boundary.dp_transition_count << '\n';
  std::cout << '\n';
  return result;
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

std::vector<double> flatten_column_major(
    const std::vector<std::vector<double>>& row_major_matrix) {
  const int n_rows = static_cast<int>(row_major_matrix.size());
  const int n_cols = static_cast<int>(row_major_matrix.front().size());
  std::vector<double> storage(
      xmvb::to_size(n_rows * n_cols),
      0.0);
  for (int row = 0; row < n_rows; ++row) {
    if (static_cast<int>(row_major_matrix[xmvb::to_size(row)].size()) != n_cols) {
      throw std::invalid_argument("row_major_matrix must be rectangular");
    }
    for (int col = 0; col < n_cols; ++col) {
      storage[xmvb::to_size(col * n_rows + row)] =
          row_major_matrix[xmvb::to_size(row)][xmvb::to_size(col)];
    }
  }
  return storage;
}

int main_impl() {
  std::vector<ComponentData> components(4);
  const std::vector<std::vector<OrbitalPair>> component_pairs{
      {{0, 1}},
      {{2, 3}},
      {{4, 5}},
      {{6, 7}},
  };
  for (int node = 0; node < 4; ++node) {
    components[xmvb::to_size(node)].graph_node = node;
    components[xmvb::to_size(node)].left_pairs =
        component_pairs[xmvb::to_size(node)];
    components[xmvb::to_size(node)].right_pairs =
        component_pairs[xmvb::to_size(node)];
    components[xmvb::to_size(node)].left_orientation_terms =
        enumerate_orientation_terms(component_pairs[xmvb::to_size(node)]);
    components[xmvb::to_size(node)].right_orientation_terms =
        enumerate_orientation_terms(component_pairs[xmvb::to_size(node)]);
  }

  // Case 1 is the smallest non-star chain:
  //   0
  //   |
  //   1
  //   |
  //   3
  //
  // If the same-spin error already appears here, the problem is inside the
  // exact subtree-as-leaf message itself rather than in the root-level
  // multi-child merge.
  const ComponentTree chain_tree{
      .root_index = 0,
      .components = {
          components[0],
          components[1],
          components[3],
      },
      .children = {
          {1},
          {2},
          {},
      },
  };

  // Case 2 is the original branched non-star tree:
  //   0
  //  / \
  // 1   2
  // |
  // 3
  //
  // Each node is one covalent pair. The support-overlap matrix keeps only the
  // tree couplings (0-1), (0-2), and (1-3), so the rooted-tree driver is
  // tested on a genuine non-star hierarchy with one internal child subtree.
  const ComponentTree branched_tree{
      .root_index = 0,
      .components = components,
      .children = {
          {1, 2},
          {3},
          {},
          {},
      },
  };

  // Case 3 is the minimal one-leaf star:
  //   0
  //   |
  //   1
  //
  // This matches the production `one_leaf_star` topology exercised heavily by
  // the C6H6 benchmark.
  const ComponentTree one_leaf_star_tree{
      .root_index = 0,
      .components = {
          components[0],
          components[1],
      },
      .children = {
          {1},
          {},
      },
  };

  // Case 4 is the smallest true multi-leaf star:
  //   0
  //  / \
  // 1   2
  //
  // This is the critical shape for the separator redesign because the
  // production Hamiltonian driver used to route stars through a separate
  // one-leaf implementation. The generic boundary-only rooted-tree channels
  // must also be exact on this simplest star instance.
  const ComponentTree star_tree{
      .root_index = 0,
      .components = {
          components[0],
          components[1],
          components[2],
      },
      .children = {
          {1, 2},
          {},
          {},
      },
  };

  const std::vector<double> support_overlap_storage = flatten_column_major({
      {1.00, 0.08, 0.12, 0.05, 0.09, 0.04, 0.00, 0.00},
      {0.11, 0.97, 0.07, 0.10, 0.03, 0.06, 0.00, 0.00},
      {0.14, 0.09, 0.95, 0.11, 0.00, 0.00, 0.13, 0.07},
      {0.06, 0.12, 0.10, 0.98, 0.00, 0.00, 0.08, 0.09},
      {0.07, 0.04, 0.00, 0.00, 0.96, 0.10, 0.00, 0.00},
      {0.05, 0.07, 0.00, 0.00, 0.09, 0.99, 0.00, 0.00},
      {0.00, 0.00, 0.11, 0.06, 0.00, 0.00, 0.94, 0.08},
      {0.00, 0.00, 0.09, 0.10, 0.00, 0.00, 0.07, 0.97},
  });
  const std::vector<double> support_one_electron_storage = flatten_column_major({
      {0.31, 0.05, 0.07, 0.03, 0.02, 0.01, 0.00, 0.00},
      {0.04, 0.28, 0.06, 0.08, 0.01, 0.03, 0.00, 0.00},
      {0.09, 0.07, 0.35, 0.04, 0.00, 0.00, 0.06, 0.02},
      {0.02, 0.08, 0.05, 0.33, 0.00, 0.00, 0.03, 0.07},
      {0.03, 0.02, 0.00, 0.00, 0.29, 0.05, 0.00, 0.00},
      {0.01, 0.04, 0.00, 0.00, 0.06, 0.32, 0.00, 0.00},
      {0.00, 0.00, 0.08, 0.03, 0.00, 0.00, 0.27, 0.04},
      {0.00, 0.00, 0.02, 0.09, 0.00, 0.00, 0.05, 0.30},
  });

  const int packed_eri_size =
      TwoElectronIndexer::two_electron_storage_index(7, 7, 7, 7) + 1;
  std::vector<double> packed_active_two_electron_integrals(
      xmvb::to_size(packed_eri_size),
      0.0);
  for (int p = 0; p < 8; ++p) {
    for (int q = 0; q < 8; ++q) {
      for (int r = 0; r < 8; ++r) {
        for (int s = 0; s < 8; ++s) {
          const double value =
              0.21 + 0.025 * (p + 1) + 0.02 * (q + 1) + 0.015 * (r + 1) +
              0.01 * (s + 1);
          packed_active_two_electron_integrals[xmvb::to_size(
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] = value;
        }
      }
    }
  }

  std::cout << std::setprecision(15);
  const CollapsedHamiltonianComponentTreeStats chain_stats = run_case(
      "chain_rooted_subtree",
      chain_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const OverlapCheckResult chain_overlap = run_overlap_case(
      "chain_rooted_subtree",
      chain_tree,
      support_overlap_storage,
      8);
  const OneElectronCheckResult chain_one_electron = run_one_electron_case(
      "chain_rooted_subtree",
      chain_tree,
      support_overlap_storage,
      support_one_electron_storage,
      8);
  const OppositeSpinCheckResult chain_opposite_spin = run_opposite_spin_case(
      "chain_rooted_subtree",
      chain_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const SameSpinCheckResult chain_same_spin = run_same_spin_case(
      "chain_rooted_subtree",
      chain_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const SameSpinDeletionPatternBreakdown chain_breakdown =
      compute_chain_same_spin_alpha_breakdown(
          chain_tree,
          support_overlap_storage,
          packed_active_two_electron_integrals,
          8);
  std::cout << "chain_alpha_breakdown.root00_child22 = "
            << chain_breakdown.root00_child22 << '\n';
  std::cout << "chain_alpha_breakdown.root11_child11 = "
            << chain_breakdown.root11_child11 << '\n';
  std::cout << "chain_alpha_breakdown.root01_child21 = "
            << chain_breakdown.root01_child21 << '\n';
  std::cout << "chain_alpha_breakdown.root10_child12 = "
            << chain_breakdown.root10_child12 << '\n';
  std::cout << "chain_alpha_breakdown.mixed_total = "
            << (chain_breakdown.root01_child21 + chain_breakdown.root10_child12) << '\n';
  std::cout << "chain_alpha_breakdown.captured_total = "
            << (chain_breakdown.root00_child22 + chain_breakdown.root11_child11) << '\n';
  std::cout << '\n';
  const CollapsedHamiltonianComponentTreeStats branched_stats = run_case(
      "branched_rooted_subtree",
      branched_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const OverlapCheckResult branched_overlap = run_overlap_case(
      "branched_rooted_subtree",
      branched_tree,
      support_overlap_storage,
      8);
  const OneElectronCheckResult branched_one_electron = run_one_electron_case(
      "branched_rooted_subtree",
      branched_tree,
      support_overlap_storage,
      support_one_electron_storage,
      8);
  const OppositeSpinCheckResult branched_opposite_spin = run_opposite_spin_case(
      "branched_rooted_subtree",
      branched_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const SameSpinCheckResult branched_same_spin = run_same_spin_case(
      "branched_rooted_subtree",
      branched_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const CollapsedHamiltonianComponentTreeStats one_leaf_star_stats = run_case(
      "one_leaf_star_rooted_subtree",
      one_leaf_star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const OverlapCheckResult one_leaf_star_overlap = run_overlap_case(
      "one_leaf_star_rooted_subtree",
      one_leaf_star_tree,
      support_overlap_storage,
      8);
  const OneElectronCheckResult one_leaf_star_one_electron = run_one_electron_case(
      "one_leaf_star_rooted_subtree",
      one_leaf_star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      8);
  const OppositeSpinCheckResult one_leaf_star_opposite_spin = run_opposite_spin_case(
      "one_leaf_star_rooted_subtree",
      one_leaf_star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const SameSpinCheckResult one_leaf_star_same_spin = run_same_spin_case(
      "one_leaf_star_rooted_subtree",
      one_leaf_star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const CollapsedHamiltonianComponentTreeStats star_stats = run_case(
      "star_rooted_subtree",
      star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const OverlapCheckResult star_overlap = run_overlap_case(
      "star_rooted_subtree",
      star_tree,
      support_overlap_storage,
      8);
  const OneElectronCheckResult star_one_electron = run_one_electron_case(
      "star_rooted_subtree",
      star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      8);
  const OppositeSpinCheckResult star_opposite_spin = run_opposite_spin_case(
      "star_rooted_subtree",
      star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);
  const SameSpinCheckResult star_same_spin = run_same_spin_case(
      "star_rooted_subtree",
      star_tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      8);

  const auto is_case_ok =
      [](const CollapsedHamiltonianComponentTreeStats& stats) {
        return stats.overlap_absolute_error <= 1.0e-10 &&
            stats.one_electron_absolute_error <= 1.0e-10 &&
            stats.same_spin_alpha_absolute_error <= 1.0e-10 &&
            stats.same_spin_beta_absolute_error <= 1.0e-10 &&
            stats.opposite_spin_absolute_error <= 1.0e-10 &&
            stats.total_two_electron_absolute_error <= 1.0e-10 &&
            stats.total_electronic_hamiltonian_absolute_error <= 1.0e-10;
      };
  if (!is_case_ok(chain_stats) ||
      !is_case_ok(branched_stats) ||
      !is_case_ok(one_leaf_star_stats) ||
      !is_case_ok(star_stats)) {
    std::cerr << "collapsed rooted-tree two-electron separator test failed\n";
    return EXIT_FAILURE;
  }
  if (chain_overlap.absolute_error > 1.0e-10 ||
      branched_overlap.absolute_error > 1.0e-10 ||
      one_leaf_star_overlap.absolute_error > 1.0e-10 ||
      star_overlap.absolute_error > 1.0e-10) {
    std::cerr << "boundary overlap exact rooted-tree check failed\n";
    return EXIT_FAILURE;
  }
  const auto is_one_electron_case_ok =
      [](const OneElectronCheckResult& result) {
        return result.overlap_absolute_error <= 1.0e-10 &&
            result.one_electron_absolute_error <= 1.0e-10;
      };
  if (!is_one_electron_case_ok(chain_one_electron) ||
      !is_one_electron_case_ok(branched_one_electron) ||
      !is_one_electron_case_ok(one_leaf_star_one_electron) ||
      !is_one_electron_case_ok(star_one_electron)) {
    std::cerr << "boundary one-electron exact rooted-tree check failed\n";
    return EXIT_FAILURE;
  }
  const auto is_opposite_spin_case_ok =
      [](const OppositeSpinCheckResult& result) {
        return result.overlap_absolute_error <= 1.0e-10 &&
            result.one_electron_absolute_error <= 1.0e-10 &&
            result.opposite_spin_absolute_error <= 1.0e-10;
      };
  if (!is_opposite_spin_case_ok(chain_opposite_spin) ||
      !is_opposite_spin_case_ok(branched_opposite_spin) ||
      !is_opposite_spin_case_ok(one_leaf_star_opposite_spin) ||
      !is_opposite_spin_case_ok(star_opposite_spin)) {
    std::cerr << "boundary opposite-spin exact rooted-tree check failed\n";
    return EXIT_FAILURE;
  }
  const auto is_same_spin_case_ok =
      [](const SameSpinCheckResult& result) {
        return result.overlap_absolute_error <= 1.0e-10 &&
            result.one_electron_absolute_error <= 1.0e-10 &&
            result.same_spin_alpha_absolute_error <= 1.0e-10 &&
            result.same_spin_beta_absolute_error <= 1.0e-10;
      };
  if (!is_same_spin_case_ok(chain_same_spin) ||
      !is_same_spin_case_ok(branched_same_spin) ||
      !is_same_spin_case_ok(one_leaf_star_same_spin) ||
      !is_same_spin_case_ok(star_same_spin)) {
    std::cerr << "boundary same-spin exact rooted-tree check failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main() {
  try {
    return main_impl();
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
