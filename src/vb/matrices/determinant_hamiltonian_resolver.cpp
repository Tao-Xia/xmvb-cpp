#include "vb/matrices/determinant_hamiltonian_resolver.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vb/matrices/biorthogonal_spin_pair.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

void build_second_order_minor_matrix(
    const ColumnMajorMatrixXd& matrix,
    int removed_row_first,
    int removed_row_second,
    int removed_column_first,
    int removed_column_second,
    Eigen::Ref<ColumnMajorMatrixXd> workspace_minor) {
  const int dimension = static_cast<int>(matrix.rows());
  int minor_column = 0;
  for (int column = 0; column < dimension; ++column) {
    if (column == removed_column_first || column == removed_column_second) {
      continue;
    }
    int minor_row = 0;
    for (int row = 0; row < dimension; ++row) {
      if (row == removed_row_first || row == removed_row_second) {
        continue;
      }
      workspace_minor(minor_row, minor_column) = matrix(row, column);
      ++minor_row;
    }
    ++minor_column;
  }
}

double compute_second_order_cofactor_from_minor(
    const ColumnMajorMatrixXd& overlap_matrix,
    int removed_right_row_first,
    int removed_right_row_second,
    int removed_left_column_first,
    int removed_left_column_second,
    Eigen::Ref<ColumnMajorMatrixXd> workspace_minor) {
  build_second_order_minor_matrix(
      overlap_matrix,
      removed_right_row_first,
      removed_right_row_second,
      removed_left_column_first,
      removed_left_column_second,
      workspace_minor);

  const double sign =
      ((removed_right_row_first + removed_right_row_second +
        removed_left_column_first + removed_left_column_second) % 2 == 0)
      ? 1.0
      : -1.0;

  const Eigen::FullPivLU<Eigen::Ref<ColumnMajorMatrixXd>> lu_factorization(workspace_minor);
  return sign * lu_factorization.determinant();
}

DeterminantHamiltonianResult resolve_original_same_spin_hamiltonian(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& determinant_overlap_submatrix,
    const DeterminantOverlapResult& overlap_result,
    const std::vector<double>& one_electron_matrix,
    int n_orbitals,
    const std::vector<double>& packed_two_electron_integrals) {
  const int n_electrons = static_cast<int>(occupied_orbitals_left.size());
  const auto overlap_matrix = map_column_major_matrix(
      determinant_overlap_submatrix,
      n_electrons);
  const auto first_order_cofactor_matrix =
      build_first_order_cofactor_matrix_from_result(overlap_result);
  const ConstColumnMajorMatrixMap one_electron_integrals(
      one_electron_matrix.data(),
      n_orbitals,
      n_orbitals);

  DeterminantHamiltonianResult result;
  result.overlap_determinant = overlap_result.overlap_determinant;
  result.nullity = overlap_result.nullity;

  if (result.nullity >= 3) {
    return result;
  }

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occupied_orbitals_left[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right =
          occupied_orbitals_right[static_cast<std::size_t>(right_row)];
      result.total_hamiltonian +=
          one_electron_integrals(orbital_index_right, orbital_index_left) *
          first_order_cofactor_matrix(right_row, left_column);
    }
  }
  result.one_electron_hamiltonian = result.total_hamiltonian;

  if (n_electrons < 2) {
    return result;
  }

  ColumnMajorMatrixXd workspace_minor(n_electrons - 2, n_electrons - 2);
  const double inv_overlap_det =
      (result.nullity == 0) ? (1.0 / overlap_result.overlap_determinant) : 0.0;

  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orb_L1 = occupied_orbitals_left[static_cast<std::size_t>(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orb_R1 = occupied_orbitals_right[static_cast<std::size_t>(right_first)];
      const double cofactor_11 = first_order_cofactor_matrix(right_first, left_first);

      const int pair_R1_L1 = TwoElectronIndexer::packed_pair_index(orb_R1, orb_L1);

      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orb_L2 = occupied_orbitals_left[static_cast<std::size_t>(left_second)];
        const double cofactor_12 = first_order_cofactor_matrix(right_first, left_second);

        const int pair_R1_L2 = TwoElectronIndexer::packed_pair_index(orb_R1, orb_L2);

        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orb_R2 = occupied_orbitals_right[static_cast<std::size_t>(right_second)];
          const double cofactor_22 = first_order_cofactor_matrix(right_second, left_second);
          const double cofactor_21 = first_order_cofactor_matrix(right_second, left_first);

          double second_order_cofactor = 0.0;
          if (result.nullity == 0) {
            second_order_cofactor =
                (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21) * inv_overlap_det;
          } else {
            second_order_cofactor = compute_second_order_cofactor_from_minor(
                overlap_matrix,
                right_first,
                right_second,
                left_first,
                left_second,
                workspace_minor);
          }

          const int pair_R2_L2 = TwoElectronIndexer::packed_pair_index(orb_R2, orb_L2);
          const int pair_R2_L1 = TwoElectronIndexer::packed_pair_index(orb_R2, orb_L1);

          const int columb_index =
              TwoElectronIndexer::packed_pair_of_pairs_index(pair_R1_L1, pair_R2_L2);
          const int exchange_index =
              TwoElectronIndexer::packed_pair_of_pairs_index(pair_R1_L2, pair_R2_L1);

          result.total_hamiltonian +=
              (packed_two_electron_integrals[static_cast<std::size_t>(columb_index)] -
               packed_two_electron_integrals[static_cast<std::size_t>(exchange_index)]) *
              second_order_cofactor;
        }
      }
    }
  }

  return result;
}

}  // namespace

DeterminantHamiltonianResolver::DeterminantHamiltonianResolver(
    VbScfAlgorithm algorithm)
    : overlap_resolver_(),
      algorithm_(algorithm) {}

DeterminantHamiltonianResolver::DeterminantHamiltonianResolver(
    DeterminantOverlapResolver overlap_resolver,
    VbScfAlgorithm algorithm)
    : overlap_resolver_(std::move(overlap_resolver)),
      algorithm_(algorithm) {}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& determinant_overlap_submatrix,
    const std::vector<double>& one_electron_matrix,
    int n_orbitals,
    const std::vector<double>& packed_two_electron_integrals) const {
  if (occupied_orbitals_left.size() != occupied_orbitals_right.size()) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (occupied_orbitals_left.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }

  const auto overlap_result = overlap_resolver_.resolve(
      determinant_overlap_submatrix,
      static_cast<int>(occupied_orbitals_left.size()));
  return resolve(
      occupied_orbitals_left,
      occupied_orbitals_right,
      determinant_overlap_submatrix,
      overlap_result,
      one_electron_matrix,
      n_orbitals,
      packed_two_electron_integrals);
}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& determinant_overlap_submatrix,
    const DeterminantOverlapResult& overlap_result,
    const std::vector<double>& one_electron_matrix,
    int n_orbitals,
    const std::vector<double>& packed_two_electron_integrals) const {
  if (occupied_orbitals_left.size() != occupied_orbitals_right.size()) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (occupied_orbitals_left.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }
  if (n_orbitals <= 0) {
    throw std::invalid_argument("n_orbitals must be positive");
  }

  const int n_electrons = static_cast<int>(occupied_orbitals_left.size());
  const std::size_t one_electron_matrix_size =
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_orbitals);
  if (one_electron_matrix.size() != one_electron_matrix_size) {
    throw std::invalid_argument("one_electron_matrix size does not match n_orbitals");
  }
  const std::size_t overlap_matrix_size =
      static_cast<std::size_t>(n_electrons) * static_cast<std::size_t>(n_electrons);
  if (determinant_overlap_submatrix.size() != overlap_matrix_size) {
    throw std::invalid_argument("determinant_overlap_submatrix size does not match electron count");
  }
  if (overlap_result.n_electrons != n_electrons) {
    throw std::invalid_argument("precomputed overlap_result electron count mismatch");
  }

  if (algorithm_ == VbScfAlgorithm::Biorthogonal) {
    DeterminantHamiltonianResult result;
    result.overlap_determinant = overlap_result.overlap_determinant;
    result.nullity = overlap_result.nullity;
    const auto hamiltonian_result = compute_same_spin_biorthogonal_hamiltonian(
        occupied_orbitals_left,
        occupied_orbitals_right,
        one_electron_matrix,
        n_orbitals,
        packed_two_electron_integrals,
        overlap_result);
    result.one_electron_hamiltonian = hamiltonian_result.one_electron_hamiltonian;
    result.total_hamiltonian = hamiltonian_result.total_hamiltonian;
    return result;
  }

  return resolve_original_same_spin_hamiltonian(
      occupied_orbitals_left,
      occupied_orbitals_right,
      determinant_overlap_submatrix,
      overlap_result,
      one_electron_matrix,
      n_orbitals,
      packed_two_electron_integrals);
}

}  // namespace xmvb::vb
