#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Input data for one determinant pair contribution.
 */
struct DeterminantPairInput {
  /**
   * @brief Zero-based determinant index on the left.
   */
  int determinant_index_left = 0;

  /**
   * @brief Zero-based determinant index on the right.
   */
  int determinant_index_right = 0;

  /**
   * @brief Zero-based occupied orbitals of the left determinant.
   */
  std::vector<int> occupied_orbitals_left;

  /**
   * @brief Zero-based occupied orbitals of the right determinant.
   */
  std::vector<int> occupied_orbitals_right;

  /**
   * @brief Column-major determinant overlap submatrix between occupied orbitals.
   */
  std::vector<double> determinant_overlap_submatrix;
};

}  // namespace xmvb::vb
