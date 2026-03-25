#pragma once

#include <vector>

#include "vb/matrices/determinant_types.hpp"

namespace xmvb::vb {

/**
 * @brief Resolves determinant overlap quantities from a determinant overlap submatrix.
 *
 * This class is the first standalone C++ replacement for the legacy
 * `MatLDR` + `Cofactor1` path in `gradient_rdm.F90`.
 */
class DeterminantOverlapResolver {
public:
  /**
   * @brief Constructs a resolver with a singular-value threshold.
   *
   * @param linear_dependence_threshold Singular values below this value are
   *   treated as zero when computing nullity.
   */
  explicit DeterminantOverlapResolver(double linear_dependence_threshold = 1.0e-12);

  /**
   * @brief Resolves determinant overlap quantities for a square overlap matrix.
   *
   * The input matrix must use column-major storage so that it remains layout
   * compatible with the existing Fortran-side matrices.
   *
   * @param overlap_submatrix Column-major determinant overlap submatrix.
   * @param n_electrons Dimension of the square matrix.
   * @return DeterminantOverlapResult Thresholded nullity plus the full SVD
   *   factors needed by the determinant-level kernels.
   */
  DeterminantOverlapResult resolve(
      const std::vector<double>& overlap_submatrix,
      int n_electrons) const;

private:
  double linear_dependence_threshold_;
};

}  // namespace xmvb::vb
