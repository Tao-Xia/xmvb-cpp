#include "core/linear_algebra/generalized_eigensolver.hpp"

#include <string>
#include <stdexcept>

#include "lapacke.h"

namespace xmvb::core {

GeneralizedEigenResult GeneralizedEigensolver::solve(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  if (dimension <= 0) {
    throw std::invalid_argument("dimension must be positive");
  }

  const std::size_t expected_size =
      static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension);
  if (hamiltonian_matrix.size() != expected_size ||
      overlap_matrix.size() != expected_size) {
    throw std::invalid_argument(
        "hamiltonian_matrix and overlap_matrix must be square matrices");
  }

  GeneralizedEigenResult result;
  result.eigenvector_matrix = hamiltonian_matrix;

  std::vector<double> overlap_matrix_copy = overlap_matrix;
  result.eigenvalues.resize(static_cast<std::size_t>(dimension));

  const lapack_int info = LAPACKE_dsygv(
      LAPACK_COL_MAJOR,
      1,
      'V',
      'U',
      dimension,
      result.eigenvector_matrix.data(),
      dimension,
      overlap_matrix_copy.data(),
      dimension,
      result.eigenvalues.data());

  if (info < 0) {
    throw std::runtime_error(
        "LAPACKE_dsygv failed: illegal argument " + std::to_string(-info));
  }
  if (info > 0) {
    throw std::runtime_error(
        "LAPACKE_dsygv failed: overlap_matrix is not positive definite or the "
        "eigensolver did not converge (info=" + std::to_string(info) + ")");
  }

  return result;
}

}  // namespace xmvb::core
