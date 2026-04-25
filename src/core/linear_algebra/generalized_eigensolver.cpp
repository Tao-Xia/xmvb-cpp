#include "core/linear_algebra/generalized_eigensolver.hpp"

#include <string>
#include <stdexcept>

#include "lapacke.h"

namespace xmvb::core {

namespace {

void validate_generalized_eigenproblem_inputs(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) {
  if (dimension <= 0) {
    throw std::invalid_argument("dimension must be positive");
  }

  const std::size_t matrix_dimension = static_cast<std::size_t>(dimension);
  const std::size_t expected_size =
      matrix_dimension * matrix_dimension;
  if (hamiltonian_matrix.size() != expected_size ||
      overlap_matrix.size() != expected_size) {
    throw std::invalid_argument(
        "hamiltonian_matrix and overlap_matrix must be square matrices");
  }
}

void check_generalized_eigensolver_info(
    lapack_int info) {
  if (info < 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: illegal argument " + std::to_string(-info));
  }
  if (info > 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: overlap_matrix is not positive definite or the "
        "eigensolver did not converge (info=" + std::to_string(info) + ")");
  }
}

}  // namespace

GeneralizedEigenResult GeneralizedEigensolver::solve(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix,
      overlap_matrix,
      dimension);

  GeneralizedEigenResult result;
  std::vector<double> hamiltonian_matrix_copy = hamiltonian_matrix;
  std::vector<double> overlap_matrix_copy = overlap_matrix;
  result.eigenvalues.resize(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR,
      1,
      'V',
      'U',
      dimension,
      hamiltonian_matrix_copy.data(),
      dimension,
      overlap_matrix_copy.data(),
      dimension,
      result.eigenvalues.data());
  check_generalized_eigensolver_info(info);
  result.eigenvector_matrix = std::move(hamiltonian_matrix_copy);
  return result;
}

std::vector<double> GeneralizedEigensolver::solve_eigenvalues_only(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix,
      overlap_matrix,
      dimension);

  std::vector<double> hamiltonian_matrix_copy = hamiltonian_matrix;
  std::vector<double> overlap_matrix_copy = overlap_matrix;
  std::vector<double> eigenvalues(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR,
      1,
      'N',
      'U',
      dimension,
      hamiltonian_matrix_copy.data(),
      dimension,
      overlap_matrix_copy.data(),
      dimension,
      eigenvalues.data());
  check_generalized_eigensolver_info(info);
  return eigenvalues;
}

}  // namespace xmvb::core
