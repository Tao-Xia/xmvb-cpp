#include "vb/orbital/legacy_jacobi_diagonalizer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace xmvb::vb {

namespace {

void require_square_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  if (!matrix.allFinite()) {
    throw std::invalid_argument(std::string(label) + " must be finite");
  }
}

}  // namespace

LegacyJacobiDiagonalizationResult diagonalize_self_adjoint_legacy_jacobi(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double tolerance,
    int max_sweeps) {
  require_square_matrix(matrix, "legacy Jacobi input");
  if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
    throw std::invalid_argument("legacy Jacobi tolerance must be finite and non-negative");
  }

  const int dimension = static_cast<int>(matrix.rows());
  LegacyJacobiDiagonalizationResult result;
  result.eigenvectors =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
  result.eigenvalues =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
  if (dimension == 0) {
    return result;
  }

  Eigen::MatrixXd working_matrix = matrix;
  if (max_sweeps <= 0) {
    // The legacy code sweeps until convergence with no explicit guard.  Keep a
    // generous quadratic cap only as a safety net for corrupted inputs.
    max_sweeps = std::max(32, 16 * dimension * dimension);
  }

  double max_off_diagonal_squared = 1.0;
  int sweep_index = 0;
  while (max_off_diagonal_squared > tolerance) {
    if (sweep_index >= max_sweeps) {
      throw std::runtime_error(
          "legacy Jacobi diagonalization did not converge within the sweep cap");
    }
    ++sweep_index;

    max_off_diagonal_squared = 0.0;
    for (int column_i = 1; column_i < dimension; ++column_i) {
      for (int column_j = 0; column_j < column_i; ++column_j) {
        const double diagonal_i = working_matrix(column_i, column_i);
        const double diagonal_j = working_matrix(column_j, column_j);
        const double off_diagonal = working_matrix(column_i, column_j);
        const double off_diagonal_squared = off_diagonal * off_diagonal;
        if (off_diagonal_squared > max_off_diagonal_squared) {
          max_off_diagonal_squared = off_diagonal_squared;
        }
        if (!(off_diagonal_squared > tolerance)) {
          continue;
        }

        double difference = diagonal_i - diagonal_j;
        double sign = 2.0;
        if (difference < 0.0) {
          sign = -2.0;
          difference = -difference;
        }
        const double tangent_denominator =
            difference +
            std::sqrt(difference * difference + 4.0 * off_diagonal_squared);
        if (!(tangent_denominator > std::numeric_limits<double>::epsilon()) ||
            !std::isfinite(tangent_denominator)) {
          throw std::runtime_error(
              "legacy Jacobi diagonalization encountered a singular rotation");
        }
        const double tangent = sign * off_diagonal / tangent_denominator;
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = cosine * tangent;

        for (int row_k = 0; row_k < dimension; ++row_k) {
          const double rotated_j =
              cosine * result.eigenvectors(row_k, column_j) -
              sine * result.eigenvectors(row_k, column_i);
          result.eigenvectors(row_k, column_i) =
              sine * result.eigenvectors(row_k, column_j) +
              cosine * result.eigenvectors(row_k, column_i);
          result.eigenvectors(row_k, column_j) = rotated_j;

          if (row_k == column_j || row_k == column_i) {
            continue;
          }
          const double rotated_matrix_j =
              cosine * working_matrix(row_k, column_j) -
              sine * working_matrix(row_k, column_i);
          working_matrix(row_k, column_i) =
              sine * working_matrix(row_k, column_j) +
              cosine * working_matrix(row_k, column_i);
          working_matrix(row_k, column_j) = rotated_matrix_j;
          working_matrix(column_i, row_k) = working_matrix(row_k, column_i);
          working_matrix(column_j, row_k) = working_matrix(row_k, column_j);
        }

        working_matrix(column_i, column_i) =
            cosine * cosine * diagonal_i +
            sine * sine * diagonal_j +
            2.0 * sine * cosine * off_diagonal;
        working_matrix(column_j, column_j) =
            cosine * cosine * diagonal_j +
            sine * sine * diagonal_i -
            2.0 * sine * cosine * off_diagonal;
        working_matrix(column_i, column_j) = 0.0;
        working_matrix(column_j, column_i) = 0.0;
      }
    }
  }

  for (int diagonal_index = 0; diagonal_index < dimension; ++diagonal_index) {
    result.eigenvalues(diagonal_index) =
        working_matrix(diagonal_index, diagonal_index);
  }
  if (!result.eigenvectors.allFinite() || !result.eigenvalues.allFinite()) {
    throw std::runtime_error(
        "legacy Jacobi diagonalization returned non-finite eigenpairs");
  }
  return result;
}

}  // namespace xmvb::vb
