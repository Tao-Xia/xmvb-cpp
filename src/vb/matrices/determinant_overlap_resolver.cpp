#include "vb/matrices/determinant_overlap_resolver.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace xmvb::vb {

namespace {

using ColumnMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ConstColumnMajorMatrixMap =
    Eigen::Map<const ColumnMajorMatrixXd>;

ConstColumnMajorMatrixMap map_column_major_matrix(
    const std::vector<double>& matrix_storage,
    int dimension) {
  return ConstColumnMajorMatrixMap(matrix_storage.data(), dimension, dimension);
}

double product_of_singular_values(const Eigen::VectorXd& singular_values) {
  double product = 1.0;
  for (int singular_index = 0; singular_index < singular_values.size(); ++singular_index) {
    product *= singular_values(singular_index);
  }
  return product;
}

}  // namespace

DeterminantOverlapResolver::DeterminantOverlapResolver(double linear_dependence_threshold)
    : linear_dependence_threshold_(linear_dependence_threshold) {
  if (linear_dependence_threshold_ <= 0.0) {
    throw std::invalid_argument("linear_dependence_threshold must be positive");
  }
}

DeterminantOverlapResult DeterminantOverlapResolver::resolve(
    const std::vector<double>& overlap_submatrix,
    int n_electrons) const {

  const std::size_t matrix_size =
      static_cast<std::size_t>(n_electrons) * static_cast<std::size_t>(n_electrons);
  if (overlap_submatrix.size() != matrix_size) {
    throw std::invalid_argument("overlap_submatrix size does not match n_electrons");
  }

  const auto overlap_matrix = map_column_major_matrix(overlap_submatrix, n_electrons);
  const Eigen::JacobiSVD<ColumnMajorMatrixXd> singular_value_decomposition(
      overlap_matrix,
      Eigen::ComputeFullU | Eigen::ComputeFullV);

  if (singular_value_decomposition.info() != Eigen::Success) {
    throw std::runtime_error("Eigen SVD failed while resolving determinant overlap");
  }

  const Eigen::VectorXd singular_values = singular_value_decomposition.singularValues();
  const ColumnMajorMatrixXd matrix_U = singular_value_decomposition.matrixU();
  const ColumnMajorMatrixXd matrix_V = singular_value_decomposition.matrixV();

  DeterminantOverlapResult result;
  result.n_electrons = n_electrons;
  result.singular_values = singular_values;
  result.matrix_U = matrix_U;
  result.matrix_V = matrix_V;
  result.nullity = 0;
  for (int singular_index = 0; singular_index < singular_values.size(); ++singular_index) {
    if (std::abs(singular_values(singular_index)) < linear_dependence_threshold_) {
      ++result.nullity;
    }
  }

  const double det_U = matrix_U.determinant();
  const double det_V = matrix_V.determinant();
  result.parity = ((det_U * det_V) >= 0.0) ? 1.0 : -1.0;

  if (result.nullity == 0) {
    result.overlap_determinant =
        result.parity * product_of_singular_values(result.singular_values);
  } else {
    result.overlap_determinant = 0.0;
  }
  return result;
}

}  // namespace xmvb::vb
