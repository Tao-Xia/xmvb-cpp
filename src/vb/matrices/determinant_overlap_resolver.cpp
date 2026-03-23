#include "vb/matrices/determinant_overlap_resolver.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace xmvb::vb {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ConstMatrixMap = Eigen::Map<const Matrix>;

DeterminantOverlapResolver::DeterminantOverlapResolver(double linear_dependence_threshold)
    : linear_dependence_threshold_(linear_dependence_threshold) {
  if (linear_dependence_threshold_ <= 0.0) {
    throw std::invalid_argument("linear_dependence_threshold must be positive");
  }
}

DeterminantOverlapResult DeterminantOverlapResolver::resolve(
    const std::vector<double>& overlap_submatrix,
    int n_electrons) const 
{
  const std::size_t matrix_size = static_cast<std::size_t>(n_electrons) * n_electrons;

  if (overlap_submatrix.size() != matrix_size) {
    throw std::invalid_argument("overlap_submatrix size does not match n_electrons");
  }

  ConstMatrixMap overlap_matrix(
    overlap_submatrix.data(),
    n_electrons,
    n_electrons
  );

  const Eigen::BDCSVD<Matrix, Eigen::ComputeFullU | Eigen::ComputeFullV> 
    SVD(overlap_matrix);
    
  if (SVD.info() != Eigen::Success) {
    throw std::runtime_error("Eigen SVD failed while resolving determinant overlap");
  }

  DeterminantOverlapResult result;
  result.n_electrons = n_electrons;
  
  result.singular_values = SVD.singularValues();
  result.matrix_U = SVD.matrixU();
  result.matrix_V = SVD.matrixV();

  result.nullity = (result.singular_values.array() < linear_dependence_threshold_).count();

  const double det_U = result.matrix_U.determinant();
  const double det_V = result.matrix_V.determinant();
  result.parity = ((det_U * det_V) >= 0.0) ? 1.0 : -1.0;

  if (result.nullity == 0) {
    result.overlap_determinant = result.parity * result.singular_values.prod();
  } else {
    result.overlap_determinant = 0.0;
  }
  
  return result;
}

}  // namespace xmvb::vb