#include "vbscf/determinants/algebra/overlap.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace xmvb::vb {
namespace {

void set_lu_determinant_state(
    const Eigen::FullPivLU<Eigen::MatrixXd>& lu,
    DeterminantOverlapResult* result) {
  const auto diagonal = lu.matrixLU().diagonal();
  double sign = static_cast<double>(
      lu.permutationP().determinant() *
      lu.permutationQ().determinant());
  double log_abs = 0.0;
  for (Eigen::Index index = 0; index < diagonal.size(); ++index) {
    const double pivot = diagonal(index);
    if (pivot == 0.0 || !std::isfinite(pivot)) {
      result->determinant_sign = 0.0;
      result->log_abs_determinant =
          -std::numeric_limits<double>::infinity();
      return;
    }
    sign = std::signbit(pivot) ? -sign : sign;
    log_abs += std::log(std::abs(pivot));
  }
  result->determinant_sign = sign >= 0.0 ? 1.0 : -1.0;
  result->log_abs_determinant = log_abs;
}

void set_svd_determinant_state(
    double parity,
    const Eigen::Ref<const Eigen::VectorXd>& singular_values,
    int nullity,
    DeterminantOverlapResult* result) {
  if (nullity != 0) {
    result->determinant_sign = 0.0;
    result->log_abs_determinant =
        -std::numeric_limits<double>::infinity();
    return;
  }
  result->determinant_sign = parity >= 0.0 ? 1.0 : -1.0;
  result->log_abs_determinant = singular_values.array().log().sum();
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
    int n_electrons) const 
{
  const std::size_t matrix_size = n_electrons * n_electrons;

  if (overlap_submatrix.size() != matrix_size) {
    throw std::invalid_argument("overlap_submatrix size does not match n_electrons");
  }

  Eigen::Map<const Eigen::MatrixXd> overlap_matrix(
    overlap_submatrix.data(),
    n_electrons,
    n_electrons
  );

  return resolve_matrix(overlap_matrix);
}

DeterminantOverlapResult DeterminantOverlapResolver::resolve_matrix(
    const Eigen::MatrixXd& overlap_matrix) const {
  if (overlap_matrix.rows() != overlap_matrix.cols()) {
    throw std::invalid_argument("overlap_submatrix must be square");
  }

  DeterminantOverlapResult result;
  result.n_electrons = overlap_matrix.rows();
  result.overlap_submatrix = overlap_matrix;

  Eigen::MatrixXd overlap_matrix_copy = overlap_matrix;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(overlap_matrix_copy);
  lu.setThreshold(linear_dependence_threshold_);

  if (lu.rank() == result.n_electrons) {
    result.nullity = 0;
    result.overlap_determinant = lu.determinant();
    set_lu_determinant_state(lu, &result);
    result.inverse_overlap_submatrix = lu.inverse();
    return result;
  }

  const Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV>
      SVD(overlap_matrix);
  if (SVD.info() != Eigen::Success) {
    throw std::runtime_error("Eigen SVD failed while resolving determinant overlap");
  }

  result.singular_values = SVD.singularValues();
  result.matrix_U = SVD.matrixU();
  result.matrix_V = SVD.matrixV();
  result.nullity = (result.singular_values.array() < linear_dependence_threshold_).count();

  if (result.nullity == 0) {
    const double det_U = result.matrix_U.determinant();
    const double det_V = result.matrix_V.determinant();
    result.parity = ((det_U * det_V) >= 0.0) ? 1.0 : -1.0;
    result.overlap_determinant = result.parity * result.singular_values.prod();
    set_svd_determinant_state(
        result.parity,
        result.singular_values,
        result.nullity,
        &result);
    const Eigen::VectorXd inverse_singular_values =
        result.singular_values.cwiseInverse();
    result.inverse_overlap_submatrix.noalias() =
        result.matrix_V * inverse_singular_values.asDiagonal() *
        result.matrix_U.transpose();
  } else {
    result.overlap_determinant = 0.0;
    const double det_U = result.matrix_U.determinant();
    const double det_V = result.matrix_V.determinant();
    result.parity = ((det_U * det_V) >= 0.0) ? 1.0 : -1.0;
    set_svd_determinant_state(
        result.parity,
        result.singular_values,
        result.nullity,
        &result);
  }
  
  return result;
}

}  // namespace xmvb::vb
