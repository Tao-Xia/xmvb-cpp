#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/contracts/types.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>
#include <Eigen/SVD>

namespace xmvb::vb {

namespace {
int pair_index(int i, int j) { return j * (j - 1) / 2 + i; }
int triple_index(int i, int j, int k) {
  return i + j * (j - 1) / 2 + k * (k - 1) * (k - 2) / 6;
}
int quadruple_index(int i, int j, int k, int l) {
  return i + j * (j - 1) / 2 + k * (k - 1) * (k - 2) / 6 +
      l * (l - 1) * (l - 2) * (l - 3) / 24;
}

double matrix_infinity_norm(const Eigen::MatrixXd& matrix) {
  if (matrix.size() == 0) {
    return 0.0;
  }
  return matrix.cwiseAbs().rowwise().sum().maxCoeff();
}
}

CofactorDifferential::CofactorDifferential(
    const DeterminantOverlapResult& overlap) {
  const int n = overlap.n_electrons;
  if (overlap.overlap_submatrix.rows() != n ||
      overlap.overlap_submatrix.cols() != n) {
    if (n != 0 || overlap.overlap_submatrix.size() != 0) {
      throw std::invalid_argument(
          "cofactor differential overlap dimensions are inconsistent");
    }
  }
  if (n == 0) {
    regular_ = true;
    determinant_ = 1.0;
    return;
  }
  const bool has_inverse =
      overlap.inverse_overlap_submatrix.rows() == n &&
      overlap.inverse_overlap_submatrix.cols() == n;
  const double condition_estimate = has_inverse
      ? matrix_infinity_norm(overlap.overlap_submatrix) *
            matrix_infinity_norm(overlap.inverse_overlap_submatrix)
      : std::numeric_limits<double>::infinity();
  // The highest regular formula below contains four inverse factors.  Bound
  // their roundoff amplification by sqrt(machine epsilon); pairs outside this
  // numerical region use the inverse-free SVD polynomial representation.
  constexpr double highest_inverse_power = 4.0;
  const double stable_condition_limit = std::pow(
      std::numeric_limits<double>::epsilon(),
      -1.0 / (2.0 * highest_inverse_power));
  const bool stable_regular =
      n >= 4 && overlap.nullity == 0 && has_inverse &&
      condition_estimate <= stable_condition_limit;
  if (stable_regular) {
    regular_ = true;
    determinant_ = overlap.overlap_determinant;
    inverse_ = overlap.inverse_overlap_submatrix;
    value_.noalias() = determinant_ * inverse_.transpose();
    return;
  }
  if (overlap.matrix_U.rows() == n && overlap.matrix_U.cols() == n &&
      overlap.matrix_V.rows() == n && overlap.matrix_V.cols() == n &&
      overlap.singular_values.size() == n) {
    u_ = overlap.matrix_U;
    v_ = overlap.matrix_V;
    singular_values_ = overlap.singular_values;
    parity_ = overlap.parity;
  } else {
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        overlap.overlap_submatrix,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success) {
      throw std::runtime_error(
          "cofactor differential SVD failed on an ill-conditioned pair");
    }
    u_ = svd.matrixU();
    v_ = svd.matrixV();
    singular_values_ = svd.singularValues();
    parity_ = u_.determinant() * v_.determinant() < 0.0 ? -1.0 : 1.0;
  }
  u2_ = pair_rotation(u_);
  v2_ = pair_rotation(v_);
  value_ = Eigen::MatrixXd::Zero(u_.rows(), u_.rows());
  for (int i = 0; i < value_.rows(); ++i) value_(i, i) = complement_product(i);
  value_ = restore(value_);
  const int pair_count = u_.rows() * (u_.rows() - 1) / 2;
  const int triple_count = n * (n - 1) * (n - 2) / 6;
  const int quadruple_count = n * (n - 1) * (n - 2) * (n - 3) / 24;
  pair_complements_.resize(pair_count);
  triple_complements_.resize(triple_count);
  quadruple_complements_.resize(quadruple_count);
  Eigen::VectorXd second_diagonal(pair_count);
  for (int j = 1; j < u_.rows(); ++j)
    for (int i = 0; i < j; ++i)
      pair_complements_(pair_index(i, j)) =
          second_diagonal(pair_index(i, j)) = complement_product(i, j);
  for (int k = 2; k < n; ++k)
    for (int j = 1; j < k; ++j)
      for (int i = 0; i < j; ++i)
        triple_complements_(triple_index(i, j, k)) =
            complement_product(i, j, k);
  for (int l = 3; l < n; ++l)
    for (int k = 2; k < l; ++k)
      for (int j = 1; j < k; ++j)
        for (int i = 0; i < j; ++i)
          quadruple_complements_(quadruple_index(i, j, k, l)) =
              complement_product(i, j, k, l);
  second_ = u2_ * second_diagonal.asDiagonal() * v2_.transpose();
}

double CofactorDifferential::complement_product(int i, int j, int k, int excluded_l) const {
  double product = parity_;
  for (int l = 0; l < singular_values_.size(); ++l)
    if (l != i && l != j && l != k && l != excluded_l) product *= singular_values_(l);
  return product;
}

double CofactorDifferential::cached_pair_complement(int i, int j) const {
  if (i > j) std::swap(i, j);
  return pair_complements_(pair_index(i, j));
}

double CofactorDifferential::cached_triple_complement(int i, int j, int k) const {
  int indices[3] = {i, j, k};
  std::sort(indices, indices + 3);
  return triple_complements_(triple_index(indices[0], indices[1], indices[2]));
}

double CofactorDifferential::cached_quadruple_complement(
    int i, int j, int k, int l) const {
  int indices[4] = {i, j, k, l};
  std::sort(indices, indices + 4);
  return quadruple_complements_(
      quadruple_index(indices[0], indices[1], indices[2], indices[3]));
}

Eigen::MatrixXd CofactorDifferential::rotate(const Eigen::MatrixXd& direction) const {
  if (regular_) {
    throw std::logic_error("regular cofactor path does not use SVD rotations");
  }
  if (direction.rows() != u_.rows() || direction.cols() != v_.rows())
    throw std::invalid_argument("cofactor direction has inconsistent dimensions");
  return u_.transpose() * direction * v_;
}

Eigen::MatrixXd CofactorDifferential::restore(const Eigen::MatrixXd& cofactor) const {
  if (regular_) {
    throw std::logic_error("regular cofactor path does not use SVD rotations");
  }
  return u_ * cofactor * v_.transpose();
}

const Eigen::MatrixXd& CofactorDifferential::value() const {
  return value_;
}

Eigen::MatrixXd CofactorDifferential::first(const Eigen::MatrixXd& direction) const {
  if (regular_) {
    if (direction.rows() != inverse_.rows() ||
        direction.cols() != inverse_.cols()) {
      throw std::invalid_argument(
          "cofactor direction has inconsistent dimensions");
    }
    const double trace = (inverse_ * direction).trace();
    return determinant_ *
        (trace * inverse_.transpose() -
         (inverse_ * direction * inverse_).transpose());
  }
  const Eigen::MatrixXd a = rotate(direction);
  Eigen::MatrixXd c = Eigen::MatrixXd::Zero(a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = i + 1; j < a.rows(); ++j) {
      const double p = cached_pair_complement(i, j);
      c(i, i) += p * a(j, j); c(j, j) += p * a(i, i);
      c(i, j) -= p * a(j, i); c(j, i) -= p * a(i, j);
    }
  }
  return restore(c);
}

Eigen::MatrixXd CofactorDifferential::mixed(
    const Eigen::MatrixXd& direction_a, const Eigen::MatrixXd& direction_b) const {
  if (regular_) {
    if (direction_a.rows() != inverse_.rows() ||
        direction_a.cols() != inverse_.cols() ||
        direction_b.rows() != inverse_.rows() ||
        direction_b.cols() != inverse_.cols()) {
      throw std::invalid_argument(
          "mixed cofactor directions have inconsistent dimensions");
    }
    const Eigen::MatrixXd ra = inverse_ * direction_a;
    const Eigen::MatrixXd rb = inverse_ * direction_b;
    const double trace_a = ra.trace();
    const double trace_b = rb.trace();
    const double mixed_trace = (ra * rb).trace();
    const Eigen::MatrixXd inverse_a = ra * inverse_;
    const Eigen::MatrixXd inverse_b = rb * inverse_;
    const Eigen::MatrixXd mixed_inverse =
        rb * inverse_a + ra * inverse_b;
    return determinant_ *
        ((trace_a * trace_b - mixed_trace) * inverse_.transpose() -
         trace_b * inverse_a.transpose() -
         trace_a * inverse_b.transpose() +
         mixed_inverse.transpose());
  }
  const Eigen::MatrixXd a = rotate(direction_a), b = rotate(direction_b);
  Eigen::MatrixXd c = Eigen::MatrixXd::Zero(a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.rows(); ++j) {
      if (i == j) continue;
      for (int k = 0; k < a.rows(); ++k) {
        if (k == i || k == j) continue;
        const double p = cached_triple_complement(i, j, k);
        c(i, j) += p * (a(j, k)*b(k, i) + b(j, k)*a(k, i)
                             - a(j, i)*b(k, k) - b(j, i)*a(k, k));
        if (j < k)
          c(i, i) += p * (a(j, j)*b(k, k) + b(j, j)*a(k, k)
                               - a(j, k)*b(k, j) - b(j, k)*a(k, j));
      }
    }
  }
  return restore(c);
}

Eigen::MatrixXd CofactorDifferential::pair_rotation(const Eigen::MatrixXd& matrix) const {
  const int n=matrix.rows(), m=n*(n-1)/2;
  Eigen::MatrixXd result(m,m);
  for (int j=1;j<n;++j) for (int i=0;i<j;++i)
    for (int l=1;l<n;++l) for (int k=0;k<l;++k)
      result(pair_index(i,j),pair_index(k,l)) =
          matrix(i,k)*matrix(j,l)-matrix(i,l)*matrix(j,k);
  return result;
}

Eigen::MatrixXd CofactorDifferential::second() const {
  if (regular_) {
    const int n = inverse_.rows();
    const int pair_count = n * (n - 1) / 2;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(pair_count, pair_count);
    for (int row_second = 1; row_second < n; ++row_second) {
      for (int row_first = 0; row_first < row_second; ++row_first) {
        const int row_pair = pair_index(row_first, row_second);
        for (int column_second = 1; column_second < n; ++column_second) {
          for (int column_first = 0;
               column_first < column_second;
               ++column_first) {
            const int column_pair = pair_index(column_first, column_second);
            result(row_pair, column_pair) = determinant_ *
                (inverse_(column_first, row_first) *
                     inverse_(column_second, row_second) -
                 inverse_(column_first, row_second) *
                     inverse_(column_second, row_first));
          }
        }
      }
    }
    return result;
  }
  return second_;
}

Eigen::MatrixXd CofactorDifferential::second_first(const Eigen::MatrixXd& direction) const {
  if (regular_) {
    if (direction.rows() != inverse_.rows() ||
        direction.cols() != inverse_.cols()) {
      throw std::invalid_argument(
          "second-cofactor direction has inconsistent dimensions");
    }
    const int n = inverse_.rows();
    const int pair_count = n * (n - 1) / 2;
    const double trace = (inverse_ * direction).trace();
    const Eigen::MatrixXd inverse_direction =
        -inverse_ * direction * inverse_;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(pair_count, pair_count);
    for (int row_second = 1; row_second < n; ++row_second) {
      for (int row_first = 0; row_first < row_second; ++row_first) {
        const int row_pair = pair_index(row_first, row_second);
        for (int column_second = 1; column_second < n; ++column_second) {
          for (int column_first = 0;
               column_first < column_second;
               ++column_first) {
            const int column_pair = pair_index(column_first, column_second);
            const double exterior =
                inverse_(column_first, row_first) *
                    inverse_(column_second, row_second) -
                inverse_(column_first, row_second) *
                    inverse_(column_second, row_first);
            const double exterior_direction =
                inverse_direction(column_first, row_first) *
                    inverse_(column_second, row_second) +
                inverse_(column_first, row_first) *
                    inverse_direction(column_second, row_second) -
                inverse_direction(column_first, row_second) *
                    inverse_(column_second, row_first) -
                inverse_(column_first, row_second) *
                    inverse_direction(column_second, row_first);
            result(row_pair, column_pair) =
                determinant_ *
                (trace * exterior + exterior_direction);
          }
        }
      }
    }
    return result;
  }
  const Eigen::MatrixXd a=rotate(direction);
  const int n=u_.rows(), m=n*(n-1)/2;
  Eigen::MatrixXd c=Eigen::MatrixXd::Zero(m,m);
  for (int k=2;k<n;++k) for (int j=1;j<k;++j) for (int i=0;i<j;++i) {
    const int index[3]={i,j,k};
    const int pairs[3]={pair_index(j,k),pair_index(i,k),pair_index(i,j)};
    const double p=cached_triple_complement(i,j,k);
    for (int r=0;r<3;++r) for (int s=0;s<3;++s)
      c(pairs[r],pairs[s]) += ((r+s)%2 ? -p:p)*a(index[r],index[s]);
  }
  return u2_*c*v2_.transpose();
}

double CofactorDifferential::second_contraction(
    const Eigen::MatrixXd& weights) const {
  if (regular_) {
    return determinant_ *
        regular_exterior_contraction(inverse_, weights).value;
  }
  if (weights.rows() != second_.rows() ||
      weights.cols() != second_.cols()) {
    throw std::invalid_argument(
        "second-cofactor weights have inconsistent dimensions");
  }
  return (weights.cwiseProduct(second_)).sum();
}

Eigen::MatrixXd CofactorDifferential::second_gradient_in_diagonal_chart(
    const Eigen::MatrixXd& weights) const {
  const int n=u_.rows(), m=n*(n-1)/2;
  if (weights.rows()!=m || weights.cols()!=m)
    throw std::invalid_argument("second-cofactor weights have inconsistent dimensions");
  Eigen::MatrixXd g=Eigen::MatrixXd::Zero(n,n);
  for (int k=2;k<n;++k) for (int j=1;j<k;++j) for (int i=0;i<j;++i) {
    const int index[3]={i,j,k};
    const int pairs[3]={pair_index(j,k),pair_index(i,k),pair_index(i,j)};
    const double p=cached_triple_complement(i,j,k);
    for (int r=0;r<3;++r) for (int s=0;s<3;++s)
      g(index[r],index[s]) += ((r+s)%2 ? -p:p)*weights(pairs[r],pairs[s]);
  }
  return g;
}

Eigen::MatrixXd CofactorDifferential::second_contraction_gradient(
    const Eigen::MatrixXd& weights) const {
  if (regular_) {
    const ExteriorContraction exterior =
        regular_exterior_contraction(inverse_, weights);
    return determinant_ *
        (exterior.value * inverse_.transpose() -
         inverse_.transpose() * exterior.gradient * inverse_.transpose());
  }
  return restore(second_gradient_in_diagonal_chart(
      u2_.transpose()*weights*v2_));
}

Eigen::MatrixXd CofactorDifferential::second_contraction_gradient_direction(
    const Eigen::MatrixXd& direction, const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& delta_weights) const {
  if (regular_) {
    if (direction.rows() != inverse_.rows() ||
        direction.cols() != inverse_.cols()) {
      throw std::invalid_argument(
          "contracted second-cofactor direction has inconsistent dimensions");
    }
    const double trace = (inverse_ * direction).trace();
    const Eigen::MatrixXd inverse_direction =
        -inverse_ * direction * inverse_;
    const ExteriorContractionDirection exterior =
        regular_exterior_contraction_direction(
            inverse_, inverse_direction, weights, delta_weights);
    const Eigen::MatrixXd base =
        exterior.base.value * inverse_.transpose() -
        inverse_.transpose() * exterior.base.gradient * inverse_.transpose();
    return determinant_ *
        (trace * base +
         exterior.direction.value * inverse_.transpose() +
         exterior.base.value * inverse_direction.transpose() -
         inverse_direction.transpose() * exterior.base.gradient * inverse_.transpose() -
         inverse_.transpose() * exterior.direction.gradient * inverse_.transpose() -
         inverse_.transpose() * exterior.base.gradient * inverse_direction.transpose());
  }
  const Eigen::MatrixXd a=rotate(direction);
  const Eigen::MatrixXd w=u2_.transpose()*weights*v2_;
  Eigen::MatrixXd g=second_gradient_in_diagonal_chart(u2_.transpose()*delta_weights*v2_);
  const int n=u_.rows();
  for (int l=3;l<n;++l) for (int k=2;k<l;++k)
    for (int j=1;j<k;++j) for (int i=0;i<j;++i) {
      const int index[4]={i,j,k,l};
      const double p=cached_quadruple_complement(i,j,k,l);
      for (int r1=0;r1<3;++r1) for (int r2=r1+1;r2<4;++r2)
        for (int c1=0;c1<3;++c1) for (int c2=c1+1;c2<4;++c2) {
          int rows[2], cols[2], nr=0, nc=0;
          for (int t=0;t<4;++t) {
            if (t!=r1 && t!=r2) rows[nr++]=index[t];
            if (t!=c1 && t!=c2) cols[nc++]=index[t];
          }
          const double factor=((r1+r2+c1+c2)%2 ? -p:p)*
              w(pair_index(index[r1],index[r2]),pair_index(index[c1],index[c2]));
          g(rows[0],cols[0]) += factor*a(rows[1],cols[1]);
          g(rows[0],cols[1]) -= factor*a(rows[1],cols[0]);
          g(rows[1],cols[0]) -= factor*a(rows[0],cols[1]);
          g(rows[1],cols[1]) += factor*a(rows[0],cols[0]);
        }
    }
  return restore(g);
}

CofactorDifferential::ExteriorContraction
CofactorDifferential::regular_exterior_contraction(
    const Eigen::MatrixXd& inverse,
    const Eigen::MatrixXd& weights) const {
  const int n = inverse.rows();
  const int pair_count = n * (n - 1) / 2;
  if (inverse.cols() != n ||
      weights.rows() != pair_count ||
      weights.cols() != pair_count) {
    throw std::invalid_argument(
        "regular exterior contraction dimensions are inconsistent");
  }
  ExteriorContraction result;
  result.gradient = Eigen::MatrixXd::Zero(n, n);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      const int row_pair = pair_index(row_first, row_second);
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0;
             column_first < column_second;
             ++column_first) {
          const int column_pair = pair_index(column_first, column_second);
          const double weight = weights(row_pair, column_pair);
          const double first = inverse(column_first, row_first);
          const double second = inverse(column_second, row_second);
          const double crossed_first = inverse(column_first, row_second);
          const double crossed_second = inverse(column_second, row_first);
          result.value +=
              weight *
              (first * second - crossed_first * crossed_second);
          result.gradient(column_first, row_first) += weight * second;
          result.gradient(column_second, row_second) += weight * first;
          result.gradient(column_first, row_second) -= weight * crossed_second;
          result.gradient(column_second, row_first) -= weight * crossed_first;
        }
      }
    }
  }
  return result;
}

CofactorDifferential::ExteriorContractionDirection
CofactorDifferential::regular_exterior_contraction_direction(
    const Eigen::MatrixXd& inverse,
    const Eigen::MatrixXd& inverse_direction,
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& delta_weights) const {
  const int n = inverse.rows();
  const int pair_count = n * (n - 1) / 2;
  if (inverse.cols() != n || inverse_direction.rows() != n ||
      inverse_direction.cols() != n || weights.rows() != pair_count ||
      weights.cols() != pair_count || delta_weights.rows() != pair_count ||
      delta_weights.cols() != pair_count) {
    throw std::invalid_argument(
        "regular exterior direction dimensions are inconsistent");
  }

  ExteriorContractionDirection result;
  result.base.gradient = Eigen::MatrixXd::Zero(n, n);
  result.direction.gradient = Eigen::MatrixXd::Zero(n, n);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      const int row_pair = pair_index(row_first, row_second);
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0;
             column_first < column_second;
             ++column_first) {
          const int column_pair = pair_index(column_first, column_second);
          const double weight = weights(row_pair, column_pair);
          const double delta_weight = delta_weights(row_pair, column_pair);
          const double first = inverse(column_first, row_first);
          const double second = inverse(column_second, row_second);
          const double crossed_first = inverse(column_first, row_second);
          const double crossed_second = inverse(column_second, row_first);
          const double delta_first =
              inverse_direction(column_first, row_first);
          const double delta_second =
              inverse_direction(column_second, row_second);
          const double delta_crossed_first =
              inverse_direction(column_first, row_second);
          const double delta_crossed_second =
              inverse_direction(column_second, row_first);
          const double exterior =
              first * second - crossed_first * crossed_second;
          const double delta_exterior =
              delta_first * second + first * delta_second -
              delta_crossed_first * crossed_second -
              crossed_first * delta_crossed_second;

          result.base.value += weight * exterior;
          result.direction.value +=
              delta_weight * exterior + weight * delta_exterior;

          result.base.gradient(column_first, row_first) += weight * second;
          result.base.gradient(column_second, row_second) += weight * first;
          result.base.gradient(column_first, row_second) -=
              weight * crossed_second;
          result.base.gradient(column_second, row_first) -=
              weight * crossed_first;

          result.direction.gradient(column_first, row_first) +=
              delta_weight * second + weight * delta_second;
          result.direction.gradient(column_second, row_second) +=
              delta_weight * first + weight * delta_first;
          result.direction.gradient(column_first, row_second) -=
              delta_weight * crossed_second + weight * delta_crossed_second;
          result.direction.gradient(column_second, row_first) -=
              delta_weight * crossed_first + weight * delta_crossed_first;
        }
      }
    }
  }
  return result;
}

const CofactorDifferential& cached_cofactor_differential(
    const SpinDeterminantPairEvaluation& pair_evaluation) {
  if (!pair_evaluation.cofactor_differential)
    throw std::runtime_error("missing accepted-point cofactor factorization");
  return *pair_evaluation.cofactor_differential;
}

std::size_t CofactorDifferential::dynamic_bytes() const {
  return sizeof(double) * static_cast<std::size_t>(
      inverse_.size() + u_.size() + v_.size() + u2_.size() + v2_.size() + value_.size() +
      second_.size() + singular_values_.size() + pair_complements_.size() +
      triple_complements_.size() + quadruple_complements_.size());
}

}  // namespace xmvb::vb
