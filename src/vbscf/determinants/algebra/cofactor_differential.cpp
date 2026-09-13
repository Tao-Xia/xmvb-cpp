#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"

#include <algorithm>
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
}

CofactorDifferential::CofactorDifferential(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols())
    throw std::invalid_argument("cofactor differential requires a square matrix");
  if (matrix.rows() == 0) {
    u_.resize(0, 0); v_.resize(0, 0); singular_values_.resize(0);
    u2_.resize(0, 0); v2_.resize(0, 0);
    value_.resize(0, 0); second_.resize(0, 0);
    pair_complements_.resize(0);
    triple_complements_.resize(0);
    quadruple_complements_.resize(0);
    return;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (svd.info() != Eigen::Success)
    throw std::runtime_error("cofactor differential SVD failed");
  u_ = svd.matrixU(); v_ = svd.matrixV(); singular_values_ = svd.singularValues();
  parity_ = u_.determinant() * v_.determinant() < 0.0 ? -1.0 : 1.0;
  u2_ = pair_rotation(u_);
  v2_ = pair_rotation(v_);
  value_ = Eigen::MatrixXd::Zero(u_.rows(), u_.rows());
  for (int i = 0; i < value_.rows(); ++i) value_(i, i) = complement_product(i);
  value_ = restore(value_);
  const int n = u_.rows();
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
  if (direction.rows() != u_.rows() || direction.cols() != v_.rows())
    throw std::invalid_argument("cofactor direction has inconsistent dimensions");
  return u_.transpose() * direction * v_;
}

Eigen::MatrixXd CofactorDifferential::restore(const Eigen::MatrixXd& cofactor) const {
  return u_ * cofactor * v_.transpose();
}

const Eigen::MatrixXd& CofactorDifferential::value() const {
  return value_;
}

Eigen::MatrixXd CofactorDifferential::first(const Eigen::MatrixXd& direction) const {
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

const Eigen::MatrixXd& CofactorDifferential::second() const {
  return second_;
}

Eigen::MatrixXd CofactorDifferential::second_first(const Eigen::MatrixXd& direction) const {
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
  return restore(second_gradient_in_diagonal_chart(
      u2_.transpose()*weights*v2_));
}

Eigen::MatrixXd CofactorDifferential::second_contraction_gradient_direction(
    const Eigen::MatrixXd& direction, const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& delta_weights) const {
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

const CofactorDifferential& cached_cofactor_differential(
    const SpinDeterminantPairEvaluation& pair_evaluation) {
  if (!pair_evaluation.cofactor_differential)
    throw std::runtime_error("missing accepted-point cofactor factorization");
  return *pair_evaluation.cofactor_differential;
}

std::size_t CofactorDifferential::dynamic_bytes() const {
  return sizeof(double) * static_cast<std::size_t>(
      u_.size() + v_.size() + u2_.size() + v2_.size() + value_.size() +
      second_.size() + singular_values_.size() + pair_complements_.size() +
      triple_complements_.size() + quadruple_complements_.size());
}

}  // namespace xmvb::vb
