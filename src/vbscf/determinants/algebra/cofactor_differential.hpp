#pragma once

#include <Eigen/Core>
#include <cstddef>

namespace xmvb::vb {

struct SpinDeterminantPairEvaluation;

// Derivatives of the polynomial C(X) = d det(X) / d X.  Orthogonal
// transformations are fixed at the accepted X; no inverse, division by a
// singular value, rank threshold, or derivative of singular vectors is used.
// These are small determinant-overlap blocks, not the orbital Hessian.
class CofactorDifferential {
 public:
  explicit CofactorDifferential(const Eigen::MatrixXd& matrix);
  const Eigen::MatrixXd& value() const;
  Eigen::MatrixXd first(const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd mixed(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const;
  // Pair indices are j*(j-1)/2+i, i<j. These are determinant cofactors
  // contracted with the antisymmetrized two-electron integrals.
  const Eigen::MatrixXd& second() const;
  Eigen::MatrixXd second_first(const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd second_contraction_gradient(const Eigen::MatrixXd& weights) const;
  Eigen::MatrixXd second_contraction_gradient_direction(
      const Eigen::MatrixXd& direction, const Eigen::MatrixXd& weights,
      const Eigen::MatrixXd& delta_weights) const;
  std::size_t dynamic_bytes() const;

 private:
  double complement_product(int i, int j = -1, int k = -1, int l = -1) const;
  double cached_pair_complement(int i, int j) const;
  double cached_triple_complement(int i, int j, int k) const;
  double cached_quadruple_complement(int i, int j, int k, int l) const;
  Eigen::MatrixXd pair_rotation(const Eigen::MatrixXd& matrix) const;
  Eigen::MatrixXd second_gradient_in_diagonal_chart(const Eigen::MatrixXd& weights) const;
  Eigen::MatrixXd rotate(const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd restore(const Eigen::MatrixXd& cofactor) const;
  Eigen::MatrixXd u_, v_, u2_, v2_, value_, second_;
  Eigen::VectorXd singular_values_;
  Eigen::VectorXd pair_complements_, triple_complements_, quadruple_complements_;
  double parity_ = 1.0;
};

const CofactorDifferential& cached_cofactor_differential(
    const SpinDeterminantPairEvaluation& pair_evaluation);

}  // namespace xmvb::vb
