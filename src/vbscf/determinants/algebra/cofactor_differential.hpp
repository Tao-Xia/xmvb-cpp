#pragma once

#include <Eigen/Core>
#include <cstddef>

namespace xmvb::vb {

struct DeterminantOverlapResult;
struct SpinDeterminantPairEvaluation;

/**
 * @brief Exact determinant-cofactor values and directional derivatives.
 *
 * Numerically regular overlap blocks use Jacobi identities without storing
 * occupied-pair compound matrices. Ill-conditioned and rank-deficient blocks
 * use an inverse-free SVD polynomial representation. These are local
 * determinant-overlap blocks, not an assembled orbital Hessian.
 */
class CofactorDifferential {
 public:
  explicit CofactorDifferential(const DeterminantOverlapResult& overlap);
  const Eigen::MatrixXd& value() const;
  Eigen::MatrixXd first(const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd mixed(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const;
  /**
   * @brief Materialize the second cofactor matrix.
   *
   * Pair indices are `j * (j - 1) / 2 + i`, with `i < j`.
   */
  Eigen::MatrixXd second() const;
  Eigen::MatrixXd second_first(const Eigen::MatrixXd& direction) const;
  /** @brief Contract the second cofactor without materializing it. */
  double second_contraction(const Eigen::MatrixXd& weights) const;
  Eigen::MatrixXd second_contraction_gradient(const Eigen::MatrixXd& weights) const;
  Eigen::MatrixXd second_contraction_gradient_direction(
      const Eigen::MatrixXd& direction, const Eigen::MatrixXd& weights,
      const Eigen::MatrixXd& delta_weights) const;
  /** @brief Whether this object uses the regular inverse/exterior form. */
  bool uses_regular_form() const noexcept { return regular_; }
  std::size_t dynamic_bytes() const;

 private:
  struct ExteriorContraction {
    double value = 0.0;
    Eigen::MatrixXd gradient;
  };

  struct ExteriorContractionDirection {
    ExteriorContraction base;
    ExteriorContraction direction;
  };

  ExteriorContraction regular_exterior_contraction(
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& weights) const;
  ExteriorContractionDirection regular_exterior_contraction_direction(
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& inverse_direction,
      const Eigen::MatrixXd& weights,
      const Eigen::MatrixXd& delta_weights) const;
  double complement_product(int i, int j = -1, int k = -1, int l = -1) const;
  double cached_pair_complement(int i, int j) const;
  double cached_triple_complement(int i, int j, int k) const;
  double cached_quadruple_complement(int i, int j, int k, int l) const;
  Eigen::MatrixXd pair_rotation(const Eigen::MatrixXd& matrix) const;
  Eigen::MatrixXd second_gradient_in_diagonal_chart(const Eigen::MatrixXd& weights) const;
  Eigen::MatrixXd rotate(const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd restore(const Eigen::MatrixXd& cofactor) const;
  bool regular_ = false;
  double determinant_ = 0.0;
  Eigen::MatrixXd inverse_;
  Eigen::MatrixXd u_, v_, u2_, v2_, value_, second_;
  Eigen::VectorXd singular_values_;
  Eigen::VectorXd pair_complements_, triple_complements_, quadruple_complements_;
  double parity_ = 1.0;
};

const CofactorDifferential& cached_cofactor_differential(
    const SpinDeterminantPairEvaluation& pair_evaluation);

}  // namespace xmvb::vb
