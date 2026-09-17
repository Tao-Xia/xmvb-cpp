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
 * occupied-pair compound matrices. Blocks with a small dangerous singular
 * subspace use exact multilinear interpolation over well-conditioned exterior
 * nodes. The remaining blocks use the inverse-free SVD polynomial form.
 * These are local determinant-overlap blocks, not an assembled orbital
 * Hessian.
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
  bool uses_regular_form() const noexcept {
    return representation_ == Representation::Regular;
  }
  /** @brief Whether this object uses exact singular-value interpolation. */
  bool uses_interpolated_form() const noexcept {
    return representation_ == Representation::Interpolated;
  }
  /** @brief Number of interpolated dangerous singular modes. */
  int dangerous_mode_count() const noexcept { return dangerous_mode_count_; }
  std::size_t dynamic_bytes() const;

 private:
  enum class Representation {
    Regular,
    Interpolated,
    Polynomial,
  };

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
  bool initialize_interpolation(double stable_condition_limit);
  std::size_t direct_node_count() const noexcept;
  void build_direct_node(
      std::size_t node_index,
      double* weighted_determinant,
      Eigen::MatrixXd* inverse) const;
  Eigen::MatrixXd direct_first(
      double determinant,
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd direct_mixed(
      double determinant,
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& direction_a,
      const Eigen::MatrixXd& direction_b) const;
  Eigen::MatrixXd direct_second(
      double determinant,
      const Eigen::MatrixXd& inverse) const;
  Eigen::MatrixXd direct_second_first(
      double determinant,
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& direction) const;
  Eigen::MatrixXd direct_second_contraction_gradient(
      double determinant,
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& weights) const;
  Eigen::MatrixXd direct_second_contraction_gradient_direction(
      double determinant,
      const Eigen::MatrixXd& inverse,
      const Eigen::MatrixXd& direction,
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
  Representation representation_ = Representation::Polynomial;
  int dangerous_mode_count_ = 0;
  double determinant_ = 0.0;
  double interpolation_tau_ = 0.0;
  double interpolation_node_determinant_ = 0.0;
  Eigen::MatrixXd inverse_;
  Eigen::MatrixXd u_, v_, u2_, v2_, value_, second_;
  Eigen::VectorXd singular_values_;
  Eigen::VectorXd pair_complements_, triple_complements_, quadruple_complements_;
  double parity_ = 1.0;
};

const CofactorDifferential& cached_cofactor_differential(
    const SpinDeterminantPairEvaluation& pair_evaluation);

}  // namespace xmvb::vb
