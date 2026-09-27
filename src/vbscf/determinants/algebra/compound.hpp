#pragma once

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace xmvb::vb {

/**
 * @brief Lexicographic subset indices shared by compound-matrix levels.
 *
 * Level @p q contains every increasing q-subset of `[0, size)`.  Removal
 * indices connect each q-subset to its q-1 subsets without hash lookups in
 * the numerical kernels.
 */
class CompoundBasis {
 public:
  CompoundBasis(int size, int max_order);

  int size() const noexcept { return size_; }
  int max_order() const noexcept { return max_order_; }
  Eigen::Index level_size(int order) const;
  const std::vector<int>& subset(int order, Eigen::Index index) const;
  Eigen::Index removed_index(
      int order, Eigen::Index index, int position) const;

 private:
  int size_ = 0;
  int max_order_ = 0;
  std::vector<std::vector<std::vector<int>>> subsets_;
  std::vector<Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>>
      removed_;
};

/**
 * @brief Compound matrices through a requested exterior order.
 *
 * For a square matrix `A`, level q stores all q-by-q minors
 * `C_q(A)[I,J] = det(A[I,J])`.  The hierarchy supports exact rank-one
 * updates, their directional derivatives, and reverse contractions.  It is
 * local to one determinant pair; callers should consume it inside a tile and
 * must not retain one hierarchy per unique-string pair.
 */
class CompoundHierarchy {
 public:
  CompoundHierarchy(int size, int max_order);

  /** @brief Rebuild all levels from an anchor matrix. */
  void assign(const Eigen::Ref<const Eigen::MatrixXd>& matrix);

  /** @brief Differentiate every level along @p direction. */
  CompoundHierarchy directional(
      const Eigen::Ref<const Eigen::MatrixXd>& direction) const;

  /** @brief Apply `A <- A + left * right.transpose()` exactly. */
  void apply_rank_one(
      const Eigen::Ref<const Eigen::VectorXd>& left,
      const Eigen::Ref<const Eigen::VectorXd>& right);

  /**
   * @brief Update accepted and directional compound levels together.
   *
   * The represented tangent changes as
   * `dA <- dA + dleft*right^T + left*dright^T`.
   */
  void apply_rank_one(
      const Eigen::Ref<const Eigen::VectorXd>& left,
      const Eigen::Ref<const Eigen::VectorXd>& right,
      const Eigen::Ref<const Eigen::VectorXd>& dleft,
      const Eigen::Ref<const Eigen::VectorXd>& dright,
      CompoundHierarchy& tangent);

  /** @brief Pull `<weights,C_q(A)>` back to the entries of `A`. */
  Eigen::MatrixXd pullback(
      int order, const Eigen::Ref<const Eigen::MatrixXd>& weights) const;

  const CompoundBasis& basis() const noexcept { return *basis_; }
  const Eigen::MatrixXd& level(int order) const;

 private:
  explicit CompoundHierarchy(std::shared_ptr<const CompoundBasis> basis);
  void validate_vector(const Eigen::Ref<const Eigen::VectorXd>& vector) const;
  void validate_compatible(const CompoundHierarchy& other) const;

  std::shared_ptr<const CompoundBasis> basis_;
  std::vector<Eigen::MatrixXd> levels_;
};

}  // namespace xmvb::vb
