#pragma once

#include <functional>
#include <string>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Matrix-free inverse update for an eliminated structure response.
 *
 * Given a base shifted-orbital inverse @f$P_\lambda@f$, orbital--structure
 * coupling @f$J@f$, and structure block @f$K@f$, this class applies
 *
 * @f[
 * P_\lambda + P_\lambda J
 * (K-J^T P_\lambda J)^\dagger J^T P_\lambda.
 * @f]
 *
 * Only the small symmetric Woodbury matrix is diagonalized.  Numerical
 * validation failures make the object explicitly unavailable; application
 * never silently falls back to the base inverse.
 */
class StructureResponseWoodburyPreconditioner {
public:
  using BlockInverseAction =
      std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

  StructureResponseWoodburyPreconditioner(
      Eigen::MatrixXd coupling,
      Eigen::MatrixXd structure_block,
      BlockInverseAction base_inverse);

  int dimension() const noexcept;
  int response_dimension() const noexcept;
  int retained_rank() const noexcept;
  double spectral_cutoff() const noexcept;
  double pseudoinverse_residual() const noexcept;
  bool available() const noexcept;
  const std::string& unavailability_reason() const noexcept;

  /** @brief Applies the certified Woodbury inverse to one covector. */
  Eigen::VectorXd apply(const Eigen::VectorXd& covector) const;

  /** @brief Applies the certified Woodbury inverse to a covector block. */
  Eigen::MatrixXd apply_block(const Eigen::MatrixXd& covectors) const;

private:
  void make_unavailable(std::string reason);
  Eigen::MatrixXd checked_base_inverse(
      const Eigen::MatrixXd& covectors) const;
  void require_available() const;

  Eigen::MatrixXd coupling_;
  BlockInverseAction base_inverse_;
  Eigen::MatrixXd base_coupling_;
  Eigen::MatrixXd small_matrix_;
  Eigen::MatrixXd small_pseudoinverse_;
  Eigen::MatrixXd small_range_projector_;
  int retained_rank_ = 0;
  double spectral_cutoff_ = 0.0;
  double pseudoinverse_residual_ = 0.0;
  bool available_ = false;
  std::string unavailability_reason_;
};

}  // namespace xmvb::vb
