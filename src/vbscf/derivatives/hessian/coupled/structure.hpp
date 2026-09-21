#pragma once

#include <memory>

#include <Eigen/Core>

namespace xmvb::vb {

struct AcceptedPointContext;
class StructureAction;

/**
 * @brief Horizontal selected-state structure displacement.
 *
 * Column @f$s@f$ stores the physical coordinate
 * @f$z_s=\sqrt{2w_s}\,q_s@f$, where @f$q_s@f$ is the corresponding
 * coefficient response. Every column is orthogonal to the complete selected
 * metric-image span, so selected-state normalization and rotations are not
 * represented as optimization degrees of freedom.
 */
struct StructureTangent {
  Eigen::MatrixXd scaled_coefficients;
};

/**
 * @brief Matrix-free structure-space block of the coupled VBSCF Hessian.
 *
 * The operator acts at one accepted VBSCF point. It supports a single state or
 * one equally weighted selected-state cluster. The structure Hessian action is
 * @f$P(H-E_sS)Pz_s@f$ and the tangent metric is @f$PSP@f$, where @f$P@f$
 * removes the span of @f$SC_{\mathrm{sel}}@f$. No structure-space matrix is
 * assembled.
 */
class StructureTangentOperator {
public:
  /**
   * @brief Constructs the immutable accepted-point structure operator.
   *
   * The borrowed action must outlive this object. Selected states must be
   * S-orthonormal. Unequally weighted multistate objectives are rejected because
   * their internal state rotations are physical directions.
   */
  explicit StructureTangentOperator(
      std::shared_ptr<const AcceptedPointContext> accepted_point,
      const StructureAction& action);

  int n_structures() const noexcept;
  int n_states() const noexcept;

  /** @brief Projects scaled coefficient columns onto the horizontal tangent. */
  StructureTangent project(
      const Eigen::Ref<const Eigen::MatrixXd>& scaled_coefficients) const;

  /**
   * @brief Converts raw coefficient responses to scaled horizontal coordinates.
   */
  StructureTangent from_coefficient_response(
      const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response) const;

  /** @brief Recovers raw coefficient responses from scaled coordinates. */
  Eigen::MatrixXd coefficient_response(
      const StructureTangent& tangent) const;

  /** @brief Applies @f$P(H-E_sS)P@f$ independently to every selected state. */
  StructureTangent apply_hessian(const StructureTangent& tangent) const;

  /** @brief Applies the structure trust-region metric @f$PSP@f$. */
  StructureTangent apply_metric(const StructureTangent& tangent) const;

  /** @brief Returns @f$\sum_s z_s^T S z_s@f$ on the horizontal tangent. */
  double squared_norm(const StructureTangent& tangent) const;

private:
  void validate_shape(const Eigen::MatrixXd& coefficients) const;
  void project_in_place(Eigen::MatrixXd* coefficients) const;

  std::shared_ptr<const AcceptedPointContext> accepted_point_;
  const StructureAction* action_ = nullptr;
  Eigen::VectorXd energies_;
  Eigen::VectorXd coordinate_scales_;
  Eigen::MatrixXd constraint_units_;
  int n_structures_ = 0;
  int n_states_ = 0;
};

}  // namespace xmvb::vb
