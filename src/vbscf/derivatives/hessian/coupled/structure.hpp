#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/QR>

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

/** @brief Structure Hessian image and inputs for the orbital coupling adjoint. */
struct StructureCouplingAction {
  StructureTangent hessian;
  Eigen::MatrixXd coefficient_response;
  Eigen::MatrixXd adjoint_multipliers;
};

/** @brief Coordinate-space structure action and orbital-adjoint inputs. */
struct StructureCoordinateCouplingAction {
  Eigen::VectorXd hessian_coordinates;
  Eigen::MatrixXd coefficient_response;
  Eigen::MatrixXd adjoint_multipliers;
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

  /** @brief Number of independent horizontal structure coordinates. */
  int tangent_size() const noexcept;

  /** @brief Packs a horizontal tangent into independent coordinates. */
  Eigen::VectorXd coordinates(const StructureTangent& tangent) const;

  /**
   * @brief Projects ambient scaled coefficients directly into coordinates.
   *
   * This is the coordinate representation of the horizontal projection. It
   * avoids materializing the projected ambient block when only its independent
   * coordinates are needed.
   */
  Eigen::VectorXd project_coordinates(
      const Eigen::Ref<const Eigen::MatrixXd>& scaled_coefficients) const;

  /** @brief Expands independent coordinates into a horizontal tangent. */
  StructureTangent expand(const Eigen::VectorXd& coordinates) const;

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

  /**
   * @brief Applies the structure block and prepares @f$B^Tz@f$ inputs.
   *
   * One H/S block action produces both the scaled horizontal Hessian image and
   * the raw response @f$q_s=z_s/\sqrt{2w_s}@f$. The returned multiplier uses
   * the sign expected by the selected-state orbital adjoint.
   */
  StructureCouplingAction apply_coupling(
      const StructureTangent& tangent) const;

  /**
   * @brief Applies the structure block directly to horizontal coordinates.
   *
   * The input is horizontal by construction. The result remains in independent
   * coordinates, so no redundant ambient projection is performed.
   */
  StructureCoordinateCouplingAction apply_coupling_coordinates(
      const Eigen::VectorXd& coordinates) const;

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
  Eigen::MatrixXd selected_;
  Eigen::MatrixXd selected_metric_inverse_;
  Eigen::HouseholderQR<Eigen::MatrixXd> constraint_qr_;
  int n_structures_ = 0;
  int n_states_ = 0;
};

}  // namespace xmvb::vb
