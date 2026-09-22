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
  /** @brief Matching structure-metric image from the same H/S action. */
  Eigen::VectorXd metric_coordinates;
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

  /** @brief Recovers raw coefficient responses directly from coordinates. */
  Eigen::MatrixXd coefficient_response_coordinates(
      const Eigen::VectorXd& coordinates) const;

  /** @brief Maximum CI-amplitude component represented by a response. */
  double maximum_coefficient_component(
      const StructureTangent& tangent) const;

  /** @brief Maximum CI-amplitude component without an ambient round trip. */
  double maximum_coefficient_component_coordinates(
      const Eigen::VectorXd& coordinates) const;

  /** @brief Applies @f$P(H-E_sS)P@f$ independently to every selected state. */
  StructureTangent apply_hessian(const StructureTangent& tangent) const;

  /**
   * @brief Applies the structure Hessian directly in horizontal coordinates.
   *
   * If @f$Q_\perp@f$ is the Euclidean-orthonormal basis of
   * @f$\ker[(SC_{\rm sel})^T]@f$, column @f$s@f$ is acted on by
   * @f$Q_\perp^T(H-E_sS)Q_\perp@f$.  The action is matrix free and does not
   * solve or eliminate a structure-response equation.
   */
  Eigen::VectorXd apply_hessian_coordinates(
      const Eigen::VectorXd& coordinates) const;

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

  /**
   * @brief Applies @f$M_s=Q_\perp^TSQ_\perp@f$ in horizontal coordinates.
   *
   * The scaled tangent variable is @f$z_s=\sqrt{2w_s}\,q_s@f$.  Consequently
   * its physical squared length is
   * @f$\sum_s z_s^TSz_s=2\sum_s w_s q_s^TSq_s@f$; the state weights are thus
   * part of the coordinates rather than an additional metric factor.
   */
  Eigen::VectorXd apply_metric_coordinates(
      const Eigen::VectorXd& coordinates) const;

  /** @brief Returns @f$x^TM_sy@f$ for two horizontal coordinate vectors. */
  double metric_inner_product(
      const Eigen::VectorXd& left,
      const Eigen::VectorXd& right) const;

  /** @brief Jacobi approximation to the dual norm induced by the overlap metric. */
  double diagonal_dual_norm(const Eigen::VectorXd& covector) const;

  /**
   * @brief Applies an SPD Jacobi inverse for @f$C+\lambda M_s@f$.
   *
   * For state @f$s@f$ the ambient diagonal is
   * @f$|H_{ii}-E_sS_{ii}|+\lambda S_{ii}@f$.  Absolute curvature makes this a
   * valid SPD Krylov preconditioner even when the structure Hessian is
   * indefinite; projection on both sides preserves symmetry on the horizontal
   * space.  This is a preconditioner only, never a static response solve.
   */
  Eigen::VectorXd apply_inverse_shifted_preconditioner(
      const Eigen::VectorXd& covector,
      double shift) const;

  /**
   * @brief Applies the signed CIAH diagonal
   * @f$(\operatorname{diag}C-\omega\operatorname{diag}M_s)^{-1}@f$.
   */
  Eigen::VectorXd apply_inverse_augmented_hessian_diagonal(
      const Eigen::VectorXd& covector,
      double eigenvalue) const;

  /** @brief Returns @f$\sum_s z_s^T S z_s@f$ on the horizontal tangent. */
  double squared_norm(const StructureTangent& tangent) const;

private:
  void validate_shape(const Eigen::MatrixXd& coefficients) const;
  void project_in_place(Eigen::MatrixXd* coefficients) const;

  std::shared_ptr<const AcceptedPointContext> accepted_point_;
  const StructureAction* action_ = nullptr;
  Eigen::VectorXd energies_;
  Eigen::VectorXd coordinate_scales_;
  Eigen::VectorXd metric_diagonal_;
  Eigen::MatrixXd absolute_hessian_diagonal_;
  Eigen::MatrixXd hessian_diagonal_;
  Eigen::MatrixXd selected_;
  Eigen::MatrixXd selected_metric_inverse_;
  Eigen::HouseholderQR<Eigen::MatrixXd> constraint_qr_;
  int n_structures_ = 0;
  int n_states_ = 0;
};

}  // namespace xmvb::vb
