#pragma once

#include <functional>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** CPU structure AH inverse uses distinct constraints for the two models. */
enum class CiInverseConstraint { Diagonal, Spin, Carrier };

/** Device actions in the original selected structure space. */
struct SingleStateCiActions {
  std::function<void(linalg::ConstMatrixView, linalg::MatrixView,
                     linalg::MatrixView)> hamiltonian_metric;
  std::function<void(linalg::ConstMatrixView, linalg::MatrixView)> metric;
  /** Current-frame unconstrained AH inverse in original structure coordinates. */
  std::function<void(linalg::ConstMatrixView, double,
                     linalg::MatrixView)> augmented_inverse;
  /** Direct-CI clipping uses the orthogonal carrier, not raw structures. */
  std::function<double(linalg::ConstMatrixView)> maximum_component;
  /** Carrier: P_L R P_R, Spin: positive-model constrained Schur inverse. */
  CiInverseConstraint inverse_constraint=CiInverseConstraint::Spin;
};

/**
 * Single selected-state horizontal CI coordinates for an exact selected-space
 * H/S action. The reflector is formed from S c and never stores a dense basis.
 * Selected structures need not span a closed carrier space. All vectors stay
 * on the device; only scalar normalization and norms cross to the host.
 * The borrowed actions and Context must outlive this object.
 */
class SingleStateCiTangent {
public:
  /** The accepted images are borrowed and must outlive the tangent. */
  SingleStateCiTangent(Context& context, int structures,
                       linalg::ConstMatrixView selected_state,
                       linalg::ConstMatrixView selected_hamiltonian_image,
                       linalg::ConstMatrixView selected_metric_image,
                       double selected_energy, double normalized_weight,
                       SingleStateCiActions actions);
  SingleStateCiTangent(const SingleStateCiTangent&) = delete;
  SingleStateCiTangent& operator=(const SingleStateCiTangent&) = delete;

  int ambient_size() const noexcept { return structures_; }
  int tangent_size() const noexcept { return structures_ - 1; }
  double coordinate_scale() const noexcept { return coordinate_scale_; }

  /** Q_perp^T x, where Q_perp spans ker((S c)^T). */
  void project_coordinates(linalg::ConstMatrixView ambient,
                           linalg::MatrixView coordinates) const;
  /** Q_perp z, an ambient scaled horizontal CI response. */
  void expand_coordinates(linalg::ConstMatrixView coordinates,
                          linalg::MatrixView ambient) const;
  /** Q_perp z / sqrt(2w), the raw coefficient response. */
  void coefficient_response_coordinates(linalg::ConstMatrixView coordinates,
                                        linalg::MatrixView response) const;
  double maximum_coefficient_component_coordinates(
      linalg::ConstMatrixView coordinates) const;

  /** Q_perp^T S Q_perp z. */
  void apply_metric_coordinates(linalg::ConstMatrixView coordinates,
                                linalg::MatrixView metric_image) const;
  /** One H/S action gives Q_perp^T(H-E S)Q_perp z and its metric image. */
  void apply_images_coordinates(linalg::ConstMatrixView coordinates,
                                linalg::MatrixView hessian_image,
                                linalg::MatrixView metric_image) const;
  /** sqrt(2w) Q_perp^T(H-E S)c from accepted Hc/Mc, including finite residual. */
  void origin_gradient(linalg::MatrixView gradient) const;
  /** CPU single-state constrained inverse: Carrier uses P_L R P_R;
   * Spin uses the positive-model one-state Schur complement.
   */
  void apply_inverse_augmented_hessian_diagonal(
      linalg::ConstMatrixView covector,double eigenvalue,
      linalg::MatrixView direction) const;

private:
  void check_ambient(linalg::ConstMatrixView view) const;
  void check_ambient(linalg::MatrixView view) const;
  void check_coordinates(linalg::ConstMatrixView view) const;
  void check_coordinates(linalg::MatrixView view) const;

  Context* context_;
  int structures_;
  double selected_energy_, coordinate_scale_, reflector_factor_, reflected_first_;
  SingleStateCiActions actions_;
  linalg::ConstMatrixView selected_hamiltonian_image_,selected_metric_image_;
  Matrix<double> selected_, reflector_;
  mutable Matrix<double> ambient_, hessian_, metric_, work_;
};

}  // namespace xmvb::cuda
