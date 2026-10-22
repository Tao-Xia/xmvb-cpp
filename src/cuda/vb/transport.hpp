#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "cuda/memory/buffer.hpp"
#include "linalg/metric_solve.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

class StructureAction;
class SelectedMetric;
class ExteriorTransform;

/** Exact F(R_left)^T F(R_right) in the original selected structure space.
 * F(R)=wedge(R) W; W is the sparse selected expansion in StructureAction.
 * No closure of the selected structures is assumed. Frame updates and all
 * vector arithmetic stay on Context's stream. Cross-frame calls leave the
 * borrowed action at the right/current frame; physical callers set their frame.
 */
class SelectedCrossOverlap {
 public:
  using ExactMetricInverse = std::function<void(
      linalg::ConstMatrixView, linalg::MatrixView)>;
  using MetricPreconditioner = ExactMetricInverse;
  SelectedCrossOverlap(Context& context, StructureAction& action,
                       int active_orbitals,
                       ExactMetricInverse exact_inverse = {},
                       MetricPreconditioner preconditioner = {});
  SelectedCrossOverlap(const SelectedCrossOverlap&) = delete;
  SelectedCrossOverlap& operator=(const SelectedCrossOverlap&) = delete;

  int size() const noexcept { return structures_; }
  int active_orbitals() const noexcept { return active_; }
  /** Refreshed before each matrix-free solve; valid until the next solve. */
  linalg::ConstMatrixView current_metric_diagonal() const noexcept {
    return {diagonal_.data(),structures_,1,structures_};
  }
  void set_preconditioner(MetricPreconditioner preconditioner) {
    preconditioner_=std::move(preconditioner);
  }

  void apply(linalg::ConstMatrixView left_frame,
             linalg::ConstMatrixView right_frame,
             linalg::ConstMatrixView coefficients,
             linalg::MatrixView image) const;

  /** Direction of the two frames, with coefficients held fixed. */
  void apply_direction(linalg::ConstMatrixView left_frame,
                       linalg::ConstMatrixView right_frame,
                       linalg::ConstMatrixView left_direction,
                       linalg::ConstMatrixView right_direction,
                       linalg::ConstMatrixView coefficients,
                       linalg::MatrixView image) const;

  /** Exact upper-frame cotangents of left^T F(L)^T F(R) right. */
  void adjoint(linalg::ConstMatrixView left_frame,
               linalg::ConstMatrixView right_frame,
               linalg::ConstMatrixView left_coefficients,
               linalg::ConstMatrixView right_coefficients,
               linalg::MatrixView left_gradient,
               linalg::MatrixView right_gradient) const;

  /** F(current)^T [F(origin) base + dF(origin)[direction] reference]. */
  void transport_image(linalg::ConstMatrixView current,linalg::ConstMatrixView origin,
      linalg::ConstMatrixView direction,linalg::ConstMatrixView base,
      linalg::ConstMatrixView reference,linalg::MatrixView image) const;

  /** Contracted finite-transport pullback and F(origin)^T F(current) lambda.
   * reference and transported are selected-structure coefficients.
   */
  void transport_gradient(linalg::ConstMatrixView current,linalg::ConstMatrixView origin,
      linalg::ConstMatrixView direction,linalg::ConstMatrixView lambda,
      linalg::ConstMatrixView base,linalg::ConstMatrixView reference,
      linalg::ConstMatrixView transported,linalg::MatrixView current_gradient,
      linalg::MatrixView origin_gradient,linalg::MatrixView image) const;

  /** Matrix-free selected Gram action, never H sigma. */
  void apply_metric(linalg::ConstMatrixView frame,
                    linalg::ConstMatrixView coefficients,
                    linalg::MatrixView image) const;

  /** Selected SPD inverse with the CPU metric solver's certified true-residual
   * and stagnation rules. A supplied immutable point metric uses its factor and
   * Gram for the certified solve without changing the shared action frame.
   * Otherwise the existing matrix-free inverse is used unchanged. No host
   * vector transfers occur; only cuBLAS scalars and diagnostics cross to host.
   */
  void solve_metric(linalg::ConstMatrixView frame,
                    linalg::ConstMatrixView rhs, linalg::MatrixView solution,
                    linalg::ConstMatrixView initial = {},
                    linalg::MetricSolveStats* stats = nullptr,
                    const SelectedMetric* metric = nullptr) const;

 private:
  void check_frame(linalg::ConstMatrixView frame) const;
  void check_structure(linalg::ConstMatrixView vector) const;
  void check_structure(linalg::MatrixView vector) const;
  void check_gradient(linalg::MatrixView matrix) const;
  void pullback(linalg::ConstMatrixView frame,
                linalg::ConstMatrixView moment,
                linalg::MatrixView gradient) const;
  bool use_product_overlap(int images,bool moments=false) const;
  void physical_carrier(linalg::ConstMatrixView coefficients,
      const ExteriorTransform& alpha,const ExteriorTransform& beta,
      linalg::MatrixView carrier) const;
  void load_sources(int begin,int count,linalg::ConstMatrixView coefficients,
                   linalg::MatrixView alpha,linalg::MatrixView beta) const;
  void contract_source(linalg::ConstMatrixView alpha,linalg::ConstMatrixView beta,
                       linalg::MatrixView image) const;
  void seeds(const ExteriorTransform& alpha_map,const ExteriorTransform& beta_map,
             linalg::ConstMatrixView coefficients,
             linalg::ConstMatrixView alpha,linalg::ConstMatrixView beta,
             linalg::MatrixView alpha_seed,linalg::MatrixView beta_seed,
             linalg::MatrixView alpha_work,linalg::MatrixView beta_work) const;
  void accumulate_moment(const ExteriorTransform& transform,
      linalg::ConstMatrixView seed,linalg::ConstMatrixView image,double scale,
      linalg::MatrixView moment) const;
  int source_extent() const;

  Context* context_;
  StructureAction* action_;
  int structures_, alpha_, beta_, active_, groups_;
  std::vector<int> source_columns_;
  Buffer<int> source_groups_,source_columns_device_,source_offsets_device_,
      alpha_offsets_device_;
  mutable Buffer<double> diagonal_;
  mutable Matrix<double> diagonal_h_scratch_;
  ExactMetricInverse exact_inverse_;
  MetricPreconditioner preconditioner_;
};

/** One immutable selected-space retraction at a trial frame. */
struct SingleStateCiTransportPoint {
  Matrix<double> frame;
  Matrix<double> origin_direction;
  Matrix<double> base_coefficients;
  Matrix<double> coefficients;
  /** Immutable exact bounded Gram and factor at this trial frame. */
  std::shared_ptr<const SelectedMetric> selected_metric;
};

/** Residual-balanced physical model outputs, all non-scalar cotangents on GPU.
 * The caller adds its physical raw-orbital gradient, pulls current_upper and
 * origin_upper through the corresponding OrbitalGeometry maps, and projects
 * structure_covector through its frozen horizontal CI tangent.
 */
struct SingleStateCiTransportModel {
  double energy = 0.0;
  Matrix<double> current_upper;
  Matrix<double> origin_upper;
  Matrix<double> structure_covector;
};

/** Single-state version of the finite selected-space CIAH CI transport.
 * The accepted frame and selected eigenvector are copied once to the device;
 * Hc0 and Mc0 must be the exact accepted selected-action images, including
 * any finite Ritz residual. The optional physical coefficient gradient is the
 * authoritative frozen residual when supplied; it must use the same weight
 * as evaluate_model. The accepted source is retained in selected structure
 * coordinates. No stationarity cutoff is imposed.
 */
class SingleStateCiTransport {
 public:
  SingleStateCiTransport(Context& context, SelectedCrossOverlap& overlap,
      linalg::ConstMatrixView origin_frame,
      linalg::ConstMatrixView selected_state,
      linalg::ConstMatrixView hamiltonian_image,
      linalg::ConstMatrixView metric_image,
      double normalized_weight,
      linalg::ConstMatrixView physical_origin_coefficient_gradient = {});

  SingleStateCiTransportPoint retract(
      linalg::ConstMatrixView trial_frame,
      linalg::ConstMatrixView origin_frame_direction,
      linalg::ConstMatrixView base_coefficients,
      std::shared_ptr<const SelectedMetric> selected_metric = {}) const;

  /** Physical coefficient gradient must include normalized_weight. */
  SingleStateCiTransportModel evaluate_model(
      const SingleStateCiTransportPoint& point, double physical_energy,
      linalg::ConstMatrixView physical_coefficient_gradient) const;

  linalg::ConstMatrixView origin_coefficient_gradient() const noexcept {
    return origin_gradient_.view();
  }
  double coordinate_scale() const noexcept { return coordinate_scale_; }

 private:
  Context* context_;
  SelectedCrossOverlap* overlap_;
  int structures_, active_;
  double coordinate_scale_;
  Matrix<double> origin_frame_,origin_gradient_,origin_coefficients_;
};

}  // namespace xmvb::cuda
