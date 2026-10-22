#pragma once

#include <memory>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::cuda {

/** Device direction of the exact raw-coefficient orbital preparation map. */
struct OrbitalDirection {
  OrbitalDirection(Context& context, int basis, int orbitals, int inactive,
                   int active);
  Matrix<double> raw, normalized, inactive_dual, inactive_density;
  Matrix<double> active_auxiliary, active_overlap, frame_transform;
  Matrix<double> orthogonal_active;
};

/**
 * Device preparation of the VB physical orbitals. The host input supplies only
 * immutable sparse-support topology; the AO overlap is a shared device view.
 * Each accepted/trial point,
 * tangent, and pullback is evaluated on the device. Raw columns retain the
 * input's fixed-width sparse-slot layout, including fixed expanded-support
 * slots. No occupied or active gauge rotation is applied here.
 */
class OrbitalGeometry {
 public:
  /** overlap is borrowed device storage and must outlive this geometry. */
  OrbitalGeometry(Context& context, const vb::OrbitalPreparationInput& input,
      linalg::ConstMatrixView overlap);
  ~OrbitalGeometry();
  OrbitalGeometry(const OrbitalGeometry&) = delete;
  OrbitalGeometry& operator=(const OrbitalGeometry&) = delete;

  /** Copy a device nbf x norb raw-slot table; prepare() must follow. */
  void set_raw(linalg::ConstMatrixView raw);
  /** Recompute the complete physical map and active Cholesky frame on GPU. */
  void prepare();
  /** Device raw-slot tangent; entries outside differentiable slots must be 0. */
  OrbitalDirection direction(linalg::ConstMatrixView raw_direction) const;

  /**
   * Pull back upstream derivatives of T_a, P_i, A=T_a^T S T_a, and
   * F=T_a R^-1, where R^T R=A. The upper-triangular R cotangent is
   * independent of F's cotangent. Any upstream view may be empty. The output
   * has the raw-slot layout; its fixed and padded slots are zero.
   * density_projection optionally supplies (gp+gp^T)*inactive_dual directly,
   * instead of gp, for consumers of accepted density-response maps.
   */
  void pullback(linalg::ConstMatrixView active_auxiliary_gradient,
                linalg::ConstMatrixView inactive_density_gradient,
                linalg::ConstMatrixView active_overlap_gradient,
                linalg::ConstMatrixView frame_transform_gradient,
                linalg::ConstMatrixView orthogonal_active_gradient,
                linalg::MatrixView raw_gradient,
                linalg::ConstMatrixView density_projection = {}) const;

  /** Exact directional derivative of pullback, including fixed-upstream
   * curvature. Upstream directions may be empty when fixed. An optional
   * density_direction_projection supplies (dgp+dgp^T)*inactive_dual instead
   * of a full dgp matrix; it does not replace gp*d(inactive_dual).
   */
  void pullback_direction(
      const OrbitalDirection& tangent,
      linalg::ConstMatrixView active_auxiliary_gradient,
      linalg::ConstMatrixView inactive_density_gradient,
      linalg::ConstMatrixView active_overlap_gradient,
      linalg::ConstMatrixView frame_transform_gradient,
      linalg::ConstMatrixView orthogonal_active_gradient,
      linalg::ConstMatrixView active_auxiliary_gradient_direction,
      linalg::ConstMatrixView inactive_density_gradient_direction,
      linalg::ConstMatrixView active_overlap_gradient_direction,
      linalg::ConstMatrixView frame_transform_gradient_direction,
      linalg::ConstMatrixView orthogonal_active_gradient_direction,
      linalg::MatrixView raw_gradient_direction,
      linalg::ConstMatrixView density_direction_projection = {}) const;

  linalg::ConstMatrixView normalized() const noexcept { return normalized_.view(); }
  linalg::ConstMatrixView raw() const noexcept { return raw_.view(); }
  linalg::ConstMatrixView ao_overlap() const noexcept { return overlap_; }
  linalg::ConstMatrixView inactive_dual() const noexcept { return dual_.view(); }
  linalg::ConstMatrixView inactive_density() const noexcept { return density_.view(); }
  linalg::ConstMatrixView active_auxiliary() const noexcept { return active_.view(); }
  linalg::ConstMatrixView active_overlap() const noexcept { return active_overlap_.view(); }
  linalg::ConstMatrixView frame_transform() const noexcept { return frame_transform_.view(); }
  linalg::ConstMatrixView orthogonal_active() const noexcept { return frame_.view(); }
  linalg::ConstMatrixView frame_inverse_transform() const noexcept { return frame_inverse_.view(); }
  int basis_size() const noexcept { return basis_; }
  int orbital_count() const noexcept { return orbitals_; }
  int inactive_count() const noexcept { return inactive_; }
  int active_count() const noexcept { return active_count_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Context* context_;
  int basis_, orbitals_, inactive_, active_count_;
  Buffer<int> support_, stored_counts_, differentiable_counts_;
  linalg::ConstMatrixView overlap_;
  Matrix<double> raw_, normalized_, overlap_normalized_;
  Matrix<double> inverse_norms_, inverse_inactive_metric_, dual_, density_;
  Matrix<double> active_, active_overlap_, frame_transform_, frame_inverse_, frame_;
  Matrix<double> inactive_active_overlap_;
  bool prepared_ = false;
};

}  // namespace xmvb::cuda
