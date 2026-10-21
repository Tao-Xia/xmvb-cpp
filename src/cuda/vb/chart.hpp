#pragma once

#include <vector>
#include <cstdint>
#include <memory>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::cuda { class OrbitalGeometry; }

namespace xmvb::cuda {

/** One exact accepted-point OrbitalChart tangent block.
 * basis is column-major [raw_slots.size() x reduced_size], using the
 * unwhitened tangent_basis already selected by the CPU chart. raw_slots are
 * zero-based addresses in the fixed-width nbf x norb raw coefficient table.
 * This is a chart export, not a request to reconstruct a quotient on GPU.
 */
struct OrbitalTangentBlock {
  std::vector<int> raw_slots;
  int reduced_offset = 0;
  int reduced_size = 0;
  std::vector<double> basis;
};

/** GPU application of a frozen accepted-point raw orbital chart.
 * B maps reduced CIAH vectors to raw sparse slots. The exact adjoint B^T
 * projects raw coefficient covectors. Retraction is x+scale*B step, with no
 * normalization or gauge repair: OrbitalGeometry handles normalization in
 * its physical forward map. Padded slots beyond stored_counts are cleared,
 * matching the CPU chart retraction. Rebuild this map whenever the accepted chart
 * changes. All numerical map applications remain on the device.
 */
class OrbitalChartMap {
 public:
  OrbitalChartMap(Context& context,int basis,int orbitals,int reduced_size,
                  const std::vector<int>& stored_counts,
                  const std::vector<OrbitalTangentBlock>& blocks);
  OrbitalChartMap(const OrbitalChartMap&)=delete;
  OrbitalChartMap& operator=(const OrbitalChartMap&)=delete;

  int reduced_size() const noexcept { return reduced_size_; }
  int raw_size() const noexcept { return basis_*orbitals_; }

  /** raw_direction receives B*reduced (nbf x norb); fixed slots are zero. */
  void expand(linalg::ConstMatrixView reduced,
              linalg::MatrixView raw_direction) const;
  /** reduced_covector receives B^T*raw_covector (nreduced x 1). */
  void project(linalg::ConstMatrixView raw_covector,
               linalg::MatrixView reduced_covector) const;
  /** trial receives accepted+scale*B*step, with original fixed slots retained. */
  void retract(linalg::ConstMatrixView accepted,
               linalg::ConstMatrixView reduced_step,double scale,
               linalg::MatrixView trial) const;

 private:
  Context* context_;
  int basis_,orbitals_,reduced_size_,mapped_rows_;
  Buffer<int> raw_slots_,row_block_,row_local_;
  Buffer<int> block_first_row_,block_row_count_,block_first_basis_;
  Buffer<int> block_reduced_offset_,block_reduced_size_;
  Buffer<int> coordinate_block_,coordinate_local_;
  Buffer<int> stored_counts_;
  mutable Buffer<int> retraction_info_;
  Buffer<double> basis_values_;
  mutable Matrix<double> scratch_;
};

/** Accepted-point unwhitened quotient chart, constructed numerically on GPU.
 * The host input contributes immutable sparse topology only. rebuild() must
 * follow OrbitalGeometry::prepare() at each accepted point. The local nullspace
 * and metric bases never leave the device; only rank-determining spectra do.
 */
class AcceptedOrbitalChart {
 public:
  AcceptedOrbitalChart(Context& context,
      const vb::OrbitalPreparationInput& input,bool complete_active_space);
  ~AcceptedOrbitalChart();
  AcceptedOrbitalChart(const AcceptedOrbitalChart&)=delete;
  AcceptedOrbitalChart& operator=(const AcceptedOrbitalChart&)=delete;

  void rebuild(const OrbitalGeometry& geometry);
  int reduced_size() const noexcept;
  int raw_size() const noexcept;
  std::uint64_t rank_signature() const noexcept;
  /** Frozen accepted AO-metric orthonormal MO frame [core,F,virtual]. */
  linalg::ConstMatrixView accepted_mo_frame() const noexcept;
  void expand(linalg::ConstMatrixView reduced,linalg::MatrixView raw) const;
  void project(linalg::ConstMatrixView raw_covector,
               linalg::MatrixView reduced_covector) const;
  /** AO-metric projection of a raw tangent vector (CPU project_vector). */
  void project_vector(linalg::ConstMatrixView raw_direction,
                      linalg::MatrixView reduced_vector) const;
  void retract(linalg::ConstMatrixView accepted,
               linalg::ConstMatrixView reduced_step,double scale,
               linalg::MatrixView trial) const;
  void inverse_metric(linalg::ConstMatrixView covector,
                      linalg::MatrixView vector) const;
  void metric(linalg::ConstMatrixView vector,
              linalg::MatrixView covector) const;
  /** Coupled selected-ray or complete-span physical metric for AH actions.
   * This is distinct from the local coordinate metric above. */
  void physical_metric(linalg::ConstMatrixView vector,
                       linalg::MatrixView covector) const;
  double coordinate_dual_norm(linalg::ConstMatrixView covector) const;
  double reference_gradient_inf_norm(linalg::ConstMatrixView covector) const;
  /** Physical external-frame rotation maximum in AO-metric units. The
   * device-built virtual frame may differ by an orthogonal rotation from
   * the CPU auxiliary frame. */
  double max_component(linalg::ConstMatrixView vector) const;
  /** CPU OrbitalChart analytic AH diagonal model in the frozen device frame. */
  void set_hessian_diagonal(const OrbitalGeometry& geometry,
                            linalg::ConstMatrixView rotation_diagonal);
  void apply_inverse_hessian_diagonal(linalg::ConstMatrixView covector,
                                      double eigenvalue,
                                      linalg::MatrixView direction) const;
  linalg::ConstMatrixView hessian_diagonal() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xmvb::cuda
