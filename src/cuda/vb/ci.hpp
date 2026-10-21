#pragma once

#include <memory>
#include <cstdint>
#include <limits>

#include "cuda/linalg/davidson.hpp"
#include "cuda/vb/metric.hpp"

namespace xmvb::cuda {

class Cholesky;

/** @brief Production structure-space CI model with device-only H/M and inverse.
 * FULL uses same-dimensional isometric coordinates. A nonclosed selection
 * retains its exact H/M and the existing selected/spin metric preconditioner.
 * Borrows the action and occupation factors for the lifetime of this model.
 */
class StructureCi {
 public:
  StructureCi(Context& context, const StructureAction& action,
              const SpinMetric& spin, bool complete_space);
  ~StructureCi();
  StructureCi(const StructureCi&) = delete;
  StructureCi& operator=(const StructureCi&) = delete;

  linalg::DavidsonResult solve(const linalg::DavidsonOptions& options);
  linalg::DavidsonResult solve(const Eigen::Ref<const Eigen::MatrixXd>& initial,
                              const linalg::DavidsonOptions& options);
  /** Device-resident production path. Initial and returned roots are in
   * original structure coordinates; only Ritz scalars reside on the host.
   * The model and selected metric refresh automatically after action changes.
   */
  DeviceDavidsonResult solve_device(linalg::ConstMatrixView initial,
                                    const linalg::DavidsonOptions& options);
  DeviceDavidsonResult solve_device(const linalg::DavidsonOptions& options);
  /** Rebuild current carrier-Jacobi model and exact selected metric if needed. */
  void refresh();
  bool has_exact_metric_inverse() const noexcept {
    return full_!=nullptr || selected_mode_;
  }
  /** Exact current-frame selected metric inverse when FULL or bounded selected.
   * Refreshes a changed frame before applying the inverse. ION has no exact
   * dense selected factor and deliberately rejects this entry point.
   */
  void inverse_metric(linalg::ConstMatrixView rhs,linalg::MatrixView image);
  /** Return Rayleigh energy and metric norm for a device structure vector.
   * Both outputs are device-resident scalar 1-by-1 matrices.
   */
  void energy_norm_device(linalg::ConstMatrixView state,
      linalg::MatrixView energy,linalg::MatrixView norm) const;
  bool identity_overlap() const noexcept { return full_!=nullptr; }
  const char* preconditioner_name() const noexcept;

 private:
  Context* context_;
  const StructureAction* action_;
  const SpinMetric* spin_;
  std::unique_ptr<FullMetric> full_;
  std::unique_ptr<Cholesky> selected_;
  Matrix<double> diagonal_h_, diagonal_m_;
  mutable Matrix<double> positive_, energy_;
  bool selected_mode_=false;
  std::uint64_t refreshed_version_=std::numeric_limits<std::uint64_t>::max();

  void precondition(linalg::ConstMatrixView residual, double energy,
                    linalg::MatrixView output) const;
};

}  // namespace xmvb::cuda
