#pragma once

#include <memory>

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

  void precondition(linalg::ConstMatrixView residual, double energy,
                    linalg::MatrixView output) const;
};

}  // namespace xmvb::cuda
