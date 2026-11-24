#pragma once

#include "tnvb/communication/process.hpp"
#include "vbscf/derivatives/gradient/active_space/backend.hpp"

namespace xmvb::tnvb {

/** Single-state SU(2) electronic evaluator; common C++ pullback stays outside. */
class ElectronicEvaluator final : public vb::ElectronicStateBackend {
 public:
  std::unique_ptr<vb::ElectronicStateForwardEvaluation> evaluate_energy(
      const vb::VbScfInput&, const vb::ElectronicStateRequest&) const override;
  vb::ActiveSpaceGradientResult complete_gradient(const vb::VbScfInput&,
      std::unique_ptr<vb::ElectronicStateForwardEvaluation>) const override;
  vb::ActiveSpaceGradientResult evaluate_energy_gradient(
      const vb::VbScfInput&, const vb::ElectronicStateRequest&) const override;
  vb::ActiveSpaceGradientResult evaluate_energy_result(
      const vb::VbScfInput&, const vb::ElectronicStateRequest&) const;

 private:
  WorkerProcess process_;
  vb::ActiveSpaceGradientResult evaluate(const vb::VbScfInput&,
      const vb::ElectronicStateRequest&, bool gradient,
      std::shared_ptr<const StateDirectory>* retain = nullptr) const;
};

}  // namespace xmvb::tnvb
