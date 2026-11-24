#pragma once

#include "vbscf/derivatives/gradient/active_space/backend.hpp"
#include "vbscf/derivatives/gradient/active_space/evaluator.hpp"

namespace xmvb::vb {

/** Default electronic-state adapter retaining the existing numerical kernels. */
class DeterminantElectronicStateBackend final : public ElectronicStateBackend {
 public:
  std::unique_ptr<ElectronicStateForwardEvaluation> evaluate_energy(
      const VbScfInput&, const ElectronicStateRequest&) const override;
  ActiveSpaceGradientResult complete_gradient(
      const VbScfInput&,
      std::unique_ptr<ElectronicStateForwardEvaluation>) const override;
  bool provides_second_order_context() const override { return true; }

 private:
  ActiveSpaceGradientEvaluator evaluator_;
};

}  // namespace xmvb::vb
