#include "vbscf/derivatives/gradient/active_space/determinant_backend.hpp"

#include <stdexcept>
#include <utility>

namespace xmvb::vb {
namespace {

class DeterminantForward final : public ElectronicStateForwardEvaluation {
 public:
  explicit DeterminantForward(ActiveSpaceForwardEvaluation value)
      : evaluation(std::move(value)) {}
  double total_energy() const override { return evaluation.total_energy(); }
  ActiveSpaceForwardEvaluation evaluation;
};

}  // namespace

std::unique_ptr<ElectronicStateForwardEvaluation>
DeterminantElectronicStateBackend::evaluate_energy(
    const VbScfInput& input, const ElectronicStateRequest& request) const {
  return std::make_unique<DeterminantForward>(evaluator_.evaluate_forward(
      input, request.selected_states, request.state_weights,
      request.nuclear_repulsion_energy, request.eigensolver, request.accuracy,
      request.initial_eigenvectors));
}

ActiveSpaceGradientResult DeterminantElectronicStateBackend::complete_gradient(
    const VbScfInput& input,
    std::unique_ptr<ElectronicStateForwardEvaluation> forward) const {
  auto* state = dynamic_cast<DeterminantForward*>(forward.get());
  if (state == nullptr) {
    throw std::invalid_argument("determinant backend received foreign forward state");
  }
  return evaluator_.complete_gradient(input, std::move(state->evaluation));
}

}  // namespace xmvb::vb
