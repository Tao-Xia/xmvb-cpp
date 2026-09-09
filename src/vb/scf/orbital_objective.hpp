#pragma once

// Compatibility header. New code should include
// "vbscf/optimization/vbscf_objective.hpp".
#include "vbscf/optimization/vbscf_objective.hpp"

namespace xmvb::vb {

using OrbitalObjectiveTrialEvaluation = VbScfObjectiveTrialEvaluation;
using OrbitalObjective = VbScfObjective;

}  // namespace xmvb::vb
