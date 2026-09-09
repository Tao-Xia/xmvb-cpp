#pragma once

// Compatibility header. New code should include
// "vbscf/optimization/vbscf_optimizer_result.hpp".
#include "vbscf/optimization/vbscf_optimizer_result.hpp"

namespace xmvb::vb {

using CppVbScfAcceptedIterationSnapshot = VbScfAcceptedIterationSnapshot;
using CppVbScfOptimizerResult = VbScfOptimizerResult;

}  // namespace xmvb::vb
