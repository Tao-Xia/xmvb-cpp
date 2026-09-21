#pragma once

#include <stdexcept>

namespace xmvb::vb {

enum class VbScfOptimizerBackend {
  Lbfgs,
  BlockLbfgs,
  Neo,
  NonredundantProjectedGradient,
  NonredundantTruncatedNewton,
};

inline const char* vbscf_optimizer_backend_name(
    VbScfOptimizerBackend backend) {
  switch (backend) {
    case VbScfOptimizerBackend::Lbfgs:
      return "lbfgs";
    case VbScfOptimizerBackend::BlockLbfgs:
      return "block_lbfgs";
    case VbScfOptimizerBackend::Neo:
      return "neo";
    case VbScfOptimizerBackend::NonredundantProjectedGradient:
      return "nonredundant_projected_gradient";
    case VbScfOptimizerBackend::NonredundantTruncatedNewton:
      return "nonredundant_truncated_newton";
  }
  throw std::invalid_argument("invalid VBSCF optimizer backend");
}

}  // namespace xmvb::vb
