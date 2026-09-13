#pragma once

#include <stdexcept>

namespace xmvb::vb {

enum class VbScfOptimizerBackend {
  Lbfgspp,
  NonredundantProjectedGradient,
  NonredundantLbfgspp,
  NonredundantTruncatedNewton,
};

inline const char* vbscf_optimizer_backend_name(
    VbScfOptimizerBackend backend) {
  switch (backend) {
    case VbScfOptimizerBackend::Lbfgspp:
      return "lbfgspp";
    case VbScfOptimizerBackend::NonredundantProjectedGradient:
      return "nonredundant_projected_gradient";
    case VbScfOptimizerBackend::NonredundantLbfgspp:
      return "nonredundant_lbfgspp";
    case VbScfOptimizerBackend::NonredundantTruncatedNewton:
      return "nonredundant_truncated_newton";
  }
  throw std::invalid_argument("invalid VBSCF optimizer backend");
}

}  // namespace xmvb::vb
