#pragma once

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
  return "unknown";
}

enum class NonredundantTruncatedNewtonHvpMode {
  FullFiniteDifference,
  ExactContextDirectAction,
};

inline const char* nonredundant_truncated_newton_hvp_mode_name(
    NonredundantTruncatedNewtonHvpMode mode) {
  switch (mode) {
    case NonredundantTruncatedNewtonHvpMode::FullFiniteDifference:
      return "full_fd";
    case NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction:
      return "exact_ctx";
  }
  return "unknown";
}

}  // namespace xmvb::vb
