#include "vb/scf/exact_ctx_strategy_profile.hpp"

#include <algorithm>

namespace xmvb::vb {

namespace {

constexpr int kLegacyOrbitalTypeHao = 1;
constexpr int kLegacyOrbitalTypeBdo = 2;

bool optimizer_chart_uses_sparse_orbital_support(
    const OrbitalPreparationInput& orbital_preparation_input) {
  if (orbital_preparation_input.orbital_type == kLegacyOrbitalTypeHao ||
      orbital_preparation_input.orbital_type == kLegacyOrbitalTypeBdo) {
    return true;
  }
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    return false;
  }
  for (const int basis_count : orbital_preparation_input.orbital_basis_counts) {
    if (basis_count > 0 && basis_count < n_basis_functions) {
      return true;
    }
  }
  return false;
}

}  // namespace

ExactCtxSystemProfile build_exact_ctx_system_profile(
    const OrbitalPreparationInput& orbital_preparation_input) {
  ExactCtxSystemProfile profile;
  profile.sparse_orbital_chart =
      optimizer_chart_uses_sparse_orbital_support(orbital_preparation_input);
  profile.open_shell = orbital_preparation_input.spin_multiplicity > 1;
  profile.n_active_orbitals = orbital_preparation_input.n_active_orbitals;
  profile.n_basis_functions = orbital_preparation_input.n_basis_functions;
  profile.active_basis_cost_proxy =
      std::max(0, profile.n_active_orbitals) *
      std::max(0, profile.n_basis_functions);
  return profile;
}

ExactCtxDefaultStrategy choose_exact_ctx_default_strategy(
    const ExactCtxSystemProfile& system_profile) {
  ExactCtxDefaultStrategy strategy;

  // Unified strategy: all systems start with cheap core-only, with adaptive
  // runtime upgrade to full-model when low trust is detected.
  strategy.prefer_internal_inactive_chart =
      !system_profile.sparse_orbital_chart;

  // Non-sparse charts (OEO): cheap model is sufficient.
  if (!system_profile.sparse_orbital_chart) {
    strategy.kind = ExactCtxDefaultStrategyKind::CheapCoreOnly;
    return strategy;
  }

  // All sparse HAO/BDO charts: one full-model calibration step to seed the
  // trust-region, then cheap model with adaptive runtime upgrade to full
  // model when low trust is detected.  The single calibration step gives
  // enough reference curvature without the wall-time penalty of a longer
  // startup window.
  constexpr int kCalibrationCount = 1;
  constexpr int kCalibrationTailCg = 10;
  strategy.kind = ExactCtxDefaultStrategyKind::StartupWindowWithGradientTail;
  strategy.startup_full_inner_solve_begin = 0;
  strategy.startup_full_inner_solve_count = kCalibrationCount;
  strategy.startup_full_inner_solve_tail_max_cg_iterations = kCalibrationTailCg;
  strategy.startup_full_inner_solve_enable_max_active_orbitals =
      system_profile.n_active_orbitals;
  strategy.startup_full_inner_solve_multi_step_max_active_orbitals =
      system_profile.n_active_orbitals;
  strategy.retry_rejected_step_with_full_operator = true;
  return strategy;
}

const char* exact_ctx_default_strategy_kind_name(
    ExactCtxDefaultStrategyKind kind) {
  switch (kind) {
    case ExactCtxDefaultStrategyKind::CheapCoreOnly:
      return "cheap_core_only";
    case ExactCtxDefaultStrategyKind::StartupWindowWithGradientTail:
      return "startup_full_window_with_gradient_tail";
  }
  return "unknown";
}

}  // namespace xmvb::vb
