#include "vb/scf/exact_ctx_strategy_profile.hpp"

#include <algorithm>

namespace xmvb::vb {

namespace {

constexpr int kLegacyOrbitalTypeHao = 1;
constexpr int kLegacyOrbitalTypeBdo = 2;

// Use a simple AO-active cost proxy so the default startup policy can
// distinguish affordable closed-shell sparse charts such as 241 from much more
// expensive closed-shell sparse charts such as 10698 before the optimizer has
// accumulated any accepted-step timing samples.
constexpr int kAffordableStartupOuterResponseCostProxy = 1024;

// Open-shell sparse-chart hybrid followup is tuned on 8-active-orbital
// transition-metal systems.  The retry gate uses a separate lower bound.
constexpr int kHybridFollowupExactActiveOrbitalThreshold = 8;
constexpr int kRetryRejectedFullOperatorMinActiveOrbitals = 8;

// Closed-shell sparse charts with more than this many active orbitals always
// get CheapCoreOnly regardless of the cost proxy.
constexpr int kStartupWindowMaxActiveOrbitals = 6;

// Startup window parameters for expensive closed-shell sparse charts:
// begin after one cheap accepted step, sample the full model for two
// iterations, allow one extra if the gradient tail is still large, and keep
// the tail Krylov budget short.
constexpr int kStartupWindowBegin = 1;
constexpr int kStartupWindowCount = 2;
constexpr int kStartupWindowMaxExtraCount = 1;
constexpr int kStartupWindowTailMaxCgIterations = 4;
constexpr int kStartupWindowMultiStepMaxActiveOrbitals = 6;

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

  if (system_profile.open_shell) {
    strategy.kind = ExactCtxDefaultStrategyKind::CheapCoreOnly;
    strategy.prefer_internal_inactive_chart = false;
    strategy.startup_full_inner_solve_enable_max_active_orbitals = 0;
    strategy.allow_hybrid_followup_full_solve =
        system_profile.sparse_orbital_chart &&
        system_profile.n_active_orbitals == kHybridFollowupExactActiveOrbitalThreshold;
    strategy.retry_rejected_step_with_full_operator =
        system_profile.sparse_orbital_chart &&
        system_profile.n_active_orbitals >= kRetryRejectedFullOperatorMinActiveOrbitals;
    return strategy;
  }

  // The internal inactive `(Q_i, T_a)` chart was intended as a cheaper
  // accepted-point model for closed-shell sparse HAO/BDO systems, but current
  // 240/241 production traces show the opposite: it sends both systems onto
  // the slow TNHVP trajectory, while staying in the physical occupied chart
  // reproduces the fast reference path. Keep the sparse-chart default on the
  // physical occupied manifold unless the user explicitly forces the internal
  // chart through the environment override.
  strategy.prefer_internal_inactive_chart =
      !system_profile.sparse_orbital_chart;
  if (!system_profile.sparse_orbital_chart ||
      system_profile.n_active_orbitals <= 0 ||
      system_profile.n_active_orbitals > kStartupWindowMaxActiveOrbitals) {
    strategy.kind = ExactCtxDefaultStrategyKind::CheapCoreOnly;
    strategy.startup_full_inner_solve_enable_max_active_orbitals = 0;
    strategy.allow_hybrid_followup_full_solve = false;
    return strategy;
  }

  strategy.startup_full_inner_solve_enable_max_active_orbitals = kStartupWindowMaxActiveOrbitals;
  if (system_profile.active_basis_cost_proxy <=
      kAffordableStartupOuterResponseCostProxy) {
    // With the current full-width AO-H1E and row-local exact-2e kernels, the
    // accepted cheap model reaches 241-class closed-shell sparse solutions with
    // lower wall time than paying for full outer-response startup solves. Keep
    // full-model corrections available through the optimizer rejection/stall
    // gates instead of spending them unconditionally at iterations 0 and 1.
    strategy.kind = ExactCtxDefaultStrategyKind::CheapCoreOnly;
    strategy.startup_full_inner_solve_enable_max_active_orbitals = 0;
    strategy.allow_hybrid_followup_full_solve = false;
    return strategy;
  }

  // Expensive closed-shell sparse charts should still sample the full model,
  // but only after one cheap accepted step and with a tighter tail Krylov
  // budget.  This captures the historical 10698-class direction without
  // reintroducing a legacy-AO-only dependency.
  strategy.kind = ExactCtxDefaultStrategyKind::StartupWindowWithGradientTail;
  strategy.startup_full_inner_solve_begin = kStartupWindowBegin;
  strategy.startup_full_inner_solve_count = kStartupWindowCount;
  strategy.startup_full_inner_solve_max_extra_count = kStartupWindowMaxExtraCount;
  strategy.startup_full_inner_solve_tail_max_cg_iterations = kStartupWindowTailMaxCgIterations;
  strategy.startup_full_inner_solve_multi_step_max_active_orbitals = kStartupWindowMultiStepMaxActiveOrbitals;
  strategy.allow_hybrid_followup_full_solve = false;
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

const char* exact_ctx_default_initial_inner_solve_policy_name(
    const ExactCtxDefaultStrategy& strategy) {
  return exact_ctx_default_strategy_kind_name(strategy.kind);
}

}  // namespace xmvb::vb
