#pragma once

#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

/**
 * @brief Static system/chart features used by exact-ctx default policy.
 *
 * The accepted-point TN strategy should not use one global default for every
 * molecule.  Open-shell systems, cheap closed-shell sparse charts, and
 * expensive closed-shell sparse charts have shown materially different wall-
 * time optima in production logs.  This profile stores the inexpensive
 * input-only features used to choose the default strategy before any timing
 * samples are available.
 */
struct ExactCtxSystemProfile {
  bool sparse_orbital_chart = false;
  bool open_shell = false;
  int n_active_orbitals = 0;
  int n_basis_functions = 0;
  int active_basis_cost_proxy = 0;
};

enum class ExactCtxDefaultStrategyKind {
  CheapCoreOnly,
  StartupWindowWithGradientTail,
};

/**
 * @brief Default exact-ctx startup and chart policy for one system profile.
 *
 * The environment may still override any of these values, but this struct is
 * the single source of truth for the repository defaults when no override is
 * present.  Keeping the policy explicit avoids diverging behavior between the
 * optimizer, the exact operator, and the CLI summary output.
 */
struct ExactCtxDefaultStrategy {
  ExactCtxDefaultStrategyKind kind =
      ExactCtxDefaultStrategyKind::CheapCoreOnly;
  bool prefer_internal_inactive_chart = false;
  int startup_full_inner_solve_begin = 0;
  int startup_full_inner_solve_count = 0;
  int startup_full_inner_solve_max_extra_count = 0;
  int startup_full_inner_solve_tail_max_cg_iterations = 8;
  int startup_full_inner_solve_multi_step_max_active_orbitals = 8;
  int startup_full_inner_solve_enable_max_active_orbitals = 0;
  bool allow_hybrid_followup_full_solve = false;
  bool retry_rejected_step_with_full_operator = false;
};

ExactCtxSystemProfile build_exact_ctx_system_profile(
    const OrbitalPreparationInput& orbital_preparation_input);

ExactCtxDefaultStrategy choose_exact_ctx_default_strategy(
    const ExactCtxSystemProfile& system_profile);

const char* exact_ctx_default_strategy_kind_name(
    ExactCtxDefaultStrategyKind kind);

const char* exact_ctx_default_initial_inner_solve_policy_name(
    const ExactCtxDefaultStrategy& strategy);

}  // namespace xmvb::vb
