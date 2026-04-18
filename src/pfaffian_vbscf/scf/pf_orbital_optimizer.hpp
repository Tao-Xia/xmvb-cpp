#pragma once

#include <string>
#include <vector>
#include <variant>

#include "pfaffian_vbscf/scf/pf_orbital_grad_eval.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "pfaffian_vbscf/types/pf_spin_adapted_basis_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfScfOptimizerOptions {
  int max_iterations = 50;
  double gradient_tolerance = 2.0e-3;
  double energy_tolerance = 1.0e-7;
  double initial_step_size = 1.0;
  double minimum_step_size = 1.0e-7;
  double armijo_constant = 1.0e-4;
  int history_size = 100;
  bool verbose = true;
};

struct PfScfOptimizerResult {
  bool converged = false;
  std::string termination_reason;
  int n_iterations = 0;
  int initial_objective_eval_count = 0;
  int total_objective_eval_count = 0;
  double initial_total_energy = 0.0;
  double final_total_energy = 0.0;
  double final_gradient_inf_norm = 0.0;
  double final_gradient_l2_norm = 0.0;
  std::vector<double> total_energy_history;
  std::vector<double> gradient_inf_norm_history;
  std::vector<double> iteration_time_history_seconds;
  std::vector<int> objective_eval_count_history;
  std::vector<int> primary_line_search_eval_count_history;
  std::vector<int> fallback_line_search_eval_count_history;
  std::vector<int> fallback_used_history;
  double total_wall_time_seconds = 0.0;
  PfScfResult scf_result;
  bool spin_adapted = false;
  PfBasisData basis;
  PfSpinAdaptedBasisData spin_adapted_basis;
  xmvb::vb::CppVbInput optimized_input;
};

class PfScfOptimizer {
public:
  explicit PfScfOptimizer(
      PfBasisData basis,
      PfScfOptimizerOptions options = {});

  explicit PfScfOptimizer(
      PfSpinAdaptedBasisData basis,
      PfScfOptimizerOptions options = {});

  PfScfOptimizer(
      PfBasisData basis,
      PfOrbitalGradEval orbital_gradient_evaluator,
      PfScfOptimizerOptions options);

  PfScfOptimizer(
      PfSpinAdaptedBasisData basis,
      PfOrbitalGradEval orbital_gradient_evaluator,
      PfScfOptimizerOptions options);

  PfScfOptimizerResult optimize(
      const xmvb::vb::CppVbInput& input,
      double nuclear_repulsion_energy = 0.0) const;

private:
  std::variant<PfBasisData, PfSpinAdaptedBasisData> basis_;
  PfOrbitalGradEval orbital_gradient_evaluator_;
  PfScfOptimizerOptions options_;
};

}  // namespace xmvb::pfaffian_vbscf
