#include "pfaffian_vbscf/scf/pf_orbital_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <variant>

#include <Eigen/Core>
#include <LBFGS.h>

namespace xmvb::pfaffian_vbscf {

namespace {

constexpr double kLineSearchExpansionFactor = 10.0;

double gradient_infinity_norm(const Eigen::VectorXd& gradient) {
  double norm = 0.0;
  for (Eigen::Index index = 0; index < gradient.size(); ++index) {
    norm = std::max(norm, std::abs(gradient[index]));
  }
  return norm;
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& input,
    int orbital_index) {
  const int n_basis_functions = input.n_basis_functions;
  const int explicit_count =
      input.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_index =
        input.orbital_basis_index_table[
            xmvb::to_size(orbital_index) * n_basis_functions +
            coefficient_count];
    if (basis_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const xmvb::vb::OrbitalPreparationInput& input) {
  std::vector<int> parameter_indices;
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      parameter_indices.push_back(
          orbital_index * input.n_basis_functions + coefficient_index);
    }
  }
  return parameter_indices;
}

Eigen::VectorXd pack_parameter_vector(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& parameter_indices) {
  Eigen::VectorXd parameters(
      static_cast<Eigen::Index>(parameter_indices.size()));
  for (Eigen::Index parameter_offset = 0;
       parameter_offset < parameters.size();
       ++parameter_offset) {
    const int parameter_index =
        parameter_indices[xmvb::to_size(parameter_offset)];
    parameters[parameter_offset] =
        input.orbital_preparation_input.orbital_value_table[
            xmvb::to_size(parameter_index)];
  }
  return parameters;
}

void unpack_parameter_vector(
    const Eigen::VectorXd& parameters,
    const std::vector<int>& parameter_indices,
    xmvb::vb::CppVbInput* input) {
  for (Eigen::Index parameter_offset = 0;
       parameter_offset < parameters.size();
       ++parameter_offset) {
    const int parameter_index =
        parameter_indices[xmvb::to_size(parameter_offset)];
    input->orbital_preparation_input.orbital_value_table[
        xmvb::to_size(parameter_index)] =
        parameters[parameter_offset];
  }
}

class OrbitalObjective {
public:
  OrbitalObjective(
      const xmvb::vb::CppVbInput& input,
      const std::vector<int>& parameter_indices,
      std::variant<PfBasisData, PfSpinAdaptedBasisData> basis,
      double nuclear_repulsion_energy,
      const PfOrbitalGradEval* orbital_gradient_evaluator)
      : working_input_(input),
        parameter_indices_(parameter_indices),
        basis_(std::move(basis)),
        nuclear_repulsion_energy_(nuclear_repulsion_energy),
        orbital_gradient_evaluator_(orbital_gradient_evaluator) {}

  double operator()(const Eigen::VectorXd& parameters, Eigen::VectorXd& gradient) {
    unpack_parameter_vector(parameters, parameter_indices_, &working_input_);
    PfOrbitalGradResult gradient_result;
    if (std::holds_alternative<PfBasisData>(basis_)) {
      gradient_result =
          orbital_gradient_evaluator_->eval(
              working_input_,
              std::get<PfBasisData>(basis_),
              nuclear_repulsion_energy_);
    } else {
      gradient_result =
          orbital_gradient_evaluator_->eval(
              working_input_,
              std::get<PfSpinAdaptedBasisData>(basis_),
              nuclear_repulsion_energy_);
    }

    gradient.resize(parameters.size());
    for (Eigen::Index parameter_offset = 0;
         parameter_offset < parameters.size();
         ++parameter_offset) {
      const int parameter_index =
          parameter_indices_[xmvb::to_size(parameter_offset)];
      gradient[parameter_offset] =
          gradient_result.sparse_orbital_energy_gradient[
              xmvb::to_size(parameter_index)];
    }

    last_gradient_result_ = gradient_result;
    energy_history_.push_back(gradient_result.scf_result.e_tot);
    gradient_inf_norm_history_.push_back(gradient_infinity_norm(gradient));
    iteration_time_history_seconds_.push_back(
        gradient_result.total_wall_time_seconds);
    ++evaluation_count_;
    return gradient_result.scf_result.e_tot;
  }

  const xmvb::vb::CppVbInput& last_input() const { return working_input_; }

  const PfOrbitalGradResult& last_gradient_result() const {
    return last_gradient_result_;
  }

  const std::vector<double>& energy_history() const { return energy_history_; }

  const std::vector<double>& gradient_inf_norm_history() const {
    return gradient_inf_norm_history_;
  }

  const std::vector<double>& iteration_time_history_seconds() const {
    return iteration_time_history_seconds_;
  }

  int evaluation_count() const { return evaluation_count_; }

private:
  xmvb::vb::CppVbInput working_input_;
  std::vector<int> parameter_indices_;
  std::variant<PfBasisData, PfSpinAdaptedBasisData> basis_;
  double nuclear_repulsion_energy_ = 0.0;
  const PfOrbitalGradEval* orbital_gradient_evaluator_ = nullptr;

  PfOrbitalGradResult last_gradient_result_;
  std::vector<double> energy_history_;
  std::vector<double> gradient_inf_norm_history_;
  std::vector<double> iteration_time_history_seconds_;
  int evaluation_count_ = 0;
};

bool try_steepest_descent_armijo_fallback(
    OrbitalObjective* objective,
    const LBFGSpp::LBFGSParam<double>& param,
    const Eigen::VectorXd& start_parameters,
    const Eigen::VectorXd& start_gradient,
    double start_energy,
    double initial_step,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy,
    double* accepted_step) {
  if (objective == nullptr ||
      accepted_parameters == nullptr ||
      accepted_gradient == nullptr ||
      accepted_energy == nullptr ||
      accepted_step == nullptr) {
    throw std::invalid_argument("fallback line search outputs must not be null");
  }

  const Eigen::VectorXd direction = -start_gradient;
  const double directional_derivative = start_gradient.dot(direction);
  if (!(directional_derivative < 0.0)) {
    return false;
  }

  double reference_step =
      std::max(param.min_step, std::min(initial_step, param.max_step));
  std::vector<double> trial_steps;
  trial_steps.push_back(reference_step);

  constexpr int kMaxExpansionTrials = 4;
  double expanded_step = reference_step;
  for (int trial = 0; trial < kMaxExpansionTrials; ++trial) {
    if (expanded_step >= param.max_step) {
      break;
    }
    const double next_step =
        std::min(param.max_step, expanded_step * 2.0);
    if (next_step <= expanded_step) {
      break;
    }
    trial_steps.push_back(next_step);
    expanded_step = next_step;
  }

  double contracted_step = reference_step;
  while (contracted_step > param.min_step) {
    contracted_step *= 0.5;
    if (contracted_step < param.min_step) {
      contracted_step = param.min_step;
    }
    if (contracted_step < trial_steps.back()) {
      trial_steps.push_back(contracted_step);
    }
    if (contracted_step <= param.min_step) {
      break;
    }
  }

  bool has_best_descent = false;
  Eigen::VectorXd best_parameters;
  Eigen::VectorXd best_gradient;
  double best_energy = start_energy;
  double best_step = reference_step;
  for (double step : trial_steps) {
    Eigen::VectorXd trial_parameters =
        (start_parameters + step * direction).eval();
    Eigen::VectorXd trial_gradient;
    const double trial_energy =
        (*objective)(trial_parameters, trial_gradient);
    if (std::isfinite(trial_energy) && trial_gradient.allFinite()) {
      if (trial_energy < best_energy) {
        has_best_descent = true;
        best_parameters = trial_parameters;
        best_gradient = trial_gradient;
        best_energy = trial_energy;
        best_step = step;
      }
    }
    if (std::isfinite(trial_energy) &&
        trial_gradient.allFinite() &&
        trial_energy <= start_energy + param.ftol * step * directional_derivative) {
      *accepted_parameters = std::move(trial_parameters);
      *accepted_gradient = std::move(trial_gradient);
      *accepted_energy = trial_energy;
      *accepted_step = step;
      return true;
    }
  }

  if (has_best_descent) {
    *accepted_parameters = std::move(best_parameters);
    *accepted_gradient = std::move(best_gradient);
    *accepted_energy = best_energy;
    *accepted_step = best_step;
    return true;
  }

  return false;
}

void sync_result_from_objective(
    const OrbitalObjective& objective,
    PfScfOptimizerResult* result) {
  result->total_energy_history = objective.energy_history();
  result->gradient_inf_norm_history = objective.gradient_inf_norm_history();
  result->iteration_time_history_seconds =
      objective.iteration_time_history_seconds();
  result->scf_result = objective.last_gradient_result().scf_result;
}

}  // namespace

PfScfOptimizer::PfScfOptimizer(
    PfBasisData basis,
    PfScfOptimizerOptions options)
    : basis_(std::move(basis)),
      orbital_gradient_evaluator_(),
      options_(options) {}

PfScfOptimizer::PfScfOptimizer(
    PfSpinAdaptedBasisData basis,
    PfScfOptimizerOptions options)
    : basis_(std::move(basis)),
      orbital_gradient_evaluator_(),
      options_(options) {}

PfScfOptimizer::PfScfOptimizer(
    PfBasisData basis,
    PfOrbitalGradEval orbital_gradient_evaluator,
    PfScfOptimizerOptions options)
    : basis_(std::move(basis)),
      orbital_gradient_evaluator_(std::move(orbital_gradient_evaluator)),
      options_(options) {}

PfScfOptimizer::PfScfOptimizer(
    PfSpinAdaptedBasisData basis,
    PfOrbitalGradEval orbital_gradient_evaluator,
    PfScfOptimizerOptions options)
    : basis_(std::move(basis)),
      orbital_gradient_evaluator_(std::move(orbital_gradient_evaluator)),
      options_(options) {}

PfScfOptimizerResult PfScfOptimizer::optimize(
    const xmvb::vb::CppVbInput& input,
    double nuclear_repulsion_energy) const {
  if (options_.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (options_.gradient_tolerance <= 0.0 ||
      options_.energy_tolerance <= 0.0 ||
      options_.initial_step_size <= 0.0 ||
      options_.minimum_step_size <= 0.0 ||
      options_.armijo_constant <= 0.0 ||
      options_.armijo_constant >= 1.0) {
    throw std::invalid_argument(
        "optimizer tolerances and step-size controls must be valid");
  }
  if (options_.minimum_step_size > options_.initial_step_size) {
    throw std::invalid_argument(
        "minimum_step_size must not exceed initial_step_size");
  }
  if (options_.history_size <= 0) {
    throw std::invalid_argument("history_size must be positive");
  }

  PfScfOptimizerResult result;
  if (std::holds_alternative<PfBasisData>(basis_)) {
    result.spin_adapted = false;
    result.basis = std::get<PfBasisData>(basis_);
  } else {
    result.spin_adapted = true;
    result.spin_adapted_basis = std::get<PfSpinAdaptedBasisData>(basis_);
    result.basis = result.spin_adapted_basis.primitive_basis;
  }
  const auto optimization_start_time = std::chrono::steady_clock::now();

  const auto parameter_indices =
      collect_differentiable_parameter_indices(
          input.orbital_preparation_input);
  Eigen::VectorXd parameters =
      pack_parameter_vector(input, parameter_indices);
  if (parameters.size() == 0) {
    throw std::invalid_argument(
        "Pf optimizer requires at least one differentiable parameter");
  }

  OrbitalObjective objective(
      input,
      parameter_indices,
      basis_,
      nuclear_repulsion_energy,
      &orbital_gradient_evaluator_);

  const int n = static_cast<int>(parameters.size());
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;

  try {
    Eigen::VectorXd gradient(parameters.size());
    double energy = objective(parameters, gradient);
    sync_result_from_objective(objective, &result);
    result.initial_objective_eval_count = objective.evaluation_count();
    result.initial_total_energy = energy;
    double previous_energy = energy;
    bool has_previous_energy = true;
    final_gradient_l2_norm = gradient.norm();

    LBFGSpp::LBFGSParam<double> param;
    param.m = options_.history_size;
    param.epsilon = 0.0;
    param.epsilon_rel = 0.0;
    param.past = 0;
    param.delta = 0.0;
    param.max_iterations = 0;
    param.max_linesearch = 20;
    param.min_step = options_.minimum_step_size;
    // Allow More-Thuente to expand beyond the initial trial step without
    // letting it explore arbitrarily large orbital-parameter displacements.
    param.max_step =
        std::max(options_.initial_step_size,
                 options_.initial_step_size * kLineSearchExpansionFactor);
    param.ftol = options_.armijo_constant;
    param.wolfe = 0.9;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    param.check_param();

    LBFGSpp::BFGSMat<double> inverse_hessian;
    inverse_hessian.reset(n, param.m);

    Eigen::VectorXd current_parameters = parameters;
    Eigen::VectorXd current_gradient = gradient;
    Eigen::VectorXd previous_parameters(n);
    Eigen::VectorXd previous_gradient(n);
    Eigen::VectorXd search_direction = -current_gradient;
    double last_robust_step = options_.initial_step_size;
    bool has_robust_step_history = false;
    bool last_iteration_used_fallback = false;
    constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();
    constexpr double kTinyStepFactor = 10.0;
    constexpr double kRobustStepShrinkRatio = 0.1;
    constexpr double kSuspiciousPrimaryStepRatio = 0.1;

    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
      if (search_direction.dot(current_gradient) >= 0.0) {
        search_direction = -current_gradient;
      }
      double directional_derivative =
          current_gradient.dot(search_direction);
      if (directional_derivative >= 0.0) {
        result.termination_reason = "lbfgspp_non_descent_direction";
        break;
      }

      previous_parameters = current_parameters;
      previous_gradient = current_gradient;
      const double line_search_start_energy = energy;
      double step = options_.initial_step_size;
      if (last_iteration_used_fallback && has_robust_step_history) {
        step = last_robust_step;
      }
      bool used_fallback = false;
      bool reset_inverse_hessian = false;
      std::string primary_line_search_error;
      const int eval_count_before_primary = objective.evaluation_count();

      try {
        LBFGSpp::LineSearchMoreThuente<double>::LineSearch(
            objective,
            param,
            previous_parameters,
            search_direction,
            param.max_step,
            step,
            energy,
            current_gradient,
            directional_derivative,
            current_parameters);
      } catch (const std::exception& error) {
        primary_line_search_error = error.what();
      }
      const int primary_eval_count =
          objective.evaluation_count() - eval_count_before_primary;
      int fallback_eval_count = 0;

      const bool suspicious_primary_step =
          primary_line_search_error.empty() &&
          has_robust_step_history &&
          step < kSuspiciousPrimaryStepRatio * last_robust_step;
      if (!primary_line_search_error.empty() ||
          step <= kTinyStepFactor * param.min_step ||
          suspicious_primary_step) {
        Eigen::VectorXd fallback_parameters;
        Eigen::VectorXd fallback_gradient;
        double fallback_energy = line_search_start_energy;
        double fallback_step =
            has_robust_step_history ? last_robust_step
                                    : options_.initial_step_size;
        const int eval_count_before_fallback = objective.evaluation_count();
        if (!try_steepest_descent_armijo_fallback(
                &objective,
                param,
                previous_parameters,
                previous_gradient,
                line_search_start_energy,
                fallback_step,
                &fallback_parameters,
                &fallback_gradient,
                &fallback_energy,
                &fallback_step)) {
          fallback_eval_count =
              objective.evaluation_count() - eval_count_before_fallback;
          sync_result_from_objective(objective, &result);
          final_gradient_l2_norm = current_gradient.norm();
          if (!primary_line_search_error.empty()) {
            result.termination_reason =
                std::string("lbfgspp_line_search: ") + primary_line_search_error;
          } else {
            result.termination_reason =
                "lbfgspp_line_search_stagnation";
          }
          break;
        }
        fallback_eval_count =
            objective.evaluation_count() - eval_count_before_fallback;

        current_parameters = std::move(fallback_parameters);
        current_gradient = std::move(fallback_gradient);
        energy = fallback_energy;
        step = fallback_step;
        used_fallback = true;
        reset_inverse_hessian = true;
      }

      ++n_iterations;
      sync_result_from_objective(objective, &result);
      final_gradient_l2_norm = current_gradient.norm();
      result.objective_eval_count_history.push_back(
          primary_eval_count + fallback_eval_count);
      result.primary_line_search_eval_count_history.push_back(
          primary_eval_count);
      result.fallback_line_search_eval_count_history.push_back(
          fallback_eval_count);
      result.fallback_used_history.push_back(used_fallback ? 1 : 0);

      if (step > kTinyStepFactor * param.min_step &&
          (!has_robust_step_history ||
           step >= kRobustStepShrinkRatio * last_robust_step)) {
        last_robust_step = step;
        has_robust_step_history = true;
      }
      last_iteration_used_fallback = used_fallback;

      const double energy_change =
          has_previous_energy ? (energy - previous_energy) : 0.0;
      previous_energy = energy;
      has_previous_energy = true;
      if (std::abs(energy_change) < options_.energy_tolerance &&
          result.gradient_inf_norm_history.back() <
              options_.gradient_tolerance) {
        result.converged = true;
        result.termination_reason = "lbfgspp_dual_tolerance";
        break;
      }

      Eigen::VectorXd parameter_step =
          current_parameters - previous_parameters;
      Eigen::VectorXd gradient_step =
          current_gradient - previous_gradient;
      if (reset_inverse_hessian) {
        inverse_hessian.reset(n, param.m);
      }
      if (!reset_inverse_hessian &&
          parameter_step.dot(gradient_step) >
              kCurvatureEpsilon * gradient_step.squaredNorm()) {
        inverse_hessian.add_correction(parameter_step, gradient_step);
      }
      inverse_hessian.apply_Hv(current_gradient, -1.0, search_direction);
    }
  } catch (const std::exception& error) {
    result.termination_reason = error.what();
  }

  if (result.total_energy_history.empty()) {
    if (!result.termination_reason.empty()) {
      throw std::runtime_error(
          std::string("Pf optimizer did not evaluate the objective: ") +
          result.termination_reason);
    }
    throw std::runtime_error("Pf optimizer did not evaluate the objective");
  }

  result.n_iterations = n_iterations;
  result.total_objective_eval_count = objective.evaluation_count();
  result.final_total_energy = result.total_energy_history.back();
  result.final_gradient_inf_norm = result.gradient_inf_norm_history.back();
  result.final_gradient_l2_norm = final_gradient_l2_norm;
  result.optimized_input = objective.last_input();

  if (result.termination_reason.empty()) {
    if (result.n_iterations >= options_.max_iterations) {
      result.termination_reason = "max_iterations";
    } else {
      result.termination_reason = "stopped";
    }
  }

  const auto optimization_end_time = std::chrono::steady_clock::now();
  result.total_wall_time_seconds =
      std::chrono::duration<double>(
          optimization_end_time - optimization_start_time)
          .count();
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
