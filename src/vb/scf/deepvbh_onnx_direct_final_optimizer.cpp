#include "vb/scf/deepvbh_onnx_direct_final_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xmvb::vb {

namespace {

double gradient_infinity_norm(
    const std::vector<double>& gradient,
    const std::vector<int>& differentiable_parameter_indices) {
  double norm = 0.0;
  for (const int parameter_index : differentiable_parameter_indices) {
    norm = std::max(norm, std::abs(gradient[parameter_index]));
  }
  return norm;
}

double gradient_l2_norm(
    const std::vector<double>& gradient,
    const std::vector<int>& differentiable_parameter_indices) {
  double squared_norm = 0.0;
  for (const int parameter_index : differentiable_parameter_indices) {
    const double value = gradient[parameter_index];
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

std::vector<int> collect_differentiable_parameter_indices(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<int> differentiable_parameter_indices;
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        differentiable_sparse_orbital_parameter_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      differentiable_parameter_indices.push_back(
          orbital_index * orbital_preparation_input.n_basis_functions + coefficient_index);
    }
  }
  return differentiable_parameter_indices;
}

CppVbScfAcceptedIterationSnapshot build_accepted_iteration_snapshot(
    const CppVbInput& input,
    const CppOrbitalGradientResult& gradient_result,
    int accepted_iteration_index) {
  CppVbScfAcceptedIterationSnapshot snapshot;
  snapshot.accepted_iteration_index = accepted_iteration_index;
  snapshot.orbital_value_table.assign(
      input.orbital_preparation_input.orbital_value_table.begin(),
      input.orbital_preparation_input.orbital_value_table.end());
  snapshot.structure_matrices = gradient_result.scf_result.structure_matrices;
  snapshot.active_orbital_overlap_matrix =
      gradient_result.active_orbital_overlap_matrix;
  snapshot.active_one_electron_integrals =
      gradient_result.active_one_electron_integrals;
  snapshot.packed_active_two_electron_integrals =
      gradient_result.packed_active_two_electron_integrals;
  snapshot.sparse_orbital_energy_gradient =
      gradient_result.sparse_orbital_energy_gradient;
  snapshot.sparse_orbital_reference_energy_gradient =
      gradient_result.sparse_orbital_reference_energy_gradient;
  snapshot.total_energy = gradient_result.scf_result.total_energy;
  snapshot.one_electron_reference_energy =
      gradient_result.scf_result.one_electron_reference_energy;
  snapshot.average_structure_overlap =
      gradient_result.scf_result.average_structure_overlap;
  return snapshot;
}

void record_accepted_iteration_snapshot(
    const CppVbInput& input,
    const CppOrbitalGradientResult& gradient_result,
    int accepted_iteration_index,
    const CppVbScfOptimizerOptions& options,
    CppVbScfOptimizerResult* result) {
  const auto snapshot =
      build_accepted_iteration_snapshot(input, gradient_result, accepted_iteration_index);
  if (options.retain_accepted_iteration_trace) {
    result->accepted_iteration_trace.push_back(snapshot);
  }
  if (options.accepted_iteration_callback) {
    options.accepted_iteration_callback(snapshot);
  }
}

bool is_finite_vector(
    const std::vector<double>& values,
    const std::vector<int>& differentiable_parameter_indices) {
  for (const int parameter_index : differentiable_parameter_indices) {
    if (!std::isfinite(values[parameter_index])) {
      return false;
    }
  }
  return true;
}

double differentiable_direction_norm(
    const std::vector<double>& direction,
    const std::vector<int>& differentiable_parameter_indices) {
  double squared_norm = 0.0;
  for (const int parameter_index : differentiable_parameter_indices) {
    const double value = direction[parameter_index];
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

CppVbInput apply_sparse_direction(
    const CppVbInput& input,
    const std::vector<double>& direction,
    const std::vector<int>& differentiable_parameter_indices,
    double step_scale) {
  CppVbInput updated_input = input;
  auto& orbital_value_table =
      updated_input.orbital_preparation_input.orbital_value_table;
  for (const int parameter_index : differentiable_parameter_indices) {
    orbital_value_table[parameter_index] +=
        step_scale * direction[parameter_index];
  }
  return updated_input;
}

void append_fallback_histories(
    const CppVbScfOptimizerResult& fallback_result,
    CppVbScfOptimizerResult* result) {
  if (fallback_result.total_energy_history.size() > 1) {
    result->total_energy_history.insert(
        result->total_energy_history.end(),
        fallback_result.total_energy_history.begin() + 1,
        fallback_result.total_energy_history.end());
  }
  if (fallback_result.gradient_inf_norm_history.size() > 1) {
    result->gradient_inf_norm_history.insert(
        result->gradient_inf_norm_history.end(),
        fallback_result.gradient_inf_norm_history.begin() + 1,
        fallback_result.gradient_inf_norm_history.end());
  }
  if (fallback_result.iteration_time_history_seconds.size() > 1) {
    result->iteration_time_history_seconds.insert(
        result->iteration_time_history_seconds.end(),
        fallback_result.iteration_time_history_seconds.begin() + 1,
        fallback_result.iteration_time_history_seconds.end());
  }
}

void append_fallback_trace(
    const CppVbScfOptimizerResult& fallback_result,
    int iteration_offset,
    CppVbScfOptimizerResult* result) {
  if (fallback_result.accepted_iteration_trace.size() <= 1) {
    return;
  }
  for (std::size_t trace_index = 1;
       trace_index < fallback_result.accepted_iteration_trace.size();
       ++trace_index) {
    auto snapshot = fallback_result.accepted_iteration_trace[trace_index];
    snapshot.accepted_iteration_index += iteration_offset;
    result->accepted_iteration_trace.push_back(std::move(snapshot));
  }
}

}  // namespace

DeepVBHOnnxDirectFinalOptimizer::DeepVBHOnnxDirectFinalOptimizer(
    DeepVBHOnnxDirectFinalOptimizerOptions options)
    : inference_runner_([&options]() {
        options.optimizer_options.backend =
            CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal;
        options.inference_options.backend = "onnx_runtime";
        options.inference_options.algorithm = options.optimizer_options.algorithm;
        return options.inference_options;
      }()),
      orbital_gradient_evaluator_(options.optimizer_options.algorithm),
      scf_evaluator_(options.optimizer_options.algorithm),
      options_(std::move(options)) {
  options_.optimizer_options.backend =
      CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal;
  options_.inference_options.backend = "onnx_runtime";
  options_.inference_options.algorithm = options_.optimizer_options.algorithm;
}

CppVbScfOptimizerResult DeepVBHOnnxDirectFinalOptimizer::optimize(
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    double nuclear_repulsion_energy) const {
  return optimize(
      input,
      raw_structure_data,
      static_molecule_metadata,
      {0},
      {1.0},
      nuclear_repulsion_energy);
}

CppVbScfOptimizerResult DeepVBHOnnxDirectFinalOptimizer::optimize(
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (options_.optimizer_options.algorithm != VBSCFAlgorithm::Original) {
    throw std::invalid_argument(
        "deepvbh_onnx_direct_final currently supports only the original VBSCF algorithm");
  }
  if (options_.optimizer_options.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (options_.optimizer_options.gradient_tolerance <= 0.0 ||
      options_.optimizer_options.energy_tolerance <= 0.0) {
    throw std::invalid_argument("optimizer tolerances must be positive");
  }
  if (options_.initial_step_scale <= 0.0 ||
      options_.minimum_step_scale <= 0.0 ||
      options_.step_shrink_factor <= 0.0 ||
      options_.step_shrink_factor >= 1.0) {
    throw std::invalid_argument("invalid DeepVBH direct-final step-scale configuration");
  }
  if (options_.minimum_step_scale > options_.initial_step_scale) {
    throw std::invalid_argument(
        "minimum_step_scale must not exceed initial_step_scale");
  }
  if (options_.max_backtracks <= 0) {
    throw std::invalid_argument("max_backtracks must be positive");
  }
  if (options_.inference_options.onnx_model_path.empty()) {
    throw std::invalid_argument(
        "deepvbh_onnx_direct_final requires inference_options.onnx_model_path");
  }

  const auto optimization_start_time = std::chrono::steady_clock::now();
  CppVbScfOptimizerResult result;
  const auto differentiable_parameter_indices =
      collect_differentiable_parameter_indices(input.orbital_preparation_input);
  if (differentiable_parameter_indices.empty()) {
    throw std::invalid_argument("optimizer requires at least one differentiable parameter");
  }

  CppVbInput current_input = input;
  const auto initial_gradient_result = orbital_gradient_evaluator_.evaluate(
      current_input,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy);
  result.scf_result = initial_gradient_result.scf_result;
  result.optimized_input = current_input;
  result.initial_total_energy = initial_gradient_result.scf_result.total_energy;
  result.total_energy_history.push_back(initial_gradient_result.scf_result.total_energy);
  const double initial_gradient_inf_norm = gradient_infinity_norm(
      initial_gradient_result.sparse_orbital_energy_gradient,
      differentiable_parameter_indices);
  result.gradient_inf_norm_history.push_back(initial_gradient_inf_norm);
  result.iteration_time_history_seconds.push_back(
      initial_gradient_result.total_wall_time_seconds);
  record_accepted_iteration_snapshot(
      current_input,
      initial_gradient_result,
      0,
      options_.optimizer_options,
      &result);

  double current_energy = initial_gradient_result.scf_result.total_energy;
  double current_gradient_inf_norm = initial_gradient_inf_norm;
  double current_gradient_l2_norm = gradient_l2_norm(
      initial_gradient_result.sparse_orbital_energy_gradient,
      differentiable_parameter_indices);
  int accepted_iterations = 0;
  bool converged = false;
  std::string pending_fallback_reason;

  if (current_gradient_inf_norm < options_.optimizer_options.gradient_tolerance) {
    converged = true;
    result.converged = true;
    result.termination_reason = "deepvbh_onnx_direct_final_initial_gradient_tolerance";
  }

  if (!converged) {
    const auto prediction = inference_runner_.predict(
        current_input,
        raw_structure_data,
        static_molecule_metadata,
        nuclear_repulsion_energy);
    const auto& direction = prediction.predicted_sparse_orbital_residual;
    if (direction.size() != current_input.orbital_preparation_input.orbital_value_table.size()) {
      pending_fallback_reason = "deepvbh_direct_final_invalid_direction_size";
    } else if (!is_finite_vector(direction, differentiable_parameter_indices)) {
      pending_fallback_reason = "deepvbh_direct_final_non_finite_direction";
    } else if (
        differentiable_direction_norm(direction, differentiable_parameter_indices) <=
        std::numeric_limits<double>::epsilon()) {
      pending_fallback_reason = "deepvbh_direct_final_zero_direction";
    } else {
      CppVbInput best_input;
      CppOrbitalGradientResult best_gradient_result;
      bool accepted = false;
      double best_step_scale = 0.0;
      double best_energy = current_energy;
      constexpr double kEnergyAcceptanceEpsilon = 1.0e-12;
      double trial_step_scale = options_.initial_step_scale;
      for (int backtrack = 0;
           backtrack < options_.max_backtracks &&
           trial_step_scale >= options_.minimum_step_scale;
           ++backtrack) {
        CppVbInput trial_input = apply_sparse_direction(
            current_input,
            direction,
            differentiable_parameter_indices,
            trial_step_scale);
        const auto trial_gradient_result = orbital_gradient_evaluator_.evaluate(
            trial_input,
            selected_state_indices,
            state_average_weights,
            nuclear_repulsion_energy);
        const double trial_energy = trial_gradient_result.scf_result.total_energy;
        if (trial_energy < best_energy - kEnergyAcceptanceEpsilon) {
          accepted = true;
          best_step_scale = trial_step_scale;
          best_energy = trial_energy;
          best_input = std::move(trial_input);
          best_gradient_result = trial_gradient_result;
        }
        trial_step_scale *= options_.step_shrink_factor;
      }

      if (!accepted) {
        pending_fallback_reason = "deepvbh_direct_final_exact_rejected_all_backtracks";
      } else {
        current_input = std::move(best_input);
        current_energy = best_gradient_result.scf_result.total_energy;
        current_gradient_inf_norm = gradient_infinity_norm(
            best_gradient_result.sparse_orbital_energy_gradient,
            differentiable_parameter_indices);
        current_gradient_l2_norm = gradient_l2_norm(
            best_gradient_result.sparse_orbital_energy_gradient,
            differentiable_parameter_indices);
        result.scf_result = best_gradient_result.scf_result;
        result.optimized_input = current_input;
        result.total_energy_history.push_back(current_energy);
        result.gradient_inf_norm_history.push_back(current_gradient_inf_norm);
        result.iteration_time_history_seconds.push_back(
            best_gradient_result.total_wall_time_seconds);

        accepted_iterations = 1;
        record_accepted_iteration_snapshot(
            current_input,
            best_gradient_result,
            accepted_iterations,
            options_.optimizer_options,
            &result);

        const double energy_change = current_energy - result.initial_total_energy;
        (void)best_step_scale;

        if (std::abs(energy_change) < options_.optimizer_options.energy_tolerance &&
            current_gradient_inf_norm < options_.optimizer_options.gradient_tolerance) {
          converged = true;
          result.converged = true;
          result.termination_reason = "deepvbh_onnx_direct_final_dual_tolerance";
        }
      }
    }
  }

  if (!converged && options_.exact_fallback_max_iterations > 0) {
    CppVbScfOptimizerOptions fallback_options = options_.optimizer_options;
    fallback_options.backend = CppVbScfOptimizerBackend::Lbfgspp;
    fallback_options.max_iterations = options_.exact_fallback_max_iterations;
    if (fallback_options.accepted_iteration_callback) {
      const auto original_callback = fallback_options.accepted_iteration_callback;
      const int iteration_offset = accepted_iterations;
      fallback_options.accepted_iteration_callback =
          [original_callback, iteration_offset](
              const CppVbScfAcceptedIterationSnapshot& snapshot) {
            if (snapshot.accepted_iteration_index == 0) {
              return;
            }
            auto shifted_snapshot = snapshot;
            shifted_snapshot.accepted_iteration_index += iteration_offset;
            original_callback(shifted_snapshot);
          };
    }
    CppVbScfOptimizer fallback_optimizer(
        orbital_gradient_evaluator_,
        scf_evaluator_,
        fallback_options);
    const auto fallback_result = fallback_optimizer.optimize(
        current_input,
        selected_state_indices,
        state_average_weights,
        nuclear_repulsion_energy);
    append_fallback_histories(fallback_result, &result);
    if (options_.optimizer_options.retain_accepted_iteration_trace) {
      append_fallback_trace(fallback_result, accepted_iterations, &result);
    }
    result.converged = fallback_result.converged;
    result.termination_reason =
        std::string("deepvbh_onnx_direct_final_fallback: ") +
        fallback_result.termination_reason;
    result.n_iterations = accepted_iterations + fallback_result.n_iterations;
    result.final_total_energy = fallback_result.final_total_energy;
    result.final_gradient_inf_norm = fallback_result.final_gradient_inf_norm;
    result.final_gradient_l2_norm = fallback_result.final_gradient_l2_norm;
    result.scf_result = fallback_result.scf_result;
    result.optimized_input = fallback_result.optimized_input;
  } else {
    result.n_iterations = accepted_iterations;
    result.final_total_energy = current_energy;
    result.final_gradient_inf_norm = current_gradient_inf_norm;
    result.final_gradient_l2_norm = current_gradient_l2_norm;
    result.optimized_input = current_input;
    if (result.termination_reason.empty()) {
      result.termination_reason =
          pending_fallback_reason.empty()
              ? "deepvbh_onnx_direct_final_stopped"
              : pending_fallback_reason;
    }
  }

  const auto optimization_end_time = std::chrono::steady_clock::now();
  result.total_wall_time_seconds =
      std::chrono::duration<double>(optimization_end_time - optimization_start_time).count();
  if (result.final_total_energy == 0.0 && !result.total_energy_history.empty()) {
    result.final_total_energy = result.total_energy_history.back();
  }
  if (result.final_gradient_inf_norm == 0.0 && !result.gradient_inf_norm_history.empty()) {
    result.final_gradient_inf_norm = result.gradient_inf_norm_history.back();
  }
  return result;
}

}  // namespace xmvb::vb
