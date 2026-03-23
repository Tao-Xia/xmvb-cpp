#include "vb/scf/cpp_vb_scf_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <LBFGS.h>

#ifdef XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
extern "C" void lbfgs_driver_(
    int* n,
    int* m,
    double* x,
    double* energy,
    double* gradient,
    int* diagco,
    double* diag,
    int* iprint,
    double* eps,
    double* xtol,
    double* workspace,
    int* iflag,
    double* gxn);
#endif

namespace xmvb::vb {

namespace {

double gradient_infinity_norm(const Eigen::VectorXd& gradient) {
  double norm = 0.0;
  for (Eigen::Index index = 0; index < gradient.size(); ++index) {
    norm = std::max(norm, std::abs(gradient[index]));
  }
  return norm;
}

void print_iteration_header() {
  std::cout << "cpp_vbscf_iterations\n";
  std::cout << "ITER           ENERGY               DE          GRAD_INF"
            << "             G_L2         DT_S\n";
  std::cout.flush();
}

void print_iteration_summary(
    int iteration_index,
    double energy,
    double energy_change,
    double gradient_inf_norm,
    double gradient_l2_norm,
    double iteration_seconds) {
  std::cout << std::setw(4) << iteration_index
            << "  " << std::setw(20) << std::setprecision(12) << std::fixed << energy
            << "  " << std::setw(16) << std::setprecision(8) << std::scientific << energy_change
            << "  " << std::setw(16) << std::setprecision(8) << gradient_inf_norm
            << "  " << std::setw(16) << std::setprecision(8) << gradient_l2_norm
            << "  " << std::setw(12) << std::setprecision(6) << std::fixed << iteration_seconds
            << '\n';
  std::cout.flush();
}

int get_sparse_coefficient_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<int> differentiable_parameter_indices;
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      differentiable_parameter_indices.push_back(
          orbital_index * orbital_preparation_input.n_basis_functions + coefficient_index);
    }
  }
  return differentiable_parameter_indices;
}

Eigen::VectorXd pack_parameter_vector(
    const CppVbInput& input,
    const std::vector<int>& differentiable_parameter_indices) {
  Eigen::VectorXd parameter_vector(
      static_cast<Eigen::Index>(differentiable_parameter_indices.size()));
  for (Eigen::Index parameter_offset = 0;
       parameter_offset < parameter_vector.size();
       ++parameter_offset) {
    const int parameter_index =
        differentiable_parameter_indices[static_cast<std::size_t>(parameter_offset)];
    parameter_vector[parameter_offset] =
        input.orbital_preparation_input.orbital_value_table[static_cast<std::size_t>(parameter_index)];
  }
  return parameter_vector;
}

void unpack_parameter_vector(
    const Eigen::VectorXd& parameter_vector,
    const std::vector<int>& differentiable_parameter_indices,
    CppVbInput* input) {
  for (Eigen::Index parameter_offset = 0;
       parameter_offset < parameter_vector.size();
       ++parameter_offset) {
    const int parameter_index =
        differentiable_parameter_indices[static_cast<std::size_t>(parameter_offset)];
    input->orbital_preparation_input.orbital_value_table[static_cast<std::size_t>(parameter_index)] =
        parameter_vector[parameter_offset];
  }
}

class OrbitalObjective {
public:
  OrbitalObjective(
      const CppVbInput& input,
      const std::vector<int>& differentiable_parameter_indices,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy,
      const CppOrbitalGradientEvaluator* orbital_gradient_evaluator)
      : working_input_(input),
        differentiable_parameter_indices_(differentiable_parameter_indices),
        selected_state_indices_(selected_state_indices),
        state_average_weights_(state_average_weights),
        nuclear_repulsion_energy_(nuclear_repulsion_energy),
        orbital_gradient_evaluator_(orbital_gradient_evaluator) {}

  double operator()(const Eigen::VectorXd& parameter_vector, Eigen::VectorXd& gradient) {
    const auto iteration_start_time = std::chrono::steady_clock::now();
    unpack_parameter_vector(parameter_vector, differentiable_parameter_indices_, &working_input_);

    const auto gradient_result = orbital_gradient_evaluator_->evaluate(
        working_input_,
        selected_state_indices_,
        state_average_weights_,
        nuclear_repulsion_energy_);

    gradient.resize(parameter_vector.size());
    for (Eigen::Index parameter_offset = 0;
         parameter_offset < parameter_vector.size();
         ++parameter_offset) {
      const int parameter_index =
          differentiable_parameter_indices_[static_cast<std::size_t>(parameter_offset)];
      gradient[parameter_offset] =
          gradient_result.sparse_orbital_energy_gradient[static_cast<std::size_t>(parameter_index)];
    }

    last_gradient_result_ = gradient_result;
    energy_history_.push_back(gradient_result.scf_result.total_energy);
    gradient_inf_norm_history_.push_back(gradient_infinity_norm(gradient));
    const auto iteration_end_time = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds =
        iteration_end_time - iteration_start_time;
    iteration_time_history_seconds_.push_back(elapsed_seconds.count());
    return gradient_result.scf_result.total_energy;
  }

  const CppVbInput& last_input() const { return working_input_; }
  const CppOrbitalGradientResult& last_gradient_result() const { return last_gradient_result_; }
  const std::vector<double>& energy_history() const { return energy_history_; }
  const std::vector<double>& gradient_inf_norm_history() const { return gradient_inf_norm_history_; }
  const std::vector<double>& iteration_time_history_seconds() const {
    return iteration_time_history_seconds_;
  }

  double last_gradient_inf_norm() const {
    if (gradient_inf_norm_history_.empty()) {
      return 0.0;
    }
    return gradient_inf_norm_history_.back();
  }

private:
  CppVbInput working_input_;
  std::vector<int> differentiable_parameter_indices_;
  std::vector<int> selected_state_indices_;
  std::vector<double> state_average_weights_;
  double nuclear_repulsion_energy_ = 0.0;
  const CppOrbitalGradientEvaluator* orbital_gradient_evaluator_ = nullptr;

  CppOrbitalGradientResult last_gradient_result_;
  std::vector<double> energy_history_;
  std::vector<double> gradient_inf_norm_history_;
  std::vector<double> iteration_time_history_seconds_;
};

void sync_result_from_objective(
    const OrbitalObjective& objective,
    CppVbScfOptimizerResult* result) {
  result->total_energy_history = objective.energy_history();
  result->gradient_inf_norm_history = objective.gradient_inf_norm_history();
  result->iteration_time_history_seconds = objective.iteration_time_history_seconds();
  result->scf_result = objective.last_gradient_result().scf_result;
}

CppVbScfAcceptedIterationSnapshot build_accepted_iteration_snapshot(
    const OrbitalObjective& objective,
    int accepted_iteration_index) {
  CppVbScfAcceptedIterationSnapshot snapshot;
  snapshot.accepted_iteration_index = accepted_iteration_index;
  snapshot.orbital_value_table.assign(
      objective.last_input().orbital_preparation_input.orbital_value_table.begin(),
      objective.last_input().orbital_preparation_input.orbital_value_table.end());
  snapshot.structure_matrices = objective.last_gradient_result().scf_result.structure_matrices;
  snapshot.sparse_orbital_energy_gradient =
      objective.last_gradient_result().sparse_orbital_energy_gradient;
  snapshot.sparse_orbital_reference_energy_gradient =
      objective.last_gradient_result().sparse_orbital_reference_energy_gradient;
  snapshot.total_energy = objective.last_gradient_result().scf_result.total_energy;
  snapshot.one_electron_reference_energy =
      objective.last_gradient_result().scf_result.one_electron_reference_energy;
  return snapshot;
}

void record_accepted_iteration_snapshot(
    const OrbitalObjective& objective,
    int accepted_iteration_index,
    const CppVbScfOptimizerOptions& options,
    CppVbScfOptimizerResult* result) {
  const auto snapshot =
      build_accepted_iteration_snapshot(objective, accepted_iteration_index);
  if (options.retain_accepted_iteration_trace) {
    result->accepted_iteration_trace.push_back(snapshot);
  }
  if (options.accepted_iteration_callback) {
    options.accepted_iteration_callback(snapshot);
  }
}

}  // namespace

CppVbScfOptimizer::CppVbScfOptimizer(
    CppVbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(options.algorithm),
      scf_evaluator_(options.algorithm),
      options_(options) {}

CppVbScfOptimizer::CppVbScfOptimizer(
    CppOrbitalGradientEvaluator orbital_gradient_evaluator,
    CppVbScfEvaluator scf_evaluator,
    CppVbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(std::move(orbital_gradient_evaluator)),
      scf_evaluator_(std::move(scf_evaluator)),
      options_(options) {}

CppVbScfOptimizerResult CppVbScfOptimizer::optimize(
    const CppVbInput& input,
    double nuclear_repulsion_energy) const {
  return optimize(input, {0}, {1.0}, nuclear_repulsion_energy);
}

CppVbScfOptimizerResult CppVbScfOptimizer::optimize(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (options_.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (options_.gradient_tolerance <= 0.0 ||
      options_.energy_tolerance <= 0.0 ||
      options_.initial_step_size <= 0.0 ||
      options_.minimum_step_size <= 0.0) {
    throw std::invalid_argument("optimizer tolerances and step sizes must be positive");
  }
  if (options_.minimum_step_size > options_.initial_step_size) {
    throw std::invalid_argument("minimum_step_size must not exceed initial_step_size");
  }
  if (options_.history_size <= 0) {
    throw std::invalid_argument("history_size must be positive");
  }
  if (!cpp_vb_scf_optimizer_backend_supported(options_.backend)) {
    throw std::invalid_argument(
        "requested optimizer backend is not enabled in this build");
  }
  if (options_.backend == CppVbScfOptimizerBackend::DeepVBHOnnx) {
    throw std::invalid_argument(
        "deepvbh_onnx requires DeepVBHOnnxHybridOptimizer and runtime metadata");
  }

  CppVbScfOptimizerResult result;
  const auto optimization_start_time = std::chrono::steady_clock::now();

  const auto differentiable_parameter_indices =
      collect_differentiable_parameter_indices(input.orbital_preparation_input);
  Eigen::VectorXd parameter_vector =
      pack_parameter_vector(input, differentiable_parameter_indices);
  if (parameter_vector.size() == 0) {
    throw std::invalid_argument("optimizer requires at least one differentiable parameter");
  }

  OrbitalObjective objective(
      input,
      differentiable_parameter_indices,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      &orbital_gradient_evaluator_);
  const int n = static_cast<int>(parameter_vector.size());
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;

  try {
    Eigen::VectorXd gradient(parameter_vector.size());
    double energy = objective(parameter_vector, gradient);
    sync_result_from_objective(objective, &result);
    record_accepted_iteration_snapshot(objective, 0, options_, &result);
    result.initial_total_energy = energy;
    double previous_energy = energy;
    bool has_previous_energy = true;
    final_gradient_l2_norm = gradient.norm();
    if (options_.verbose) {
      print_iteration_header();
    }

    switch (options_.backend) {
      case CppVbScfOptimizerBackend::LegacyFortran: {
#ifdef XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
        const int m = options_.history_size;
        std::vector<double> x(static_cast<std::size_t>(n));
        std::memcpy(x.data(), parameter_vector.data(), sizeof(double) * static_cast<std::size_t>(n));
        std::vector<double> gradient_buffer(static_cast<std::size_t>(n), 0.0);
        std::memcpy(
            gradient_buffer.data(),
            gradient.data(),
            sizeof(double) * static_cast<std::size_t>(n));
        std::vector<double> workspace(
            static_cast<std::size_t>(n) * static_cast<std::size_t>(2 * m + 1) +
                static_cast<std::size_t>(2 * m),
            0.0);
        std::vector<double> diag(static_cast<std::size_t>(n), 1.0);

        int diagco = 0;
        int iprint[2] = {-1, 0};
        int iflag = 0;
        double eps = 1.0e-5;
        double xtol = 1.0e-16;
        double gxn = final_gradient_l2_norm;

        auto evaluate_current_point = [&]() {
          Eigen::Map<Eigen::VectorXd> parameter_map(x.data(), n);
          Eigen::VectorXd gradient_map;
          energy = objective(parameter_map, gradient_map);
          std::memcpy(
              gradient_buffer.data(),
              gradient_map.data(),
              sizeof(double) * static_cast<std::size_t>(n));
          sync_result_from_objective(objective, &result);
          final_gradient_l2_norm = gradient_map.norm();
        };

        while (true) {
          lbfgs_driver_(
              const_cast<int*>(&n),
              const_cast<int*>(&m),
              x.data(),
              &energy,
              gradient_buffer.data(),
              &diagco,
              diag.data(),
              iprint,
              &eps,
              &xtol,
              workspace.data(),
              &iflag,
              &gxn);

          if (iflag == 1) {
            evaluate_current_point();
            ++n_iterations;
            record_accepted_iteration_snapshot(objective, n_iterations, options_, &result);

            const double de = has_previous_energy ? (energy - previous_energy) : 0.0;
            previous_energy = energy;
            has_previous_energy = true;
            if (options_.verbose) {
              print_iteration_summary(
                  n_iterations,
                  energy,
                  de,
                  result.gradient_inf_norm_history.back(),
                  gxn,
                  result.iteration_time_history_seconds.back());
            }

            if (std::abs(de) < options_.energy_tolerance &&
                gxn < options_.gradient_tolerance) {
              result.converged = true;
              result.termination_reason = "legacy_dual_tolerance";
              iflag = 0;
              break;
            }

            if (n_iterations >= options_.max_iterations) {
              result.termination_reason = "max_iterations";
              break;
            }

            continue;
          }

          if (iflag == 0) {
            result.converged = true;
            if (result.termination_reason.empty()) {
              result.termination_reason = "lbfgs_driver_finished";
            }
            break;
          }

          if (iflag == 2) {
            result.termination_reason = "unexpected_diag_request";
            break;
          }

          result.termination_reason = "lbfgs_driver_error";
          break;
        }
        final_gradient_l2_norm = gxn;
        break;
#else
        result.termination_reason = "legacy_fortran_backend_disabled";
        break;
#endif
      }

      case CppVbScfOptimizerBackend::Lbfgspp: {
        LBFGSpp::LBFGSParam<double> param;
        param.m = options_.history_size;
        param.epsilon = 0.0;
        param.epsilon_rel = 0.0;
        param.past = 0;
        param.delta = 0.0;
        param.max_iterations = 0;
        param.max_linesearch = 20;
        param.min_step = options_.minimum_step_size;
        param.max_step = options_.initial_step_size;
        param.ftol = options_.armijo_constant;
        param.wolfe = 0.9;
        param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
        param.check_param();

        LBFGSpp::BFGSMat<double> inverse_hessian;
        inverse_hessian.reset(n, param.m);

        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        Eigen::VectorXd previous_parameters(n);
        Eigen::VectorXd previous_gradient(n);
        Eigen::VectorXd search_direction = -current_gradient;
        constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (search_direction.dot(current_gradient) >= 0.0) {
            search_direction = -current_gradient;
          }
          double directional_derivative = current_gradient.dot(search_direction);
          if (directional_derivative >= 0.0) {
            result.termination_reason = "lbfgspp_non_descent_direction";
            break;
          }

          previous_parameters = current_parameters;
          previous_gradient = current_gradient;
          double step = std::min(1.0, options_.initial_step_size);

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
            sync_result_from_objective(objective, &result);
            final_gradient_l2_norm = current_gradient.norm();
            result.termination_reason = std::string("lbfgspp_line_search: ") + error.what();
            break;
          }

          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();

          const double de = has_previous_energy ? (energy - previous_energy) : 0.0;
          previous_energy = energy;
          has_previous_energy = true;
          if (options_.verbose) {
            print_iteration_summary(
                n_iterations,
                energy,
                de,
                result.gradient_inf_norm_history.back(),
                final_gradient_l2_norm,
                result.iteration_time_history_seconds.back());
          }

          if (std::abs(de) < options_.energy_tolerance &&
              result.gradient_inf_norm_history.back() < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason = "lbfgspp_dual_tolerance";
            break;
          }

          Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
          Eigen::VectorXd gradient_step = current_gradient - previous_gradient;
          if (parameter_step.dot(gradient_step) >
              kCurvatureEpsilon * gradient_step.squaredNorm()) {
            inverse_hessian.add_correction(parameter_step, gradient_step);
          }
          inverse_hessian.apply_Hv(current_gradient, -1.0, search_direction);
        }
        break;
      }

      case CppVbScfOptimizerBackend::DeepVBHOnnx:
        result.termination_reason =
            "deepvbh_onnx_requires_hybrid_optimizer";
        break;
    }
  } catch (const std::exception& error) {
    result.termination_reason = error.what();
  }

  if (result.total_energy_history.empty()) {
    throw std::runtime_error("optimizer did not evaluate the objective");
  }

  result.n_iterations = n_iterations;
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
  const std::chrono::duration<double> total_elapsed_seconds =
      optimization_end_time - optimization_start_time;
  result.total_wall_time_seconds = total_elapsed_seconds.count();

  return result;
}

}  // namespace xmvb::vb
