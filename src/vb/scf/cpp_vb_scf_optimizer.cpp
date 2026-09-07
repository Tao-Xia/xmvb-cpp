#include "vb/scf/cpp_vb_scf_optimizer.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <LBFGS.h>

#include "vb/orbital/localized_representative_selector.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/runtime_utils.hpp"
#include "vb/orbital/support_aware_mo_gauge_fix.hpp"
#include "vb/scf/exact_orbital_second_order_operator.hpp"
#include "vb/scf/orbital_objective.hpp"
#include "vb/scf/optimizer_types.hpp"
#include "vb/scf/scf_vector_utilities.hpp"

namespace xmvb::vb {

namespace {

constexpr double kLineSearchExpansionFactor = 10.0;

bool optimizer_backend_uses_nonredundant_space(
    CppVbScfOptimizerBackend backend) {
  switch (backend) {
    case CppVbScfOptimizerBackend::NonredundantProjectedGradient:
    case CppVbScfOptimizerBackend::NonredundantLbfgspp:
    case CppVbScfOptimizerBackend::NonredundantTruncatedNewton:
      return true;
    case CppVbScfOptimizerBackend::Lbfgspp:
    case CppVbScfOptimizerBackend::DeepVBHOnnx:
    case CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal:
      return false;
  }
  return false;
}


const char* bool_name(bool value) {
  return value ? "true" : "false";
}

double inexact_newton_forcing_term(double gradient_inf_norm) {
  if (!std::isfinite(gradient_inf_norm) || gradient_inf_norm <= 0.0) {
    return 0.5;
  }
  // Dembo-Eisenstat-Steihaug style inexact Newton forcing. Far from a
  // stationary point the linear system is deliberately solved loosely; the
  // tolerance tightens continuously as the nonlinear gradient contracts.
  // This replaces molecule/chart-specific Krylov budgets with a residual
  // condition intrinsic to the current optimization state.
  constexpr double kMinimumForcingTerm = 1.0e-3;
  constexpr double kMaximumForcingTerm = 0.5;
  return std::clamp(
      std::sqrt(gradient_inf_norm),
      kMinimumForcingTerm,
      kMaximumForcingTerm);
}

NonredundantOrbitalSpace build_nonredundant_space(
    const OrbitalObjective& objective,
    const SparseOrbitalParameterView& parameter_view) {
  const auto& orbital_preparation_input =
      objective.last_input().orbital_preparation_input;
  const auto& orbital_preparation_result =
      objective.last_gradient_result().orbital_preparation_result;
  const auto& normalized_orbital_matrix =
      orbital_preparation_result.physical_orbital_frame.normalized_orbital_matrix;
  if (normalized_orbital_matrix.size() == 0) {
    throw std::runtime_error(
        "nonredundant space requires the cached physical orbital frame");
  }
  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_occupied_orbitals =
      n_inactive_doubly_occupied_orbitals +
      orbital_preparation_input.n_active_orbitals;
  return NonredundantOrbitalSpace(
      orbital_preparation_input,
      parameter_view,
      orbital_preparation_result.auxiliary_orbital_matrix.leftCols(n_occupied_orbitals),
      normalized_orbital_matrix,
      &objective.last_gradient_result().ao_effective_one_electron_result.ao_effective_h1e);
}

Eigen::VectorXd build_nonredundant_preconditioned_gradient_direction(
    const NonredundantOrbitalSpace& space,
    const NonredundantOrbitalSpace::ProjectionResult& projection) {
  // The reduced coordinates are either the orthogonalized tangent chart or the
  // direct nonredundant orbital-replacement amplitudes used by the full-AO
  // fast path. Use the full block preconditioner rather than only the
  // curvature diagonal so large reduced sparse blocks keep the accepted-point
  // block metric in the search direction.
  return -space.expand_step(
      space.apply_inverse_reduced_block_preconditioner(
          projection.reduced_gradient));
}

// The optimizer objective lives in packed differentiable sparse coefficients,
// while `expand_retract_input_tangent()` returns the full stored-orbital
// tangent. Gather the actual finite-retraction tangent back to packed
// coordinates whenever a directional derivative or finite-difference scale must
// match `retract_step()` rather than the raw additive chart.
Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  const Eigen::VectorXd full_tangent =
      space.expand_retract_input_tangent(
          orbital_preparation_input,
          reduced_step);
  Eigen::VectorXd packed_tangent =
      Eigen::VectorXd::Zero(
          static_cast<Eigen::Index>(parameter_view.size()));
  const auto& differentiable_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_tangent.size();
       ++packed_index) {
    packed_tangent[packed_index] =
        full_tangent[differentiable_indices[packed_index]];
  }
  return packed_tangent;
}

double compute_nonredundant_retract_tangent_norm(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  if (reduced_step.size() == 0) {
    return 0.0;
  }
  const double tangent_norm =
      gather_nonredundant_retract_tangent(
          orbital_preparation_input,
          space,
          parameter_view,
          reduced_step)
          .norm();
  return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
}

class NonredundantRetractionMetric {
public:
  NonredundantRetractionMetric(
      const OrbitalPreparationInput& orbital_preparation_input,
      const NonredundantOrbitalSpace& space,
      const SparseOrbitalParameterView& parameter_view)
      : orbital_preparation_input_(orbital_preparation_input),
        space_(space),
        parameter_view_(parameter_view) {}

  Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const {
    return gather_nonredundant_retract_tangent(
        orbital_preparation_input_,
        space_,
        parameter_view_,
        reduced_step);
  }

  double norm(const Eigen::VectorXd& reduced_step) const {
    const double tangent_norm = tangent(reduced_step).norm();
    return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
  }

  Eigen::VectorXd clip_to_radius(
      const Eigen::VectorXd& reduced_step,
      double trust_radius) const {
    if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
      return Eigen::VectorXd::Zero(reduced_step.size());
    }
    const double tangent_norm = norm(reduced_step);
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
      return Eigen::VectorXd::Zero(reduced_step.size());
    }
    if (tangent_norm <= trust_radius) {
      return reduced_step;
    }
    return (trust_radius / tangent_norm) * reduced_step;
  }

private:
  const OrbitalPreparationInput& orbital_preparation_input_;
  const NonredundantOrbitalSpace& space_;
  const SparseOrbitalParameterView& parameter_view_;
};

Eigen::VectorXd clip_nonredundant_reduced_step_to_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step,
    double trust_radius) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm =
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          space,
          parameter_view,
          reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm <= trust_radius) {
    return reduced_step;
  }
  return (trust_radius / tangent_norm) * reduced_step;
}

Eigen::VectorXd shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step,
    double trust_radius) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  constexpr double kInitialStepSafetyFraction = 0.95;
  const double target_radius = kInitialStepSafetyFraction * trust_radius;
  if (!(target_radius > 0.0) || !std::isfinite(target_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm =
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          space,
          parameter_view,
          reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm < target_radius) {
    return reduced_step;
  }
  return (target_radius / tangent_norm) * reduced_step;
}

int choose_nonredundant_truncated_newton_max_cg_iterations(
    const CppVbScfOptimizerOptions& options,
    int reduced_size) {
  if (options.nonredundant_truncated_newton_max_cg_iterations > 0) {
    return std::min(
        std::max(1, reduced_size),
        options.nonredundant_truncated_newton_max_cg_iterations);
  }
  constexpr int kDefaultKrylovSafetyLimit = 32;
  return std::min(std::max(1, reduced_size), kDefaultKrylovSafetyLimit);
}

int choose_nonredundant_truncated_newton_transport_history_size(
    const CppVbScfOptimizerOptions& options,
    const OrbitalObjective& objective) {
  (void)objective;
  return std::max(
      0,
      options.nonredundant_truncated_newton_transport_history_size);
}

class TransportedReducedLbfgsPreconditioner {
public:
  explicit TransportedReducedLbfgsPreconditioner(
      const NonredundantOrbitalSpace* space)
      : space_(space) {}

  bool try_add_pair(
      Eigen::VectorXd reduced_step,
      Eigen::VectorXd reduced_gradient_change) {
    const double step_norm = reduced_step.norm();
    const double gradient_change_norm = reduced_gradient_change.norm();
    const double secant_curvature =
        reduced_step.dot(reduced_gradient_change);
    const double minimum_secant_alignment =
        std::sqrt(std::numeric_limits<double>::epsilon());
    if (!(step_norm > 0.0) ||
        !(gradient_change_norm > 0.0) ||
        !std::isfinite(step_norm) ||
        !std::isfinite(gradient_change_norm) ||
        !std::isfinite(secant_curvature) ||
        secant_curvature <=
            minimum_secant_alignment * step_norm * gradient_change_norm) {
      return false;
    }

    Pair pair;
    pair.reduced_step = std::move(reduced_step);
    pair.reduced_gradient_change = std::move(reduced_gradient_change);
    pair.inverse_curvature = 1.0 / secant_curvature;
    pairs_.push_back(std::move(pair));
    return true;
  }

  bool empty() const noexcept {
    return pairs_.empty();
  }

  int size() const noexcept {
    return static_cast<int>(pairs_.size());
  }

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_vector) const {
    if (pairs_.empty()) {
      return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
    }

    Eigen::VectorXd q = reduced_vector;
    std::vector<double> alphas(pairs_.size(), 0.0);
    for (std::size_t pair_index = pairs_.size(); pair_index-- > 0;) {
      const auto& pair = pairs_[pair_index];
      const double alpha =
          pair.inverse_curvature * pair.reduced_step.dot(q);
      if (!std::isfinite(alpha)) {
        return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
      }
      alphas[pair_index] = alpha;
      q.noalias() -= alpha * pair.reduced_gradient_change;
    }

    Eigen::VectorXd z =
        space_->apply_inverse_reduced_block_preconditioner(q);
    for (std::size_t pair_index = 0;
         pair_index < pairs_.size();
         ++pair_index) {
      const auto& pair = pairs_[pair_index];
      const double beta =
          pair.inverse_curvature * pair.reduced_gradient_change.dot(z);
      if (!std::isfinite(beta)) {
        return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
      }
      z.noalias() += pair.reduced_step * (alphas[pair_index] - beta);
    }
    return z;
  }

private:
  struct Pair {
    Eigen::VectorXd reduced_step;
    Eigen::VectorXd reduced_gradient_change;
    double inverse_curvature = 0.0;
  };

  const NonredundantOrbitalSpace* space_ = nullptr;
  std::vector<Pair> pairs_;
};

TransportedReducedLbfgsPreconditioner
build_nonredundant_truncated_newton_preconditioner(
    const NonredundantOrbitalSpace& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size) {
  TransportedReducedLbfgsPreconditioner preconditioner(&current_space);
  if (max_history_size <= 0 || packed_secant_history.empty()) {
    return preconditioner;
  }

  const std::size_t history_begin =
      packed_secant_history.size() >
              static_cast<std::size_t>(max_history_size)
          ? packed_secant_history.size() -
                static_cast<std::size_t>(max_history_size)
          : 0;
  for (std::size_t pair_index = history_begin;
       pair_index < packed_secant_history.size();
       ++pair_index) {
    const auto& packed_pair = packed_secant_history[pair_index];
    // The reduced basis changes after every accepted orbital update. Reproject
    // each ambient packed secant pair into the current tangent space so the
    // L-BFGS recursion only uses directions that still survive the latest
    // nonredundant parameterization.
    preconditioner.try_add_pair(
        current_space.project_vector(packed_pair.packed_step).reduced_gradient,
        current_space
            .project_vector(packed_pair.packed_projected_gradient_change)
            .reduced_gradient);
  }
  return preconditioner;
}

Eigen::VectorXd apply_nonredundant_truncated_newton_preconditioner(
    const NonredundantOrbitalSpace& current_space,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd& reduced_vector) {
  if (transported_preconditioner == nullptr ||
      transported_preconditioner->empty()) {
    return current_space.apply_inverse_reduced_block_preconditioner(
        reduced_vector);
  }

  const Eigen::VectorXd preconditioned =
      transported_preconditioner->apply(reduced_vector);
  const double curvature = reduced_vector.dot(preconditioned);
  if (!std::isfinite(curvature) || curvature <= 0.0) {
    return current_space.apply_inverse_reduced_block_preconditioner(
        reduced_vector);
  }
  return preconditioned;
}

void append_nonredundant_truncated_newton_secant_pair(
    Eigen::VectorXd packed_step,
    Eigen::VectorXd packed_projected_gradient_change,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history) {
  if (max_history_size <= 0) {
    return;
  }
  const double step_norm = packed_step.norm();
  const double gradient_change_norm = packed_projected_gradient_change.norm();
  const double secant_curvature =
      packed_step.dot(packed_projected_gradient_change);
  const double minimum_secant_alignment =
      std::sqrt(std::numeric_limits<double>::epsilon());
  if (!(step_norm > 0.0) ||
      !(gradient_change_norm > 0.0) ||
      !std::isfinite(step_norm) ||
      !std::isfinite(gradient_change_norm) ||
      !std::isfinite(secant_curvature) ||
      secant_curvature <=
          minimum_secant_alignment * step_norm * gradient_change_norm) {
    return;
  }

  packed_secant_history->push_back(
      PackedSecantPair{
          std::move(packed_step),
          std::move(packed_projected_gradient_change)});
  if (packed_secant_history->size() >
      static_cast<std::size_t>(max_history_size)) {
    packed_secant_history->erase(packed_secant_history->begin());
  }
}

double solve_trust_region_metric_boundary_tau(
    const Eigen::VectorXd& current_step_tangent,
    const Eigen::VectorXd& search_direction_tangent,
    double trust_radius) {
  if (current_step_tangent.size() != search_direction_tangent.size()) {
    return 0.0;
  }
  const double a = search_direction_tangent.squaredNorm();
  if (!(a > 0.0) || !std::isfinite(a) ||
      !(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return 0.0;
  }
  // Solve ||J(s + tau p)||^2 = Delta^2 without materializing G = J^T J.
  const double b = current_step_tangent.dot(search_direction_tangent);
  const double c =
      current_step_tangent.squaredNorm() - trust_radius * trust_radius;
  const double discriminant = std::max(0.0, b * b - a * c);
  const double tau =
      (-b + std::sqrt(discriminant)) / a;
  if (!std::isfinite(tau)) {
    return 0.0;
  }
  return std::max(0.0, tau);
}

class ReducedHvpOperator {
public:
  virtual ~ReducedHvpOperator() = default;
  virtual Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) = 0;
  virtual Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
    Eigen::MatrixXd responses(
        reduced_directions.rows(),
        reduced_directions.cols());
    for (Eigen::Index column = 0;
         column < reduced_directions.cols();
         ++column) {
      responses.col(column) = apply(reduced_directions.col(column));
    }
    return responses;
  }
};

class FullFiniteDifferenceReducedHvpOperator final : public ReducedHvpOperator {
public:
  FullFiniteDifferenceReducedHvpOperator(
      const OrbitalObjective& objective,
      const NonredundantOrbitalSpace& current_space,
      const NonredundantOrbitalSpace::ProjectionResult& current_projection,
      const OrbitalPreparationInput& current_orbital_input,
      const SparseOrbitalParameterView& parameter_view,
      const Eigen::VectorXd& current_parameters,
      double hvp_step_size)
      : probe_objective_(objective.make_probe_copy()),
        current_space_(current_space),
        current_reduced_gradient_(current_projection.reduced_gradient),
        current_orbital_input_(current_orbital_input),
        parameter_view_(parameter_view),
        current_parameters_(current_parameters),
        hvp_step_size_(hvp_step_size) {}

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override {
    if (reduced_direction.size() == 0) {
      return Eigen::VectorXd::Zero(0);
    }

    const Eigen::VectorXd packed_direction =
        gather_nonredundant_retract_tangent(
            current_orbital_input_,
            current_space_,
            parameter_view_,
            reduced_direction);
    const double packed_direction_norm = packed_direction.norm();
    if (!(packed_direction_norm > 0.0) || !std::isfinite(packed_direction_norm)) {
      return Eigen::VectorXd::Zero(reduced_direction.size());
    }

    const double epsilon =
        hvp_step_size_ / std::max(1.0, packed_direction_norm);
    if (!(epsilon > 0.0) || !std::isfinite(epsilon)) {
      throw std::runtime_error("invalid finite-difference step for reduced HVP");
    }

    const OrbitalPreparationInput trial_orbital_input =
        current_space_.retract_step(
            current_orbital_input_,
            reduced_direction,
            epsilon);
    const Eigen::VectorXd trial_parameters =
        parameter_view_.pack(trial_orbital_input);
    const OrbitalObjective::TrialEvaluation trial_evaluation =
        probe_objective_.evaluate_trial_without_committing(trial_parameters);
    return
        (current_space_.project_reduced_gradient(trial_evaluation.gradient) -
         current_reduced_gradient_) /
        epsilon;
  }

private:
  OrbitalObjective probe_objective_;
  const NonredundantOrbitalSpace& current_space_;
  Eigen::VectorXd current_reduced_gradient_;
  OrbitalPreparationInput current_orbital_input_;
  SparseOrbitalParameterView parameter_view_;
  Eigen::VectorXd current_parameters_;
  double hvp_step_size_ = 0.0;
};

class ExactContextReducedHvpOperator final : public ReducedHvpOperator {
public:
  ExactContextReducedHvpOperator(
      const OrbitalObjective& objective,
      const NonredundantOrbitalSpace& current_space)
      : exact_operator_(
            objective.last_second_order_context(),
            &objective.last_input(),
            SparseOrbitalParameterView(
                objective.last_input().orbital_preparation_input),
            &current_space) {}

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override {
    return exact_operator_.apply_reduced(reduced_direction);
  }

  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) override {
    return exact_operator_.apply_reduced_batch(reduced_directions);
  }

  bool supports_analytic_core_model() const noexcept {
    return exact_operator_.supports_analytic_core_model();
  }

  ExactOrbitalSecondOrderOperator::Diagnostics diagnostics() const {
    return exact_operator_.diagnostics();
  }

private:
  ExactOrbitalSecondOrderOperator exact_operator_;
};

std::string build_exact_ctx_unavailable_message(
    const ExactContextReducedHvpOperator& hvp_operator) {
  const auto info = hvp_operator.diagnostics();
  std::ostringstream message;
  message << "exact_ctx HVP is unavailable"
          << ": supports_analytic_core_model="
          << bool_name(info.supports_analytic_core_model)
          << " outer_response_enabled="
          << bool_name(info.outer_response_enabled)
          << " has_same_spin_matrix_form="
          << bool_name(info.has_same_spin_matrix_form)
          << " has_opposite_spin_matrix_form="
          << bool_name(info.has_opposite_spin_matrix_form)
          << " n_selected_states=" << info.n_selected_states
          << " n_active_orbitals=" << info.n_active_orbitals
          << " n_blocks=" << info.n_blocks;
  return message.str();
}

struct TruncatedNewtonKrylovSubspace {
  Eigen::MatrixXd orthonormal_basis;
  Eigen::MatrixXd tangent_basis;
  Eigen::MatrixXd hessian_basis;
  Eigen::MatrixXd reduced_hessian;
  Eigen::VectorXd projected_gradient;
};

struct TruncatedNewtonStepResult {
  Eigen::VectorXd reduced_step;
  Eigen::VectorXd reduced_hessian_times_step;
  TruncatedNewtonKrylovSubspace krylov_subspace;
  double retract_tangent_norm = 0.0;
  bool reached_boundary = false;
  bool encountered_negative_curvature = false;
  bool used_krylov_rescue = false;
  int cg_iterations = 0;
  double projected_model_gradient_norm = 0.0;
  double model_spectral_radius = 0.0;
  double trust_region_shift = 0.0;
  double predicted_decrease = 0.0;
};

double truncated_newton_step_effective_norm(
    const TruncatedNewtonStepResult& step) {
  if (std::isfinite(step.retract_tangent_norm) &&
      step.retract_tangent_norm > 0.0) {
    return step.retract_tangent_norm;
  }
  const double reduced_norm = step.reduced_step.norm();
  return std::isfinite(reduced_norm) ? reduced_norm : 0.0;
}

void clamp_nonredundant_step_result_to_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& current_space,
    const SparseOrbitalParameterView& parameter_view,
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    TruncatedNewtonStepResult* step) {
  if (step == nullptr ||
      step->reduced_step.size() != current_projection.reduced_gradient.size() ||
      step->reduced_step.size() == 0 ||
      !step->reduced_step.allFinite()) {
    return;
  }

  const double tangent_norm =
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          current_space,
          parameter_view,
          step->reduced_step);
  step->retract_tangent_norm = tangent_norm;
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !(tangent_norm > 0.0) ||
      !std::isfinite(tangent_norm)) {
    return;
  }

  if (tangent_norm > trust_radius) {
    const double scale = trust_radius / tangent_norm;
    step->reduced_step *= scale;
    if (step->reduced_hessian_times_step.size() ==
            current_projection.reduced_gradient.size() &&
        step->reduced_hessian_times_step.allFinite()) {
      step->reduced_hessian_times_step *= scale;
      step->predicted_decrease =
          -current_projection.reduced_gradient.dot(step->reduced_step) -
          0.5 * step->reduced_step.dot(step->reduced_hessian_times_step);
    } else {
      step->predicted_decrease = 0.0;
    }
    step->retract_tangent_norm = trust_radius;
    step->reached_boundary = true;
    return;
  }

  step->reached_boundary =
      step->reached_boundary ||
      tangent_norm >= (1.0 - 1.0e-8) * trust_radius;
}

struct TruncatedNewtonTrialEvaluation {
  double actual_decrease = 0.0;
  double predicted_decrease = 0.0;
};

struct RejectedTruncatedNewtonStepCache {
  Eigen::VectorXd cached_step;

  bool has_cached_step(Eigen::Index expected_size) const {
    return finite_nonzero_vector_matches_size(cached_step, expected_size);
  }

  void clear() {
    cached_step.resize(0);
  }

  void update(
      const OrbitalPreparationInput& orbital_preparation_input,
      const NonredundantOrbitalSpace& current_space,
      const SparseOrbitalParameterView& parameter_view,
      const TruncatedNewtonStepResult& model_step,
      Eigen::Index expected_size,
      double trust_radius) {
    cached_step =
        finite_nonzero_vector_matches_size(
                model_step.reduced_step,
                expected_size)
            ? shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
                  orbital_preparation_input,
                  current_space,
                  parameter_view,
                  model_step.reduced_step,
                  trust_radius)
            : Eigen::VectorXd();
  }
};

bool truncated_newton_krylov_subspace_is_usable(
    const TruncatedNewtonKrylovSubspace& krylov_subspace,
    Eigen::Index reduced_size) {
  return
      reduced_size >= 0 &&
      krylov_subspace.orthonormal_basis.rows() == reduced_size &&
      krylov_subspace.orthonormal_basis.cols() > 0 &&
      krylov_subspace.orthonormal_basis.allFinite() &&
      krylov_subspace.tangent_basis.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.tangent_basis.allFinite() &&
      krylov_subspace.hessian_basis.rows() == reduced_size &&
      krylov_subspace.hessian_basis.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.hessian_basis.allFinite() &&
      krylov_subspace.reduced_hessian.rows() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.reduced_hessian.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.reduced_hessian.allFinite() &&
      krylov_subspace.projected_gradient.size() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.projected_gradient.allFinite();
}

bool truncated_newton_step_is_usable(
    const TruncatedNewtonStepResult& step,
    const Eigen::VectorXd& reduced_gradient) {
  return
      step.reduced_step.size() == reduced_gradient.size() &&
      step.reduced_step.allFinite() &&
      step.reduced_step.squaredNorm() > 0.0 &&
      std::isfinite(step.predicted_decrease) &&
      step.predicted_decrease > 0.0 &&
      reduced_gradient.dot(step.reduced_step) < 0.0;
}

bool append_truncated_newton_krylov_basis_vector(
    const NonredundantRetractionMetric& retraction_metric,
    const Eigen::VectorXd& candidate_vector,
    const Eigen::VectorXd& hessian_times_candidate,
    std::vector<Eigen::VectorXd>* basis_vectors,
    std::vector<Eigen::VectorXd>* tangent_basis_vectors,
    std::vector<Eigen::VectorXd>* hessian_basis_vectors) {
  if (basis_vectors == nullptr ||
      tangent_basis_vectors == nullptr ||
      hessian_basis_vectors == nullptr ||
      candidate_vector.size() != hessian_times_candidate.size() ||
      candidate_vector.size() == 0 ||
      !candidate_vector.allFinite() ||
      !hessian_times_candidate.allFinite()) {
    return false;
  }

  Eigen::VectorXd orthogonal_vector = candidate_vector;
  Eigen::VectorXd orthogonal_hessian_vector = hessian_times_candidate;
  Eigen::VectorXd orthogonal_tangent =
      retraction_metric.tangent(candidate_vector);
  const double candidate_norm = orthogonal_tangent.norm();
  if (!(candidate_norm > 0.0) || !std::isfinite(candidate_norm)) {
    return false;
  }

  // Orthonormalize in the accepted-point retraction metric
  //   <u,v>_G = (J u)^T (J v).
  // The reduced basis columns are G-orthonormal, while the cached tangent
  // columns keep reorthogonalization and cached Ritz solves matrix-free.
  for (int orthogonalization_pass = 0;
       orthogonalization_pass < 2;
       ++orthogonalization_pass) {
    for (std::size_t basis_index = 0;
         basis_index < basis_vectors->size();
         ++basis_index) {
      const double coefficient =
          (*tangent_basis_vectors)[basis_index].dot(orthogonal_tangent);
      if (!std::isfinite(coefficient)) {
        return false;
      }
      orthogonal_vector.noalias() -=
          coefficient * (*basis_vectors)[basis_index];
      orthogonal_tangent.noalias() -=
          coefficient * (*tangent_basis_vectors)[basis_index];
      orthogonal_hessian_vector.noalias() -=
          coefficient * (*hessian_basis_vectors)[basis_index];
    }
  }

  const double linear_dependence_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon());
  const double orthogonal_norm = orthogonal_tangent.norm();
  if (!(orthogonal_norm >
        linear_dependence_tolerance * candidate_norm) ||
      !std::isfinite(orthogonal_norm)) {
    return false;
  }

  basis_vectors->push_back(orthogonal_vector / orthogonal_norm);
  tangent_basis_vectors->push_back(orthogonal_tangent / orthogonal_norm);
  hessian_basis_vectors->push_back(orthogonal_hessian_vector / orthogonal_norm);
  return true;
}

TruncatedNewtonKrylovSubspace build_truncated_newton_krylov_subspace(
    const Eigen::VectorXd& reduced_gradient,
    const std::vector<Eigen::VectorXd>& basis_vectors,
    const std::vector<Eigen::VectorXd>& tangent_basis_vectors,
    const std::vector<Eigen::VectorXd>& hessian_basis_vectors) {
  TruncatedNewtonKrylovSubspace krylov_subspace;
  if (basis_vectors.empty() ||
      basis_vectors.size() != tangent_basis_vectors.size() ||
      basis_vectors.size() != hessian_basis_vectors.size()) {
    return krylov_subspace;
  }

  const Eigen::Index reduced_size = reduced_gradient.size();
  const Eigen::Index basis_size =
      static_cast<Eigen::Index>(basis_vectors.size());
  krylov_subspace.orthonormal_basis.resize(reduced_size, basis_size);
  krylov_subspace.tangent_basis.resize(
      tangent_basis_vectors.front().size(),
      basis_size);
  krylov_subspace.hessian_basis.resize(reduced_size, basis_size);
  krylov_subspace.reduced_hessian.resize(basis_size, basis_size);
  krylov_subspace.projected_gradient.resize(basis_size);

  for (Eigen::Index column = 0; column < basis_size; ++column) {
    krylov_subspace.orthonormal_basis.col(column) =
        basis_vectors[column];
    krylov_subspace.tangent_basis.col(column) =
        tangent_basis_vectors[column];
    krylov_subspace.hessian_basis.col(column) =
        hessian_basis_vectors[column];
    krylov_subspace.projected_gradient[column] =
        basis_vectors[column].dot(reduced_gradient);
  }

  for (Eigen::Index row = 0; row < basis_size; ++row) {
    for (Eigen::Index column = 0; column < basis_size; ++column) {
      krylov_subspace.reduced_hessian(row, column) =
          basis_vectors[row].dot(
              hessian_basis_vectors[column]);
    }
  }
  krylov_subspace.reduced_hessian =
      0.5 *
      (krylov_subspace.reduced_hessian +
       krylov_subspace.reduced_hessian.transpose());
  if (!truncated_newton_krylov_subspace_is_usable(
          krylov_subspace,
          reduced_size)) {
    return TruncatedNewtonKrylovSubspace();
  }
  return krylov_subspace;
}

TruncatedNewtonStepResult solve_trust_region_in_krylov_subspace(
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    const TruncatedNewtonKrylovSubspace& krylov_subspace) {
  TruncatedNewtonStepResult result;
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !truncated_newton_krylov_subspace_is_usable(
          krylov_subspace,
          current_projection.reduced_gradient.size())) {
    return result;
  }

  const Eigen::MatrixXd reduced_hessian =
      0.5 *
      (krylov_subspace.reduced_hessian +
       krylov_subspace.reduced_hessian.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(reduced_hessian);
  if (eigensolver.info() != Eigen::Success) {
    return result;
  }

  const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
  const Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
  if (eigenvalues.size() == 0 ||
      !eigenvalues.allFinite() ||
      !eigenvectors.allFinite()) {
    return result;
  }
  const Eigen::VectorXd projected_gradient_in_eigenbasis =
      eigenvectors.transpose() * krylov_subspace.projected_gradient;
  if (!projected_gradient_in_eigenbasis.allFinite()) {
    return result;
  }

  const double radius_squared = trust_radius * trust_radius;
  const double spectral_scale = reduced_hessian.cwiseAbs().maxCoeff();
  constexpr double kShiftToleranceFactor =
      64.0 * std::numeric_limits<double>::epsilon();
  constexpr double kRelativeRadiusTolerance = 1.0e-10;
  auto solve_shifted_subspace_system =
      [&](double lambda,
          Eigen::VectorXd* eigen_coordinates,
          double* squared_norm) -> bool {
        eigen_coordinates->resize(eigenvalues.size());
        *squared_norm = 0.0;
        for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
          const double denominator = eigenvalues[index] + lambda;
          const double denominator_floor =
              kShiftToleranceFactor *
              std::max(
                  spectral_scale,
                  std::abs(eigenvalues[index]) + std::abs(lambda));
          if (!(denominator > denominator_floor) ||
              !std::isfinite(denominator)) {
            return false;
          }
          (*eigen_coordinates)[index] =
              -projected_gradient_in_eigenbasis[index] / denominator;
          *squared_norm +=
              (*eigen_coordinates)[index] * (*eigen_coordinates)[index];
        }
        return std::isfinite(*squared_norm) &&
            eigen_coordinates->allFinite();
      };

  Eigen::Index minimum_eigenvalue_index = 0;
  const double minimum_eigenvalue =
      eigenvalues.minCoeff(&minimum_eigenvalue_index);
  Eigen::VectorXd eigen_coordinates;
  double coordinate_squared_norm = 0.0;
  double trust_region_shift = 0.0;
  bool solved_subproblem = false;
  if (minimum_eigenvalue > kShiftToleranceFactor * spectral_scale &&
      solve_shifted_subspace_system(
          0.0,
          &eigen_coordinates,
          &coordinate_squared_norm) &&
      coordinate_squared_norm <=
          radius_squared * (1.0 + kRelativeRadiusTolerance)) {
    solved_subproblem = true;
  } else {
    double lower_shift = std::max(0.0, -minimum_eigenvalue);
    if (lower_shift > 0.0 || minimum_eigenvalue <= 0.0) {
      lower_shift +=
          kShiftToleranceFactor *
          spectral_scale;
    }
    if (!solve_shifted_subspace_system(
            lower_shift,
            &eigen_coordinates,
            &coordinate_squared_norm)) {
      return result;
    }

    if (coordinate_squared_norm <=
        radius_squared * (1.0 + kRelativeRadiusTolerance)) {
      solved_subproblem = true;
      trust_region_shift = lower_shift;
      if (minimum_eigenvalue < 0.0) {
        const double minimum_coordinate =
            eigen_coordinates[minimum_eigenvalue_index];
        const double other_coordinate_squared_norm =
            std::max(
                0.0,
                coordinate_squared_norm -
                    minimum_coordinate * minimum_coordinate);
        const double available_minimum_coordinate_squared =
            radius_squared - other_coordinate_squared_norm;
        if (available_minimum_coordinate_squared >
            radius_squared * kRelativeRadiusTolerance) {
          const double augmentation_sign =
              projected_gradient_in_eigenbasis[minimum_eigenvalue_index] > 0.0
                  ? -1.0
                  : 1.0;
          eigen_coordinates[minimum_eigenvalue_index] =
              augmentation_sign *
              std::sqrt(available_minimum_coordinate_squared);
          coordinate_squared_norm = radius_squared;
        }
      }
    } else {
      double upper_shift =
          std::max(1.0, std::max(2.0 * lower_shift, lower_shift + 1.0));
      Eigen::VectorXd upper_coordinates;
      double upper_squared_norm = 0.0;
      bool bracketed = false;
      for (int expansion_iteration = 0;
           expansion_iteration < 64;
           ++expansion_iteration) {
        if (!solve_shifted_subspace_system(
                upper_shift,
                &upper_coordinates,
                &upper_squared_norm)) {
          return result;
        }
        if (upper_squared_norm <= radius_squared) {
          bracketed = true;
          break;
        }
        upper_shift = std::max(2.0 * upper_shift, upper_shift + 1.0);
      }
      if (!bracketed) {
        return result;
      }

      double bisection_lower_shift = lower_shift;
      double bisection_upper_shift = upper_shift;
      eigen_coordinates = upper_coordinates;
      coordinate_squared_norm = upper_squared_norm;
      for (int bisection_iteration = 0;
           bisection_iteration < 64;
           ++bisection_iteration) {
        const double mid_shift =
            0.5 * (bisection_lower_shift + bisection_upper_shift);
        Eigen::VectorXd mid_coordinates;
        double mid_squared_norm = 0.0;
        if (!solve_shifted_subspace_system(
                mid_shift,
                &mid_coordinates,
                &mid_squared_norm)) {
          return result;
        }
        if (mid_squared_norm > radius_squared) {
          bisection_lower_shift = mid_shift;
        } else {
          bisection_upper_shift = mid_shift;
          eigen_coordinates = std::move(mid_coordinates);
          coordinate_squared_norm = mid_squared_norm;
        }
      }
      trust_region_shift = bisection_upper_shift;
      solved_subproblem = true;
    }
  }

  if (!solved_subproblem || !eigen_coordinates.allFinite()) {
    return result;
  }

  Eigen::VectorXd subspace_coordinates =
      eigenvectors * eigen_coordinates;
  Eigen::VectorXd reduced_step =
      krylov_subspace.orthonormal_basis * subspace_coordinates;
  double step_metric_norm = subspace_coordinates.norm();
  if (!(step_metric_norm > 0.0) || !std::isfinite(step_metric_norm)) {
    return result;
  }
  if (step_metric_norm >
      trust_radius * (1.0 + 1.0e-8)) {
    const double scale = trust_radius / step_metric_norm;
    subspace_coordinates *= scale;
    reduced_step *= scale;
    step_metric_norm = subspace_coordinates.norm();
  }
  const Eigen::VectorXd reduced_hessian_times_step =
      krylov_subspace.hessian_basis * subspace_coordinates;
  if (reduced_hessian_times_step.size() !=
          current_projection.reduced_gradient.size() ||
      !reduced_hessian_times_step.allFinite()) {
    return result;
  }

  const Eigen::VectorXd reduced_model_hessian_step =
      reduced_hessian * subspace_coordinates;
  const double predicted_decrease =
      -krylov_subspace.projected_gradient.dot(subspace_coordinates) -
      0.5 * subspace_coordinates.dot(reduced_model_hessian_step);
  if (!std::isfinite(predicted_decrease) ||
      predicted_decrease <= 0.0 ||
      current_projection.reduced_gradient.dot(reduced_step) >= 0.0) {
    return result;
  }

  result.reduced_step = std::move(reduced_step);
  result.reduced_hessian_times_step = reduced_hessian_times_step;
  result.krylov_subspace = krylov_subspace;
  result.retract_tangent_norm = step_metric_norm;
  result.reached_boundary =
      step_metric_norm >= (1.0 - 1.0e-8) * trust_radius;
  result.encountered_negative_curvature =
      minimum_eigenvalue <= -kShiftToleranceFactor * spectral_scale;
  result.projected_model_gradient_norm =
      krylov_subspace.projected_gradient.norm();
  result.model_spectral_radius = eigenvalues.cwiseAbs().maxCoeff();
  result.trust_region_shift = trust_region_shift;
  result.predicted_decrease = predicted_decrease;
  return result;
}

Eigen::VectorXd build_nonredundant_preconditioned_reduced_gradient_step(
    const NonredundantRetractionMetric& retraction_metric,
    const NonredundantOrbitalSpace& space,
    const NonredundantOrbitalSpace::ProjectionResult& projection,
    double trust_radius,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner) {
  const Eigen::VectorXd reduced_preconditioned_gradient =
      apply_nonredundant_truncated_newton_preconditioner(
          space,
          transported_preconditioner,
          projection.reduced_gradient);
  return retraction_metric.clip_to_radius(
      -reduced_preconditioned_gradient,
      trust_radius);
}

double update_nonredundant_truncated_newton_trust_radius(
    double trust_radius,
    double minimum_step_size,
    const TruncatedNewtonTrialEvaluation& trial,
    const TruncatedNewtonStepResult& model_step,
    bool accepted) {
  const double step_norm = truncated_newton_step_effective_norm(model_step);
  const auto geometric_fallback = [&]() {
    if (accepted && step_norm > 0.0 && std::isfinite(step_norm)) {
      return std::max(minimum_step_size, step_norm);
    }
    return std::max(
        minimum_step_size,
        std::sqrt(minimum_step_size * trust_radius));
  };
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius) ||
      !(step_norm > 0.0) || !std::isfinite(step_norm) ||
      !(trial.predicted_decrease > 0.0) ||
      !std::isfinite(trial.predicted_decrease) ||
      !std::isfinite(trial.actual_decrease)) {
    return geometric_fallback();
  }

  if (!accepted) {
    // Estimate how much of the trial scale remains trustworthy from the
    // observed Taylor-model remainder.  This continuously contracts more for
    // worse disagreement instead of applying a fixed rejection multiplier.
    const double model_error =
        std::abs(trial.actual_decrease - trial.predicted_decrease);
    const double retained_model_fraction =
        trial.predicted_decrease /
        (trial.predicted_decrease + model_error);
    const double candidate_radius = step_norm * retained_model_fraction;
    return std::max(
        minimum_step_size,
        std::min(trust_radius, candidate_radius));
  }

  if (model_step.used_krylov_rescue) {
    return std::max(minimum_step_size, step_norm);
  }

  // An interior minimizer contains no evidence that the current radius is
  // restrictive.  Preserve it; repeatedly rescaling an interior radius by
  // rho would turn otherwise valid Newton convergence into tiny first-order
  // steps whenever the nonlinear retraction makes rho slightly smaller than
  // one.
  if (!model_step.reached_boundary) {
    return std::max(minimum_step_size, trust_radius);
  }

  // The Ritz spectrum supplies a Cauchy-like length scale for the local
  // quadratic model.  On a boundary step, extrapolate only as far as both the
  // observed model agreement and that curvature length support.
  const double effective_curvature =
      model_step.model_spectral_radius + model_step.trust_region_shift;
  const double curvature_floor =
      std::numeric_limits<double>::epsilon() *
      model_step.model_spectral_radius;
  double spectral_radius = step_norm;
  if (model_step.projected_model_gradient_norm > 0.0 &&
      std::isfinite(model_step.projected_model_gradient_norm) &&
      effective_curvature > 0.0 &&
      std::isfinite(effective_curvature)) {
    spectral_radius =
        model_step.projected_model_gradient_norm /
        std::max(curvature_floor, effective_curvature);
  }
  const double trust_ratio = trial.actual_decrease / trial.predicted_decrease;
  const double overprediction_fraction = std::max(0.0, 1.0 - trust_ratio);
  const double agreement_denominator =
      std::max(
          std::sqrt(std::numeric_limits<double>::epsilon()),
          overprediction_fraction);
  const double agreement_radius = step_norm / agreement_denominator;
  const double curvature_radius = step_norm + spectral_radius;
  const double candidate_radius =
      std::max(
          step_norm,
          std::min(agreement_radius, curvature_radius));
  if (!(candidate_radius > 0.0) || !std::isfinite(candidate_radius)) {
    return geometric_fallback();
  }
  return std::max(minimum_step_size, candidate_radius);
}

double estimate_nonredundant_reduced_model_decrease(
    const NonredundantOrbitalSpace::ProjectionResult& projection,
    const Eigen::VectorXd& reduced_step,
    ReducedHvpOperator* hvp_operator) {
  const Eigen::VectorXd reduced_hessian_step =
      hvp_operator->apply(reduced_step);
  return
      -projection.reduced_gradient.dot(reduced_step) -
      0.5 * reduced_step.dot(reduced_hessian_step);
}

TruncatedNewtonStepResult solve_nonredundant_truncated_newton_step(
    const NonredundantRetractionMetric& retraction_metric,
    const NonredundantOrbitalSpace& current_space,
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    int max_cg_iterations,
    ReducedHvpOperator* hvp_operator,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step = nullptr) {
  TruncatedNewtonStepResult result;
  const Eigen::VectorXd fallback_step =
      build_nonredundant_preconditioned_reduced_gradient_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          transported_preconditioner);
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  std::vector<Eigen::VectorXd> krylov_basis_vectors;
  std::vector<Eigen::VectorXd> krylov_tangent_basis_vectors;
  std::vector<Eigen::VectorXd> krylov_hessian_basis_vectors;
  krylov_basis_vectors.reserve(
      std::max(0, max_cg_iterations) + (initial_reduced_step != nullptr ? 1 : 0));
  krylov_tangent_basis_vectors.reserve(krylov_basis_vectors.capacity());
  krylov_hessian_basis_vectors.reserve(krylov_basis_vectors.capacity());
  auto finalize_result = [&]() -> TruncatedNewtonStepResult {
    result.krylov_subspace =
        build_truncated_newton_krylov_subspace(
            current_projection.reduced_gradient,
            krylov_basis_vectors,
            krylov_tangent_basis_vectors,
            krylov_hessian_basis_vectors);
    return result;
  };
  if (current_projection.reduced_gradient.size() == 0 ||
      max_cg_iterations <= 0) {
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    return finalize_result();
  }

  const Eigen::VectorXd rhs = -current_projection.reduced_gradient;
  Eigen::VectorXd residual = rhs;
  if (initial_reduced_step != nullptr) {
    const double initial_step_norm = retraction_metric.norm(*initial_reduced_step);
    if (std::isfinite(initial_step_norm) &&
        initial_step_norm > 0.0 &&
        initial_step_norm < trust_radius) {
      const Eigen::VectorXd hessian_times_initial_step =
          hvp_operator->apply(*initial_reduced_step);

      if (hessian_times_initial_step.allFinite()) {
        result.reduced_step = *initial_reduced_step;
        result.reduced_hessian_times_step = hessian_times_initial_step;
        result.predicted_decrease =
            rhs.dot(result.reduced_step) -
            0.5 * result.reduced_step.dot(hessian_times_initial_step);
        residual.noalias() -= hessian_times_initial_step;
        append_truncated_newton_krylov_basis_vector(
            retraction_metric,
            *initial_reduced_step,
            hessian_times_initial_step,
            &krylov_basis_vectors,
            &krylov_tangent_basis_vectors,
            &krylov_hessian_basis_vectors);
      }
    }
  }

  Eigen::VectorXd preconditioned_residual =
      apply_nonredundant_truncated_newton_preconditioner(
          current_space,
          transported_preconditioner,
          residual);
  Eigen::VectorXd search_direction = preconditioned_residual;
  double residual_dot_preconditioned =
      residual.dot(preconditioned_residual);
  if (!std::isfinite(residual_dot_preconditioned) ||
      residual_dot_preconditioned <= 0.0) {
    if (result.reduced_step.squaredNorm() > 0.0 &&
        current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
        std::isfinite(result.predicted_decrease) &&
        result.predicted_decrease > 0.0) {
      return finalize_result();
    }
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
    return finalize_result();
  }

  const double initial_residual_inf_norm = gradient_infinity_norm(residual);
  const double outer_gradient_inf_norm =
      gradient_infinity_norm(current_projection.reduced_gradient);
  // The inexact-Newton forcing term is an outer-iteration condition:
  // ||H s + g|| <= eta_k ||g||.  In particular, a cached same-point trial
  // changes the initial residual but must not redefine the requested Newton
  // accuracy.
  const double residual_inf_target =
      inexact_newton_forcing_term(outer_gradient_inf_norm) *
      outer_gradient_inf_norm;
  if (initial_residual_inf_norm <= residual_inf_target) {
    if (result.reduced_step.squaredNorm() > 0.0 &&
        current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
        std::isfinite(result.predicted_decrease) &&
        result.predicted_decrease > 0.0) {
      return finalize_result();
    }
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
    return finalize_result();
  }

  constexpr double kCurvatureTolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  for (int cg_iteration = 0;
       cg_iteration < max_cg_iterations;
       ++cg_iteration) {
    const Eigen::VectorXd hessian_times_direction =
        hvp_operator->apply(search_direction);
    append_truncated_newton_krylov_basis_vector(
        retraction_metric,
        search_direction,
        hessian_times_direction,
        &krylov_basis_vectors,
        &krylov_tangent_basis_vectors,
        &krylov_hessian_basis_vectors);
    const double curvature =
        search_direction.dot(hessian_times_direction);
    const Eigen::VectorXd current_step_tangent =
        retraction_metric.tangent(result.reduced_step);
    const Eigen::VectorXd search_direction_tangent =
        retraction_metric.tangent(search_direction);
    const double search_direction_metric_norm_squared =
        search_direction_tangent.squaredNorm();
    const double curvature_scale =
        search_direction.norm() * hessian_times_direction.norm();
    if (!std::isfinite(curvature) ||
        !std::isfinite(curvature_scale) ||
        curvature <=
            kCurvatureTolerance * curvature_scale) {
      result.encountered_negative_curvature = true;
      result.reached_boundary = true;
      if (search_direction_metric_norm_squared > 0.0 &&
          std::isfinite(search_direction_metric_norm_squared)) {
        const double tau =
            solve_trust_region_metric_boundary_tau(
                current_step_tangent,
                search_direction_tangent,
                trust_radius);
        result.predicted_decrease +=
            tau * residual.dot(search_direction) -
            0.5 * tau * tau * curvature;
        result.reduced_step.noalias() +=
            tau *
            search_direction;
        result.reduced_hessian_times_step.noalias() +=
            tau *
            hessian_times_direction;
      }
      break;
    }

    const double alpha =
        residual_dot_preconditioned / curvature;
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      result.reduced_step = fallback_step;
      result.reduced_hessian_times_step.resize(0);
      return finalize_result();
    }

    const Eigen::VectorXd candidate_step =
        result.reduced_step + alpha * search_direction;
    const double candidate_step_metric_norm =
        retraction_metric.norm(candidate_step);
    if (candidate_step_metric_norm >= trust_radius) {
      result.reached_boundary = true;
      const double tau =
          solve_trust_region_metric_boundary_tau(
              current_step_tangent,
              search_direction_tangent,
              trust_radius);
      result.predicted_decrease +=
          tau * residual.dot(search_direction) -
          0.5 * tau * tau * curvature;
      result.reduced_step.noalias() +=
          tau *
          search_direction;
      result.reduced_hessian_times_step.noalias() +=
          tau *
          hessian_times_direction;
      break;
    }

    result.predicted_decrease +=
        alpha * residual.dot(search_direction) -
        0.5 * alpha * alpha * curvature;
    result.reduced_step = candidate_step;
    result.retract_tangent_norm = candidate_step_metric_norm;
    result.reduced_hessian_times_step.noalias() +=
        alpha *
        hessian_times_direction;
    residual.noalias() -= alpha * hessian_times_direction;
    result.cg_iterations = cg_iteration + 1;
    if (gradient_infinity_norm(residual) <= residual_inf_target) {
      break;
    }

    preconditioned_residual =
        apply_nonredundant_truncated_newton_preconditioner(
            current_space,
            transported_preconditioner,
            residual);
    const double next_residual_dot_preconditioned =
        residual.dot(preconditioned_residual);
    if (!std::isfinite(next_residual_dot_preconditioned) ||
        next_residual_dot_preconditioned <= 0.0) {
      break;
    }
    const double beta =
        next_residual_dot_preconditioned / residual_dot_preconditioned;
    if (!std::isfinite(beta) || beta < 0.0) {
      break;
    }
    search_direction =
        preconditioned_residual + beta * search_direction;
    residual_dot_preconditioned = next_residual_dot_preconditioned;
  }

  // CG is used here to collect a local HVP subspace. The final candidate must be
  // the trust-region minimizer in the accepted-point retraction metric
  // ||J d||, not the raw Euclidean PCG accumulation, otherwise sparse charts
  // solve the wrong spherical subproblem and tend to exhaust the radius.
  const TruncatedNewtonKrylovSubspace krylov_subspace =
      build_truncated_newton_krylov_subspace(
          current_projection.reduced_gradient,
          krylov_basis_vectors,
          krylov_tangent_basis_vectors,
          krylov_hessian_basis_vectors);
  auto metric_trust_region_step =
      solve_trust_region_in_krylov_subspace(
          current_projection,
          trust_radius,
          krylov_subspace);
  if (truncated_newton_step_is_usable(
          metric_trust_region_step,
          current_projection.reduced_gradient)) {
    metric_trust_region_step.encountered_negative_curvature =
        metric_trust_region_step.encountered_negative_curvature ||
        result.encountered_negative_curvature;
    metric_trust_region_step.cg_iterations = result.cg_iterations;
    return metric_trust_region_step;
  }

  const bool result_step_is_usable =
      std::isfinite(result.reduced_step.norm()) &&
      result.reduced_step.squaredNorm() > 0.0 &&
      current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
      std::isfinite(result.predicted_decrease) &&
      result.predicted_decrease > 0.0;
  if (!result_step_is_usable) {
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
  }
  return finalize_result();
}

bool try_armijo_backtracking_direction(
    OrbitalObjective* objective,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    const Eigen::VectorXd& search_direction,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy) {
  const double directional_derivative =
      current_gradient.dot(search_direction);
  if (!std::isfinite(directional_derivative) ||
      directional_derivative >= 0.0) {
    return false;
  }

  double step = std::max(minimum_step, initial_step);
  Eigen::VectorXd trial_parameters(current_parameters.size());
  // This helper is shared by the L-BFGS stall fallback and the nonredundant
  // projected-gradient backend. Both use the same sufficient-decrease test on
  // the full relaxed orbital objective.
  while (step >= minimum_step) {
    trial_parameters.noalias() =
        current_parameters + step * search_direction;
    auto trial_evaluation =
        objective->evaluate_trial_without_committing(trial_parameters);
    const double trial_energy = trial_evaluation.energy;
    const double armijo_upper_bound =
        current_energy + armijo_constant * step * directional_derivative;
    if (std::isfinite(trial_energy) && trial_energy <= armijo_upper_bound) {
      *accepted_parameters = trial_parameters;
      *accepted_gradient = trial_evaluation.gradient;
      *accepted_energy = trial_energy;
      objective->commit_trial_evaluation(std::move(trial_evaluation));
      return true;
    }
    step *= 0.5;
  }

  return false;
}

bool try_build_nonredundant_lifted_trial_parameters(
    const OrbitalPreparationInput& current_orbital_input,
    const NonredundantOrbitalSpace& current_space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step,
    Eigen::VectorXd* trial_parameters) {
  const Eigen::VectorXd current_parameters =
      parameter_view.pack(current_orbital_input);
  // The reduced coordinates parameterize a local tangent vector on the
  // accepted sparse-orbital chart. Build finite trial points with the same
  // retraction used by the reduced HVP finite-difference operator, then pack
  // the resulting orbital table back into the optimizer coordinate vector.
  try {
    const OrbitalPreparationInput trial_orbital_input =
        current_space.retract_step(
            current_orbital_input,
            reduced_step);
    *trial_parameters =
        parameter_view.pack(trial_orbital_input);
  } catch (const std::exception&) {
    // In a line search or trust-radius retry, leaving the local retraction
    // chart means this trial is too large; callers can shrink and retry.
    return false;
  }
  if (trial_parameters->size() != current_parameters.size() ||
      !trial_parameters->allFinite()) {
    return false;
  }
  return true;
}

bool try_armijo_backtracking_nonredundant_direction(
    OrbitalObjective* objective,
    const OrbitalPreparationInput& current_orbital_input,
    const NonredundantOrbitalSpace& current_space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    const Eigen::VectorXd& reduced_search_direction,
    const Eigen::VectorXd& packed_tangent_search_direction,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy) {
  const double directional_derivative =
      current_gradient.dot(packed_tangent_search_direction);
  if (!std::isfinite(directional_derivative) ||
      directional_derivative >= 0.0) {
    return false;
  }

  double step = std::max(minimum_step, initial_step);
  Eigen::VectorXd trial_parameters(current_parameters.size());
  // Keep the accepted-point state in the original sparse-coefficient chart,
  // but generate finite trial points with the nonredundant orbital-increment
  // lift rather than by adding the reduced direction directly in packed sparse
  // coordinates.
  while (step >= minimum_step) {
    if (!try_build_nonredundant_lifted_trial_parameters(
            current_orbital_input,
            current_space,
            parameter_view,
            step * reduced_search_direction,
            &trial_parameters)) {
      step *= 0.5;
      continue;
    }
    if (is_effectively_zero_step(
            trial_parameters - current_parameters,
            current_parameters)) {
      step *= 0.5;
      continue;
    }

    auto trial_evaluation =
        objective->evaluate_trial_without_committing(trial_parameters);
    const double trial_energy = trial_evaluation.energy;
    const double armijo_upper_bound =
        current_energy + armijo_constant * step * directional_derivative;
    if (std::isfinite(trial_energy) && trial_energy <= armijo_upper_bound) {
      *accepted_parameters = trial_parameters;
      *accepted_gradient = trial_evaluation.gradient;
      *accepted_energy = trial_energy;
      objective->commit_trial_evaluation(std::move(trial_evaluation));
      return true;
    }
    step *= 0.5;
  }

  return false;
}

bool try_armijo_backtracking_step(
    OrbitalObjective* objective,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy) {
  // This fallback intentionally switches to steepest descent with Armijo-only
  // decrease. It is more tolerant to small gradient inconsistencies from
  // highly parallel reductions than a failed quasi-Newton line search.
  const Eigen::VectorXd search_direction = -current_gradient;
  return try_armijo_backtracking_direction(
      objective,
      current_parameters,
      current_energy,
      current_gradient,
      search_direction,
      initial_step,
      minimum_step,
      armijo_constant,
      accepted_parameters,
      accepted_gradient,
      accepted_energy);
}

void sync_result_from_objective(
    const OrbitalObjective& objective,
    CppVbScfOptimizerResult* result);

void record_accepted_iteration_snapshot(
    OrbitalObjective* objective,
    int iteration,
    const CppVbScfOptimizerOptions& options,
    CppVbScfOptimizerResult* result);

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
    if (is_effectively_zero_step(
            trial_parameters - start_parameters,
            start_parameters)) {
      continue;
    }
    Eigen::VectorXd trial_gradient;
    const double trial_energy =
        (*objective)(trial_parameters, trial_gradient);
    if (std::isfinite(trial_energy) &&
        trial_gradient.allFinite() &&
        trial_energy < best_energy) {
      has_best_descent = true;
      best_parameters = trial_parameters;
      best_gradient = trial_gradient;
      best_energy = trial_energy;
      best_step = step;
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
    CppVbScfOptimizerResult* result) {
  result->total_energy_history = objective.energy_history();
  result->gradient_inf_norm_history = objective.gradient_inf_norm_history();
  result->iteration_time_history_seconds = objective.iteration_time_history_seconds();
  result->scf_result = objective.last_gradient_result().scf_result;
}

void record_accepted_iteration_snapshot(
    OrbitalObjective* objective,
    int accepted_iteration_index,
    const CppVbScfOptimizerOptions& options,
    CppVbScfOptimizerResult* result) {
  if (!options.retain_accepted_iteration_trace && !options.accepted_iteration_callback) {
    return;
  }
  const bool include_reference_energy_gradient =
      options.retain_accepted_iteration_trace ||
      options.accepted_iteration_callback_requires_reference_gradient;
  const bool include_full_payload =
      options.retain_accepted_iteration_trace ||
      options.accepted_iteration_callback_requires_full_snapshot;
  // Lightweight callback snapshots only need scalar iteration summaries. Skip
  // the heavyweight matrix/integral deep copies unless the retained trace or a
  // callback explicitly requested the full payload.
  if (include_reference_energy_gradient) {
    objective->ensure_last_reference_energy_gradient();
  }
  const auto& gradient_result = objective->last_gradient_result();
  CppVbScfAcceptedIterationSnapshot snapshot;
  snapshot.accepted_iteration_index = accepted_iteration_index;
  snapshot.has_full_payload = include_full_payload;
  for (const double value : gradient_result.sparse_orbital_energy_gradient) {
    snapshot.sparse_orbital_energy_gradient_inf_norm =
        std::max(snapshot.sparse_orbital_energy_gradient_inf_norm, std::abs(value));
    snapshot.sparse_orbital_energy_gradient_l2_norm += value * value;
  }
  snapshot.sparse_orbital_energy_gradient_l2_norm =
      std::sqrt(snapshot.sparse_orbital_energy_gradient_l2_norm);
  if (include_full_payload) {
    snapshot.orbital_value_table.assign(
        objective->last_input().orbital_preparation_input.orbital_value_table.begin(),
        objective->last_input().orbital_preparation_input.orbital_value_table.end());
    snapshot.structure_matrices = gradient_result.scf_result.structure_matrices;
    snapshot.active_orbital_overlap_matrix =
        gradient_result.active_orbital_overlap_matrix;
    snapshot.active_one_electron_integrals =
        gradient_result.active_one_electron_integrals;
    snapshot.packed_active_two_electron_integrals =
        gradient_result.packed_active_two_electron_integrals;
    snapshot.sparse_orbital_energy_gradient =
        gradient_result.sparse_orbital_energy_gradient;
    if (include_reference_energy_gradient) {
      snapshot.sparse_orbital_reference_energy_gradient =
          gradient_result.sparse_orbital_reference_energy_gradient;
    }
  }
  snapshot.total_energy = gradient_result.scf_result.total_energy;
  snapshot.one_electron_reference_energy =
      gradient_result.scf_result.one_electron_reference_energy;
  snapshot.average_structure_overlap =
      gradient_result.scf_result.average_structure_overlap;
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
  if (options_.nonredundant_truncated_newton_max_cg_iterations < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_max_cg_iterations must be nonnegative");
  }
  if (options_.nonredundant_truncated_newton_hvp_step_size <= 0.0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_hvp_step_size must be positive");
  }
  if (options_.nonredundant_truncated_newton_transport_history_size < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_transport_history_size must be nonnegative");
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

  // For `guess=mo`, numerical parity with the legacy VBSCF implementation is
  // more important than any temporary convergence-speed heuristic. Keep the
  // nonredundant optimizer on the original legacy sparse chart and exact-
  // support block partition so the reduced coordinates, projected gradients,
  // and exact-context orbital derivatives all live on the same variational
  // manifold as the reference `.xmo` calculation.
  std::optional<CppVbInput> adapted_optimizer_input;
  const CppVbInput* optimizer_input = &input;
  if (optimizer_backend_uses_nonredundant_space(options_.backend)) {
    adapted_optimizer_input = build_nonredundant_optimizer_input(input);
    optimizer_input = &adapted_optimizer_input.value();
  }
  const SparseOrbitalParameterView parameter_view(
      optimizer_input->orbital_preparation_input);
  Eigen::VectorXd parameter_vector =
      parameter_view.pack(optimizer_input->orbital_preparation_input);
  Eigen::MatrixXd initial_normalized_orbital_matrix;

  OrbitalObjective objective(
      *optimizer_input,
      parameter_view,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      &orbital_gradient_evaluator_,
      &scf_evaluator_);
  const int n = static_cast<int>(parameter_vector.size());
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;
  bool final_projected_gradient_ready = false;

  try {
    Eigen::VectorXd gradient(parameter_vector.size());
    double energy = objective(parameter_vector, gradient);
    initial_normalized_orbital_matrix =
        objective.last_gradient_result()
            .orbital_preparation_result
            .physical_orbital_frame
            .normalized_orbital_matrix;
    sync_result_from_objective(objective, &result);
    record_accepted_iteration_snapshot(&objective, 0, options_, &result);
    result.initial_total_energy = energy;
    result.initial_one_electron_reference_energy =
        objective.last_gradient_result().scf_result.one_electron_reference_energy;
    double previous_energy = energy;
    final_gradient_l2_norm = gradient.norm();
    switch (options_.backend) {

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
        // Let the primary line search expand beyond the nominal unit trial
        // step, while still clamping it to a finite orbital-parameter radius.
        param.max_step =
            std::max(
                options_.initial_step_size,
                options_.initial_step_size * kLineSearchExpansionFactor);
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
        double last_robust_step = std::min(1.0, options_.initial_step_size);
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
          double directional_derivative = current_gradient.dot(search_direction);
          if (directional_derivative >= 0.0) {
            result.termination_reason = "lbfgspp_non_descent_direction";
            break;
          }

          previous_parameters = current_parameters;
          previous_gradient = current_gradient;
          const double reference_energy = energy;
          const double previous_gradient_inf_norm =
              gradient_infinity_norm(previous_gradient);
          double step = std::min(1.0, options_.initial_step_size);
          if (last_iteration_used_fallback && has_robust_step_history) {
            step = std::max(
                param.min_step,
                std::min(last_robust_step, param.max_step));
          }
          bool used_fallback = false;
          bool reset_inverse_hessian = false;
          std::string primary_line_search_error;

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

          Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
          const double primary_gradient_inf_norm =
              gradient_infinity_norm(current_gradient);
          const bool stalled_line_search =
              primary_line_search_error.empty() &&
              (is_effectively_zero_step(parameter_step, previous_parameters) ||
               line_search_made_no_meaningful_progress(
                   reference_energy,
                   energy,
                   previous_gradient_inf_norm,
                   primary_gradient_inf_norm,
                   options_.gradient_tolerance));
          const bool suspicious_primary_step =
              primary_line_search_error.empty() &&
              has_robust_step_history &&
              step < kSuspiciousPrimaryStepRatio * last_robust_step;
          const bool should_try_fallback =
              !primary_line_search_error.empty() ||
              (previous_gradient_inf_norm >= options_.gradient_tolerance &&
               (stalled_line_search ||
                step <= kTinyStepFactor * param.min_step ||
                suspicious_primary_step));
          if (should_try_fallback) {
            const Eigen::VectorXd primary_parameters = current_parameters;
            const Eigen::VectorXd primary_gradient = current_gradient;
            const double primary_energy = energy;
            double fallback_step =
                has_robust_step_history
                    ? last_robust_step
                    : std::max(
                          param.min_step,
                          std::min(
                              std::min(1.0, options_.initial_step_size),
                              param.max_step));
            Eigen::VectorXd fallback_parameters;
            Eigen::VectorXd fallback_gradient;
            double fallback_energy = reference_energy;
            if (try_steepest_descent_armijo_fallback(
                    &objective,
                    param,
                    previous_parameters,
                    previous_gradient,
                    reference_energy,
                    fallback_step,
                    &fallback_parameters,
                    &fallback_gradient,
                    &fallback_energy,
                    &fallback_step)) {
              const bool should_replace_primary =
                  !primary_line_search_error.empty() ||
                  stalled_line_search ||
                  step <= kTinyStepFactor * param.min_step ||
                  suspicious_primary_step ||
                  fallback_energy < primary_energy;
              if (should_replace_primary) {
                current_parameters = std::move(fallback_parameters);
                current_gradient = std::move(fallback_gradient);
                energy = fallback_energy;
                step = fallback_step;
                parameter_step = current_parameters - previous_parameters;
                used_fallback = true;
                reset_inverse_hessian = true;
              } else {
                current_parameters = primary_parameters;
                current_gradient = primary_gradient;
                energy = primary_energy;
              }
            } else if (!primary_line_search_error.empty() ||
                       stalled_line_search) {
              energy = objective(previous_parameters, current_gradient);
              current_parameters = previous_parameters;
              sync_result_from_objective(objective, &result);
              final_gradient_l2_norm = current_gradient.norm();
              result.termination_reason =
                  !primary_line_search_error.empty()
                      ? std::string("lbfgspp_line_search: ") +
                            primary_line_search_error
                      : "lbfgspp_line_search_stalled";
              break;
            }
          }

          if (step > kTinyStepFactor * param.min_step &&
              (!has_robust_step_history ||
               step >= kRobustStepShrinkRatio * last_robust_step)) {
            last_robust_step = step;
            has_robust_step_history = true;
          }
          last_iteration_used_fallback = used_fallback;

          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient);
          if (accepted_point_chart_reset) {
            reset_inverse_hessian = true;
          }

          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          if (std::abs(de) < options_.energy_tolerance &&
              final_gradient_l2_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason = "lbfgspp_dual_tolerance";
            break;
          }

          Eigen::VectorXd gradient_step = current_gradient - previous_gradient;
          if (reset_inverse_hessian) {
            inverse_hessian.reset(n, param.m);
          }
          if (!reset_inverse_hessian &&
              parameter_step.dot(gradient_step) >
              kCurvatureEpsilon * gradient_step.squaredNorm()) {
            inverse_hessian.add_correction(parameter_step, gradient_step);
          }
          inverse_hessian.apply_Hv(current_gradient, -1.0, search_direction);
          if (used_fallback &&
              search_direction.dot(current_gradient) >= 0.0) {
            search_direction = -current_gradient;
          }
        }
        break;
      }

      case CppVbScfOptimizerBackend::NonredundantProjectedGradient: {
        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        NonredundantOrbitalSpace current_space =
            build_nonredundant_space(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          if (iteration == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_projected_gradient_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const Eigen::VectorXd reduced_search_direction =
              -current_space.apply_inverse_reduced_block_preconditioner(
                  current_projection.reduced_gradient);
          const Eigen::VectorXd search_direction =
              gather_nonredundant_retract_tangent(
                  objective.last_input().orbital_preparation_input,
                  current_space,
                  parameter_view,
                  reduced_search_direction);
          // The accepted point remains in packed sparse coefficients, so the
          // projected-gradient backend must use the actual retraction tangent
          // rather than the raw additive chart when testing descent and Armijo.
          const double directional_derivative =
              current_gradient.dot(search_direction);
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0) {
            result.termination_reason =
                "nonredundant_projected_gradient_non_descent_direction";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const OrbitalPreparationInput current_orbital_input =
              objective.last_input().orbital_preparation_input;
          Eigen::VectorXd accepted_parameters(current_parameters.size());
          Eigen::VectorXd accepted_gradient(current_gradient.size());
          double accepted_energy = energy;
          if (!try_armijo_backtracking_nonredundant_direction(
                  &objective,
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  current_parameters,
                  energy,
                  current_gradient,
                  reduced_search_direction,
                  search_direction,
                  std::min(1.0, options_.initial_step_size),
                  options_.minimum_step_size,
                  options_.armijo_constant,
                  &accepted_parameters,
                  &accepted_gradient,
                  &accepted_energy)) {
            result.termination_reason =
                "nonredundant_projected_gradient_line_search_failed";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          current_parameters = std::move(accepted_parameters);
          current_gradient = std::move(accepted_gradient);
          energy = accepted_energy;
          objective.canonicalize_orbital_chart_at_current_point(
              &current_parameters,
              &current_gradient);
          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          NonredundantOrbitalSpace next_space =
              build_nonredundant_space(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          if (std::abs(de) < options_.energy_tolerance &&
              gradient_infinity_norm(next_projection.reduced_gradient) <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_projected_gradient_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }
        break;
      }

      case CppVbScfOptimizerBackend::NonredundantLbfgspp: {
        const int history_size = options_.history_size;
        LBFGSpp::BFGSMat<double> inverse_hessian;
        inverse_hessian.reset(n, history_size);

        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        NonredundantOrbitalSpace current_space =
            build_nonredundant_space(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);
        Eigen::VectorXd previous_parameters(n);
        Eigen::VectorXd previous_gradient(n);
        Eigen::VectorXd previous_projected_gradient(n);
        constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          if (iteration == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_lbfgspp_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const Eigen::VectorXd fallback_reduced_direction =
              -current_projection.reduced_gradient;
          const OrbitalPreparationInput previous_orbital_input =
              objective.last_input().orbital_preparation_input;
          const Eigen::VectorXd fallback_direction =
              gather_nonredundant_retract_tangent(
                  previous_orbital_input,
                  current_space,
                  parameter_view,
                  fallback_reduced_direction);
          Eigen::VectorXd search_direction;
          Eigen::VectorXd reduced_search_direction;
          inverse_hessian.apply_Hv(
              current_projection.packed_projected_gradient,
              -1.0,
              search_direction);
          {
            const auto search_projection =
                current_space.project_vector(search_direction);
            reduced_search_direction = search_projection.reduced_gradient;
            search_direction =
                gather_nonredundant_retract_tangent(
                    previous_orbital_input,
                    current_space,
                    parameter_view,
                    reduced_search_direction);
          }
          double directional_derivative =
              current_gradient.dot(search_direction);
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0 ||
              is_effectively_zero_step(search_direction, current_parameters)) {
            inverse_hessian.reset(n, history_size);
            search_direction = fallback_direction;
            reduced_search_direction = fallback_reduced_direction;
            directional_derivative = current_gradient.dot(search_direction);
          }
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0) {
            result.termination_reason =
                "nonredundant_lbfgspp_non_descent_direction";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          previous_parameters = current_parameters;
          previous_gradient = current_gradient;
          previous_projected_gradient = current_projection.packed_projected_gradient;
          const Eigen::VectorXd previous_reduced_gradient =
              current_projection.reduced_gradient;
          const double reference_energy = energy;
          Eigen::VectorXd accepted_parameters(current_parameters.size());
          Eigen::VectorXd accepted_gradient(current_gradient.size());
          double accepted_energy = energy;
          if (!try_armijo_backtracking_nonredundant_direction(
                  &objective,
                  previous_orbital_input,
                  current_space,
                  parameter_view,
                  current_parameters,
                  energy,
                  current_gradient,
                  reduced_search_direction,
                  search_direction,
                  std::min(1.0, options_.initial_step_size),
                  options_.minimum_step_size,
                  options_.armijo_constant,
                  &accepted_parameters,
                  &accepted_gradient,
                  &accepted_energy)) {
            result.termination_reason =
                "nonredundant_lbfgspp_line_search_failed";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          current_parameters = std::move(accepted_parameters);
          current_gradient = std::move(accepted_gradient);
          energy = accepted_energy;

          NonredundantOrbitalSpace next_space =
              build_nonredundant_space(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          double next_reduced_gradient_inf_norm =
              gradient_infinity_norm(next_projection.reduced_gradient);

          Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
          bool recovered_from_stall = false;
          const bool stalled_line_search =
              is_effectively_zero_step(parameter_step, previous_parameters) ||
              line_search_made_no_meaningful_progress(
                  reference_energy,
                  energy,
                  reduced_gradient_inf_norm,
                  next_reduced_gradient_inf_norm,
                  options_.gradient_tolerance);
          if (stalled_line_search &&
              reduced_gradient_inf_norm >= options_.gradient_tolerance) {
            const double fallback_initial_step =
                std::max(options_.minimum_step_size,
                         std::min(
                             std::min(1.0, options_.initial_step_size),
                             1.0 / std::max(1.0, reduced_gradient_inf_norm)));
            if (!try_armijo_backtracking_nonredundant_direction(
                    &objective,
                    previous_orbital_input,
                    current_space,
                    parameter_view,
                    previous_parameters,
                    reference_energy,
                    previous_gradient,
                    -previous_reduced_gradient,
                    gather_nonredundant_retract_tangent(
                        previous_orbital_input,
                        current_space,
                        parameter_view,
                        -previous_reduced_gradient),
                    fallback_initial_step,
                    options_.minimum_step_size,
                    options_.armijo_constant,
                    &current_parameters,
                    &current_gradient,
                    &energy)) {
              energy = objective(previous_parameters, current_gradient);
              current_parameters = previous_parameters;
              sync_result_from_objective(objective, &result);
              final_gradient_l2_norm = current_gradient.norm();
              result.termination_reason = "nonredundant_lbfgspp_line_search_stalled";
              break;
            }
            next_space = build_nonredundant_space(objective, parameter_view);
            next_projection =
                next_space.project_gradient(current_gradient);
            next_reduced_gradient_inf_norm =
                gradient_infinity_norm(next_projection.reduced_gradient);
            parameter_step = current_parameters - previous_parameters;
            inverse_hessian.reset(n, history_size);
            recovered_from_stall = true;
          }

          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient);
          if (accepted_point_chart_reset) {
            next_space = build_nonredundant_space(objective, parameter_view);
            next_projection =
                next_space.project_gradient(current_gradient);
            next_reduced_gradient_inf_norm =
                gradient_infinity_norm(next_projection.reduced_gradient);
          }

          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          if (std::abs(de) < options_.energy_tolerance &&
              next_reduced_gradient_inf_norm <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_lbfgspp_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          if (accepted_point_chart_reset) {
            inverse_hessian.reset(n, history_size);
          } else {
            parameter_step = current_parameters - previous_parameters;
            const Eigen::VectorXd projected_gradient_step =
                next_projection.packed_projected_gradient -
                previous_projected_gradient;
            if (parameter_step.dot(projected_gradient_step) >
                kCurvatureEpsilon * projected_gradient_step.squaredNorm()) {
              inverse_hessian.add_correction(
                  parameter_step,
                  projected_gradient_step);
            } else if (recovered_from_stall) {
              inverse_hessian.reset(n, history_size);
            }
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }

        break;
      }

      case CppVbScfOptimizerBackend::NonredundantTruncatedNewton: {
        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        NonredundantOrbitalSpace current_space =
            build_nonredundant_space(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);
        double final_projected_gradient_inf_norm =
            gradient_infinity_norm(current_projection.reduced_gradient);
        double final_projected_gradient_l2_norm =
            current_projection.reduced_gradient.norm();
        std::vector<PackedSecantPair> packed_secant_history;
        packed_secant_history.reserve(
            std::max(
                0,
                options_.nonredundant_truncated_newton_transport_history_size));
        double trust_radius =
            std::max(options_.minimum_step_size, options_.initial_step_size);
        int rejected_trial_step_count_for_current_point = 0;
        RejectedTruncatedNewtonStepCache rejected_step_cache;
        TruncatedNewtonKrylovSubspace cached_krylov_subspace;

        while (n_iterations < options_.max_iterations) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            final_gradient_l2_norm = 0.0;
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          final_projected_gradient_inf_norm = reduced_gradient_inf_norm;
          final_projected_gradient_l2_norm =
              current_projection.reduced_gradient.norm();
          if (n_iterations == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_truncated_newton_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
            result.termination_reason =
                "nonredundant_truncated_newton_invalid_trust_radius";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          std::unique_ptr<ReducedHvpOperator> hvp_operator;
          switch (options_.nonredundant_truncated_newton_hvp_mode) {
            case NonredundantTruncatedNewtonHvpMode::FullFiniteDifference:
              hvp_operator = std::make_unique<FullFiniteDifferenceReducedHvpOperator>(
                  objective,
                  current_space,
                  current_projection,
                  objective.last_input().orbital_preparation_input,
                  parameter_view,
                  current_parameters,
                  options_.nonredundant_truncated_newton_hvp_step_size);
              break;
            case NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction: {
              auto exact_ctx_hvp_operator =
                  std::make_unique<ExactContextReducedHvpOperator>(
                      objective,
                      current_space);
              if (!exact_ctx_hvp_operator->supports_analytic_core_model()) {
                throw std::runtime_error(
                    build_exact_ctx_unavailable_message(*exact_ctx_hvp_operator));
              }
              hvp_operator = std::move(exact_ctx_hvp_operator);
              break;
            }
          }
          const int transport_history_size =
              choose_nonredundant_truncated_newton_transport_history_size(
                  options_,
                  objective);
          const auto transported_preconditioner =
              build_nonredundant_truncated_newton_preconditioner(
                  current_space,
                  packed_secant_history,
                  transport_history_size);
          const int max_cg_iterations =
              choose_nonredundant_truncated_newton_max_cg_iterations(
                  options_,
                  current_projection.reduced_gradient.size());
          const OrbitalPreparationInput current_orbital_input =
              objective.last_input().orbital_preparation_input;
          const NonredundantRetractionMetric retraction_metric(
              current_orbital_input,
              current_space,
              parameter_view);
          const auto admit_energy_only_trial_screen = [&]() {
            const auto& objective_time_history =
                objective.iteration_time_history_seconds();
            const double last_objective_seconds =
                objective_time_history.empty()
                    ? 0.0
                    : objective_time_history.back();
            // Measure the two available trial-evaluation paths directly.
            // After one sample, use energy-only screening exactly when its
            // observed cost is below a full objective-and-gradient call.
            if (objective.energy_only_call_count() == 0) {
              return true;
            }
            const double last_energy_only_seconds =
                objective.last_energy_only_wall_time_seconds();
            return
                std::isfinite(last_energy_only_seconds) &&
                last_energy_only_seconds > 0.0 &&
                last_energy_only_seconds < last_objective_seconds;
          };
          auto try_truncated_newton_trial_step =
              [&](const Eigen::VectorXd& candidate_reduced_step,
                  double candidate_predicted_decrease,
                  bool screen_with_energy_only,
                  TruncatedNewtonTrialEvaluation* trial_evaluation,
                  Eigen::VectorXd* accepted_packed_step,
                  OrbitalObjective::TrialEvaluation* accepted_trial_evaluation,
                  Eigen::VectorXd* accepted_trial_parameters,
                  Eigen::VectorXd* accepted_trial_gradient,
                  double* accepted_trial_energy) -> bool {
                if (trial_evaluation != nullptr) {
                  *trial_evaluation = TruncatedNewtonTrialEvaluation();
                }
                if (!std::isfinite(candidate_predicted_decrease) ||
                    candidate_predicted_decrease <= 0.0) {
                  return false;
                }

                Eigen::VectorXd candidate_trial_parameters(
                    current_parameters.size());
                if (!try_build_nonredundant_lifted_trial_parameters(
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        candidate_reduced_step,
                        &candidate_trial_parameters)) {
                  return false;
                }
                const Eigen::VectorXd candidate_packed_step =
                    candidate_trial_parameters - current_parameters;
                if (is_effectively_zero_step(
                        candidate_packed_step,
                        current_parameters)) {
                  return false;
                }

                double effective_predicted_decrease =
                    candidate_predicted_decrease;
                if (candidate_reduced_step.size() ==
                        current_projection.reduced_gradient.size()) {
                  const double reduced_linear_decrease =
                      -current_projection.reduced_gradient.dot(
                          candidate_reduced_step);
                  const double packed_retraction_linear_decrease =
                      -current_gradient.dot(candidate_packed_step);
                  if (std::isfinite(reduced_linear_decrease) &&
                      std::isfinite(packed_retraction_linear_decrease)) {
                    const double curvature_decrease =
                        candidate_predicted_decrease -
                        reduced_linear_decrease;
                    const double retraction_predicted_decrease =
                        packed_retraction_linear_decrease +
                        curvature_decrease;
                    if (std::isfinite(retraction_predicted_decrease) &&
                        retraction_predicted_decrease > 0.0) {
                      effective_predicted_decrease =
                          retraction_predicted_decrease;
                    }
                  }
                }
                if (!std::isfinite(effective_predicted_decrease) ||
                    effective_predicted_decrease <= 0.0) {
                  return false;
                }
                if (trial_evaluation != nullptr) {
                  trial_evaluation->predicted_decrease =
                      effective_predicted_decrease;
                }

                if (screen_with_energy_only) {
                  const double candidate_trial_energy =
                      objective.evaluate_energy_only(
                          candidate_trial_parameters);
                  const double actual_decrease =
                      energy - candidate_trial_energy;
                  const double candidate_trust_ratio =
                      actual_decrease / effective_predicted_decrease;
                  if (trial_evaluation != nullptr) {
                    trial_evaluation->actual_decrease = actual_decrease;
                  }
                  if (!std::isfinite(candidate_trial_energy) ||
                      !std::isfinite(candidate_trust_ratio) ||
                      actual_decrease <= 0.0) {
                    return false;
                  }
                }
                OrbitalObjective::TrialEvaluation candidate_trial_evaluation =
                    objective.evaluate_trial_without_committing(
                        candidate_trial_parameters);
                const double candidate_trial_energy =
                    candidate_trial_evaluation.energy;
                const double actual_decrease =
                    energy - candidate_trial_energy;
                const double candidate_trust_ratio =
                    actual_decrease / effective_predicted_decrease;
                if (trial_evaluation != nullptr) {
                  trial_evaluation->actual_decrease = actual_decrease;
                }
                if (!std::isfinite(candidate_trial_energy) ||
                    !std::isfinite(candidate_trust_ratio) ||
                    actual_decrease <= 0.0) {
                  return false;
                }

                *accepted_packed_step = candidate_packed_step;
                *accepted_trial_evaluation =
                    std::move(candidate_trial_evaluation);
                *accepted_trial_parameters = candidate_trial_parameters;
                *accepted_trial_gradient =
                    std::move(accepted_trial_evaluation->gradient);
                *accepted_trial_energy = candidate_trial_energy;
                return true;
              };
          auto try_nonredundant_descent_fallback_step =
              [&](Eigen::VectorXd* accepted_packed_step,
                  OrbitalObjective* accepted_trial_objective,
                  Eigen::VectorXd* accepted_trial_parameters,
                  Eigen::VectorXd* accepted_trial_gradient,
                  double* accepted_trial_energy) -> bool {
                Eigen::VectorXd fallback_reduced_direction =
                    -apply_nonredundant_truncated_newton_preconditioner(
                        current_space,
                        &transported_preconditioner,
                        current_projection.reduced_gradient);
                Eigen::VectorXd search_direction =
                    gather_nonredundant_retract_tangent(
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        fallback_reduced_direction);
                double directional_derivative =
                    current_gradient.dot(search_direction);
                if (!std::isfinite(directional_derivative) ||
                    directional_derivative >= 0.0 ||
                    is_effectively_zero_step(
                        search_direction,
                        current_parameters)) {
                  fallback_reduced_direction =
                      -current_projection.reduced_gradient;
                  search_direction =
                      gather_nonredundant_retract_tangent(
                          current_orbital_input,
                          current_space,
                          parameter_view,
                          fallback_reduced_direction);
                  directional_derivative =
                      current_gradient.dot(search_direction);
                }
                if (!std::isfinite(directional_derivative) ||
                    directional_derivative >= 0.0 ||
                    is_effectively_zero_step(
                        search_direction,
                        current_parameters)) {
                  return false;
                }

                const double reduced_search_direction_norm =
                    search_direction.norm();
                const double trust_radius_limited_initial_step =
                    std::isfinite(reduced_search_direction_norm) &&
                            reduced_search_direction_norm > 0.0
                        ? trust_radius / reduced_search_direction_norm
                        : options_.minimum_step_size;
                const double initial_fallback_step =
                    std::max(
                        options_.minimum_step_size,
                        std::min(
                            std::min(
                                std::min(1.0, options_.initial_step_size),
                                1.0 / std::max(1.0, reduced_gradient_inf_norm)),
                            trust_radius_limited_initial_step));
                // The descent fallback is entered only after the current
                // accepted-point Newton model already failed to produce an
                // acceptable trust-region step. Starting the Armijo backtrack
                // from a reduced step that already fits inside the current
                // trust radius avoids burning many full objective evaluations
                // just to rediscover the same radius contraction.
                OrbitalObjective fallback_objective =
                    objective.make_probe_copy();
                Eigen::VectorXd fallback_parameters(current_parameters.size());
                Eigen::VectorXd fallback_gradient(current_gradient.size());
                double fallback_energy = energy;
                if (!try_armijo_backtracking_nonredundant_direction(
                        &fallback_objective,
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        current_parameters,
                        energy,
                        current_gradient,
                        fallback_reduced_direction,
                        search_direction,
                        initial_fallback_step,
                        options_.minimum_step_size,
                        options_.armijo_constant,
                        &fallback_parameters,
                        &fallback_gradient,
                        &fallback_energy)) {
                  return false;
                }

                *accepted_packed_step =
                    fallback_parameters - current_parameters;
                *accepted_trial_objective =
                    std::move(fallback_objective);
                *accepted_trial_parameters = std::move(fallback_parameters);
                *accepted_trial_gradient = std::move(fallback_gradient);
                *accepted_trial_energy = fallback_energy;
                return true;
              };
          const Eigen::Index reduced_size =
              current_projection.reduced_gradient.size();
          TruncatedNewtonTrialEvaluation trial_evaluation_cache;
          const Eigen::VectorXd* initial_reduced_step_for_current_solve = nullptr;
          if (rejected_step_cache.has_cached_step(reduced_size)) {
            initial_reduced_step_for_current_solve =
                &rejected_step_cache.cached_step;
          }

          auto truncated_newton_step =
              solve_trust_region_in_krylov_subspace(
                  current_projection,
                  trust_radius,
                  cached_krylov_subspace);
          const bool reused_krylov_subspace =
              truncated_newton_step_is_usable(
                  truncated_newton_step,
                  current_projection.reduced_gradient);
          if (!reused_krylov_subspace) {
            cached_krylov_subspace = TruncatedNewtonKrylovSubspace();
            truncated_newton_step =
                solve_nonredundant_truncated_newton_step(
                    retraction_metric,
                    current_space,
                    current_projection,
                    trust_radius,
                    max_cg_iterations,
                    hvp_operator.get(),
                    &transported_preconditioner,
                    initial_reduced_step_for_current_solve);
          }
          clamp_nonredundant_step_result_to_retract_tangent_radius(
              current_orbital_input,
              current_space,
              parameter_view,
              current_projection,
              trust_radius,
              &truncated_newton_step);
          if (truncated_newton_krylov_subspace_is_usable(
                  truncated_newton_step.krylov_subspace,
                  current_projection.reduced_gradient.size())) {
            cached_krylov_subspace =
                truncated_newton_step.krylov_subspace;
          }
          const TruncatedNewtonStepResult model_step =
              truncated_newton_step;
          Eigen::VectorXd reduced_step = truncated_newton_step.reduced_step;
          if (reduced_step.size() != current_projection.reduced_gradient.size()) {
            result.termination_reason =
                "nonredundant_truncated_newton_invalid_step_dimension";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }
          double predicted_decrease =
              truncated_newton_step.predicted_decrease;
          if (!std::isfinite(predicted_decrease) ||
              predicted_decrease <= 0.0) {
            reduced_step =
                build_nonredundant_preconditioned_reduced_gradient_step(
                    retraction_metric,
                    current_space,
                    current_projection,
                    trust_radius,
                    &transported_preconditioner);
            predicted_decrease =
                estimate_nonredundant_reduced_model_decrease(
                    current_projection,
                    reduced_step,
                    hvp_operator.get());
            truncated_newton_step.reduced_step = reduced_step;
            truncated_newton_step.reduced_hessian_times_step.resize(0);
            truncated_newton_step.retract_tangent_norm =
                compute_nonredundant_retract_tangent_norm(
                    current_orbital_input,
                    current_space,
                    parameter_view,
                    reduced_step);
            truncated_newton_step.reached_boundary =
                truncated_newton_step.retract_tangent_norm >=
                (1.0 - 1.0e-8) * trust_radius;
            truncated_newton_step.predicted_decrease = predicted_decrease;
          }
          TruncatedNewtonStepResult trial_step_for_current_trial =
              truncated_newton_step;

          Eigen::VectorXd packed_step(current_parameters.size());
          OrbitalObjective::TrialEvaluation accepted_trial_evaluation;
          Eigen::VectorXd trial_parameters(current_parameters.size());
          Eigen::VectorXd trial_gradient(current_gradient.size());
          double trial_energy = energy;
          const bool screen_rejected_trials_with_energy_only =
              rejected_trial_step_count_for_current_point > 0 &&
              admit_energy_only_trial_screen();
          bool accepted_trial =
              try_truncated_newton_trial_step(
                  reduced_step,
                  predicted_decrease,
                  screen_rejected_trials_with_energy_only,
                  &trial_evaluation_cache,
                  &packed_step,
                  &accepted_trial_evaluation,
                  &trial_parameters,
                  &trial_gradient,
                  &trial_energy);
          if (!accepted_trial &&
              rejected_trial_step_count_for_current_point == 0 &&
              reduced_gradient_inf_norm >=
                  8.0 * options_.gradient_tolerance &&
              model_step.encountered_negative_curvature) {
            if (try_nonredundant_descent_fallback_step(
                    &packed_step,
                    &objective,
                    &trial_parameters,
                    &trial_gradient,
                    &trial_energy)) {
              accepted_trial = true;
              const double fallback_actual_decrease = energy - trial_energy;
              trial_evaluation_cache.actual_decrease =
                  fallback_actual_decrease;
              trial_evaluation_cache.predicted_decrease =
                  fallback_actual_decrease;
              truncated_newton_step.used_krylov_rescue = true;
              truncated_newton_step.reached_boundary = false;
              truncated_newton_step.encountered_negative_curvature = false;
              truncated_newton_step.cg_iterations = 0;
              truncated_newton_step.reduced_step =
                  current_space.project_vector(packed_step).reduced_gradient;
              truncated_newton_step.retract_tangent_norm = packed_step.norm();
              truncated_newton_step.predicted_decrease =
                  std::max(options_.energy_tolerance, energy - trial_energy);
            }
          }
          if (!accepted_trial) {
            ++rejected_trial_step_count_for_current_point;
            trust_radius =
                update_nonredundant_truncated_newton_trust_radius(
                    trust_radius,
                    options_.minimum_step_size,
                    trial_evaluation_cache,
                    trial_step_for_current_trial,
                    false);
            if (trust_radius <= options_.minimum_step_size) {
              result.termination_reason =
                  "nonredundant_truncated_newton_trust_radius_exhausted";
              final_gradient_l2_norm = current_projection.reduced_gradient.norm();
              break;
            }
            rejected_step_cache.update(
                current_orbital_input,
                current_space,
                parameter_view,
                model_step,
                reduced_size,
                trust_radius);
            continue;
          }
          if (accepted_trial_evaluation.valid) {
            objective.commit_trial_evaluation(
                std::move(accepted_trial_evaluation));
          }
          current_parameters = trial_parameters;
          current_gradient = std::move(trial_gradient);
          energy = trial_energy;
          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient,
                  &packed_secant_history);
          ++n_iterations;
          rejected_trial_step_count_for_current_point = 0;
          rejected_step_cache.clear();
          cached_krylov_subspace = TruncatedNewtonKrylovSubspace();
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          NonredundantOrbitalSpace next_space =
              build_nonredundant_space(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          const bool nonredundant_rank_changed =
              current_space.reduced_size() != next_space.reduced_size() ||
              current_space.rank_signature() != next_space.rank_signature();
          final_projected_gradient_inf_norm =
              gradient_infinity_norm(next_projection.reduced_gradient);
          final_projected_gradient_l2_norm =
              next_projection.reduced_gradient.norm();
          trust_radius =
              update_nonredundant_truncated_newton_trust_radius(
                  trust_radius,
                  options_.minimum_step_size,
                  trial_evaluation_cache,
                  truncated_newton_step,
                  true);
          if (nonredundant_rank_changed) {
            packed_secant_history.clear();
          }
          if (!accepted_point_chart_reset && !nonredundant_rank_changed) {
            const Eigen::VectorXd packed_projected_gradient_change =
                next_projection.packed_projected_gradient -
                current_projection.packed_projected_gradient;
            append_nonredundant_truncated_newton_secant_pair(
                packed_step,
                packed_projected_gradient_change,
                transport_history_size,
                &packed_secant_history);
          }
          if (std::abs(de) < options_.energy_tolerance &&
              gradient_infinity_norm(next_projection.reduced_gradient) <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_truncated_newton_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }
        result.final_projected_gradient_inf_norm =
            final_projected_gradient_inf_norm;
        result.final_projected_gradient_l2_norm =
            final_projected_gradient_l2_norm;
        final_projected_gradient_ready = true;
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
    if (!result.termination_reason.empty()) {
      throw std::runtime_error(
          std::string("optimizer did not evaluate the objective: ") +
          result.termination_reason);
    }
    throw std::runtime_error("optimizer did not evaluate the objective");
  }

  result.n_iterations = n_iterations;
  result.final_total_energy = result.total_energy_history.back();
  result.final_one_electron_reference_energy =
      result.scf_result.one_electron_reference_energy;
  result.final_gradient_inf_norm = result.gradient_inf_norm_history.back();
  result.final_gradient_l2_norm = final_gradient_l2_norm;
  if (optimizer_backend_uses_nonredundant_space(options_.backend) &&
      !final_projected_gradient_ready) {
    const NonredundantOrbitalSpace final_space =
        build_nonredundant_space(objective, parameter_view);
    const Eigen::VectorXd final_packed_gradient =
        parameter_view.gather_from_full(
            objective.last_gradient_result().sparse_orbital_energy_gradient);
    const auto final_projection =
        final_space.project_gradient(final_packed_gradient);
    result.final_projected_gradient_inf_norm =
        gradient_infinity_norm(final_projection.reduced_gradient);
    result.final_projected_gradient_l2_norm =
        final_projection.reduced_gradient.norm();
  }
  result.optimized_input = objective.last_input();
  Eigen::MatrixXd final_normalized_orbital_matrix =
      objective.last_gradient_result()
          .orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix;
  if (final_normalized_orbital_matrix.size() != 0) {
    // The evaluator always works with the normalized physical orbital frame,
    // mirroring legacy `normalize(...)`. Store that same frame in the final
    // sparse slots before exporting so Molden / restart artifacts see the
    // actual accepted physical orbitals rather than a pre-normalization raw
    // parameter vector.
    //
    // The OEO representative is an export gauge, not an optimizer mutation.
    // Rebuild it once from the converged auxiliary block for Molden/restart
    // output without perturbing the accepted-point tangent chart.
    final_normalized_orbital_matrix =
        build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
            result.optimized_input.orbital_preparation_input,
            objective.last_gradient_result().orbital_preparation_result,
            initial_normalized_orbital_matrix);
    overwrite_sparse_orbitals_from_dense_physical_frame(
        final_normalized_orbital_matrix,
        &result.optimized_input.orbital_preparation_input);
  }
  enforce_strict_sparse_orbital_support(
      &result.optimized_input.orbital_preparation_input);

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
