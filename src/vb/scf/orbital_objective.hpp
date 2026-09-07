#ifndef XMVB_VB_SCF_ORBITAL_OBJECTIVE_HPP_
#define XMVB_VB_SCF_ORBITAL_OBJECTIVE_HPP_

// OrbitalObjective wraps the SCF + orbital-gradient evaluators into the
// L-BFGS / truncated-Newton objective surface used by cpp_vb_scf_optimizer.
// The class owns the accepted orbital point, the last committed gradient
// result, and the energy / gradient-norm histories; trial evaluations are
// staged into a scratch buffer so rejected trust-region steps do not pay for
// full gradient, adjoint, and second-order-context construction.
//
// The helpers declared below this class are the orbital-chart canonicalization
// machinery that the class drives at each accepted point. They are promoted
// out of cpp_vb_scf_optimizer.cpp's anonymous namespace so that the objective
// can live in its own TU; they remain internal to the optimizer subsystem and
// should not be picked up by other modules.

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_orbital_gradient_result.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/scf/optimizer_types.hpp"

namespace xmvb::vb {

struct CppActiveSpaceSecondOrderContext;
class LocalizedRepresentativeSelector;
struct OrbitalPreparationResult;
struct SupportAwareInactiveMoGaugeTransform;

// One accepted or trialed orbital point evaluated without committing state.
// Carries everything the optimizer needs to decide whether to accept the
// step: gradient in packed coordinates, SCF total energy, infinity-norm of
// the gradient, wall time, and the inputs that produced it.
struct OrbitalObjectiveTrialEvaluation {
  OrbitalPreparationInput orbital_preparation_input;
  CppOrbitalGradientResult gradient_result;
  Eigen::VectorXd gradient;
  double energy = 0.0;
  double gradient_inf_norm = 0.0;
  double wall_time_seconds = 0.0;
  bool valid = false;
};

class OrbitalObjective {
 public:
  using TrialEvaluation = OrbitalObjectiveTrialEvaluation;

  OrbitalObjective(
      const CppVbInput& input,
      SparseOrbitalParameterView parameter_view,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy,
      const CppOrbitalGradientEvaluator* orbital_gradient_evaluator,
      const CppVbScfEvaluator* scf_evaluator);

  double operator()(const Eigen::VectorXd& parameter_vector,
                    Eigen::VectorXd& gradient);

  const CppVbInput& last_input() const { return working_input_; }
  const CppOrbitalGradientResult& last_gradient_result() const {
    return last_gradient_result_;
  }
  const std::shared_ptr<CppActiveSpaceSecondOrderContext>&
  last_second_order_context() const {
    return last_gradient_result_.second_order_context;
  }

  void ensure_last_reference_energy_gradient();

  const std::vector<double>& energy_history() const { return energy_history_; }
  const std::vector<double>& gradient_inf_norm_history() const {
    return gradient_inf_norm_history_;
  }
  const std::vector<double>& iteration_time_history_seconds() const {
    return iteration_time_history_seconds_;
  }
  std::size_t call_count() const { return energy_history_.size(); }
  double objective_wall_time_seconds() const {
    return objective_wall_time_seconds_;
  }
  std::size_t energy_only_call_count() const { return energy_only_call_count_; }
  double energy_only_wall_time_seconds() const {
    return energy_only_wall_time_seconds_;
  }
  double last_energy_only_wall_time_seconds() const {
    return last_energy_only_wall_time_seconds_;
  }
  double last_gradient_inf_norm() const {
    if (gradient_inf_norm_history_.empty()) {
      return 0.0;
    }
    return gradient_inf_norm_history_.back();
  }

  TrialEvaluation evaluate_trial_without_committing(
      const Eigen::VectorXd& parameter_vector) const;

  void commit_trial_evaluation(TrialEvaluation evaluation);

  double evaluate_energy_only(const Eigen::VectorXd& parameter_vector) const;

  bool canonicalize_orbital_chart_at_current_point(
      Eigen::VectorXd* parameter_vector,
      Eigen::VectorXd* gradient,
      std::vector<PackedSecantPair>* packed_secant_history = nullptr);

  OrbitalObjective make_probe_copy() const;

 private:
  CppVbInput working_input_;
  mutable CppVbInput probe_input_buffer_;
  SparseOrbitalParameterView parameter_view_;
  std::vector<int> selected_state_indices_;
  std::vector<double> state_average_weights_;
  double nuclear_repulsion_energy_ = 0.0;
  const CppOrbitalGradientEvaluator* orbital_gradient_evaluator_ = nullptr;
  const CppVbScfEvaluator* scf_evaluator_ = nullptr;

  CppOrbitalGradientResult last_gradient_result_;
  std::vector<double> energy_history_;
  std::vector<double> gradient_inf_norm_history_;
  std::vector<double> iteration_time_history_seconds_;
  double objective_wall_time_seconds_ = 0.0;
  mutable std::size_t energy_only_call_count_ = 0;
  mutable double energy_only_wall_time_seconds_ = 0.0;
  mutable double last_energy_only_wall_time_seconds_ = 0.0;
};

// ---- Orbital-chart canonicalization helpers -------------------------------
//
// All helpers below operate on the accepted-point orbital chart and are
// promoted out of cpp_vb_scf_optimizer.cpp's anonymous namespace so that
// OrbitalObjective (now in its own TU) can call them. They remain internal
// to the optimizer subsystem; other modules should not pick them up.

bool orbital_uses_full_ao_support(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index);

void require_full_ao_orbital_block(
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    int orbital_count,
    const char* label);

Eigen::MatrixXd build_dense_orbital_block_from_sparse_input(
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    int orbital_count);

Eigen::MatrixXd build_dense_orbital_block_from_full_vector(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& full_vector,
    int first_orbital,
    int orbital_count);

void scatter_dense_orbital_block_to_full_vector(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_block,
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    std::vector<double>* full_vector);

void transform_sparse_oeo_active_representative_gradient(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_gradient);

void transform_sparse_inactive_orbital_step(
    const SupportAwareInactiveMoGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_step);

void transform_sparse_oeo_active_representative_step(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_step);

std::vector<double> build_full_sparse_vector_from_packed(
    const SparseOrbitalParameterView& parameter_view,
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& packed_vector);

void overwrite_packed_vector_from_full_sparse(
    const SparseOrbitalParameterView& parameter_view,
    const std::vector<double>& full_sparse_vector,
    Eigen::VectorXd* packed_vector);

void transport_packed_secant_history_with_support_aware_inactive_gauge(
    const SupportAwareInactiveMoGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseOrbitalParameterView& parameter_view,
    std::vector<PackedSecantPair>* packed_secant_history);

void transport_packed_secant_history_with_oeo_active_representative_reset(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseOrbitalParameterView& parameter_view,
    std::vector<PackedSecantPair>* packed_secant_history);

void refresh_cached_localized_representative_selector(
    const OrbitalPreparationInput& orbital_preparation_input,
    OrbitalPreparationResult* orbital_result);

void overwrite_sparse_orbitals_from_dense_physical_frame(
    const Eigen::MatrixXd& dense_orbitals,
    OrbitalPreparationInput* orbital_preparation_input);

Eigen::MatrixXd build_self_adjoint_matrix_power(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double exponent,
    const char* label);

Eigen::MatrixXd build_inactive_metric_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

Eigen::MatrixXd build_metric_preserving_inactive_repaired_active_physical_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

Eigen::MatrixXd build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalPreparationResult& orbital_result,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_normalized_orbital_matrix);

}  // namespace xmvb::vb

#endif  // XMVB_VB_SCF_ORBITAL_OBJECTIVE_HPP_
