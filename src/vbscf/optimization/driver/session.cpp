#include "vbscf/optimization/driver/session.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {

bool uses_nonredundant_space(VbScfOptimizerBackend backend) {
  switch (backend) {
    case VbScfOptimizerBackend::NonredundantProjectedGradient:
    case VbScfOptimizerBackend::NonredundantLbfgspp:
    case VbScfOptimizerBackend::NonredundantTruncatedNewton:
      return true;
    case VbScfOptimizerBackend::Lbfgspp:
      return false;
  }
  return false;
}

OrbitalChart build_orbital_chart(
    const VbScfObjective& objective,
    const SparseParameterLayout& parameter_view) {
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
  return OrbitalChart(
      orbital_preparation_input,
      parameter_view,
      orbital_preparation_result.auxiliary_orbital_matrix.leftCols(n_occupied_orbitals),
      normalized_orbital_matrix,
      &objective.last_gradient_result().ao_effective_one_electron_result.ao_effective_h1e);
}

int choose_truncated_newton_max_cg_iterations(
    const VbScfOptimizerOptions& options,
    int reduced_size) {
  if (options.nonredundant_truncated_newton_max_cg_iterations > 0) {
    return std::min(
        std::max(1, reduced_size),
        options.nonredundant_truncated_newton_max_cg_iterations);
  }
  constexpr int kDefaultKrylovSafetyLimit = 32;
  return std::min(std::max(1, reduced_size), kDefaultKrylovSafetyLimit);
}

int choose_truncated_newton_transport_history_size(
    const VbScfOptimizerOptions& options) {
  return std::max(
      0,
      options.nonredundant_truncated_newton_transport_history_size);
}

void sync_result_from_objective(
    const VbScfObjective& objective,
    VbScfOptimizerResult* result) {
  result->total_energy_history = objective.energy_history();
  result->gradient_inf_norm_history = objective.gradient_inf_norm_history();
  result->iteration_time_history_seconds = objective.iteration_time_history_seconds();
  result->scf_result = objective.last_gradient_result().scf_result;
}

void record_accepted_iteration_snapshot(
    VbScfObjective* objective,
    int accepted_iteration_index,
    const VbScfOptimizerOptions& options,
    VbScfOptimizerResult* result) {
  if (!options.retain_accepted_iteration_trace && !options.accepted_iteration_callback) {
    return;
  }
  const bool include_reference_energy_gradient =
      options.retain_accepted_iteration_trace ||
      options.accepted_iteration_callback_requires_reference_gradient;
  const bool include_full_payload =
      options.retain_accepted_iteration_trace ||
      options.accepted_iteration_callback_requires_full_snapshot;
  if (include_reference_energy_gradient) {
    objective->ensure_last_reference_energy_gradient();
  }
  const auto& gradient_result = objective->last_gradient_result();
  VbScfAcceptedIterationSnapshot snapshot;
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

}  // namespace xmvb::vb::optimizer_detail
