#include "vb/scf/exact_ctx_memory_accounting.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "vb/matrices/determinant_types.hpp"
#include "vb/orbital/active_space_one_electron_result.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"
#include "vb/orbital/ao_effective_one_electron_result.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"
#include "vb/orbital/physical_orbital_frame.hpp"

namespace xmvb::vb {

namespace {

bool parse_enabled_env(const char* env_name) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

std::size_t parse_positive_size_t_env(
    const char* env_name,
    std::size_t default_value) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  const long long parsed = std::stoll(value);
  if (parsed <= 0) {
    throw std::invalid_argument(
        std::string(env_name) + " must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

int exact_ctx_memory_accounting_top_entries() {
  return static_cast<int>(parse_positive_size_t_env(
      "XMVB_CPP_LOG_EXACT_CTX_MEMORY_TOP",
      32));
}

std::string format_bytes(std::size_t bytes) {
  static constexpr double kKiB = 1024.0;
  static constexpr double kMiB = 1024.0 * 1024.0;
  static constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  char buffer[64];
  if (bytes >= static_cast<std::size_t>(kGiB)) {
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%.2f GiB",
        static_cast<double>(bytes) / kGiB);
    return buffer;
  }
  if (bytes >= static_cast<std::size_t>(kMiB)) {
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%.2f MiB",
        static_cast<double>(bytes) / kMiB);
    return buffer;
  }
  if (bytes >= static_cast<std::size_t>(kKiB)) {
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%.2f KiB",
        static_cast<double>(bytes) / kKiB);
    return buffer;
  }
  std::snprintf(buffer, sizeof(buffer), "%zu B", bytes);
  return buffer;
}

std::size_t projection_bytes(const OppositeSpinPackedPairProjection& projection) {
  return exact_ctx_vector_capacity_bytes(projection.packed_pair_indices) +
      exact_ctx_vector_capacity_bytes(projection.packed_pair_values) +
      exact_ctx_vector_capacity_bytes(projection.projected_pair_values);
}

std::size_t overlap_result_bytes(const DeterminantOverlapResult& result) {
  return exact_ctx_matrix_bytes(result.inverse_overlap_submatrix) +
      exact_ctx_matrix_bytes(result.first_order_cofactor_matrix) +
      exact_ctx_vector_bytes(result.singular_values) +
      exact_ctx_matrix_bytes(result.matrix_U) +
      exact_ctx_matrix_bytes(result.matrix_V);
}

std::size_t same_spin_pair_dynamic_bytes(
    const SpinDeterminantPairEvaluation& evaluation) {
  return exact_ctx_matrix_bytes(evaluation.same_spin_inverse_overlap_gradient) +
      exact_ctx_matrix_bytes(evaluation.same_spin_overlap_hamiltonian_gradient) +
      overlap_result_bytes(evaluation.overlap_result) +
      projection_bytes(evaluation.opposite_spin_pair_cache.first_order_cofactor_projection) +
      projection_bytes(evaluation.opposite_spin_pair_cache.inverse_overlap_projection);
}

std::size_t physical_orbital_frame_bytes(const PhysicalOrbitalFrame& frame) {
  return exact_ctx_matrix_bytes(frame.normalized_orbital_matrix) +
      exact_ctx_matrix_bytes(frame.inactive_physical_orbital_matrix) +
      exact_ctx_matrix_bytes(frame.inactive_orthonormal_orbital_matrix) +
      exact_ctx_matrix_bytes(frame.active_physical_orbital_matrix) +
      exact_ctx_matrix_bytes(frame.inactive_orthonormal_gauge_transform) +
      exact_ctx_matrix_bytes(
          frame.localized_representative_selector.inactive_right_transform) +
      exact_ctx_matrix_bytes(
          frame.localized_representative_selector
              .inactive_inverse_transpose_right_transform) +
      exact_ctx_matrix_bytes(
          frame.localized_representative_selector.active_inactive_coefficients);
}

void append_orbital_preparation_result_memory_breakdown(
    const std::string& prefix,
    const OrbitalPreparationResult& result,
    ExactCtxMemoryBreakdown* breakdown) {
  breakdown->add_prefixed(
      prefix,
      "active_sparse_row_offsets",
      exact_ctx_vector_capacity_bytes(result.active_sparse_row_offsets));
  breakdown->add_prefixed(
      prefix,
      "active_sparse_orbital_indices",
      exact_ctx_vector_capacity_bytes(result.active_sparse_orbital_indices));
  breakdown->add_prefixed(
      prefix,
      "active_sparse_values",
      exact_ctx_vector_capacity_bytes(result.active_sparse_values));
  breakdown->add_prefixed(
      prefix,
      "auxiliary_orbital_matrix",
      exact_ctx_matrix_bytes(result.auxiliary_orbital_matrix));
  breakdown->add_prefixed(
      prefix,
      "active_orbital_overlap_matrix",
      exact_ctx_vector_capacity_bytes(result.active_orbital_overlap_matrix));
  breakdown->add_prefixed(
      prefix,
      "inactive_density_matrix",
      exact_ctx_matrix_bytes(result.inactive_density_matrix));
  breakdown->add_prefixed(
      prefix,
      "inactive_orthonormal_projector_matrix",
      exact_ctx_matrix_bytes(result.inactive_orthonormal_projector_matrix));
  breakdown->add_prefixed(
      prefix,
      "inactive_density_low_rank_factors",
      exact_ctx_matrix_bytes(result.inactive_density_low_rank_factors));
  breakdown->add_prefixed(
      prefix,
      "occupied_space_projector",
      exact_ctx_vector_capacity_bytes(result.occupied_space_projector));
  breakdown->add_prefixed(
      prefix,
      "inactive_auxiliary_transform",
      exact_ctx_vector_capacity_bytes(result.inactive_auxiliary_transform));
  breakdown->add_prefixed(
      prefix,
      "inactive_active_overlap_matrix",
      exact_ctx_vector_capacity_bytes(result.inactive_active_overlap_matrix));
  breakdown->add_prefixed(
      prefix,
      "projected_active_overlap_matrix",
      exact_ctx_vector_capacity_bytes(result.projected_active_overlap_matrix));
  breakdown->add_prefixed(
      prefix,
      "auxiliary_orbital_inverse_matrix",
      exact_ctx_vector_capacity_bytes(result.auxiliary_orbital_inverse_matrix));
  breakdown->add_prefixed(
      prefix,
      "physical_orbital_frame",
      physical_orbital_frame_bytes(result.physical_orbital_frame));
}

void append_ao_effective_one_electron_result_memory_breakdown(
    const std::string& prefix,
    const AoEffectiveOneElectronResult& result,
    ExactCtxMemoryBreakdown* breakdown) {
  breakdown->add_prefixed(
      prefix,
      "ao_coulomb_exchange_matrix",
      exact_ctx_matrix_bytes(result.ao_coulomb_exchange_matrix));
  breakdown->add_prefixed(
      prefix,
      "ao_effective_h1e",
      exact_ctx_matrix_bytes(result.ao_effective_h1e));
}

void append_active_space_one_electron_result_memory_breakdown(
    const std::string& prefix,
    const ActiveSpaceOneElectronResult& result,
    ExactCtxMemoryBreakdown* breakdown) {
  breakdown->add_prefixed(
      prefix,
      "h1e_act",
      exact_ctx_matrix_bytes(result.h1e_act));
}

void append_active_space_two_electron_result_memory_breakdown(
    const std::string& prefix,
    const ActiveSpaceTwoElectronResult& result,
    ExactCtxMemoryBreakdown* breakdown) {
  breakdown->add_prefixed(
      prefix,
      "packed_active_two_electron_integrals",
      exact_ctx_vector_capacity_bytes(result.packed_active_two_electron_integrals));
  breakdown->add_prefixed(
      prefix,
      "ri_active_pair_factors",
      exact_ctx_matrix_bytes(result.ri_active_pair_factors));
  breakdown->add_prefixed(
      prefix,
      "dense_active_coefficients",
      exact_ctx_matrix_bytes(result.dense_active_coefficients));
  breakdown->add_prefixed(
      prefix,
      "dense_ao_pair_products",
      exact_ctx_matrix_bytes(result.dense_ao_pair_products));
}

void append_spin_determinant_reuse_table_memory_breakdown(
    const std::string& prefix,
    const SpinDeterminantReuseTable& table,
    ExactCtxMemoryBreakdown* breakdown) {
  breakdown->add_prefixed(
      prefix,
      "unique_determinants",
      exact_ctx_nested_vector_capacity_bytes(table.unique_determinants));
  breakdown->add_prefixed(
      prefix,
      "determinant_to_unique_id",
      exact_ctx_vector_capacity_bytes(table.determinant_to_unique_id));
}

void append_same_spin_pair_vector_memory_breakdown(
    const std::string& prefix,
    const std::vector<SpinDeterminantPairEvaluation>& evaluations,
    ExactCtxMemoryBreakdown* breakdown) {
  std::size_t dynamic_bytes = 0;
  std::size_t inverse_overlap_gradient_bytes = 0;
  std::size_t overlap_hamiltonian_gradient_bytes = 0;
  std::size_t overlap_payload_bytes = 0;
  std::size_t opposite_spin_projection_bytes = 0;
  for (const auto& evaluation : evaluations) {
    inverse_overlap_gradient_bytes +=
        exact_ctx_matrix_bytes(evaluation.same_spin_inverse_overlap_gradient);
    overlap_hamiltonian_gradient_bytes +=
        exact_ctx_matrix_bytes(evaluation.same_spin_overlap_hamiltonian_gradient);
    overlap_payload_bytes += overlap_result_bytes(evaluation.overlap_result);
    opposite_spin_projection_bytes +=
        projection_bytes(
            evaluation.opposite_spin_pair_cache.first_order_cofactor_projection) +
        projection_bytes(
            evaluation.opposite_spin_pair_cache.inverse_overlap_projection);
    dynamic_bytes += same_spin_pair_dynamic_bytes(evaluation);
  }
  breakdown->add_prefixed(
      prefix,
      "container",
      exact_ctx_vector_capacity_bytes(evaluations));
  breakdown->add_prefixed(
      prefix,
      "same_spin_inverse_overlap_gradients",
      inverse_overlap_gradient_bytes);
  breakdown->add_prefixed(
      prefix,
      "same_spin_overlap_hamiltonian_gradients",
      overlap_hamiltonian_gradient_bytes);
  breakdown->add_prefixed(
      prefix,
      "overlap_payloads",
      overlap_payload_bytes);
  breakdown->add_prefixed(
      prefix,
      "opposite_spin_projections",
      opposite_spin_projection_bytes);
  breakdown->add_prefixed(
      prefix,
      "dynamic_total",
      dynamic_bytes);
}

std::size_t selected_state_determinant_coefficients_bytes(
    const SelectedStateDeterminantCoefficients& state) {
  return exact_ctx_vector_capacity_bytes(state.determinant_coefficients) +
      exact_ctx_matrix_bytes(state.coefficient_matrix) +
      exact_ctx_vector_capacity_bytes(state.diagonal_coefficients) +
      exact_ctx_vector_capacity_bytes(state.alpha_support) +
      exact_ctx_vector_capacity_bytes(state.beta_support) +
      exact_ctx_matrix_bytes(state.local_coefficient_matrix) +
      exact_ctx_vector_capacity_bytes(state.local_diagonal_coefficients);
}

std::size_t selected_state_directional_column_cache_bytes(
    const AcceptedSelectedStateEigenResponseColumnCache& cache) {
  return exact_ctx_vector_bytes(cache.energy_gaps) +
      exact_ctx_vector_bytes(cache.gap_tolerances) +
      exact_ctx_array_bytes(cache.uses_equal_weight_gauge);
}

std::size_t structure_coefficient_block_bytes(const StructureCoefficientBlock& block) {
  return exact_ctx_vector_capacity_bytes(block.alpha_support) +
      exact_ctx_vector_capacity_bytes(block.beta_support) +
      exact_ctx_matrix_bytes(block.local_coefficients) +
      exact_ctx_vector_capacity_bytes(block.local_diagonal_coefficients);
}

}  // namespace

void ExactCtxMemoryBreakdown::add(std::string label, std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  total_bytes_ += bytes;
  entries_.push_back({std::move(label), bytes});
}

void ExactCtxMemoryBreakdown::add_prefixed(
    const std::string& prefix,
    const std::string& label,
    std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (prefix.empty()) {
    add(label, bytes);
    return;
  }
  add(prefix + "." + label, bytes);
}

bool exact_ctx_memory_accounting_enabled() {
  return parse_enabled_env("XMVB_CPP_LOG_EXACT_CTX_MEMORY");
}

std::size_t exact_ctx_memory_accounting_min_bytes() {
  static constexpr std::size_t kDefaultMinBytes = 1u << 20;
  return parse_positive_size_t_env(
      "XMVB_CPP_LOG_EXACT_CTX_MEMORY_MIN_BYTES",
      kDefaultMinBytes);
}

void maybe_log_exact_ctx_memory_breakdown(
    const char* scope,
    const ExactCtxMemoryBreakdown& breakdown) {
  if (!exact_ctx_memory_accounting_enabled()) {
    return;
  }
  const char* safe_scope =
      (scope != nullptr && scope[0] != '\0') ? scope : "exact_ctx";
  std::vector<ExactCtxMemoryEntry> sorted_entries = breakdown.entries();
  std::stable_sort(
      sorted_entries.begin(),
      sorted_entries.end(),
      [](const ExactCtxMemoryEntry& left, const ExactCtxMemoryEntry& right) {
        return left.bytes > right.bytes;
      });
  const std::size_t min_bytes = exact_ctx_memory_accounting_min_bytes();
  const int top_entries = exact_ctx_memory_accounting_top_entries();
  std::fprintf(
      stderr,
      "[exact_ctx_mem] %s total=%s (%zu bytes)\n",
      safe_scope,
      format_bytes(breakdown.total_bytes()).c_str(),
      breakdown.total_bytes());
  int printed = 0;
  int omitted = 0;
  for (const auto& entry : sorted_entries) {
    if (entry.bytes < min_bytes || printed >= top_entries) {
      ++omitted;
      continue;
    }
    std::fprintf(
        stderr,
        "[exact_ctx_mem]   %s = %s (%zu bytes)\n",
        entry.label.c_str(),
        format_bytes(entry.bytes).c_str(),
        entry.bytes);
    ++printed;
  }
  if (omitted > 0) {
    std::fprintf(
        stderr,
        "[exact_ctx_mem]   omitted=%d entries below threshold/limit\n",
        omitted);
  }
  std::fflush(stderr);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const PreparedActiveSpaceContext& context,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  append_orbital_preparation_result_memory_breakdown(
      prefix + ".orbital_result",
      context.orbital_result,
      breakdown);
  append_ao_effective_one_electron_result_memory_breakdown(
      prefix + ".ao_effective_one_electron_result",
      context.ao_effective_one_electron_result,
      breakdown);
  append_active_space_one_electron_result_memory_breakdown(
      prefix + ".active_space_one_electron_result",
      context.active_space_one_electron_result,
      breakdown);
  append_active_space_two_electron_result_memory_breakdown(
      prefix + ".active_space_two_electron_result",
      context.active_space_two_electron_result,
      breakdown);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const SameSpinPairCacheContext& context,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  append_spin_determinant_reuse_table_memory_breakdown(
      prefix + ".alpha_reuse_table",
      context.alpha_reuse_table,
      breakdown);
  append_spin_determinant_reuse_table_memory_breakdown(
      prefix + ".beta_reuse_table",
      context.beta_reuse_table,
      breakdown);
  append_same_spin_pair_vector_memory_breakdown(
      prefix + ".alpha_pair_cache",
      context.alpha_pair_cache,
      breakdown);
  append_same_spin_pair_vector_memory_breakdown(
      prefix + ".beta_pair_cache",
      context.beta_pair_cache,
      breakdown);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const StructureAccumulationResult& structure_matrices,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "overlap_matrix",
      exact_ctx_vector_capacity_bytes(structure_matrices.overlap_matrix));
  breakdown->add_prefixed(
      prefix,
      "hamiltonian_matrix",
      exact_ctx_vector_capacity_bytes(structure_matrices.hamiltonian_matrix));
  breakdown->add_prefixed(
      prefix,
      "determinant_overlap_cache",
      exact_ctx_vector_capacity_bytes(
          structure_matrices.determinant_overlap_cache));
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const xmvb::core::GeneralizedEigenResult& eigen_result,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "eigenvalues",
      exact_ctx_vector_capacity_bytes(eigen_result.eigenvalues));
  breakdown->add_prefixed(
      prefix,
      "eigenvector_matrix",
      exact_ctx_vector_capacity_bytes(eigen_result.eigenvector_matrix));
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "selected_state_indices",
      exact_ctx_vector_capacity_bytes(selected_state_matrices.selected_state_indices));
  breakdown->add_prefixed(
      prefix,
      "normalized_state_weights",
      exact_ctx_vector_capacity_bytes(selected_state_matrices.normalized_state_weights));
  breakdown->add_prefixed(
      prefix,
      "states_container",
      exact_ctx_vector_capacity_bytes(selected_state_matrices.states));
  breakdown->add_prefixed(
      prefix,
      "determinant_to_unique_alpha_id",
      exact_ctx_vector_capacity_bytes(
          selected_state_matrices.determinant_to_unique_alpha_id));
  breakdown->add_prefixed(
      prefix,
      "determinant_to_unique_beta_id",
      exact_ctx_vector_capacity_bytes(
          selected_state_matrices.determinant_to_unique_beta_id));
  std::size_t states_payload_bytes = 0;
  for (const auto& state : selected_state_matrices.states) {
    states_payload_bytes += selected_state_determinant_coefficients_bytes(state);
  }
  breakdown->add_prefixed(
      prefix,
      "states_payload",
      states_payload_bytes);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const CppActiveSpaceSecondOrderContext& context,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  append_exact_ctx_memory_breakdown(
      prefix + ".prepared_active_space",
      context.prepared_active_space,
      breakdown);
  append_exact_ctx_memory_breakdown(
      prefix + ".same_spin_pair_cache",
      context.same_spin_pair_cache,
      breakdown);
  append_exact_ctx_memory_breakdown(
      prefix + ".structure_matrices",
      context.structure_matrices,
      breakdown);
  breakdown->add_prefixed(
      prefix,
      "active_orbital_overlap_gradient",
      exact_ctx_vector_capacity_bytes(context.active_orbital_overlap_gradient));
  breakdown->add_prefixed(
      prefix,
      "active_one_electron_gradient",
      exact_ctx_vector_capacity_bytes(context.active_one_electron_gradient));
  breakdown->add_prefixed(
      prefix,
      "packed_active_two_electron_gradient",
      exact_ctx_vector_capacity_bytes(context.packed_active_two_electron_gradient));
  append_exact_ctx_memory_breakdown(
      prefix + ".eigen_result",
      context.eigen_result,
      breakdown);
  breakdown->add_prefixed(
      prefix,
      "selected_state_indices",
      exact_ctx_vector_capacity_bytes(context.selected_state_indices));
  breakdown->add_prefixed(
      prefix,
      "normalized_state_weights",
      exact_ctx_vector_capacity_bytes(context.normalized_state_weights));
  breakdown->add_prefixed(
      prefix,
      "selected_state_energies",
      exact_ctx_vector_capacity_bytes(context.selected_state_energies));
  append_exact_ctx_memory_breakdown(
      prefix + ".selected_state_matrices",
      context.selected_state_matrices,
      breakdown);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const std::vector<StructureCoefficientBlock>& blocks,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "container",
      exact_ctx_vector_capacity_bytes(blocks));
  std::size_t payload_bytes = 0;
  for (const auto& block : blocks) {
    payload_bytes += structure_coefficient_block_bytes(block);
  }
  breakdown->add_prefixed(
      prefix,
      "payload",
      payload_bytes);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "ao_pair_first_indices",
      exact_ctx_vector_capacity_bytes(cache.ao_pair_first_indices));
  breakdown->add_prefixed(
      prefix,
      "ao_pair_second_indices",
      exact_ctx_vector_capacity_bytes(cache.ao_pair_second_indices));
  breakdown->add_prefixed(
      prefix,
      "active_pair_first_indices",
      exact_ctx_vector_capacity_bytes(cache.active_pair_first_indices));
  breakdown->add_prefixed(
      prefix,
      "active_pair_second_indices",
      exact_ctx_vector_capacity_bytes(cache.active_pair_second_indices));
  breakdown->add_prefixed(
      prefix,
      "active_pair_gradient_matrix",
      exact_ctx_matrix_bytes(cache.active_pair_gradient_matrix));
  breakdown->add_prefixed(
      prefix,
      "accepted_active_pair_gradient_backprop_rows_buffer",
      exact_ctx_vector_capacity_bytes(
          cache.accepted_active_pair_gradient_backprop_rows_buffer));
  breakdown->add_prefixed(
      prefix,
      "accepted_dense_active_coefficients",
      exact_ctx_matrix_bytes(
          cache.accepted_dense_active_coefficients));
  breakdown->add_prefixed(
      prefix,
      "accepted_pair_coefficients",
      exact_ctx_matrix_bytes(cache.accepted_pair_coefficients));
  breakdown->add_prefixed(
      prefix,
      "accepted_base_pair_products",
      exact_ctx_matrix_bytes(cache.accepted_base_pair_products));
  breakdown->add_prefixed(
      prefix,
      "accepted_base_pair_gradients",
      exact_ctx_matrix_bytes(cache.accepted_base_pair_gradients));
  breakdown->add_prefixed(
      prefix,
      "accepted_base_pair_gradient_matrices_buffer",
      exact_ctx_vector_capacity_bytes(
          cache.accepted_base_pair_gradient_matrices_buffer));
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const ExactPackedActiveTwoElectronApplyWorkspace& workspace,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "dense_active_gradient_direction",
      exact_ctx_matrix_bytes(workspace.dense_active_gradient_direction));
  breakdown->add_prefixed(
      prefix,
      "dense_active_direction",
      exact_ctx_matrix_bytes(workspace.dense_active_direction));
  breakdown->add_prefixed(
      prefix,
      "mixed_pair_coefficients",
      exact_ctx_matrix_bytes(workspace.mixed_pair_coefficients));
  breakdown->add_prefixed(
      prefix,
      "transformed_pair_coefficients",
      exact_ctx_matrix_bytes(
          workspace.transformed_pair_coefficients));
  breakdown->add_prefixed(
      prefix,
      "pair_gradients",
      exact_ctx_matrix_bytes(workspace.pair_gradients));
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace& workspace,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "dense_active_direction",
      exact_ctx_matrix_bytes(workspace.dense_active_direction));
  breakdown->add_prefixed(
      prefix,
      "pair_coefficients",
      exact_ctx_matrix_bytes(workspace.pair_coefficients));
  breakdown->add_prefixed(
      prefix,
      "base_pair_products",
      exact_ctx_matrix_bytes(workspace.base_pair_products));
  breakdown->add_prefixed(
      prefix,
      "directional_pair_coefficients",
      exact_ctx_matrix_bytes(
          workspace.directional_pair_coefficients));
  breakdown->add_prefixed(
      prefix,
      "directional_pair_products",
      exact_ctx_matrix_bytes(workspace.directional_pair_products));
  breakdown->add_prefixed(
      prefix,
      "delta_active_pair_matrix",
      exact_ctx_matrix_bytes(workspace.delta_active_pair_matrix));
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const AcceptedOuterResponseLinearResponseCache& cache,
    ExactCtxMemoryBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("memory breakdown must not be null");
  }
  breakdown->add_prefixed(
      prefix,
      "selected_state_indices",
      exact_ctx_vector_capacity_bytes(cache.selected_state_indices));
  breakdown->add_prefixed(
      prefix,
      "accepted_selected_eigenvector_columns",
      exact_ctx_matrix_bytes(cache.accepted_selected_eigenvector_columns));
  breakdown->add_prefixed(
      prefix,
      "selected_state_eigen_response_operator.accepted_eigenvalues",
      exact_ctx_vector_bytes(
          cache.selected_state_eigen_response_operator.accepted_eigenvalues));
  breakdown->add_prefixed(
      prefix,
      "selected_state_eigen_response_operator.selected_columns_container",
      exact_ctx_vector_capacity_bytes(
          cache.selected_state_eigen_response_operator.selected_columns));
  std::size_t selected_columns_payload_bytes = 0;
  for (const auto& selected_column :
       cache.selected_state_eigen_response_operator.selected_columns) {
    selected_columns_payload_bytes +=
        selected_state_directional_column_cache_bytes(selected_column);
  }
  breakdown->add_prefixed(
      prefix,
      "selected_state_eigen_response_operator.selected_columns_payload",
      selected_columns_payload_bytes);
}

}  // namespace xmvb::vb
