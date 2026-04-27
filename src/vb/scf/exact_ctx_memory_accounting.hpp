#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/exact_orbital_second_order_operator_outer_response_internal.hpp"

namespace xmvb::vb {

/**
 * @brief One labeled owned-memory contribution in the exact_ctx pipeline.
 *
 * The memory debugging path reports resident bytes by object/field instead of
 * only a process-wide RSS.  Each entry stores a stable label together with the
 * owned buffer size estimated from `std::vector::capacity()` and Eigen dense
 * dimensions so accepted-point caches can be ranked directly in logs.
 */
struct ExactCtxMemoryEntry {
  std::string label;
  std::size_t bytes = 0;
};

/**
 * @brief Collects a flat exact_ctx memory breakdown before logging.
 *
 * The breakdown is intentionally flat: every caller contributes
 * `label -> owned bytes` pairs, and the logger sorts those pairs by size. This
 * keeps accepted-context and accepted-cache reports easy to compare between
 * MnF2 runs without reconstructing a tree from nested JSON-like output.
 */
class ExactCtxMemoryBreakdown {
public:
  void add(std::string label, std::size_t bytes);

  void add_prefixed(
      const std::string& prefix,
      const std::string& label,
      std::size_t bytes);

  std::size_t total_bytes() const noexcept {
    return total_bytes_;
  }

  const std::vector<ExactCtxMemoryEntry>& entries() const noexcept {
    return entries_;
  }

private:
  std::vector<ExactCtxMemoryEntry> entries_;
  std::size_t total_bytes_ = 0;
};

bool exact_ctx_memory_accounting_enabled();

std::size_t exact_ctx_memory_accounting_min_bytes();

void maybe_log_exact_ctx_memory_breakdown(
    const char* scope,
    const ExactCtxMemoryBreakdown& breakdown);

template <typename T>
inline std::size_t exact_ctx_vector_capacity_bytes(const std::vector<T>& values) {
  return values.capacity() * sizeof(T);
}

template <typename T>
inline std::size_t exact_ctx_nested_vector_capacity_bytes(
    const std::vector<std::vector<T>>& values) {
  std::size_t bytes = exact_ctx_vector_capacity_bytes(values);
  for (const auto& inner : values) {
    bytes += exact_ctx_vector_capacity_bytes(inner);
  }
  return bytes;
}

template <typename Derived>
inline std::size_t exact_ctx_matrix_bytes(const Eigen::MatrixBase<Derived>& matrix) {
  return static_cast<std::size_t>(matrix.size()) *
      sizeof(typename Derived::Scalar);
}

inline std::size_t exact_ctx_vector_bytes(const Eigen::VectorXd& vector) {
  return static_cast<std::size_t>(vector.size()) * sizeof(double);
}

inline std::size_t exact_ctx_array_bytes(const Eigen::ArrayXi& array) {
  return static_cast<std::size_t>(array.size()) * sizeof(Eigen::ArrayXi::Scalar);
}

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const PreparedActiveSpaceContext& context,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const SameSpinPairCacheContext& context,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const StructureAccumulationResult& structure_matrices,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const xmvb::core::GeneralizedEigenResult& eigen_result,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const CppActiveSpaceSecondOrderContext& context,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const std::vector<StructureCoefficientBlock>& blocks,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const ExactPackedActiveTwoElectronApplyWorkspace& workspace,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace& workspace,
    ExactCtxMemoryBreakdown* breakdown);

void append_exact_ctx_memory_breakdown(
    const std::string& prefix,
    const AcceptedOuterResponseLinearResponseCache& cache,
    ExactCtxMemoryBreakdown* breakdown);

}  // namespace xmvb::vb
