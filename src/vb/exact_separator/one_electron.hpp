#pragma once

#include <cstdint>
#include <vector>

#include "vb/exact_separator/component_data.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Exact validation statistics for the component-ordered one-electron recurrence.
 *
 * `exact_*` are the determinant-pair reference values. `collapsed_*` are the
 * values reconstructed from the separator recurrence. The work counters track
 * the size of the reduced state space visited by the current implementation.
 * In the current one-leaf boundary path, `collapsed_leaf_state_count` counts
 * visited one-spin boundary sectors and `dp_transition_count` is zero.
 */
struct CollapsedOneElectronStarPairStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double overlap_absolute_error = 0.0;
  double exact_one_electron = 0.0;
  double collapsed_one_electron = 0.0;
  double one_electron_absolute_error = 0.0;
  double constructed_one_electron = 0.0;
  double constructed_one_electron_absolute_error = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Production one-electron totals for one component-ordered star pair.
 *
 * This is the exact separator recurrence result without any determinant-space
 * reference bookkeeping. The values correspond to the collapsed overlap and
 * one-electron matrix element only, plus the work counters needed for
 * performance benchmarking. In the current one-leaf boundary path,
 * `collapsed_leaf_state_count` counts visited one-spin boundary sectors and
 * `dp_transition_count` is zero.
 */
struct OneElectronStarPairResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Evaluates the exact component-ordered open-state one-electron recurrence.
 *
 * The support matrices use the repository-wide column-major storage convention
 * `storage[column * support_size + row]` with rows on the ket/right side and
 * columns on the bra/left side. `ordered_components.front()` must be the root
 * component, and the remaining entries must be the leaf components in the same
 * order used by the separator recurrence.
 */
CollapsedOneElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_one_electron(
    double exact_overlap,
    double exact_one_electron,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the collapsed one-electron recurrence without exact reference work.
 *
 * This wrapper is the production entry point for the component-ordered
 * one-electron separator kernel. It returns only the collapsed overlap,
 * collapsed one-electron matrix element, and work counters.
 */
OneElectronStarPairResult
evaluate_component_ordered_open_state_star_pair_one_electron_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

}  // namespace xmvb::vb::exact_separator
