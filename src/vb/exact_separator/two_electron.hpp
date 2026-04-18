#pragma once

#include <cstdint>
#include <vector>

#include "vb/exact_separator/component_data.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Exact determinant-space two-electron totals for one component-ordered star pair.
 *
 * `exact_overlap` is the exact raw-VB overlap reconstructed from the merged
 * global determinant terms on the bra and ket sides. `exact_two_electron` is
 * the exact two-electron Hamiltonian matrix element only, evaluated with the
 * same global determinant basis. The channel-resolved fields split that pure
 * two-electron value into same-spin alpha, same-spin beta, and opposite-spin
 * Coulomb parts. The counters describe the size of the merged determinant
 * expansions and the number of determinant pairs actually visited.
 */
struct ExactTwoElectronStarPairStats {
  double exact_overlap = 0.0;
  double exact_two_electron = 0.0;
  double exact_same_spin_alpha_two_electron = 0.0;
  double exact_same_spin_beta_two_electron = 0.0;
  double exact_opposite_spin_two_electron = 0.0;
  int left_global_term_count = 0;
  int right_global_term_count = 0;
  std::uint64_t evaluated_determinant_pair_count = 0;
  std::uint64_t skipped_zero_weight_pair_count = 0;
};

/**
 * @brief Exact-vs-collapsed validation totals for the one-leaf opposite-spin path.
 *
 * The current exact-separator implementation target is the opposite-spin
 * two-electron channel for a component-ordered star with exactly one root
 * component and one leaf component. `exact_*` are determinant-space reference
 * values reconstructed by the baseline exact routine. `collapsed_*` are the
 * corresponding quantities rebuilt from one-spin boundary-message aggregates
 * without an explicit determinant-pair Cartesian product. The counters
 * describe the amount of one-spin aggregate work and dense minor/cofactor
 * evaluations performed by the collapsed path. In the current boundary path,
 * `alpha_mask_state_count` and `beta_mask_state_count` count visited
 * one-spin boundary sectors.
 */
struct CollapsedOppositeSpinOneLeafStarPairStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double overlap_absolute_error = 0.0;
  double exact_opposite_spin_two_electron = 0.0;
  double collapsed_opposite_spin_two_electron = 0.0;
  double opposite_spin_absolute_error = 0.0;
  std::uint64_t processed_term_quadruple_count = 0;
  std::uint64_t alpha_mask_state_count = 0;
  std::uint64_t beta_mask_state_count = 0;
  std::uint64_t skipped_zero_alpha_mask_state_count = 0;
  std::uint64_t skipped_zero_beta_mask_state_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
};

/**
 * @brief Exact-vs-collapsed totals for the full one-leaf two-electron path.
 *
 * This extends the opposite-spin validation to all three pure two-electron
 * channels of one component-ordered star with exactly one root component and
 * one leaf component. The collapsed path reconstructs the exact one-spin
 * overlap, first-cofactor, and same-spin aggregates from one-spin boundary
 * messages, then evaluates the same-spin alpha, same-spin beta, and
 * opposite-spin contractions without an explicit determinant-pair Cartesian
 * product. `alpha_mask_state_count` and `beta_mask_state_count` count the
 * visited one-spin boundary sectors.
 */
struct CollapsedTwoElectronOneLeafStarPairStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double overlap_absolute_error = 0.0;
  double exact_same_spin_alpha_two_electron = 0.0;
  double collapsed_same_spin_alpha_two_electron = 0.0;
  double same_spin_alpha_absolute_error = 0.0;
  double exact_same_spin_beta_two_electron = 0.0;
  double collapsed_same_spin_beta_two_electron = 0.0;
  double same_spin_beta_absolute_error = 0.0;
  double exact_opposite_spin_two_electron = 0.0;
  double collapsed_opposite_spin_two_electron = 0.0;
  double opposite_spin_absolute_error = 0.0;
  double exact_two_electron = 0.0;
  double collapsed_two_electron = 0.0;
  double total_two_electron_absolute_error = 0.0;
  std::uint64_t processed_term_quadruple_count = 0;
  std::uint64_t alpha_mask_state_count = 0;
  std::uint64_t beta_mask_state_count = 0;
  std::uint64_t skipped_zero_alpha_mask_state_count = 0;
  std::uint64_t skipped_zero_beta_mask_state_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
};

/**
 * @brief Exact-vs-collapsed totals for a general component-ordered star pair.
 *
 * `ordered_components.front()` is the root component and the remaining entries
 * are leaf components in the star order used by the separator recurrence. The
 * collapsed path reconstructs the exact overlap, same-spin alpha channel,
 * same-spin beta channel, opposite-spin channel, and total pure two-electron
 * matrix element without an explicit determinant-pair Cartesian product.
 */
struct CollapsedTwoElectronStarPairStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double overlap_absolute_error = 0.0;
  double exact_same_spin_alpha_two_electron = 0.0;
  double collapsed_same_spin_alpha_two_electron = 0.0;
  double same_spin_alpha_absolute_error = 0.0;
  double exact_same_spin_beta_two_electron = 0.0;
  double collapsed_same_spin_beta_two_electron = 0.0;
  double same_spin_beta_absolute_error = 0.0;
  double exact_opposite_spin_two_electron = 0.0;
  double collapsed_opposite_spin_two_electron = 0.0;
  double opposite_spin_absolute_error = 0.0;
  double exact_two_electron = 0.0;
  double collapsed_two_electron = 0.0;
  double total_two_electron_absolute_error = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Production two-electron totals for one component-ordered star pair.
 *
 * This is the collapsed separator recurrence result without determinant-space
 * reference bookkeeping. It returns the overlap reconstructed by the same
 * deleted-minor payload, the three pure two-electron channels, the total
 * two-electron matrix element, and the associated work counters.
 */
struct TwoElectronStarPairResult {
  double overlap = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
  double two_electron = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Production one-leaf collapsed Hamiltonian totals built from one table pass.
 *
 * This is the fused one-leaf production path used by the rooted-tree driver.
 * It constructs the exact one-spin boundary aggregate tables once, then
 * reuses them to contract overlap, one-electron, same-spin, and opposite-spin
 * channels without rebuilding the same one-leaf state-pair aggregates twice.
 */
struct OneLeafHamiltonianStarPairResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
  double two_electron = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Evaluates the exact component-ordered two-electron matrix element.
 *
 * This routine is the clean determinant-space baseline for the exact-separator
 * two-electron work. It expands the component-ordered star pair to the merged
 * global determinant basis on each side, then contracts the exact determinant
 * pair two-electron matrix elements using the standard packed active-space
 * two-electron integrals.
 */
ExactTwoElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact one-leaf collapsed opposite-spin separator path.
 *
 * `ordered_components` must contain exactly two entries: the root component
 * followed by one leaf component. The routine rebuilds the exact opposite-spin
 * two-electron channel by summing signed one-spin first-cofactor matrices over
 * exact one-leaf boundary sectors, then contracting the final alpha and beta
 * cofactor sums against the packed active-space ERIs.
 */
CollapsedOppositeSpinOneLeafStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_opposite_spin_one_leaf(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact one-leaf collapsed two-electron separator path.
 *
 * `ordered_components` must contain exactly two entries: the root component
 * followed by one leaf component. The routine rebuilds the exact overlap and
 * all three two-electron channels from one-spin boundary-message aggregates,
 * without an explicit determinant-pair Cartesian product.
 */
CollapsedTwoElectronOneLeafStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the collapsed one-leaf two-electron recurrence only.
 *
 * This is the production entry point for the exact one-leaf separator kernel.
 * It skips determinant-space reference work and returns only the collapsed
 * overlap, pure two-electron channels, and work counters.
 */
TwoElectronStarPairResult
evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the fused one-leaf collapsed Hamiltonian recurrence.
 *
 * `ordered_components` must contain exactly two entries: the root component
 * followed by one leaf component. The routine builds the exact one-leaf
 * alpha/beta boundary aggregate tables once and contracts all Hamiltonian
 * channels from that shared state-pair basis.
 */
OneLeafHamiltonianStarPairResult
evaluate_component_ordered_open_state_star_pair_hamiltonian_one_leaf_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact collapsed multi-leaf star two-electron recurrence.
 *
 * This is the general component-ordered star driver:
 * - `ordered_components.front()` is the root component;
 * - the remaining entries are leaf components in star order.
 *
 * For one leaf, the implementation reuses the specialized one-leaf path. For
 * multiple leaves, it propagates exact one-spin deleted-minor sectors up to
 * second order through the frontier merge and reconstructs the full pure
 * two-electron Hamiltonian matrix element at the root closure.
 */
CollapsedTwoElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the collapsed star two-electron recurrence only.
 *
 * This is the production entry point for the component-ordered star
 * two-electron separator kernel. It avoids determinant-space reference work
 * and returns only the collapsed channels and work counters.
 */
TwoElectronStarPairResult
evaluate_component_ordered_open_state_star_pair_two_electron_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver);

}  // namespace xmvb::vb::exact_separator
