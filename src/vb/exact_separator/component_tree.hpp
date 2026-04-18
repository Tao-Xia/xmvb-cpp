#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "vb/exact_separator/component_data.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Rooted component tree used by the non-star exact-separator driver.
 *
 * `components[node]` stores the local raw-VB component data for one tree node.
 * `children[node]` lists the directed children of that node in the local tree
 * order used by the separator recurrence. The current implementation assumes
 * that `root_index` is a valid node and that every node is reachable from the
 * root exactly once.
 */
struct ComponentTree {
  int root_index = 0;
  std::vector<ComponentData> components;
  std::vector<std::vector<int>> children;
};

/**
 * @brief Exact-vs-collapsed Hamiltonian totals for a rooted non-star component tree.
 *
 * `exact_*` are determinant-space reference values rebuilt from the preorder
 * component list of the tree. `collapsed_*` are produced by the current
 * rooted-tree separator driver. The work counters describe the exact subtree
 * messages materialized at child edges and the root-level DP transitions.
 */
struct CollapsedHamiltonianComponentTreeStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double overlap_absolute_error = 0.0;
  double exact_one_electron = 0.0;
  double collapsed_one_electron = 0.0;
  double one_electron_absolute_error = 0.0;
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
  double exact_total_electronic_hamiltonian = 0.0;
  double collapsed_total_electronic_hamiltonian = 0.0;
  double total_electronic_hamiltonian_absolute_error = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief One rooted component-tree Hamiltonian value without reference bookkeeping.
 *
 * This is the production result shared by the exact determinant-space path and
 * the collapsed subtree-message recurrence. It carries the total overlap, the
 * one-electron term, the spin-resolved two-electron channels, and the total
 * electronic Hamiltonian. The work counters are populated by the collapsed
 * recurrence and left at zero by the exact determinant-space baseline.
 */
struct ComponentTreeHamiltonianResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
  double two_electron = 0.0;
  double total_electronic_hamiltonian = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Exact rooted-tree active-space gradient blocks for one structure pair.
 *
 * `hamiltonian` stores the same exact rooted-tree scalar channels returned by
 * `ComponentTreeHamiltonianResult`. The gradient buffers use the repository
 * storage conventions:
 *
 * - `active_orbital_overlap_gradient` is column-major with respect to the
 *   support-space overlap matrix storage `S[col * support_size + row]`;
 * - `active_one_electron_gradient` is column-major with respect to the
 *   support-space one-electron matrix storage `h[col * support_size + row]`;
 * - `packed_active_two_electron_gradient` matches the packed ERI indexing used
 *   by `TwoElectronIndexer`.
 *
 * The convenience `evaluate_rooted_component_tree_hamiltonian_gradient_*`
 * wrappers return the unweighted derivative of the total electronic
 * Hamiltonian matrix element, i.e. `d(H^(1) + H^(2))`. The weighted
 * `evaluate_rooted_component_tree_active_space_gradient_*` routines instead
 * return the gradient of
 *
 * `hamiltonian_weight * (H^(1) + H^(2)) + overlap_weight * S`.
 */
struct ComponentTreeHamiltonianGradientResult {
  ComponentTreeHamiltonianResult hamiltonian;
  std::vector<double> active_orbital_overlap_gradient;
  std::vector<double> active_one_electron_gradient;
  std::vector<double> packed_active_two_electron_gradient;
};

/**
 * @brief Exact rooted-tree overlap value with overlap-only work counters.
 *
 * `overlap` is the scalar overlap matrix element. The counters describe either
 * the determinant-space exact reference work or the exact boundary-only
 * overlap recurrence, depending on which routine populated the result.
 */
struct ComponentTreeOverlapResult {
  double overlap = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Exact rooted-tree overlap and one-electron value with light counters.
 *
 * This is the overlap plus total one-electron analogue of
 * `ComponentTreeOverlapResult`. It is used by the new boundary-only scalar
 * message recurrence before same-spin and opposite-spin are migrated.
 */
struct ComponentTreeOneElectronResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Exact rooted-tree overlap, one-electron, and opposite-spin value.
 *
 * This is the degree-1 boundary-message analogue of
 * `ComponentTreeOneElectronResult`. It carries exactly the channels that are
 * reconstructible from overlap plus first-cofactor sectors:
 * `S`, total `H^(1)`, and opposite-spin `H^(2)_{alpha,beta}`.
 */
struct ComponentTreeOppositeSpinResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Debug aggregate of the degree-1 rooted-tree boundary recurrence.
 *
 * This helper exposes the exact support-space first-cofactor matrices rebuilt
 * by the degree-1 boundary-message path:
 * - `alpha_first_cofactor` is the total alpha first-order deleted-minor sum
 *   already weighted by the spectator beta overlap and component coefficients.
 * - `beta_first_cofactor` is the analogous beta aggregate.
 *
 * The scalar channels are provided alongside the matrices so that tools can
 * cross-check both the contracted totals and the underlying tensor payload.
 */
struct ComponentTreeOppositeSpinDebugResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
  Eigen::MatrixXd alpha_first_cofactor;
  Eigen::MatrixXd beta_first_cofactor;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

/**
 * @brief Exact rooted-tree overlap, one-electron, and same-spin values.
 *
 * This is the degree-2 boundary-message analogue of
 * `ComponentTreeOneElectronResult`. It carries exactly the channels
 * reconstructible from overlap, first-order cofactors, and same-spin
 * second-order cofactors, but excludes opposite-spin.
 */
struct ComponentTreeSameSpinResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

using CollapsedTwoElectronComponentTreeStats = CollapsedHamiltonianComponentTreeStats;

/**
 * @brief Evaluates the exact rooted component-tree electronic Hamiltonian.
 *
 * The collapsed path uses one recursive subtree-message recurrence and one
 * joint deleted-minor payload to reconstruct all of
 * `S`, `H^(1)`, `H^(2)`, and `H_total = H^(1) + H^(2)` at the root closure.
 */
CollapsedHamiltonianComponentTreeStats
evaluate_rooted_component_tree_hamiltonian(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the determinant-space exact rooted-tree Hamiltonian only.
 *
 * This routine rebuilds the preorder component list of the rooted tree,
 * expands the corresponding exact determinant terms, and contracts the exact
 * determinant-pair Hamiltonian channels. It is the clean baseline used for
 * correctness checks and performance comparisons against the collapsed path.
 */
ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the collapsed rooted-tree Hamiltonian recurrence only.
 *
 * This is the production entry point for the rooted component-tree separator
 * kernel. It skips determinant-space reference work and returns only the
 * collapsed overlap, one-electron term, two-electron channels, and work
 * counters.
 */
ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact boundary-only Hamiltonian path without determinant fallback.
 *
 * This entry point is reserved for boundary-message implementations only. It
 * may dispatch to the best exact separator kernel currently available for the
 * given tree topology, and otherwise falls back to the generic recursive
 * typed boundary bundle. It does not call the determinant-pair reference
 * Hamiltonian builder.
 */
ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_bundle_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates one weighted rooted-tree active-space gradient exactly.
 *
 * This is the determinant-space exact reference for the full separator
 * active-space local kernel. It returns the exact derivatives of
 *
 * `hamiltonian_weight * (H^(1) + H^(2)) + overlap_weight * S`
 *
 * with respect to `S_act`, `h_act`, and the packed active-space ERIs.
 */
ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_active_space_gradient_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    double hamiltonian_weight,
    double overlap_weight);

/**
 * @brief Evaluates one weighted rooted-tree active-space gradient from messages.
 *
 * This is the exact-separator production local gradient kernel for rooted
 * trees. The `h_act` and ERI derivatives are accumulated from the current
 * typed Hamiltonian payload. The overlap-gradient block is currently kept
 * exact by the rooted-tree deleted-minor reverse together with temporary
 * exact subtree/root closures at the unresolved recursive root boundary.
 * The long-term target is to replace those closures by the canonical boundary
 * bundle algebra described in `gradient.md`.
 */
ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_active_space_gradient_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    double hamiltonian_weight,
    double overlap_weight);

/**
 * @brief Evaluates the exact rooted-tree active-space gradient of `H^(1) + H^(2)`.
 *
 * This wrapper keeps the earlier G1 entry point, but now also populates the
 * overlap-gradient block corresponding to `d(H^(1) + H^(2)) / dS_act`.
 */
ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_hamiltonian_gradient_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact rooted-tree active-space gradient of `H^(1) + H^(2)`.
 *
 * This wrapper keeps the earlier G1 entry point, but now also populates the
 * overlap-gradient block corresponding to `d(H^(1) + H^(2)) / dS_act`.
 */
ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_hamiltonian_gradient_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact rooted component-tree two-electron matrix element.
 *
 * This is the first non-star rooted-tree driver. The root-level recurrence is
 * collapsed over child subtree messages, while the current child message
 * builder materializes exact subtree frontier contributions. The interface is
 * intentionally tree-based so that later fully recursive subtree-message
 * implementations can reuse the same API.
 */
CollapsedTwoElectronComponentTreeStats
evaluate_rooted_component_tree_two_electron(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the determinant-space exact rooted-tree overlap only.
 *
 * This is the clean exact reference for the new overlap-only boundary-message
 * prototype. It expands the global determinant terms, contracts only the
 * overlap channel, and avoids the extra Hamiltonian work of the full exact
 * reference driver.
 */
ComponentTreeOverlapResult evaluate_rooted_component_tree_overlap_exact(
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact boundary-only rooted-tree overlap recurrence.
 *
 * This is the first true `2^n -> 2^m` rooted-tree implementation in the
 * component-tree code path. Each subtree is summarized only by its exact
 * boundary sector table on the parent interface, with the subtree interior
 * fully integrated out.
 */
ComponentTreeOverlapResult evaluate_rooted_component_tree_overlap_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the determinant-space exact rooted-tree overlap plus one-electron only.
 */
ComponentTreeOneElectronResult evaluate_rooted_component_tree_one_electron_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the exact boundary-only rooted-tree overlap plus one-electron recurrence.
 */
ComponentTreeOneElectronResult
evaluate_rooted_component_tree_one_electron_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the determinant-space exact rooted-tree overlap, one-electron, and opposite-spin only.
 */
ComponentTreeOppositeSpinResult evaluate_rooted_component_tree_opposite_spin_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the opposite-spin projection of the exact rooted-tree boundary bundle.
 *
 * The production implementation now performs the exact full Hamiltonian
 * boundary closure and returns only the overlap, one-electron, and
 * opposite-spin channels from that exact bundle.
 */
ComponentTreeOppositeSpinResult
evaluate_rooted_component_tree_opposite_spin_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Debug helper exposing the exact rooted-tree first-cofactor aggregates.
 *
 * This uses the same exact full-boundary closure as
 * `evaluate_rooted_component_tree_opposite_spin_boundary_collapsed(...)`, but
 * also returns the fully aggregated alpha/beta first-cofactor matrices before
 * the final one-electron / opposite-spin contractions.
 */
ComponentTreeOppositeSpinDebugResult
evaluate_rooted_component_tree_opposite_spin_boundary_debug(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the determinant-space exact rooted-tree overlap, one-electron, and same-spin only.
 */
ComponentTreeSameSpinResult evaluate_rooted_component_tree_same_spin_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

/**
 * @brief Evaluates the same-spin projection of the exact rooted-tree boundary bundle.
 *
 * The production implementation now performs the exact full Hamiltonian
 * boundary closure and returns only the overlap, one-electron, and same-spin
 * channels from that exact bundle.
 */
ComponentTreeSameSpinResult
evaluate_rooted_component_tree_same_spin_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver);

}  // namespace xmvb::vb::exact_separator
