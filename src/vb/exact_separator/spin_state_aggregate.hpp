#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/exact_separator/leaf_coefficient_operator.hpp"
#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Exact full-state one-spin aggregate in support-orbital coordinates.
 *
 * This is the direct Phase-3 replacement for the old one-leaf mask-summed
 * state aggregate. The quantities correspond to one unique full one-spin
 * occupied-state pair `(left_occ, right_occ)`:
 *
 * - `overlap` is the exact overlap determinant;
 * - `left_occ` / `right_occ` are the occupied support-orbital labels of this
 *   full state;
 * - `local_first_cofactor` is the exact first-cofactor block in the natural
 *   occupied ordering `(rows = right_occ, cols = left_occ)`;
 * - `first_cofactor` is the same object scattered into support-orbital
 *   coordinates with rows = ket/right orbitals and cols = bra/left orbitals.
 *   Some fast one-leaf paths intentionally leave this support-space matrix
 *   empty and contract the exact local representation directly;
 * - `one_electron` is the exact one-electron scalar
 *   `sum_{ij} h_{ij} C_{ij}` for the supplied support-space one-electron
 *   matrix;
 * - `same_spin_two_electron` is the exact pure same-spin two-electron scalar
 *   of that full state.
 */
struct DirectSpinStateAggregate {
  double overlap = 0.0;
  std::vector<int> left_occ;
  std::vector<int> right_occ;
  Eigen::MatrixXd local_first_cofactor;
  Eigen::MatrixXd first_cofactor;
  double one_electron = 0.0;
  double same_spin_two_electron = 0.0;
};

/**
 * @brief Builds the exact one-spin aggregate directly from one full state.
 *
 * This routine avoids the old separator-mask reconstruction layer entirely.
 * It factorizes the full overlap block once, recovers the exact first
 * cofactor directly, contracts the exact one-electron scalar against the
 * supplied one-electron support matrix, and optionally evaluates the exact
 * same-spin scalar from the same full-state determinant pair.
 */
DirectSpinStateAggregate build_direct_spin_state_aggregate(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    const DeterminantOverlapResolver& overlap_resolver,
    const DeterminantHamiltonianResolver& hamiltonian_resolver,
    std::uint64_t* subdeterminant_evaluations);

/**
 * @brief Dense one-leaf table of direct full-state aggregates.
 *
 * The aggregates are stored in Cartesian-product order
 * `index = root_state_index * leaf_state_count + leaf_state_index`.
 */
struct IndexedOneLeafDirectSpinAggregateTable {
  int root_state_count = 0;
  int leaf_state_count = 0;
  std::vector<DirectSpinStateAggregate> aggregates;
  std::uint64_t total_state_count = 0;

  int index(int root_state_index, int leaf_state_index) const;
};

/**
 * @brief Builds all one-leaf direct aggregates with root/leaf block reuse.
 *
 * The builder pre-factorizes the reusable root and leaf overlap blocks once
 * per unique component state. Regular full-state pairs use exact
 * Schur-complement reconstruction; singular pairs fall back to the exact full
 * local direct builder without invoking the determinant-pair production path.
 * When `materialize_support_first_cofactor == false`, the returned aggregates
 * still contain exact local occupied-space first cofactors but skip the extra
 * support-space scattering step used by older generic contractions.
 */
IndexedOneLeafDirectSpinAggregateTable
build_indexed_one_leaf_direct_spin_aggregate_table(
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool enable_reused_block_family,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

/**
 * @brief Builds alpha and beta one-leaf direct tables with a shared full-state cache.
 *
 * The alpha and beta one-spin channels sometimes generate the same final
 * occupied-orbital pair `(left_occ, right_occ)`. This paired builder reuses the
 * already materialized exact direct aggregate across the two channel tables
 * instead of rebuilding the same determinant/cofactor object twice.
 */
struct PairedOneLeafDirectSpinAggregateTables {
  IndexedOneLeafDirectSpinAggregateTable alpha_table;
  IndexedOneLeafDirectSpinAggregateTable beta_table;
};

PairedOneLeafDirectSpinAggregateTables
build_paired_one_leaf_direct_spin_aggregate_tables(
    const std::vector<SpinPairStateKey>& root_alpha_states,
    const std::vector<SpinPairStateKey>& leaf_alpha_states,
    const std::vector<SpinPairStateKey>& root_beta_states,
    const std::vector<SpinPairStateKey>& leaf_beta_states,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool enable_reused_block_family,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

/**
 * @brief Fused one-leaf contraction totals read directly from direct aggregate tables.
 *
 * The fields are the exact channel totals reconstructed by a single pass over
 * the compressed root/leaf coefficient operators. When the underlying tables
 * were built without one-electron or same-spin payloads, the corresponding
 * totals remain zero. When `packed_active_two_electron_integrals == nullptr`,
 * `opposite_spin` is left at zero.
 */
struct OneLeafDirectAggregateContractionResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
};

/**
 * @brief Contracts all available one-leaf channels in one direct-table pass.
 */
OneLeafDirectAggregateContractionResult contract_one_leaf_direct_aggregate_channels(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const IndexedOneLeafDirectSpinAggregateTable& alpha_table,
    const IndexedOneLeafDirectSpinAggregateTable& beta_table,
    const std::vector<double>* packed_active_two_electron_integrals);

/**
 * @brief Applies the opposite-spin ERI kernel to one first-cofactor matrix.
 *
 * Both the input first cofactor and the returned potential matrix live in
 * support-orbital coordinates with rows = ket/right orbitals and
 * cols = bra/left orbitals.
 */
Eigen::MatrixXd apply_opposite_spin_kernel_to_first_cofactor(
    const Eigen::MatrixXd& beta_first_cofactor,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals);

/**
 * @brief Contracts one opposite-spin pair from exact support-space matrices.
 */
double contract_direct_opposite_spin_channel(
    const Eigen::MatrixXd& alpha_first_cofactor,
    const Eigen::MatrixXd& beta_potential_first_cofactor);

}  // namespace xmvb::vb::exact_separator
