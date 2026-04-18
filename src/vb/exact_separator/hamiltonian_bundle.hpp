#pragma once

#include <vector>

#include "vb/exact_separator/bundle.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief One projected one-spin deleted-sector coefficient in orbital labels.
 *
 * `row_orbitals` and `col_orbitals` are the deleted row/column orbital labels
 * in support-space indexing. Degree is `row_orbitals.size()` and must match
 * `col_orbitals.size()`.
 */
struct HamiltonianProjectedSpinSectorValue {
  std::vector<int> row_orbitals;
  std::vector<int> col_orbitals;
  double value = 0.0;
};

/**
 * @brief One projected one-spin Hamiltonian payload used by leaf scatter.
 *
 * This is the dense-carrier friendly equivalent of the current projected
 * one-spin payload values:
 *
 * - `overlap` is the degree-0 closed contribution;
 * - `same_spin_sectors` holds degree-1 and degree-2 channels for alpha/beta
 *   one-electron and same-spin contractions;
 * - `first_order_sectors` holds degree-1 channels for alpha/beta mixed
 *   opposite-spin outer products.
 */
struct HamiltonianProjectedSpinValues {
  double overlap = 0.0;
  std::vector<HamiltonianProjectedSpinSectorValue> same_spin_sectors;
  std::vector<HamiltonianProjectedSpinSectorValue> first_order_sectors;
};

/**
 * @brief Dense generic forward Hamiltonian carrier on boundary bundle basis.
 *
 * This mirrors the one-leaf packed path channel decomposition:
 *
 * - `overlap`: closed scalar channel,
 * - `alpha`: alpha-active one-spin bundle weighted by spectator beta overlap,
 * - `beta`: beta-active one-spin bundle weighted by spectator alpha overlap,
 * - `mixed`: alpha/beta degree-1 second moment for opposite-spin contraction.
 */
struct HamiltonianBoundaryBundle {
  double overlap = 0.0;
  BoundarySpinBundle alpha;
  BoundarySpinBundle beta;
  BoundaryMixedBundle mixed;
};

/**
 * @brief Readout of contracted Hamiltonian channels from dense bundle carrier.
 */
struct HamiltonianBoundaryBundleReadout {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
};

/**
 * @brief Allocates a zero-filled dense Hamiltonian boundary bundle.
 */
HamiltonianBoundaryBundle make_zero_hamiltonian_boundary_bundle(
    const BoundarySpinBundleLayout& alpha_layout,
    const BoundarySpinBundleLayout& beta_layout);

/**
 * @brief Checks dense channel storage compatibility against bundle layouts.
 */
bool has_compatible_layout(const HamiltonianBoundaryBundle& bundle);

/**
 * @brief Adds one dense Hamiltonian bundle into another with scalar weight.
 */
void add_scaled_hamiltonian_boundary_bundle(
    const HamiltonianBoundaryBundle& source,
    double scale,
    HamiltonianBoundaryBundle* destination);

/**
 * @brief Scatters one projected one-spin payload into one fixed flat sector.
 *
 * The sector coordinate comes from the outer boundary-mask traversal
 * (`flat_sector_index`). This routine only scatters orbital-deleted
 * coefficients into degree-0/1/2 dense basis entries of `destination`.
 */
void scatter_projected_spin_values_to_flat_sector(
    const HamiltonianProjectedSpinValues& projected,
    int flat_sector_index,
    double scale,
    BoundarySpinBundle* destination);

/**
 * @brief Accumulates projected alpha/beta degree-1 outer products to mixed channel.
 *
 * Both projected inputs are interpreted at fixed outer sector coordinates
 * (`alpha_flat_sector_index`, `beta_flat_sector_index`).
 */
void accumulate_projected_mixed_first_order_outer_product(
    const HamiltonianProjectedSpinValues& alpha_projected,
    int alpha_flat_sector_index,
    const BoundarySpinBundleLayout& alpha_layout,
    const HamiltonianProjectedSpinValues& beta_projected,
    int beta_flat_sector_index,
    const BoundarySpinBundleLayout& beta_layout,
    double scale,
    BoundaryMixedBundle* destination);

/**
 * @brief Adds one weighted projected alpha/beta spin pair to Hamiltonian channels.
 *
 * This is the projected-value analogue of the one-leaf packed accumulation:
 *
 * - `*overlap` gets `scale * alpha_overlap * beta_overlap`,
 * - `alpha` gets `scale * beta_overlap * alpha_projected`,
 * - `beta` gets `scale * alpha_overlap * beta_projected`,
 * - `mixed` gets `scale * (alpha_degree1 outer beta_degree1)`.
 *
 * The destination channel pointers must all be non-null and already allocated
 * with layouts compatible with `alpha_layout` / `beta_layout`.
 */
void accumulate_projected_spin_pair_into_hamiltonian_channels(
    const HamiltonianProjectedSpinValues& alpha_projected,
    int alpha_flat_sector_index,
    const BoundarySpinBundleLayout& alpha_layout,
    const HamiltonianProjectedSpinValues& beta_projected,
    int beta_flat_sector_index,
    const BoundarySpinBundleLayout& beta_layout,
    double scale,
    double* overlap,
    BoundarySpinBundle* alpha,
    BoundarySpinBundle* beta,
    BoundaryMixedBundle* mixed);

/**
 * @brief Adds one weighted one-spin pair directly to Hamiltonian channels.
 *
 * The destination channel pointers must all be non-null and already allocated
 * with layouts compatible with `alpha_bundle` / `beta_bundle`.
 */
void accumulate_weighted_spin_pair_into_hamiltonian_channels(
    const BoundarySpinBundle& alpha_bundle,
    const BoundarySpinBundle& beta_bundle,
    double coefficient,
    double* overlap,
    BoundarySpinBundle* alpha,
    BoundarySpinBundle* beta,
    BoundaryMixedBundle* mixed);

/**
 * @brief Adds one weighted one-spin pair directly to dense Hamiltonian channels.
 *
 * This is the dense analogue of the one-leaf packed accumulation:
 *
 * - overlap gets `coefficient * alpha_overlap * beta_overlap`,
 * - alpha channel gets `coefficient * beta_overlap * alpha_bundle`,
 * - beta channel gets `coefficient * alpha_overlap * beta_bundle`,
 * - mixed gets `coefficient * (beta_degree1 outer alpha_degree1)`.
 */
void accumulate_weighted_spin_pair_into_hamiltonian_bundle(
    const BoundarySpinBundle& alpha_bundle,
    const BoundarySpinBundle& beta_bundle,
    double coefficient,
    HamiltonianBoundaryBundle* destination);

/**
 * @brief Contracts scalar Hamiltonian channels from dense boundary bundle carrier.
 */
HamiltonianBoundaryBundleReadout contract_hamiltonian_boundary_bundle_channels(
    const HamiltonianBoundaryBundle& bundle,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size);

}  // namespace xmvb::vb::exact_separator
