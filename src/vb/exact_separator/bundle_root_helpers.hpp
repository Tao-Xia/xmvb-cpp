#pragma once

#include "vb/exact_separator/bundle.hpp"
#include "vb/exact_separator/hamiltonian_bundle.hpp"

namespace xmvb::vb::exact_separator {

struct SpinInterfaceLayout {
  std::vector<int> frontier_interface_rows;
  std::vector<int> frontier_interface_cols;
  std::vector<int> local_remainder_rows;
  std::vector<int> local_remainder_cols;
  int frontier_rows = 0;
  int frontier_cols = 0;
};

/**
 * @brief Layout descriptor for root interface reordering.
 *
 * Records the interface/frontier partitioning for alpha/beta channels so the
 * bundle reorder helper can compute parity adjustments exactly instead of
 * leaving an identity placeholder.
 */
struct RootInterfaceLayout {
  SpinInterfaceLayout alpha;
  SpinInterfaceLayout beta;
};

/**
 * @brief Bundle-level Hamiltonian contract result from the closed root block.
 */
struct BundleRootReadout {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
};

/**
 * @brief Applies the canonical reorder for propagated root closure.
 *
 * Converts the natural bundle order into the parent-oriented order by
 * applying the parity adjustments associated with moving the interface block
 * across the descendant frontier body. The helper currently only uses layout
 * metadata, so integrations can start wiring data into `RootInterfaceLayout`.
 */
BoundaryBundle reorder_root_bundle(
    const BoundaryBundle& natural_payload,
    const RootInterfaceLayout& layout);

/**
 * @brief Applies the canonical reorder to dense Hamiltonian channels.
 *
 * This is the Hamiltonian-channel analogue of `reorder_root_bundle(...)`:
 *
 * - `overlap` is left unchanged;
 * - `alpha.degree1/degree2`, `beta.degree1/degree2`, and `mixed.values`
 *   receive the exact interface/frontier parity adjustment.
 */
HamiltonianBoundaryBundle reorder_root_hamiltonian_bundle(
    const HamiltonianBoundaryBundle& natural_bundle,
    const RootInterfaceLayout& layout);

/**
 * @brief Contracts bundle-level Hamiltonian scalars.
 *
 * Accepts a fully closed root bundle (after reorder) and returns the four
 * Hamiltonian channels via the helpers in `bundle_channels.cpp`.
 */
BundleRootReadout contract_root_bundle_channels(
    const BoundaryBundle& bundle,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size);

}  // namespace xmvb::vb::exact_separator
