#pragma once

#include "vb/exact_separator/bundle.hpp"
#include "vb/exact_separator/leaf_boundary_message.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Builds the canonical dense bundle layout induced by one one-leaf message.
 *
 * The exact one-leaf boundary message already enumerates the same imbalance
 * families and selected-mask sectors needed by `BoundarySpinBundleLayout`.
 * This helper turns that public message metadata into the canonical dense
 * layout used by the bundle carrier.
 */
BoundarySpinBundleLayout make_one_leaf_boundary_spin_bundle_layout(
    const OneLeafBoundarySpinMessage& message);

/**
 * @brief Reorders one sparse one-leaf boundary message into a dense one-spin bundle.
 *
 * `degree0`, `degree1`, and `degree2` are filled directly from the exported
 * exact sector amplitudes carried by `OneLeafBoundarySpinMessage`.
 */
BoundarySpinBundle build_one_leaf_boundary_spin_bundle(
    const OneLeafBoundarySpinMessage& message);

}  // namespace xmvb::vb::exact_separator
