#pragma once

#include "vb/exact_separator/bundle.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Builds the canonical output layout for one binary one-spin merge.
 *
 * The merged block concatenates the left and right boundary label orders, so:
 *
 * - row count adds;
 * - column count adds;
 * - the base overlap-family imbalance adds.
 *
 * `min_family_delta` / `max_family_delta` let the caller choose whether to
 * keep the full exact family range of both inputs or a truncated range such as
 * `[-2, 2]`.
 */
BoundarySpinBundleLayout make_merged_boundary_spin_bundle_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right,
    int min_family_delta,
    int max_family_delta);

/**
 * @brief Convenience exact family range for one binary one-spin merge.
 *
 * This keeps every family that can arise from combining one family of `left`
 * and one family of `right`.
 */
BoundarySpinBundleLayout make_full_merged_boundary_spin_bundle_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right);

/**
 * @brief Exact binary merge of one-spin boundary bundles.
 *
 * This is the dense canonical bundle analogue of the old deleted-minor merge:
 *
 * - degree-0 times degree-0;
 * - degree-1 by the exact Leibniz rule;
 * - degree-2 from same-child degree-2 terms plus the cross term from two
 *   degree-1 insertions.
 */
BoundarySpinBundle merge_boundary_spin_bundles(
    const BoundarySpinBundle& left,
    const BoundarySpinBundle& right,
    const BoundarySpinBundleLayout& merged_layout);

/**
 * @brief Reverse of the exact binary one-spin bundle merge.
 *
 * `left_adjoint` / `right_adjoint` are accumulated in place.
 */
void reverse_merge_boundary_spin_bundles(
    const BoundarySpinBundle& left,
    const BoundarySpinBundle& right,
    const BoundarySpinBundle& merged_adjoint,
    BoundarySpinBundle* left_adjoint,
    BoundarySpinBundle* right_adjoint);

/**
 * @brief Exact binary merge of the full alpha/beta boundary bundle.
 *
 * The alpha and beta one-spin channels use the corresponding one-spin merge.
 * The mixed channel is merged from:
 *
 * 1. left mixed times right overlap;
 * 2. right mixed times left overlap;
 * 3. left alpha-degree1 with right beta-degree1;
 * 4. right alpha-degree1 with left beta-degree1.
 */
BoundaryBundle merge_boundary_bundles(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundarySpinBundleLayout& merged_alpha_layout,
    const BoundarySpinBundleLayout& merged_beta_layout);

/**
 * @brief Reverse of the exact binary full boundary bundle merge.
 *
 * This accumulates adjoints for:
 *
 * - alpha degree-0/1/2 channels;
 * - beta degree-0/1/2 channels;
 * - the mixed alpha/beta degree-1 second moment.
 */
void reverse_merge_boundary_bundles(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundaryBundle& merged_adjoint,
    BoundaryBundle* left_adjoint,
    BoundaryBundle* right_adjoint);

}  // namespace xmvb::vb::exact_separator
