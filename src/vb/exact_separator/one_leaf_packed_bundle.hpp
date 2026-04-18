#pragma once

#include <cstdint>
#include <vector>

#include "vb/exact_separator/bundle.hpp"
#include "vb/exact_separator/leaf_boundary_message.hpp"
#include "vb/exact_separator/leaf_coefficient_operator.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Packed one-leaf Hamiltonian bundle in dense boundary storage.
 *
 * This is not the raw factorized determinant bundle of one single state pair.
 * Instead it is the fully accumulated one-leaf Hamiltonian message in dense
 * boundary-bundle coordinates:
 *
 * - `overlap` is the final closed overlap scalar;
 * - `alpha` stores the exact alpha degree-0/1/2 bundle already weighted by
 *   the matching beta overlap channel;
 * - `beta` stores the symmetric beta-weighted channel;
 * - `mixed` stores the exact accumulated alpha/beta degree-1 second moment.
 *
 * These are exactly the same forward Hamiltonian channels previously carried
 * by `HamiltonianBoundaryPayload`, but expressed on the canonical dense bundle
 * basis instead of sparse deleted-label maps.
 */
struct PackedOneLeafBundle {
  double overlap = 0.0;
  BoundarySpinBundle alpha;
  BoundarySpinBundle beta;
  BoundaryMixedBundle mixed;
  int support_size = 0;
  std::uint64_t alpha_state_pair_count = 0;
  std::uint64_t beta_state_pair_count = 0;
  std::uint64_t total_sector_count = 0;
};

struct PackedOneLeafBundleContractionResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
};

PackedOneLeafBundle build_one_leaf_packed_bundle(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

PackedOneLeafBundleContractionResult contract_one_leaf_packed_bundle(
    const PackedOneLeafBundle& packed,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals);

}  // namespace xmvb::vb::exact_separator
