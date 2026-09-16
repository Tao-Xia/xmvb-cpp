#pragma once

#include <vector>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/local_contractions.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb::detail {

enum class PrimarySpin {
  Alpha,
  Beta,
};

/**
 * @brief Sparse coefficient graph for one directional selected-state response.
 *
 * Rows are indexed by primary-spin unique strings. Traversing one ordered row
 * pair contracts the two coefficient products
 * `delta_C(left,:) * C(right,:)` and `C(left,:) * delta_C(right,:)` directly
 * into the sparse first-order cofactor channels of the partner-spin pairs.
 */
class DirectionalPairGraph {
 public:
  DirectionalPairGraph(
      const SelectedStateDeterminantMatrices& selected_states,
      const SelectedStateDeterminantMatrices& directional_selected_states,
      PrimarySpin primary_spin);

  void accumulate_partner_projection(
      int primary_left,
      int primary_right,
      const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
      int n_unique_partner,
      std::vector<double>* partner_image,
      std::vector<unsigned char>* touched_flags,
      std::vector<int>* touched_channels) const;

 private:
  struct State {
    const SparseLocalCoefficientMatrix* accepted_coefficients = nullptr;
    const SparseLocalCoefficientMatrix* directional_coefficients = nullptr;
    const std::vector<int>* accepted_partner_support = nullptr;
    const std::vector<int>* directional_partner_support = nullptr;
    std::vector<int> accepted_primary_to_local;
    std::vector<int> directional_primary_to_local;
    double weight = 0.0;
  };

  std::vector<State> states_;
};

}  // namespace xmvb::vb::detail
