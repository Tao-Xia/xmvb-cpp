#pragma once

#include <vector>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/local_contractions.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb::detail {

struct DirectionalOppositeSpinPairData;

enum class PrimarySpin {
  Alpha,
  Beta,
};

/**
 * @brief Sparse coefficient graph for accepted or directional selected states.
 *
 * Rows are indexed by primary-spin unique strings. Traversing one ordered row
 * pair contracts either `C(left,:) * C(right,:)` or the directional sum
 * `delta_C(left,:) * C(right,:) + C(left,:) * delta_C(right,:)` directly into
 * the sparse first-order cofactor channels of the partner-spin pairs.
 */
class SelectedStatePairGraph {
 public:
  SelectedStatePairGraph(
      const SelectedStateDeterminantMatrices& selected_states,
      PrimarySpin primary_spin);

  SelectedStatePairGraph(
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

  void accumulate_partner_projected_values(
      int primary_left,
      int primary_right,
      const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
      int n_unique_partner,
      const std::vector<int>& target_channels,
      std::vector<double>* target_values) const;

  void accumulate_partner_projected_values(
      int primary_left,
      int primary_right,
      const std::vector<DirectionalOppositeSpinPairData>& partner_pair_data,
      int n_unique_partner,
      const std::vector<int>& target_channels,
      std::vector<double>* target_values) const;

  void accumulate_partner_projection(
      int primary_left,
      int primary_right,
      const std::vector<DirectionalOppositeSpinPairData>& partner_pair_data,
      int n_unique_partner,
      std::vector<double>* partner_image,
      std::vector<unsigned char>* touched_flags,
      std::vector<int>* touched_channels) const;

 private:
  struct Term {
    const SparseLocalCoefficientMatrix* left_coefficients = nullptr;
    const SparseLocalCoefficientMatrix* right_coefficients = nullptr;
    const std::vector<int>* left_partner_support = nullptr;
    const std::vector<int>* right_partner_support = nullptr;
    std::vector<int> left_primary_to_local;
    std::vector<int> right_primary_to_local;
    double weight = 0.0;
  };

  void add_term(
      const SelectedStateDeterminantCoefficients& left,
      const SelectedStateDeterminantCoefficients& right,
      double weight,
      PrimarySpin primary_spin,
      int n_primary);

  template <typename ProjectionAt>
  void accumulate_partner_projection_impl(
      int primary_left,
      int primary_right,
      ProjectionAt&& projection_at,
      std::vector<double>* partner_image,
      std::vector<unsigned char>* touched_flags,
      std::vector<int>* touched_channels) const;

  template <typename AccumulatePair>
  void for_each_partner_pair(
      int primary_left,
      int primary_right,
      AccumulatePair&& accumulate_pair) const;

  std::vector<Term> terms_;
};

}  // namespace xmvb::vb::detail
