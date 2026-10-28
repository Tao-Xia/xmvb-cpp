#pragma once

#include <cstdint>

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
    SelectedStatePairGraph(const SelectedStateDeterminantMatrices& selected_states,
                           PrimarySpin primary_spin);

    SelectedStatePairGraph(const SelectedStateDeterminantMatrices& selected_states,
                           const SelectedStateDeterminantMatrices& directional_selected_states,
                           PrimarySpin primary_spin);

    void accumulate_partner_projection(
        Eigen::Index primary_left,
        Eigen::Index primary_right,
        const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
        std::int64_t n_unique_partner,
        std::vector<double>* partner_image,
        std::vector<unsigned char>* touched_flags,
        std::vector<int>* touched_channels) const;

    void accumulate_partner_projected_values(
        Eigen::Index primary_left,
        Eigen::Index primary_right,
        const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
        std::int64_t n_unique_partner,
        const std::vector<int>& target_channels,
        std::vector<double>* target_values) const;

    void accumulate_partner_projected_values(
        Eigen::Index primary_left,
        Eigen::Index primary_right,
        const std::vector<DirectionalOppositeSpinPairData>& partner_pair_data,
        std::int64_t n_unique_partner,
        const std::vector<int>& target_channels,
        std::vector<double>* target_values) const;

    void accumulate_partner_projection(
        Eigen::Index primary_left,
        Eigen::Index primary_right,
        const std::vector<DirectionalOppositeSpinPairData>& partner_pair_data,
        std::int64_t n_unique_partner,
        std::vector<double>* partner_image,
        std::vector<unsigned char>* touched_flags,
        std::vector<int>* touched_channels) const;

  private:
    struct Term {
        const SparseLocalCoefficientMatrix* left_coefficients = nullptr;
        const SparseLocalCoefficientMatrix* right_coefficients = nullptr;
        const std::vector<Eigen::Index>* left_partner_support = nullptr;
        const std::vector<Eigen::Index>* right_partner_support = nullptr;
        std::vector<Eigen::Index> left_primary_to_local;
        std::vector<Eigen::Index> right_primary_to_local;
        double weight = 0.0;
    };

    void add_term(const SelectedStateDeterminantCoefficients& left,
                  const SelectedStateDeterminantCoefficients& right,
                  double weight,
                  PrimarySpin primary_spin,
                  Eigen::Index n_primary);

    template <typename ProjectionAt>
    void accumulate_partner_projection_impl(Eigen::Index primary_left,
                                            Eigen::Index primary_right,
                                            ProjectionAt&& projection_at,
                                            std::vector<double>* partner_image,
                                            std::vector<unsigned char>* touched_flags,
                                            std::vector<int>* touched_channels) const;

    template <typename AccumulatePair>
    void for_each_partner_pair(Eigen::Index primary_left,
                               Eigen::Index primary_right,
                               AccumulatePair&& accumulate_pair) const;

    std::vector<Term> terms_;
};

} // namespace xmvb::vb::detail
