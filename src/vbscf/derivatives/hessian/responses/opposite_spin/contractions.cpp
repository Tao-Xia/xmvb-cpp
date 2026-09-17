#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/packed_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

namespace xmvb::vb {

using detail::DirectionalOppositeSpinPairData;

namespace detail {
namespace {

constexpr double kContributionTolerance = 1.0e-15;

const OppositeSpinPackedPairProjection& packed_projection(
    const SpinDeterminantPairEvaluation& pair) {
  return pair.opposite_spin_pair_cache.first_order_cofactor_projection;
}

const OppositeSpinPackedPairProjection& packed_projection(
    const DirectionalOppositeSpinPairData& pair) {
  return pair.delta_first_order_cofactor_projection;
}

template <typename PrimaryPairData, typename PartnerPairData>
void accumulate_ordered_packed_gradient_from_pair_graph(
    const std::vector<PrimaryPairData>& primary_pairs,
    const std::vector<PartnerPairData>& partner_pairs,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStatePairGraph& pair_graph,
    int n_packed_active_pairs,
    Eigen::MatrixXd* ordered_gradient) {
  const int n_threads = std::min(
      xmvb::effective_openmp_thread_count(),
      selected_states.n_unique_alpha);
  std::vector<Eigen::MatrixXd> partial_gradients;
  partial_gradients.reserve(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    partial_gradients.emplace_back(Eigen::MatrixXd::Zero(
        n_packed_active_pairs,
        n_packed_active_pairs));
  }
#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
#ifdef _OPENMP
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    std::vector<double> beta_image(n_packed_active_pairs, 0.0);
    std::vector<int> touched_beta_channels;
    touched_beta_channels.reserve(n_packed_active_pairs);
    std::vector<unsigned char> beta_channel_touched(
        n_packed_active_pairs,
        0u);
#pragma omp for schedule(static)
    for (int alpha_right = 0;
         alpha_right < selected_states.n_unique_alpha;
         ++alpha_right) {
      for (int alpha_left = 0;
           alpha_left < selected_states.n_unique_alpha;
           ++alpha_left) {
      const auto& alpha_projection = packed_projection(
          primary_pairs[ordered_spin_pair_storage_index(
              alpha_left,
              alpha_right,
              selected_states.n_unique_alpha)]);
      if (alpha_projection.packed_pair_indices.empty()) {
        continue;
      }

      pair_graph.accumulate_partner_projection(
          alpha_left,
          alpha_right,
          partner_pairs,
          selected_states.n_unique_beta,
          &beta_image,
          &beta_channel_touched,
          &touched_beta_channels);

      for (std::size_t alpha_entry = 0;
           alpha_entry < alpha_projection.packed_pair_indices.size();
           ++alpha_entry) {
        const int alpha_channel =
            alpha_projection.packed_pair_indices[alpha_entry];
        const double alpha_value =
            alpha_projection.packed_pair_values[alpha_entry];
        for (const int beta_channel : touched_beta_channels) {
          partial_gradients[thread](alpha_channel, beta_channel) +=
              alpha_value * beta_image[beta_channel];
        }
      }
      for (const int beta_channel : touched_beta_channels) {
        beta_image[beta_channel] = 0.0;
        beta_channel_touched[beta_channel] = 0u;
      }
      touched_beta_channels.clear();
      }
    }
  }
  for (const Eigen::MatrixXd& partial : partial_gradients) {
    *ordered_gradient += partial;
  }
}

void pack_ordered_gradient(
    const Eigen::MatrixXd& ordered_gradient,
    int n_packed_active_pairs,
    std::vector<double>* packed_active_two_electron_gradient) {
  for (int beta_channel = 0;
       beta_channel < n_packed_active_pairs;
       ++beta_channel) {
    for (int alpha_channel = 0;
         alpha_channel < n_packed_active_pairs;
         ++alpha_channel) {
      const double value = ordered_gradient(alpha_channel, beta_channel);
      if (std::abs(value) <= kContributionTolerance) {
        continue;
      }
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              beta_channel,
              alpha_channel);
      (*packed_active_two_electron_gradient)[
          packed_pair_of_pairs_index] += value;
    }
  }
}

}  // namespace

void accumulate_opposite_spin_packed_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    std::vector<double>* packed_active_two_electron_gradient) {
  const SelectedStatePairGraph pair_graph(
      selected_states,
      PrimarySpin::Alpha);
  Eigen::MatrixXd ordered_gradient = Eigen::MatrixXd::Zero(
      n_packed_active_pairs,
      n_packed_active_pairs);
  accumulate_ordered_packed_gradient_from_pair_graph(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_states,
      pair_graph,
      n_packed_active_pairs,
      &ordered_gradient);
  pack_ordered_gradient(
      ordered_gradient,
      n_packed_active_pairs,
      packed_active_two_electron_gradient);
}

void accumulate_directional_opposite_spin_packed_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    std::vector<double>* packed_active_two_electron_gradient) {
  const SelectedStatePairGraph pair_graph(
      selected_states,
      directional_selected_states,
      PrimarySpin::Alpha);
  Eigen::MatrixXd ordered_gradient = Eigen::MatrixXd::Zero(
      n_packed_active_pairs,
      n_packed_active_pairs);
  accumulate_ordered_packed_gradient_from_pair_graph(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_states,
      pair_graph,
      n_packed_active_pairs,
      &ordered_gradient);
  pack_ordered_gradient(
      ordered_gradient,
      n_packed_active_pairs,
      packed_active_two_electron_gradient);
}

void accumulate_local_opposite_spin_packed_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    std::vector<double>* packed_active_two_electron_gradient) {
  const SelectedStatePairGraph pair_graph(
      selected_states,
      PrimarySpin::Alpha);
  Eigen::MatrixXd ordered_gradient = Eigen::MatrixXd::Zero(
      n_packed_active_pairs,
      n_packed_active_pairs);
  accumulate_ordered_packed_gradient_from_pair_graph(
      alpha_directional_pair_data,
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_states,
      pair_graph,
      n_packed_active_pairs,
      &ordered_gradient);
  accumulate_ordered_packed_gradient_from_pair_graph(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      beta_directional_pair_data,
      selected_states,
      pair_graph,
      n_packed_active_pairs,
      &ordered_gradient);
  pack_ordered_gradient(
      ordered_gradient,
      n_packed_active_pairs,
      packed_active_two_electron_gradient);
}

}  // namespace detail

}  // namespace xmvb::vb
