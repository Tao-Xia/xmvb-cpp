#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/packed_contractions_internal.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace {

using xmvb::vb::OppositeSpinPackedPairProjection;
using xmvb::vb::SelectedStateDeterminantCoefficients;
using xmvb::vb::SelectedStateDeterminantMatrices;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

SelectedStateDeterminantCoefficients make_state(
    const Eigen::MatrixXd& coefficients,
    double weight) {
  SelectedStateDeterminantCoefficients state;
  state.normalized_state_weight = weight;
  state.coefficient_matrix = coefficients;
  state.alpha_support.resize(coefficients.rows());
  state.beta_support.resize(coefficients.cols());
  for (int row = 0; row < coefficients.rows(); ++row) {
    state.alpha_support[row] = row;
  }
  for (int column = 0; column < coefficients.cols(); ++column) {
    state.beta_support[column] = column;
  }

  std::vector<Eigen::Triplet<double, int>> triplets;
  for (int row = 0; row < coefficients.rows(); ++row) {
    for (int column = 0; column < coefficients.cols(); ++column) {
      if (coefficients(row, column) != 0.0) {
        triplets.emplace_back(row, column, coefficients(row, column));
      }
    }
  }
  state.nonzero_coefficient_count = static_cast<int>(triplets.size());
  state.local_sparse_coefficient_matrix.resize(
      coefficients.rows(),
      coefficients.cols());
  state.local_sparse_coefficient_matrix.setFromTriplets(
      triplets.begin(),
      triplets.end());
  state.local_sparse_coefficient_transpose =
      state.local_sparse_coefficient_matrix.transpose();
  return state;
}

void fill_pair_cache(
    int n_unique,
    int n_channels,
    int spin_offset,
    std::vector<xmvb::vb::SpinDeterminantPairEvaluation>* cache) {
  cache->resize(static_cast<std::size_t>(n_unique) * n_unique);
  for (int right = 0; right < n_unique; ++right) {
    for (int left = 0; left < n_unique; ++left) {
      auto& pair = (*cache)[xmvb::vb::ordered_spin_pair_storage_index(
          left, right, n_unique)];
      auto& projection =
          pair.opposite_spin_pair_cache.first_order_cofactor_projection;
      pair.opposite_spin_pair_cache.n_packed_active_pairs = n_channels;
      for (int channel = 0; channel < n_channels; ++channel) {
        if ((left + 2 * right + channel + spin_offset) % 3 == 0) {
          continue;
        }
        projection.packed_pair_indices.push_back(channel);
        projection.packed_pair_values.push_back(
            0.11 * (1 + left) - 0.07 * (1 + right) +
            0.05 * (1 + channel + spin_offset));
      }
    }
  }
}

double channel_value(
    const OppositeSpinPackedPairProjection& projection,
    int channel) {
  for (std::size_t entry = 0;
       entry < projection.packed_pair_indices.size();
       ++entry) {
    if (projection.packed_pair_indices[entry] == channel) {
      return projection.packed_pair_values[entry];
    }
  }
  return 0.0;
}

std::vector<double> direct_packed_gradient(
    const xmvb::vb::SameSpinPairCacheContext& cache,
    const Eigen::MatrixXd& accepted_coefficients,
    const Eigen::MatrixXd& directional_coefficients,
    int n_channels) {
  const int n_alpha = accepted_coefficients.rows();
  const int n_beta = accepted_coefficients.cols();
  std::vector<double> result(
      static_cast<std::size_t>(n_channels) * (n_channels + 1) / 2,
      0.0);
  for (int alpha_right = 0; alpha_right < n_alpha; ++alpha_right) {
    for (int alpha_left = 0; alpha_left < n_alpha; ++alpha_left) {
      const auto& alpha_projection =
          cache.alpha_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
              alpha_left, alpha_right, n_alpha)]
              .opposite_spin_pair_cache.first_order_cofactor_projection;
      for (int beta_right = 0; beta_right < n_beta; ++beta_right) {
        for (int beta_left = 0; beta_left < n_beta; ++beta_left) {
          const auto& beta_projection =
              cache.beta_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
                  beta_left, beta_right, n_beta)]
                  .opposite_spin_pair_cache.first_order_cofactor_projection;
          const double coefficient =
              directional_coefficients(alpha_left, beta_left) *
                  accepted_coefficients(alpha_right, beta_right) +
              accepted_coefficients(alpha_left, beta_left) *
                  directional_coefficients(alpha_right, beta_right);
          for (int alpha_channel = 0;
               alpha_channel < n_channels;
               ++alpha_channel) {
            const double alpha_value =
                channel_value(alpha_projection, alpha_channel);
            for (int beta_channel = 0;
                 beta_channel < n_channels;
                 ++beta_channel) {
              result[xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                  beta_channel,
                  alpha_channel)] +=
                  coefficient * alpha_value *
                  channel_value(beta_projection, beta_channel);
            }
          }
        }
      }
    }
  }
  return result;
}

std::vector<xmvb::vb::detail::DirectionalOppositeSpinPairData>
make_directional_pairs(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& accepted_pairs,
    double pair_scale) {
  std::vector<xmvb::vb::detail::DirectionalOppositeSpinPairData> result(
      accepted_pairs.size());
  for (std::size_t pair = 0; pair < accepted_pairs.size(); ++pair) {
    const auto& accepted = accepted_pairs[pair]
        .opposite_spin_pair_cache.first_order_cofactor_projection;
    auto& directional = result[pair].delta_first_order_cofactor_projection;
    directional.packed_pair_indices = accepted.packed_pair_indices;
    directional.packed_pair_values.resize(accepted.packed_pair_values.size());
    for (std::size_t entry = 0;
         entry < accepted.packed_pair_values.size();
         ++entry) {
      directional.packed_pair_values[entry] =
          pair_scale * accepted.packed_pair_values[entry] +
          0.01 * static_cast<double>(pair + entry + 1);
    }
  }
  return result;
}

std::vector<double> direct_local_packed_gradient(
    const xmvb::vb::SameSpinPairCacheContext& cache,
    const std::vector<xmvb::vb::detail::DirectionalOppositeSpinPairData>&
        alpha_directional_pairs,
    const std::vector<xmvb::vb::detail::DirectionalOppositeSpinPairData>&
        beta_directional_pairs,
    const Eigen::MatrixXd& coefficients,
    int n_channels) {
  const int n_alpha = coefficients.rows();
  const int n_beta = coefficients.cols();
  std::vector<double> result(
      static_cast<std::size_t>(n_channels) * (n_channels + 1) / 2,
      0.0);
  for (int alpha_right = 0; alpha_right < n_alpha; ++alpha_right) {
    for (int alpha_left = 0; alpha_left < n_alpha; ++alpha_left) {
      const std::size_t alpha_pair = xmvb::vb::ordered_spin_pair_storage_index(
          alpha_left, alpha_right, n_alpha);
      const auto& alpha = cache.alpha_pair_cache[alpha_pair]
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      const auto& delta_alpha = alpha_directional_pairs[alpha_pair]
          .delta_first_order_cofactor_projection;
      for (int beta_right = 0; beta_right < n_beta; ++beta_right) {
        for (int beta_left = 0; beta_left < n_beta; ++beta_left) {
          const std::size_t beta_pair = xmvb::vb::ordered_spin_pair_storage_index(
              beta_left, beta_right, n_beta);
          const auto& beta = cache.beta_pair_cache[beta_pair]
              .opposite_spin_pair_cache.first_order_cofactor_projection;
          const auto& delta_beta = beta_directional_pairs[beta_pair]
              .delta_first_order_cofactor_projection;
          const double coefficient =
              coefficients(alpha_left, beta_left) *
              coefficients(alpha_right, beta_right);
          for (int alpha_channel = 0;
               alpha_channel < n_channels;
               ++alpha_channel) {
            for (int beta_channel = 0;
                 beta_channel < n_channels;
                 ++beta_channel) {
              result[xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                  beta_channel,
                  alpha_channel)] += coefficient *
                  (channel_value(delta_alpha, alpha_channel) *
                       channel_value(beta, beta_channel) +
                   channel_value(alpha, alpha_channel) *
                       channel_value(delta_beta, beta_channel));
            }
          }
        }
      }
    }
  }
  return result;
}

void fill_regular_overlap_data(
    int n_unique,
    int n_channels,
    std::vector<xmvb::vb::SpinDeterminantPairEvaluation>* cache) {
  for (int right = 0; right < n_unique; ++right) {
    for (int left = 0; left < n_unique; ++left) {
      auto& pair = (*cache)[xmvb::vb::ordered_spin_pair_storage_index(
          left, right, n_unique)];
      pair.overlap_result.n_electrons = 1;
      pair.overlap_result.nullity = 0;
      pair.overlap_result.overlap_determinant =
          0.8 + 0.03 * left - 0.02 * right;
      pair.overlap_result.inverse_overlap_submatrix =
          Eigen::MatrixXd::Constant(
              1,
              1,
              1.1 - 0.04 * left + 0.05 * right);
      auto& inverse =
          pair.opposite_spin_pair_cache.inverse_overlap_projection;
      for (int channel = 0; channel < n_channels; ++channel) {
        if ((left + right + channel) % 3 == 1) {
          continue;
        }
        inverse.packed_pair_indices.push_back(channel);
        inverse.packed_pair_values.push_back(
            0.09 + 0.02 * left - 0.01 * right + 0.03 * channel);
      }
    }
  }
}

std::vector<double> direct_overlap_gradient(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& primary_cache,
    const std::vector<std::vector<int>>& primary_determinants,
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& partner_cache,
    const Eigen::MatrixXd& accepted_primary_partner,
    const Eigen::MatrixXd& directional_primary_partner,
    const std::vector<double>& packed_kernel,
    int n_active_orbitals) {
  const int n_primary = accepted_primary_partner.rows();
  const int n_partner = accepted_primary_partner.cols();
  const int n_channels =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  std::vector<double> result(
      static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals,
      0.0);
  std::vector<double> partner_image(n_channels, 0.0);
  std::vector<double> projected_image(n_channels, 0.0);

  for (int primary_right = 0; primary_right < n_primary; ++primary_right) {
    for (int primary_left = 0; primary_left < n_primary; ++primary_left) {
      std::fill(partner_image.begin(), partner_image.end(), 0.0);
      for (int partner_right = 0; partner_right < n_partner; ++partner_right) {
        for (int partner_left = 0; partner_left < n_partner; ++partner_left) {
          const double coefficient =
              directional_primary_partner(primary_left, partner_left) *
                  accepted_primary_partner(primary_right, partner_right) +
              accepted_primary_partner(primary_left, partner_left) *
                  directional_primary_partner(primary_right, partner_right);
          const auto& projection =
              partner_cache[xmvb::vb::ordered_spin_pair_storage_index(
                  partner_left, partner_right, n_partner)]
                  .opposite_spin_pair_cache.first_order_cofactor_projection;
          for (int channel = 0; channel < n_channels; ++channel) {
            partner_image[channel] +=
                coefficient * channel_value(projection, channel);
          }
        }
      }
      for (int row = 0; row < n_channels; ++row) {
        projected_image[row] = 0.0;
        for (int column = 0; column < n_channels; ++column) {
          projected_image[row] +=
              packed_kernel[
                  xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                      row, column)] *
              partner_image[column];
        }
      }

      const auto& pair =
          primary_cache[xmvb::vb::ordered_spin_pair_storage_index(
              primary_left, primary_right, n_primary)];
      const auto& inverse =
          pair.opposite_spin_pair_cache.inverse_overlap_projection;
      double determinant_weight = 0.0;
      for (std::size_t entry = 0;
           entry < inverse.packed_pair_indices.size();
           ++entry) {
        determinant_weight +=
            inverse.packed_pair_values[entry] *
            projected_image[inverse.packed_pair_indices[entry]];
      }
      const int occ_left = primary_determinants[primary_left][0];
      const int occ_right = primary_determinants[primary_right][0];
      const int occupied_channel =
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              occ_right, occ_left);
      const double inverse_overlap =
          pair.overlap_result.inverse_overlap_submatrix(0, 0);
      result[occ_left * n_active_orbitals + occ_right] +=
          determinant_weight * pair.overlap_result.overlap_determinant *
              inverse_overlap -
          pair.overlap_result.overlap_determinant * inverse_overlap *
              projected_image[occupied_channel] * inverse_overlap;
    }
  }
  return result;
}

}  // namespace

int main() {
  constexpr int n_alpha = 2;
  constexpr int n_beta = 3;
  constexpr int n_channels = 3;

  xmvb::vb::SameSpinPairCacheContext cache;
  fill_pair_cache(n_alpha, n_channels, 0, &cache.alpha_pair_cache);
  fill_pair_cache(n_beta, n_channels, 1, &cache.beta_pair_cache);
  fill_regular_overlap_data(n_alpha, n_channels, &cache.alpha_pair_cache);
  fill_regular_overlap_data(n_beta, n_channels, &cache.beta_pair_cache);
  cache.alpha_reuse_table.unique_determinants = {{0}, {1}};
  cache.beta_reuse_table.unique_determinants = {{0}, {1}, {0}};
  cache.use_same_spin_pair_cache = true;

  Eigen::Matrix<double, n_alpha, n_beta> accepted_coefficients;
  accepted_coefficients <<
      0.7, 0.0, -0.2,
      0.3, -0.4, 0.5;
  Eigen::Matrix<double, n_alpha, n_beta> directional_coefficients;
  directional_coefficients <<
      0.0, 0.6, 0.1,
      -0.8, 0.0, 0.2;

  SelectedStateDeterminantMatrices accepted;
  accepted.n_unique_alpha = n_alpha;
  accepted.n_unique_beta = n_beta;
  accepted.states.push_back(make_state(accepted_coefficients, 1.0));
  SelectedStateDeterminantMatrices directional;
  directional.n_unique_alpha = n_alpha;
  directional.n_unique_beta = n_beta;
  directional.states.push_back(make_state(directional_coefficients, 1.0));

  const int packed_size = n_channels * (n_channels + 1) / 2;
  std::vector<double> actual(packed_size, 0.0);
  xmvb::vb::detail::accumulate_directional_opposite_spin_packed_gradient_by_pair_graph(
      cache,
      accepted,
      directional,
      n_channels,
      &actual);

  const std::vector<double> expected = direct_packed_gradient(
      cache,
      accepted_coefficients,
      directional_coefficients,
      n_channels);

  double max_error = 0.0;
  double reference_scale = 1.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    max_error = std::max(max_error, std::abs(actual[index] - expected[index]));
    reference_scale = std::max(reference_scale, std::abs(expected[index]));
  }
  require(
      max_error <= 1.0e-12 * reference_scale,
      "pair-major opposite-spin contraction disagrees with direct sum");

  xmvb::vb::ActiveSpaceTwoElectronResult two_electron_result;
  two_electron_result.packed_active_two_electron_integrals.resize(packed_size);
  for (int row = 0; row < n_channels; ++row) {
    for (int column = 0; column <= row; ++column) {
      two_electron_result.packed_active_two_electron_integrals[
          xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
              row, column)] =
          0.4 + 0.03 * row - 0.02 * column;
    }
  }
  const auto contribution =
      xmvb::vb::build_directional_opposite_spin_backward_contribution(
          cache,
          accepted,
          directional,
          2,
          two_electron_result);
  std::vector<double> expected_overlap = direct_overlap_gradient(
      cache.alpha_pair_cache,
      cache.alpha_reuse_table.unique_determinants,
      cache.beta_pair_cache,
      accepted_coefficients,
      directional_coefficients,
      two_electron_result.packed_active_two_electron_integrals,
      2);
  const std::vector<double> expected_beta_overlap = direct_overlap_gradient(
      cache.beta_pair_cache,
      cache.beta_reuse_table.unique_determinants,
      cache.alpha_pair_cache,
      accepted_coefficients.transpose(),
      directional_coefficients.transpose(),
      two_electron_result.packed_active_two_electron_integrals,
      2);
  for (std::size_t index = 0; index < expected_overlap.size(); ++index) {
    expected_overlap[index] += expected_beta_overlap[index];
  }
  max_error = 0.0;
  reference_scale = 1.0;
  for (std::size_t index = 0; index < expected_overlap.size(); ++index) {
    max_error = std::max(
        max_error,
        std::abs(
            contribution.active_orbital_overlap_gradient[index] -
            expected_overlap[index]));
    reference_scale = std::max(
        reference_scale,
        std::abs(expected_overlap[index]));
  }
  require(
      max_error <= 1.0e-12 * reference_scale,
      "pair-major opposite-spin overlap adjoint disagrees with direct sum");

  const auto accepted_contribution =
      xmvb::vb::build_opposite_spin_backward_contribution(
          cache,
          accepted,
          2,
          two_electron_result);
  const Eigen::MatrixXd half_accepted_coefficients =
      0.5 * accepted_coefficients;
  const std::vector<double> expected_accepted_packed = direct_packed_gradient(
      cache,
      accepted_coefficients,
      half_accepted_coefficients,
      n_channels);
  max_error = 0.0;
  reference_scale = 1.0;
  for (std::size_t index = 0;
       index < expected_accepted_packed.size();
       ++index) {
    max_error = std::max(
        max_error,
        std::abs(
            accepted_contribution.packed_active_two_electron_gradient[index] -
            expected_accepted_packed[index]));
    reference_scale = std::max(
        reference_scale,
        std::abs(expected_accepted_packed[index]));
  }
  require(
      max_error <= 1.0e-12 * reference_scale,
      "accepted pair-major packed adjoint disagrees with direct sum");

  std::vector<double> expected_accepted_overlap = direct_overlap_gradient(
      cache.alpha_pair_cache,
      cache.alpha_reuse_table.unique_determinants,
      cache.beta_pair_cache,
      accepted_coefficients,
      half_accepted_coefficients,
      two_electron_result.packed_active_two_electron_integrals,
      2);
  const std::vector<double> expected_accepted_beta_overlap =
      direct_overlap_gradient(
          cache.beta_pair_cache,
          cache.beta_reuse_table.unique_determinants,
          cache.alpha_pair_cache,
          accepted_coefficients.transpose(),
          half_accepted_coefficients.transpose(),
          two_electron_result.packed_active_two_electron_integrals,
          2);
  for (std::size_t index = 0;
       index < expected_accepted_overlap.size();
       ++index) {
    expected_accepted_overlap[index] +=
        expected_accepted_beta_overlap[index];
  }
  max_error = 0.0;
  reference_scale = 1.0;
  for (std::size_t index = 0;
       index < expected_accepted_overlap.size();
       ++index) {
    max_error = std::max(
        max_error,
        std::abs(
            accepted_contribution.active_orbital_overlap_gradient[index] -
            expected_accepted_overlap[index]));
    reference_scale = std::max(
        reference_scale,
        std::abs(expected_accepted_overlap[index]));
  }
  require(
      max_error <= 1.0e-12 * reference_scale,
      "accepted pair-major overlap adjoint disagrees with direct sum");

  const auto alpha_directional_pairs =
      make_directional_pairs(cache.alpha_pair_cache, 0.17);
  const auto beta_directional_pairs =
      make_directional_pairs(cache.beta_pair_cache, -0.23);
  std::vector<double> local_actual(packed_size, 0.0);
  xmvb::vb::detail::accumulate_local_opposite_spin_packed_gradient_by_pair_graph(
      cache,
      alpha_directional_pairs,
      beta_directional_pairs,
      accepted,
      n_channels,
      &local_actual);
  const std::vector<double> local_expected = direct_local_packed_gradient(
      cache,
      alpha_directional_pairs,
      beta_directional_pairs,
      accepted_coefficients,
      n_channels);
  max_error = 0.0;
  reference_scale = 1.0;
  for (std::size_t index = 0; index < local_expected.size(); ++index) {
    max_error = std::max(
        max_error,
        std::abs(local_actual[index] - local_expected[index]));
    reference_scale = std::max(reference_scale, std::abs(local_expected[index]));
  }
  require(
      max_error <= 1.0e-12 * reference_scale,
      "local pair-major packed adjoint disagrees with direct sum");
  return 0;
}
