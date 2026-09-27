#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/packed_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/ri_factor_adjoint.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

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

  const std::vector<std::vector<int>> one_electron_determinants = {
      {0}, {1}};
  const std::vector<double> active_overlap = {
      1.0, 0.2,
      0.2, 1.0};
  Eigen::Matrix2d active_one_electron;
  active_one_electron << -0.7, 0.1,
                          0.1, -0.4;
  xmvb::vb::ActiveSpaceTwoElectronResult cache_two_electron_result;
  cache_two_electron_result.packed_active_two_electron_integrals = {
      0.8, 0.1, 0.6, 0.2, 0.15, 0.5};
  xmvb::vb::DeterminantPairEvaluator pair_evaluator;
  auto eager_cache = xmvb::vb::build_same_spin_pair_cache_context(
      one_electron_determinants,
      one_electron_determinants,
      pair_evaluator,
      active_overlap,
      active_one_electron,
      2,
      cache_two_electron_result);
  auto lazy_cache = xmvb::vb::build_same_spin_pair_cache_context(
      one_electron_determinants,
      one_electron_determinants,
      pair_evaluator,
      active_overlap,
      active_one_electron,
      2,
      cache_two_electron_result,
      xmvb::vb::SameSpinPairCacheBuildOptions{
          xmvb::vb::PairProjectionCache::Both,
          false});
  require(
      eager_cache.alpha_pair_cache.front().cofactor_differential != nullptr,
      "eager pair cache omitted its derivative payload");
  require(
      lazy_cache.alpha_pair_cache.front().cofactor_differential == nullptr,
      "forward-only pair cache constructed a derivative payload");
  require(
      !eager_cache.alpha_pair_cache.front()
           .opposite_spin_pair_cache.inverse_overlap_projection
           .packed_pair_indices.empty(),
      "regular pair cache omitted its sparse inverse-overlap projection");
  require(
      eager_cache.alpha_pair_cache.front()
          .opposite_spin_pair_cache.inverse_overlap_projection
          .projected_pair_values.empty(),
      "pair cache retained an unused dense inverse-overlap image");
  for (std::size_t pair = 0;
       pair < eager_cache.alpha_pair_cache.size();
       ++pair) {
    const auto& eager = eager_cache.alpha_pair_cache[pair];
    const auto& lazy = lazy_cache.alpha_pair_cache[pair];
    require(
        std::abs(
            eager.overlap_result.overlap_determinant -
            lazy.overlap_result.overlap_determinant) <= 1.0e-14 &&
            std::abs(eager.total_hamiltonian - lazy.total_hamiltonian) <=
                1.0e-14,
        "transpose-generated forward pair scalar disagrees with direct evaluation");
    for (int channel = 0; channel < n_channels; ++channel) {
      require(
          std::abs(
              channel_value(
                  eager.opposite_spin_pair_cache
                      .first_order_cofactor_projection,
                  channel) -
              channel_value(
                  lazy.opposite_spin_pair_cache
                      .first_order_cofactor_projection,
                  channel)) <= 1.0e-14,
          "transpose-generated pair projection disagrees with direct evaluation");
    }
  }
  xmvb::vb::populate_same_spin_phi_cache(
      &eager_cache,
      active_one_electron,
      2,
      cache_two_electron_result);
  xmvb::vb::populate_same_spin_phi_cache(
      &lazy_cache,
      active_one_electron,
      2,
      cache_two_electron_result);
  for (std::size_t pair = 0;
       pair < eager_cache.alpha_pair_cache.size();
       ++pair) {
    const auto& eager = eager_cache.alpha_pair_cache[pair];
    const auto& lazy = lazy_cache.alpha_pair_cache[pair];
    require(
        lazy.cofactor_differential != nullptr,
        "lazy pair cache did not complete its derivative payload");
    require(
        std::abs(eager.total_hamiltonian - lazy.total_hamiltonian) <= 1.0e-14 &&
            std::abs(
                eager.same_spin_total_phi - lazy.same_spin_total_phi) <=
                1.0e-14 &&
            (eager.same_spin_inverse_overlap_gradient -
             lazy.same_spin_inverse_overlap_gradient).norm() <= 1.0e-14 &&
            (eager.same_spin_overlap_hamiltonian_gradient -
             lazy.same_spin_overlap_hamiltonian_gradient).norm() <= 1.0e-14,
        "lazy derivative cache disagrees with eager construction");
  }

  xmvb::vb::ActiveSpaceTwoElectronResult ri_pair_result;
  ri_pair_result.representation =
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
  ri_pair_result.n_auxiliary_functions = 2;
  ri_pair_result.ri_active_pair_factors.resize(2, n_channels);
  ri_pair_result.ri_active_pair_factors <<
      0.31, -0.17, 0.23,
      -0.08, 0.29, 0.41;
  Eigen::MatrixXd directional_ri_factors(2, n_channels);
  directional_ri_factors <<
      -0.13, 0.07, 0.19,
      0.11, -0.05, 0.03;
  const Eigen::MatrixXd one_sided_ri_direction =
      ri_pair_result.ri_active_pair_factors.transpose() *
      directional_ri_factors;
  const Eigen::MatrixXd ri_kernel =
      ri_pair_result.ri_active_pair_factors.transpose() *
      ri_pair_result.ri_active_pair_factors;
  xmvb::vb::ActiveSpaceTwoElectronResult packed_ri_result = ri_pair_result;
  packed_ri_result.representation =
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::PackedExact;
  packed_ri_result.packed_active_two_electron_integrals.assign(
      static_cast<std::size_t>(n_channels) * (n_channels + 1) / 2,
      0.0);
  for (int row = 0; row < n_channels; ++row) {
    for (int column = 0; column <= row; ++column) {
      packed_ri_result.packed_active_two_electron_integrals[
          xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
              row, column)] = ri_kernel(row, column);
    }
  }
  std::vector<double> packed_ri_direction(
      static_cast<std::size_t>(n_channels) * (n_channels + 1) / 2,
      0.0);
  for (int row = 0; row < n_channels; ++row) {
    for (int column = 0; column <= row; ++column) {
      packed_ri_direction[
          xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
              row,
              column)] =
          one_sided_ri_direction(row, column) +
          one_sided_ri_direction(column, row);
    }
  }
  std::vector<xmvb::vb::SameSpinPolynomialDirectionalPairData>
      polynomial_pair_directions(eager_cache.alpha_pair_cache.size());
  for (std::size_t pair = 0;
       pair < polynomial_pair_directions.size();
       ++pair) {
    polynomial_pair_directions[pair].delta_cofactor_1st =
        Eigen::MatrixXd::Constant(1, 1, 0.04 * (pair + 1));
  }
  const std::vector<double> zero_overlap_direction(4, 0.0);
  const std::vector<double> zero_one_electron_direction(4, 0.0);
  const xmvb::vb::ActiveSpaceIntegralDirectionView ri_direction_view{
      zero_overlap_direction,
      zero_one_electron_direction,
      packed_ri_direction};
  xmvb::vb::detail::SameSpinDirectionalPairTile same_spin_tile;
  same_spin_tile.left_begin = 0;
  same_spin_tile.right_begin = 0;
  same_spin_tile.delta_overlap = Eigen::MatrixXd::Zero(2, 2);
  same_spin_tile.delta_regular_hamiltonian = Eigen::MatrixXd::Zero(2, 2);
  same_spin_tile.delta_singular_hamiltonian = Eigen::MatrixXd::Zero(2, 2);
  same_spin_tile.pairs = polynomial_pair_directions;
  const auto packed_directional_tile =
      xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
          one_electron_determinants,
          eager_cache.alpha_pair_cache,
          2,
          2,
          packed_ri_result,
          ri_direction_view,
          same_spin_tile.view());
  const auto factor_directional_tile =
      xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
          one_electron_determinants,
          eager_cache.alpha_pair_cache,
          2,
          2,
          ri_pair_result,
          ri_direction_view,
          same_spin_tile.view(),
          &ri_pair_result.ri_active_pair_factors,
          &directional_ri_factors);
  require(
      packed_directional_tile.projected_channel_values.rows() ==
          factor_directional_tile.projected_channel_values.rows() &&
      packed_directional_tile.projected_channel_values.cols() ==
          factor_directional_tile.projected_channel_values.cols() &&
      (packed_directional_tile.projected_channel_values -
       factor_directional_tile.projected_channel_values)
              .cwiseAbs()
              .maxCoeff() <= 1.0e-13,
      "RI-native opposite-spin direction disagrees with packed reference");

  xmvb::vb::SameSpinPairCacheContext cache;
  fill_pair_cache(n_alpha, n_channels, 0, &cache.alpha_pair_cache);
  fill_pair_cache(n_beta, n_channels, 1, &cache.beta_pair_cache);
  fill_regular_overlap_data(n_alpha, n_channels, &cache.alpha_pair_cache);
  fill_regular_overlap_data(n_beta, n_channels, &cache.beta_pair_cache);
  for (auto& pair : cache.alpha_pair_cache) {
    pair.has_same_spin_phi_cache = true;
  }
  for (auto& pair : cache.beta_pair_cache) {
    pair.has_same_spin_phi_cache = true;
  }
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
  require(
      std::abs(
          xmvb::vb::estimate_selected_state_contraction_work(accepted) -
          30.0) <= 1.0e-12,
      "dense unique-string contraction work estimate is incorrect");
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

  Eigen::MatrixXd accepted_ri_factors(4, n_channels);
  accepted_ri_factors <<
      0.31, -0.07, 0.18,
      -0.11, 0.29, 0.06,
      0.17, 0.13, -0.23,
      0.04, -0.19, 0.27;
  const Eigen::MatrixXd factor_adjoint =
      xmvb::vb::apply_regular_ri_pair_space_adjoint(
          cache,
          accepted,
          std::vector<double>{0.0},
          2,
          accepted_ri_factors);
  Eigen::MatrixXd packed_adjoint_matrix(n_channels, n_channels);
  for (int row = 0; row < n_channels; ++row) {
    for (int column = 0; column < n_channels; ++column) {
      const double packed_value =
          accepted_contribution.packed_active_two_electron_gradient[
              xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                  row,
                  column)];
      packed_adjoint_matrix(row, column) =
          row == column ? 2.0 * packed_value : packed_value;
    }
  }
  const Eigen::MatrixXd expected_factor_adjoint =
      accepted_ri_factors * packed_adjoint_matrix;
  require(
      (factor_adjoint - expected_factor_adjoint).norm() <=
          1.0e-12 * std::max(1.0, expected_factor_adjoint.norm()),
      "RI-native pair-space adjoint disagrees with packed reference");

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
