#include "vbscf/integrals/active/two_electron/response/directional.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/response/internal.hpp"
#include "vbscf/integrals/active/two_electron/transformation/ao_pair_operator.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {
namespace {

using detail::accumulate_pair_gradient_rows;
using detail::apply_exact_ao_pair_kernel;
using detail::apply_generated_pair_rows;
using detail::build_mixed_pair_rows;
using detail::build_mixed_ao_pair_to_active_pair_coefficients_from_cache;
using detail::resize_for_overwrite;

}  // namespace

void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals) {
  if (workspace == nullptr) {
    throw std::invalid_argument("exact packed delta GGO workspace must not be null");
  }
  if (delta_packed_active_two_electron_integrals == nullptr) {
    throw std::invalid_argument("exact packed delta GGO output must not be null");
  }

  const int n_bf = ao_integral_input.n_basis_functions;
  const int n_ao = accepted_cache.n_active_orbitals;
  if (n_bf <= 0 || n_ao <= 0) {
    throw std::invalid_argument(
        "exact packed delta GGO cache dimensions must be positive");
  }
  if (accepted_cache.n_basis_functions != n_bf) {
    throw std::invalid_argument("exact packed delta GGO cache basis mismatch");
  }
  if (dense_active_direction.rows() != n_bf ||
      dense_active_direction.cols() != n_ao) {
    throw std::invalid_argument(
        "dense active direction shape mismatch in cached delta GGO");
  }

  if (accepted_cache.accepted_dense_active_coefficients.rows() !=
          n_bf ||
      accepted_cache.accepted_dense_active_coefficients.cols() !=
          n_ao) {
    throw std::invalid_argument(
        "cached dense active coefficient size mismatch in delta GGO");
  }
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact packed delta GGO cache pair-index mismatch");
  }
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  if (accepted_cache.active_pair_gradient_matrix.rows() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.active_pair_gradient_matrix.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument(
        "cached active-pair adjoint size mismatch in delta GGO");
  }
  if (accepted_cache.accepted_pair_products == nullptr) {
    workspace->dense_active_direction = dense_active_direction;
    workspace->active_pair_contraction.setZero(
        static_cast<Eigen::Index>(n_active_pairs),
        static_cast<Eigen::Index>(n_active_pairs));
    workspace->dense_fixed_adjoint_direction.setZero(n_bf, n_ao);
    const Eigen::Index tile_rows = n_bf;
    for (Eigen::Index row_begin = 0;
         row_begin < static_cast<Eigen::Index>(n_bf_pairs);
         row_begin += tile_rows) {
      const Eigen::Index row_count = std::min(
          tile_rows,
          static_cast<Eigen::Index>(n_bf_pairs) - row_begin);
      build_mixed_pair_rows(
          accepted_cache.accepted_dense_active_coefficients,
          dense_active_direction,
          accepted_cache,
          row_begin,
          row_count,
          &workspace->directional_pair_coefficients);
      apply_generated_pair_rows(
          ao_integral_input,
          accepted_cache.accepted_dense_active_coefficients,
          &workspace->dense_active_direction,
          accepted_cache,
          row_begin,
          row_count,
          &workspace->accepted_pair_products_tile,
          &workspace->directional_pair_products);
      workspace->active_pair_contraction.noalias() +=
          workspace->directional_pair_coefficients.transpose() *
          workspace->accepted_pair_products_tile;

      workspace->pair_gradients_tile.noalias() =
          workspace->accepted_pair_products_tile *
          accepted_cache.active_pair_gradient_matrix;
      accumulate_pair_gradient_rows(
          workspace->pair_gradients_tile,
          row_begin,
          dense_active_direction,
          accepted_cache,
          &workspace->dense_fixed_adjoint_direction);

      workspace->pair_gradients_tile.noalias() =
          workspace->directional_pair_products *
          accepted_cache.active_pair_gradient_matrix;
      accumulate_pair_gradient_rows(
          workspace->pair_gradients_tile,
          row_begin,
          accepted_cache.accepted_dense_active_coefficients,
          accepted_cache,
          &workspace->dense_fixed_adjoint_direction);
    }
    workspace->directional_pair_products.resize(0, 0);
  } else {
    workspace->dense_fixed_adjoint_direction.resize(0, 0);
    if (accepted_cache.accepted_pair_products->rows() !=
            static_cast<Eigen::Index>(n_bf_pairs) ||
        accepted_cache.accepted_pair_products->cols() !=
            static_cast<Eigen::Index>(n_active_pairs)) {
      throw std::invalid_argument(
          "cached accepted pair buffers size mismatch in delta GGO");
    }

    build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
        accepted_cache.accepted_dense_active_coefficients,
        dense_active_direction,
        accepted_cache,
        &workspace->directional_pair_coefficients);
    apply_exact_ao_pair_kernel(
        ao_integral_input,
        workspace->directional_pair_coefficients,
        n_bf,
        n_active_pairs,
        &workspace->directional_pair_products);

    workspace->active_pair_contraction.resize(
        static_cast<Eigen::Index>(n_active_pairs),
        static_cast<Eigen::Index>(n_active_pairs));
    workspace->active_pair_contraction.noalias() =
        workspace->directional_pair_coefficients.transpose() *
        *accepted_cache.accepted_pair_products;
  }

  const std::size_t packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  resize_for_overwrite(
      delta_packed_active_two_electron_integrals,
      packed_size);
  for (std::size_t left_active_pair_index = 0;
       left_active_pair_index < n_active_pairs;
       ++left_active_pair_index) {
    for (std::size_t right_active_pair_index = 0;
         right_active_pair_index <= left_active_pair_index;
         ++right_active_pair_index) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              static_cast<int>(left_active_pair_index),
              static_cast<int>(right_active_pair_index));
      (*delta_packed_active_two_electron_integrals)[
          packed_index] =
          workspace->active_pair_contraction(
              static_cast<Eigen::Index>(left_active_pair_index),
              static_cast<Eigen::Index>(right_active_pair_index)) +
          workspace->active_pair_contraction(
              static_cast<Eigen::Index>(right_active_pair_index),
              static_cast<Eigen::Index>(left_active_pair_index));
    }
  }
}

Eigen::MatrixXd
compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions,
    const AoIntegralInput& ao_integral_input,
    std::vector<ExactCtxPairMatrix>* directional_pair_products,
    std::vector<Eigen::MatrixXd>* fixed_adjoint_directions) {
  const int n_bf = accepted_cache.n_basis_functions;
  const int n_ao = accepted_cache.n_active_orbitals;
  const Eigen::Index n_directions =
      static_cast<Eigen::Index>(dense_active_directions.size());
  const Eigen::Index n_active_pairs =
      static_cast<Eigen::Index>(
          accepted_cache.active_pair_first_indices.size());
  const Eigen::Index n_bf_pairs =
      static_cast<Eigen::Index>(n_bf) *
      (n_bf + 1) / 2;
  const Eigen::Index packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  Eigen::MatrixXd packed_directions(packed_size, n_directions);
  if (n_directions == 0) {
    if (directional_pair_products != nullptr) {
      directional_pair_products->clear();
    }
    if (fixed_adjoint_directions != nullptr) {
      fixed_adjoint_directions->clear();
    }
    return packed_directions;
  }
  if (n_bf <= 0 || n_ao <= 0 ||
      n_active_pairs <= 0 ||
      accepted_cache.active_pair_second_indices.size() !=
          static_cast<std::size_t>(n_active_pairs) ||
      (accepted_cache.accepted_pair_products != nullptr &&
       (accepted_cache.accepted_pair_products->rows() != n_bf_pairs ||
        accepted_cache.accepted_pair_products->cols() != n_active_pairs))) {
    throw std::invalid_argument(
        "cached exact delta GGO batch has inconsistent accepted dimensions");
  }

  if (accepted_cache.accepted_pair_products == nullptr) {
    if (directional_pair_products != nullptr) {
      directional_pair_products->assign(
          static_cast<std::size_t>(n_directions),
          ExactCtxPairMatrix{});
    }
    if (fixed_adjoint_directions != nullptr) {
      fixed_adjoint_directions->resize(
          static_cast<std::size_t>(n_directions));
    }
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace workspace;
    std::vector<double> packed_direction;
    for (Eigen::Index direction = 0;
         direction < n_directions;
         ++direction) {
      compute_exact_packed_active_two_electron_integral_directional_derivative(
          accepted_cache,
          dense_active_directions[direction],
          ao_integral_input,
          &workspace,
          &packed_direction);
      packed_directions.col(direction) =
          Eigen::Map<const Eigen::VectorXd>(
              packed_direction.data(),
              static_cast<Eigen::Index>(packed_direction.size()));
      if (fixed_adjoint_directions != nullptr) {
        (*fixed_adjoint_directions)[direction] =
            workspace.dense_fixed_adjoint_direction;
      }
    }
    return packed_directions;
  }
  if (fixed_adjoint_directions != nullptr) {
    fixed_adjoint_directions->assign(
        static_cast<std::size_t>(n_directions),
        Eigen::MatrixXd{});
  }

  ExactCtxPairMatrix combined_directional_coefficients(
      n_bf_pairs,
      n_active_pairs * n_directions);
  for (Eigen::Index direction = 0;
       direction < n_directions;
       ++direction) {
    if (dense_active_directions[direction].rows() != n_bf ||
        dense_active_directions[direction].cols() != n_ao) {
      throw std::invalid_argument(
          "dense active direction shape mismatch in delta GGO batch");
    }
    ExactCtxPairMatrix directional_coefficients;
    build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
        accepted_cache.accepted_dense_active_coefficients,
        dense_active_directions[direction],
        accepted_cache,
        &directional_coefficients);
    combined_directional_coefficients.middleCols(
        direction * n_active_pairs,
        n_active_pairs) = directional_coefficients;
  }

  ExactCtxPairMatrix combined_directional_products;
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      combined_directional_coefficients,
      n_bf,
      static_cast<std::size_t>(n_active_pairs * n_directions),
      &combined_directional_products);
  if (directional_pair_products != nullptr) {
    directional_pair_products->resize(n_directions);
  }

  for (Eigen::Index direction = 0;
       direction < n_directions;
       ++direction) {
    const auto directional_coefficients =
        combined_directional_coefficients.middleCols(
            direction * n_active_pairs,
            n_active_pairs);
    const auto directional_products =
        combined_directional_products.middleCols(
            direction * n_active_pairs,
            n_active_pairs);
    if (directional_pair_products != nullptr) {
      (*directional_pair_products)[direction] = directional_products;
    }
    const Eigen::MatrixXd directional_active_pair_contraction =
        directional_coefficients.transpose() *
        *accepted_cache.accepted_pair_products;
    // Apply the same symmetric-kernel identity independently to every block
    // direction; the directional products remain available for direct-core HVP.
    const Eigen::MatrixXd delta_active_pair_matrix =
        directional_active_pair_contraction +
        directional_active_pair_contraction.transpose();
    for (Eigen::Index left = 0; left < n_active_pairs; ++left) {
      for (Eigen::Index right = 0; right <= left; ++right) {
        const int packed_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                static_cast<int>(left),
                static_cast<int>(right));
        packed_directions(packed_index, direction) =
            delta_active_pair_matrix(left, right);
      }
    }
  }
  return packed_directions;
}

}  // namespace xmvb::vb
