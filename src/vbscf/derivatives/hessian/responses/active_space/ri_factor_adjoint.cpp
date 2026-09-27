#include "vbscf/derivatives/hessian/responses/active_space/ri_factor_adjoint.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/openmp.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/matrix_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb {
namespace {

constexpr double kContributionTolerance = 1.0e-15;
constexpr std::size_t kOppositeSpinWorkspaceBytes = 64u * 1024u * 1024u;

Eigen::MatrixXd gather_transition(
    const std::vector<int>& occ_left,
    const std::vector<int>& occ_right,
    const Eigen::Ref<const Eigen::MatrixXd>& factors,
    int auxiliary) {
  Eigen::MatrixXd transition(occ_right.size(), occ_left.size());
  for (int left = 0; left < static_cast<int>(occ_left.size()); ++left) {
    for (int right = 0; right < static_cast<int>(occ_right.size()); ++right) {
      transition(right, left) = factors(
          auxiliary,
          TwoElectronIndexer::packed_pair_index(
              occ_right[right], occ_left[left]));
    }
  }
  return transition;
}

void accumulate_same_spin_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& hamiltonian_weights,
    int left_begin,
    int right_begin,
    double spin_scale,
    const Eigen::Ref<const Eigen::MatrixXd>& factors,
    Eigen::MatrixXd* factor_adjoint) {
  const int n_unique = static_cast<int>(unique_determinants.size());
  const int n_auxiliary = static_cast<int>(factors.rows());
#pragma omp parallel for schedule(static) if(n_auxiliary > 1)
  for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
    for (int left_local = 0; left_local < hamiltonian_weights.rows(); ++left_local) {
      const int left = left_begin + left_local;
      for (int right_local = 0;
           right_local < hamiltonian_weights.cols();
           ++right_local) {
        const double weight =
            spin_scale * hamiltonian_weights(left_local, right_local);
        if (std::abs(weight) <= kContributionTolerance) {
          continue;
        }
        const int right = right_begin + right_local;
        const auto& pair = pair_cache[ordered_spin_pair_storage_index(
            left, right, n_unique)];
        const auto& inverse = pair.overlap_result.inverse_overlap_submatrix;
        const auto& occ_left = unique_determinants[left];
        const auto& occ_right = unique_determinants[right];
        const Eigen::MatrixXd transition = gather_transition(
            occ_left, occ_right, factors, auxiliary);
        const Eigen::MatrixXd contracted = inverse * transition;
        const Eigen::MatrixXd transition_adjoint =
            weight * pair.overlap_result.overlap_determinant *
            (contracted.trace() * inverse - contracted * inverse).transpose();
        for (int left_occ = 0;
             left_occ < static_cast<int>(occ_left.size());
             ++left_occ) {
          for (int right_occ = 0;
               right_occ < static_cast<int>(occ_right.size());
               ++right_occ) {
            const int packed_pair = TwoElectronIndexer::packed_pair_index(
                occ_right[right_occ], occ_left[left_occ]);
            (*factor_adjoint)(auxiliary, packed_pair) +=
                transition_adjoint(right_occ, left_occ);
          }
        }
      }
    }
  }
}

void accumulate_same_spin_adjoint(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& factors,
    Eigen::MatrixXd* factor_adjoint) {
  const bool close_shell = cache.close_shell_reuses_same_spin_pair_cache();
  const bool sparse = should_use_sparse_selected_state_contractions(selected_states);
  if (!sparse) {
    const detail::SameSpinExactWeightMatrices weights =
        detail::build_dense_exact_same_spin_weight_matrices(
            cache, selected_states, selected_state_energies);
    accumulate_same_spin_tile(
        cache.alpha_reuse_table.unique_determinants,
        cache.alpha_pair_cache_ref(),
        weights.alpha_hamiltonian_weight_matrix,
        0,
        0,
        close_shell ? 2.0 : 1.0,
        factors,
        factor_adjoint);
    if (!close_shell) {
      accumulate_same_spin_tile(
          cache.beta_reuse_table.unique_determinants,
          cache.beta_pair_cache_ref(),
          weights.beta_hamiltonian_weight_matrix,
          0,
          0,
          1.0,
          factors,
          factor_adjoint);
    }
    return;
  }

  const int n_active_orbitals =
      infer_active_orbital_count_from_packed_pair_count(
          static_cast<int>(factors.cols()));
  const detail::PairTileExtents tile_extents =
      detail::plan_pair_tile_extents(
          cache,
          n_active_orbitals,
          ActiveSpaceTwoElectronResult{},
          false);
  detail::SameSpinAcceptedTileWeights tile;
  const int n_alpha = selected_states.n_unique_alpha;
  for (int left = 0; left < n_alpha; left += tile_extents.alpha) {
    const int left_end = std::min(n_alpha, left + tile_extents.alpha);
    for (int right = 0; right < n_alpha; right += tile_extents.alpha) {
      const int right_end = std::min(n_alpha, right + tile_extents.alpha);
      detail::accumulate_alpha_accepted_tile_weights(
          selected_states,
          selected_state_energies,
          close_shell ? cache.alpha_pair_cache_ref()
                      : cache.beta_pair_cache_ref(),
          close_shell ? n_alpha : selected_states.n_unique_beta,
          left,
          left_end,
          right,
          right_end,
          &tile);
      accumulate_same_spin_tile(
          cache.alpha_reuse_table.unique_determinants,
          cache.alpha_pair_cache_ref(),
          tile.hamiltonian,
          left,
          right,
          close_shell ? 2.0 : 1.0,
          factors,
          factor_adjoint);
    }
  }
  if (close_shell) {
    return;
  }
  const int n_beta = selected_states.n_unique_beta;
  for (int left = 0; left < n_beta; left += tile_extents.beta) {
    const int left_end = std::min(n_beta, left + tile_extents.beta);
    for (int right = 0; right < n_beta; right += tile_extents.beta) {
      const int right_end = std::min(n_beta, right + tile_extents.beta);
      detail::accumulate_beta_accepted_tile_weights(
          selected_states,
          selected_state_energies,
          cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha,
          left,
          left_end,
          right,
          right_end,
          &tile);
      accumulate_same_spin_tile(
          cache.beta_reuse_table.unique_determinants,
          cache.beta_pair_cache_ref(),
          tile.hamiltonian,
          left,
          right,
          1.0,
          factors,
          factor_adjoint);
    }
  }
}

void accumulate_opposite_spin_adjoint(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::Ref<const Eigen::MatrixXd>& factors,
    Eigen::MatrixXd* factor_adjoint) {
  const int n_pairs = static_cast<int>(factors.cols());
  const int n_auxiliary = static_cast<int>(factors.rows());
  const int n_threads = std::max(
      1,
      std::min(
          xmvb::effective_openmp_thread_count(),
          selected_states.n_unique_alpha));
  const std::size_t row_bytes =
      static_cast<std::size_t>(n_threads) * n_pairs * sizeof(double);
  const int auxiliary_block = std::max(
      1,
      std::min(
          n_auxiliary,
          static_cast<int>(kOppositeSpinWorkspaceBytes /
              std::max<std::size_t>(row_bytes, 1u))));
  const detail::SelectedStatePairGraph pair_graph(
      selected_states, detail::PrimarySpin::Alpha);

  for (int auxiliary_begin = 0;
       auxiliary_begin < n_auxiliary;
       auxiliary_begin += auxiliary_block) {
    const int auxiliary_end =
        std::min(n_auxiliary, auxiliary_begin + auxiliary_block);
    const int block_rows = auxiliary_end - auxiliary_begin;
    std::vector<Eigen::MatrixXd> partials(
        n_threads,
        Eigen::MatrixXd::Zero(block_rows, n_pairs));
#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
#ifdef _OPENMP
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      auto& partial = partials[thread];
      std::vector<double> beta_image(n_pairs, 0.0);
      std::vector<unsigned char> touched_mask(n_pairs, 0u);
      std::vector<int> touched_beta;
      touched_beta.reserve(n_pairs);
      Eigen::VectorXd alpha_auxiliary(block_rows);
      Eigen::VectorXd beta_auxiliary(block_rows);
#pragma omp for schedule(static)
      for (int alpha_right = 0;
           alpha_right < selected_states.n_unique_alpha;
           ++alpha_right) {
        for (int alpha_left = 0;
             alpha_left < selected_states.n_unique_alpha;
             ++alpha_left) {
          const auto& alpha_projection =
              cache.alpha_pair_cache_ref()[ordered_spin_pair_storage_index(
                  alpha_left,
                  alpha_right,
                  selected_states.n_unique_alpha)]
                  .opposite_spin_pair_cache.first_order_cofactor_projection;
          if (alpha_projection.packed_pair_indices.empty()) {
            continue;
          }
          pair_graph.accumulate_partner_projection(
              alpha_left,
              alpha_right,
              cache.beta_pair_cache_ref(),
              selected_states.n_unique_beta,
              &beta_image,
              &touched_mask,
              &touched_beta);
          alpha_auxiliary.setZero();
          beta_auxiliary.setZero();
          for (std::size_t entry = 0;
               entry < alpha_projection.packed_pair_indices.size();
               ++entry) {
            alpha_auxiliary.noalias() +=
                factors.block(
                    auxiliary_begin,
                    alpha_projection.packed_pair_indices[entry],
                    block_rows,
                    1) *
                alpha_projection.packed_pair_values[entry];
          }
          for (const int pair : touched_beta) {
            beta_auxiliary.noalias() +=
                factors.block(auxiliary_begin, pair, block_rows, 1) *
                beta_image[pair];
          }
          for (std::size_t entry = 0;
               entry < alpha_projection.packed_pair_indices.size();
               ++entry) {
            partial.col(alpha_projection.packed_pair_indices[entry]) +=
                alpha_projection.packed_pair_values[entry] * beta_auxiliary;
          }
          for (const int pair : touched_beta) {
            partial.col(pair) += beta_image[pair] * alpha_auxiliary;
            beta_image[pair] = 0.0;
            touched_mask[pair] = 0u;
          }
          touched_beta.clear();
        }
      }
    }
    for (const auto& partial : partials) {
      factor_adjoint->middleRows(auxiliary_begin, block_rows) += partial;
    }
  }
}

}  // namespace

Eigen::MatrixXd apply_regular_ri_pair_space_adjoint(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_pair_factors) {
  if (!same_spin_pair_cache.enabled() ||
      has_polynomial_same_spin_response_pairs(same_spin_pair_cache)) {
    throw std::invalid_argument(
        "RI factor adjoint requires an all-regular same-spin pair cache");
  }
  if (selected_states.states.size() != selected_state_energies.size() ||
      active_pair_factors.cols() !=
          packed_active_pair_count(n_active_orbitals) ||
      active_pair_factors.rows() <= 0) {
    throw std::invalid_argument("RI factor adjoint dimensions are inconsistent");
  }
  const auto validate_regular_pairs = [](
      const std::vector<std::vector<int>>& determinants,
      const std::vector<SpinDeterminantPairEvaluation>& pairs) {
    const int n_unique = static_cast<int>(determinants.size());
    if (pairs.size() != static_cast<std::size_t>(n_unique) * n_unique) {
      throw std::invalid_argument(
          "RI factor adjoint pair-cache dimensions are inconsistent");
    }
    for (int left = 0; left < n_unique; ++left) {
      for (int right = 0; right < n_unique; ++right) {
        const auto& pair = pairs[ordered_spin_pair_storage_index(
            left, right, n_unique)];
        if (!pair.has_same_spin_phi_cache ||
            pair.overlap_result.inverse_overlap_submatrix.rows() !=
                static_cast<int>(determinants[left].size()) ||
            pair.overlap_result.inverse_overlap_submatrix.cols() !=
                static_cast<int>(determinants[right].size())) {
          throw std::invalid_argument(
              "RI factor adjoint encountered a non-regular same-spin pair");
        }
      }
    }
  };
  validate_regular_pairs(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref());
  if (!same_spin_pair_cache.shares_same_spin_pair_cache_between_spins()) {
    validate_regular_pairs(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref());
  }
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      active_pair_factors.rows(), active_pair_factors.cols());
  accumulate_same_spin_adjoint(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies,
      active_pair_factors,
      &result);
  accumulate_opposite_spin_adjoint(
      same_spin_pair_cache,
      selected_states,
      active_pair_factors,
      &result);
  return result;
}

}  // namespace xmvb::vb
