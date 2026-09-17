#include "vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph_internal.hpp"

namespace xmvb::vb {

using detail::DirectionalOppositeSpinPairData;
using detail::PrimarySpin;
using detail::SelectedStatePairGraph;

namespace {

constexpr double kContributionTolerance = 1.0e-15;

void accumulate_spin_overlap_gradient_direction(
    const std::vector<int>& occ_left,
    const std::vector<int>& occ_right,
    const CofactorDifferential& cofactor,
    const Eigen::MatrixXd& delta_overlap_submatrix,
    const Eigen::MatrixXd& cofactor_weight,
    const Eigen::MatrixXd& delta_cofactor_weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  const Eigen::MatrixXd gradient_direction =
      cofactor.mixed(delta_overlap_submatrix, cofactor_weight.transpose()) +
      cofactor.first(delta_cofactor_weight.transpose());
  for (int left = 0; left < static_cast<int>(occ_left.size()); ++left) {
    for (int right = 0; right < static_cast<int>(occ_right.size()); ++right) {
      (*active_orbital_overlap_gradient)[
          occ_left[left] * n_active_orbitals + occ_right[right]] +=
          gradient_direction(right, left);
    }
  }
}

std::vector<int> build_retained_minor_indices_local(
    int dimension,
    const std::vector<int>& deleted_indices) {
  std::vector<int> retained_indices;
  retained_indices.reserve(
      dimension - static_cast<int>(deleted_indices.size()));
  for (int index = 0; index < dimension; ++index) {
    if (std::find(deleted_indices.begin(), deleted_indices.end(), index) ==
        deleted_indices.end()) {
      retained_indices.push_back(index);
    }
  }
  return retained_indices;
}

void scatter_minor_cofactor_to_overlap_block_gradient_local(
    const Eigen::MatrixXd& minor_cofactor,
    const std::vector<int>& retained_rows,
    const std::vector<int>& retained_cols,
    double scale,
    Eigen::MatrixXd* overlap_block_gradient) {
  if (overlap_block_gradient == nullptr) {
    throw std::invalid_argument("overlap_block_gradient must not be null");
  }
  if (minor_cofactor.rows() != static_cast<int>(retained_rows.size()) ||
      minor_cofactor.cols() != static_cast<int>(retained_cols.size())) {
    throw std::invalid_argument(
        "minor cofactor dimensions do not match retained deleted-minor indices");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  for (int retained_col = 0;
       retained_col < static_cast<int>(retained_cols.size());
       ++retained_col) {
    const int overlap_col = retained_cols[retained_col];
    for (int retained_row = 0;
         retained_row < static_cast<int>(retained_rows.size());
         ++retained_row) {
      const int overlap_row = retained_rows[retained_row];
      (*overlap_block_gradient)(overlap_row, overlap_col) +=
          scale * minor_cofactor(retained_row, retained_col);
    }
  }
}

void accumulate_deleted_minor_pullback_to_overlap_block_gradient_local(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    double scale,
    const DeterminantOverlapResolver& overlap_resolver,
    Eigen::MatrixXd* overlap_block_gradient) {
  if (overlap_block_gradient == nullptr) {
    throw std::invalid_argument("overlap_block_gradient must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  const std::vector<int> retained_rows =
      build_retained_minor_indices_local(overlap_block.rows(), deleted_rows);
  const std::vector<int> retained_cols =
      build_retained_minor_indices_local(overlap_block.cols(), deleted_cols);
  if (retained_rows.empty() || retained_cols.empty()) {
    return;
  }

  const Eigen::MatrixXd minor =
      build_deleted_minor_matrix(overlap_block, deleted_rows, deleted_cols);
  const DeterminantOverlapResult minor_result =
      overlap_resolver.resolve_matrix(minor);
  const Eigen::MatrixXd minor_cofactor =
      calc_cofactor_1st(minor_result);
  if (minor_cofactor.size() == 0) {
    return;
  }

  scatter_minor_cofactor_to_overlap_block_gradient_local(
      minor_cofactor,
      retained_rows,
      retained_cols,
      scale * calc_deleted_minor_sign(
                  overlap_block.rows(),
                  overlap_block.cols(),
                  deleted_rows,
                  deleted_cols),
      overlap_block_gradient);
}

void accumulate_singular_spin_overlap_gradient_from_dense_image_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& overlap_result,
    const Eigen::MatrixXd& pair_dense_image,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (pair_dense_image.rows() != static_cast<int>(occ_R.size()) ||
      pair_dense_image.cols() != static_cast<int>(occ_L.size())) {
    throw std::invalid_argument(
        "pair_dense_image shape does not match the singular same-spin overlap block");
  }
  if (overlap_result.nullity != 1) {
    return;
  }

  const Eigen::MatrixXd overlap_block =
      build_overlap_submatrix_from_result(overlap_result);
  Eigen::MatrixXd overlap_block_gradient =
      Eigen::MatrixXd::Zero(overlap_block.rows(), overlap_block.cols());
  const DeterminantOverlapResolver overlap_resolver;
  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const double coefficient = pair_dense_image(right_row, left_column);
      if (std::abs(coefficient) <= kContributionTolerance) {
        continue;
      }
      accumulate_deleted_minor_pullback_to_overlap_block_gradient_local(
          overlap_block,
          {right_row},
          {left_column},
          coefficient,
          overlap_resolver,
          &overlap_block_gradient);
    }
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_orbital_overlap_gradient)[orbital_index_left *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_block_gradient(right_row, left_column);
    }
  }
}

void accumulate_overlap_gradient_by_pair_graph(
    const std::vector<SpinDeterminantPairEvaluation>& primary_pair_cache,
    const std::vector<std::vector<int>>& unique_primary_determinants,
    int n_unique_primary,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    const SelectedStatePairGraph& pair_graph,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronView& two_electron_view,
    std::vector<double>* active_orbital_overlap_gradient) {
  // For each primary-spin string pair, contract the accepted/directional
  // coefficient rows directly with compatible partner-spin pairs.  This first
  // forms the sparse raw partner projection y, then evaluates only the entries
  // of G y required by the inverse-overlap pullback.  The former P-channel
  // sweep repeated the coefficient contraction once for every active pair.
  const int n_packed_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  std::vector<double> partner_image(n_packed_active_pairs, 0.0);
  std::vector<unsigned char> touched_flags(n_packed_active_pairs, 0u);
  std::vector<int> touched_channels;
  std::vector<double> touched_values;
  touched_channels.reserve(n_packed_active_pairs);
  touched_values.reserve(n_packed_active_pairs);
  std::vector<unsigned char> target_flags(n_packed_active_pairs, 0u);
  std::vector<int> target_channels;
  target_channels.reserve(n_packed_active_pairs);
  std::vector<double> projected_image(n_packed_active_pairs, 0.0);
  Eigen::MatrixXd pair_dense_image;
  Eigen::MatrixXd inverse_overlap_gradient;

  for (int primary_right = 0;
       primary_right < n_unique_primary;
       ++primary_right) {
    for (int primary_left = 0;
         primary_left < n_unique_primary;
         ++primary_left) {
      const auto& pair_evaluation =
          primary_pair_cache[ordered_spin_pair_storage_index(
              primary_left,
              primary_right,
              n_unique_primary)];
      if (pair_evaluation.overlap_result.nullity > 1) {
        continue;
      }
      const auto& inverse_projection =
          pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;

      pair_graph.accumulate_partner_projection(
          primary_left,
          primary_right,
          partner_pair_cache,
          n_unique_partner,
          &partner_image,
          &touched_flags,
          &touched_channels);
      if (touched_channels.empty()) {
        continue;
      }

      touched_values.clear();
      for (const int channel : touched_channels) {
        touched_values.push_back(partner_image[channel]);
      }
      const auto& occ_L = unique_primary_determinants[primary_left];
      const auto& occ_R = unique_primary_determinants[primary_right];
      const auto add_target = [&](int channel) {
        if (target_flags[channel] == 0u) {
          target_flags[channel] = 1u;
          target_channels.push_back(channel);
        }
      };
      for (const int channel : inverse_projection.packed_pair_indices) {
        add_target(channel);
      }
      for (const int orbital_left : occ_L) {
        for (const int orbital_right : occ_R) {
          add_target(TwoElectronIndexer::packed_pair_index(
              orbital_right,
              orbital_left));
        }
      }
      const Eigen::VectorXd projected_targets =
          apply_active_space_two_electron_kernel_to_sparse_projection_subset(
              two_electron_view,
              n_active_orbitals,
              touched_channels,
              touched_values,
              target_channels);

      for (std::size_t target = 0;
           target < target_channels.size();
           ++target) {
        projected_image[target_channels[target]] =
            projected_targets(static_cast<int>(target));
      }

      const int n_electrons = static_cast<int>(occ_L.size());
      if (pair_evaluation.overlap_result.nullity == 1) {
        pair_dense_image.resize(n_electrons, n_electrons);
        for (int left_column = 0; left_column < n_electrons; ++left_column) {
          for (int right_row = 0; right_row < n_electrons; ++right_row) {
            const int channel = TwoElectronIndexer::packed_pair_index(
                occ_R[right_row],
                occ_L[left_column]);
            pair_dense_image(right_row, left_column) =
                projected_image[channel];
          }
        }
        accumulate_singular_spin_overlap_gradient_from_dense_image_local(
            occ_L,
            occ_R,
            pair_evaluation.overlap_result,
            pair_dense_image,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      } else {
        double determinant_overlap_weight = 0.0;
        for (std::size_t entry = 0;
             entry < inverse_projection.packed_pair_indices.size();
             ++entry) {
          const int channel = inverse_projection.packed_pair_indices[entry];
          determinant_overlap_weight +=
              inverse_projection.packed_pair_values[entry] *
              projected_image[channel];
        }
        inverse_overlap_gradient.setZero(n_electrons, n_electrons);
        for (int left_column = 0; left_column < n_electrons; ++left_column) {
          for (int right_row = 0; right_row < n_electrons; ++right_row) {
            const int channel = TwoElectronIndexer::packed_pair_index(
                occ_R[right_row],
                occ_L[left_column]);
            inverse_overlap_gradient(left_column, right_row) =
                projected_image[channel];
          }
        }
        accumulate_spin_overlap_gradient(
            occ_L,
            occ_R,
            pair_evaluation.overlap_result,
            determinant_overlap_weight,
            inverse_overlap_gradient,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }

      for (const int channel : target_channels) {
        projected_image[channel] = 0.0;
        target_flags[channel] = 0u;
      }
      target_channels.clear();
      for (const int channel : touched_channels) {
        partner_image[channel] = 0.0;
        touched_flags[channel] = 0u;
      }
      touched_channels.clear();
    }
  }
}

void accumulate_local_overlap_gradient_by_pair_graph(
    const std::vector<SpinDeterminantPairEvaluation>& primary_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& primary_directional_pairs,
    const std::vector<std::vector<int>>& unique_primary_determinants,
    int n_unique_primary,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& partner_directional_pairs,
    int n_unique_partner,
    const SelectedStatePairGraph& pair_graph,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  std::vector<int> target_channels;
  std::vector<double> accepted_values;
  std::vector<double> directional_values;
  Eigen::MatrixXd cofactor_weight;
  Eigen::MatrixXd delta_cofactor_weight;

  for (int primary_right = 0; primary_right < n_unique_primary; ++primary_right) {
    for (int primary_left = 0; primary_left < n_unique_primary; ++primary_left) {
      const auto& occ_left = unique_primary_determinants[primary_left];
      const auto& occ_right = unique_primary_determinants[primary_right];
      const int n_electrons = static_cast<int>(occ_left.size());
      target_channels.clear();
      target_channels.reserve(
          static_cast<std::size_t>(n_electrons) * n_electrons);
      for (const int orbital_left : occ_left) {
        for (const int orbital_right : occ_right) {
          target_channels.push_back(TwoElectronIndexer::packed_pair_index(
              orbital_right,
              orbital_left));
        }
      }
      accepted_values.assign(target_channels.size(), 0.0);
      directional_values.assign(target_channels.size(), 0.0);
      pair_graph.accumulate_partner_projected_values(
          primary_left,
          primary_right,
          partner_pair_cache,
          n_unique_partner,
          target_channels,
          &accepted_values);
      pair_graph.accumulate_partner_projected_values(
          primary_left,
          primary_right,
          partner_directional_pairs,
          n_unique_partner,
          target_channels,
          &directional_values);

      cofactor_weight.resize(n_electrons, n_electrons);
      delta_cofactor_weight.resize(n_electrons, n_electrons);
      std::size_t target = 0;
      for (int left = 0; left < n_electrons; ++left) {
        for (int right = 0; right < n_electrons; ++right, ++target) {
          cofactor_weight(left, right) = accepted_values[target];
          delta_cofactor_weight(left, right) = directional_values[target];
        }
      }

      const std::size_t pair_index = ordered_spin_pair_storage_index(
          primary_left,
          primary_right,
          n_unique_primary);
      accumulate_spin_overlap_gradient_direction(
          occ_left,
          occ_right,
          cached_cofactor_differential(primary_pair_cache[pair_index]),
          primary_directional_pairs[pair_index].delta_overlap_submatrix,
          cofactor_weight,
          delta_cofactor_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

}  // namespace

namespace detail {

void accumulate_alpha_overlap_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const SelectedStatePairGraph pair_graph(
      selected_states,
      PrimarySpin::Alpha);
  accumulate_overlap_gradient_by_pair_graph(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      selected_states.n_unique_alpha,
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_states.n_unique_beta,
      pair_graph,
      n_active_orbitals,
      make_active_space_two_electron_view(active_space_two_electron_result),
      active_orbital_overlap_gradient);
}

void accumulate_beta_overlap_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const SelectedStatePairGraph pair_graph(
      selected_states,
      PrimarySpin::Beta);
  accumulate_overlap_gradient_by_pair_graph(
      same_spin_pair_cache.beta_pair_cache_ref(),
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      selected_states.n_unique_beta,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      selected_states.n_unique_alpha,
      pair_graph,
      n_active_orbitals,
      make_active_space_two_electron_view(active_space_two_electron_result),
      active_orbital_overlap_gradient);
}

void accumulate_directional_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const SelectedStatePairGraph pair_graph(
      selected_states,
      directional_selected_states,
      PrimarySpin::Alpha);
  accumulate_overlap_gradient_by_pair_graph(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      selected_states.n_unique_alpha,
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_states.n_unique_beta,
      pair_graph,
      n_active_orbitals,
      make_active_space_two_electron_view(active_space_two_electron_result),
      active_orbital_overlap_gradient);
}

void accumulate_directional_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const SelectedStatePairGraph pair_graph(
      selected_states,
      directional_selected_states,
      PrimarySpin::Beta);
  accumulate_overlap_gradient_by_pair_graph(
      same_spin_pair_cache.beta_pair_cache_ref(),
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      selected_states.n_unique_beta,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      selected_states.n_unique_alpha,
      pair_graph,
      n_active_orbitals,
      make_active_space_two_electron_view(active_space_two_electron_result),
      active_orbital_overlap_gradient);
}

void accumulate_local_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const SelectedStatePairGraph pair_graph(
      selected_states,
      PrimarySpin::Alpha);
  accumulate_local_overlap_gradient_by_pair_graph(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      alpha_directional_pair_data,
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      selected_states.n_unique_alpha,
      same_spin_pair_cache.beta_pair_cache_ref(),
      beta_directional_pair_data,
      selected_states.n_unique_beta,
      pair_graph,
      n_active_orbitals,
      active_orbital_overlap_gradient);
}

void accumulate_local_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const SelectedStatePairGraph pair_graph(
      selected_states,
      PrimarySpin::Beta);
  accumulate_local_overlap_gradient_by_pair_graph(
      same_spin_pair_cache.beta_pair_cache_ref(),
      beta_directional_pair_data,
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      selected_states.n_unique_beta,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      alpha_directional_pair_data,
      selected_states.n_unique_alpha,
      pair_graph,
      n_active_orbitals,
      active_orbital_overlap_gradient);
}

}  // namespace detail

}  // namespace xmvb::vb
