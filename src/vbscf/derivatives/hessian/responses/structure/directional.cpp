#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb {
namespace {

constexpr double kProjectionZeroTolerance = 1.0e-15;

void require_finite(const Eigen::MatrixXd& values, const char* label) {
  if (!values.allFinite()) {
    throw std::runtime_error(std::string(label) + " contains non-finite values");
  }
}

Eigen::MatrixXd pair_scalar_matrix(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    bool hamiltonian) {
  if (pair_cache.size() !=
      static_cast<std::size_t>(n_unique) * n_unique) {
    throw std::invalid_argument(
        "same-spin pair cache has inconsistent dimensions");
  }
  Eigen::MatrixXd result(n_unique, n_unique);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& pair = pair_cache[ordered_spin_pair_storage_index(
          left, right, n_unique)];
      result(left, right) = hamiltonian
          ? pair.total_hamiltonian
          : pair.overlap_result.overlap_determinant;
    }
  }
  return result;
}

OppositeSpinPackedPairProjection build_directional_projection(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const Eigen::MatrixXd& delta_cofactor,
    int n_active_orbitals) {
  OppositeSpinPackedPairProjection result;
  if (occupied_left.empty()) return result;
  if (delta_cofactor.rows() !=
          static_cast<Eigen::Index>(occupied_right.size()) ||
      delta_cofactor.cols() !=
          static_cast<Eigen::Index>(occupied_left.size())) {
    throw std::invalid_argument(
        "directional cofactor has inconsistent occupied dimensions");
  }

  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> dense_values(n_pairs, 0.0);
  for (int left_column = 0;
       left_column < static_cast<int>(occupied_left.size());
       ++left_column) {
    for (int right_row = 0;
         right_row < static_cast<int>(occupied_right.size());
         ++right_row) {
      const int packed_pair = TwoElectronIndexer::packed_pair_index(
          occupied_right[right_row],
          occupied_left[left_column]);
      dense_values[packed_pair] +=
          delta_cofactor(right_row, left_column);
    }
  }
  for (int packed_pair = 0; packed_pair < n_pairs; ++packed_pair) {
    if (std::abs(dense_values[packed_pair]) <= kProjectionZeroTolerance) {
      continue;
    }
    result.packed_pair_indices.push_back(packed_pair);
    result.packed_pair_values.push_back(dense_values[packed_pair]);
  }
  return result;
}

struct ChannelEntry {
  int row = 0;
  int column = 0;
  double value = 0.0;
};

/** Channel-major exact nonzeros used to stream one packed-pair channel. */
using SparseChannels = std::vector<std::vector<ChannelEntry>>;

template <typename ProjectionProvider>
SparseChannels index_channels(
    int n_unique,
    int n_pairs,
    ProjectionProvider&& projection_at) {
  SparseChannels channels(n_pairs);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& projection = projection_at(left, right);
      if (projection.packed_pair_indices.size() !=
          projection.packed_pair_values.size()) {
        throw std::invalid_argument(
            "packed-pair projection indices and values differ in size");
      }
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        const int packed_pair = projection.packed_pair_indices[entry];
        if (packed_pair < 0 || packed_pair >= n_pairs) {
          throw std::out_of_range(
              "packed-pair projection index is out of range");
        }
        channels[packed_pair].push_back(ChannelEntry{
            left,
            right,
            projection.packed_pair_values[entry]});
      }
    }
  }
  return channels;
}

SparseChannels accepted_channels(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    int n_pairs) {
  return index_channels(
      n_unique,
      n_pairs,
      [&](int left, int right) -> const OppositeSpinPackedPairProjection& {
        return pair_cache[ordered_spin_pair_storage_index(
            left, right, n_unique)]
            .opposite_spin_pair_cache.first_order_cofactor_projection;
      });
}

SparseChannels directional_channels(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SameSpinPolynomialDirectionalPairData>& directional_pairs,
    int n_active_orbitals) {
  const int n_unique = static_cast<int>(unique_determinants.size());
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  if (directional_pairs.size() !=
      static_cast<std::size_t>(n_unique) * n_unique) {
    throw std::invalid_argument(
        "directional same-spin pair cache has inconsistent dimensions");
  }
  return index_channels(
      n_unique,
      n_pairs,
      [&](int left, int right) {
        const std::size_t index = ordered_spin_pair_storage_index(
            left, right, n_unique);
        return build_directional_projection(
            unique_determinants[left],
            unique_determinants[right],
            directional_pairs[index].delta_cofactor_1st,
            n_active_orbitals);
      });
}

Eigen::MatrixXd dense_channel(
    const SparseChannels& channels,
    int channel,
    int n_unique) {
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_unique, n_unique);
  for (const ChannelEntry& entry : channels[channel]) {
    result(entry.row, entry.column) = entry.value;
  }
  return result;
}

template <typename KernelValue>
void accumulate_projected_channel(
    const SparseChannels& raw_channels,
    int target,
    KernelValue&& kernel_value,
    Eigen::MatrixXd* projected) {
  if (projected == nullptr) {
    throw std::invalid_argument("projected channel output must not be null");
  }
  for (int source = 0;
       source < static_cast<int>(raw_channels.size());
       ++source) {
    const double weight = kernel_value(target, source);
    if (weight == 0.0) {
      continue;
    }
    for (const ChannelEntry& entry : raw_channels[source]) {
      (*projected)(entry.row, entry.column) += weight * entry.value;
    }
  }
}

void scatter_spin_image(
    const FullDeterminantStructureData& structure_data,
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& spin_image,
    int state_offset,
    Eigen::MatrixXd* structure_image) {
  for (int determinant = 0;
       determinant < selected_states.n_determinants;
       ++determinant) {
    const double value = spin_image(
        selected_states.determinant_to_unique_alpha_id[determinant],
        selected_states.determinant_to_unique_beta_id[determinant]);
    if (value == 0.0) continue;
    for (const auto& term :
         structure_data.determinant_to_structure_terms[determinant]) {
      (*structure_image)(term.structure_index, state_offset) +=
          term.coefficient * value;
    }
  }
}

}  // namespace

SelectedStateDirectionalStructureImages
build_selected_structure_direction(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairCache& directional_pair_cache) {
  if (accepted.input == nullptr ||
      accepted.accepted_point_context == nullptr) {
    throw std::invalid_argument(
        "factorized structure direction requires a complete accepted context");
  }
  const auto& input = *accepted.input;
  const auto& accepted_point = *accepted.accepted_point_context;
  const auto& same_spin = accepted_point.same_spin_pair_cache;
  if (!same_spin.enabled()) {
    throw std::invalid_argument(
        "factorized structure direction requires the same-spin cache");
  }

  const int n_structures = input.structure_data.n_structures;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  const int n_alpha = static_cast<int>(
      same_spin.alpha_reuse_table.unique_determinants.size());
  const int n_beta = static_cast<int>(
      same_spin.beta_reuse_table.unique_determinants.size());
  const auto& selected_columns =
      accepted.selected_state_eigen_response_operator.selected_eigenvectors;
  if (selected_columns.rows() != n_structures ||
      selected_columns.cols() !=
          static_cast<int>(accepted_point.selected_state_indices.size())) {
    throw std::invalid_argument(
        "selected structure coefficients have inconsistent dimensions");
  }

  const auto selected_states =
      build_selected_state_determinant_matrices_from_selected_columns(
          input.structure_data,
          selected_columns,
          accepted_point.selected_state_indices,
          accepted_point.normalized_state_weights,
          same_spin);

  const auto& alpha_cache = same_spin.alpha_pair_cache_ref();
  const auto& beta_cache = same_spin.beta_pair_cache_ref();
  const Eigen::MatrixXd alpha_overlap =
      pair_scalar_matrix(alpha_cache, n_alpha, false);
  const Eigen::MatrixXd alpha_hamiltonian =
      pair_scalar_matrix(alpha_cache, n_alpha, true);
  const Eigen::MatrixXd beta_overlap =
      pair_scalar_matrix(beta_cache, n_beta, false);
  const Eigen::MatrixXd beta_hamiltonian =
      pair_scalar_matrix(beta_cache, n_beta, true);
  const Eigen::MatrixXd delta_alpha_overlap =
      directional_pair_cache.alpha.delta_overlap_determinant_matrix;
  const Eigen::MatrixXd delta_alpha_hamiltonian =
      directional_pair_cache.alpha.delta_regular_total_hamiltonian_matrix +
      directional_pair_cache.alpha.delta_singular_total_hamiltonian_matrix;
  const auto& beta_direction = directional_pair_cache.close_shell_same_spin
      ? directional_pair_cache.alpha
      : directional_pair_cache.beta;
  const Eigen::MatrixXd delta_beta_overlap =
      beta_direction.delta_overlap_determinant_matrix;
  const Eigen::MatrixXd delta_beta_hamiltonian =
      beta_direction.delta_regular_total_hamiltonian_matrix +
      beta_direction.delta_singular_total_hamiltonian_matrix;

  const auto alpha_channels = accepted_channels(
      alpha_cache, n_alpha, n_pairs);
  const auto beta_channels = accepted_channels(
      beta_cache, n_beta, n_pairs);
  const auto delta_alpha_channels = directional_channels(
      same_spin.alpha_reuse_table.unique_determinants,
      directional_pair_cache.alpha.ordered_pair_data,
      n_active_orbitals);
  const auto delta_beta_channels = directional_channels(
      same_spin.beta_reuse_table.unique_determinants,
      beta_direction.ordered_pair_data,
      n_active_orbitals);

  const ActiveSpaceTwoElectronView accepted_kernel =
      make_active_space_two_electron_view(
          accepted_point.prepared_active_space.active_space_two_electron_result);

  SelectedStateDirectionalStructureImages result;
  const int n_selected_states =
      static_cast<int>(selected_states.states.size());
  result.delta_hamiltonian_selected =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  result.delta_overlap_selected =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  std::vector<Eigen::MatrixXd> delta_hamiltonian_images;
  std::vector<Eigen::MatrixXd> delta_overlap_images;
  delta_hamiltonian_images.reserve(n_selected_states);
  delta_overlap_images.reserve(n_selected_states);
  for (int state = 0; state < n_selected_states; ++state) {
    const Eigen::MatrixXd& coefficients =
        selected_states.states[state].coefficient_matrix;
    delta_overlap_images.push_back(
        delta_alpha_overlap * coefficients * beta_overlap.transpose());
    delta_overlap_images.back().noalias() +=
        alpha_overlap * coefficients * delta_beta_overlap.transpose();

    delta_hamiltonian_images.push_back(
        delta_alpha_hamiltonian * coefficients * beta_overlap.transpose());
    delta_hamiltonian_images.back().noalias() +=
        alpha_hamiltonian * coefficients * delta_beta_overlap.transpose();
    delta_hamiltonian_images.back().noalias() +=
        delta_alpha_overlap * coefficients * beta_hamiltonian.transpose();
    delta_hamiltonian_images.back().noalias() +=
        alpha_overlap * coefficients * delta_beta_hamiltonian.transpose();
  }

  for (int target = 0; target < n_pairs; ++target) {
    Eigen::MatrixXd alpha_projected =
        Eigen::MatrixXd::Zero(n_alpha, n_alpha);
    accumulate_projected_channel(
        alpha_channels,
        target,
        [&](int target_pair, int source_pair) {
          return lookup_active_space_two_electron_kernel_value(
              accepted_kernel,
              target_pair,
              source_pair,
              n_active_orbitals);
        },
        &alpha_projected);
    Eigen::MatrixXd delta_alpha_projected =
        Eigen::MatrixXd::Zero(n_alpha, n_alpha);
    accumulate_projected_channel(
        delta_alpha_channels,
        target,
        [&](int target_pair, int source_pair) {
          return lookup_active_space_two_electron_kernel_value(
              accepted_kernel,
              target_pair,
              source_pair,
              n_active_orbitals);
        },
        &delta_alpha_projected);
    accumulate_projected_channel(
        alpha_channels,
        target,
        [&](int target_pair, int source_pair) {
          return direction.packed_two_electron[
              TwoElectronIndexer::packed_pair_of_pairs_index(
                  target_pair, source_pair)];
        },
        &delta_alpha_projected);

    const Eigen::MatrixXd beta =
        dense_channel(beta_channels, target, n_beta);
    const Eigen::MatrixXd delta_beta =
        dense_channel(delta_beta_channels, target, n_beta);
    const bool first_term_is_zero =
        delta_alpha_projected.isZero(0.0) || beta.isZero(0.0);
    const bool second_term_is_zero =
        alpha_projected.isZero(0.0) || delta_beta.isZero(0.0);
    for (int state = 0; state < n_selected_states; ++state) {
      const Eigen::MatrixXd& coefficients =
          selected_states.states[state].coefficient_matrix;
      if (!first_term_is_zero) {
        delta_hamiltonian_images[state].noalias() +=
            delta_alpha_projected * coefficients * beta.transpose();
      }
      if (!second_term_is_zero) {
        delta_hamiltonian_images[state].noalias() +=
            alpha_projected * coefficients * delta_beta.transpose();
      }
    }
  }

  for (int state = 0; state < n_selected_states; ++state) {
    scatter_spin_image(
        input.structure_data,
        selected_states,
        delta_hamiltonian_images[state],
        state,
        &result.delta_hamiltonian_selected);
    scatter_spin_image(
        input.structure_data,
        selected_states,
        delta_overlap_images[state],
        state,
        &result.delta_overlap_selected);
  }

  require_finite(
      result.delta_hamiltonian_selected,
      "factorized directional Hamiltonian images");
  require_finite(
      result.delta_overlap_selected,
      "factorized directional overlap images");
  return result;
}

}  // namespace xmvb::vb
