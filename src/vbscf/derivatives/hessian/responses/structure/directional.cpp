#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "core/openmp.hpp"
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

}  // namespace

struct AcceptedStructureResponseFactors {
  SelectedStateDeterminantMatrices selected_states;
  Eigen::MatrixXd alpha_overlap;
  Eigen::MatrixXd alpha_hamiltonian;
  Eigen::MatrixXd beta_overlap;
  Eigen::MatrixXd beta_hamiltonian;
  SparseChannels alpha_channels;
  SparseChannels beta_channels;
  bool beta_channels_reuse_alpha = false;
};

std::shared_ptr<const AcceptedStructureResponseFactors>
build_accepted_structure_response_factors(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point,
    const Eigen::MatrixXd& selected_columns) {
  const auto& same_spin = accepted_point.same_spin_pair_cache;
  if (!same_spin.enabled()) {
    throw std::invalid_argument(
        "accepted structure response requires the same-spin cache");
  }
  const int n_alpha = static_cast<int>(
      same_spin.alpha_reuse_table.unique_determinants.size());
  const int n_beta = static_cast<int>(
      same_spin.beta_reuse_table.unique_determinants.size());
  const int n_pairs = packed_active_pair_count(
      input.orbital_preparation_input.n_active_orbitals);

  auto factors = std::make_shared<AcceptedStructureResponseFactors>();
  factors->selected_states =
      build_selected_state_determinant_matrices_from_selected_columns(
          input.structure_data,
          selected_columns,
          accepted_point.selected_state_indices,
          accepted_point.normalized_state_weights,
          same_spin);
  const auto& alpha_cache = same_spin.alpha_pair_cache_ref();
  const auto& beta_cache = same_spin.beta_pair_cache_ref();
  factors->alpha_overlap = pair_scalar_matrix(alpha_cache, n_alpha, false);
  factors->alpha_hamiltonian = pair_scalar_matrix(alpha_cache, n_alpha, true);
  factors->beta_overlap = pair_scalar_matrix(beta_cache, n_beta, false);
  factors->beta_hamiltonian = pair_scalar_matrix(beta_cache, n_beta, true);
  factors->alpha_channels = accepted_channels(alpha_cache, n_alpha, n_pairs);
  factors->beta_channels_reuse_alpha =
      same_spin.shares_same_spin_pair_cache_between_spins();
  if (!factors->beta_channels_reuse_alpha) {
    factors->beta_channels = accepted_channels(beta_cache, n_beta, n_pairs);
  }
  return factors;
}

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

  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  const int n_alpha = static_cast<int>(
      same_spin.alpha_reuse_table.unique_determinants.size());
  const int n_beta = static_cast<int>(
      same_spin.beta_reuse_table.unique_determinants.size());
  const StructureAction* structure_action = accepted
      .selected_state_eigen_response_operator.structure_action;
  if (structure_action == nullptr) {
    throw std::invalid_argument(
        "factorized structure direction requires the structure action");
  }
  if (structure_action->supports_integral_direction()) {
    if (!accepted_point.structure_adjoint_state.has_value()) {
      accepted_point.structure_adjoint_state =
          structure_action->prepare_active_adjoint(
              accepted_point.selected_state_matrices,
              accepted_point.selected_state_energies);
    }
    StructureIntegralDirection direct_ci_direction;
    const StructureActionResult direct_direction =
        structure_action->apply_integral_direction(
            *accepted_point.structure_adjoint_state,
            direction.overlap,
            direction.one_electron,
            direction.packed_two_electron,
            &direct_ci_direction);
    require_finite(
        direct_direction.hamiltonian,
        "orthogonal direct-CI directional Hamiltonian images");
    require_finite(
        direct_direction.overlap,
        "orthogonal direct-CI directional overlap images");
    return SelectedStateDirectionalStructureImages{
        direct_direction.hamiltonian,
        direct_direction.overlap,
        std::move(direct_ci_direction)};
  }
  if (!same_spin.enabled()) {
    throw std::invalid_argument(
        "factorized structure direction requires the same-spin cache");
  }
  if (accepted.structure_factors == nullptr) {
    throw std::invalid_argument(
        "factorized structure direction requires accepted factors");
  }
  const auto& factors = *accepted.structure_factors;
  const auto& selected_states = factors.selected_states;
  const auto& alpha_overlap = factors.alpha_overlap;
  const auto& alpha_hamiltonian = factors.alpha_hamiltonian;
  const auto& beta_overlap = factors.beta_overlap;
  const auto& beta_hamiltonian = factors.beta_hamiltonian;
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

  const auto& alpha_channels = factors.alpha_channels;
  const auto& beta_channels = factors.beta_channels_reuse_alpha
      ? factors.alpha_channels
      : factors.beta_channels;
  const auto delta_alpha_channels = directional_channels(
      same_spin.alpha_reuse_table.unique_determinants,
      directional_pair_cache.alpha.ordered_pair_data,
      n_active_orbitals);
  SparseChannels delta_beta_storage;
  if (!directional_pair_cache.close_shell_same_spin) {
    delta_beta_storage = directional_channels(
        same_spin.beta_reuse_table.unique_determinants,
        beta_direction.ordered_pair_data,
        n_active_orbitals);
  }
  const auto& delta_beta_channels = directional_pair_cache.close_shell_same_spin
      ? delta_alpha_channels
      : delta_beta_storage;

  const ActiveSpaceTwoElectronView accepted_kernel =
      make_active_space_two_electron_view(
          accepted_point.prepared_active_space.active_space_two_electron_result);

  SelectedStateDirectionalStructureImages result;
  const int n_selected_states =
      static_cast<int>(selected_states.states.size());
  Eigen::MatrixXd delta_hamiltonian_images =
      Eigen::MatrixXd::Zero(n_alpha, n_selected_states * n_beta);
  Eigen::MatrixXd delta_overlap_images =
      Eigen::MatrixXd::Zero(n_alpha, n_selected_states * n_beta);
  for (int state = 0; state < n_selected_states; ++state) {
    const Eigen::MatrixXd& coefficients =
        selected_states.states[state].coefficient_matrix;
    auto delta_overlap_image = delta_overlap_images.middleCols(
        state * n_beta, n_beta);
    delta_overlap_image.noalias() =
        delta_alpha_overlap * coefficients * beta_overlap.transpose();
    delta_overlap_image.noalias() +=
        alpha_overlap * coefficients * delta_beta_overlap.transpose();

    auto delta_hamiltonian_image = delta_hamiltonian_images.middleCols(
        state * n_beta, n_beta);
    delta_hamiltonian_image.noalias() =
        delta_alpha_hamiltonian * coefficients * beta_overlap.transpose();
    delta_hamiltonian_image.noalias() +=
        alpha_hamiltonian * coefficients * delta_beta_overlap.transpose();
    delta_hamiltonian_image.noalias() +=
        delta_alpha_overlap * coefficients * beta_hamiltonian.transpose();
    delta_hamiltonian_image.noalias() +=
        alpha_overlap * coefficients * delta_beta_hamiltonian.transpose();
  }

  const int n_threads = std::min(
      xmvb::effective_openmp_thread_count(),
      n_pairs);
  std::vector<Eigen::MatrixXd> partial_hamiltonian_images;
  partial_hamiltonian_images.reserve(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    partial_hamiltonian_images.emplace_back(Eigen::MatrixXd::Zero(
        n_alpha,
        n_selected_states * n_beta));
  }
#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
#ifdef _OPENMP
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    Eigen::MatrixXd alpha_projected(n_alpha, n_alpha);
    Eigen::MatrixXd delta_alpha_projected(n_alpha, n_alpha);
#pragma omp for schedule(static)
    for (int target = 0; target < n_pairs; ++target) {
      alpha_projected.setZero();
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
      delta_alpha_projected.setZero();
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
        auto delta_hamiltonian_image =
            partial_hamiltonian_images[thread].middleCols(
                state * n_beta,
                n_beta);
        if (!first_term_is_zero) {
          delta_hamiltonian_image.noalias() +=
              delta_alpha_projected * coefficients * beta.transpose();
        }
        if (!second_term_is_zero) {
          delta_hamiltonian_image.noalias() +=
              alpha_projected * coefficients * delta_beta.transpose();
        }
      }
    }
  }
  for (const Eigen::MatrixXd& partial : partial_hamiltonian_images) {
    delta_hamiltonian_images += partial;
  }

  result.delta_hamiltonian_selected =
      structure_action->contract_spin_product_block(
          delta_hamiltonian_images);
  result.delta_overlap_selected =
      structure_action->contract_spin_product_block(delta_overlap_images);

  require_finite(
      result.delta_hamiltonian_selected,
      "factorized directional Hamiltonian images");
  require_finite(
      result.delta_overlap_selected,
      "factorized directional overlap images");
  return result;
}

}  // namespace xmvb::vb
