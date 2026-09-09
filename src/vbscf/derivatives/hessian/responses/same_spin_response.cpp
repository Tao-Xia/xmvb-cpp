#include "vbscf/derivatives/hessian/responses/same_spin_response.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_weight_builder_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/determinants/cofactor_differential.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"

namespace xmvb::vb {

namespace {

using detail::accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local;
using detail::accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local;
using detail::accumulate_directional_one_electron_gradient_contribution_local;
using detail::accumulate_one_electron_gradient_contribution_local;
using detail::accumulate_overlap_block_gradient_contribution_local;
using detail::accumulate_alpha_accepted_tile_weights;
using detail::accumulate_alpha_directional_tile_weights;
using detail::accumulate_alpha_local_tile_weights;
using detail::accumulate_beta_accepted_tile_weights;
using detail::accumulate_beta_directional_tile_weights;
using detail::accumulate_beta_local_tile_weights;
using detail::build_dense_directional_exact_same_spin_weight_matrices;
using detail::build_directional_pair_scalar_matrices;
using detail::build_exact_same_spin_weight_matrices;
using detail::build_local_same_spin_response_weight_matrices;
using detail::build_polynomial_spin_directional_data;
using detail::build_support_sparse_directional_exact_same_spin_weight_matrices;
using detail::SameSpinAcceptedTileWeights;
using detail::SameSpinExactWeightMatrices;
using detail::SameSpinLocalResponseWeightMatrices;
using detail::SameSpinLocalTileWeights;
using detail::validate_directional_selected_state_inputs;
using detail::validate_full_matrix_same_spin_inputs;

constexpr double kContributionTolerance = 1.0e-15;
constexpr int kSameSpinBackwardPairTileSize = 64;
constexpr std::size_t kSameSpinAcceptedTileMinDenseBytes =
    256ull * 1024ull * 1024ull;

std::size_t square_storage_size(int dimension) {
  return (dimension) * (dimension);
}

int same_spin_backward_pair_tile_size() {
  return kSameSpinBackwardPairTileSize;
}

std::size_t same_spin_accepted_tile_min_dense_bytes() {
  return kSameSpinAcceptedTileMinDenseBytes;
}

bool should_use_same_spin_accepted_tile_backward(
    const SelectedStateDeterminantMatrices& selected_states) {
  const std::size_t alpha_size =
      static_cast<std::size_t>(selected_states.n_unique_alpha);
  const std::size_t beta_size =
      static_cast<std::size_t>(selected_states.n_unique_beta);
  const std::size_t dense_weight_bytes =
      4ull * sizeof(double) *
      (alpha_size * alpha_size + beta_size * beta_size);
  return dense_weight_bytes >= same_spin_accepted_tile_min_dense_bytes();
}

bool should_use_same_spin_local_tile_backward(
    const SelectedStateDeterminantMatrices& selected_states) {
  return should_use_same_spin_accepted_tile_backward(selected_states);
}

bool should_use_same_spin_directional_tile_backward(
    const SelectedStateDeterminantMatrices& selected_states) {
  return should_use_same_spin_accepted_tile_backward(selected_states);
}

bool same_spin_local_tile_has_any_weight(
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end) {
  for (int left_id = left_begin; left_id < left_end; ++left_id) {
    for (int right_id = right_begin; right_id < right_end; ++right_id) {
      if (std::abs(hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(partner_total_transfer_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_partner_total_transfer_matrix(left_id, right_id)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

template <typename HWeight, typename SWeight, typename TWeight>
bool same_spin_weight_tile_has_any_weight(
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile) {
  for (int column = 0; column < hamiltonian_weight_tile.cols(); ++column) {
    for (int row = 0; row < hamiltonian_weight_tile.rows(); ++row) {
      if (std::abs(hamiltonian_weight_tile(row, column)) > kContributionTolerance ||
          std::abs(overlap_weight_tile(row, column)) > kContributionTolerance ||
          std::abs(partner_total_transfer_tile(row, column)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

template <
    typename HWeight,
    typename SWeight,
    typename TWeight,
    typename DHWeight,
    typename DSWeight,
    typename DTWeight>
bool same_spin_local_weight_tile_has_any_weight(
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile,
    const DHWeight& delta_hamiltonian_weight_tile,
    const DSWeight& delta_overlap_weight_tile,
    const DTWeight& delta_partner_total_transfer_tile) {
  for (int column = 0; column < hamiltonian_weight_tile.cols(); ++column) {
    for (int row = 0; row < hamiltonian_weight_tile.rows(); ++row) {
      if (std::abs(hamiltonian_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(overlap_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(partner_total_transfer_tile(row, column)) >
              kContributionTolerance ||
          std::abs(delta_hamiltonian_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(delta_overlap_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(delta_partner_total_transfer_tile(row, column)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

template <typename HWeight, typename SWeight, typename TWeight>
void accumulate_spin_matrix_backward_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile,
    int left_begin,
    int right_begin,
    int n_unique_determinants,
    int n_active_orbitals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  for (int left_local = 0;
       left_local < hamiltonian_weight_tile.rows();
       ++left_local) {
    const int left_id = left_begin + left_local;
    for (int right_local = 0;
         right_local < hamiltonian_weight_tile.cols();
         ++right_local) {
      const int right_id = right_begin + right_local;
      const double hamiltonian_weight =
          hamiltonian_weight_tile(left_local, right_local);
      const double overlap_weight =
          overlap_weight_tile(left_local, right_local);
      const double partner_total =
          partner_total_transfer_tile(left_local, right_local);
      if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
          std::abs(overlap_weight) <= kContributionTolerance &&
          std::abs(partner_total) <= kContributionTolerance) {
        continue;
      }

      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      const auto& occ_L = unique_determinants[left_id];
      const auto& occ_R = unique_determinants[right_id];
      const auto& overlap_result = pair_evaluation.overlap_result;
      const bool has_hamiltonian_weight =
          std::abs(hamiltonian_weight) > kContributionTolerance;

      const double determinant_overlap_weight =
          overlap_weight + partner_total;
      const bool need_first_order_cofactor =
          has_hamiltonian_weight ||
          std::abs(determinant_overlap_weight) > kContributionTolerance;
      Eigen::MatrixXd cofactor_1st;
      if (need_first_order_cofactor) {
        cofactor_1st = calc_cofactor_1st(overlap_result);
      }

      if (has_hamiltonian_weight) {
        if (cofactor_1st.size() != 0) {
          accumulate_one_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              cofactor_1st,
              hamiltonian_weight,
              active_one_electron_gradient);
        }
        accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
            occ_L,
            occ_R,
            cached_cofactor_differential(pair_evaluation),
            hamiltonian_weight,
            packed_active_two_electron_gradient);
        if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
                static_cast<int>(occ_R.size()) ||
            pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
                static_cast<int>(occ_L.size())) {
          throw std::runtime_error(
              "matrix-form same-spin backward requires cached overlap Hamiltonian gradients");
        }
        accumulate_overlap_block_gradient_contribution_local(
            occ_L,
            occ_R,
            pair_evaluation.same_spin_overlap_hamiltonian_gradient,
            hamiltonian_weight,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }

      if (std::abs(determinant_overlap_weight) <= kContributionTolerance ||
          cofactor_1st.size() == 0) {
        continue;
      }
      accumulate_overlap_block_gradient_contribution_local(
          occ_L,
          occ_R,
          cofactor_1st,
          determinant_overlap_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

void accumulate_spin_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (active_one_electron_gradient == nullptr ||
      active_orbital_overlap_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("same-spin matrix backward output buffers must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);
      // Tile the ordered unique-pair sweep so sparse/support-aware same-spin
      // weights can skip large zero regions without touching every cached pair.
      const auto hamiltonian_weight_tile =
          hamiltonian_weight_matrix.block(
              left_begin,
              right_begin,
              left_end - left_begin,
              right_end - right_begin);
      const auto overlap_weight_tile =
          overlap_weight_matrix.block(
              left_begin,
              right_begin,
              left_end - left_begin,
              right_end - right_begin);
      const auto partner_total_transfer_tile =
          partner_total_transfer_matrix.block(
              left_begin,
              right_begin,
              left_end - left_begin,
              right_end - right_begin);
      if (!same_spin_weight_tile_has_any_weight(
              hamiltonian_weight_tile,
              overlap_weight_tile,
              partner_total_transfer_tile)) {
        continue;
      }
      accumulate_spin_matrix_backward_tile(
          unique_determinants,
          ordered_pair_cache,
          hamiltonian_weight_tile,
          overlap_weight_tile,
          partner_total_transfer_tile,
          left_begin,
          right_begin,
          n_unique_determinants,
          n_active_orbitals,
          active_one_electron_gradient,
          active_orbital_overlap_gradient,
          packed_active_two_electron_gradient);
    }
  }
}

template <
    typename HWeight,
    typename SWeight,
    typename TWeight,
    typename DHWeight,
    typename DSWeight,
    typename DTWeight>
void accumulate_spin_local_matrix_backward_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const std::vector<SameSpinPolynomialDirectionalPairData>& ordered_directional_data,
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile,
    const DHWeight& delta_hamiltonian_weight_tile,
    const DSWeight& delta_overlap_weight_tile,
    const DTWeight& delta_partner_total_transfer_tile,
    int left_begin,
    int right_begin,
    int n_unique_determinants,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Pair-local directional data still depends on the target ordered pair, but
  // all six scalar adjoints are tile-local. Reuse matrix workspaces across the
  // ordered-pair sweep to avoid allocating temporary inverse-gradient matrices
  // inside the hot loop.
  for (int left_local = 0;
       left_local < hamiltonian_weight_tile.rows();
       ++left_local) {
    const int left_id = left_begin + left_local;
    for (int right_local = 0;
         right_local < hamiltonian_weight_tile.cols();
         ++right_local) {
      const int right_id = right_begin + right_local;
      const double hamiltonian_weight =
          hamiltonian_weight_tile(left_local, right_local);
      const double overlap_weight =
          overlap_weight_tile(left_local, right_local);
      const double partner_total =
          partner_total_transfer_tile(left_local, right_local);
      const double delta_hamiltonian_weight =
          delta_hamiltonian_weight_tile(left_local, right_local);
      const double delta_overlap_weight =
          delta_overlap_weight_tile(left_local, right_local);
      const double delta_partner_total =
          delta_partner_total_transfer_tile(left_local, right_local);
      if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
          std::abs(overlap_weight) <= kContributionTolerance &&
          std::abs(partner_total) <= kContributionTolerance &&
          std::abs(delta_hamiltonian_weight) <= kContributionTolerance &&
          std::abs(delta_overlap_weight) <= kContributionTolerance &&
          std::abs(delta_partner_total) <= kContributionTolerance) {
        continue;
      }

      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      const auto& occ_L = unique_determinants[left_id];
      const auto& occ_R = unique_determinants[right_id];
      const auto& overlap_result = pair_evaluation.overlap_result;

      const SameSpinPolynomialDirectionalPairData& directional_data =
          ordered_directional_data[ordered_spin_pair_storage_index(
              left_id, right_id, n_unique_determinants)];

      accumulate_directional_one_electron_gradient_contribution_local(
          occ_L,
          occ_R,
          directional_data.cofactor_1st,
          directional_data.delta_cofactor_1st,
          hamiltonian_weight,
          delta_hamiltonian_weight,
          active_one_electron_gradient);
      accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
          occ_L,
          occ_R,
          cached_cofactor_differential(pair_evaluation),
          delta_ao_overlap_matrix,
          n_active_orbitals,
          hamiltonian_weight,
          delta_hamiltonian_weight,
          packed_active_two_electron_gradient);

      if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
              static_cast<int>(occ_R.size()) ||
          pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
              static_cast<int>(occ_L.size())) {
        throw std::runtime_error(
            "same-spin local-response requires cached overlap Hamiltonian gradients");
      }
      accumulate_overlap_block_gradient_contribution_local(
          occ_L,
          occ_R,
          pair_evaluation.same_spin_overlap_hamiltonian_gradient,
          delta_hamiltonian_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);
      accumulate_overlap_block_gradient_contribution_local(
          occ_L,
          occ_R,
          directional_data.delta_same_spin_overlap_hamiltonian_gradient,
          hamiltonian_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);

      const double determinant_overlap_weight =
          overlap_weight + partner_total;
      const double delta_determinant_overlap_weight =
          delta_overlap_weight + delta_partner_total;
      if (directional_data.cofactor_1st.size() != 0) {
        accumulate_overlap_block_gradient_contribution_local(
            occ_L,
            occ_R,
            directional_data.cofactor_1st,
            delta_determinant_overlap_weight,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }
      if (directional_data.delta_cofactor_1st.size() != 0 &&
          std::abs(determinant_overlap_weight) > kContributionTolerance) {
        accumulate_overlap_block_gradient_contribution_local(
            occ_L,
            occ_R,
            directional_data.delta_cofactor_1st,
            determinant_overlap_weight,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }
    }
  }
}

void accumulate_spin_local_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const std::vector<SameSpinPolynomialDirectionalPairData>& ordered_directional_data,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // The compressed same-spin local weights are symmetric on the unique-spin
  // spaces, but the local-response kernels still depend on the ordered
  // left/right role of each unique-determinant pair through the directional
  // cofactors and overlap-gradient adjoints. Reusing the accepted-state
  // half-pair shortcut here corrupts the SSO/HHO local response. Sweep the
  // full ordered pair grid just like the validated accepted/directional
  // same-spin backward path.
  if (active_one_electron_gradient == nullptr ||
      active_orbital_overlap_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument(
        "same-spin local matrix backward output buffers must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);
      if (!same_spin_local_tile_has_any_weight(
              hamiltonian_weight_matrix,
              overlap_weight_matrix,
              partner_total_transfer_matrix,
              delta_hamiltonian_weight_matrix,
              delta_overlap_weight_matrix,
              delta_partner_total_transfer_matrix,
              left_begin,
              left_end,
              right_begin,
              right_end)) {
        continue;
      }

      for (int left_id = left_begin; left_id < left_end; ++left_id) {
        for (int right_id = right_begin; right_id < right_end; ++right_id) {
          const double hamiltonian_weight =
              hamiltonian_weight_matrix(left_id, right_id);
          const double overlap_weight =
              overlap_weight_matrix(left_id, right_id);
          const double partner_total =
              partner_total_transfer_matrix(left_id, right_id);
          const double delta_hamiltonian_weight =
              delta_hamiltonian_weight_matrix(left_id, right_id);
          const double delta_overlap_weight =
              delta_overlap_weight_matrix(left_id, right_id);
          const double delta_partner_total =
              delta_partner_total_transfer_matrix(left_id, right_id);
          if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
              std::abs(overlap_weight) <= kContributionTolerance &&
              std::abs(partner_total) <= kContributionTolerance &&
              std::abs(delta_hamiltonian_weight) <= kContributionTolerance &&
              std::abs(delta_overlap_weight) <= kContributionTolerance &&
              std::abs(delta_partner_total) <= kContributionTolerance) {
            continue;
          }

          const auto& pair_evaluation =
              ordered_pair_cache[ordered_spin_pair_storage_index(
                  left_id,
                  right_id,
                  n_unique_determinants)];
          const auto& occ_L = unique_determinants[left_id];
          const auto& occ_R = unique_determinants[right_id];
          const auto& overlap_result = pair_evaluation.overlap_result;

          const SameSpinPolynomialDirectionalPairData& directional_data =
              ordered_directional_data[ordered_spin_pair_storage_index(
                  left_id, right_id, n_unique_determinants)];

          accumulate_directional_one_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              directional_data.cofactor_1st,
              directional_data.delta_cofactor_1st,
              hamiltonian_weight,
              delta_hamiltonian_weight,
              active_one_electron_gradient);
          accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              cached_cofactor_differential(pair_evaluation),
              delta_ao_overlap_matrix,
              n_active_orbitals,
              hamiltonian_weight,
              delta_hamiltonian_weight,
              packed_active_two_electron_gradient);

          if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
                  static_cast<int>(occ_R.size()) ||
              pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
                  static_cast<int>(occ_L.size())) {
            throw std::runtime_error(
                "same-spin local-response requires cached overlap Hamiltonian gradients");
          }
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              pair_evaluation.same_spin_overlap_hamiltonian_gradient,
              delta_hamiltonian_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              directional_data.delta_same_spin_overlap_hamiltonian_gradient,
              hamiltonian_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);

          const double determinant_overlap_weight =
              overlap_weight + partner_total;
          const double delta_determinant_overlap_weight =
              delta_overlap_weight + delta_partner_total;
          if (directional_data.cofactor_1st.size() != 0) {
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.cofactor_1st,
                delta_determinant_overlap_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
          if (directional_data.delta_cofactor_1st.size() != 0 &&
              std::abs(determinant_overlap_weight) > kContributionTolerance) {
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.delta_cofactor_1st,
                determinant_overlap_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

SameSpinMatrixBackwardContribution
build_support_sparse_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals) {
  // Support-sparse accepted backward in true tile form:
  //   build W_H/W_S/W_partner only for the current unique-spin tile,
  //   consume that tile immediately in pair-cache backward,
  //   then discard the tile workspace.
  //
  // This removes the persistent O(N_unique^2) same-spin weight matrices from
  // the accepted-point support-sparse path while keeping the exact pair-local
  // gradient formulas unchanged.
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int tile_size = same_spin_backward_pair_tile_size();
  SameSpinAcceptedTileWeights tile_weights;

  const int alpha_tile_size =
      std::min(selected_states.n_unique_alpha, tile_size);
  for (int left_begin = 0;
       left_begin < selected_states.n_unique_alpha;
       left_begin += alpha_tile_size) {
    const int left_end =
        std::min(selected_states.n_unique_alpha, left_begin + alpha_tile_size);
    for (int right_begin = 0;
         right_begin < selected_states.n_unique_alpha;
         right_begin += alpha_tile_size) {
      const int right_end =
          std::min(selected_states.n_unique_alpha, right_begin + alpha_tile_size);
      accumulate_alpha_accepted_tile_weights(
          selected_states,
          selected_state_energies,
          close_shell_same_spin
              ? same_spin_pair_cache.alpha_pair_cache_ref()
              : same_spin_pair_cache.beta_pair_cache_ref(),
          close_shell_same_spin
              ? selected_states.n_unique_alpha
              : selected_states.n_unique_beta,
          left_begin,
          left_end,
          right_begin,
          right_end,
          &tile_weights);
      if (!same_spin_weight_tile_has_any_weight(
              tile_weights.hamiltonian,
              tile_weights.overlap,
              tile_weights.partner_total)) {
        continue;
      }
      accumulate_spin_matrix_backward_tile(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          tile_weights.hamiltonian,
          tile_weights.overlap,
          tile_weights.partner_total,
          left_begin,
          right_begin,
          selected_states.n_unique_alpha,
          n_active_orbitals,
          &active_one_electron_gradient,
          &result.active_orbital_overlap_gradient,
          &result.packed_active_two_electron_gradient);
    }
  }

  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const int beta_tile_size =
        std::min(selected_states.n_unique_beta, tile_size);
    for (int left_begin = 0;
         left_begin < selected_states.n_unique_beta;
         left_begin += beta_tile_size) {
      const int left_end =
          std::min(selected_states.n_unique_beta, left_begin + beta_tile_size);
      for (int right_begin = 0;
           right_begin < selected_states.n_unique_beta;
           right_begin += beta_tile_size) {
        const int right_end =
            std::min(selected_states.n_unique_beta, right_begin + beta_tile_size);
        accumulate_beta_accepted_tile_weights(
            selected_states,
            selected_state_energies,
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            left_begin,
            left_end,
            right_begin,
            right_end,
            &tile_weights);
        if (!same_spin_weight_tile_has_any_weight(
                tile_weights.hamiltonian,
                tile_weights.overlap,
                tile_weights.partner_total)) {
          continue;
        }
        accumulate_spin_matrix_backward_tile(
            same_spin_pair_cache.beta_reuse_table.unique_determinants,
            same_spin_pair_cache.beta_pair_cache_ref(),
            tile_weights.hamiltonian,
            tile_weights.overlap,
            tile_weights.partner_total,
            left_begin,
            right_begin,
            selected_states.n_unique_beta,
            n_active_orbitals,
            &active_one_electron_gradient,
            &result.active_orbital_overlap_gradient,
            &result.packed_active_two_electron_gradient);
      }
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_support_sparse_directional_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    int n_active_orbitals) {
  // Directional selected-state HVP in true tile form:
  //   build only dW_H/dW_S/dW_partner for the current unique-spin tile,
  //   consume that tile immediately in the same-spin pair-cache backward,
  //   and discard it. The product rule terms use support-local mixed
  //   contractions, so no union-support global weight matrix is allocated.
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int tile_size = same_spin_backward_pair_tile_size();
  SameSpinAcceptedTileWeights tile_weights;

  const int alpha_tile_size =
      std::min(selected_states.n_unique_alpha, tile_size);
  for (int left_begin = 0;
       left_begin < selected_states.n_unique_alpha;
       left_begin += alpha_tile_size) {
    const int left_end =
        std::min(selected_states.n_unique_alpha, left_begin + alpha_tile_size);
    for (int right_begin = 0;
         right_begin < selected_states.n_unique_alpha;
         right_begin += alpha_tile_size) {
      const int right_end =
          std::min(selected_states.n_unique_alpha, right_begin + alpha_tile_size);
      accumulate_alpha_directional_tile_weights(
          selected_states,
          directional_selected_states,
          selected_state_energies,
          directional_selected_state_energies,
          close_shell_same_spin
              ? same_spin_pair_cache.alpha_pair_cache_ref()
              : same_spin_pair_cache.beta_pair_cache_ref(),
          close_shell_same_spin
              ? selected_states.n_unique_alpha
              : selected_states.n_unique_beta,
          left_begin,
          left_end,
          right_begin,
          right_end,
          &tile_weights);
      if (!same_spin_weight_tile_has_any_weight(
              tile_weights.hamiltonian,
              tile_weights.overlap,
              tile_weights.partner_total)) {
        continue;
      }
      accumulate_spin_matrix_backward_tile(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          tile_weights.hamiltonian,
          tile_weights.overlap,
          tile_weights.partner_total,
          left_begin,
          right_begin,
          selected_states.n_unique_alpha,
          n_active_orbitals,
          &active_one_electron_gradient,
          &result.active_orbital_overlap_gradient,
          &result.packed_active_two_electron_gradient);
    }
  }

  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const int beta_tile_size =
        std::min(selected_states.n_unique_beta, tile_size);
    for (int left_begin = 0;
         left_begin < selected_states.n_unique_beta;
         left_begin += beta_tile_size) {
      const int left_end =
          std::min(selected_states.n_unique_beta, left_begin + beta_tile_size);
      for (int right_begin = 0;
           right_begin < selected_states.n_unique_beta;
           right_begin += beta_tile_size) {
        const int right_end =
            std::min(selected_states.n_unique_beta, right_begin + beta_tile_size);
        accumulate_beta_directional_tile_weights(
            selected_states,
            directional_selected_states,
            selected_state_energies,
            directional_selected_state_energies,
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            left_begin,
            left_end,
            right_begin,
            right_end,
            &tile_weights);
        if (!same_spin_weight_tile_has_any_weight(
                tile_weights.hamiltonian,
                tile_weights.overlap,
                tile_weights.partner_total)) {
          continue;
        }
        accumulate_spin_matrix_backward_tile(
            same_spin_pair_cache.beta_reuse_table.unique_determinants,
            same_spin_pair_cache.beta_pair_cache_ref(),
            tile_weights.hamiltonian,
            tile_weights.overlap,
            tile_weights.partner_total,
            left_begin,
            right_begin,
            selected_states.n_unique_beta,
            n_active_orbitals,
            &active_one_electron_gradient,
            &result.active_orbital_overlap_gradient,
            &result.packed_active_two_electron_gradient);
      }
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_support_sparse_local_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache* precomputed_directional_pair_cache) {
  // Local HVP tile path:
  //   keep only partner directional scalar matrices,
  //   build accepted/directional W tiles on demand from support-local C blocks,
  //   immediately consume those tiles in pair-local directional backward.
  //
  // This removes the global accepted W and local-response dW matrices from the
  // support-sparse local HVP path. Tile workspaces are declared once and reset
  // per tile rather than reallocated inside the pair loops.
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  SameSpinDirectionalPairCache owned_directional_pair_cache;
  if (precomputed_directional_pair_cache == nullptr) {
    owned_directional_pair_cache = build_same_spin_directional_pair_cache(
        same_spin_pair_cache,
        n_active_orbitals,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals);
    precomputed_directional_pair_cache = &owned_directional_pair_cache;
  }
  if (precomputed_directional_pair_cache->close_shell_same_spin !=
      close_shell_same_spin) {
    throw std::invalid_argument(
        "precomputed same-spin directional cache has inconsistent spin sharing");
  }
  const auto& alpha_directional_scalars =
      precomputed_directional_pair_cache->alpha;
  const auto& beta_directional_scalars = close_shell_same_spin
      ? precomputed_directional_pair_cache->alpha
      : precomputed_directional_pair_cache->beta;

  const int tile_size = same_spin_backward_pair_tile_size();
  SameSpinLocalTileWeights tile_weights;
  const int alpha_tile_size =
      std::min(selected_states.n_unique_alpha, tile_size);
  for (int left_begin = 0;
       left_begin < selected_states.n_unique_alpha;
       left_begin += alpha_tile_size) {
    const int left_end =
        std::min(selected_states.n_unique_alpha, left_begin + alpha_tile_size);
    for (int right_begin = 0;
         right_begin < selected_states.n_unique_alpha;
         right_begin += alpha_tile_size) {
      const int right_end =
          std::min(selected_states.n_unique_alpha, right_begin + alpha_tile_size);
      accumulate_alpha_local_tile_weights(
          selected_states,
          selected_state_energies,
          close_shell_same_spin
              ? same_spin_pair_cache.alpha_pair_cache_ref()
              : same_spin_pair_cache.beta_pair_cache_ref(),
          close_shell_same_spin
              ? alpha_directional_scalars
              : beta_directional_scalars,
          close_shell_same_spin
              ? selected_states.n_unique_alpha
              : selected_states.n_unique_beta,
          left_begin,
          left_end,
          right_begin,
          right_end,
          &tile_weights);
      if (!same_spin_local_weight_tile_has_any_weight(
              tile_weights.hamiltonian,
              tile_weights.overlap,
              tile_weights.partner_total,
              tile_weights.delta_hamiltonian,
              tile_weights.delta_overlap,
              tile_weights.delta_partner_total)) {
        continue;
      }
      accumulate_spin_local_matrix_backward_tile(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          alpha_directional_scalars.ordered_pair_data,
          tile_weights.hamiltonian,
          tile_weights.overlap,
          tile_weights.partner_total,
          tile_weights.delta_hamiltonian,
          tile_weights.delta_overlap,
          tile_weights.delta_partner_total,
          left_begin,
          right_begin,
          selected_states.n_unique_alpha,
          n_active_orbitals,
          active_one_electron_matrix,
          active_space_two_electron_result,
          delta_ao_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals,
          &active_one_electron_gradient,
          &result.active_orbital_overlap_gradient,
          &result.packed_active_two_electron_gradient);
    }
  }

  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const int beta_tile_size =
        std::min(selected_states.n_unique_beta, tile_size);
    for (int left_begin = 0;
         left_begin < selected_states.n_unique_beta;
         left_begin += beta_tile_size) {
      const int left_end =
          std::min(selected_states.n_unique_beta, left_begin + beta_tile_size);
      for (int right_begin = 0;
           right_begin < selected_states.n_unique_beta;
           right_begin += beta_tile_size) {
        const int right_end =
            std::min(selected_states.n_unique_beta, right_begin + beta_tile_size);
        accumulate_beta_local_tile_weights(
            selected_states,
            selected_state_energies,
            same_spin_pair_cache.alpha_pair_cache_ref(),
            alpha_directional_scalars,
            selected_states.n_unique_alpha,
            left_begin,
            left_end,
            right_begin,
            right_end,
            &tile_weights);
        if (!same_spin_local_weight_tile_has_any_weight(
                tile_weights.hamiltonian,
                tile_weights.overlap,
                tile_weights.partner_total,
                tile_weights.delta_hamiltonian,
                tile_weights.delta_overlap,
                tile_weights.delta_partner_total)) {
          continue;
        }
        accumulate_spin_local_matrix_backward_tile(
            same_spin_pair_cache.beta_reuse_table.unique_determinants,
            same_spin_pair_cache.beta_pair_cache_ref(),
            beta_directional_scalars.ordered_pair_data,
            tile_weights.hamiltonian,
            tile_weights.overlap,
            tile_weights.partner_total,
            tile_weights.delta_hamiltonian,
            tile_weights.delta_overlap,
            tile_weights.delta_partner_total,
            left_begin,
            right_begin,
            selected_states.n_unique_beta,
            n_active_orbitals,
            active_one_electron_matrix,
            active_space_two_electron_result,
            delta_ao_overlap_matrix,
            delta_active_one_electron_matrix,
            delta_packed_active_two_electron_integrals,
            &active_one_electron_gradient,
            &result.active_orbital_overlap_gradient,
            &result.packed_active_two_electron_gradient);
      }
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

}  // namespace

SameSpinDirectionalPairCache build_same_spin_directional_pair_cache(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    int n_active_orbitals,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  SameSpinDirectionalPairCache result;
  result.close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  result.alpha = build_directional_pair_scalar_matrices(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()),
      n_active_orbitals,
      delta_ao_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals);
  if (!result.close_shell_same_spin) {
    result.beta = build_directional_pair_scalar_matrices(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()),
        n_active_orbitals,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals);
  }
  return result;
}

SameSpinMatrixBackwardContribution build_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states) &&
      should_use_same_spin_accepted_tile_backward(selected_states)) {
    return build_support_sparse_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies,
        n_active_orbitals);
  }

  // Compress the selected-state determinant coefficients directly onto the
  // unique alpha/beta same-spin channels. This keeps the backward algebra exact
  // while replacing the previous full determinant-pair scatter with dense
  // matrix products on the unique-spin spaces.
  const SameSpinExactWeightMatrices exact_weight_matrices =
      build_exact_same_spin_weight_matrices(
          same_spin_pair_cache,
          selected_states,
          selected_state_energies);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      exact_weight_matrices.alpha_partner_total_transfer_matrix +
      exact_weight_matrices.alpha_singular_partner_transfer_matrix;

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      exact_weight_matrices.alpha_hamiltonian_weight_matrix,
      exact_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const Eigen::MatrixXd beta_total_partner_transfer_matrix =
        exact_weight_matrices.beta_partner_total_transfer_matrix +
        exact_weight_matrices.beta_singular_partner_transfer_matrix;
    accumulate_spin_matrix_backward(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        exact_weight_matrices.beta_hamiltonian_weight_matrix,
        exact_weight_matrices.beta_overlap_weight_matrix,
        beta_total_partner_transfer_matrix,
        selected_states.n_unique_beta,
        n_active_orbitals,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_directional_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states) &&
      should_use_same_spin_directional_tile_backward(selected_states)) {
    return build_support_sparse_directional_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        directional_selected_states,
        selected_state_energies,
        directional_selected_state_energies,
        n_active_orbitals);
  }

  const SameSpinExactWeightMatrices directional_weight_matrices =
      should_use_support_sparse_selected_state_contractions(selected_states)
          ? build_support_sparse_directional_exact_same_spin_weight_matrices(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                selected_state_energies,
                directional_selected_state_energies)
          : build_dense_directional_exact_same_spin_weight_matrices(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                selected_state_energies,
                directional_selected_state_energies);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      directional_weight_matrices.alpha_partner_total_transfer_matrix +
      directional_weight_matrices.alpha_singular_partner_transfer_matrix;

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      directional_weight_matrices.alpha_hamiltonian_weight_matrix,
      directional_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const Eigen::MatrixXd beta_total_partner_transfer_matrix =
        directional_weight_matrices.beta_partner_total_transfer_matrix +
        directional_weight_matrices.beta_singular_partner_transfer_matrix;
    accumulate_spin_matrix_backward(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        directional_weight_matrices.beta_hamiltonian_weight_matrix,
        directional_weight_matrices.beta_overlap_weight_matrix,
        beta_total_partner_transfer_matrix,
        selected_states.n_unique_beta,
        n_active_orbitals,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_local_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache* precomputed_directional_pair_cache) {
  // Matrix-form local same-spin HVP:
  // 1. compress accepted determinant/structure adjoints onto unique-spin pair weights,
  // 2. compress partner directional scalars onto `δW`,
  // 3. sweep only the ordered unique-spin pairs to accumulate the exact local
  //    response on `(δS_act, δh_act, δg_act)`.
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states) &&
      should_use_same_spin_local_tile_backward(selected_states)) {
    return build_support_sparse_local_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        precomputed_directional_pair_cache);
  }

  const SameSpinExactWeightMatrices exact_weight_matrices =
      build_exact_same_spin_weight_matrices(
          same_spin_pair_cache,
          selected_states,
          selected_state_energies);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      exact_weight_matrices.alpha_partner_total_transfer_matrix +
      exact_weight_matrices.alpha_singular_partner_transfer_matrix;

  SameSpinDirectionalPairCache owned_directional_pair_cache;
  if (precomputed_directional_pair_cache == nullptr) {
    owned_directional_pair_cache = build_same_spin_directional_pair_cache(
        same_spin_pair_cache,
        n_active_orbitals,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals);
    precomputed_directional_pair_cache = &owned_directional_pair_cache;
  }
  if (precomputed_directional_pair_cache->close_shell_same_spin !=
      close_shell_same_spin) {
    throw std::invalid_argument(
        "precomputed same-spin directional cache has inconsistent spin sharing");
  }
  const auto& alpha_directional_scalars =
      precomputed_directional_pair_cache->alpha;
  const auto& beta_directional_scalars = close_shell_same_spin
      ? precomputed_directional_pair_cache->alpha
      : precomputed_directional_pair_cache->beta;
  const SameSpinLocalResponseWeightMatrices local_weight_matrices =
      build_local_same_spin_response_weight_matrices(
          selected_states,
          selected_state_energies,
          alpha_directional_scalars,
          close_shell_same_spin
              ? alpha_directional_scalars
              : beta_directional_scalars,
          close_shell_same_spin);

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_local_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      alpha_directional_scalars.ordered_pair_data,
      exact_weight_matrices.alpha_hamiltonian_weight_matrix,
      exact_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      local_weight_matrices.alpha_delta_hamiltonian_weight_matrix,
      local_weight_matrices.alpha_delta_overlap_weight_matrix,
      local_weight_matrices.alpha_delta_partner_total_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      active_one_electron_matrix,
      active_space_two_electron_result,
      delta_ao_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const Eigen::MatrixXd beta_total_partner_transfer_matrix =
        exact_weight_matrices.beta_partner_total_transfer_matrix +
        exact_weight_matrices.beta_singular_partner_transfer_matrix;
    accumulate_spin_local_matrix_backward(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        beta_directional_scalars.ordered_pair_data,
        exact_weight_matrices.beta_hamiltonian_weight_matrix,
        exact_weight_matrices.beta_overlap_weight_matrix,
        beta_total_partner_transfer_matrix,
        local_weight_matrices.beta_delta_hamiltonian_weight_matrix,
        local_weight_matrices.beta_delta_overlap_weight_matrix,
        local_weight_matrices.beta_delta_partner_total_transfer_matrix,
        selected_states.n_unique_beta,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

}  // namespace xmvb::vb
