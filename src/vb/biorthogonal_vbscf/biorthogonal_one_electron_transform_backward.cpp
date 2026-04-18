#include "vb/biorthogonal_vbscf/biorthogonal_one_electron_transform_backward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb::biorthogonal_vbscf {

namespace {


constexpr double kContributionTolerance = 1.0e-15;

struct SpinExcitation {
  std::vector<int> removed_orbitals;
  std::vector<int> added_orbitals;
  int rank = 0;
};

struct OneElectronSpinWeightMatrices {
  Eigen::MatrixXd alpha_weight_matrix;
  Eigen::MatrixXd beta_weight_matrix;
};

void validate_local_matrix_shape(
    const Eigen::MatrixXd& matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    const char* label) {
  if (matrix.rows() != static_cast<int>(alpha_support.size()) ||
      matrix.cols() != static_cast<int>(beta_support.size())) {
    throw std::invalid_argument(
        std::string(label) + " shape does not match support dimensions");
  }
}

void scatter_add_dense_submatrix(
    const Eigen::MatrixXd& local_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    double scale,
    Eigen::MatrixXd* global_matrix) {
  if (global_matrix == nullptr) {
    throw std::invalid_argument("global_matrix must not be null");
  }
  if (local_matrix.rows() != static_cast<int>(row_indices.size()) ||
      local_matrix.cols() != static_cast<int>(column_indices.size())) {
    throw std::invalid_argument(
        "local_matrix shape does not match scatter support dimensions");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      (*global_matrix)(row_global, column_global) +=
          scale * local_matrix(row_local, column_local);
    }
  }
}

OneElectronSpinWeightMatrices build_one_electron_spin_weight_matrices(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "one-electron transform backward requires an enabled same-spin cache");
  }
  if (selected_state_matrices.n_unique_alpha != same_spin_pair_cache.n_unique_alpha ||
      selected_state_matrices.n_unique_beta != same_spin_pair_cache.n_unique_beta) {
    throw std::invalid_argument(
        "selected_state_matrices dimensions do not match same-spin cache dimensions");
  }
  if (selected_state_matrices.states.empty()) {
    throw std::invalid_argument("selected_state_matrices.states must not be empty");
  }

  OneElectronSpinWeightMatrices weights;
  weights.alpha_weight_matrix =
      Eigen::MatrixXd::Zero(selected_state_matrices.n_unique_alpha,
                        selected_state_matrices.n_unique_alpha);
  weights.beta_weight_matrix =
      Eigen::MatrixXd::Zero(selected_state_matrices.n_unique_beta,
                        selected_state_matrices.n_unique_beta);

  // The exact selected-state Hamiltonian adjoint is
  //   W_H(J, I) = sum_n w_n l_J^(n) r_I^(n).
  // Summing over the untouched partner spin immediately compresses the full
  // determinant-pair weights onto ordered unique same-spin pairs:
  //   W_alpha(aL, aR) = sum_n w_n [L^(n) R^(n)^T]_{aL, aR},
  //   W_beta (bL, bR) = sum_n w_n [L^(n)^T R^(n)]_{bL, bR}.
  for (const auto& state : selected_state_matrices.states) {
    const double state_weight = state.normalized_state_weight;
    if (std::abs(state_weight) <= kContributionTolerance) {
      continue;
    }
    validate_local_matrix_shape(
        state.local_left_coefficient_matrix,
        state.alpha_support,
        state.beta_support,
        "local_left_coefficient_matrix");
    validate_local_matrix_shape(
        state.local_right_coefficient_matrix,
        state.alpha_support,
        state.beta_support,
        "local_right_coefficient_matrix");
    if (state.alpha_support.empty() || state.beta_support.empty()) {
      continue;
    }

    Eigen::MatrixXd alpha_local_image =
        state.local_left_coefficient_matrix *
        state.local_right_coefficient_matrix.transpose();
    scatter_add_dense_submatrix(
        alpha_local_image,
        state.alpha_support,
        state.alpha_support,
        state_weight,
        &weights.alpha_weight_matrix);

    Eigen::MatrixXd beta_local_image =
        state.local_left_coefficient_matrix.transpose() *
        state.local_right_coefficient_matrix;
    scatter_add_dense_submatrix(
        beta_local_image,
        state.beta_support,
        state.beta_support,
        state_weight,
        &weights.beta_weight_matrix);
  }

  return weights;
}

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

double apply_annihilation(
    int orbital,
    std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  const auto iterator =
      std::lower_bound(occupied_orbitals->begin(), occupied_orbitals->end(), orbital);
  if (iterator == occupied_orbitals->end() || *iterator != orbital) {
    return 0.0;
  }
  const int parity = static_cast<int>(std::distance(occupied_orbitals->begin(), iterator));
  occupied_orbitals->erase(iterator);
  return parity_sign(parity);
}

double apply_creation(
    int orbital,
    std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  const auto iterator =
      std::lower_bound(occupied_orbitals->begin(), occupied_orbitals->end(), orbital);
  if (iterator != occupied_orbitals->end() && *iterator == orbital) {
    return 0.0;
  }
  const int parity = static_cast<int>(std::distance(occupied_orbitals->begin(), iterator));
  occupied_orbitals->insert(iterator, orbital);
  return parity_sign(parity);
}

SpinExcitation build_spin_excitation(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals) {
  SpinExcitation excitation;
  std::set_difference(
      right_occupied_orbitals.begin(),
      right_occupied_orbitals.end(),
      left_occupied_orbitals.begin(),
      left_occupied_orbitals.end(),
      std::back_inserter(excitation.removed_orbitals));
  std::set_difference(
      left_occupied_orbitals.begin(),
      left_occupied_orbitals.end(),
      right_occupied_orbitals.begin(),
      right_occupied_orbitals.end(),
      std::back_inserter(excitation.added_orbitals));
  if (excitation.removed_orbitals.size() != excitation.added_orbitals.size()) {
    throw std::runtime_error("left/right determinants carry inconsistent electron counts");
  }
  excitation.rank = static_cast<int>(excitation.removed_orbitals.size());
  return excitation;
}

double compute_excitation_phase(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals,
    const SpinExcitation& excitation) {
  if (excitation.rank == 0) {
    return left_occupied_orbitals == right_occupied_orbitals ? 1.0 : 0.0;
  }

  std::vector<int> occupied_orbitals = right_occupied_orbitals;
  double phase = 1.0;
  for (int removed_orbital : excitation.removed_orbitals) {
    phase *= apply_annihilation(removed_orbital, &occupied_orbitals);
    if (phase == 0.0) {
      return 0.0;
    }
  }
  for (auto added_iterator = excitation.added_orbitals.rbegin();
       added_iterator != excitation.added_orbitals.rend();
       ++added_iterator) {
    phase *= apply_creation(*added_iterator, &occupied_orbitals);
    if (phase == 0.0) {
      return 0.0;
    }
  }
  return occupied_orbitals == left_occupied_orbitals ? phase : 0.0;
}

void accumulate_spin_left_right_one_electron_gradient(
    const std::vector<std::vector<int>>& unique_determinants,
    const Eigen::MatrixXd& weight_matrix,
    Eigen::MatrixXd* left_right_one_electron_gradient) {
  if (left_right_one_electron_gradient == nullptr) {
    throw std::invalid_argument("left_right_one_electron_gradient must not be null");
  }
  if (weight_matrix.rows() != static_cast<int>(unique_determinants.size()) ||
      weight_matrix.cols() != static_cast<int>(unique_determinants.size())) {
    throw std::invalid_argument("weight_matrix shape does not match unique determinants");
  }

  // In the exact biorthogonal determinant Hamiltonian, the one-electron piece
  // obeys the standard Slater-Condon pattern in the left/right biorthogonal
  // orbital basis:
  // - diagonal pairs contribute `h^{LR}_{o,o}` for each occupied orbital,
  // - rank-1 pairs contribute `phase * h^{LR}_{a,i}`,
  // - higher-rank pairs carry no one-electron contribution.
  for (int left_id = 0; left_id < weight_matrix.rows(); ++left_id) {
    const auto& occ_L = unique_determinants[xmvb::to_size(left_id)];
    for (int right_id = 0; right_id < weight_matrix.cols(); ++right_id) {
      const double weight = weight_matrix(left_id, right_id);
      if (std::abs(weight) <= kContributionTolerance) {
        continue;
      }

      const auto& occ_R = unique_determinants[xmvb::to_size(right_id)];
      const SpinExcitation excitation = build_spin_excitation(occ_L, occ_R);
      if (excitation.rank == 0) {
        for (const int occupied_orbital : occ_L) {
          (*left_right_one_electron_gradient)(
              occupied_orbital,
              occupied_orbital) += weight;
        }
        continue;
      }
      if (excitation.rank != 1) {
        continue;
      }

      const double phase =
          compute_excitation_phase(occ_L, occ_R, excitation);
      if (phase == 0.0) {
        throw std::runtime_error(
            "failed to reproduce left determinant from one-electron excitation");
      }
      (*left_right_one_electron_gradient)(
          excitation.added_orbitals.front(),
          excitation.removed_orbitals.front()) += weight * phase;
    }
  }
}

BiorthogonalOneElectronTransformBackwardContribution
pull_back_left_right_one_electron_gradient(
    const Eigen::MatrixXd& left_right_one_electron_gradient,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const std::vector<double>& right_right_one_electron) {
  validate_biorthogonal_orbital_integrals(orbital_integrals);
  const int n_orbitals = orbital_integrals.n_orbitals;
  const std::size_t expected_size =
      xmvb::to_size(n_orbitals) * n_orbitals;
  if (right_right_one_electron.size() != expected_size) {
    throw std::invalid_argument(
        "right_right_one_electron size does not match orbital_integrals.n_orbitals");
  }
  if (left_right_one_electron_gradient.rows() != n_orbitals ||
      left_right_one_electron_gradient.cols() != n_orbitals) {
    throw std::invalid_argument(
        "left_right_one_electron_gradient dimensions are inconsistent");
  }

  const Eigen::Map<const Eigen::MatrixXd> right_right_one_electron_matrix(
      right_right_one_electron.data(),
      n_orbitals,
      n_orbitals);
  const Eigen::MatrixXd& left_dual_from_right_transform =
      orbital_integrals.left_dual_from_right_transform;

  // Reverse-mode pullback through
  //   h^{LR} = X^{-1} HHO,
  //   X^{-1} = SSO^{-1}.
  // The first line contributes
  //   dHHO += (X^{-1})^T d h^{LR},
  //   dX^{-1} += d h^{LR} HHO^T,
  // and the inverse map contributes
  //   dSSO += -(X^{-1})^T dX^{-1} (X^{-1})^T.
  const Eigen::MatrixXd active_one_electron_gradient =
      left_dual_from_right_transform.transpose() *
      left_right_one_electron_gradient;
  const Eigen::MatrixXd left_dual_from_right_transform_gradient =
      left_right_one_electron_gradient *
      right_right_one_electron_matrix.transpose();
  const Eigen::MatrixXd active_orbital_overlap_gradient =
      -left_dual_from_right_transform.transpose() *
      left_dual_from_right_transform_gradient *
      left_dual_from_right_transform.transpose();

  BiorthogonalOneElectronTransformBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      active_orbital_overlap_gradient.data(),
      active_orbital_overlap_gradient.data() +
          active_orbital_overlap_gradient.size());
  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() +
          active_one_electron_gradient.size());
  return result;
}

}  // namespace

BiorthogonalOneElectronTransformBackwardContribution
build_biorthogonal_one_electron_transform_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const std::vector<double>& right_right_one_electron) {
  const OneElectronSpinWeightMatrices weight_matrices =
      build_one_electron_spin_weight_matrices(
          same_spin_pair_cache,
          selected_state_matrices);

  Eigen::MatrixXd left_right_one_electron_gradient =
      Eigen::MatrixXd::Zero(
          orbital_integrals.n_orbitals,
          orbital_integrals.n_orbitals);
  accumulate_spin_left_right_one_electron_gradient(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      weight_matrices.alpha_weight_matrix,
      &left_right_one_electron_gradient);
  accumulate_spin_left_right_one_electron_gradient(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      weight_matrices.beta_weight_matrix,
      &left_right_one_electron_gradient);

  return pull_back_left_right_one_electron_gradient(
      left_right_one_electron_gradient,
      orbital_integrals,
      right_right_one_electron);
}

}  // namespace xmvb::vb::biorthogonal_vbscf
