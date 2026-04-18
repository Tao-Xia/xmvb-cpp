#include "vb/biorthogonal_vbscf/biorthogonal_two_electron_transform_backward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {


constexpr double kContributionTolerance = 1.0e-15;

struct SpinExcitation {
  std::vector<int> removed_orbitals;
  std::vector<int> added_orbitals;
  int rank = 0;
};

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

void accumulate_transformed_two_electron_integral_adjoint(
    int right_first_orbital,
    int left_first_orbital,
    int right_second_orbital,
    int left_second_orbital,
    double transformed_integral_gradient,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (left_dual_from_right_transform_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("two-electron transform backward outputs must not be null");
  }
  if (std::abs(transformed_integral_gradient) <= kContributionTolerance) {
    return;
  }

  const Eigen::MatrixXd& left_dual_from_right_transform =
      orbital_integrals.left_dual_from_right_transform;
  const int n_orbitals = orbital_integrals.n_orbitals;
  for (int right_basis_left = 0; right_basis_left < n_orbitals; ++right_basis_left) {
    const double left_first_scale =
        left_dual_from_right_transform(
            right_basis_left,
            left_first_orbital);
    if (std::abs(left_first_scale) <= kContributionTolerance) {
      continue;
    }
    const int first_pair_index = xmvb::vb::TwoElectronIndexer::packed_pair_index(
        right_first_orbital,
        right_basis_left);
    for (int right_basis_right = 0;
         right_basis_right < n_orbitals;
         ++right_basis_right) {
      const double left_second_scale =
          left_dual_from_right_transform(
              right_basis_right,
              left_second_orbital);
      if (std::abs(left_second_scale) <= kContributionTolerance) {
        continue;
      }
      const int second_pair_index = xmvb::vb::TwoElectronIndexer::packed_pair_index(
          right_second_orbital,
          right_basis_right);
      const double scaled_gradient =
          transformed_integral_gradient *
          left_first_scale *
          left_second_scale;
      const int packed_pair_of_pairs_index =
          xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
              first_pair_index,
              second_pair_index);
      (*packed_active_two_electron_gradient)[xmvb::to_size(
          packed_pair_of_pairs_index)] += scaled_gradient;

      const double original_kernel_value =
          xmvb::vb::lookup_active_space_two_electron_kernel_value(
              right_right_two_electron_view,
              first_pair_index,
              second_pair_index,
              n_orbitals);
      (*left_dual_from_right_transform_gradient)(
          right_basis_left,
          left_first_orbital) +=
          transformed_integral_gradient *
          left_second_scale *
          original_kernel_value;
      (*left_dual_from_right_transform_gradient)(
          right_basis_right,
          left_second_orbital) +=
          transformed_integral_gradient *
          left_first_scale *
          original_kernel_value;
    }
  }
}

void accumulate_diagonal_same_spin_terms(
    const std::vector<int>& occupied_orbitals,
    double pair_weight,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  for (std::size_t first_index = 0; first_index < occupied_orbitals.size(); ++first_index) {
    const int first_orbital = occupied_orbitals[first_index];
    for (std::size_t second_index = first_index + 1;
         second_index < occupied_orbitals.size();
         ++second_index) {
      const int second_orbital = occupied_orbitals[second_index];
      accumulate_transformed_two_electron_integral_adjoint(
          first_orbital,
          first_orbital,
          second_orbital,
          second_orbital,
          pair_weight,
          orbital_integrals,
          right_right_two_electron_view,
          left_dual_from_right_transform_gradient,
          packed_active_two_electron_gradient);
      accumulate_transformed_two_electron_integral_adjoint(
          first_orbital,
          second_orbital,
          second_orbital,
          first_orbital,
          -pair_weight,
          orbital_integrals,
          right_right_two_electron_view,
          left_dual_from_right_transform_gradient,
          packed_active_two_electron_gradient);
    }
  }
}

void accumulate_diagonal_opposite_spin_terms(
    const std::vector<int>& alpha_occupied_orbitals,
    const std::vector<int>& beta_occupied_orbitals,
    double pair_weight,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  for (const int alpha_orbital : alpha_occupied_orbitals) {
    for (const int beta_orbital : beta_occupied_orbitals) {
      accumulate_transformed_two_electron_integral_adjoint(
          alpha_orbital,
          alpha_orbital,
          beta_orbital,
          beta_orbital,
          pair_weight,
          orbital_integrals,
          right_right_two_electron_view,
          left_dual_from_right_transform_gradient,
          packed_active_two_electron_gradient);
    }
  }
}

void accumulate_alpha_single_terms(
    const std::vector<int>& alpha_occupied_orbitals_right,
    const std::vector<int>& beta_occupied_orbitals_right,
    const SpinExcitation& alpha_excitation,
    double pair_weight,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int removed_orbital = alpha_excitation.removed_orbitals.front();
  const int added_orbital = alpha_excitation.added_orbitals.front();
  for (const int spectator_orbital : alpha_occupied_orbitals_right) {
    if (spectator_orbital == removed_orbital) {
      continue;
    }
    accumulate_transformed_two_electron_integral_adjoint(
        removed_orbital,
        added_orbital,
        spectator_orbital,
        spectator_orbital,
        pair_weight,
        orbital_integrals,
        right_right_two_electron_view,
        left_dual_from_right_transform_gradient,
        packed_active_two_electron_gradient);
    accumulate_transformed_two_electron_integral_adjoint(
        removed_orbital,
        spectator_orbital,
        spectator_orbital,
        added_orbital,
        -pair_weight,
        orbital_integrals,
        right_right_two_electron_view,
        left_dual_from_right_transform_gradient,
        packed_active_two_electron_gradient);
  }
  for (const int spectator_orbital : beta_occupied_orbitals_right) {
    accumulate_transformed_two_electron_integral_adjoint(
        removed_orbital,
        added_orbital,
        spectator_orbital,
        spectator_orbital,
        pair_weight,
        orbital_integrals,
        right_right_two_electron_view,
        left_dual_from_right_transform_gradient,
        packed_active_two_electron_gradient);
  }
}

void accumulate_beta_single_terms(
    const std::vector<int>& alpha_occupied_orbitals_right,
    const std::vector<int>& beta_occupied_orbitals_right,
    const SpinExcitation& beta_excitation,
    double pair_weight,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int removed_orbital = beta_excitation.removed_orbitals.front();
  const int added_orbital = beta_excitation.added_orbitals.front();
  for (const int spectator_orbital : beta_occupied_orbitals_right) {
    if (spectator_orbital == removed_orbital) {
      continue;
    }
    accumulate_transformed_two_electron_integral_adjoint(
        removed_orbital,
        added_orbital,
        spectator_orbital,
        spectator_orbital,
        pair_weight,
        orbital_integrals,
        right_right_two_electron_view,
        left_dual_from_right_transform_gradient,
        packed_active_two_electron_gradient);
    accumulate_transformed_two_electron_integral_adjoint(
        removed_orbital,
        spectator_orbital,
        spectator_orbital,
        added_orbital,
        -pair_weight,
        orbital_integrals,
        right_right_two_electron_view,
        left_dual_from_right_transform_gradient,
        packed_active_two_electron_gradient);
  }
  for (const int spectator_orbital : alpha_occupied_orbitals_right) {
    accumulate_transformed_two_electron_integral_adjoint(
        spectator_orbital,
        spectator_orbital,
        removed_orbital,
        added_orbital,
        pair_weight,
        orbital_integrals,
        right_right_two_electron_view,
        left_dual_from_right_transform_gradient,
        packed_active_two_electron_gradient);
  }
}

void accumulate_same_spin_double_terms(
    const SpinExcitation& excitation,
    double pair_weight,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  accumulate_transformed_two_electron_integral_adjoint(
      excitation.removed_orbitals[0],
      excitation.added_orbitals[0],
      excitation.removed_orbitals[1],
      excitation.added_orbitals[1],
      pair_weight,
      orbital_integrals,
      right_right_two_electron_view,
      left_dual_from_right_transform_gradient,
      packed_active_two_electron_gradient);
  accumulate_transformed_two_electron_integral_adjoint(
      excitation.removed_orbitals[0],
      excitation.added_orbitals[1],
      excitation.removed_orbitals[1],
      excitation.added_orbitals[0],
      -pair_weight,
      orbital_integrals,
      right_right_two_electron_view,
      left_dual_from_right_transform_gradient,
      packed_active_two_electron_gradient);
}

void accumulate_opposite_spin_double_terms(
    const SpinExcitation& alpha_excitation,
    const SpinExcitation& beta_excitation,
    double pair_weight,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* left_dual_from_right_transform_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  accumulate_transformed_two_electron_integral_adjoint(
      alpha_excitation.removed_orbitals.front(),
      alpha_excitation.added_orbitals.front(),
      beta_excitation.removed_orbitals.front(),
      beta_excitation.added_orbitals.front(),
      pair_weight,
      orbital_integrals,
      right_right_two_electron_view,
      left_dual_from_right_transform_gradient,
      packed_active_two_electron_gradient);
}

}  // namespace

BiorthogonalTwoElectronTransformBackwardContribution
build_biorthogonal_two_electron_transform_backward_contribution(
    const xmvb::vb::CppVbInput& input,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronResult& right_right_two_electron_result) {
  validate_biorthogonal_orbital_integrals(orbital_integrals);
  if (selected_state_matrices.n_determinants !=
      static_cast<int>(input.structure_data.alpha_det.size()) ||
      selected_state_matrices.n_determinants !=
          static_cast<int>(input.structure_data.beta_det.size())) {
    throw std::invalid_argument(
        "selected_state_matrices determinant count does not match input");
  }

  const auto pair_weights =
      build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
          selected_state_matrices);
  const int n_orbitals = orbital_integrals.n_orbitals;
  Eigen::MatrixXd left_dual_from_right_transform_gradient =
      Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  std::vector<double> packed_active_two_electron_gradient(
      xmvb::vb::packed_active_two_electron_integral_count(n_orbitals),
      0.0);
  const xmvb::vb::ActiveSpaceTwoElectronView right_right_two_electron_view =
      xmvb::vb::make_active_space_two_electron_view(right_right_two_electron_result);

  for (int determinant_index_left = 0;
       determinant_index_left < pair_weights.n_determinants;
       ++determinant_index_left) {
    const auto& alpha_occ_L =
        input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)];
    const auto& beta_occ_L =
        input.structure_data.beta_det[xmvb::to_size(determinant_index_left)];
    for (int determinant_index_right = 0;
         determinant_index_right < pair_weights.n_determinants;
         ++determinant_index_right) {
      const double weight =
          pair_weights.ordered_hamiltonian_weights[
              xmvb::to_size(determinant_index_left) *
                  pair_weights.n_determinants +
              xmvb::to_size(determinant_index_right)];
      if (std::abs(weight) <= kContributionTolerance) {
        continue;
      }

      const auto& alpha_occ_R =
          input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)];
      const auto& beta_occ_R =
          input.structure_data.beta_det[xmvb::to_size(determinant_index_right)];
      const SpinExcitation alpha_excitation =
          build_spin_excitation(alpha_occ_L, alpha_occ_R);
      const SpinExcitation beta_excitation =
          build_spin_excitation(beta_occ_L, beta_occ_R);

      if (alpha_excitation.rank > 2 || beta_excitation.rank > 2 ||
          alpha_excitation.rank + beta_excitation.rank > 2) {
        continue;
      }

      const double alpha_phase =
          alpha_excitation.rank == 0
              ? 1.0
              : compute_excitation_phase(alpha_occ_L, alpha_occ_R, alpha_excitation);
      const double beta_phase =
          beta_excitation.rank == 0
              ? 1.0
              : compute_excitation_phase(beta_occ_L, beta_occ_R, beta_excitation);
      if ((alpha_excitation.rank > 0 && alpha_phase == 0.0) ||
          (beta_excitation.rank > 0 && beta_phase == 0.0)) {
        throw std::runtime_error(
            "failed to reproduce left determinant from excitation operator");
      }

      if (alpha_excitation.rank == 0 && beta_excitation.rank == 0) {
        accumulate_diagonal_same_spin_terms(
            alpha_occ_L,
            weight,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        accumulate_diagonal_same_spin_terms(
            beta_occ_L,
            weight,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        accumulate_diagonal_opposite_spin_terms(
            alpha_occ_L,
            beta_occ_L,
            weight,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        continue;
      }

      if (alpha_excitation.rank == 1 && beta_excitation.rank == 0) {
        accumulate_alpha_single_terms(
            alpha_occ_R,
            beta_occ_R,
            alpha_excitation,
            weight * alpha_phase,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        continue;
      }

      if (alpha_excitation.rank == 0 && beta_excitation.rank == 1) {
        accumulate_beta_single_terms(
            alpha_occ_R,
            beta_occ_R,
            beta_excitation,
            weight * beta_phase,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        continue;
      }

      if (alpha_excitation.rank == 2 && beta_excitation.rank == 0) {
        accumulate_same_spin_double_terms(
            alpha_excitation,
            weight * alpha_phase,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        continue;
      }

      if (alpha_excitation.rank == 0 && beta_excitation.rank == 2) {
        accumulate_same_spin_double_terms(
            beta_excitation,
            weight * beta_phase,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
        continue;
      }

      if (alpha_excitation.rank == 1 && beta_excitation.rank == 1) {
        accumulate_opposite_spin_double_terms(
            alpha_excitation,
            beta_excitation,
            weight * alpha_phase * beta_phase,
            orbital_integrals,
            right_right_two_electron_view,
            &left_dual_from_right_transform_gradient,
            &packed_active_two_electron_gradient);
      }
    }
  }

  const Eigen::MatrixXd& left_dual_from_right_transform =
      orbital_integrals.left_dual_from_right_transform;
  const Eigen::MatrixXd active_orbital_overlap_gradient =
      -left_dual_from_right_transform.transpose() *
      left_dual_from_right_transform_gradient *
      left_dual_from_right_transform.transpose();

  BiorthogonalTwoElectronTransformBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      active_orbital_overlap_gradient.data(),
      active_orbital_overlap_gradient.data() +
          active_orbital_overlap_gradient.size());
  result.packed_active_two_electron_gradient =
      std::move(packed_active_two_electron_gradient);
  return result;
}

}  // namespace xmvb::vb::biorthogonal_vbscf
