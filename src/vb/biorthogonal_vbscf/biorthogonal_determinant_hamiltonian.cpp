#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

struct SpinExcitation {
  std::vector<int> removed_orbitals;
  std::vector<int> added_orbitals;
  int rank = 0;
};

void validate_occupied_orbitals(
    const std::vector<int>& occupied_orbitals,
    int n_orbitals,
    const char* label) {
  if (n_orbitals <= 0) {
    throw std::invalid_argument("n_orbitals must be positive");
  }
  for (std::size_t orbital_index = 0; orbital_index < occupied_orbitals.size(); ++orbital_index) {
    const int orbital = occupied_orbitals[orbital_index];
    if (orbital < 0 || orbital >= n_orbitals) {
      throw std::out_of_range(std::string(label) + " contains an out-of-range orbital");
    }
    if (orbital_index > 0 &&
        occupied_orbitals[orbital_index - 1] >= occupied_orbitals[orbital_index]) {
      throw std::invalid_argument(
          std::string(label) + " must be strictly increasing without duplicates");
    }
  }
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

double compute_excitation_phase(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals,
    const SpinExcitation& excitation) {
  if (excitation.rank == 0) {
    return left_occupied_orbitals == right_occupied_orbitals ? 1.0 : 0.0;
  }

  // The canonical excitation operator is
  // `a^\dagger_{a_1} ... a^\dagger_{a_k} a_{i_k} ... a_{i_1}` with ascending
  // `i_1 < ... < i_k` and `a_1 < ... < a_k`. Acting on the ket therefore means:
  // 1. annihilate removed orbitals in ascending order,
  // 2. create added orbitals in descending order.
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

double same_spin_diagonal_two_electron_energy(
    const std::vector<int>& occupied_orbitals,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  double total_energy = 0.0;
  for (std::size_t left_index = 0; left_index < occupied_orbitals.size(); ++left_index) {
    const int first_orbital = occupied_orbitals[left_index];
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals.size();
         ++right_index) {
      const int second_orbital = occupied_orbitals[right_index];
      total_energy +=
          evaluate_biorthogonal_two_electron_integral(
              first_orbital,
              first_orbital,
              second_orbital,
              second_orbital,
              orbital_integrals,
              right_right_two_electron_view) -
          evaluate_biorthogonal_two_electron_integral(
              first_orbital,
              second_orbital,
              second_orbital,
              first_orbital,
              orbital_integrals,
              right_right_two_electron_view);
    }
  }
  return total_energy;
}

double opposite_spin_diagonal_two_electron_energy(
    const std::vector<int>& alpha_occupied_orbitals,
    const std::vector<int>& beta_occupied_orbitals,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  double total_energy = 0.0;
  for (int alpha_orbital : alpha_occupied_orbitals) {
    for (int beta_orbital : beta_occupied_orbitals) {
      total_energy += evaluate_biorthogonal_two_electron_integral(
          alpha_orbital,
          alpha_orbital,
          beta_orbital,
          beta_orbital,
          orbital_integrals,
          right_right_two_electron_view);
    }
  }
  return total_energy;
}

double evaluate_alpha_single_total_hamiltonian(
    const BiorthogonalDeterminant& right_determinant,
    const SpinExcitation& alpha_excitation,
    double alpha_phase,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  const int removed_orbital = alpha_excitation.removed_orbitals.front();
  const int added_orbital = alpha_excitation.added_orbitals.front();

  double matrix_element =
      orbital_integrals.left_right_one_electron(added_orbital, removed_orbital);

  for (int spectator_orbital : right_determinant.alpha_occupied_orbitals) {
    if (spectator_orbital == removed_orbital) {
      continue;
    }
    matrix_element +=
        evaluate_biorthogonal_two_electron_integral(
            removed_orbital,
            added_orbital,
            spectator_orbital,
            spectator_orbital,
            orbital_integrals,
            right_right_two_electron_view) -
        evaluate_biorthogonal_two_electron_integral(
            removed_orbital,
            spectator_orbital,
            spectator_orbital,
            added_orbital,
            orbital_integrals,
            right_right_two_electron_view);
  }

  for (int spectator_orbital : right_determinant.beta_occupied_orbitals) {
    matrix_element += evaluate_biorthogonal_two_electron_integral(
        removed_orbital,
        added_orbital,
        spectator_orbital,
        spectator_orbital,
        orbital_integrals,
        right_right_two_electron_view);
  }

  return alpha_phase * matrix_element;
}

double evaluate_beta_single_total_hamiltonian(
    const BiorthogonalDeterminant& right_determinant,
    const SpinExcitation& beta_excitation,
    double beta_phase,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  const int removed_orbital = beta_excitation.removed_orbitals.front();
  const int added_orbital = beta_excitation.added_orbitals.front();

  double matrix_element =
      orbital_integrals.left_right_one_electron(added_orbital, removed_orbital);

  for (int spectator_orbital : right_determinant.beta_occupied_orbitals) {
    if (spectator_orbital == removed_orbital) {
      continue;
    }
    matrix_element +=
        evaluate_biorthogonal_two_electron_integral(
            removed_orbital,
            added_orbital,
            spectator_orbital,
            spectator_orbital,
            orbital_integrals,
            right_right_two_electron_view) -
        evaluate_biorthogonal_two_electron_integral(
            removed_orbital,
            spectator_orbital,
            spectator_orbital,
            added_orbital,
            orbital_integrals,
            right_right_two_electron_view);
  }

  for (int spectator_orbital : right_determinant.alpha_occupied_orbitals) {
    matrix_element += evaluate_biorthogonal_two_electron_integral(
        spectator_orbital,
        spectator_orbital,
        removed_orbital,
        added_orbital,
        orbital_integrals,
        right_right_two_electron_view);
  }

  return beta_phase * matrix_element;
}

double evaluate_same_spin_double_hamiltonian(
    const SpinExcitation& excitation,
    double excitation_phase,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  const int removed_first = excitation.removed_orbitals[0];
  const int removed_second = excitation.removed_orbitals[1];
  const int added_first = excitation.added_orbitals[0];
  const int added_second = excitation.added_orbitals[1];
  const double matrix_element =
      evaluate_biorthogonal_two_electron_integral(
          removed_first,
          added_first,
          removed_second,
          added_second,
          orbital_integrals,
          right_right_two_electron_view) -
      evaluate_biorthogonal_two_electron_integral(
          removed_first,
          added_second,
          removed_second,
          added_first,
          orbital_integrals,
          right_right_two_electron_view);
  return excitation_phase * matrix_element;
}

double evaluate_opposite_spin_double_hamiltonian(
    const SpinExcitation& alpha_excitation,
    const SpinExcitation& beta_excitation,
    double alpha_phase,
    double beta_phase,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  return alpha_phase * beta_phase *
      evaluate_biorthogonal_two_electron_integral(
          alpha_excitation.removed_orbitals.front(),
          alpha_excitation.added_orbitals.front(),
          beta_excitation.removed_orbitals.front(),
          beta_excitation.added_orbitals.front(),
          orbital_integrals,
          right_right_two_electron_view);
}

}  // namespace

void validate_biorthogonal_determinant(
    const BiorthogonalDeterminant& determinant,
    int n_orbitals) {
  validate_occupied_orbitals(
      determinant.alpha_occupied_orbitals,
      n_orbitals,
      "alpha_occupied_orbitals");
  validate_occupied_orbitals(
      determinant.beta_occupied_orbitals,
      n_orbitals,
      "beta_occupied_orbitals");
}

BiorthogonalDeterminantHamiltonianEntry
evaluate_biorthogonal_determinant_hamiltonian(
    const BiorthogonalDeterminant& left_determinant,
    const BiorthogonalDeterminant& right_determinant,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  validate_biorthogonal_orbital_integrals(orbital_integrals);
  validate_biorthogonal_determinant(left_determinant, orbital_integrals.n_orbitals);
  validate_biorthogonal_determinant(right_determinant, orbital_integrals.n_orbitals);
  if (left_determinant.alpha_occupied_orbitals.size() !=
          right_determinant.alpha_occupied_orbitals.size() ||
      left_determinant.beta_occupied_orbitals.size() !=
          right_determinant.beta_occupied_orbitals.size()) {
    throw std::invalid_argument(
        "left and right determinants must have the same alpha/beta electron counts");
  }

  const SpinExcitation alpha_excitation = build_spin_excitation(
      left_determinant.alpha_occupied_orbitals,
      right_determinant.alpha_occupied_orbitals);
  const SpinExcitation beta_excitation = build_spin_excitation(
      left_determinant.beta_occupied_orbitals,
      right_determinant.beta_occupied_orbitals);

  BiorthogonalDeterminantHamiltonianEntry entry;
  entry.alpha_excitation_rank = alpha_excitation.rank;
  entry.beta_excitation_rank = beta_excitation.rank;
  entry.overlap =
      (alpha_excitation.rank == 0 && beta_excitation.rank == 0) ? 1.0 : 0.0;

  if (alpha_excitation.rank > 2 || beta_excitation.rank > 2 ||
      alpha_excitation.rank + beta_excitation.rank > 2) {
    return entry;
  }

  const double alpha_phase = compute_excitation_phase(
      left_determinant.alpha_occupied_orbitals,
      right_determinant.alpha_occupied_orbitals,
      alpha_excitation);
  const double beta_phase = compute_excitation_phase(
      left_determinant.beta_occupied_orbitals,
      right_determinant.beta_occupied_orbitals,
      beta_excitation);

  if ((alpha_excitation.rank > 0 && alpha_phase == 0.0) ||
      (beta_excitation.rank > 0 && beta_phase == 0.0)) {
    throw std::runtime_error("failed to reproduce left determinant from excitation operator");
  }

  if (alpha_excitation.rank == 0 && beta_excitation.rank == 0) {
    for (int occupied_orbital : left_determinant.alpha_occupied_orbitals) {
      entry.one_electron_hamiltonian +=
          orbital_integrals.left_right_one_electron(occupied_orbital, occupied_orbital);
    }
    for (int occupied_orbital : left_determinant.beta_occupied_orbitals) {
      entry.one_electron_hamiltonian +=
          orbital_integrals.left_right_one_electron(occupied_orbital, occupied_orbital);
    }

    entry.total_hamiltonian =
        entry.one_electron_hamiltonian +
        same_spin_diagonal_two_electron_energy(
            left_determinant.alpha_occupied_orbitals,
            orbital_integrals,
            right_right_two_electron_view) +
        same_spin_diagonal_two_electron_energy(
            left_determinant.beta_occupied_orbitals,
            orbital_integrals,
            right_right_two_electron_view) +
        opposite_spin_diagonal_two_electron_energy(
            left_determinant.alpha_occupied_orbitals,
            left_determinant.beta_occupied_orbitals,
            orbital_integrals,
            right_right_two_electron_view);
    return entry;
  }

  if (alpha_excitation.rank == 1 && beta_excitation.rank == 0) {
    entry.one_electron_hamiltonian =
        alpha_phase *
        orbital_integrals.left_right_one_electron(
            alpha_excitation.added_orbitals.front(),
            alpha_excitation.removed_orbitals.front());
    entry.total_hamiltonian =
        evaluate_alpha_single_total_hamiltonian(
            right_determinant,
            alpha_excitation,
            alpha_phase,
            orbital_integrals,
            right_right_two_electron_view);
    return entry;
  }

  if (alpha_excitation.rank == 0 && beta_excitation.rank == 1) {
    entry.one_electron_hamiltonian =
        beta_phase *
        orbital_integrals.left_right_one_electron(
            beta_excitation.added_orbitals.front(),
            beta_excitation.removed_orbitals.front());
    entry.total_hamiltonian =
        evaluate_beta_single_total_hamiltonian(
            right_determinant,
            beta_excitation,
            beta_phase,
            orbital_integrals,
            right_right_two_electron_view);
    return entry;
  }

  if (alpha_excitation.rank == 2 && beta_excitation.rank == 0) {
    entry.total_hamiltonian =
        evaluate_same_spin_double_hamiltonian(
            alpha_excitation,
            alpha_phase,
            orbital_integrals,
            right_right_two_electron_view);
    return entry;
  }

  if (alpha_excitation.rank == 0 && beta_excitation.rank == 2) {
    entry.total_hamiltonian =
        evaluate_same_spin_double_hamiltonian(
            beta_excitation,
            beta_phase,
            orbital_integrals,
            right_right_two_electron_view);
    return entry;
  }

  if (alpha_excitation.rank == 1 && beta_excitation.rank == 1) {
    entry.total_hamiltonian =
        evaluate_opposite_spin_double_hamiltonian(
            alpha_excitation,
            beta_excitation,
            alpha_phase,
            beta_phase,
            orbital_integrals,
            right_right_two_electron_view);
    return entry;
  }

  return entry;
}

BiorthogonalDeterminantHamiltonianEntry
evaluate_biorthogonal_determinant_hamiltonian(
    const BiorthogonalDeterminant& left_determinant,
    const BiorthogonalDeterminant& right_determinant,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result) {
  return evaluate_biorthogonal_determinant_hamiltonian(
      left_determinant,
      right_determinant,
      orbital_integrals,
      make_active_space_two_electron_view(right_right_two_electron_result));
}

Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  return build_biorthogonal_determinant_hamiltonian_matrix(
      determinants,
      determinants,
      orbital_integrals,
      right_right_two_electron_view);
}

Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result) {
  return build_biorthogonal_determinant_hamiltonian_matrix(
      determinants,
      orbital_integrals,
      make_active_space_two_electron_view(right_right_two_electron_result));
}

Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& left_determinants,
    const std::vector<BiorthogonalDeterminant>& right_determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  validate_biorthogonal_orbital_integrals(orbital_integrals);
  if (left_determinants.empty() || right_determinants.empty()) {
    throw std::invalid_argument("determinant lists must not be empty");
  }

  Eigen::MatrixXd determinant_hamiltonian = Eigen::MatrixXd::Zero(
      static_cast<int>(left_determinants.size()),
      static_cast<int>(right_determinants.size()));

  for (std::size_t left_index = 0; left_index < left_determinants.size(); ++left_index) {
    validate_biorthogonal_determinant(
        left_determinants[left_index],
        orbital_integrals.n_orbitals);
    for (std::size_t right_index = 0; right_index < right_determinants.size(); ++right_index) {
      validate_biorthogonal_determinant(
          right_determinants[right_index],
          orbital_integrals.n_orbitals);
      determinant_hamiltonian(
          static_cast<int>(left_index),
          static_cast<int>(right_index)) =
          evaluate_biorthogonal_determinant_hamiltonian(
              left_determinants[left_index],
              right_determinants[right_index],
              orbital_integrals,
              right_right_two_electron_view)
              .total_hamiltonian;
    }
  }

  return determinant_hamiltonian;
}

Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& left_determinants,
    const std::vector<BiorthogonalDeterminant>& right_determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result) {
  return build_biorthogonal_determinant_hamiltonian_matrix(
      left_determinants,
      right_determinants,
      orbital_integrals,
      make_active_space_two_electron_view(right_right_two_electron_result));
}

Eigen::MatrixXd build_biorthogonal_determinant_metric_matrix(
    const std::vector<BiorthogonalDeterminant>& determinants,
    int n_orbitals) {
  if (determinants.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }
  for (const auto& determinant : determinants) {
    validate_biorthogonal_determinant(determinant, n_orbitals);
  }
  return Eigen::MatrixXd::Identity(
      static_cast<int>(determinants.size()),
      static_cast<int>(determinants.size()));
}

}  // namespace xmvb::vb::biorthogonal_vbscf
