#include "pfaffian_vbscf/scf/pf_structure_pattern_utils.hpp"

#include <cmath>
#include <stdexcept>

#include "pfaffian_vbscf/math/antisymm_codec.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

Matrix build_pairing_matrix(const ConstMatrixRef& alpha_beta_block) {
  const int n_active_orbitals = static_cast<int>(alpha_beta_block.rows());
  Matrix pairing_matrix =
      Matrix::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  pairing_matrix.topRightCorner(n_active_orbitals, n_active_orbitals) =
      alpha_beta_block;
  pairing_matrix.bottomLeftCorner(n_active_orbitals, n_active_orbitals) =
      -alpha_beta_block.transpose();
  return pairing_matrix;
}

}  // namespace

PfElectronPartition build_pf_electron_partition(
    const xmvb::vb::RawStructureData& raw_structure_data) {
  if (raw_structure_data.n_total_electrons <= 0 ||
      raw_structure_data.n_active_electrons <= 0) {
    throw std::invalid_argument(
        "raw structure data must define positive electron counts");
  }
  if (raw_structure_data.spin_multiplicity <= 0) {
    throw std::invalid_argument("spin multiplicity must be positive");
  }

  const int inactive_electrons =
      raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons;
  if ((inactive_electrons % 2) != 0) {
    throw std::invalid_argument(
        "inactive electron count must be even for closed-shell core orbitals");
  }

  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int singlet_electron_count =
      raw_structure_data.n_active_electrons - n_open_shell_electrons;
  if (singlet_electron_count < 0 || (singlet_electron_count % 2) != 0) {
    throw std::invalid_argument(
        "raw structure active-electron partition is inconsistent with the spin multiplicity");
  }

  PfElectronPartition partition;
  partition.n_inactive_doubly_occupied_orbitals = inactive_electrons / 2;
  partition.n_singlet_pairs = singlet_electron_count / 2;
  partition.n_open_shell_electrons = n_open_shell_electrons;
  partition.active_start = 2 * partition.n_inactive_doubly_occupied_orbitals;
  return partition;
}

PfActiveStructurePattern decode_pf_structure_pattern(
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfElectronPartition& partition,
    int n_active_orbitals,
    int structure_index) {
  if (structure_index < 0 || structure_index >= raw_structure_data.n_structures) {
    throw std::out_of_range("raw structure index is out of range");
  }

  const int active_stop =
      partition.active_start + raw_structure_data.n_active_electrons;
  if (partition.active_start < 0 ||
      active_stop > raw_structure_data.n_total_electrons) {
    throw std::runtime_error("active-electron window is out of range");
  }

  PfActiveStructurePattern pattern;
  std::vector<bool> orbital_is_assigned(
      xmvb::to_size(n_active_orbitals),
      false);
  const int* structure_orbitals =
      raw_structure_data.structure_orbitals_data(structure_index);

  for (int pair_index = 0; pair_index < partition.n_singlet_pairs; ++pair_index) {
    const int left_orbital =
        structure_orbitals[partition.active_start + 2 * pair_index] -
        partition.n_inactive_doubly_occupied_orbitals - 1;
    const int right_orbital =
        structure_orbitals[partition.active_start + 2 * pair_index + 1] -
        partition.n_inactive_doubly_occupied_orbitals - 1;
    if (left_orbital < 0 || left_orbital >= n_active_orbitals ||
        right_orbital < 0 || right_orbital >= n_active_orbitals) {
      throw std::invalid_argument(
          "raw VB structure references an orbital outside the active-space window");
    }

    if (left_orbital == right_orbital) {
      if (orbital_is_assigned[xmvb::to_size(left_orbital)]) {
        throw std::invalid_argument(
            "raw VB structure reuses an active orbital across multiple singlet pairs");
      }
      orbital_is_assigned[xmvb::to_size(left_orbital)] = true;
      pattern.doubly_occupied.push_back(left_orbital);
      continue;
    }

    if (orbital_is_assigned[xmvb::to_size(left_orbital)] ||
        orbital_is_assigned[xmvb::to_size(right_orbital)]) {
      throw std::invalid_argument(
          "raw VB structure reuses an active orbital across multiple covalent bonds");
    }
    orbital_is_assigned[xmvb::to_size(left_orbital)] = true;
    orbital_is_assigned[xmvb::to_size(right_orbital)] = true;
    pattern.covalent_pairs.emplace_back(left_orbital, right_orbital);
  }

  for (int open_shell_index = 0;
       open_shell_index < partition.n_open_shell_electrons;
       ++open_shell_index) {
    const int orbital =
        structure_orbitals[
            partition.active_start + 2 * partition.n_singlet_pairs + open_shell_index] -
        partition.n_inactive_doubly_occupied_orbitals - 1;
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument(
          "raw VB structure references an open-shell orbital outside the active-space window");
    }
    if (orbital_is_assigned[xmvb::to_size(orbital)]) {
      throw std::invalid_argument(
          "raw VB structure reuses an active orbital between singlet-pair and open-shell sectors");
    }
    orbital_is_assigned[xmvb::to_size(orbital)] = true;
    pattern.open_shell_orbitals.push_back(orbital);
  }

  return pattern;
}

Matrix build_pf_alpha_beta_block(
    const PfActiveStructurePattern& pattern,
    int n_active_orbitals) {
  Matrix alpha_beta_block =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);

  for (const int orbital : pattern.doubly_occupied) {
    alpha_beta_block(orbital, orbital) = 1.0;
  }
  for (const auto& bond : pattern.covalent_pairs) {
    alpha_beta_block(bond.first, bond.second) = 1.0;
    alpha_beta_block(bond.second, bond.first) = 1.0;
  }
  return alpha_beta_block;
}

void add_pf_symmetric_noise(
    double pairing_noise,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution,
    Matrix* alpha_beta_block) {
  if (alpha_beta_block == nullptr) {
    throw std::invalid_argument("alpha_beta_block must not be null");
  }
  if (pairing_noise <= 0.0) {
    return;
  }
  if (generator == nullptr || distribution == nullptr) {
    throw std::invalid_argument("noise generator inputs must not be null");
  }
  for (int col = 0; col < alpha_beta_block->cols(); ++col) {
    for (int row = 0; row <= col; ++row) {
      const double delta = pairing_noise * (*distribution)(*generator);
      (*alpha_beta_block)(row, col) += delta;
      if (row != col) {
        (*alpha_beta_block)(col, row) += delta;
      }
    }
  }
}

PfState build_pf_state_from_alpha_beta_block(
    int n_active_orbitals,
    int n_singlet_pairs,
    const ConstMatrixRef& alpha_beta_block,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals) {
  PfState state;
  state.n_active_orbitals = n_active_orbitals;
  state.n_spin_orbitals = 2 * n_active_orbitals;
  state.n_singlet_pairs = n_singlet_pairs;
  state.blocked_alpha_orbitals = blocked_alpha_orbitals;
  state.blocked_beta_orbitals = blocked_beta_orbitals;

  Matrix normalized_block = alpha_beta_block;
  const double norm = std::sqrt(std::max(1.0e-30, normalized_block.squaredNorm()));
  normalized_block /= norm;
  state.packed_entries = encode_antisymm(build_pairing_matrix(normalized_block));
  return state;
}

}  // namespace xmvb::pfaffian_vbscf
