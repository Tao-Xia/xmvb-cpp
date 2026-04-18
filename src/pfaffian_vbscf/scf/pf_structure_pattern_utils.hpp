#pragma once

#include <random>
#include <utility>
#include <vector>

#include "pfaffian_vbscf/data/pf_state.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfActiveStructurePattern {
  std::vector<int> doubly_occupied;
  std::vector<std::pair<int, int>> covalent_pairs;
  std::vector<int> open_shell_orbitals;
};

struct PfElectronPartition {
  int n_inactive_doubly_occupied_orbitals = 0;
  int n_singlet_pairs = 0;
  int n_open_shell_electrons = 0;
  int active_start = 0;
};

PfElectronPartition build_pf_electron_partition(
    const xmvb::vb::RawStructureData& raw_structure_data);

PfActiveStructurePattern decode_pf_structure_pattern(
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfElectronPartition& partition,
    int n_active_orbitals,
    int structure_index);

Matrix build_pf_alpha_beta_block(
    const PfActiveStructurePattern& pattern,
    int n_active_orbitals);

void add_pf_symmetric_noise(
    double pairing_noise,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution,
    Matrix* alpha_beta_block);

PfState build_pf_state_from_alpha_beta_block(
    int n_active_orbitals,
    int n_singlet_pairs,
    const ConstMatrixRef& alpha_beta_block,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals);

}  // namespace xmvb::pfaffian_vbscf
