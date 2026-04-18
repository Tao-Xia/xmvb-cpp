#include "pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pfaffian_vbscf/scf/pf_spin_coupling_builder.hpp"
#include "pfaffian_vbscf/scf/pf_structure_pattern_utils.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

int resolve_target_spin_multiplicity(
    int requested_spin_multiplicity,
    int raw_spin_multiplicity) {
  if (requested_spin_multiplicity == 0) {
    return raw_spin_multiplicity;
  }
  return requested_spin_multiplicity;
}

int resolve_spin_adapted_ms_twice(
    int requested_ms_twice,
    int n_open_shell_electrons) {
  if (requested_ms_twice == std::numeric_limits<int>::min()) {
    return n_open_shell_electrons % 2;
  }
  return requested_ms_twice;
}

}  // namespace

PfSpinAdaptedBasisData build_structure_pf_spin_adapted_basis(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfSpinAdaptedBasisFactoryOptions& options) {
  if (options.n_structures < 0) {
    throw std::invalid_argument("Pf spin-adapted basis structure count must be non-negative");
  }
  if (options.pairing_noise < 0.0) {
    throw std::invalid_argument("Pf spin-adapted basis pairing_noise must be non-negative");
  }

  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  if (raw_structure_data.n_structures <= 0 ||
      raw_structure_data.raw_structure_orbitals.size() !=
          raw_structure_data.flat_orbital_count()) {
    throw std::invalid_argument(
        "raw structure count does not match stored VB structures");
  }

  const PfElectronPartition partition =
      build_pf_electron_partition(raw_structure_data);
  const int target_spin_multiplicity =
      resolve_target_spin_multiplicity(
          options.target_spin_multiplicity,
          raw_structure_data.spin_multiplicity);
  const int ms_twice =
      resolve_spin_adapted_ms_twice(
          options.ms_twice,
          partition.n_open_shell_electrons);
  PfSpinCouplingBuilder coupling_builder;
  const PfSpinCouplingBlock coupling_block =
      coupling_builder.build(
          partition.n_open_shell_electrons,
          target_spin_multiplicity,
          ms_twice);

  const int available_structures = raw_structure_data.n_structures;
  const int n_selected_structures =
      (options.n_structures == 0) ? available_structures : options.n_structures;
  if (n_selected_structures <= 0) {
    throw std::invalid_argument("Pf spin-adapted basis structure count must be positive");
  }
  if (n_selected_structures > available_structures) {
    throw std::invalid_argument(
        "requested Pf spin-adapted basis size exceeds the number of selected raw VB structures");
  }

  PfSpinAdaptedBasisData basis;
  basis.spin_multiplicity = target_spin_multiplicity;
  basis.ms_twice = ms_twice;
  basis.n_states =
      n_selected_structures *
      coupling_block.primitive_to_adapted_coefficients.cols();
  basis.primitive_basis.n_active_orbitals = n_active_orbitals;
  basis.primitive_basis.n_alpha =
      partition.n_singlet_pairs + coupling_block.n_blocked_alpha;
  basis.primitive_basis.n_beta =
      partition.n_singlet_pairs + coupling_block.n_blocked_beta;
  basis.primitive_basis.n_singlet_pairs = partition.n_singlet_pairs;
  basis.primitive_basis.n_blocked_alpha = coupling_block.n_blocked_alpha;
  basis.primitive_basis.n_blocked_beta = coupling_block.n_blocked_beta;
  basis.primitive_basis.n_states =
      n_selected_structures *
      coupling_block.primitive_to_adapted_coefficients.rows();
  basis.primitive_basis.states.reserve(
      xmvb::to_size(basis.primitive_basis.n_states));
  basis.states.reserve(xmvb::to_size(basis.n_states));
  basis.primitive_to_adapted_coefficients =
      Matrix::Zero(
          basis.primitive_basis.n_states,
          basis.n_states);

  std::mt19937 generator(options.seed);
  std::normal_distribution<double> distribution(0.0, 1.0);
  constexpr double kCoefficientTolerance = 1.0e-12;
  int primitive_offset = 0;
  int adapted_offset = 0;

  for (int structure_index = 0;
       structure_index < n_selected_structures;
       ++structure_index) {
    const PfActiveStructurePattern pattern =
        decode_pf_structure_pattern(
            raw_structure_data,
            partition,
            n_active_orbitals,
            structure_index);
    if (static_cast<int>(pattern.open_shell_orbitals.size()) !=
        partition.n_open_shell_electrons) {
      throw std::runtime_error(
          "decoded open-shell orbital count does not match the spin-coupling sector");
    }

    Matrix alpha_beta_block = build_pf_alpha_beta_block(
        pattern,
        n_active_orbitals);
    add_pf_symmetric_noise(
        options.pairing_noise,
        &generator,
        &distribution,
        &alpha_beta_block);

    for (const auto& spin_string : coupling_block.primitive_spin_strings) {
      std::vector<bool> beta_mask(
          xmvb::to_size(partition.n_open_shell_electrons),
          false);
      for (const int beta_position : spin_string.blocked_beta_positions) {
        beta_mask[xmvb::to_size(beta_position)] = true;
      }

      std::vector<int> blocked_alpha_orbitals;
      std::vector<int> blocked_beta_orbitals;
      blocked_alpha_orbitals.reserve(
          xmvb::to_size(coupling_block.n_blocked_alpha));
      blocked_beta_orbitals.reserve(
          xmvb::to_size(coupling_block.n_blocked_beta));
      for (int open_shell_index = 0;
           open_shell_index < partition.n_open_shell_electrons;
           ++open_shell_index) {
        const int orbital =
            pattern.open_shell_orbitals[xmvb::to_size(open_shell_index)];
        if (beta_mask[xmvb::to_size(open_shell_index)]) {
          blocked_beta_orbitals.push_back(orbital);
        } else {
          blocked_alpha_orbitals.push_back(orbital);
        }
      }

      basis.primitive_basis.states.push_back(
          build_pf_state_from_alpha_beta_block(
              n_active_orbitals,
              partition.n_singlet_pairs,
              alpha_beta_block,
              blocked_alpha_orbitals,
              blocked_beta_orbitals));
    }

    const int block_primitive_dim =
        coupling_block.primitive_to_adapted_coefficients.rows();
    const int block_adapted_dim =
        coupling_block.primitive_to_adapted_coefficients.cols();
    basis.primitive_to_adapted_coefficients.block(
        primitive_offset,
        adapted_offset,
        block_primitive_dim,
        block_adapted_dim) =
        coupling_block.primitive_to_adapted_coefficients;

    for (int adapted_local_index = 0;
         adapted_local_index < block_adapted_dim;
         ++adapted_local_index) {
      PfSpinAdaptedState state;
      state.spin_multiplicity = target_spin_multiplicity;
      state.ms_twice = ms_twice;
      for (int primitive_local_index = 0;
           primitive_local_index < block_primitive_dim;
           ++primitive_local_index) {
        const double coefficient =
            coupling_block.primitive_to_adapted_coefficients(
                primitive_local_index,
                adapted_local_index);
        if (std::abs(coefficient) <= kCoefficientTolerance) {
          continue;
        }
        state.primitive_state_indices.push_back(
            primitive_offset + primitive_local_index);
        state.primitive_coefficients.push_back(coefficient);
      }
      basis.states.push_back(std::move(state));
    }

    primitive_offset += block_primitive_dim;
    adapted_offset += block_adapted_dim;
  }

  return basis;
}

}  // namespace xmvb::pfaffian_vbscf
