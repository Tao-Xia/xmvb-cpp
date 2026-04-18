#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"

#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pfaffian_vbscf/scf/pf_structure_pattern_utils.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

std::vector<std::vector<int>> enumerate_combinations(
    int n_items,
    int n_selected) {
  if (n_items < 0 || n_selected < 0 || n_selected > n_items) {
    throw std::invalid_argument("invalid combination enumeration request");
  }

  std::vector<std::vector<int>> combinations;
  std::vector<int> current;
  current.reserve(xmvb::to_size(n_selected));

  const auto recurse =
      [&](const auto& self, int start_index, int remaining) -> void {
    if (remaining == 0) {
      combinations.push_back(current);
      return;
    }
    for (int index = start_index; index <= n_items - remaining; ++index) {
      current.push_back(index);
      self(self, index + 1, remaining - 1);
      current.pop_back();
    }
  };
  recurse(recurse, 0, n_selected);
  return combinations;
}

int resolve_fixed_ms_twice(
    int requested_ms_twice,
    int n_open_shell_electrons) {
  if (requested_ms_twice == std::numeric_limits<int>::max()) {
    return n_open_shell_electrons;
  }
  return requested_ms_twice;
}

void validate_fixed_ms_sector(
    int n_open_shell_electrons,
    int ms_twice) {
  if (n_open_shell_electrons < 0) {
    throw std::invalid_argument("n_open_shell_electrons must be non-negative");
  }
  if (ms_twice < 0 || ms_twice > n_open_shell_electrons) {
    throw std::invalid_argument("requested 2*M_s is outside the open-shell sector");
  }
  if (((n_open_shell_electrons - ms_twice) % 2) != 0) {
    throw std::invalid_argument(
        "requested 2*M_s is incompatible with the open-shell electron parity");
  }
}

}  // namespace

PfBasisData build_fixed_ms_pf_basis(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfFixedMsBasisFactoryOptions& options) {
  if (options.n_structures < 0) {
    throw std::invalid_argument("Pf basis structure count must be non-negative");
  }
  if (options.pairing_noise < 0.0) {
    throw std::invalid_argument("Pf basis pairing_noise must be non-negative");
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
  const int ms_twice =
      resolve_fixed_ms_twice(
          options.ms_twice,
          partition.n_open_shell_electrons);
  validate_fixed_ms_sector(partition.n_open_shell_electrons, ms_twice);
  const int n_blocked_beta =
      (partition.n_open_shell_electrons - ms_twice) / 2;
  const int n_blocked_alpha =
      partition.n_open_shell_electrons - n_blocked_beta;
  const auto beta_position_combinations =
      enumerate_combinations(
          partition.n_open_shell_electrons,
          n_blocked_beta);

  const int available_structures = raw_structure_data.n_structures;
  const int n_selected_structures =
      (options.n_structures == 0) ? available_structures : options.n_structures;
  if (n_selected_structures <= 0) {
    throw std::invalid_argument("Pf basis structure count must be positive");
  }
  if (n_selected_structures > available_structures) {
    throw std::invalid_argument(
        "requested Pf basis size exceeds the number of selected raw VB structures");
  }

  PfBasisData basis;
  basis.n_active_orbitals = n_active_orbitals;
  basis.n_alpha = partition.n_singlet_pairs + n_blocked_alpha;
  basis.n_beta = partition.n_singlet_pairs + n_blocked_beta;
  basis.n_singlet_pairs = partition.n_singlet_pairs;
  basis.n_blocked_alpha = n_blocked_alpha;
  basis.n_blocked_beta = n_blocked_beta;
  basis.n_states =
      n_selected_structures *
      static_cast<int>(beta_position_combinations.size());
  basis.states.reserve(xmvb::to_size(basis.n_states));

  std::mt19937 generator(options.seed);
  std::normal_distribution<double> distribution(0.0, 1.0);

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
          "decoded open-shell orbital count does not match the requested fixed-M_s sector");
    }

    Matrix alpha_beta_block = build_pf_alpha_beta_block(
        pattern,
        n_active_orbitals);
    add_pf_symmetric_noise(
        options.pairing_noise,
        &generator,
        &distribution,
        &alpha_beta_block);

    for (const auto& beta_positions : beta_position_combinations) {
      std::vector<bool> beta_mask(
          xmvb::to_size(partition.n_open_shell_electrons),
          false);
      for (const int beta_position : beta_positions) {
        beta_mask[xmvb::to_size(beta_position)] = true;
      }

      std::vector<int> blocked_alpha_orbitals;
      std::vector<int> blocked_beta_orbitals;
      blocked_alpha_orbitals.reserve(xmvb::to_size(n_blocked_alpha));
      blocked_beta_orbitals.reserve(xmvb::to_size(n_blocked_beta));
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

      basis.states.push_back(
          build_pf_state_from_alpha_beta_block(
              n_active_orbitals,
              partition.n_singlet_pairs,
              alpha_beta_block,
              blocked_alpha_orbitals,
              blocked_beta_orbitals));
    }
  }

  return basis;
}

PfBasisData build_structure_pf_basis(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfBasisFactoryOptions& options) {
  PfFixedMsBasisFactoryOptions fixed_ms_options;
  fixed_ms_options.n_structures = options.n_states;
  fixed_ms_options.seed = options.seed;
  fixed_ms_options.pairing_noise = options.pairing_noise;
  return build_fixed_ms_pf_basis(
      input,
      raw_structure_data,
      fixed_ms_options);
}

}  // namespace xmvb::pfaffian_vbscf
