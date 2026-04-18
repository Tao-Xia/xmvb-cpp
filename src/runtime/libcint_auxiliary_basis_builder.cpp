#include "runtime/libcint_auxiliary_basis_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

extern "C" {
#include "cint.h"
}

namespace xmvb::vb {

namespace {

struct AuxiliaryShellSpecification {
  int atom_index = 0;
  int angular_momentum = 0;
  double exponent = 0.0;
};

void validate_primary_input(const LibcintInput& input) {
  if (input.n_atoms < 0) {
    throw std::invalid_argument("LibcintInput n_atoms must be non-negative");
  }
  if (input.n_shells <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one shell");
  }
  if (input.atm.size() != xmvb::to_size(input.n_atoms) * ATM_SLOTS) {
    throw std::invalid_argument("LibcintInput atm size mismatch");
  }
  if (input.bas.size() != xmvb::to_size(input.n_shells) * BAS_SLOTS) {
    throw std::invalid_argument("LibcintInput bas size mismatch");
  }
  if (input.env.empty()) {
    throw std::invalid_argument("LibcintInput env must not be empty");
  }
}

int shell_cartesian_count(int angular_momentum) {
  return (angular_momentum + 1) * (angular_momentum + 2) / 2;
}

void append_generated_shell_block(
    std::vector<AuxiliaryShellSpecification>& shell_specs,
    int atom_index,
    const std::vector<int>& angular_momenta,
    double first_exponent,
    int n_exponents,
    double ratio,
    double prefactor) {
  if (!(first_exponent > 0.0)) {
    throw std::invalid_argument("generated auxiliary exponent must be positive");
  }
  if (n_exponents < 2) {
    throw std::invalid_argument("generated auxiliary shell block requires at least 2 exponents");
  }

  double exponent_zero = first_exponent;
  double exponent_one = prefactor * exponent_zero;
  double exponent_two = exponent_zero / ratio;
  for (int angular_momentum : angular_momenta) {
    shell_specs.push_back({atom_index, angular_momentum, exponent_one});
  }
  for (int angular_momentum : angular_momenta) {
    shell_specs.push_back({atom_index, angular_momentum, exponent_two});
  }
  double current_exponent = exponent_two;
  for (int exponent_index = 2; exponent_index < n_exponents; ++exponent_index) {
    current_exponent /= ratio;
    for (int angular_momentum : angular_momenta) {
      shell_specs.push_back({atom_index, angular_momentum, current_exponent});
    }
  }
}

}  // namespace

LibcintInput LibcintAuxiliaryBasisBuilder::build(
    const LibcintInput& primary_input,
    const LibcintAuxiliaryBasisBuilderOptions& options) const {
  validate_primary_input(primary_input);
  if (options.level < 2 || options.level > 4) {
    throw std::invalid_argument("GEN-A_n level must satisfy 2 <= n <= 4");
  }

  const double ratio = 6.0 - static_cast<double>(options.level);
  const double prefactor =
      1.0 + static_cast<double>(options.level) /
                (12.0 - 2.0 * static_cast<double>(options.level));

  std::vector<double> max_exponents(xmvb::to_size(primary_input.n_atoms), 0.0);
  std::vector<double> min_exponents(
      xmvb::to_size(primary_input.n_atoms),
      1.0e300);
  for (int shell_index = 0; shell_index < primary_input.n_shells; ++shell_index) {
    const std::size_t shell_offset = xmvb::to_size(shell_index) * BAS_SLOTS;
    const int atom_index = primary_input.bas[shell_offset + ATOM_OF];
    const int n_primitives = primary_input.bas[shell_offset + NPRIM_OF];
    const int exponent_offset = primary_input.bas[shell_offset + PTR_EXP];
    if (atom_index < 0 || atom_index >= primary_input.n_atoms) {
      throw std::invalid_argument("primary basis shell atom index out of range");
    }
    if (n_primitives <= 0) {
      throw std::invalid_argument("primary basis shell must contain at least one primitive");
    }
    for (int primitive_index = 0; primitive_index < n_primitives; ++primitive_index) {
      const double exponent =
          primary_input.env[xmvb::to_size(exponent_offset + primitive_index)];
      if (!(exponent > 0.0)) {
        throw std::invalid_argument("primary basis exponent must be positive");
      }
      const std::size_t atom_offset = xmvb::to_size(atom_index);
      max_exponents[atom_offset] = std::max(max_exponents[atom_offset], exponent);
      min_exponents[atom_offset] = std::min(min_exponents[atom_offset], exponent);
    }
  }

  std::vector<AuxiliaryShellSpecification> shell_specs;
  shell_specs.reserve(xmvb::to_size(primary_input.n_atoms) * 24);
  for (int atom_index = 0; atom_index < primary_input.n_atoms; ++atom_index) {
    const std::size_t atom_offset = xmvb::to_size(atom_index);
    const double max_exponent = max_exponents[atom_offset];
    const double min_exponent = min_exponents[atom_offset];
    if (!(max_exponent > 0.0) || !(min_exponent > 0.0) || max_exponent < min_exponent) {
      throw std::invalid_argument("failed to infer primary-basis exponent range per atom");
    }

    int n_s_exponents = static_cast<int>(
        std::log(max_exponent / min_exponent) / std::log(ratio) + 0.5);
    n_s_exponents = std::min(std::max(n_s_exponents, 3), 5);
    const double s_seed_exponent =
        2.0 * min_exponent * std::pow(ratio, static_cast<double>(n_s_exponents - 1));
    append_generated_shell_block(
        shell_specs,
        atom_index,
        {0},
        s_seed_exponent,
        n_s_exponents,
        ratio,
        prefactor);

    const int atomic_charge =
        primary_input.atm[xmvb::to_size(atom_index) * ATM_SLOTS + CHARGE_OF];
    if (atomic_charge <= 2) {
      continue;
    }

    double tail_exponent = s_seed_exponent / std::pow(ratio, static_cast<double>(n_s_exponents));
    int n_spd_exponents = std::max(n_s_exponents - 1, 3);
    append_generated_shell_block(
        shell_specs,
        atom_index,
        {0, 1, 2},
        tail_exponent,
        n_spd_exponents,
        ratio,
        prefactor);

    if (!options.use_star) {
      continue;
    }

    tail_exponent /= std::pow(ratio, static_cast<double>(n_spd_exponents));
    const int n_spdf_exponents = std::max(n_spd_exponents - 1, 3);
    append_generated_shell_block(
        shell_specs,
        atom_index,
        {0, 1, 2, 3},
        tail_exponent,
        n_spdf_exponents,
        ratio,
        prefactor);
  }

  if (shell_specs.empty()) {
    throw std::runtime_error("generated auxiliary basis is empty");
  }

  const std::size_t primary_env_size = primary_input.env.size();
  const int coefficient_offset = static_cast<int>(primary_env_size + shell_specs.size());
  std::vector<int> auxiliary_basis;
  auxiliary_basis.reserve(shell_specs.size() * BAS_SLOTS);
  std::vector<int> auxiliary_basidx;
  auxiliary_basidx.reserve(shell_specs.size() * 2);
  std::vector<double> auxiliary_env;
  auxiliary_env.reserve(primary_env_size + shell_specs.size() + 1);
  auxiliary_env.insert(
      auxiliary_env.end(),
      primary_input.env.begin(),
      primary_input.env.end());

  int ao_offset = 0;
  for (const auto& shell_spec : shell_specs) {
    const int exponent_offset = static_cast<int>(auxiliary_env.size());
    auxiliary_env.push_back(shell_spec.exponent);

    auxiliary_basis.push_back(shell_spec.atom_index);
    auxiliary_basis.push_back(shell_spec.angular_momentum);
    auxiliary_basis.push_back(1);
    auxiliary_basis.push_back(1);
    auxiliary_basis.push_back(0);
    auxiliary_basis.push_back(exponent_offset);
    auxiliary_basis.push_back(coefficient_offset);
    auxiliary_basis.push_back(0);

    auxiliary_basidx.push_back(ao_offset);
    const int ao_count = shell_cartesian_count(shell_spec.angular_momentum);
    auxiliary_basidx.push_back(ao_count);
    ao_offset += ao_count;
  }
  auxiliary_env.push_back(1.0);

  LibcintInput auxiliary_input;
  auxiliary_input.n_atoms = primary_input.n_atoms;
  auxiliary_input.n_shells = static_cast<int>(shell_specs.size());
  auxiliary_input.n_gaussian_primitives = static_cast<int>(shell_specs.size());
  auxiliary_input.atm.assign(primary_input.atm.begin(), primary_input.atm.end());
  auxiliary_input.bas = std::move(auxiliary_basis);
  auxiliary_input.basidx = std::move(auxiliary_basidx);
  auxiliary_input.env = std::move(auxiliary_env);
  return auxiliary_input;
}

}  // namespace xmvb::vb
