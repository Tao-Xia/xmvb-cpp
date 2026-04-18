#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
  xmvb::vb::AoIntegralSource ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
  xmvb::vb::OrbitalGuessSource orbital_guess_source =
      xmvb::vb::OrbitalGuessSource::Cpp;
};

void print_usage() {
  std::cerr << "usage: check_exact_ao_h1e_builder <input.xmi>"
               " [--standard-two-electron-mode exact|auto]"
               " [--ao-integral-source legacy|libcint_cpp]"
               " [--orbital-guess-source legacy|cpp]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string name = argv[argument_index];
    const std::string value = argv[argument_index + 1];
    if (name == "--standard-two-electron-mode") {
      if (value == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (value == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else {
        throw std::invalid_argument(
            "invalid --standard-two-electron-mode value: " + value);
      }
      continue;
    }
    if (name == "--ao-integral-source") {
      if (value == "legacy") {
        options.ao_integral_source = xmvb::vb::AoIntegralSource::LegacyRuntime;
      } else if (value == "libcint_cpp") {
        options.ao_integral_source =
            xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
      } else {
        throw std::invalid_argument(
            "invalid --ao-integral-source value: " + value);
      }
      continue;
    }
    if (name == "--orbital-guess-source") {
      if (value == "legacy") {
        options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
      } else if (value == "cpp") {
        options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
      } else {
        throw std::invalid_argument(
            "invalid --orbital-guess-source value: " + value);
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  return options;
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch");
  }
  double max_abs_diff = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_abs_diff = std::max(max_abs_diff, std::abs(left[index] - right[index]));
  }
  return max_abs_diff;
}

double compute_one_electron_reference_energy(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  const Eigen::Map<const Matrix> inactive_density(
      inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> effective_h1e(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> core_hamiltonian(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return (inactive_density.array() *
          (effective_h1e.array() + core_hamiltonian.array())).sum();
}

xmvb::vb::AoEffectiveOneElectronResult build_reference_ao_effective_one_electron(
    const std::vector<double>& inactive_density_matrix,
    const xmvb::vb::AoIntegralInput& ao_integral_input) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size ||
      ao_integral_input.ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("AO matrix size mismatch");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value size mismatch");
  }

  const Eigen::Map<const Matrix> inactive_density(
      inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> core_hamiltonian(
      ao_integral_input.ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  Matrix g11 = Matrix::Zero(n_basis_functions, n_basis_functions);

  for (std::size_t integral_index = 0;
       integral_index < ao_integral_input.ao_two_electron_integral_values.size();
       ++integral_index) {
    double two_electron_value =
        ao_integral_input.ao_two_electron_integral_values[integral_index];
    const int i = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4];
    const int j = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 1];
    const int k = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 2];
    const int l = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 3];
    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      throw std::invalid_argument("AO two-electron index out of range");
    }

    if (i == j) {
      two_electron_value *= 0.5;
    }
    if (k == l) {
      two_electron_value *= 0.5;
    }
    if (i == k && j == l) {
      two_electron_value *= 0.5;
    }

    const double a0 = inactive_density(i, j) * two_electron_value * 4.0;
    const double a1 = inactive_density(k, l) * two_electron_value * 4.0;
    g11(i, j) += a1;
    g11(k, l) += a0;
    g11(i, k) -= inactive_density(l, j) * two_electron_value;
    g11(j, l) -= inactive_density(k, i) * two_electron_value;
    g11(i, l) -= inactive_density(k, j) * two_electron_value;
    g11(j, k) -= inactive_density(l, i) * two_electron_value;
  }

  for (int row = 0; row < n_basis_functions; ++row) {
    for (int column = 0; column <= row; ++column) {
      g11(row, column) = g11(row, column) + g11(column, row);
      g11(column, row) = g11(row, column);
    }
  }

  xmvb::vb::AoEffectiveOneElectronResult result;
  result.ao_coulomb_exchange_matrix.assign(
      g11.data(),
      g11.data() + g11.size());
  Matrix ao_effective_h1e = core_hamiltonian;
  ao_effective_h1e.noalias() += g11;
  result.ao_effective_h1e.assign(
      ao_effective_h1e.data(),
      ao_effective_h1e.data() + ao_effective_h1e.size());
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    load_options.ao_integral_source = options.ao_integral_source;
    load_options.orbital_guess_source = options.orbital_guess_source;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const auto& input = load_result.input;

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    const auto orbital_result =
        orbital_preparer.prepare(input.orbital_preparation_input);
    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) /
        2;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;

    xmvb::vb::AoEffectiveOneElectronBuilder ao_builder;
    const auto current_ao_result =
        ao_builder.build(
            orbital_result.inactive_density_matrix,
            input.ao_integral_input);
    const auto reference_ao_result =
        build_reference_ao_effective_one_electron(
            orbital_result.inactive_density_matrix,
            input.ao_integral_input);

    xmvb::vb::ActiveSpaceOneElectronBuilder active_h1e_builder;
    const auto current_active_h1e_result =
        active_h1e_builder.build(
            current_ao_result.ao_effective_h1e,
            orbital_result.auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const auto reference_active_h1e_result =
        active_h1e_builder.build(
            reference_ao_result.ao_effective_h1e,
            orbital_result.auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);

    const double current_reference_energy =
        compute_one_electron_reference_energy(
            orbital_result.inactive_density_matrix,
            current_ao_result.ao_effective_h1e,
            input.ao_integral_input.ao_core_hamiltonian_matrix.vector(),
            n_basis_functions);
    const double reference_reference_energy =
        compute_one_electron_reference_energy(
            orbital_result.inactive_density_matrix,
            reference_ao_result.ao_effective_h1e,
            input.ao_integral_input.ao_core_hamiltonian_matrix.vector(),
            n_basis_functions);

    std::cout << std::setprecision(15);
    std::cout << "ao_integral_source = "
              << xmvb::vb::ao_integral_source_name(load_result.ao_integral_source)
              << '\n';
    std::cout << "orbital_guess_source = "
              << xmvb::vb::orbital_guess_source_name(load_result.orbital_guess_source)
              << '\n';
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     load_result.standard_two_electron_mode)
              << '\n';
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "n_ao_two_electron_integrals = "
              << input.ao_integral_input.ao_two_electron_integral_values.size() << '\n';
    std::cout << "max_abs_g11_diff = "
              << max_abs_difference(
                     current_ao_result.ao_coulomb_exchange_matrix,
                     reference_ao_result.ao_coulomb_exchange_matrix)
              << '\n';
    std::cout << "max_abs_ao_f11_diff = "
              << max_abs_difference(
                     current_ao_result.ao_effective_h1e,
                     reference_ao_result.ao_effective_h1e)
              << '\n';
    std::cout << "max_abs_active_h1e_diff = "
              << max_abs_difference(
                     current_active_h1e_result.h1e_act,
                     reference_active_h1e_result.h1e_act)
              << '\n';
    std::cout << "current_one_electron_reference_energy = "
              << current_reference_energy << '\n';
    std::cout << "reference_one_electron_reference_energy = "
              << reference_reference_energy << '\n';
    std::cout << "one_electron_reference_energy_diff = "
              << (current_reference_energy - reference_reference_energy) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
