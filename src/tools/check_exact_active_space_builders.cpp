#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
  bool skip_active_eri = false;
};

void print_usage() {
  std::cerr
      << "usage: check_exact_active_space_builders <input.xmi>"
      << " [--standard-two-electron-mode exact|auto]"
      << " [--skip-active-eri true|false]\n";
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
    if (name == "--skip-active-eri") {
      if (value == "true" || value == "1") {
        options.skip_active_eri = true;
      } else if (value == "false" || value == "0") {
        options.skip_active_eri = false;
      } else {
        throw std::invalid_argument("invalid --skip-active-eri value: " + value);
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

std::vector<double> build_dense_active_coefficients(
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (auxiliary_orbital_matrix.size() !=
      n_basis_functions * n_basis_functions) {
    throw std::invalid_argument("auxiliary orbital matrix size mismatch");
  }

  const Eigen::Map<const Matrix> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);

  std::vector<double> dense_active_coefficients(
      n_basis_functions * n_active_orbitals,
      0.0);
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    for (int active_orbital_index = 0;
         active_orbital_index < n_active_orbitals;
         ++active_orbital_index) {
      dense_active_coefficients[basis_function_index * n_active_orbitals +
                                active_orbital_index] =
          active_auxiliary_orbitals(basis_function_index, active_orbital_index);
    }
  }
  return dense_active_coefficients;
}

struct UniquePermutationSet {
  std::array<std::array<int, 4>, 8> values{};
  int count = 0;
};

UniquePermutationSet enumerate_unique_symmetry_permutations(
    int i,
    int j,
    int k,
    int l) {
  const std::array<std::array<int, 4>, 8> permutations = {{
      {{i, j, k, l}},
      {{j, i, k, l}},
      {{i, j, l, k}},
      {{j, i, l, k}},
      {{k, l, i, j}},
      {{l, k, i, j}},
      {{k, l, j, i}},
      {{l, k, j, i}},
  }};

  UniquePermutationSet unique_permutations;
  for (const auto& permutation : permutations) {
    bool already_seen = false;
    for (int permutation_index = 0;
         permutation_index < unique_permutations.count;
         ++permutation_index) {
      if (unique_permutations.values[permutation_index] == permutation) {
        already_seen = true;
        break;
      }
    }
    if (!already_seen) {
      unique_permutations.values[unique_permutations.count] = permutation;
      ++unique_permutations.count;
    }
  }
  return unique_permutations;
}

xmvb::vb::AoEffectiveOneElectronResult build_reference_ao_effective_one_electron(
    const std::vector<double>& inactive_density_matrix,
    const xmvb::vb::AoIntegralInput& ao_integral_input) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
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

xmvb::vb::ActiveSpaceTwoElectronResult build_reference_active_space_two_electron(
    const xmvb::vb::AoIntegralInput& ao_integral_input,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space dimensions must be positive");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value size mismatch");
  }

  const auto dense_active_coefficients =
      build_dense_active_coefficients(
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const std::size_t active_tensor_size =
      n_active_orbitals *
      n_active_orbitals *
      n_active_orbitals *
      n_active_orbitals;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<std::vector<double>> partial_tensors(
      n_threads,
      std::vector<double>(active_tensor_size, 0.0));

  const auto active_tensor_index =
      [n_active_orbitals](int p, int q, int r, int s) -> std::size_t {
    return (((p * n_active_orbitals + q) * n_active_orbitals + r) *
            n_active_orbitals) +
        s;
  };

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_tensor = partial_tensors[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(
                               ao_integral_input.ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const double ao_integral_value =
          ao_integral_input.ao_two_electron_integral_values[integral_index];
      const int i = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4];
      const int j =
          ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 1];
      const int k =
          ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 2];
      const int l =
          ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 3];
      if (i < 0 || i >= n_basis_functions ||
          j < 0 || j >= n_basis_functions ||
          k < 0 || k >= n_basis_functions ||
          l < 0 || l >= n_basis_functions) {
        continue;
      }

      const auto permutations =
          enumerate_unique_symmetry_permutations(i, j, k, l);
      for (int permutation_index = 0;
           permutation_index < permutations.count;
           ++permutation_index) {
        const auto& permutation =
            permutations.values[permutation_index];
        const int a = permutation[0];
        const int b = permutation[1];
        const int c = permutation[2];
        const int d = permutation[3];

        const double* coeff_a =
            dense_active_coefficients.data() + a * n_active_orbitals;
        const double* coeff_b =
            dense_active_coefficients.data() + b * n_active_orbitals;
        const double* coeff_c =
            dense_active_coefficients.data() + c * n_active_orbitals;
        const double* coeff_d =
            dense_active_coefficients.data() + d * n_active_orbitals;
        for (int p = 0; p < n_active_orbitals; ++p) {
          if (coeff_a[p] == 0.0) {
            continue;
          }
          for (int q = 0; q < n_active_orbitals; ++q) {
            if (coeff_b[q] == 0.0) {
              continue;
            }
            const double coefficient_ab = ao_integral_value * coeff_a[p] * coeff_b[q];
            for (int r = 0; r < n_active_orbitals; ++r) {
              if (coeff_c[r] == 0.0) {
                continue;
              }
              const double coefficient_abc = coefficient_ab * coeff_c[r];
              for (int s = 0; s < n_active_orbitals; ++s) {
                if (coeff_d[s] == 0.0) {
                  continue;
                }
                local_tensor[active_tensor_index(p, q, r, s)] +=
                    coefficient_abc * coeff_d[s];
              }
            }
          }
        }
      }
    }
  }

  std::vector<double> active_two_electron_tensor(active_tensor_size, 0.0);
  for (const auto& partial_tensor : partial_tensors) {
    for (std::size_t tensor_index = 0;
         tensor_index < active_two_electron_tensor.size();
         ++tensor_index) {
      active_two_electron_tensor[tensor_index] += partial_tensor[tensor_index];
    }
  }

  const std::size_t packed_size =
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
          n_active_orbitals - 1,
          n_active_orbitals - 1,
          n_active_orbitals - 1,
          n_active_orbitals - 1) +
      1;
  std::vector<double> packed_active_two_electron_integrals(packed_size, 0.0);
  for (int p = 0; p < n_active_orbitals; ++p) {
    for (int q = 0; q <= p; ++q) {
      for (int r = 0; r <= p; ++r) {
        const int s_upper = (r == p) ? q : r;
        for (int s = 0; s <= s_upper; ++s) {
          const int packed_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  p,
                  q,
                  r,
                  s);
          packed_active_two_electron_integrals[packed_index] =
              active_two_electron_tensor[active_tensor_index(p, q, r, s)];
        }
      }
    }
  }

  xmvb::vb::ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals =
      std::move(packed_active_two_electron_integrals);
  return result;
}

std::vector<double> build_reference_active_space_one_electron(
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const Eigen::Map<const Matrix> ao_f11_matrix(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Matrix h1e_act =
      active_auxiliary_orbitals.transpose() * ao_f11_matrix *
      active_auxiliary_orbitals;
  return std::vector<double>(h1e_act.data(), h1e_act.data() + h1e_act.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
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
    const auto reference_h1e_act =
        build_reference_active_space_one_electron(
            reference_ao_result.ao_effective_h1e,
            orbital_result.auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);

    std::cout << std::setprecision(15);
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
                     reference_h1e_act)
              << '\n';
    if (!options.skip_active_eri) {
      xmvb::vb::ActiveSpaceTwoElectronBuilder active_eri_builder;
      const auto current_active_eri_result =
          active_eri_builder.build(
              input.ao_integral_input,
              orbital_result,
              n_active_orbitals);
      const auto reference_active_eri_result =
          build_reference_active_space_two_electron(
              input.ao_integral_input,
              orbital_result.auxiliary_orbital_matrix,
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals);
      std::cout << "max_abs_packed_active_eri_diff = "
                << max_abs_difference(
                       current_active_eri_result.packed_active_two_electron_integrals,
                       reference_active_eri_result.packed_active_two_electron_integrals)
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
