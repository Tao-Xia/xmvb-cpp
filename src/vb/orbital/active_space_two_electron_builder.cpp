#include "vb/orbital/active_space_two_electron_builder.hpp"

#include <array>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct SparseActiveCoefficient {
  int active_orbital_index = 0;
  double value = 0.0;
};

struct UniquePermutationSet {
  std::array<std::array<int, 4>, 8> values;
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
      if (unique_permutations.values[static_cast<std::size_t>(permutation_index)] ==
          permutation) {
        already_seen = true;
        break;
      }
    }
    if (!already_seen) {
      unique_permutations.values[static_cast<std::size_t>(unique_permutations.count)] =
          permutation;
      ++unique_permutations.count;
    }
  }
  return unique_permutations;
}

std::vector<std::vector<SparseActiveCoefficient>> build_sparse_active_coefficients_by_basis(
    const Eigen::Ref<const Matrix>& active_auxiliary_orbitals,
    int n_basis_functions,
    int n_active_orbitals) {
  std::vector<std::vector<SparseActiveCoefficient>> coefficients_by_basis(
      static_cast<std::size_t>(n_basis_functions));
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    auto& coefficients =
        coefficients_by_basis[static_cast<std::size_t>(basis_function_index)];
    for (int active_orbital_index = 0;
         active_orbital_index < n_active_orbitals;
         ++active_orbital_index) {
      const double coefficient =
          active_auxiliary_orbitals(basis_function_index, active_orbital_index);
      if (coefficient == 0.0) {
        continue;
      }
      coefficients.push_back({active_orbital_index, coefficient});
    }
  }
  return coefficients_by_basis;
}

std::vector<std::vector<SparseActiveCoefficient>> build_sparse_active_coefficients_by_basis(
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions) {
  if (orbital_preparation_result.active_sparse_row_offsets.size() !=
      static_cast<std::size_t>(n_basis_functions) + 1) {
    throw std::invalid_argument("active_sparse_row_offsets size mismatch");
  }
  if (orbital_preparation_result.active_sparse_orbital_indices.size() !=
      orbital_preparation_result.active_sparse_values.size()) {
    throw std::invalid_argument("active sparse index/value sizes are inconsistent");
  }

  std::vector<std::vector<SparseActiveCoefficient>> coefficients_by_basis(
      static_cast<std::size_t>(n_basis_functions));
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    const int begin =
        orbital_preparation_result.active_sparse_row_offsets[static_cast<std::size_t>(basis_function_index)];
    const int end =
        orbital_preparation_result.active_sparse_row_offsets[static_cast<std::size_t>(basis_function_index + 1)];
    auto& coefficients =
        coefficients_by_basis[static_cast<std::size_t>(basis_function_index)];
    coefficients.reserve(static_cast<std::size_t>(std::max(0, end - begin)));
    for (int offset = begin; offset < end; ++offset) {
      coefficients.push_back({
          orbital_preparation_result.active_sparse_orbital_indices[static_cast<std::size_t>(offset)],
          orbital_preparation_result.active_sparse_values[static_cast<std::size_t>(offset)]});
    }
  }
  return coefficients_by_basis;
}

}  // namespace

ActiveSpaceTwoElectronResult ActiveSpaceTwoElectronBuilder::build(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron dimensions must be positive");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  const std::size_t auxiliary_matrix_size =
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
  if (auxiliary_orbital_matrix.size() != auxiliary_matrix_size) {
    throw std::invalid_argument("auxiliary orbital matrix size mismatch");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const Eigen::Map<const Matrix> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const auto sparse_coefficients_by_basis = build_sparse_active_coefficients_by_basis(
      active_auxiliary_orbitals,
      n_basis_functions,
      n_active_orbitals);
  const std::size_t active_tensor_size =
      static_cast<std::size_t>(n_active_orbitals) *
      n_active_orbitals *
      n_active_orbitals *
      n_active_orbitals;
  std::vector<double> active_two_electron_tensor(active_tensor_size, 0.0);

  auto active_tensor_index =
      [n_active_orbitals](int p, int q, int r, int s) -> std::size_t {
    return (((static_cast<std::size_t>(p) * n_active_orbitals + q) * n_active_orbitals + r) *
            n_active_orbitals) +
           s;
  };

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<std::vector<double>> partial_tensors(
      static_cast<std::size_t>(n_threads),
      std::vector<double>(active_tensor_size, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_active_two_electron_tensor =
        partial_tensors[static_cast<std::size_t>(thread_index)];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = static_cast<std::size_t>(integral_offset);
      const double ao_integral_value = ao_two_electron_integral_values[integral_index];
      const int i = ao_two_electron_integral_indices[integral_index * 4];
      const int j = ao_two_electron_integral_indices[integral_index * 4 + 1];
      const int k = ao_two_electron_integral_indices[integral_index * 4 + 2];
      const int l = ao_two_electron_integral_indices[integral_index * 4 + 3];

      if (i < 0 || i >= n_basis_functions ||
          j < 0 || j >= n_basis_functions ||
          k < 0 || k >= n_basis_functions ||
          l < 0 || l >= n_basis_functions) {
        continue;
      }

      const auto permutations = enumerate_unique_symmetry_permutations(i, j, k, l);
      for (int permutation_index = 0;
           permutation_index < permutations.count;
           ++permutation_index) {
        const auto& permutation =
            permutations.values[static_cast<std::size_t>(permutation_index)];
        const int a = permutation[0];
        const int b = permutation[1];
        const int c = permutation[2];
        const int d = permutation[3];

        const auto& coefficients_a = sparse_coefficients_by_basis[static_cast<std::size_t>(a)];
        const auto& coefficients_b = sparse_coefficients_by_basis[static_cast<std::size_t>(b)];
        const auto& coefficients_c = sparse_coefficients_by_basis[static_cast<std::size_t>(c)];
        const auto& coefficients_d = sparse_coefficients_by_basis[static_cast<std::size_t>(d)];
        for (const auto& coefficient_a : coefficients_a) {
          for (const auto& coefficient_b : coefficients_b) {
            const double coefficient_ab =
                ao_integral_value * coefficient_a.value * coefficient_b.value;
            for (const auto& coefficient_c : coefficients_c) {
              const double coefficient_abc = coefficient_ab * coefficient_c.value;
              const std::size_t pqr_index =
                  (static_cast<std::size_t>(coefficient_a.active_orbital_index) *
                       n_active_orbitals +
                   coefficient_b.active_orbital_index) *
                      n_active_orbitals +
                  coefficient_c.active_orbital_index;
              for (const auto& coefficient_d : coefficients_d) {
                local_active_two_electron_tensor[pqr_index * n_active_orbitals +
                                                 coefficient_d.active_orbital_index] +=
                    coefficient_abc * coefficient_d.value;
              }
            }
          }
        }
      }
    }
  }

  for (const auto& partial_tensor : partial_tensors) {
    for (std::size_t tensor_index = 0;
         tensor_index < active_two_electron_tensor.size();
         ++tensor_index) {
      active_two_electron_tensor[tensor_index] += partial_tensor[tensor_index];
    }
  }

  const std::size_t packed_size =
      static_cast<std::size_t>(
          TwoElectronIndexer::two_electron_storage_index(
              n_active_orbitals - 1,
              n_active_orbitals - 1,
              n_active_orbitals - 1,
              n_active_orbitals - 1)) +
      1;
  std::vector<double> packed_active_two_electron_integrals(packed_size, 0.0);

  for (int p = 0; p < n_active_orbitals; ++p) {
    for (int q = 0; q <= p; ++q) {
      for (int r = 0; r <= p; ++r) {
        const int s_upper = (r == p) ? q : r;
        for (int s = 0; s <= s_upper; ++s) {
          const int packed_index =
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s);
          packed_active_two_electron_integrals[static_cast<std::size_t>(packed_index)] =
              active_two_electron_tensor[active_tensor_index(p, q, r, s)];
        }
      }
    }
  }

  ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals = std::move(packed_active_two_electron_integrals);
  return result;
}

ActiveSpaceTwoElectronResult ActiveSpaceTwoElectronBuilder::build(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron dimensions must be positive");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  if (orbital_preparation_result.active_sparse_row_offsets.size() !=
      static_cast<std::size_t>(n_basis_functions) + 1) {
    throw std::invalid_argument("active_sparse_row_offsets size mismatch");
  }
  if (orbital_preparation_result.active_sparse_orbital_indices.size() !=
      orbital_preparation_result.active_sparse_values.size()) {
    throw std::invalid_argument("active sparse index/value sizes are inconsistent");
  }
  const std::vector<int>& active_sparse_row_offsets =
      orbital_preparation_result.active_sparse_row_offsets;
  const std::vector<int>& active_sparse_orbital_indices =
      orbital_preparation_result.active_sparse_orbital_indices;
  const std::vector<double>& active_sparse_values =
      orbital_preparation_result.active_sparse_values;
  const std::size_t active_tensor_size =
      static_cast<std::size_t>(n_active_orbitals) *
      n_active_orbitals *
      n_active_orbitals *
      n_active_orbitals;
  std::vector<double> active_two_electron_tensor(active_tensor_size, 0.0);

  auto active_tensor_index =
      [n_active_orbitals](int p, int q, int r, int s) -> std::size_t {
    return (((static_cast<std::size_t>(p) * n_active_orbitals + q) * n_active_orbitals + r) *
            n_active_orbitals) +
           s;
  };

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<std::vector<double>> partial_tensors(
      static_cast<std::size_t>(n_threads),
      std::vector<double>(active_tensor_size, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_active_two_electron_tensor =
        partial_tensors[static_cast<std::size_t>(thread_index)];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = static_cast<std::size_t>(integral_offset);
      const double ao_integral_value = ao_two_electron_integral_values[integral_index];
      const int i = ao_two_electron_integral_indices[integral_index * 4];
      const int j = ao_two_electron_integral_indices[integral_index * 4 + 1];
      const int k = ao_two_electron_integral_indices[integral_index * 4 + 2];
      const int l = ao_two_electron_integral_indices[integral_index * 4 + 3];

      if (i < 0 || i >= n_basis_functions ||
          j < 0 || j >= n_basis_functions ||
          k < 0 || k >= n_basis_functions ||
          l < 0 || l >= n_basis_functions) {
        continue;
      }

      const auto permutations = enumerate_unique_symmetry_permutations(i, j, k, l);
      for (int permutation_index = 0;
           permutation_index < permutations.count;
           ++permutation_index) {
        const auto& permutation =
            permutations.values[static_cast<std::size_t>(permutation_index)];
        const int a = permutation[0];
        const int b = permutation[1];
        const int c = permutation[2];
        const int d = permutation[3];
        const int begin_a = active_sparse_row_offsets[static_cast<std::size_t>(a)];
        const int end_a = active_sparse_row_offsets[static_cast<std::size_t>(a + 1)];
        const int begin_b = active_sparse_row_offsets[static_cast<std::size_t>(b)];
        const int end_b = active_sparse_row_offsets[static_cast<std::size_t>(b + 1)];
        const int begin_c = active_sparse_row_offsets[static_cast<std::size_t>(c)];
        const int end_c = active_sparse_row_offsets[static_cast<std::size_t>(c + 1)];
        const int begin_d = active_sparse_row_offsets[static_cast<std::size_t>(d)];
        const int end_d = active_sparse_row_offsets[static_cast<std::size_t>(d + 1)];

        for (int offset_a = begin_a; offset_a < end_a; ++offset_a) {
          const int p =
              active_sparse_orbital_indices[static_cast<std::size_t>(offset_a)];
          const double value_a = active_sparse_values[static_cast<std::size_t>(offset_a)];
          for (int offset_b = begin_b; offset_b < end_b; ++offset_b) {
            const int q =
                active_sparse_orbital_indices[static_cast<std::size_t>(offset_b)];
            const double coefficient_ab =
                ao_integral_value * value_a *
                active_sparse_values[static_cast<std::size_t>(offset_b)];
            for (int offset_c = begin_c; offset_c < end_c; ++offset_c) {
              const int r =
                  active_sparse_orbital_indices[static_cast<std::size_t>(offset_c)];
              const double coefficient_abc =
                  coefficient_ab * active_sparse_values[static_cast<std::size_t>(offset_c)];
              const std::size_t pqr_index =
                  (static_cast<std::size_t>(p) * n_active_orbitals + q) *
                      n_active_orbitals +
                  r;
              for (int offset_d = begin_d; offset_d < end_d; ++offset_d) {
                const int s =
                    active_sparse_orbital_indices[static_cast<std::size_t>(offset_d)];
                local_active_two_electron_tensor[pqr_index * n_active_orbitals + s] +=
                    coefficient_abc * active_sparse_values[static_cast<std::size_t>(offset_d)];
              }
            }
          }
        }
      }
    }
  }

  for (const auto& partial_tensor : partial_tensors) {
    for (std::size_t tensor_index = 0;
         tensor_index < active_two_electron_tensor.size();
         ++tensor_index) {
      active_two_electron_tensor[tensor_index] += partial_tensor[tensor_index];
    }
  }

  const std::size_t packed_size =
      static_cast<std::size_t>(
          TwoElectronIndexer::two_electron_storage_index(
              n_active_orbitals - 1,
              n_active_orbitals - 1,
              n_active_orbitals - 1,
              n_active_orbitals - 1)) +
      1;
  std::vector<double> packed_active_two_electron_integrals(packed_size, 0.0);

  for (int p = 0; p < n_active_orbitals; ++p) {
    for (int q = 0; q <= p; ++q) {
      for (int r = 0; r <= p; ++r) {
        const int s_upper = (r == p) ? q : r;
        for (int s = 0; s <= s_upper; ++s) {
          const int packed_index =
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s);
          packed_active_two_electron_integrals[static_cast<std::size_t>(packed_index)] =
              active_two_electron_tensor[active_tensor_index(p, q, r, s)];
        }
      }
    }
  }

  ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals = std::move(packed_active_two_electron_integrals);
  return result;
}

}  // namespace xmvb::vb
