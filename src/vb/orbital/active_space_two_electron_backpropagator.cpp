#include "vb/orbital/active_space_two_electron_backpropagator.hpp"

#include <array>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

using ColumnMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

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
    const Eigen::Ref<const ColumnMajorMatrixXd>& active_auxiliary_orbitals,
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

void accumulate_factor_gradient(
    int differentiated_position,
    int differentiated_basis_function,
    const std::vector<SparseActiveCoefficient>& coefficients_first,
    const std::vector<SparseActiveCoefficient>& coefficients_second,
    const std::vector<SparseActiveCoefficient>& coefficients_third,
    const std::vector<double>& packed_active_two_electron_gradient,
    double ao_integral_value,
    int n_basis_functions,
    int n_active_orbitals,
    std::vector<double>* active_auxiliary_gradient) {
  for (int active_orbital_index = 0;
       active_orbital_index < n_active_orbitals;
       ++active_orbital_index) {
    double coefficient_gradient = 0.0;
    for (const auto& coefficient_first : coefficients_first) {
      for (const auto& coefficient_second : coefficients_second) {
        for (const auto& coefficient_third : coefficients_third) {
          const double coefficient_product =
              ao_integral_value *
              coefficient_first.value *
              coefficient_second.value *
              coefficient_third.value;
          int p = coefficient_first.active_orbital_index;
          int q = coefficient_second.active_orbital_index;
          int r = coefficient_third.active_orbital_index;
          int s = active_orbital_index;
          switch (differentiated_position) {
            case 0:
              p = active_orbital_index;
              q = coefficient_first.active_orbital_index;
              r = coefficient_second.active_orbital_index;
              s = coefficient_third.active_orbital_index;
              break;
            case 1:
              p = coefficient_first.active_orbital_index;
              q = active_orbital_index;
              r = coefficient_second.active_orbital_index;
              s = coefficient_third.active_orbital_index;
              break;
            case 2:
              p = coefficient_first.active_orbital_index;
              q = coefficient_second.active_orbital_index;
              r = active_orbital_index;
              s = coefficient_third.active_orbital_index;
              break;
            case 3:
              p = coefficient_first.active_orbital_index;
              q = coefficient_second.active_orbital_index;
              r = coefficient_third.active_orbital_index;
              s = active_orbital_index;
              break;
            default:
              throw std::invalid_argument("differentiated_position must be in [0, 3]");
          }
          if (q > p || r > p) {
            continue;
          }
          const int s_upper = (r == p) ? q : r;
          if (s > s_upper) {
            continue;
          }
          const int packed_index = TwoElectronIndexer::two_electron_storage_index(
              p,
              q,
              r,
              s);
          coefficient_gradient +=
              packed_active_two_electron_gradient[static_cast<std::size_t>(packed_index)] *
              coefficient_product;
        }
      }
    }
    (*active_auxiliary_gradient)[static_cast<std::size_t>(active_orbital_index) *
                                     n_basis_functions +
                                 differentiated_basis_function] +=
        coefficient_gradient;
  }
}

}  // namespace

ActiveSpaceTwoElectronBackpropagationResult
ActiveSpaceTwoElectronBackpropagator::backpropagate(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron backprop dimensions must be positive");
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

  const Eigen::Map<const ColumnMajorMatrixXd> auxiliary_matrix(
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
  std::vector<double> active_auxiliary_gradient(
      static_cast<std::size_t>(n_basis_functions) * n_active_orbitals,
      0.0);

  for (std::size_t integral_index = 0;
       integral_index < ao_two_electron_integral_values.size();
       ++integral_index) {
    const double ao_integral_value = ao_two_electron_integral_values[integral_index];
    const int i = ao_two_electron_integral_indices[integral_index * 4];
    const int j = ao_two_electron_integral_indices[integral_index * 4 + 1];
    const int k = ao_two_electron_integral_indices[integral_index * 4 + 2];
    const int l = ao_two_electron_integral_indices[integral_index * 4 + 3];

    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      throw std::invalid_argument("AO two-electron index out of range");
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
      accumulate_factor_gradient(
          0,
          a,
          coefficients_b,
          coefficients_c,
          coefficients_d,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
      accumulate_factor_gradient(
          1,
          b,
          coefficients_a,
          coefficients_c,
          coefficients_d,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
      accumulate_factor_gradient(
          2,
          c,
          coefficients_a,
          coefficients_b,
          coefficients_d,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
      accumulate_factor_gradient(
          3,
          d,
          coefficients_a,
          coefficients_b,
          coefficients_c,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
    }
  }

  std::vector<double> auxiliary_gradient(
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions,
      0.0);
  for (int active_orbital_index = 0;
       active_orbital_index < n_active_orbitals;
       ++active_orbital_index) {
    const int column_index =
        n_inactive_doubly_occupied_orbitals + active_orbital_index;
    const double* source_column =
        active_auxiliary_gradient.data() +
        static_cast<std::size_t>(active_orbital_index) * n_basis_functions;
    double* target_column =
        auxiliary_gradient.data() +
        static_cast<std::size_t>(column_index) * n_basis_functions;
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      target_column[basis_function_index] = source_column[basis_function_index];
    }
  }

  ActiveSpaceTwoElectronBackpropagationResult result;
  result.auxiliary_orbital_gradient = std::move(auxiliary_gradient);
  return result;
}

ActiveSpaceTwoElectronBackpropagationResult
ActiveSpaceTwoElectronBackpropagator::backpropagate(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron backprop dimensions must be positive");
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
  std::vector<double> active_auxiliary_gradient(
      static_cast<std::size_t>(n_basis_functions) * n_active_orbitals,
      0.0);

  for (std::size_t integral_index = 0;
       integral_index < ao_two_electron_integral_values.size();
       ++integral_index) {
    const double ao_integral_value = ao_two_electron_integral_values[integral_index];
    const int i = ao_two_electron_integral_indices[integral_index * 4];
    const int j = ao_two_electron_integral_indices[integral_index * 4 + 1];
    const int k = ao_two_electron_integral_indices[integral_index * 4 + 2];
    const int l = ao_two_electron_integral_indices[integral_index * 4 + 3];

    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      throw std::invalid_argument("AO two-electron index out of range");
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

      std::vector<SparseActiveCoefficient> coefficients_b;
      std::vector<SparseActiveCoefficient> coefficients_c;
      std::vector<SparseActiveCoefficient> coefficients_d;
      coefficients_b.reserve(static_cast<std::size_t>(std::max(0, end_b - begin_b)));
      coefficients_c.reserve(static_cast<std::size_t>(std::max(0, end_c - begin_c)));
      coefficients_d.reserve(static_cast<std::size_t>(std::max(0, end_d - begin_d)));
      for (int offset_b = begin_b; offset_b < end_b; ++offset_b) {
        coefficients_b.push_back({
            active_sparse_orbital_indices[static_cast<std::size_t>(offset_b)],
            active_sparse_values[static_cast<std::size_t>(offset_b)]});
      }
      for (int offset_c = begin_c; offset_c < end_c; ++offset_c) {
        coefficients_c.push_back({
            active_sparse_orbital_indices[static_cast<std::size_t>(offset_c)],
            active_sparse_values[static_cast<std::size_t>(offset_c)]});
      }
      for (int offset_d = begin_d; offset_d < end_d; ++offset_d) {
        coefficients_d.push_back({
            active_sparse_orbital_indices[static_cast<std::size_t>(offset_d)],
            active_sparse_values[static_cast<std::size_t>(offset_d)]});
      }

      std::vector<SparseActiveCoefficient> coefficients_a;
      coefficients_a.reserve(static_cast<std::size_t>(std::max(0, end_a - begin_a)));
      for (int offset_a = begin_a; offset_a < end_a; ++offset_a) {
        coefficients_a.push_back({
            active_sparse_orbital_indices[static_cast<std::size_t>(offset_a)],
            active_sparse_values[static_cast<std::size_t>(offset_a)]});
      }

      accumulate_factor_gradient(
          0,
          a,
          coefficients_b,
          coefficients_c,
          coefficients_d,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
      accumulate_factor_gradient(
          1,
          b,
          coefficients_a,
          coefficients_c,
          coefficients_d,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
      accumulate_factor_gradient(
          2,
          c,
          coefficients_a,
          coefficients_b,
          coefficients_d,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
      accumulate_factor_gradient(
          3,
          d,
          coefficients_a,
          coefficients_b,
          coefficients_c,
          packed_active_two_electron_gradient,
          ao_integral_value,
          n_basis_functions,
          n_active_orbitals,
          &active_auxiliary_gradient);
    }
  }

  std::vector<double> auxiliary_gradient(
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions,
      0.0);
  for (int active_orbital_index = 0;
       active_orbital_index < n_active_orbitals;
       ++active_orbital_index) {
    const int column_index =
        n_inactive_doubly_occupied_orbitals + active_orbital_index;
    const double* source_column =
        active_auxiliary_gradient.data() +
        static_cast<std::size_t>(active_orbital_index) * n_basis_functions;
    double* target_column =
        auxiliary_gradient.data() +
        static_cast<std::size_t>(column_index) * n_basis_functions;
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      target_column[basis_function_index] = source_column[basis_function_index];
    }
  }

  ActiveSpaceTwoElectronBackpropagationResult result;
  result.auxiliary_orbital_gradient = std::move(auxiliary_gradient);
  return result;
}

}  // namespace xmvb::vb
