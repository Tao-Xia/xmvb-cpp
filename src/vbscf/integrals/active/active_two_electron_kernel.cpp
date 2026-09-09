#include "vbscf/integrals/active/active_two_electron_operator.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

const std::vector<double>* find_packed_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  if (two_electron_view.packed_active_two_electron_integrals == nullptr) {
    return nullptr;
  }
  const auto& packed_active_two_electron_integrals =
      *two_electron_view.packed_active_two_electron_integrals;
  const std::size_t expected_size =
      packed_active_two_electron_integral_count(n_active_orbitals);
  if (packed_active_two_electron_integrals.size() != expected_size) {
    return nullptr;
  }
  return &packed_active_two_electron_integrals;
}

const std::vector<double>& require_packed_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  const std::vector<double>* packed_active_two_electron_integrals =
      find_packed_integrals(two_electron_view, n_active_orbitals);
  if (packed_active_two_electron_integrals == nullptr) {
    throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
  }
  return *packed_active_two_electron_integrals;
}

const Eigen::MatrixXd& require_ri_active_pair_factors(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  if (two_electron_view.ri_active_pair_factors == nullptr) {
    throw std::invalid_argument("RI active-pair-factor buffer is unavailable");
  }
  const auto& ri_active_pair_factors = *two_electron_view.ri_active_pair_factors;
  const std::size_t expected_size =
      two_electron_view.n_auxiliary_functions *
      packed_active_pair_count(n_active_orbitals);
  if (two_electron_view.n_auxiliary_functions <= 0 ||
      ri_active_pair_factors.rows() != two_electron_view.n_auxiliary_functions ||
      ri_active_pair_factors.cols() != packed_active_pair_count(n_active_orbitals) ||
      ri_active_pair_factors.size() != expected_size) {
    throw std::invalid_argument("RI active-pair-factor buffer size mismatch");
  }
  return ri_active_pair_factors;
}

double lookup_ri_active_space_two_electron_kernel_value(
    const Eigen::MatrixXd& ri_active_pair_factors,
    int row_packed_pair_index,
    int column_packed_pair_index) {
  double kernel_value = 0.0;
  for (int auxiliary_function_index = 0;
       auxiliary_function_index < ri_active_pair_factors.rows();
       ++auxiliary_function_index) {
    kernel_value +=
        ri_active_pair_factors(auxiliary_function_index, row_packed_pair_index) *
        ri_active_pair_factors(auxiliary_function_index, column_packed_pair_index);
  }
  return kernel_value;
}

std::vector<double> apply_ri_active_space_two_electron_kernel_to_sparse_projection(
    const Eigen::MatrixXd& ri_active_pair_factors,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values) {
  const int n_auxiliary_functions = static_cast<int>(ri_active_pair_factors.rows());
  const int n_packed_active_pairs = static_cast<int>(ri_active_pair_factors.cols());
  std::vector<double> auxiliary_projection(
      n_auxiliary_functions,
      0.0);
  for (std::size_t entry_index = 0;
       entry_index < packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index = packed_pair_indices[entry_index];
    const double packed_pair_value = packed_pair_values[entry_index];
    if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
      throw std::invalid_argument("packed active-pair index out of range");
    }
    for (int auxiliary_function_index = 0;
         auxiliary_function_index < n_auxiliary_functions;
         ++auxiliary_function_index) {
      auxiliary_projection[auxiliary_function_index] +=
          ri_active_pair_factors(auxiliary_function_index, packed_pair_index) *
          packed_pair_value;
    }
  }

  std::vector<double> projected_pair_values(
      n_packed_active_pairs,
      0.0);
  for (int packed_pair_index = 0;
       packed_pair_index < n_packed_active_pairs;
       ++packed_pair_index) {
    double projected_value = 0.0;
    for (int auxiliary_function_index = 0;
         auxiliary_function_index < n_auxiliary_functions;
         ++auxiliary_function_index) {
      projected_value +=
          ri_active_pair_factors(auxiliary_function_index, packed_pair_index) *
          auxiliary_projection[auxiliary_function_index];
    }
    projected_pair_values[packed_pair_index] = projected_value;
  }

  return projected_pair_values;
}

}  // namespace

ActiveSpaceTwoElectronView make_active_space_two_electron_view(
    const std::vector<double>& packed_active_two_electron_integrals) {
  ActiveSpaceTwoElectronView view;
  view.representation = ActiveSpaceTwoElectronRepresentation::PackedExact;
  view.packed_active_two_electron_integrals = &packed_active_two_electron_integrals;
  return view;
}

ActiveSpaceTwoElectronView make_active_space_two_electron_view(
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
  ActiveSpaceTwoElectronView view;
  view.representation = active_space_two_electron_result.representation;
  view.packed_active_two_electron_integrals =
      &active_space_two_electron_result.packed_active_two_electron_integrals;
  view.n_auxiliary_functions = active_space_two_electron_result.n_auxiliary_functions;
  view.ri_active_pair_factors = &active_space_two_electron_result.ri_active_pair_factors;
  return view;
}

int packed_active_pair_count(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  return n_active_orbitals * (n_active_orbitals + 1) / 2;
}

std::size_t packed_active_two_electron_integral_count(int n_active_orbitals) {
  const std::size_t n_packed_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  return n_packed_active_pairs * (n_packed_active_pairs + 1) / 2;
}

int infer_active_orbital_count_from_packed_pair_count(int n_packed_active_pairs) {
  if (n_packed_active_pairs <= 0) {
    throw std::invalid_argument("n_packed_active_pairs must be positive");
  }
  const double discriminant = 1.0 + 8.0 * static_cast<double>(n_packed_active_pairs);
  const int n_active_orbitals =
      static_cast<int>((std::sqrt(discriminant) - 1.0) / 2.0 + 0.5);
  if (n_active_orbitals * (n_active_orbitals + 1) / 2 != n_packed_active_pairs) {
    throw std::invalid_argument("packed active-pair count is not triangular");
  }
  return n_active_orbitals;
}

int infer_active_orbital_count_from_packed_integral_count(std::size_t packed_integral_count) {
  if (packed_integral_count == 0) {
    throw std::invalid_argument("packed_integral_count must be positive");
  }
  const double discriminant = 1.0 + 8.0 * static_cast<double>(packed_integral_count);
  const int n_packed_active_pairs =
      static_cast<int>((std::sqrt(discriminant) - 1.0) / 2.0 + 0.5);
  if (n_packed_active_pairs *
          (n_packed_active_pairs + 1) / 2 !=
      packed_integral_count) {
    throw std::invalid_argument("packed active-space two-electron count is not triangular");
  }
  return infer_active_orbital_count_from_packed_pair_count(n_packed_active_pairs);
}

double lookup_active_space_two_electron_kernel_value(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int row_packed_pair_index,
    int column_packed_pair_index,
    int n_active_orbitals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (row_packed_pair_index < 0 || row_packed_pair_index >= n_packed_active_pairs ||
      column_packed_pair_index < 0 || column_packed_pair_index >= n_packed_active_pairs) {
    throw std::invalid_argument("packed active-pair index out of range");
  }

  if (const std::vector<double>* packed_active_two_electron_integrals =
          find_packed_integrals(two_electron_view, n_active_orbitals);
      packed_active_two_electron_integrals != nullptr) {
    // A prebuilt packed `GGO` buffer turns the determinant-pair hot path back
    // into a direct table lookup, avoiding an auxiliary-length RI dot product
    // for every requested pair-of-pairs kernel value.
    const int packed_pair_of_pairs_index =
        TwoElectronIndexer::packed_pair_of_pairs_index(
            row_packed_pair_index,
            column_packed_pair_index);
    return (*packed_active_two_electron_integrals)[
        packed_pair_of_pairs_index];
  }

  switch (two_electron_view.representation) {
    case ActiveSpaceTwoElectronRepresentation::PackedExact: {
      throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    }
    case ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity: {
      const auto& ri_active_pair_factors =
          require_ri_active_pair_factors(two_electron_view, n_active_orbitals);
      return lookup_ri_active_space_two_electron_kernel_value(
          ri_active_pair_factors,
          row_packed_pair_index,
          column_packed_pair_index);
    }
  }
  throw std::invalid_argument("unknown active-space two-electron representation");
}

std::vector<double> apply_active_space_two_electron_kernel_to_sparse_projection(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values) {
  if (packed_pair_indices.size() != packed_pair_values.size()) {
    throw std::invalid_argument("packed active-pair projection sizes are inconsistent");
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (const std::vector<double>* packed_active_two_electron_integrals =
          find_packed_integrals(two_electron_view, n_active_orbitals);
      packed_active_two_electron_integrals != nullptr) {
    std::vector<double> projected_pair_values(
        n_packed_active_pairs,
        0.0);
    for (std::size_t entry_index = 0;
         entry_index < packed_pair_indices.size();
         ++entry_index) {
      const int packed_pair_index = packed_pair_indices[entry_index];
      const double packed_pair_value = packed_pair_values[entry_index];
      if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
        throw std::invalid_argument("packed active-pair index out of range");
      }
      for (int row_packed_pair_index = 0;
           row_packed_pair_index < n_packed_active_pairs;
           ++row_packed_pair_index) {
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                packed_pair_index);
        projected_pair_values[row_packed_pair_index] +=
            (*packed_active_two_electron_integrals)[
                packed_pair_of_pairs_index] *
            packed_pair_value;
      }
    }
    return projected_pair_values;
  }

  switch (two_electron_view.representation) {
    case ActiveSpaceTwoElectronRepresentation::PackedExact: {
      throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    }
    case ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity: {
      const auto& ri_active_pair_factors =
          require_ri_active_pair_factors(two_electron_view, n_active_orbitals);
      return apply_ri_active_space_two_electron_kernel_to_sparse_projection(
          ri_active_pair_factors,
          packed_pair_indices,
          packed_pair_values);
    }
  }
  throw std::invalid_argument("unknown active-space two-electron representation");
}

Eigen::VectorXd apply_active_space_two_electron_kernel_to_sparse_projection_subset(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<int>& target_packed_pair_indices) {
  if (packed_pair_indices.size() != packed_pair_values.size()) {
    throw std::invalid_argument("packed active-pair projection sizes are inconsistent");
  }

  Eigen::VectorXd projected_pair_values =
      Eigen::VectorXd::Zero(static_cast<int>(target_packed_pair_indices.size()));
  if (target_packed_pair_indices.empty()) {
    return projected_pair_values;
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (const std::vector<double>* packed_active_two_electron_integrals =
          find_packed_integrals(two_electron_view, n_active_orbitals);
      packed_active_two_electron_integrals != nullptr) {
    for (std::size_t target_index = 0;
         target_index < target_packed_pair_indices.size();
         ++target_index) {
      const int row_packed_pair_index =
          target_packed_pair_indices[target_index];
      if (row_packed_pair_index < 0 || row_packed_pair_index >= n_packed_active_pairs) {
        throw std::invalid_argument("packed active-pair index out of range");
      }
      double projected_value = 0.0;
      for (std::size_t entry_index = 0;
           entry_index < packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = packed_pair_indices[entry_index];
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::invalid_argument("packed active-pair index out of range");
        }
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                packed_pair_index);
        projected_value +=
            (*packed_active_two_electron_integrals)[
                packed_pair_of_pairs_index] *
            packed_pair_values[entry_index];
      }
      projected_pair_values(static_cast<int>(target_index)) = projected_value;
    }
    return projected_pair_values;
  }

  switch (two_electron_view.representation) {
    case ActiveSpaceTwoElectronRepresentation::PackedExact: {
      throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    }
    case ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity: {
      const auto& ri_active_pair_factors =
          require_ri_active_pair_factors(two_electron_view, n_active_orbitals);
      Eigen::VectorXd auxiliary_projection =
          Eigen::VectorXd::Zero(ri_active_pair_factors.rows());
      for (std::size_t entry_index = 0;
           entry_index < packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = packed_pair_indices[entry_index];
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::invalid_argument("packed active-pair index out of range");
        }
        auxiliary_projection.noalias() +=
            packed_pair_values[entry_index] *
            ri_active_pair_factors.col(packed_pair_index);
      }

      for (std::size_t target_index = 0;
           target_index < target_packed_pair_indices.size();
           ++target_index) {
        const int row_packed_pair_index =
            target_packed_pair_indices[target_index];
        if (row_packed_pair_index < 0 || row_packed_pair_index >= n_packed_active_pairs) {
          throw std::invalid_argument("packed active-pair index out of range");
        }
        projected_pair_values(static_cast<int>(target_index)) =
            ri_active_pair_factors.col(row_packed_pair_index).dot(auxiliary_projection);
      }
      return projected_pair_values;
    }
  }
  throw std::invalid_argument("unknown active-space two-electron representation");
}

std::vector<double> reconstruct_packed_active_two_electron_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  if (two_electron_view.representation ==
      ActiveSpaceTwoElectronRepresentation::PackedExact) {
    return require_packed_integrals(two_electron_view, n_active_orbitals);
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> packed_active_two_electron_integrals(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);
  for (int column_packed_pair_index = 0;
       column_packed_pair_index < n_packed_active_pairs;
       ++column_packed_pair_index) {
    for (int row_packed_pair_index = 0;
         row_packed_pair_index <= column_packed_pair_index;
         ++row_packed_pair_index) {
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_packed_pair_index,
              column_packed_pair_index);
      packed_active_two_electron_integrals[
          packed_pair_of_pairs_index] =
          lookup_active_space_two_electron_kernel_value(
              two_electron_view,
              row_packed_pair_index,
              column_packed_pair_index,
              n_active_orbitals);
    }
  }
  return packed_active_two_electron_integrals;
}


}  // namespace xmvb::vb

