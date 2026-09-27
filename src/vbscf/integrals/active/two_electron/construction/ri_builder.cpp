#include "vbscf/integrals/active/two_electron/construction/ri_builder.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"

namespace xmvb::vb {

namespace {

Eigen::MatrixXd build_dense_active_coefficients(
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_active_orbitals) {
  if (orbital_preparation_result.active_sparse_row_offsets.size() !=
      n_basis_functions + 1) {
    throw std::invalid_argument("active_sparse_row_offsets size mismatch");
  }
  if (orbital_preparation_result.active_sparse_orbital_indices.size() !=
      orbital_preparation_result.active_sparse_values.size()) {
    throw std::invalid_argument("active sparse index/value sizes are inconsistent");
  }

  Eigen::MatrixXd dense_active_coefficients =
      Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    const int begin =
        orbital_preparation_result.active_sparse_row_offsets[
            basis_function_index];
    const int end =
        orbital_preparation_result.active_sparse_row_offsets[
            basis_function_index + 1];
    for (int offset = begin; offset < end; ++offset) {
      const int active_orbital_index =
          orbital_preparation_result.active_sparse_orbital_indices[
              offset];
      if (active_orbital_index < 0 || active_orbital_index >= n_active_orbitals) {
        throw std::invalid_argument("active sparse orbital index out of range");
      }
      dense_active_coefficients(basis_function_index, active_orbital_index) =
          orbital_preparation_result.active_sparse_values[offset];
    }
  }
  return dense_active_coefficients;
}

std::vector<double> build_memory_dominated_pair_kernel(
    const Eigen::Ref<const Eigen::MatrixXd>& active_pair_factors) {
  const std::size_t n_pairs =
      static_cast<std::size_t>(active_pair_factors.cols());
  const std::size_t factor_elements =
      static_cast<std::size_t>(active_pair_factors.size());
  const std::size_t gram_elements = n_pairs * n_pairs;
  const std::size_t packed_elements = n_pairs * (n_pairs + 1) / 2;

  // A materialized pair kernel is useful only when its construction workspace
  // plus persistent packed result does not exceed one additional copy of the
  // already resident RI factor matrix. This dimension-only rule gives small
  // and medium active spaces one cache-friendly GEMM while preventing an
  // unconditional O(n_active^4) allocation for large active spaces.
  if (gram_elements + packed_elements > factor_elements) {
    return {};
  }

  const Eigen::MatrixXd gram =
      active_pair_factors.transpose() * active_pair_factors;
  std::vector<double> packed(packed_elements, 0.0);
  for (int column = 0; column < active_pair_factors.cols(); ++column) {
    for (int row = 0; row <= column; ++row) {
      packed[TwoElectronIndexer::packed_pair_of_pairs_index(row, column)] =
          gram(row, column);
    }
  }
  return packed;
}

}  // namespace

ActiveSpaceTwoElectronResult RiActiveSpaceTwoElectronBuilder::build(
    const RiAoFactorization& ao_ri_result,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("RI active-space dimensions must be positive");
  }
  if (ao_ri_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("AO RI result basis-function count mismatch");
  }

  const std::size_t n_active_pairs =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  if (ao_ri_result.metric_whitened_ao_pair_factors.rows() !=
          ao_ri_result.n_auxiliary_functions ||
      ao_ri_result.metric_whitened_ao_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_ao_pairs)) {
    throw std::invalid_argument("AO RI factor matrix shape mismatch");
  }

  const auto dense_active_coefficients =
      build_dense_active_coefficients(
          orbital_preparation_result,
          n_basis_functions,
          n_active_orbitals);
  PackedOrbitalPairMapMatrix ao_pair_to_active_pair_coefficients;
  build_packed_orbital_pair_map(
      dense_active_coefficients,
      &ao_pair_to_active_pair_coefficients);
  const Eigen::MatrixXd active_pair_factors =
      ao_ri_result.metric_whitened_ao_pair_factors *
      ao_pair_to_active_pair_coefficients;

  ActiveSpaceTwoElectronResult result;
  result.representation =
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
  result.n_auxiliary_functions = ao_ri_result.n_auxiliary_functions;
  result.ri_active_pair_factors = active_pair_factors;
  result.packed_active_two_electron_integrals =
      build_memory_dominated_pair_kernel(active_pair_factors);
  result.dense_active_coefficients = dense_active_coefficients;

  return result;
}

}  // namespace xmvb::vb
