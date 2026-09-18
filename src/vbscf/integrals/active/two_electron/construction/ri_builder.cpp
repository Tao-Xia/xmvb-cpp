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

}  // namespace

ActiveSpaceTwoElectronResult RiActiveSpaceTwoElectronBuilder::build(
    const RiAoFactorization& ao_ri_result,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_active_orbitals,
    const RiActiveSpaceTwoElectronBuilderOptions& options) const {
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
  result.dense_active_coefficients = dense_active_coefficients;

  if (options.reconstruct_packed_integrals) {
    const Eigen::MatrixXd active_pair_gram =
        active_pair_factors.transpose() * active_pair_factors;
    std::vector<double> packed_active_two_electron_integrals(
        n_active_pairs * (n_active_pairs + 1) / 2,
        0.0);
    for (int pair_column = 0; pair_column < static_cast<int>(n_active_pairs); ++pair_column) {
      for (int pair_row = 0; pair_row <= pair_column; ++pair_row) {
        const int packed_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(pair_column, pair_row);
        packed_active_two_electron_integrals[packed_index] =
            active_pair_gram(pair_column, pair_row);
      }
    }
    result.packed_active_two_electron_integrals =
        std::move(packed_active_two_electron_integrals);
  }

  return result;
}

}  // namespace xmvb::vb
