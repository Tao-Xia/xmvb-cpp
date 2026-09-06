#include "vb/approx/approx_vbscf_cluster_quotient.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace xmvb::vb {

namespace {

double safe_local_quotient(
    const ApproxVbScfPairClusterMatrixElement& matrix_element) {
  constexpr double kTinyPositive = 1.0e-14;
  if (std::abs(matrix_element.overlap) <= kTinyPositive) {
    return 0.0;
  }
  return matrix_element.total_hamiltonian / matrix_element.overlap;
}

}  // namespace

ApproxVbScfClusterQuotientModel build_approx_vbscf_cluster_quotient_model(
    const ApproxVbScfPairClusterHamiltonianInput& input) {
  const int n_active_pairs =
      approx_vbscf_active_pair_count(input.n_active_orbitals);

  ApproxVbScfClusterQuotientModel model;
  model.one_pair_energy.assign(n_active_pairs, 0.0);
  model.one_pair_overlap.assign(n_active_pairs, 0.0);
  model.pair_pair_energy =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);
  model.pair_pair_overlap =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);

  for (int pair_index = 0; pair_index < n_active_pairs; ++pair_index) {
    const std::vector<std::pair<int, int>> cluster = {
        unpack_approx_vbscf_active_pair(pair_index)};
    const auto matrix_element =
        evaluate_approx_vbscf_pair_cluster_matrix_element(
            cluster,
            input);
    model.one_pair_overlap[pair_index] = matrix_element.overlap;
    model.one_pair_energy[pair_index] =
        safe_local_quotient(matrix_element);
  }

  for (int column_pair = 0; column_pair < n_active_pairs; ++column_pair) {
    const auto column_orbitals =
        unpack_approx_vbscf_active_pair(column_pair);
    for (int row_pair = 0; row_pair < column_pair; ++row_pair) {
      const auto row_orbitals =
          unpack_approx_vbscf_active_pair(row_pair);
      const std::vector<std::pair<int, int>> cluster = {
          row_orbitals,
          column_orbitals};
      if (!approx_vbscf_pair_cluster_is_compatible(cluster)) {
        continue;
      }

      const auto matrix_element =
          evaluate_approx_vbscf_pair_cluster_matrix_element(
              cluster,
              input);
      const double pair_pair_quotient =
          safe_local_quotient(matrix_element) -
          model.one_pair_energy[row_pair] -
          model.one_pair_energy[column_pair];
      model.pair_pair_overlap(row_pair, column_pair) =
          matrix_element.overlap;
      model.pair_pair_overlap(column_pair, row_pair) =
          matrix_element.overlap;
      model.pair_pair_energy(row_pair, column_pair) =
          pair_pair_quotient;
      model.pair_pair_energy(column_pair, row_pair) =
          pair_pair_quotient;
    }
  }

  return model;
}

}  // namespace xmvb::vb
