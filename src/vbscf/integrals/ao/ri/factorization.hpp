#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

/**
 * @brief Molecule-static metric-whitened AO-pair RI factorization.
 *
 * This is a numerical data contract consumed by VBSCF.  It deliberately does
 * not expose which integral backend produced the factors.
 */
struct RiAoFactorization {
  int n_basis_functions = 0;
  int n_auxiliary_functions = 0;
  int n_packed_ao_pairs = 0;

  /** Dense auxiliary Coulomb metric in column-major storage. */
  Eigen::MatrixXd auxiliary_metric_matrix;

  /** Metric-whitened AO-pair factors `L_{A,P}`. */
  Eigen::MatrixXd metric_whitened_ao_pair_factors;

  /** Optional packed AO-pair metric `g_{PQ} = sum_A L_{A,P} L_{A,Q}`. */
  std::vector<double> packed_ao_pair_metric;
};

}  // namespace xmvb::vb
