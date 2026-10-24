#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::integrals {

/**
 * @brief Molecule-static metric-whitened AO-pair RI factorization.
 *
 * This method-independent numerical data contract deliberately does
 * not expose which integral backend produced the factors.
 */
struct RiAoFactorization {
    int n_basis_functions = 0;
    /** Number of retained fitting directions (factor rows), not basis size. */
    int n_auxiliary_functions = 0;
    int n_packed_ao_pairs = 0;

    /** Untruncated Coulomb metric in unit-radial Cartesian auxiliary AOs.
   * Its dimension is the auxiliary basis size and can exceed the factor rank.
   */
    Eigen::MatrixXd auxiliary_metric_matrix;

    /** Metric-whitened AO-pair factors `L_{A,P}`. */
    Eigen::MatrixXd metric_whitened_ao_pair_factors;

    /** Optional packed AO-pair metric `g_{PQ} = sum_A L_{A,P} L_{A,Q}`. */
    std::vector<double> packed_ao_pair_metric;
};

} // namespace xmvb::integrals
