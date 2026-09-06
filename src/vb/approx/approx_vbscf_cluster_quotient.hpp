#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/approx/approx_vbscf_pair_cluster.hpp"

namespace xmvb::vb {

struct ApproxVbScfClusterQuotientModel {
  std::vector<double> one_pair_energy;
  Eigen::MatrixXd pair_pair_energy;
  std::vector<double> one_pair_overlap;
  Eigen::MatrixXd pair_pair_overlap;
};

/**
 * @brief Builds the local quotient Hamiltonian cumulants for aVBSCF.
 *
 * The one-pair entries store `H_p / S_p`.  The pair-pair matrix stores the
 * connected quotient cumulant `H_pq / S_pq - e_p - e_q`.  These entries are the
 * Hamiltonian-numerator counterpart of the local metric cumulants.
 */
ApproxVbScfClusterQuotientModel build_approx_vbscf_cluster_quotient_model(
    const ApproxVbScfPairClusterHamiltonianInput& input);

}  // namespace xmvb::vb
