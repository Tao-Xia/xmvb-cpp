#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

struct ApproxVbScfMetricInput {
  int n_active_orbitals = 0;
  const std::vector<double>* active_orbital_overlap_matrix = nullptr;
};

struct ApproxVbScfMetricModel {
  std::vector<double> one_pair_metric_log;
  Eigen::MatrixXd pair_pair_metric_log;
};

/**
 * @brief Builds the pair-cluster approximation to the nonorthogonal VB metric.
 *
 * The one-pair entries are local metric cumulants `a_p`; the two-pair matrix
 * stores connected cumulants `b_pq`.  Both are computed from tiny spin-adapted
 * VB overlap clusters and are independent of the full structure list.  The
 * reference metric is the unit orthogonal active-overlap limit.
 */
ApproxVbScfMetricModel build_approx_vbscf_metric_model(
    const ApproxVbScfMetricInput& input);

}  // namespace xmvb::vb
