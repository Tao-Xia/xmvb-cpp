#include "vb/approx/approx_vbscf_metric.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include "vb/approx/approx_vbscf_pair_cluster.hpp"

namespace xmvb::vb {

namespace {

double safe_metric_log_ratio(double numerator, double denominator) {
  constexpr double kTinyPositive = 1.0e-14;
  if (numerator <= kTinyPositive || denominator <= kTinyPositive) {
    return 0.0;
  }
  return std::log(numerator / denominator);
}

double build_one_pair_metric_log(
    const std::pair<int, int>& pair,
    const std::vector<double>& active_overlap,
    const std::vector<double>& orthogonal_overlap,
    int n_active_orbitals) {
  const std::vector<std::pair<int, int>> cluster = {pair};
  const double metric =
      evaluate_approx_vbscf_pair_cluster_overlap(
          cluster,
          active_overlap,
          n_active_orbitals);
  const double orthogonal_metric =
      evaluate_approx_vbscf_pair_cluster_overlap(
          cluster,
          orthogonal_overlap,
          n_active_orbitals);
  return safe_metric_log_ratio(metric, orthogonal_metric);
}

double build_pair_pair_metric_log(
    const std::pair<int, int>& first_pair,
    const std::pair<int, int>& second_pair,
    double first_pair_log,
    double second_pair_log,
    const std::vector<double>& active_overlap,
    const std::vector<double>& orthogonal_overlap,
    int n_active_orbitals) {
  const std::vector<std::pair<int, int>> cluster = {
      first_pair,
      second_pair};
  if (!approx_vbscf_pair_cluster_is_compatible(cluster)) {
    return 0.0;
  }
  const double metric =
      evaluate_approx_vbscf_pair_cluster_overlap(
          cluster,
          active_overlap,
          n_active_orbitals);
  const double orthogonal_metric =
      evaluate_approx_vbscf_pair_cluster_overlap(
          cluster,
          orthogonal_overlap,
          n_active_orbitals);
  return safe_metric_log_ratio(metric, orthogonal_metric) -
      first_pair_log -
      second_pair_log;
}

}  // namespace

ApproxVbScfMetricModel build_approx_vbscf_metric_model(
    const ApproxVbScfMetricInput& input) {
  const int n_active_orbitals = input.n_active_orbitals;
  const int n_active_pairs =
      approx_vbscf_active_pair_count(n_active_orbitals);
  // The unit orthogonal reference matches the R0 gauge used by the metric
  // functional derivation and the exact VB structure-overlap convention.
  std::vector<double> orthogonal_overlap(
      n_active_orbitals * n_active_orbitals,
      0.0);
  for (int orbital_index = 0;
       orbital_index < n_active_orbitals;
       ++orbital_index) {
    orthogonal_overlap[
        orbital_index * n_active_orbitals + orbital_index] = 1.0;
  }

  ApproxVbScfMetricModel model;
  model.one_pair_metric_log.assign(n_active_pairs, 0.0);
  model.pair_pair_metric_log =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);

  for (int pair_index = 0; pair_index < n_active_pairs; ++pair_index) {
    model.one_pair_metric_log[pair_index] =
        build_one_pair_metric_log(
            unpack_approx_vbscf_active_pair(pair_index),
            *input.active_orbital_overlap_matrix,
            orthogonal_overlap,
            n_active_orbitals);
  }

  for (int column_pair = 0; column_pair < n_active_pairs; ++column_pair) {
    const auto column_orbitals =
        unpack_approx_vbscf_active_pair(column_pair);
    for (int row_pair = 0; row_pair < column_pair; ++row_pair) {
      const auto row_orbitals =
          unpack_approx_vbscf_active_pair(row_pair);
      const double metric_log =
          build_pair_pair_metric_log(
              row_orbitals,
              column_orbitals,
              model.one_pair_metric_log[row_pair],
              model.one_pair_metric_log[column_pair],
              *input.active_orbital_overlap_matrix,
              orthogonal_overlap,
              n_active_orbitals);
      model.pair_pair_metric_log(row_pair, column_pair) = metric_log;
      model.pair_pair_metric_log(column_pair, row_pair) = metric_log;
    }
  }

  return model;
}

}  // namespace xmvb::vb
