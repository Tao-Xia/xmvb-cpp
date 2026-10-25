#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

/**
 * @brief Accepted-point metric with the same gauge as the orbital chart.
 *
 * Selected structures use the coupled ray metric of Eq. (31); complete OEO
 * uses the inactive/active span metric of Section 14 of the theory note.
 * It measures the physical derivative of the same raw-coefficient
 * curve used by the optimizer and annihilates support-admissible orbital gauge
 * directions. No orbital Hessian or AO-by-AO directional projector is formed.
 */
class OrbitalPhysicalMetric {
  public:
    OrbitalPhysicalMetric(const OrbitalPreparationInput& input, bool active_subspace_gauge);

    /** @brief Squared physical length of one packed raw-coefficient direction. */
    double squared_norm(const SparseParameterLayout& layout,
                        const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const;

    /** @brief Matrix-free cotangent action of the coupled pullback metric. */
    Eigen::VectorXd apply(const SparseParameterLayout& layout,
                          const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const;

    /** @brief Whether every occupied coefficient is a variational full-AO slot. */
    bool has_full_support(const SparseParameterLayout& layout) const;

    /** @brief Exact flag-metric dual norm for a gauge-annihilating full-AO covector. */
    double subspace_dual_norm(const SparseParameterLayout& layout,
                              const Eigen::Ref<const Eigen::VectorXd>& packed_covector) const;

    /**
   * @brief Exact single-orbital diagonal block of the physical metric.
   *
   * Contracts the support-local lift analytically, including inactive--active
   * feedback. No full reduced metric or per-column metric actions are needed.
   */
    Eigen::MatrixXd orbital_block(int orbital,
                                  const std::vector<int>& support,
                                  const Eigen::Ref<const Eigen::MatrixXd>& lift) const;

  private:
    Eigen::VectorXd
    physical_feature(const SparseParameterLayout& layout,
                     const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const;
    /** @brief Exact reverse of physical_feature with the same accepted-point factors. */
    Eigen::VectorXd feature_adjoint(const SparseParameterLayout& layout,
                                    const Eigen::Ref<const Eigen::VectorXd>& feature) const;
    Eigen::MatrixXd overlap_;
    Eigen::MatrixXd ao_metric_upper_;
    Eigen::MatrixXd inactive_;
    Eigen::MatrixXd active_;
    Eigen::MatrixXd inactive_inverse_;
    Eigen::MatrixXd inactive_inverse_upper_;
    Eigen::MatrixXd complement_;
    Eigen::MatrixXd projected_active_;
    /** @brief Accepted-point contractions reused by feature and cotangent actions. */
    Eigen::MatrixXd inactive_inverse_active_overlap_;
    Eigen::MatrixXd inactive_times_inverse_;
    Eigen::MatrixXd overlap_active_;
    Eigen::MatrixXd overlap_projected_active_;
    Eigen::MatrixXd transposed_overlap_projected_active_;
    Eigen::MatrixXd active_complement_;
    Eigen::MatrixXd active_inverse_;
    Eigen::MatrixXd active_inverse_upper_;
    Eigen::VectorXd projected_norm_squared_;
    std::vector<std::vector<int>> orbital_supports_;
    int n_basis_functions_ = 0;
    int n_inactive_ = 0;
    int n_active_ = 0;
    bool active_subspace_gauge_;
};

} // namespace xmvb::vb
