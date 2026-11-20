#pragma once
#include <Eigen/Core>
#include <cstddef>

namespace xmvb::integrals::lri {
struct FitResult {
    Eigen::MatrixXd coefficients;
    std::size_t rank;
    double relative_residual;
};
// Local auxiliary Coulomb-metric least squares. The rank cutoff is an explicit
// basis-preparation parameter, fixed throughout orbital optimization.
FitResult fit_pair(const Eigen::MatrixXd& metric,
                   const Eigen::MatrixXd& three_center,
                   double relative_rank_cutoff);
} // namespace xmvb::integrals::lri
