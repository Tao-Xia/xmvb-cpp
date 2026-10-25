#include "dft/grid/partition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::dft {

BeckePartition::BeckePartition(const Eigen::MatrixXd& centers)
    : centers_(centers), inverse_separations_(centers.rows(), centers.rows()) {
    if (centers.cols() != 3 || centers.rows() == 0 || !centers.allFinite()) {
        throw std::invalid_argument("Becke centers must be a nonempty finite N x 3 matrix");
    }
    for (Eigen::Index i = 0; i < centers.rows(); ++i) {
        inverse_separations_(i, i) = 0.0;
        for (Eigen::Index j = 0; j < i; ++j) {
            const double separation = (centers.row(i) - centers.row(j)).norm();
            if (!(separation > 0.0) || !std::isfinite(separation)) {
                throw std::invalid_argument(
                    "Becke centers must be distinct with finite separations");
            }
            inverse_separations_(i, j) = inverse_separations_(j, i) = 1.0 / separation;
        }
    }
}

void BeckePartition::weights(const Eigen::Vector3d& point,
                             std::vector<double>& distances,
                             std::vector<double>& factors) const {
    if (!point.allFinite()) {
        throw std::invalid_argument("Becke partition point must be finite");
    }
    const int n = n_atoms();
    distances.resize(n);
    factors.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        distances[i] = (point.transpose() - centers_.row(i)).norm();
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double mu = (distances[j] - distances[i]) * inverse_separations_(i, j);
            mu = std::clamp(mu, -1.0, 1.0);
            for (int smooth = 0; smooth < 3; ++smooth) {
                mu = 0.5 * mu * (3.0 - mu * mu);
            }
            const double s_j = 0.5 * (1.0 - mu);
            // Accumulate the same Becke product in log space; no nearest-atom
            // replacement is needed when many small factors would underflow.
            factors[j] += s_j > 0.0 ? std::log(s_j) : -std::numeric_limits<double>::infinity();
            factors[i] += s_j < 1.0 ? std::log1p(-s_j) : -std::numeric_limits<double>::infinity();
        }
    }
    const double maximum = *std::max_element(factors.begin(), factors.end());
    double total = 0.0;
    for (double& factor : factors) {
        factor = std::exp(factor - maximum);
        total += factor;
    }
    for (double& factor : factors)
        factor /= total;
}

} // namespace xmvb::dft
