#include "dft/integration/density.hpp"

#include "dft/integration/cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::dft {

void evaluate_density_tile(const BasisTile& basis,
                           const Eigen::Ref<const Eigen::MatrixXd>& density_matrix,
                           bool with_gradient,
                           Eigen::Index spin,
                           Eigen::MatrixXd* projected,
                           Eigen::MatrixXd* rho,
                           std::array<Eigen::MatrixXd, 3>* density_gradient,
                           DensityTileMode mode) {
    const auto& values = basis.value;
    if (!projected || !rho || !density_gradient || spin < 0 ||
        density_matrix.rows() != density_matrix.cols() || values.cols() != density_matrix.rows() ||
        rho->rows() != values.rows() || spin >= rho->cols()) {
        throw std::invalid_argument("AO density tile shape mismatch");
    }
    if (with_gradient) {
        for (int axis = 0; axis < 3; ++axis) {
            if (basis.gradient[axis].rows() != values.rows() ||
                basis.gradient[axis].cols() != values.cols() ||
                (*density_gradient)[axis].rows() != values.rows() ||
                (*density_gradient)[axis].cols() != rho->cols()) {
                throw std::invalid_argument("AO density gradient tile shape mismatch");
            }
        }
    }

    projected->resize(values.rows(), values.cols());
    projected->noalias() = values * density_matrix;
    for (Eigen::Index point = 0; point < values.rows(); ++point) {
        const auto products = values.row(point).array() * projected->row(point).array();
        const double raw_rho = products.sum();
        if (!std::isfinite(raw_rho)) {
            throw std::runtime_error("nonfinite AO density on DFT grid");
        }
        if (mode == DensityTileMode::Physical) {
            const double absolute_sum = products.abs().sum();
            const double noise = 64.0 * std::numeric_limits<double>::epsilon() * absolute_sum;
            if (!std::isfinite(absolute_sum)) {
                throw std::runtime_error("nonfinite AO density on DFT grid");
            }
            if (raw_rho < -noise) {
                throw std::invalid_argument("AO density is negative on DFT grid");
            }
            (*rho)(point, spin) = std::max(0.0, raw_rho);
        } else {
            (*rho)(point, spin) = raw_rho;
        }
    }
    // Eigen's grid arrays are column-major. Keep the contraction on contiguous
    // point columns instead of taking three strided AO row dots at every point.
    if (with_gradient) {
        for (int axis = 0; axis < 3; ++axis) {
            auto gradient = (*density_gradient)[axis].col(spin);
            gradient.setZero();
            for (Eigen::Index j = 0; j < values.cols(); ++j)
                gradient.array() += basis.gradient[axis].col(j).array() * projected->col(j).array();
            gradient *= 2.0;
            if (!gradient.allFinite())
                throw std::runtime_error("nonfinite AO density gradient on DFT grid");
        }
    }
}

} // namespace xmvb::dft
