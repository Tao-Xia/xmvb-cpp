#include "dft/integration/potential.hpp"

#include "core/workspace.hpp"
#include "dft/grid/atomic.hpp"
#include "dft/integration/cache.hpp"
#include "dft/integration/density.hpp"
#include "dft/xc/functional.hpp"
#include "integrals/basis/values.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace xmvb::dft {
namespace {

void compensated_add(double term, double* sum, double* correction) {
    const double adjusted = term - *correction;
    const double next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

void symmetrize_sum(Eigen::MatrixXd* matrix) {
    for (Eigen::Index i = 0; i < matrix->rows(); ++i) {
        (*matrix)(i, i) *= 2.0;
        for (Eigen::Index j = 0; j < i; ++j) {
            const double value = (*matrix)(i, j) + (*matrix)(j, i);
            (*matrix)(i, j) = value;
            (*matrix)(j, i) = value;
        }
    }
}

/** @brief Shared weak-form contraction in contiguous grid columns.
 * Reuses A*D scratch; neither AO data nor spin channels are copied.
 */
void accumulate_potential(const BasisTile& basis,
                          const Eigen::MatrixXd& vrho,
                          const std::array<Eigen::MatrixXd, 3>& gradient,
                          bool gga,
                          int spin,
                          Eigen::MatrixXd* projected,
                          Eigen::MatrixXd* potential) {
    for (Eigen::Index j = 0; j < basis.value.cols(); ++j) {
        projected->col(j).array() = vrho.col(spin).array() * basis.value.col(j).array();
        if (gga)
            for (int axis = 0; axis < 3; ++axis)
                projected->col(j).array() +=
                    gradient[axis].col(spin).array() * basis.gradient[axis].col(j).array();
    }
    if (!projected->allFinite())
        throw std::runtime_error("nonfinite weighted XC potential or response");
    potential->noalias() += basis.value.transpose() * (*projected);
}

} // namespace

XcIntegrator::XcIntegrator(const integrals::LibcintInput& input,
                           const Grid& grid,
                           const Functional& functional)
    : grid_(grid), functional_(functional),
      basis_(std::make_unique<GridBasis>(input, grid, functional.is_gga())),
      response_(std::make_unique<ResponseCache>()) {
    if (grid_.points.rows() == 0 || grid_.points.cols() != 3 ||
        grid_.weights.size() != grid_.points.rows() || !grid_.points.allFinite() ||
        !grid_.weights.allFinite() || (grid_.weights.array() < 0.0).any()) {
        throw std::invalid_argument("invalid DFT grid geometry or weights");
    }
}

XcIntegrator::~XcIntegrator() = default;

XcResult XcIntegrator::evaluate(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density) const {
    const int spins = functional_.n_spin();
    const bool gga = functional_.is_gga();
    const int sigma_columns = gga ? (spins == 1 ? 1 : 3) : 0;
    const Eigen::Index n = basis_->n_bf();
    if (density.size() != static_cast<std::size_t>(spins)) {
        throw std::invalid_argument("AO density spin count does not match XC functional");
    }
    for (const auto& matrix : density) {
        if (matrix.rows() != n || matrix.cols() != n || !matrix.allFinite() ||
            !matrix.isApprox(matrix.transpose(), 1.0e-12)) {
            throw std::invalid_argument("AO density must be finite and symmetric");
        }
    }

    const std::size_t nb = static_cast<std::size_t>(n);
    const std::size_t max_doubles = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (nb && nb > max_doubles / nb / static_cast<std::size_t>(spins)) {
        throw std::length_error("DFT XC matrix size overflow");
    }
    const std::size_t output_doubles = static_cast<std::size_t>(spins) * nb * nb;
    // Simultaneously live tile arrays:
    // LDA: A, projected/AP, rho, exc, vrho, Libxc's packed rho/exc/vrho.
    // GGA: A+3 grad(A), projected/AP, rho+3 grad(rho), sigma,
    //      exc+vrho+vsigma, and Libxc's packed rho/sigma/exc/vrho/vsigma.
    const std::size_t basis_factor = gga ? 5 : 2;
    const std::size_t scalar_factor =
        gga ? static_cast<std::size_t>(2 + 7 * spins + 4 * sigma_columns)
            : static_cast<std::size_t>(2 + 4 * spins);
    if (nb > (max_doubles - scalar_factor) / basis_factor) {
        throw std::length_error("DFT XC tile size overflow");
    }
    const std::size_t point_doubles = basis_factor * nb + scalar_factor;
    const auto minimum = (output_doubles + point_doubles) * sizeof(double);
    if (workspace::available() < minimum)
        response_->clear();
    basis_->prepare(output_doubles * sizeof(double), point_doubles * sizeof(double));
    const std::size_t available = workspace::available();
    if (output_doubles > available / sizeof(double) ||
        point_doubles > (available / sizeof(double) - output_doubles)) {
        throw std::runtime_error("DFT XC resource_limit: insufficient shared workspace");
    }
    const std::size_t grid_size = static_cast<std::size_t>(grid_.points.rows());
    const std::size_t rows_per_tile =
        std::min(grid_size, (available / sizeof(double) - output_doubles) / point_doubles);
    workspace::Reservation reservation;
    if (!reservation.try_resize((output_doubles + rows_per_tile * point_doubles) *
                                sizeof(double))) {
        throw std::runtime_error("DFT XC resource_limit: workspace reservation failed");
    }

    XcResult result;
    for (int spin = 0; spin < spins; ++spin) {
        result.potential[spin] = Eigen::MatrixXd::Zero(n, n);
    }
    integrals::BasisValues basis_values;
    Eigen::MatrixXd projected;
    Eigen::MatrixXd rho;
    std::array<Eigen::MatrixXd, 3> rho_gradient;
    Eigen::MatrixXd sigma;
    XcValues xc;
    double energy_correction = 0.0;
    std::array<double, 2> electron_correction{};

    for (Eigen::Index begin = 0; begin < grid_.points.rows();) {
        const Eigen::Index rows =
            basis_->rows(begin,
                         std::min<Eigen::Index>(grid_.points.rows() - begin,
                                                static_cast<Eigen::Index>(rows_per_tile)));
        const auto tile = basis_->tile(begin, rows, &basis_values);
        rho.resize(rows, spins);
        if (gga) {
            for (auto& component : rho_gradient)
                component.resize(rows, spins);
        }
        for (int spin = 0; spin < spins; ++spin) {
            evaluate_density_tile(tile, density[spin], gga, spin, &projected, &rho, &rho_gradient);
        }

        sigma.resize(rows, sigma_columns);
        if (gga) {
            for (Eigen::Index point = 0; point < rows; ++point) {
                double aa = 0.0;
                double ab = 0.0;
                double bb = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double alpha = rho_gradient[axis](point, 0);
                    aa += alpha * alpha;
                    if (spins == 2) {
                        const double beta = rho_gradient[axis](point, 1);
                        ab += alpha * beta;
                        bb += beta * beta;
                    }
                }
                sigma(point, 0) = aa;
                if (spins == 2) {
                    sigma(point, 1) = ab;
                    sigma(point, 2) = bb;
                }
            }
            if (!sigma.allFinite()) {
                throw std::runtime_error("nonfinite density-gradient invariant on DFT grid");
            }
        }
        functional_.evaluate(rho, sigma, &xc);

        for (Eigen::Index point = 0; point < rows; ++point) {
            const double weight = grid_.weights[begin + point];
            double total_rho = 0.0;
            for (int spin = 0; spin < spins; ++spin) {
                const double spin_rho = rho(point, spin);
                total_rho += spin_rho;
                compensated_add(
                    weight * spin_rho, &result.electron_count[spin], &electron_correction[spin]);
            }
            compensated_add(weight * total_rho * xc.exc[point], &result.energy, &energy_correction);
        }

        // Density/gradient inputs are no longer needed after energy integration.
        // Reuse them for weak-form weights, updating both spins before overwriting.
        for (Eigen::Index point = 0; point < rows; ++point) {
            const double weight = grid_.weights[begin + point];
            for (int spin = 0; spin < spins; ++spin)
                xc.vrho(point, spin) *= 0.5 * weight;
            if (gga)
                for (int axis = 0; axis < 3; ++axis) {
                    const double a = rho_gradient[axis](point, 0);
                    if (spins == 1) {
                        rho_gradient[axis](point, 0) = weight * (2.0 * xc.vsigma(point, 0) * a);
                    } else {
                        const double b = rho_gradient[axis](point, 1);
                        rho_gradient[axis](point, 0) =
                            weight * (2.0 * xc.vsigma(point, 0) * a + xc.vsigma(point, 1) * b);
                        rho_gradient[axis](point, 1) =
                            weight * (2.0 * xc.vsigma(point, 2) * b + xc.vsigma(point, 1) * a);
                    }
                }
        }
        for (int spin = 0; spin < spins; ++spin)
            accumulate_potential(
                tile, xc.vrho, rho_gradient, gga, spin, &projected, &result.potential[spin]);
        begin += rows;
    }

    if (!std::isfinite(result.energy)) {
        throw std::runtime_error("nonfinite XC grid integral");
    }
    for (int spin = 0; spin < spins; ++spin) {
        symmetrize_sum(&result.potential[spin]);
        if (!std::isfinite(result.electron_count[spin]) || !result.potential[spin].allFinite()) {
            throw std::runtime_error("nonfinite XC grid integral");
        }
    }
    return result;
}

std::array<Eigen::MatrixXd, 2>
XcIntegrator::response(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density,
                       std::span<const Eigen::Ref<const Eigen::MatrixXd>> delta_density) const {
    const int spins = functional_.n_spin();
    const bool gga = functional_.is_gga();
    const int sigma_columns = gga ? (spins == 1 ? 1 : 3) : 0;
    const Eigen::Index n = basis_->n_bf();
    if (density.size() != static_cast<std::size_t>(spins) ||
        delta_density.size() != static_cast<std::size_t>(spins)) {
        throw std::invalid_argument("AO density spin count does not match XC functional");
    }
    const auto validate_matrix = [n](const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
        if (matrix.rows() != n || matrix.cols() != n || !matrix.allFinite() ||
            !matrix.isApprox(matrix.transpose(), 1.0e-12)) {
            throw std::invalid_argument("AO density and direction must be finite and symmetric");
        }
    };
    for (int spin = 0; spin < spins; ++spin) {
        validate_matrix(density[spin]);
        validate_matrix(delta_density[spin]);
    }

    const std::size_t nb = static_cast<std::size_t>(n);
    const std::size_t max_doubles = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (nb && nb > max_doubles / nb / static_cast<std::size_t>(spins)) {
        throw std::length_error("DFT XC matrix size overflow");
    }
    const std::size_t output_doubles = static_cast<std::size_t>(spins) * nb * nb;
    // Basis values, gradients and the one reusable projected AO matrix are
    // the only point-by-AO arrays. The scalar allowance covers base and
    // directional densities/invariants, XC outputs, and Libxc buffers.
    const std::size_t basis_factor = gga ? 5 : 2;
    const std::size_t scalar_factor =
        gga ? static_cast<std::size_t>(2 + 20 * spins + 14 * sigma_columns)
            : static_cast<std::size_t>(2 + 10 * spins);
    if (nb > (max_doubles - scalar_factor) / basis_factor) {
        throw std::length_error("DFT XC tile size overflow");
    }
    const std::size_t point_doubles = basis_factor * nb + scalar_factor;
    response_->select(density);
    const auto minimum = (output_doubles + point_doubles) * sizeof(double);
    if (workspace::available() < minimum)
        response_->clear();
    basis_->prepare(output_doubles * sizeof(double), point_doubles * sizeof(double));
    response_->prepare(density,
                       basis_->resident_rows() > 0 ? basis_->resident_rows() : grid_.points.rows(),
                       output_doubles * sizeof(double),
                       point_doubles * sizeof(double),
                       gga);
    const std::size_t available = workspace::available();
    if (output_doubles > available / sizeof(double) ||
        point_doubles > (available / sizeof(double) - output_doubles)) {
        throw std::runtime_error("DFT XC response resource_limit: insufficient shared workspace");
    }
    const std::size_t grid_size = static_cast<std::size_t>(grid_.points.rows());
    const std::size_t rows_per_tile =
        std::min(grid_size, (available / sizeof(double) - output_doubles) / point_doubles);
    workspace::Reservation reservation;
    if (!reservation.try_resize((output_doubles + rows_per_tile * point_doubles) *
                                sizeof(double))) {
        throw std::runtime_error("DFT XC response resource_limit: workspace reservation failed");
    }

    std::array<Eigen::MatrixXd, 2> result;
    for (int spin = 0; spin < spins; ++spin) {
        result[spin] = Eigen::MatrixXd::Zero(n, n);
    }
    integrals::BasisValues basis_values;
    Eigen::MatrixXd projected;
    Eigen::MatrixXd rho, delta_rho, sigma, delta_sigma;
    std::array<Eigen::MatrixXd, 3> delta_rho_gradient;
    Eigen::MatrixXd delta_vrho, delta_vsigma;

    for (Eigen::Index begin = 0; begin < grid_.points.rows();) {
        const Eigen::Index rows = response_->rows(
            begin,
            basis_->rows(begin,
                         std::min<Eigen::Index>(grid_.points.rows() - begin,
                                                static_cast<Eigen::Index>(rows_per_tile))));
        const auto tile = basis_->tile(begin, rows, &basis_values);
        ReferenceTile transient;
        auto* reference = response_->tile(begin, rows);
        if (!reference)
            reference = &transient;
        const bool build_reference = reference->kernel.rho2.rows() == 0;
        auto& rho_gradient = reference->gradient;
        if (build_reference)
            rho.resize(rows, spins);
        delta_rho.resize(rows, spins);
        if (gga) {
            for (int axis = 0; axis < 3; ++axis) {
                if (build_reference)
                    rho_gradient[axis].resize(rows, spins);
                delta_rho_gradient[axis].resize(rows, spins);
            }
        }
        for (int spin = 0; spin < spins; ++spin) {
            if (build_reference)
                evaluate_density_tile(
                    tile, density[spin], gga, spin, &projected, &rho, &rho_gradient);
            evaluate_density_tile(tile,
                                  delta_density[spin],
                                  gga,
                                  spin,
                                  &projected,
                                  &delta_rho,
                                  &delta_rho_gradient,
                                  DensityTileMode::Directional);
        }

        if (build_reference)
            sigma.resize(rows, sigma_columns);
        delta_sigma.resize(rows, sigma_columns);
        if (gga) {
            for (Eigen::Index point = 0; point < rows; ++point) {
                double aa = 0.0, ab = 0.0, bb = 0.0;
                double delta_aa = 0.0, delta_ab = 0.0, delta_bb = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double alpha = rho_gradient[axis](point, 0);
                    const double delta_alpha = delta_rho_gradient[axis](point, 0);
                    if (build_reference)
                        aa += alpha * alpha;
                    delta_aa += 2.0 * alpha * delta_alpha;
                    if (spins == 2) {
                        const double beta = rho_gradient[axis](point, 1);
                        const double delta_beta = delta_rho_gradient[axis](point, 1);
                        if (build_reference)
                            ab += alpha * beta;
                        delta_ab += delta_alpha * beta + alpha * delta_beta;
                        if (build_reference)
                            bb += beta * beta;
                        delta_bb += 2.0 * beta * delta_beta;
                    }
                }
                if (build_reference)
                    sigma(point, 0) = aa;
                delta_sigma(point, 0) = delta_aa;
                if (spins == 2) {
                    if (build_reference) {
                        sigma(point, 1) = ab;
                        sigma(point, 2) = bb;
                    }
                    delta_sigma(point, 1) = delta_ab;
                    delta_sigma(point, 2) = delta_bb;
                }
            }
            if ((build_reference && !sigma.allFinite()) || !delta_sigma.allFinite()) {
                throw std::runtime_error("nonfinite density-gradient invariant on DFT grid");
            }
        }
        if (build_reference) {
            if (gga) {
                XcValues xc;
                functional_.evaluate(rho, sigma, &xc);
                reference->vsigma = std::move(xc.vsigma);
            }
            functional_.kernel(rho, sigma, &reference->kernel);
        }
        reference->kernel.apply(delta_rho, delta_sigma, &delta_vrho, &delta_vsigma);

        // Overwrite only directional scratch, never the cached reference gradient.
        for (Eigen::Index point = 0; point < rows; ++point) {
            const double weight = grid_.weights[begin + point];
            for (int spin = 0; spin < spins; ++spin)
                delta_vrho(point, spin) *= 0.5 * weight;
            if (gga)
                for (int axis = 0; axis < 3; ++axis) {
                    const double a = rho_gradient[axis](point, 0);
                    const double da = delta_rho_gradient[axis](point, 0);
                    double ta =
                        2.0 * (delta_vsigma(point, 0) * a + reference->vsigma(point, 0) * da);
                    if (spins == 1) {
                        delta_rho_gradient[axis](point, 0) = weight * ta;
                    } else {
                        const double b = rho_gradient[axis](point, 1);
                        const double db = delta_rho_gradient[axis](point, 1);
                        double tb =
                            2.0 * (delta_vsigma(point, 2) * b + reference->vsigma(point, 2) * db);
                        ta += delta_vsigma(point, 1) * b + reference->vsigma(point, 1) * db;
                        tb += delta_vsigma(point, 1) * a + reference->vsigma(point, 1) * da;
                        delta_rho_gradient[axis](point, 0) = weight * ta;
                        delta_rho_gradient[axis](point, 1) = weight * tb;
                    }
                }
        }
        for (int spin = 0; spin < spins; ++spin)
            accumulate_potential(
                tile, delta_vrho, delta_rho_gradient, gga, spin, &projected, &result[spin]);
        begin += rows;
    }

    for (int spin = 0; spin < spins; ++spin) {
        symmetrize_sum(&result[spin]);
        if (!result[spin].allFinite()) {
            throw std::runtime_error("nonfinite XC grid response");
        }
    }
    return result;
}

} // namespace xmvb::dft
