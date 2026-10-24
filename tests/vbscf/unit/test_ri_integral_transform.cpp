#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "integrals/active/ri_transform.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

bool close(const Eigen::Ref<const Eigen::MatrixXd>& actual,
           const Eigen::Ref<const Eigen::MatrixXd>& expected,
           double tolerance) {
    return actual.rows() == expected.rows() && actual.cols() == expected.cols() &&
           actual.allFinite() &&
           (actual - expected).norm() <= tolerance * std::max(1.0, expected.norm());
}

double factor_at(const Eigen::MatrixXd& factors, int auxiliary, int i, int j) {
    const int high = std::max(i, j);
    const int low = std::min(i, j);
    return factors(auxiliary, high * (high + 1) / 2 + low);
}

// The reference uses explicit scalar contractions, independent of the
// implementation's unpack-and-multiply route.
Eigen::MatrixXd scalar_forward(const Eigen::MatrixXd& factors, const Eigen::MatrixXd& frame) {
    const int n = static_cast<int>(frame.rows());
    Eigen::MatrixXd result(factors.rows(), factors.cols());
    for (int a = 0; a < factors.rows(); ++a) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = 0.0;
                for (int mu = 0; mu < n; ++mu) {
                    for (int nu = 0; nu < n; ++nu) {
                        value += frame(mu, i) * factor_at(factors, a, mu, nu) * frame(nu, j);
                    }
                }
                result(a, i * (i + 1) / 2 + j) = value;
            }
        }
    }
    return result;
}

Eigen::MatrixXd scalar_direction(const Eigen::MatrixXd& factors,
                                 const Eigen::MatrixXd& factor_direction,
                                 const Eigen::MatrixXd& frame,
                                 const Eigen::MatrixXd& frame_direction) {
    const int n = static_cast<int>(frame.rows());
    Eigen::MatrixXd result(factors.rows(), factors.cols());
    for (int a = 0; a < factors.rows(); ++a) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = 0.0;
                for (int mu = 0; mu < n; ++mu) {
                    for (int nu = 0; nu < n; ++nu) {
                        const double b = factor_at(factors, a, mu, nu);
                        value +=
                            frame(mu, i) * factor_at(factor_direction, a, mu, nu) * frame(nu, j);
                        value += frame_direction(mu, i) * b * frame(nu, j);
                        value += frame(mu, i) * b * frame_direction(nu, j);
                    }
                }
                result(a, i * (i + 1) / 2 + j) = value;
            }
        }
    }
    return result;
}

void test_transforms(int n, int n_auxiliary, std::size_t workspace_bytes) {
    const int pairs = n * (n + 1) / 2;
    Eigen::MatrixXd factors(n_auxiliary, pairs);
    Eigen::MatrixXd factor_direction(n_auxiliary, pairs);
    for (int a = 0; a < n_auxiliary; ++a) {
        for (int p = 0; p < pairs; ++p) {
            factors(a, p) = std::sin(0.19 * (1 + 7 * a + p)) + 0.04 * p;
            factor_direction(a, p) = std::cos(0.13 * (2 + 3 * a + 2 * p));
        }
    }
    Eigen::MatrixXd frame(n, n);
    Eigen::MatrixXd frame_direction(n, n);
    Eigen::MatrixXd connection(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            frame(i, j) = (i == j ? 1.2 + 0.12 * i : 0.0) + 0.04 * std::sin(1.0 + 3 * i + j);
            frame_direction(i, j) = 0.17 * std::cos(0.3 + 2 * i - j);
            connection(i, j) = 0.11 * std::sin(0.7 + i + 3 * j);
        }
    }
    require((frame.transpose() * frame - Eigen::MatrixXd::Identity(n, n)).norm() > 0.1,
            "test fixture unexpectedly orthogonal");

    const Eigen::MatrixXd expected_forward = scalar_forward(factors, frame);
    const Eigen::MatrixXd transformed =
        xmvb::integrals::transform_ri_pair_factors(factors, frame, workspace_bytes);
    require(close(transformed, expected_forward, 2e-13),
            "RI factor transform differs from scalar oracle");

    const Eigen::MatrixXd expected_direction =
        scalar_direction(factors, factor_direction, frame, frame_direction);
    const Eigen::MatrixXd actual_direction = xmvb::integrals::transform_ri_pair_factor_direction(
        factors, factor_direction, frame, frame_direction, workspace_bytes);
    require(close(actual_direction, expected_direction, 2e-13),
            "RI factor direction differs from scalar oracle");

    constexpr double step = 1e-6;
    const Eigen::MatrixXd plus =
        scalar_forward(factors + step * factor_direction, frame + step * frame_direction);
    const Eigen::MatrixXd minus =
        scalar_forward(factors - step * factor_direction, frame - step * frame_direction);
    require(close(actual_direction, (plus - minus) / (2 * step), 2e-9),
            "RI factor direction failed centered difference");

    const Eigen::MatrixXd orthogonal_frame_direction = -frame * connection;
    const Eigen::MatrixXd expected_orthogonal_direction =
        scalar_direction(factors, factor_direction, frame, orthogonal_frame_direction);
    const Eigen::MatrixXd from_orthogonal =
        xmvb::integrals::transform_ri_pair_factor_direction_from_orthogonal(
            transformed, factor_direction, frame, connection, workspace_bytes);
    require(close(from_orthogonal, expected_orthogonal_direction, 2e-13),
            "accepted-factor RI direction differs from scalar oracle");
    const Eigen::MatrixXd orthogonal_plus = scalar_forward(
        factors + step * factor_direction, frame + step * orthogonal_frame_direction);
    const Eigen::MatrixXd orthogonal_minus = scalar_forward(
        factors - step * factor_direction, frame - step * orthogonal_frame_direction);
    require(close(from_orthogonal, (orthogonal_plus - orthogonal_minus) / (2 * step), 2e-9),
            "accepted-factor RI direction failed centered difference");
}

void test_invalid_dimensions_and_budget() {
    const Eigen::MatrixXd factors = Eigen::MatrixXd::Ones(2, 10);
    const Eigen::MatrixXd frame = Eigen::MatrixXd::Identity(4, 4);
    bool rejected = false;
    try {
        (void)xmvb::integrals::transform_ri_pair_factors(factors, Eigen::MatrixXd::Identity(3, 3));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "RI transform accepted mismatched pair dimensions");
    rejected = false;
    try {
        (void)xmvb::integrals::transform_ri_pair_factor_direction(
            factors, Eigen::MatrixXd::Zero(1, 10), frame, frame);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "RI direction accepted mismatched factor dimensions");
    rejected = false;
    try {
        (void)xmvb::integrals::transform_ri_pair_factor_direction_from_orthogonal(
            factors, factors, frame, frame, 255);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "RI transform exceeded a one-row workspace limit");
}

} // namespace

int main() {
    // 384 bytes permits exactly one auxiliary row in the three-buffer path.
    test_transforms(4, 11, 384);
    test_transforms(4, 11, xmvb::workspace::available());
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    test_transforms(4, 11, 384);
    omp_set_num_threads(3);
    test_transforms(4, 11, 384);
    omp_set_num_threads(original_threads);
#endif
    test_invalid_dimensions_and_budget();
}
