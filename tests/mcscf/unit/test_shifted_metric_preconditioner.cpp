/** @file Common-solver regression; links without the VBSCF library. */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "mcscf/optimization/preconditioners/shifted_metric.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_scaled_inner_budget() {
    Eigen::Matrix3d model;
    model << 3.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 0.9;
    Eigen::Matrix3d metric;
    metric << 1.4, 0.35, -0.1, 0.35, 1.1, 0.2, -0.1, 0.2, 0.8;
    const Eigen::Vector3d rhs(0.7, -1.2, 0.4);
    constexpr double shift = 0.65;
    constexpr double relative_target = 1.0e-8;
    const Eigen::Matrix3d block = model + shift * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d block_inverse = block.ldlt().solve(Eigen::Matrix3d::Identity());
    const Eigen::Vector3d result = xmvb::mcscf::apply_shifted_metric_preconditioner(
        rhs,
        shift,
        relative_target,
        [model](const Eigen::VectorXd& vector) { return (model * vector).eval(); },
        [metric](const Eigen::VectorXd& vector) { return (metric * vector).eval(); },
        [block_inverse](const Eigen::VectorXd& vector) { return (block_inverse * vector).eval(); });
    const Eigen::Vector3d residual = rhs - (model + shift * metric) * result;
    const double rhs_norm = std::sqrt(rhs.dot(block_inverse * rhs));
    const double residual_norm = std::sqrt(residual.dot(block_inverse * residual));
    require(residual_norm <= std::sqrt(relative_target) * rhs_norm,
            "preconditioner exceeded its scaled inner residual budget");
}

void check_coordinate_rescaling_preserves_direction() {
    Eigen::Matrix3d model;
    model << 3.0, 0.1, -0.2, 0.1, 1.8, 0.3, -0.2, 0.3, 1.1;
    Eigen::Matrix3d metric;
    metric << 1.4, 0.35, -0.1, 0.35, 1.1, 0.2, -0.1, 0.2, 0.8;
    const Eigen::Vector3d rhs(0.7, -1.2, 0.4);
    const Eigen::Vector3d scales(0.01, 8.0, 100.0);
    const Eigen::Matrix3d coordinates = scales.asDiagonal();
    constexpr double shift = 0.65;
    constexpr double relative_target = 0.01;
    const auto solve = [shift, relative_target](const Eigen::Matrix3d& local_model,
                                                const Eigen::Matrix3d& local_metric,
                                                const Eigen::Matrix3d& local_block_inverse,
                                                const Eigen::Vector3d& local_rhs) {
        return xmvb::mcscf::apply_shifted_metric_preconditioner(
            local_rhs,
            shift,
            relative_target,
            [local_model](const Eigen::VectorXd& vector) { return (local_model * vector).eval(); },
            [local_metric](const Eigen::VectorXd& vector) {
                return (local_metric * vector).eval();
            },
            [local_block_inverse](const Eigen::VectorXd& vector) {
                return (local_block_inverse * vector).eval();
            });
    };
    const Eigen::Matrix3d block = model + shift * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d block_inverse = block.ldlt().solve(Eigen::Matrix3d::Identity());
    const Eigen::Vector3d original = solve(model, metric, block_inverse, rhs);
    const Eigen::Matrix3d rescaled_model = coordinates * model * coordinates;
    const Eigen::Matrix3d rescaled_metric = coordinates * metric * coordinates;
    const Eigen::Matrix3d inverse_coordinates = scales.cwiseInverse().asDiagonal();
    const Eigen::Matrix3d rescaled_inverse =
        inverse_coordinates * block_inverse * inverse_coordinates;
    const Eigen::Vector3d rescaled =
        solve(rescaled_model, rescaled_metric, rescaled_inverse, coordinates * rhs);
    require((coordinates * rescaled - original).norm() <= 1.0e-9 * std::max(1.0, original.norm()),
            "coordinate rescaling changed the shifted preconditioner direction");
}

void check_zero_shift_uses_block_inverse() {
    const Eigen::Vector2d diagonal(2.0, 5.0);
    const Eigen::Vector2d rhs(1.0, -3.0);
    int model_calls = 0;
    const Eigen::Vector2d result = xmvb::mcscf::apply_shifted_metric_preconditioner(
        rhs,
        0.0,
        0.1,
        [diagonal, &model_calls](const Eigen::VectorXd& vector) {
            ++model_calls;
            return (diagonal.asDiagonal() * vector).eval();
        },
        [](const Eigen::VectorXd& vector) { return vector; },
        [diagonal](const Eigen::VectorXd& vector) {
            return (vector.array() / diagonal.array()).matrix();
        });
    require(model_calls == 0, "zero shift performed an iterative solve");
    require((result - Eigen::Vector2d(0.5, -0.6)).norm() < 1.0e-15,
            "zero-shift block inverse is incorrect");
}

/** @brief A shifted model must use the physical metric, not an identity shift. */
void check_physical_metric_shift() {
    Eigen::Matrix3d model;
    model << 2.0, 0.2, 0.0, 0.2, 3.0, 0.1, 0.0, 0.1, 4.0;
    Eigen::Matrix3d metric;
    metric << 4.0, 0.7, 0.1, 0.7, 1.0, 0.2, 0.1, 0.2, 0.5;
    const Eigen::Vector3d rhs(1.0, -2.0, 0.5);
    constexpr double shift = 0.8;
    const Eigen::Matrix3d shifted = model + shift * metric;
    const Eigen::Matrix3d identity_shift = model + shift * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d block_inverse = identity_shift.ldlt().solve(Eigen::Matrix3d::Identity());
    const Eigen::Vector3d result = xmvb::mcscf::apply_shifted_metric_preconditioner(
        rhs,
        shift,
        1.0e-20,
        [model](const Eigen::VectorXd& x) { return (model * x).eval(); },
        [metric](const Eigen::VectorXd& x) { return (metric * x).eval(); },
        [block_inverse](const Eigen::VectorXd& x) { return (block_inverse * x).eval(); });
    require((result - shifted.ldlt().solve(rhs)).norm() < 1.0e-10,
            "shifted model did not invert the complete physical metric term");
    require((result - block_inverse * rhs).norm() > 1.0e-2,
            "fixture does not distinguish physical and identity metric shifts");
}

} // namespace

int main() {
    try {
        check_scaled_inner_budget();
        check_coordinate_rescaling_preserves_direction();
        check_zero_shift_uses_block_inverse();
        check_physical_metric_shift();
        std::cout << "shifted-metric preconditioner tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shifted-metric preconditioner test failed: " << error.what() << '\n';
        return 1;
    }
}
