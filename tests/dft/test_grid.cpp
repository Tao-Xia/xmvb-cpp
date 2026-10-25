#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "cint.h"
}

#include "dft/grid/atomic.hpp"
#include "dft/grid/partition.hpp"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double got, double expected, double tolerance, const char* message) {
    if (!(std::abs(got - expected) <= tolerance)) {
        throw std::runtime_error(std::string(message) + ": got " + std::to_string(got) +
                                 ", expected " + std::to_string(expected));
    }
}

xmvb::integrals::LibcintInput molecule(const std::vector<Eigen::Vector3d>& centers,
                                       const std::vector<int>& atomic_numbers) {
    require(centers.size() == atomic_numbers.size(), "test fixture size mismatch");
    xmvb::integrals::LibcintInput input;
    input.n_atoms = static_cast<int>(centers.size());
    input.atm.assign(centers.size() * ATM_SLOTS, 0);
    input.env.assign(PTR_ENV_START + 3 * centers.size(), 0.0);
    for (std::size_t atom = 0; atom < centers.size(); ++atom) {
        input.atm[atom * ATM_SLOTS + CHARGE_OF] = atomic_numbers[atom];
        input.atm[atom * ATM_SLOTS + PTR_COORD] = static_cast<int>(PTR_ENV_START + 3 * atom);
        for (int k = 0; k < 3; ++k) {
            input.env[PTR_ENV_START + 3 * atom + k] = centers[atom][k];
        }
    }
    return input;
}

double gaussian_integral(const xmvb::dft::Grid& grid, const Eigen::Vector3d& center, double alpha) {
    double sum = 0.0;
    for (Eigen::Index i = 0; i < grid.weights.size(); ++i) {
        sum += grid.weights[i] *
               std::exp(-alpha * (grid.points.row(i).transpose() - center).squaredNorm());
    }
    return sum;
}

template <class Action> void require_throws(Action&& action, const char* message) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void check_angular_rule(int angular_points) {
    constexpr int nr = 80;
    const auto grid = xmvb::dft::make_atomic_grid({nr, angular_points});
    require(grid.points.rows() == nr * angular_points, "atomic grid point count");
    require(grid.weights.size() == grid.points.rows(), "atomic grid weight count");

    // Recover dOmega from the first radial shell. This checks the sphere rule
    // independently of its angular weights and of later molecular partitioning.
    const double radius = grid.points.row(0).norm();
    const double theta = kPi / (nr + 1.0);
    const double sine = std::sin(theta);
    const double one_plus_xi = std::exp((1.0 - radius) * std::log(2.0));
    const double dr = 16.0 / (3.0 * (nr + 1.0)) * std::pow(sine, 4) / (std::log(2.0) * one_plus_xi);
    double m0 = 0.0, mx = 0.0, mx2 = 0.0, mx4 = 0.0, mx2y2 = 0.0;
    for (int j = 0; j < angular_points; ++j) {
        const Eigen::Vector3d direction = grid.points.row(j).transpose() / radius;
        const double w = grid.weights[j] / (radius * radius * dr);
        near(direction.squaredNorm(), 1.0, 2e-14, "sphere node radius");
        require(w > 0.0, "sphere weight positive");
        m0 += w;
        mx += w * direction.x();
        mx2 += w * direction.x() * direction.x();
        mx4 += w * std::pow(direction.x(), 4);
        mx2y2 += w * direction.x() * direction.x() * direction.y() * direction.y();
    }
    near(m0, 4.0 * kPi, 2e-12, "sphere area");
    near(mx, 0.0, 2e-13, "sphere odd moment");
    near(mx2, 4.0 * kPi / 3.0, 2e-12, "sphere x^2 moment");
    near(mx4, 4.0 * kPi / 5.0, 2e-12, "sphere x^4 moment");
    near(mx2y2, 4.0 * kPi / 15.0, 2e-12, "sphere x^2y^2 moment");
}

void check_radial_and_single_atom() {
    const auto input = molecule({Eigen::Vector3d(0.7, -1.2, 0.3)}, {1});
    const auto grid = xmvb::dft::make_grid(input);
    require(grid.weights.size() == 80 * 590, "default grid point count");
    for (Eigen::Index i = 0; i < grid.weights.size(); ++i) {
        require(grid.weights[i] > 0.0 && std::isfinite(grid.weights[i]),
                "single atom finite positive weight");
    }
    for (double alpha : {0.5, 1.0, 2.0}) {
        const double exact = std::pow(kPi / alpha, 1.5);
        near(gaussian_integral(grid, Eigen::Vector3d(0.7, -1.2, 0.3), alpha),
             exact,
             2e-7 * exact,
             "single atom Gaussian integral");
    }
}

void check_partition_and_molecule() {
    const Eigen::Vector3d left(-0.7, 0.2, -0.1);
    const Eigen::Vector3d right(0.7, 0.2, -0.1);
    Eigen::MatrixXd centers(2, 3);
    centers.row(0) = left.transpose();
    centers.row(1) = right.transpose();
    xmvb::dft::BeckePartition partition(centers);
    std::vector<double> distances, factors;
    for (const Eigen::Vector3d& point :
         {left, right, Eigen::Vector3d(0.0, 0.2, -0.1), Eigen::Vector3d(0.1, 0.9, 0.7)}) {
        partition.weights(point, distances, factors);
        near(factors[0] + factors[1], 1.0, 2e-15, "Becke partition of unity");
        require(factors[0] >= 0.0 && factors[1] >= 0.0, "Becke partition nonnegative");
    }
    near(factors[0] + factors[1], 1.0, 2e-15, "Becke normalization");
    partition.weights(left, distances, factors);
    near(factors[0], 1.0, 2e-15, "Becke at left atom");

    const auto input = molecule({left, right}, {1, 1});
    const auto grid = xmvb::dft::make_grid(input);
    require(grid.weights.size() == 2 * 80 * 590, "molecular point count");
    const Eigen::Vector3d midpoint = 0.5 * (left + right);
    const double exact = std::pow(kPi, 1.5);
    near(gaussian_integral(grid, midpoint, 1.0),
         exact,
         2e-6 * exact,
         "two atom midpoint Gaussian integral");
    near(gaussian_integral(grid, left, 1.0),
         exact,
         2e-6 * exact,
         "two atom centered Gaussian integral");

    const auto swapped = xmvb::dft::make_grid(molecule({right, left}, {1, 1}));
    const Eigen::Index block = 80 * 590;
    for (Eigen::Index i = 0; i < block; ++i) {
        near((grid.points.row(i) - swapped.points.row(i + block)).norm(),
             0.0,
             2e-14,
             "atom permutation point");
        near(grid.weights[i],
             swapped.weights[i + block],
             2e-13 * std::max(1.0, grid.weights[i]),
             "atom permutation weight");
    }
    const Eigen::Vector3d shift(1.1, -2.3, 0.8);
    const auto translated = xmvb::dft::make_grid(molecule({left + shift, right + shift}, {1, 1}));
    near(gaussian_integral(translated, midpoint + shift, 1.0),
         gaussian_integral(grid, midpoint, 1.0),
         2e-12,
         "translation invariant integral");
}

void check_invalid_input() {
    require_throws([] { xmvb::dft::make_atomic_grid({0, 302}); }, "zero radial count must fail");
    require_throws([] { xmvb::dft::make_atomic_grid({80, 300}); },
                   "unsupported angular count must fail");
    require_throws([] { xmvb::dft::make_grid(molecule({Eigen::Vector3d::Zero()}, {0})); },
                   "zero atomic number must fail");
    require_throws(
        [] {
            xmvb::dft::make_grid(
                molecule({Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}, {1, 1}));
        },
        "coincident atoms must fail");
}

} // namespace

int main() {
    try {
        check_angular_rule(302);
        check_angular_rule(590);
        check_radial_and_single_atom();
        check_partition_and_molecule();
        check_invalid_input();
    } catch (const std::exception& error) {
        std::cerr << "DFT grid test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
