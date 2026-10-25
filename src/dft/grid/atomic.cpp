#include "dft/grid/atomic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

extern "C" {
#include "cint.h"
}

#include "dft/grid/partition.hpp"

namespace xmvb::dft {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Orbit {
    int code;
    double a;
    double b;
    double weight;
};

// Coefficients adapted from PySCF dft/LebedevGrid.py MakeAngularGrid_302 and
// MakeAngularGrid_590, Copyright 2014-2018 The PySCF Developers, Apache-2.0:
// https://pyscf.org/_modules/pyscf/dft/LebedevGrid.html
// PySCF's orbit codes: 0=(1,0,0), 2=(a,a,a), 3=(a,a,c),
// 4=(a,c,0), 5=(a,b,c). Its tabulated weights sum to one.
constexpr Orbit kLebedev302[] = {
    {0, 0, 0, 0.8545911725128148e-3},
    {2, 0, 0, 0.3599119285025571e-2},
    {3, 0.3515640345570105e+0, 0, 0.3449788424305883e-2},
    {3, 0.6566329410219612e+0, 0, 0.3604822601419882e-2},
    {3, 0.4729054132581005e+0, 0, 0.3576729661743367e-2},
    {3, 0.9618308522614784e-1, 0, 0.2352101413689164e-2},
    {3, 0.2219645236294178e+0, 0, 0.3108953122413675e-2},
    {3, 0.7011766416089545e+0, 0, 0.3650045807677255e-2},
    {4, 0.2644152887060663e+0, 0, 0.2982344963171804e-2},
    {4, 0.5718955891878961e+0, 0, 0.3600820932216460e-2},
    {5, 0.2510034751770465e+0, 0.8000727494073952e+0, 0.3571540554273387e-2},
    {5, 0.1233548532583327e+0, 0.4127724083168531e+0, 0.3392312205006170e-2},
};

constexpr Orbit kLebedev590[] = {
    {0, 0, 0, 0.3095121295306187e-3},
    {2, 0, 0, 0.1852379698597489e-2},
    {3, 0.7040954938227469e+0, 0, 0.1871790639277744e-2},
    {3, 0.6807744066455243e+0, 0, 0.1858812585438317e-2},
    {3, 0.6372546939258752e+0, 0, 0.1852028828296213e-2},
    {3, 0.5044419707800358e+0, 0, 0.1846715956151242e-2},
    {3, 0.4215761784010967e+0, 0, 0.1818471778162769e-2},
    {3, 0.3317920736472123e+0, 0, 0.1749564657281154e-2},
    {3, 0.2384736701421887e+0, 0, 0.1617210647254411e-2},
    {3, 0.1459036449157763e+0, 0, 0.1384737234851692e-2},
    {3, 0.6095034115507196e-1, 0, 0.9764331165051050e-3},
    {4, 0.6116843442009876e+0, 0, 0.1857161196774078e-2},
    {4, 0.3964755348199858e+0, 0, 0.1705153996395864e-2},
    {4, 0.1724782009907724e+0, 0, 0.1300321685886048e-2},
    {5, 0.5610263808622060e+0, 0.3518280927733519e+0, 0.1842866472905286e-2},
    {5, 0.4742392842551980e+0, 0.2634716655937950e+0, 0.1802658934377451e-2},
    {5, 0.5984126497885380e+0, 0.1816640840360209e+0, 0.1849830560443660e-2},
    {5, 0.3791035407695563e+0, 0.1720795225656878e+0, 0.1713904507106709e-2},
    {5, 0.2778673190586244e+0, 0.8213021581932511e-1, 0.1555213603396808e-2},
    {5, 0.5033564271075117e+0, 0.8999205842074875e-1, 0.1802239128008525e-2},
};

void append_orbit(const Orbit& orbit,
                  std::vector<Eigen::Vector3d>& points,
                  std::vector<double>& weights) {
    std::array<double, 3> coordinates;
    switch (orbit.code) {
        case 0:
            coordinates = {0.0, 0.0, 1.0};
            break;
        case 2:
            coordinates = {std::sqrt(1.0 / 3.0), std::sqrt(1.0 / 3.0), std::sqrt(1.0 / 3.0)};
            break;
        case 3:
            coordinates = {orbit.a, orbit.a, std::sqrt(1.0 - 2.0 * orbit.a * orbit.a)};
            break;
        case 4:
            coordinates = {0.0, orbit.a, std::sqrt(1.0 - orbit.a * orbit.a)};
            break;
        case 5:
            coordinates = {
                orbit.a, orbit.b, std::sqrt(1.0 - orbit.a * orbit.a - orbit.b * orbit.b)};
            break;
        default:
            throw std::logic_error("invalid Lebedev orbit code");
    }
    std::sort(coordinates.begin(), coordinates.end());
    do {
        for (int signs = 0; signs < 8; ++signs) {
            if ((coordinates[0] == 0.0 && (signs & 1)) || (coordinates[1] == 0.0 && (signs & 2)) ||
                (coordinates[2] == 0.0 && (signs & 4))) {
                continue;
            }
            points.emplace_back((signs & 1) ? -coordinates[0] : coordinates[0],
                                (signs & 2) ? -coordinates[1] : coordinates[1],
                                (signs & 4) ? -coordinates[2] : coordinates[2]);
            weights.push_back(4.0 * kPi * orbit.weight);
        }
    } while (std::next_permutation(coordinates.begin(), coordinates.end()));
}

void make_angular_rule(int n, std::vector<Eigen::Vector3d>& points, std::vector<double>& weights) {
    const Orbit* orbits = nullptr;
    std::size_t n_orbits = 0;
    if (n == 302) {
        orbits = kLebedev302;
        n_orbits = std::size(kLebedev302);
    } else if (n == 590) {
        orbits = kLebedev590;
        n_orbits = std::size(kLebedev590);
    } else {
        throw std::invalid_argument("supported Lebedev angular_points are 302 and 590");
    }
    points.reserve(n);
    weights.reserve(n);
    for (std::size_t i = 0; i < n_orbits; ++i) {
        append_orbit(orbits[i], points, weights);
    }
    if (points.size() != static_cast<std::size_t>(n)) {
        throw std::logic_error("Lebedev orbit table has the wrong point count");
    }
}

} // namespace

Grid make_atomic_grid(const GridOptions& options) {
    if (options.radial_points <= 0) {
        throw std::invalid_argument("radial_points must be positive");
    }
    std::vector<Eigen::Vector3d> directions;
    std::vector<double> angular_weights;
    make_angular_rule(options.angular_points, directions, angular_weights);
    const Eigen::Index nr = options.radial_points;
    const Eigen::Index na = options.angular_points;
    if (nr > Eigen::Index(std::numeric_limits<Eigen::Index>::max() / na)) {
        throw std::length_error("atomic grid point count overflows Eigen::Index");
    }
    Grid grid{Eigen::MatrixXd(nr * na, 3), Eigen::VectorXd(nr * na)};

    // Krack-Koster Gauss-Chebyshev radial rule, following PySCF dft/radi.py
    // gauss_chebyshev (PySCF Developers, Apache-2.0):
    // https://pyscf.org/_modules/pyscf/dft/radi.html
    const double n = static_cast<double>(options.radial_points);
    const double inv_log2 = 1.0 / std::log(2.0);
    std::vector<double> xi(options.radial_points);
    std::vector<double> sin_theta(options.radial_points);
    for (int i = 0; i < options.radial_points; ++i) {
        const double theta = (i + 1) * kPi / (n + 1.0);
        const double sine = std::sin(theta);
        sin_theta[i] = sine;
        xi[i] = (n - 1.0 - 2.0 * i) / (n + 1.0) +
                (1.0 + (2.0 / 3.0) * sine * sine) * std::sin(2.0 * theta) / kPi;
    }
    for (int i = 0; i < options.radial_points; ++i) {
        const double symmetric_xi = 0.5 * (xi[i] - xi[options.radial_points - 1 - i]);
        const double radius = 1.0 - std::log1p(symmetric_xi) * inv_log2;
        const double sine2 = sin_theta[i] * sin_theta[i];
        const double radial_weight =
            (16.0 / (3.0 * (n + 1.0))) * sine2 * sine2 * inv_log2 / (1.0 + symmetric_xi);
        for (Eigen::Index j = 0; j < na; ++j) {
            const Eigen::Index row = static_cast<Eigen::Index>(i) * na + j;
            grid.points.row(row) = radius * directions[j].transpose();
            grid.weights[row] = radius * radius * radial_weight * angular_weights[j];
        }
    }
    return grid;
}

Grid make_grid(const integrals::LibcintInput& input, const GridOptions& options) {
    if (input.n_atoms <= 0 ||
        input.atm.size() != static_cast<std::size_t>(input.n_atoms) * ATM_SLOTS) {
        throw std::invalid_argument("invalid LibcintInput atom table");
    }
    Eigen::MatrixXd centers(input.n_atoms, 3);
    for (int atom = 0; atom < input.n_atoms; ++atom) {
        const std::size_t offset = static_cast<std::size_t>(atom) * ATM_SLOTS;
        if (input.atm[offset + CHARGE_OF] <= 0) {
            throw std::invalid_argument("DFT grid requires positive atomic numbers");
        }
        const int coord = input.atm[offset + PTR_COORD];
        if (coord < 0 || static_cast<std::size_t>(coord) > input.env.size() ||
            input.env.size() - static_cast<std::size_t>(coord) < 3) {
            throw std::invalid_argument("invalid LibcintInput coordinate pointer");
        }
        for (int k = 0; k < 3; ++k)
            centers(atom, k) = input.env[coord + k];
    }
    const BeckePartition partition(centers);
    const Grid atomic = make_atomic_grid(options);
    const Eigen::Index nlocal = atomic.weights.size();
    if (input.n_atoms > std::numeric_limits<Eigen::Index>::max() / nlocal) {
        throw std::length_error("molecular grid point count overflows Eigen::Index");
    }
    Grid grid{Eigen::MatrixXd(input.n_atoms * nlocal, 3), Eigen::VectorXd(input.n_atoms * nlocal)};
#pragma omp parallel
    {
        std::vector<double> distances;
        std::vector<double> factors;
        distances.reserve(input.n_atoms);
        factors.reserve(input.n_atoms);
#pragma omp for schedule(static)
        for (Eigen::Index row = 0; row < grid.weights.size(); ++row) {
            const int owner = static_cast<int>(row / nlocal);
            const Eigen::Index local = row % nlocal;
            const Eigen::Vector3d point =
                atomic.points.row(local).transpose() + centers.row(owner).transpose();
            partition.weights(point, distances, factors);
            grid.points.row(row) = point.transpose();
            grid.weights[row] = atomic.weights[local] * factors[owner];
        }
    }
    return grid;
}

} // namespace xmvb::dft
