#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "scf/optimization/orbitals.hpp"

namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using xmvb::scf::ApplyUnrestrictedResponse;
using xmvb::scf::OrbitalModel;
using xmvb::scf::SpinMatrices;
using xmvb::scf::UnrestrictedEvaluation;

constexpr int kBasis = 5;
constexpr double kCoulomb = 0.21;
constexpr double kExchange = 0.13;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
}

Matrix make_factor() {
    Matrix upper = Matrix::Identity(kBasis, kBasis);
    upper.diagonal() << 1.12, 0.91, 1.19, 1.04, 0.86;
    upper(0, 1) = 0.17;
    upper(0, 3) = -0.11;
    upper(1, 2) = 0.14;
    upper(1, 4) = -0.08;
    upper(2, 3) = 0.12;
    upper(3, 4) = -0.16;
    return upper;
}

Matrix make_orthogonal(int spin) {
    Matrix q = Matrix::Identity(kBasis, kBasis);
    const auto plane = [&](int left, int right, double angle) {
        Matrix rotation = Matrix::Identity(kBasis, kBasis);
        const double cosine = std::cos(angle), sine = std::sin(angle);
        rotation(left, left) = cosine;
        rotation(right, right) = cosine;
        rotation(left, right) = -sine;
        rotation(right, left) = sine;
        q *= rotation;
    };
    plane(0, 3, 0.28 + 0.11 * spin);
    plane(1, 4, -0.34 + 0.07 * spin);
    plane(0, 2, 0.19 - 0.12 * spin);
    plane(2, 4, -0.23 - 0.06 * spin);
    return q;
}

Matrix make_core() {
    Matrix h(kBasis, kBasis);
    h << -1.31, 0.08, -0.04, 0.03, -0.02, 0.08, -0.84, 0.06, -0.05, 0.04, -0.04, 0.06, -0.26, 0.09,
        -0.03, 0.03, -0.05, 0.09, 0.31, 0.07, -0.02, 0.04, -0.03, 0.07, 0.88;
    return h;
}

UnrestrictedEvaluation evaluate(const SpinMatrices& density, const Matrix& h, int channels) {
    Matrix total = Matrix::Zero(kBasis, kBasis);
    for (int spin = 0; spin < channels; ++spin)
        total += density[spin];
    UnrestrictedEvaluation result;
    result.electronic_energy = 0.5 * kCoulomb * total.squaredNorm();
    for (int spin = 0; spin < channels; ++spin) {
        result.electronic_energy += (h.array() * density[spin].array()).sum() -
                                    0.5 * kExchange * density[spin].squaredNorm();
        result.fock[spin] = h + kCoulomb * total - kExchange * density[spin];
    }
    return result;
}

Vector make_direction(Eigen::Index size, double phase) {
    Vector direction(size);
    for (Eigen::Index i = 0; i < size; ++i) {
        const double index = static_cast<double>(i + 1);
        direction[i] =
            0.17 * std::sin(0.57 * index + phase) + 0.09 * std::cos(0.31 * index - 0.7 * phase);
    }
    return direction;
}

void check_density(const SpinMatrices& orbitals,
                   const SpinMatrices& density,
                   const Matrix& overlap,
                   const std::array<int, 2>& occupied,
                   int channels,
                   double occupation,
                   const std::string& label) {
    const Matrix identity = Matrix::Identity(kBasis, kBasis);
    for (int spin = 0; spin < channels; ++spin) {
        const std::string tag = label + " spin " + std::to_string(spin);
        require((orbitals[spin].transpose() * overlap * orbitals[spin] - identity).norm() < 3e-12,
                tag + ": C^T S C != I");
        near((density[spin] * overlap).trace(),
             occupation * occupied[spin],
             3e-12,
             tag + ": electron count");
        require((density[spin] * overlap * density[spin] - occupation * density[spin]).norm() <
                    3e-12,
                tag + ": generalized density idempotency");
        if (occupied[spin] == 0)
            require(density[spin].norm() == 0.0, tag + ": empty spin is nonzero");
    }
}

void check_case(const std::string& label,
                int channels,
                const std::array<int, 2>& occupied,
                double occupation) {
    const Matrix factor = make_factor();
    const Matrix overlap = factor.transpose() * factor;
    const Matrix h = make_core();
    require(std::abs(overlap(0, 1)) > 0.1, label + ": trivial overlap");

    SpinMatrices orbitals, initial_density;
    for (int spin = 0; spin < channels; ++spin) {
        orbitals[spin] = factor.triangularView<Eigen::Upper>().solve(make_orthogonal(spin));
        const Matrix c_occ = orbitals[spin].leftCols(occupied[spin]);
        initial_density[spin] = occupation * c_occ * c_occ.transpose();
    }
    check_density(
        orbitals, initial_density, overlap, occupied, channels, occupation, label + " initial");

    const UnrestrictedEvaluation initial = evaluate(initial_density, h, channels);
    const ApplyUnrestrictedResponse response = [channels](const SpinMatrices& reference,
                                                          const SpinMatrices& direction) {
        (void)reference;
        Matrix total = Matrix::Zero(kBasis, kBasis);
        for (int spin = 0; spin < channels; ++spin)
            total += direction[spin];
        SpinMatrices image;
        for (int spin = 0; spin < channels; ++spin)
            image[spin] = kCoulomb * total - kExchange * direction[spin];
        return image;
    };
    const OrbitalModel model(
        orbitals, occupied, channels, occupation, initial_density, initial.fock, response);

    const Eigen::Index size = model.gradient().size();
    require(size > 0, label + ": no orbital rotation coordinates");
    const Vector zero = Vector::Zero(size);
    const Vector first = make_direction(size, 0.22);
    const Vector second = make_direction(size, 1.39);
    const Vector displacement = 0.55 * make_direction(size, -0.61);

    const auto energy_at = [&](const Vector& p) {
        const SpinMatrices rotated = model.rotate(p);
        return evaluate(model.density(rotated), h, channels).electronic_energy;
    };
    const auto gradient_at = [&](const Vector& p) {
        const SpinMatrices rotated = model.rotate(p);
        const SpinMatrices density = model.density(rotated);
        return model.pullback(evaluate(density, h, channels).fock, p, rotated);
    };

    check_density(model.rotate(zero),
                  model.density(model.rotate(zero)),
                  overlap,
                  occupied,
                  channels,
                  occupation,
                  label + " at origin");
    const SpinMatrices rotated = model.rotate(displacement);
    check_density(rotated,
                  model.density(rotated),
                  overlap,
                  occupied,
                  channels,
                  occupation,
                  label + " at nonzero displacement");
    require((gradient_at(zero) - model.gradient()).norm() < 2e-12,
            label + ": origin pullback differs from model gradient");

    constexpr double step = 1e-6;
    for (Eigen::Index i = 0; i < size; ++i) {
        Vector basis = Vector::Zero(size);
        basis[i] = 1.0;
        const double finite_difference =
            (energy_at(step * basis) - energy_at(-step * basis)) / (2 * step);
        near(model.gradient()[i],
             finite_difference,
             3e-8,
             label + ": energy-gradient finite difference coordinate " + std::to_string(i));
    }

    const Vector h_first = model.apply(first);
    const Vector h_second = model.apply(second);
    const Vector finite_h_first =
        (gradient_at(step * first) - gradient_at(-step * first)) / (2 * step);
    require((h_first - finite_h_first).norm() < 2e-7,
            label + ": Hessian-vector product differs from pullback derivative");
    near(first.dot(h_second),
         second.dot(h_first),
         2e-10,
         label + ": Hessian bilinear form is asymmetric");

    const double finite_nonzero =
        (energy_at(displacement + step * second) - energy_at(displacement - step * second)) /
        (2 * step);
    near(gradient_at(displacement).dot(second),
         finite_nonzero,
         3e-8,
         label + ": nonzero-displacement pullback directional derivative");

    // Constrained normal descent must use a positive inverse, even when some
    // signed orbital gaps are negative. It preserves the selected soft coordinate.
    const Vector u = first.normalized();
    const Vector finite_gradient = gradient_at(displacement);
    const Vector pu = model.positive_precondition(u);
    require(u.dot(pu) > 0.0, label + ": nonpositive descent inverse");
    Vector correction = -model.positive_precondition(finite_gradient);
    correction -= pu * (u.dot(correction) / u.dot(pu));
    require(std::abs(u.dot(correction)) < 1e-11 * std::max(1.0, correction.norm()),
            label + ": normal correction changes soft coordinate");
    require(finite_gradient.dot(correction) <= 0.0,
            label + ": normal correction is not a descent direction");

    constexpr double projection_step = 1e-6;
    const SpinMatrices nearby = model.density(model.rotate(projection_step * first));
    SpinMatrices change;
    for (int spin = 0; spin < channels; ++spin)
        change[spin] = nearby[spin] - initial_density[spin];
    require((model.displacement(change, overlap) / projection_step - first).norm() < 3e-6,
            label + ": tangent density projection");
}

} // namespace

int main() {
    try {
        check_case("unrestricted unequal spins", 2, {2, 1}, 1.0);
        check_case("restricted occupation two", 1, {2, 0}, 2.0);
        check_case("fully polarized empty beta", 2, {2, 0}, 1.0);
        std::cout << "Orbital model finite-difference tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Orbital model test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
