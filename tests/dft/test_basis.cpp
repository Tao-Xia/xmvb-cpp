#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

extern "C" {
#include "cint.h"
}

#include "integrals/basis/values.hpp"

namespace {

using xmvb::integrals::BasisEvaluator;
using xmvb::integrals::BasisValues;
using xmvb::integrals::LibcintInput;

struct ShellSpec {
    int l;
    std::vector<double> exponents;
    std::vector<std::vector<double>> contractions;
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, const char* message) {
    require(std::abs(actual - expected) <=
                tolerance * std::max({1.0, std::abs(actual), std::abs(expected)}),
            message);
}

// Explicit low-l Cartesian order provides an independent ordering oracle.
std::vector<std::array<int, 3>> monomials(int l) {
    switch (l) {
        case 0:
            return {{{0, 0, 0}}};
        case 1:
            return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        case 2:
            return {{{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}}};
        case 3:
            return {{{3, 0, 0},
                     {2, 1, 0},
                     {2, 0, 1},
                     {1, 2, 0},
                     {1, 1, 1},
                     {1, 0, 2},
                     {0, 3, 0},
                     {0, 2, 1},
                     {0, 1, 2},
                     {0, 0, 3}}};
        default:
            throw std::invalid_argument("test oracle only covers s through f");
    }
}

LibcintInput make_input(const std::array<double, 3>& center, const std::vector<ShellSpec>& shells) {
    LibcintInput input;
    input.n_atoms = 1;
    input.n_shells = static_cast<int>(shells.size());
    input.atm.assign(ATM_SLOTS, 0);
    input.atm[CHARGE_OF] = 1;
    input.atm[PTR_COORD] = PTR_ENV_START;
    input.env.assign(PTR_ENV_START, 0.0);
    input.env.insert(input.env.end(), center.begin(), center.end());
    input.bas.assign(shells.size() * BAS_SLOTS, 0);
    input.basidx.assign(shells.size() * 2, 0);

    int ao_offset = 0;
    for (std::size_t shell_index = 0; shell_index < shells.size(); ++shell_index) {
        const ShellSpec& shell = shells[shell_index];
        const int nprim = static_cast<int>(shell.exponents.size());
        const int nctr = static_cast<int>(shell.contractions.size());
        require(nprim > 0 && nctr > 0, "test shell must have primitives and contractions");
        const int exponent_offset = static_cast<int>(input.env.size());
        input.env.insert(input.env.end(), shell.exponents.begin(), shell.exponents.end());
        const int coefficient_offset = static_cast<int>(input.env.size());
        for (const auto& contraction : shell.contractions) {
            require(static_cast<int>(contraction.size()) == nprim,
                    "test contraction length is invalid");
            input.env.insert(input.env.end(), contraction.begin(), contraction.end());
        }

        const int basis_offset = static_cast<int>(shell_index) * BAS_SLOTS;
        input.bas[basis_offset + ANG_OF] = shell.l;
        input.bas[basis_offset + NPRIM_OF] = nprim;
        input.bas[basis_offset + NCTR_OF] = nctr;
        input.bas[basis_offset + PTR_EXP] = exponent_offset;
        input.bas[basis_offset + PTR_COEFF] = coefficient_offset;
        input.basidx[2 * shell_index] = ao_offset;
        const int ao_count = static_cast<int>(monomials(shell.l).size()) * nctr;
        input.basidx[2 * shell_index + 1] = ao_count;
        ao_offset += ao_count;
        input.n_gaussian_primitives += nprim;
    }
    return input;
}

double odd_double_factorial(int n) {
    double result = 1.0;
    for (int value = n; value > 1; value -= 2) {
        result *= value;
    }
    return result;
}

double normalized_reference(const ShellSpec& shell,
                            int contraction,
                            const std::array<int, 3>& powers,
                            const std::array<double, 3>& displacement) {
    const int l = powers[0] + powers[1] + powers[2];
    const double angular_integral_factor = odd_double_factorial(2 * powers[0] - 1) *
                                           odd_double_factorial(2 * powers[1] - 1) *
                                           odd_double_factorial(2 * powers[2] - 1);
    const auto& coefficients = shell.contractions[contraction];
    double norm_squared = 0.0;
    for (std::size_t i = 0; i < shell.exponents.size(); ++i) {
        for (std::size_t j = 0; j < shell.exponents.size(); ++j) {
            const double beta = shell.exponents[i] + shell.exponents[j];
            const double integral = angular_integral_factor * std::pow(std::acos(-1.0), 1.5) /
                                    (std::pow(2.0, l) * std::pow(beta, l + 1.5));
            norm_squared += coefficients[i] * coefficients[j] * integral;
        }
    }
    require(norm_squared > 0.0, "test contraction norm is non-positive");

    const double radius_squared = displacement[0] * displacement[0] +
                                  displacement[1] * displacement[1] +
                                  displacement[2] * displacement[2];
    double radial = 0.0;
    for (std::size_t i = 0; i < shell.exponents.size(); ++i) {
        radial += coefficients[i] * std::exp(-shell.exponents[i] * radius_squared);
    }
    double polynomial = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        polynomial *= std::pow(displacement[axis], powers[axis]);
    }
    return polynomial * radial / std::sqrt(norm_squared);
}

void check_analytic_values(const std::vector<ShellSpec>& shells,
                           const std::array<double, 3>& center,
                           const Eigen::MatrixXd& points) {
    const LibcintInput input = make_input(center, shells);
    const BasisEvaluator evaluator(input);
    BasisValues result;
    evaluator.evaluate(points, 1, &result);
    require(result.value.rows() == points.rows() && result.value.cols() == evaluator.n_bf(),
            "basis output dimensions are invalid");
    for (const auto& gradient : result.gradient) {
        require(gradient.rows() == points.rows() && gradient.cols() == evaluator.n_bf(),
                "gradient output dimensions are invalid");
    }

    int ao_offset = 0;
    for (const ShellSpec& shell : shells) {
        const auto powers_by_ao = monomials(shell.l);
        for (std::size_t contraction = 0; contraction < shell.contractions.size(); ++contraction) {
            for (std::size_t cartesian = 0; cartesian < powers_by_ao.size(); ++cartesian) {
                const int ao =
                    ao_offset + static_cast<int>(contraction * powers_by_ao.size() + cartesian);
                for (Eigen::Index point = 0; point < points.rows(); ++point) {
                    const std::array<double, 3> displacement = {points(point, 0) - center[0],
                                                                points(point, 1) - center[1],
                                                                points(point, 2) - center[2]};
                    near(result.value(point, ao),
                         normalized_reference(shell,
                                              static_cast<int>(contraction),
                                              powers_by_ao[cartesian],
                                              displacement),
                         2.0e-11,
                         "normalized Cartesian AO value differs from analytic oracle");
                }
            }
        }
        ao_offset += static_cast<int>(powers_by_ao.size() * shell.contractions.size());
    }
}

void check_gradients_and_reuse(const LibcintInput& input, const Eigen::MatrixXd& points) {
    const BasisEvaluator evaluator(input);
    BasisValues result;
    evaluator.evaluate(points, 1, &result);
    const double step = 1.0e-5;
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::MatrixXd plus = points;
        Eigen::MatrixXd minus = points;
        plus.col(axis).array() += step;
        minus.col(axis).array() -= step;
        BasisValues plus_result;
        BasisValues minus_result;
        evaluator.evaluate(plus, 0, &plus_result);
        evaluator.evaluate(minus, 0, &minus_result);
        require(plus_result.gradient[axis].size() == 0 && minus_result.gradient[axis].size() == 0,
                "order-zero evaluation retained gradients");
        for (Eigen::Index point = 0; point < points.rows(); ++point) {
            for (int ao = 0; ao < evaluator.n_bf(); ++ao) {
                near(result.gradient[axis](point, ao),
                     (plus_result.value(point, ao) - minus_result.value(point, ao)) / (2.0 * step),
                     2.0e-8,
                     "Cartesian AO gradient differs from finite difference");
            }
        }
    }

    Eigen::MatrixXd one_point = points.topRows(1);
    evaluator.evaluate(one_point, 0, &result);
    require(result.value.rows() == 1 && result.value.cols() == evaluator.n_bf(),
            "block reuse did not resize values");
    for (const auto& gradient : result.gradient) {
        require(gradient.rows() == 0 && gradient.cols() == 0,
                "block reuse did not clear gradients");
    }
    evaluator.evaluate(points, 1, &result);
    require(result.gradient[0].rows() == points.rows(), "block reuse did not restore gradients");

    Eigen::MatrixXd empty(0, 3);
    evaluator.evaluate(empty, 1, &result);
    require(result.value.rows() == 0 && result.value.cols() == evaluator.n_bf(),
            "empty block has invalid value dimensions");
    require(result.gradient[0].rows() == 0 && result.gradient[0].cols() == evaluator.n_bf(),
            "empty block has invalid gradient dimensions");
}

} // namespace

int main() {
    try {
        const std::array<double, 3> center = {0.25, -0.4, 0.15};
        Eigen::MatrixXd points(4, 3);
        points << 0.25, -0.4, 0.15, 0.37, -0.62, 0.48, -0.21, 0.09, 0.61, 0.25, -0.18, 0.15;

        const std::vector<ShellSpec> single = {
            {0, {0.7}, {{1.3}}}, {1, {1.1}, {{0.8}}}, {2, {0.9}, {{1.4}}}, {3, {1.5}, {{0.6}}}};
        check_analytic_values(single, center, points);
        const LibcintInput single_input = make_input(center, single);
        check_gradients_and_reuse(single_input, points);

        const std::vector<ShellSpec> contracted = {
            {0, {0.5, 1.7}, {{0.8, -0.2}, {0.25, 1.1}}},
            {2, {0.4, 1.2, 2.1}, {{0.5, 0.6, 0.1}, {0.7, -0.3, 0.9}}}};
        check_analytic_values(contracted, center, points);
        const LibcintInput contracted_input = make_input(center, contracted);
        check_gradients_and_reuse(contracted_input, points);
        std::cout << "Cartesian basis values and gradients passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
