#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace {
using Matrix = Eigen::MatrixXd;
using Transform = xmvb::vb::ExteriorOrbitalTransform;

void near(const Matrix& a, const Matrix& b, const char* name) {
    const double error = (a - b).norm() / std::max(1.0, b.norm());
    if (!a.allFinite() || error > 4e-13)
        throw std::runtime_error(name);
}

// Independent elementary-factor reference, with the original ordering. This is
// a correctness oracle, not a replayed performance baseline.
void reference(const Transform& transform,
               Matrix& value,
               Matrix* tangent,
               bool right,
               bool adjoint,
               bool inverse,
               const xmvb::vb::ExteriorTransformDirection* direction = nullptr) {
    const auto scale = [&] {
        for (int spin = 0; spin < transform.dimension(); ++spin)
            for (int lane = 0; lane < (right ? value.rows() : value.cols()); ++lane) {
                auto& v = right ? value(lane, spin) : value(spin, lane);
                const double c = transform.determinant_scales()[spin];
                if (tangent) {
                    auto& d = right ? (*tangent)(lane, spin) : (*tangent)(spin, lane);
                    d = c * d + direction->determinant_scales[spin] * v;
                }
                v = inverse ? v / c : v * c;
            }
    };
    const bool reverse = adjoint != inverse;
    if (!reverse)
        scale();
    for (std::size_t step = 0; step < transform.shear_count(); ++step) {
        const auto index = reverse ? transform.shear_count() - 1 - step : step;
        const double c = transform.shear_coefficient(index);
        for (const auto& edge : transform.shear_edges(index)) {
            const int target = adjoint ? edge.source : edge.target;
            const int source = adjoint ? edge.target : edge.source;
            for (int lane = 0; lane < (right ? value.rows() : value.cols()); ++lane) {
                auto& v = right ? value(lane, target) : value(target, lane);
                const double x = right ? value(lane, source) : value(source, lane);
                if (tangent) {
                    auto& d = right ? (*tangent)(lane, target) : (*tangent)(target, lane);
                    const double dx = right ? (*tangent)(lane, source) : (*tangent)(source, lane);
                    d += edge.sign * (direction->shear_coefficients[index] * x + c * dx);
                }
                v += (inverse ? -c : c) * edge.sign * x;
            }
        }
    }
    if (reverse)
        scale();
}

void check(int n, int electrons, bool shuffled) {
    xmvb::mcscf::SpinStrings strings(n, electrons);
    if (shuffled) {
        auto masks = strings.masks();
        std::reverse(masks.begin(), masks.end());
        strings = xmvb::mcscf::SpinStrings(n, masks);
    }
    Matrix upper = Matrix::Zero(n, n), du = upper;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) {
            upper(i, j) = i == j ? 1.0 + 0.003 * j : 0.017 * std::sin(i + 2 * j + 1.0);
            if (i < j && (i + j) % 3 == 0)
                upper(i, j) = 0.0;
            du(i, j) = 0.013 * std::cos(2 * i + j + 0.3);
        }
    Transform transform(strings, upper);
    const auto direction = transform.direction(du);
    Matrix initial(strings.size(), n <= 8 ? 9 : 3);
    for (int j = 0; j < initial.cols(); ++j)
        for (int i = 0; i < initial.rows(); ++i)
            initial(i, j) = std::sin(0.07 * i + 0.19 * j);
    if (n <= 8) {
        const auto occupations = strings.decode();
        Matrix compound(strings.size(), strings.size());
        for (int j = 0; j < strings.size(); ++j)
            for (int i = 0; i < strings.size(); ++i) {
                Matrix minor(electrons, electrons);
                for (int b = 0; b < electrons; ++b)
                    for (int a = 0; a < electrons; ++a)
                        minor(a, b) = upper(occupations[i][a], occupations[j][b]);
                compound(i, j) = electrons ? minor.determinant() : 1.0;
            }
        Matrix actual = initial;
        transform.apply_left(&actual);
        near(actual, compound * initial, "compound-matrix determinant mismatch");
        const double step = 1e-5;
        Transform plus(strings, upper + step * du), minus(strings, upper - step * du);
        Matrix vp = initial * (1.0 + step * 0.23);
        Matrix vm = initial * (1.0 - step * 0.23);
        plus.apply_left(&vp);
        minus.apply_left(&vm);
        Matrix primal = initial, tangent = initial * 0.23;
        transform.apply_directional_left(direction, &primal, &tangent);
        const Matrix fd = (vp - vm) / (2 * step);
        if ((tangent - fd).norm() / std::max(1.0, fd.norm()) > 2e-8)
            throw std::runtime_error("zero-shear tangent finite difference mismatch");
    }
    for (bool right : {false, true}) {
        Matrix input = initial;
        if (right) {
            // Enough contiguous lanes to exercise packet and scalar tails.
            input.resize(n <= 8 ? 41 : 3, strings.size());
            for (int j = 0; j < input.cols(); ++j)
                for (int i = 0; i < input.rows(); ++i)
                    input(i, j) = std::sin(0.07 * j + 0.19 * i);
        }
        for (bool adjoint : {false, true}) {
            for (bool inverse : {false, true}) {
                Matrix expected = input, actual = input;
                reference(transform, expected, nullptr, right, adjoint, inverse);
                if (right) {
                    if (inverse) {
                        if (adjoint)
                            transform.apply_inverse_adjoint_right_block(actual);
                        else
                            transform.apply_inverse_right_block(actual);
                    } else if (adjoint)
                        transform.apply_adjoint_right(&actual);
                    else
                        transform.apply_right(&actual);
                } else {
                    if (inverse) {
                        if (adjoint)
                            transform.apply_inverse_adjoint_left(&actual);
                        else
                            transform.apply_inverse_left(&actual);
                    } else if (adjoint)
                        transform.apply_adjoint_left(&actual);
                    else
                        transform.apply_left(&actual);
                }
                near(actual, expected, "ordinary exterior column mismatch");
            }
            Matrix value = input, delta = input * 0.23, expected = value, expected_delta = delta;
            reference(transform, expected, &expected_delta, right, adjoint, false, &direction);
            if (right) {
                if (adjoint)
                    transform.apply_directional_adjoint_right(direction, &value, &delta);
                else
                    transform.apply_directional_right(direction, &value, &delta);
            } else {
                if (adjoint)
                    transform.apply_directional_adjoint_left(direction, &value, &delta);
                else
                    transform.apply_directional_left(direction, &value, &delta);
            }
            near(value, expected, "directional exterior primal mismatch");
            near(delta, expected_delta, "directional exterior tangent mismatch");
        }
    }
    std::cout << "sector=" << n << ',' << electrons << ",shuffled=" << shuffled
              << ",bytes=" << transform.dynamic_bytes() << '\n';
}
} // namespace
int main() {
    try {
        for (auto [n, e] : {std::pair{4, 0}, {4, 4}, {6, 2}, {8, 5}, {14, 7}, {18, 9}}) {
            check(n, e, false);
            if (n <= 8)
                check(n, e, true);
        }
        std::cout << "Exterior transforms: compounds, finite differences and ordered-factor "
                     "references pass\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
