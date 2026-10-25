#include "integrals/basis/values.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

extern "C" {
#include "cint.h"
}

#include "integrals/ao/libcint.hpp"

namespace xmvb::integrals {
namespace {

double integer_power(double x, int n) {
    double result = 1.0;
    for (int i = 0; i < n; ++i) {
        result *= x;
    }
    return result;
}

bool valid_env_span(int offset, std::int64_t count, std::size_t env_size) {
    return offset >= 0 && count >= 0 && static_cast<std::uint64_t>(offset) <= env_size &&
           static_cast<std::uint64_t>(count) <= env_size - static_cast<std::size_t>(offset);
}

} // namespace

BasisEvaluator::BasisEvaluator(const LibcintInput& input) : input_(input) {
    validate_libcint_input_shape(input_);
    shells_.reserve(input_.n_shells);

    std::int64_t next_ao = 0;
    for (int shell_index = 0; shell_index < input_.n_shells; ++shell_index) {
        const int* basis = input_.bas.data() + shell_index * BAS_SLOTS;
        const int atom = basis[ATOM_OF];
        const int l = basis[ANG_OF];
        const int nprim = basis[NPRIM_OF];
        const int nctr = basis[NCTR_OF];
        if (atom < 0 || atom >= input_.n_atoms || l < 0 || nprim <= 0 || nctr <= 0) {
            throw std::invalid_argument("invalid Cartesian shell metadata");
        }

        const std::int64_t ncart =
            (static_cast<std::int64_t>(l) + 1) * (static_cast<std::int64_t>(l) + 2) / 2;
        if (ncart > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Cartesian shell angular momentum is too large");
        }
        const std::int64_t shell_count = ncart * nctr;
        if (shell_count > std::numeric_limits<int>::max() - next_ao ||
            input_.basidx[2 * shell_index] != next_ao ||
            input_.basidx[2 * shell_index + 1] != shell_count) {
            throw std::invalid_argument("Cartesian shell AO offsets or counts are invalid");
        }

        const int coord_offset = input_.atm[atom * ATM_SLOTS + PTR_COORD];
        const int exponent_offset = basis[PTR_EXP];
        const int coefficient_offset = basis[PTR_COEFF];
        if (!valid_env_span(coord_offset, 3, input_.env.size()) ||
            !valid_env_span(exponent_offset, nprim, input_.env.size()) ||
            !valid_env_span(
                coefficient_offset, static_cast<std::int64_t>(nprim) * nctr, input_.env.size())) {
            throw std::invalid_argument("Cartesian shell env pointer is out of range");
        }

        Shell shell;
        shell.center = input_.env.data() + coord_offset;
        shell.exponents = input_.env.data() + exponent_offset;
        shell.coefficients = input_.env.data() + coefficient_offset;
        shell.nprim = nprim;
        shell.nctr = nctr;
        shell.ao_offset = static_cast<int>(next_ao);
        // These are libcint's CINTcommon_fac_sp factors. The input coefficients
        // already carry the parser's primitive normalization and are used as-is.
        shell.common_factor = l == 0 ? 0.282094791773878143 : l == 1 ? 0.488602511902919921 : 1.0;
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(shell.center[axis])) {
                throw std::invalid_argument("non-finite Cartesian shell center");
            }
        }
        for (int primitive = 0; primitive < nprim; ++primitive) {
            if (!std::isfinite(shell.exponents[primitive]) || !(shell.exponents[primitive] > 0.0)) {
                throw std::invalid_argument("invalid Cartesian primitive exponent");
            }
        }
        for (std::int64_t coefficient = 0; coefficient < static_cast<std::int64_t>(nprim) * nctr;
             ++coefficient) {
            if (!std::isfinite(shell.coefficients[coefficient])) {
                throw std::invalid_argument("non-finite Cartesian contraction coefficient");
            }
        }

        shell.monomials.reserve(static_cast<std::size_t>(ncart));
        for (int lx = l; lx >= 0; --lx) {
            for (int ly = l - lx; ly >= 0; --ly) {
                shell.monomials.push_back({lx, ly, l - lx - ly});
            }
        }
        shells_.push_back(std::move(shell));
        next_ao += shell_count;
    }

    n_bf_ = static_cast<int>(next_ao);
    ao_normalization_ = build_cartesian_ao_normalization(input_);
    if (ao_normalization_.size() != static_cast<std::size_t>(n_bf_)) {
        throw std::runtime_error("Cartesian AO normalization count is invalid");
    }
}

void BasisEvaluator::evaluate(const Eigen::Ref<const Eigen::MatrixXd>& points,
                              int derivative_order,
                              BasisValues* output) const {
    if (output == nullptr) {
        throw std::invalid_argument("BasisValues output must not be null");
    }
    if (points.cols() != 3 || (derivative_order != 0 && derivative_order != 1)) {
        throw std::invalid_argument("basis points must be N x 3 and derivative order 0 or 1");
    }

    output->value.resize(points.rows(), n_bf_);
    for (auto& component : output->gradient) {
        if (derivative_order == 1) {
            component.resize(points.rows(), n_bf_);
        } else {
            component.resize(0, 0);
        }
    }

#pragma omp parallel for schedule(static)
    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        for (const Shell& shell : shells_) {
            const double dx = points(point, 0) - shell.center[0];
            const double dy = points(point, 1) - shell.center[1];
            const double dz = points(point, 2) - shell.center[2];
            const double radius_squared = dx * dx + dy * dy + dz * dz;
            const double displacement[3] = {dx, dy, dz};
            const int ncart = static_cast<int>(shell.monomials.size());

            for (int contraction = 0; contraction < shell.nctr; ++contraction) {
                double radial = 0.0;
                double radial_derivative = 0.0;
                for (int primitive = 0; primitive < shell.nprim; ++primitive) {
                    const double exponent = shell.exponents[primitive];
                    const double term = shell.coefficients[contraction * shell.nprim + primitive] *
                                        std::exp(-exponent * radius_squared);
                    radial += term;
                    if (derivative_order == 1) {
                        radial_derivative -= 2.0 * exponent * term;
                    }
                }

                for (int cartesian = 0; cartesian < ncart; ++cartesian) {
                    const Monomial& monomial = shell.monomials[cartesian];
                    const int powers[3] = {monomial.lx, monomial.ly, monomial.lz};
                    const double coordinate_powers[3] = {integer_power(dx, powers[0]),
                                                         integer_power(dy, powers[1]),
                                                         integer_power(dz, powers[2])};
                    const double polynomial =
                        coordinate_powers[0] * coordinate_powers[1] * coordinate_powers[2];
                    const int ao = shell.ao_offset + contraction * ncart + cartesian;
                    const double scale = shell.common_factor * ao_normalization_[ao];
                    output->value(point, ao) = scale * polynomial * radial;

                    if (derivative_order == 1) {
                        for (int axis = 0; axis < 3; ++axis) {
                            const int other1 = (axis + 1) % 3;
                            const int other2 = (axis + 2) % 3;
                            const double polynomial_derivative =
                                powers[axis] == 0
                                    ? 0.0
                                    : powers[axis] *
                                          integer_power(displacement[axis], powers[axis] - 1) *
                                          coordinate_powers[other1] * coordinate_powers[other2];
                            output->gradient[axis](point, ao) =
                                scale * (polynomial_derivative * radial +
                                         polynomial * radial_derivative * displacement[axis]);
                        }
                    }
                }
            }
        }
    }
}

} // namespace xmvb::integrals
