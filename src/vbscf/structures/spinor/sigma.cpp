#include "vbscf/structures/spinor/sigma.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xmvb::vb::spinor {
namespace {

std::size_t eri_index(int p, int q, int r, int s, int n) {
    const std::size_t width = static_cast<std::size_t>(n);
    return (((static_cast<std::size_t>(p) * width + q) * width + r) * width + s);
}

double hermitian_tolerance(double scale, int n) {
    return 128.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(1, n)) *
           std::max(1.0, scale);
}

int prefix_parity(std::uint64_t mask, int orbital) {
    const std::uint64_t prefix = (std::uint64_t{1} << orbital) - 1;
    return __builtin_popcountll(mask & prefix) & 1;
}

} // namespace

Sigma::Sigma(std::shared_ptr<const Strings> strings,
             Eigen::MatrixXcd h,
             std::vector<std::complex<double>> eri)
    : strings_(std::move(strings)), h_(std::move(h)), eri_(std::move(eri)) {
    validate_common();
    validate_exact();
}

Sigma::Sigma(std::shared_ptr<const Strings> strings,
             Eigen::MatrixXcd h,
             std::vector<Eigen::MatrixXcd> factors)
    : strings_(std::move(strings)), h_(std::move(h)), factors_(std::move(factors)), ri_(true) {
    validate_common();
    validate_ri();
}

void Sigma::validate_common() const {
    if (!strings_) {
        throw std::invalid_argument("spinor sigma requires a strings carrier");
    }
    const int n = strings_->n_spinor();
    if (h_.rows() != n || h_.cols() != n || !h_.allFinite()) {
        throw std::invalid_argument("invalid spinor one-electron Hamiltonian");
    }
    double scale = 1.0;
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            scale = std::max(scale, std::abs(h_(p, q)));
        }
    }
    const double tolerance = hermitian_tolerance(scale, n);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (std::abs(h_(p, q) - std::conj(h_(q, p))) > tolerance) {
                throw std::invalid_argument("spinor one-electron Hamiltonian is not Hermitian");
            }
        }
    }
}

void Sigma::validate_exact() const {
    const int n = strings_->n_spinor();
    const std::size_t width = static_cast<std::size_t>(n);
    const std::size_t expected = width * width * width * width;
    if (eri_.size() != expected) {
        throw std::invalid_argument("spinor ERI size must equal n_spinor^4");
    }
    double scale = 1.0;
    for (const auto& value : eri_) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw std::invalid_argument("spinor ERI contains a non-finite value");
        }
        scale = std::max(scale, std::abs(value));
    }
    const double tolerance = hermitian_tolerance(scale, n);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    const auto value = eri_[eri_index(p, q, r, s, n)];
                    if (std::abs(value - eri_[eri_index(r, s, p, q, n)]) > tolerance ||
                        std::abs(value - std::conj(eri_[eri_index(q, p, s, r, n)])) > tolerance) {
                        throw std::invalid_argument(
                            "spinor ERI violates density-pair exchange or Hermitian reversal");
                    }
                }
            }
        }
    }
}

void Sigma::validate_ri() const {
    const int n = strings_->n_spinor();
    for (const auto& factor : factors_) {
        if (factor.rows() != n || factor.cols() != n || !factor.allFinite()) {
            throw std::invalid_argument("invalid spinor RI factor dimensions or values");
        }
        double scale = 1.0;
        for (int p = 0; p < n; ++p) {
            for (int q = 0; q < n; ++q) {
                scale = std::max(scale, std::abs(factor(p, q)));
            }
        }
        const double tolerance = hermitian_tolerance(scale, n);
        for (int p = 0; p < n; ++p) {
            for (int q = 0; q < n; ++q) {
                if (std::abs(factor(p, q) - std::conj(factor(q, p))) > tolerance) {
                    throw std::invalid_argument("spinor RI factor is not Hermitian");
                }
            }
        }
    }
}

Eigen::MatrixXcd
Sigma::apply_exact_two_body(const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const {
    const int n = strings_->n_spinor();
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(strings_->dimension(), matrix.cols());
    for (int source = 0; source < strings_->dimension(); ++source) {
        const std::uint64_t source_mask = strings_->masks()[source];
        for (std::uint64_t q_bits = source_mask; q_bits != 0; q_bits &= q_bits - 1) {
            const int q = __builtin_ctzll(q_bits);
            const std::uint64_t after_q = source_mask ^ (std::uint64_t{1} << q);
            const int q_parity = prefix_parity(source_mask, q);
            for (std::uint64_t s_bits = after_q; s_bits != 0; s_bits &= s_bits - 1) {
                const int s = __builtin_ctzll(s_bits);
                const std::uint64_t after_s = after_q ^ (std::uint64_t{1} << s);
                const int annihilation_parity = q_parity ^ prefix_parity(after_q, s);
                for (int r = 0; r < n; ++r) {
                    const std::uint64_t r_bit = std::uint64_t{1} << r;
                    if ((after_s & r_bit) != 0)
                        continue;
                    const std::uint64_t after_r = after_s | r_bit;
                    const int r_parity = annihilation_parity ^ prefix_parity(after_s, r);
                    for (int p = 0; p < n; ++p) {
                        const std::uint64_t p_bit = std::uint64_t{1} << p;
                        if ((after_r & p_bit) != 0)
                            continue;
                        std::complex<double> coefficient = 0.5 * eri_[eri_index(p, q, r, s, n)];
                        if (coefficient == std::complex<double>{0.0, 0.0})
                            continue;
                        if ((r_parity ^ prefix_parity(after_r, p)) != 0) {
                            coefficient = -coefficient;
                        }
                        const int target = strings_->index(after_r | p_bit);
                        result.row(target) += coefficient * matrix.row(source);
                    }
                }
            }
        }
    }
    return result;
}

Eigen::MatrixXcd Sigma::apply(const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const {
    if (matrix.rows() != strings_->dimension() || !matrix.allFinite()) {
        throw std::invalid_argument("invalid spinor sigma block dimensions or values");
    }
    Eigen::MatrixXcd result = strings_->apply_one_body_unchecked(h_, matrix);
    if (!ri_) {
        result += apply_exact_two_body(matrix);
        return result;
    }
    for (const auto& factor : factors_) {
        const Eigen::MatrixXcd intermediate = strings_->apply_one_body_unchecked(factor, matrix);
        const Eigen::MatrixXcd squared = strings_->apply_one_body_unchecked(factor, intermediate);
        const Eigen::MatrixXcd counterterm = factor * factor;
        result += 0.5 * (squared - strings_->apply_one_body_unchecked(counterterm, matrix));
    }
    return result;
}

} // namespace xmvb::vb::spinor
