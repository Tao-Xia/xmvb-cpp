#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "mcscf/ci/factors.hpp"
#include "mcscf/ci/factors_internal.hpp"
#include "mcscf/ci/strings.hpp"
#include "parallel/openmp.hpp"

namespace {

using xmvb::mcscf::SpinFactorSource;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

Eigen::MatrixXd coefficients(int rows, int rank, double phase) {
    Eigen::MatrixXd result(rows, rank);
    for (int j = 0; j < rank; ++j)
        for (int i = 0; i < rows; ++i)
            result(i, j) = std::sin(phase + 0.17 * (i + 1) * (j + 2)) / std::sqrt(rows);
    return result;
}

SpinFactorSource source(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return {
        static_cast<int>(a.rows()),
        static_cast<int>(b.rows()),
        static_cast<int>(a.cols()),
        [&](int first, int width, Eigen::Ref<Eigen::MatrixXd> x, Eigen::Ref<Eigen::MatrixXd> y) {
            x = a.middleCols(first, width);
            y = b.middleCols(first, width);
        }};
}

xmvb::mcscf::ActiveIntegrals integrals(int n, double phase) {
    const int d = n * (n + 1) / 2;
    xmvb::mcscf::ActiveIntegrals result;
    const auto h = coefficients(n, n, phase), v = coefficients(d, d, phase + 0.31);
    result.one_electron = 0.5 * (h + h.transpose());
    result.pair_kernel = 0.05 * (v + v.transpose());
    return result;
}

void check_case(int n, int ea, int eb) {
    const xmvb::mcscf::SpinStrings alpha(n, ea), beta(n, eb);
    const auto active = integrals(n, 0.4);
    const xmvb::mcscf::DirectCiSigmaAction sigma(alpha, beta, active);
    const int d = sigma.channel_kernel().rows(), rows = alpha.size() + beta.size();
    for (bool spin : {false, true}) {
        const int count = spin ? beta.size() : alpha.size();
        const auto a = coefficients(count, 3, 0.27), b = coefficients(count, 3, 0.82);
        Eigen::MatrixXd padded_a(count + 2, 3), padded_b(count + 3, 3), image(count, 3);
        padded_a.topRows(count) = a;
        padded_b.topRows(count) = b;
        const Eigen::MatrixXd kappa = coefficients(n, n, 0.91);
        sigma.apply_spin_generator(spin, padded_b.topRows(count), kappa, image);
        const auto density =
            sigma.spin_generator_adjoint(spin, padded_a.topRows(count), padded_b.topRows(count));
        require(std::abs((density.array() * kappa.array()).sum() -
                         (a.array() * image.array()).sum()) < 1e-12,
                "strided single-spin generator/adjoint pairing changed oriented indices");
        for (int p = 0; p < n; ++p)
            for (int q = 0; q <= p; ++q) {
                Eigen::MatrixXd generator = Eigen::MatrixXd::Zero(n, n), channel(count, 3);
                generator(p, q) = generator(q, p) = 1.0;
                sigma.scatter_spin(spin, b, p * (p + 1) / 2 + q, 1, channel);
                sigma.apply_spin_generator(spin, b, generator, image);
                require((channel - image).norm() < 1e-12,
                        "sorted channel range missed a diagonal or replacement link");
            }
    }
    for (const int rank : {0, 1, 5}) {
        auto a = coefficients(alpha.size(), rank, 0.2);
        auto b = coefficients(beta.size(), rank, 0.7);
        const auto factors = source(a, b);
        for (const auto budget :
             {64ULL * 1024 * 1024,
              static_cast<unsigned long long>(15 * rows * sizeof(double)),
              static_cast<unsigned long long>(2 * rows * (3 * d + 6) * sizeof(double))}) {
            xmvb::mcscf::FactorActionWorkspace workspace(factors, d, budget);
            for (int repeat = 0; repeat < 2; ++repeat) {
                Eigen::MatrixXd h = Eigen::MatrixXd::Zero(alpha.size(), beta.size()), m = h;
                const auto work = workspace.apply(
                    sigma,
                    factors,
                    [&](xmvb::mcscf::FactorCrossBlock block) {
                        h.noalias() += 2 * block.alpha_channels * block.mixed_beta.transpose();
                    },
                    [&](xmvb::mcscf::FactorImageBlock block) {
                        h.noalias() += block.same_alpha * block.beta.transpose();
                        h.noalias() += block.alpha * block.same_beta.transpose();
                        m.noalias() += block.alpha * block.beta.transpose();
                    });
                const Eigen::MatrixXd y = a * b.transpose(), expected = sigma.apply(y);
                require((h - expected).norm() < 4e-12 * std::max(1.0, expected.norm()),
                        "factor action differs from complete-carrier sigma");
                require((m - y).norm() < 2e-12,
                        "factor metric reconstruction changed the amplitude");
                require(work.workspace_bound <= budget, "factor action exceeded its slab budget");
                a.array() *= -0.63;
            }
        }
    }
    for (const auto ranks : {std::pair{1, 1}, std::pair{2, 3}, std::pair{5, 3}}) {
        const auto a = coefficients(alpha.size(), ranks.first, 0.2);
        const auto b = coefficients(beta.size(), ranks.first, 0.6);
        const auto c = coefficients(alpha.size(), ranks.second, 0.8);
        const auto f = coefficients(beta.size(), ranks.second, 1.1);
        const auto bra = source(a, b), ket = source(c, f);
        const Eigen::MatrixXd x = a * b.transpose(), y = c * f.transpose();
        const Eigen::MatrixXd generator_reference = sigma.one_body_generator_adjoint(x, y);
        const Eigen::MatrixXd residual = sigma.apply(x) - 0.37 * x;
        const Eigen::MatrixXd residual_reference = sigma.one_body_generator_adjoint(residual, y);
        auto mapped = bra;
        mapped.contract_work = 1;
        mapped.contract_rows = rows;
        mapped.contract = [&](bool transpose,
                              const Eigen::Ref<const Eigen::MatrixXd>& input,
                              Eigen::Ref<Eigen::MatrixXd> output,
                              double scale) {
            if (transpose)
                output.noalias() += scale * x.transpose() * input;
            else
                output.noalias() += scale * x * input;
        };
        for (const std::size_t budget :
             {64ULL * 1024 * 1024,
              static_cast<unsigned long long>(
                  (18 * rows + 2 + (xmvb::effective_openmp_thread_count() + 1) * n * n) *
                  sizeof(double))}) {
            const auto plan = xmvb::mcscf::detail::plan_factor_panels(
                sigma, mapped, ket, budget, xmvb::mcscf::detail::FactorContraction::Residual);
            require(plan.workspace_bytes <= budget && plan.source_columns == 0,
                    "support-map plan retained all-source storage or unaccounted scratch");
            const auto actual = xmvb::mcscf::factor_generator_adjoint(sigma, mapped, ket, budget);
            const auto relaxed =
                xmvb::mcscf::factor_residual_generator_adjoint(sigma, mapped, ket, 0.37, budget);
            require((actual - generator_reference).norm() <
                            8e-12 * std::max(1.0, generator_reference.norm()) &&
                        (relaxed - residual_reference).norm() <
                            8e-12 * std::max(1.0, residual_reference.norm()),
                    "support-map generator or H-left residual adjoint mismatch");
            const auto integral = xmvb::mcscf::factor_integral_adjoint(sigma, mapped, ket, budget);
            const auto reference = sigma.integral_adjoint(x, y);
            require((integral.one_electron - reference.one_electron).norm() < 8e-12 &&
                        (integral.pair_kernel - reference.pair_kernel).norm() < 8e-12,
                    "support-map integral adjoint mismatch");
        }
        const std::size_t reductions = (xmvb::effective_openmp_thread_count() + 1) * n * n;
        for (const auto bytes :
             {64ULL * 1024 * 1024,
              static_cast<unsigned long long>((13 * rows + 1 + reductions) * sizeof(double))}) {
            const auto actual = xmvb::mcscf::factor_generator_adjoint(sigma, bra, ket, bytes);
            auto traced = bra;
            std::size_t loaded = 0;
            traced.load = [&](int first,
                              int width,
                              Eigen::Ref<Eigen::MatrixXd> x,
                              Eigen::Ref<Eigen::MatrixXd> y) {
                loaded += width;
                bra.load(first, width, x, y);
            };
            const auto plan = xmvb::mcscf::detail::plan_factor_panels(
                sigma, traced, ket, bytes, xmvb::mcscf::detail::FactorContraction::Residual);
            const auto relaxed =
                xmvb::mcscf::factor_residual_generator_adjoint(sigma, traced, ket, 0.37, bytes);
            require(plan.workspace_bytes <= bytes && loaded == plan.source_columns,
                    "joint plan exceeded its budget or miscounted source regeneration");
            require((actual - generator_reference).norm() <
                        8e-12 * std::max(1.0, generator_reference.norm()),
                    "factor generator adjoint lost an oriented one-body index");
            require(
                (relaxed - residual_reference).norm() <
                    8e-12 * std::max(1.0, residual_reference.norm()),
                "factor residual generator changed H-left ordering or partial-channel contraction");
        }
        const auto reference = sigma.integral_adjoint(x, y);
        const auto direction = integrals(n, 1.3);
        const xmvb::mcscf::DirectCiSigmaAction delta(sigma, direction);
        const double exact_direction = (x.array() * delta.apply(y).array()).sum();
        const auto payload = [&](int l, int r, int p) {
            return sizeof(double) * (rows * (l + 2 * r + (p < d ? 3 : 2) * r * p) + l * r * p);
        };
        for (const auto budget : {64ULL * 1024 * 1024,
                                  static_cast<unsigned long long>(payload(1, 1, 1)),
                                  static_cast<unsigned long long>(payload(1, 1, 3)),
                                  static_cast<unsigned long long>(payload(2, 2, d))}) {
            const auto actual = xmvb::mcscf::factor_integral_adjoint(sigma, bra, ket, budget);
            require((actual.one_electron - reference.one_electron).norm() <
                        5e-12 * std::max(1.0, reference.one_electron.norm()),
                    "factor one-electron adjoint mismatch");
            require((actual.pair_kernel - reference.pair_kernel).norm() <
                        5e-12 * std::max(1.0, reference.pair_kernel.norm()),
                    "factor pair-kernel adjoint mismatch");
            const double slope =
                (actual.one_electron.array() * direction.one_electron.array()).sum() +
                (actual.pair_kernel.array() * direction.pair_kernel.array()).sum();
            require(std::abs(slope - exact_direction) <
                        1e-11 * std::max(1.0, std::abs(exact_direction)),
                    "factor adjoint violates the independent integral-direction identity");
        }
    }
    const auto a = coefficients(alpha.size(), 1, 0.2), b = coefficients(beta.size(), 1, 0.6);
    {
        const auto u = coefficients(alpha.size(), 3, 0.3), v = coefficients(beta.size(), 3, 0.8);
        const auto base = source(u, v);
        const Eigen::MatrixXd kappa = coefficients(n, n, 0.61); // intentionally nonsymmetric
        const auto direction = xmvb::mcscf::generator_factors(sigma, base, kappa);
        const auto combined = xmvb::mcscf::sum_factors({direction, base});
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(alpha.size(), beta.size());
        for (int first = 0; first < combined.count; first += 4) {
            const int count = std::min(4, combined.count - first);
            Eigen::MatrixXd x(alpha.size(), count), y(beta.size(), count);
            combined.load(first, count, x, y);
            full.noalias() += x * y.transpose();
        }
        const Eigen::MatrixXd raw = u * v.transpose();
        const Eigen::MatrixXd expected = sigma.apply_one_body_generator(raw, kappa) + raw;
        require((full - expected).norm() < 4e-12,
                "factor generator/sum lost an oriented spin link or source boundary");
        const auto gradient = sigma.integral_adjoint(expected, raw);
        const std::size_t budget = sizeof(double) * (12 * rows + combined.workspace_rows);
        const auto actual = xmvb::mcscf::factor_integral_adjoint(sigma, combined, base, budget);
        require((actual.one_electron - gradient.one_electron).norm() < 5e-12 &&
                    (actual.pair_kernel - gradient.pair_kernel).norm() < 5e-12,
                "directional factor adjoint differs from complete carrier");
        xmvb::mcscf::FactorActionWorkspace workspace(combined, d, budget);
        Eigen::MatrixXd image = Eigen::MatrixXd::Zero(alpha.size(), beta.size());
        const auto work = workspace.apply(
            sigma,
            combined,
            [&](xmvb::mcscf::FactorCrossBlock block) {
                image.noalias() += 2 * block.alpha_channels * block.mixed_beta.transpose();
            },
            [&](xmvb::mcscf::FactorImageBlock block) {
                image.noalias() += block.same_alpha * block.beta.transpose();
                image.noalias() += block.alpha * block.same_beta.transpose();
            });
        require(work.workspace_bound <= budget && (image - sigma.apply(expected)).norm() < 5e-12,
                "directional factor sweep changed sigma or exceeded its loading budget");
    }
    bool limited = false;
    try {
        xmvb::mcscf::factor_integral_adjoint(sigma, source(a, b), source(a, b), 1);
    } catch (const std::length_error&) {
        limited = true;
    }
    require(limited, "factor adjoint silently expanded an insufficient workspace budget");
}

} // namespace

int main() {
    try {
        check_case(4, 2, 2);
        check_case(5, 2, 1);
        check_case(4, 1, 0);
        std::cout << "factor actions and integral adjoints: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
