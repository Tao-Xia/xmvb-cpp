#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

extern "C" {
#include "cint.h"
}

#include "integrals/ao/exact.hpp"
#include "integrals/response/ao_pair_operator.hpp"
#include "libcint/direct_shell.hpp"
#include "libcint/materialized_provider.hpp"
#include "libcint/shell_normalization.hpp"

namespace {

constexpr double kTolerance = 1.0e-10;
constexpr int kShells = 3;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

/** @brief One s, p, and Cartesian d shell with a common coefficient scale. */
xmvb::integrals::LibcintInput make_input(double coefficient_scale) {
    xmvb::integrals::LibcintInput input;
    input.n_atoms = 1;
    input.n_shells = kShells;
    input.n_gaussian_primitives = kShells;
    input.atm.assign(ATM_SLOTS, 0);
    input.atm[CHARGE_OF] = 1;
    input.atm[PTR_COORD] = PTR_ENV_START;
    input.env.assign(PTR_ENV_START + 3, 0.0);
    input.bas.assign(kShells * BAS_SLOTS, 0);
    input.basidx.resize(2 * kShells);
    const int counts[kShells] = {1, 3, 6};
    int ao_offset = 0;
    for (int shell = 0; shell < kShells; ++shell) {
        const int environment_offset = static_cast<int>(input.env.size());
        input.env.push_back(1.0);
        input.env.push_back(coefficient_scale * xmvb::libcint::shell_normalization(shell, 1.0));
        input.bas[shell * BAS_SLOTS + ANG_OF] = shell;
        input.bas[shell * BAS_SLOTS + NPRIM_OF] = 1;
        input.bas[shell * BAS_SLOTS + NCTR_OF] = 1;
        input.bas[shell * BAS_SLOTS + PTR_EXP] = environment_offset;
        input.bas[shell * BAS_SLOTS + PTR_COEFF] = environment_offset + 1;
        input.basidx[2 * shell] = ao_offset;
        input.basidx[2 * shell + 1] = counts[shell];
        ao_offset += counts[shell];
    }
    return input;
}

std::size_t quartet_index(int i, int j, int k, int l, int ni, int nj, int nk) {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(ni) *
               (static_cast<std::size_t>(j) +
                static_cast<std::size_t>(nj) *
                    (static_cast<std::size_t>(k) +
                     static_cast<std::size_t>(nk) * static_cast<std::size_t>(l)));
}

void check_bound_and_schwarz(const xmvb::libcint::LibcintDirectShellEvaluator& evaluator) {
    xmvb::libcint::LibcintShellWorkspace workspace;
    double bounds[kShells][kShells] = {};
    for (int i = 0; i < kShells; ++i) {
        for (int j = 0; j < kShells; ++j) {
            bounds[i][j] = evaluator.evaluate_two_electron_shell_pair_bound(i, j, workspace);
            const auto& self = evaluator.evaluate_two_electron_shell_quartet(i, j, i, j, workspace);
            double direct_maximum = 0.0;
            for (int local_j = 0; local_j < self.ao_count_j; ++local_j) {
                for (int local_i = 0; local_i < self.ao_count_i; ++local_i) {
                    direct_maximum =
                        std::max(direct_maximum,
                                 std::abs(self.values[quartet_index(local_i,
                                                                    local_j,
                                                                    local_i,
                                                                    local_j,
                                                                    self.ao_count_i,
                                                                    self.ao_count_j,
                                                                    self.ao_count_k)]));
                }
            }
            require(std::abs(bounds[i][j] - direct_maximum) <=
                        1.0e-12 * std::max(1.0, direct_maximum),
                    "shell-pair bound differs from normalized self diagonal");
            require(bounds[i][j] > 0.0, "shell-pair bound must be positive");
        }
    }

    for (int i = 0; i < kShells; ++i) {
        for (int j = 0; j < kShells; ++j) {
            for (int k = 0; k < kShells; ++k) {
                for (int l = 0; l < kShells; ++l) {
                    const double schwarz = std::sqrt(bounds[i][j] * bounds[k][l]);
                    const auto& quartet =
                        evaluator.evaluate_two_electron_shell_quartet(i, j, k, l, workspace);
                    for (double value : quartet.values) {
                        require(std::abs(value) <= schwarz + 1.0e-12 * std::max(1.0, schwarz),
                                "normalized shell-pair bound violates Schwarz");
                    }
                }
            }
        }
    }
}

void check_normalized_quartets_are_scale_invariant(
    const xmvb::libcint::LibcintDirectShellEvaluator& original,
    const xmvb::libcint::LibcintDirectShellEvaluator& scaled) {
    xmvb::libcint::LibcintShellWorkspace original_workspace;
    xmvb::libcint::LibcintShellWorkspace scaled_workspace;
    for (int i = 0; i < kShells; ++i) {
        for (int j = 0; j < kShells; ++j) {
            for (int k = 0; k < kShells; ++k) {
                for (int l = 0; l < kShells; ++l) {
                    const auto& a = original.evaluate_two_electron_shell_quartet(
                        i, j, k, l, original_workspace);
                    const auto& b =
                        scaled.evaluate_two_electron_shell_quartet(i, j, k, l, scaled_workspace);
                    require(a.values.size() == b.values.size(),
                            "coefficient scaling changed quartet size");
                    for (std::size_t index = 0; index < a.values.size(); ++index) {
                        require(std::abs(a.values[index] - b.values[index]) <= 1.0e-11,
                                "normalized ERI changed under primitive coefficient scaling");
                    }
                }
            }
        }
    }
}

void check_materialized_equal(const xmvb::libcint::MaterializedAoIntegralBuffers& a,
                              const xmvb::libcint::MaterializedAoIntegralBuffers& b) {
    require(a.n_basis_functions == b.n_basis_functions, "materialized AO counts differ");
    require(a.left_pair_indices == b.left_pair_indices &&
                a.right_pair_indices == b.right_pair_indices,
            "materialized integral indices differ");
    require(!a.two_electron_values.empty(), "materialized ERI stream is empty");
    require(a.two_electron_values.size() == b.two_electron_values.size(),
            "materialized ERI counts differ");
    for (std::size_t index = 0; index < a.two_electron_values.size(); ++index) {
        require(std::abs(a.two_electron_values[index] - b.two_electron_values[index]) <= 1.0e-11,
                "materialized ERI values differ");
    }
}

void check_tiled_operators(const xmvb::libcint::MaterializedAoIntegralBuffers& reference,
                           xmvb::libcint::MaterializedAoIntegralBuffers tiled) {
    using namespace xmvb::integrals;
    require(tiled.tiles && tiled.two_electron_values.empty() && tiled.chunks.empty(),
            "CPU libcint did not build one triangular payload");
    const int n = reference.n_basis_functions;
    AoIntegralInput csr;
    csr.n_basis_functions = n;
    csr.pair_graph = build_ao_pair_graph(reference.left_pair_indices,
                                        reference.right_pair_indices,
                                        reference.two_electron_values,
                                        n);
    const auto blocks = xmvb::libcint::build_materialized_ao_integral_input(std::move(tiled));
    const auto& graph = blocks.pair_graph;
    const auto size = graph.pair_first.size();
    require(graph.integral_count() == csr.pair_graph.integral_count(),
            "triangular integral count differs");
    require(graph.directed_count() == csr.pair_graph.directed_count(),
            "triangular directed count differs");
    require(graph.row_offsets.empty() && !graph.has_compact_integral_stream(),
            "triangular data were expanded");
    Eigen::MatrixXd exact = Eigen::MatrixXd::Zero(size, size);
    for (std::size_t p = 0; p < size; ++p)
        csr.pair_graph.visit_row(p, [&](std::size_t q, double v) { exact(p, q) = v; });
    for (std::size_t p = 0; p < size; ++p)
        graph.visit_row(p, [&](std::size_t q, double v) {
            require(v == exact(p, q), "triangular row value differs");
        });

    const ExactCtxPairMatrix rhs = ExactCtxPairMatrix::NullaryExpr(size, 11,
        [](Eigen::Index i, Eigen::Index j) { return std::cos((i + 1) * (j + 1)); });
    ExactCtxPairMatrix image;
    detail::apply_exact_ao_pair_graph(graph, rhs, n, rhs.cols(), &image);
    require(image.isApprox(exact * rhs, 1e-12), "triangular AO-pair product differs");

    // Two symmetric density channels exercise canonical traversal and the
    // output-owned projection used by the analytical Hessian diagonal.
    Eigen::MatrixXd densities(n * n, 2);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            densities(i + j * n, 0) = std::sin((i + 1) * (j + 1));
            densities(i + j * n, 1) = std::cos((i + 1) * (j + 1));
        }
    AoH1eBatchWorkspace workspace;
    Eigen::MatrixXd expected, actual;
    apply_symmetric_ao_h1e_batch(densities, csr, 2, &workspace, &expected);
    apply_symmetric_ao_h1e_batch(densities, blocks, 2, &workspace, &actual);
    require(actual.isApprox(expected, 1e-12), "triangular multi-density response differs");

    AoDensityDiagonalWorkspace diagonal;
    Eigen::MatrixXd j_ref, k_ref, j, k;
    const AoH1eBatchWorkspace::Channels sources = densities;
    const Eigen::MatrixXd frame = rhs.topLeftCorner(n, 3);
    apply_ao_density_diagonals(csr, sources, frame, 1, 2, &diagonal, &k_ref, &j_ref);
    apply_ao_density_diagonals(blocks, sources, frame, 1, 2, &diagonal, &k, &j);
    require(j.isApprox(j_ref, 1e-12) && k.isApprox(k_ref, 1e-12),
            "triangular projected density diagonal differs");
}
} // namespace

int main() {
    try {
        const auto original_input = make_input(1.0);
        auto scaled_input = make_input(1.0e-3);
        xmvb::libcint::LibcintDirectShellEvaluator original(original_input);
        xmvb::libcint::LibcintDirectShellEvaluator scaled(scaled_input);
        check_bound_and_schwarz(original);
        check_bound_and_schwarz(scaled);
        check_normalized_quartets_are_scale_invariant(original, scaled);

        // The old raw-pair prescreen rejected this physically nonzero s quartet.
        FINT shells[4] = {0, 0, 0, 0};
        double raw_s_quartet = 0.0;
        cint2e_cart(&raw_s_quartet,
                    shells,
                    scaled_input.atm.data(),
                    scaled_input.n_atoms,
                    scaled_input.bas.data(),
                    scaled_input.n_shells,
                    scaled_input.env.data(),
                    nullptr);
        require(std::abs(raw_s_quartet) < kTolerance,
                "scaled raw s quartet does not exercise the old prescreen failure");
        xmvb::libcint::LibcintShellWorkspace workspace;
        const auto& normalized_s =
            scaled.evaluate_two_electron_shell_quartet(0, 0, 0, 0, workspace);
        require(std::abs(normalized_s.values[0]) > 1.0e-2,
                "scaled normalized s quartet unexpectedly vanished");

        xmvb::libcint::LibcintMaterializedIntegralProvider provider;
        xmvb::libcint::LibcintMaterializedIntegralProviderOptions screened;
        screened.integral_tolerance = kTolerance;
        screened.use_shell_pair_prescreen = true;
        const auto tiled = screened;
        screened.ao_pair_layout = xmvb::integrals::AoPairLayout::Automatic;
        auto unscreened = screened;
        unscreened.use_shell_pair_prescreen = false;
        const auto original_with_screen = provider.build(original_input, screened);
        const auto original_without_screen = provider.build(original_input, unscreened);
        const auto scaled_with_screen = provider.build(scaled_input, screened);
        const auto scaled_without_screen = provider.build(scaled_input, unscreened);
        check_materialized_equal(original_with_screen, original_without_screen);
        check_materialized_equal(scaled_with_screen, scaled_without_screen);
        check_materialized_equal(original_with_screen, scaled_with_screen);
        check_tiled_operators(original_with_screen, provider.build(original_input, tiled));
        auto chunked = screened;
        chunked.ao_pair_layout = xmvb::integrals::AoPairLayout::Chunked;
        auto buffered = provider.build(original_input, chunked);
        auto streamed = xmvb::libcint::build_materialized_ao_integral_input(std::move(buffered),
                                                                            chunked.ao_pair_layout);
        require(streamed.pair_graph.valid_unique_storage(), "invalid chunked libcint storage");
        require(streamed.pair_graph.integral_count() ==
                    original_with_screen.two_electron_values.size(),
                "chunked libcint integral count changed");
        for (std::size_t i = 0; i < streamed.pair_graph.integral_count(); ++i) {
            const auto eri = streamed.pair_graph.integral(i);
            require(eri.left_pair == original_with_screen.left_pair_indices[i] &&
                        eri.right_pair == original_with_screen.right_pair_indices[i] &&
                        eri.value == original_with_screen.two_electron_values[i],
                    "chunked libcint integral order or FP64 value changed");
        }
        std::cout << "normalized libcint shell screening regression passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
