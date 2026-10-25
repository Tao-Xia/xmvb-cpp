#include "core/workspace.hpp"
#include "dft/driver/potential.hpp"
#include "dft/grid/atomic.hpp"
#include "dft/integration/cache.hpp"
#include "dft/integration/potential.hpp"
#include "dft/xc/functional.hpp"
#include "integrals/ao/libcint_input.hpp"
#include "integrals/ao/graph.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

extern "C" {
#include "cint.h"
}

namespace {

using Matrix = Eigen::MatrixXd;
using MatrixRef = Eigen::Ref<const Matrix>;
using xmvb::dft::XcResult;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Callable> void require_invalid(Callable&& call, const char* message) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

struct RestoreWorkspace {
    std::size_t original = xmvb::workspace::limit();
    ~RestoreWorkspace() {
        xmvb::workspace::set_limit(original);
    }
};

xmvb::integrals::LibcintInput make_two_s_basis() {
    xmvb::integrals::LibcintInput input;
    input.n_atoms = 1;
    input.n_shells = 2;
    input.n_gaussian_primitives = 2;
    input.atm.assign(ATM_SLOTS, 0);
    input.atm[CHARGE_OF] = 1;
    input.atm[PTR_COORD] = PTR_ENV_START;
    input.env.assign(PTR_ENV_START + 3, 0.0);
    input.bas.assign(2 * BAS_SLOTS, 0);
    input.basidx = {0, 1, 1, 1};
    for (int shell = 0; shell < 2; ++shell) {
        const int offset = static_cast<int>(input.env.size());
        input.env.push_back(shell == 0 ? 0.7 : 1.4);
        input.env.push_back(1.0);
        input.bas[shell * BAS_SLOTS + ATOM_OF] = 0;
        input.bas[shell * BAS_SLOTS + ANG_OF] = 0;
        input.bas[shell * BAS_SLOTS + NPRIM_OF] = 1;
        input.bas[shell * BAS_SLOTS + NCTR_OF] = 1;
        input.bas[shell * BAS_SLOTS + PTR_EXP] = offset;
        input.bas[shell * BAS_SLOTS + PTR_COEFF] = offset + 1;
    }
    return input;
}

xmvb::dft::Grid make_grid() {
    xmvb::dft::Grid grid;
    grid.points.resize(12, 3);
    grid.weights.resize(12);
    for (Eigen::Index i = 0; i < 12; ++i) {
        grid.points(i, 0) = 0.13 * static_cast<double>(i - 5);
        grid.points(i, 1) = 0.07 * static_cast<double>((i * 3) % 7 - 3);
        grid.points(i, 2) = 0.11 * static_cast<double>((i * 5) % 9 - 4);
        grid.weights[i] = 0.04 + 0.002 * static_cast<double>(i);
    }
    return grid;
}

XcResult integrate(const xmvb::dft::XcIntegrator& integrator, const Matrix& density) {
    const std::array<MatrixRef, 1> refs{density};
    return integrator.evaluate(refs);
}

XcResult
integrate(const xmvb::dft::XcIntegrator& integrator, const Matrix& alpha, const Matrix& beta) {
    const std::array<MatrixRef, 2> refs{alpha, beta};
    return integrator.evaluate(refs);
}

std::array<Matrix, 2>
respond(const xmvb::dft::XcIntegrator& integrator, const Matrix& density, const Matrix& direction) {
    const std::array<MatrixRef, 1> densities{density};
    const std::array<MatrixRef, 1> directions{direction};
    return integrator.response(densities, directions);
}

std::array<Matrix, 2> respond(const xmvb::dft::XcIntegrator& integrator,
                              const Matrix& alpha,
                              const Matrix& beta,
                              const Matrix& direction_alpha,
                              const Matrix& direction_beta) {
    const std::array<MatrixRef, 2> densities{alpha, beta};
    const std::array<MatrixRef, 2> directions{direction_alpha, direction_beta};
    return integrator.response(densities, directions);
}

double contraction(const Matrix& potential, const Matrix& direction) {
    return (potential.array() * direction.array()).sum();
}

void check_functional() {
    Matrix rho(3, 1);
    rho << 0.01, 0.5, 2.0;
    Matrix no_sigma(3, 0);
    xmvb::dft::XcValues exchange_values;
    xmvb::dft::Functional exchange("LDA_X");
    exchange.evaluate(rho, no_sigma, &exchange_values);
    require(exchange.n_spin() == 1 && !exchange.is_gga() && exchange_values.vsigma.rows() == 3 &&
                exchange_values.vsigma.cols() == 0,
            "restricted LDA functional shape mismatch");
    const double coefficient = std::cbrt(3.0 / std::acos(-1.0));
    for (Eigen::Index i = 0; i < rho.rows(); ++i) {
        const double scale = coefficient * std::cbrt(rho(i, 0));
        require(std::abs(exchange_values.exc[i] + 0.75 * scale) < 2.0e-12,
                "Dirac exchange energy per particle mismatch");
        require(std::abs(exchange_values.vrho(i, 0) + scale) < 2.0e-12,
                "Dirac exchange potential mismatch");
    }
    xmvb::dft::XcValues correlation_values, combined_values;
    xmvb::dft::Functional correlation("LDA_C_VWN");
    correlation.evaluate(rho, no_sigma, &correlation_values);
    xmvb::dft::Functional combined;
    combined.evaluate(rho, no_sigma, &combined_values);
    require((combined_values.exc - exchange_values.exc - correlation_values.exc).norm() < 1.0e-13 &&
                (combined_values.vrho - exchange_values.vrho - correlation_values.vrho).norm() <
                    1.0e-13,
            "default LDA composition mismatch");

    xmvb::dft::Functional pbe("GGA_X_PBE,GGA_C_PBE", 2);
    Matrix spin_rho(3, 2);
    spin_rho << 0.8, 0.3, 0.6, 0.2, 0.4, 0.1;
    Matrix sigma(3, 3);
    sigma << 0.20, 0.04, 0.10, 0.12, 0.02, 0.08, 0.07, 0.01, 0.03;
    xmvb::dft::XcValues pbe_values;
    pbe.evaluate(spin_rho, sigma, &pbe_values);
    require(pbe.n_spin() == 2 && pbe.is_gga() && pbe_values.vrho.rows() == 3 &&
                pbe_values.vrho.cols() == 2 && pbe_values.vsigma.rows() == 3 &&
                pbe_values.vsigma.cols() == 3 &&
                pbe_values.vsigma.col(1).cwiseAbs().maxCoeff() > 1.0e-10,
            "polarized PBE mixed-gradient derivative is missing");

    require_invalid([] { xmvb::dft::Functional bad("LDA_K_TF"); },
                    "kinetic functional accepted as XC");
    require_invalid([] { xmvb::dft::Functional bad("LDA_X,"); }, "empty XC component accepted");
    require_invalid([] { xmvb::dft::Functional bad("NOT_A_LIBXC_NAME"); },
                    "unknown XC functional accepted");
    rho(0, 0) = -1.0e-5;
    require_invalid([&] { exchange.evaluate(rho, no_sigma, &exchange_values); },
                    "negative density accepted");
    rho(0, 0) = std::numeric_limits<double>::quiet_NaN();
    require_invalid([&] { exchange.evaluate(rho, no_sigma, &exchange_values); },
                    "nonfinite density accepted");
}

void check_directional_derivative(const char* name, bool polarized) {
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    const xmvb::dft::Functional functional(name, polarized ? 2 : 1);
    const xmvb::dft::XcIntegrator integrator(input, grid, functional);
    Matrix alpha(2, 2), beta(2, 2), direction_alpha(2, 2), direction_beta(2, 2);
    alpha << 0.9, 0.1, 0.1, 0.5;
    beta << 0.25, 0.03, 0.03, 0.35;
    direction_alpha << 0.02, -0.01, -0.01, -0.015;
    direction_beta << -0.008, 0.006, 0.006, 0.013;

    const XcResult base =
        polarized ? integrate(integrator, alpha, beta) : integrate(integrator, alpha);
    const double step = 1.0e-5;
    const Matrix alpha_plus = alpha + step * direction_alpha;
    const Matrix alpha_minus = alpha - step * direction_alpha;
    const double plus_alpha = polarized ? integrate(integrator, alpha_plus, beta).energy
                                        : integrate(integrator, alpha_plus).energy;
    const double minus_alpha = polarized ? integrate(integrator, alpha_minus, beta).energy
                                         : integrate(integrator, alpha_minus).energy;
    const double derivative_alpha = (plus_alpha - minus_alpha) / (2.0 * step);
    const double expected_alpha = contraction(base.potential[0], direction_alpha);
    require(std::abs(derivative_alpha - expected_alpha) <
                3.0e-7 * std::max(1.0, std::abs(expected_alpha)),
            "alpha XC potential is not the energy derivative");
    if (polarized) {
        const Matrix beta_plus = beta + step * direction_beta;
        const Matrix beta_minus = beta - step * direction_beta;
        const double plus_beta = integrate(integrator, alpha, beta_plus).energy;
        const double minus_beta = integrate(integrator, alpha, beta_minus).energy;
        const double derivative_beta = (plus_beta - minus_beta) / (2.0 * step);
        const double expected_beta = contraction(base.potential[1], direction_beta);
        require(std::abs(derivative_beta - expected_beta) <
                    3.0e-7 * std::max(1.0, std::abs(expected_beta)),
                "beta XC potential is not the energy derivative");
        require(base.electron_count[1] > 0.0, "missing beta electron count");
    }
    require(base.electron_count[0] > 0.0 && std::isfinite(base.energy) &&
                base.potential[0].allFinite(),
            "invalid XC grid result");
}

void check_unpolarized_equivalence(const char* name) {
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    const xmvb::dft::Functional restricted(name, 1);
    const xmvb::dft::Functional polarized(name, 2);
    const xmvb::dft::XcIntegrator rks(input, grid, restricted);
    const xmvb::dft::XcIntegrator uks(input, grid, polarized);
    Matrix density(2, 2);
    density << 0.9, 0.1, 0.1, 0.5;
    const Matrix half = 0.5 * density;
    const auto r = integrate(rks, density);
    const auto u = integrate(uks, half, half);
    require(std::abs(r.energy - u.energy) < 1.0e-11 &&
                std::abs(r.electron_count[0] - u.electron_count[0] - u.electron_count[1]) <
                    1.0e-12 &&
                (r.potential[0] - u.potential[0]).norm() < 1.0e-10 &&
                (r.potential[0] - u.potential[1]).norm() < 1.0e-10,
            "unpolarized UKS does not equal restricted XC");
}

void check_tiling_and_budget() {
    RestoreWorkspace restore;
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    const xmvb::dft::Functional functional("GGA_X_PBE,GGA_C_PBE", 2);
    const xmvb::dft::XcIntegrator integrator(input, grid, functional);
    Matrix alpha(2, 2), beta(2, 2);
    alpha << 0.9, 0.1, 0.1, 0.5;
    beta << 0.25, 0.03, 0.03, 0.35;
    xmvb::workspace::set_limit(10000);
    const auto full = integrate(integrator, alpha, beta);
    // nbf=2, spin=2: 2*2*2*8=64 output bytes; one GGA row uses
    // (5*2 + 2 + 7*2 + 4*3)*8=304 scratch bytes.
    xmvb::workspace::set_limit(368);
    const auto tiled = integrate(integrator, alpha, beta);
    require(std::abs(full.energy - tiled.energy) < 1.0e-12 &&
                std::abs(full.electron_count[0] - tiled.electron_count[0]) < 1.0e-12 &&
                std::abs(full.electron_count[1] - tiled.electron_count[1]) < 1.0e-12 &&
                (full.potential[0] - tiled.potential[0]).norm() < 1.0e-11 &&
                (full.potential[1] - tiled.potential[1]).norm() < 1.0e-11,
            "XC integral depends on tile size");
    xmvb::workspace::set_limit(367);
    bool exhausted = false;
    try {
        (void)integrate(integrator, alpha, beta);
    } catch (const std::runtime_error& error) {
        exhausted = std::string(error.what()).find("resource_limit") != std::string::npos;
    }
    require(exhausted, "insufficient XC workspace was not reported");
}

/** @brief Test the production full J/K+XC model, not a copied hybrid formula. */
void check_hybrid_model(const char* name, int n_spin) {
    const auto basis = make_two_s_basis();
    const auto grid = make_grid();
    const xmvb::dft::Functional functional(name, n_spin);
    xmvb::integrals::AoIntegralInput ao;
    ao.n_basis_functions = 2;
    ao.ao_core_hamiltonian_matrix.resize(2, 2);
    ao.ao_core_hamiltonian_matrix << -1.2, 0.1, 0.1, -0.4;
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(
        {0, 1, 1, 2, 2, 2}, {0, 0, 1, 0, 1, 2}, {0.7, 0.12, 0.22, 0.3, 0.04, 0.5}, 2);
    xmvb::dft::KsPotential potential(basis, grid, functional, ao);
    using Spin = xmvb::scf::SpinMatrices;
    Spin density, direction, second, plus, minus;
    density[0] = (Matrix(2, 2) << 0.9, 0.1, 0.1, 0.5).finished();
    direction[0] = (Matrix(2, 2) << 0.13, -0.07, -0.07, -0.11).finished();
    second[0] = (Matrix(2, 2) << -0.04, 0.02, 0.02, 0.08).finished();
    if (n_spin == 2) {
        density[1] = (Matrix(2, 2) << 0.25, 0.03, 0.03, 0.35).finished();
        direction[1] = (Matrix(2, 2) << -0.06, 0.03, 0.03, 0.09).finished();
        second[1] = (Matrix(2, 2) << 0.11, -0.05, -0.05, -0.03).finished();
    }
    const auto refs = [](const Spin& matrices) {
        return std::array<MatrixRef, 2>{matrices[0], matrices[1]};
    };
    const auto evaluate = [&](const Spin& at) {
        const auto channels = refs(at);
        return potential.evaluate(std::span(channels).first(n_spin));
    };
    const auto apply = [&](const Spin& v) {
        const auto channels = refs(density), delta = refs(v);
        return potential.response(std::span(channels).first(n_spin),
                                  std::span(delta).first(n_spin));
    };
    constexpr double step = 1e-5;
    for (int spin = 0; spin < n_spin; ++spin) {
        plus[spin] = density[spin] + step * direction[spin];
        minus[spin] = density[spin] - step * direction[spin];
    }
    const auto physical = evaluate(density), ep = evaluate(plus), em = evaluate(minus);
    const auto hp = apply(direction), hq = apply(second);
    double derivative = 0.0, reciprocity = 0.0;
    for (int spin = 0; spin < n_spin; ++spin) {
        derivative += (physical.fock[spin].array() * direction[spin].array()).sum();
        require((hp[spin] - (ep.fock[spin] - em.fock[spin]) / (2 * step)).norm() < 2e-8,
                "full KS response differs from finite-difference physical Fock");
        reciprocity += (second[spin].array() * hp[spin].array()).sum() -
                       (direction[spin].array() * hq[spin].array()).sum();
    }
    require(std::abs(derivative - (ep.electronic_energy - em.electronic_energy) / (2 * step)) <
                2e-9,
            "full hybrid energy derivative differs from physical Fock");
    require(std::abs(reciprocity) < 1e-11, "full hybrid response lost cross-spin reciprocity");
}

void check_zero_beta() {
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    Matrix alpha(2, 2);
    alpha << 0.9, 0.1, 0.1, 0.5;
    const Matrix beta = Matrix::Zero(2, 2);
    for (const char* name : {"LDA_X,LDA_C_VWN", "GGA_X_PBE,GGA_C_PBE"}) {
        const xmvb::dft::Functional functional(name, 2);
        const xmvb::dft::XcIntegrator integrator(input, grid, functional);
        const auto result = integrate(integrator, alpha, beta);
        require(result.electron_count[1] == 0.0 && std::isfinite(result.energy) &&
                    result.potential[0].allFinite() && result.potential[1].allFinite(),
                "zero-beta XC result is invalid");
    }
}

void check_potential_response(const char* name, bool polarized) {
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    const xmvb::dft::Functional functional(name, polarized ? 2 : 1);
    const xmvb::dft::XcIntegrator integrator(input, grid, functional);
    Matrix alpha(2, 2), beta(2, 2), direction_alpha(2, 2), direction_beta(2, 2), second_alpha(2, 2),
        second_beta(2, 2);
    alpha << 0.9, 0.1, 0.1, 0.5;
    beta << 0.25, 0.03, 0.03, 0.35;
    // The alpha perturbation is negative definite. Its directional grid
    // density must remain signed through the AO density evaluation.
    direction_alpha << -0.12, 0.02, 0.02, -0.04;
    direction_beta << 0.05, -0.01, -0.01, 0.025;
    second_alpha << 0.017, -0.009, -0.009, 0.013;
    second_beta << -0.011, 0.004, 0.004, 0.019;

    const auto response = polarized
                              ? respond(integrator, alpha, beta, direction_alpha, direction_beta)
                              : respond(integrator, alpha, direction_alpha);
    const double step = 2.0e-5;
    const auto plus =
        polarized
            ? integrate(integrator, alpha + step * direction_alpha, beta + step * direction_beta)
            : integrate(integrator, alpha + step * direction_alpha);
    const auto minus =
        polarized
            ? integrate(integrator, alpha - step * direction_alpha, beta - step * direction_beta)
            : integrate(integrator, alpha - step * direction_alpha);
    const int spins = polarized ? 2 : 1;
    for (int spin = 0; spin < spins; ++spin) {
        const Matrix difference = (plus.potential[spin] - minus.potential[spin]) / (2.0 * step);
        require(response[spin].allFinite() && (response[spin] - difference).norm() <
                                                  2.0e-6 * std::max(1.0, difference.norm()),
                "analytic XC potential response differs from finite difference");
    }

    const auto second_response = polarized
                                     ? respond(integrator, alpha, beta, second_alpha, second_beta)
                                     : respond(integrator, alpha, second_alpha);
    double forward = contraction(response[0], second_alpha);
    double reverse = contraction(second_response[0], direction_alpha);
    if (polarized) {
        forward += contraction(response[1], second_beta);
        reverse += contraction(second_response[1], direction_beta);
    }
    require(std::abs(forward - reverse) <
                3.0e-8 * std::max({1.0, std::abs(forward), std::abs(reverse)}),
            "XC response violates bilinear symmetry");

    if (polarized) {
        const Matrix zero = Matrix::Zero(2, 2);
        const auto alpha_only = respond(integrator, alpha, beta, direction_alpha, zero);
        const auto beta_only = respond(integrator, alpha, beta, zero, direction_beta);
        require(alpha_only[1].norm() > 1.0e-10 && beta_only[0].norm() > 1.0e-10,
                "polarized XC response lost cross-spin coupling");
        require((response[0] - alpha_only[0] - beta_only[0]).norm() < 1.0e-10 &&
                    (response[1] - alpha_only[1] - beta_only[1]).norm() < 1.0e-10,
                "XC response is not linear in signed AO directions");
    }

    // The same integrator must invalidate reference-only data when the accepted
    // density changes, even if its shape and allocation stay unchanged.
    alpha += 0.03 * direction_alpha;
    const auto changed = polarized
                             ? respond(integrator, alpha, beta, direction_alpha, direction_beta)
                             : respond(integrator, alpha, direction_alpha);
    const auto next_plus =
        polarized
            ? integrate(integrator, alpha + step * direction_alpha, beta + step * direction_beta)
            : integrate(integrator, alpha + step * direction_alpha);
    const auto next_minus =
        polarized
            ? integrate(integrator, alpha - step * direction_alpha, beta - step * direction_beta)
            : integrate(integrator, alpha - step * direction_alpha);
    for (int spin = 0; spin < spins; ++spin)
        require((changed[spin] -
                 (next_plus.potential[spin] - next_minus.potential[spin]) / (2.0 * step))
                        .norm() < 2e-6,
                "XC response reused a stale reference density");
}

void check_response_tiling_and_budget() {
    RestoreWorkspace restore;
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    const xmvb::dft::Functional functional("GGA_X_PBE,GGA_C_PBE", 2);
    const xmvb::dft::XcIntegrator integrator(input, grid, functional);
    Matrix alpha(2, 2), beta(2, 2), delta_alpha(2, 2), delta_beta(2, 2);
    alpha << 0.9, 0.1, 0.1, 0.5;
    beta << 0.25, 0.03, 0.03, 0.35;
    delta_alpha << -0.12, 0.02, 0.02, -0.04;
    delta_beta << 0.05, -0.01, -0.01, 0.025;
    xmvb::workspace::set_limit(10000);
    const auto full = respond(integrator, alpha, beta, delta_alpha, delta_beta);
    // nbf=2, spin=2: 64 output bytes. One GGA row reserves
    // (5*2 + 2 + 20*2 + 14*3)*8 = 752 scratch bytes.
    xmvb::workspace::set_limit(816);
    const auto tiled = respond(integrator, alpha, beta, delta_alpha, delta_beta);
    require((full[0] - tiled[0]).norm() < 1.0e-10 && (full[1] - tiled[1]).norm() < 1.0e-10,
            "XC response depends on tile size");
    xmvb::workspace::set_limit(815);
    bool exhausted = false;
    try {
        (void)respond(integrator, alpha, beta, delta_alpha, delta_beta);
    } catch (const std::runtime_error& error) {
        exhausted = std::string(error.what()).find("resource_limit") != std::string::npos;
    }
    require(exhausted, "insufficient XC response workspace was not reported");
}

void check_residency() {
    RestoreWorkspace restore;
    const auto before = xmvb::workspace::retained();
    const auto input = make_two_s_basis();
    const auto grid = make_grid();
    xmvb::workspace::set_limit(before + 10000);
    {
        xmvb::dft::GridBasis cache(input, grid, true);
        cache.prepare(64, 304);
        require(cache.resident_rows() == 12, "AO grid did not become resident");
        xmvb::integrals::BasisValues scratch;
        const auto first = cache.tile(0, 12, &scratch);
        const auto repeated = cache.tile(0, 12, &scratch);
        require(first.value.data() == repeated.value.data() && scratch.value.size() == 0,
                "resident AO tile was copied or regenerated");
        const auto slice = cache.tile(3, 4, &scratch);
        require((slice.value - first.value.middleRows(3, 4)).norm() == 0.0,
                "strided resident AO view is incorrect");
        require(xmvb::workspace::retained() == before + 12 * 2 * 4 * sizeof(double),
                "resident AO bytes are not charged once");

        xmvb::dft::ResponseCache response;
        Matrix a = Matrix::Identity(2, 2), b = 0.5 * a;
        const std::array<MatrixRef, 2> key{a, b};
        response.prepare(key, 12, 64, 752, true);
        auto* panel = response.tile(0, response.rows(0, 12));
        require(panel != nullptr, "reference response tile was not reserved");
        panel->kernel.rho2 = Matrix::Ones(1, 1);
        response.select(key);
        require(response.tile(0, response.rows(0, 12)) == panel && panel->kernel.rho2.size() == 1,
                "unchanged reference was invalidated");
        b(0, 0) += 1e-14;
        response.select(key);
        require(response.tile(0, 12) == nullptr,
                "changed reference incorrectly reused the old response");
    }
    require(xmvb::workspace::retained() == before, "XC residency leaked its reservation");
}

} // namespace

int main() {
    check_functional();
    check_directional_derivative("LDA_X,LDA_C_VWN", false);
    check_directional_derivative("LDA_X,LDA_C_VWN", true);
    check_directional_derivative("GGA_X_PBE,GGA_C_PBE", false);
    check_directional_derivative("GGA_X_PBE,GGA_C_PBE", true);
    check_directional_derivative("GGA_C_PBE", true);
    check_unpolarized_equivalence("LDA_X,LDA_C_VWN");
    check_unpolarized_equivalence("GGA_X_PBE,GGA_C_PBE");
    check_tiling_and_budget();
    check_zero_beta();
    check_potential_response("LDA_X,LDA_C_VWN", false);
    check_potential_response("LDA_X,LDA_C_VWN", true);
    check_potential_response("GGA_X_PBE,GGA_C_PBE", false);
    check_potential_response("GGA_X_PBE,GGA_C_PBE", true);
    for (const char* name : {"PBE", "BLYP", "PBE0", "B3LYP", "B3LYP5", "HYB_LDA_XC_LDA0"})
        for (int spin : {1, 2})
            check_hybrid_model(name, spin);
    check_response_tiling_and_budget();
    check_residency();
    std::cout << "dft_xc=PASS\n";
}
