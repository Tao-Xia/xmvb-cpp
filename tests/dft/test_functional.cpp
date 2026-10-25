#include "dft/xc/functional.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using xmvb::dft::Functional;
using xmvb::dft::XcValues;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char* message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(std::string(message) + ": actual " + std::to_string(actual) +
                                 ", expected " + std::to_string(expected));
    }
}

template <class Action> void require_invalid(Action&& action, const char* message) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

XcValues
evaluate(const Functional& functional, const Eigen::MatrixXd& rho, const Eigen::MatrixXd& sigma) {
    XcValues values;
    functional.evaluate(rho, sigma, &values);
    return values;
}

double total_energy(const Functional& functional,
                    const Eigen::MatrixXd& rho,
                    const Eigen::MatrixXd& sigma) {
    const XcValues values = evaluate(functional, rho, sigma);
    double energy = 0.0;
    for (Eigen::Index i = 0; i < rho.rows(); ++i) {
        energy += rho.row(i).sum() * values.exc[i];
    }
    return energy;
}

void check_lda_exchange() {
    static_assert(!std::is_copy_constructible_v<Functional>);
    static_assert(std::is_move_constructible_v<Functional>);
    const double coefficient = std::cbrt(3.0 / std::acos(-1.0));
    const double spin_factor = std::cbrt(2.0);

    Functional restricted("LDA_X");
    require(restricted.n_spin() == 1 && !restricted.is_gga(), "LDA restricted metadata");
    Eigen::MatrixXd rho(3, 1);
    rho << 0.01, 0.5, 2.0;
    const Eigen::MatrixXd no_sigma(3, 0);
    const auto values = evaluate(restricted, rho, no_sigma);
    require(values.exc.size() == 3 && values.vrho.rows() == 3 && values.vrho.cols() == 1 &&
                values.vsigma.rows() == 3 && values.vsigma.cols() == 0,
            "LDA output shapes");
    for (Eigen::Index i = 0; i < rho.rows(); ++i) {
        const double scale = coefficient * std::cbrt(rho(i, 0));
        near(values.exc[i], -0.75 * scale, 2e-12, "restricted Dirac exchange energy");
        near(values.vrho(i, 0), -scale, 2e-12, "restricted Dirac exchange potential");
    }

    Functional polarized("LDA_X", 2);
    require(polarized.n_spin() == 2 && !polarized.is_gga(), "LDA polarized metadata");
    Eigen::MatrixXd spin_rho(3, 2);
    spin_rho << 0.4, 0.2, 0.7, 0.0, 0.03, 1.2;
    const auto spin_values = evaluate(polarized, spin_rho, Eigen::MatrixXd(3, 0));
    for (Eigen::Index i = 0; i < spin_rho.rows(); ++i) {
        const double alpha = spin_rho(i, 0);
        const double beta = spin_rho(i, 1);
        const double expected_exc = -0.75 * spin_factor * coefficient *
                                    (alpha * std::cbrt(alpha) + beta * std::cbrt(beta)) /
                                    (alpha + beta);
        near(spin_values.exc[i], expected_exc, 2e-12, "polarized Dirac exchange energy");
        near(spin_values.vrho(i, 0),
             -spin_factor * coefficient * std::cbrt(alpha),
             2e-12,
             "alpha Dirac exchange potential");
        near(spin_values.vrho(i, 1),
             -spin_factor * coefficient * std::cbrt(beta),
             2e-12,
             "beta Dirac exchange potential");
    }
    Eigen::MatrixXd full_rho(1, 1);
    full_rho(0, 0) = 0.7;
    near(spin_values.exc[1],
         spin_factor * evaluate(restricted, full_rho, Eigen::MatrixXd(1, 0)).exc[0],
         2e-12,
         "full-polarization exchange scaling");

    Eigen::MatrixXd ignored_sigma = Eigen::MatrixXd::Constant(3, 1, 42.0);
    const auto ignored = evaluate(restricted, rho, ignored_sigma);
    near((ignored.exc - values.exc).norm(), 0.0, 1e-14, "LDA ignores sigma energy");
    near((ignored.vrho - values.vrho).norm(), 0.0, 1e-14, "LDA ignores sigma potential");
}

void check_composition() {
    Eigen::MatrixXd rho(2, 2);
    rho << 0.7, 0.2, 0.3, 0.8;
    Eigen::MatrixXd sigma(2, 3);
    sigma << 0.04, -0.01, 0.07, 0.08, 0.01, 0.03;
    Functional lda("LDA_X", 2);
    Functional gga("GGA_C_PBE", 2);
    Functional mixed(" LDA_X , GGA_C_PBE ", 2);
    require(mixed.is_gga(), "mixed LDA/GGA metadata");
    const auto a = evaluate(lda, rho, Eigen::MatrixXd(2, 0));
    const auto b = evaluate(gga, rho, sigma);
    const auto sum = evaluate(mixed, rho, sigma);
    near((sum.exc - a.exc - b.exc).norm(), 0.0, 1e-12, "mixed XC energy composition");
    near(
        (sum.vrho - a.vrho - b.vrho).norm(), 0.0, 1e-12, "mixed XC density derivative composition");
    near((sum.vsigma - b.vsigma).norm(), 0.0, 1e-12, "mixed XC sigma derivative composition");
}

void check_derivatives(const Functional& functional, Eigen::MatrixXd rho, Eigen::MatrixXd sigma) {
    const auto values = evaluate(functional, rho, sigma);
    const double step = 1e-6;
    for (Eigen::Index i = 0; i < rho.rows(); ++i) {
        for (Eigen::Index channel = 0; channel < rho.cols(); ++channel) {
            rho(i, channel) += step;
            const double plus = total_energy(functional, rho, sigma);
            rho(i, channel) -= 2.0 * step;
            const double minus = total_energy(functional, rho, sigma);
            rho(i, channel) += step;
            const double derivative = (plus - minus) / (2.0 * step);
            near(derivative,
                 values.vrho(i, channel),
                 2e-7 * std::max(1.0, std::abs(derivative)),
                 "Libxc density finite difference");
        }
        for (Eigen::Index channel = 0; channel < sigma.cols(); ++channel) {
            sigma(i, channel) += step;
            const double plus = total_energy(functional, rho, sigma);
            sigma(i, channel) -= 2.0 * step;
            const double minus = total_energy(functional, rho, sigma);
            sigma(i, channel) += step;
            const double derivative = (plus - minus) / (2.0 * step);
            near(derivative,
                 values.vsigma(i, channel),
                 2e-7 * std::max(1.0, std::abs(derivative)),
                 "Libxc sigma finite difference");
        }
    }
}

void check_gga() {
    Functional restricted("GGA_X_PBE,GGA_C_PBE");
    require(restricted.is_gga() && restricted.n_spin() == 1, "restricted PBE metadata");
    Eigen::MatrixXd rho(3, 1), sigma(3, 1);
    rho << 0.3, 0.8, 1.2;
    sigma << 0.017, 0.09, 0.21;
    check_derivatives(restricted, rho, sigma);

    Functional polarized("GGA_X_PBE,GGA_C_PBE", 2);
    Eigen::MatrixXd spin_rho(3, 2), spin_sigma(3, 3);
    spin_rho << 0.4, 0.2, 0.8, 0.3, 0.35, 0.9;
    spin_sigma << 0.018, 0.005, 0.025, 0.06, -0.01, 0.04, 0.13, -0.02, 0.09;
    check_derivatives(polarized, spin_rho, spin_sigma);
    check_derivatives(Functional("LDA_X,GGA_C_PBE", 2), spin_rho, spin_sigma);

    const auto batch = evaluate(polarized, spin_rho, spin_sigma);
    for (Eigen::Index i = 0; i < spin_rho.rows(); ++i) {
        Eigen::MatrixXd one_rho = spin_rho.row(i);
        Eigen::MatrixXd one_sigma = spin_sigma.row(i);
        const auto one = evaluate(polarized, one_rho, one_sigma);
        near(batch.exc[i], one.exc[0], 1e-13, "interleaved XC energy");
        near((batch.vrho.row(i) - one.vrho.row(0)).norm(), 0.0, 1e-13, "interleaved XC vrho");
        near((batch.vsigma.row(i) - one.vsigma.row(0)).norm(), 0.0, 1e-13, "interleaved XC vsigma");
    }

    Eigen::MatrixXd edge_rho(2, 2), edge_sigma(2, 3);
    edge_rho << 0.7, 0.0, 0.2, 0.1;
    edge_sigma << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    const auto edge = evaluate(polarized, edge_rho, edge_sigma);
    require(edge.exc.allFinite() && edge.vrho.allFinite() && edge.vsigma.allFinite(),
            "zero spin density and gradient must remain finite");
}

void check_response_fd(const Functional& functional,
                       const Eigen::MatrixXd& rho,
                       const Eigen::MatrixXd& sigma,
                       const Eigen::MatrixXd& delta_rho,
                       const Eigen::MatrixXd& delta_sigma) {
    Eigen::MatrixXd delta_vrho, delta_vsigma;
    xmvb::dft::XcKernel kernel;
    functional.kernel(rho, sigma, &kernel);
    kernel.apply(delta_rho, delta_sigma, &delta_vrho, &delta_vsigma);
    require(delta_vrho.rows() == rho.rows() && delta_vrho.cols() == rho.cols() &&
                delta_vsigma.rows() == rho.rows() &&
                delta_vsigma.cols() == (functional.is_gga() ? sigma.cols() : 0),
            "XC response output shapes");
    const double step = 1e-5;
    const auto plus = evaluate(functional, rho + step * delta_rho, sigma + step * delta_sigma);
    const auto minus = evaluate(functional, rho - step * delta_rho, sigma - step * delta_sigma);
    const Eigen::MatrixXd fd_rho = (plus.vrho - minus.vrho) / (2.0 * step);
    const Eigen::MatrixXd fd_sigma = (plus.vsigma - minus.vsigma) / (2.0 * step);
    for (Eigen::Index i = 0; i < rho.rows(); ++i) {
        for (Eigen::Index spin = 0; spin < rho.cols(); ++spin) {
            near(delta_vrho(i, spin),
                 fd_rho(i, spin),
                 3e-7 * std::max(1.0, std::abs(fd_rho(i, spin))),
                 "XC response density finite difference");
        }
        for (Eigen::Index channel = 0; channel < delta_vsigma.cols(); ++channel) {
            near(delta_vsigma(i, channel),
                 fd_sigma(i, channel),
                 3e-7 * std::max(1.0, std::abs(fd_sigma(i, channel))),
                 "XC response sigma finite difference");
        }
    }
}

void check_response() {
    Eigen::MatrixXd rho(3, 1), sigma(3, 1), delta_rho(3, 1);
    rho << 0.4, 0.8, 1.3;
    sigma << 0.02, 0.07, 0.15;
    delta_rho << 0.11, -0.23, 0.17;
    Eigen::MatrixXd delta_sigma(3, 1);
    delta_sigma << 0.008, -0.012, 0.019;
    const Eigen::MatrixXd no_sigma(3, 0);
    check_response_fd(Functional("LDA_X,LDA_C_VWN"), rho, no_sigma, delta_rho, no_sigma);
    check_response_fd(Functional("HYB_LDA_XC_LDA0"), rho, no_sigma, delta_rho, no_sigma);
    check_response_fd(Functional("GGA_X_PBE,GGA_C_PBE"), rho, sigma, delta_rho, delta_sigma);
    for (const char* name :
         {"BLYP", "BP86", "PW91", "revPBE", "PBEsol", "PBE0", "B3LYP", "B3LYP5"}) {
        check_derivatives(Functional(name), rho, sigma);
        check_response_fd(Functional(name), rho, sigma, delta_rho, delta_sigma);
    }

    Eigen::MatrixXd spin_rho(3, 2), spin_sigma(3, 3);
    Eigen::MatrixXd spin_direction(3, 2), sigma_direction(3, 3);
    spin_rho << 0.7, 0.3, 0.4, 0.9, 1.1, 0.6;
    spin_sigma << 0.05, 0.01, 0.03, 0.08, 0.02, 0.06, 0.12, 0.03, 0.10;
    spin_direction << 0.08, -0.12, -0.16, 0.10, 0.21, 0.07;
    sigma_direction << 0.012, -0.005, 0.007, -0.008, 0.011, -0.009, 0.015, 0.006, -0.013;
    check_response_fd(Functional("LDA_X,LDA_C_VWN", 2),
                      spin_rho,
                      Eigen::MatrixXd(3, 0),
                      spin_direction,
                      Eigen::MatrixXd(3, 0));
    check_response_fd(Functional("HYB_LDA_XC_LDA0", 2),
                      spin_rho,
                      Eigen::MatrixXd(3, 0),
                      spin_direction,
                      Eigen::MatrixXd(3, 0));
    check_response_fd(Functional("GGA_X_PBE,GGA_C_PBE", 2),
                      spin_rho,
                      spin_sigma,
                      spin_direction,
                      sigma_direction);
    for (const char* name :
         {"BLYP", "BP86", "PW91", "revPBE", "PBEsol", "PBE0", "B3LYP", "B3LYP5"}) {
        check_derivatives(Functional(name, 2), spin_rho, spin_sigma);
        check_response_fd(
            Functional(name, 2), spin_rho, spin_sigma, spin_direction, sigma_direction);
    }
    check_response_fd(
        Functional("LDA_X,GGA_C_PBE", 2), spin_rho, spin_sigma, spin_direction, sigma_direction);

    // Isolate the off-diagonal gradient channel and the beta density column.
    Eigen::MatrixXd cross_rho = Eigen::MatrixXd::Zero(3, 2);
    Eigen::MatrixXd cross_sigma = Eigen::MatrixXd::Zero(3, 3);
    cross_rho.col(1) << 0.13, -0.17, 0.09;
    cross_sigma.col(1) << 0.007, -0.006, 0.011;
    check_response_fd(
        Functional("GGA_C_PBE", 2), spin_rho, spin_sigma, cross_rho, Eigen::MatrixXd::Zero(3, 3));
    check_response_fd(
        Functional("GGA_C_PBE", 2), spin_rho, spin_sigma, Eigen::MatrixXd::Zero(3, 2), cross_sigma);

    // A pointwise Hessian must be symmetric across the rho/sigma blocks.
    Functional polarized("LDA_X,GGA_C_PBE", 2);
    const Eigen::MatrixXd one_rho = spin_rho.topRows(1);
    const Eigen::MatrixXd one_sigma = spin_sigma.topRows(1);
    const Eigen::MatrixXd a_rho = spin_direction.topRows(1);
    const Eigen::MatrixXd a_sigma = sigma_direction.topRows(1);
    const Eigen::MatrixXd b_rho = cross_rho.topRows(1);
    const Eigen::MatrixXd b_sigma = cross_sigma.topRows(1);
    Eigen::MatrixXd ha_rho, ha_sigma, hb_rho, hb_sigma;
    xmvb::dft::XcKernel kernel;
    polarized.kernel(one_rho, one_sigma, &kernel);
    kernel.apply(a_rho, a_sigma, &ha_rho, &ha_sigma);
    kernel.apply(b_rho, b_sigma, &hb_rho, &hb_sigma);
    const double a_h_b =
        (a_rho.cwiseProduct(hb_rho)).sum() + (a_sigma.cwiseProduct(hb_sigma)).sum();
    const double b_h_a =
        (b_rho.cwiseProduct(ha_rho)).sum() + (b_sigma.cwiseProduct(ha_sigma)).sum();
    near(a_h_b, b_h_a, 1e-12, "XC response Hessian symmetry");

    // The final point crosses the internal 128-point Libxc tile boundary.
    Eigen::MatrixXd tiled_rho(129, 2), tiled_sigma(129, 3);
    Eigen::MatrixXd tiled_direction_rho(129, 2), tiled_direction_sigma(129, 3);
    for (Eigen::Index i = 0; i < 129; ++i) {
        tiled_rho.row(i) = one_rho.row(0);
        tiled_sigma.row(i) = one_sigma.row(0);
        tiled_direction_rho.row(i) = a_rho.row(0);
        tiled_direction_sigma.row(i) = a_sigma.row(0);
    }
    Eigen::MatrixXd tiled_vrho, tiled_vsigma;
    polarized.kernel(tiled_rho, tiled_sigma, &kernel);
    kernel.apply(tiled_direction_rho, tiled_direction_sigma, &tiled_vrho, &tiled_vsigma);
    for (Eigen::Index i = 0; i < 129; ++i) {
        near((tiled_vrho.row(i) - ha_rho.row(0)).norm(),
             0.0,
             1e-13,
             "XC response tile density parity");
        near((tiled_vsigma.row(i) - ha_sigma.row(0)).norm(),
             0.0,
             1e-13,
             "XC response tile sigma parity");
    }
}

void check_aliases() {
    for (const char* name : {"LDA",
                             "SVWN",
                             "SVWN5",
                             "PBE",
                             "BLYP",
                             "BP86",
                             "PW91",
                             "revPBE",
                             "PBEsol",
                             "PBE0",
                             "B3LYP",
                             "B3LYP5"}) {
        const std::string resolved = Functional::canonical_name(name);
        require(Functional::canonical_name(resolved) == resolved, "XC alias is not canonical");
        for (int spin : {1, 2}) {
            const Functional functional(name, spin);
            const Functional explicit_name(resolved, spin);
            require(functional.n_spin() == spin && functional.is_gga() == explicit_name.is_gga(),
                    "XC alias metadata differs from explicit components");
            near(functional.exact_exchange(),
                 explicit_name.exact_exchange(),
                 0.0,
                 "XC alias changed exact exchange");
        }
    }
    near(Functional("PBE0").exact_exchange(), 0.25, 1e-15, "PBE0 exact exchange fraction");
    near(Functional(" b3lyp ").exact_exchange(), 0.20, 1e-15, "B3LYP exact exchange fraction");
    near(Functional("B3LYP5").exact_exchange(), 0.20, 1e-15, "B3LYP5 exact exchange fraction");
    near(Functional("BLYP").exact_exchange(), 0.0, 0.0, "pure GGA has exact exchange");
    require(Functional::canonical_name("B3LYP") != Functional::canonical_name("B3LYP5"),
            "different B3LYP VWN definitions were conflated");
}

void check_rejections() {
    require_invalid([] { Functional bad("LDA_X", 0); }, "n_spin=0 accepted");
    require_invalid([] { Functional bad("LDA_X", 3); }, "n_spin=3 accepted");
    require_invalid([] { Functional bad("LDA_K_TF"); }, "kinetic component accepted");
    for (const char* name :
         {"HYB_GGA_XC_CAM_B3LYP", "HYB_GGA_XC_HSE06", "MGGA_X_SCAN", "GGA_XC_VV10"})
        require_invalid([&] { Functional bad(name); },
                        "unsupported nonlocal/meta-GGA component accepted");
    require_invalid([] { Functional bad("MGGA_X_TPSS"); }, "meta-GGA component accepted");
    require_invalid([] { Functional bad("NOT_A_LIBXC_NAME"); }, "unknown component accepted");
    require_invalid([] { Functional bad("LDA_X,"); }, "empty component accepted");

    Functional gga("GGA_X_PBE", 2);
    Eigen::MatrixXd rho = Eigen::MatrixXd::Constant(1, 2, 0.5);
    Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(1, 3);
    XcValues values;
    require_invalid([&] { gga.evaluate(rho, Eigen::MatrixXd(1, 0), &values); },
                    "missing GGA sigma accepted");
    rho(0, 0) = -1e-4;
    require_invalid([&] { gga.evaluate(rho, sigma, &values); }, "negative density accepted");
    rho(0, 0) = std::numeric_limits<double>::quiet_NaN();
    require_invalid([&] { gga.evaluate(rho, sigma, &values); }, "nonfinite density accepted");
    rho(0, 0) = 0.5;
    sigma(0, 2) = -1e-4;
    require_invalid([&] { gga.evaluate(rho, sigma, &values); }, "negative diagonal sigma accepted");
    sigma(0, 2) = 0.0;
    sigma(0, 1) = std::numeric_limits<double>::infinity();
    require_invalid([&] { gga.evaluate(rho, sigma, &values); }, "nonfinite cross sigma accepted");
    sigma(0, 0) = 0.02;
    sigma(0, 2) = 0.02;
    sigma(0, 1) = -0.01;
    require(evaluate(gga, rho, sigma).exc.allFinite(), "negative cross sigma rejected");

    rho(0, 1) = -1e-15;
    require(evaluate(Functional("LDA_X", 2), rho, Eigen::MatrixXd(1, 0)).exc.allFinite(),
            "roundoff-scale negative density rejected");
    rho(0, 1) = 0.5;
    sigma(0, 0) = -1e-16;
    sigma(0, 1) = 0.0;
    const auto rounded = evaluate(gga, rho, sigma);
    require(rounded.exc.allFinite() && rounded.vrho.allFinite() && rounded.vsigma.allFinite(),
            "roundoff-scale negative sigma rejected");

    Eigen::MatrixXd delta_rho = Eigen::MatrixXd::Constant(1, 2, 0.1);
    Eigen::MatrixXd delta_sigma = Eigen::MatrixXd::Constant(1, 3, 0.01);
    Eigen::MatrixXd delta_vrho, delta_vsigma;
    xmvb::dft::XcKernel kernel;
    require_invalid([&] { gga.kernel(rho, Eigen::MatrixXd(1, 0), &kernel); },
                    "response accepted missing GGA sigma");
    gga.kernel(rho, sigma, &kernel);
    require_invalid(
        [&] { kernel.apply(Eigen::MatrixXd(1, 1), delta_sigma, &delta_vrho, &delta_vsigma); },
        "response accepted wrong density direction shape");
    delta_sigma(0, 1) = std::numeric_limits<double>::quiet_NaN();
    require_invalid([&] { kernel.apply(delta_rho, delta_sigma, &delta_vrho, &delta_vsigma); },
                    "response accepted nonfinite sigma direction");
}

} // namespace

int main() {
    try {
        check_lda_exchange();
        check_composition();
        check_gga();
        check_response();
        check_aliases();
        check_rejections();
    } catch (const std::exception& error) {
        std::cerr << "DFT functional test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "dft_functional=PASS\n";
}
