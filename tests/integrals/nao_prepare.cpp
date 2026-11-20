#include "integrals/nao/prepare.hpp"
#include "integrals/lri/fit.hpp"
#include <Eigen/Cholesky>
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace {
using namespace xmvb::integrals;
constexpr double pi = std::numbers::pi;
void near(double value, double expected, double tolerance, const char* what) {
    const double error = std::abs(value - expected);
    std::cout << what << " error=" << error << '\n';
    if (!std::isfinite(value) || error > tolerance)
        throw std::runtime_error(what);
}
std::shared_ptr<nao::Radial>
gaussian(int l, double exponent, double coefficient, std::size_t count = 4001) {
    std::vector<double> r(count), f(count);
    for (std::size_t i = 0; i < count; ++i) {
        r[i] = 8.0 * i / (count - 1);
        f[i] = coefficient * std::exp(-exponent * r[i] * r[i]);
    }
    f.back() = 0;
    return std::make_shared<nao::Radial>(l, r, f);
}
double coulomb(double a, double b, double r2) {
    const double t = a * b / (a + b) * r2;
    const double boys = t == 0 ? 1 : 0.5 * std::sqrt(pi / t) * std::erf(std::sqrt(t));
    return 2 * std::pow(pi, 2.5) / (a * b * std::sqrt(a + b)) * boys;
}
double norm_s(double a) {
    return std::pow(2 * a / pi, 0.75);
}

nao::Prepared dimer(int radial_points, std::size_t panel_points, std::size_t knots = 4001) {
    const std::vector<double> a{0.8, 1.1}, b{1.6, 2.2};
    const std::vector<Eigen::Vector3d> x{{0, 0, 0}, {0, 0, 1.4}};
    std::vector<nao::Center> centers;
    for (std::size_t i = 0; i < 2; ++i)
        centers.push_back({x[i],
                           {{gaussian(0, a[i], norm_s(a[i]) * std::sqrt(4 * pi), knots), 0}},
                           {{gaussian(0, b[i], std::sqrt(4 * pi), knots), 0}}});
    const auto start = std::chrono::steady_clock::now();
    const auto data = nao::prepare(centers, {{radial_points, 590}, panel_points, 1e-10});
    std::cout << "DIMER radial=" << radial_points << " panel=" << panel_points
              << " evaluation_panel_bytes=" << data.panel_bytes << '\n';
    double max_overlap = 0, max_kinetic = 0, max_v = 0, max_b = 0;
    Eigen::MatrixXd metric(2, 2), factors = Eigen::MatrixXd::Zero(2, 4);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
            const double r2 = (x[i] - x[j]).squaredNorm(), p = a[i] + a[j], mu = a[i] * a[j] / p;
            const double overlap =
                norm_s(a[i]) * norm_s(a[j]) * std::pow(pi / p, 1.5) * std::exp(-mu * r2);
            max_overlap = std::max(max_overlap, std::abs(data.overlap(i, j) - overlap));
            max_kinetic = std::max(max_kinetic,
                                   std::abs(data.kinetic(i, j) - mu * (3 - 2 * mu * r2) * overlap));
            metric(i, j) = coulomb(b[i], b[j], r2);
        }
    for (const auto& v : data.coulomb)
        max_v = std::max(max_v, std::abs(v.values(0, 0) - metric(v.left, v.right)));
    Eigen::MatrixXd numeric_metric(2, 2);
    for (const auto& v : data.coulomb)
        numeric_metric(v.left, v.right) = numeric_metric(v.right, v.left) = v.values(0, 0);
    for (const auto& fit : data.fits) {
        const auto i = fit.left, j = fit.right;
        const double p = a[i] + a[j], mu = a[i] * a[j] / p;
        const Eigen::Vector3d center = (a[i] * x[i] + a[j] * x[j]) / p;
        const double factor =
            norm_s(a[i]) * norm_s(a[j]) * std::exp(-mu * (x[i] - x[j]).squaredNorm());
        Eigen::VectorXd rhs(i == j ? 1 : 2), coeff(rhs.size());
        rhs[0] = factor * coulomb(b[i], p, (x[i] - center).squaredNorm());
        coeff[0] = fit.left_fit(0, 0);
        if (i != j) {
            rhs[1] = factor * coulomb(b[j], p, (x[j] - center).squaredNorm());
            coeff[1] = fit.right_fit(0, 0);
            max_b = std::max(max_b, (numeric_metric * coeff - rhs).cwiseAbs().maxCoeff());
            coeff = metric.ldlt().solve(rhs);
            factors(0, i * 2 + j) = factors(0, j * 2 + i) = coeff[0];
            factors(1, i * 2 + j) = factors(1, j * 2 + i) = coeff[1];
        } else {
            max_b = std::max(max_b, std::abs(numeric_metric(i, i) * coeff[0] - rhs[0]));
            factors(i, i * 2 + i) = rhs[0] / metric(i, i);
        }
    }
    near(max_overlap, 0, 2e-10, "overlap vs analytic");
    near(max_kinetic, 0, knots > 4001 ? 3e-8 : 5e-7, "kinetic vs analytic");
    near(max_v, 0, 2e-9, "auxiliary Coulomb vs analytic");
    near(max_b, 0, 2e-9, "three-center vs analytic");
    near(data.max_fit_residual, 0, 2e-12, "local fit residual");
    const Eigen::MatrixXd eri = factors.transpose() * metric * factors;
    Eigen::Matrix2d density;
    density << 0.7, -0.4, -0.4, -0.2;
    Eigen::Matrix2d expected = Eigen::Matrix2d::Zero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    expected(i, j) += eri(i * 2 + k, j * 2 + l) * density(k, l);
    lri::Exchange exchange(data.atoms, data.fits, data.coulomb);
    near(
        (exchange.apply(density) - expected).cwiseAbs().maxCoeff(), 0, 2e-9, "NAO-fit-LibRI chain");
    std::cout << "DIMER seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << '\n';
    return data;
}

void angular_potential() {
    const double a = 0.9;
    auto p = gaussian(1, a, std::sqrt(4 * pi / 3));
    auto d = gaussian(2, a, std::sqrt(16 * pi / 5));
    const Eigen::Vector3d x(0.3, 0.4, 0.7);
    const double r = x.norm(), q = std::pow(pi / a, 1.5);
    const double e = std::exp(-a * r * r), c = 2 * std::sqrt(a / pi),
                 erf = std::erf(std::sqrt(a) * r);
    const double first = q * (c * e / r - erf / (r * r));
    const double second_minus_first_over_r =
        q * (-2 * a * c * e - 3 * c * e / (r * r) + 3 * erf / (r * r * r));
    near(p->potential(x, 0), -first * x.z() / r / (2 * a), 2e-10, "p Coulomb potential");
    near(d->potential(x, 0),
         second_minus_first_over_r * (3 * x.z() * x.z() / (r * r) - 1) / (4 * a * a),
         2e-10,
         "d Coulomb potential");
}

void angular_and_rank() {
    constexpr double a = 0.9;
    auto s = gaussian(0, a, norm_s(a) * std::sqrt(4 * pi));
    auto p = gaussian(1, a, 2 * std::sqrt(a) * norm_s(a) * std::sqrt(4 * pi / 3));
    auto aux = gaussian(0, 2 * a, std::sqrt(4 * pi));
    const nao::Center atom{{0, 0, 0}, {{s, 0}, {p, 0}, {p, 1}, {p, -1}}, {{aux, 0}, {aux, 0}}};
    const auto data = nao::prepare({atom}, {{160, 590}, 1024, 1e-10});
    near((data.overlap - Eigen::Matrix4d::Identity()).norm(), 0, 3e-6, "s/p angular normalization");
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity() * (2.5 * a);
    t(0, 0) = 1.5 * a;
    near((data.kinetic - t).norm(), 0, 3e-5, "s/p kinetic");
    if (data.fit_ranks != std::vector<std::size_t>{1})
        throw std::runtime_error("duplicate auxiliary rank");
    near(p->value({0.3, 0, 0}, 1),
         -2 * std::sqrt(a) * norm_s(a) * 0.3 * std::exp(-a * 0.09),
         1e-8,
         "px convention");
    // Non-Gaussian compact radial: f=(1-r^2)^3 on [0,1]. Norm integral is B(3/2,7)/2.
    std::vector<double> r(2001), f(2001);
    for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] = double(i) / (r.size() - 1);
        f[i] = std::pow(1 - r[i] * r[i], 3);
    }
    auto compact = std::make_shared<nao::Radial>(0, r, f);
    auto compact_data =
        nao::prepare({{{0, 0, 0}, {{compact, 0}}, {{compact, 0}}}}, {{300, 590}, 512, 1e-10});
    const double expected_norm = 0.5 * std::tgamma(1.5) * std::tgamma(7.) / std::tgamma(8.5);
    near(compact_data.overlap(0, 0), expected_norm, 2e-8, "non-Gaussian compact normalization");
    near(compact->value({1.01, 0, 0}, 0), 0, 0, "compact support");
    if (!(compact->potential({2, 0, 0}, 0) > 0))
        throw std::runtime_error("Coulomb tail incorrectly truncated");
}

void higher_shells() {
    const double a = 0.9;
    auto aux = gaussian(0, 2 * a, std::sqrt(4 * pi));
    for (int l = 2; l <= 4; ++l) {
        const double radial_norm = std::sqrt(2 * std::pow(2 * a, l + 1.5) / std::tgamma(l + 1.5));
        auto radial = gaussian(l, a, radial_norm, 16001);
        nao::Center atom{{0, 0, 0}, {}, {{aux, 0}}};
        for (int m = -l; m <= l; ++m)
            atom.orbitals.push_back({radial, m});
        const auto data = nao::prepare({atom}, {{160, 590}, 1024, 1e-10});
        const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2 * l + 1, 2 * l + 1);
        std::cout << "SHELL l=" << l << '\n';
        near((data.overlap - identity).norm(), 0, 1e-10, "higher-shell overlap");
        near((data.kinetic - (l + 1.5) * a * identity).norm(), 0, 3e-7, "higher-shell kinetic");
    }
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int status = 0;
    try {
        std::cout << std::setprecision(12);
        const auto full = dimer(100, 1024);
        const auto narrow = dimer(100, 257);
        near((full.overlap - narrow.overlap).norm(), 0, 2e-13, "panel invariance S");
        near((full.kinetic - narrow.kinetic).norm(), 0, 2e-13, "panel invariance T");
        for (std::size_t i = 0; i < full.fits.size(); ++i) {
            near((full.fits[i].left_fit - narrow.fits[i].left_fit).norm(),
                 0,
                 2e-13,
                 "panel invariance fit");
            near((full.coulomb[i].values - narrow.coulomb[i].values).norm(),
                 0,
                 2e-13,
                 "panel invariance V");
        }
        dimer(160, 257);
        dimer(160, 1024, 16001);
        angular_potential();
        angular_and_rank();
        higher_shells();
        std::cout << "PASS numerical AO preparation\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
