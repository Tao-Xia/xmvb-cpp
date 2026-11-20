#include "integrals/nao/radial.hpp"
#include <cubic_spline.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

namespace xmvb::integrals::nao {
namespace {
using Spline = ModuleBase::CubicSpline;
constexpr double pi = std::numbers::pi;
// Eight-point Gauss-Legendre integrates each cubic-spline moment exactly
// through l=4 (maximum polynomial degree 13).
constexpr std::array<double, 4> nodes{
    0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
constexpr std::array<double, 4> weights{
    0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
double moment(const Spline& spline, double a, double b, int power) {
    double sum = 0;
    for (std::size_t q = 0; q < nodes.size(); ++q)
        for (double sign : {-1.0, 1.0}) {
            const double r = 0.5 * ((b + a) + sign * (b - a) * nodes[q]);
            double f;
            spline.eval(1, &r, &f);
            sum += weights[q] * f * std::pow(r, power);
        }
    return 0.5 * (b - a) * sum;
}
double harmonic(int l, int m, const Eigen::Vector3d& x, double r) {
    if (m < -l || m > l)
        throw std::invalid_argument("invalid real harmonic index");
    if (r == 0)
        return l == 0 ? 1 / std::sqrt(4 * pi) : 0;
    const int am = std::abs(m);
    // Restore the Condon-Shortley phase omitted by C++ associated Legendre.
    // This matches ABACUS: p(m=0,+1,-1) = sqrt(3/4pi) * (z,-x,-y)/r.
    double value = std::sqrt((2 * l + 1) / (4 * pi) *
                             std::exp(std::lgamma(l - am + 1) - std::lgamma(l + am + 1))) *
                   std::assoc_legendre(l, am, std::clamp(x.z() / r, -1.0, 1.0));
    if (am) {
        const double phi = std::atan2(x.y(), x.x());
        value *= (am % 2 ? -1.0 : 1.0) * std::sqrt(2.0) *
                 (m > 0 ? std::cos(am * phi) : std::sin(am * phi));
    }
    return value;
}
} // namespace

struct Radial::Impl {
    int angular;
    double rc;
    std::optional<Spline> function, coulomb;
    double tail_moment;
    Impl(int l, std::span<const double> grid, std::span<const double> f) : angular(l) {
        if (l < 0 || l > 4 || grid.size() < 4 || grid.size() != f.size() ||
            grid.size() > static_cast<std::size_t>(INT_MAX / 2) || grid.front() != 0 ||
            f.back() != 0)
            throw std::invalid_argument("invalid NAO radial grid, l or compact support");
        for (std::size_t i = 0; i < grid.size(); ++i)
            if (!std::isfinite(grid[i]) || !std::isfinite(f[i]) || (i && grid[i] <= grid[i - 1]))
                throw std::invalid_argument("NAO radial grid must be finite and increasing");
        rc = grid.back();
        const int count = static_cast<int>(grid.size());
        const Spline::BoundaryCondition zero_derivative(Spline::BoundaryType::first_deriv, 0);
        function.emplace(count, grid.data(), f.data(), zero_derivative, zero_derivative);
        std::vector<double> inner(grid.size(), 0), outer(grid.size(), 0), potential(grid.size());
        for (std::size_t i = 1; i < grid.size(); ++i)
            inner[i] = inner[i - 1] + moment(*function, grid[i - 1], grid[i], 2 * l + 2);
        for (std::size_t i = grid.size() - 1; i > 0; --i)
            outer[i - 1] = outer[i] + moment(*function, grid[i - 1], grid[i], 1);
        const double factor = 4 * pi / (2 * l + 1);
        potential[0] = factor * outer[0];
        for (std::size_t i = 1; i < grid.size(); ++i)
            potential[i] = factor * (inner[i] / std::pow(grid[i], 2 * l + 1) + outer[i]);
        tail_moment = factor * inner.back();
        coulomb.emplace(count,
                        grid.data(),
                        potential.data(),
                        zero_derivative,
                        Spline::BoundaryCondition(Spline::BoundaryType::first_deriv,
                                                  -(2 * l + 1) * potential.back() / rc));
    }
};

Radial::Radial(int l, std::span<const double> grid, std::span<const double> values)
    : impl_(std::make_unique<Impl>(l, grid, values)) {}
Radial::~Radial() = default;
int Radial::l() const {
    return impl_->angular;
}
double Radial::cutoff() const {
    return impl_->rc;
}
double Radial::value(const Eigen::Vector3d& x, int m) const {
    const double r = x.norm();
    const double y = harmonic(l(), m, x, r);
    if (r >= cutoff())
        return 0;
    double f;
    impl_->function->eval(1, &r, &f);
    return std::pow(r, l()) * f * y;
}
double Radial::kinetic(const Eigen::Vector3d& x, int m) const {
    const double r = x.norm();
    const double y = harmonic(l(), m, x, r);
    if (r >= cutoff())
        return 0;
    double df, ddf;
    impl_->function->eval(1, &r, nullptr, &df, &ddf);
    const double laplacian = ddf + 2 * (l() + 1) * (r == 0 ? ddf : df / r);
    return -0.5 * std::pow(r, l()) * laplacian * y;
}
double Radial::potential(const Eigen::Vector3d& x, int m) const {
    const double r = x.norm();
    const double y = harmonic(l(), m, x, r);
    if (r >= cutoff())
        return impl_->tail_moment * y / std::pow(r, l() + 1);
    double v;
    impl_->coulomb->eval(1, &r, &v);
    return std::pow(r, l()) * v * y;
}
} // namespace xmvb::integrals::nao
