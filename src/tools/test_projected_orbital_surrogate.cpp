#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>

#include "vb/orbital/normalized_orbital_curvature.hpp"
#include "vb/orbital/projected_orbital_surrogate.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double occupied_energy(const Eigen::MatrixXd& f, const Eigen::MatrixXd& s,
                       const Eigen::MatrixXd& occupied) {
  if (occupied.cols() == 0) return 0.0;
  const Eigen::MatrixXd gram = occupied.transpose() * s * occupied;
  return gram.llt().solve(occupied.transpose() * f * occupied).trace();
}

void check(const Eigen::MatrixXd& f, const Eigen::MatrixXd& s,
           const Eigen::MatrixXd& fixed, const std::vector<int>& support,
           const Eigen::VectorXd& c, const Eigen::MatrixXd& u) {
  const auto model = xmvb::vb::projected_orbital_surrogate(f, s, fixed, support);
  const Eigen::MatrixXd h = xmvb::vb::normalized_orbital_curvature(
      model.one_electron, model.overlap, c, u);
  const double reference = occupied_energy(f, s, fixed);
  auto physical_energy = [&](const Eigen::VectorXd& x) {
    Eigen::MatrixXd occupied(f.rows(), fixed.cols() + 1);
    occupied.leftCols(fixed.cols()) = fixed;
    occupied.col(fixed.cols()).setZero();
    for (std::size_t j = 0; j < support.size(); ++j)
      occupied(support[j], fixed.cols()) = x[j];
    return occupied_energy(f, s, occupied) - reference;
  };
  const double model_energy = c.dot(model.one_electron * c) / c.dot(model.overlap * c);
  require(std::abs(model_energy - physical_energy(c)) < 1e-12,
          "projected Rayleigh quotient differs from independent projector trace");
  Eigen::VectorXd direction(u.cols());
  for (Eigen::Index j = 0; j < direction.size(); ++j) direction[j] = std::sin(j + 0.7);
  direction.normalize();
  const double exact = direction.dot(h * direction);
  const double step = 1e-4;
  const double numerical = (physical_energy(c + step * u * direction) -
      2.0 * physical_energy(c) + physical_energy(c - step * u * direction)) / (step * step);
  require(std::abs(exact - numerical) < 1e-6 * std::max(1.0, std::abs(exact)),
          "curvature differs from independent physical-energy second difference");
}

void check_closed_shell_differential(const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& s, const Eigen::MatrixXd& fixed,
    const Eigen::VectorXd& c, const Eigen::MatrixXd& interaction) {
  auto density = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
    Eigen::MatrixXd occupied(s.rows(), fixed.cols() + 1);
    occupied.leftCols(fixed.cols()) = fixed;
    occupied.col(fixed.cols()) = x;
    const Eigen::MatrixXd gram = occupied.transpose() * s * occupied;
    return occupied * gram.llt().solve(occupied.transpose());
  };
  // A self-adjoint linear mean-field map, independent of the curvature code.
  auto field = [&](const Eigen::MatrixXd& p) -> Eigen::MatrixXd {
    return interaction * p * interaction;
  };
  auto energy = [&](const Eigen::MatrixXd& p) {
    return ((2.0 * core + field(p)) * p).trace();
  };
  Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(c.size(), -0.4, 0.7);
  direction.normalize();
  const double step = 1e-4;
  const Eigen::MatrixXd p = density(c), plus = density(c + step * direction),
      minus = density(c - step * direction);
  const Eigen::MatrixXd dp = (plus - minus) / (2.0 * step);
  const Eigen::MatrixXd f = core + field(p);
  const double first = 2.0 * (f * dp).trace();
  require(std::abs((energy(plus) - energy(minus)) / (2.0 * step) - first) < 1e-7,
          "closed-shell density differential does not equal twice F11");
  std::vector<int> support(c.size());
  for (int j = 0; j < c.size(); ++j) support[j] = j;
  const auto model = xmvb::vb::projected_orbital_surrogate(f, s, fixed, support);
  const Eigen::MatrixXd curvature = xmvb::vb::normalized_orbital_curvature(
      model.one_electron, model.overlap, c, Eigen::MatrixXd::Identity(c.size(), c.size()));
  const double frozen = 2.0 * direction.dot(curvature * direction);
  const double response = 2.0 * (field(dp) * dp).trace();
  const double second = (energy(plus) - 2.0 * energy(p) + energy(minus)) / (step * step);
  require(std::abs(second - frozen - response) < 2e-6 * std::max(1.0, std::abs(second)),
          "double-occupancy geometry plus field response fails energy second difference");
  require(std::abs(frozen) > 1e-2 && std::abs(response) > 1e-3,
          "closed-shell fixture cannot detect missing occupation or field response");
  std::cout << "Closed-shell differential and frozen/response curvature decomposition: passed\n";
}
}  // namespace

int main() {
  try {
    constexpr int n = 6;
    Eigen::MatrixXd f(n, n), a(n, n);
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
      a(i, j) = 0.1 * std::sin(1.3 * (i + 1) * (j + 1));
      f(i, j) = 0.2 * std::cos(0.7 * (i + j)) + (i == j ? j - 2.0 : 0.0);
    }
    const Eigen::MatrixXd s = Eigen::MatrixXd::Identity(n, n) + a.transpose() * a;
    Eigen::MatrixXd fixed(n, 2);
    fixed << 1.0, 0.2, 0.1, 1.0, 0.3, -0.2, 0.0, 0.4, 0.2, 0.0, 0.1, 0.3;
    Eigen::VectorXd c(n);
    c << 0.3, -0.1, 0.7, 0.2, 0.9, -0.3;
    const std::vector<int> full_support{0, 1, 2, 3, 4, 5};
    const auto model = xmvb::vb::projected_orbital_surrogate(f, s, fixed, full_support);
    check(f, s, fixed, full_support, c, Eigen::MatrixXd::Identity(n, n));
    require((model.one_electron * fixed).norm() < 1e-12 &&
            (model.overlap * fixed).norm() < 1e-12,
            "fixed inactive gauge was not annihilated");
    Eigen::Matrix2d change;
    change << 2.0, 0.4, -0.3, 0.7;
    const auto changed = xmvb::vb::projected_orbital_surrogate(f, s, fixed * change, full_support);
    require((model.one_electron - changed.one_electron).norm() < 1e-12 &&
            (model.overlap - changed.overlap).norm() < 1e-12,
            "surrogate depends on the basis of the excluded inactive span");
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
    check_closed_shell_differential(f, s, fixed, c, 0.5 * identity + a);
    const Eigen::VectorXd shifted_c = c + fixed * Eigen::Vector2d(0.4, -0.7);
    const Eigen::MatrixXd h = xmvb::vb::normalized_orbital_curvature(
        model.one_electron, model.overlap, c, identity);
    const Eigen::MatrixXd raw_h = xmvb::vb::normalized_orbital_curvature(f, s, c, identity);
    require((raw_h * fixed).norm() > 1e-2,
            "fixture does not distinguish raw from inactive-projected curvature");
    const Eigen::MatrixXd shifted_h = xmvb::vb::normalized_orbital_curvature(
        model.one_electron, model.overlap, shifted_c, identity);
    require((h - shifted_h).norm() < 1e-11 && (h * fixed).norm() < 1e-11,
            "inactive gauge shift changed the frozen-target model curvature");
    std::cout << "Projector trace, curvature, inactive-basis and gauge invariance: passed\n";

    Eigen::VectorXd sparse_c(4);
    sparse_c << 0.3, 0.7, 0.9, -0.3;
    check(f, s, fixed, {0, 2, 4, 5}, sparse_c, Eigen::MatrixXd::Identity(4, 3));
    check(f, s, fixed.leftCols(1), {0, 2, 4, 5}, sparse_c, Eigen::MatrixXd::Identity(4, 3));
    check(f, s, Eigen::MatrixXd::Zero(n, 0), full_support, c, identity);
    std::cout << "Strict support, frozen tail, excluded target and empty inactive span: passed\n";

    Eigen::MatrixXd singular = fixed;
    singular.col(1) = singular.col(0);
    bool rejected = false;
    try { xmvb::vb::projected_orbital_surrogate(f, s, singular, full_support); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "singular fixed inactive span was accepted");
    std::cout << "Singular inactive-span guard: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
