#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vbscf/optimization/trust_region/spectral_trust_region.hpp"

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(const std::string& name, const Eigen::VectorXd& d,
           const Eigen::VectorXd& g, double radius, bool boundary,
           bool hard_case = false) {
  const auto result = xmvb::vb::solve_spectral_trust_region(d, g, radius);
  const double norm = result.step.stableNorm();
  const Eigen::VectorXd residual =
      ((d.array() + result.shift) * result.step.array()).matrix() + g;
  const double scale = std::max(g.stableNorm(),
      (d.cwiseAbs().maxCoeff() + result.shift) * radius);
  require(result.step.allFinite() && std::isfinite(result.shift), name + ": finite");
  require(norm <= radius * (1.0 + 1e-12), name + ": feasible");
  require(result.shift >= 0.0, name + ": nonnegative multiplier");
  require(d.minCoeff() + result.shift >= -1e-12 * d.cwiseAbs().maxCoeff(),
          name + ": shifted Hessian must be positive semidefinite");
  require(residual.stableNorm() <= 1e-12 * scale, name + ": stationarity");
  require(result.shift == 0.0 || std::abs(norm / radius - 1.0) < 1e-12,
          name + ": complementarity");
  require(result.boundary == boundary, name + ": boundary classification");
  require(result.hard_case == hard_case, name + ": hard-case classification");
  std::cout << name << ": passed\n";
}
}  // namespace

int main() {
  try {
    using V = Eigen::Vector2d;
    check("positive definite interior", V(2, 4), V(1, 2), 2, false);
    check("positive definite boundary", V(2, 4), V(1, 2), 0.1, true);
    check("zero-multiplier boundary", V(2, 4), V(2, 0), 1, true);
    check("semidefinite compatible", V(0, 2), V(0, 1), 1, false);
    check("semidefinite incompatible", V(0, 2), V(1, 1), 1, true);
    check("indefinite regular", V(-2, 3), V(1, 2), 1, true);
    check("indefinite hard case", V(-2, 3), V(0, 1), 1, true, true);
    check("pure negative curvature", V(-2, 3), V(0, 0), 1, true, true);
    check("zero model", V(0, 0), V(0, 0), 1, false);
    check("linear model", V(0, 0), V(1, 2), 1, true);
    check("repeated lowest eigenvalue", V(-2, -2), V(0, 0), 1, true, true);
    check("tiny positive mode", V(1e-60, 1), V(2e-60, 0.1), 1, true);
    check("large radius near spectral pole", V(-2, 3), V(1, 2), 1e20, true);
    check("small radius", V(-2, 3), V(1, 2), 1e-20, true);
    check("small radius with disparate scales", V(0, 1e200), V(1e-200, 0), 1e-200, true);
    for (double scale : {1e-100, 1e100}) {
      check("energy rescaling", scale * V(-2, 3), scale * V(1, 2), 1, true);
      const auto reference = xmvb::vb::solve_spectral_trust_region(V(-2, 3), V(1, 2), 1);
      const auto scaled = xmvb::vb::solve_spectral_trust_region(
          scale * V(-2, 3), scale * V(1, 2), 1);
      require((reference.step - scaled.step).norm() < 1e-12,
              "solution changed under energy rescaling");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
