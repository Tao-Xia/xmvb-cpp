#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/QR>

#include "vbscf/orbitals/charts/normalized_orbital_curvature.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Eigen::VectorXd gradient(const Eigen::MatrixXd& f, const Eigen::MatrixXd& s,
                         const Eigen::VectorXd& x, const Eigen::MatrixXd& u) {
  const double norm = x.dot(s * x);
  const double e = x.dot(f * x) / norm;
  return u.transpose() * (2.0 / norm * (f * x - e * s * x));
}

void check(const Eigen::MatrixXd& f, const Eigen::MatrixXd& s,
           const Eigen::VectorXd& x, const Eigen::MatrixXd& u) {
  const Eigen::MatrixXd h = xmvb::vb::normalized_orbital_curvature(f, s, x, u);
  Eigen::MatrixXd finite_difference(h.rows(), h.cols());
  const double step = 1e-5;
  for (Eigen::Index j = 0; j < u.cols(); ++j) {
    finite_difference.col(j) = (gradient(f, s, x + step * u.col(j), u) -
        gradient(f, s, x - step * u.col(j), u)) / (2.0 * step);
  }
  require((h - finite_difference).norm() < 1e-8 * std::max(1.0, h.norm()),
          "normalized curvature differs from independent gradient differences");
  require((h - h.transpose()).norm() < 1e-13 * std::max(1.0, h.norm()),
          "normalized curvature is not symmetric");
  for (const double scale : {0.01, -3.0, 100.0}) {
    const Eigen::MatrixXd scaled = xmvb::vb::normalized_orbital_curvature(
        f, s, scale * x, u);
    require((scale * scale * scaled - h).norm() < 1e-11 * std::max(1.0, h.norm()),
            "orbital-rescaling covariance failed");
  }
  Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(u.cols(), u.cols());
  if (u.cols() >= 2) {
    rotation(0, 0) = rotation(1, 1) = std::cos(0.37);
    rotation(0, 1) = -std::sin(0.37);
    rotation(1, 0) = std::sin(0.37);
  }
  const Eigen::MatrixXd rotated = xmvb::vb::normalized_orbital_curvature(
      f, s, x, u * rotation);
  require((rotated - rotation.transpose() * h * rotation).norm() <
              1e-12 * std::max(1.0, h.norm()), "quotient-basis covariance failed");
}
}  // namespace

int main() {
  try {
    Eigen::MatrixXd f(4, 4), s(4, 4);
    f << -2.0, 0.3, -0.1, 0.2,
          0.3, 0.5, 0.4, -0.2,
         -0.1, 0.4, 1.5, 0.1,
          0.2, -0.2, 0.1, 3.0;
    s << 1.0, 0.2, 0.1, 0.0,
         0.2, 1.3, 0.0, 0.1,
         0.1, 0.0, 0.8, 0.2,
         0.0, 0.1, 0.2, 1.2;
    Eigen::VectorXd x(4);
    x << 0.9, -0.4, 0.2, 0.7;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(x);
    const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(4, 4);
    const Eigen::MatrixXd u = q.rightCols(3);
    require((u.transpose() * x).norm() < 1e-14, "invalid Euclidean quotient fixture");
    require((u.transpose() * s * x).norm() > 1e-2,
            "fixture does not distinguish Euclidean and overlap orthogonality");
    check(f, s, x, u);
    const double norm = x.dot(s * x);
    const double energy = x.dot(f * x) / norm;
    const Eigen::MatrixXd incomplete = 2.0 / norm * u.transpose() * (f - energy * s) * u;
    const Eigen::MatrixXd exact = xmvb::vb::normalized_orbital_curvature(f, s, x, u);
    require((incomplete - exact).norm() > 1e-2,
            "fixture failed to expose omitted normalization derivatives");
    std::cout << "Euclidean quotient and normalization derivatives: passed\n";

    Eigen::MatrixXd frozen = Eigen::MatrixXd::Identity(4, 3);
    check(f, s, x, frozen);
    const Eigen::MatrixXd correct = xmvb::vb::normalized_orbital_curvature(f, s, x, frozen);
    const Eigen::MatrixXd missing_tail = xmvb::vb::normalized_orbital_curvature(
        f.topLeftCorner(3, 3), s.topLeftCorner(3, 3), x.head(3),
        Eigen::MatrixXd::Identity(3, 3));
    require((correct - missing_tail).norm() > 1e-2,
            "fixture failed to expose omitted frozen coefficients");
    std::cout << "Stored frozen tail, finite differences, scaling and rotation: passed\n";

    check(f, s, x, Eigen::MatrixXd::Zero(4, 0));
    bool rejected = false;
    try {
      xmvb::vb::normalized_orbital_curvature(f, s, Eigen::VectorXd::Zero(4), u);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "zero metric norm was accepted");
    std::cout << "Empty tangent and invalid norm: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
