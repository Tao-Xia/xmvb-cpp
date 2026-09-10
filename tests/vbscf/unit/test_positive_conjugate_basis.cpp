#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/QR>

#include "vbscf/optimization/krylov/orthonormal_basis.hpp"
#include "vbscf/optimization/krylov/positive_conjugate_basis.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check(double condition_number) {
  constexpr int n = 24;
  const double pi = std::acos(-1.0);
  Eigen::MatrixXd rotation(n, n);
  Eigen::VectorXd spectrum(n), rhs(n);
  for (int j = 0; j < n; ++j) {
    spectrum[j] = std::pow(condition_number, double(j) / (n - 1));
    rhs[j] = 1.0 + std::sin(0.7 * j);
    for (int i = 0; i < n; ++i) {
      rotation(i, j) = std::cos(pi * (i + 0.5) * j / n) *
          std::sqrt((j == 0 ? 1.0 : 2.0) / n);
    }
  }
  const Eigen::MatrixXd h = rotation * spectrum.asDiagonal() * rotation.transpose();
  const Eigen::VectorXd inverse_diagonal = h.diagonal().cwiseInverse();
  xmvb::vb::PositiveConjugateBasis conjugate;
  std::vector<Eigen::VectorXd> q, hq;
  Eigen::MatrixXd p(n, n), hp(n, n);
  Eigen::VectorXd step = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd image = step;
  Eigen::VectorXd residual = rhs;
  int calls = 0;
  for (int j = 0; j < n; ++j) {
    const Eigen::VectorXd direction = conjugate.orthogonalize(
        inverse_diagonal.cwiseProduct(residual));
    Eigen::VectorXd hd;
    require(xmvb::vb::append_orthonormal_hvp_direction(
        direction, [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
          ++calls;
          return h * v;
        }, &q, &hq, &hd), "independent CG direction rejected before full dimension");
    const double curvature = direction.dot(hd);
    const double alpha = residual.dot(direction) / curvature;
    step += alpha * direction;
    image += alpha * hd;
    residual = rhs - image;
    conjugate.append(direction, hd);
    p.col(j) = direction / std::sqrt(curvature);
    hp.col(j) = hd / std::sqrt(curvature);
  }
  require(calls == n, "conjugacy restoration consumed extra HVPs");
  const Eigen::MatrixXd gram = p.transpose() * hp;
  require((gram - Eigen::MatrixXd::Identity(n, n)).norm() < 1e-7,
          "positive search directions lost H-conjugacy");
  require((h * step - rhs).norm() / rhs.norm() < 1e-7,
          "full-space CG solution failed fresh residual check");
  std::cout << "SPD condition " << condition_number << ": passed (" << calls << " HVPs)\n";
}
}  // namespace

int main() {
  try {
    check(100.0);
    check(1e6);
    xmvb::vb::PositiveConjugateBasis basis;
    bool rejected = false;
    try { basis.append(Eigen::VectorXd::Ones(2), -Eigen::VectorXd::Ones(2)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "negative curvature was admitted to the positive basis");
    std::cout << "Negative-curvature guard: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
