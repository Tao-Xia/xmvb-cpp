#include <iostream>
#include <stdexcept>
#include "vbscf/diagnostics/orbitals/curvature.hpp"

int main() {
  try {
    Eigen::Matrix4d diagonal = Eigen::Matrix4d::Zero();
    diagonal.diagonal() << 2.0, 3.0, 4.0, 5.0;
    diagonal(0, 1) = diagonal(1, 0) = 0.4;
    Eigen::Matrix4d coupling = Eigen::Matrix4d::Zero();
    coupling(1, 2) = coupling(2, 1) = -0.7;
    // Computational components need not be separately symmetric.
    coupling(2, 1) = 0.25;
    const Eigen::Vector4d w(1.0, 2.0, -1.0, 0.5);
    const Eigen::Matrix4d outer = -0.2 * w * w.transpose();
    const Eigen::Matrix4d model = 2.0 * Eigen::Matrix4d::Identity();
    const Eigen::VectorXd v = Eigen::Vector4d(0.3, -0.5, 0.1, 0.8);
    int core_calls = 0;
    const auto d = xmvb::diagnostics::decompose_curvature(v, 2,
        [&](const Eigen::VectorXd& x) -> Eigen::VectorXd { return (diagonal + coupling + outer) * x; },
        [&](const Eigen::VectorXd& x) -> Eigen::VectorXd { ++core_calls; return (diagonal + coupling) * x; },
        [&](const Eigen::VectorXd& x) -> Eigen::VectorXd { return model * x; },
        [](const Eigen::VectorXd& x, int p) -> Eigen::VectorXd {
          Eigen::VectorXd y = Eigen::VectorXd::Zero(4);
          y.segment(2*p, 2) = x.segment(2*p, 2);
          return y;
        });
    if ((d.local_error - (diagonal-model)*v).norm() > 1e-13 ||
        (d.coupling - coupling*v).norm() > 1e-13 ||
        (d.outer - outer*v).norm() > 1e-13 || core_calls != 3 ||
        (d.local_error+d.coupling+d.outer - (d.full-d.model)).norm() > 1e-13)
      throw std::runtime_error("independent block decomposition failed");
    std::cout << "Local error, cross-orbital coupling and outer response: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
