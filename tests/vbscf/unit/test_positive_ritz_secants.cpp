#include <cmath>
#include <iostream>
#include <stdexcept>

#include "vbscf/optimization/krylov/positive_ritz_secants.hpp"
#include "vbscf/optimization/preconditioners/block_inverse_bfgs.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
Eigen::MatrixXd projector(const std::vector<xmvb::vb::RitzSecant>& pairs, int n) {
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n, n);
  for (const auto& pair : pairs) p += pair.direction * pair.direction.transpose();
  return p;
}
}  // namespace

int main() {
  try {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(5, 5);
    h.diagonal() << -3.0, 0.0, 0.2, 1.0, 9.0;
    h(4, 2) = h(2, 4) = 0.03;
    h(4, 3) = h(3, 4) = 0.04;
    const Eigen::MatrixXd q = Eigen::MatrixXd::Identity(5, 4);
    const auto pairs = xmvb::vb::positive_ritz_secants(q, h * q, q.transpose() * h * q, 2);
    require(pairs.size() == 2, "incorrect positive Ritz rank");
    for (const auto& pair : pairs) {
      require((pair.image - h * pair.direction).norm() < 1e-13,
              "Ritz image lost full HVP information");
      require(std::abs(pair.image[4]) > 0.02,
              "off-subspace HVP component was discarded");
      require(pair.direction.dot(pair.image) > 0.0,
              "nonpositive mode admitted as a positive secant");
    }
    require(std::abs(pairs.back().direction.dot(pairs.back().image) - 0.2) < 1e-13,
            "softest mode must be newest in FIFO history");
    Eigen::MatrixXd positive_directions(5, pairs.size());
    Eigen::MatrixXd positive_images(5, pairs.size());
    for (std::size_t column = 0; column < pairs.size(); ++column) {
      positive_directions.col(static_cast<Eigen::Index>(column)) =
          pairs[column].direction;
      positive_images.col(static_cast<Eigen::Index>(column)) =
          pairs[column].image;
    }
    const xmvb::vb::BlockInverseBfgs positive_update(
        positive_directions,
        positive_images);
    const Eigen::VectorXd negative_mode = Eigen::VectorXd::Unit(5, 0);
    const Eigen::VectorXd updated_negative = positive_update.apply(
        negative_mode,
        [](const Eigen::VectorXd& vector) { return vector; });
    require((updated_negative - negative_mode).norm() < 1.0e-13,
            "negative curvature leaked into the positive inverse update");
    require((positive_update.apply_block(
                 positive_images,
                 [](const Eigen::MatrixXd& block) { return block; }) -
             positive_directions).norm() < 1.0e-13,
            "selected positive Ritz modes lost their exact secants");
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(4, 4);
    rotation(0, 0) = rotation(2, 2) = std::cos(0.43);
    rotation(0, 2) = -std::sin(0.43);
    rotation(2, 0) = std::sin(0.43);
    const Eigen::MatrixXd rotated_q = q * rotation;
    const auto rotated = xmvb::vb::positive_ritz_secants(rotated_q, h * rotated_q,
        rotated_q.transpose() * h * rotated_q, 2);
    require((projector(pairs, 5) - projector(rotated, 5)).norm() < 1e-12,
            "selected subspace depends on its orthogonal basis");
    std::cout << "Positive soft modes, full images and basis covariance: passed\n";

    Eigen::MatrixXd spd = Eigen::MatrixXd::Zero(4, 4);
    spd.diagonal() << 0.01, 0.2, 3.0, 20.0;
    spd = (rotation * spd * rotation.transpose()).eval();
    const auto full = xmvb::vb::positive_ritz_secants(
        Eigen::MatrixXd::Identity(4, 4), spd, spd, 4);
    Eigen::MatrixXd inverse = Eigen::MatrixXd::Identity(4, 4);
    for (const auto& pair : full) {
      const double rho = 1.0 / pair.direction.dot(pair.image);
      const Eigen::MatrixXd v = Eigen::MatrixXd::Identity(4, 4) -
          rho * pair.direction * pair.image.transpose();
      inverse = (v * inverse * v.transpose() +
          rho * pair.direction * pair.direction.transpose()).eval();
    }
    require((inverse * spd - Eigen::MatrixXd::Identity(4, 4)).norm() < 1e-11,
            "complete conjugate Ritz secants do not recover the quadratic inverse");
    std::cout << "Full SPD model inverse-BFGS recovery: passed\n";

    require(xmvb::vb::positive_ritz_secants(q, h * q, q.transpose() * h * q, 0).empty(),
            "zero history budget was ignored");
    require(xmvb::vb::positive_ritz_secants(Eigen::MatrixXd::Zero(5, 0),
        Eigen::MatrixXd::Zero(5, 0), Eigen::MatrixXd::Zero(0, 0), 8).empty(),
        "empty subspace was not handled");
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(4, 4);
    require(xmvb::vb::positive_ritz_secants(identity, -identity, -identity, 4).empty(),
            "negative model was altered into positive secants");
    std::cout << "Empty, disabled and negative-only models: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
