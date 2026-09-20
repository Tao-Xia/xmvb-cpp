#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/optimization/preconditioners/block_inverse_bfgs.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::MatrixXd& actual,
    const Eigen::MatrixXd& expected,
    double tolerance,
    const char* message) {
  require(
      (actual - expected).stableNorm() <=
          tolerance * std::max(1.0, expected.stableNorm()),
      message);
}

Eigen::MatrixXd explicit_update(
    const Eigen::MatrixXd& base,
    const Eigen::MatrixXd& directions,
    const Eigen::MatrixXd& images) {
  const Eigen::MatrixXd curvature = directions.transpose() * images;
  const Eigen::MatrixXd inverse_curvature = curvature.inverse();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
      directions.rows(), directions.rows());
  const Eigen::MatrixXd left =
      identity - directions * inverse_curvature * images.transpose();
  return left * base * left.transpose() +
      directions * inverse_curvature * directions.transpose();
}

Eigen::MatrixXd applied_matrix(
    const xmvb::vb::BlockInverseBfgs& update,
    const Eigen::MatrixXd& base) {
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(base.rows(), base.cols());
  return update.apply_block(
      identity,
      [&base](const Eigen::MatrixXd& block) {
        return (base * block).eval();
      });
}

}  // namespace

int main() {
  try {
    Eigen::MatrixXd generator(5, 5);
    generator <<
        1.2,  0.1, -0.2,  0.3,  0.0,
        0.1,  1.5,  0.2, -0.1,  0.4,
       -0.2,  0.2,  1.1,  0.2, -0.3,
        0.3, -0.1,  0.2,  1.4,  0.1,
        0.0,  0.4, -0.3,  0.1,  1.3;
    const Eigen::MatrixXd hessian =
        generator.transpose() * generator +
        0.4 * Eigen::MatrixXd::Identity(5, 5);
    const Eigen::MatrixXd base =
        Eigen::VectorXd::LinSpaced(5, 0.2, 0.6).asDiagonal();
    Eigen::MatrixXd directions(5, 2);
    directions <<
        1.0,  0.2,
       -0.3,  0.8,
        0.4, -0.1,
        0.2,  0.5,
       -0.6,  0.3;
    const Eigen::MatrixXd images = hessian * directions;
    const xmvb::vb::BlockInverseBfgs update(directions, images);
    require(update.dimension() == 5 && update.rank() == 2,
            "block inverse-BFGS dimensions are wrong");

    const Eigen::MatrixXd actual = applied_matrix(update, base);
    const Eigen::MatrixXd expected =
        explicit_update(base, directions, images);
    require_close(actual, expected, 2.0e-13,
                  "matrix-free block inverse-BFGS differs from its formula");
    require_close(actual * images, directions, 3.0e-13,
                  "block inverse-BFGS violates its exact block secants");
    require_close(actual, actual.transpose(), 2.0e-13,
                  "block inverse-BFGS lost symmetry");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(actual);
    require(spectrum.info() == Eigen::Success &&
                spectrum.eigenvalues().minCoeff() > 0.0,
            "positive block inverse-BFGS update lost definiteness");

    const Eigen::VectorXd probe =
        Eigen::VectorXd::LinSpaced(5, -0.7, 0.9);
    const Eigen::VectorXd vector_action = update.apply(
        probe,
        [&base](const Eigen::VectorXd& vector) {
          return (base * vector).eval();
        });
    require_close(vector_action, actual * probe, 2.0e-13,
                  "vector and block inverse-BFGS actions disagree");

    Eigen::Matrix2d block_map;
    block_map << 1.1, -0.4,
                 0.3,  0.9;
    const xmvb::vb::BlockInverseBfgs remapped_update(
        directions * block_map,
        images * block_map);
    require_close(applied_matrix(remapped_update, base), actual, 4.0e-13,
                  "block inverse-BFGS depends on the sampled block basis");

    Eigen::MatrixXd coordinate_map = Eigen::MatrixXd::Identity(5, 5);
    coordinate_map(0, 1) = 0.3;
    coordinate_map(2, 4) = -0.2;
    coordinate_map(3, 0) = 0.4;
    coordinate_map(4, 2) = 0.1;
    const Eigen::MatrixXd inverse_coordinate_map = coordinate_map.inverse();
    const Eigen::MatrixXd transformed_base =
        inverse_coordinate_map * base * inverse_coordinate_map.transpose();
    const Eigen::MatrixXd transformed_directions =
        inverse_coordinate_map * directions;
    const Eigen::MatrixXd transformed_images =
        coordinate_map.transpose() * images;
    const xmvb::vb::BlockInverseBfgs transformed_update(
        transformed_directions,
        transformed_images);
    const Eigen::MatrixXd transformed_action =
        applied_matrix(transformed_update, transformed_base);
    require_close(
        coordinate_map * transformed_action * coordinate_map.transpose(),
        actual,
        8.0e-13,
        "block inverse-BFGS is not vector-covector coordinate covariant");

    const Eigen::MatrixXd full_directions =
        Eigen::MatrixXd::Identity(5, 5);
    const xmvb::vb::BlockInverseBfgs complete_update(
        full_directions,
        hessian);
    require_close(
        applied_matrix(complete_update, base),
        hessian.inverse(),
        2.0e-12,
        "complete exact-curvature block did not recover the dense inverse");

    bool rejected_negative = false;
    try {
      const Eigen::MatrixXd negative_images = -directions;
      const xmvb::vb::BlockInverseBfgs invalid(
          directions, negative_images);
      static_cast<void>(invalid);
    } catch (const std::invalid_argument&) {
      rejected_negative = true;
    }
    require(rejected_negative,
            "nonpositive curvature entered the inverse-BFGS update");

    std::cout << "Matrix-free block inverse-BFGS algebra: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
