#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vbscf/determinants/algebra/compound.hpp"
#include "vbscf/determinants/algebra/transition_density.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Eigen::MatrixXd submatrix(
    const Eigen::MatrixXd& matrix, const std::vector<int>& rows,
    const std::vector<int>& columns) {
  Eigen::MatrixXd result(rows.size(), columns.size());
  for (Eigen::Index row = 0; row < result.rows(); ++row) {
    for (Eigen::Index column = 0; column < result.cols(); ++column) {
      result(row, column) = matrix(rows[row], columns[column]);
    }
  }
  return result;
}

Eigen::MatrixXd direct_compound(
    const Eigen::MatrixXd& matrix, const xmvb::vb::CompoundBasis& basis,
    int order) {
  const Eigen::Index count = basis.level_size(order);
  Eigen::MatrixXd result(count, count);
  if (order == 0) {
    result(0, 0) = 1.0;
    return result;
  }
  for (Eigen::Index row = 0; row < count; ++row) {
    for (Eigen::Index column = 0; column < count; ++column) {
      result(row, column) =
          submatrix(
              matrix, basis.subset(order, row),
              basis.subset(order, column))
              .determinant();
    }
  }
  return result;
}

double relative_error(const Eigen::MatrixXd& actual,
                      const Eigen::MatrixXd& expected) {
  return (actual - expected).norm() / std::max(1.0, expected.norm());
}

}  // namespace

int main() {
  try {
    constexpr int size = 6;
    constexpr int max_order = size;
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(size, size);
    matrix.diagonal().array() += 2.0;
    const Eigen::MatrixXd direction =
        0.2 * Eigen::MatrixXd::Random(size, size);

    xmvb::vb::CompoundHierarchy hierarchy(size, max_order);
    hierarchy.assign(matrix);
    for (int order = 0; order <= max_order; ++order) {
      require(
          relative_error(
              hierarchy.level(order),
              direct_compound(matrix, hierarchy.basis(), order)) < 2.0e-14,
          "anchor compound level differs from direct minors");
    }

    xmvb::vb::CompoundHierarchy tangent = hierarchy.directional(direction);
    constexpr double finite_difference_step = 1.0e-6;
    xmvb::vb::CompoundHierarchy plus(size, max_order);
    xmvb::vb::CompoundHierarchy minus(size, max_order);
    plus.assign(matrix + finite_difference_step * direction);
    minus.assign(matrix - finite_difference_step * direction);
    for (int order = 1; order <= max_order; ++order) {
      const Eigen::MatrixXd finite_difference =
          (plus.level(order) - minus.level(order)) /
          (2.0 * finite_difference_step);
      require(
          relative_error(tangent.level(order), finite_difference) < 2.0e-9,
          "directional compound level fails finite differences");
    }

    const Eigen::VectorXd left = Eigen::VectorXd::Random(size);
    const Eigen::VectorXd right = Eigen::VectorXd::Random(size);
    const Eigen::VectorXd dleft = 0.2 * Eigen::VectorXd::Random(size);
    const Eigen::VectorXd dright = 0.2 * Eigen::VectorXd::Random(size);
    hierarchy.apply_rank_one(left, right, dleft, dright, tangent);

    const Eigen::MatrixXd updated_matrix = matrix + left * right.transpose();
    const Eigen::MatrixXd updated_direction = direction +
        dleft * right.transpose() + left * dright.transpose();
    xmvb::vb::CompoundHierarchy updated(size, max_order);
    updated.assign(updated_matrix);
    const xmvb::vb::CompoundHierarchy updated_tangent =
        updated.directional(updated_direction);
    for (int order = 0; order <= max_order; ++order) {
      require(
          relative_error(hierarchy.level(order), updated.level(order)) <
              3.0e-14,
          "rank-one compound update differs from anchor rebuild");
      require(
          relative_error(
              tangent.level(order), updated_tangent.level(order)) < 3.0e-14,
          "rank-one directional update differs from anchor rebuild");
    }

    for (int order = 1; order <= max_order; ++order) {
      const Eigen::Index count = hierarchy.basis().level_size(order);
      const Eigen::MatrixXd weights = Eigen::MatrixXd::Random(count, count);
      const double forward =
          (weights.array() * tangent.level(order).array()).sum();
      const double reverse =
          (hierarchy.pullback(order, weights).array() *
           updated_direction.array())
              .sum();
      require(
          std::abs(forward - reverse) <
              2.0e-12 * std::max({1.0, std::abs(forward), std::abs(reverse)}),
          "compound pullback fails the adjoint identity");
    }

    const Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(size, size) +
        0.15 * Eigen::MatrixXd::Random(size, size);
    const double overlap_determinant = overlap.determinant();
    const Eigen::MatrixXd inverse_overlap = overlap.inverse();
    xmvb::vb::TransitionDensityHierarchy densities(size, max_order);
    densities.assign(overlap_determinant, inverse_overlap);
    const Eigen::MatrixXd overlap_direction =
        0.1 * Eigen::MatrixXd::Random(size, size);
    for (int order = 0; order <= max_order; ++order) {
      const Eigen::MatrixXd expected = overlap_determinant *
          direct_compound(
              inverse_overlap.transpose(), hierarchy.basis(), order);
      require(
          relative_error(densities.level(order), expected) < 3.0e-14,
          "transition density differs from det(X) C_q(X^-T)");

      constexpr double density_step = 1.0e-5;
      const Eigen::MatrixXd overlap_plus =
          overlap + density_step * overlap_direction;
      const Eigen::MatrixXd overlap_minus =
          overlap - density_step * overlap_direction;
      xmvb::vb::TransitionDensityHierarchy density_plus(size, max_order);
      xmvb::vb::TransitionDensityHierarchy density_minus(size, max_order);
      density_plus.assign(overlap_plus.determinant(), overlap_plus.inverse());
      density_minus.assign(
          overlap_minus.determinant(), overlap_minus.inverse());
      const Eigen::MatrixXd density_finite_difference =
          (density_plus.level(order) - density_minus.level(order)) /
          (2.0 * density_step);
      require(
          relative_error(
              densities.directional_level(order, overlap_direction),
              density_finite_difference) < 2.0e-8,
          "transition-density direction fails finite differences at order " +
              std::to_string(order));

      const Eigen::Index count = hierarchy.basis().level_size(order);
      const Eigen::MatrixXd density_weights =
          Eigen::MatrixXd::Random(count, count);
      const double contraction_direction =
          (density_weights.array() *
           densities.directional_level(order, overlap_direction).array())
              .sum();
      const double contraction_pullback =
          (densities.contraction_gradient(order, density_weights).array() *
           overlap_direction.array())
              .sum();
      require(
          std::abs(contraction_direction - contraction_pullback) <
              3.0e-12 * std::max(
                  {1.0, std::abs(contraction_direction),
                   std::abs(contraction_pullback)}),
          "transition-density pullback fails finite differences");
    }

    const Eigen::VectorXd overlap_left =
        0.05 * Eigen::VectorXd::Random(size);
    const Eigen::VectorXd overlap_right =
        0.05 * Eigen::VectorXd::Random(size);
    require(
        densities.apply_overlap_rank_one(overlap_left, overlap_right),
        "regular transition-density update was unexpectedly singular");
    const Eigen::MatrixXd updated_overlap =
        overlap + overlap_left * overlap_right.transpose();
    const double updated_overlap_determinant = updated_overlap.determinant();
    const Eigen::MatrixXd updated_inverse = updated_overlap.inverse();
    for (int order = 0; order <= max_order; ++order) {
      const Eigen::MatrixXd expected = updated_overlap_determinant *
          direct_compound(
              updated_inverse.transpose(), hierarchy.basis(), order);
      require(
          relative_error(densities.level(order), expected) < 8.0e-14,
          "low-rank transition-density update differs from refactorization");
    }

    std::cout << "arbitrary-order compound hierarchy checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_compound: " << error.what() << '\n';
    return 1;
  }
}
