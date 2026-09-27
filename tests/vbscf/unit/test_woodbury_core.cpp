#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/QR>

#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/pairs/woodbury_core.hpp"

namespace {

void require_close(
    double value,
    double reference,
    double tolerance,
    const char* label) {
  const double scale = std::max({1.0, std::abs(value), std::abs(reference)});
  if (std::abs(value - reference) > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

void require_matrix_close(
    const Eigen::MatrixXd& value,
    const Eigen::MatrixXd& reference,
    double tolerance,
    const char* label) {
  const double scale = std::max({1.0, value.norm(), reference.norm()});
  if (value.rows() != reference.rows() || value.cols() != reference.cols() ||
      (value - reference).norm() > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

Eigen::MatrixXd exterior_square(const Eigen::MatrixXd& matrix) {
  const int n = matrix.rows();
  const int pairs = n * (n - 1) / 2;
  Eigen::MatrixXd result(pairs, pairs);
  for (int j = 1; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      const int row = j * (j - 1) / 2 + i;
      for (int l = 1; l < n; ++l) {
        for (int k = 0; k < l; ++k) {
          const int column = l * (l - 1) / 2 + k;
          result(row, column) =
              matrix(i, k) * matrix(j, l) -
              matrix(i, l) * matrix(j, k);
        }
      }
    }
  }
  return result;
}

void check_case(const Eigen::MatrixXd& overlap, const char* label) {
  const int n = overlap.rows();
  Eigen::MatrixXd transition(n, n);
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < n; ++row) {
      transition(row, column) =
          0.07 * (row + 1) - 0.03 * (column + 2) +
          0.011 * ((row + 2) * (column + 3) % 5);
    }
  }

  xmvb::vb::DeterminantOverlapResolver resolver;
  const auto resolved = resolver.resolve_matrix(overlap);
  const xmvb::vb::CofactorDifferential reference(resolved);
  const xmvb::vb::WoodburyCore core(overlap);

  require_close(
      core.determinant(),
      resolved.overlap_determinant,
      2.0e-11,
      label);
  require_matrix_close(
      core.first_cofactor(), reference.value(), 2.0e-10, label);
  require_close(
      core.first_contraction(transition),
      (reference.value().cwiseProduct(transition)).sum(),
      2.0e-10,
      label);
  require_close(
      core.second_factor_contraction(transition),
      reference.second_contraction(exterior_square(transition)),
      5.0e-9,
      label);

  const xmvb::vb::WoodburyContraction first_gradient =
      core.first_contraction_gradient(transition);
  const xmvb::vb::WoodburyContraction second_gradient =
      core.second_factor_contraction_gradient(transition);
  Eigen::MatrixXd overlap_direction(n, n);
  Eigen::MatrixXd transition_direction(n, n);
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < n; ++row) {
      overlap_direction(row, column) =
          0.017 * (row + 1) - 0.009 * (column + 2);
      transition_direction(row, column) =
          0.012 * (column + 1) + 0.004 * (row - column);
    }
  }
  constexpr double epsilon = 2.0e-6;
  const xmvb::vb::WoodburyCore plus_core(
      overlap + epsilon * overlap_direction);
  const xmvb::vb::WoodburyCore minus_core(
      overlap - epsilon * overlap_direction);
  const double first_difference =
      (plus_core.first_contraction(
           transition + epsilon * transition_direction) -
       minus_core.first_contraction(
           transition - epsilon * transition_direction)) /
      (2.0 * epsilon);
  const double second_difference =
      (plus_core.second_factor_contraction(
           transition + epsilon * transition_direction) -
       minus_core.second_factor_contraction(
           transition - epsilon * transition_direction)) /
      (2.0 * epsilon);
  require_close(
      (first_gradient.overlap_gradient.cwiseProduct(overlap_direction)).sum() +
          (first_gradient.transition_gradient
               .cwiseProduct(transition_direction)).sum(),
      first_difference,
      2.0e-7,
      "first contraction gradient");
  require_close(
      (second_gradient.overlap_gradient.cwiseProduct(overlap_direction)).sum() +
          (second_gradient.transition_gradient
               .cwiseProduct(transition_direction)).sum(),
      second_difference,
      2.0e-6,
      "second contraction gradient");

  const auto first_gradient_direction =
      core.first_contraction_gradient_direction(
          transition, overlap_direction, transition_direction);
  const auto second_gradient_direction =
      core.second_factor_contraction_gradient_direction(
          transition, overlap_direction, transition_direction);
  const auto plus_first_gradient = plus_core.first_contraction_gradient(
      transition + epsilon * transition_direction);
  const auto minus_first_gradient = minus_core.first_contraction_gradient(
      transition - epsilon * transition_direction);
  const auto plus_second_gradient = plus_core.second_factor_contraction_gradient(
      transition + epsilon * transition_direction);
  const auto minus_second_gradient = minus_core.second_factor_contraction_gradient(
      transition - epsilon * transition_direction);
  require_close(
      first_gradient_direction.value,
      first_difference,
      2.0e-7,
      "first contraction value direction");
  require_matrix_close(
      first_gradient_direction.overlap_gradient,
      (plus_first_gradient.overlap_gradient -
       minus_first_gradient.overlap_gradient) /
          (2.0 * epsilon),
      3.0e-6,
      "first overlap-gradient direction");
  require_matrix_close(
      first_gradient_direction.transition_gradient,
      (plus_first_gradient.transition_gradient -
       minus_first_gradient.transition_gradient) /
          (2.0 * epsilon),
      3.0e-6,
      "first transition-gradient direction");
  require_close(
      second_gradient_direction.value,
      second_difference,
      2.0e-6,
      "second contraction value direction");
  require_matrix_close(
      second_gradient_direction.overlap_gradient,
      (plus_second_gradient.overlap_gradient -
       minus_second_gradient.overlap_gradient) /
          (2.0 * epsilon),
      3.0e-5,
      "second overlap-gradient direction");
  require_matrix_close(
      second_gradient_direction.transition_gradient,
      (plus_second_gradient.transition_gradient -
       minus_second_gradient.transition_gradient) /
          (2.0 * epsilon),
      3.0e-5,
      "second transition-gradient direction");

  require_matrix_close(
      core.first_cofactor_direction(overlap_direction),
      (plus_core.first_cofactor() - minus_core.first_cofactor()) /
          (2.0 * epsilon),
      3.0e-6,
      "first cofactor direction");

  const Eigen::MatrixXd updated = overlap +
      Eigen::VectorXd::LinSpaced(n, -0.02, 0.03) *
      Eigen::RowVectorXd::LinSpaced(n, 0.04, -0.01);
  xmvb::vb::WoodburyCore propagated(overlap);
  const Eigen::VectorXd update_left =
      Eigen::VectorXd::LinSpaced(n, -0.02, 0.03);
  const Eigen::VectorXd update_right =
      Eigen::VectorXd::LinSpaced(n, 0.04, -0.01);
  const xmvb::vb::WoodburyBaseUpdate base_update = propagated.append(
      update_left, update_right);
  const auto updated_resolved = resolver.resolve_matrix(updated);
  const xmvb::vb::CofactorDifferential updated_reference(updated_resolved);
  require_matrix_close(
      propagated.overlap(), updated, 2.0e-13, "propagated overlap");
  require_close(
      propagated.determinant(),
      updated_resolved.overlap_determinant,
      2.0e-10,
      "propagated determinant");
  require_matrix_close(
      propagated.first_cofactor(),
      updated_reference.value(),
      5.0e-9,
      "propagated first cofactor");
  require_close(
      propagated.second_factor_contraction(transition),
      updated_reference.second_contraction(exterior_square(transition)),
      2.0e-8,
      "propagated second contraction");

  Eigen::MatrixXd channel = core.inverse_base() * transition;
  channel.noalias() +=
      base_update.left * (base_update.right.transpose() * channel);
  require_close(
      propagated.second_channel_contraction(channel),
      propagated.second_factor_contraction(transition),
      2.0e-10,
      "propagated channel contraction");
}

}  // namespace

int main() {
  try {
    constexpr int n = 6;
    Eigen::MatrixXd left_seed(n, n);
    Eigen::MatrixXd right_seed(n, n);
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < n; ++row) {
        left_seed(row, column) =
            std::sin(0.31 * (row + 1) * (column + 2));
        right_seed(row, column) =
            std::cos(0.27 * (row + 2) * (column + 1));
      }
    }
    const Eigen::MatrixXd left =
        left_seed.householderQr().householderQ();
    const Eigen::MatrixXd right =
        right_seed.householderQr().householderQ();

    Eigen::VectorXd regular_values(n);
    regular_values << 1.2, 1.0, 0.83, 0.61, 0.44, 0.29;
    check_case(
        left * regular_values.asDiagonal() * right.transpose(),
        "regular core");

    Eigen::VectorXd ill_conditioned_values(n);
    ill_conditioned_values << 1.2, 1.0, 0.83, 0.61, 1.0e-5, 2.0e-7;
    check_case(
        left * ill_conditioned_values.asDiagonal() * right.transpose(),
        "ill-conditioned core");

    Eigen::VectorXd singular_values(n);
    singular_values << 1.2, 1.0, 0.83, 0.61, 1.0e-5, 0.0;
    check_case(
        left * singular_values.asDiagonal() * right.transpose(),
        "singular core");

    std::cout << "woodbury core contraction tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
