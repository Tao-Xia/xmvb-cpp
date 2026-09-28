#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/determinants/algebra/contracted_density.hpp"

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

Eigen::MatrixXd make_matrix(int dimension, double phase) {
  Eigen::MatrixXd result(dimension, dimension);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      result(row, column) =
          std::sin(phase * (row + 1) * (column + 2)) +
          0.2 * std::cos((phase + 0.17) * (row + 2) * (column + 1));
    }
  }
  return result;
}

void check_gradients(
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& transition,
    int maximum_order) {
  const Eigen::MatrixXd inverse = overlap.inverse();
  const double determinant = overlap.determinant();
  const xmvb::vb::ContractedDensityState state(
      inverse, transition, maximum_order);
  const Eigen::MatrixXd overlap_direction = make_matrix(overlap.rows(), 0.11);
  const Eigen::MatrixXd transition_direction =
      make_matrix(overlap.rows(), 0.07);
  constexpr double epsilon = 2.0e-6;
  for (int order = 0; order <= maximum_order; ++order) {
    const Eigen::MatrixXd plus_overlap =
        overlap + epsilon * overlap_direction;
    const Eigen::MatrixXd minus_overlap =
        overlap - epsilon * overlap_direction;
    const xmvb::vb::ContractedDensityState plus(
        plus_overlap.inverse(),
        transition + epsilon * transition_direction,
        maximum_order);
    const xmvb::vb::ContractedDensityState minus(
        minus_overlap.inverse(),
        transition - epsilon * transition_direction,
        maximum_order);
    const double finite_difference =
        (plus.contraction(order, plus_overlap.determinant()) -
         minus.contraction(order, minus_overlap.determinant())) /
        (2.0 * epsilon);
    const double analytic =
        (state.overlap_gradient(order, determinant)
             .cwiseProduct(overlap_direction)).sum() +
        (state.transition_gradient(order, determinant)
             .cwiseProduct(transition_direction)).sum();
    require_close(
        analytic,
        finite_difference,
        2.0e-6,
        "contracted-density gradient");
  }
}

void check_low_rank_update(
    const Eigen::MatrixXd& inverse,
    const Eigen::MatrixXd& transition,
    int maximum_order,
    int inverse_rank,
    int channel_rank) {
  const int n = inverse.rows();
  xmvb::vb::ContractedDensityState updated(
      inverse, transition, maximum_order);
  const Eigen::MatrixXd old_channel = inverse * transition;
  const Eigen::MatrixXd inverse_left =
      make_matrix(n, 0.031).leftCols(inverse_rank);
  const Eigen::MatrixXd inverse_right =
      make_matrix(n, 0.047).leftCols(inverse_rank);
  const Eigen::MatrixXd channel_left =
      make_matrix(n, 0.023).leftCols(channel_rank);
  const Eigen::MatrixXd channel_right =
      make_matrix(n, 0.059).leftCols(channel_rank);
  updated.update(
      inverse_left, inverse_right, channel_left, channel_right);

  const Eigen::MatrixXd new_inverse =
      inverse + inverse_left * inverse_right.transpose();
  const Eigen::MatrixXd new_channel =
      old_channel + channel_left * channel_right.transpose();
  const Eigen::MatrixXd new_transition =
      new_inverse.fullPivLu().solve(new_channel);
  const xmvb::vb::ContractedDensityState rebuilt(
      new_inverse, new_transition, maximum_order);
  require_matrix_close(
      updated.channel(), rebuilt.channel(), 2.0e-12, "updated channel");
  for (int order = 0; order <= maximum_order; ++order) {
    require_close(
        updated.coefficient(order),
        rebuilt.coefficient(order),
        2.0e-10,
        "updated arbitrary-order contraction");
    require_matrix_close(
        updated.transition_gradient(order, 0.73),
        rebuilt.transition_gradient(order, 0.73),
        3.0e-10,
        "updated transition adjoint");
    require_matrix_close(
        updated.overlap_gradient(order, 0.73),
        rebuilt.overlap_gradient(order, 0.73),
        3.0e-10,
        "updated overlap adjoint");
  }
}

}  // namespace

int main() {
  try {
    constexpr int n = 7;
    Eigen::MatrixXd overlap = make_matrix(n, 0.19);
    overlap.diagonal().array() += 3.0;
    const Eigen::MatrixXd transition = make_matrix(n, 0.13);
    check_gradients(overlap, transition, 6);

    const Eigen::MatrixXd inverse = overlap.inverse();
    for (int order = 0; order <= 6; ++order) {
      check_low_rank_update(inverse, transition, order, 1, 1);
      check_low_rank_update(inverse, transition, order, 1, 2);
      check_low_rank_update(inverse, transition, order, 2, 2);
    }
    std::cout << "contracted density tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
