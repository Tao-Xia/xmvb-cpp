#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/contracted_exterior_jet.hpp"

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

void check_update(int dimension, int rank, int maximum_order) {
  Eigen::MatrixXd channel(dimension, dimension);
  Eigen::MatrixXd left(dimension, rank);
  Eigen::MatrixXd right(dimension, rank);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      channel(row, column) =
          0.031 * (row + 1) - 0.017 * (column + 2) +
          0.004 * ((row + 3) * (column + 1) % 7);
    }
  }
  for (int column = 0; column < rank; ++column) {
    for (int row = 0; row < dimension; ++row) {
      left(row, column) =
          0.019 * (row + 1) + 0.006 * (column + 2);
      right(row, column) =
          -0.013 * (row + 2) + 0.005 * (column + 1);
    }
  }

  xmvb::vb::ContractedExteriorJet updated =
      xmvb::vb::build_contracted_exterior_jet(channel, maximum_order);
  xmvb::vb::update_contracted_exterior_jet(
      channel, left, right, &updated);
  const xmvb::vb::ContractedExteriorJet rebuilt =
      xmvb::vb::build_contracted_exterior_jet(
          channel + left * right.transpose(), maximum_order);
  for (int order = 0; order <= maximum_order; ++order) {
    require_close(
        updated.coefficients[order],
        rebuilt.coefficients[order],
        2.0e-11,
        "low-rank exterior coefficient");
  }
}

}  // namespace

int main() {
  try {
    for (int maximum_order = 0; maximum_order <= 8; ++maximum_order) {
      check_update(8, 1, maximum_order);
      check_update(8, 2, maximum_order);
      check_update(8, 3, maximum_order);
    }

    // The channel itself may be singular.  The graph update identity only
    // uses the formal inverse of I + t A, whose constant term is always the
    // identity.
    Eigen::MatrixXd singular = Eigen::MatrixXd::Zero(8, 8);
    singular.diagonal().head(4).setLinSpaced(0.2, 0.8);
    const Eigen::MatrixXd left =
        Eigen::MatrixXd::Random(8, 2) * 0.03;
    const Eigen::MatrixXd right =
        Eigen::MatrixXd::Random(8, 2) * 0.02;
    xmvb::vb::ContractedExteriorJet updated =
        xmvb::vb::build_contracted_exterior_jet(singular, 8);
    xmvb::vb::update_contracted_exterior_jet(
        singular, left, right, &updated);
    const xmvb::vb::ContractedExteriorJet rebuilt =
        xmvb::vb::build_contracted_exterior_jet(
            singular + left * right.transpose(), 8);
    for (int order = 0; order <= 8; ++order) {
      require_close(
          updated.coefficients[order],
          rebuilt.coefficients[order],
          2.0e-11,
          "singular low-rank exterior coefficient");
    }

    std::cout << "contracted exterior jet tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
