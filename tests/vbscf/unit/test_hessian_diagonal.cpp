#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    constexpr int n = 7;
    Eigen::MatrixXd hessian(n, n);
    for (int row = 0; row < n; ++row) {
      for (int column = 0; column < n; ++column) {
        hessian(row, column) =
            row == column ? 0.5 + row : 0.01 * (row + column + 1);
      }
    }
    const std::vector<xmvb::vb::OrbitalChart::ReducedBlock> blocks{
        {0, 2}, {2, 1}, {3, 4}};
    std::vector<int> observed_widths;
    const Eigen::VectorXd diagonal =
        xmvb::vb::extract_reduced_hessian_diagonal(
            n,
            blocks,
            [&](const Eigen::MatrixXd& directions) {
              observed_widths.push_back(
                  static_cast<int>(directions.cols()));
              return hessian * directions;
            });
    require((diagonal - hessian.diagonal()).norm() == 0.0,
            "block-HVP Hessian diagonal is incorrect");
    require(observed_widths == std::vector<int>({2, 1, 4}),
            "Hessian diagonal did not preserve natural orbital blocks");
    std::cout << "reduced block-HVP Hessian diagonal: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
