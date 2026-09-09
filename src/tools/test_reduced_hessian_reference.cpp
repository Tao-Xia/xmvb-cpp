#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/diagnostics/reduced_hessian_reference.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    constexpr int n = 17;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
      h(i, i) = 0.25 + i;
      if (i > 0) h(i - 1, i) = h(i, i - 1) = -0.13 * i;
    }
    int calls = 0;
    const auto reference = xmvb::vb::assemble_reduced_hessian_reference(
        n, 4, [&](const Eigen::Ref<const Eigen::MatrixXd>& directions) {
          ++calls;
          return h * directions;
        });
    require(calls == 5, "coordinate HVPs were not assembled in blocks");
    require((reference.raw_hessian - h).norm() == 0.0,
            "assembled reduced Hessian is incorrect");
    require(reference.relative_skew_norm == 0.0,
            "symmetric reference reported a skew component");
    std::cout << "reduced Hessian reference: passed (17 coordinates, width 4)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
