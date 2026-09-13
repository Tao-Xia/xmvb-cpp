#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

// Diagnostic gold standard for small reduced spaces. Production optimization
// must not call this routine: it explicitly assembles H by applying the same
// matrix-free operator to blocks of coordinate vectors. Keeping assembly on
// the HVP interface makes comparisons isolate the subproblem algorithm from
// orbital/structure derivative correctness.
struct ReducedHessianReference {
  Eigen::MatrixXd raw_hessian;
  Eigen::MatrixXd symmetric_hessian;
  double relative_skew_norm = 0.0;
};

template <typename BlockHvpAction>
ReducedHessianReference assemble_reduced_hessian_reference(
    Eigen::Index reduced_dimension,
    Eigen::Index block_width,
    BlockHvpAction&& apply_hvp_block) {
  if (reduced_dimension < 0 || block_width <= 0) {
    throw std::invalid_argument("invalid reduced-Hessian reference dimensions");
  }
  ReducedHessianReference reference;
  reference.raw_hessian =
      Eigen::MatrixXd::Zero(reduced_dimension, reduced_dimension);
  for (Eigen::Index begin = 0; begin < reduced_dimension;
       begin += block_width) {
    const Eigen::Index width =
        std::min(block_width, reduced_dimension - begin);
    Eigen::MatrixXd coordinate_block =
        Eigen::MatrixXd::Zero(reduced_dimension, width);
    for (Eigen::Index column = 0; column < width; ++column) {
      coordinate_block(begin + column, column) = 1.0;
    }
    const Eigen::MatrixXd image_block = apply_hvp_block(coordinate_block);
    if (image_block.rows() != reduced_dimension ||
        image_block.cols() != width || !image_block.allFinite()) {
      throw std::runtime_error("invalid block while assembling reduced Hessian");
    }
    reference.raw_hessian.middleCols(begin, width) = image_block;
  }
  reference.symmetric_hessian =
      0.5 * (reference.raw_hessian + reference.raw_hessian.transpose()).eval();
  const double hessian_norm = reference.raw_hessian.norm();
  const double skew_norm =
      (reference.raw_hessian - reference.raw_hessian.transpose()).norm();
  reference.relative_skew_norm =
      hessian_norm > 0.0 ? skew_norm / hessian_norm : skew_norm;
  if (!std::isfinite(reference.relative_skew_norm)) {
    throw std::runtime_error("non-finite reduced-Hessian symmetry audit");
  }
  return reference;
}

}  // namespace xmvb::vb
