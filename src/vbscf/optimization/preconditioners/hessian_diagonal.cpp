#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

#include <stdexcept>

#include "vbscf/derivatives/hessian/exact/operator.hpp"

namespace xmvb::vb {

Eigen::VectorXd extract_reduced_hessian_diagonal(
    int reduced_size,
    const std::vector<OrbitalChart::ReducedBlock>& blocks,
    const ReducedBlockHvp& apply_block) {
  if (reduced_size < 0 || !apply_block) {
    throw std::invalid_argument("invalid reduced Hessian diagonal input");
  }
  Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(reduced_size);
  std::vector<char> covered(static_cast<std::size_t>(reduced_size), 0);
  for (const auto& block : blocks) {
    if (block.offset < 0 || block.size <= 0 ||
        block.offset + block.size > reduced_size) {
      throw std::invalid_argument("invalid reduced Hessian diagonal block");
    }
    Eigen::MatrixXd directions =
        Eigen::MatrixXd::Zero(reduced_size, block.size);
    for (int column = 0; column < block.size; ++column) {
      const int coordinate = block.offset + column;
      if (covered[static_cast<std::size_t>(coordinate)] != 0) {
        throw std::invalid_argument("overlapping reduced Hessian blocks");
      }
      covered[static_cast<std::size_t>(coordinate)] = 1;
      directions(coordinate, column) = 1.0;
    }
    const Eigen::MatrixXd images = apply_block(directions);
    if (images.rows() != reduced_size || images.cols() != block.size ||
        !images.allFinite()) {
      throw std::runtime_error("invalid reduced Hessian block image");
    }
    for (int column = 0; column < block.size; ++column) {
      diagonal[block.offset + column] = images(block.offset + column, column);
    }
  }
  for (const char present : covered) {
    if (present == 0) {
      throw std::invalid_argument("reduced Hessian blocks do not cover the chart");
    }
  }
  if (!diagonal.allFinite()) {
    throw std::runtime_error("non-finite reduced Hessian diagonal");
  }
  return diagonal;
}

Eigen::VectorXd build_reduced_hessian_diagonal(
    const ExactHvpOperator& hessian,
    const OrbitalChart& chart) {
  const HvpComponents orbital_block{
      .direct_core_response = true,
      .fixed_upstream_pullback = true,
      .local_active_response = true,
      .structure_response = false};
  return extract_reduced_hessian_diagonal(
      chart.reduced_size(),
      chart.reduced_blocks(),
      [&hessian, orbital_block](const Eigen::MatrixXd& directions) {
        return hessian.apply_reduced_batch(directions, orbital_block);
      });
}

}  // namespace xmvb::vb
