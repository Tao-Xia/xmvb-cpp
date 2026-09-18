#include "vbscf/optimization/objective/reduced_hvp.hpp"

namespace xmvb::vb {

Eigen::MatrixXd ReducedHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  Eigen::MatrixXd responses(
      reduced_directions.rows(),
      reduced_directions.cols());
  for (Eigen::Index column = 0;
       column < reduced_directions.cols();
       ++column) {
    responses.col(column) = apply(reduced_directions.col(column));
  }
  return responses;
}

}  // namespace xmvb::vb
