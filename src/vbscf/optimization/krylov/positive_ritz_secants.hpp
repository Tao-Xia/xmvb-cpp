#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>

namespace xmvb::vb {

struct RitzSecant {
  Eigen::VectorXd direction;
  Eigen::VectorXd image;
};

// Keep the soft positive Ritz modes of an already evaluated HVP subspace.
// The images are H*(Q*z), not Ritz-value multiples of Q*z: components outside
// span(Q) must not be discarded. These pairs may approximate the next-point
// preconditioner after transport; they are never next-point exact HVPs.
inline std::vector<RitzSecant> positive_ritz_secants(
    const Eigen::MatrixXd& q, const Eigen::MatrixXd& hq,
    const Eigen::MatrixXd& projected_hessian, int maximum_pairs) {
  if (q.rows() != hq.rows() || q.cols() != hq.cols() ||
      projected_hessian.rows() != q.cols() || projected_hessian.cols() != q.cols() ||
      !q.allFinite() || !hq.allFinite() || !projected_hessian.allFinite()) {
    throw std::invalid_argument("invalid Ritz-secant subspace");
  }
  std::vector<RitzSecant> pairs;
  if (maximum_pairs <= 0 || q.cols() == 0) return pairs;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ritz(projected_hessian);
  if (ritz.info() != Eigen::Success) return pairs;
  const auto& values = ritz.eigenvalues();
  const double floor = std::numeric_limits<double>::epsilon() *
      values.size() * values.cwiseAbs().maxCoeff();
  std::vector<Eigen::Index> selected;
  for (Eigen::Index j = 0; j < values.size() &&
       selected.size() < static_cast<std::size_t>(maximum_pairs); ++j) {
    if (values[j] > floor) selected.push_back(j);
  }
  // Append the softest mode last so ordinary FIFO history truncation keeps it.
  for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
    pairs.push_back({q * ritz.eigenvectors().col(*it),
                     hq * ritz.eigenvectors().col(*it)});
  }
  return pairs;
}

}  // namespace xmvb::vb
