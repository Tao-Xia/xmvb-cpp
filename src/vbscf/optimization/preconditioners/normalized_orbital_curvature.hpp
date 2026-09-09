#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

// Exact pullback Hessian of the frozen one-electron surrogate
// e(x) = (x^T F x)/(x^T S x), x(d) = x + U d.
// F and S are real symmetric matrices; x^T S x must be positive.
// Include stored but frozen coefficients in x; their rows in U are zero.
// This is a local preconditioning model, NOT the relaxed VBSCF Hessian.
inline Eigen::MatrixXd normalized_orbital_curvature(
    const Eigen::MatrixXd& f, const Eigen::MatrixXd& overlap,
    const Eigen::VectorXd& x, const Eigen::MatrixXd& tangent) {
  const Eigen::Index n = x.size();
  if (f.rows() != n || f.cols() != n || overlap.rows() != n ||
      overlap.cols() != n || tangent.rows() != n || !f.allFinite() ||
      !overlap.allFinite() || !x.allFinite() || !tangent.allFinite()) {
    throw std::invalid_argument("invalid normalized-orbital curvature input");
  }
  const Eigen::VectorXd sx = overlap * x;
  const Eigen::VectorXd fx = f * x;
  const double norm_squared = x.dot(sx);
  if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
    throw std::invalid_argument("nonpositive normalized-orbital metric norm");
  }
  const double energy = x.dot(fx) / norm_squared;
  const Eigen::VectorXd a = tangent.transpose() * ((fx - energy * sx) / norm_squared);
  const Eigen::VectorXd b = tangent.transpose() * (sx / norm_squared);
  Eigen::MatrixXd curvature = (2.0 / norm_squared) *
      (tangent.transpose() * (f * tangent - energy * (overlap * tangent)));
  curvature.noalias() -= 4.0 * (a * b.transpose() + b * a.transpose());
  curvature = (0.5 * (curvature + curvature.transpose())).eval();
  if (!curvature.allFinite()) {
    throw std::runtime_error("non-finite normalized-orbital curvature");
  }
  return curvature;
}

}  // namespace xmvb::vb
