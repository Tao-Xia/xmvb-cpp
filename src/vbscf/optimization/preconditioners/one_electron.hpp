#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace xmvb::vb {

struct ProjectedOrbitalSurrogate {
  Eigen::MatrixXd one_electron;
  Eigen::MatrixXd overlap;
};

/**
 * @brief Projects a one-electron model out of the fixed inactive span.
 *
 * The sparse support is unchanged. The result is a local preconditioning
 * model, not the relaxed VBSCF Hessian.
 */
inline ProjectedOrbitalSurrogate projected_orbital_surrogate(
    const Eigen::Ref<const Eigen::MatrixXd>& f,
    const Eigen::Ref<const Eigen::MatrixXd>& s,
    const Eigen::Ref<const Eigen::MatrixXd>& fixed_inactive,
    const std::vector<int>& support) {
  if (f.rows() != f.cols() || s.rows() != f.rows() || s.cols() != f.cols() ||
      fixed_inactive.rows() != f.rows() || !f.allFinite() || !s.allFinite() ||
      !fixed_inactive.allFinite()) {
    throw std::invalid_argument("invalid projected orbital surrogate input");
  }
  Eigen::MatrixXd injection = Eigen::MatrixXd::Zero(f.rows(), support.size());
  for (std::size_t column = 0; column < support.size(); ++column) {
    if (support[column] < 0 || support[column] >= f.rows()) {
      throw std::invalid_argument("invalid projected orbital support");
    }
    injection(support[column], column) = 1.0;
  }
  Eigen::MatrixXd projected = injection;
  if (fixed_inactive.cols() > 0) {
    const Eigen::MatrixXd sc = s * fixed_inactive;
    Eigen::MatrixXd gram = fixed_inactive.transpose() * sc;
    gram = (0.5 * (gram + gram.transpose())).eval();
    Eigen::LDLT<Eigen::MatrixXd> factor(gram);
    if (factor.info() != Eigen::Success || !factor.vectorD().allFinite() ||
        !(factor.vectorD().minCoeff() > 0.0)) {
      throw std::runtime_error("rank-deficient fixed inactive surrogate span");
    }
    projected.noalias() -=
        fixed_inactive * factor.solve(sc.transpose() * injection);
  }
  ProjectedOrbitalSurrogate result;
  result.one_electron = projected.transpose() * f * projected;
  result.overlap = projected.transpose() * s * projected;
  result.one_electron =
      (0.5 * (result.one_electron + result.one_electron.transpose())).eval();
  result.overlap =
      (0.5 * (result.overlap + result.overlap.transpose())).eval();
  return result;
}

/**
 * @brief Exact pullback Hessian of a normalized one-electron Rayleigh quotient.
 *
 * This evaluates the Hessian of
 * \f$e(d)=(x+Ud)^T F(x+Ud)/(x+Ud)^T S(x+Ud)\f$ at \f$d=0\f$.
 * Frozen stored coefficients remain in `x` and have zero rows in `tangent`.
 */
inline Eigen::MatrixXd normalized_orbital_curvature(
    const Eigen::MatrixXd& f,
    const Eigen::MatrixXd& overlap,
    const Eigen::VectorXd& x,
    const Eigen::MatrixXd& tangent) {
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
  const Eigen::VectorXd a =
      tangent.transpose() * ((fx - energy * sx) / norm_squared);
  const Eigen::VectorXd b = tangent.transpose() * (sx / norm_squared);
  Eigen::MatrixXd curvature = (2.0 / norm_squared) *
      (tangent.transpose() * (f * tangent - energy * overlap * tangent));
  curvature.noalias() -= 4.0 * (a * b.transpose() + b * a.transpose());
  curvature = (0.5 * (curvature + curvature.transpose())).eval();
  if (!curvature.allFinite()) {
    throw std::runtime_error("non-finite normalized-orbital curvature");
  }
  return curvature;
}

}  // namespace xmvb::vb
