#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace xmvb::vb {

struct ProjectedOrbitalSurrogate {
  Eigen::MatrixXd one_electron;
  Eigen::MatrixXd overlap;
};

// Pull back the one-electron Rayleigh quotient of R*c, where R projects out
// the fixed inactive span in the AO overlap metric. For an inactive target,
// fixed_inactive excludes that target; for an active target it includes all
// inactive orbitals. The sparse input support is unchanged by this model.
inline ProjectedOrbitalSurrogate projected_orbital_surrogate(
    const Eigen::Ref<const Eigen::MatrixXd>& f,
    const Eigen::Ref<const Eigen::MatrixXd>& s,
    const Eigen::Ref<const Eigen::MatrixXd>& fixed_inactive,
    const std::vector<int>& support) {
  if (f.rows() != f.cols() || s.rows() != f.rows() || s.cols() != f.cols() ||
      fixed_inactive.rows() != f.rows() || !f.allFinite() || !s.allFinite() ||
      !fixed_inactive.allFinite())
    throw std::invalid_argument("invalid projected orbital surrogate input");
  Eigen::MatrixXd injection = Eigen::MatrixXd::Zero(f.rows(), support.size());
  for (std::size_t j = 0; j < support.size(); ++j) {
    if (support[j] < 0 || support[j] >= f.rows())
      throw std::invalid_argument("invalid projected orbital support");
    injection(support[j], j) = 1.0;
  }
  Eigen::MatrixXd projected = injection;
  if (fixed_inactive.cols() > 0) {
    const Eigen::MatrixXd sc = s * fixed_inactive;
    Eigen::MatrixXd gram = fixed_inactive.transpose() * sc;
    gram = (0.5 * (gram + gram.transpose())).eval();
    Eigen::LDLT<Eigen::MatrixXd> factor(gram);
    if (factor.info() != Eigen::Success || !factor.vectorD().allFinite() ||
        !(factor.vectorD().minCoeff() > 0.0))
      throw std::runtime_error("rank-deficient fixed inactive surrogate span");
    projected.noalias() -= fixed_inactive * factor.solve(sc.transpose() * injection);
  }
  ProjectedOrbitalSurrogate result;
  result.one_electron = projected.transpose() * f * projected;
  result.overlap = projected.transpose() * s * projected;
  result.one_electron = (0.5 * (result.one_electron + result.one_electron.transpose())).eval();
  result.overlap = (0.5 * (result.overlap + result.overlap.transpose())).eval();
  return result;
}

}  // namespace xmvb::vb
