#pragma once

#include <Eigen/Core>

#include "integrals/ao/input.hpp"
#include "integrals/ao/ri.hpp"
#include "scf/driver/solver.hpp"

namespace xmvb::scf {

/** Restricted HF evaluation for total density P: F = h + J[P] - K[P]/2. */
Evaluation evaluate_hf(const Eigen::Ref<const Eigen::MatrixXd>& density,
                       const integrals::AoIntegralInput& ao);

Evaluation evaluate_hf(const Eigen::Ref<const Eigen::MatrixXd>& density,
                       const Eigen::Ref<const Eigen::MatrixXd>& core,
                       const integrals::RiAoFactorization& ri);

} // namespace xmvb::scf
