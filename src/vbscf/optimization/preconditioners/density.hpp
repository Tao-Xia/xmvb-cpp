#pragma once

#include <vector>
#include <Eigen/Core>
#include "integrals/response/density_diagonal.hpp"
#include "integrals/response/types.hpp"

namespace xmvb::vb {
struct VbScfInput;

using integrals::DensityDiagonals;

/** @brief Exact source-span contraction without spectral density truncation. */
DensityDiagonals contract_diagonal_densities(const VbScfInput& input,
                                             const Eigen::MatrixXd& frame,
                                             int n_core,
                                             const Eigen::MatrixXd& dm1,
                                             const std::vector<double>& dm2,
                                             const std::vector<double>& active_coulomb,
                                             const integrals::ExactCtxPairMatrix* exchange,
                                             const Eigen::MatrixXd* active_to_exchange);
} // namespace xmvb::vb
