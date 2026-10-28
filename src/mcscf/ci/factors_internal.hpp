#pragma once

#include "mcscf/ci/factors.hpp"

namespace xmvb::mcscf::detail {

enum class FactorContraction { Integral, Generator, Residual };

/** @brief Compare two exact amplitude multiplication orders, never precision. */
bool use_amplitude_map(const SpinFactorSource& source);

/** @brief Shared scratch doubles for one amplitude multiplication panel. */
std::size_t
amplitude_workspace(const SpinFactorSource& source, Eigen::Index tile, Eigen::Index columns);

bool integral_bra_is_left(const DirectCiSigmaAction& sigma,
                          const SpinFactorSource& left,
                          const SpinFactorSource& right,
                          std::size_t workspace_bytes);

/** @brief Joint source/channel schedule; counts work, never changes the operator. */
struct FactorPanels {
    Eigen::Index left = 0, right = 0, channels = 0;
    std::size_t workspace_bytes = 0;
    long double work = 0, source_columns = 0;
};

FactorPanels plan_factor_panels(const DirectCiSigmaAction& sigma,
                                const SpinFactorSource& left,
                                const SpinFactorSource& right,
                                std::size_t workspace_bytes,
                                FactorContraction contraction);

/** @brief Integral work including the chosen slab schedule and provider replay.
 * Uses the same self-adjoint bra/ket order as factor_integral_adjoint().
 * This is an arithmetic/traffic estimate, not an accuracy or admission test.
 */
long double factor_integral_work(const DirectCiSigmaAction& sigma,
                                 const SpinFactorSource& left,
                                 const SpinFactorSource& right,
                                 std::size_t workspace_bytes);

void validate_source(const DirectCiSigmaAction& sigma, const SpinFactorSource& source);
void add_product(const Eigen::Ref<const Eigen::MatrixXd>& left,
                 const Eigen::Ref<const Eigen::MatrixXd>& right,
                 Eigen::Ref<Eigen::MatrixXd> result);
void transpose_product(const Eigen::Ref<const Eigen::MatrixXd>& left,
                       const Eigen::Ref<const Eigen::MatrixXd>& right,
                       Eigen::Ref<Eigen::MatrixXd> result);

} // namespace xmvb::mcscf::detail
