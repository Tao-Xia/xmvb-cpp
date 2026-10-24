#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::integrals {
struct AoIntegralInput;
struct AoH1eBatchWorkspace;
struct ExactPackedActiveTwoElectronAdjointCache;
} // namespace xmvb::integrals

namespace xmvb::vb {

namespace detail {

/**
 * @brief Builds one combined AO pullback and one active projection per direction.
 *
 * Inputs are flattened columns: symmetric density directions, symmetric
 * pullback sources, and W=delta(C)*(G_h+G_h^T). Outputs are the full combined
 * inactive-density covector and L(delta D)*C. Consumers share these outputs.
 */
void apply_exact_ao_response(const Eigen::Ref<const Eigen::MatrixXd>& density_sources,
                             const Eigen::Ref<const Eigen::MatrixXd>& pullback_sources,
                             const Eigen::Ref<const Eigen::MatrixXd>& weighted_directions,
                             const xmvb::integrals::ExactPackedActiveTwoElectronAdjointCache& cache,
                             const xmvb::integrals::AoIntegralInput& ao_integral_input,
                             xmvb::integrals::AoH1eBatchWorkspace* workspace,
                             Eigen::MatrixXd* combined_response,
                             Eigen::MatrixXd* active_projection);

} // namespace detail
} // namespace xmvb::vb
