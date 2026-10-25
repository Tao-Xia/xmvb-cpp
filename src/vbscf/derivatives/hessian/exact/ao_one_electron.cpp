#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"

#include <stdexcept>
#include <vector>

#include "parallel/openmp.hpp"
#include "integrals/ao/input.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/response/adjoint.hpp"

namespace xmvb::vb::detail {

void apply_exact_ao_response(const Eigen::Ref<const Eigen::MatrixXd>& density,
                             const Eigen::Ref<const Eigen::MatrixXd>& pullback,
                             const Eigen::Ref<const Eigen::MatrixXd>& weighted_directions,
                             const xmvb::integrals::ExactPackedActiveTwoElectronAdjointCache& cache,
                             const xmvb::integrals::AoIntegralInput& ao,
                             xmvb::integrals::AoH1eBatchWorkspace* workspace,
                             Eigen::MatrixXd* combined_response,
                             Eigen::MatrixXd* active_projection) {
    const int n_bf = ao.n_basis_functions;
    const int n = cache.n_active_orbitals;
    if (n_bf <= 0 || density.rows() != n_bf * n_bf || pullback.rows() != density.rows() ||
        pullback.cols() != density.cols() || weighted_directions.rows() != n_bf * n ||
        weighted_directions.cols() != density.cols() || combined_response == nullptr ||
        active_projection == nullptr)
        throw std::invalid_argument("invalid exact AO response arguments");
    const auto& coefficients = *cache.accepted_active_coefficients;
    active_projection->resize(n_bf * n, density.cols());
    if (cache.exchange_integrals.size() != 0) {
        // T=delta D+W C^T, pullback=T+T^T. By linearity,
        // R=L(delta D)+L(T+T^T)/2=L(2 delta D+(W C^T+C W^T)/2).
        // Therefore L(delta D) C=R C/2-L(W C^T+C W^T) C/4.
        // Only R traverses the AO integral graph; the correction uses accepted
        // half-transformed images in O(n_bf^2*n_active^2) arithmetic.
        const Eigen::MatrixXd source = density + 0.5 * pullback;
        xmvb::integrals::apply_symmetric_ao_h1e_batch(
            source, ao, effective_openmp_thread_count(), workspace, combined_response);
        for (Eigen::Index column = 0; column < density.cols(); ++column) {
            const Eigen::Map<const Eigen::MatrixXd> response(
                combined_response->col(column).data(), n_bf, n_bf);
            const Eigen::Map<const Eigen::MatrixXd> weights(
                weighted_directions.col(column).data(), n_bf, n);
            Eigen::Map<Eigen::MatrixXd> projection(active_projection->col(column).data(), n_bf, n);
            projection.noalias() = 0.5 * response * coefficients;
            projection -=
                0.25 * xmvb::integrals::project_exact_active_density_response(cache, weights);
        }
    } else {
        // When the bounded accepted image is not admitted, keep the streamed
        // integral representation. Consumers still share a single projection.
        Eigen::MatrixXd forward, adjoint;
        xmvb::integrals::apply_symmetric_ao_h1e_fused_batch(
            density, pullback, ao, effective_openmp_thread_count(), workspace, &forward, &adjoint);
        *combined_response = forward + adjoint;
        for (Eigen::Index column = 0; column < density.cols(); ++column) {
            const Eigen::Map<const Eigen::MatrixXd> response(
                forward.col(column).data(), n_bf, n_bf);
            Eigen::Map<Eigen::MatrixXd> projection(active_projection->col(column).data(), n_bf, n);
            projection.noalias() = response * coefficients;
        }
    }
}

} // namespace xmvb::vb::detail
