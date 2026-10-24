#include "vbscf/integrals/ao/one_electron/backpropagator.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "parallel/openmp.hpp"
#include "linalg/storage.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/ao/ri_operator.hpp"
#include "integrals/response/ri_ao.hpp"

namespace xmvb::vb {
namespace {

std::vector<double> encode_symmetric_gradient(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
    if (matrix.rows() <= 0 || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("invalid symmetric AO gradient size");
    }
    Eigen::MatrixXd lower = matrix.triangularView<Eigen::Lower>();
    lower.diagonal() *= 0.5;
    return {lower.data(), lower.data() + lower.size()};
}

AoEffectiveOneElectronBackpropagationResult
backpropagate_ri(const std::vector<double>& gradient_storage,
                 const xmvb::integrals::RiAoFactorization& ri,
                 int n_bf) {
    const std::size_t size = static_cast<std::size_t>(n_bf) * static_cast<std::size_t>(n_bf);
    if (n_bf <= 0 || gradient_storage.size() != size || ri.n_basis_functions != n_bf) {
        throw std::invalid_argument("invalid RI AO-H1E backpropagation dimensions");
    }
    const Eigen::Map<const Eigen::MatrixXd> gradient(gradient_storage.data(), n_bf, n_bf);
    const Eigen::MatrixXd symmetric = gradient + gradient.transpose();
    auto density_gradient = xmvb::integrals::apply_ao_effective_one_electron_ri_operator(
        std::vector<double>(symmetric.data(), symmetric.data() + symmetric.size()),
        ri,
        n_bf,
        {.attempt_spectral_factorization = true});
    AoEffectiveOneElectronBackpropagationResult result;
    result.inactive_density_gradient = encode_symmetric_gradient(
        Eigen::Map<const Eigen::MatrixXd>(density_gradient.data(), n_bf, n_bf));
    return result;
}

AoEffectiveOneElectronBackpropagationResult
backpropagate_ri(const std::vector<double>& active_gradient,
                 const integrals::RiActiveTwoElectronResponseCache& cache,
                 const AoEffectiveOneElectronResult* reference) {
    const int n_active = cache.n_active_orbitals;
    if (active_gradient.size() != static_cast<std::size_t>(n_active) * n_active)
        throw std::invalid_argument("invalid active-H1E RI adjoint shape");
    const Eigen::Map<const Eigen::MatrixXd> gradient(active_gradient.data(), n_active, n_active);
    const Eigen::MatrixXd symmetric = gradient + gradient.transpose();
    Eigen::MatrixXd density_gradient =
        integrals::apply_ri_active_density_response(cache, symmetric);
    if (reference)
        density_gradient += 2 * reference->ao_coulomb_exchange_matrix;
    AoEffectiveOneElectronBackpropagationResult result;
    result.inactive_density_gradient = encode_symmetric_gradient(density_gradient);
    return result;
}

} // namespace

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(const std::vector<double>& gradient,
                                                    const xmvb::integrals::RiAoFactorization& ri,
                                                    int n_bf) const {
    return backpropagate_ri(gradient, ri, n_bf);
}

AoEffectiveOneElectronBackpropagationResult AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& gradient,
    const xmvb::integrals::RiAoFactorization& ri,
    int n_bf) const {
    return backpropagate_ri(xmvb::linalg::flatten_matrix_column_major(gradient), ri, n_bf);
}

AoEffectiveOneElectronBackpropagationResult AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& active_gradient,
    const integrals::RiActiveTwoElectronResponseCache& cache,
    const AoEffectiveOneElectronResult* reference) const {
    return backpropagate_ri(active_gradient, cache, reference);
}

AoEffectiveOneElectronBackpropagationResult AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& gradient_storage, const xmvb::integrals::AoIntegralInput& ao) const {
    const int n_bf = ao.n_basis_functions;
    const std::size_t size = static_cast<std::size_t>(n_bf) * static_cast<std::size_t>(n_bf);
    if (n_bf <= 0 || gradient_storage.size() != size) {
        throw std::invalid_argument("invalid AO-H1E gradient dimensions");
    }
    const Eigen::Map<const Eigen::MatrixXd> gradient(gradient_storage.data(), n_bf, n_bf);
    const Eigen::MatrixXd symmetric = gradient + gradient.transpose();
    AoEffectiveOneElectronBackpropagationResult result;
    result.inactive_density_gradient = xmvb::integrals::apply_ao_h1e_transpose(
        symmetric.data(), ao, effective_openmp_thread_count());
    return result;
}

AoEffectiveOneElectronBackpropagationResult AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& gradient,
    const xmvb::integrals::AoIntegralInput& ao) const {
    if (gradient.rows() != ao.n_basis_functions || gradient.cols() != ao.n_basis_functions) {
        throw std::invalid_argument("invalid AO-H1E gradient dimensions");
    }
    return backpropagate(xmvb::linalg::flatten_matrix_column_major(gradient), ao);
}

} // namespace xmvb::vb
