#include "integrals/response/density_diagonal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::integrals {
namespace {

/** @brief Project core Coulomb images once; no complete AO response per core. */
Eigen::MatrixXd core_coulomb_diagonals(const Eigen::MatrixXd& factors,
                                       const Eigen::MatrixXd& charges,
                                       const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                                       std::size_t allowance) {
    const int n_core = charges.cols(), n_mo = orbitals.cols();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_mo, n_core);
    if (!n_core)
        return result;
    const int n_pairs = factors.cols();
    const int width = std::min<std::size_t>(n_pairs, allowance / (n_core + n_mo));
    if (width <= 0)
        throw std::length_error("RI core diagonal pair tile exceeds workspace");
    const int workers = std::min(effective_openmp_thread_count(), n_mo);
    for (int first = 0; first < n_pairs; first += width) {
        const int count = std::min(width, n_pairs - first);
        OrbitalPairMatrix products(count, n_core), map(count, n_mo);
#pragma omp parallel num_threads(workers)
        {
            int worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            const int begin = worker * count / workers, end = (worker + 1) * count / workers;
            products.middleRows(begin, end - begin).noalias() =
                factors.middleCols(first + begin, end - begin).transpose() * charges;
#pragma omp for schedule(static)
            for (int row = 0; row < count; ++row) {
                const std::size_t pair = first + row;
                std::size_t mu = (std::sqrt(8.0 * pair + 1) - 1) / 2;
                while ((mu + 1) * (mu + 2) / 2 <= pair)
                    ++mu;
                while (mu * (mu + 1) / 2 > pair)
                    --mu;
                const int nu = pair - mu * (mu + 1) / 2;
                map.row(row) = (mu == static_cast<std::size_t>(nu) ? 1.0 : 2.0) *
                               orbitals.row(mu).cwiseProduct(orbitals.row(nu));
            }
            // The map loop's barrier also completes every disjoint product row.
            const int p = worker * n_mo / workers, q = (worker + 1) * n_mo / workers;
            result.middleRows(p, q - p).noalias() +=
                map.middleCols(p, q - p).transpose() * products;
        }
    }
    return result;
}

} // namespace

DensityDiagonals
contract_ri_density_diagonals(const RiAoFactorization& ri,
                              const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                              int n_core,
                              int n_active,
                              const Eigen::Ref<const Eigen::MatrixXd>& active_weights,
                              const Eigen::Ref<const OrbitalPairMatrix>& active_coulomb,
                              std::size_t workspace_bytes) {
    profiling::ScopedPhase phase(profiling::Phase::RiDensityDiagonal);
    const int n_bf = orbitals.rows(), n_mo = orbitals.cols();
    const int n_occ = n_core + n_active, n_aux = ri.n_auxiliary_functions;
    if (n_bf <= 0 || n_core < 0 || n_active <= 0 || n_occ > n_mo ||
        active_weights.rows() != n_active * n_active || active_coulomb.rows() != n_mo ||
        active_coulomb.cols() != n_active * n_active)
        throw std::invalid_argument("invalid RI density diagonal dimensions");
    const int workers = std::min(effective_openmp_thread_count(), n_bf);
    const std::size_t allowance = workspace_bytes / sizeof(double);
    const std::size_t charge_elements = static_cast<std::size_t>(n_aux) * n_core;
    const std::size_t fixed =
        charge_elements + static_cast<std::size_t>(workers) * n_active * n_active;
    const std::size_t per_aux = static_cast<std::size_t>(n_bf) * (n_occ + workers) + n_occ * n_mo;
    if (fixed + per_aux > allowance)
        throw std::length_error("minimum RI density diagonal workspace exceeds allowance");
    const int width = std::min<std::size_t>(n_aux, (allowance - fixed) / per_aux);
    if (width <= 0)
        throw std::length_error("RI density diagonal auxiliary tile exceeds workspace");
    DensityDiagonals result{
        Eigen::MatrixXd::Zero(n_mo, n_core), 2 * active_coulomb * active_weights, {}};
    Eigen::MatrixXd charges(n_aux, n_core);
    Eigen::MatrixXd one_index;
    for (int first = 0; first < n_aux; first += width) {
        const int count = std::min(width, n_aux - first);
        one_index.resize(static_cast<Eigen::Index>(count) * n_occ, n_bf);
        transform_ri_one_index(
            ri.metric_whitened_ao_pair_factors, orbitals.leftCols(n_occ), first, count, one_index);
        const int rows = count * n_occ;
        Eigen::MatrixXd projected(rows, n_mo);
#pragma omp parallel num_threads(workers)
        {
            int worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            const int begin = worker * rows / workers, end = (worker + 1) * rows / workers;
            projected.middleRows(begin, end - begin).noalias() =
                one_index.middleRows(begin, end - begin) * orbitals;
#pragma omp barrier
#pragma omp for schedule(static) nowait
            for (int c = 0; c < n_core; ++c)
                charges.col(c).segment(first, count) = projected.col(c).segment(c * count, count);
            Eigen::MatrixXd moment(n_active, n_active);
#pragma omp for schedule(static)
            for (int p = 0; p < n_mo; ++p) {
                for (int c = 0; c < n_core; ++c)
                    result.core(p, c) -= projected.col(p).segment(c * count, count).squaredNorm();
                const Eigen::Map<const Eigen::MatrixXd> active(
                    projected.col(p).data() + n_core * count, count, n_active);
                moment.noalias() = active.transpose() * active;
                const Eigen::Map<const Eigen::VectorXd> flat(moment.data(), moment.size());
                result.active.row(p).noalias() -= flat.transpose() * active_weights;
            }
        }
    }
    result.core_coulomb = core_coulomb_diagonals(
        ri.metric_whitened_ao_pair_factors, charges, orbitals, allowance - charge_elements);
    result.core.noalias() += 2 * result.core_coulomb;
    return result;
}

} // namespace xmvb::integrals
