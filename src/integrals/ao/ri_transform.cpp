#include "integrals/ao/ri_transform.hpp"

#include <algorithm>
#include <stdexcept>
#include "parallel/openmp.hpp"
#include "core/timing.hpp"

namespace xmvb::integrals {

void transform_ri_one_index(const Eigen::Ref<const Eigen::MatrixXd>& packed_factors,
                            const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                            int first_aux,
                            int count,
                            Eigen::Ref<Eigen::MatrixXd> transformed) {
    profiling::ScopedPhase phase(profiling::Phase::RiOneIndex);
    const int n = coefficients.rows(), orbitals = coefficients.cols();
    if (n <= 0 || first_aux < 0 || count <= 0 || first_aux + count > packed_factors.rows() ||
        packed_factors.cols() != static_cast<Eigen::Index>(n) * (n + 1) / 2 ||
        transformed.rows() != static_cast<Eigen::Index>(count) * orbitals ||
        transformed.cols() != n)
        throw std::invalid_argument("invalid RI one-index transform dimensions");
    const int workers = std::min(effective_openmp_thread_count(), n);
#pragma omp parallel num_threads(workers)
    {
        Eigen::MatrixXd ao_slice(count, n);
#pragma omp for schedule(static)
        for (int mu = 0; mu < n; ++mu) {
            for (int nu = 0; nu < n; ++nu) {
                const int hi = std::max(mu, nu), lo = std::min(mu, nu);
                const Eigen::Index pair = static_cast<Eigen::Index>(hi) * (hi + 1) / 2 + lo;
                ao_slice.col(nu) = packed_factors.col(pair).segment(first_aux, count);
            }
            Eigen::Map<Eigen::MatrixXd> output(transformed.col(mu).data(), count, orbitals);
            output.noalias() = ao_slice * coefficients;
        }
    }
}

void fill_ri_one_index_tiled(const Eigen::Ref<const Eigen::MatrixXd>& packed_factors,
                             const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                             Eigen::Ref<Eigen::MatrixXd> result,
                             std::size_t workspace_bytes) {
    const int n_bf = coefficients.rows(), orbitals = coefficients.cols();
    const int n_aux = packed_factors.rows();
    if (result.rows() != static_cast<Eigen::Index>(n_aux) * orbitals || result.cols() != n_bf)
        throw std::invalid_argument("RI one-index destination has the wrong shape");
    const int workers = std::min(effective_openmp_thread_count(), n_bf);
    const std::size_t per_aux = static_cast<std::size_t>(n_bf) * (orbitals + workers);
    if (workspace_bytes / sizeof(double) < per_aux)
        throw std::length_error("minimum RI transform tile exceeds workspace");
    const int width = std::min<std::size_t>(
        n_aux, std::max<std::size_t>(1, workspace_bytes / sizeof(double) / per_aux));
    Eigen::MatrixXd tile;
    for (int first = 0; first < n_aux; first += width) {
        const int count = std::min(width, n_aux - first);
        tile.resize(static_cast<Eigen::Index>(count) * orbitals, n_bf);
        transform_ri_one_index(packed_factors, coefficients, first, count, tile);
        for (int a = 0; a < orbitals; ++a)
            result.middleRows(static_cast<Eigen::Index>(a) * n_aux + first, count) =
                tile.middleRows(a * count, count);
    }
}

} // namespace xmvb::integrals
