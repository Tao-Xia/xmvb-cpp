#include "integrals/active/ri.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include "core/pair_index.hpp"
#include "integrals/ao/ri_transform.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::integrals {

std::vector<double>
pack_ri_active_integrals(const Eigen::Ref<const Eigen::MatrixXd>& active_pair_factors) {
    if (active_pair_factors.rows() <= 0 || active_pair_factors.cols() <= 0)
        throw std::invalid_argument("RI active kernel requires nonempty pair factors");
    const Eigen::Index pairs = active_pair_factors.cols();
    std::vector<double> packed(static_cast<std::size_t>(pairs) * (pairs + 1) / 2);
    constexpr Eigen::Index panel_rows = 64;
    for (Eigen::Index first = 0; first < pairs; first += panel_rows) {
        const Eigen::Index count = std::min(panel_rows, pairs - first);
        const Eigen::MatrixXd panel = active_pair_factors.middleCols(first, count).transpose() *
                                      active_pair_factors.leftCols(first + count);
        for (Eigen::Index row = 0; row < count; ++row) {
            const Eigen::Index p = first + row;
            for (Eigen::Index q = 0; q <= p; ++q)
                packed[static_cast<std::size_t>(p) * (p + 1) / 2 + q] = panel(row, q);
        }
    }
    return packed;
}

ActiveSpaceTwoElectronResult
transform_ri_active_integrals(const RiAoFactorization& ao_ri_result,
                              const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients) {
    const int n_basis_functions = static_cast<int>(dense_active_coefficients.rows());
    const int n_active_orbitals = static_cast<int>(dense_active_coefficients.cols());
    if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
        throw std::invalid_argument("RI active-space dimensions must be positive");
    }
    if (ao_ri_result.n_basis_functions != n_basis_functions) {
        throw std::invalid_argument("AO RI result basis-function count mismatch");
    }

    const std::size_t n_ao_pairs = static_cast<std::size_t>(n_basis_functions) * (n_basis_functions + 1ULL) / 2;
    if (ao_ri_result.metric_whitened_ao_pair_factors.rows() != ao_ri_result.n_auxiliary_functions ||
        ao_ri_result.metric_whitened_ao_pair_factors.cols() !=
            static_cast<Eigen::Index>(n_ao_pairs)) {
        throw std::invalid_argument("AO RI factor matrix shape mismatch");
    }

    ActiveSpaceTwoElectronResult result;
    result.representation = ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    result.n_auxiliary_functions = ao_ri_result.n_auxiliary_functions;
    const int n_aux = ao_ri_result.n_auxiliary_functions;
    const Eigen::Index pairs = static_cast<Eigen::Index>(n_active_orbitals) * (n_active_orbitals + 1LL) / 2;
    result.ri_active_pair_factors.resize(n_aux, pairs);
    const int workers = std::min(effective_openmp_thread_count(), n_basis_functions);
    const std::size_t per_aux =
        static_cast<std::size_t>(n_basis_functions) * (n_active_orbitals + workers) +
        n_active_orbitals * n_active_orbitals;
    auto resident =
        workspace::make_matrix<Eigen::MatrixXd>(static_cast<std::size_t>(n_aux) * n_active_orbitals,
                                                n_basis_functions,
                                                per_aux * sizeof(double));
    const auto scratch = workspace::available() / sizeof(double);
    if (scratch < per_aux)
        throw std::length_error("minimum RI active transform tile exceeds workspace");
    const int tile = std::min<std::size_t>(n_aux, scratch / per_aux);
    Eigen::MatrixXd one_index;
    for (int first = 0; first < n_aux; first += tile) {
        const int count = std::min(tile, n_aux - first);
        one_index.resize(static_cast<Eigen::Index>(count) * n_active_orbitals, n_basis_functions);
        transform_ri_one_index(ao_ri_result.metric_whitened_ao_pair_factors,
                               dense_active_coefficients,
                               first,
                               count,
                               one_index);
        const Eigen::MatrixXd active = one_index * dense_active_coefficients;
        for (int a = 0; a < n_active_orbitals; ++a) {
            if (resident)
                resident->middleRows(static_cast<Eigen::Index>(a) * n_aux + first, count) =
                    one_index.middleRows(static_cast<Eigen::Index>(a) * count, count);
            const Eigen::Index pair_begin = static_cast<Eigen::Index>(a) * (a + 1LL) / 2;
            for (int b = 0; b <= a; ++b)
                result.ri_active_pair_factors.col(pair_begin + b).segment(first, count) =
                    active.col(b).segment(static_cast<Eigen::Index>(a) * count, count);
        }
    }
    result.ri_one_index_factors = std::move(resident);
    result.packed_active_two_electron_integrals =
        pack_ri_active_integrals(result.ri_active_pair_factors);
    result.dense_active_coefficients = dense_active_coefficients;

    return result;
}

} // namespace xmvb::integrals
