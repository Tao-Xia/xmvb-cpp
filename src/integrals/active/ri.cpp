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
    const int pairs = static_cast<int>(active_pair_factors.cols());
    std::vector<double> packed(static_cast<std::size_t>(pairs) * (pairs + 1) / 2);
    constexpr int panel_rows = 64;
    for (int first = 0; first < pairs; first += panel_rows) {
        const int count = std::min(panel_rows, pairs - first);
        const Eigen::MatrixXd panel = active_pair_factors.middleCols(first, count).transpose() *
                                      active_pair_factors.leftCols(first + count);
        for (int row = 0; row < count; ++row) {
            const int p = first + row;
            for (int q = 0; q <= p; ++q)
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

    const std::size_t n_ao_pairs = n_basis_functions * (n_basis_functions + 1) / 2;
    if (ao_ri_result.metric_whitened_ao_pair_factors.rows() != ao_ri_result.n_auxiliary_functions ||
        ao_ri_result.metric_whitened_ao_pair_factors.cols() !=
            static_cast<Eigen::Index>(n_ao_pairs)) {
        throw std::invalid_argument("AO RI factor matrix shape mismatch");
    }

    ActiveSpaceTwoElectronResult result;
    result.representation = ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    result.n_auxiliary_functions = ao_ri_result.n_auxiliary_functions;
    const int n_aux = ao_ri_result.n_auxiliary_functions;
    const int pairs = n_active_orbitals * (n_active_orbitals + 1) / 2;
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
                resident->middleRows(a * n_aux + first, count) =
                    one_index.middleRows(a * count, count);
            for (int b = 0; b <= a; ++b)
                result.ri_active_pair_factors.col(a * (a + 1) / 2 + b).segment(first, count) =
                    active.col(b).segment(a * count, count);
        }
    }
    result.ri_one_index_factors = std::move(resident);
    result.packed_active_two_electron_integrals =
        pack_ri_active_integrals(result.ri_active_pair_factors);
    result.dense_active_coefficients = dense_active_coefficients;

    return result;
}

} // namespace xmvb::integrals
