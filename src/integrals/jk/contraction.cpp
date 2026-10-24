#include "integrals/jk/contraction.hpp"

#include <cmath>
#include <stdexcept>

#include "integrals/ao/exact.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::integrals {

Eigen::MatrixXd contract_jk(const Eigen::Ref<const Eigen::MatrixXd>& density,
                            const AoIntegralInput& ao,
                            JkWeights weights) {
    const int n = ao.n_basis_functions;
    if (n <= 0 || density.rows() != n || density.cols() != n || !density.allFinite() ||
        !density.isApprox(density.transpose()) || !std::isfinite(weights.coulomb) ||
        !std::isfinite(weights.exchange)) {
        throw std::invalid_argument("invalid symmetric AO density or JK weights");
    }
    Eigen::MatrixXd output;
    if (weights.coulomb == 1.0 && weights.exchange == -0.5) {
        // Preserve the existing unique-ERI 2J-K sweep and triangular fold.
        auto storage = apply_ao_h1e(density.data(), ao, effective_openmp_thread_count());
        Eigen::Map<Eigen::MatrixXd> folded(storage.data(), n, n);
        for (int row = 0; row < n; ++row)
            for (int column = 0; column <= row; ++column) {
                folded(row, column) += folded(column, row);
                folded(column, row) = folded(row, column);
            }
        output = 0.5 * folded;
        return output;
    }
    apply_weighted_symmetric_ao_jk(
        density, ao, weights.coulomb, weights.exchange, effective_openmp_thread_count(), &output);
    return output;
}

} // namespace xmvb::integrals
