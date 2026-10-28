#pragma once

#include <cstddef>
#include "core/index.hpp"
#include <cstdint>
#include <span>
#include <vector>

#include <Eigen/Core>
#include "mcscf/ci/exterior_row.hpp"

namespace xmvb::vb {

/** @brief Exact compound-matrix rows by descending occupied-row recursion.
 * Only Pascal coefficients and the caller's column permutation are retained.
 * Each worker uses two single-spin vectors, never a spin-product matrix.
 */
class ExteriorRowPlan {
  public:
    using Rows = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    ExteriorRowPlan(int n_orbitals, int n_electrons, std::span<const std::uint64_t> masks);

    Rows apply(const Eigen::MatrixXd& upper,
               std::span<const std::uint64_t> masks,
               std::span<const Eigen::Index> rows) const;
    std::size_t dynamic_bytes() const noexcept;
    mcscf::detail::ExteriorRowTopology topology() const noexcept {
        return {n_,
                electrons_,
                core::checked_index<std::int64_t>(column_ranks_.size()),
                binomial_.data(),
                column_ranks_.data()};
    }
    std::span<const std::uint64_t> binomial() const noexcept {
        return binomial_;
    }
    std::span<const std::int64_t> column_ranks() const noexcept {
        return column_ranks_;
    }

  private:
    std::uint64_t choose(int n, int k) const noexcept;
    int n_, electrons_;
    std::vector<std::uint64_t> binomial_;
    std::vector<std::int64_t> column_ranks_;
};

} // namespace xmvb::vb
