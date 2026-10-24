#pragma once

#include <complex>
#include <memory>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vbscf/structures/spinor/strings.hpp"

namespace xmvb::vb::spinor {

/** @brief Explicit selected-VB parent map B on occupation strings.
 *
 * This map records a prescribed structure dictionary, not variational CI
 * coefficients. Rows use Strings ordering and columns are VB structures.
 * Linear independence is not asserted here; dependent maps still have valid
 * actions and must be handled by the future structure eigensolver.
 */
class Basis {
  public:
    Basis(std::shared_ptr<const Strings> strings,
          Eigen::SparseMatrix<std::complex<double>> mapping);

    int size() const noexcept {
        return static_cast<int>(mapping_.cols());
    }
    const std::shared_ptr<const Strings>& strings() const noexcept {
        return strings_;
    }
    Eigen::MatrixXcd expand(const Eigen::Ref<const Eigen::MatrixXcd>& structures) const;
    Eigen::MatrixXcd contract(const Eigen::Ref<const Eigen::MatrixXcd>& carrier) const;

  private:
    std::shared_ptr<const Strings> strings_;
    Eigen::SparseMatrix<std::complex<double>> mapping_;
};

} // namespace xmvb::vb::spinor
