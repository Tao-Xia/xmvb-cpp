#pragma once

#include <memory>
#include <Eigen/Core>
#include "mcscf/ci/strings.hpp"

namespace xmvb::mcscf {

/** @brief Matrix-free S^2 and exact finite-spectrum spin projection.
 * Coefficients are alpha rows and beta columns. S^2 is evaluated as
 * Ms(Ms+1) + S_- S_+. Only spin-string lookup tables are retained; no
 * product-space transition graph or spin matrix is assembled.
 */
class SpinSpace {
  public:
    SpinSpace(int n_orbitals, int n_alpha, int n_beta);
    const SpinStrings& alpha() const noexcept {
        return alpha_;
    }
    const SpinStrings& beta() const noexcept {
        return beta_;
    }
    Eigen::MatrixXd apply_squared(const Eigen::Ref<const Eigen::MatrixXd>& x) const;
    Eigen::MatrixXd project(const Eigen::Ref<const Eigen::MatrixXd>& x, int twice_spin) const;
    std::size_t dimension(int twice_spin) const;

  private:
    int n_, na_, nb_;
    SpinStrings alpha_, beta_;
    std::unique_ptr<SpinStrings> raised_alpha_, raised_beta_;
};

} // namespace xmvb::mcscf
