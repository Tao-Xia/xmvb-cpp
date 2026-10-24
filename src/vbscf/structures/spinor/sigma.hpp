#pragma once

#include <complex>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vbscf/structures/spinor/strings.hpp"

namespace xmvb::vb::spinor {

/**
 * @brief Fixed-N complex Hamiltonian action on the complete spinor carrier.
 *
 * Exact integrals use g(p,q,r,s) at (((p*n+q)*n+r)*n+s), where
 * H2 = 1/2 sum_pqrs g(p,q,r,s) a_p^dagger a_r^dagger a_s a_q.
 * RI factors use g(p,q,r,s) = sum_P L_P(p,q)L_P(r,s), without conjugating
 * the second factor. The full carrier still costs C(n,N) times block width
 * complex amplitudes; neither representation removes that limitation.
 */
class Sigma {
  public:
    Sigma(std::shared_ptr<const Strings> strings,
          Eigen::MatrixXcd h,
          std::vector<std::complex<double>> eri);

    Sigma(std::shared_ptr<const Strings> strings,
          Eigen::MatrixXcd h,
          std::vector<Eigen::MatrixXcd> factors);

    const std::shared_ptr<const Strings>& strings() const noexcept {
        return strings_;
    }

    /** @brief Apply H to each RHS column in masks() row order. */
    Eigen::MatrixXcd apply(const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const;

  private:
    void validate_common() const;
    void validate_exact() const;
    void validate_ri() const;
    Eigen::MatrixXcd apply_exact_two_body(const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const;

    std::shared_ptr<const Strings> strings_;
    Eigen::MatrixXcd h_;
    std::vector<std::complex<double>> eri_;
    std::vector<Eigen::MatrixXcd> factors_;
    bool ri_ = false;
};

} // namespace xmvb::vb::spinor
