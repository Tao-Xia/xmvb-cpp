#pragma once

#include <Eigen/Core>

namespace xmvb::vb::spinor {

/** @brief Orthonormal computational frame for independent nonorthogonal spinors.
 *
 * Coefficients are (2*n_bf, n_spinor), with alpha AO rows followed by beta AO
 * rows. The physical AO supports are unchanged: this frame is only a change
 * of computational basis. A singular Gram matrix is rejected, not regularized.
 */
class Frame {
  public:
    Frame(const Eigen::Ref<const Eigen::MatrixXcd>& coefficients,
          const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap);

    const Eigen::MatrixXcd& gram() const noexcept {
        return gram_;
    }
    const Eigen::MatrixXcd& upper() const noexcept {
        return upper_;
    }
    const Eigen::MatrixXcd& coefficients() const noexcept {
        return coefficients_;
    }

  private:
    Eigen::MatrixXcd gram_;
    Eigen::MatrixXcd upper_;
    Eigen::MatrixXcd coefficients_;
};

} // namespace xmvb::vb::spinor
